"""
Ensemble Weighting
==================
Dynamic, walk-forward ensemble construction for the model zoo.

Weights are NOT hardcoded. They are learned from rolling OOS performance
on previous walk-forward windows using five signal-quality dimensions:

    ic_score      : rolling mean rank IC (signal direction quality)
    tstat_score   : IC t-stat (statistical significance)
    spread_score  : decile spread (cross-sectional separation)
    sharpe_score  : long-only quintile Sharpe (return quality)
    stability_pen : IC coefficient-of-variation penalty (unstable models penalized)
    corr_pen      : signal correlation penalty (redundant models downweighted)
    turnover_pen  : turnover penalty (high-churn models downweighted)

Per-window weight computation
------------------------------
    raw_score[m] = (
        w_ic    × ic_score[m]     +
        w_tstat × tstat_score[m]  +
        w_spread× spread_score[m] +
        w_sharpe× sharpe_score[m]
    )
    penalized[m] = raw_score[m]
                   × exp(−λ_stab  × ic_cv[m])
                   × exp(−λ_corr  × mean_pairwise_corr[m])
                   × exp(−λ_to    × turnover[m])
    weight[m] = softmax(penalized) with temperature T

The weights from the most recent K windows are EMA-averaged into the
final ensemble weight for the next test window (recency bias toward
more recent OOS performance).

Usage
-----
    from model_selection.ensemble_weighting import (
        EnsembleConfig, build_dynamic_ensemble,
        evaluate_ensemble, format_ensemble_report,
    )
    from model_selection.objective_comparison import WalkForwardConfig

    ens_cfg = EnsembleConfig()
    wf_cfg = WalkForwardConfig()

    result = build_dynamic_ensemble(panel_df, models, feature_cols, wf_cfg, ens_cfg)
    print(format_ensemble_report(result))
"""

from __future__ import annotations

import copy
import math
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import softmax as _softmax

from model_selection.objective_comparison import (
    WalkForwardConfig,
    _WindowPrep,
    _aggregate_window_metrics,
    _daily_rank_ic,
    _decile_metrics,
    _fit_one_window,
    _long_quintile_sharpe,
    _top_quintile_turnover,
    _walk_forward_dates,
    prepare_window,
)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class EnsembleConfig:
    # Metric weights for raw score
    w_ic: float = 0.30          # IC mean weight
    w_tstat: float = 0.25       # IC t-stat weight
    w_spread: float = 0.20      # decile spread weight
    w_sharpe: float = 0.25      # long-only Sharpe weight

    # Penalty lambdas (higher = more aggressive penalty)
    lambda_stability: float = 0.5   # IC_CV penalty
    lambda_corr: float = 1.0        # pairwise signal correlation penalty
    lambda_turnover: float = 0.3    # top-quintile turnover penalty

    # Softmax temperature: lower T → sharper concentration on best model
    temperature: float = 0.5

    # EMA lookback for weight smoothing across windows
    ema_alpha: float = 0.4          # higher = more recency weight

    # Minimum windows before a model receives nonzero weight
    min_windows: int = 1

    # Minimum IC t-stat to consider a model's score valid
    min_ic_tstat: float = 0.5

    # Whether to normalize each model's score cross-sectionally before combining
    cross_section_normalize: bool = True

    # Evaluation: top-quintile fraction for long-only Sharpe
    top_q: float = 0.20

    # Score correlation window: number of test dates to use for signal correlation
    corr_window_dates: int = 20


# ---------------------------------------------------------------------------
# Per-window weight computation
# ---------------------------------------------------------------------------


def _cross_section_normalize_scores(
    df: pd.DataFrame, score_col: str = "score"
) -> pd.DataFrame:
    """Z-score normalize scores within each date cross-section."""
    out = df.copy()
    grouped = out.groupby("date")[score_col]
    means = grouped.transform("mean")
    stds = grouped.transform("std").fillna(1.0).clip(lower=1e-10)
    out[score_col] = (out[score_col] - means) / stds
    return out


def _compute_window_metrics(
    scored_frame: pd.DataFrame,
    cfg: WalkForwardConfig,
) -> dict[str, float]:
    """Compute IC, t-stat, spread, Sharpe, turnover for a single OOS window."""
    ic_vals = _daily_rank_ic(scored_frame, "score", cfg.target_col)
    ic_arr = np.array(ic_vals, dtype=float)
    ic_arr = ic_arr[np.isfinite(ic_arr)]

    if len(ic_arr) < 3:
        return {
            "ic_mean": float("nan"),
            "ic_tstat": float("nan"),
            "ic_cv": float("nan"),
            "decile_spread": float("nan"),
            "long_sharpe": float("nan"),
            "turnover": float("nan"),
        }

    ic_mean = float(ic_arr.mean())
    ic_std = float(ic_arr.std(ddof=1)) if len(ic_arr) > 1 else 1e-8
    ic_tstat = ic_mean / (ic_std / math.sqrt(len(ic_arr))) if ic_std > 1e-10 else 0.0
    ic_cv = ic_std / abs(ic_mean) if abs(ic_mean) > 1e-10 else float("nan")

    # Calibrate direction from IC sign so inverted models evaluate correctly
    model_direction: int = -1 if ic_mean < 0.0 else 1
    if model_direction == -1:
        cal_frame = scored_frame.copy()
        cal_frame["score"] = -cal_frame["score"]
    else:
        cal_frame = scored_frame

    spread, _ = _decile_metrics(cal_frame, "score", cfg.target_col, sign=1)
    sharpe = _long_quintile_sharpe(
        cal_frame, "score", cfg.raw_return_col, holding_days=cfg.holding_days
    )
    turnover = _top_quintile_turnover(cal_frame, "score")

    return {
        "ic_mean": ic_mean,
        "ic_tstat": ic_tstat,
        "ic_cv": ic_cv if math.isfinite(ic_cv) else 5.0,
        "decile_spread": spread,
        "long_sharpe": sharpe,
        "turnover": turnover,
        "model_direction": float(model_direction),
    }


def _rank_normalize(values: dict[str, float]) -> dict[str, float]:
    """Rank-normalize a dict of scores to [0, 1]."""
    items = [(k, v) for k, v in values.items() if math.isfinite(v)]
    if not items:
        return {k: 0.0 for k in values}
    vals = [v for _, v in items]
    vmin, vmax = min(vals), max(vals)
    if vmax == vmin:
        return {k: 0.5 for k in values}
    normed = {k: (v - vmin) / (vmax - vmin) for k, v in items}
    return {k: normed.get(k, 0.0) for k in values}


def compute_window_weights(
    per_model_window_metrics: dict[str, dict[str, float]],
    per_model_corr: dict[str, float],
    cfg: EnsembleConfig,
) -> dict[str, float]:
    """
    Compute ensemble weights for one OOS window from per-model metrics.

    Parameters
    ----------
    per_model_window_metrics : {model_name: {metric: value}} for this window
    per_model_corr           : {model_name: mean pairwise abs corr}
    cfg                      : EnsembleConfig

    Returns
    -------
    {model_name: weight} — sum=1, non-negative
    """
    names = list(per_model_window_metrics.keys())
    if not names:
        return {}

    # Extract per-metric dicts — use absolute values so inverted models rank correctly
    ic_vals = {n: abs(m.get("ic_mean", float("nan"))) for n, m in per_model_window_metrics.items()}
    tstat_vals = {n: abs(m.get("ic_tstat", float("nan"))) for n, m in per_model_window_metrics.items()}
    spread_vals = {n: m.get("decile_spread", float("nan")) for n, m in per_model_window_metrics.items()}
    sharpe_vals = {n: m.get("long_sharpe", float("nan")) for n, m in per_model_window_metrics.items()}
    cv_vals = {n: m.get("ic_cv", 5.0) for n, m in per_model_window_metrics.items()}
    turnover_vals = {n: m.get("turnover", 0.0) for n, m in per_model_window_metrics.items()}

    # Rank-normalize each metric
    ic_n = _rank_normalize(ic_vals)
    tstat_n = _rank_normalize(tstat_vals)
    spread_n = _rank_normalize(spread_vals)
    sharpe_n = _rank_normalize(sharpe_vals)

    # Raw score per model
    raw: dict[str, float] = {}
    for name in names:
        # Zero-out models below minimum t-stat threshold
        tstat = tstat_vals.get(name, 0.0)
        if not math.isfinite(tstat) or abs(tstat) < cfg.min_ic_tstat:
            raw[name] = 0.0
            continue
        raw[name] = (
            cfg.w_ic     * ic_n.get(name, 0.0)
            + cfg.w_tstat  * tstat_n.get(name, 0.0)
            + cfg.w_spread * spread_n.get(name, 0.0)
            + cfg.w_sharpe * sharpe_n.get(name, 0.0)
        )

    # Apply penalties multiplicatively
    penalized: dict[str, float] = {}
    for name in names:
        score = raw.get(name, 0.0)
        cv = cv_vals.get(name, 5.0)
        corr = per_model_corr.get(name, 0.0)
        to = turnover_vals.get(name, 0.0)
        if not math.isfinite(cv):
            cv = 5.0
        if not math.isfinite(corr):
            corr = 0.0
        if not math.isfinite(to):
            to = 0.0

        stability_pen = math.exp(-cfg.lambda_stability * max(cv - 1.0, 0.0))
        corr_pen = math.exp(-cfg.lambda_corr * corr)
        to_pen = math.exp(-cfg.lambda_turnover * to)

        penalized[name] = score * stability_pen * corr_pen * to_pen

    # Softmax with temperature to produce final weights
    names_sorted = sorted(penalized.keys())
    scores_arr = np.array([penalized[n] for n in names_sorted], dtype=float)
    scores_arr = np.nan_to_num(scores_arr, nan=0.0, posinf=0.0, neginf=0.0)

    if scores_arr.max() == scores_arr.min() == 0.0:
        weights_arr = np.ones(len(names_sorted)) / len(names_sorted)
    else:
        weights_arr = _softmax(scores_arr / max(cfg.temperature, 1e-4))

    return {n: float(w) for n, w in zip(names_sorted, weights_arr)}


# ---------------------------------------------------------------------------
# EMA weight smoothing across windows
# ---------------------------------------------------------------------------


def ema_smooth_weights(
    weight_history: list[dict[str, float]],
    alpha: float,
) -> dict[str, float]:
    """
    EMA-smooth a list of per-window weight dicts into a single weight vector.
    Most recent window gets the most weight.
    """
    if not weight_history:
        return {}
    all_names = sorted({n for w in weight_history for n in w})
    ema: dict[str, float] = {}

    for t, w_dict in enumerate(weight_history):
        a_t = alpha * (1.0 - alpha) ** (len(weight_history) - 1 - t)
        for name in all_names:
            ema[name] = ema.get(name, 0.0) + a_t * w_dict.get(name, 0.0)

    total = sum(ema.values())
    if total > 1e-10:
        ema = {k: v / total for k, v in ema.items()}
    else:
        ema = {k: 1.0 / len(all_names) for k in all_names}
    return ema


# ---------------------------------------------------------------------------
# Walk-forward ensemble training
# ---------------------------------------------------------------------------


@dataclass
class EnsembleResult:
    # Final ensemble OOS scored frames (concatenated across windows)
    oos_frames: list[pd.DataFrame]
    # Per-window weight history: [{model: weight} for each window]
    weight_history: list[dict[str, float]]
    # Final smoothed weights (used for last window)
    final_weights: dict[str, float]
    # Per-window per-model raw metrics
    window_metrics: list[dict[str, dict[str, float]]]
    # Per-model aggregated metrics (baseline, no ensemble)
    individual_metrics: dict[str, dict[str, float]]
    # Ensemble aggregated metrics
    ensemble_metrics: dict[str, float]
    # Model contribution table
    contribution_table: pd.DataFrame
    cfg: EnsembleConfig
    wf_cfg: WalkForwardConfig


def build_dynamic_ensemble(
    df: pd.DataFrame,
    models: list[tuple[str, Any, bool, str]],
    feature_cols: list[str],
    wf_cfg: WalkForwardConfig | None = None,
    ens_cfg: EnsembleConfig | None = None,
) -> EnsembleResult:
    """
    Build a dynamic walk-forward ensemble.

    For each OOS window:
    1. Fit all models on [train_start, train_end)
    2. Score test window with each model
    3. Compute per-model OOS metrics
    4. Derive weights for the NEXT window via penalized softmax
    5. Combine model scores using the weights from the PREVIOUS window

    Final ensemble score on window t uses weights learned from windows 1..t-1
    (pure OOS — no lookahead).  Window 1 uses equal weights.

    Parameters
    ----------
    df          : panel DataFrame
    models      : list of (name, estimator, uses_proba, kind) from build_models()
    feature_cols: alpha feature column names
    wf_cfg      : WalkForwardConfig
    ens_cfg     : EnsembleConfig

    Returns
    -------
    EnsembleResult
    """
    if wf_cfg is None:
        wf_cfg = WalkForwardConfig()
    if ens_cfg is None:
        ens_cfg = EnsembleConfig()

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date"]).sort_values(["date", "ticker"]).reset_index(drop=True)

    active_feat = [c for c in feature_cols if c in work.columns]
    windows = _walk_forward_dates(work, wf_cfg)
    if not windows:
        empty_metrics: dict[str, float] = {
            k: float("nan") for k in [
                "ic_mean", "ic_tstat", "ic_ir", "decile_spread",
                "monotonicity", "long_sharpe", "turnover", "alpha_cap_proxy",
            ]
        }
        return EnsembleResult([], [], {}, [], {}, empty_metrics, pd.DataFrame(), ens_cfg, wf_cfg)

    valid_models = [(n, m, u, k) for n, m, u, k in models if k != "short_classifier"]
    model_names = [n for n, _, _, _ in valid_models]

    # Storage: per_window_scored[window_idx][model_name] = scored_frame | None
    per_window_scored: list[dict[str, pd.DataFrame | None]] = []
    weight_history: list[dict[str, float]] = []
    window_metrics: list[dict[str, dict[str, float]]] = []
    oos_ensemble_frames: list[pd.DataFrame] = []

    # Individual model storage for final aggregate
    individual_frames: dict[str, list[pd.DataFrame]] = {n: [] for n in model_names}

    for w_idx, (train_start, train_end, test_start, test_end) in enumerate(windows):
        train_df = work[(work["date"] >= train_start) & (work["date"] <= train_end)]
        test_df = work[(work["date"] >= test_start) & (work["date"] <= test_end)]
        if len(train_df) < wf_cfg.min_train_obs or test_df.empty:
            per_window_scored.append({n: None for n in model_names})
            continue

        # --- Precompute preprocessing once, share across all models in this window ---
        window_prep: _WindowPrep | None = prepare_window(train_df, test_df, active_feat, wf_cfg)

        # --- Fit all models and collect OOS scores ---
        window_scored: dict[str, pd.DataFrame | None] = {}
        for name, model, _uses_proba, kind in valid_models:
            scored = _fit_one_window(
                name, model, kind, train_df, test_df, active_feat, wf_cfg.target_col, wf_cfg,
                _precomputed=window_prep,
            )
            window_scored[name] = scored
            if scored is not None and not scored.empty:
                individual_frames[name].append(scored)
        per_window_scored.append(window_scored)

        # --- Compute per-model metrics for this window ---
        w_metrics: dict[str, dict[str, float]] = {}
        for name in model_names:
            scored = window_scored.get(name)
            if scored is None or scored.empty:
                w_metrics[name] = {
                    "ic_mean": float("nan"),
                    "ic_tstat": float("nan"),
                    "ic_cv": float("nan"),
                    "decile_spread": float("nan"),
                    "long_sharpe": float("nan"),
                    "turnover": float("nan"),
                }
            else:
                w_metrics[name] = _compute_window_metrics(scored, wf_cfg)
        window_metrics.append(w_metrics)

        # --- Compute pairwise signal correlations ---
        # Build correlation from current window scored frames
        cur_scores: dict[str, pd.Series] = {}
        for name in model_names:
            s = window_scored.get(name)
            if s is not None and not s.empty and "score" in s.columns:
                pivot = s.set_index(["date", "ticker"])["score"]
                cur_scores[name] = pivot
        pairwise_corr = _compute_pairwise_corr(cur_scores)

        # --- Extract per-model directions from this window's metrics ---
        window_directions: dict[str, int] = {
            name: int(w_metrics[name].get("model_direction", 1.0))
            for name in model_names
            if math.isfinite(w_metrics[name].get("model_direction", 1.0))
        }

        # --- Derive weights for NEXT window ---
        new_weights = compute_window_weights(w_metrics, pairwise_corr, ens_cfg)
        weight_history.append(new_weights)

        # --- Ensemble score for THIS window using PREVIOUS weights ---
        if w_idx == 0:
            # First window: equal weights (no prior history)
            curr_weights = {n: 1.0 / len(model_names) for n in model_names}
        else:
            # EMA over all previous windows
            curr_weights = ema_smooth_weights(weight_history[:-1], ens_cfg.ema_alpha)

        ensemble_scored = _combine_scores(
            window_scored, curr_weights, test_df, wf_cfg, ens_cfg,
            model_directions=window_directions,
        )
        if ensemble_scored is not None and not ensemble_scored.empty:
            oos_ensemble_frames.append(ensemble_scored)

    # --- Final smoothed weights ---
    final_weights = ema_smooth_weights(weight_history, ens_cfg.ema_alpha) if weight_history else {}

    # --- Aggregate individual model metrics ---
    individual_metrics: dict[str, dict[str, float]] = {}
    for name, frames in individual_frames.items():
        if frames:
            individual_metrics[name] = _aggregate_window_metrics(frames, wf_cfg)
        else:
            individual_metrics[name] = {k: float("nan") for k in [
                "ic_mean", "ic_tstat", "ic_ir", "decile_spread",
                "monotonicity", "long_sharpe", "turnover", "alpha_cap_proxy",
            ]}

    # --- Aggregate ensemble metrics ---
    ensemble_metrics = _aggregate_window_metrics(oos_ensemble_frames, wf_cfg) if oos_ensemble_frames else {
        k: float("nan") for k in [
            "ic_mean", "ic_tstat", "ic_ir", "decile_spread",
            "monotonicity", "long_sharpe", "turnover", "alpha_cap_proxy",
        ]
    }

    # --- Contribution table ---
    contribution_table = _build_contribution_table(
        individual_metrics, final_weights, window_metrics
    )

    return EnsembleResult(
        oos_frames=oos_ensemble_frames,
        weight_history=weight_history,
        final_weights=final_weights,
        window_metrics=window_metrics,
        individual_metrics=individual_metrics,
        ensemble_metrics=ensemble_metrics,
        contribution_table=contribution_table,
        cfg=ens_cfg,
        wf_cfg=wf_cfg,
    )


def _compute_pairwise_corr(
    cur_scores: dict[str, pd.Series],
) -> dict[str, float]:
    """Compute mean pairwise absolute Spearman correlation for each model."""
    names = list(cur_scores.keys())
    if len(names) < 2:
        return {n: 0.0 for n in names}

    combined = pd.concat(cur_scores.values(), axis=1, keys=names).dropna(how="all")
    if combined.empty or len(combined) < 5:
        return {n: 0.0 for n in names}

    corr_matrix = combined.rank().corr(method="spearman").abs()
    return {
        n: float(corr_matrix.loc[n, [m for m in names if m != n]].mean())
        if n in corr_matrix.index else 0.0
        for n in names
    }


def _combine_scores(
    window_scored: dict[str, pd.DataFrame | None],
    weights: dict[str, float],
    test_df: pd.DataFrame,
    wf_cfg: WalkForwardConfig,
    ens_cfg: EnsembleConfig,
    model_directions: dict[str, int] | None = None,
) -> pd.DataFrame | None:
    """Combine per-model scores for one OOS window using given weights.

    Each model's score is flipped by its direction before combining so all
    scores share the same polarity (higher = stronger long signal).
    """
    score_arrays: list[tuple[float, np.ndarray]] = []
    ref_frame: pd.DataFrame | None = None
    _dirs = model_directions or {}

    for name, scored in window_scored.items():
        w = weights.get(name, 0.0)
        if scored is None or scored.empty or w <= 0:
            continue
        s = pd.to_numeric(scored["score"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(s).any():
            continue
        # Apply direction calibration exactly once per model
        direction = _dirs.get(name, 1)
        s = s * direction
        # Optional cross-section normalize before combining
        if ens_cfg.cross_section_normalize and "date" in scored.columns:
            tmp = scored.copy()
            tmp["score"] = s
            tmp = _cross_section_normalize_scores(tmp, "score")
            s = tmp["score"].to_numpy(dtype=float)
        score_arrays.append((w, s))
        if ref_frame is None:
            ref_frame = scored

    if not score_arrays or ref_frame is None:
        return None

    total_w = sum(w for w, _ in score_arrays)
    if total_w <= 0:
        return None

    ensemble_score = sum(w * s for w, s in score_arrays) / total_w

    out = ref_frame[["date", "ticker"]].copy()
    if wf_cfg.target_col in ref_frame.columns:
        out[wf_cfg.target_col] = pd.to_numeric(ref_frame[wf_cfg.target_col], errors="coerce").to_numpy()
    if wf_cfg.raw_return_col in ref_frame.columns:
        out[wf_cfg.raw_return_col] = pd.to_numeric(ref_frame[wf_cfg.raw_return_col], errors="coerce").to_numpy()
    for col in ("raw_5d", "target_return", "target_rank"):
        if col in ref_frame.columns and col not in out.columns:
            out[col] = pd.to_numeric(ref_frame[col], errors="coerce").to_numpy()
    out["score"] = ensemble_score
    return out


def _build_contribution_table(
    individual_metrics: dict[str, dict[str, float]],
    final_weights: dict[str, float],
    window_metrics: list[dict[str, dict[str, float]]],
) -> pd.DataFrame:
    """Build per-model contribution diagnostic table."""
    rows = []
    all_names = sorted(individual_metrics.keys())

    for name in all_names:
        m = individual_metrics.get(name, {})
        w = final_weights.get(name, 0.0)

        # IC stability across windows
        window_ics = [
            wm.get(name, {}).get("ic_mean", float("nan"))
            for wm in window_metrics
        ]
        window_ics_valid = [x for x in window_ics if math.isfinite(x)]
        ic_stability = (
            float(np.std(window_ics_valid, ddof=1) / abs(np.mean(window_ics_valid)))
            if len(window_ics_valid) > 1 and abs(np.mean(window_ics_valid)) > 1e-8
            else float("nan")
        )

        # Average window-level IC t-stat
        window_tstats = [
            wm.get(name, {}).get("ic_tstat", float("nan"))
            for wm in window_metrics
        ]
        window_tstats_valid = [x for x in window_tstats if math.isfinite(x)]
        mean_window_tstat = float(np.mean(window_tstats_valid)) if window_tstats_valid else float("nan")

        # Modal direction across windows (-1 or +1)
        window_dirs = [
            int(wm.get(name, {}).get("model_direction", 1.0))
            for wm in window_metrics
            if math.isfinite(wm.get(name, {}).get("model_direction", float("nan")))
        ]
        modal_direction = int(np.sign(sum(window_dirs))) if window_dirs else 1

        rows.append(
            {
                "model": name,
                "final_weight": round(w, 4),
                "model_direction": modal_direction,
                "ic_mean": round(float(m.get("ic_mean", float("nan"))), 5),
                "ic_tstat": round(float(m.get("ic_tstat", float("nan"))), 3),
                "ic_stability_cv": round(ic_stability, 3) if math.isfinite(ic_stability) else float("nan"),
                "mean_window_tstat": round(mean_window_tstat, 3) if math.isfinite(mean_window_tstat) else float("nan"),
                "long_sharpe": round(float(m.get("long_sharpe", float("nan"))), 3),
                "decile_spread": round(float(m.get("decile_spread", float("nan"))), 5),
                "turnover": round(float(m.get("turnover", float("nan"))), 3),
                "alpha_cap_proxy": round(float(m.get("alpha_cap_proxy", float("nan"))), 3),
            }
        )

    if not rows:
        return pd.DataFrame()
    return (
        pd.DataFrame(rows)
        .sort_values("final_weight", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# Ensemble evaluation
# ---------------------------------------------------------------------------


def evaluate_ensemble(result: EnsembleResult) -> pd.DataFrame:
    """
    Compare ensemble vs each individual model on IC, stability, Sharpe, drawdown.
    Returns a comparison DataFrame sorted by ic_tstat descending.
    """
    rows = []

    # Ensemble row
    em = result.ensemble_metrics
    rows.append(
        {
            "model": "** ENSEMBLE **",
            "ic_mean": em.get("ic_mean", float("nan")),
            "ic_tstat": em.get("ic_tstat", float("nan")),
            "ic_ir": em.get("ic_ir", float("nan")),
            "long_sharpe": em.get("long_sharpe", float("nan")),
            "decile_spread": em.get("decile_spread", float("nan")),
            "monotonicity": em.get("monotonicity", float("nan")),
            "turnover": em.get("turnover", float("nan")),
            "alpha_cap_proxy": em.get("alpha_cap_proxy", float("nan")),
        }
    )

    # Individual model rows
    for name, m in result.individual_metrics.items():
        rows.append(
            {
                "model": name,
                "ic_mean": m.get("ic_mean", float("nan")),
                "ic_tstat": m.get("ic_tstat", float("nan")),
                "ic_ir": m.get("ic_ir", float("nan")),
                "long_sharpe": m.get("long_sharpe", float("nan")),
                "decile_spread": m.get("decile_spread", float("nan")),
                "monotonicity": m.get("monotonicity", float("nan")),
                "turnover": m.get("turnover", float("nan")),
                "alpha_cap_proxy": m.get("alpha_cap_proxy", float("nan")),
            }
        )

    df = pd.DataFrame(rows)
    df["_sort"] = df["ic_tstat"].fillna(-99)
    df = df.sort_values("_sort", ascending=False).drop(columns=["_sort"])
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

_W = 30
_NW = 10


def format_ensemble_report(result: EnsembleResult) -> str:
    lines: list[str] = []

    # ── Configuration ────────────────────────────────────────────────────────
    c = result.cfg
    lines.append("=" * 90)
    lines.append("DYNAMIC ENSEMBLE WEIGHTING REPORT")
    lines.append("=" * 90)
    lines.append(
        f"  Weights from: rolling IC ({c.w_ic:.2f}) + IC_tstat ({c.w_tstat:.2f}) "
        f"+ spread ({c.w_spread:.2f}) + Sharpe ({c.w_sharpe:.2f})"
    )
    lines.append(
        f"  Penalties: stability λ={c.lambda_stability:.2f} | "
        f"corr λ={c.lambda_corr:.2f} | turnover λ={c.lambda_turnover:.2f}"
    )
    lines.append(f"  Softmax T={c.temperature:.2f} | EMA α={c.ema_alpha:.2f} | "
                 f"min_IC_tstat={c.min_ic_tstat:.1f}")
    lines.append(f"  Walk-forward windows: {len(result.weight_history)}")
    lines.append("")

    # ── Weight evolution ─────────────────────────────────────────────────────
    lines.append("WEIGHT EVOLUTION  (per window, most recent last)")
    lines.append("-" * 90)
    if result.weight_history:
        model_names = sorted(result.final_weights.keys())
        if model_names:
            hdr = f"  {'Window':>8} " + "".join(f"{n[:12]:>{_NW}}" for n in model_names)
            lines.append(hdr)
            lines.append("  " + "-" * (8 + _NW * len(model_names)))
            for i, w_dict in enumerate(result.weight_history):
                vals = "".join(f"{w_dict.get(n, 0.0):>{_NW}.3f}" for n in model_names)
                lines.append(f"  {i+1:>8} {vals}")
            lines.append(f"  {'FINAL':>8} " + "".join(f"{result.final_weights.get(n, 0.0):>{_NW}.3f}" for n in model_names))
    else:
        lines.append("  [No windows completed]")

    # ── Contribution table ────────────────────────────────────────────────────
    lines.append("")
    lines.append("MODEL CONTRIBUTION  (sorted by final weight)")
    lines.append("-" * 90)
    if not result.contribution_table.empty:
        ct = result.contribution_table
        num_cols = [c for c in ["final_weight", "ic_mean", "ic_tstat",
                                 "ic_stability_cv", "long_sharpe",
                                 "decile_spread", "turnover", "alpha_cap_proxy"]
                    if c in ct.columns]
        hdr = f"  {'Model':<{_W}}" + "".join(f"{c:>{_NW}}" for c in num_cols)
        lines.append(hdr)
        lines.append("  " + "-" * (len(hdr) - 2))
        for _, row in ct.iterrows():
            vals = "".join(
                f"{float(row.get(c, float('nan'))):>{_NW}.4f}" for c in num_cols
            )
            lines.append(f"  {str(row.get('model', '')):<{_W}}{vals}")
    else:
        lines.append("  [No contribution data]")

    # ── Ensemble vs individual comparison ────────────────────────────────────
    lines.append("")
    lines.append("ENSEMBLE vs INDIVIDUAL MODELS  (sorted by IC t-stat)")
    lines.append("-" * 90)
    comp = evaluate_ensemble(result)
    if not comp.empty:
        num_cols_c = [c for c in ["ic_mean", "ic_tstat", "ic_ir",
                                   "long_sharpe", "decile_spread",
                                   "monotonicity", "turnover", "alpha_cap_proxy"]
                     if c in comp.columns]
        hdr_c = f"  {'Model':<{_W}}" + "".join(f"{c:>{_NW}}" for c in num_cols_c)
        lines.append(hdr_c)
        lines.append("  " + "-" * (len(hdr_c) - 2))
        for _, row in comp.iterrows():
            model_str = str(row.get("model", ""))
            vals = "".join(f"{float(row.get(c, float('nan'))):>{_NW}.4f}" for c in num_cols_c)
            prefix = "►" if model_str.startswith("**") else " "
            lines.append(f" {prefix} {model_str:<{_W}}{vals}")
    else:
        lines.append("  [No comparison data]")

    # ── Δ over best individual ───────────────────────────────────────────────
    lines.append("")
    lines.append("LIFT vs BEST INDIVIDUAL MODEL")
    lines.append("-" * 90)
    lines.append(_lift_verdict(result, comp))

    # ── Verdict ──────────────────────────────────────────────────────────────
    lines.append("")
    lines.append("=" * 90)
    lines.append("VERDICT")
    lines.append("=" * 90)
    lines.append(_ensemble_verdict(result, comp))

    return "\n".join(lines)


def _lift_verdict(result: EnsembleResult, comp: pd.DataFrame) -> str:
    if comp.empty or "ic_tstat" not in comp.columns:
        return "  Insufficient data."

    ens_row = comp[comp["model"].str.startswith("**")]
    ind_rows = comp[~comp["model"].str.startswith("**")]
    if ens_row.empty or ind_rows.empty:
        return "  Insufficient data."

    best_ind = ind_rows.loc[ind_rows["ic_tstat"].idxmax()]
    ens = ens_row.iloc[0]
    parts = []
    for col in ["ic_mean", "ic_tstat", "long_sharpe", "decile_spread"]:
        ev = float(ens.get(col, float("nan")))
        bv = float(best_ind.get(col, float("nan")))
        if math.isfinite(ev) and math.isfinite(bv):
            parts.append(f"  {col:<20} ensemble={ev:.4f}  best_individual={bv:.4f}  Δ={ev-bv:+.4f}")
    return "\n".join(parts) if parts else "  No comparable metrics."


def _ensemble_verdict(result: EnsembleResult, comp: pd.DataFrame) -> str:
    em = result.ensemble_metrics
    parts = []

    n_models = len(result.individual_metrics)
    n_windows = len(result.weight_history)
    parts.append(f"  {n_models} models evaluated over {n_windows} walk-forward windows.")

    if result.final_weights:
        top3 = sorted(result.final_weights.items(), key=lambda x: -x[1])[:3]
        parts.append(
            "  Top-weighted models: "
            + ", ".join(f"{n} ({w:.2f})" for n, w in top3)
        )

    if not comp.empty and "ic_tstat" in comp.columns:
        ens_rows = comp[comp["model"].str.startswith("**")]
        ind_rows = comp[~comp["model"].str.startswith("**")]
        if not ens_rows.empty and not ind_rows.empty:
            ens_t = float(ens_rows["ic_tstat"].iloc[0])
            best_t = float(ind_rows["ic_tstat"].max())
            mean_t = float(ind_rows["ic_tstat"].mean())
            lift_vs_best = ens_t - best_t
            lift_vs_mean = ens_t - mean_t
            parts.append(
                f"  Ensemble IC t-stat: {ens_t:.3f} | "
                f"best individual: {best_t:.3f} (Δ={lift_vs_best:+.3f}) | "
                f"mean individual: {mean_t:.3f} (Δ={lift_vs_mean:+.3f})"
            )
            if lift_vs_best > 0.3:
                parts.append("  RECOMMEND: ensemble materially outperforms best individual.")
            elif lift_vs_best > 0:
                parts.append("  MARGINAL GAIN: ensemble slightly better than best individual.")
            else:
                parts.append(
                    "  ENSEMBLE DOES NOT BEAT BEST INDIVIDUAL on IC t-stat.  "
                    "Consider: (a) fewer/better models, (b) lower temperature, "
                    "(c) stronger per-model differentiation."
                )

    return "\n".join(parts) if parts else "  Insufficient data for verdict."
