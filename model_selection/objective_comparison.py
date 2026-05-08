"""
Objective Comparison
====================
Walk-forward signal quality comparison across model objectives:

    binary classification  (baseline — predicts P(return > 0))
    Ridge / ElasticNet     (continuous regression on residualized return)
    XGBoost / LightGBM     (ranking objectives: pairwise, NDCG, lambdarank)
    Quantile regression    (robust tail-weighted signal)

For each model the evaluator runs a rolling walk-forward:
    Train: [t − train_days, t)
    Test:  [t, t + test_days)   × n_windows

Per OOS window the following metrics are computed cross-sectionally per date:

    ic_mean         : mean daily rank IC(score, target_col)
    ic_tstat        : Newey–West HAC t-stat on IC series
    ic_ir           : |IC mean| / IC std
    decile_spread   : mean(top-decile target) − mean(bottom-decile target)
    monotonicity    : fraction of adjacent decile pairs with correct sign
    long_sharpe     : annualized Sharpe — top-quintile by score on raw_5d
    turnover        : mean fraction of top-quintile that changes per period
    alpha_cap_proxy : IC_mean × (1 − turnover_cost_ratio)
    composite       : weighted combination for final ranking

Two comparison outputs are produced:
    1. Full diagnostics table — one row per model, sorted by composite score
    2. Δ-vs-classifier table — regression/ranker lift over classifier baseline

Usage
-----
    from model_selection.objective_comparison import compare_objectives, format_comparison_table
    from model_selection.validation import ExecutionCostConfig
    from model_selection.model_registry import build_models

    models = build_models(cfg)
    table, delta = compare_objectives(
        panel_df,
        models=models,
        feature_cols=["f_trend", "ret_5d", "quality_score"],
        target_col="target_rank",
        train_days=252,
        test_days=21,
        n_windows=8,
    )
    print(format_comparison_table(table, delta))
"""

from __future__ import annotations

import copy
import logging
import math
import time
import warnings
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from model_selection.alpha_research import _hac_tstat
from model_selection.alpha_model import assert_pure_alpha_fit_kwargs
from model_selection.training import FeaturePreprocessor, make_training_target

_EPS = 1e-12
_log = logging.getLogger(__name__)


def _safe_clone(model: Any) -> Any:
    """Try sklearn.base.clone (fast, hyperparams-only), fall back to deepcopy."""
    try:
        from sklearn.base import clone
        return clone(model)
    except Exception:
        return copy.deepcopy(model)


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class WalkForwardConfig:
    """Walk-forward evaluation parameters."""

    train_days: int = 252
    test_days: int = 21
    n_windows: int = 8
    min_train_obs: int = 200
    target_col: str = "target_rank"
    raw_return_col: str = "raw_5d"
    avg_roundtrip_cost_bps: float = 15.0
    holding_days: int = 5
    min_cross_section: int = 10
    winsor_q: float = 0.01


# ── Cross-sectional IC helpers ────────────────────────────────────────────────

def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _daily_rank_ic(df: pd.DataFrame, score_col: str, target_col: str) -> list[float]:
    """Return per-date cross-sectional rank IC values."""
    vals: list[float] = []
    for _, g in df[["date", score_col, target_col]].dropna().groupby("date", sort=True):
        if len(g) < 8 or g[score_col].nunique() < 2 or g[target_col].nunique() < 2:
            continue
        corr = float(g[score_col].corr(g[target_col], method="spearman"))
        if np.isfinite(corr):
            vals.append(corr)
    return vals


def _decile_metrics(
    df: pd.DataFrame, score_col: str, target_col: str, *, sign: int = 1
) -> tuple[float, float]:
    """Return (decile_spread, monotonicity) using score for ranking."""
    spreads: list[float] = []
    mono: list[float] = []
    for _, g in df[["date", score_col, target_col]].dropna().groupby("date", sort=True):
        if len(g) < 30 or g[score_col].nunique() < 10:
            continue
        try:
            bucket = pd.qcut(g[score_col].rank(method="first"), 10, labels=False, duplicates="drop")
        except ValueError:
            continue
        by_bucket = g.groupby(bucket, observed=True)[target_col].mean().sort_index()
        if len(by_bucket) < 3:
            continue
        spreads.append(float(sign) * float(by_bucket.iloc[-1] - by_bucket.iloc[0]))
        diffs = np.diff(by_bucket.to_numpy(dtype=float))
        if len(diffs):
            mono.append(float(((float(sign) * diffs) >= 0.0).mean()))
    return (
        float(np.nanmean(spreads)) if spreads else float("nan"),
        float(np.nanmean(mono)) if mono else float("nan"),
    )


def _long_quintile_sharpe(
    df: pd.DataFrame, score_col: str, raw_col: str, *, holding_days: int = 5
) -> float:
    """Annualized Sharpe of top-quintile portfolio (by score) on raw_col returns."""
    period_rets: list[float] = []
    for _, g in df[["date", score_col, raw_col]].dropna().groupby("date", sort=True):
        if len(g) < 10:
            continue
        top = g.loc[g[score_col] >= g[score_col].quantile(0.80), raw_col]
        if len(top) < 2:
            continue
        period_rets.append(float(top.mean()))
    if len(period_rets) < 10:
        return float("nan")
    arr = np.asarray([v for v in period_rets if np.isfinite(v)], dtype=float)
    if len(arr) < 10:
        return float("nan")
    mu, sigma = float(arr.mean()), float(arr.std(ddof=1))
    if sigma < _EPS:
        return float("nan")
    return float((mu / sigma) * math.sqrt(252.0 / max(holding_days, 1)))


def _top_quintile_turnover(df: pd.DataFrame, score_col: str) -> float:
    """
    Mean fraction of top-quintile tickers that change per consecutive date pair.

    Turnover = 1 − Jaccard(top_Q_t, top_Q_{t-1}), averaged over transitions.
    High turnover → more trading → more cost drag on theoretical IC.
    """
    dates = sorted(df["date"].dropna().unique())
    if len(dates) < 2:
        return float("nan")
    prev_top: set[str] = set()
    turnovers: list[float] = []
    for dt in dates:
        g = df.loc[df["date"] == dt, ["ticker", score_col]].dropna()
        if len(g) < 10:
            prev_top = set()
            continue
        threshold = g[score_col].quantile(0.80)
        cur_top = set(g.loc[g[score_col] >= threshold, "ticker"].astype(str))
        if prev_top and cur_top:
            union_size = len(prev_top | cur_top)
            intersect_size = len(prev_top & cur_top)
            turnover = 1.0 - intersect_size / max(union_size, 1)
            turnovers.append(turnover)
        prev_top = cur_top
    return float(np.nanmean(turnovers)) if turnovers else float("nan")


def _alpha_capture_proxy(
    ic_mean: float, turnover: float, avg_cost_bps: float, holding_days: int = 5
) -> float:
    """
    Estimate net alpha efficiency after turnover-driven cost drag.

    Signal return (IC proxy): IC × 252/H days of signal
    TC drag: turnover × avg_roundtrip_cost / holding_period
    alpha_capture_proxy = max(0, 1 − cost_drag/signal_return)

    IC_mean is in rank-correlation units (typically 0.02–0.08 for good signals).
    avg_cost_bps in basis points per round trip.
    """
    if not (math.isfinite(ic_mean) and math.isfinite(turnover)):
        return float("nan")
    signal_proxy = abs(ic_mean) * math.sqrt(252.0 / max(holding_days, 1))
    if signal_proxy < _EPS:
        return 0.0
    cost_drag = turnover * avg_cost_bps / 10_000.0 * (252.0 / max(holding_days, 1))
    return float(max(0.0, 1.0 - cost_drag / signal_proxy))


# ── Date-group utilities for rankers ─────────────────────────────────────────

def _build_date_groups(df: pd.DataFrame) -> np.ndarray:
    """Map rows to integer cross-section indices in date order."""
    dates = pd.to_datetime(df["date"], errors="coerce")
    unique_dates, inverse = np.unique(dates.to_numpy(dtype="datetime64[ns]"), return_inverse=True)
    return inverse.astype(np.int32)


# ── Walk-forward core ─────────────────────────────────────────────────────────

def _get_target_for_model(
    df: pd.DataFrame,
    model_name: str,
    model_kind: str,
    target_col: str,
) -> np.ndarray:
    """
    Return per-row training labels aligned to the model's objective.

    Classifiers use binary target_up; rankers/regressors use target_col.
    The target_col is what the caller specifies (typically target_rank or
    factor_residual_5d from target_construction).
    """
    if model_kind in {"classifier"}:
        if "target_up" in df.columns:
            return df["target_up"].fillna(0).astype(int).to_numpy()
        return (
            pd.to_numeric(df.get(target_col, pd.Series(0, index=df.index)), errors="coerce")
            .fillna(0.0) > 0.0
        ).astype(int).to_numpy()

    if model_kind == "short_classifier":
        if "target_down_decile" in df.columns:
            return df["target_down_decile"].fillna(0).astype(int).to_numpy()
        return (
            pd.to_numeric(df.get(target_col, pd.Series(0, index=df.index)), errors="coerce")
            .fillna(0.0) < 0.0
        ).astype(int).to_numpy()

    # Rankers and regressors: use the caller-specified continuous target
    if target_col in df.columns:
        return pd.to_numeric(df[target_col], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    # Fall back through available targets
    for fallback in ("target_rank", "target_return", "target_return_net"):
        if fallback in df.columns:
            return pd.to_numeric(df[fallback], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    return np.zeros(len(df), dtype=float)


def _score_model(
    model: Any,
    model_kind: str,
    uses_proba: bool,
    x_arr: np.ndarray,
) -> np.ndarray:
    """Convert model output to a continuous score (higher = long-biased)."""
    is_ranker = hasattr(model, "_preloaded_date_groups") or hasattr(model, "set_date_context")
    if model_kind in {"regressor", "long_alpha", "overlay_alpha", "short_alpha"} or is_ranker:
        return np.asarray(model.predict(x_arr), dtype=float)
    if model_kind == "short_classifier":
        if uses_proba and hasattr(model, "predict_proba"):
            return model.predict_proba(x_arr)[:, 1].astype(float) - 0.5
        return model.predict(x_arr).astype(float) - 0.5
    # Classifier (long direction): convert P(up) to a centered score
    if uses_proba and hasattr(model, "predict_proba"):
        return model.predict_proba(x_arr)[:, 1].astype(float) - 0.5
    if hasattr(model, "decision_function"):
        return model.decision_function(x_arr).astype(float)
    return model.predict(x_arr).astype(float) - 0.5


@dataclass(frozen=True)
class _WindowPrep:
    """Preprocessing artifacts shared across all model/target fits in the same window."""

    x_train: np.ndarray
    x_test: np.ndarray
    active_features: tuple[str, ...]
    date_groups: np.ndarray  # integer cross-section indices for rankers


def prepare_window(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    cfg: WalkForwardConfig,
) -> _WindowPrep | None:
    """Precompute preprocessing once per window; share across model and target fits."""
    t0 = time.perf_counter()
    active = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]
    if not active:
        return None
    try:
        preproc = FeaturePreprocessor.fit(train_df, active, winsor_q=cfg.winsor_q)
    except Exception:
        return None
    if not preproc.active_features:
        return None
    x_train = preproc.transform(train_df)
    x_test = preproc.transform(test_df)
    if x_train.shape[0] < cfg.min_train_obs or x_train.shape[1] == 0:
        return None
    date_groups = _build_date_groups(train_df)
    elapsed = time.perf_counter() - t0
    _log.debug(
        "PREP BUILD %.3fs  train=%d  test=%d  feats=%d",
        elapsed, len(train_df), len(test_df), len(preproc.active_features),
    )
    return _WindowPrep(
        x_train=x_train,
        x_test=x_test,
        active_features=tuple(preproc.active_features),
        date_groups=date_groups,
    )


def _fit_one_window(
    model_name: str,
    model: Any,
    model_kind: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    cfg: WalkForwardConfig,
    _precomputed: _WindowPrep | None = None,
) -> pd.DataFrame | None:
    """
    Fit model on train_df, score test_df.

    Returns test_df with a 'score' column, or None on failure.
    Pass _precomputed to skip FeaturePreprocessor.fit/transform (shared across models in one window).
    """
    if _precomputed is not None:
        # HIT: use shared preprocessing artifacts
        x_train = _precomputed.x_train
        x_test = _precomputed.x_test
        date_groups = _precomputed.date_groups
        _log.debug("PREP HIT  model=%s  target=%s", model_name, target_col)
    else:
        # MISS: fit preprocessing for this call only
        active = [c for c in feature_cols if c in train_df.columns and c in test_df.columns]
        if not active:
            return None
        try:
            preproc = FeaturePreprocessor.fit(train_df, active, winsor_q=cfg.winsor_q)
        except Exception:
            return None
        if not preproc.active_features:
            return None
        x_train = preproc.transform(train_df)
        x_test = preproc.transform(test_df)
        date_groups = _build_date_groups(train_df)
        _log.debug("PREP MISS  model=%s  target=%s", model_name, target_col)

    if x_train.shape[0] < cfg.min_train_obs or x_train.shape[1] == 0:
        return None

    y_train = _get_target_for_model(train_df, model_name, model_kind, target_col)
    if len(y_train) != x_train.shape[0]:
        return None

    est = _safe_clone(model)

    # Pass date-group context for rankers
    _needs_groups = hasattr(est, "set_date_context") or hasattr(est, "_preloaded_date_groups")
    if _needs_groups:
        if hasattr(est, "set_date_context"):
            est.set_date_context(date_groups)

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            if _needs_groups:
                fit_kwargs = {"_date_groups": date_groups}
                assert_pure_alpha_fit_kwargs(fit_kwargs)
                est.fit(x_train, y_train, **fit_kwargs)
            else:
                assert_pure_alpha_fit_kwargs({})
                est.fit(x_train, y_train)
    except Exception:
        return None

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            scores = _score_model(est, model_kind, uses_proba=False, x_arr=x_test)
    except Exception:
        return None

    if not np.isfinite(scores).any():
        return None

    out = test_df[["date", "ticker"]].copy()
    if cfg.target_col in test_df.columns:
        out[cfg.target_col] = pd.to_numeric(test_df[cfg.target_col], errors="coerce").to_numpy()
    if cfg.raw_return_col in test_df.columns:
        out[cfg.raw_return_col] = pd.to_numeric(test_df[cfg.raw_return_col], errors="coerce").to_numpy()
    # Preserve any raw_5d even when target_col differs
    for col in ("raw_5d", "target_return", "target_rank"):
        if col in test_df.columns and col not in out.columns:
            out[col] = pd.to_numeric(test_df[col], errors="coerce").to_numpy()
    out["score"] = scores
    return out


# ── Walk-forward orchestration ────────────────────────────────────────────────

def _walk_forward_dates(
    df: pd.DataFrame, cfg: WalkForwardConfig
) -> list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]]:
    """
    Return list of (train_start, train_end, test_start, test_end) date tuples.

    Windows are constructed from the end of the data backward so that the final
    OOS window includes the most recent observations.
    """
    dates = sorted(pd.to_datetime(df["date"], errors="coerce").dropna().unique())
    if not dates:
        return []
    biz_days = pd.bdate_range(dates[0], dates[-1])
    total = len(biz_days)
    windows: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp, pd.Timestamp]] = []
    step = max(1, cfg.test_days)
    for w in range(cfg.n_windows):
        # Anchor test windows from end of available data
        test_end_idx = total - w * step
        test_start_idx = test_end_idx - step
        train_end_idx = test_start_idx
        train_start_idx = max(0, train_end_idx - cfg.train_days)
        if test_start_idx <= 0 or train_end_idx - train_start_idx < cfg.min_train_obs // 5:
            break
        windows.append((
            biz_days[train_start_idx],
            biz_days[min(train_end_idx - 1, total - 1)],
            biz_days[test_start_idx],
            biz_days[min(test_end_idx - 1, total - 1)],
        ))
    return list(reversed(windows))


def _aggregate_window_metrics(
    scored_frames: list[pd.DataFrame],
    cfg: WalkForwardConfig,
) -> dict[str, float]:
    """Aggregate per-window IC series and portfolio metrics into summary stats."""
    if not scored_frames:
        return {k: float("nan") for k in ("ic_mean", "ic_tstat", "ic_ir", "decile_spread",
                                            "monotonicity", "long_sharpe", "turnover", "alpha_cap_proxy")}

    all_scored = pd.concat(scored_frames, ignore_index=True)
    all_scored["date"] = pd.to_datetime(all_scored["date"], errors="coerce")
    all_scored["score"] = pd.to_numeric(all_scored["score"], errors="coerce")

    target_col = cfg.target_col
    raw_col = cfg.raw_return_col

    # Use raw_5d if present; fall back to target_col for IC (should be the same in basic setups)
    ic_target = target_col if target_col in all_scored.columns else "target_rank"
    raw_ret_col = raw_col if raw_col in all_scored.columns else ic_target

    # Per-date rank IC
    ic_vals_list = _daily_rank_ic(all_scored, "score", ic_target)
    ic_arr = np.asarray(ic_vals_list, dtype=float)
    ic_arr = ic_arr[np.isfinite(ic_arr)]

    if len(ic_arr) < 5:
        return {k: float("nan") for k in ("ic_mean", "ic_tstat", "ic_ir", "decile_spread",
                                            "monotonicity", "long_sharpe", "turnover", "alpha_cap_proxy")}

    ic_mean = float(ic_arr.mean())
    ic_std = float(ic_arr.std(ddof=1))
    ic_ir = abs(ic_mean) / ic_std if ic_std > _EPS else float("nan")
    ic_tstat = _hac_tstat(ic_arr)

    spread, mono = _decile_metrics(all_scored, "score", ic_target, sign=1)
    lo_sharpe = _long_quintile_sharpe(all_scored, "score", raw_ret_col, holding_days=cfg.holding_days)
    turnover = _top_quintile_turnover(all_scored, "score")
    ac_proxy = _alpha_capture_proxy(ic_mean, turnover, cfg.avg_roundtrip_cost_bps, cfg.holding_days)

    return {
        "ic_mean": round(ic_mean, 5),
        "ic_tstat": round(ic_tstat, 3) if math.isfinite(ic_tstat) else float("nan"),
        "ic_ir": round(ic_ir, 3) if math.isfinite(ic_ir) else float("nan"),
        "decile_spread": round(spread, 5) if math.isfinite(spread) else float("nan"),
        "monotonicity": round(mono, 3) if math.isfinite(mono) else float("nan"),
        "long_sharpe": round(lo_sharpe, 3) if math.isfinite(lo_sharpe) else float("nan"),
        "turnover": round(turnover, 3) if math.isfinite(turnover) else float("nan"),
        "alpha_cap_proxy": round(ac_proxy, 3) if math.isfinite(ac_proxy) else float("nan"),
    }


def _composite_score(row: pd.Series) -> float:
    """
    Weighted composite for ranking objectives:
      0.35 × IC t-stat   (statistical evidence)
      0.25 × long_sharpe (realized return quality)
      0.20 × alpha_cap_proxy (net efficiency after TC)
      0.15 × monotonicity (cross-sectional structure)
      0.05 × ic_ir       (consistency)
    """
    weights = {
        "ic_tstat": 0.35,
        "long_sharpe": 0.25,
        "alpha_cap_proxy": 0.20,
        "monotonicity": 0.15,
        "ic_ir": 0.05,
    }
    score, total_w = 0.0, 0.0
    for key, w in weights.items():
        v = float(row.get(key, float("nan")))
        if math.isfinite(v):
            score += w * v
            total_w += w
    return score / total_w if total_w > 0.5 else float("nan")


# ── Public API ────────────────────────────────────────────────────────────────

def compare_objectives(
    df: pd.DataFrame,
    models: list[tuple[str, Any, bool, str]],
    feature_cols: list[str],
    *,
    cfg: WalkForwardConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Walk-forward evaluation of all supplied models.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with date, ticker, features, and target columns
        (target_rank, target_return, raw_5d, target_up etc).
    models : list[tuple[name, estimator, uses_proba, model_kind]]
        From build_models().
    feature_cols : list[str]
        Feature columns to use (DO NOT add or remove — these are the fixed features).
    cfg : WalkForwardConfig | None
        Walk-forward parameters. Uses defaults if None.

    Returns
    -------
    table : pd.DataFrame
        Full metrics table, one row per model, sorted by composite_score.
    delta : pd.DataFrame
        Δ vs. classifier baseline — lift / drag for each regression / ranking model.
    """
    wf = cfg or WalkForwardConfig()
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["date", "ticker"]).reset_index(drop=True)

    windows = _walk_forward_dates(df, wf)
    if not windows:
        empty = pd.DataFrame(columns=["model_name", "model_kind", "ic_mean", "ic_tstat",
                                       "ic_ir", "decile_spread", "monotonicity", "long_sharpe",
                                       "turnover", "alpha_cap_proxy", "composite_score"])
        return empty, empty

    rows: list[dict] = []
    for model_name, model, uses_proba, model_kind in models:
        if model_kind == "short_classifier":
            continue  # short models evaluated separately

        scored_frames: list[pd.DataFrame] = []
        for train_start, train_end, test_start, test_end in windows:
            train_df = df[(df["date"] >= train_start) & (df["date"] <= train_end)]
            test_df = df[(df["date"] >= test_start) & (df["date"] <= test_end)]
            if len(train_df) < wf.min_train_obs or test_df.empty:
                continue
            result = _fit_one_window(
                model_name, model, model_kind,
                train_df, test_df, feature_cols, wf.target_col, wf,
            )
            if result is not None and not result.empty:
                scored_frames.append(result)

        metrics = _aggregate_window_metrics(scored_frames, wf)
        metrics["model_name"] = model_name
        metrics["model_kind"] = model_kind
        metrics["n_oos_windows"] = len(scored_frames)
        rows.append(metrics)

    col_order = ["model_name", "model_kind", "n_oos_windows",
                 "ic_mean", "ic_tstat", "ic_ir", "decile_spread",
                 "monotonicity", "long_sharpe", "turnover", "alpha_cap_proxy"]
    table = pd.DataFrame(rows)[col_order]
    table["composite_score"] = table.apply(_composite_score, axis=1)
    table = table.sort_values("composite_score", ascending=False).reset_index(drop=True)
    table.insert(0, "rank", range(1, len(table) + 1))

    # ── Delta vs classifier baseline ──────────────────────────────────────────
    clf_rows = table[table["model_kind"] == "classifier"]
    if not clf_rows.empty:
        baseline = clf_rows.iloc[0]  # best classifier by composite score
        metric_cols = ["ic_mean", "ic_tstat", "ic_ir", "decile_spread",
                       "monotonicity", "long_sharpe", "alpha_cap_proxy", "composite_score"]
        delta_rows: list[dict] = []
        for _, row in table[table["model_kind"] != "classifier"].iterrows():
            dr: dict = {"model_name": row["model_name"], "model_kind": row["model_kind"]}
            for col in metric_cols:
                rv = float(row.get(col, float("nan")))
                bv = float(baseline.get(col, float("nan")))
                dr[f"delta_{col}"] = round(rv - bv, 5) if (math.isfinite(rv) and math.isfinite(bv)) else float("nan")
            dr["improves_over_classifier"] = (
                float(row.get("composite_score", float("nan"))) >
                float(baseline.get("composite_score", float("nan")))
                if math.isfinite(float(row.get("composite_score", float("nan"))))
                   and math.isfinite(float(baseline.get("composite_score", float("nan"))))
                else None
            )
            delta_rows.append(dr)
        delta = pd.DataFrame(delta_rows).reset_index(drop=True)
    else:
        delta = pd.DataFrame(columns=["model_name", "model_kind", "improves_over_classifier"])

    return table, delta


# ── Formatting ────────────────────────────────────────────────────────────────

def _fmt(v: object, width: int = 8, prec: int = 4) -> str:
    if isinstance(v, float) and math.isfinite(v):
        return f"{v:{width}.{prec}f}"
    if isinstance(v, (int, np.integer)):
        return f"{int(v):{width}d}"
    return " " * (width - 3) + "NaN"


def format_comparison_table(table: pd.DataFrame, delta: pd.DataFrame | None = None) -> str:
    """Render the comparison table and optional delta table as a fixed-width report."""
    sep = "=" * 106
    lines = [
        "Model Objective Comparison — Walk-Forward Signal Quality",
        sep,
        f"{'Rk':>2}  {'Model':<22}  {'Kind':<14}  "
        f"{'W':>2}  {'IC Mean':>8}  {'IC t':>7}  {'IC IR':>6}  "
        f"{'D10 Sprd':>8}  {'Mono':>5}  {'LO Sharpe':>9}  "
        f"{'Turnover':>8}  {'α-Cap':>6}  {'Score':>7}",
        "-" * 106,
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{int(row.get('rank', 0)):>2}  "
            f"{str(row['model_name']):<22}  "
            f"{str(row['model_kind']):<14}  "
            f"{_fmt(row.get('n_oos_windows', 0), 2, 0)}  "
            f"{_fmt(row.get('ic_mean'))}  "
            f"{_fmt(row.get('ic_tstat'), 7, 3)}  "
            f"{_fmt(row.get('ic_ir'), 6, 3)}  "
            f"{_fmt(row.get('decile_spread'))}  "
            f"{_fmt(row.get('monotonicity'), 5, 3)}  "
            f"{_fmt(row.get('long_sharpe'), 9, 3)}  "
            f"{_fmt(row.get('turnover'), 8, 3)}  "
            f"{_fmt(row.get('alpha_cap_proxy'), 6, 3)}  "
            f"{_fmt(row.get('composite_score'), 7, 4)}"
        )
    lines += [
        sep,
        "IC Mean      = mean daily cross-sectional rank IC(score, target_rank)",
        "IC t-stat    = Newey–West HAC t-stat on IC series (≥2.0 = significant)",
        "IC IR        = |IC mean| / IC std (consistency; ≥0.5 = stable)",
        "D10 Spread   = mean(top-decile target) − mean(bottom-decile target)",
        "Mono         = fraction of adjacent decile pairs with correct sign",
        "LO Sharpe    = annualized Sharpe — top-quintile portfolio on raw_5d",
        "Turnover     = mean top-quintile member change fraction per period",
        "α-Cap        = IC_mean × (1 − turnover_cost_ratio) — net efficiency",
        "Score        = 0.35×IC-t + 0.25×LO-Sharpe + 0.20×α-Cap + 0.15×Mono + 0.05×IC-IR",
    ]

    if delta is not None and not delta.empty:
        lines += ["", "Δ vs. Best Classifier Baseline", "=" * 80]
        delta_metric_cols = [c for c in delta.columns if c.startswith("delta_")]
        hdr = f"  {'Model':<22}  {'Kind':<14}  " + "  ".join(f"{c.replace('delta_',''):>9}" for c in delta_metric_cols) + "  Beats?"
        lines += [hdr, "-" * 80]
        for _, row in delta.iterrows():
            vals = "  ".join(
                f"{float(row[c]):+9.4f}" if math.isfinite(float(row[c])) else "      NaN"
                for c in delta_metric_cols
            )
            beats = "YES" if row.get("improves_over_classifier") else " NO"
            lines.append(f"  {str(row['model_name']):<22}  {str(row['model_kind']):<14}  {vals}  {beats}")
        lines.append("=" * 80)

    return "\n".join(lines)


def summarize_verdict(table: pd.DataFrame, delta: pd.DataFrame) -> str:
    """
    One-paragraph verdict: do regression/ranking objectives improve over classification?

    Compares the best regression/ranking model vs the best classifier on
    IC t-stat and long_sharpe — the two most policy-relevant metrics.
    """
    clf_rows = table[table["model_kind"] == "classifier"]
    non_clf = table[table["model_kind"] != "classifier"]
    if clf_rows.empty or non_clf.empty:
        return "Insufficient models to compute verdict."

    best_clf = clf_rows.iloc[0]
    best_reg = non_clf.iloc[0]

    def _g(row: pd.Series, col: str) -> float:
        return float(row.get(col, float("nan")))

    ic_lift = _g(best_reg, "ic_tstat") - _g(best_clf, "ic_tstat")
    sharpe_lift = _g(best_reg, "long_sharpe") - _g(best_clf, "long_sharpe")
    turnover_delta = _g(best_reg, "turnover") - _g(best_clf, "turnover")

    verdict = (
        f"Best regression/ranking model: {best_reg['model_name']} ({best_reg['model_kind']})\n"
        f"Best classifier baseline:       {best_clf['model_name']} ({best_clf['model_kind']})\n\n"
    )

    def _sign(v: float) -> str:
        return f"+{v:.3f}" if v >= 0 else f"{v:.3f}"

    verdict += (
        f"IC t-stat lift:    {_sign(ic_lift)}  {'← regression wins' if ic_lift > 0 else '← classifier wins'}\n"
        f"Long-only Sharpe:  {_sign(sharpe_lift)}  {'← regression wins' if sharpe_lift > 0 else '← classifier wins'}\n"
        f"Turnover delta:    {_sign(turnover_delta)}  "
        f"({'lower turnover' if turnover_delta < 0 else 'higher turnover'} for regression)\n\n"
    )

    n_improve = int((delta["improves_over_classifier"] == True).sum()) if not delta.empty else 0  # noqa: E712
    n_total = len(delta)

    if ic_lift > 0.5 and sharpe_lift > 0.1:
        verdict += (
            f"VERDICT: Regression/ranking objectives are clearly superior ({n_improve}/{n_total} models beat baseline). "
            "Replace binary classifiers with continuous targets."
        )
    elif ic_lift > 0.0 or sharpe_lift > 0.0:
        verdict += (
            f"VERDICT: Marginal improvement ({n_improve}/{n_total} beat baseline). "
            "Regression preferred; retain classifier for ensemble diversity."
        )
    else:
        verdict += (
            f"VERDICT: Classifier holds edge on this data ({n_improve}/{n_total} beat baseline). "
            "Investigate signal horizon alignment before switching objectives."
        )

    return verdict
