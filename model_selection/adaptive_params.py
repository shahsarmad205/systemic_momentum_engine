"""Adaptive parameter calibration for institutional-grade model selection.

Three calibrations derived from signal and cost data rather than hardcoded constants:
  1. Kelly gamma        – γ = σ²_cs_daily / (2 × E[IC²])         (Grinold-Kahn)
  2. GP no-trade band   – band ∝ c / predicted_alpha              (Garleanu-Pedersen spirit)
  3. Optimal horizon    – argmax  median_IC(τ) × sqrt(τ)          (IC-decay fit)

Design note — GP band formula choice
--------------------------------------
The literal GP (2013) formula  half_width = c / (2γ × σ²_pos)  uses position-level
variance in dollar space.  Translated to portfolio-weight space at typical equity
vol (σ=2%/day) and transaction cost (c=4bps), this gives half_widths >> 30%, always
hitting the upper bound regardless of IC.  The formula is correct for continuous-time
dollar-denominated positions but loses resolution in the [0%, 3%] band range relevant
to equity model selection.

Instead we use the economically equivalent IC-cost ratio interpretation:
  cost_alpha_ratio = c_roundtrip / max(IC_mean × σ_cs_daily, c_roundtrip)   ∈ [0, 1]
  no_trade_band    = cost_alpha_ratio × max_name_weight × BASE_EFFICIENCY

BASE_EFFICIENCY = 0.05 is calibrated so that at the breakeven IC (where
IC_mean × σ_cs = c_roundtrip), band = max_name_weight × 0.05 = 0.5% (for 10% max
position).  The band tightens quadratically as IC strengthens above breakeven.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Any

# ── Bounds ─────────────────────────────────────────────────────────────────────
_KELLY_GAMMA_LO:   float = 0.5
_KELLY_GAMMA_HI:   float = 8.0
_GP_BAND_WD_LO:    float = 0.001   # 10bp floor  — numerical stability
_GP_BAND_WD_HI:    float = 0.015   # 1.5% ceiling — wider than this, optimizer freezes
_GP_DRIFT_FACTOR:  float = 3.0     # total_drift = factor × weight_diff
_GP_BASE_EFF:      float = 0.05    # see module docstring
_KELLY_MIN_WINDOWS: int  = 2       # warm-up: fall back to config for first N windows


def compute_sigma_cs_daily(train_df: pd.DataFrame) -> float:
    """Mean daily cross-sectional standard deviation of DAILY returns.

    Uses ``daily_return`` column when available; falls back to ``target_return``.
    Returns nan if neither column nor ``date`` is present.
    """
    col = "daily_return" if "daily_return" in train_df.columns else "target_return"
    if col not in train_df.columns or "date" not in train_df.columns:
        return float("nan")
    daily_cs_std = (
        train_df.groupby("date")[col]
        .std(ddof=1)
        .dropna()
    )
    if daily_cs_std.empty:
        return float("nan")
    return float(daily_cs_std.mean())


def compute_sigma_sq_cs(train_df: pd.DataFrame, target_col: str = "target_return") -> float:
    """Mean daily cross-sectional variance of target_col.

    Returns nan if target_col or date not present.
    """
    if target_col not in train_df.columns or "date" not in train_df.columns:
        return float("nan")
    daily_var = train_df.groupby("date")[target_col].var(ddof=1).dropna()
    if daily_var.empty:
        return float("nan")
    return float(daily_var.mean())


def calibrate_kelly_gamma(
    ic_history: list[float],
    train_df: pd.DataFrame,
    target_col: str = "target_return",
    config_gamma: float = 2.0,
) -> float:
    """Return Kelly-optimal turnover penalty γ = σ²_cs_daily / (2 × E[IC²]).

    σ²_cs is computed from daily_return (not target_return) so it measures
    single-day cross-sectional return risk — the correct denominator for Kelly
    when IC is expressed as daily IC (or de-annualised h-day IC).

    Requires at least _KELLY_MIN_WINDOWS prior IC observations for stability.
    Falls back to config_gamma when data are insufficient or E[IC²] ≈ 0.
    Clamped to [_KELLY_GAMMA_LO, _KELLY_GAMMA_HI].
    """
    if len(ic_history) < _KELLY_MIN_WINDOWS:
        return float(config_gamma)

    finite_ics = [v for v in ic_history if np.isfinite(v)]
    if not finite_ics:
        return float(config_gamma)

    e_ic_sq = float(np.mean(np.square(finite_ics)))
    if e_ic_sq < 1e-8:
        return float(config_gamma)

    # Prefer daily_return for σ²_cs; fall back to target_return
    sigma_cs_daily = compute_sigma_cs_daily(train_df)
    if not np.isfinite(sigma_cs_daily) or sigma_cs_daily <= 0:
        # Second fallback: compute from target_return
        sigma_sq_cs = compute_sigma_sq_cs(train_df, target_col)
        if not np.isfinite(sigma_sq_cs) or sigma_sq_cs <= 0:
            return float(config_gamma)
        sigma_sq_cs_daily = sigma_sq_cs
    else:
        sigma_sq_cs_daily = sigma_cs_daily ** 2

    kelly_gamma = sigma_sq_cs_daily / (2.0 * e_ic_sq)
    return float(np.clip(kelly_gamma, _KELLY_GAMMA_LO, _KELLY_GAMMA_HI))


def calibrate_gp_band(
    one_way_cost_bps: float,
    ic_mean: float,
    sigma_cs_daily: float,
    max_name_weight: float = 0.10,
) -> tuple[float, float]:
    """Return (no_trade_band_weight_diff, no_trade_band_total_drift).

    Cost-alpha ratio approach (GP spirit, weight-space adapted):
        predicted_alpha  = |IC_mean| × sigma_cs_daily
        cost_alpha_ratio = c_roundtrip / max(predicted_alpha, c_roundtrip)   ∈ [0, 1]
        weight_diff      = cost_alpha_ratio × max_name_weight × BASE_EFFICIENCY

    Behaviour:
        IC weak  (pred_alpha < c): ratio → 1, band → max_name_weight × BASE_EFFICIENCY
        IC strong (pred_alpha >> c): ratio → 0, band → floor (tight, always trade)

    With BASE_EFFICIENCY = 0.05 and max_name_weight = 0.10:
        At breakeven IC → band ≈ 0.5%
        Strong IC (ratio=0.4) → band ≈ 0.2%
        Very strong IC (ratio→0) → band ≈ 0.1% (floor)
    """
    if not np.isfinite(ic_mean) or not np.isfinite(sigma_cs_daily) or max_name_weight <= 0:
        return (0.005, 0.025)

    round_trip = 2.0 * float(one_way_cost_bps) / 10_000.0
    predicted_alpha = abs(float(ic_mean)) * float(sigma_cs_daily)

    # ratio ∈ [0, 1]: 1 when cost ≥ alpha (signal covers costs), 0 when alpha >> cost
    cost_alpha_ratio = round_trip / max(predicted_alpha, round_trip) if round_trip > 0 else 1.0

    weight_diff = float(np.clip(
        cost_alpha_ratio * max_name_weight * _GP_BASE_EFF,
        _GP_BAND_WD_LO,
        _GP_BAND_WD_HI,
    ))
    total_drift = float(weight_diff * _GP_DRIFT_FACTOR)
    return weight_diff, total_drift


def optimal_horizon_from_decay(
    alpha_decay: pd.DataFrame,
    candidate_horizons: list[int],
    target_type: str = "net_residual_return",
    *,
    min_features_per_horizon: int = 2,
    min_median_ic: float = 0.002,
    tie_tolerance_pct: float = 10.0,
) -> tuple[int, dict[str, Any]]:
    """Select horizon maximising median_IC(τ) × sqrt(τ) across positive-IC features.

    The IC-IR proxy  median_IC(τ) × sqrt(τ)  encodes the Grinold information ratio:
    longer horizons have fewer non-overlapping observations (÷ sqrt(τ)), so a longer
    horizon only wins if its per-period IC is proportionally higher.

    P16 institutional guardrails:
      - min_features_per_horizon : require at least N features with positive IC
      - min_median_ic            : selected horizon must have meaningful IC
      - tie_tolerance_pct        : prefer shorter horizon within X% of the best score
        (shorter horizons mean more non-overlapping observations and faster iteration)

    candidate_horizons must overlap with horizons present in alpha_decay.
    Returns (optimal_horizon, diagnostics_dict).
    """
    default_h = int(candidate_horizons[0]) if candidate_horizons else 10
    diag_base: dict[str, Any] = {"candidate_horizons": candidate_horizons}

    if alpha_decay is None or alpha_decay.empty:
        return default_h, {**diag_base, "reason": "empty_decay_table"}

    ic_col = "daily_spearman_ic"
    h_col = "horizon_days"
    type_col = "target_type"

    if ic_col not in alpha_decay.columns or h_col not in alpha_decay.columns:
        return default_h, {**diag_base, "reason": "missing_columns"}

    if type_col in alpha_decay.columns and target_type in alpha_decay[type_col].unique():
        subset = alpha_decay[alpha_decay[type_col] == target_type]
    else:
        subset = alpha_decay

    horizon_stats: dict[int, dict[str, float]] = {}
    for h in candidate_horizons:
        h_rows = subset[subset[h_col] == int(h)]
        if h_rows.empty:
            continue
        ics = pd.to_numeric(h_rows[ic_col], errors="coerce").dropna()
        positive_ics = ics[ics > 0]
        n_positive = len(positive_ics)
        if n_positive < min_features_per_horizon:
            continue
        median_ic = float(positive_ics.median())
        mean_ic = float(positive_ics.mean())
        n_features = len(ics)
        if np.isfinite(median_ic) and median_ic >= min_median_ic:
            horizon_stats[h] = {
                "score": float(median_ic * np.sqrt(float(h))),
                "median_ic": round(median_ic, 6),
                "mean_ic": round(mean_ic, 6),
                "n_positive_features": n_positive,
                "n_total_features": n_features,
            }

    if not horizon_stats:
        return default_h, {
            **diag_base,
            "reason": "no_horizon_passes_guardrails",
            "min_features_required": min_features_per_horizon,
            "min_median_ic_required": min_median_ic,
        }

    # P16: Tie-breaking — prefer shorter horizon within tolerance
    # Shorter horizons → more non-overlapping observations → faster research iteration.
    # A 20d horizon is only selected if it's >10% better than 10d (or 5d).
    scores = {h: s["score"] for h, s in horizon_stats.items()}
    best_h = max(scores, key=lambda h: scores[h])
    best_score = scores[best_h]

    # Check if a shorter horizon is within tolerance
    shorter_contenders = {
        h: s for h, s in scores.items()
        if h < best_h
        and abs(s - best_score) / max(abs(best_score), 1e-9) * 100 < tie_tolerance_pct
    }
    if shorter_contenders:
        best_h = min(shorter_contenders)  # Prefer the shortest within tolerance
        best_score = scores[best_h]
        tie_reason = f"tie_broken_to_shorter_within_{tie_tolerance_pct}pct"
    else:
        tie_reason = "clear_winner"

    diag: dict[str, Any] = {
        **diag_base,
        "reason": f"ic_decay_argmax_{tie_reason}",
        "scores": {h: round(v, 6) for h, v in sorted(scores.items())},
        "horizon_details": horizon_stats,
        "best_horizon": int(best_h),
        "best_ir_proxy": round(best_score, 6),
        "guardrail": {
            "min_features_per_horizon": min_features_per_horizon,
            "min_median_ic": min_median_ic,
            "tie_tolerance_pct": tie_tolerance_pct,
            "tie_decision": tie_reason,
        },
    }
    return int(best_h), diag


# ═══════════════════════════════════════════════════════════════════════════════
# P34: Execution-Aware Horizon Frontier — advisory diagnostic
# ═══════════════════════════════════════════════════════════════════════════════
# Builds a multi-metric frontier table from the IC-decay table so researchers
# can see which horizons are IC-strong vs execution-viable BEFORE models are
# trained.  All metrics are computed from already-available data (IC-decay table).
# Metrics that require model training (turnover, alpha capture, exec Sharpe) are
# explicitly marked 'not_yet_available' rather than silently omitted.
#
# Scoring weights are loaded from YAML config (execution_aware_horizon section)
# and sum-normalised at runtime.  The frontier is advisory — it does NOT change
# the selected horizon unless --auto-horizon or adaptive_horizon.apply is active.

_METRIC_NOT_AVAILABLE = "not_yet_available"


@dataclass(frozen=True)
class ExecutionAwareHorizonConfig:
    """P34: Config for execution-aware horizon frontier scoring."""

    enabled: bool = True
    candidate_horizons: tuple[int, ...] = (5, 10, 20, 40, 60, 63)
    target_type: str = "net_residual_return"

    # Scoring weights — sum-normalised at runtime
    weight_ic_strength: float = 0.25
    weight_ic_consistency: float = 0.15
    weight_halflife_persistence: float = 0.20
    weight_cost_adjusted_ic: float = 0.20
    weight_alpha_capture: float = 0.10
    weight_execution_sharpe: float = 0.10

    # Guardrails
    min_features_per_horizon: int = 2
    min_median_ic: float = 0.002

    # Persistence formula: expected rank survival = 2^(-rebalance / halflife)
    halflife_persistence_threshold: float = 0.30

    # Rebalance frequencies to evaluate per horizon (for rebalance frontier)
    candidate_rebalance_frequencies: tuple[int, ...] = (2, 3, 5, 10, 20, 63)

    # Max rebalance / horizon ratio
    max_rebalance_to_horizon_ratio: float = 1.0

    # Output paths
    horizon_frontier_path: str = "output/models/horizon_frontier.csv"
    rebalance_frontier_path: str = "output/models/rebalance_frontier.csv"
    report_path: str = "output/models/execution_aware_horizon_report.txt"


def execution_aware_horizon_frontier(
    alpha_decay: pd.DataFrame,
    *,
    cfg: ExecutionAwareHorizonConfig,
    rebalance_frequency_days: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """P34: Build execution-aware horizon frontier from IC-decay evidence.

    Computes all available metrics per horizon and composite score. Metrics that
    require model training (turnover, alpha capture, exec Sharpe) are marked as
    'not_yet_available' — they will be populated after ExecutionValidation.
    The frontier does NOT change the selected horizon.

    rebalance_frequency_days: when provided, rank persistence is computed at
    the actual rebalance frequency rather than at the horizon.  This unifies
    the halflife/persistence contract so that persistence_at_rebalance always
    reflects the execution contract, not the research horizon.

    Returns:
        frontier_df  — one row per candidate horizon with all metrics
        diagnostics  — scoring metadata, flag summaries, recommendations
    """
    ic_col = "daily_spearman_ic"
    h_col = "horizon_days"
    type_col = "target_type"
    halflife_col = "signal_halflife_days"
    tstat_col = "daily_spearman_ic_tstat"
    ic_std_col = "daily_spearman_ic_std"
    feature_col = "feature"

    rows: list[dict[str, Any]] = []
    diagnostics: dict[str, Any] = {
        "config": {
            "candidate_horizons": list(cfg.candidate_horizons),
            "target_type": cfg.target_type,
            "weights": {k: getattr(cfg, k) for k in dir(cfg) if k.startswith("weight_")},
        },
        "flags": [],
        "recommendations": [],
    }

    if alpha_decay is None or alpha_decay.empty:
        diagnostics["flags"].append("empty_decay_table")
        return pd.DataFrame(rows), diagnostics

    required = {ic_col, h_col}
    if not required.issubset(alpha_decay.columns):
        diagnostics["flags"].append("missing_columns")
        return pd.DataFrame(rows), diagnostics

    # Filter to target type if present
    if type_col in alpha_decay.columns and cfg.target_type in alpha_decay[type_col].unique():
        subset = alpha_decay[alpha_decay[type_col] == cfg.target_type]
    else:
        subset = alpha_decay

    # ── Per-horizon metrics from IC-decay table ────────────────────────────
    horizon_metrics: dict[int, dict[str, float]] = {}
    for h in cfg.candidate_horizons:
        h_rows = subset[subset[h_col] == int(h)]
        if h_rows.empty:
            continue
        ics = pd.to_numeric(h_rows[ic_col], errors="coerce").dropna()
        positive_ics = ics[ics > 0]
        n_positive = len(positive_ics)
        if n_positive < cfg.min_features_per_horizon:
            continue
        median_ic = float(positive_ics.median())
        if not np.isfinite(median_ic) or median_ic < cfg.min_median_ic:
            continue

        mean_ic = float(ics.mean())
        n_features = len(ics)

        # IC consistency: Information Ratio = mean / std across features
        ic_ir = mean_ic / float(ics.std(ddof=1)) if len(ics) > 1 and ics.std(ddof=1) > 1e-12 else float("nan")

        # IC t-stat: mean per-feature tstat
        tstats = pd.to_numeric(h_rows[tstat_col], errors="coerce").dropna() if tstat_col in h_rows.columns else pd.Series(dtype=float)
        ic_tstat = float(tstats.mean()) if len(tstats) > 0 else float("nan")

        # Signal halflife: median per-feature halflife at this horizon
        halflives = pd.to_numeric(h_rows[halflife_col], errors="coerce").dropna() if halflife_col in h_rows.columns else pd.Series(dtype=float)
        halflife = float(halflives.median()) if len(halflives) > 0 and np.isfinite(halflives.median()) else float("nan")

        # Cost-adjusted IC: net_residual target already embeds costs at alpha research level
        # For the frontier, we use the net_residual IC as the cost-adjusted proxy.
        # Post-model cost-adjusted IC requires execution validation — marked NA.
        cost_ic = mean_ic  # net_residual return is already net-of-costs

        # Rank persistence at rebalance (halflife-based estimate)
        # P39: Use actual rebalance frequency, not horizon, to compute
        # persistence.  This unifies the halflife/persistence contract so
        # that persistence_at_rebalance always reflects the execution contract.
        _rebal_for_persistence = rebalance_frequency_days if rebalance_frequency_days is not None else int(h)
        rank_persistence = float("nan")
        if np.isfinite(halflife) and halflife > 0:
            rank_persistence = 2.0 ** (-float(_rebal_for_persistence) / halflife)

        # IC-decay based IR proxy (Grinold: IR ≈ IC × √breadth)
        breadth = max(1.0, 252.0 / max(1, float(h)))
        ic_ir_proxy = median_ic * np.sqrt(float(h))

        m = {
            "horizon_days": int(h),
            "n_features": n_features,
            "n_positive_features": n_positive,
            "median_ic": round(median_ic, 6),
            "mean_ic": round(mean_ic, 6),
            "ic_ir": round(ic_ir, 4) if np.isfinite(ic_ir) else float("nan"),
            "ic_tstat": round(ic_tstat, 4) if np.isfinite(ic_tstat) else float("nan"),
            "signal_halflife_days": round(halflife, 2) if np.isfinite(halflife) else float("nan"),
            "rank_persistence_at_rebalance": round(rank_persistence, 4) if np.isfinite(rank_persistence) else float("nan"),
            "cost_adjusted_ic": round(cost_ic, 6),
            "ic_decay_ir_proxy": round(ic_ir_proxy, 6),

            # Metrics not yet available — require model training
            "avg_turnover": _METRIC_NOT_AVAILABLE,
            "alpha_capture": _METRIC_NOT_AVAILABLE,
            "gross_alpha": _METRIC_NOT_AVAILABLE,
            "execution_cost_pnl": _METRIC_NOT_AVAILABLE,
            "net_execution_sharpe": _METRIC_NOT_AVAILABLE,
            "psr": _METRIC_NOT_AVAILABLE,
            "dsr": _METRIC_NOT_AVAILABLE,
            "score_direction_stability": _METRIC_NOT_AVAILABLE,

            # Persistence flag
            "halflife_persistence_ok": None,
        }

        if np.isfinite(rank_persistence):
            m["halflife_persistence_ok"] = rank_persistence >= cfg.halflife_persistence_threshold
        else:
            m["halflife_persistence_ok"] = None  # unknown — no halflife evidence

        horizon_metrics[h] = m
        rows.append(m)

    frontier = pd.DataFrame(rows)

    # ── Composite scoring from available metrics ──────────────────────────
    if not frontier.empty:
        # Normalise each metric to [0, 1] across horizons
        frontiers_scored = frontier.copy()
        _norm_columns = [
            ("median_ic", cfg.weight_ic_strength, True),
            ("ic_ir", cfg.weight_ic_consistency, True),
            ("cost_adjusted_ic", cfg.weight_cost_adjusted_ic, True),
        ]
        for col, weight, higher_better in _norm_columns:
            vals = pd.to_numeric(frontiers_scored[col], errors="coerce")
            finite = vals.dropna()
            if len(finite) < 2:
                frontiers_scored[f"{col}_norm"] = 0.0
                continue
            mn, mx = finite.min(), finite.max()
            rng = mx - mn
            if rng < 1e-12:
                frontiers_scored[f"{col}_norm"] = 0.5
            elif higher_better:
                frontiers_scored[f"{col}_norm"] = (vals - mn) / rng
            else:
                frontiers_scored[f"{col}_norm"] = (mx - vals) / rng

        # Halflife persistence: use as multiplier if available, else neutral
        pers_vals = pd.to_numeric(frontiers_scored["rank_persistence_at_rebalance"], errors="coerce")
        pers_valid = pers_vals[pers_vals.notna()]
        if len(pers_valid) >= 1:
            pers_norm = (pers_vals - pers_valid.min()) / max(pers_valid.max() - pers_valid.min(), 1e-12)
        else:
            pers_norm = pd.Series(0.5, index=frontiers_scored.index)

        # Composite: available weights only, sum-normalised
        total_w = cfg.weight_ic_strength + cfg.weight_ic_consistency + cfg.weight_cost_adjusted_ic + cfg.weight_halflife_persistence
        total_w = max(total_w, 1e-12)

        frontiers_scored["composite_score"] = (
            (cfg.weight_ic_strength / total_w) * frontiers_scored["median_ic_norm"]
            + (cfg.weight_ic_consistency / total_w) * frontiers_scored["ic_ir_norm"]
            + (cfg.weight_cost_adjusted_ic / total_w) * frontiers_scored["cost_adjusted_ic_norm"]
            + (cfg.weight_halflife_persistence / total_w) * pers_norm
        )
        frontiers_scored = frontiers_scored.drop(columns=[c for c in frontiers_scored.columns if c.endswith("_norm")], errors="ignore")
        frontiers_scored = frontiers_scored.sort_values("composite_score", ascending=False)
        frontier = frontiers_scored

        # Best by composite
        best = frontier.iloc[0]
        diagnostics["best_horizon_composite"] = int(best["horizon_days"])
        diagnostics["best_composite_score"] = round(float(best["composite_score"]), 6)

        # Persistence flags
        high_ic_low_persistence = frontier[
            (frontier["halflife_persistence_ok"] == False)
        ]
        if not high_ic_low_persistence.empty:
            flagged_h = [int(h) for h in high_ic_low_persistence["horizon_days"]]
            diagnostics["flags"].append(f"high_ic_low_persistence_horizons: {flagged_h}")
            diagnostics["recommendations"].append(
                f"Horizon(s) {flagged_h} have strong IC but rank persistence "
                f"< {cfg.halflife_persistence_threshold:.0%} at rebalance. "
                "Consider faster rebalance or longer-halflife features."
            )

    return frontier, diagnostics


def compute_rebalance_frontier(
    alpha_decay: pd.DataFrame,
    *,
    cfg: ExecutionAwareHorizonConfig,
) -> pd.DataFrame:
    """P34: Build rebalance frequency frontier for each target horizon.

    For each target horizon in candidate_horizons, evaluate each rebalance
    frequency from candidate_rebalance_frequencies (clamped by max_ratio).
    Computes halflife-based rank persistence at each rebalance frequency.
    Returns a DataFrame with rows: (horizon, rebalance_freq, rank_persistence).
    """
    ic_col = "daily_spearman_ic"
    h_col = "horizon_days"
    type_col = "target_type"
    halflife_col = "signal_halflife_days"

    rows: list[dict[str, Any]] = []
    if alpha_decay is None or alpha_decay.empty:
        return pd.DataFrame(rows)

    if type_col in alpha_decay.columns and cfg.target_type in alpha_decay[type_col].unique():
        subset = alpha_decay[alpha_decay[type_col] == cfg.target_type]
    else:
        subset = alpha_decay

    for target_h in cfg.candidate_horizons:
        h_rows = subset[subset[h_col] == int(target_h)]
        if h_rows.empty:
            continue
        halflives = pd.to_numeric(h_rows[halflife_col], errors="coerce").dropna() if halflife_col in h_rows.columns else pd.Series(dtype=float)
        median_halflife = float(halflives.median()) if len(halflives) > 0 and np.isfinite(halflives.median()) else float("nan")

        for reb_freq in cfg.candidate_rebalance_frequencies:
            if reb_freq > target_h * cfg.max_rebalance_to_horizon_ratio:
                continue
            persistence = float("nan")
            if np.isfinite(median_halflife) and median_halflife > 0:
                persistence = 2.0 ** (-float(reb_freq) / median_halflife)

            rows.append({
                "target_horizon_days": int(target_h),
                "rebalance_frequency_days": int(reb_freq),
                "median_signal_halflife_days": round(median_halflife, 2) if np.isfinite(median_halflife) else float("nan"),
                "rank_persistence_at_rebalance": round(persistence, 4) if np.isfinite(persistence) else float("nan"),
                "persistence_ok": bool(persistence >= cfg.halflife_persistence_threshold) if np.isfinite(persistence) else None,
            })

    return pd.DataFrame(rows)
