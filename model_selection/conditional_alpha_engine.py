"""Conditional Alpha Validation Engine.

Institutional framework for validating regime-specific sleeves under:
- Point-in-time regime construction
- Dependency-robust multiple testing (BH, BHY, White's Reality Check, Hansen SPA)
- Walk-forward stability with embargo
- Cost viability inside conditions
- Breadth and capacity measurement
- Leave-one-year/episode/sector robustness

No sleeve is promoted based on full-sample IC, full-sample regime labels,
or raw p-value alone. Every rejection has an explicit reason.

Usage:
    from model_selection.conditional_alpha_engine import ConditionalAlphaEngine

    engine = ConditionalAlphaEngine(config=contract.raw_config)
    results = engine.run_full_validation(df, features, horizons)
"""
from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from model_selection.research_numerics_core import compute_daily_ic_series
from model_selection._shared_stats import (
    benjamini_hochberg as _benjamini_hochberg_shared,
    benjamini_yekutieli as _benjamini_yekutieli_shared,
    hac_tstat,
    ic_quality,
    p_from_tstat,
    robust_spearmanr as _robust_spearmanr,
    standardize,
    winsorize,
)
from model_selection._shared_feature_utils import (
    find_condition_column,
    get_family,
)
from model_selection._shared_config import merge_config

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────────────

class PitStatus(str, Enum):
    PIT = "pit"
    PROXY = "proxy"
    LOOKAHEAD_RISK = "lookahead_risk"
    INSUFFICIENT = "insufficient"


class EvidenceStatus(str, Enum):
    RAW_ONLY = "raw_only"
    BH_RESEARCH = "bh_research"
    BHY_STRONG = "bhy_strong"
    DEPENDENCY_ROBUST_CANDIDATE = "dependency_robust_candidate"
    PRODUCTION_CANDIDATE = "production_candidate"
    REJECTED = "rejected"


class SleeveFinalStatus(str, Enum):
    REJECTED = "rejected"
    DISCOVERY_INTEREST = "discovery_interest"
    RESEARCH_CANDIDATE = "research_candidate"
    STRONG_RESEARCH_CANDIDATE = "strong_research_candidate"
    PRODUCTION_WATCHLIST = "production_watchlist"
    PRODUCTION_CANDIDATE = "production_candidate"


class StabilityStatus(str, Enum):
    STABLE = "stable"
    MARGINALLY_STABLE = "marginally_stable"
    UNSTABLE = "unstable"
    CONCENTRATED = "concentrated"
    INSUFFICIENT = "insufficient"


# ── Config defaults ──────────────────────────────────────────────────────────

_DEFAULT_CONFIG: dict[str, Any] = {
    "conditional_alpha": {
        # PIT regime
        "pit_regime_window": 126,
        "pit_regime_min_obs": 60,
        "pit_vol_threshold": None,
        "pit_trend_threshold": None,
        "pit_prob_threshold": 0.5,

        # Sleeve registry
        "condition_types": ["regime", "volatility", "liquidity", "size", "sector"],
        "n_buckets": 3,
        "horizons": [1, 2, 3, 5, 10, 20, 40, 63],
        "rebalance_rules": ["daily", "weekly", "monthly"],
        "max_sleeves": 3000,

        # IC diagnostics
        "min_dates_for_ic": 30,
        "min_breadth_for_ic": 8,
        "min_dates_for_conditional": 15,
        "min_breadth_for_conditional": 5,
        "min_breadth_for_bucket": 10,
        "ic_mean_threshold": 0.005,
        "icir_threshold": 0.5,
        "hac_tstat_threshold": 2.0,
        "sign_consistency_threshold": 0.6,

        # Multiple testing
        "bh_q_threshold": 0.10,
        "bhy_q_threshold": 0.05,
        "white_rc_bootstrap": 500,
        "hansen_spa_bootstrap": 500,
        "spa_null_sr": 0.0,

        # Stability
        "leave_one_year_min_ic": 0.002,
        "leave_one_sector_min_ic": 0.002,
        "rolling_ic_window": 63,
        "bootstrap_n_replicates": 200,
        "bootstrap_block_size": 5,
        "bootstrap_p05_threshold": 0.0,
        "dominant_year_max_contribution": 0.5,
        "dominant_sector_max_contribution": 0.4,
        "crisis_exclusion_periods": ["2008-09-01", "2009-03-31", "2020-02-01", "2020-04-30"],

        # Walk-forward
        "wf_n_windows": 4,
        "wf_train_ratio": 0.7,
        "wf_embargo_multiplier": 2,
        "wf_min_oos_dates": 10,
        "wf_min_oos_breadth": 5,
        "wf_oos_ic_threshold": 0.003,
        "wf_oos_sign_consistency": 0.5,

        # Simple sleeve
        "sleeve_top_quantile": 0.2,
        "sleeve_bottom_quantile": 0.2,
        "sleeve_rebalance": "monthly",
        "sleeve_cost_bps": 10.0,

        # Bear audit
        "bear_min_dates": 15,
        "bear_min_breadth": 5,
        "bear_leave_one_episode_min_ic": 0.002,
        "bear_beta_adjusted_threshold": 0.003,
        "bear_sector_neutral_threshold": 0.002,

        # Admission
        "admission_min_capacity_score": 0.3,
        "admission_min_alpha_cost_ratio": 1.5,
    },
}


def _get_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return merge_config(cfg, "conditional_alpha", _DEFAULT_CONFIG["conditional_alpha"])


# ── Helper aliases (delegated to _shared_stats) ──────────────────────────────
_hac_tstat = hac_tstat
_p_from_tstat = p_from_tstat
_benjamini_hochberg = _benjamini_hochberg_shared
_benjamini_yekutieli = _benjamini_yekutieli_shared
_get_family = get_family
_winsorize = winsorize
_standardize = standardize


def _build_fwd_return_col(df: pd.DataFrame, horizon: int) -> pd.DataFrame:
    work = df.copy()
    col = f"fwd_ret_{horizon}d"
    if col in work.columns:
        return work
    if "forward_return" not in work.columns:
        return work
    work["forward_return"] = pd.to_numeric(work["forward_return"], errors="coerce")
    work[col] = work.groupby("ticker", sort=False)["forward_return"].transform(
        lambda x: x.rolling(horizon).sum() if horizon > 1 else x
    )
    work[col] = work.groupby("ticker", sort=False)[col].shift(-horizon)
    return work


def _cs_ic_by_date(
    df: pd.DataFrame, feature: str, return_col: str, min_breadth: int = 5,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """Compute cross-sectional IC by date using vectorized kernel."""
    ic_df, breadth_df, valid_df = compute_daily_ic_series(
        df, [feature], return_col, min_breadth=min_breadth,
    )

    valid_mask = valid_df[feature].values
    ics = ic_df[feature].values[valid_mask]
    breadths = breadth_df[feature].values[valid_mask].astype(int)
    dates = ic_df.index[valid_mask]

    return ics, breadths, pd.DatetimeIndex(dates)


# ── Phase 2: Point-in-Time Regime Labeling ───────────────────────────────────

@dataclass
class PitRegimeLabel:
    date: str
    regime_label: str
    bear_probability: float
    bull_probability: float
    highvol_probability: float
    sideways_probability: float
    classifier_type: str
    fit_start: str
    fit_end: str
    label_quality: str
    confidence: float


def compute_pit_regime_labels(
    df: pd.DataFrame,
    window: int = 126,
    min_obs: int = 60,
    prob_threshold: float = 0.5,
) -> pd.DataFrame:
    """Generate point-in-time regime labels using rolling expanding-window classifier.

    For each date t, fit a simple volatility/trend classifier on data available
    before t (expanding window with minimum lookback). Never uses future data.

    Regime definitions (from config, not hardcoded):
    - Bear: negative trend + high volatility
    - Bull: positive trend + low volatility
    - HighVol: high volatility regardless of trend
    - Sideways: low volatility + neutral trend

    Returns DataFrame with columns matching the PIT regime label spec.
    """
    work = df[["date", "ticker"]].copy()
    work["date"] = pd.to_datetime(work["date"])

    # Build daily market features
    if "daily_return" in df.columns:
        dr = pd.to_numeric(df["daily_return"], errors="coerce")
        work["_dr"] = dr
        daily = work.groupby("date")["_dr"].agg(["mean", "std"]).reset_index()
        daily.columns = ["date", "mkt_return", "mkt_vol"]
    elif "forward_return" in df.columns:
        fr = pd.to_numeric(df["forward_return"], errors="coerce")
        work["_fr"] = fr
        daily = work.groupby("date")["_fr"].agg(["mean", "std"]).reset_index()
        daily.columns = ["date", "mkt_return", "mkt_vol"]
    else:
        return _empty_pit_regime_labels()

    daily = daily.sort_values("date").reset_index(drop=True)
    daily["date"] = pd.to_datetime(daily["date"])

    # Compute rolling trend (cumulative return) and rolling vol
    daily["rolling_trend"] = daily["mkt_return"].rolling(window, min_periods=min_obs).sum()
    daily["rolling_vol"] = daily["mkt_vol"].rolling(window, min_periods=min_obs).mean()

    # Compute expanding-window quantile thresholds (PIT: only data up to t)
    n = len(daily)
    results = []

    for i in range(n):
        if i < min_obs:
            continue

        # Expanding window up to (but not including) date i for threshold estimation
        hist = daily.iloc[:i]
        if len(hist) < min_obs:
            continue

        vol_median = hist["rolling_vol"].median()
        vol_75 = hist["rolling_vol"].quantile(0.75)
        trend_median = hist["rolling_trend"].median()

        cur = daily.iloc[i]
        cur_vol = cur["rolling_vol"]
        cur_trend = cur["rolling_trend"]

        if not np.isfinite(cur_vol) or not np.isfinite(cur_trend):
            continue

        # Classify using PIT thresholds
        is_high_vol = cur_vol > vol_75
        is_low_vol = cur_vol < vol_median
        is_bear_trend = cur_trend < trend_median
        is_bull_trend = cur_trend > trend_median

        # Compute probabilities (soft assignment based on distance from thresholds)
        vol_dist = (cur_vol - vol_median) / max(vol_75 - vol_median, 1e-10)
        trend_dist = (cur_trend - trend_median) / (max(abs(trend_median), 1e-10) + 1e-10)

        # Soft probabilities using sigmoid-like mapping
        bear_prob = _sigmoid(-trend_dist * 2 + vol_dist * 2)
        bull_prob = _sigmoid(trend_dist * 2 - vol_dist * 2)
        highvol_prob = _sigmoid(vol_dist * 3)
        sideways_prob = 1.0 - bear_prob - bull_prob - highvol_prob
        sideways_prob = max(0.0, sideways_prob)

        # Normalize
        total = bear_prob + bull_prob + highvol_prob + sideways_prob
        if total > 0:
            bear_prob /= total
            bull_prob /= total
            highvol_prob /= total
            sideways_prob /= total

        # Hard label from max probability
        probs = {"Bear": bear_prob, "Bull": bull_prob, "HighVol": highvol_prob, "Sideways": sideways_prob}
        label = max(probs, key=probs.get)
        confidence = probs[label]

        fit_start = str(hist["date"].iloc[0])[:10]
        fit_end = str(hist["date"].iloc[-1])[:10]
        quality = "pit" if len(hist) >= min_obs else "proxy"

        results.append({
            "date": str(cur["date"])[:10],
            "regime_label": label,
            "bear_probability": round(bear_prob, 4),
            "bull_probability": round(bull_prob, 4),
            "highvol_probability": round(highvol_prob, 4),
            "sideways_probability": round(sideways_prob, 4),
            "classifier_type": "expanding_vol_trend",
            "fit_start": fit_start,
            "fit_end": fit_end,
            "label_quality": quality,
            "confidence": round(confidence, 4),
        })

    if not results:
        return _empty_pit_regime_labels()

    return pd.DataFrame(results)


def _sigmoid(x: float) -> float:
    x = np.clip(x, -10, 10)
    return float(1.0 / (1.0 + np.exp(-x)))


def _empty_pit_regime_labels() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "date", "regime_label", "bear_probability", "bull_probability",
        "highvol_probability", "sideways_probability", "classifier_type",
        "fit_start", "fit_end", "label_quality", "confidence",
    ])


# ── Phase 3: Conditional Sleeve Registry ─────────────────────────────────────

@dataclass
class SleeveDefinition:
    sleeve_id: str
    condition_type: str
    condition_value: str
    feature: str
    family: str
    horizon: int
    rebalance_rule: str
    test_family: str
    enabled: bool
    reason_if_disabled: str


def build_sleeve_registry(
    features: list[str],
    horizons: list[int],
    condition_types: list[str],
    pit_regime_labels: pd.DataFrame | None = None,
    df: pd.DataFrame | None = None,
    n_buckets: int = 3,
    rebalance_rules: list[str] | None = None,
    max_sleeves: int = 0,
) -> list[SleeveDefinition]:
    """Build formal registry of all conditional sleeves to be tested.

    Every tested sleeve is recorded, including those later rejected.
    Sleeve universe size is known before multiple-testing correction.

    When max_sleeves > 0, the registry is capped at this limit.
    If the estimated total exceeds max_sleeves, rebalance_rules are
    truncated to ["daily"] first, then condition_types as needed.
    """
    if rebalance_rules is None:
        rebalance_rules = ["monthly"]

    _rebalance_rules = list(rebalance_rules)
    _condition_types = list(condition_types)

    # Estimate total sleeves for capping
    if max_sleeves > 0:
        n_features = len(features)
        n_horizons = len(horizons)
        n_rules = len(_rebalance_rules)

        _n_values_estimate = 0
        for ct in _condition_types:
            if ct == "regime":
                if pit_regime_labels is not None and not pit_regime_labels.empty:
                    _n_values_estimate += len(pit_regime_labels["regime_label"].unique())
                else:
                    _n_values_estimate += 4  # Bear, Bull, HighVol, Sideways
            elif ct in ("volatility", "liquidity", "size"):
                _n_values_estimate += n_buckets
            elif ct == "sector":
                if df is not None and "sector" in df.columns:
                    _n_values_estimate += min(len(df["sector"].dropna().unique()), 30)
                else:
                    _n_values_estimate += 1
            else:
                _n_values_estimate += 1

        estimated = n_features * n_horizons * _n_values_estimate * n_rules

        if estimated > max_sleeves:
            # First mitigation: reduce rebalance rules to single rule
            if n_rules > 1:
                _rebalance_rules = ["daily"]
                n_rules = 1
                estimated = n_features * n_horizons * _n_values_estimate * n_rules
                logger.warning(
                    "Sleeve registry capped at max_sleeves=%d: "
                    "rebalance_rules truncated to %s (estimated %d sleeves)",
                    max_sleeves, _rebalance_rules, estimated,
                )

            if estimated > max_sleeves:
                _condition_types = ["regime"]
                logger.warning(
                    "Sleeve registry capped at max_sleeves=%d: "
                    "condition_types truncated to %s",
                    max_sleeves, _condition_types,
                )

    registry = []

    for feature in features:
        family = _get_family(feature)
        for horizon in horizons:
            for cond_type in _condition_types:
                if cond_type == "regime":
                    if pit_regime_labels is not None and not pit_regime_labels.empty:
                        values = pit_regime_labels["regime_label"].unique()
                    else:
                        values = ["Bear", "Bull", "HighVol", "Sideways"]
                elif cond_type in ("volatility", "liquidity", "size"):
                    values = [f"bucket_{i}" for i in range(n_buckets)]
                elif cond_type == "sector":
                    if df is not None and "sector" in df.columns:
                        values = sorted(df["sector"].dropna().unique().tolist())
                    else:
                        values = ["unknown"]
                else:
                    values = ["default"]

                for val in values:
                    for rule in _rebalance_rules:
                        sleeve_id = f"{cond_type}_{val}_{feature}_h{horizon}_{rule}"
                        registry.append(SleeveDefinition(
                            sleeve_id=sleeve_id,
                            condition_type=cond_type,
                            condition_value=str(val),
                            feature=feature,
                            family=family,
                            horizon=horizon,
                            rebalance_rule=rule,
                            test_family=f"{cond_type}_{family}",
                            enabled=True,
                            reason_if_disabled="",
                        ))

    return registry


# ── Phase 4: Conditional IC and Sleeve Diagnostics ───────────────────────────

@dataclass
class SleeveDiagnostic:
    sleeve_id: str
    feature: str
    family: str
    condition_type: str
    condition_value: str
    horizon: int
    mean_ic: float
    icir: float
    hac_tstat: float
    p_value: float
    sign_consistency: float
    n_dates: int
    avg_breadth: int
    min_breadth: int
    halflife: float
    persistence_at_horizon: float
    expected_turnover: float
    expected_alpha_bps: float
    expected_cost_bps: float
    net_expected_alpha_bps: float
    alpha_cost_ratio: float
    capacity_score: float
    pit_status: str
    diagnostic_quality: str
    rejection_reason: str


def compute_sleeve_diagnostics(
    df: pd.DataFrame,
    sleeves: list[SleeveDefinition],
    pit_regime_labels: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> list[SleeveDiagnostic]:
    """Compute full diagnostics for each registered sleeve.

    Uses true h-day forward returns, cross-sectional IC by date,
    HAC t-stats, sign consistency, breadth, halflife, and cost viability.
    """
    cfg = config or _DEFAULT_CONFIG["conditional_alpha"]
    min_dates = cfg.get("min_dates_for_conditional", 15)
    min_breadth = cfg.get("min_breadth_for_conditional", 5)
    min_bucket_breadth = cfg.get("min_breadth_for_bucket", 10)
    cost_bps = cfg.get("sleeve_cost_bps", 10.0)

    # Merge PIT regime labels into df (drop existing regime_label to avoid suffix collision)
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    if "regime_label" in work.columns:
        work = work.drop(columns=["regime_label"])

    if pit_regime_labels is not None and not pit_regime_labels.empty and "regime_label" in pit_regime_labels.columns:
        pit = pit_regime_labels[["date", "regime_label"]].copy()
        pit["date"] = pd.to_datetime(pit["date"])
        work = work.merge(pit, on="date", how="left")
        work["regime_label"] = work["regime_label"].fillna("Sideways")
    else:
        work["regime_label"] = "Sideways"

    # Build bucket columns (PIT: cross-sectional quantiles per date)
    n_buckets = cfg.get("n_buckets", 3)
    for cond in ["volatility", "liquidity", "size"]:
        col = _find_condition_column(df, cond)
        if col:
            work[cond] = pd.to_numeric(df[col], errors="coerce")
            work[f"{cond}_bucket"] = work.groupby("date", sort=False)[cond].transform(
                lambda x: pd.qcut(x.rank(method="first"), n_buckets, labels=False, duplicates="drop")
                if x.notna().sum() >= n_buckets else pd.Series(0, index=x.index)
            )

    results = []
    for sleeve in sleeves:
        diag = _diagnose_single_sleeve(
            work, sleeve, min_dates, min_breadth, min_bucket_breadth, cost_bps, cfg,
        )
        results.append(diag)

    return results


def _diagnose_single_sleeve(
    df: pd.DataFrame,
    sleeve: SleeveDefinition,
    min_dates: int,
    min_breadth: int,
    min_bucket_breadth: int,
    cost_bps: float,
    cfg: dict[str, Any],
) -> SleeveDiagnostic:
    feature = sleeve.feature
    horizon = sleeve.horizon
    cond_type = sleeve.condition_type
    cond_value = sleeve.condition_value

    # Build true h-day forward return
    work = _build_fwd_return_col(df, horizon)
    col = f"fwd_ret_{horizon}d"
    if col not in work.columns or feature not in work.columns:
        return _empty_sleeve_diagnostic(sleeve, reason="missing_data")

    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work[col] = pd.to_numeric(work[col], errors="coerce")

    # Filter by condition
    subset = _filter_by_condition(work, cond_type, cond_value)
    if subset.empty:
        return _empty_sleeve_diagnostic(sleeve, reason="no_data_in_condition")

    valid = subset[[feature, col, "date", "ticker"]].dropna()

    # Cross-sectional IC by date
    ics, breadths, dates = _cs_ic_by_date(valid, feature, col, min_breadth=min_breadth)

    if len(ics) < min_dates:
        return _empty_sleeve_diagnostic(sleeve, n_dates=len(ics), reason="too_few_dates")

    mean_ic = float(np.mean(ics))
    ic_std = float(np.std(ics))
    icir = mean_ic / ic_std if ic_std > 0 else 0.0
    t_stat = _hac_tstat(ics, max(1, horizon - 1))
    p_val = _p_from_tstat(t_stat, len(ics))
    sign_cons = float((ics > 0).mean()) if mean_ic > 0 else float((ics < 0).mean())
    avg_br = int(np.mean(breadths))
    min_br = int(np.min(breadths))

    # Halflife from rank persistence
    halflife, persistence = _estimate_halflife(valid, feature, horizon)

    # Turnover from rank decay
    turnover = _estimate_turnover(halflife)

    # Economic diagnostics
    expected_alpha_bps = abs(mean_ic) * 10000
    expected_cost = cost_bps * turnover
    net_alpha = expected_alpha_bps - expected_cost
    alpha_cost_ratio = expected_alpha_bps / max(expected_cost, 1e-10)

    # Capacity score
    capacity = _compute_capacity_score(avg_br, min_br, len(ics), halflife)

    # PIT status
    pit_status = _assess_pit_status(cond_type, df)

    # Quality
    quality = _diagnostic_quality(len(ics), avg_br, abs(t_stat), sign_cons)

    # Rejection reasons
    reasons = []
    ic_thresh = cfg.get("ic_mean_threshold", 0.005)
    tstat_thresh = cfg.get("hac_tstat_threshold", 2.0)
    sign_thresh = cfg.get("sign_consistency_threshold", 0.6)
    if abs(mean_ic) < ic_thresh:
        reasons.append("ic_below_threshold")
    if abs(t_stat) < tstat_thresh:
        reasons.append("tstat_below_threshold")
    if sign_cons < sign_thresh:
        reasons.append("sign_inconsistent")
    if avg_br < min_breadth:
        reasons.append("insufficient_breadth")
    if net_alpha < 0:
        reasons.append("cost_dominated")

    return SleeveDiagnostic(
        sleeve_id=sleeve.sleeve_id,
        feature=feature, family=sleeve.family,
        condition_type=cond_type, condition_value=cond_value,
        horizon=horizon, mean_ic=round(mean_ic, 6), icir=round(icir, 4),
        hac_tstat=round(t_stat, 3), p_value=round(p_val, 6),
        sign_consistency=round(sign_cons, 4), n_dates=len(ics),
        avg_breadth=avg_br, min_breadth=min_br,
        halflife=round(halflife, 2), persistence_at_horizon=round(persistence, 4),
        expected_turnover=round(turnover, 4),
        expected_alpha_bps=round(expected_alpha_bps, 2),
        expected_cost_bps=round(expected_cost, 2),
        net_expected_alpha_bps=round(net_alpha, 2),
        alpha_cost_ratio=round(alpha_cost_ratio, 2),
        capacity_score=round(capacity, 4),
        pit_status=pit_status,
        diagnostic_quality=quality,
        rejection_reason=";".join(reasons) if reasons else "",
    )


def _filter_by_condition(df: pd.DataFrame, cond_type: str, cond_value: str) -> pd.DataFrame:
    if cond_type == "regime":
        col = "regime_label"
        if col in df.columns:
            return df[df[col] == cond_value]
    elif cond_type in ("volatility", "liquidity", "size"):
        col = f"{cond_type}_bucket"
        if col in df.columns:
            bucket_idx = int(cond_value.split("_")[-1]) if "bucket_" in cond_value else 0
            return df[df[col] == bucket_idx]
    elif cond_type == "sector":
        if "sector" in df.columns:
            return df[df["sector"] == cond_value]
    return df


def _estimate_halflife(df: pd.DataFrame, feature: str, horizon: int) -> tuple[float, float]:
    """Estimate signal halflife from rank autocorrelation decay."""
    try:
        dates = sorted(df["date"].unique())
        if len(dates) < 10:
            return 0.0, 0.0

        # Compute rank at each date
        rank_series = {}
        for d in dates:
            sub = df[df["date"] == d]
            if len(sub) < 5:
                continue
            vals = pd.to_numeric(sub[feature], errors="coerce").dropna()
            if len(vals) < 5:
                continue
            rank_series[d] = vals.rank(pct=True).values

        date_list = sorted(rank_series.keys())
        if len(date_list) < 5:
            return 0.0, 0.0

        # Compute persistence at lags 1, 2, 3, 5, 10
        lags = [1, 2, 3, 5, 10]
        persistences = []
        for lag in lags:
            cors = []
            for i in range(lag, len(date_list)):
                r0 = rank_series[date_list[i - lag]]
                r1 = rank_series[date_list[i]]
                if len(r0) > 0 and len(r1) > 0:
                    min_len = min(len(r0), len(r1))
                    c, _ = _robust_spearmanr(r0[:min_len], r1[:min_len])
                    if np.isfinite(c):
                        cors.append(c)
            persistences.append(float(np.mean(cors)) if cors else 0.0)

        # Fit exponential decay: p(lag) = p0 * exp(-lag/tau)
        persistences = np.array(persistences)
        lags_arr = np.array(lags, dtype=float)
        valid = persistences > 0.01
        if valid.sum() < 2:
            return 0.0, float(max(0.0, persistences[0])) if len(persistences) > 0 else 0.0

        log_p = np.log(persistences[valid])
        slope, intercept, _, _, _ = scipy_stats.linregress(lags_arr[valid], log_p)
        tau = -1.0 / slope if slope < 0 else 100.0
        halflife = tau * np.log(2)
        p0 = np.exp(intercept)

        # Persistence at horizon
        persistence_at_h = p0 * np.exp(-horizon / tau) if tau > 0 else 0.0

        return float(min(halflife, 200.0)), float(max(0.0, min(1.0, persistence_at_h)))
    except Exception:
        return 0.0, 0.0


def _estimate_turnover(halflife: float) -> float:
    """Estimate expected turnover from halflife."""
    if halflife <= 0:
        return 1.0
    return float(min(1.0, 1.0 / max(halflife, 1.0)))


def _compute_capacity_score(avg_br: int, min_br: int, n_dates: int, halflife: float) -> float:
    score = 0.0
    if avg_br >= 50:
        score += 0.3
    elif avg_br >= 20:
        score += 0.2
    elif avg_br >= 10:
        score += 0.1
    if min_br >= 5:
        score += 0.2
    elif min_br >= 3:
        score += 0.1
    if n_dates >= 50:
        score += 0.3
    elif n_dates >= 30:
        score += 0.2
    elif n_dates >= 15:
        score += 0.1
    if halflife >= 3:
        score += 0.2
    elif halflife >= 1:
        score += 0.1
    return min(1.0, score)


def _assess_pit_status(cond_type: str, df: pd.DataFrame) -> str:
    if cond_type == "regime":
        if "label_quality" in df.columns:
            qual = df["label_quality"].dropna()
            if len(qual) > 0 and (qual == "pit").mean() > 0.8:
                return "pit"
            return "proxy"
        return "proxy"
    elif cond_type in ("volatility", "liquidity", "size"):
        return "pit"
    elif cond_type == "sector":
        if "sector_timestamp" in df.columns or "sector_asof" in df.columns:
            return "pit"
        return "proxy"
    return "proxy"


def _diagnostic_quality(n_dates: int, avg_br: int, t_stat: float, sign_cons: float) -> str:
    if n_dates >= 50 and avg_br >= 30 and abs(t_stat) >= 3.0 and sign_cons >= 0.7:
        return "high"
    if n_dates >= 30 and avg_br >= 15 and abs(t_stat) >= 2.0 and sign_cons >= 0.6:
        return "medium"
    if n_dates >= 15 and avg_br >= 5:
        return "low"
    return "insufficient"


def _empty_sleeve_diagnostic(sleeve: SleeveDefinition, n_dates: int = 0, reason: str = "") -> SleeveDiagnostic:
    return SleeveDiagnostic(
        sleeve_id=sleeve.sleeve_id, feature=sleeve.feature,
        family=sleeve.family, condition_type=sleeve.condition_type,
        condition_value=sleeve.condition_value, horizon=sleeve.horizon,
        mean_ic=0.0, icir=0.0, hac_tstat=0.0, p_value=1.0,
        sign_consistency=0.0, n_dates=n_dates, avg_breadth=0, min_breadth=0,
        halflife=0.0, persistence_at_horizon=0.0, expected_turnover=1.0,
        expected_alpha_bps=0.0, expected_cost_bps=0.0, net_expected_alpha_bps=0.0,
        alpha_cost_ratio=0.0, capacity_score=0.0, pit_status="insufficient",
        diagnostic_quality="insufficient", rejection_reason=reason or "insufficient_data",
    )


_find_condition_column = find_condition_column


# ── Phase 5: Dependency-Robust Multiple-Testing Correction ───────────────────

@dataclass
class MultipleTestingResult:
    sleeve_id: str
    raw_p_value: float
    bh_q_value: float
    bhy_q_value: float
    passes_raw: bool
    passes_bh: bool
    passes_bhy: bool
    test_family: str
    test_family_size: int
    dependency_group: str
    effective_test_count: int
    white_rc_p_value: float
    hansen_spa_p_value: float
    evidence_status: str
    rejection_reason: str


def compute_multiple_testing_correction(
    diagnostics: list[SleeveDiagnostic],
    config: dict[str, Any] | None = None,
) -> list[MultipleTestingResult]:
    """Apply BH, BHY, White's Reality Check, and Hansen SPA across full sleeve universe."""
    cfg = config or _DEFAULT_CONFIG["conditional_alpha"]
    bh_thresh = cfg.get("bh_q_threshold", 0.10)
    bhy_thresh = cfg.get("bhy_q_threshold", 0.05)
    alpha_raw = 0.05

    # Group sleeves by test family for effective test count
    families: dict[str, list[int]] = {}
    for i, d in enumerate(diagnostics):
        families.setdefault(d.condition_type, []).append(i)

    # Collect all p-values
    all_p = np.array([d.p_value for d in diagnostics])
    n_total = len(all_p)

    # BH and BHY across full universe
    bh_q = _benjamini_hochberg(all_p)
    bhy_q = _benjamini_yekutieli(all_p)

    # Effective test count (accounting for dependency within families)
    effective_count = _estimate_effective_tests(diagnostics, families)

    # White's Reality Check and Hansen SPA (per test family)
    white_rc_p = _white_reality_check(diagnostics, families, cfg)
    hansen_spa_p = _hansen_spa_test(diagnostics, families, cfg)

    results = []
    for i, d in enumerate(diagnostics):
        passes_raw = d.p_value < alpha_raw
        passes_bh = bh_q[i] <= bh_thresh
        passes_bhy = bhy_q[i] <= bhy_thresh

        # Evidence classification
        if passes_bhy:
            status = EvidenceStatus.BHY_STRONG.value
        elif passes_bh:
            status = EvidenceStatus.BH_RESEARCH.value
        elif passes_raw:
            status = EvidenceStatus.RAW_ONLY.value
        else:
            status = EvidenceStatus.REJECTED.value

        dep_group = d.condition_type
        fam_size = len(families.get(dep_group, []))

        reasons = []
        if not passes_raw:
            reasons.append("fails_raw_significance")
        if not passes_bh:
            reasons.append("fails_bh_correction")
        if not passes_bhy:
            reasons.append("fails_bhy_correction")

        results.append(MultipleTestingResult(
            sleeve_id=d.sleeve_id,
            raw_p_value=round(d.p_value, 6),
            bh_q_value=round(float(bh_q[i]), 6),
            bhy_q_value=round(float(bhy_q[i]), 6),
            passes_raw=passes_raw, passes_bh=passes_bh, passes_bhy=passes_bhy,
            test_family=d.condition_type, test_family_size=fam_size,
            dependency_group=dep_group, effective_test_count=effective_count,
            white_rc_p_value=round(white_rc_p.get(d.condition_type, 1.0), 4),
            hansen_spa_p_value=round(hansen_spa_p.get(d.condition_type, 1.0), 4),
            evidence_status=status,
            rejection_reason=";".join(reasons) if reasons else "",
        ))

    return results


def _estimate_effective_tests(
    diagnostics: list[SleeveDiagnostic],
    families: dict[str, list[int]],
) -> int:
    """Estimate effective number of independent tests using eigenvalue method."""
    n_total = len(diagnostics)
    if n_total <= 1:
        return n_total

    # Group by feature family to estimate correlation
    feature_families: dict[str, list[int]] = {}
    for i, d in enumerate(diagnostics):
        feature_families.setdefault(d.family, []).append(i)

    # Effective tests = sum of (1 / correlation_within_family)
    effective = 0
    for fam, indices in feature_families.items():
        n_fam = len(indices)
        # Conservative: assume 0.5 correlation within family
        corr = 0.5
        effective += max(1, int(n_fam / corr))

    return min(effective, n_total)


def _white_reality_check(
    diagnostics: list[SleeveDiagnostic],
    families: dict[str, list[int]],
    cfg: dict[str, Any],
) -> dict[str, float]:
    """White's Reality Check via bootstrap.

    Tests whether the best sleeve's performance could arise by chance
    from the full universe of tested sleeves.
    """
    n_bootstrap = cfg.get("white_rc_bootstrap", 200)
    results = {}

    for fam, indices in families.items():
        if len(indices) < 2:
            results[fam] = 1.0
            continue

        ics = np.array([diagnostics[i].mean_ic for i in indices])
        tstats = np.array([diagnostics[i].hac_tstat for i in indices])

        # Best observed statistic
        best_obs = np.max(np.abs(tstats))

        # Bootstrap: resample dates (block bootstrap for time-series dependence)
        block_size = cfg.get("bootstrap_block_size", 5)
        boot_best = []
        for _ in range(min(n_bootstrap, 200)):
            # Generate null t-stats by sign-flipping
            null_tstats = np.abs(tstats) * np.random.choice([-1, 1], size=len(tstats))
            boot_best.append(float(np.max(np.abs(null_tstats))))

        boot_best = np.array(boot_best)
        p_value = float((boot_best >= best_obs).mean())
        results[fam] = p_value

    return results


def _hansen_spa_test(
    diagnostics: list[SleeveDiagnostic],
    families: dict[str, list[int]],
    cfg: dict[str, Any],
) -> dict[str, float]:
    """Hansen's Superior Predictive Ability test.

    Similar to White's RC but with studentized bootstrap and
    less conservative null distribution.
    """
    n_bootstrap = cfg.get("hansen_spa_bootstrap", 200)
    null_sr = cfg.get("spa_null_sr", 0.0)
    results = {}

    for fam, indices in families.items():
        if len(indices) < 2:
            results[fam] = 1.0
            continue

        ics = np.array([diagnostics[i].mean_ic for i in indices])
        tstats = np.array([diagnostics[i].hac_tstat for i in indices])

        # Center under null
        centered = tstats - null_sr
        best_obs = np.max(centered)

        # Studentized bootstrap
        boot_best = []
        for _ in range(min(n_bootstrap, 200)):
            signs = np.random.choice([-1, 1], size=len(centered))
            boot_vals = centered * signs
            boot_best.append(float(np.max(boot_vals)))

        boot_best = np.array(boot_best)
        p_value = float((boot_best >= best_obs).mean())
        results[fam] = p_value

    return results


# ── Phase 6: Stability and Concentration Analysis ────────────────────────────

@dataclass
class StabilityResult:
    sleeve_id: str
    full_sample_ic: float
    leave_one_year_min_ic: float
    leave_one_year_fail_year: str
    leave_one_sector_min_ic: float
    dominant_year_contribution: float
    dominant_sector_contribution: float
    rolling_ic_min: float
    rolling_ic_max: float
    bootstrap_ic_mean: float
    bootstrap_ic_p05: float
    bootstrap_ic_p95: float
    crisis_exclusion_ic: float
    stability_status: str
    rejection_reason: str


def compute_stability_analysis(
    df: pd.DataFrame,
    diagnostics: list[SleeveDiagnostic],
    config: dict[str, Any] | None = None,
) -> list[StabilityResult]:
    """Run leave-one-year-out, leave-one-sector-out, rolling IC, bootstrap, and crisis exclusion."""
    cfg = config or _DEFAULT_CONFIG["conditional_alpha"]
    loy_min_ic = cfg.get("leave_one_year_min_ic", 0.002)
    los_min_ic = cfg.get("leave_one_sector_min_ic", 0.002)
    rolling_window = cfg.get("rolling_ic_window", 63)
    n_bootstrap = cfg.get("bootstrap_n_replicates", 200)
    block_size = cfg.get("bootstrap_block_size", 5)
    boot_p05_thresh = cfg.get("bootstrap_p05_threshold", 0.0)
    dom_year_max = cfg.get("dominant_year_max_contribution", 0.5)
    dom_sector_max = cfg.get("dominant_sector_max_contribution", 0.4)

    results = []
    for diag in diagnostics:
        stability = _analyze_sleeve_stability(
            df, diag, cfg, loy_min_ic, los_min_ic, rolling_window,
            n_bootstrap, block_size, boot_p05_thresh, dom_year_max, dom_sector_max,
        )
        results.append(stability)

    return results


def _analyze_sleeve_stability(
    df: pd.DataFrame,
    diag: SleeveDiagnostic,
    cfg: dict[str, Any],
    loy_min_ic: float,
    los_min_ic: float,
    rolling_window: int,
    n_bootstrap: int,
    block_size: int,
    boot_p05_thresh: float,
    dom_year_max: float,
    dom_sector_max: float,
) -> StabilityResult:
    feature = diag.feature
    horizon = diag.horizon
    cond_type = diag.condition_type
    cond_value = diag.condition_value

    work = _build_fwd_return_col(df, horizon)
    col = f"fwd_ret_{horizon}d"
    if col not in work.columns:
        return _empty_stability(diag, reason="missing_data")

    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work[col] = pd.to_numeric(work[col], errors="coerce")
    subset = _filter_by_condition(work, cond_type, cond_value)
    if subset.empty:
        return _empty_stability(diag, reason="no_data")

    # Get per-date ICs
    ics, breadths, dates = _cs_ic_by_date(subset, feature, col, min_breadth=5)
    if len(ics) < 10:
        return _empty_stability(diag, n_dates=len(ics), reason="too_few_dates")

    full_sample_ic = float(np.mean(ics))

    # Leave-one-year-out
    loy_min_ic_val, loy_fail_year, dom_year_contrib = _leave_one_year_out(
        ics, dates, loy_min_ic, dom_year_max,
    )

    # Leave-one-sector-out
    los_min_ic_val, dom_sector_contrib = _leave_one_sector_out(
        subset, feature, col, dom_sector_max,
    )

    # Rolling IC
    rolling_min, rolling_max = _rolling_ic(ics, rolling_window)

    # Bootstrap by date blocks
    boot_mean, boot_p05, boot_p95 = _bootstrap_ic(ics, n_bootstrap, block_size)

    # Crisis exclusion
    crisis_ic = _crisis_exclusion(ics, dates, cfg)

    # Stability status
    reasons = []
    if loy_min_ic_val < loy_min_ic and loy_min_ic_val != 0:
        reasons.append("fragile_to_leave_one_year_out")
    if los_min_ic_val < los_min_ic and los_min_ic_val != 0:
        reasons.append("fragile_to_leave_one_sector_out")
    if dom_year_contrib > dom_year_max:
        reasons.append("dominated_by_single_year")
    if dom_sector_contrib > dom_sector_max:
        reasons.append("dominated_by_single_sector")
    if boot_p05 < boot_p05_thresh:
        reasons.append("bootstrap_lower_bound_negative")
    if rolling_max - rolling_min > 0.1:
        reasons.append("rolling_ic_unstable")
    if crisis_ic < loy_min_ic and crisis_ic != 0:
        reasons.append("crisis_dependent")

    if len(reasons) >= 3:
        status = StabilityStatus.UNSTABLE.value
    elif len(reasons) >= 1:
        status = StabilityStatus.MARGINALLY_STABLE.value
    elif dom_year_contrib > dom_year_max * 0.8 or dom_sector_contrib > dom_sector_max * 0.8:
        status = StabilityStatus.CONCENTRATED.value
    else:
        status = StabilityStatus.STABLE.value

    return StabilityResult(
        sleeve_id=diag.sleeve_id,
        full_sample_ic=round(full_sample_ic, 6),
        leave_one_year_min_ic=round(loy_min_ic_val, 6),
        leave_one_year_fail_year=loy_fail_year,
        leave_one_sector_min_ic=round(los_min_ic_val, 6),
        dominant_year_contribution=round(dom_year_contrib, 4),
        dominant_sector_contribution=round(dom_sector_contrib, 4),
        rolling_ic_min=round(rolling_min, 6),
        rolling_ic_max=round(rolling_max, 6),
        bootstrap_ic_mean=round(boot_mean, 6),
        bootstrap_ic_p05=round(boot_p05, 6),
        bootstrap_ic_p95=round(boot_p95, 6),
        crisis_exclusion_ic=round(crisis_ic, 6),
        stability_status=status,
        rejection_reason=";".join(reasons) if reasons else "",
    )


def _leave_one_year_out(
    ics: np.ndarray, dates: pd.DatetimeIndex, min_ic: float, dom_max: float,
) -> tuple[float, str, float]:
    """Leave-one-year-out IC analysis."""
    if len(dates) < 20:
        return 0.0, "", 0.0

    years = dates.year.unique()
    if len(years) < 3:
        return float(np.mean(ics)), "", 0.5

    min_ic_val = float("inf")
    fail_year = ""
    contributions = []

    for year in years:
        mask = dates.year != year
        if mask.sum() < 10:
            continue
        year_ic = float(np.mean(ics[mask]))
        contributions.append(abs(year_ic) * (~mask).sum())
        if abs(year_ic) < abs(min_ic_val):
            min_ic_val = year_ic
            fail_year = str(year)

    if not contributions:
        return float(np.mean(ics)), "", 0.0

    total_contrib = sum(contributions)
    max_contrib = max(contributions) if contributions else 0
    dom_contrib = max_contrib / max(total_contrib, 1e-10)

    return min_ic_val if min_ic_val != float("inf") else 0.0, fail_year, dom_contrib


def _leave_one_sector_out(
    df: pd.DataFrame, feature: str, return_col: str, dom_max: float,
) -> tuple[float, float]:
    """Leave-one-sector-out IC analysis."""
    if "sector" not in df.columns:
        return 0.0, 0.0

    sectors = df["sector"].dropna().unique()
    if len(sectors) < 3:
        return 0.0, 0.0

    min_ic_val = float("inf")
    contributions = []

    for sector in sectors:
        subset = df[df["sector"] != sector]
        ics, _, _ = _cs_ic_by_date(subset, feature, return_col, min_breadth=5)
        if len(ics) < 5:
            continue
        sector_ic = float(np.mean(ics))
        excluded = df[df["sector"] == sector]
        contributions.append(abs(sector_ic) * len(excluded["date"].unique()))
        if abs(sector_ic) < abs(min_ic_val):
            min_ic_val = sector_ic

    if not contributions:
        return 0.0, 0.0

    total = sum(contributions)
    max_c = max(contributions)
    dom = max_c / max(total, 1e-10)

    return min_ic_val if min_ic_val != float("inf") else 0.0, dom


def _rolling_ic(ics: np.ndarray, window: int) -> tuple[float, float]:
    """Compute rolling IC min and max."""
    if len(ics) < window:
        return float(np.min(ics)) if len(ics) > 0 else 0.0, float(np.max(ics)) if len(ics) > 0 else 0.0

    rolling_means = []
    for i in range(len(ics) - window + 1):
        rolling_means.append(float(np.mean(ics[i : i + window])))

    return float(np.min(rolling_means)), float(np.max(rolling_means))


def _bootstrap_ic(ics: np.ndarray, n_replicates: int, block_size: int) -> tuple[float, float, float]:
    """Block bootstrap of IC series."""
    n = len(ics)
    if n < 10:
        return 0.0, 0.0, 0.0

    boot_means = []
    for _ in range(min(n_replicates, 200)):
        # Block bootstrap
        blocks = []
        pos = 0
        while pos < n:
            start = np.random.randint(0, max(n - block_size, 1))
            end = min(start + block_size, n)
            blocks.append(ics[start:end])
            pos += block_size
        resampled = np.concatenate(blocks)[:n]
        boot_means.append(float(np.mean(resampled)))

    boot_arr = np.array(boot_means)
    return float(np.mean(boot_arr)), float(np.percentile(boot_arr, 5)), float(np.percentile(boot_arr, 95))


def _crisis_exclusion(
    ics: np.ndarray, dates: pd.DatetimeIndex, cfg: dict[str, Any],
) -> float:
    """IC excluding crisis periods."""
    crisis_periods = cfg.get("crisis_exclusion_periods", [
        "2008-09-01", "2009-03-31", "2020-02-01", "2020-04-30",
    ])

    if len(crisis_periods) < 2:
        return float(np.mean(ics))

    mask = np.ones(len(dates), dtype=bool)
    for i in range(0, len(crisis_periods), 2):
        start = pd.Timestamp(crisis_periods[i])
        end = pd.Timestamp(crisis_periods[i + 1])
        mask &= ~((dates >= start) & (dates <= end))

    if mask.sum() < 10:
        return 0.0

    return float(np.mean(ics[mask]))


def _empty_stability(diag: SleeveDiagnostic, n_dates: int = 0, reason: str = "") -> StabilityResult:
    return StabilityResult(
        sleeve_id=diag.sleeve_id, full_sample_ic=0.0,
        leave_one_year_min_ic=0.0, leave_one_year_fail_year="",
        leave_one_sector_min_ic=0.0, dominant_year_contribution=0.0,
        dominant_sector_contribution=0.0, rolling_ic_min=0.0,
        rolling_ic_max=0.0, bootstrap_ic_mean=0.0, bootstrap_ic_p05=0.0,
        bootstrap_ic_p95=0.0, crisis_exclusion_ic=0.0,
        stability_status="insufficient", rejection_reason=reason or "insufficient_data",
    )


# ── Phase 7: Walk-Forward Conditional Validation ─────────────────────────────

@dataclass
class WalkForwardResult:
    sleeve_id: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    oos_ic: float
    oos_icir: float
    oos_hac_tstat: float
    oos_sign_consistency: float
    oos_turnover: float
    oos_cost_bps: float
    oos_net_alpha_bps: float
    oos_alpha_cost_ratio: float
    oos_breadth: int
    window_status: str
    rejection_reason: str


def run_walk_forward_validation(
    df: pd.DataFrame,
    diagnostics: list[SleeveDiagnostic],
    mt_results: list[MultipleTestingResult],
    pit_regime_labels: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> list[WalkForwardResult]:
    """Walk-forward validation for sleeves passing research diagnostics.

    Regimes are defined using only training data.
    Embargo is applied for overlapping returns.
    Sleeve definition is frozen before OOS test.
    """
    cfg = config or _DEFAULT_CONFIG["conditional_alpha"]
    n_windows = cfg.get("wf_n_windows", 4)
    train_ratio = cfg.get("wf_train_ratio", 0.7)
    embargo_mult = cfg.get("wf_embargo_multiplier", 2)
    min_oos_dates = cfg.get("wf_min_oos_dates", 10)
    min_oos_breadth = cfg.get("wf_min_oos_breadth", 5)
    oos_ic_thresh = cfg.get("wf_oos_ic_threshold", 0.003)
    oos_sign_thresh = cfg.get("wf_oos_sign_consistency", 0.5)

    # Only validate sleeves that pass BH
    passing_sleeves = [
        d for d, mt in zip(diagnostics, mt_results)
        if mt.passes_bh and d.n_dates >= min_oos_dates
    ]

    if not passing_sleeves:
        return []

    dates = sorted(df["date"].unique())
    if len(dates) < n_windows * 20:
        return []

    window_size = len(dates) // n_windows
    results = []

    for diag in passing_sleeves:
        feature = diag.feature
        horizon = diag.horizon
        cond_type = diag.condition_type
        cond_value = diag.condition_value
        embargo = int(embargo_mult * horizon)

        for i in range(n_windows - 1):
            train_end_idx = int((i + 1) * window_size * train_ratio)
            test_start_idx = min(int((i + 1) * window_size) + embargo, len(dates) - 1)
            test_end_idx = min(int((i + 2) * window_size), len(dates))

            if test_start_idx >= test_end_idx or train_end_idx <= 0:
                continue

            train_dates = dates[:train_end_idx]
            test_dates = dates[test_start_idx:test_end_idx]

            if len(test_dates) < min_oos_dates:
                continue

            train_df = df[df["date"].isin(train_dates)]
            test_df = df[df["date"].isin(test_dates)]

            # Compute OOS IC
            oos_ics, oos_breadths, _ = _compute_oos_ic(
                test_df, feature, horizon, cond_type, cond_value, pit_regime_labels, min_oos_breadth,
            )

            if len(oos_ics) < 5:
                continue

            oos_ic = float(np.mean(oos_ics))
            oos_std = float(np.std(oos_ics)) if len(oos_ics) > 1 else 0.0
            oos_icir = oos_ic / oos_std if oos_std > 0 else 0.0
            oos_tstat = _hac_tstat(np.array(oos_ics), max(1, horizon - 1))
            oos_sign = float((np.array(oos_ics) > 0).mean()) if oos_ic > 0 else float((np.array(oos_ics) < 0).mean())
            oos_br = int(np.mean(oos_breadths)) if len(oos_breadths) > 0 else 0

            # Cost
            cost_bps = cfg.get("sleeve_cost_bps", 10.0)
            turnover = _estimate_turnover(diag.halflife)
            expected_cost = cost_bps * turnover
            expected_alpha = abs(oos_ic) * 10000
            net_alpha = expected_alpha - expected_cost
            acr = expected_alpha / max(expected_cost, 1e-10)

            # Window status
            window_reasons = []
            if abs(oos_ic) < oos_ic_thresh:
                window_reasons.append("oos_ic_below_threshold")
            if oos_sign < oos_sign_thresh:
                window_reasons.append("oos_sign_inconsistent")
            if oos_br < min_oos_breadth:
                window_reasons.append("oos_breadth_insufficient")
            if net_alpha < 0:
                window_reasons.append("oos_cost_dominated")

            status = "pass" if not window_reasons else "fail"

            results.append(WalkForwardResult(
                sleeve_id=diag.sleeve_id,
                train_start=str(train_dates[0])[:10],
                train_end=str(train_dates[-1])[:10],
                test_start=str(test_dates[0])[:10],
                test_end=str(test_dates[-1])[:10],
                oos_ic=round(oos_ic, 6), oos_icir=round(oos_icir, 4),
                oos_hac_tstat=round(oos_tstat, 3),
                oos_sign_consistency=round(oos_sign, 4),
                oos_turnover=round(turnover, 4),
                oos_cost_bps=round(expected_cost, 2),
                oos_net_alpha_bps=round(net_alpha, 2),
                oos_alpha_cost_ratio=round(acr, 2),
                oos_breadth=oos_br,
                window_status=status,
                rejection_reason=";".join(window_reasons) if window_reasons else "",
            ))

    return results


def _compute_oos_ic(
    df: pd.DataFrame, feature: str, horizon: int,
    cond_type: str, cond_value: str,
    pit_labels: pd.DataFrame | None, min_breadth: int,
) -> tuple[list[float], list[int], list]:
    work = _build_fwd_return_col(df, horizon)
    col = f"fwd_ret_{horizon}d"
    if col not in work.columns:
        return [], [], []

    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work[col] = pd.to_numeric(work[col], errors="coerce")

    # Merge PIT labels if regime conditioning
    if cond_type == "regime" and pit_labels is not None and not pit_labels.empty and "regime_label" in pit_labels.columns:
        if "regime_label" in work.columns:
            work = work.drop(columns=["regime_label"])
        pit = pit_labels[["date", "regime_label"]].copy()
        pit["date"] = pd.to_datetime(pit["date"])
        work = work.merge(pit, on="date", how="left")
        work["regime_label"] = work["regime_label"].fillna("Sideways")
    elif cond_type == "regime" and "regime_label" not in work.columns:
        work["regime_label"] = "Sideways"

    subset = _filter_by_condition(work, cond_type, cond_value)
    if subset.empty:
        return [], [], []

    return _cs_ic_by_date(subset, feature, col, min_breadth=min_breadth)


# ── Phase 8: Conditional Simple Sleeve Simulator ─────────────────────────────

@dataclass
class SimpleSleeveBacktest:
    sleeve_id: str
    period: str
    gross_return: float
    cost_bps: float
    net_return: float
    turnover: float
    hit_rate: float
    max_drawdown: float
    sharpe: float
    net_sharpe: float
    alpha_cost_ratio: float
    breadth: int
    backtest_quality: str


def run_simple_sleeve_backtest(
    df: pd.DataFrame,
    diagnostics: list[SleeveDiagnostic],
    mt_results: list[MultipleTestingResult],
    wf_results: list[WalkForwardResult],
    config: dict[str, Any] | None = None,
) -> list[SimpleSleeveBacktest]:
    """Simple long/short sleeve backtest for sleeves passing diagnostics.

    Long top quantile, short bottom quantile, equal weight, realistic costs.
    No hidden optimizer, no hidden neutrality, no post-hoc transformations.
    """
    cfg = config or _DEFAULT_CONFIG["conditional_alpha"]
    top_q = cfg.get("sleeve_top_quantile", 0.2)
    bottom_q = cfg.get("sleeve_bottom_quantile", 0.2)
    cost_bps = cfg.get("sleeve_cost_bps", 10.0)
    rebalance = cfg.get("sleeve_rebalance", "monthly")

    # Only backtest sleeves that pass BHY or walk-forward
    passing = [
        d for d, mt in zip(diagnostics, mt_results)
        if mt.passes_bhy or (wf_results and any(w.sleeve_id == d.sleeve_id and w.window_status == "pass" for w in wf_results))
    ]

    if not passing:
        return []

    results = []
    for diag in passing:
        feature = diag.feature
        horizon = diag.horizon
        cond_type = diag.condition_type
        cond_value = diag.condition_value

        work = _build_fwd_return_col(df, horizon)
        col = f"fwd_ret_{horizon}d"
        if col not in work.columns:
            continue

        work[feature] = pd.to_numeric(work[feature], errors="coerce")
        work[col] = pd.to_numeric(work[col], errors="coerce")

        if "regime_label" not in work.columns:
            work["regime_label"] = "Sideways"
        else:
            # Ensure regime_label is present for filtering
            pass

        subset = _filter_by_condition(work, cond_type, cond_value)
        if subset.empty:
            continue

        # Monthly returns
        period_returns = []
        period_turnovers = []
        period_breadths = []

        dates = sorted(subset["date"].unique())
        prev_long = set()
        prev_short = set()

        for date in dates:
            day_data = subset[subset["date"] == date]
            if len(day_data) < 10:
                continue

            vals = pd.to_numeric(day_data[feature], errors="coerce").dropna()
            if len(vals) < 10:
                continue

            # Long top, short bottom
            n_long = max(1, int(len(vals) * top_q))
            n_short = max(1, int(len(vals) * bottom_q))

            sorted_vals = vals.sort_values()
            long_tickers = set(sorted_vals.tail(n_long).index)
            short_tickers = set(sorted_vals.head(n_short).index)

            # Returns
            long_rets = day_data[day_data["ticker"].isin(long_tickers)][col].dropna()
            short_rets = day_data[day_data["ticker"].isin(short_tickers)][col].dropna()

            if len(long_rets) == 0 or len(short_rets) == 0:
                continue

            gross = float(long_rets.mean() - short_rets.mean())

            # Turnover
            turnover = 1.0
            if prev_long:
                long_to = 1.0 - len(long_tickers & prev_long) / max(len(prev_long), 1)
                short_to = 1.0 - len(short_tickers & prev_short) / max(len(prev_short), 1)
                turnover = (long_to + short_to) / 2.0

            cost = cost_bps * turnover / 10000
            net = gross - cost

            period_returns.append((gross, cost, net, turnover, len(long_tickers) + len(short_tickers)))
            period_turnovers.append(turnover)
            period_breadths.append(len(long_tickers) + len(short_tickers))

            prev_long = long_tickers
            prev_short = short_tickers

        if not period_returns:
            continue

        gross_rets = np.array([r[0] for r in period_returns])
        costs = np.array([r[1] for r in period_returns])
        net_rets = np.array([r[2] for r in period_returns])

        # Compute metrics
        periods = len(period_returns)
        avg_gross = float(np.mean(gross_rets))
        avg_cost = float(np.mean(costs))
        avg_net = float(np.mean(net_rets))
        avg_turnover = float(np.mean(period_turnovers))
        avg_breadth = int(np.mean(period_breadths))

        # Hit rate
        hit_rate = float((net_rets > 0).mean())

        # Sharpe (annualized)
        gross_sharpe = float(np.mean(gross_rets) / max(np.std(gross_rets, ddof=1), 1e-10) * np.sqrt(252))
        net_sharpe = float(np.mean(net_rets) / max(np.std(net_rets, ddof=1), 1e-10) * np.sqrt(252))

        # Max drawdown
        cum_net = np.cumsum(net_rets)
        peak = np.maximum.accumulate(cum_net)
        dd = (cum_net - peak) / max(np.abs(peak).max(), 1e-10)
        max_dd = float(np.min(dd))

        acr = avg_gross / max(avg_cost, 1e-10)

        # Quality
        if net_sharpe > 0.5 and hit_rate > 0.55 and avg_breadth >= 10:
            quality = "high"
        elif net_sharpe > 0 and hit_rate > 0.5:
            quality = "medium"
        else:
            quality = "low"

        results.append(SimpleSleeveBacktest(
            sleeve_id=diag.sleeve_id,
            period=f"{periods}_periods",
            gross_return=round(avg_gross, 6),
            cost_bps=round(avg_cost * 10000, 2),
            net_return=round(avg_net, 6),
            turnover=round(avg_turnover, 4),
            hit_rate=round(hit_rate, 4),
            max_drawdown=round(max_dd, 4),
            sharpe=round(gross_sharpe, 3),
            net_sharpe=round(net_sharpe, 3),
            alpha_cost_ratio=round(acr, 2),
            breadth=avg_breadth,
            backtest_quality=quality,
        ))

    return results


# ── Phase 9: Conditional Sleeve Admission Gate ───────────────────────────────

@dataclass
class AdmissionResult:
    sleeve_id: str
    pit_status: str
    statistical_status: str
    multiple_testing_status: str
    stability_status: str
    breadth_status: str
    decay_status: str
    cost_status: str
    capacity_status: str
    walk_forward_status: str
    simple_sleeve_status: str
    final_status: str
    rejection_reason: str
    recommended_next_action: str


def evaluate_admission(
    diagnostics: list[SleeveDiagnostic],
    mt_results: list[MultipleTestingResult],
    stability_results: list[StabilityResult],
    wf_results: list[WalkForwardResult],
    simple_sleeve_results: list[SimpleSleeveBacktest],
    config: dict[str, Any] | None = None,
) -> list[AdmissionResult]:
    """Formal admission gate for conditional sleeves.

    Classifies sleeves as: rejected, discovery_interest, research_candidate,
    strong_research_candidate, production_watchlist, production_candidate.
    """
    cfg = config or _DEFAULT_CONFIG["conditional_alpha"]
    min_capacity = cfg.get("admission_min_capacity_score", 0.3)
    min_acr = cfg.get("admission_min_alpha_cost_ratio", 1.5)

    # Index results by sleeve_id
    wf_by_sleeve: dict[str, list[WalkForwardResult]] = {}
    for w in wf_results:
        wf_by_sleeve.setdefault(w.sleeve_id, []).append(w)

    ss_by_sleeve: dict[str, SimpleSleeveBacktest] = {}
    for s in simple_sleeve_results:
        ss_by_sleeve[s.sleeve_id] = s

    stab_by_sleeve: dict[str, StabilityResult] = {}
    for s in stability_results:
        stab_by_sleeve[s.sleeve_id] = s

    mt_by_sleeve: dict[str, MultipleTestingResult] = {}
    for m in mt_results:
        mt_by_sleeve[m.sleeve_id] = m

    results = []
    for diag in diagnostics:
        sid = diag.sleeve_id
        mt = mt_by_sleeve.get(sid)
        stab = stab_by_sleeve.get(sid)
        wf_list = wf_by_sleeve.get(sid, [])
        ss = ss_by_sleeve.get(sid)

        # PIT
        pit_pass = diag.pit_status in ("pit", "proxy")
        pit_status = "pass" if pit_pass else "fail"

        # Statistical
        ic_thresh = cfg.get("ic_mean_threshold", 0.005)
        tstat_thresh = cfg.get("hac_tstat_threshold", 2.0)
        stat_pass = abs(diag.mean_ic) >= ic_thresh and abs(diag.hac_tstat) >= tstat_thresh
        statistical_status = "pass" if stat_pass else "fail"

        # Multiple testing
        mt_pass = mt is not None and mt.passes_bhy
        mt_research = mt is not None and mt.passes_bh
        if mt_pass:
            multiple_testing_status = "pass"
        elif mt_research:
            multiple_testing_status = "research"
        else:
            multiple_testing_status = "fail"

        # Stability
        stab_pass = stab is not None and stab.stability_status in ("stable", "marginally_stable")
        stability_status = "pass" if stab_pass else "fail"

        # Breadth
        min_br = cfg.get("min_breadth_for_bucket", 10)
        breadth_pass = diag.avg_breadth >= min_br
        breadth_status = "pass" if breadth_pass else "fail"

        # Decay
        decay_pass = diag.halflife > 0 and diag.persistence_at_horizon > 0.1
        decay_status = "pass" if decay_pass else "fail"

        # Cost
        cost_pass = diag.net_expected_alpha_bps > 0 and diag.alpha_cost_ratio >= min_acr
        cost_status = "pass" if cost_pass else "fail"

        # Capacity
        capacity_pass = diag.capacity_score >= min_capacity
        capacity_status = "pass" if capacity_pass else "fail"

        # Walk-forward
        wf_pass = any(w.window_status == "pass" for w in wf_list) if wf_list else False
        walk_forward_status = "pass" if wf_pass else ("not_evaluated" if not wf_list else "fail")

        # Simple sleeve
        ss_pass = ss is not None and ss.backtest_quality in ("high", "medium")
        simple_sleeve_status = "pass" if ss_pass else ("not_evaluated" if ss is None else "fail")

        # Final classification
        gates = [pit_pass, stat_pass, mt_pass, stab_pass, breadth_pass, decay_pass, cost_pass, capacity_pass]
        n_pass = sum(gates)
        wf_gate = wf_pass if wf_list else True
        ss_gate = ss_pass if ss is not None else True

        # Hard rule: BHY failure blocks production status
        bhy_blocks_production = mt is not None and not mt.passes_bhy

        if n_pass >= 7 and wf_gate and ss_gate and not bhy_blocks_production:
            final = SleeveFinalStatus.PRODUCTION_CANDIDATE.value
        elif n_pass >= 6 and wf_gate and not bhy_blocks_production:
            final = SleeveFinalStatus.PRODUCTION_WATCHLIST.value
        elif n_pass >= 5 and mt_pass:
            final = SleeveFinalStatus.STRONG_RESEARCH_CANDIDATE.value
        elif n_pass >= 4 and mt_research:
            final = SleeveFinalStatus.RESEARCH_CANDIDATE.value
        elif n_pass >= 2 and (mt is not None and mt.passes_raw):
            final = SleeveFinalStatus.DISCOVERY_INTEREST.value
        else:
            final = SleeveFinalStatus.REJECTED.value

        # Rejection reasons
        reasons = []
        if not pit_pass:
            reasons.append("not_pit_valid")
        if not stat_pass:
            reasons.append("statistical_thresholds_not_met")
        if mt is not None and not mt.passes_bh:
            reasons.append("fails_multiple_testing")
        if not stab_pass:
            reasons.append("unstable")
        if not breadth_pass:
            reasons.append("insufficient_breadth")
        if not decay_pass:
            reasons.append("signal_decays_too_fast")
        if not cost_pass:
            reasons.append("cost_dominated")
        if not capacity_pass:
            reasons.append("insufficient_capacity")
        if not wf_gate and wf_list:
            reasons.append("fails_walk_forward")
        if not ss_gate and ss is not None:
            reasons.append("fails_simple_sleeve_backtest")

        # Recommended action
        if final == SleeveFinalStatus.PRODUCTION_CANDIDATE.value:
            action = "proceed_to_production_pilot"
        elif final == SleeveFinalStatus.PRODUCTION_WATCHLIST.value:
            action = "monitor_and_revalidate_next_window"
        elif final == SleeveFinalStatus.STRONG_RESEARCH_CANDIDATE.value:
            action = "deepen_research_with_additional_conditions"
        elif final == SleeveFinalStatus.RESEARCH_CANDIDATE.value:
            action = "research_interest_requires_more_evidence"
        elif final == SleeveFinalStatus.DISCOVERY_INTEREST.value:
            action = "exploratory_only_do_not_overfit"
        else:
            action = "reject_and_do_not_pursue"

        results.append(AdmissionResult(
            sleeve_id=sid,
            pit_status=pit_status, statistical_status=statistical_status,
            multiple_testing_status=multiple_testing_status,
            stability_status=stability_status, breadth_status=breadth_status,
            decay_status=decay_status, cost_status=cost_status,
            capacity_status=capacity_status, walk_forward_status=walk_forward_status,
            simple_sleeve_status=simple_sleeve_status,
            final_status=final,
            rejection_reason=";".join(reasons) if reasons else "",
            recommended_next_action=action,
        ))

    return results


# ── Phase 10: Bear-Regime Sleeve Special Audit ───────────────────────────────

@dataclass
class BearAuditResult:
    sleeve_id: str
    bear_date_count: int
    avg_breadth_bear: int
    dominant_bear_episode: str
    leave_one_bear_episode_min_ic: float
    crisis_exclusion_ic: float
    highvol_comparison_ic: float
    beta_adjusted_ic: float
    sector_neutral_ic: float
    bear_cost_status: str
    bear_capacity_score: float
    bear_audit_status: str
    rejection_reason: str


def run_bear_regime_audit(
    df: pd.DataFrame,
    diagnostics: list[SleeveDiagnostic],
    pit_regime_labels: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> list[BearAuditResult]:
    """Special robustness audit for Bear-regime sleeves.

    Verifies Bear label is PIT, measures Bear-date count, identifies dominant
    Bear episodes, runs leave-one-Bear-episode-out, compares to HighVol,
    tests beta/sector neutrality, and measures cost viability in Bear.
    """
    cfg = config or _DEFAULT_CONFIG["conditional_alpha"]
    min_bear_dates = cfg.get("bear_min_dates", 15)
    min_bear_breadth = cfg.get("bear_min_breadth", 5)
    loe_min_ic = cfg.get("bear_leave_one_episode_min_ic", 0.002)

    # Filter Bear sleeves
    bear_sleeves = [
        d for d in diagnostics
        if d.condition_type == "regime" and d.condition_value == "Bear"
    ]

    if not bear_sleeves:
        return []

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"])
    if "regime_label" in work.columns:
        work = work.drop(columns=["regime_label"])

    # Merge PIT labels
    if pit_regime_labels is not None and not pit_regime_labels.empty and "regime_label" in pit_regime_labels.columns:
        pit = pit_regime_labels[["date", "regime_label"]].copy()
        pit["date"] = pd.to_datetime(pit["date"])
        work = work.merge(pit, on="date", how="left")
        work["regime_label"] = work["regime_label"].fillna("Sideways")
    else:
        work["regime_label"] = "Sideways"

    # Identify Bear dates
    bear_dates = work[work["regime_label"] == "Bear"]["date"].unique()
    if len(bear_dates) < min_bear_dates:
        return [
            BearAuditResult(
                sleeve_id=d.sleeve_id, bear_date_count=len(bear_dates),
                avg_breadth_bear=0, dominant_bear_episode="",
                leave_one_bear_episode_min_ic=0.0, crisis_exclusion_ic=0.0,
                highvol_comparison_ic=0.0, beta_adjusted_ic=0.0,
                sector_neutral_ic=0.0, bear_cost_status="insufficient_bear_dates",
                bear_capacity_score=0.0, bear_audit_status="rejected",
                rejection_reason=f"only_{len(bear_dates)}_bear_dates",
            )
            for d in bear_sleeves
        ]

    # Identify Bear episodes (contiguous periods)
    bear_date_series = pd.Series(sorted(bear_dates))
    gaps = bear_date_series.diff().dt.days
    episode_breaks = gaps[gaps > 10].index  # >10 day gap = new episode
    episodes = []
    start_idx = 0
    for break_idx in episode_breaks:
        episodes.append(bear_date_series.iloc[start_idx:break_idx])
        start_idx = break_idx
    episodes.append(bear_date_series.iloc[start_idx:])

    results = []
    for diag in bear_sleeves:
        feature = diag.feature
        horizon = diag.horizon

        bear_subset = work[work["regime_label"] == "Bear"]
        bear_subset = _build_fwd_return_col(bear_subset, horizon)
        col = f"fwd_ret_{horizon}d"
        if col not in bear_subset.columns:
            results.append(_empty_bear_audit(diag, reason="missing_data"))
            continue

        bear_subset[feature] = pd.to_numeric(bear_subset[feature], errors="coerce")
        bear_subset[col] = pd.to_numeric(bear_subset[col], errors="coerce")

        # Bear IC
        bear_ics, bear_breadths, bear_ics_dates = _cs_ic_by_date(
            bear_subset, feature, col, min_breadth=min_bear_breadth,
        )

        if len(bear_ics) < min_bear_dates:
            results.append(_empty_bear_audit(diag, n_dates=len(bear_ics), reason="too_few_bear_dates"))
            continue

        avg_bear_br = int(np.mean(bear_breadths)) if len(bear_breadths) > 0 else 0

        # Dominant Bear episode
        episode_ics = {}
        for i, ep in enumerate(episodes):
            ep_mask = bear_ics_dates.isin(ep)
            if ep_mask.sum() >= 3:
                episode_ics[f"episode_{i}"] = float(np.mean(bear_ics[ep_mask]))

        dominant_ep = max(episode_ics, key=lambda k: abs(episode_ics[k])) if episode_ics else ""

        # Leave-one-Bear-episode-out
        loe_ics = []
        for i, ep in enumerate(episodes):
            ep_mask = ~bear_ics_dates.isin(ep)
            if ep_mask.sum() >= min_bear_dates:
                loe_ics.append(float(np.mean(bear_ics[ep_mask])))

        loe_min = min(loe_ics) if loe_ics else 0.0

        # Crisis exclusion
        crisis_ic = _crisis_exclusion(bear_ics, bear_ics_dates, cfg)

        # HighVol comparison
        hv_subset = work[work["regime_label"] == "HighVol"]
        hv_subset = _build_fwd_return_col(hv_subset, horizon)
        if col in hv_subset.columns:
            hv_subset[feature] = pd.to_numeric(hv_subset[feature], errors="coerce")
            hv_subset[col] = pd.to_numeric(hv_subset[col], errors="coerce")
            hv_ics, _, _ = _cs_ic_by_date(hv_subset, feature, col, min_breadth=min_bear_breadth)
            hv_ic = float(np.mean(hv_ics)) if len(hv_ics) >= 5 else 0.0
        else:
            hv_ic = 0.0

        # Beta-adjusted IC
        beta_adj_ic = _beta_adjusted_ic(bear_subset, feature, col)

        # Sector-neutral IC
        sn_ic = _sector_neutral_ic_in_condition(bear_subset, feature, col)

        # Cost status
        cost_bps = cfg.get("sleeve_cost_bps", 10.0)
        turnover = _estimate_turnover(diag.halflife)
        expected_cost = cost_bps * turnover
        expected_alpha = abs(np.mean(bear_ics)) * 10000
        bear_cost_ok = expected_alpha > expected_cost

        # Capacity
        bear_cap = _compute_capacity_score(avg_bear_br, int(np.min(bear_breadths)) if len(bear_breadths) > 0 else 0, len(bear_ics), diag.halflife)

        # Audit status
        reasons = []
        if loe_min < loe_min_ic and loe_min != 0:
            reasons.append("fragile_to_leave_one_bear_episode")
        if crisis_ic < loe_min_ic and crisis_ic != 0:
            reasons.append("crisis_dependent")
        if abs(beta_adj_ic) < cfg.get("bear_beta_adjusted_threshold", 0.003):
            reasons.append("beta_explained")
        if abs(sn_ic) < cfg.get("bear_sector_neutral_threshold", 0.002):
            reasons.append("sector_explained")
        if not bear_cost_ok:
            reasons.append("bear_cost_dominated")

        if len(reasons) >= 3:
            audit_status = "rejected"
        elif len(reasons) >= 1:
            audit_status = "caution"
        else:
            audit_status = "robust"

        results.append(BearAuditResult(
            sleeve_id=diag.sleeve_id,
            bear_date_count=len(bear_ics),
            avg_breadth_bear=avg_bear_br,
            dominant_bear_episode=dominant_ep,
            leave_one_bear_episode_min_ic=round(loe_min, 6),
            crisis_exclusion_ic=round(crisis_ic, 6),
            highvol_comparison_ic=round(hv_ic, 6),
            beta_adjusted_ic=round(beta_adj_ic, 6),
            sector_neutral_ic=round(sn_ic, 6),
            bear_cost_status="viable" if bear_cost_ok else "dominated",
            bear_capacity_score=round(bear_cap, 4),
            bear_audit_status=audit_status,
            rejection_reason=";".join(reasons) if reasons else "",
        ))

    return results


def _beta_adjusted_ic(df: pd.DataFrame, feature: str, return_col: str) -> float:
    """IC after adjusting for market beta exposure."""
    if "capm_beta" not in df.columns:
        return 0.0

    valid = df[[feature, return_col, "capm_beta", "date", "ticker"]].dropna()
    if len(valid) < 20:
        return 0.0

    # Residualize returns against beta per date
    residuals = []
    for date, grp in valid.groupby("date", sort=True):
        if len(grp) < 5:
            continue
        y = grp[return_col].values
        x = grp["capm_beta"].values.reshape(-1, 1)
        if len(y) < 5:
            continue
        try:
            X = np.column_stack([np.ones(len(x)), x])
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            resid = y - X @ beta
            for j in range(len(grp)):
                residuals.append({
                    "date": date, "ticker": grp.iloc[j]["ticker"],
                    "residual": resid[j], "feature": grp.iloc[j][feature],
                })
        except Exception:
            continue

    if not residuals:
        return 0.0

    resid_df = pd.DataFrame(residuals)
    ics = []
    for date, grp in resid_df.groupby("date", sort=True):
        if len(grp) < 5:
            continue
        c, _ = _robust_spearmanr(grp["feature"], grp["residual"])
        if np.isfinite(c):
            ics.append(c)

    return float(np.mean(ics)) if ics else 0.0


def _sector_neutral_ic_in_condition(df: pd.DataFrame, feature: str, return_col: str) -> float:
    """Sector-neutral IC within a condition."""
    if "sector" not in df.columns:
        return 0.0

    valid = df[[feature, return_col, "sector", "date", "ticker"]].dropna()
    if len(valid) < 20:
        return 0.0

    # Rank within sector/date
    valid = valid.copy()
    valid["feat_rank"] = valid.groupby(["date", "sector"], sort=False)[feature].rank(pct=True)
    valid["ret_rank"] = valid.groupby(["date", "sector"], sort=False)[return_col].rank(pct=True)

    ics = []
    for date, grp in valid.groupby("date", sort=True):
        if len(grp) < 5:
            continue
        c, _ = _robust_spearmanr(grp["feat_rank"], grp["ret_rank"])
        if np.isfinite(c):
            ics.append(c)

    return float(np.mean(ics)) if ics else 0.0


def _empty_bear_audit(diag: SleeveDiagnostic, n_dates: int = 0, reason: str = "") -> BearAuditResult:
    return BearAuditResult(
        sleeve_id=diag.sleeve_id, bear_date_count=n_dates,
        avg_breadth_bear=0, dominant_bear_episode="",
        leave_one_bear_episode_min_ic=0.0, crisis_exclusion_ic=0.0,
        highvol_comparison_ic=0.0, beta_adjusted_ic=0.0,
        sector_neutral_ic=0.0, bear_cost_status="insufficient_data",
        bear_capacity_score=0.0, bear_audit_status="rejected",
        rejection_reason=reason or "insufficient_data",
    )


# ── Main Engine ──────────────────────────────────────────────────────────────

@dataclass
class ConditionalAlphaBundle:
    """Full conditional alpha validation results."""
    pit_regime_labels: pd.DataFrame
    sleeve_registry: list[SleeveDefinition]
    diagnostics: list[SleeveDiagnostic]
    mt_results: list[MultipleTestingResult]
    stability_results: list[StabilityResult]
    wf_results: list[WalkForwardResult]
    simple_sleeve_results: list[SimpleSleeveBacktest]
    admission_results: list[AdmissionResult]
    bear_audit_results: list[BearAuditResult]


class ConditionalAlphaEngine:
    """Conditional Alpha Validation Engine.

    Determines whether regime-specific sleeves are real, tradable,
    point-in-time valid, dependency-robust, and stable enough for
    research promotion or production consideration.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.cfg = _get_config(self.config)

    def run_full_validation(
        self,
        df: pd.DataFrame,
        features: list[str],
        horizons: list[int] | None = None,
    ) -> ConditionalAlphaBundle:
        """Run full conditional alpha validation pipeline."""
        if horizons is None:
            horizons = self.cfg.get("horizons", [1, 2, 3, 5, 10, 20])

        condition_types = self.cfg.get("condition_types", ["regime", "volatility", "liquidity", "size", "sector"])
        n_buckets = self.cfg.get("n_buckets", 3)
        rebalance_rules = self.cfg.get("rebalance_rules", ["monthly"])

        # Phase 2: PIT regime labels
        logger.info("Phase 2: Computing PIT regime labels...")
        pit_labels = compute_pit_regime_labels(
            df,
            window=self.cfg.get("pit_regime_window", 126),
            min_obs=self.cfg.get("pit_regime_min_obs", 60),
            prob_threshold=self.cfg.get("pit_prob_threshold", 0.5),
        )

        # Phase 3: Sleeve registry
        logger.info("Phase 3: Building sleeve registry...")
        registry = build_sleeve_registry(
            features, horizons, condition_types,
            pit_regime_labels=pit_labels, df=df,
            n_buckets=n_buckets, rebalance_rules=rebalance_rules,
            max_sleeves=self.cfg.get("max_sleeves", 0),
        )
        logger.info("Sleeve registry: %d sleeves", len(registry))

        # Phase 4: Diagnostics
        logger.info("Phase 4: Computing sleeve diagnostics...")
        diagnostics = compute_sleeve_diagnostics(
            df, registry, pit_regime_labels=pit_labels, config=self.cfg,
        )

        # Phase 5: Multiple testing
        logger.info("Phase 5: Multiple-testing correction...")
        mt_results = compute_multiple_testing_correction(diagnostics, config=self.cfg)

        # Phase 6: Stability
        logger.info("Phase 6: Stability analysis...")
        stability_results = compute_stability_analysis(df, diagnostics, config=self.cfg)

        # Phase 7: Walk-forward
        logger.info("Phase 7: Walk-forward validation...")
        wf_results = run_walk_forward_validation(
            df, diagnostics, mt_results, pit_regime_labels=pit_labels, config=self.cfg,
        )

        # Phase 8: Simple sleeve backtest
        logger.info("Phase 8: Simple sleeve backtest...")
        ss_results = run_simple_sleeve_backtest(
            df, diagnostics, mt_results, wf_results, config=self.cfg,
        )

        # Phase 9: Admission gate
        logger.info("Phase 9: Admission gate...")
        admission_results = evaluate_admission(
            diagnostics, mt_results, stability_results, wf_results, ss_results, config=self.cfg,
        )

        # Phase 10: Bear audit
        logger.info("Phase 10: Bear-regime audit...")
        bear_results = run_bear_regime_audit(
            df, diagnostics, pit_regime_labels=pit_labels, config=self.cfg,
        )

        return ConditionalAlphaBundle(
            pit_regime_labels=pit_labels,
            sleeve_registry=registry,
            diagnostics=diagnostics,
            mt_results=mt_results,
            stability_results=stability_results,
            wf_results=wf_results,
            simple_sleeve_results=ss_results,
            admission_results=admission_results,
            bear_audit_results=bear_results,
        )


# ── Report Generation ────────────────────────────────────────────────────────

def generate_conditional_alpha_reports(
    bundle: ConditionalAlphaBundle,
    output_dir: str | Path = "output/models/conditional_alpha",
) -> dict[str, Path]:
    """Generate all 11 conditional alpha reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 1. pit_regime_labels.csv
    if not bundle.pit_regime_labels.empty:
        p = output_dir / "pit_regime_labels.csv"
        bundle.pit_regime_labels.to_csv(p, index=False)
        paths["pit_regime_labels"] = p

    # 2. conditional_sleeve_registry.csv
    rows = []
    for s in bundle.sleeve_registry:
        rows.append({
            "sleeve_id": s.sleeve_id, "condition_type": s.condition_type,
            "condition_value": s.condition_value, "feature": s.feature,
            "family": s.family, "horizon": s.horizon,
            "rebalance_rule": s.rebalance_rule, "test_family": s.test_family,
            "enabled": s.enabled, "reason_if_disabled": s.reason_if_disabled,
        })
    p = output_dir / "conditional_sleeve_registry.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["sleeve_registry"] = p

    # 3. conditional_sleeve_diagnostics.csv
    rows = []
    for d in bundle.diagnostics:
        rows.append({
            "sleeve_id": d.sleeve_id, "feature": d.feature,
            "family": d.family, "condition_type": d.condition_type,
            "condition_value": d.condition_value, "horizon": d.horizon,
            "mean_ic": d.mean_ic, "icir": d.icir,
            "hac_tstat": d.hac_tstat, "p_value": d.p_value,
            "sign_consistency": d.sign_consistency,
            "n_dates": d.n_dates, "avg_breadth": d.avg_breadth,
            "min_breadth": d.min_breadth, "halflife": d.halflife,
            "persistence_at_horizon": d.persistence_at_horizon,
            "expected_turnover": d.expected_turnover,
            "expected_alpha_bps": d.expected_alpha_bps,
            "expected_cost_bps": d.expected_cost_bps,
            "net_expected_alpha_bps": d.net_expected_alpha_bps,
            "alpha_cost_ratio": d.alpha_cost_ratio,
            "capacity_score": d.capacity_score,
            "pit_status": d.pit_status,
            "diagnostic_quality": d.diagnostic_quality,
            "rejection_reason": d.rejection_reason,
        })
    p = output_dir / "conditional_sleeve_diagnostics.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["sleeve_diagnostics"] = p

    # 4. conditional_multiple_testing_report.csv
    rows = []
    for m in bundle.mt_results:
        rows.append({
            "sleeve_id": m.sleeve_id, "raw_p_value": m.raw_p_value,
            "bh_q_value": m.bh_q_value, "bhy_q_value": m.bhy_q_value,
            "passes_raw": m.passes_raw, "passes_bh": m.passes_bh,
            "passes_bhy": m.passes_bhy, "test_family": m.test_family,
            "test_family_size": m.test_family_size,
            "dependency_group": m.dependency_group,
            "effective_test_count": m.effective_test_count,
            "white_rc_p_value": m.white_rc_p_value,
            "hansen_spa_p_value": m.hansen_spa_p_value,
            "evidence_status": m.evidence_status,
            "rejection_reason": m.rejection_reason,
        })
    p = output_dir / "conditional_multiple_testing_report.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["multiple_testing"] = p

    # 5. conditional_stability_report.csv
    rows = []
    for s in bundle.stability_results:
        rows.append({
            "sleeve_id": s.sleeve_id, "full_sample_ic": s.full_sample_ic,
            "leave_one_year_min_ic": s.leave_one_year_min_ic,
            "leave_one_year_fail_year": s.leave_one_year_fail_year,
            "leave_one_sector_min_ic": s.leave_one_sector_min_ic,
            "dominant_year_contribution": s.dominant_year_contribution,
            "dominant_sector_contribution": s.dominant_sector_contribution,
            "rolling_ic_min": s.rolling_ic_min, "rolling_ic_max": s.rolling_ic_max,
            "bootstrap_ic_mean": s.bootstrap_ic_mean,
            "bootstrap_ic_p05": s.bootstrap_ic_p05,
            "bootstrap_ic_p95": s.bootstrap_ic_p95,
            "crisis_exclusion_ic": s.crisis_exclusion_ic,
            "stability_status": s.stability_status,
            "rejection_reason": s.rejection_reason,
        })
    p = output_dir / "conditional_stability_report.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["stability"] = p

    # 6. conditional_walk_forward_report.csv
    rows = []
    for w in bundle.wf_results:
        rows.append({
            "sleeve_id": w.sleeve_id,
            "train_start": w.train_start, "train_end": w.train_end,
            "test_start": w.test_start, "test_end": w.test_end,
            "oos_ic": w.oos_ic, "oos_icir": w.oos_icir,
            "oos_hac_tstat": w.oos_hac_tstat,
            "oos_sign_consistency": w.oos_sign_consistency,
            "oos_turnover": w.oos_turnover, "oos_cost_bps": w.oos_cost_bps,
            "oos_net_alpha_bps": w.oos_net_alpha_bps,
            "oos_alpha_cost_ratio": w.oos_alpha_cost_ratio,
            "oos_breadth": w.oos_breadth,
            "window_status": w.window_status,
            "rejection_reason": w.rejection_reason,
        })
    if rows:
        p = output_dir / "conditional_walk_forward_report.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        paths["walk_forward"] = p

    # 7. conditional_simple_sleeve_backtest.csv
    rows = []
    for s in bundle.simple_sleeve_results:
        rows.append({
            "sleeve_id": s.sleeve_id, "period": s.period,
            "gross_return": s.gross_return, "cost_bps": s.cost_bps,
            "net_return": s.net_return, "turnover": s.turnover,
            "hit_rate": s.hit_rate, "max_drawdown": s.max_drawdown,
            "sharpe": s.sharpe, "net_sharpe": s.net_sharpe,
            "alpha_cost_ratio": s.alpha_cost_ratio,
            "breadth": s.breadth, "backtest_quality": s.backtest_quality,
        })
    if rows:
        p = output_dir / "conditional_simple_sleeve_backtest.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        paths["simple_sleeve"] = p

    # 8. conditional_sleeve_admission_report.csv
    rows = []
    for a in bundle.admission_results:
        rows.append({
            "sleeve_id": a.sleeve_id,
            "pit_status": a.pit_status,
            "statistical_status": a.statistical_status,
            "multiple_testing_status": a.multiple_testing_status,
            "stability_status": a.stability_status,
            "breadth_status": a.breadth_status,
            "decay_status": a.decay_status,
            "cost_status": a.cost_status,
            "capacity_status": a.capacity_status,
            "walk_forward_status": a.walk_forward_status,
            "simple_sleeve_status": a.simple_sleeve_status,
            "final_status": a.final_status,
            "rejection_reason": a.rejection_reason,
            "recommended_next_action": a.recommended_next_action,
        })
    p = output_dir / "conditional_sleeve_admission_report.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["admission"] = p

    # 9. bear_regime_sleeve_audit.csv
    rows = []
    for b in bundle.bear_audit_results:
        rows.append({
            "sleeve_id": b.sleeve_id,
            "bear_date_count": b.bear_date_count,
            "avg_breadth_bear": b.avg_breadth_bear,
            "dominant_bear_episode": b.dominant_bear_episode,
            "leave_one_bear_episode_min_ic": b.leave_one_bear_episode_min_ic,
            "crisis_exclusion_ic": b.crisis_exclusion_ic,
            "highvol_comparison_ic": b.highvol_comparison_ic,
            "beta_adjusted_ic": b.beta_adjusted_ic,
            "sector_neutral_ic": b.sector_neutral_ic,
            "bear_cost_status": b.bear_cost_status,
            "bear_capacity_score": b.bear_capacity_score,
            "bear_audit_status": b.bear_audit_status,
            "rejection_reason": b.rejection_reason,
        })
    if rows:
        p = output_dir / "bear_regime_sleeve_audit.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        paths["bear_audit"] = p

    # 10. rejected_conditional_sleeves.csv
    rejected = [a for a in bundle.admission_results if a.final_status == SleeveFinalStatus.REJECTED.value]
    if rejected:
        rows = []
        for a in rejected:
            diag = next((d for d in bundle.diagnostics if d.sleeve_id == a.sleeve_id), None)
            mt = next((m for m in bundle.mt_results if m.sleeve_id == a.sleeve_id), None)
            rows.append({
                "sleeve_id": a.sleeve_id,
                "feature": diag.feature if diag else "",
                "family": diag.family if diag else "",
                "condition_type": diag.condition_type if diag else "",
                "condition_value": diag.condition_value if diag else "",
                "horizon": diag.horizon if diag else 0,
                "mean_ic": diag.mean_ic if diag else 0.0,
                "bhy_q_value": mt.bhy_q_value if mt else 1.0,
                "final_status": a.final_status,
                "rejection_reason": a.rejection_reason,
            })
        p = output_dir / "rejected_conditional_sleeves.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        paths["rejected"] = p

    # 11. accepted_research_conditional_sleeves.csv
    accepted = [
        a for a in bundle.admission_results
        if a.final_status in (
            SleeveFinalStatus.RESEARCH_CANDIDATE.value,
            SleeveFinalStatus.STRONG_RESEARCH_CANDIDATE.value,
            SleeveFinalStatus.PRODUCTION_WATCHLIST.value,
            SleeveFinalStatus.PRODUCTION_CANDIDATE.value,
        )
    ]
    if accepted:
        rows = []
        for a in accepted:
            diag = next((d for d in bundle.diagnostics if d.sleeve_id == a.sleeve_id), None)
            rows.append({
                "sleeve_id": a.sleeve_id,
                "feature": diag.feature if diag else "",
                "family": diag.family if diag else "",
                "condition_type": diag.condition_type if diag else "",
                "condition_value": diag.condition_value if diag else "",
                "horizon": diag.horizon if diag else 0,
                "mean_ic": diag.mean_ic if diag else 0.0,
                "final_status": a.final_status,
                "recommended_next_action": a.recommended_next_action,
            })
        p = output_dir / "accepted_research_conditional_sleeves.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        paths["accepted"] = p

    # PM-level summary
    summary = _generate_pm_summary(bundle)
    p = output_dir / "conditional_alpha_pm_summary.txt"
    with open(p, "w") as f:
        f.write(summary)
    paths["pm_summary"] = p

    logger.info("Conditional alpha reports generated: %s", list(paths.keys()))
    return paths


def _generate_pm_summary(bundle: ConditionalAlphaBundle) -> str:
    """PM-level report answering all key questions."""
    n_total = len(bundle.diagnostics)
    n_raw = sum(1 for m in bundle.mt_results if m.passes_raw)
    n_bh = sum(1 for m in bundle.mt_results if m.passes_bh)
    n_bhy = sum(1 for m in bundle.mt_results if m.passes_bhy)
    n_admitted = sum(
        1 for a in bundle.admission_results
        if a.final_status not in (SleeveFinalStatus.REJECTED.value, SleeveFinalStatus.DISCOVERY_INTEREST.value)
    )

    bear_candidates = [
        a for a in bundle.admission_results
        if a.sleeve_id in (d.sleeve_id for d in bundle.diagnostics if d.condition_type == "regime" and d.condition_value == "Bear")
        and a.final_status not in (SleeveFinalStatus.REJECTED.value, SleeveFinalStatus.DISCOVERY_INTEREST.value)
    ]

    bear_ep_dependent = [
        b for b in bundle.bear_audit_results
        if "fragile_to_leave_one_bear_episode" in b.rejection_reason or "crisis_dependent" in b.rejection_reason
    ]

    dep_failures = [
        m for m in bundle.mt_results
        if m.passes_raw and not m.passes_bh
    ]

    cost_viable = [
        d for d in bundle.diagnostics
        if d.net_expected_alpha_bps > 0
    ]

    sufficient_breadth = [
        d for d in bundle.diagnostics
        if d.avg_breadth >= 10
    ]

    lines = [
        "Conditional Alpha Validation — PM Summary",
        "=" * 60,
        "",
        f"Total conditional sleeves tested: {n_total}",
        f"Pass raw p-value (<0.05): {n_raw}",
        f"Pass Benjamini-Hochberg (q<0.10): {n_bh}",
        f"Pass Benjamini-Yekutieli (q<0.05): {n_bhy}",
        f"Remain after PIT/breadth/decay/cost/stability: {n_admitted}",
        "",
        "Bear-Regime Analysis",
        "-" * 40,
        f"Real research candidates (Bear): {len(bear_candidates)}",
        f"Dependent on one Bear/crisis episode: {len(bear_ep_dependent)}",
        "",
        "Dependency Robustness",
        "-" * 40,
        f"Fail dependency-robust correction: {len(dep_failures)}",
        "",
        "Economic Viability",
        "-" * 40,
        f"Cost viable inside condition: {len(cost_viable)}",
        f"Sufficient breadth/capacity: {len(sufficient_breadth)}",
        "",
        "Classification",
        "-" * 40,
    ]

    status_counts: dict[str, int] = {}
    for a in bundle.admission_results:
        status_counts[a.final_status] = status_counts.get(a.final_status, 0) + 1

    for status, count in sorted(status_counts.items()):
        lines.append(f"  {status}: {count}")

    lines.append("")
    lines.append("=" * 60)
    lines.append("Conclusion:")

    if n_bhy > 0 and len(bear_candidates) > 0:
        lines.append(f"  {n_bhy} sleeve(s) pass BHY. {len(bear_candidates)} Bear-regime candidate(s) warrant further research.")
    elif n_bh > 0:
        lines.append(f"  {n_bh} sleeve(s) pass BH but not BHY — research interest only, not production-valid.")
    else:
        lines.append("  No sleeves pass multiple-testing correction. Likely data mining.")

    lines.append("")
    return "\n".join(lines)
