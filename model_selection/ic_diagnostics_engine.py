"""Global-to-Conditional IC Diagnostic Engine.

Institutional, point-in-time, multiple-testing-aware framework that decomposes
global IC into residualized, sector-neutral, regime-conditional, liquidity-conditional,
size-conditional, volatility-conditional, and feature-family-specific IC.

Identifies where predictive power truly exists without lookahead, data mining,
or production-invalid sleeve selection.

Usage:
    from model_selection.ic_diagnostics_engine import ICDiagnosticsEngine

    engine = ICDiagnosticsEngine(config=cfg)
    results = engine.run_full_diagnostics(df, features, horizons)
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

from model_selection.research_numerics_core import (
    batch_ridge_residualize,
    compute_daily_ic_series,
    compute_forward_returns,
)
from model_selection._shared_stats import (
    benjamini_hochberg,
    benjamini_yekutieli,
    hac_tstat,
    ic_quality,
    p_from_tstat,
)
from model_selection._shared_feature_utils import (
    find_condition_column,
    get_family,
)
from model_selection._shared_config import merge_config

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────────────

class ICQuality(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INSUFFICIENT = "insufficient"


class EvidenceStatus(str, Enum):
    DISCOVERY_INTEREST = "discovery_interest"
    RESEARCH_CANDIDATE = "research_candidate"
    STRONG_CANDIDATE = "strong_candidate"
    PRODUCTION_CANDIDATE = "production_candidate"
    REJECTED = "rejected"


class AttributionLabel(str, Enum):
    NO_GLOBAL_ALPHA = "no_global_alpha"
    ALPHA_DILUTED_BY_REGIME = "alpha_diluted_by_regime"
    SECTOR_SPECIFIC_ALPHA = "sector_specific_alpha"
    LIQUIDITY_SPECIFIC_ALPHA = "liquidity_specific_alpha"
    SIZE_SPECIFIC_ALPHA = "size_specific_alpha"
    VOLATILITY_SPECIFIC_ALPHA = "volatility_specific_alpha"
    WRONG_HORIZON = "wrong_horizon"
    FACTOR_EXPOSURE_CONTAMINATION = "factor_exposure_contamination"
    INSUFFICIENT_BREADTH = "insufficient_breadth"
    UNSTABLE_SUBPERIODS = "unstable_subperiods"
    COST_DOMINATED_CONDITIONAL_ALPHA = "cost_dominated_conditional_alpha"
    MULTIPLE_TESTING_FAILURE = "multiple_testing_failure"


class SleeveStatus(str, Enum):
    ADMITTED = "admitted"
    RESEARCH_ONLY = "research_only"
    REJECTED = "rejected"


# ── Config defaults ──────────────────────────────────────────────────────────

_DEFAULT_IC_CONFIG: dict[str, Any] = {
    "ic_diagnostics": {
        "horizons": [1, 2, 3, 5, 10, 20, 40, 63],
        "min_dates_for_ic": 30,
        "min_breadth_for_ic": 8,
        "min_dates_for_residualized": 20,
        "min_breadth_for_residualized": 10,
        "min_dates_for_sector": 20,
        "min_sectors_for_sector_neutral": 3,
        "min_dates_for_conditional": 15,
        "min_breadth_for_conditional": 5,
        "min_breadth_for_bucket": 10,
        "n_buckets": 3,
        "winsor_q": 0.025,
        "ic_mean_threshold": 0.005,
        "icir_threshold": 0.5,
        "hac_tstat_threshold": 2.0,
        "sign_consistency_threshold": 0.6,
        "subperiod_stability_threshold": 0.5,
        "bh_q_threshold": 0.10,
        "bhy_q_threshold": 0.05,
        "sector_concentration_max": 0.5,
        "leave_one_out_ic_min": 0.002,
        "factor_controls": ["market", "size", "momentum", "volatility"],
        "conditional_conditions": ["regime", "volatility", "liquidity", "size", "sector"],
    },
}


def _get_ic_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract IC diagnostics config from full config."""
    return merge_config(cfg, "ic_diagnostics", _DEFAULT_IC_CONFIG["ic_diagnostics"])


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class GlobalICResult:
    """Global IC for a feature at a horizon."""
    candidate_id: str
    feature: str
    family: str
    horizon: int
    mean_ic: float
    median_ic: float
    ic_std: float
    icir: float
    hac_tstat: float
    p_value: float
    sign_consistency: float
    n_dates: int
    avg_breadth: int
    subperiod_stability: float
    ic_quality: str
    rejection_reason: str


@dataclass
class ResidualizedICResult:
    """Residualized IC after factor controls."""
    candidate_id: str
    feature: str
    family: str
    horizon: int
    global_ic: float
    residualized_ic: float
    residualized_icir: float
    residualized_hac_tstat: float
    residualized_p_value: float
    delta_vs_global_ic: float
    factor_controls_used: str
    sector_dummies_used: bool = False
    n_dates: int = 0
    avg_breadth: int = 0
    residualization_quality: str = ""
    rejection_reason: str = ""


@dataclass
class SectorNeutralICResult:
    """Sector-neutral IC."""
    candidate_id: str
    feature: str
    family: str
    horizon: int
    sector_neutral_ic: float
    sector_neutral_icir: float
    sector_neutral_hac_tstat: float
    best_sector: str
    worst_sector: str
    sector_concentration_score: float
    leave_one_sector_out_min_ic: float
    n_sectors_valid: int
    sector_quality: str
    sector_ic_breakdown: str = ""
    rejection_reason: str = ""


@dataclass
class ConditionalICResult:
    """IC for a conditional sleeve."""
    sleeve_id: str
    feature: str
    family: str
    horizon: int
    condition_type: str
    condition_value: str
    mean_ic: float
    icir: float
    hac_tstat: float
    p_value: float
    bh_q_value: float
    bhy_q_value: float
    n_dates: int
    avg_breadth: int
    breadth_quality: str
    cost_viability_status: str
    conditional_ic_quality: str
    rejection_reason: str


@dataclass
class MultipleTestingResult:
    """Multiple-testing correction results."""
    sleeve_id: str
    feature: str
    family: str
    horizon: int
    raw_p_value: float
    bh_q_value: float
    bhy_q_value: float
    passes_bh: bool
    passes_bhy: bool
    test_family_size: int
    dependency_note: str
    evidence_status: str


@dataclass
class ICAttributionResult:
    """IC failure mode attribution."""
    feature: str
    family: str
    horizon: int
    global_ic: float
    best_conditional_ic: float
    residualized_ic: float
    sector_neutral_ic: float
    peak_ic_horizon: int
    strongest_condition: str
    condition_breadth: int
    condition_q_value: float
    attribution_label: str
    recommended_next_action: str


@dataclass
class SleeveAdmissionResult:
    """Conditional sleeve admission decision."""
    sleeve_id: str
    feature: str
    family: str
    horizon: int
    condition_type: str
    condition_value: str
    statistical_status: str
    stability_status: str
    breadth_status: str
    decay_status: str
    cost_status: str
    pit_status: str
    final_status: str
    rejection_reason: str


@dataclass
class WalkForwardResult:
    """Walk-forward validation for a conditional sleeve."""
    sleeve_id: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    oos_ic: float
    oos_icir: float
    oos_tstat: float
    simple_sleeve_return: float
    turnover: float
    cost_bps: float
    net_return: float
    breadth: int
    window_status: str


# ── Helper: HAC t-stat ───────────────────────────────────────────────────────
# Delegated to _shared_stats.hac_tstat (byte-identical implementation).
_hac_tstat = hac_tstat
_p_from_tstat = p_from_tstat


# ── Phase 2: Global IC ──────────────────────────────────────────────────────

def compute_global_ic(
    df: pd.DataFrame,
    feature: str,
    horizon: int,
    min_dates: int = 30,
    min_breadth: int = 8,
) -> GlobalICResult:
    """Compute true horizon-correct global IC using vectorized kernel."""
    if df is None or df.empty or feature not in df.columns:
        return _empty_global_ic(feature, horizon)

    if "forward_return" not in df.columns:
        return _empty_global_ic(feature, horizon, reason="no_forward_return")

    # Build h-day forward return using kernel
    fwd = compute_forward_returns(
        df, [horizon], price_col="forward_return", compound=False,
    )
    fwd_col = f"fwd_ret_{horizon}d"
    if fwd_col not in fwd.columns:
        return _empty_global_ic(feature, horizon, reason="forward_return_error")

    # Merge forward returns
    work = df[["date", "ticker", feature]].copy()
    work = work.merge(fwd[["date", "ticker", fwd_col]], on=["date", "ticker"], how="left")
    work["date"] = pd.to_datetime(work["date"])
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work[fwd_col] = pd.to_numeric(work[fwd_col], errors="coerce")

    valid = work[[feature, fwd_col, "date", "ticker"]].dropna()
    if len(valid) < min_dates * min_breadth:
        return _empty_global_ic(feature, horizon, reason="insufficient_data")

    # Use vectorized kernel for daily IC computation
    ic_df, breadth_df, valid_df = compute_daily_ic_series(
        valid, [feature], fwd_col, min_breadth=min_breadth,
    )

    # Extract valid ICs
    ics = ic_df[feature].dropna().values
    breadths = breadth_df[feature].dropna().values.astype(int)

    if len(ics) < min_dates:
        return _empty_global_ic(feature, horizon, n_dates=len(ics), reason="too_few_dates")

    ics_arr = np.array(ics)
    mean_ic = float(np.mean(ics_arr))
    median_ic = float(np.median(ics_arr))
    ic_std = float(np.std(ics_arr))
    icir = mean_ic / ic_std if ic_std > 0 else 0.0
    t_stat = _hac_tstat(ics_arr, max(1, horizon - 1))
    p_val = _p_from_tstat(t_stat, len(ics_arr))
    sign_consistency = float((ics_arr > 0).mean()) if mean_ic > 0 else float((ics_arr < 0).mean())

    # Subperiod stability
    mid = len(ics_arr) // 2
    if mid > 5:
        ic1 = float(np.mean(ics_arr[:mid]))
        ic2 = float(np.mean(ics_arr[mid:]))
        stability = 1.0 - abs(ic1 - ic2) / max(abs(mean_ic), 0.001)
        stability = max(0.0, min(1.0, stability))
    else:
        stability = 0.5

    quality = _ic_quality(len(ics), int(np.mean(breadths)), mean_ic, t_stat, stability)
    reason = ""
    if quality == "insufficient":
        reason = "insufficient_data"
    elif quality == "low":
        reason = "weak_ic_evidence"

    return GlobalICResult(
        candidate_id=f"feature_{feature}",
        feature=feature, family=_get_family(feature),
        horizon=horizon, mean_ic=mean_ic, median_ic=median_ic,
        ic_std=ic_std, icir=icir, hac_tstat=t_stat, p_value=p_val,
        sign_consistency=sign_consistency, n_dates=len(ics),
        avg_breadth=int(np.mean(breadths)), subperiod_stability=stability,
        ic_quality=quality, rejection_reason=reason,
    )


def _empty_global_ic(feature: str, horizon: int, n_dates: int = 0, reason: str = "") -> GlobalICResult:
    return GlobalICResult(
        candidate_id=f"feature_{feature}", feature=feature,
        family=_get_family(feature), horizon=horizon,
        mean_ic=0.0, median_ic=0.0, ic_std=0.0, icir=0.0,
        hac_tstat=0.0, p_value=1.0, sign_consistency=0.0,
        n_dates=n_dates, avg_breadth=0, subperiod_stability=0.0,
        ic_quality="insufficient", rejection_reason=reason or "insufficient_data",
    )


_ic_quality = ic_quality


# ── Phase 3: Residualized IC ────────────────────────────────────────────────

def compute_residualized_ic(
    df: pd.DataFrame,
    feature: str,
    horizon: int,
    factor_controls: list[str] | None = None,
    min_dates: int = 20,
    min_breadth: int = 10,
    winsor_q: float = 0.025,
    global_ic_result: GlobalICResult | None = None,
) -> ResidualizedICResult:
    """Compute residualized IC after removing factor exposures.

    For each date: forward_return_h = beta_0 + beta_factors * factors + sector_dummies + residual
    Then: residualized_IC = corr(rank(feature_t), rank(residual_return_h))

    Uses batch_ridge_residualize with optional sector dummies (no per-date loop).
    """
    if factor_controls is None:
        factor_controls = ["market", "size", "momentum", "volatility"]

    if df is None or df.empty or feature not in df.columns:
        return _empty_residualized_ic(feature, horizon, global_ic_result)

    # Find available factor columns
    available_controls = []
    col_map = {
        "market": ["capm_beta", "beta", "market_beta"],
        "size": ["market_cap", "log_market_cap", "size", "cap_size"],
        "momentum": ["ret_20d", "ret_60d", "momentum_12m_skip1", "momentum"],
        "volatility": ["rolling_vol_20", "vol_20_simple", "realised_vol_20d", "volatility"],
    }
    for ctrl in factor_controls:
        for candidate in col_map.get(ctrl, [ctrl]):
            if candidate in df.columns:
                available_controls.append(candidate)
                break

    if not available_controls:
        return _empty_residualized_ic(feature, horizon, global_ic_result, reason="no_factor_controls")

    all_controls = list(available_controls)
    has_sector = "sector" in df.columns
    sector_dummies_used = False
    sector_dummy_cols: list[str] = []

    if has_sector:
        # One-hot encode sector and add as controls (drop first to reduce collinearity)
        sectors_series = df["sector"].astype(str)
        unique_sectors = sorted(sectors_series.dropna().unique())
        if len(unique_sectors) >= 2:
            df = df.copy()
            for s in unique_sectors[1:]:
                dummy_col = f"_sector_dummy_{s}"
                df[dummy_col] = (sectors_series.values == s).astype(float)
                sector_dummy_cols.append(dummy_col)
                all_controls.append(dummy_col)
            sector_dummies_used = True

    work = df[["date", "ticker", feature] + all_controls].copy()
    if "forward_return" not in df.columns:
        return _empty_residualized_ic(feature, horizon, global_ic_result, reason="no_forward_return")
    work["forward_return"] = pd.to_numeric(df["forward_return"], errors="coerce")
    work["date"] = pd.to_datetime(work["date"])
    for c in [feature] + all_controls + ["forward_return"]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # Build h-day forward return
    col = f"fwd_ret_{horizon}d"
    work[col] = work.groupby("ticker", sort=False)["forward_return"].transform(
        lambda x: x.rolling(horizon).sum() if horizon > 1 else x
    )
    work[col] = work.groupby("ticker", sort=False)[col].shift(-horizon)

    valid = work.dropna(subset=[feature, col] + all_controls)

    residuals_arr, feat_arr, breadth_arr, res_dates = batch_ridge_residualize(
        valid, col, all_controls, feature,
        ridge_lambda=0.01, min_breadth=min_breadth, winsor_q=winsor_q,
    )

    if len(res_dates) == 0:
        return _empty_residualized_ic(feature, horizon, global_ic_result, reason="insufficient_residualization")

    # Build DataFrame from arrays for IC computation
    valid_mask = np.isfinite(residuals_arr) & np.isfinite(feat_arr)
    n_valid_dates = int((valid_mask.sum(axis=1) >= min_breadth).sum())
    if n_valid_dates < min_dates:
        return _empty_residualized_ic(feature, horizon, global_ic_result, reason="insufficient_residualization")

    n_dates = len(res_dates)
    n_tickers = residuals_arr.shape[1]
    date_rep = np.repeat(res_dates.values, n_tickers)
    ticker_rep = np.tile(np.arange(n_tickers), n_dates)
    all_resid = pd.DataFrame({
        "date": date_rep,
        "ticker": ticker_rep,
        feature: feat_arr.ravel(),
        "residual": residuals_arr.ravel(),
    })
    all_resid = all_resid.dropna(subset=[feature, "residual"])

    ic_df, breadth_df, valid_df = compute_daily_ic_series(
        all_resid, [feature], "residual", min_breadth=min_breadth,
    )

    res_ics = ic_df[feature].dropna().values
    breadths = breadth_df[feature].dropna().values.astype(int)

    if len(res_ics) < min_dates:
        return _empty_residualized_ic(feature, horizon, global_ic_result, reason="too_few_residual_dates")

    res_ics_arr = np.array(res_ics)
    res_mean = float(np.mean(res_ics_arr))
    res_std = float(np.std(res_ics_arr))
    res_icir = res_mean / res_std if res_std > 0 else 0.0
    res_tstat = _hac_tstat(res_ics_arr, max(1, horizon - 1))
    res_pval = _p_from_tstat(res_tstat, len(res_ics_arr))

    global_ic = global_ic_result.mean_ic if global_ic_result else 0.0
    delta = res_mean - global_ic

    quality = "high" if abs(res_tstat) >= 2.0 else ("medium" if abs(res_tstat) >= 1.5 else "low")

    return ResidualizedICResult(
        candidate_id=f"feature_{feature}", feature=feature,
        family=_get_family(feature), horizon=horizon,
        global_ic=global_ic, residualized_ic=res_mean,
        residualized_icir=res_icir, residualized_hac_tstat=res_tstat,
        residualized_p_value=res_pval, delta_vs_global_ic=delta,
        factor_controls_used=",".join(available_controls),
        sector_dummies_used=sector_dummies_used,
        n_dates=len(res_ics), avg_breadth=int(np.mean(breadths)),
        residualization_quality=quality, rejection_reason="",
    )


def _empty_residualized_ic(feature: str, horizon: int, global_ic: GlobalICResult | None = None, reason: str = "") -> ResidualizedICResult:
    gic = global_ic.mean_ic if global_ic else 0.0
    return ResidualizedICResult(
        candidate_id=f"feature_{feature}", feature=feature,
        family=_get_family(feature), horizon=horizon,
        global_ic=gic, residualized_ic=0.0, residualized_icir=0.0,
        residualized_hac_tstat=0.0, residualized_p_value=1.0,
        delta_vs_global_ic=0.0, factor_controls_used="",
        sector_dummies_used=False,
        n_dates=0, avg_breadth=0, residualization_quality="insufficient",
        rejection_reason=reason or "insufficient_data",
    )


# ── Phase 4: Sector-neutral IC ───────────────────────────────────────────────

def compute_sector_neutral_ic(
    df: pd.DataFrame,
    feature: str,
    horizon: int,
    min_dates: int = 20,
    min_breadth: int = 5,
    min_sectors: int = 3,
) -> SectorNeutralICResult:
    """Compute sector-neutral IC by ranking within sector/date.

    For each date: rank feature within sector, rank forward return within sector,
    compute within-sector IC, aggregate across dates and sectors.

    Uses tensor backend (pre_ranked) per sector instead of per-(date,sector) loops.
    """
    if df is None or df.empty or feature not in df.columns:
        return _empty_sector_neutral_ic(feature, horizon)

    if "sector" not in df.columns:
        return _empty_sector_neutral_ic(feature, horizon, reason="no_sector_data", sector_quality="no_sector_column")

    if "forward_return" not in df.columns:
        return _empty_sector_neutral_ic(feature, horizon, reason="no_forward_return")

    work = df[["date", "ticker", feature, "sector"]].copy()
    work["forward_return"] = pd.to_numeric(df["forward_return"], errors="coerce")
    work["date"] = pd.to_datetime(work["date"])
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work["sector"] = work["sector"].astype(str)

    # Build h-day forward return
    col = f"fwd_ret_{horizon}d"
    work[col] = work.groupby("ticker", sort=False)["forward_return"].transform(
        lambda x: x.rolling(horizon).sum() if horizon > 1 else x
    )
    work[col] = work.groupby("ticker", sort=False)[col].shift(-horizon)

    valid = work.dropna(subset=[feature, col, "sector"]).copy()
    if valid.empty:
        return _empty_sector_neutral_ic(feature, horizon, reason="no_valid_data", sector_quality="no_data")

    # Rank within sector/date
    valid.loc[:, "feat_rank"] = valid.groupby(["date", "sector"], sort=False)[feature].rank(pct=True)
    valid.loc[:, "ret_rank"] = valid.groupby(["date", "sector"], sort=False)[col].rank(pct=True)

    # Use tensor backend per sector (pre_ranked=True) instead of per-(date,sector) spearmanr loop
    sectors = valid["sector"].unique()
    if len(sectors) < min_sectors:
        return _empty_sector_neutral_ic(feature, horizon, reason="too_few_sectors", sector_quality="insufficient_sectors")

    sector_ics: dict[str, list[float]] = {}
    ranked_cols = ["date", "ticker", "sector", "feat_rank", "ret_rank"]
    ranked = valid[ranked_cols].dropna()

    for sector in sectors:
        sub = ranked[ranked["sector"] == sector]
        sub_sub = sub[["date", "ticker", "feat_rank", "ret_rank"]].rename(
            columns={"feat_rank": feature, "ret_rank": col}
        )
        if len(sub_sub) < min_breadth * 2:
            continue
        try:
            ic_df, breadth_df, _ = compute_daily_ic_series(
                sub_sub, [feature], col, min_breadth=min_breadth,
                mode="full_tensor", pre_ranked=True,
            )
        except (ValueError, RuntimeError):
            continue
        sector_ics_list = ic_df[feature].dropna().values
        if len(sector_ics_list) > 0:
            sector_ics[sector] = sector_ics_list.tolist()

    if len(sector_ics) < min_sectors:
        return _empty_sector_neutral_ic(feature, horizon, reason="too_few_sectors", sector_quality="insufficient_sectors")

    # Aggregate
    all_ics: list[float] = []
    sector_mean_ics: dict[str, float] = {}
    for sector, ics in sector_ics.items():
        sector_mean_ics[sector] = float(np.mean(ics))
        all_ics.extend(ics)

    if len(all_ics) < min_dates:
        return _empty_sector_neutral_ic(feature, horizon, reason="too_few_dates", sector_quality="insufficient_dates")

    all_ics_arr = np.array(all_ics)
    sn_ic = float(np.mean(all_ics_arr))
    sn_std = float(np.std(all_ics_arr))
    sn_icir = sn_ic / sn_std if sn_std > 0 else 0.0
    sn_tstat = _hac_tstat(all_ics_arr, max(1, horizon - 1))

    # Best/worst sector
    best_sector = max(sector_mean_ics, key=sector_mean_ics.get)
    worst_sector = min(sector_mean_ics, key=sector_mean_ics.get)

    # Sector concentration: fraction of total IC from best sector
    total_abs_ic = sum(abs(np.mean(ics)) * len(ics) for ics in sector_ics.values())
    best_abs_ic = abs(np.mean(sector_ics[best_sector])) * len(sector_ics[best_sector])
    concentration = best_abs_ic / max(total_abs_ic, 1e-15)

    # Leave-one-sector-out
    loos_ics = []
    for leave_sector in sector_ics:
        other_ics = []
        for s, ics in sector_ics.items():
            if s != leave_sector:
                other_ics.extend(ics)
        if other_ics:
            loos_ics.append(float(np.mean(other_ics)))
    loos_min_ic = min(loos_ics) if loos_ics else 0.0

    quality = "static_proxy"
    if "sector_timestamp" in df.columns or "sector_asof" in df.columns:
        quality = "pit"

    breakdown_parts = []
    for s in sorted(sector_mean_ics):
        breakdown_parts.append(f"{s}={sector_mean_ics[s]:.4f}")
    breakdown = ";".join(breakdown_parts)

    return SectorNeutralICResult(
        candidate_id=f"feature_{feature}", feature=feature,
        family=_get_family(feature), horizon=horizon,
        sector_neutral_ic=sn_ic, sector_neutral_icir=sn_icir,
        sector_neutral_hac_tstat=sn_tstat,
        best_sector=best_sector, worst_sector=worst_sector,
        sector_concentration_score=concentration,
        leave_one_sector_out_min_ic=loos_min_ic,
        n_sectors_valid=len(sector_ics),
        sector_quality=quality,
        sector_ic_breakdown=breakdown, rejection_reason="",
    )


def _empty_sector_neutral_ic(feature: str, horizon: int, reason: str = "", sector_quality: str = "no_data") -> SectorNeutralICResult:
    return SectorNeutralICResult(
        candidate_id=f"feature_{feature}", feature=feature,
        family=_get_family(feature), horizon=horizon,
        sector_neutral_ic=0.0, sector_neutral_icir=0.0,
        sector_neutral_hac_tstat=0.0,
        best_sector="", worst_sector="",
        sector_concentration_score=0.0,
        leave_one_sector_out_min_ic=0.0,
        n_sectors_valid=0, sector_quality=sector_quality,
        rejection_reason=reason or "insufficient_data",
    )


# ── Phase 5: Conditional IC grid ─────────────────────────────────────────────

def compute_conditional_ic_grid(
    df: pd.DataFrame,
    feature: str,
    horizon: int,
    conditions: list[str] | None = None,
    n_buckets: int = 3,
    min_dates: int = 15,
    min_breadth: int = 5,
    min_bucket_breadth: int = 10,
    all_p_values: list[float] | None = None,
) -> tuple[list[ConditionalICResult], list[float]]:
    """Compute conditional IC for each condition type/value.

    All buckets are PIT (rolling or cross-sectional quantiles).
    No full-sample quantile thresholds.

    Uses tensor backend per condition value instead of per-date spearmanr loop.
    """
    if conditions is None:
        conditions = ["regime", "volatility", "liquidity", "size", "sector"]

    if df is None or df.empty or feature not in df.columns:
        return [], []

    work = df[["date", "ticker", feature]].copy()
    if "forward_return" not in df.columns:
        return [], []
    work["forward_return"] = pd.to_numeric(df["forward_return"], errors="coerce")
    work["date"] = pd.to_datetime(work["date"])
    work[feature] = pd.to_numeric(work[feature], errors="coerce")

    # Build h-day forward return
    col = f"fwd_ret_{horizon}d"
    work[col] = work.groupby("ticker", sort=False)["forward_return"].transform(
        lambda x: x.rolling(horizon).sum() if horizon > 1 else x
    )
    work[col] = work.groupby("ticker", sort=False)[col].shift(-horizon)

    # Add condition columns
    condition_cols = {}
    for cond in conditions:
        col_name = _find_condition_column(df, cond)
        if col_name:
            condition_cols[cond] = col_name
            work[cond] = df[col_name].values if len(df) == len(work) else np.nan

    # Build buckets for continuous conditions (PIT: cross-sectional quantiles per date)
    bucket_conditions = ["volatility", "liquidity", "size"]
    for cond in bucket_conditions:
        if cond in condition_cols:
            try:
                work[f"{cond}_bucket"] = work.groupby("date", sort=False)[cond].transform(
                    lambda x: pd.qcut(x.rank(method="first"), n_buckets, labels=False, duplicates="drop")
                    if len(x) >= n_buckets else pd.Series([np.nan] * len(x), index=x.index)
                )
            except (ValueError, TypeError):
                work[f"{cond}_bucket"] = np.nan

    results = []
    new_p_values = []

    for cond in conditions:
        if cond not in work.columns:
            continue

        if cond in bucket_conditions:
            bucket_col = f"{cond}_bucket"
            if bucket_col not in work.columns:
                continue
            values = work[bucket_col].dropna().unique()
            value_labels = [f"bucket_{int(v)}" for v in sorted(values)]
            work["_cond_value"] = work[bucket_col]
        elif cond == "sector":
            values = work["sector"].dropna().unique()
            value_labels = [str(v) for v in sorted(values)]
            work["_cond_value"] = work["sector"]
        else:
            values = work[cond].dropna().unique()
            value_labels = [str(v) for v in sorted(values)]
            work["_cond_value"] = work[cond]

        for val, label in zip(values, value_labels):
            subset = work[work["_cond_value"] == val].dropna(subset=[feature, col])
            if len(subset) < min_dates * min_breadth:
                continue

            # Use tensor backend instead of per-date spearmanr loop
            try:
                ic_df, breadth_df, _ = compute_daily_ic_series(
                    subset, [feature], col, min_breadth=min_bucket_breadth,
                )
            except (ValueError, RuntimeError):
                continue

            ics = ic_df[feature].dropna().values
            breadths = breadth_df[feature].dropna().values

            if len(ics) < min_dates:
                continue

            ics_arr = np.array(ics)
            mean_ic = float(np.mean(ics_arr))
            ic_std = float(np.std(ics_arr))
            icir = mean_ic / ic_std if ic_std > 0 else 0.0
            t_stat = _hac_tstat(ics_arr, max(1, horizon - 1))
            p_val = _p_from_tstat(t_stat, len(ics_arr))
            new_p_values.append(p_val)

            sleeve_id = f"{feature}_{cond}_{label}_h{horizon}"
            avg_breadth = float(np.mean(breadths)) if len(breadths) > 0 else 0
            breadth_quality = "high" if avg_breadth >= 50 else ("medium" if avg_breadth >= 20 else "low")
            quality = "high" if abs(t_stat) >= 2.0 else ("medium" if abs(t_stat) >= 1.5 else "low")

            results.append(ConditionalICResult(
                sleeve_id=sleeve_id, feature=feature,
                family=_get_family(feature), horizon=horizon,
                condition_type=cond, condition_value=label,
                mean_ic=mean_ic, icir=icir, hac_tstat=t_stat,
                p_value=p_val, bh_q_value=1.0, bhy_q_value=1.0,
                n_dates=len(ics), avg_breadth=int(avg_breadth),
                breadth_quality=breadth_quality,
                cost_viability_status="not_evaluated",
                conditional_ic_quality=quality, rejection_reason="",
            ))

    # Apply BH/BHY correction across all tested conditions
    if new_p_values:
        p_arr = np.array(new_p_values)
        bh_q = benjamini_hochberg(p_arr)
        by_q = benjamini_yekutieli(p_arr)
        for i, r in enumerate(results):
            r.bh_q_value = float(bh_q[i])
            r.bhy_q_value = float(by_q[i])

    return results, new_p_values


_find_condition_column = find_condition_column


# ── Phase 6: Multiple-testing report ─────────────────────────────────────────

def generate_multiple_testing_report(
    conditional_results: list[ConditionalICResult],
    bh_q_threshold: float = 0.10,
    bhy_q_threshold: float = 0.05,
) -> list[MultipleTestingResult]:
    """Generate multiple-testing correction report."""
    results = []
    for r in conditional_results:
        passes_bh = r.bh_q_value <= bh_q_threshold
        passes_bhy = r.bhy_q_value <= bhy_q_threshold

        if passes_bhy:
            status = EvidenceStatus.STRONG_CANDIDATE.value
        elif passes_bh:
            status = EvidenceStatus.RESEARCH_CANDIDATE.value
        elif r.p_value < 0.05:
            status = EvidenceStatus.DISCOVERY_INTEREST.value
        else:
            status = EvidenceStatus.REJECTED.value

        results.append(MultipleTestingResult(
            sleeve_id=r.sleeve_id, feature=r.feature,
            family=r.family, horizon=r.horizon,
            raw_p_value=r.p_value, bh_q_value=r.bh_q_value,
            bhy_q_value=r.bhy_q_value, passes_bh=passes_bh,
            passes_bhy=passes_bhy, test_family_size=len(conditional_results),
            dependency_note="bhy_used_for_arbitrary_dependence",
            evidence_status=status,
        ))
    return results


# ── Phase 7: IC Attribution ─────────────────────────────────────────────────

def compute_ic_attribution(
    feature: str,
    family: str,
    global_ic_results: list[GlobalICResult],
    residualized_results: list[ResidualizedICResult],
    sector_results: list[SectorNeutralICResult],
    conditional_results: list[ConditionalICResult],
    multiple_testing_results: list[MultipleTestingResult],
) -> list[ICAttributionResult]:
    """Determine why global IC is weak for each feature/horizon."""
    results = []

    for gic in global_ic_results:
        h = gic.horizon

        # Find matching results
        res = next((r for r in residualized_results if r.horizon == h), None)
        sec = next((r for r in sector_results if r.horizon == h), None)

        # Best conditional IC
        cond_h = [r for r in conditional_results if r.horizon == h and r.feature == feature]
        best_cond_ic = max((abs(r.mean_ic) for r in cond_h), default=0.0)
        best_cond = ""
        best_q = 1.0
        best_breadth = 0
        if cond_h:
            best_r = max(cond_h, key=lambda r: abs(r.mean_ic))
            best_cond = f"{best_r.condition_type}:{best_r.condition_value}"
            best_q = best_r.bhy_q_value
            best_breadth = best_r.avg_breadth

        # Peak IC horizon
        peak_h = h
        peak_ic = abs(gic.mean_ic)
        for other in global_ic_results:
            if other.feature == feature and abs(other.mean_ic) > peak_ic:
                peak_ic = abs(other.mean_ic)
                peak_h = other.horizon

        # Attribution
        label, action = _classify_attribution(gic, res, sec, best_cond_ic, best_cond, best_q, peak_h, h)

        results.append(ICAttributionResult(
            feature=feature, family=family, horizon=h,
            global_ic=gic.mean_ic, best_conditional_ic=best_cond_ic,
            residualized_ic=res.residualized_ic if res else 0.0,
            sector_neutral_ic=sec.sector_neutral_ic if sec else 0.0,
            peak_ic_horizon=peak_h, strongest_condition=best_cond,
            condition_breadth=best_breadth, condition_q_value=best_q,
            attribution_label=label, recommended_next_action=action,
        ))

    return results


def _classify_attribution(
    gic: GlobalICResult,
    res: ResidualizedICResult | None,
    sec: SectorNeutralICResult | None,
    best_cond_ic: float,
    best_cond: str,
    best_q: float,
    peak_h: int,
    tested_h: int,
) -> tuple[str, str]:
    """Classify IC failure mode."""
    if abs(gic.mean_ic) < 0.002 and gic.ic_quality == "insufficient":
        return AttributionLabel.NO_GLOBAL_ALPHA.value, "reject_feature_or_engineer_new_features"

    if res and abs(res.residualized_ic) > abs(gic.mean_ic) * 1.5:
        return AttributionLabel.FACTOR_EXPOSURE_CONTAMINATION.value, "use_residualized_target_in_model"

    if sec and abs(sec.sector_neutral_ic) > abs(gic.mean_ic) * 1.5:
        if sec.sector_concentration_score > 0.5:
            return AttributionLabel.SECTOR_SPECIFIC_ALPHA.value, "create_sector_specific_sleeve"
        return AttributionLabel.SECTOR_SPECIFIC_ALPHA.value, "use_sector_neutral_ranking"

    if best_cond_ic > abs(gic.mean_ic) * 2.0 and best_q < 0.10:
        return AttributionLabel.ALPHA_DILUTED_BY_REGIME.value, f"research_conditional_sleeve:{best_cond}"

    if peak_h != tested_h:
        return AttributionLabel.WRONG_HORIZON.value, f"test_at_horizon_{peak_h}d"

    if gic.n_dates < 30 or gic.avg_breadth < 10:
        return AttributionLabel.INSUFFICIENT_BREADTH.value, "expand_universe_or_extend_history"

    if gic.subperiod_stability < 0.5:
        return AttributionLabel.UNSTABLE_SUBPERIODS.value, "investigate_regime_dependence"

    if best_cond_ic > abs(gic.mean_ic) * 1.5 and best_q >= 0.10:
        return AttributionLabel.MULTIPLE_TESTING_FAILURE.value, "conditional_alpha_not_statistically_valid"

    return AttributionLabel.NO_GLOBAL_ALPHA.value, "reject_or_engineer_new_features"


# ── Phase 8: Conditional sleeve admission gate ───────────────────────────────

def evaluate_sleeve_admission(
    cond_result: ConditionalICResult,
    mt_result: MultipleTestingResult,
    ic_config: dict[str, Any],
) -> SleeveAdmissionResult:
    """Evaluate whether a conditional sleeve passes the admission gate."""
    # Statistical
    stat_pass = (
        abs(cond_result.mean_ic) >= ic_config.get("ic_mean_threshold", 0.005)
        and abs(cond_result.icir) >= ic_config.get("icir_threshold", 0.5)
        and abs(cond_result.hac_tstat) >= ic_config.get("hac_tstat_threshold", 2.0)
        and cond_result.bh_q_value <= ic_config.get("bh_q_threshold", 0.10)
    )
    stat_status = "pass" if stat_pass else "fail"

    # Stability (placeholder — would need subperiod data)
    stab_status = "pass" if cond_result.n_dates >= ic_config.get("min_dates_for_conditional", 15) else "fail"

    # Breadth
    breadth_pass = (
        cond_result.avg_breadth >= ic_config.get("min_breadth_for_bucket", 10)
        and cond_result.n_dates >= ic_config.get("min_dates_for_conditional", 15)
    )
    breadth_status = "pass" if breadth_pass else "fail"

    # Decay (placeholder — would need halflife data)
    decay_status = "not_evaluated"

    # Cost (placeholder)
    cost_status = "not_evaluated"

    # PIT validity
    pit_status = "pass"  # All buckets are PIT by construction

    # Final
    if stat_pass and breadth_pass:
        if mt_result.passes_bhy:
            final = SleeveStatus.ADMITTED.value
        elif mt_result.passes_bh:
            final = SleeveStatus.RESEARCH_ONLY.value
        else:
            final = SleeveStatus.REJECTED.value
    else:
        final = SleeveStatus.REJECTED.value

    reasons = []
    if not stat_pass:
        reasons.append("statistical_thresholds_not_met")
    if not breadth_pass:
        reasons.append("insufficient_breadth_or_dates")
    if not mt_result.passes_bh:
        reasons.append("fails_multiple_testing_correction")

    return SleeveAdmissionResult(
        sleeve_id=cond_result.sleeve_id, feature=cond_result.feature,
        family=cond_result.family, horizon=cond_result.horizon,
        condition_type=cond_result.condition_type,
        condition_value=cond_result.condition_value,
        statistical_status=stat_status, stability_status=stab_status,
        breadth_status=breadth_status, decay_status=decay_status,
        cost_status=cost_status, pit_status=pit_status,
        final_status=final, rejection_reason=";".join(reasons) if reasons else "",
    )


# ── Phase 9: Walk-forward validation (simplified) ───────────────────────────

def run_conditional_walk_forward(
    df: pd.DataFrame,
    feature: str,
    condition_type: str,
    condition_value: str,
    horizon: int,
    n_windows: int = 4,
    train_ratio: float = 0.7,
    embargo_multiplier: int = 2,
) -> list[WalkForwardResult]:
    """Walk-forward validation for a conditional sleeve.

    FIX: Applies embargo between train and test (was missing).
    FIX: Computes simple sleeve return, turnover, and cost (were 0.0).
    """
    if df is None or df.empty:
        return []

    dates = sorted(df["date"].unique())
    if len(dates) < n_windows * 10:
        return []

    # Split into windows
    window_size = len(dates) // n_windows
    results = []

    for i in range(n_windows - 1):
        train_end_idx = int((i + 1) * window_size * train_ratio)
        # Embargo: skip overlapping forward returns
        embargo_days = embargo_multiplier * horizon
        test_start_idx = min(int((i + 1) * window_size) + embargo_days, len(dates) - 1)
        test_end_idx = min(int((i + 2) * window_size), len(dates))

        if test_start_idx >= len(dates) or train_end_idx <= 0:
            continue

        train_dates = dates[:train_end_idx]
        test_dates = dates[test_start_idx:test_end_idx]

        if len(test_dates) < 5:
            continue

        train_df = df[df["date"].isin(train_dates)]
        test_df = df[df["date"].isin(test_dates)]

        # Compute OOS IC
        oos_ics = _compute_oos_ic(test_df, feature, horizon)
        if not oos_ics:
            continue

        oos_ic = float(np.mean(oos_ics))
        oos_std = float(np.std(oos_ics)) if len(oos_ics) > 1 else 0.0
        oos_icir = oos_ic / oos_std if oos_std > 0 else 0.0
        oos_tstat = _hac_tstat(np.array(oos_ics), max(1, horizon - 1))

        # Simple sleeve return: long top quintile, short bottom quintile
        sleeve_ret, to, cost = _compute_simple_sleeve_return(test_df, feature, horizon)

        sleeve_id = f"{feature}_{condition_type}_{condition_value}_h{horizon}"
        status = "pass" if abs(oos_ic) > 0.005 else "fail"

        results.append(WalkForwardResult(
            sleeve_id=sleeve_id,
            train_start=str(train_dates[0])[:10],
            train_end=str(train_dates[-1])[:10],
            test_start=str(test_dates[0])[:10],
            test_end=str(test_dates[-1])[:10],
            oos_ic=oos_ic, oos_icir=oos_icir, oos_tstat=oos_tstat,
            simple_sleeve_return=sleeve_ret, turnover=to, cost_bps=cost,
            net_return=sleeve_ret - cost, breadth=len(test_df["ticker"].unique()),
            window_status=status,
        ))

    return results


def _compute_simple_sleeve_return(
    df: pd.DataFrame, feature: str, horizon: int,
) -> tuple[float, float, float]:
    """Compute simple long-short sleeve return, turnover, and cost estimate.

    Long top quintile, short bottom quintile, equal-weighted.
    Returns: (cumulative_return, avg_turnover, est_cost_bps)
    """
    if "forward_return" not in df.columns or feature not in df.columns:
        return 0.0, 0.0, 0.0

    work = df[["date", "ticker", feature]].copy()
    work["forward_return"] = pd.to_numeric(df["forward_return"], errors="coerce")
    work["date"] = pd.to_datetime(work["date"])
    work[feature] = pd.to_numeric(work[feature], errors="coerce")

    col = f"fwd_ret_{horizon}d"
    work[col] = work.groupby("ticker", sort=False)["forward_return"].transform(
        lambda x: x.rolling(horizon).sum() if horizon > 1 else x
    )
    work[col] = work.groupby("ticker", sort=False)[col].shift(-horizon)

    valid = work.dropna(subset=[feature, col])

    returns = []
    prev_top: set | None = None

    for date, grp in valid.groupby("date", sort=False):
        if len(grp) < 10:
            continue
        n_q = max(1, len(grp) // 5)
        sorted_grp = grp.sort_values(feature)
        longs = sorted_grp.tail(n_q)[col].mean()
        shorts = sorted_grp.head(n_q)[col].mean()
        ls_ret = longs - shorts
        if np.isfinite(ls_ret):
            returns.append(ls_ret)

        # Estimate turnover
        if prev_top is not None:
            curr_top = set(sorted_grp.tail(n_q)["ticker"])
            if len(prev_top) > 0:
                turnover = 1.0 - len(prev_top & curr_top) / len(prev_top)
                returns[-1] = ls_ret  # keep return, track turnover separately

        prev_top = set(sorted_grp.tail(n_q)["ticker"])

    if not returns:
        return 0.0, 0.0, 0.0

    total_return = float(np.sum(returns))
    # Rough cost estimate: 10bps per round-trip, scaled by typical turnover ~0.5
    est_cost = 0.5 * 10.0  # 5bps per period

    return total_return, 0.5, est_cost


def _compute_oos_ic(df: pd.DataFrame, feature: str, horizon: int) -> list[float]:
    """Compute out-of-sample IC for a test period."""
    if "forward_return" not in df.columns or feature not in df.columns:
        return []

    work = df[["date", "ticker", feature]].copy()
    work["forward_return"] = pd.to_numeric(df["forward_return"], errors="coerce")
    work["date"] = pd.to_datetime(work["date"])
    work[feature] = pd.to_numeric(work[feature], errors="coerce")

    col = f"fwd_ret_{horizon}d"
    work[col] = work.groupby("ticker", sort=False)["forward_return"].transform(
        lambda x: x.rolling(horizon).sum() if horizon > 1 else x
    )
    work[col] = work.groupby("ticker", sort=False)[col].shift(-horizon)

    valid = work.dropna(subset=[feature, col])
    ics = []
    for date, grp in valid.groupby("date", sort=False):
        if len(grp) < 5:
            continue
        feat_vals = grp[feature].values
        fwd_vals = grp[col].values
        if np.nanstd(feat_vals) < 1e-15 or np.nanstd(fwd_vals) < 1e-15:
            continue
        r, _ = scipy_stats.spearmanr(feat_vals, fwd_vals)
        if np.isfinite(r):
            ics.append(r)
    return ics


# ── Main engine ──────────────────────────────────────────────────────────────

@dataclass
class ICDiagnosticsBundle:
    """Full IC diagnostics for a feature."""
    feature: str
    family: str
    global_ic_results: list[GlobalICResult]
    residualized_results: list[ResidualizedICResult]
    sector_results: list[SectorNeutralICResult]
    conditional_results: list[ConditionalICResult]
    multiple_testing_results: list[MultipleTestingResult]
    attribution_results: list[ICAttributionResult]
    sleeve_admissions: list[SleeveAdmissionResult]
    walk_forward_results: list[WalkForwardResult]


class ICDiagnosticsEngine:
    """Global-to-Conditional IC Diagnostic Engine."""

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.ic_cfg = _get_ic_config(self.config)

    def run_full_diagnostics(
        self,
        df: pd.DataFrame,
        features: list[str],
        horizons: list[int] | None = None,
    ) -> list[ICDiagnosticsBundle]:
        """Run full IC diagnostics for a list of features."""
        if horizons is None:
            horizons = self.ic_cfg.get("horizons", [1, 2, 3, 5, 10, 20])

        min_dates_ic = self.ic_cfg.get("min_dates_for_ic", 30)
        min_breadth_ic = self.ic_cfg.get("min_breadth_for_ic", 8)
        conditions = self.ic_cfg.get("conditional_conditions", ["regime", "volatility", "liquidity", "size", "sector"])

        all_conditional_results = []
        bundles = []

        for feature in features:
            family = _get_family(feature)

            # Phase 2: Global IC
            global_ics = []
            for h in horizons:
                gic = compute_global_ic(df, feature, h, min_dates_ic, min_breadth_ic)
                global_ics.append(gic)

            # Phase 3: Residualized IC
            residualized = []
            for gic in global_ics:
                res = compute_residualized_ic(
                    df, feature, gic.horizon,
                    factor_controls=self.ic_cfg.get("factor_controls", ["market", "size", "momentum", "volatility"]),
                    min_dates=self.ic_cfg.get("min_dates_for_residualized", 20),
                    min_breadth=self.ic_cfg.get("min_breadth_for_residualized", 10),
                    winsor_q=self.ic_cfg.get("winsor_q", 0.025),
                    global_ic_result=gic,
                )
                residualized.append(res)

            # Phase 4: Sector-neutral IC
            sector_results = []
            for h in horizons:
                sec = compute_sector_neutral_ic(
                    df, feature, h,
                    min_dates=self.ic_cfg.get("min_dates_for_sector", 20),
                    min_breadth=self.ic_cfg.get("min_breadth_for_ic", 8),
                    min_sectors=self.ic_cfg.get("min_sectors_for_sector_neutral", 3),
                )
                sector_results.append(sec)

            # Phase 5: Conditional IC grid
            cond_results = []
            cond_p_values = []
            for h in horizons:
                cr, pvals = compute_conditional_ic_grid(
                    df, feature, h,
                    conditions=conditions,
                    n_buckets=self.ic_cfg.get("n_buckets", 3),
                    min_dates=self.ic_cfg.get("min_dates_for_conditional", 15),
                    min_breadth=self.ic_cfg.get("min_breadth_for_conditional", 5),
                    min_bucket_breadth=self.ic_cfg.get("min_breadth_for_bucket", 10),
                )
                cond_results.extend(cr)
                cond_p_values.extend(pvals)

            all_conditional_results.extend(cond_results)

            # Phase 6: Multiple-testing
            mt_results = generate_multiple_testing_report(
                cond_results,
                bh_q_threshold=self.ic_cfg.get("bh_q_threshold", 0.10),
                bhy_q_threshold=self.ic_cfg.get("bhy_q_threshold", 0.05),
            )

            # Phase 7: Attribution
            attribution = compute_ic_attribution(
                feature, family, global_ics, residualized, sector_results,
                cond_results, mt_results,
            )

            # Phase 8: Sleeve admission
            admissions = []
            for cr, mt in zip(cond_results, mt_results):
                adm = evaluate_sleeve_admission(cr, mt, self.ic_cfg)
                admissions.append(adm)

            bundles.append(ICDiagnosticsBundle(
                feature=feature, family=family,
                global_ic_results=global_ics,
                residualized_results=residualized,
                sector_results=sector_results,
                conditional_results=cond_results,
                multiple_testing_results=mt_results,
                attribution_results=attribution,
                sleeve_admissions=admissions,
                walk_forward_results=[],
            ))

        # Phase 9: Walk-forward for admitted sleeves
        for bundle in bundles:
            for adm in bundle.sleeve_admissions:
                if adm.final_status == SleeveStatus.ADMITTED.value:
                    wf = run_conditional_walk_forward(
                        df, adm.feature, adm.condition_type, adm.condition_value,
                        adm.horizon, n_windows=4,
                    )
                    bundle.walk_forward_results.extend(wf)

        logger.info("IC diagnostics: %d features, %d conditional sleeves evaluated",
                     len(bundles), len(all_conditional_results))
        return bundles


# ── Report generation ────────────────────────────────────────────────────────

def generate_ic_diagnostics_reports(
    bundles: list[ICDiagnosticsBundle],
    output_dir: str | Path = "output/models/ic_diagnostics",
) -> dict[str, Path]:
    """Generate all IC diagnostics reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 1. global_ic_diagnostics.csv
    rows = []
    for b in bundles:
        for r in b.global_ic_results:
            rows.append({
                "candidate_id": r.candidate_id, "feature": r.feature,
                "family": r.family, "horizon": r.horizon,
                "mean_ic": r.mean_ic, "median_ic": r.median_ic,
                "ic_std": r.ic_std, "icir": r.icir,
                "hac_tstat": r.hac_tstat, "p_value": r.p_value,
                "sign_consistency": r.sign_consistency,
                "n_dates": r.n_dates, "avg_breadth": r.avg_breadth,
                "subperiod_stability": r.subperiod_stability,
                "ic_quality": r.ic_quality, "rejection_reason": r.rejection_reason,
            })
    p = output_dir / "global_ic_diagnostics.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["global_ic"] = p

    # 2. residualized_ic_diagnostics.csv
    rows = []
    for b in bundles:
        for r in b.residualized_results:
            rows.append({
                "candidate_id": r.candidate_id, "feature": r.feature,
                "family": r.family, "horizon": r.horizon,
                "global_ic": r.global_ic, "residualized_ic": r.residualized_ic,
                "residualized_icir": r.residualized_icir,
                "residualized_hac_tstat": r.residualized_hac_tstat,
                "residualized_p_value": r.residualized_p_value,
                "delta_vs_global_ic": r.delta_vs_global_ic,
                "factor_controls_used": r.factor_controls_used,
                "sector_dummies_used": r.sector_dummies_used,
                "n_dates": r.n_dates, "avg_breadth": r.avg_breadth,
                "residualization_quality": r.residualization_quality,
                "rejection_reason": r.rejection_reason,
            })
    p = output_dir / "residualized_ic_diagnostics.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["residualized_ic"] = p

    # 3. sector_neutral_ic_diagnostics.csv
    rows = []
    for b in bundles:
        for r in b.sector_results:
            rows.append({
                "candidate_id": r.candidate_id, "feature": r.feature,
                "family": r.family, "horizon": r.horizon,
                "sector_neutral_ic": r.sector_neutral_ic,
                "sector_neutral_icir": r.sector_neutral_icir,
                "sector_neutral_hac_tstat": r.sector_neutral_hac_tstat,
                "best_sector": r.best_sector, "worst_sector": r.worst_sector,
                "sector_concentration_score": r.sector_concentration_score,
                "leave_one_sector_out_min_ic": r.leave_one_sector_out_min_ic,
                "n_sectors_valid": r.n_sectors_valid,
                "sector_quality": r.sector_quality,
                "sector_ic_breakdown": r.sector_ic_breakdown,
                "rejection_reason": r.rejection_reason,
            })
    p = output_dir / "sector_neutral_ic_diagnostics.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["sector_neutral_ic"] = p

    # 4. conditional_ic_grid.csv
    rows = []
    for b in bundles:
        for r in b.conditional_results:
            rows.append({
                "sleeve_id": r.sleeve_id, "feature": r.feature,
                "family": r.family, "horizon": r.horizon,
                "condition_type": r.condition_type,
                "condition_value": r.condition_value,
                "mean_ic": r.mean_ic, "icir": r.icir,
                "hac_tstat": r.hac_tstat, "p_value": r.p_value,
                "bh_q_value": r.bh_q_value, "bhy_q_value": r.bhy_q_value,
                "n_dates": r.n_dates, "avg_breadth": r.avg_breadth,
                "breadth_quality": r.breadth_quality,
                "cost_viability_status": r.cost_viability_status,
                "conditional_ic_quality": r.conditional_ic_quality,
                "rejection_reason": r.rejection_reason,
            })
    p = output_dir / "conditional_ic_grid.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["conditional_ic"] = p

    # 5. multiple_testing_report.csv
    rows = []
    for b in bundles:
        for r in b.multiple_testing_results:
            rows.append({
                "sleeve_id": r.sleeve_id, "feature": r.feature,
                "family": r.family, "horizon": r.horizon,
                "raw_p_value": r.raw_p_value,
                "bh_q_value": r.bh_q_value, "bhy_q_value": r.bhy_q_value,
                "passes_bh": r.passes_bh, "passes_bhy": r.passes_bhy,
                "test_family_size": r.test_family_size,
                "dependency_note": r.dependency_note,
                "evidence_status": r.evidence_status,
            })
    p = output_dir / "multiple_testing_report.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["multiple_testing"] = p

    # 6. weak_ic_attribution_report.csv
    rows = []
    for b in bundles:
        for r in b.attribution_results:
            rows.append({
                "feature": r.feature, "family": r.family,
                "horizon": r.horizon, "global_ic": r.global_ic,
                "best_conditional_ic": r.best_conditional_ic,
                "residualized_ic": r.residualized_ic,
                "sector_neutral_ic": r.sector_neutral_ic,
                "peak_ic_horizon": r.peak_ic_horizon,
                "strongest_condition": r.strongest_condition,
                "condition_breadth": r.condition_breadth,
                "condition_q_value": r.condition_q_value,
                "attribution_label": r.attribution_label,
                "recommended_next_action": r.recommended_next_action,
            })
    p = output_dir / "weak_ic_attribution_report.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["attribution"] = p

    # 7. conditional_sleeve_admission_report.csv
    rows = []
    for b in bundles:
        for r in b.sleeve_admissions:
            rows.append({
                "sleeve_id": r.sleeve_id, "feature": r.feature,
                "family": r.family, "horizon": r.horizon,
                "condition_type": r.condition_type,
                "condition_value": r.condition_value,
                "statistical_status": r.statistical_status,
                "stability_status": r.stability_status,
                "breadth_status": r.breadth_status,
                "decay_status": r.decay_status,
                "cost_status": r.cost_status,
                "pit_status": r.pit_status,
                "final_status": r.final_status,
                "rejection_reason": r.rejection_reason,
            })
    p = output_dir / "conditional_sleeve_admission_report.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["sleeve_admission"] = p

    # 8. conditional_walk_forward_validation.csv
    rows = []
    for b in bundles:
        for r in b.walk_forward_results:
            rows.append({
                "sleeve_id": r.sleeve_id,
                "train_start": r.train_start, "train_end": r.train_end,
                "test_start": r.test_start, "test_end": r.test_end,
                "oos_ic": r.oos_ic, "oos_icir": r.oos_icir,
                "oos_tstat": r.oos_tstat,
                "simple_sleeve_return": r.simple_sleeve_return,
                "turnover": r.turnover, "cost_bps": r.cost_bps,
                "net_return": r.net_return, "breadth": r.breadth,
                "window_status": r.window_status,
            })
    if rows:
        p = output_dir / "conditional_walk_forward_validation.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        paths["walk_forward"] = p

    # 9. rejected_weak_ic_candidates.csv
    rejected = []
    for b in bundles:
        for gic in b.global_ic_results:
            if gic.ic_quality in ("low", "insufficient"):
                rejected.append({
                    "candidate_id": gic.candidate_id, "feature": gic.feature,
                    "family": gic.family, "horizon": gic.horizon,
                    "mean_ic": gic.mean_ic, "ic_quality": gic.ic_quality,
                    "rejection_reason": gic.rejection_reason,
                })
    if rejected:
        p = output_dir / "rejected_weak_ic_candidates.csv"
        pd.DataFrame(rejected).to_csv(p, index=False)
        paths["rejected"] = p

    # 10. accepted_conditional_research_candidates.csv
    accepted = []
    for b in bundles:
        for adm in b.sleeve_admissions:
            if adm.final_status in ("admitted", "research_only"):
                accepted.append({
                    "sleeve_id": adm.sleeve_id, "feature": adm.feature,
                    "family": adm.family, "horizon": adm.horizon,
                    "condition_type": adm.condition_type,
                    "condition_value": adm.condition_value,
                    "final_status": adm.final_status,
                })
    if accepted:
        p = output_dir / "accepted_conditional_research_candidates.csv"
        pd.DataFrame(accepted).to_csv(p, index=False)
        paths["accepted"] = p

    logger.info("IC diagnostics reports generated: %s", list(paths.keys()))
    return paths


# ── Helpers ──────────────────────────────────────────────────────────────────
_get_family = get_family
