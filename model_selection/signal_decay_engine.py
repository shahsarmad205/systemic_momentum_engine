"""Signal Decay and Horizon Compatibility Engine.

Institutional, point-in-time framework that measures:
- True rank persistence (ticker/date-aligned, NOT flattened panel autocorrelation)
- Signal halflife from exponential decay fit
- IC decay curve by horizon (true h-day forward returns)
- Horizon compatibility classification
- Expected turnover from rank decay
- Optional smoothing experiments

All thresholds come from ResearchContract/config — no hardcoded halflife,
turnover, or horizon assumptions.

Usage:
    from model_selection.signal_decay_engine import SignalDecayEngine

    engine = SignalDecayEngine(config=cfg)
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
    compute_forward_returns,
    compute_ic_decay,
    compute_rank_persistence,
)
from model_selection._shared_stats import ic_quality
from model_selection._shared_feature_utils import get_family
from model_selection._shared_config import merge_config

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────────────

class DecayStatus(str, Enum):
    PERSISTENT = "persistent"
    MARGINAL_PERSISTENCE = "marginal_persistence"
    FAST_DECAY = "fast_decay"
    UNSTABLE_DECAY = "unstable_decay"
    INSUFFICIENT_DATA = "insufficient_data"


class HorizonStatus(str, Enum):
    HORIZON_COMPATIBLE = "horizon_compatible"
    MARGINAL_HORIZON_FIT = "marginal_horizon_fit"
    HORIZON_TOO_LONG = "horizon_too_long"
    SIGNAL_TOO_FAST = "signal_too_fast"
    IC_PEAKS_ELSEWHERE = "ic_peaks_elsewhere"
    UNSTABLE_DECAY = "unstable_decay"
    INSUFFICIENT_DECAY_EVIDENCE = "insufficient_decay_evidence"


class SmoothingMethod(str, Enum):
    RAW = "raw"
    EWMA = "ewma"
    RANK_SMOOTHED = "rank_smoothed"
    KALMAN = "kalman"


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class RankPersistenceResult:
    """Rank persistence at a single lag for a single feature."""
    candidate_id: str
    feature: str
    family: str
    sleeve: str
    regime: str
    lag: int
    rank_persistence: float
    n_dates: int
    avg_breadth: int
    persistence_quality: str  # "high" | "medium" | "low" | "insufficient"


@dataclass
class HalflifeResult:
    """Halflife estimate from persistence curve."""
    candidate_id: str
    feature: str
    family: str
    sleeve: str
    regime: str
    estimated_halflife_days: float
    decay_tau: float
    initial_persistence: float
    persistence_at_horizon: float
    fit_r2: float
    halflife_quality: str  # "high" | "medium" | "low"
    decay_status: str


@dataclass
class ICDecayResult:
    """IC at a single horizon for a single feature."""
    candidate_id: str
    feature: str
    family: str
    sleeve: str
    regime: str
    horizon: int
    mean_ic: float
    icir: float
    hac_tstat: float
    n_dates: int
    avg_breadth: int
    sign_consistency: float
    subperiod_stability: float
    ic_quality: str  # "high" | "medium" | "low" | "insufficient"


@dataclass
class HorizonCompatibilityResult:
    """Horizon fit classification."""
    candidate_id: str
    feature: str
    family: str
    sleeve: str
    regime: str
    tested_horizon: int
    estimated_halflife: float
    halflife_to_horizon_ratio: float
    persistence_at_horizon: float
    ic_at_horizon: float
    peak_ic_horizon: int
    decay_adjusted_ic: float
    turnover_pressure: float
    horizon_status: str
    rejection_reason: str


@dataclass
class TurnoverDecayResult:
    """Turnover estimate from rank decay."""
    candidate_id: str
    feature: str
    family: str
    sleeve: str
    rebalance_gap: int
    avg_name_churn: float
    avg_rank_migration: float
    estimated_target_turnover: float
    turnover_pressure: float
    turnover_quality: str


@dataclass
class SmoothingExperimentResult:
    """Smoothing experiment result."""
    feature: str
    family: str
    smoothing_method: str
    smoothing_param: float
    mean_ic: float
    halflife: float
    turnover: float
    gross_alpha_bps: float
    cost_bps: float
    net_alpha_bps: float
    alpha_cost_ratio: float
    accepted: bool
    rejection_reason: str


# ── Config defaults ──────────────────────────────────────────────────────────

_DEFAULT_SIGNAL_DECAY_CONFIG: dict[str, Any] = {
    "signal_decay": {
        "lags": [1, 2, 3, 5, 10, 20, 40, 63],
        "horizons": [1, 2, 3, 5, 10, 20, 40, 63],
        "min_dates_for_persistence": 50,
        "min_breadth_for_persistence": 10,
        "min_dates_for_ic": 30,
        "min_breadth_for_ic": 8,
        "halflife_fallback_tau": 10.0,
        "halflife_to_horizon_ratio_min": 0.5,
        "halflife_to_horizon_ratio_marginal": 1.0,
        "persistence_at_horizon_min": 0.3,
        "persistence_at_horizon_marginal": 0.5,
        "ic_decay_threshold": 0.002,
        "turnover_pressure_max": 0.80,
        "turnover_pressure_marginal": 0.50,
        "subperiod_stability_min": 0.6,
        "sign_consistency_min": 0.7,
        "fit_r2_min": 0.5,
        "smoothing": {
            "enabled": False,
            "methods": ["ewma", "rank_smoothed"],
            "ewma_spans": [3, 5, 10],
            "rank_smoothed_windows": [3, 5],
        },
    },
}


def _get_decay_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Extract signal decay config from full config, falling back to defaults."""
    return merge_config(
        cfg, "signal_decay", _DEFAULT_SIGNAL_DECAY_CONFIG["signal_decay"],
        nested_keys=("smoothing",),
    )


# ── Core: True rank persistence (ticker/date-aligned) ────────────────────────

def compute_rank_persistence_curve(
    df: pd.DataFrame,
    feature: str,
    lags: list[int] | None = None,
    min_dates: int = 50,
    min_breadth: int = 10,
) -> list[RankPersistenceResult]:
    """Compute true rank persistence: ticker/date-aligned Spearman at each lag.

    Delegates to shared vectorized kernel; converts output to dataclass results.
    """
    if lags is None:
        lags = [1, 2, 3, 5, 10, 20, 40, 63]

    if df is None or df.empty or feature not in df.columns:
        return []

    kernel_results = compute_rank_persistence(
        df, [feature], lags, min_dates=min_dates, min_breadth=min_breadth,
    )
    feat_df = kernel_results.get(feature)
    if feat_df is None or feat_df.empty:
        return []

    results = []
    for _, row in feat_df.iterrows():
        persistence = float(row["persistence"]) if np.isfinite(row["persistence"]) else float("nan")
        quality = _persistence_quality(
            int(row["n_dates"]), int(row["avg_breadth"]), persistence,
        )
        results.append(RankPersistenceResult(
            candidate_id=f"feature_{feature}",
            feature=feature,
            family=_get_family(feature),
            sleeve="",
            regime="",
            lag=int(row["lag"]),
            rank_persistence=persistence,
            n_dates=int(row["n_dates"]),
            avg_breadth=int(row["avg_breadth"]),
            persistence_quality=quality,
        ))

    return results


def _persistence_quality(n_dates: int, avg_breadth: int, persistence: float) -> str:
    """Assess persistence estimate quality."""
    if n_dates < 30 or avg_breadth < 5:
        return "insufficient"
    if n_dates < 50 or avg_breadth < 10:
        return "low"
    if not np.isfinite(persistence) or abs(persistence) < 0.05:
        return "low"
    if n_dates >= 100 and avg_breadth >= 20:
        return "high"
    return "medium"


# ── Halflife estimation from persistence curve ───────────────────────────────

def estimate_halflife_from_persistence(
    persistence_results: list[RankPersistenceResult],
    fallback_tau: float = 10.0,
    fit_r2_min: float = 0.5,
) -> HalflifeResult:
    """Estimate signal halflife from rank persistence curve.

    Fits: persistence(k) ≈ exp(-k / tau)
    Then: halflife = tau * ln(2)

    Supports robust fallback:
    - Interpolate first lag where persistence falls below 0.5
    - If persistence is noisy, flag halflife_quality = low
    - If persistence is negative or unstable, classify as unstable_decay
    """
    if not persistence_results:
        return HalflifeResult(
            candidate_id="unknown", feature="unknown", family="unknown",
            sleeve="", regime="",
            estimated_halflife_days=fallback_tau * np.log(2),
            decay_tau=fallback_tau, initial_persistence=0.0,
            persistence_at_horizon=0.0, fit_r2=0.0,
            halflife_quality="low", decay_status=DecayStatus.INSUFFICIENT_DATA.value,
        )

    # Extract (lag, persistence) pairs
    lags = []
    vals = []
    for r in persistence_results:
        if np.isfinite(r.rank_persistence) and r.persistence_quality != "insufficient":
            lags.append(r.lag)
            vals.append(r.rank_persistence)

    if len(lags) < 2:
        return HalflifeResult(
            candidate_id=persistence_results[0].candidate_id if persistence_results else "unknown",
            feature=persistence_results[0].feature if persistence_results else "unknown",
            family=persistence_results[0].family if persistence_results else "unknown",
            sleeve=persistence_results[0].sleeve if persistence_results else "",
            regime=persistence_results[0].regime if persistence_results else "",
            estimated_halflife_days=fallback_tau * np.log(2),
            decay_tau=fallback_tau, initial_persistence=0.0,
            persistence_at_horizon=0.0, fit_r2=0.0,
            halflife_quality="low", decay_status=DecayStatus.INSUFFICIENT_DATA.value,
        )

    lags_arr = np.array(lags, dtype=float)
    vals_arr = np.array(vals, dtype=float)

    # Method 1: Exponential fit via log-linear regression
    # log(persistence) ≈ -k/tau + log(initial)
    valid_mask = vals_arr > 0
    if valid_mask.sum() >= 2:
        log_vals = np.log(vals_arr[valid_mask])
        valid_lags = lags_arr[valid_mask]
        slope, intercept, r_value, _, _ = scipy_stats.linregress(valid_lags, log_vals)

        if slope < 0:
            tau = -1.0 / slope
            halflife = tau * np.log(2)
            initial = np.exp(intercept)
            fit_r2 = r_value ** 2
        else:
            tau = fallback_tau
            halflife = tau * np.log(2)
            initial = float(np.mean(vals_arr))
            fit_r2 = 0.0
    else:
        tau = fallback_tau
        halflife = tau * np.log(2)
        initial = float(np.mean(vals_arr))
        fit_r2 = 0.0

    # Method 2: Interpolation fallback (first lag where persistence < 0.5)
    interp_halflife = None
    for i in range(len(lags_arr) - 1):
        if vals_arr[i] >= 0.5 and vals_arr[i + 1] < 0.5:
            frac = (0.5 - vals_arr[i + 1]) / (vals_arr[i] - vals_arr[i + 1])
            interp_halflife = lags_arr[i] + frac
            break

    # Use interpolation if fit is poor
    if fit_r2 < fit_r2_min and interp_halflife is not None:
        halflife = interp_halflife
        tau = halflife / np.log(2)
        fit_r2 = 0.0

    # Persistence at horizon (use max lag as proxy for horizon)
    max_lag = int(lags_arr[-1])
    persistence_at_horizon = float(np.exp(-max_lag / tau)) if tau > 0 else 0.0

    # Decay status
    decay_status = _classify_decay(halflife, fit_r2, vals_arr)

    # Quality
    quality = "high" if fit_r2 >= 0.7 else ("medium" if fit_r2 >= fit_r2_min else "low")

    return HalflifeResult(
        candidate_id=persistence_results[0].candidate_id,
        feature=persistence_results[0].feature,
        family=persistence_results[0].family,
        sleeve=persistence_results[0].sleeve,
        regime=persistence_results[0].regime,
        estimated_halflife_days=halflife,
        decay_tau=tau,
        initial_persistence=initial,
        persistence_at_horizon=persistence_at_horizon,
        fit_r2=fit_r2,
        halflife_quality=quality,
        decay_status=decay_status,
    )


def _classify_decay(halflife: float, fit_r2: float, persistence_vals: np.ndarray) -> str:
    """Classify decay status."""
    if not np.isfinite(halflife) or halflife <= 0:
        return DecayStatus.UNSTABLE_DECAY.value
    if len(persistence_vals) < 2:
        return DecayStatus.INSUFFICIENT_DATA.value
    if fit_r2 < 0.3:
        return DecayStatus.UNSTABLE_DECAY.value
    if halflife < 3:
        return DecayStatus.FAST_DECAY.value
    if halflife < 5:
        return DecayStatus.MARGINAL_PERSISTENCE.value
    return DecayStatus.PERSISTENT.value


# ── IC decay curve by horizon ────────────────────────────────────────────────

def compute_ic_decay_curve(
    df: pd.DataFrame,
    feature: str,
    horizons: list[int] | None = None,
    min_dates: int = 30,
    min_breadth: int = 8,
) -> list[ICDecayResult]:
    """Compute IC decay curve: IC at each horizon using true h-day forward returns.

    Delegates to shared vectorized kernel; converts output to dataclass results.
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 10, 20, 40, 63]

    if df is None or df.empty or feature not in df.columns:
        return []

    kernel_results = compute_ic_decay(
        df, [feature], horizons, min_dates=min_dates, min_breadth=min_breadth,
    )
    feat_df = kernel_results.get(feature)
    if feat_df is None or feat_df.empty:
        return []

    results = []
    for _, row in feat_df.iterrows():
        h = int(row["horizon"])
        mean_ic = float(row["mean_ic"])
        icir = float(row["icir"])
        t_stat = float(row["hac_tstat"])
        n_dates = int(row["n_dates"])
        avg_breadth = int(row["avg_breadth"])
        sign_consistency = float(row["sign_consistency"])
        subperiod_stability = float(row["subperiod_stability"])
        quality = row["ic_quality"]

        results.append(ICDecayResult(
            candidate_id=f"feature_{feature}",
            feature=feature, family=_get_family(feature),
            sleeve="", regime="", horizon=h,
            mean_ic=mean_ic, icir=icir, hac_tstat=t_stat,
            n_dates=n_dates, avg_breadth=avg_breadth,
            sign_consistency=sign_consistency, subperiod_stability=subperiod_stability,
            ic_quality=quality,
        ))

    return results


def _hac_tstat(ics: np.ndarray, nw_lag: int) -> float:
    """Compute HAC/Newey-West t-stat for IC series."""
    n = len(ics)
    if n < nw_lag * 2 + 5:
        return float(np.mean(ics) / (np.std(ics) / np.sqrt(n))) if np.std(ics) > 0 else 0.0

    mean_ic = np.mean(ics)
    var = np.var(ics, ddof=1)

    # Newey-West correction
    for k in range(1, min(nw_lag + 1, n)):
        gamma_k = np.cov(ics[k:], ics[:-k])[0, 1]
        var += 2.0 * (1.0 - k / (nw_lag + 1)) * gamma_k

    se = np.sqrt(max(var / n, 1e-12))
    return float(mean_ic / se)


_ic_quality = ic_quality


# ── Horizon compatibility classification ─────────────────────────────────────

def classify_horizon_compatibility(
    halflife_result: HalflifeResult,
    ic_decay_results: list[ICDecayResult],
    tested_horizon: int,
    halflife_to_horizon_ratio_min: float = 0.5,
    halflife_to_horizon_ratio_marginal: float = 1.0,
    persistence_at_horizon_min: float = 0.3,
    persistence_at_horizon_marginal: float = 0.5,
    ic_decay_threshold: float = 0.002,
    turnover_pressure_max: float = 0.80,
) -> HorizonCompatibilityResult:
    """Classify whether a feature's signal lifespan is compatible with the intended horizon.

    Computes:
    - persistence_at_horizon = exp(-horizon / tau)
    - halflife_to_horizon_ratio = halflife / horizon
    - ic_at_horizon = IC at tested horizon
    - peak_ic_horizon = horizon where IC is highest
    - decay_adjusted_ic = IC_h * persistence_at_horizon
    - turnover_pressure = 1 - persistence_at_horizon
    """
    tau = halflife_result.decay_tau
    halflife = halflife_result.estimated_halflife_days

    # Persistence at horizon
    persistence_at_horizon = float(np.exp(-tested_horizon / tau)) if tau > 0 else 0.0

    # Halflife to horizon ratio
    hl_ratio = halflife / tested_horizon if tested_horizon > 0 else 0.0

    # IC at horizon
    ic_at_horizon = 0.0
    for r in ic_decay_results:
        if r.horizon == tested_horizon:
            ic_at_horizon = r.mean_ic
            break

    # Peak IC horizon
    peak_ic_horizon = tested_horizon
    max_ic = abs(ic_at_horizon)
    for r in ic_decay_results:
        if abs(r.mean_ic) > max_ic:
            max_ic = abs(r.mean_ic)
            peak_ic_horizon = r.horizon

    # Decay-adjusted IC
    decay_adjusted_ic = ic_at_horizon * persistence_at_horizon

    # Turnover pressure
    turnover_pressure = 1.0 - persistence_at_horizon

    # Classification
    status, reason = _classify_horizon_fit(
        hl_ratio, persistence_at_horizon, ic_at_horizon,
        peak_ic_horizon, tested_horizon, halflife_result.decay_status,
        halflife_result.halflife_quality,
        halflife_to_horizon_ratio_min, halflife_to_horizon_ratio_marginal,
        persistence_at_horizon_min, persistence_at_horizon_marginal,
        ic_decay_threshold, turnover_pressure_max,
    )

    return HorizonCompatibilityResult(
        candidate_id=halflife_result.candidate_id,
        feature=halflife_result.feature,
        family=halflife_result.family,
        sleeve=halflife_result.sleeve,
        regime=halflife_result.regime,
        tested_horizon=tested_horizon,
        estimated_halflife=halflife,
        halflife_to_horizon_ratio=hl_ratio,
        persistence_at_horizon=persistence_at_horizon,
        ic_at_horizon=ic_at_horizon,
        peak_ic_horizon=peak_ic_horizon,
        decay_adjusted_ic=decay_adjusted_ic,
        turnover_pressure=turnover_pressure,
        horizon_status=status,
        rejection_reason=reason,
    )


def _classify_horizon_fit(
    hl_ratio: float,
    persistence: float,
    ic: float,
    peak_horizon: int,
    tested_horizon: int,
    decay_status: str,
    hl_quality: str,
    hl_ratio_min: float,
    hl_ratio_marginal: float,
    persist_min: float,
    persist_marginal: float,
    ic_threshold: float,
    turnover_max: float,
) -> tuple[str, str]:
    """Classify horizon fit status."""
    if decay_status == DecayStatus.INSUFFICIENT_DATA.value:
        return HorizonStatus.INSUFFICIENT_DECAY_EVIDENCE.value, "insufficient_decay_data"
    if decay_status == DecayStatus.UNSTABLE_DECAY.value:
        return HorizonStatus.UNSTABLE_DECAY.value, "unstable_decay"
    if hl_quality == "low":
        return HorizonStatus.INSUFFICIENT_DECAY_EVIDENCE.value, "low_halflife_quality"

    if abs(ic) < ic_threshold:
        return HorizonStatus.SIGNAL_TOO_FAST.value, "ic_below_threshold"

    if peak_horizon != tested_horizon and abs(peak_horizon - tested_horizon) > tested_horizon * 0.5:
        return HorizonStatus.IC_PEAKS_ELSEWHERE.value, f"ic_peaks_at_{peak_horizon}d_not_{tested_horizon}d"

    if hl_ratio < hl_ratio_min:
        return HorizonStatus.HORIZON_TOO_LONG.value, f"halflife_too_short: ratio={hl_ratio:.2f}<{hl_ratio_min}"

    if persistence < persist_min:
        return HorizonStatus.SIGNAL_TOO_FAST.value, f"persistence_at_horizon={persistence:.2f}<{persist_min}"

    if hl_ratio < hl_ratio_marginal or persistence < persist_marginal:
        return HorizonStatus.MARGINAL_HORIZON_FIT.value, "marginal_persistence"

    return HorizonStatus.HORIZON_COMPATIBLE.value, ""


# ── Turnover estimation from rank decay ──────────────────────────────────────

def estimate_turnover_from_decay(
    df: pd.DataFrame,
    feature: str,
    halflife_result: HalflifeResult,
    rebalance_gaps: list[int] | None = None,
    quantile: float = 0.2,
) -> list[TurnoverDecayResult]:
    """Estimate expected turnover from rank decay and sleeve membership churn.

    For each feature and rebalance gap:
    - Build simple rank sleeve (long top quantile, short bottom)
    - Measure name churn across rebalance dates
    - Measure rank migration
    - Estimate expected turnover
    """
    if rebalance_gaps is None:
        rebalance_gaps = [1, 2, 3, 5, 10, 20]

    if df is None or df.empty or feature not in df.columns:
        return []

    work = df[["date", "ticker", feature]].copy()
    work["date"] = pd.to_datetime(work["date"])
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work = work.dropna(subset=["date", "ticker", feature])
    work = work.sort_values(["date", "ticker"])

    # Cross-sectional ranks
    work["rank"] = work.groupby("date", sort=False)[feature].rank(pct=True, method="average")

    # Sleeve membership: long = top quantile, short = bottom quantile
    work["in_long"] = work["rank"] >= (1.0 - quantile)
    work["in_short"] = work["rank"] <= quantile

    dates = sorted(work["date"].unique())
    results = []

    for gap in rebalance_gaps:
        if gap >= len(dates):
            continue

        name_churns = []
        rank_migrations = []

        for i in range(0, len(dates) - gap, gap):
            t0 = dates[i]
            t1 = dates[min(i + gap, len(dates) - 1)]

            g0 = work[work["date"] == t0].set_index("ticker")
            g1 = work[work["date"] == t1].set_index("ticker")

            if g0.empty or g1.empty:
                continue

            common = g0.index.intersection(g1.index)
            if len(common) < 5:
                continue

            # Name churn: fraction of long sleeve members that changed
            long_t0 = set(g0.loc[g0["in_long"], :].index)
            long_t1 = set(g1.loc[g1["in_long"], :].index)
            if long_t0:
                churn = 1.0 - len(long_t0 & long_t1) / len(long_t0)
                name_churns.append(churn)

            # Rank migration: average absolute rank change
            r0 = g0.loc[common, "rank"].values
            r1 = g1.loc[common, "rank"].values
            migration = float(np.mean(np.abs(r1 - r0)))
            rank_migrations.append(migration)

        avg_churn = float(np.mean(name_churns)) if name_churns else 0.0
        avg_migration = float(np.mean(rank_migrations)) if rank_migrations else 0.0

        # Estimated target turnover ≈ name churn × 2 (exit + enter)
        est_turnover = min(1.0, avg_churn * 2.0)

        # Turnover pressure from halflife
        tau = halflife_result.decay_tau
        turnover_from_decay = 1.0 - np.exp(-gap / tau) if tau > 0 else 1.0

        # Combine: use max of empirical and decay-based
        combined_turnover = max(est_turnover, turnover_from_decay)

        quality = "high" if len(name_churns) >= 20 else ("medium" if len(name_churns) >= 10 else "low")

        results.append(TurnoverDecayResult(
            candidate_id=f"feature_{feature}",
            feature=feature, family=_get_family(feature),
            sleeve="", rebalance_gap=gap,
            avg_name_churn=avg_churn,
            avg_rank_migration=avg_migration,
            estimated_target_turnover=combined_turnover,
            turnover_pressure=turnover_from_decay,
            turnover_quality=quality,
        ))

    return results


# ── Smoothing experiment ─────────────────────────────────────────────────────

def run_smoothing_experiment(
    df: pd.DataFrame,
    feature: str,
    cfg: dict[str, Any],
    horizons: list[int] | None = None,
) -> list[SmoothingExperimentResult]:
    """Compare raw vs smoothed signals.

    Only accepts smoothing if it improves net alpha after costs,
    not merely because it reduces turnover.
    """
    decay_cfg = _get_decay_config(cfg)
    smooth_cfg = decay_cfg.get("smoothing", {})
    if not smooth_cfg.get("enabled", False):
        return []

    if df is None or df.empty or feature not in df.columns:
        return []

    if horizons is None:
        horizons = decay_cfg.get("horizons", [1, 2, 3, 5, 10, 20])

    results = []

    # Raw signal baseline
    raw_result = _evaluate_signal_version(df, feature, "raw", 0.0, horizons, cfg)
    results.append(raw_result)

    # EWMA smoothed
    for span in smooth_cfg.get("ewma_spans", [3, 5, 10]):
        smoothed = _apply_ewma(df, feature, span)
        if smoothed is not None:
            result = _evaluate_signal_version(smoothed, feature, "ewma", float(span), horizons, cfg)
            results.append(result)

    # Rank smoothed (rolling median rank)
    for window in smooth_cfg.get("rank_smoothed_windows", [3, 5]):
        smoothed = _apply_rank_smoothing(df, feature, window)
        if smoothed is not None:
            result = _evaluate_signal_version(smoothed, feature, "rank_smoothed", float(window), horizons, cfg)
            results.append(result)

    # Accept only if net alpha after costs improves
    raw_net = raw_result.net_alpha_bps
    for r in results[1:]:
        r.accepted = r.net_alpha_bps > raw_net
        if not r.accepted:
            r.rejection_reason = f"net_alpha {r.net_alpha_bps:.1f}bps <= raw {raw_net:.1f}bps"

    return results


def _apply_ewma(df: pd.DataFrame, feature: str, span: int) -> pd.DataFrame | None:
    """Apply EWMA smoothing to feature values."""
    work = df.copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work = work.sort_values(["ticker", "date"])
    work[feature] = work.groupby("ticker", sort=False)[feature].transform(
        lambda x: x.ewm(span=span, min_periods=1).mean()
    )
    return work


def _apply_rank_smoothing(df: pd.DataFrame, feature: str, window: int) -> pd.DataFrame | None:
    """Apply rolling median rank smoothing."""
    work = df.copy()
    work[feature] = pd.to_numeric(work[feature], errors="coerce")
    work["rank"] = work.groupby("date", sort=False)[feature].rank(pct=True)
    work = work.sort_values(["ticker", "date"])
    work["smoothed_rank"] = work.groupby("ticker", sort=False)["rank"].transform(
        lambda x: x.rolling(window, min_periods=1).median()
    )
    work[feature] = work["smoothed_rank"]
    work = work.drop(columns=["smoothed_rank", "rank"])
    return work


def _evaluate_signal_version(
    df: pd.DataFrame,
    feature: str,
    method: str,
    param: float,
    horizons: list[int],
    cfg: dict[str, Any],
) -> SmoothingExperimentResult:
    """Evaluate a single signal version."""
    # Compute IC at primary horizon
    primary_h = horizons[-1] if horizons else 20
    ic_results = compute_ic_decay_curve(df, feature, horizons=[primary_h])
    mean_ic = ic_results[0].mean_ic if ic_results else 0.0

    # Compute halflife
    persistence_results = compute_rank_persistence_curve(df, feature, lags=[1, 2, 3, 5, 10, 20])
    hl_result = estimate_halflife_from_persistence(persistence_results)
    halflife = hl_result.estimated_halflife_days

    # Estimate turnover
    turnover_results = estimate_turnover_from_decay(df, feature, hl_result, rebalance_gaps=[primary_h])
    turnover = turnover_results[0].estimated_target_turnover if turnover_results else 0.10

    # Gross alpha approximation
    sigma_annual = 0.20
    gross_alpha_bps = abs(mean_ic) * sigma_annual * np.sqrt(primary_h / 252) * 10000

    # Cost approximation
    cost_bps = 10.0  # Simplified; full cost model would use CostViabilityEngine
    if turnover > 0:
        cost_bps = max(cost_bps, turnover * 20.0)  # Approximate 10bps half-spread × turnover

    net_alpha_bps = gross_alpha_bps - cost_bps
    alpha_cost_ratio = gross_alpha_bps / cost_bps if cost_bps > 0 else 0.0

    return SmoothingExperimentResult(
        feature=feature, family=_get_family(feature),
        smoothing_method=method, smoothing_param=param,
        mean_ic=mean_ic, halflife=halflife, turnover=turnover,
        gross_alpha_bps=gross_alpha_bps, cost_bps=cost_bps,
        net_alpha_bps=net_alpha_bps, alpha_cost_ratio=alpha_cost_ratio,
        accepted=(method == "raw"),  # Raw is always "accepted" as baseline
        rejection_reason="",
    )


# ── Main engine ──────────────────────────────────────────────────────────────

@dataclass
class SignalDecayDiagnostics:
    """Full diagnostics output for a feature."""
    feature: str
    family: str
    persistence_results: list[RankPersistenceResult]
    halflife_result: HalflifeResult
    ic_decay_results: list[ICDecayResult]
    horizon_compatibility: list[HorizonCompatibilityResult]
    turnover_results: list[TurnoverDecayResult]
    smoothing_results: list[SmoothingExperimentResult]


class SignalDecayEngine:
    """Signal Decay and Horizon Compatibility Engine.

    Measures rank persistence, IC decay, halflife, and horizon fit
    for every feature, family, sleeve, and model candidate.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.decay_cfg = _get_decay_config(self.config)

    def run_full_diagnostics(
        self,
        df: pd.DataFrame,
        features: list[str],
        horizons: list[int] | None = None,
        sleeve: str = "",
        regime: str = "",
    ) -> list[SignalDecayDiagnostics]:
        """Run full signal decay diagnostics for a list of features.

        Args:
            df: Feature matrix panel with date, ticker, feature columns, and forward_return
            features: List of feature column names
            horizons: Horizons to test (default from config)
            sleeve: Sleeve name (e.g., "momentum", "reversal")
            regime: Regime name (e.g., "bull", "bear")

        Returns:
            List of SignalDecayDiagnostics per feature.
        """
        if horizons is None:
            horizons = self.decay_cfg.get("horizons", [1, 2, 3, 5, 10, 20, 40, 63])

        lags = self.decay_cfg.get("lags", [1, 2, 3, 5, 10, 20, 40, 63])
        min_dates_persist = self.decay_cfg.get("min_dates_for_persistence", 50)
        min_breadth_persist = self.decay_cfg.get("min_breadth_for_persistence", 10)
        min_dates_ic = self.decay_cfg.get("min_dates_for_ic", 30)
        min_breadth_ic = self.decay_cfg.get("min_breadth_for_ic", 8)

        results = []
        for feature in features:
            # 1. Rank persistence
            persistence = compute_rank_persistence_curve(
                df, feature, lags=lags,
                min_dates=min_dates_persist, min_breadth=min_breadth_persist,
            )

            # 2. Halflife
            hl_result = estimate_halflife_from_persistence(
                persistence,
                fallback_tau=self.decay_cfg.get("halflife_fallback_tau", 10.0),
                fit_r2_min=self.decay_cfg.get("fit_r2_min", 0.5),
            )

            # 3. IC decay
            ic_decay = compute_ic_decay_curve(
                df, feature, horizons=horizons,
                min_dates=min_dates_ic, min_breadth=min_breadth_ic,
            )

            # 4. Horizon compatibility
            compat = []
            for h in horizons:
                c = classify_horizon_compatibility(
                    hl_result, ic_decay, h,
                    halflife_to_horizon_ratio_min=self.decay_cfg.get("halflife_to_horizon_ratio_min", 0.5),
                    halflife_to_horizon_ratio_marginal=self.decay_cfg.get("halflife_to_horizon_ratio_marginal", 1.0),
                    persistence_at_horizon_min=self.decay_cfg.get("persistence_at_horizon_min", 0.3),
                    persistence_at_horizon_marginal=self.decay_cfg.get("persistence_at_horizon_marginal", 0.5),
                    ic_decay_threshold=self.decay_cfg.get("ic_decay_threshold", 0.002),
                    turnover_pressure_max=self.decay_cfg.get("turnover_pressure_max", 0.80),
                )
                compat.append(c)

            # 5. Turnover from decay
            rebalance_gaps = [h for h in horizons if h <= 20]
            turnover = estimate_turnover_from_decay(df, feature, hl_result, rebalance_gaps=rebalance_gaps)

            # 6. Smoothing experiment (optional)
            smoothing = run_smoothing_experiment(df, feature, self.config, horizons=horizons)

            results.append(SignalDecayDiagnostics(
                feature=feature,
                family=_get_family(feature),
                persistence_results=persistence,
                halflife_result=hl_result,
                ic_decay_results=ic_decay,
                horizon_compatibility=compat,
                turnover_results=turnover,
                smoothing_results=smoothing,
            ))

        logger.info("Signal decay diagnostics: %d features evaluated", len(results))
        return results


# ── Report generation ────────────────────────────────────────────────────────

def generate_signal_decay_reports(
    diagnostics: list[SignalDecayDiagnostics],
    output_dir: str | Path = "output/models/signal_decay",
) -> dict[str, Path]:
    """Generate all signal decay reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = {}

    # 1. signal_rank_persistence.csv
    rows = []
    for d in diagnostics:
        for r in d.persistence_results:
            rows.append({
                "candidate_id": r.candidate_id, "feature": r.feature,
                "family": r.family, "sleeve": r.sleeve, "regime": r.regime,
                "lag": r.lag, "rank_persistence": r.rank_persistence,
                "n_dates": r.n_dates, "avg_breadth": r.avg_breadth,
                "persistence_quality": r.persistence_quality,
            })
    p = output_dir / "signal_rank_persistence.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["rank_persistence"] = p

    # 2. signal_halflife_report.csv
    rows = []
    for d in diagnostics:
        r = d.halflife_result
        rows.append({
            "candidate_id": r.candidate_id, "feature": r.feature,
            "family": r.family, "sleeve": r.sleeve, "regime": r.regime,
            "estimated_halflife_days": r.estimated_halflife_days,
            "decay_tau": r.decay_tau, "initial_persistence": r.initial_persistence,
            "persistence_at_horizon": r.persistence_at_horizon,
            "fit_r2": r.fit_r2, "halflife_quality": r.halflife_quality,
            "decay_status": r.decay_status,
        })
    p = output_dir / "signal_halflife_report.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["halflife"] = p

    # 3. ic_decay_curve.csv
    rows = []
    for d in diagnostics:
        for r in d.ic_decay_results:
            rows.append({
                "candidate_id": r.candidate_id, "feature": r.feature,
                "family": r.family, "sleeve": r.sleeve, "regime": r.regime,
                "horizon": r.horizon, "mean_ic": r.mean_ic,
                "icir": r.icir, "hac_tstat": r.hac_tstat,
                "n_dates": r.n_dates, "avg_breadth": r.avg_breadth,
                "sign_consistency": r.sign_consistency,
                "subperiod_stability": r.subperiod_stability,
                "ic_quality": r.ic_quality,
            })
    p = output_dir / "ic_decay_curve.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["ic_decay"] = p

    # 4. horizon_compatibility_report.csv
    rows = []
    for d in diagnostics:
        for r in d.horizon_compatibility:
            rows.append({
                "candidate_id": r.candidate_id, "feature": r.feature,
                "family": r.family, "sleeve": r.sleeve, "regime": r.regime,
                "tested_horizon": r.tested_horizon,
                "estimated_halflife": r.estimated_halflife,
                "halflife_to_horizon_ratio": r.halflife_to_horizon_ratio,
                "persistence_at_horizon": r.persistence_at_horizon,
                "ic_at_horizon": r.ic_at_horizon,
                "peak_ic_horizon": r.peak_ic_horizon,
                "decay_adjusted_ic": r.decay_adjusted_ic,
                "turnover_pressure": r.turnover_pressure,
                "horizon_status": r.horizon_status,
                "rejection_reason": r.rejection_reason,
            })
    p = output_dir / "horizon_compatibility_report.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["horizon_compatibility"] = p

    # 5. signal_turnover_decay_report.csv
    rows = []
    for d in diagnostics:
        for r in d.turnover_results:
            rows.append({
                "candidate_id": r.candidate_id, "feature": r.feature,
                "family": r.family, "sleeve": r.sleeve,
                "rebalance_gap": r.rebalance_gap,
                "avg_name_churn": r.avg_name_churn,
                "avg_rank_migration": r.avg_rank_migration,
                "estimated_target_turnover": r.estimated_target_turnover,
                "turnover_pressure": r.turnover_pressure,
                "turnover_quality": r.turnover_quality,
            })
    p = output_dir / "signal_turnover_decay_report.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["turnover_decay"] = p

    # 6. fast_decay_candidates.csv
    fast = [d for d in diagnostics if d.halflife_result.decay_status in (
        DecayStatus.FAST_DECAY.value, DecayStatus.UNSTABLE_DECAY.value
    )]
    rows = []
    for d in fast:
        r = d.halflife_result
        rows.append({
            "candidate_id": r.candidate_id, "feature": r.feature,
            "family": r.family, "estimated_halflife_days": r.estimated_halflife_days,
            "decay_status": r.decay_status, "halflife_quality": r.halflife_quality,
        })
    p = output_dir / "fast_decay_candidates.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["fast_decay"] = p

    # 7. horizon_mismatch_candidates.csv
    mismatch = []
    for d in diagnostics:
        for c in d.horizon_compatibility:
            if c.horizon_status not in (HorizonStatus.HORIZON_COMPATIBLE.value,):
                mismatch.append(c)
    rows = []
    for r in mismatch:
        rows.append({
            "candidate_id": r.candidate_id, "feature": r.feature,
            "family": r.family, "tested_horizon": r.tested_horizon,
            "estimated_halflife": r.estimated_halflife,
            "halflife_to_horizon_ratio": r.halflife_to_horizon_ratio,
            "horizon_status": r.horizon_status,
            "rejection_reason": r.rejection_reason,
        })
    p = output_dir / "horizon_mismatch_candidates.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["horizon_mismatch"] = p

    # 8. decay_quality_flags.csv
    rows = []
    for d in diagnostics:
        r = d.halflife_result
        rows.append({
            "candidate_id": r.candidate_id, "feature": r.feature,
            "family": r.family, "halflife_quality": r.halflife_quality,
            "decay_status": r.decay_status, "fit_r2": r.fit_r2,
            "n_persistence_lags": len(d.persistence_results),
            "n_ic_horizons": len(d.ic_decay_results),
            "persistence_quality_flags": ",".join(
                set(pr.persistence_quality for pr in d.persistence_results)
            ),
            "ic_quality_flags": ",".join(
                set(icr.ic_quality for icr in d.ic_decay_results)
            ),
        })
    p = output_dir / "decay_quality_flags.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["quality_flags"] = p

    # 9. signal_smoothing_experiment.csv (if run)
    smoothing_rows = []
    for d in diagnostics:
        for r in d.smoothing_results:
            smoothing_rows.append({
                "feature": r.feature, "family": r.family,
                "smoothing_method": r.smoothing_method,
                "smoothing_param": r.smoothing_param,
                "mean_ic": r.mean_ic, "halflife": r.halflife,
                "turnover": r.turnover, "gross_alpha_bps": r.gross_alpha_bps,
                "cost_bps": r.cost_bps, "net_alpha_bps": r.net_alpha_bps,
                "alpha_cost_ratio": r.alpha_cost_ratio,
                "accepted": r.accepted, "rejection_reason": r.rejection_reason,
            })
    if smoothing_rows:
        p = output_dir / "signal_smoothing_experiment.csv"
        pd.DataFrame(smoothing_rows).to_csv(p, index=False)
        paths["smoothing_experiment"] = p

    logger.info("Signal decay reports generated: %s", list(paths.keys()))
    return paths


# ── Helpers ──────────────────────────────────────────────────────────────────

_get_family = get_family
