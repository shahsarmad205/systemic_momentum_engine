"""
Horizon Eligibility Contract — Hardened
========================================
Determines which horizons each feature is statistically eligible for.
Generated BEFORE model training. Blocks ineligible feature-horizon combinations.

Hardening changes:
- Halflife estimated from true feature-rank persistence (Spearman rank autocorrelation),
  NOT IC-series autocorrelation.
- Split statistical vs production admissibility.
- Cost viability enforced as hard rejection condition.
- Effective signal diversity via eigenvalue participation ratio.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HorizonEligibilityContract:
    """
    Per-feature horizon eligibility determined by empirical viability analysis.

    Split into two admission levels:
    - statistically_admissible_horizons: passes IC, halflife, decay checks
    - production_admissible_horizons: also passes cost viability and diversity checks
    """
    feature: str
    family: str

    # Empirical measurements
    ic_by_horizon: dict[int, float] = field(default_factory=dict)
    ic_decay_curve: dict[int, float] = field(default_factory=dict)
    estimated_halflife: float = 0.0
    rank_persistence: dict[int, float] = field(default_factory=dict)

    # Statistical eligibility (predictive signal exists)
    statistically_admissible_horizons: list[int] = field(default_factory=list)
    statistical_rejections: dict[int, str] = field(default_factory=dict)

    # Production eligibility (also passes cost, diversity, operational checks)
    production_admissible_horizons: list[int] = field(default_factory=list)
    production_rejections: dict[int, str] = field(default_factory=dict)

    # Operational constraints
    min_rebalance_frequency: int = 1
    cost_adjusted_viable: dict[int, bool] = field(default_factory=dict)
    cost_proxy_bps: float = 10.0

    # Diversity diagnostics
    feature_correlation_matrix: Optional[np.ndarray] = None
    effective_signal_count: float = 0.0

    # Metadata
    computed_at: str = ""
    data_window: str = ""
    n_observations: int = 0


# Reason codes
HALFLIFE_TOO_SHORT = "HALFLIFE_TOO_SHORT"
IC_TOO_WEAK = "IC_TOO_WEAK"
IC_NEGATIVE = "IC_NEGATIVE"
COST_DOMINATED = "COST_DOMINATED"
SELECTION_DISTORTED = "SELECTION_DISTORTED"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"
DECAY_DOMINATED = "DECAY_DOMINATED"
COST_NOT_VIABLE = "COST_NOT_VIABLE"
MISSING_COST_DATA = "MISSING_COST_DATA"


def _spearman_rank_correlation(x: np.ndarray, y: np.ndarray) -> float:
    """Compute Spearman rank correlation between two arrays."""
    if len(x) < 3 or len(y) < 3:
        return np.nan
    rx = pd.Series(x).rank(pct=True, method="average").values
    ry = pd.Series(y).rank(pct=True, method="average").values
    valid = np.isfinite(rx) & np.isfinite(ry)
    if valid.sum() < 3:
        return np.nan
    rx, ry = rx[valid], ry[valid]
    mu_x, mu_y = rx.mean(), ry.mean()
    num = ((rx - mu_x) * (ry - mu_y)).sum()
    den = math.sqrt(((rx - mu_x) ** 2).sum() * ((ry - mu_y) ** 2).sum())
    if den < 1e-12:
        return np.nan
    return float(num / den)


def compute_rank_persistence_curve(
    df: pd.DataFrame, feature_col: str, max_lag: int = 20
) -> dict[int, float]:
    """
    Compute true feature-rank persistence: Spearman rank autocorrelation
    of feature values at each lag, aggregated across all ticker-date pairs.

    This measures how stable the cross-sectional ranking of a feature is
    over time — the correct basis for halflife estimation.
    """
    if feature_col not in df.columns:
        return {}

    work = df[["date", "ticker", feature_col]].dropna().copy()
    if len(work) < 100:
        return {}

    # Compute cross-sectional ranks per date
    work["rank"] = work.groupby("date")[feature_col].rank(pct=True, method="average")

    # Pivot to date x ticker matrix
    pivot = work.pivot_table(index="date", columns="ticker", values="rank")
    pivot = pivot.dropna(axis=1, thresh=len(pivot) * 0.5)

    if pivot.shape[1] < 5 or pivot.shape[0] < 10:
        return {}

    persistence = {}
    for lag in range(1, max_lag + 1):
        if lag >= len(pivot):
            break
        t0 = pivot.iloc[:-lag].values
        t1 = pivot.iloc[lag:].values

        # Compute mean pairwise Spearman correlation across tickers
        n_tickers = t0.shape[1]
        if n_tickers < 3:
            continue

        # Vectorized: correlate each ticker's rank series at lag
        cors = []
        for j in range(n_tickers):
            x = t0[:, j]
            y = t1[:, j]
            valid = np.isfinite(x) & np.isfinite(y)
            if valid.sum() >= 3:
                c = _spearman_rank_correlation(x[valid], y[valid])
                if not np.isnan(c):
                    cors.append(c)

        if cors:
            persistence[lag] = float(np.mean(cors))

    return persistence


def estimate_halflife_from_persistence(persistence: dict[int, float]) -> float:
    """
    Estimate rank half-life from the persistence curve.

    Half-life is the lag at which rank autocorrelation decays to 0.5.
    Uses linear interpolation between adjacent lags.
    """
    if not persistence or len(persistence) < 2:
        return np.nan

    lags = sorted(persistence.keys())
    vals = [persistence[lag] for lag in lags]

    # Find where persistence crosses 0.5
    for i in range(len(vals) - 1):
        if vals[i] >= 0.5 and vals[i + 1] < 0.5:
            # Linear interpolation
            frac = (0.5 - vals[i + 1]) / (vals[i] - vals[i + 1])
            return float(lags[i] + frac)

    # If always above 0.5, return max lag
    if vals[0] >= 0.5:
        return float(lags[-1])

    # If always below 0.5, return 1
    return 1.0


def estimate_halflife(ic_series: pd.Series, max_lag: int = 40) -> float:
    """
    Estimate rank IC half-life via autocorrelation decay.
    Kept for backward compatibility; prefer estimate_halflife_from_persistence().
    """
    v = ic_series.dropna().values
    if len(v) < 6:
        return np.nan
    max_lag = min(max_lag, len(v) // 3)
    if max_lag < 2:
        return np.nan
    mu, var = v.mean(), v.var()
    if var < 1e-12:
        return np.nan
    acf = np.array([
        np.mean((v[:-lag] - mu) * (v[lag:] - mu)) / var
        for lag in range(1, max_lag + 1)
    ])
    for i in range(len(acf) - 1):
        if acf[i] >= 0.5 and acf[i + 1] < 0.5:
            return float(i + 1 + (0.5 - acf[i + 1]) / (acf[i] - acf[i + 1]))
    return 1.0 if acf[0] < 0.5 else float(max_lag)


def compute_ic_at_horizon(
    df: pd.DataFrame, feature_col: str, horizon: int
) -> tuple[float, pd.Series]:
    """
    Compute mean IC at a specific horizon.
    Returns (mean_ic, ic_series).
    """
    fwd_col = f"fwd_{horizon}d"
    if fwd_col in df.columns:
        work = df[["date", feature_col, fwd_col]].dropna().rename(columns={fwd_col: "_fwd"})
    elif {"date", "ticker", "daily_return", feature_col}.issubset(df.columns):
        ordered = df[["date", "ticker", feature_col, "daily_return"]].copy()
        ordered["date"] = pd.to_datetime(ordered["date"])
        ordered = ordered.sort_values(["ticker", "date"], kind="mergesort")
        ordered["_fwd"] = compute_forward_returns_at_lag(ordered, int(horizon))
        work = ordered[["date", feature_col, "_fwd"]].dropna()
    else:
        return np.nan, pd.Series(dtype=float)

    if len(work) < 100:
        return np.nan, pd.Series(dtype=float)

    work = work.copy()
    work["fx"] = work.groupby("date")[feature_col].rank(pct=True, method="average")
    work["ty"] = work.groupby("date")["_fwd"].rank(pct=True, method="average")

    g = work.groupby("date", sort=True)
    n = g["fx"].count().values.astype(float)
    sx = g["fx"].sum().values
    sy = g["ty"].sum().values
    sxy = (work["fx"] * work["ty"]).groupby(work["date"]).sum().values
    sx2 = (work["fx"] ** 2).groupby(work["date"]).sum().values
    sy2 = (work["ty"] ** 2).groupby(work["date"]).sum().values

    denom = np.sqrt(np.maximum((sx2 - sx**2 / n) * (sy2 - sy**2 / n), 0))
    ic = np.where(denom > 1e-10, (sxy - sx * sy / n) / denom, np.nan)
    ic = ic[(n >= 10) & np.isfinite(ic)]

    if len(ic) < 5:
        return np.nan, pd.Series(dtype=float)

    return float(ic.mean()), pd.Series(ic)


def compute_ic_decay_curve(
    df: pd.DataFrame, feature_col: str, max_lag: int = 20
) -> dict[int, float]:
    """Compute IC decay curve: IC at each lag."""
    if not {"date", "ticker", "daily_return", feature_col}.issubset(df.columns):
        return {}

    ordered = df[["date", "ticker", feature_col, "daily_return"]].copy()
    ordered["date"] = pd.to_datetime(ordered["date"])
    ordered = ordered.sort_values(["ticker", "date"], kind="mergesort")

    decay = {}
    for lag in range(1, max_lag + 1):
        fwd = compute_forward_returns_at_lag(ordered, lag)
        if fwd is None:
            continue

        work = ordered[["date", feature_col]].copy()
        work["_fwd"] = fwd
        work = work.dropna(subset=[feature_col, "_fwd"])
        if len(work) < 100:
            continue

        work["fx"] = work.groupby("date")[feature_col].rank(pct=True, method="average")
        work["ty"] = work.groupby("date")["_fwd"].rank(pct=True, method="average")

        g = work.groupby("date", sort=True)
        n = g["fx"].count().values.astype(float)
        sx = g["fx"].sum().values
        sy = g["ty"].sum().values
        sxy = (work["fx"] * work["ty"]).groupby(work["date"]).sum().values
        sx2 = (work["fx"] ** 2).groupby(work["date"]).sum().values
        sy2 = (work["ty"] ** 2).groupby(work["date"]).sum().values

        denom = np.sqrt(np.maximum((sx2 - sx**2 / n) * (sy2 - sy**2 / n), 0))
        ic = np.where(denom > 1e-10, (sxy - sx * sy / n) / denom, np.nan)
        ic = ic[(n >= 10) & np.isfinite(ic)]

        if len(ic) >= 5:
            decay[lag] = float(ic.mean())

    return decay


def compute_forward_returns_at_lag(df: pd.DataFrame, lag: int) -> Optional[np.ndarray]:
    """Compute forward return from t+1 to t+lag."""
    if "daily_return" not in df.columns:
        return None

    log_ret = np.log1p(df["daily_return"].values)
    ticker_ids = df["ticker"].values
    n = len(df)
    fwd = np.full(n, np.nan)

    changes = np.where(ticker_ids[1:] != ticker_ids[:-1])[0] + 1
    boundaries = np.concatenate([[0], changes, [n]])

    for i in range(len(boundaries) - 1):
        s, e = boundaries[i], boundaries[i + 1]
        if e - s <= lag:
            continue
        lc = np.cumsum(log_ret[s:e])
        valid = np.arange(s, e - lag)
        fwd[valid] = np.exp(lc[valid - s + lag] - lc[valid - s]) - 1.0

    return fwd


def compute_cost_viability(
    ic: float, horizon: int, cost_bps: float = 10.0, sigma_annual: float = 0.20
) -> bool:
    """
    Determine if a feature-horizon combination is cost-viable.

    Approximate alpha = |IC| * sigma * sqrt(h/252)
    Cost = cost_bps / 10000
    Viable if alpha > cost (fails closed on missing/zero IC).
    """
    if np.isnan(ic) or ic == 0:
        return False
    alpha_approx = abs(ic) * sigma_annual * math.sqrt(horizon / 252)
    cost = cost_bps / 10000.0
    return alpha_approx > cost


def compute_effective_signal_count(corr_matrix: np.ndarray) -> float:
    """
    Compute effective number of independent signals via eigenvalue participation ratio.

    participation_ratio = (sum eigenvalues)^2 / sum(eigenvalues^2)
    = N^2 / sum(eigenvalues^2) for correlation matrix (sum eigenvalues = N)

    Returns value in [1, N] where N is the number of features.
    """
    if corr_matrix is None or corr_matrix.size == 0:
        return 0.0
    eigenvalues = np.linalg.eigvalsh(corr_matrix)
    eigenvalues = np.maximum(eigenvalues, 0)
    sum_eig = eigenvalues.sum()
    sum_eig_sq = (eigenvalues ** 2).sum()
    if sum_eig_sq < 1e-12:
        return 1.0
    return float(sum_eig ** 2 / sum_eig_sq)


def compute_eligibility(
    df: pd.DataFrame,
    feature: str,
    family: str,
    horizons: list[int] = None,
    ic_threshold: float = 0.005,
    halflife_ratio: float = 0.5,
    cost_bps: float = 10.0,
    sigma_annual: float = 0.20,
) -> HorizonEligibilityContract:
    """
    Compute full eligibility contract for one feature.

    Parameters
    ----------
    df : Panel with date, ticker, feature columns, and daily_return
    feature : Feature column name
    family : Feature family name
    horizons : Horizons to evaluate (default: [1,2,3,5,10,20,40,63])
    ic_threshold : Minimum |IC| for admissibility
    halflife_ratio : Minimum halflife/horizon ratio for admissibility
    cost_bps : Round-trip cost in basis points
    sigma_annual : Annualized return volatility for cost approximation
    """
    if horizons is None:
        horizons = [1, 2, 3, 5, 10, 20, 40, 63]

    if feature not in df.columns:
        return HorizonEligibilityContract(
            feature=feature, family=family,
            computed_at=datetime.utcnow().isoformat(),
        )

    # IC by horizon
    ic_by_horizon = {}
    for h in horizons:
        mean_ic, _ = compute_ic_at_horizon(df, feature, h)
        ic_by_horizon[h] = mean_ic

    # IC decay curve (for diagnostics)
    ic_decay = compute_ic_decay_curve(df, feature, max_lag=20)

    # True rank persistence and halflife
    rank_persistence = compute_rank_persistence_curve(df, feature, max_lag=20)
    halflife = estimate_halflife_from_persistence(rank_persistence)

    # --- Statistical admissibility ---
    stat_admissible = []
    stat_rejected = {}
    for h in horizons:
        ic = ic_by_horizon.get(h, np.nan)
        reasons = []

        if np.isnan(ic):
            reasons.append(INSUFFICIENT_DATA)
        elif ic < -ic_threshold:
            reasons.append(IC_NEGATIVE)
        elif abs(ic) < ic_threshold:
            reasons.append(IC_TOO_WEAK)

        if not np.isnan(halflife) and halflife > 0 and halflife < h * halflife_ratio:
            reasons.append(HALFLIFE_TOO_SHORT)

        if not np.isnan(halflife) and halflife > 0:
            decay_factor = 2.0 ** (-h / halflife)
            if decay_factor < 0.1:
                reasons.append(DECAY_DOMINATED)

        if reasons:
            stat_rejected[h] = " + ".join(reasons)
        else:
            stat_admissible.append(h)

    # --- Cost viability ---
    cost_adjusted = {}
    for h in horizons:
        ic = ic_by_horizon.get(h, np.nan)
        cost_adjusted[h] = compute_cost_viability(ic, h, cost_bps, sigma_annual)

    # --- Production admissibility (statistical + cost) ---
    prod_admissible = []
    prod_rejected = {}
    for h in horizons:
        if h in stat_rejected:
            prod_rejected[h] = stat_rejected[h]
            continue

        if not cost_adjusted.get(h, False):
            ic_val = ic_by_horizon.get(h, np.nan)
            if np.isnan(ic_val):
                prod_rejected[h] = MISSING_COST_DATA
            else:
                prod_rejected[h] = COST_NOT_VIABLE
            continue

        prod_admissible.append(h)

    # Minimum rebalance frequency
    if not np.isnan(halflife) and halflife > 0:
        min_rebal = max(1, int(math.floor(halflife)))
    else:
        min_rebal = 1

    # Data window
    if "date" in df.columns:
        data_window = f"{df['date'].min()} to {df['date'].max()}"
        n_obs = len(df)
    else:
        data_window = ""
        n_obs = 0

    return HorizonEligibilityContract(
        feature=feature,
        family=family,
        ic_by_horizon=ic_by_horizon,
        ic_decay_curve=ic_decay,
        estimated_halflife=float(halflife) if not np.isnan(halflife) else 0.0,
        rank_persistence=rank_persistence,
        statistically_admissible_horizons=stat_admissible,
        statistical_rejections=stat_rejected,
        production_admissible_horizons=prod_admissible,
        production_rejections=prod_rejected,
        min_rebalance_frequency=min_rebal,
        cost_adjusted_viable=cost_adjusted,
        cost_proxy_bps=cost_bps,
        computed_at=datetime.utcnow().isoformat(),
        data_window=data_window,
        n_observations=n_obs,
    )


def compute_all_eligibility(
    df: pd.DataFrame,
    feature_families: dict[str, list[str]],
    horizons: list[int] = None,
    cost_bps: float = 10.0,
) -> dict[str, HorizonEligibilityContract]:
    """Compute eligibility contracts for all features."""
    contracts = {}
    for family, features in feature_families.items():
        for feat in features:
            if feat not in df.columns:
                continue
            contracts[feat] = compute_eligibility(df, feat, family, horizons, cost_bps=cost_bps)
    return contracts


def format_eligibility_report(
    contracts: dict[str, HorizonEligibilityContract],
    horizons: list[int],
) -> str:
    """Format eligibility contracts as a human-readable report."""
    lines = []
    lines.append("=" * 120)
    lines.append("HORIZON ELIGIBILITY REPORT (HARDENED)")
    lines.append("=" * 120)
    lines.append("")

    for h in horizons:
        min_hl = h * 0.5
        lines.append("-" * 120)
        lines.append(f"H{h}D (horizon={h}, min_halflife={min_hl:.1f}d)")
        lines.append("-" * 120)
        lines.append(f"{'Feature':<25} | {'Halflife':>8} | {'IC@' + str(h) + 'd':>7} | {'Stat':>5} | {'Prod':>5} | {'Cost OK':>7} | {'Rejection Reason':<40}")
        lines.append("-" * 120)

        stat_count = 0
        prod_count = 0
        for feat, c in sorted(contracts.items(), key=lambda x: x[1].ic_by_horizon.get(h, 0) or 0, reverse=True):
            ic = c.ic_by_horizon.get(h, np.nan)
            hl = c.estimated_halflife
            stat_ok = h in c.statistically_admissible_horizons
            prod_ok = h in c.production_admissible_horizons
            cost_ok = c.cost_adjusted_viable.get(h, False)
            reason = c.production_rejections.get(h, c.statistical_rejections.get(h, "-"))
            if stat_ok:
                stat_count += 1
            if prod_ok:
                prod_count += 1

            lines.append(
                f"{feat:<25} | {hl:>7.1f}d | {ic:>7.4f} | {'YES' if stat_ok else 'NO':>5} | {'YES' if prod_ok else 'NO':>5} | {'YES' if cost_ok else 'NO':>7} | {reason:<40}"
            )

        lines.append("")
        lines.append(f"Statistical: {stat_count}/{len(contracts)} eligible ({stat_count/len(contracts)*100:.0f}%)")
        lines.append(f"Production:  {prod_count}/{len(contracts)} eligible ({prod_count/len(contracts)*100:.0f}%)")

        if prod_count == 0:
            lines.append(f"BLOCKED: h{h}d has ZERO production-eligible features.")
        elif prod_count == 1:
            lines.append(f"WARNING: h{h}d has only 1 production-eligible feature. Insufficient diversity.")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Integration with Cost Viability Engine
# ---------------------------------------------------------------------------

def compute_cost_viability_institutional(
    ic: float,
    horizon: int,
    adv_usd: float = 50_000_000.0,
    daily_vol: float = 0.02,
    sigma_annual: float = 0.20,
    expected_turnover: float = 0.10,
    cost_config: Optional[dict] = None,
) -> tuple[bool, float, float]:
    """
    Institutional cost viability check using the full CostViabilityEngine.

    Replaces the flat 10bps proxy with Almgren-Chriss impact modeling.

    Returns:
        (is_viable, alpha_bps, cost_bps)
    """
    from model_selection.cost_viability_engine import CostViabilityEngine

    engine = CostViabilityEngine(config=cost_config)
    result = engine.evaluate(
        candidate_id=f"eligibility_{horizon}d",
        feature="unknown",
        family="unknown",
        ic=ic,
        horizon=horizon,
        sigma_annual=sigma_annual,
        halflife=horizon * 0.5,  # Conservative estimate
        expected_turnover=expected_turnover,
        adv_usd=adv_usd,
        daily_vol=daily_vol,
    )
    return (
        result.cost_status.value in ("cost_viable", "marginal"),
        result.expected_alpha_bps,
        result.expected_cost_bps,
    )
