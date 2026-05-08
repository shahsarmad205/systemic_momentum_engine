"""Pipeline stage: PIT (point-in-time) diagnostics.

Responsibility: Compute per-candidate/date PIT metrics needed for cost viability.
All values must be observable at signal time — no forward-looking data.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class PITDiagnostic:
    candidate_id: str
    feature: str
    horizon: int
    date: pd.Timestamp
    adv_usd: float
    daily_vol: float
    annual_vol: float
    spread_proxy_bps: float
    expected_turnover: float
    rank_halflife: float
    ic_decay_1d: float
    ic_decay_5d: float
    ic_decay_20d: float
    liquidity_bucket: str
    spread_bucket: str
    capacity_estimate_usd: float
    data_quality: str  # "complete" | "degraded" | "missing"


def compute_pit_diagnostics(
    df: pd.DataFrame,
    feature: str,
    horizon: int,
    candidate_id: str,
) -> list[PITDiagnostic]:
    """Compute PIT diagnostics for a single feature-horizon candidate.

    All values are point-in-time observable.
    Missing data produces explicit degraded-data-quality status.
    """
    if df is None or df.empty or feature not in df.columns:
        return []

    work = df[["date", "ticker", feature]].copy()
    work["date"] = pd.to_datetime(work["date"])
    work[feature] = pd.to_numeric(work[feature], errors="coerce")

    # ADV if available
    adv_col = _find_adv_column(df)
    if adv_col:
        work["adv_usd"] = pd.to_numeric(df[adv_col], errors="coerce")
    else:
        work["adv_usd"] = np.nan

    # Vol if available
    vol_col = _find_vol_column(df)
    if vol_col:
        work["daily_vol"] = pd.to_numeric(df[vol_col], errors="coerce")
    else:
        work["daily_vol"] = np.nan

    # Spread proxy if available
    spread_col = _find_spread_column(df)
    if spread_col:
        work["spread_proxy"] = pd.to_numeric(df[spread_col], errors="coerce")
    else:
        work["spread_proxy"] = np.nan

    # Compute rank-based turnover proxy
    work["rank"] = work.groupby("date")[feature].rank(pct=True, method="average")

    # IC decay curve (autocorrelation of ranks)
    ac1, ac5, ac20 = _compute_ic_decay(work)

    results = []
    for date, group in work.groupby("date"):
        adv = group["adv_usd"].median() if group["adv_usd"].notna().any() else np.nan
        daily_vol = group["daily_vol"].median() if group["daily_vol"].notna().any() else np.nan
        spread = group["spread_proxy"].median() if group["spread_proxy"].notna().any() else np.nan

        annual_vol = daily_vol * np.sqrt(252) if np.isfinite(daily_vol) else np.nan

        # Expected turnover from rank autocorrelation
        expected_turnover = max(0.01, min(0.80, 1.0 - abs(ac1))) if np.isfinite(ac1) else 0.10

        # Rank halflife
        halflife = -1.0 / np.log2(ac1) if 0 < ac1 < 1.0 else 1.0

        # Liquidity bucket
        liq_bucket = _liquidity_bucket(adv)

        # Spread bucket
        spread_bucket = _spread_bucket(spread)

        # Capacity estimate
        capacity = adv * 0.05 if np.isfinite(adv) else np.nan  # 5% ADV participation

        # Data quality
        n_missing = group[["adv_usd", "daily_vol"]].isna().all(axis=1).sum()
        data_quality = "complete" if n_missing == 0 else ("degraded" if n_missing < len(group) else "missing")

        results.append(PITDiagnostic(
            candidate_id=candidate_id,
            feature=feature,
            horizon=horizon,
            date=date,
            adv_usd=adv if np.isfinite(adv) else 0.0,
            daily_vol=daily_vol if np.isfinite(daily_vol) else 0.0,
            annual_vol=annual_vol if np.isfinite(annual_vol) else 0.0,
            spread_proxy_bps=spread if np.isfinite(spread) else 0.0,
            expected_turnover=expected_turnover,
            rank_halflife=halflife,
            ic_decay_1d=ac1 if np.isfinite(ac1) else 0.0,
            ic_decay_5d=ac5 if np.isfinite(ac5) else 0.0,
            ic_decay_20d=ac20 if np.isfinite(ac20) else 0.0,
            liquidity_bucket=liq_bucket,
            spread_bucket=spread_bucket,
            capacity_estimate_usd=capacity if np.isfinite(capacity) else 0.0,
            data_quality=data_quality,
        ))

    return results


def _compute_ic_decay(work: pd.DataFrame) -> tuple[float, float, float]:
    """Compute rank autocorrelation at 1, 5, 20 day lags."""
    ranks = work.dropna(subset=["rank"])
    if len(ranks) < 100:
        return (np.nan, np.nan, np.nan)

    # Per-ticker autocorrelation
    ac1_vals, ac5_vals, ac20_vals = [], [], []
    for ticker, grp in ranks.groupby("ticker"):
        r = grp["rank"].values
        if len(r) < 30:
            continue
        ac1_vals.append(_autocorr(r, 1))
        ac5_vals.append(_autocorr(r, 5))
        ac20_vals.append(_autocorr(r, 20))

    ac1 = float(np.nanmean(ac1_vals)) if ac1_vals else np.nan
    ac5 = float(np.nanmean(ac5_vals)) if ac5_vals else np.nan
    ac20 = float(np.nanmean(ac20_vals)) if ac20_vals else np.nan

    return (ac1, ac5, ac20)


def _autocorr(x: np.ndarray, lag: int) -> float:
    """Compute autocorrelation at given lag."""
    if len(x) <= lag:
        return np.nan
    x = x - np.nanmean(x)
    num = np.nanmean(x[lag:] * x[:-lag])
    den = np.nanmean(x * x)
    if den == 0:
        return np.nan
    return num / den


def _liquidity_bucket(adv_usd: float) -> str:
    if not np.isfinite(adv_usd):
        return "unknown"
    if adv_usd < 1_000_000:
        return "micro"
    if adv_usd < 10_000_000:
        return "small"
    if adv_usd < 50_000_000:
        return "mid"
    if adv_usd < 200_000_000:
        return "large"
    return "mega"


def _spread_bucket(spread_bps: float) -> str:
    if not np.isfinite(spread_bps):
        return "unknown"
    if spread_bps < 1.0:
        return "tight"
    if spread_bps < 5.0:
        return "normal"
    if spread_bps < 15.0:
        return "wide"
    return "very_wide"


def _find_adv_column(df: pd.DataFrame) -> str | None:
    for c in ["adv_dollar_20", "adv_dollar", "adv_usd"]:
        if c in df.columns:
            return c
    return None


def _find_vol_column(df: pd.DataFrame) -> str | None:
    for c in ["realised_vol_20d", "vol_20_simple", "daily_vol", "rolling_vol_20"]:
        if c in df.columns:
            return c
    return None


def _find_spread_column(df: pd.DataFrame) -> str | None:
    for c in ["spread_bps", "bid_ask_spread", "spread"]:
        if c in df.columns:
            return c
    return None
