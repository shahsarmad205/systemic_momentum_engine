"""
Regime Detection for Weight Learning
======================================
Classifies the market into regimes using SPY data only.
Uses only historical data at each date (no look-ahead bias).

Regimes:
    Bull       — SPY close above 200-day moving average
    Bear       — SPY close below 200-day moving average
    HighVol    — SPY 20-day volatility significantly above long-term average
    Normal     — default (e.g. near the 200 MA or insufficient history)

When multiple conditions could apply, HighVol takes precedence, then Bull/Bear.
"""

from __future__ import annotations

import pandas as pd

SPY_TICKER = "SPY"
LOOKBACK_DAYS = 250  # 200 MA + buffer
VOL_WINDOW = 20
LONG_TERM_VOL_WINDOW = 200
HIGH_VOL_MULTIPLIER = 1.5  # 20d vol > this × long-term avg → HighVol


def detect_regimes(start_date: str, end_date: str) -> pd.Series:
    """
    Detect market regime for each trading day in [start_date, end_date].

    Returns a Series with DatetimeIndex and values in {"Bull", "Bear", "HighVol", "Normal"}.
    Uses only data available at or before each date (no look-ahead).
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    import yfinance as yf  # lazy: avoids pulling websockets stack at import time

    dl_start = start - pd.Timedelta(days=LOOKBACK_DAYS)
    dl_end = end + pd.Timedelta(days=5)

    raw = yf.download(SPY_TICKER, start=dl_start, end=dl_end, progress=False)
    if raw.empty:
        return pd.Series(dtype=object)
    if isinstance(raw.columns, pd.MultiIndex):
        raw.columns = raw.columns.get_level_values(0)

    close = raw["Close"]
    daily_ret = close.pct_change()

    # All backward-looking: no look-ahead
    sma200 = close.rolling(200).mean()
    vol_20 = daily_ret.rolling(VOL_WINDOW).std()
    vol_20_long_term_avg = vol_20.rolling(LONG_TERM_VOL_WINDOW).mean()
    high_vol = vol_20 > (HIGH_VOL_MULTIPLIER * vol_20_long_term_avg)

    regime = pd.Series("Normal", index=close.index, dtype=object)
    # High volatility takes precedence
    regime.loc[high_vol.fillna(False)] = "HighVol"
    # Bull: price above 200 MA (where not already HighVol)
    bull_mask = (close > sma200) & (regime == "Normal")
    regime.loc[bull_mask] = "Bull"
    # Bear: price below 200 MA (where not already HighVol)
    bear_mask = (close < sma200) & (regime == "Normal")
    regime.loc[bear_mask] = "Bear"

    # Restrict to requested range
    mask = (regime.index >= start) & (regime.index <= end)
    return regime[mask].copy()


def get_regime_series_for_dates(dates: pd.DatetimeIndex, start_date: str, end_date: str) -> pd.Series:
    """
    Return a Series mapping each date in *dates* to its regime.
    Uses detect_regimes(start_date, end_date) and reindexes to *dates*.
    """
    full = detect_regimes(start_date, end_date)
    return full.reindex(dates).fillna("Normal")
