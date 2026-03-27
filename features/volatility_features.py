"""
Volatility feature construction for single-name time series.

This module measures realised volatility over multiple horizons and
derives simple regime-style indicators (spikes, percentile, and trend)
to capture the structure of risk in the return distribution.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _get_returns_for_vol(df: pd.DataFrame) -> pd.Series:
    """Prefer an existing daily_return column; otherwise use log returns."""
    if "daily_return" in df.columns:
        return pd.to_numeric(df["daily_return"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    return np.log(close).diff()


def calculate_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add realised volatility and volatility-structure features.

    Financial intuition
    -------------------
    - Annualised 20d realised volatility summarises recent risk.
    - Volatility spikes versus a 60d median highlight stress episodes.
    - A long-horizon percentile of volatility shows where current risk
      sits relative to the past year.
    - Volatility trend (sign of 5d change) indicates whether risk is
      rising or falling.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with at least a ``'close'`` column and
        optionally a ``'daily_return'`` column.

    Returns
    -------
    pd.DataFrame
        Same object with new columns appended:
        ``volatility_20``, ``volatility_spike``,
        ``volatility_percentile``, ``volatility_trend``.
    """
    if "close" not in df.columns:
        raise KeyError("calculate_volatility_features requires a 'close' column.")

    out = df.copy()
    rets = _get_returns_for_vol(out)

    vol_20 = (
        rets.rolling(window=20, min_periods=10)
        .std()
        .astype(float)
        * np.sqrt(252.0)
    )
    out["volatility_20"] = vol_20

    med_60 = vol_20.rolling(window=60, min_periods=20).median()
    spike = (vol_20 > 1.5 * med_60).astype(float)
    out["volatility_spike"] = spike

    def _pct_in_window(x: pd.Series) -> float:
        r = x.rank(pct=True)
        return float(r.iloc[-1])

    vol_pct = vol_20.rolling(window=252, min_periods=20).apply(
        _pct_in_window, raw=False
    )
    out["volatility_percentile"] = vol_pct.clip(lower=0.0, upper=1.0)

    out["volatility_trend"] = np.sign(vol_20 - vol_20.shift(5))

    return out

