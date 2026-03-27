"""
Momentum feature construction for single-name price series.

This module derives short-, medium-, and longer-horizon momentum
signals plus a simple trend-strength score, which help quantify the
direction and persistence of recent returns.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add medium/long horizon momentum and trend-strength features.

    Financial intuition
    -------------------
    - Short-horizon momentum (5d) captures very recent moves.
    - Medium (20d) and longer (60d) momentum proxy for trend persistence.
    - Momentum acceleration highlights whether momentum is speeding up
      or fading.
    - Trend strength ranks each day's 20d/60d momentum ratio within a
      rolling window and rescales it to [-1, 1], so you can compare
      how strong the trend is relative to its own recent history.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing at least a ``'close'`` column.
        Index is typically a DatetimeIndex, but any index is accepted.

    Returns
    -------
    pd.DataFrame
        Same object with new columns appended:
        ``momentum_5``, ``momentum_20``, ``momentum_60``,
        ``momentum_acceleration``, ``trend_strength``.
    """
    if "close" not in df.columns:
        raise KeyError("calculate_momentum_features requires a 'close' column.")

    out = df.copy()
    close = out["close"].astype(float)

    out["momentum_5"] = close.pct_change(5)
    out["momentum_20"] = close.pct_change(20)
    out["momentum_60"] = close.pct_change(60)

    out["momentum_acceleration"] = out["momentum_5"] - out["momentum_20"]

    ratio = out["momentum_20"] / out["momentum_60"].replace(0.0, np.nan)

    window = 63

    def _rolling_percentile(x: pd.Series) -> float:
        r = x.rank(pct=True)
        return float(r.iloc[-1])

    pct_rank = ratio.rolling(window=window, min_periods=10).apply(
        _rolling_percentile, raw=False
    )
    out["trend_strength"] = pct_rank * 2.0 - 1.0

    return out

