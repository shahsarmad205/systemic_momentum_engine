"""
Market breadth indicators across a universe of tickers.

This module aggregates per-name prices into simple breadth measures:
    - percentage of names above key moving averages,
    - an advance/decline ratio based on daily price moves.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_market_breadth(
    market_data_dict: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Compute simple market breadth metrics across a set of tickers.

    Financial intuition
    -------------------
    - The fraction of names above their 50d/200d moving averages
      gives a cross-sectional view of trend strength.
    - The advance/decline ratio compares the number of rising vs
      falling stocks on each day, a classic internal market health
      indicator.

    Parameters
    ----------
    market_data_dict : dict[str, pd.DataFrame]
        Mapping of ticker -> OHLCV DataFrame. Each DataFrame must at
        least contain a ``'close'`` column; index should be dates.

    Returns
    -------
    pd.DataFrame
        Date-indexed DataFrame with columns:
        ``percent_above_50ma``, ``percent_above_200ma``,
        ``advance_decline_ratio``.
        Returns an empty DataFrame with those columns if fewer than
        two tickers are provided.
    """
    tickers = list(market_data_dict.keys())
    if len(tickers) < 2:
        return pd.DataFrame(
            columns=["percent_above_50ma", "percent_above_200ma", "advance_decline_ratio"]
        )

    close_frames = []
    for tk, df in market_data_dict.items():
        if df is None or "close" not in df.columns:
            continue
        s = pd.to_numeric(df["close"], errors="coerce")
        s.name = tk
        close_frames.append(s)

    if not close_frames:
        return pd.DataFrame(
            columns=["percent_above_50ma", "percent_above_200ma", "advance_decline_ratio"]
        )

    closes = pd.concat(close_frames, axis=1).sort_index().ffill()

    ma50 = closes.rolling(window=50, min_periods=10).mean()
    ma200 = closes.rolling(window=200, min_periods=40).mean()

    above_50 = (closes > ma50) & ma50.notna()
    above_200 = (closes > ma200) & ma200.notna()

    valid_50 = above_50.sum(axis=1)
    valid_50 = valid_50.replace(0, np.nan)
    pct_above_50 = above_50.sum(axis=1) / valid_50

    valid_200 = above_200.sum(axis=1)
    valid_200 = valid_200.replace(0, np.nan)
    pct_above_200 = above_200.sum(axis=1) / valid_200

    rets = closes.pct_change()
    up = (rets > 0).sum(axis=1)
    down = (rets < 0).sum(axis=1)
    ad_ratio = up / (down + 1.0)

    breadth = pd.DataFrame(
        {
            "percent_above_50ma": pct_above_50,
            "percent_above_200ma": pct_above_200,
            "advance_decline_ratio": ad_ratio,
        }
    )
    return breadth

