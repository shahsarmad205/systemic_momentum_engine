"""
Liquidity feature construction from OHLCV data.

This module measures trading activity and simple turnover proxies,
which help control for how easy it is to enter and exit positions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def calculate_liquidity_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add volume- and dollar-volume-based liquidity features.

    Financial intuition
    -------------------
    - 20d average volume captures typical trading activity.
    - Volume spikes indicate unusually heavy participation or flow.
    - Dollar volume scales volume by price, making liquidity
      comparable across stocks.
    - Turnover ratio compares recent dollar flow to a long-run
      baseline, flagging regime shifts in liquidity.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with ``'close'`` and ``'volume'`` columns.

    Returns
    -------
    pd.DataFrame
        Same object with new columns appended:
        ``volume_mean``, ``volume_spike``, ``dollar_volume``,
        ``turnover_ratio``.
    """
    if "close" not in df.columns or "volume" not in df.columns:
        raise KeyError("calculate_liquidity_features requires 'close' and 'volume' columns.")

    out = df.copy()
    vol = pd.to_numeric(out["volume"], errors="coerce")
    close = pd.to_numeric(out["close"], errors="coerce")

    vol_mean_20 = vol.rolling(window=20, min_periods=5).mean()
    raw_vol_mean = vol_mean_20.copy()
    # Z-score normalise volume_mean so scale matches other features
    vm_mean = raw_vol_mean.rolling(252, min_periods=60).mean()
    vm_std = raw_vol_mean.rolling(252, min_periods=60).std().replace(0, np.nan)
    out["volume_mean"] = ((raw_vol_mean - vm_mean) / vm_std).replace([np.inf, -np.inf], np.nan)

    # Volume spike: short-term volume / 20d mean, then z-score over 252d window
    raw_spike = (vol / vol_mean_20).replace([np.inf, -np.inf], np.nan)
    spike_mean = raw_spike.rolling(252, min_periods=60).mean()
    spike_std = raw_spike.rolling(252, min_periods=60).std().replace(0, np.nan)
    out["volume_spike"] = ((raw_spike - spike_mean) / spike_std).replace([np.inf, -np.inf], np.nan)

    dollar_vol = close * vol
    dv_20 = dollar_vol.rolling(window=20, min_periods=5).mean()
    dv_252 = dollar_vol.rolling(window=252, min_periods=60).mean()
    raw_ratio = (dv_20 / dv_252).replace([np.inf, -np.inf], np.nan)
    ratio_mean = raw_ratio.rolling(252, min_periods=60).mean()
    ratio_std = raw_ratio.rolling(252, min_periods=60).std().replace(0, np.nan)
    turnover_ratio = ((raw_ratio - ratio_mean) / ratio_std).replace([np.inf, -np.inf], np.nan)
    # Z-score normalise dollar_volume so scale matches other features
    dv_mean = dollar_vol.rolling(252, min_periods=60).mean()
    dv_std = dollar_vol.rolling(252, min_periods=60).std().replace(0, np.nan)
    out["dollar_volume"] = ((dollar_vol - dv_mean) / dv_std).replace([np.inf, -np.inf], np.nan)
    out["turnover_ratio"] = turnover_ratio

    return out

