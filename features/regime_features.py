"""
Single-name regime labelling based on price trend and volatility regime.

This module derives:
    - a coarse trend regime (bull / bear / sideways),
    - a volatility regime (low / normal / high),
    - a combined regime score that down-weights signals in high-vol
      environments and emphasises strong, low-vol trends.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def detect_market_regime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Detect per-name trend and volatility regimes and add a regime score.

    Financial intuition
    -------------------
    - Bull/bear/sideways labels follow classic moving-average
      relationships (50d vs 200d and price vs trend).
    - Volatility regime is derived from a percentile of realised
      volatility, distinguishing quiet vs stressed environments.
    - The regime score multiplies a trend sign (+1, 0, -1) by a
      volatility multiplier (low=1, normal=0.5, high=0), effectively
      muting directional conviction in very high-vol periods.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with at least a ``'close'`` column and typically the
        trend-agent outputs ``'ma_50'`` and ``'ma_200'`` and the
        volatility feature ``'volatility_20'`` / ``'volatility_percentile'``.

    Returns
    -------
    pd.DataFrame
        Same object with new columns:
        ``trend_regime``, ``volatility_regime``, ``regime_score``.
    """
    if "close" not in df.columns:
        raise KeyError("detect_market_regime requires a 'close' column.")

    out = df.copy()
    close = pd.to_numeric(out["close"], errors="coerce")

    if "ma_50" in out.columns and "ma_200" in out.columns:
        ma50 = pd.to_numeric(out["ma_50"], errors="coerce")
        ma200 = pd.to_numeric(out["ma_200"], errors="coerce")
    else:
        ma50 = close.rolling(window=50, min_periods=10).mean()
        ma200 = close.rolling(window=200, min_periods=40).mean()

    bull_mask = (close > ma50) & (close > ma200) & (ma50 > ma200)
    bear_mask = (close < ma50) & (close < ma200) & (ma50 < ma200)

    trend_regime = pd.Series("sideways", index=out.index, dtype=object)
    trend_regime.loc[bull_mask] = "bull_trend"
    trend_regime.loc[bear_mask] = "bear_trend"
    out["trend_regime"] = trend_regime

    if "volatility_percentile" in out.columns:
        vol_pct = pd.to_numeric(out["volatility_percentile"], errors="coerce")
    elif "volatility_20" in out.columns:
        vol = pd.to_numeric(out["volatility_20"], errors="coerce")

        def _pct(x: pd.Series) -> float:
            r = x.rank(pct=True)
            return float(r.iloc[-1])

        vol_pct = vol.rolling(window=252, min_periods=20).apply(_pct, raw=False)
    else:
        vol_pct = pd.Series(0.5, index=out.index)

    vol_regime = pd.Series("normal_vol", index=out.index, dtype=object)
    vol_regime.loc[vol_pct < 0.33] = "low_vol"
    vol_regime.loc[vol_pct > 0.67] = "high_vol"
    out["volatility_regime"] = vol_regime

    trend_score = pd.Series(0.0, index=out.index)
    trend_score.loc[trend_regime == "bull_trend"] = 1.0
    trend_score.loc[trend_regime == "bear_trend"] = -1.0

    vol_mult = pd.Series(0.5, index=out.index)
    vol_mult.loc[vol_regime == "low_vol"] = 1.0
    vol_mult.loc[vol_regime == "high_vol"] = 0.0

    regime_score = (trend_score * vol_mult).clip(-1.0, 1.0)
    out["regime_score"] = regime_score

    return out

