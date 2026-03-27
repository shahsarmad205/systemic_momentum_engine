"""
Volatility-scaled position sizing utilities.

These helpers adjust raw position weights so that higher-volatility
names receive smaller allocations and lower-volatility names receive
larger allocations, targeting a more uniform risk contribution per
position.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def compute_vol_scaled_weight(
    raw_weight: float,
    volatility_20: float,
    target_vol: float = 0.15,
    min_vol_floor: float = 0.05,
    max_scale_cap: float = 3.0,
) -> float:
    """
    Scale a raw position weight inversely by realised 20d volatility.

    Financial intuition
    -------------------
    - ``target_vol`` is the desired annualised volatility contribution
      per position; typical equity CTA values are around 10–20% (0.10
      to 0.20).
    - ``min_vol_floor`` prevents division by extremely small
      volatility (e.g. illiquid or stale-priced names) which would
      otherwise explode the position size.
    - ``max_scale_cap`` limits how much we can lever up low-vol names
      to avoid over-concentration in supposedly safe assets.

    Parameters
    ----------
    raw_weight : float
        Baseline position weight or scale before volatility adjustment.
    volatility_20 : float
        Annualised 20-day realised volatility for the instrument.
    target_vol : float, default 0.15
        Target annual volatility contribution per position.
    min_vol_floor : float, default 0.05
        Minimum volatility level used in the denominator to avoid
        division by near-zero volatility.
    max_scale_cap : float, default 3.0
        Maximum allowed leverage multiplier applied to ``raw_weight``.

    Returns
    -------
    float
        Volatility-scaled weight.
    """
    vol = max(float(volatility_20), float(min_vol_floor))
    scale = float(target_vol) / vol
    if scale > max_scale_cap:
        scale = float(max_scale_cap)
    result = float(raw_weight) * scale
    # Normalise tiny floating-point noise so simple equality checks pass.
    result = float(round(result, 12))
    if result == -0.0:  # normalise signed zero to plain zero
        result = 0.0
    return result


def compute_portfolio_vol_weights(
    weights: dict[str, float],
    volatilities: dict[str, float],
    target_vol: float = 0.15,
) -> dict[str, float]:
    """
    Apply volatility scaling to a portfolio and renormalise absolute weights.

    Financial intuition
    -------------------
    - Each name's raw weight is first scaled by its volatility via
      ``compute_vol_scaled_weight`` so that high-vol names shrink and
      low-vol names grow.
    - We then normalise so that the sum of absolute weights equals 1,
      preserving a consistent gross exposure budget.

    Parameters
    ----------
    weights : dict[str, float]
        Mapping of ticker -> raw weight (can be positive or negative).
    volatilities : dict[str, float]
        Mapping of ticker -> annualised 20d volatility.  Missing
        entries fall back to ``target_vol`` for that ticker.
    target_vol : float, default 0.15
        Target annual volatility per position used in the scaling.

    Returns
    -------
    dict[str, float]
        New mapping of ticker -> volatility-scaled, normalised weight.
    """
    scaled: dict[str, float] = {}
    for ticker, w in weights.items():
        vol = float(volatilities.get(ticker, target_vol))
        scaled[ticker] = compute_vol_scaled_weight(w, vol, target_vol=target_vol)

    total_abs = sum(abs(v) for v in scaled.values())
    if total_abs <= 0.0:
        return {k: 0.0 for k in weights.keys()}

    return {k: v / total_abs for k, v in scaled.items()}


def compute_realized_vol_annualized(returns: pd.Series, window: int = 20) -> pd.Series:
    """Rolling std of daily returns, annualised (sqrt 252)."""
    s = returns.astype(float)
    std = s.rolling(window, min_periods=1).std(ddof=0)
    return (std * np.sqrt(252.0)).astype(float)


def compute_vol_target_scaling_factor(
    vol: float | np.floating[Any],
    *,
    target_vol: float = 0.15,
    min_vol_floor: float = 0.05,
    max_scale_cap: float = 3.0,
) -> float:
    v = max(float(vol), float(min_vol_floor))
    scale = float(target_vol) / v
    return float(min(scale, float(max_scale_cap)))


def apply_vol_kill_switch(
    positions: float | np.ndarray | pd.Series,
    vol: float | np.ndarray | pd.Series,
    *,
    threshold_annual: float,
    cut_factor: float,
) -> float | np.ndarray | pd.Series:
    """
    When annualised vol exceeds ``threshold_annual``, scale position by ``cut_factor``.
    Supports scalar, numpy array, or aligned Series.
    """
    thr = float(threshold_annual)
    cf = float(cut_factor)

    if isinstance(positions, pd.Series):
        vv = pd.to_numeric(vol, errors="coerce").astype(float)
        v_arr = vv.to_numpy()
        trig = np.isfinite(v_arr) & (v_arr > thr)
        out = positions.astype(float).where(~pd.Series(trig, index=positions.index), positions.astype(float) * cf)
        return out

    if isinstance(positions, np.ndarray):
        v_arr = np.asarray(vol, dtype=float)
        p_arr = np.asarray(positions, dtype=float)
        trig = np.isfinite(v_arr) & (v_arr > thr)
        out = np.where(trig, p_arr * cf, p_arr)
        return out.astype(float)

    pv = float(vol)
    p = float(positions)
    if not np.isfinite(pv) or pv <= thr:
        return p
    return p * cf

