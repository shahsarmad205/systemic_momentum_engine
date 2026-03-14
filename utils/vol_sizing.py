"""
Volatility-scaled position sizing utilities.

These helpers adjust raw position weights so that higher-volatility
names receive smaller allocations and lower-volatility names receive
larger allocations, targeting a more uniform risk contribution per
position.
"""

from __future__ import annotations

from typing import Dict


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
    weights: Dict[str, float],
    volatilities: Dict[str, float],
    target_vol: float = 0.15,
) -> Dict[str, float]:
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
    scaled: Dict[str, float] = {}
    for ticker, w in weights.items():
        vol = float(volatilities.get(ticker, target_vol))
        scaled[ticker] = compute_vol_scaled_weight(w, vol, target_vol=target_vol)

    total_abs = sum(abs(v) for v in scaled.values())
    if total_abs <= 0.0:
        return {k: 0.0 for k in weights.keys()}

    return {k: v / total_abs for k, v in scaled.items()}

