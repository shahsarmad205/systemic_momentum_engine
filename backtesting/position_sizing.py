from __future__ import annotations

"""Position sizing utilities (equal, vol-scaled, Kelly, risk parity, compose)."""

import math
from dataclasses import dataclass


@dataclass
class PositionSizingParams:
    method: str = "equal"
    max_position_pct_of_equity: float = 0.25
    target_risk_fraction: float = 0.15  # fraction of equity risk budget per position (for vol_scaled / risk_parity)
    kelly_fraction: float = 0.5
    kelly_win_rate: float = 0.55
    kelly_avg_win_return: float = 0.02
    kelly_avg_loss_return: float = 0.015


def _base_equal_size(equity: float, max_positions: int, position_scale: float) -> float:
    if max_positions <= 0:
        return 0.0
    base = equity / max_positions
    return max(0.0, base * position_scale)


def _apply_cap(size: float, equity: float, max_position_pct_of_equity: float) -> float:
    cap = equity * max_position_pct_of_equity
    if cap > 0:
        size = min(size, cap)
    return max(0.0, size)


def equal_size(
    equity: float,
    max_positions: int,
    position_scale: float,
    params: PositionSizingParams,
) -> float:
    size = _base_equal_size(equity, max_positions, position_scale)
    return _apply_cap(size, equity, params.max_position_pct_of_equity)


def vol_scaled_size(
    equity: float,
    max_positions: int,
    position_scale: float,
    stock_vol_annual: float | None,
    params: PositionSizingParams,
) -> float:
    """
    Volatility-scaled position sizing.

    Approximate formula:
        size_dollars ≈ (target_risk_fraction * equity) / stock_volatility
    capped by equal-weight size and max_position_pct_of_equity.
    """
    size_eq = _base_equal_size(equity, max_positions, position_scale)
    if stock_vol_annual is None or stock_vol_annual <= 1e-8:
        return _apply_cap(size_eq, equity, params.max_position_pct_of_equity)

    risk_budget = equity * max(params.target_risk_fraction, 0.0)
    # Avoid pathological huge leverage when vol is extremely small
    eff_vol = max(stock_vol_annual, 1e-3)
    size_vol = risk_budget / eff_vol
    # Do not exceed 3x the equal-weight allocation before global cap
    size = min(size_vol, size_eq * 3.0)
    return _apply_cap(size, equity, params.max_position_pct_of_equity)


def kelly_size(
    equity: float,
    max_positions: int,
    position_scale: float,
    params: PositionSizingParams,
) -> float:
    """
    Kelly-style sizing: f* = (p*b - q)/b, half-Kelly f = f* * kelly_fraction, position = equity * f.

    p = win_rate, q = 1-p, b = avg_win_return / avg_loss_return.
    Uses params (config or rolling from trade history). Position size = equity * f, capped.
    """
    p = params.kelly_win_rate
    q = 1.0 - p
    if params.kelly_avg_loss_return <= 0:
        b = 2.0
    else:
        b = params.kelly_avg_win_return / params.kelly_avg_loss_return

    full_kelly = (p * b - q) / b if b > 0 else 0.0
    full_kelly = max(0.0, min(full_kelly, 1.0))
    # Half-Kelly (or other fraction): f = f* * kelly_fraction; position size = equity * f
    f = full_kelly * params.kelly_fraction
    size = equity * f * position_scale
    # Minimum: at least half of equal-weight so we still open positions when Kelly is small
    size_eq = _base_equal_size(equity, max_positions, 1.0)
    size = max(size, size_eq * 0.5) if size_eq > 0 else size
    return _apply_cap(size, equity, params.max_position_pct_of_equity)


def risk_parity_size(
    equity: float,
    max_positions: int,
    position_scale: float,
    stock_vol_annual: float | None,
    params: PositionSizingParams,
) -> float:
    """
    Single-asset approximation of risk parity.

    In the multi-asset case, equal risk contribution requires using the
    full covariance matrix. For a single asset in isolation, this
    degenerates to inverse-vol weighting, which is equivalent to the
    vol_scaled formulation with the same target_risk_fraction.
    """
    return vol_scaled_size(
        equity=equity,
        max_positions=max_positions,
        position_scale=position_scale,
        stock_vol_annual=stock_vol_annual,
        params=params,
    )


def compose_position_size(
    equity: float,
    weight: float,
    vol_scaling: float,
    regime_scaling: float,
    max_single_position_pct: float = 0.12,
    *,
    long_only: bool = True,
    min_single_position_pct: float | None = None,
) -> float:
    """
    Dollar size from multiplicative factors, capped per name as fraction of equity.

    Negative weights represent shorts when ``long_only`` is False.
    """
    if equity <= 0 or not math.isfinite(equity):
        return 0.0
    if not all(
        math.isfinite(x) for x in (weight, vol_scaling, regime_scaling, max_single_position_pct)
    ):
        return 0.0
    if long_only and weight < 0:
        return 0.0

    raw = float(equity) * float(weight) * float(vol_scaling) * float(regime_scaling)
    if min_single_position_pct is not None:
        thresh = float(equity) * float(min_single_position_pct)
        if abs(raw) < thresh:
            return 0.0

    cap = float(equity) * float(max_single_position_pct)
    if raw >= 0:
        return float(min(raw, cap))
    return float(max(raw, -cap))

