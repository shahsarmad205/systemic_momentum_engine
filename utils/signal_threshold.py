"""
Signal threshold utilities.

These helpers adjust signal classification thresholds based on
transaction costs, expected holding period and volatility regime.
"""

from __future__ import annotations

from typing import Tuple


def compute_cost_adjusted_threshold(
    commission_bps: float = 10.0,
    spread_bps: float = 5.0,
    slippage_bps: float = 5.0,
    holding_period_days: int = 5,
    base_threshold: float = 0.5,
    vol_percentile: float = 0.5,
) -> float:
    """
    Compute the minimum |adjusted_score| required for a trade to be worth taking.

    Example
    -------
    A 20 bps round-trip cost with a 5-day holding period implies a
    4 bps/day drag:
        round_trip_cost = 0.002
        daily_cost_drag = 0.002 / 5 = 0.0004
        cost_penalty    = 0.0004 * 100 = 0.04
    so the effective threshold becomes base_threshold + 0.04.

    Rationale
    ---------
    - Holding period matters because transaction costs must be
      amortised over the number of holding days; a shorter expected
      hold means higher per-day drag and therefore a higher score
      hurdle.
    - vol_percentile matters because high-volatility regimes tend to
      have wider effective spreads and noisier signals, so we widen
      the band before acting.
    """
    total_side_bps = float(commission_bps) + float(spread_bps) + float(slippage_bps)
    round_trip_cost = (total_side_bps * 2.0) / 10_000.0

    hp = max(1, int(holding_period_days))
    daily_cost_drag = round_trip_cost / float(hp)
    cost_penalty = daily_cost_drag * 100.0

    vol_adj = (float(vol_percentile) - 0.5) * 0.2
    vol_penalty = max(vol_adj, 0.0)  # only tighten in high-vol regimes

    threshold = float(base_threshold) + cost_penalty + vol_penalty
    # Never relax below the base threshold.
    return max(float(base_threshold), threshold)


def compute_dynamic_thresholds(config, vol_percentile: float = 0.5) -> Tuple[float, float]:
    """
    Return (bullish_threshold, bearish_threshold) using BacktestConfig.

    Uses config fields:
        execution_costs_commission_bps,
        execution_costs_spread_bps,
        execution_costs_slippage_bps,
        holding_period_days,
        dynamic_thresholds_enabled,
        base_signal_threshold,
        min_signal_strength        # optional; overrides base_signal_threshold when present.

    bearish_threshold is the symmetric negative of bullish_threshold by default.
    """
    # Prefer user-configured min_signal_strength if available; otherwise fall
    # back to base_signal_threshold (and finally to 0.5). This keeps the
    # effective thresholds tunable from YAML instead of hard-coded.
    if hasattr(config, "min_signal_strength"):
        base = float(getattr(config, "min_signal_strength"))
    else:
        base = float(getattr(config, "base_signal_threshold", 0.5))
    if not getattr(config, "dynamic_thresholds_enabled", True):
        return base, -base

    commission_bps = float(getattr(config, "execution_costs_commission_bps", 10.0))
    spread_bps = float(getattr(config, "execution_costs_spread_bps", 5.0))
    slippage_bps = float(getattr(config, "execution_costs_slippage_bps", 5.0))
    holding_period_days = int(getattr(config, "holding_period_days", 5))

    bull = compute_cost_adjusted_threshold(
        commission_bps=commission_bps,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        holding_period_days=holding_period_days,
        base_threshold=base,
        vol_percentile=vol_percentile,
    )
    bear = -bull
    return bull, bear

