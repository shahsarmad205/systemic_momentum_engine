"""
Transaction Cost Model
======================
Calculates execution cost per trade leg (entry or exit) using commission,
bid-ask spread, and slippage, all expressed in basis points relative to trade size.

Formula: total_cost_bps = commission_bps + spread_bps + slippage_bps
         cost_dollars = position_size * (total_cost_bps / 10_000)
"""

from __future__ import annotations


class TransactionCostModel:
    """
    Compute total execution cost per trade leg in bps and dollars.

    Parameters
    ----------
    commission_bps : float
        Commission cost in basis points (e.g. 2 = 0.02% of notional per side).
    spread_bps : float
        Bid-ask spread cost in basis points.
    slippage_bps : float
        Slippage cost in basis points (market impact / worse fill).
    """

    def __init__(
        self,
        commission_bps: float = 2.0,
        spread_bps: float = 2.0,
        slippage_bps: float = 1.0,
    ) -> None:
        self.commission_bps = float(commission_bps)
        self.spread_bps = float(spread_bps)
        self.slippage_bps = float(slippage_bps)

    @property
    def total_bps(self) -> float:
        """Total cost per leg in basis points: commission + spread + slippage."""
        return self.commission_bps + self.spread_bps + self.slippage_bps

    def cost_bps(self) -> float:
        """Alias for total_bps (cost in basis points per leg)."""
        return self.total_bps

    def cost_dollars(self, position_size: float) -> float:
        """
        Cost in dollars for one leg (entry or exit) given position size in dollars.

        cost_dollars = position_size * (total_bps / 10_000)
        """
        if position_size <= 0:
            return 0.0
        return position_size * (self.total_bps / 10_000.0)

    def cost_fraction(self) -> float:
        """Cost as a fraction of notional (total_bps / 10_000)."""
        return self.total_bps / 10_000.0
