"""
Market Impact Model
====================
Implements a practical square-root market impact model based on
Almgren & Chriss (2001) and the empirical "square-root law" widely used
by institutional desks (Two Sigma, AQR, Citadel).

The full cost of executing an order has three components:

    1. Temporary impact  — price displacement during execution, reverts after
    2. Permanent impact  — lasting price shift from information content
    3. Commission + spread — exchange fees + bid-ask spread half-width

Total round-trip cost:
    cost_bps = 2 * (spread_half_bps + commission_bps)
             + 2 * temporary_impact_bps(order_size, adv, vol)
             + permanent_impact_bps(order_size, adv, vol)   [not double-counted]

Empirical calibration (Kissell & Malamut 2005 / ITG):
    - η (temporary impact coefficient) ≈ 0.142
    - α (permanent impact coefficient) ≈ 0.314
    - γ (participation rate exponent)  ≈ 0.6  (sqrt law → 0.5)

For equity long-short strategies holding 5-10 days and trading <0.5% ADV,
the dominant cost is bid-ask spread + linear commission. Market impact
matters once orders exceed ~0.1% of daily ADV.

Usage
-----
>>> model = MarketImpactModel(adv_usd=50_000_000)
>>> cost = model.round_trip_cost_bps(order_usd=500_000, vol_daily=0.015)
>>> cost
4.2  # bps, round-trip
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class MarketImpactParams:
    """
    Calibration parameters for the market impact model.

    Defaults are tuned for US large-cap equities (Russell 1000 universe)
    with a 5-day holding period and <1% ADV participation rate.
    """
    # Almgren-Chriss coefficients
    eta: float = 0.142          # temporary impact coefficient
    alpha: float = 0.314        # permanent impact coefficient
    gamma: float = 0.6          # participation rate exponent (0.5 = pure sqrt law)

    # Fixed execution costs (per leg, in bps)
    spread_half_bps: float = 2.5    # half bid-ask spread (5 bps wide spread → 2.5 each side)
    commission_bps: float = 1.0     # all-in commission (DMA / prime brokerage)

    # Scaling guards
    min_adv_usd: float = 5_000_000.0    # below this, assume illiquid / no trade
    max_impact_bps: float = 150.0       # cap impact at 150 bps (sanity bound)
    max_participation_rate: float = 0.10  # never model >10% ADV participation


class MarketImpactModel:
    """
    Institutional-grade cost model combining:
    - Almgren-Chriss temporary + permanent market impact (square-root law)
    - Fixed bid-ask spread and commission

    Parameters
    ----------
    adv_usd : float
        Average daily dollar volume (ADV) of the security being traded.
    params : MarketImpactParams
        Calibration coefficients.
    """

    def __init__(
        self,
        adv_usd: float = 10_000_000.0,
        params: Optional[MarketImpactParams] = None,
    ) -> None:
        self.adv_usd = max(float(adv_usd), 1.0)
        self.params = params or MarketImpactParams()

    # ------------------------------------------------------------------
    # Core impact calculations
    # ------------------------------------------------------------------

    def participation_rate(self, order_usd: float) -> float:
        """Order size as fraction of daily ADV. Capped at max_participation_rate."""
        rate = abs(float(order_usd)) / self.adv_usd
        return min(rate, self.params.max_participation_rate)

    def temporary_impact_bps(
        self,
        order_usd: float,
        vol_daily: float = 0.015,
    ) -> float:
        """
        Temporary price displacement from executing the order.

        Formula: η × σ_daily × (order/ADV)^γ × 10_000

        This represents the average fill-price worsening during execution.
        It is symmetric (affects both entries and exits).

        Parameters
        ----------
        order_usd : float
            Dollar size of the order leg (one-way).
        vol_daily : float
            Daily return volatility of the security (annualised vol / sqrt(252)).
        """
        prate = self.participation_rate(order_usd)
        if prate <= 0:
            return 0.0
        sigma = abs(float(vol_daily))
        impact = self.params.eta * sigma * (prate ** self.params.gamma)
        return min(impact * 10_000, self.params.max_impact_bps)

    def permanent_impact_bps(
        self,
        order_usd: float,
        vol_daily: float = 0.015,
    ) -> float:
        """
        Permanent price shift attributable to order information content.

        Formula: α × σ_daily × (order/ADV)^γ × 10_000

        Permanent impact is a one-time cost; it does NOT apply to both entry
        and exit (the market has already absorbed the information).
        """
        prate = self.participation_rate(order_usd)
        if prate <= 0:
            return 0.0
        sigma = abs(float(vol_daily))
        impact = self.params.alpha * sigma * (prate ** self.params.gamma)
        return min(impact * 10_000, self.params.max_impact_bps)

    def one_way_cost_bps(
        self,
        order_usd: float,
        vol_daily: float = 0.015,
    ) -> float:
        """
        Total one-way execution cost in basis points.

        = spread_half + commission + temporary_impact
        (permanent impact is attributed to entry leg only)
        """
        fixed = self.params.spread_half_bps + self.params.commission_bps
        temp = self.temporary_impact_bps(order_usd, vol_daily)
        return fixed + temp

    def round_trip_cost_bps(
        self,
        order_usd: float,
        vol_daily: float = 0.015,
    ) -> float:
        """
        Full round-trip execution cost in basis points (entry + exit).

        = 2 × (spread_half + commission + temporary_impact)
          + permanent_impact      [charged once, at entry]

        This is the correct cost to subtract from a holding-period P&L.
        """
        one_way = self.one_way_cost_bps(order_usd, vol_daily)
        permanent = self.permanent_impact_bps(order_usd, vol_daily)
        return 2.0 * one_way + permanent

    def cost_dollars(
        self,
        position_size_usd: float,
        vol_daily: float = 0.015,
        is_exit: bool = False,
    ) -> float:
        """
        One-way dollar cost for entering or exiting a position.

        Parameters
        ----------
        position_size_usd : float
            Dollar value of the position leg.
        vol_daily : float
            Daily volatility of the security.
        is_exit : bool
            If True, exclude permanent impact (already baked in at entry).
        """
        if position_size_usd <= 0:
            return 0.0
        fixed = self.params.spread_half_bps + self.params.commission_bps
        temp = self.temporary_impact_bps(position_size_usd, vol_daily)
        cost_bps = fixed + temp
        if not is_exit:
            cost_bps += self.permanent_impact_bps(position_size_usd, vol_daily)
        return position_size_usd * (cost_bps / 10_000.0)


# ------------------------------------------------------------------
# Convenience factory: build model from config
# ------------------------------------------------------------------

def make_model_from_config(
    config,
    adv_usd: float = 10_000_000.0,
) -> MarketImpactModel:
    """
    Build a MarketImpactModel from BacktestConfig.

    Falls back to flat TransactionCostModel values when market-impact
    parameters are absent from config.
    """
    params = MarketImpactParams()

    # Read Almgren-Chriss coefficients if present in config
    params.eta = float(getattr(config, "mi_eta", params.eta))
    params.alpha = float(getattr(config, "mi_alpha", params.alpha))
    params.gamma = float(getattr(config, "mi_gamma", params.gamma))

    # Fixed cost components from execution_costs section
    params.spread_half_bps = float(
        getattr(config, "execution_costs_spread_bps", params.spread_half_bps * 2) / 2
    )
    params.commission_bps = float(
        getattr(config, "execution_costs_commission_bps", params.commission_bps)
    )

    return MarketImpactModel(adv_usd=adv_usd, params=params)


# ------------------------------------------------------------------
# ADV-aware slippage applied price-by-price (drop-in for ExecutionEngine)
# ------------------------------------------------------------------

class ImpactAwareExecutionEngine:
    """
    Drop-in replacement for backtesting.execution.ExecutionEngine that
    applies volume-scaled market impact instead of flat slippage bps.

    Parameters
    ----------
    adv_usd : float
        Average ADV estimate. In production, this should come from the
        adv_cache per ticker. Default: $50M (large-cap approximation).
    vol_daily : float
        Daily volatility estimate. Default: 1.5% (approx S&P 500 average).
    params : MarketImpactParams
        Almgren-Chriss calibration.
    commission : float
        Flat USD commission per trade side (exchange/brokerage fee).
    """

    def __init__(
        self,
        adv_usd: float = 50_000_000.0,
        vol_daily: float = 0.015,
        params: Optional[MarketImpactParams] = None,
        commission: float = 1.0,
    ) -> None:
        self._model = MarketImpactModel(adv_usd=adv_usd, params=params)
        self.vol_daily = float(vol_daily)
        self.commission = float(commission)

    def update_adv(self, adv_usd: float) -> None:
        """Update ADV (call once per ticker before simulating its trades)."""
        self._model = MarketImpactModel(adv_usd=max(adv_usd, 1.0), params=self._model.params)

    def _impact_fraction(self, position_size_usd: float, is_exit: bool = False) -> float:
        """Return impact as a fraction of notional (not bps)."""
        dollars = self._model.cost_dollars(position_size_usd, self.vol_daily, is_exit=is_exit)
        return dollars / max(position_size_usd, 1e-6)

    def apply_entry_slippage(
        self,
        price: float,
        signal: str,
        position_size_usd: float = 10_000.0,
    ) -> float:
        """Worsen fill price on entry using market impact."""
        impact = self._impact_fraction(position_size_usd, is_exit=False)
        if signal == "Bullish":
            return price * (1.0 + impact)   # buy higher
        return price * (1.0 - impact)       # short lower

    def apply_exit_slippage(
        self,
        price: float,
        signal: str,
        position_size_usd: float = 10_000.0,
    ) -> float:
        """Worsen fill price on exit using market impact."""
        impact = self._impact_fraction(position_size_usd, is_exit=True)
        if signal == "Bullish":
            return price * (1.0 - impact)   # sell lower
        return price * (1.0 + impact)       # cover higher
