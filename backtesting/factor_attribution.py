"""
Factor Attribution and P&L Decomposition
==========================================
Institutional performance attribution system that decomposes realized P&L into:
  1. Factor contributions (market beta, size, momentum, value, quality)
  2. Alpha vs. beta decomposition
  3. Transaction cost attribution
  4. Sector allocation effects

Implements the Brinson-Fachler attribution methodology adapted for
quantitative factor portfolios.

References:
  - Brinson, Hood, Beebower (1986) "Determinants of Portfolio Performance"
  - Brinson, Singer, Beebower (1991) "Determinants of Portfolio Performance II"
  - Menchero (2002) "A Multi-Factor Approach to Performance Attribution"
  - AQR "Understanding Factor Performance" (2018)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class FactorAttributionResult:
    """Results from factor attribution analysis."""
    date: pd.Timestamp
    total_return: float
    # Factor contributions
    factor_returns: dict[str, float] = field(default_factory=dict)
    factor_contributions: dict[str, float] = field(default_factory=dict)
    # Alpha decomposition
    alpha_component: float = 0.0
    beta_component: float = 0.0
    residual: float = 0.0
    # Transaction costs
    total_costs: float = 0.0
    market_impact_cost: float = 0.0
    spread_cost: float = 0.0
    commission_cost: float = 0.0
    # Sector allocation
    sector_allocation_effect: float = 0.0
    stock_selection_effect: float = 0.0
    interaction_effect: float = 0.0
    # Risk-adjusted metrics
    information_ratio: float = 0.0
    active_return: float = 0.0
    active_risk: float = 0.0
    # Metadata
    n_positions: int = 0
    gross_exposure: float = 0.0
    net_exposure: float = 0.0


class FactorAttribution:
    """
    Factor-based P&L attribution engine.

    Decomposes portfolio returns into:
      r_p = Σ (w_f × r_f) + α + ε

    where:
      w_f = portfolio exposure to factor f
      r_f = factor f return
      α = residual alpha (not explained by factors)
      ε = noise
    """

    def __init__(
        self,
        factor_names: list[str] | None = None,
        benchmark_ticker: str = "SPY",
        risk_free_rate: float = 0.02,
    ):
        self.factor_names = factor_names or ["market_beta", "size", "momentum", "value", "quality"]
        self.benchmark_ticker = benchmark_ticker
        self.risk_free_rate = risk_free_rate
        self._history: list[FactorAttributionResult] = []

    def compute_attribution(
        self,
        weights: dict[str, float],
        factor_exposures: dict[str, dict[str, float]],
        factor_returns: dict[str, float],
        realized_returns: dict[str, float],
        transaction_costs: dict[str, float] | None = None,
        sector_weights: dict[str, float] | None = None,
        benchmark_sector_weights: dict[str, float] | None = None,
        benchmark_return: float = 0.0,
        date: pd.Timestamp | None = None,
    ) -> FactorAttributionResult:
        """
        Compute factor attribution for a given period.

        Parameters
        ----------
        weights : dict
            Portfolio weights by ticker
        factor_exposures : dict
            Factor exposures by ticker: {factor: {ticker: exposure}}
        factor_returns : dict
            Factor returns: {factor: return}
        realized_returns : dict
            Realized returns by ticker
        transaction_costs : dict, optional
            Costs by ticker
        sector_weights : dict, optional
            Portfolio sector weights
        benchmark_sector_weights : dict, optional
            Benchmark sector weights
        benchmark_return : float
            Benchmark portfolio return
        date : pd.Timestamp, optional
            Attribution date

        Returns
        -------
        FactorAttributionResult
        """
        date = date or pd.Timestamp.today()

        # Portfolio return
        port_return = sum(weights.get(t, 0) * realized_returns.get(t, 0) for t in weights)

        # Factor contributions: exposure × factor return
        factor_contributions = {}
        for fname in self.factor_names:
            exposures = factor_exposures.get(fname, {})
            exposure = sum(weights.get(t, 0) * exposures.get(t, 0) for t in weights)
            f_return = factor_returns.get(fname, 0.0)
            factor_contributions[fname] = exposure * f_return

        # Beta component: market beta × market return
        market_beta = factor_contributions.get("market_beta", 0.0)
        market_return = factor_returns.get("market_beta", benchmark_return)
        beta_component = market_beta

        # Alpha component: return not explained by factors
        total_factor_return = sum(factor_contributions.values())
        alpha_component = port_return - total_factor_return
        residual = alpha_component  # In factor model, alpha = residual

        # Transaction cost attribution
        total_costs = 0.0
        market_impact_cost = 0.0
        spread_cost = 0.0
        commission_cost = 0.0
        if transaction_costs:
            for ticker, cost in transaction_costs.items():
                total_costs += cost
                # Simplified cost decomposition
                market_impact_cost += cost * 0.6  # ~60% market impact
                spread_cost += cost * 0.25        # ~25% spread
                commission_cost += cost * 0.15    # ~15% commission

        # Brinson-Fachler sector attribution
        sector_alloc = 0.0
        stock_select = 0.0
        interaction = 0.0
        if sector_weights and benchmark_sector_weights:
            all_sectors = set(sector_weights.keys()) | set(benchmark_sector_weights.keys())
            for sector in all_sectors:
                w_p = sector_weights.get(sector, 0.0)
                w_b = benchmark_sector_weights.get(sector, 0.0)
                r_p = sum(
                    weights.get(t, 0) * realized_returns.get(t, 0)
                    for t in weights
                    if t in factor_exposures.get(f"sector:{sector}", {})
                )
                r_b = benchmark_return / max(len(benchmark_sector_weights), 1)  # Simplified

                sector_alloc += (w_p - w_b) * (r_b - benchmark_return)
                stock_select += w_b * (r_p - r_b)
                interaction += (w_p - w_b) * (r_p - r_b)

        # Active return and risk
        active_return = port_return - benchmark_return

        # Information ratio (annualized, using 1-day estimate)
        active_risk = abs(active_return) * np.sqrt(252) * 0.1  # Rough estimate
        information_ratio = active_return / active_risk if active_risk > 1e-8 else 0.0

        result = FactorAttributionResult(
            date=date,
            total_return=port_return,
            factor_returns=factor_returns,
            factor_contributions=factor_contributions,
            alpha_component=alpha_component,
            beta_component=beta_component,
            residual=residual,
            total_costs=total_costs,
            market_impact_cost=market_impact_cost,
            spread_cost=spread_cost,
            commission_cost=commission_cost,
            sector_allocation_effect=sector_alloc,
            stock_selection_effect=stock_select,
            interaction_effect=interaction,
            information_ratio=information_ratio,
            active_return=active_return,
            active_risk=active_risk,
            n_positions=sum(1 for w in weights.values() if abs(w) > 1e-6),
            gross_exposure=sum(abs(w) for w in weights.values()),
            net_exposure=sum(weights.values()),
        )

        self._history.append(result)
        return result

    def summarize(self, window: int | None = None) -> dict[str, Any]:
        """
        Create summary attribution report.

        Parameters
        ----------
        window : int, optional
            Number of recent periods to include
        """
        history = self._history[-window:] if window else self._history
        if not history:
            return {"n_periods": 0}

        # Aggregate factor contributions
        factor_agg: dict[str, float] = {}
        for result in history:
            for fname, contrib in result.factor_contributions.items():
                factor_agg[fname] = factor_agg.get(fname, 0.0) + contrib

        total_return = sum(r.total_return for r in history)
        total_alpha = sum(r.alpha_component for r in history)
        total_beta = sum(r.beta_component for r in history)
        total_costs = sum(r.total_costs for r in history)

        # Average information ratio
        ir_vals = [r.information_ratio for r in history if r.active_risk > 1e-8]
        avg_ir = float(np.mean(ir_vals)) if ir_vals else 0.0

        return {
            "n_periods": len(history),
            "total_return": total_return,
            "total_alpha": total_alpha,
            "total_beta": total_beta,
            "alpha_pct": total_alpha / total_return if total_return != 0 else 0.0,
            "beta_pct": total_beta / total_return if total_return != 0 else 0.0,
            "total_costs": total_costs,
            "net_return": total_return - total_costs,
            "avg_information_ratio": avg_ir,
            "factor_contributions": factor_agg,
            "avg_positions": float(np.mean([r.n_positions for r in history])),
            "avg_gross": float(np.mean([r.gross_exposure for r in history])),
            "avg_net": float(np.mean([r.net_exposure for r in history])),
            "period_details": [
                {
                    "date": str(r.date.date()),
                    "return": r.total_return,
                    "alpha": r.alpha_component,
                    "beta": r.beta_component,
                    "costs": r.total_costs,
                    "n_pos": r.n_positions,
                }
                for r in history
            ],
        }
