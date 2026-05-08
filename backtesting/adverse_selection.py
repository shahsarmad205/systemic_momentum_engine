"""
Adverse Selection and Implementation Shortfall Analysis
=========================================================
Institutional analysis of whether signals are strong enough to overcome
trading costs and whether the strategy exhibits systematic patterns
that could be detected and front-run by other market participants.

Implements:
  1. Implementation shortfall benchmark (Perold 1988)
  2. Signal-to-cost ratio analysis
  3. Adverse selection detection (predictability of trading patterns)
  4. Market impact footprint analysis

References:
  - Perold (1988) "The Implementation Shortfall: Paper vs. Reality"
  - Hasbrouck (2007) "Empirical Market Microstructure"
  - Kissell (2013) "The Science of Algorithmic Trading and Portfolio Management"
  - AQR "The Limits of Arbitrage" (2016)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdverseSelectionConfig:
    """Configuration for adverse selection analysis."""
    enabled: bool = True
    min_signal_to_cost_ratio: float = 2.0  # Signal must be 2x cost to trade
    max_adverse_selection: float = 0.001  # Max acceptable adverse selection
    monitoring_window: int = 60  # Days for monitoring analysis
    predictability_threshold: float = 0.02  # R² threshold for pattern detection


@dataclass
class ImplementationShortfall:
    """
    Implementation shortfall decomposition for a single trade.

    IS = (P_execution - P_decision) × quantity + costs

    Components:
      - Delay cost: (P_order - P_decision) × quantity
      - Market impact: (P_fill - P_order) × quantity
      - Commission: explicit trading costs
      - Opportunity cost: (P_final - P_decision) × missed_quantity
    """
    ticker: str
    decision_price: float
    order_price: float
    fill_price: float
    final_price: float
    quantity: float
    commission: float = 0.0
    # Decomposed costs
    delay_cost: float = 0.0
    market_impact: float = 0.0
    opportunity_cost: float = 0.0
    # Total
    total_shortfall: float = 0.0
    shortfall_bps: float = 0.0  # As bps of notional


@dataclass
class AdverseSelectionReport:
    """Report on adverse selection risk."""
    date: pd.Timestamp
    avg_signal_to_cost_ratio: float = 0.0
    pct_trades_below_threshold: float = 0.0
    avg_implementation_shortfall_bps: float = 0.0
    adverse_selection_measure: float = 0.0
    pattern_predictability: float = 0.0  # R² of order flow predictability
    high_risk_trades: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)


class AdverseSelectionAnalyzer:
    """
    Analyzes adverse selection risk and implementation shortfall.

    Key institutional metrics:
      1. Signal-to-Cost Ratio: |α| / cost_per_trade
         - If < 1: expected alpha < cost → don't trade
         - If 1-2: marginal → only trade with high conviction
         - If > 2: profitable after costs

      2. Implementation Shortfall: difference between paper and actual returns
         - Decomposes into delay, impact, commission, opportunity costs

      3. Adverse Selection: correlation between order flow and subsequent returns
         - If positive: market moves against us after we trade → possible front-running
         - If negative: market moves in our favor → we're providing liquidity

      4. Pattern Predictability: R² of regressing order flow on lagged signals
         - High R² → our trading is predictable → others can front-run
    """

    def __init__(self, config: AdverseSelectionConfig | None = None):
        self.cfg = config or AdverseSelectionConfig()
        self._trades: list[ImplementationShortfall] = []
        self._signal_history: list[dict] = []

    def analyze_signal_strength(
        self,
        alphas: dict[str, float],
        costs_per_ticker: dict[str, float],
    ) -> dict[str, dict[str, float]]:
        """
        Analyze whether signals are strong enough to overcome costs.

        Parameters
        ----------
        alphas : dict
            Expected alpha per ticker
        costs_per_ticker : dict
            Expected round-trip cost per ticker

        Returns
        -------
        dict
            {ticker: {alpha, cost, signal_to_cost_ratio, should_trade}}
        """
        results = {}
        for ticker, alpha in alphas.items():
            cost = costs_per_ticker.get(ticker, 0.0)
            if cost <= 0:
                results[ticker] = {
                    "alpha": alpha,
                    "cost": cost,
                    "signal_to_cost_ratio": float("inf"),
                    "should_trade": abs(alpha) > 1e-8,
                }
            else:
                ratio = abs(alpha) / cost
                results[ticker] = {
                    "alpha": alpha,
                    "cost": cost,
                    "signal_to_cost_ratio": ratio,
                    "should_trade": ratio >= self.cfg.min_signal_to_cost_ratio,
                }

        return results

    def record_trade(
        self,
        ticker: str,
        decision_price: float,
        order_price: float,
        fill_price: float,
        final_price: float,
        quantity: float,
        commission: float = 0.0,
    ) -> ImplementationShortfall:
        """
        Record a trade and compute implementation shortfall.

        Parameters
        ----------
        ticker : str
            Security identifier
        decision_price : float
            Price at decision time (when signal generated)
        order_price : float
            Price when order was sent
        fill_price : float
            Actual execution price
        final_price : float
            Price at end of measurement period
        quantity : float
            Number of shares traded
        commission : float
            Explicit commission cost

        Returns
        -------
        ImplementationShortfall
        """
        notional = abs(quantity) * decision_price
        if notional < 1e-8:
            notional = 1.0

        direction = np.sign(quantity)
        delay = (order_price - decision_price) * direction * quantity
        impact = (fill_price - order_price) * direction * quantity
        opportunity = (final_price - decision_price) * direction * quantity * max(0, -np.sign(quantity))  # Missed portion
        total = delay + impact + commission + opportunity

        shortfall = ImplementationShortfall(
            ticker=ticker,
            decision_price=decision_price,
            order_price=order_price,
            fill_price=fill_price,
            final_price=final_price,
            quantity=quantity,
            commission=commission,
            delay_cost=float(delay),
            market_impact=float(impact),
            opportunity_cost=float(opportunity),
            total_shortfall=float(total),
            shortfall_bps=float(total / notional * 10000),
        )

        self._trades.append(shortfall)
        return shortfall

    def compute_adverse_selection(
        self,
        order_flow: pd.Series,
        subsequent_returns: pd.Series,
        window: int | None = None,
    ) -> dict[str, float]:
        """
        Compute adverse selection measure: correlation between order flow
        and subsequent returns.

        Positive correlation → market moves against us → adverse selection
        Negative correlation → market moves in our favor → providing liquidity

        Parameters
        ----------
        order_flow : pd.Series
            Signed order flow over time
        subsequent_returns : pd.Series
            Returns after orders are executed
        window : int, optional
            Rolling window for calculation

        Returns
        -------
        dict
            {correlation, t_stat, p_value, interpretation}
        """
        aligned = pd.DataFrame({
            "flow": order_flow,
            "returns": subsequent_returns,
        }).dropna()

        if len(aligned) < 10:
            return {"correlation": 0.0, "t_stat": 0.0, "p_value": 1.0, "interpretation": "insufficient_data"}

        corr = aligned["flow"].corr(aligned["returns"])
        if not np.isfinite(corr):
            return {"correlation": 0.0, "t_stat": 0.0, "p_value": 1.0, "interpretation": "invalid_correlation"}

        # T-statistic for correlation
        n = len(aligned)
        t_stat = corr * np.sqrt((n - 2) / (1 - corr**2 + 1e-12))
        p_value = 2.0 * (1.0 - _norm_cdf(abs(t_stat)))

        if corr > self.cfg.max_adverse_selection:
            interpretation = "adverse_selection_detected"
        elif corr < -self.cfg.max_adverse_selection:
            interpretation = "providing_liquidity"
        else:
            interpretation = "neutral"

        return {
            "correlation": float(corr),
            "t_stat": float(t_stat),
            "p_value": float(p_value),
            "interpretation": interpretation,
            "n_observations": n,
        }

    def check_pattern_predictability(
        self,
        signals: pd.DataFrame,
        order_flows: pd.DataFrame,
    ) -> dict[str, float]:
        """
        Check if order flows are predictable from past signals.

        High predictability → our trading patterns can be front-run.

        Returns R² for each ticker's order flow predicted by lagged signals.
        """
        predictability = {}
        for ticker in order_flows.columns:
            if ticker not in signals.columns:
                continue
            sig = signals[ticker].shift(1).dropna()  # Lagged signal
            flow = order_flows[ticker].loc[sig.index].dropna()
            aligned = pd.DataFrame({"sig": sig, "flow": flow}).dropna()

            if len(aligned) < 20:
                predictability[ticker] = 0.0
                continue

            # Simple linear regression: flow = β × sig + ε
            x = aligned["sig"].values
            y = aligned["flow"].values
            beta = float(np.dot(x, y) / (np.dot(x, x) + 1e-12))
            predicted = beta * x
            r_squared = float(1.0 - np.sum((y - predicted)**2) / (np.sum((y - np.mean(y))**2) + 1e-12))
            predictability[ticker] = max(0.0, r_squared)

        high_risk = {t: r2 for t, r2 in predictability.items() if r2 > self.cfg.predictability_threshold}

        return {
            "avg_r_squared": float(np.mean(list(predictability.values()))) if predictability else 0.0,
            "high_risk_tickers": high_risk,
            "ticker_r_squared": predictability,
        }

    def generate_report(
        self,
        date: pd.Timestamp,
        alphas: dict[str, float],
        costs: dict[str, float],
        order_flow: pd.Series | None = None,
        subsequent_returns: pd.Series | None = None,
    ) -> AdverseSelectionReport:
        """Generate comprehensive adverse selection report."""
        # Signal-to-cost analysis
        signal_analysis = self.analyze_signal_strength(alphas, costs)
        ratios = [v["signal_to_cost_ratio"] for v in signal_analysis.values() if np.isfinite(v["signal_to_cost_ratio"])]
        avg_ratio = float(np.mean(ratios)) if ratios else 0.0
        below_threshold = sum(1 for r in ratios if r < self.cfg.min_signal_to_cost_ratio)
        pct_below = below_threshold / len(ratios) if ratios else 0.0

        # Implementation shortfall from recorded trades
        if self._trades:
            avg_is = float(np.mean([t.shortfall_bps for t in self._trades]))
        else:
            avg_is = 0.0

        # Adverse selection
        adv_sel = 0.0
        if order_flow is not None and subsequent_returns is not None:
            adv_result = self.compute_adverse_selection(order_flow, subsequent_returns)
            adv_sel = adv_result.get("correlation", 0.0)

        # High-risk trades
        high_risk = [
            t for t, analysis in signal_analysis.items()
            if not analysis["should_trade"] and abs(analysis["alpha"]) > 1e-6
        ]

        # Recommendations
        recommendations = []
        if avg_ratio < 1.0:
            recommendations.append("Average signal-to-cost ratio < 1. Consider reducing trade frequency or improving signal quality.")
        if pct_below > 0.5:
            recommendations.append(f"{pct_below:.0%} of trades have signal-to-cost ratio below threshold. Review trade eligibility criteria.")
        if adv_sel > self.cfg.max_adverse_selection:
            recommendations.append("Adverse selection detected. Consider randomizing execution timing or using more aggressive algorithms.")
        if avg_is > 10:
            recommendations.append(f"Average implementation shortfall is {avg_is:.1f}bps. Review execution algorithms.")

        return AdverseSelectionReport(
            date=date,
            avg_signal_to_cost_ratio=avg_ratio,
            pct_trades_below_threshold=pct_below,
            avg_implementation_shortfall_bps=avg_is,
            adverse_selection_measure=adv_sel,
            pattern_predictability=0.0,  # Computed separately if needed
            high_risk_trades=high_risk,
            recommendations=recommendations,
        )


def _norm_cdf(x: float) -> float:
    """Approximation of standard normal CDF."""
    return 0.5 * (1.0 + np.tanh(x * np.sqrt(2.0 / np.pi) * (0.866 + 0.04 * x**2)))
