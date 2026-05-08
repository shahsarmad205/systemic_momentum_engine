"""
Stress Testing and Scenario Analysis
======================================
Institutional portfolio stress testing integrated into the risk management loop.

Implements:
  1. Historical VaR and Expected Shortfall (CVaR)
  2. Stress scenarios (2008-style, liquidity dry-up, vol spike)
  3. Correlation shock analysis ("what if 2008 correlations return?")
  4. Liquidity stress testing (ADV depletion)

These tests run as part of the portfolio construction pipeline, not as
a standalone utility. The optimizer receives stress-aware constraints.

References:
  - Basel III stress testing requirements
  - AQR "Stress Testing for Factor Portfolios" (2016)
  - Fama-French correlation regime analysis
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StressScenario:
    """Definition of a stress test scenario."""
    name: str
    description: str
    vol_shock: float = 0.0        # Multiplicative vol shock (1.5 = 50% vol increase)
    correlation_shock: float = 0.0  # Additive correlation increase
    market_shock: float = 0.0     # Market return shock
    adv_shrink: float = 0.0       # ADV reduction fraction (0.3 = 30% less liquidity)
    sector_shocks: dict[str, float] = field(default_factory=dict)
    factor_shocks: dict[str, float] = field(default_factory=dict)


@dataclass
class StressTestResult:
    """Results from a stress test."""
    scenario_name: str
    portfolio_return: float
    portfolio_vol: float
    portfolio_var: float
    portfolio_cvar: float  # Expected Shortfall
    max_drawdown: float
    liquidity_impact: float  # Estimated cost increase from ADV shrink
    factor_impact: dict[str, float] = field(default_factory=dict)
    constraint_violations: list[str] = field(default_factory=list)
    passed: bool = True


class StressTestEngine:
    """
    Portfolio stress testing engine.

    Applies scenario shocks to the covariance matrix, expected returns,
    and liquidity constraints to estimate portfolio behavior under stress.
    """

    def __init__(
        self,
        confidence_level: float = 0.95,
        n_simulations: int = 10000,
    ):
        self.confidence_level = float(confidence_level)
        self.n_simulations = int(n_simulations)

        # Pre-built scenarios
        self.scenarios: list[StressScenario] = [
            StressScenario(
                name="2008_gfc",
                description="Global Financial Crisis: vol spike + correlation flight to quality",
                vol_shock=2.5,
                correlation_shock=0.3,
                market_shock=-0.40,
                adv_shrink=0.5,
            ),
            StressScenario(
                name="2020_covid",
                description="COVID crash: extreme vol + liquidity freeze",
                vol_shock=3.0,
                correlation_shock=0.4,
                market_shock=-0.35,
                adv_shrink=0.7,
            ),
            StressScenario(
                name="vol_spike",
                description="VIX doubles: vol regime shift without crash",
                vol_shock=2.0,
                correlation_shock=0.15,
                market_shock=-0.05,
                adv_shrink=0.2,
            ),
            StressScenario(
                name="liquidity_dry_up",
                description="Liquidity evaporation: ADV halved, spreads widen 3x",
                vol_shock=1.2,
                correlation_shock=0.1,
                market_shock=-0.05,
                adv_shrink=0.5,
            ),
            StressScenario(
                name="mild_recession",
                description="Moderate downturn: -15% market, vol up 50%",
                vol_shock=1.5,
                correlation_shock=0.2,
                market_shock=-0.15,
                adv_shrink=0.3,
            ),
        ]

    def run_stress_tests(
        self,
        weights: np.ndarray,
        cov: np.ndarray,
        expected_returns: np.ndarray | None = None,
        adv_by_ticker: dict[str, float] | None = None,
        factor_exposures: dict[str, np.ndarray] | None = None,
        tickers: list[str] | None = None,
    ) -> list[StressTestResult]:
        """
        Run all stress scenarios against the current portfolio.

        Parameters
        ----------
        weights : np.ndarray
            Portfolio weights (N,)
        cov : np.ndarray
            Covariance matrix (N, N)
        expected_returns : np.ndarray, optional
            Expected returns (N,)
        adv_by_ticker : dict, optional
            ADV by ticker for liquidity stress
        factor_exposures : dict, optional
            Factor exposures for factor stress
        tickers : list, optional
            Ticker list for reporting

        Returns
        -------
        list[StressTestResult]
            Results for each scenario
        """
        w = np.asarray(weights, dtype=float)
        cov_matrix = np.asarray(cov, dtype=float)

        results = []
        for scenario in self.scenarios:
            result = self._run_single_scenario(
                w, cov_matrix, expected_returns, scenario,
                adv_by_ticker, factor_exposures, tickers,
            )
            results.append(result)

        return results

    def _run_single_scenario(
        self,
        weights: np.ndarray,
        cov: np.ndarray,
        expected_returns: np.ndarray | None,
        scenario: StressScenario,
        adv_by_ticker: dict[str, float] | None,
        factor_exposures: dict[str, np.ndarray] | None,
        tickers: list[str] | None,
    ) -> StressTestResult:
        """Run a single stress scenario."""
        N = len(weights)

        # Apply vol shock
        stressed_cov = cov * (scenario.vol_shock ** 2)

        # Apply correlation shock: increase off-diagonal elements
        if scenario.correlation_shock > 0:
            D = np.sqrt(np.diag(stressed_cov))
            corr = stressed_cov / np.outer(D, D)
            corr = np.clip(corr, -1.0, 1.0)
            # Increase correlations toward 1
            stressed_corr = corr + scenario.correlation_shock * (1.0 - corr)
            np.fill_diagonal(stressed_corr, 1.0)
            stressed_cov = stressed_corr * np.outer(D, D)

        # Apply market shock to expected returns
        if expected_returns is not None:
            stressed_mu = expected_returns + scenario.market_shock / 252.0
        else:
            stressed_mu = np.zeros(N) + scenario.market_shock / 252.0

        # Portfolio-level metrics
        port_return = float(np.dot(weights, stressed_mu)) * 252  # Annualized
        port_var = float(np.dot(weights, np.dot(stressed_cov, weights)))
        port_vol = float(np.sqrt(port_var))

        # Monte Carlo VaR/ES
        sims = self._simulate_returns(weights, stressed_mu, stressed_cov)
        var = float(np.percentile(sims, (1 - self.confidence_level) * 100))
        cvar = float(np.mean(sims[sims <= var]))

        # Max drawdown from simulation
        max_dd = float(self._estimate_max_drawdown(sims))

        # Liquidity impact
        liq_impact = 0.0
        if adv_by_ticker and tickers and scenario.adv_shrink > 0:
            liq_impact = self._estimate_liquidity_impact(
                weights, adv_by_ticker, tickers, scenario.adv_shrink
            )

        # Factor impact
        factor_impact: dict[str, float] = {}
        if factor_exposures and scenario.factor_shocks:
            for fname, shock in scenario.factor_shocks.items():
                if fname in factor_exposures:
                    exp = factor_exposures[fname]
                    impact = float(np.dot(weights, exp)) * shock
                    factor_impact[fname] = impact

        # Check constraint violations
        violations = []
        if port_vol > 0.40:
            violations.append(f"vol_exceeded: {port_vol:.1%} > 40%")
        if max_dd > 0.25:
            violations.append(f"dd_exceeded: {max_dd:.1%} > 25%")
        if port_return < -0.30:
            violations.append(f"return_exceeded: {port_return:.1%} < -30%")

        passed = len(violations) == 0

        return StressTestResult(
            scenario_name=scenario.name,
            portfolio_return=port_return,
            portfolio_vol=port_vol,
            portfolio_var=var * np.sqrt(252),
            portfolio_cvar=cvar * np.sqrt(252),
            max_drawdown=max_dd,
            liquidity_impact=liq_impact,
            factor_impact=factor_impact,
            constraint_violations=violations,
            passed=passed,
        )

    def _simulate_returns(
        self,
        weights: np.ndarray,
        mu: np.ndarray,
        cov: np.ndarray,
    ) -> np.ndarray:
        """Monte Carlo simulation of portfolio returns."""
        N = len(weights)
        try:
            L = np.linalg.cholesky(cov)
        except np.linalg.LinAlgError:
            # Add small regularization if not PSD
            cov_reg = cov + np.eye(N) * 1e-8
            L = np.linalg.cholesky(cov_reg)

        Z = np.random.randn(self.n_simulations, N)
        correlated = Z @ L.T
        asset_returns = mu[None, :] + correlated
        port_returns = asset_returns @ weights
        return port_returns

    def _estimate_max_drawdown(self, returns: np.ndarray) -> float:
        """Estimate max drawdown from simulated returns."""
        cumulative = np.cumprod(1.0 + returns)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (cumulative - running_max) / running_max
        return float(np.abs(np.min(drawdowns)))

    def _estimate_liquidity_impact(
        self,
        weights: np.ndarray,
        adv_by_ticker: dict[str, float],
        tickers: list[str],
        adv_shrink: float,
    ) -> float:
        """Estimate cost increase from ADV shrinkage."""
        impact = 0.0
        for i, ticker in enumerate(tickers):
            if ticker in adv_by_ticker:
                adv = adv_by_ticker[ticker] * (1.0 - adv_shrink)
                weight = abs(weights[i])
                # Simplified impact model: cost increases inversely with ADV
                if adv > 0:
                    cost = (weight * 1e8) / adv * 0.001  # 10bps per 100% ADV
                    impact += cost
        return impact

    def summarize(self, results: list[StressTestResult]) -> dict[str, Any]:
        """Create summary report of stress test results."""
        return {
            "n_scenarios": len(results),
            "passed": sum(1 for r in results if r.passed),
            "failed": sum(1 for r in results if not r.passed),
            "worst_return": min(r.portfolio_return for r in results),
            "worst_vol": max(r.portfolio_vol for r in results),
            "worst_dd": max(r.max_drawdown for r in results),
            "worst_cvar": min(r.portfolio_cvar for r in results),
            "scenarios": [
                {
                    "name": r.scenario_name,
                    "return": r.portfolio_return,
                    "vol": r.portfolio_vol,
                    "var": r.portfolio_var,
                    "cvar": r.portfolio_cvar,
                    "max_dd": r.max_drawdown,
                    "passed": r.passed,
                    "violations": r.constraint_violations,
                }
                for r in results
            ],
        }
