"""
CostAwareLambdaCalibrator
==========================
Dynamically calibrates the FISTA turnover penalty γ from observed execution data.

Theory
------
The portfolio optimizer solves:
    maximize  w'α_net − λ w'Σ_idio w − γ ||w − w₀||²

The term γ||Δw||² penalizes trading.  When γ is a fixed constant it cannot
adapt to the actual cost environment, which causes one of two failure modes:

  γ too low → optimizer trades freely → turnover × costs exceed gross alpha
  γ too high → optimizer barely trades → alpha sits unrealised in signal space

Calibration rule
----------------
Let R_t = (daily_cost) / (|daily_gross_pnl| + ε)

  R_t > 1 : costs consumed more than gross alpha → γ should rise (trade less)
  R_t < 1 : gross alpha exceeded costs → γ can fall (trade more aggressively)

EMA smoothing over 20 days (monthly) avoids reacting to day-to-day noise.

    γ_t = γ_base × EMA(R_t, span=20)
    γ_t ∈ [γ_base / 8,  γ_base × 10]

Integration
-----------
Called from _simulate_continuous in backtester.py:

    calibrator = CostAwareLambdaCalibrator(gamma_base=optimizer.gamma_turnover)
    # ... at end of each day:
    calibrator.update(day_cost, day_turnover, abs(daily_pnl))
    optimizer.gamma_turnover = calibrator.gamma
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class CostAwareLambdaCalibrator:
    """
    Maintain a rolling cost/alpha ratio and calibrate γ each trading day.

    Parameters
    ----------
    gamma_base : float
        Starting γ (from optimizer config). Acts as the neutral-point.
    ema_span : int
        EMA span for cost/alpha ratio smoothing. 20 = monthly half-life.
    gamma_floor_factor : float
        γ lower bound = gamma_base × factor. Prevents reckless trading.
    gamma_ceil_factor : float
        γ upper bound = gamma_base × factor. Prevents complete freeze.
    warmup_days : int
        Days before calibration activates. Uses gamma_base during warmup.
    """

    def __init__(
        self,
        gamma_base: float,
        *,
        ema_span: int = 20,
        gamma_floor_factor: float = 0.125,
        gamma_ceil_factor: float = 10.0,
        warmup_days: int = 20,
    ) -> None:
        self._gamma_base = float(gamma_base)
        self._ema_alpha = 2.0 / (float(ema_span) + 1.0)
        self._gamma_floor = self._gamma_base * float(gamma_floor_factor)
        self._gamma_ceil = self._gamma_base * float(gamma_ceil_factor)
        self._warmup_days = int(warmup_days)
        self._days_seen: int = 0
        self._ratio_ema: float | None = None
        self.gamma: float = self._gamma_base

    # ------------------------------------------------------------------

    def update(
        self,
        cost_today: float,
        turnover_today: float,
        abs_gross_pnl_today: float,
    ) -> float:
        """
        Update γ from today's observed cost and gross return.

        Parameters
        ----------
        cost_today : float
            Absolute cost paid today (fraction of equity, positive).
        turnover_today : float
            Total |Δw| executed today.
        abs_gross_pnl_today : float
            |daily_pnl| before costs (fraction of equity).

        Returns
        -------
        float : updated γ (also stored in self.gamma)
        """
        self._days_seen += 1

        if turnover_today < 1e-9:
            return self.gamma

        if abs_gross_pnl_today < 1e-9:
            # Zero-alpha day: treat ratio as 1 (neutral), don't spike γ
            ratio_today = 1.0
        else:
            ratio_today = float(cost_today) / float(abs_gross_pnl_today)
            # Cap ratio at ceiling factor to prevent single-day blowups
            ratio_today = min(ratio_today, self._gamma_ceil / self._gamma_base)

        if self._ratio_ema is None:
            self._ratio_ema = ratio_today
        else:
            self._ratio_ema = (
                ratio_today * self._ema_alpha
                + self._ratio_ema * (1.0 - self._ema_alpha)
            )

        if self._days_seen < self._warmup_days:
            return self.gamma

        gamma_new = self._gamma_base * float(self._ratio_ema)
        self.gamma = float(np.clip(gamma_new, self._gamma_floor, self._gamma_ceil))
        logger.debug(
            "CostAwareLambda: ratio_ema=%.3f γ=%.2f (cost=%.5f gross_pnl=%.5f)",
            self._ratio_ema, self.gamma, cost_today, abs_gross_pnl_today,
        )
        return self.gamma

    @property
    def ratio_ema(self) -> float:
        return float("nan") if self._ratio_ema is None else float(self._ratio_ema)
