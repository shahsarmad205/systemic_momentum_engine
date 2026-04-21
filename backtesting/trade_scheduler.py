"""
Trade Scheduler (Gradual Execution)
=====================================
Implements Almgren-Chriss-style gradual execution: spreads the weight delta
from optimizer target to actual portfolio over T trading days.

    trade_step_t = (w_target - w_current) / T

This reduces:
  - Realized slippage vs a single-day full rebalance
  - Alpha destruction from market impact on large orders
  - Turnover spikes from optimizer pivots

Reference:
  Almgren & Chriss (2001) "Optimal Execution of Portfolio Transactions".
  Grinold & Kahn (2000) Active Portfolio Management, Ch. 14–15.
"""

from __future__ import annotations

import numpy as np


class TradeScheduler:
    """
    Spreads portfolio weight changes over `horizon` trading days.

    The scheduler maintains a `pending_delta` dict: the remaining weight
    change that has not yet been executed. Each call to `step()` executes
    1/horizon of the remaining delta and returns the actual weight change.

    Usage in the simulation loop:
        scheduler = TradeScheduler(horizon=3)
        for date in trading_days:
            w_target = optimizer.optimize(...)
            actual_delta = scheduler.step(w_target, w_current)
            w_current = {t: w_current.get(t,0) + actual_delta.get(t,0) for t in all_t}

    Parameters
    ----------
    horizon : int
        Execution horizon in trading days. 1 = immediate (instant fill).
        2–5 = gradual execution (recommended for reducing market impact).
    min_trade : float
        Weight changes below this threshold are ignored (avoids sub-cent trades).
    """

    def __init__(self, horizon: int = 3, min_trade: float = 0.0005, short_close_horizon: int | None = None):
        self.horizon = max(1, int(horizon))
        # Faster close horizon for short positions: reversal IC decays past day 2-3,
        # so holding shorts beyond that flips the expected return negative.
        self.short_close_horizon = max(1, int(short_close_horizon)) if short_close_horizon is not None else self.horizon
        self.min_trade = float(min_trade)
        # Remaining unexecuted weight delta per ticker
        self._pending: dict[str, float] = {}

    def step(
        self,
        w_target: dict[str, float],
        w_current: dict[str, float],
    ) -> dict[str, float]:
        """
        Execute one day's slice of the path from w_current → w_target.

        Logic:
          1. Compute the NEW desired delta: w_target − w_current.
          2. Overwrite pending with this new delta (optimizer is re-run each day,
             so we always move toward the most recent optimal weights).
          3. Execute 1/horizon of the pending delta.

        Returns
        -------
        actual_delta : dict[ticker, float]
            Weight change to apply to w_current today.
        """
        all_tickers = set(w_target) | set(w_current) | set(self._pending)

        # Recompute desired delta from current → target
        desired_delta: dict[str, float] = {}
        for t in all_tickers:
            wt = float(w_target.get(t, 0.0))
            wc = float(w_current.get(t, 0.0))
            d = wt - wc
            if abs(d) > self.min_trade:
                desired_delta[t] = d

        # Update pending: new optimizer solution supersedes old pending
        self._pending = desired_delta.copy()

        # Execute 1/horizon of pending.
        # For short positions being closed/reduced, use short_close_horizon so
        # we exit before the reversal signal decays and momentum takes over.
        actual_delta: dict[str, float] = {}
        for t, remaining in list(self._pending.items()):
            wc = float(w_current.get(t, 0.0))
            # Closing/reducing a short: current weight is negative, delta is positive
            is_closing_short = wc < -self.min_trade and remaining > 0
            h = self.short_close_horizon if is_closing_short else self.horizon
            step = remaining / h
            actual_delta[t] = step
            new_remaining = remaining - step
            if abs(new_remaining) < self.min_trade:
                del self._pending[t]
            else:
                self._pending[t] = new_remaining

        return actual_delta

    def reset(self) -> None:
        """Clear all pending executions (e.g. on regime transition)."""
        self._pending.clear()

    @property
    def pending_gross_delta(self) -> float:
        """Total unexecuted weight change magnitude (for monitoring)."""
        return float(sum(abs(v) for v in self._pending.values()))
