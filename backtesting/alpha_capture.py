"""
Alpha Capture Tracker
======================
Measures how much of the theoretical signal alpha the portfolio actually realizes,
and decomposes the gap into cost, lag (execution delay), and timing (churn) losses.

Definitions
-----------
theoretical_alpha_rate:
    Σ_i |w_target_i × α_i|  — alpha we'd earn with perfect instantaneous execution.
    Uses absolute values because target weights are already signed correctly.

realized_alpha_rate:
    Σ_i w_current_i × α_i  — alpha from actual (partially-executed) holdings.
    Positive when long × positive-α or short × negative-α.

alpha_capture:
    Σ realized_alpha_rate / Σ theoretical_alpha_rate   (cumulative)

Loss decomposition (each component as fraction of theoretical):
    1. cost_fraction:   Σ cost_today / Σ theoretical_alpha    [direct TC paid]
    2. lag_fraction:    Σ (theoretical − realized) / Σ theoretical
                        [alpha sitting in pending trades, not yet captured]
    3. decay_fraction:  lag_fraction × (1 − exp(−E_lag / τ))
                        [portion of lag alpha that will decay before execution]
    4. timing_fraction: residual = 1 − capture − cost − decay
                        [unnecessary reversals / churn that consumed alpha]

Interpretation:
    alpha_capture ≈ 1 − cost_fraction − decay_fraction − timing_fraction

A well-tuned system should achieve:
    alpha_capture > 0.6 (60%+ of theoretical)
    timing_fraction < 0.05 (< 5% wasted on churn)
"""

from __future__ import annotations

import math
import logging
from collections import deque

import numpy as np

logger = logging.getLogger(__name__)

_EPS = 1e-12


class AlphaCaptureTracker:
    """
    Records per-day alpha metrics and computes the four-way decomposition.

    Parameters
    ----------
    tau_days : float
        Signal decay time constant (trading days).  Used to estimate
        decay_fraction from the lag.
    ema_span : int
        EMA span for smoothed daily alpha_capture display.
    """

    def __init__(self, tau_days: float = 6.0, ema_span: int = 20) -> None:
        self.tau_days = max(0.5, float(tau_days))
        self._ema_alpha = 2.0 / (max(1, int(ema_span)) + 1.0)

        self._theoretical: list[float] = []
        self._realized: list[float] = []
        self._cost: list[float] = []
        self._dates: list = []
        self._n_trades: list[int] = []
        self._n_gated: list[int] = []      # trades blocked by persistence gate

        self._capture_ema: float | None = None

    # ------------------------------------------------------------------
    # Per-day recording
    # ------------------------------------------------------------------

    def record_day(
        self,
        date,
        forecasts: dict[str, float],
        w_current: dict[str, float],
        w_target: dict[str, float],
        cost_today: float,
        n_trades: int = 0,
        n_gated: int = 0,
    ) -> None:
        """
        Record one simulation day's alpha metrics.

        Parameters
        ----------
        forecasts : dict[str, float]
            Today's per-ticker forecasts (α_i).
        w_current : dict[str, float]
            Actual holdings AFTER today's execution.
        w_target : dict[str, float]
            Optimizer target weights.
        cost_today : float
            Total transaction cost paid today (as fraction of equity).
        n_trades : int
            Number of tickers traded today.
        n_gated : int
            Number of trades blocked by the persistence gate.
        """
        theoretical = 0.0
        realized = 0.0

        for t, alpha in forecasts.items():
            wt = float(w_target.get(t, 0.0))
            wc = float(w_current.get(t, 0.0))
            if not (math.isfinite(alpha) and math.isfinite(wt) and math.isfinite(wc)):
                continue
            # Theoretical: magnitude of alpha × target weight (signed correctly)
            theoretical += abs(wt * alpha)
            # Realized: how much of that alpha we actually hold
            realized += wc * alpha

        self._theoretical.append(theoretical)
        self._realized.append(max(realized, 0.0))  # floor at 0: losses are noise, not -α
        self._cost.append(max(float(cost_today), 0.0))
        self._dates.append(date)
        self._n_trades.append(int(n_trades))
        self._n_gated.append(int(n_gated))

        # Update EMA for live display
        cap_today = realized / max(theoretical, _EPS)
        if self._capture_ema is None:
            self._capture_ema = cap_today
        else:
            self._capture_ema = (
                self._ema_alpha * cap_today + (1.0 - self._ema_alpha) * self._capture_ema
            )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> dict:
        """
        Compute cumulative alpha capture and four-way loss decomposition.

        Returns
        -------
        dict with keys:
            alpha_capture       : float  cumulative realized / theoretical
            cost_fraction       : float  costs as fraction of theoretical
            lag_fraction        : float  unharvested alpha in pending trades
            decay_fraction      : float  estimated decay cost of lag
            timing_fraction     : float  unexplained residual (churn)
            theoretical_total   : float  Σ theoretical_alpha_rate
            realized_total      : float  Σ realized_alpha_rate
            cost_total          : float  Σ cost_today
            n_days              : int
            gated_pct           : float  % of trade signals blocked by persistence gate
            capture_ema         : float  exponentially-smoothed recent capture
        """
        n = len(self._theoretical)
        if n == 0:
            return {"alpha_capture": float("nan"), "n_days": 0}

        total_theoretical = sum(self._theoretical)
        total_realized = sum(self._realized)
        total_cost = sum(self._cost)
        total_trades = sum(self._n_trades)
        total_gated = sum(self._n_gated)

        if total_theoretical < _EPS:
            return {"alpha_capture": float("nan"), "n_days": n}

        alpha_capture = total_realized / total_theoretical
        cost_fraction = total_cost / total_theoretical

        # Lag fraction: theoretical − realized = alpha in pending, not yet held
        lag_fraction = max(0.0, 1.0 - alpha_capture)

        # Decay estimate: of lag_fraction alpha, proportion that decays before execution.
        # Assume mean execution lag ~ τ_exec days (worst case: signal half-life).
        # decay_fraction ≈ lag_fraction × (1 − exp(−τ/τ)) = lag × (1 − 1/e) ≈ lag × 0.63
        # More conservative: use (1 − exp(−1)) ≈ 0.63 as the decay fraction of lag alpha.
        decay_fraction = lag_fraction * (1.0 - math.exp(-1.0))

        # Cost-adjusted alpha capture
        net_alpha_capture = (total_realized - total_cost) / total_theoretical

        # Timing loss: churn that doesn't improve alpha
        # = 1 − capture − cost_fraction − decay_fraction
        timing_fraction = max(0.0, 1.0 - alpha_capture - cost_fraction - decay_fraction)

        gated_pct = 100.0 * total_gated / max(total_trades + total_gated, 1)

        return {
            "alpha_capture": round(alpha_capture, 4),
            "net_alpha_capture": round(net_alpha_capture, 4),
            "cost_fraction": round(cost_fraction, 4),
            "lag_fraction": round(lag_fraction, 4),
            "decay_fraction": round(decay_fraction, 4),
            "timing_fraction": round(timing_fraction, 4),
            "theoretical_total": round(total_theoretical, 6),
            "realized_total": round(total_realized, 6),
            "cost_total": round(total_cost, 6),
            "n_days": n,
            "gated_pct": round(gated_pct, 1),
            "capture_ema": round(float(self._capture_ema or 0.0), 4),
        }

    def log_summary(self, prefix: str = "") -> None:
        """Log the alpha capture summary at INFO level."""
        s = self.summary()
        if math.isnan(s.get("alpha_capture", float("nan"))):
            logger.info("%sAlphaCaptureTracker: insufficient data", prefix)
            return
        logger.info(
            "%sAlpha capture: %.3f (net: %.3f) | "
            "cost: %.3f  lag: %.3f  decay: %.3f  timing: %.3f | "
            "gated: %.1f%% of signals",
            prefix,
            s["alpha_capture"], s["net_alpha_capture"],
            s["cost_fraction"], s["lag_fraction"],
            s["decay_fraction"], s["timing_fraction"],
            s["gated_pct"],
        )
