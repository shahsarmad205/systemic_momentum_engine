"""
Trade Scheduler (Decay-Aware + Liquidity-Adaptive Execution)
=============================================================
Per-ticker execution fraction derived jointly from signal decay and liquidity:

    f_i(t) = min(f_signal, f_liq_i) × max(0.05, urgency_i)

    f_signal  = 1 − exp(−1/τ_exec)            [signal-decay rate, global]
    f_liq_i   = P_max × ADV_i / (|remaining_i| × E)   [liquidity rate, per-ticker]

The binding constraint switches automatically:
  Liquid names (P_i < P_max):   f_i = f_signal  → signal-paced execution
  Illiquid names (P_i > P_max): f_i = f_liq_i   → participation-rate-limited

Implied execution horizon per ticker:
    T_i^signal = 1 / f_signal  ≈ τ_exec days
    T_i^liq    = |remaining_i| × E / (P_max × ADV_i)
    T_i^eff    = max(T_i^signal, T_i^liq)

Alpha-capture under liquidity constraint:
    capture(T, τ) = (τ/T) × (1 − exp(−T/τ)) / (1 − exp(−1/τ))

  When T_eff ≈ τ (liquid or small position):   capture ≈ 1
  When T_eff >> τ (illiquid, large position):   capture ≈ τ/T_eff

Impact reduction from participation capping:
  TC ∝ σ × η × √participation
  Capping participation at P_max < P_unconstrained:
    TC_ratio = √(P_max / P_unconstrained)
  At P_unconstrained=10%, P_max=2%: TC_ratio=√(0.2)=45% → 55% cost reduction.

Execution modes:
  Uniform (tau_exec=None):    f_signal = 1/horizon  (legacy fallback)
  Exponential (tau_exec set): f_signal = 1−exp(−1/τ)  (decay-aligned)

Per-ticker urgency scaling:
    urgency_i = min(2, |α_i| / α_p75)
  High conviction: execute at up to 2× rate (within liquidity limit).
  Low conviction:  execute at sub-base rate (floor 0.05 so positions clear).

Short-close path:
  Uses τ_short_close = 0.4 × τ_exec by default (2.5× faster than entry)
  because reversal IC decays more rapidly than trend IC.

References
----------
  Almgren & Chriss (2001) "Optimal Execution of Portfolio Transactions".
  Grinold & Kahn (2000) Active Portfolio Management, Ch. 14–15.
"""

from __future__ import annotations

import math


class TradeScheduler:
    """
    Spreads portfolio weight changes at rates derived from signal decay and liquidity.

    Parameters
    ----------
    horizon : int
        Fallback execution horizon (days) when tau_exec is None.
    min_trade : float
        Weight changes below this threshold are skipped (avoids micro-trades).
    short_close_horizon : int | None
        Legacy fallback horizon for closing short positions.
    tau_exec : float | None
        Alpha decay tau (trading days). Drives f_signal = 1−exp(−1/τ).
        When None: falls back to uniform 1/horizon.
    short_close_tau : float | None
        Execution tau for short-close trades. Default: 0.4 × tau_exec.
    max_participation : float
        Maximum fraction of ADV to trade in a single day per ticker.
        Acts as an upper cap on participation rate: trades are slowed when
        |remaining| × equity / ADV_i > max_participation.
        Default 0.02 (2% of ADV).  Set 1.0 to disable.
    persistence_multiplier : float
        Dead-band threshold for the alpha-vs-cost persistence gate.
        Trade is initiated only when:
            |α_i| × τ_exec  >  persistence_multiplier × 2 × κ_i
        i.e., the expected alpha over the decay horizon must exceed the
        round-trip cost scaled by this multiplier.
        persistence_multiplier=1.0 (default): gate opens when α×τ > 2κ.
        persistence_multiplier=0.5: more permissive (α×τ > κ).
        persistence_multiplier=2.0: conservative (α×τ > 4κ).
        Set 0.0 to disable (no persistence filtering).
        Requires alpha_map and cost_map to be passed to step().
    """

    def __init__(
        self,
        horizon: int = 3,
        min_trade: float = 0.0005,
        short_close_horizon: int | None = None,
        tau_exec: float | None = None,
        short_close_tau: float | None = None,
        max_participation: float = 0.02,
        persistence_multiplier: float = 1.0,
    ) -> None:
        self.horizon = max(1, int(horizon))
        self.min_trade = float(min_trade)
        self.max_participation = max(1e-4, float(max_participation))
        self.persistence_multiplier = max(0.0, float(persistence_multiplier))

        # ── Execution fractions ──────────────────────────────────────────
        if tau_exec is not None:
            self.tau_exec: float | None = max(0.5, float(tau_exec))
            self._frac_base = 1.0 - math.exp(-1.0 / self.tau_exec)
        else:
            self.tau_exec = None
            self._frac_base = 1.0 / self.horizon  # uniform fallback

        if short_close_tau is not None:
            _sc_tau = max(0.3, float(short_close_tau))
            self._frac_short_close = 1.0 - math.exp(-1.0 / _sc_tau)
        elif short_close_horizon is not None:
            self._frac_short_close = 1.0 / max(1, int(short_close_horizon))
        elif tau_exec is not None:
            # Default: short-close at 2.5× the long-entry rate (0.4 × tau → faster)
            _sc_tau = max(0.3, float(tau_exec) * 0.4)
            self._frac_short_close = 1.0 - math.exp(-1.0 / _sc_tau)
        else:
            self._frac_short_close = self._frac_base

        self.short_close_horizon: int = (
            short_close_horizon if short_close_horizon is not None else self.horizon
        )

        # Remaining unexecuted weight delta per ticker
        self._pending: dict[str, float] = {}

    # ------------------------------------------------------------------

    def step(
        self,
        w_target: dict[str, float],
        w_current: dict[str, float],
        urgency_map: dict[str, float] | None = None,
        adv_map: dict[str, float] | None = None,
        equity: float = 0.0,
        alpha_map: dict[str, float] | None = None,
        cost_map: dict[str, float] | None = None,
    ) -> tuple[dict[str, float], int]:
        """
        Execute one day's slice of the path from w_current → w_target.

        Per-ticker execution fraction:
            f_i = min(f_signal × urgency_i, f_liq_i)

        Persistence gate (dead-band):
            Trade only if  |α_i| × τ_exec  >  persistence_multiplier × 2 × κ_i
            i.e., expected alpha over the decay horizon > round-trip cost.
            Blocks low-conviction incremental adjustments that consume alpha.

        Parameters
        ----------
        urgency_map : dict[ticker, float] | None
            Per-ticker urgency ∈ [0.05, 2].  urgency_i = min(2, |α_i|/α_p75).
        adv_map : dict[ticker, float] | None
            Per-ticker average daily dollar volume.  When provided with equity>0,
            enables the liquidity-rate hard cap.
        equity : float
            Portfolio equity in dollars.
        alpha_map : dict[ticker, float] | None
            Per-ticker forecast |α_i|.  Required for persistence gate.
        cost_map : dict[ticker, float] | None
            Per-ticker one-way cost rate κ_i.  Required for persistence gate.

        Returns
        -------
        actual_delta : dict[ticker, float]
            Weight change to apply to w_current today.
        n_gated : int
            Number of tickers blocked by the persistence gate today.
        """
        all_tickers = set(w_target) | set(w_current) | set(self._pending)

        # ── Persistence gate parameters ──────────────────────────────────
        _gate_enabled = (
            alpha_map is not None
            and cost_map is not None
            and self.tau_exec is not None
            and self.persistence_multiplier > 0.0
        )
        _tau = max(float(self.tau_exec or self.horizon), 1.0)
        _threshold_factor = self.persistence_multiplier * 2.0  # multiplier × round-trip

        # Recompute desired delta from current → target
        desired_delta: dict[str, float] = {}
        n_gated = 0
        for t in all_tickers:
            wt = float(w_target.get(t, 0.0))
            wc = float(w_current.get(t, 0.0))
            d = wt - wc
            if abs(d) <= self.min_trade:
                continue

            # ── Persistence gate ─────────────────────────────────────────
            # Block trade if |α_i| × τ < threshold × κ_i.
            # Applies uniformly to entries, exits, and rebalances.
            # Small signal flips are blocked; large reversals pass through.
            if _gate_enabled:
                alpha_i = abs(float(alpha_map.get(t, 0.0)))  # type: ignore[union-attr]
                kappa_i = float(cost_map.get(t, 0.0))        # type: ignore[union-attr]
                if kappa_i > 0.0 and alpha_i * _tau < _threshold_factor * kappa_i:
                    n_gated += 1
                    continue  # Expected alpha < round-trip cost: hold position

            desired_delta[t] = d

        # New optimizer solution supersedes old pending
        self._pending = desired_delta.copy()

        _use_liq = adv_map is not None and equity > 0.0
        actual_delta: dict[str, float] = {}

        for t, remaining in list(self._pending.items()):
            wc = float(w_current.get(t, 0.0))
            is_closing_short = wc < -self.min_trade and remaining > 0

            frac = self._frac_short_close if is_closing_short else self._frac_base

            # ── Urgency scaling (signal strength relative to cross-section) ─
            # Applied to f_signal BEFORE the liquidity cap so that urgency
            # can accelerate execution on liquid names but cannot override
            # market capacity on illiquid ones.
            if urgency_map is not None:
                u = float(urgency_map.get(t, 1.0))
                frac = min(1.0, frac * max(0.05, u))

            # ── Liquidity-rate hard cap ──────────────────────────────────
            # Enforced AFTER urgency: participation cannot exceed P_max
            # regardless of signal strength.  This is the market-capacity
            # ceiling — no amount of urgency can absorb more than ADV allows.
            if _use_liq:
                adv_i = float(adv_map.get(t, 0.0))  # type: ignore[union-attr]
                if adv_i > 0.0:
                    participation = abs(remaining) * equity / adv_i
                    if participation > self.max_participation:
                        f_liq = self.max_participation / participation
                        frac = min(frac, f_liq)

            step = remaining * frac
            actual_delta[t] = step
            new_remaining = remaining - step
            if abs(new_remaining) < self.min_trade:
                del self._pending[t]
            else:
                self._pending[t] = new_remaining

        return actual_delta, n_gated

    def reset(self) -> None:
        """Clear all pending executions (e.g. on regime transition)."""
        self._pending.clear()

    @property
    def pending_gross_delta(self) -> float:
        """Total unexecuted weight change magnitude (for monitoring)."""
        return float(sum(abs(v) for v in self._pending.values()))
