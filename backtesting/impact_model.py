"""
Almgren-Chriss Impact Model
============================
Three-component total-cost model per ticker per trade:

    TC_i(Δw) = σ_i · c_spread · √(ADV_med/ADV_i)          [spread proxy]
             + σ_i · η_perm  · (|Δw_i|·E / ADV_i)          [permanent impact: linear]
             + σ_i · η_temp  · √(|Δw_i|·E / ADV_i)         [temporary impact: √-law]

where:
  σ_i        = 20-day realized daily return vol
  ADV_i      = 20-day average daily dollar volume
  E          = current portfolio equity
  ADV_med    = cross-sectional median ADV (for robust normalization)

FISTA embedding — heterogeneous turnover penalty
-------------------------------------------------
The √-impact term is a concave function of |Δw|. Embedding it in FISTA's
quadratic objective (∑ γ_i (w_i - w₀_i)²) requires a local quadratic
approximation at a reference trade size Δw_ref.

First-order condition with impact:
  μ_i = 2λ(Σw)_i + 2γ_i(w_i - w₀_i) + ∂TC_temp/∂|Δw_i|

Setting 2γ_i_impact·|Δw_ref| = ∂TC_temp/∂|Δw_ref| at the reference:
  γ_i_impact = σ_i·η_temp·√(E/ADV_i) / (4·√|Δw_ref|)

In cross-sectional normalized form (dimensionless, self-calibrating):
  γ_i^eff = γ_base · (1 + η_impact · (σ_i/σ_med) · √(ADV_med/ADV_i))

Effects on sizing:
  Liquid (ADV_i >> ADV_med):  multiplier ≈ 1         → minimal size reduction
  Median liquidity:           multiplier = 1+η_impact  → ~η_impact/2 smaller position
  Illiquid (ADV_i = ADV_med/k): multiplier = 1+η√k    → position ∝ 1/multiplier

At η_impact=0.5, k=10 (10× less liquid than median):
  multiplier = 1 + 0.5×√10 ≈ 2.58
  Position shrinks to ≈ 1/2.58 ≈ 39% of unconstrained size
  TC ∝ |Δw|^1.5 → (0.39)^1.5 ≈ 24% of original → 76% impact cost reduction

References
----------
  Almgren & Chriss (2001) "Optimal Execution of Portfolio Transactions."
  Amihud (2002) "Illiquidity and stock returns."
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_ADV_WINDOW = 20
_VOL_WINDOW = 20
_VOL_FALLBACK = 0.015     # 1.5%/day ≈ S&P 500 median
_ADV_FALLBACK = 50_000_000.0  # $50M default


class ImpactModel:
    """
    Computes per-ticker (cost_rate, effective_gamma) for the FISTA optimizer.

    Parameters
    ----------
    eta_temp : float
        Temporary impact coefficient. η_temp≈0.10 calibrated from Almgren et al. (2005).
    eta_perm : float
        Permanent impact coefficient. η_perm≈0.03 (one-third of temporary).
    c_spread : float
        Spread proxy scaling. Spread ≈ c_spread·σ·√(ADV_med/ADV_i).
        c_spread≈0.05 gives ~5bp spread at median liquidity for σ=1%.
    eta_impact : float
        Scale of heterogeneous-γ amplification. Controls how strongly illiquid
        names are penalized relative to γ_base.
        η_impact=0.5: 50% extra penalty at median liquidity, 158% at ADV/10.
    adv_fallback : float
        Default ADV when price data is unavailable.
    vol_fallback : float
        Default daily vol when price data is unavailable.
    """

    def __init__(
        self,
        eta_temp: float = 0.10,
        eta_perm: float = 0.03,
        c_spread: float = 0.05,
        eta_impact: float = 0.5,
        adv_fallback: float = _ADV_FALLBACK,
        vol_fallback: float = _VOL_FALLBACK,
    ) -> None:
        self.eta_temp = float(eta_temp)
        self.eta_perm = float(eta_perm)
        self.c_spread = float(c_spread)
        self.eta_impact = float(eta_impact)
        self.adv_fallback = float(adv_fallback)
        self.vol_fallback = float(vol_fallback)

    # ------------------------------------------------------------------
    # Public: compute both outputs for the full optimizer universe
    # ------------------------------------------------------------------

    def compute_universe(
        self,
        tickers: list[str],
        equity: float,
        price_data: dict,
        date: pd.Timestamp,
        gamma_base: float,
        w_scale: float = 0.02,
    ) -> tuple[dict[str, float], dict[str, float]]:
        """
        Compute per-ticker cost rate and effective gamma for all tickers.

        Parameters
        ----------
        tickers : list[str]
            Optimizer universe for this date.
        equity : float
            Portfolio equity in dollars.
        price_data : dict[str, pd.DataFrame]
            OHLCV data by ticker.
        date : pd.Timestamp
            Current simulation date.
        gamma_base : float
            Scalar turnover penalty (from calibrator). Used as the baseline
            that per-ticker multipliers are applied to.
        w_scale : float
            Reference trade size for linearizing √-impact. Default 0.02 (2%
            position size, half the 4% typical active weight).

        Returns
        -------
        per_ticker_cost : dict[str, float]
            Linear cost rate (spread/2 + permanent impact) for forecast adjustment.
            Units: fraction of trade notional. Used as κ_i in μ̃_i = μ_i - κ_i·dir_i.
        per_ticker_gamma : dict[str, float]
            Effective turnover penalty γ_i^eff = γ_base × impact_multiplier_i.
            Replaces scalar gamma_turnover in FISTA. Encodes √-impact as quadratic.
        """
        equity = max(float(equity), 1.0)

        # ── Gather vol and ADV for all tickers ──────────────────────────
        vols: dict[str, float] = {}
        advs: dict[str, float] = {}
        for t in tickers:
            vols[t] = self._daily_vol(t, price_data, date)
            advs[t] = self._adv_usd(t, price_data, date)

        # ── Cross-sectional medians (robust normalization) ───────────────
        vol_vals = [v for v in vols.values() if v > 0]
        adv_vals = [a for a in advs.values() if a > 0]
        sigma_med = float(np.median(vol_vals)) if vol_vals else self.vol_fallback
        adv_med = float(np.median(adv_vals)) if adv_vals else self.adv_fallback
        sigma_med = max(sigma_med, 1e-6)
        adv_med = max(adv_med, 1.0)

        per_ticker_cost: dict[str, float] = {}
        per_ticker_gamma: dict[str, float] = {}

        for t in tickers:
            sigma_i = max(vols[t], 1e-6)
            adv_i = max(advs[t], 1.0)

            # ── Spread proxy: σ·c_spread·√(ADV_med/ADV_i) ──────────────
            spread_cost = sigma_i * self.c_spread * np.sqrt(adv_med / adv_i)

            # ── Permanent impact: linear in participation ────────────────
            participation_ref = w_scale * equity / adv_i
            permanent_cost = sigma_i * self.eta_perm * participation_ref

            # ── per_ticker_cost: spread/2 + permanent component ──────────
            # (temporary impact is non-linear; embedded in gamma instead)
            per_ticker_cost[t] = spread_cost * 0.5 + permanent_cost

            # ── Heterogeneous gamma: γ_base × (1 + η·(σ/σ_med)·√(ADV_med/ADV)) ─
            illiq_mult = np.sqrt(adv_med / adv_i)
            vol_ratio = sigma_i / sigma_med
            impact_multiplier = 1.0 + self.eta_impact * vol_ratio * illiq_mult
            per_ticker_gamma[t] = gamma_base * max(impact_multiplier, 1.0)

        return per_ticker_cost, per_ticker_gamma

    # ------------------------------------------------------------------
    # Private helpers (mirror backtester._compute_vol_daily / _compute_adv_usd)
    # ------------------------------------------------------------------

    def _daily_vol(self, ticker: str, price_data: dict, date: pd.Timestamp) -> float:
        """20-day realized daily return vol. Falls back to self.vol_fallback."""
        try:
            df = price_data.get(ticker)
            if df is None or df.empty or "Close" not in df.columns:
                return self.vol_fallback
            hist = df["Close"].loc[df.index <= date].tail(_VOL_WINDOW + 1)
            if len(hist) < 3:
                return self.vol_fallback
            vol = float(hist.pct_change().dropna().std())
            return vol if vol > 0 else self.vol_fallback
        except Exception:
            return self.vol_fallback

    def _adv_usd(self, ticker: str, price_data: dict, date: pd.Timestamp) -> float:
        """20-day average daily dollar volume. Falls back to self.adv_fallback."""
        try:
            df = price_data.get(ticker)
            if df is None or df.empty or "Volume" not in df.columns or "Close" not in df.columns:
                return self.adv_fallback
            hist = df.loc[df.index <= date]
            if len(hist) < 2:
                return self.adv_fallback
            dvol = (hist["Volume"] * hist["Close"]).tail(_ADV_WINDOW)
            adv = float(dvol.mean())
            return adv if adv > 0 else self.adv_fallback
        except Exception:
            return self.adv_fallback
