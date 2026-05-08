"""
Time-Series Risk Management Overlay
=====================================
Institutional portfolio-level exposure management based on:
  1. Aggregate signal strength (cross-sectional conviction)
  2. Market trend regime (time-series momentum)
  3. Volatility regime (vol clustering and crisis detection)

Unlike purely cross-sectional systems, this overlay adjusts total
portfolio exposure dynamically, combining:
  - AQR's "time-series momentum + cross-sectional value" approach
  - Volatility-targeted scaling
  - Signal-strength-aware exposure modulation

This implements the "risk management overlay" pattern used by
systematic macro funds and multi-strat platforms.

References:
  - Moskowitz, Ooi, Pedersen (2012) "Time Series Momentum"
  - AQR "A New Core Equity Paradigm" (2014)
  - GEMs (Global Equity Model) time-series overlay patterns
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeSeriesRiskConfig:
    """Configuration for time-series risk overlay."""
    enabled: bool = True

    # Signal strength scaling
    signal_strength_window: int = 60
    signal_strength_percentile: int = 75
    signal_ema_span: int = 5
    signal_min_exposure: float = 0.1
    signal_max_exposure: float = 1.3

    # Market trend filter
    market_trend_enabled: bool = True
    market_ticker: str = "SPY"
    trend_lookback: int = 200
    trend_ema_span: int = 50
    trend_min_exposure: float = 0.0
    trend_max_exposure: float = 1.0
    trend_threshold: float = 0.0

    # Volatility regime scaling
    vol_regime_enabled: bool = True
    vol_lookback: int = 20
    vol_target: float = 0.15
    vol_min_scale: float = 0.2
    vol_max_scale: float = 1.5
    vol_garch_halflife: int = 20

    # Combined exposure bounds
    min_total_exposure: float = 0.05
    max_total_exposure: float = 1.5

    # Crisis detection (VIX or vol spike)
    crisis_detection_enabled: bool = True
    crisis_vol_threshold: float = 0.04  # Daily vol > 4% (~63% annualized)
    crisis_exposure_scale: float = 0.1
    crisis_recovery_window: int = 10


@dataclass
class RiskOverlayState:
    """State tracking for risk overlay decisions."""
    signal_strength: float = 1.0
    trend_signal: float = 1.0
    vol_scale: float = 1.0
    crisis_active: bool = False
    combined_exposure: float = 1.0
    regime_label: str = "normal"
    diagnostics: dict[str, float] = field(default_factory=dict)


class TimeSeriesRiskOverlay:
    """
    Institutional time-series risk management overlay.

    Produces a daily exposure multiplier that scales all portfolio weights:
      w_adjusted = exposure_multiplier × w_base

    The multiplier is the product of:
      1. Signal strength factor (cross-sectional conviction)
      2. Trend factor (market direction alignment)
      3. Volatility factor (risk adjustment)
      4. Crisis flag (emergency deleveraging)

    Each factor operates independently and is bounded to prevent
    extreme leverage or complete shutdown.
    """

    def __init__(self, config: TimeSeriesRiskConfig):
        self.cfg = config
        self._state = RiskOverlayState()
        self._signal_history: list[float] = []
        self._market_returns: list[float] = []
        self._vol_history: list[float] = []
        self._crisis_cooldown = 0

    def compute_exposure(
        self,
        date: pd.Timestamp,
        scores: pd.Series,
        market_price: float | None = None,
        market_vol: float | None = None,
        prev_weights: dict[str, float] | None = None,
        cov: np.ndarray | None = None,
    ) -> tuple[float, RiskOverlayState]:
        """
        Compute the time-series exposure multiplier for today.

        Parameters
        ----------
        date : pd.Timestamp
            Current date
        scores : pd.Series
            Cross-section of model scores (ticker → score)
        market_price : float, optional
            Current market index price (for trend filter)
        market_vol : float, optional
            Current market implied/realized vol (for vol scaling)
        prev_weights : dict, optional
            Previous portfolio weights (for ex-ante vol calculation)
        cov : np.ndarray, optional
            Covariance matrix (for ex-ante vol calculation)

        Returns
        -------
        tuple[float, RiskOverlayState]
            (exposure_multiplier, overlay_state_with_diagnostics)
        """
        # 1. Signal strength scaling
        signal_scale = self._compute_signal_scale(scores)

        # 2. Market trend filter
        trend_scale = self._compute_trend_scale(market_price)

        # 3. Volatility regime scaling
        vol_scale = self._compute_vol_scale(market_vol, prev_weights, cov)

        # 4. Crisis detection
        crisis_scale = self._compute_crisis_scale(market_vol)

        # Combine multiplicatively with bounds
        raw_exposure = signal_scale * trend_scale * vol_scale * crisis_scale
        exposure = float(np.clip(raw_exposure, self.cfg.min_total_exposure, self.cfg.max_total_exposure))

        # Update state
        regime = "crisis" if crisis_scale < 0.5 else ("low_vol" if vol_scale > 1.2 else ("weak_trend" if trend_scale < 0.5 else "normal"))
        self._state = RiskOverlayState(
            signal_strength=signal_scale,
            trend_signal=trend_scale,
            vol_scale=vol_scale,
            crisis_active=crisis_scale < 0.5,
            combined_exposure=exposure,
            regime_label=regime,
            diagnostics={
                "raw_exposure": raw_exposure,
                "signal_scale": signal_scale,
                "trend_scale": trend_scale,
                "vol_scale": vol_scale,
                "crisis_scale": crisis_scale,
                "n_signals": float(len(scores.dropna())),
                "signal_mean": float(scores.mean()),
                "signal_std": float(scores.std()),
            }
        )

        return exposure, self._state

    def _compute_signal_scale(self, scores: pd.Series) -> float:
        """Scale exposure based on aggregate cross-sectional signal strength."""
        valid_scores = scores.dropna()
        if len(valid_scores) < 3:
            return self.cfg.signal_min_exposure

        # Signal strength = cross-sectional std of scores (dispersion = conviction)
        strength = float(valid_scores.std())
        self._signal_history.append(strength)

        # Keep rolling window
        if len(self._signal_history) > self.cfg.signal_strength_window:
            self._signal_history = self._signal_history[-self.cfg.signal_strength_window:]

        if len(self._signal_history) < 10:
            return 1.0

        # Normalize against historical percentile
        hist = np.array(self._signal_history)
        percentile = float(np.percentile(hist, self.cfg.signal_strength_percentile))
        if percentile < 1e-12:
            return 1.0

        raw_scale = strength / percentile

        # EMA smoothing
        ema = pd.Series(self._signal_history).ewm(span=self.cfg.signal_ema_span).mean().iloc[-1]
        ema_percentile = float(np.percentile(hist, self.cfg.signal_strength_percentile))
        if ema_percentile < 1e-12:
            return 1.0

        smoothed_scale = ema / ema_percentile

        return float(np.clip(smoothed_scale, self.cfg.signal_min_exposure, self.cfg.signal_max_exposure))

    def _compute_trend_scale(self, market_price: float | None) -> float:
        """Scale exposure based on market trend alignment."""
        if not self.cfg.market_trend_enabled or market_price is None:
            return 1.0

        self._market_returns.append(float(market_price))
        if len(self._market_returns) < self.cfg.trend_lookback:
            return 1.0

        self._market_returns = self._market_returns[-self.cfg.trend_lookback:]
        prices = np.array(self._market_returns)

        # Trend = price vs moving average
        ma = float(np.mean(prices[-self.cfg.trend_ema_span:]))
        current = prices[-1]

        if ma < 1e-12:
            return 1.0

        trend_ratio = current / ma
        # Above MA → full exposure, below MA → reduced
        if trend_ratio > 1.0 + self.cfg.trend_threshold:
            return self.cfg.trend_max_exposure
        elif trend_ratio < 1.0 - self.cfg.trend_threshold:
            # Linear scaling based on how far below MA
            deviation = (1.0 - trend_ratio) / self.cfg.trend_threshold
            scale = self.cfg.trend_min_exposure + (1.0 - self.cfg.trend_min_exposure) * max(0.0, 1.0 - deviation)
            return float(np.clip(scale, self.cfg.trend_min_exposure, self.cfg.trend_max_exposure))
        return 1.0

    def _compute_vol_scale(
        self,
        market_vol: float | None,
        prev_weights: dict[str, float] | None,
        cov: np.ndarray | None,
    ) -> float:
        """Scale exposure based on volatility regime."""
        if not self.cfg.vol_regime_enabled:
            return 1.0

        if market_vol is not None:
            # Use provided market vol
            vol = market_vol
        elif prev_weights is not None and cov is not None:
            # Compute ex-ante portfolio vol
            w = np.array(list(prev_weights.values()))
            port_vol = float(np.sqrt(max(np.dot(w, np.dot(cov, w)), 1e-12)))
            vol = port_vol
        else:
            return 1.0

        self._vol_history.append(float(vol))
        if len(self._vol_history) > 252:
            self._vol_history = self._vol_history[-252:]

        # Vol targeting: scale inversely to vol
        if vol < 1e-8:
            return self.cfg.vol_max_scale

        raw_scale = self.cfg.vol_target / vol
        return float(np.clip(raw_scale, self.cfg.vol_min_scale, self.cfg.vol_max_scale))

    def _compute_crisis_scale(self, market_vol: float | None) -> float:
        """Emergency deleveraging during crisis conditions."""
        if not self.cfg.crisis_detection_enabled:
            return 1.0

        if self._crisis_cooldown > 0:
            self._crisis_cooldown -= 1
            return self.cfg.crisis_exposure_scale

        if market_vol is not None and market_vol > self.cfg.crisis_vol_threshold:
            self._crisis_cooldown = self.cfg.crisis_recovery_window
            return self.cfg.crisis_exposure_scale

        return 1.0

    def reset(self) -> None:
        """Reset overlay state for new evaluation period."""
        self._state = RiskOverlayState()
        self._signal_history.clear()
        self._market_returns.clear()
        self._vol_history.clear()
        self._crisis_cooldown = 0
