"""
Forecast Engine
================
Transforms raw model signals into calibrated expected-return forecasts,
and tracks per-position alpha decay over the holding horizon.

Design principles (Grinold & Kahn, AQR research):
  - Signals live in score-space; forecasts live in return-space.
  - Alpha decays exponentially with a characteristic time tau.
  - Per-ticker rolling-vol normalisation preserves cross-time signal strength:
      strong days → bigger allocations, weak days → smaller allocations.
  - Cross-sectional demeaning (optional) removes market-level bias.
  - NO forced unit-variance rescaling: signal magnitude is economically meaningful.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ------------------------------------------------------------------
# Forecast Engine
# ------------------------------------------------------------------

class ForecastEngine:
    """
    Maps adjusted_score series → calibrated expected-return series.

    Parameters
    ----------
    tau_days : float
        Alpha decay time constant in trading days (half-life ≈ 0.693 × tau).
    smoothing_span : int
        EMA span for signal smoothing. Separates noise from persistent alpha.
    scale_factor : float
        Converts vol-normalised score units → fractional expected return.
        Calibration at actual large-cap vol=30%, lambda=2:
          w* = scale×z / (2λσ²)  →  scale=0.012 gives w*(z=1)=3.3%, w*(z=2)=6.7%.
    vol_scale_window : int
        Lookback window (days) for per-ticker rolling signal volatility.
        Dividing by rolling vol normalises each ticker to its own historical
        signal amplitude.  Zero or negative disables vol scaling.
        Default 60 matches the covariance estimation window.
    vol_scale_floor : float
        Minimum rolling vol allowed before division (prevents division by near-zero).
        Default 1e-4 (appropriate for adjusted_score which is ~[-0.1, +0.06]).
    """

    def __init__(
        self,
        tau_days: float = 6.0,
        smoothing_span: int = 5,
        scale_factor: float = 0.012,
        vol_scale_window: int = 60,
        vol_scale_floor: float = 1e-4,
    ):
        self.tau_days = float(tau_days)
        self.smoothing_span = int(smoothing_span)
        self.scale_factor = float(scale_factor)
        self.vol_scale_window = int(vol_scale_window)
        self.vol_scale_floor = float(vol_scale_floor)

    # ------------------------------------------------------------------
    # Vectorised: full time-series → full forecast series (no look-ahead)
    # ------------------------------------------------------------------

    def signal_to_forecast(self, score_series: pd.Series) -> pd.Series:
        """
        Vectorised: score_series (daily adjusted_score) → forecast series.

        Pipeline:
          1. EMA smoothing (removes day-to-day noise)
          2. Per-ticker vol normalisation (preserves cross-time signal strength)
          3. scale_factor mapping → expected-return units

        EMA smoothing is causal: ewm().mean() at time t uses only t, t-1, ...
        Rolling vol is also causal (no look-ahead).
        """
        # Step 1: EMA smoothing
        smoothed = (
            score_series
            .ewm(span=max(1, self.smoothing_span), min_periods=min(3, self.smoothing_span))
            .mean()
        )

        # Step 2: per-ticker rolling vol normalisation
        if self.vol_scale_window > 0:
            rolling_vol = (
                score_series
                .rolling(window=self.vol_scale_window, min_periods=max(5, self.vol_scale_window // 4))
                .std()
            )
            # Floor: avoid division by near-zero during warmup or flat-score periods
            rolling_vol = rolling_vol.where(
                rolling_vol > self.vol_scale_floor, self.vol_scale_floor
            )
            # Fill NaN in early rows with the floor (not enough history)
            rolling_vol = rolling_vol.fillna(self.vol_scale_floor)
            normalised = smoothed / rolling_vol
        else:
            normalised = smoothed

        return normalised * self.scale_factor

    # ------------------------------------------------------------------
    # Alpha decay: per-position remaining expected return
    # ------------------------------------------------------------------

    def decay_factor(self, age_days: int | float) -> float:
        """exp(-age / tau). At age=tau: 36.8% remaining; at age=2*tau: 13.5%."""
        return float(np.exp(-float(age_days) / self.tau_days))

    def remaining_alpha(self, entry_forecast: float, age_days: int | float) -> float:
        """
        Expected return remaining in a position opened at entry_forecast,
        now age_days old.  Used by caller to compare against new_forecast.
        """
        return entry_forecast * self.decay_factor(age_days)

    # ------------------------------------------------------------------
    # Cross-section snapshot: all tickers for a single date
    # ------------------------------------------------------------------

    def current_forecasts(
        self,
        signal_data: dict[str, pd.DataFrame],
        date: pd.Timestamp,
        score_col: str = "adjusted_score",
    ) -> dict[str, float]:
        """
        Returns pre-smoothed, vol-normalised forecast for each ticker on `date`.
        Looks for `smoothed_forecast` column (pre-built by build_forecast_series).
        Falls back to raw adjusted_score × scale_factor if absent.
        """
        out: dict[str, float] = {}
        for ticker, df in signal_data.items():
            if df is None or df.empty or date not in df.index:
                continue
            row = df.loc[date]
            if "smoothed_forecast" in df.columns:
                v = float(row["smoothed_forecast"])
            elif score_col in df.columns:
                v = float(row[score_col]) * self.scale_factor
            else:
                continue
            if np.isfinite(v):
                out[ticker] = v
        return out

    # ------------------------------------------------------------------
    # Pre-compute: attach smoothed_forecast column to all signal DataFrames
    # ------------------------------------------------------------------

    def build_forecast_series(
        self,
        signal_data: dict[str, pd.DataFrame],
        score_col: str = "adjusted_score",
    ) -> dict[str, pd.DataFrame]:
        """
        Adds `smoothed_forecast` column to each ticker's signal DataFrame.
        Called once before the simulation loop — no per-bar overhead.
        Returns a new dict (does not mutate the input).
        """
        result: dict[str, pd.DataFrame] = {}
        for ticker, df in signal_data.items():
            if df is None or df.empty or score_col not in df.columns:
                result[ticker] = df
                continue
            df_new = df.copy()
            df_new["smoothed_forecast"] = self.signal_to_forecast(
                pd.to_numeric(df_new[score_col], errors="coerce").fillna(0.0)
            )
            result[ticker] = df_new
        return result
