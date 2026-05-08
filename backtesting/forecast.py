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

Decay calibration:
  tau_from_span(span) — convert EMA span to equivalent tau (no free parameters).
  estimate_halflife()  — measure observed signal persistence from score autocorrelation.
  Both are used by _simulate_continuous to auto-calibrate execution speed.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ForecastCalibrationConfig:
    enabled: bool = False
    method: str = "linear"
    window_days: int = 504
    min_obs: int = 250
    horizon_days: int = 5
    target_col: str = "forward_return"
    clip_quantile: float = 0.995
    output_dir: str = "output/forecast_calibration"
    fallback_scale_factor: float = 0.012
    fallback_max_alpha: float = 0.05


@dataclass(frozen=True)
class ForecastCalibrationResult:
    signal_data: dict[str, pd.DataFrame]
    curve: pd.DataFrame
    distribution: pd.DataFrame
    diagnostics: pd.DataFrame


def _safe_corr(x: pd.Series, y: pd.Series) -> float:
    try:
        out = float(pd.to_numeric(x, errors="coerce").corr(pd.to_numeric(y, errors="coerce")))
        return out if np.isfinite(out) else float("nan")
    except Exception:
        return float("nan")


class ForecastCalibrator:
    """Causal score-to-return calibration for forecast scaling.

    The model never changes predictions or features. It only maps an existing
    score-derived forecast proxy into realized-return units using historical
    pairs that would have been known at the forecast date.
    """

    def __init__(self, cfg: ForecastCalibrationConfig):
        self.cfg = cfg

    @staticmethod
    def from_config(raw: dict | None, *, fallback_scale_factor: float, fallback_max_alpha: float) -> "ForecastCalibrator":
        raw = raw or {}
        cfg = ForecastCalibrationConfig(
            enabled=bool(raw.get("enabled", False)),
            method=str(raw.get("method", "linear")).strip().lower() or "linear",
            window_days=int(raw.get("window_days", 504) or 504),
            min_obs=int(raw.get("min_obs", 250) or 250),
            horizon_days=int(raw.get("horizon_days", raw.get("holding_period_days", 5)) or 5),
            target_col=str(raw.get("target_col", "forward_return") or "forward_return"),
            clip_quantile=float(raw.get("clip_quantile", 0.995) or 0.995),
            output_dir=str(raw.get("output_dir", "output/forecast_calibration") or "output/forecast_calibration"),
            fallback_scale_factor=float(fallback_scale_factor),
            fallback_max_alpha=float(fallback_max_alpha),
        )
        return ForecastCalibrator(cfg)

    def calibrate(
        self,
        signal_data: dict[str, pd.DataFrame],
        *,
        raw_forecast_col: str = "raw_smoothed_forecast",
    ) -> ForecastCalibrationResult:
        if not self.cfg.enabled:
            return ForecastCalibrationResult(
                signal_data=signal_data,
                curve=pd.DataFrame(),
                distribution=pd.DataFrame(),
                diagnostics=pd.DataFrame(),
            )

        panel_parts: list[pd.DataFrame] = []
        for ticker, df in signal_data.items():
            if df is None or df.empty or raw_forecast_col not in df.columns or self.cfg.target_col not in df.columns:
                continue
            tmp = df[[raw_forecast_col, self.cfg.target_col]].copy()
            tmp["date"] = pd.to_datetime(tmp.index, errors="coerce")
            tmp["ticker"] = str(ticker)
            panel_parts.append(tmp.reset_index(drop=True))
        if not panel_parts:
            logger.warning("ForecastCalibrator: no score/return pairs found; using fixed fallback scaling.")
            return ForecastCalibrationResult(signal_data=signal_data, curve=pd.DataFrame(), distribution=pd.DataFrame(), diagnostics=pd.DataFrame())

        panel = pd.concat(panel_parts, ignore_index=True)
        panel = panel.rename(columns={raw_forecast_col: "raw_forecast", self.cfg.target_col: "realized_return"})
        panel["raw_forecast"] = pd.to_numeric(panel["raw_forecast"], errors="coerce")
        panel["realized_return"] = pd.to_numeric(panel["realized_return"], errors="coerce")
        panel = panel.dropna(subset=["date", "raw_forecast", "realized_return"]).sort_values(["date", "ticker"])
        dates = pd.DatetimeIndex(sorted(panel["date"].unique()))
        if panel.empty or dates.empty:
            return ForecastCalibrationResult(signal_data=signal_data, curve=pd.DataFrame(), distribution=pd.DataFrame(), diagnostics=pd.DataFrame())

        calibrated_parts: list[pd.DataFrame] = []
        diag_rows: list[dict] = []
        horizon_bday = pd.offsets.BDay(max(1, int(self.cfg.horizon_days)))
        window_days = max(1, int(self.cfg.window_days))
        min_obs = max(10, int(self.cfg.min_obs))
        q = min(0.9999, max(0.50, float(self.cfg.clip_quantile)))

        for dt in dates:
            cutoff = pd.Timestamp(dt) - horizon_bday
            start = cutoff - pd.offsets.BDay(window_days)
            hist = panel[(panel["date"] >= start) & (panel["date"] <= cutoff)]
            today = panel[panel["date"].eq(dt)]
            if today.empty:
                continue
            intercept = 0.0
            slope = 1.0
            clip_abs = float(self.cfg.fallback_max_alpha)
            method_used = "fallback"
            n_obs = int(len(hist))
            if n_obs >= min_obs and hist["raw_forecast"].std(ddof=0) > 1e-12:
                x = hist["raw_forecast"].to_numpy(dtype=float)
                y = hist["realized_return"].to_numpy(dtype=float)
                slope, intercept = np.polyfit(x, y, 1)
                if not (np.isfinite(slope) and np.isfinite(intercept)):
                    slope, intercept = 1.0, 0.0
                    method_used = "fallback"
                else:
                    method_used = "linear"
                if self.cfg.method == "piecewise":
                    # Piecewise through median split; use the side appropriate
                    # for each observation below.
                    method_used = "piecewise"
                clip_abs = float(np.nanquantile(np.abs(y), q)) if len(y) else clip_abs
                if not np.isfinite(clip_abs) or clip_abs <= 0:
                    clip_abs = float(self.cfg.fallback_max_alpha)

            out = today[["date", "ticker", "raw_forecast", "realized_return"]].copy()
            if method_used == "piecewise" and n_obs >= min_obs:
                median_x = float(hist["raw_forecast"].median())
                low = hist[hist["raw_forecast"] <= median_x]
                high = hist[hist["raw_forecast"] > median_x]
                calibrated = np.zeros(len(out), dtype=float)
                for mask_value, subset in ((False, low), (True, high)):
                    mask = (out["raw_forecast"].to_numpy(dtype=float) > median_x) == mask_value
                    if subset.shape[0] >= max(20, min_obs // 4) and subset["raw_forecast"].std(ddof=0) > 1e-12:
                        s, b = np.polyfit(subset["raw_forecast"].to_numpy(dtype=float), subset["realized_return"].to_numpy(dtype=float), 1)
                    else:
                        s, b = slope, intercept
                    calibrated[mask] = b + s * out.loc[mask, "raw_forecast"].to_numpy(dtype=float)
                out["calibrated_forecast"] = calibrated
            else:
                out["calibrated_forecast"] = intercept + slope * out["raw_forecast"]
            out["calibrated_forecast"] = out["calibrated_forecast"].clip(-clip_abs, clip_abs)
            calibrated_parts.append(out)
            diag_rows.append(
                {
                    "date": pd.Timestamp(dt),
                    "method": method_used,
                    "n_obs": n_obs,
                    "slope": float(slope),
                    "intercept": float(intercept),
                    "clip_abs": float(clip_abs),
                    "raw_mean": float(out["raw_forecast"].mean()),
                    "calibrated_mean": float(out["calibrated_forecast"].mean()),
                    "realized_mean": float(out["realized_return"].mean()),
                    "raw_realized_corr": _safe_corr(out["raw_forecast"], out["realized_return"]),
                    "calibrated_realized_corr": _safe_corr(out["calibrated_forecast"], out["realized_return"]),
                }
            )

        if not calibrated_parts:
            return ForecastCalibrationResult(signal_data=signal_data, curve=pd.DataFrame(), distribution=pd.DataFrame(), diagnostics=pd.DataFrame())

        calibrated_panel = pd.concat(calibrated_parts, ignore_index=True)
        result_data: dict[str, pd.DataFrame] = {}
        for ticker, df in signal_data.items():
            if df is None or df.empty:
                result_data[ticker] = df
                continue
            df_new = df.copy()
            ticker_cal = calibrated_panel[calibrated_panel["ticker"].eq(str(ticker))].set_index("date")
            aligned = ticker_cal.reindex(pd.DatetimeIndex(df_new.index))
            if "calibrated_forecast" in aligned:
                df_new["smoothed_forecast"] = pd.to_numeric(aligned["calibrated_forecast"], errors="coerce").fillna(df_new.get("smoothed_forecast", 0.0)).to_numpy(dtype=float)
            result_data[ticker] = df_new

        curve = self._calibration_curve(calibrated_panel)
        distribution = self._distribution_summary(calibrated_panel)
        diagnostics = pd.DataFrame(diag_rows)
        self._write_outputs(curve, distribution, diagnostics)
        return ForecastCalibrationResult(result_data, curve, distribution, diagnostics)

    def _calibration_curve(self, panel: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
        if panel.empty:
            return pd.DataFrame()
        work = panel.dropna(subset=["raw_forecast", "calibrated_forecast", "realized_return"]).copy()
        if work.empty:
            return pd.DataFrame()
        work["bin"] = pd.qcut(work["raw_forecast"].rank(method="first"), q=min(n_bins, len(work)), duplicates="drop")
        return (
            work.groupby("bin", observed=False)
            .agg(
                n=("realized_return", "size"),
                raw_forecast_mean=("raw_forecast", "mean"),
                calibrated_forecast_mean=("calibrated_forecast", "mean"),
                realized_return_mean=("realized_return", "mean"),
                realized_return_std=("realized_return", "std"),
            )
            .reset_index(drop=True)
        )

    def _distribution_summary(self, panel: pd.DataFrame) -> pd.DataFrame:
        rows: list[dict] = []
        for col in ("raw_forecast", "calibrated_forecast", "realized_return"):
            s = pd.to_numeric(panel.get(col), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                continue
            rows.append(
                {
                    "series": col,
                    "mean": float(s.mean()),
                    "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
                    "p01": float(s.quantile(0.01)),
                    "p05": float(s.quantile(0.05)),
                    "p50": float(s.quantile(0.50)),
                    "p95": float(s.quantile(0.95)),
                    "p99": float(s.quantile(0.99)),
                }
            )
        return pd.DataFrame(rows)

    def _write_outputs(self, curve: pd.DataFrame, distribution: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
        try:
            out = Path(self.cfg.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            if not curve.empty:
                curve.to_csv(out / "forecast_calibration_curve.csv", index=False)
            if not distribution.empty:
                distribution.to_csv(out / "forecast_distribution_before_after.csv", index=False)
            if not diagnostics.empty:
                diagnostics.to_csv(out / "forecast_calibration_diagnostics.csv", index=False)
        except Exception as exc:
            logger.warning("ForecastCalibrator: failed to write calibration outputs: %s", exc)


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
    # Static: EMA span ↔ tau conversion
    # ------------------------------------------------------------------

    @staticmethod
    def tau_from_span(span: int) -> float:
        """
        Convert EMA smoothing span to equivalent alpha decay tau.

        EMA(span) assigns weight α = 2/(span+1) to the most recent observation,
        which is equivalent to exponential decay  exp(-1/τ) = 1 − α.

        For tau_exec in the TradeScheduler, this gives the execution fraction
        f = 1 − exp(−1/τ) = α = 2/(span+1) — the exact fraction that matches
        the signal smoothing speed.

        Parameters
        ----------
        span : int  EMA span (as used in pd.Series.ewm(span=...))

        Returns
        -------
        float  tau in trading days
        """
        alpha_ema = 2.0 / (max(1, int(span)) + 1.0)
        alpha_ema = min(alpha_ema, 0.999)  # avoid log(0)
        return float(-1.0 / np.log(1.0 - alpha_ema))

    # ------------------------------------------------------------------
    # Classmethod: estimate signal halflife from score autocorrelation
    # ------------------------------------------------------------------

    @classmethod
    def estimate_halflife(
        cls,
        signal_data: dict[str, "pd.DataFrame"],
        score_col: str = "adjusted_score",
        max_tickers: int = 100,
        min_obs: int = 30,
    ) -> float:
        """
        Estimate alpha decay tau from median lag-1 rank autocorrelation.

        For each ticker, computes Spearman(score_t, score_{t-1}) — how
        much of yesterday's ranking persists today.  The median across
        tickers is a robust estimate of cross-sectional persistence.

        tau = −1 / ln(median_autocorr)
        halflife = tau × ln(2)

        A fast-decaying signal (r ≈ 0.5) gives tau ≈ 1.4d (halflife ≈ 1d).
        A persistent signal (r ≈ 0.8) gives tau ≈ 4.5d (halflife ≈ 3d).

        Returns
        -------
        float : tau in trading days, or nan if insufficient data.
        """
        acorrs: list[float] = []
        sample = list(signal_data.items())[:max_tickers]
        for _ticker, df in sample:
            if df is None or df.empty or score_col not in df.columns:
                continue
            s = pd.to_numeric(df[score_col], errors="coerce").dropna()
            if len(s) < min_obs:
                continue
            r_rank = float(s.rank().corr(s.shift(1).rank()))
            if np.isfinite(r_rank) and 0.0 < r_rank < 1.0:
                acorrs.append(r_rank)

        if len(acorrs) < 3:
            return float("nan")

        r_med = float(np.median(acorrs))
        if not (0.0 < r_med < 1.0):
            return float("nan")

        tau = -1.0 / np.log(r_med)
        logger.debug(
            "ForecastEngine.estimate_halflife: r_med=%.3f τ=%.2fd halflife=%.2fd (n=%d tickers)",
            r_med, tau, tau * np.log(2), len(acorrs),
        )
        return float(tau)

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
            df_new["raw_smoothed_forecast"] = self.signal_to_forecast(
                pd.to_numeric(df_new[score_col], errors="coerce").fillna(0.0)
            )
            df_new["smoothed_forecast"] = df_new["raw_smoothed_forecast"]
            result[ticker] = df_new
        return result
