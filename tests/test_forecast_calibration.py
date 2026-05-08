from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.forecast import ForecastCalibrationConfig, ForecastCalibrator, ForecastEngine


def _synthetic_signal_data(n_days: int = 90, n_tickers: int = 6) -> dict[str, pd.DataFrame]:
    dates = pd.bdate_range("2020-01-01", periods=n_days)
    out: dict[str, pd.DataFrame] = {}
    for i in range(n_tickers):
        rng = np.random.default_rng(100 + i)
        score = np.linspace(-1.0, 1.0, n_days) + rng.normal(0.0, 0.05, n_days)
        realized = 0.002 + 0.015 * score + rng.normal(0.0, 0.002, n_days)
        out[f"T{i}"] = pd.DataFrame(
            {
                "adjusted_score": score,
                "forward_return": realized,
            },
            index=dates,
        )
    return out


def test_forecast_calibrator_maps_scores_to_realized_return_units(tmp_path) -> None:
    signal_data = _synthetic_signal_data()
    engine = ForecastEngine(scale_factor=0.001, smoothing_span=2, vol_scale_window=0)
    raw = engine.build_forecast_series(signal_data)
    calibrator = ForecastCalibrator(
        ForecastCalibrationConfig(
            enabled=True,
            method="linear",
            window_days=40,
            min_obs=80,
            horizon_days=5,
            output_dir=str(tmp_path),
            fallback_scale_factor=0.001,
            fallback_max_alpha=0.05,
        )
    )

    result = calibrator.calibrate(raw)

    assert not result.curve.empty
    assert not result.distribution.empty
    assert not result.diagnostics.empty
    assert (tmp_path / "forecast_calibration_curve.csv").exists()
    assert (tmp_path / "forecast_distribution_before_after.csv").exists()

    combined = []
    for ticker, df in result.signal_data.items():
        tmp = df[["raw_smoothed_forecast", "smoothed_forecast", "forward_return"]].dropna().copy()
        tmp["ticker"] = ticker
        combined.append(tmp)
    panel = pd.concat(combined)
    raw_scale = float(panel["raw_smoothed_forecast"].abs().mean())
    calibrated_scale = float(panel["smoothed_forecast"].abs().mean())
    realized_scale = float(panel["forward_return"].abs().mean())

    assert calibrated_scale > raw_scale
    assert abs(calibrated_scale - realized_scale) < abs(raw_scale - realized_scale)


def test_forecast_calibrator_is_causal_for_early_dates(tmp_path) -> None:
    signal_data = _synthetic_signal_data(n_days=30, n_tickers=3)
    engine = ForecastEngine(scale_factor=0.001, smoothing_span=2, vol_scale_window=0)
    raw = engine.build_forecast_series(signal_data)
    calibrator = ForecastCalibrator(
        ForecastCalibrationConfig(
            enabled=True,
            method="linear",
            window_days=20,
            min_obs=500,
            horizon_days=5,
            output_dir=str(tmp_path),
            fallback_scale_factor=0.001,
            fallback_max_alpha=0.05,
        )
    )

    result = calibrator.calibrate(raw)

    assert not result.diagnostics.empty
    assert set(result.diagnostics["method"].unique()) == {"fallback"}
    for ticker, df in result.signal_data.items():
        pd.testing.assert_series_equal(
            df["smoothed_forecast"],
            raw[ticker]["raw_smoothed_forecast"],
            check_names=False,
        )
