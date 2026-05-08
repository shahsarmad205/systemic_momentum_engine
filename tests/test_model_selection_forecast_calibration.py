from __future__ import annotations

import numpy as np
import pandas as pd

from model_selection.forecast_calibration import calibrate_scores, fit_score_calibrator


def test_score_calibrator_maps_raw_scores_to_return_units_cross_sectionally() -> None:
    dates = pd.bdate_range("2021-01-01", periods=20)
    rows = []
    raw_scores = []
    for d_idx, date in enumerate(dates):
        for i in range(8):
            score = float(i - 3.5)
            rows.append(
                {
                    "date": date,
                    "ticker": f"T{i}",
                    "target_return": 0.001 * d_idx + 0.02 * score,
                }
            )
            raw_scores.append(score)
    panel = pd.DataFrame(rows)

    result = fit_score_calibrator(panel, np.asarray(raw_scores), target_col="target_return")
    transformed = result.transform(np.asarray([-2.0, 0.0, 2.0]))

    assert result.method == "cross_sectional_linear_shrunk"
    assert result.n_dates == len(dates)
    assert result.slope > 0.015
    assert transformed[2] > transformed[1] > transformed[0]


def test_score_calibrator_shrinks_noisy_mapping_without_fixed_thresholds() -> None:
    rng = np.random.default_rng(42)
    dates = np.repeat(pd.bdate_range("2022-01-01", periods=30), 10)
    scores = rng.normal(size=len(dates))
    targets = rng.normal(scale=0.03, size=len(dates))
    panel = pd.DataFrame({"date": dates, "target_return": targets})

    result = fit_score_calibrator(panel, scores, target_col="target_return")

    assert 0.0 <= result.shrinkage <= 1.0
    assert abs(result.slope) <= abs(np.dot(scores - scores.mean(), targets - targets.mean()) / np.dot(scores - scores.mean(), scores - scores.mean()))


def test_calibrate_scores_uses_training_panel_only_for_eval_transform() -> None:
    train = pd.DataFrame(
        {
            "date": pd.bdate_range("2020-01-01", periods=6).repeat(4),
            "target_return": np.tile(np.array([-0.03, -0.01, 0.01, 0.03]), 6),
        }
    )
    train_scores = np.tile(np.array([-3.0, -1.0, 1.0, 3.0]), 6)
    eval_scores = np.array([-2.0, 0.0, 2.0])

    calibrated, result = calibrate_scores(train, train_scores, eval_scores, target_col="target_return")

    assert result.n_obs == len(train)
    assert calibrated.shape == eval_scores.shape
    assert calibrated[2] > calibrated[1] > calibrated[0]
