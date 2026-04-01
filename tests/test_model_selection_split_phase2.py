from __future__ import annotations

import pandas as pd

from run_model_selection import _pick_split_best_models, _target_name_for_model_kind


def test_pick_split_best_models_uses_ranked_rows() -> None:
    report = pd.DataFrame(
        [
            {"model_name": "LogisticRegression", "model_kind": "classifier", "_selection_metric": 0.9},
            {"model_name": "ShortXGB", "model_kind": "short_classifier", "_selection_metric": 0.8},
            {"model_name": "ShortLogistic", "model_kind": "short_classifier", "_selection_metric": 0.7},
            {"model_name": "Ridge", "model_kind": "regressor", "_selection_metric": 0.6},
        ]
    )

    best_long, best_short = _pick_split_best_models(report)
    assert best_long == "LogisticRegression"
    assert best_short == "ShortXGB"


def test_pick_split_best_models_handles_missing_side() -> None:
    report = pd.DataFrame(
        [
            {"model_name": "Ridge", "model_kind": "regressor", "_selection_metric": 1.0},
            {"model_name": "ShortLogistic", "model_kind": "short_classifier", "_selection_metric": 0.5},
        ]
    )

    best_long, best_short = _pick_split_best_models(report)
    assert best_long is None
    assert best_short == "ShortLogistic"


def test_target_name_for_model_kind_split_enabled() -> None:
    assert _target_name_for_model_kind("classifier", True) == "y_long"
    assert _target_name_for_model_kind("short_classifier", True) == "y_short"
    assert _target_name_for_model_kind("regressor", True) == "forward_return"


def test_target_name_for_model_kind_split_disabled() -> None:
    assert _target_name_for_model_kind("classifier", False) == "forward_return"
    assert _target_name_for_model_kind("short_classifier", False) == "forward_return"
    assert _target_name_for_model_kind("regressor", False) == "forward_return"
