from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from run_model_selection import (
    _complexity_screen_model_template,
    _prescreen_qp_candidates,
    execution_validation,
    feasibility_filter,
    signal_screening,
)


class _ToyBoostedEstimator(BaseEstimator):
    def __init__(self, n_estimators: int = 100, max_depth: int = 6, num_leaves: int = 63) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.num_leaves = num_leaves

    def get_params(self, deep: bool = False) -> dict[str, int]:
        return {
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "num_leaves": self.num_leaves,
        }

    def set_params(self, **params: int) -> "_ToyBoostedEstimator":
        for key, value in params.items():
            setattr(self, key, value)
        return self


def _spec(name: str, path: str = "long_short_spread") -> tuple[str, Any, bool, str]:
    kind = {
        "long_short_spread": "regressor",
        "short_side": "short_classifier",
        "long_only_overlay": "overlay_alpha",
    }[path]
    return (name, object(), False, kind)


def test_signal_discovery_uses_feature_dimension_complexity_proxy() -> None:
    full = _ToyBoostedEstimator(n_estimators=100, max_depth=8, num_leaves=255)

    proxy = _complexity_screen_model_template(full, active_feature_count=7)

    assert proxy is not full
    assert proxy.n_estimators < full.n_estimators
    assert proxy.max_depth < full.max_depth
    assert proxy.num_leaves <= (2**proxy.max_depth) - 1
    assert full.n_estimators == 100


def test_numpy_qp_prescreen_matches_groupby_rank_contract() -> None:
    dates = pd.bdate_range("2022-01-03", periods=3)
    n_names = 80
    base_scores = np.linspace(-1.0, 1.0, n_names)
    base_scores[3] = np.nan
    base_scores[10] = base_scores[11]
    scored = pd.DataFrame(
        {
            "date": np.repeat(dates, n_names),
            "ticker": [f"T{i}" for _ in dates for i in range(n_names)],
            "score": np.tile(base_scores, len(dates)),
        }
    )
    k = 50
    rank_asc = scored.groupby("date")["score"].rank(ascending=True, method="first")
    rank_desc = scored.groupby("date")["score"].rank(ascending=False, method="first")
    expected_ls = scored[(rank_desc <= k) | (rank_asc <= k)].reset_index(drop=True)
    expected_long = scored[rank_desc <= k].reset_index(drop=True)

    fast_ls = _prescreen_qp_candidates(
        scored,
        primary_path="long_short_spread",
        max_positions=1,
        qp_top_k_multiplier=2,
    )
    fast_long = _prescreen_qp_candidates(
        scored,
        primary_path="long_only_overlay",
        max_positions=1,
        qp_top_k_multiplier=2,
    )

    pd.testing.assert_frame_equal(fast_ls, expected_ls)
    pd.testing.assert_frame_equal(fast_long, expected_long)


def test_feasibility_filter_reduces_candidate_count_by_path() -> None:
    specs = [_spec("A"), _spec("B"), _spec("C")]
    signal_results = [
        {"model_name": "A", "model_kind": "regressor", "primary_path": "long_short_spread", "feasibility_score": 0.30},
        {"model_name": "B", "model_kind": "regressor", "primary_path": "long_short_spread", "feasibility_score": 0.20},
        {"model_name": "C", "model_kind": "regressor", "primary_path": "long_short_spread", "feasibility_score": 0.10},
    ]

    shortlisted, audit = feasibility_filter(
        signal_results,
        specs,
        screening_cfg={"enabled": True, "shortlist_top_k_per_path": 1, "min_keep_per_path": 1},
    )

    assert [spec[0] for spec in shortlisted] == ["A"]
    assert audit[0]["candidate_count_in"] == 3
    assert audit[0]["candidate_count_out"] == 1


def test_funnel_keeps_full_simulation_work_on_stage_c_survivors_only() -> None:
    specs = [_spec("A"), _spec("B"), _spec("C")]
    calls: list[tuple[str, int]] = []

    def fake_runner(phase_name: str, phase_models: list[tuple[str, Any, bool, str]], *, evaluator: Any) -> list[dict[str, Any]]:
        calls.append((phase_name, len(phase_models)))
        if phase_name == "SignalDiscovery":
            return [
                {"model_name": "A", "model_kind": "regressor", "primary_path": "long_short_spread", "feasibility_score": 0.30},
                {"model_name": "B", "model_kind": "regressor", "primary_path": "long_short_spread", "feasibility_score": 0.20},
                {"model_name": "C", "model_kind": "regressor", "primary_path": "long_short_spread", "feasibility_score": 0.10},
            ]
        return [{"model_name": spec[0]} for spec in phase_models]

    signal_results = signal_screening(specs, phase_runner=fake_runner)
    shortlisted, _ = feasibility_filter(
        signal_results,
        specs,
        screening_cfg={"enabled": True, "shortlist_top_k_per_path": 1, "min_keep_per_path": 1},
    )
    execution_validation(shortlisted, phase_runner=fake_runner)

    assert calls == [("SignalDiscovery", 3), ("ExecutionValidation", 1)]
    assert calls[-1][1] < calls[0][1]


def test_execution_validation_results_unchanged_when_candidate_set_is_unchanged() -> None:
    specs = [_spec("A"), _spec("B")]

    def fake_runner(phase_name: str, phase_models: list[tuple[str, Any, bool, str]], *, evaluator: Any) -> list[dict[str, Any]]:
        return [{"phase": phase_name, "model_name": spec[0], "rank": idx} for idx, spec in enumerate(phase_models)]

    through_funnel, _ = feasibility_filter(
        [
            {"model_name": "A", "model_kind": "regressor", "primary_path": "long_short_spread", "feasibility_score": 0.30},
            {"model_name": "B", "model_kind": "regressor", "primary_path": "long_short_spread", "feasibility_score": 0.20},
        ],
        specs,
        screening_cfg={"enabled": True, "shortlist_top_k_per_path": 2, "min_keep_per_path": 2},
    )

    assert execution_validation(through_funnel, phase_runner=fake_runner) == execution_validation(specs, phase_runner=fake_runner)


def test_feasibility_filter_excludes_classifier_stock_selectors_from_stage_c() -> None:
    specs = [
        ("XGBClassifier", object(), True, "classifier"),
        ("XGBRegressor", object(), False, "regressor"),
        ("ShortXGB", object(), True, "short_classifier"),
    ]
    signal_results = [
        {"model_name": "XGBClassifier", "model_kind": "classifier", "primary_path": "long_short_spread", "feasibility_score": 0.90},
        {"model_name": "XGBRegressor", "model_kind": "regressor", "primary_path": "long_short_spread", "feasibility_score": 0.20},
        {"model_name": "ShortXGB", "model_kind": "short_classifier", "primary_path": "short_side", "feasibility_score": 0.80},
    ]

    shortlisted, audit = feasibility_filter(
        signal_results,
        specs,
        screening_cfg={"enabled": True, "shortlist_top_k_per_path": 2, "min_keep_per_path": 1},
    )

    assert [spec[0] for spec in shortlisted] == ["XGBRegressor"]
    diagnostic = {row["primary_path"]: row.get("diagnostic_only_models", []) for row in audit}
    assert diagnostic["long_short_spread"] == ["XGBClassifier"]
    assert diagnostic["short_side"] == ["ShortXGB"]
