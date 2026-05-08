from __future__ import annotations

import numpy as np
import pandas as pd

from model_selection.economic_objectives import DateGroupedEconomicModel
from model_selection.training import make_training_target
from run_model_selection import (
    _active_features_for_model_kind,
    _deployment_primary_path_for_model,
    _primary_evaluation_path,
    _validation_target_col_for_path,
)


def _panel(n_dates: int = 18, n_names: int = 24) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(7)
    rows = []
    groups = []
    tickers = []
    target = []
    costs = []
    for d in range(n_dates):
        quality = np.linspace(-1.0, 1.0, n_names) + rng.normal(0.0, 0.03, n_names)
        rows.append(np.c_[quality, rng.normal(0.0, 0.20, n_names)])
        groups.extend([d] * n_names)
        tickers.extend([f"T{i:03d}" for i in range(n_names)])
        target.extend(quality + rng.normal(0.0, 0.04, n_names))
        costs.extend(np.where(quality > 0.7, 0.02, 0.001))
    return (
        np.vstack(rows),
        np.asarray(target, dtype=float),
        np.asarray(groups, dtype=int),
        np.asarray(tickers, dtype=str),
        np.asarray(costs, dtype=float),
    )


def test_date_grouped_spread_objective_learns_cross_sectional_rank() -> None:
    x, y, groups, tickers, cost = _panel()
    model = DateGroupedEconomicModel(
        objective="long_short_spread",
        cost_penalty=0.0,
        turnover_penalty=0.0,
        learning_rate=0.03,
        max_iter=120,
        random_state=3,
        allow_portfolio_objective_training=True,
    )
    model.fit(x, y, _date_groups=groups, _tickers=tickers, _cost=cost)
    score = model.predict(x)
    assert np.corrcoef(score, y)[0, 1] > 0.80


def test_short_objective_keeps_low_scores_for_bad_short_candidates() -> None:
    x, y, groups, tickers, cost = _panel()
    model = DateGroupedEconomicModel(
        objective="short_side",
        cost_penalty=0.0,
        turnover_penalty=0.0,
        learning_rate=0.03,
        max_iter=120,
        random_state=3,
        allow_portfolio_objective_training=True,
    )
    model.fit(x, y, _date_groups=groups, _tickers=tickers, _cost=cost)
    score = model.predict(x)
    assert np.corrcoef(score, y)[0, 1] > 0.80
    assert float(score[y <= np.quantile(y, 0.10)].mean()) < float(score[y >= np.quantile(y, 0.90)].mean())


def test_economic_model_routing_and_targets_are_family_specific() -> None:
    df = pd.DataFrame(
        {
            "target_rank": [-1.0, 0.0, 1.0],
            "target_down_decile": [1, 0, 0],
            "target_return": [-0.02, 0.0, 0.03],
            "target_up": [0, 0, 1],
        }
    )
    assert _primary_evaluation_path("long_alpha") == "long_short_spread"
    assert _primary_evaluation_path("overlay_alpha") == "long_only_overlay"
    assert _primary_evaluation_path("short_alpha") == "short_side"
    assert _deployment_primary_path_for_model("regressor", "long_only_overlay") == "long_only_overlay"
    assert _deployment_primary_path_for_model("regressor", "long_short_spread") == "long_short_spread"
    assert _deployment_primary_path_for_model("short_alpha", "long_only_overlay") == "short_side"
    assert _validation_target_col_for_path("long_only_overlay") == "forward_return"
    assert _validation_target_col_for_path("long_short_spread") == "target_return"
    assert make_training_target(df, model_name="EconomicSpreadAlpha", model_kind="long_alpha", use_risk_adj=False).tolist() == [-1.0, 0.0, 1.0]
    assert make_training_target(df, model_name="EconomicOverlayAlpha", model_kind="overlay_alpha", use_risk_adj=False).tolist() == [-1.0, 0.0, 1.0]
    assert make_training_target(df, model_name="EconomicShortAlpha", model_kind="short_alpha", use_risk_adj=False).tolist() == [-1.0, 0.0, 1.0]


def test_feature_views_are_evidence_admitted_not_static_by_mandate() -> None:
    base = ["mom", "quality", "value"]
    short = ["short_mom"]
    overlay = ["quality", "low_vol"]
    assert _active_features_for_model_kind(
        "long_alpha",
        base,
        short_feature_subset=short,
        overlay_feature_subset=overlay,
    ) == base
    assert _active_features_for_model_kind(
        "short_alpha",
        base,
        short_feature_subset=short,
        overlay_feature_subset=overlay,
    ) == base
    assert _active_features_for_model_kind(
        "overlay_alpha",
        base,
        short_feature_subset=short,
        overlay_feature_subset=overlay,
    ) == base
