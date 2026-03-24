from __future__ import annotations

import pytest
import pandas as pd

from strategy.portfolio_construction import (
    RegimeExposureConfig,
    compute_rank_based_weights,
    select_high_conviction_assets,
    construct_regime_aware_portfolio,
    construct_top_k_weights,
)


@pytest.fixture
def synthetic_scores() -> dict[str, dict[str, float]]:
    return {
        "equal_scores": {"A": 1.0, "B": 1.0, "C": 1.0, "D": 1.0},
        "all_negative": {"A": -0.1, "B": -0.2, "C": -0.4, "D": -0.8},
        "one_dominant": {"A": 10.0, "B": 1.0, "C": 0.5, "D": 0.2},
    }


def test_weights_sum_to_one(synthetic_scores: dict[str, dict[str, float]]) -> None:
    w = construct_top_k_weights(synthetic_scores["one_dominant"], top_k=3)
    assert w
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-12)


def test_only_top_k_selected(synthetic_scores: dict[str, dict[str, float]]) -> None:
    w = construct_top_k_weights(synthetic_scores["one_dominant"], top_k=2)
    assert set(w.keys()) == {"A", "B"}
    assert len(w) == 2


def test_higher_score_gets_higher_weight(
    synthetic_scores: dict[str, dict[str, float]],
) -> None:
    w = construct_top_k_weights(synthetic_scores["one_dominant"], top_k=4)
    assert w["A"] > w["B"] > w["C"] > w["D"]


def test_stability_when_scores_equal(synthetic_scores: dict[str, dict[str, float]]) -> None:
    w = construct_top_k_weights(synthetic_scores["equal_scores"], top_k=3)
    assert len(w) == 3
    assert all(v == pytest.approx(1.0 / 3.0, abs=1e-12) for v in w.values())
    # deterministic tie-break by ticker
    assert list(w.keys()) == ["A", "B", "C"]


def test_all_negative_scores_do_not_break_normalization(
    synthetic_scores: dict[str, dict[str, float]],
) -> None:
    w = construct_top_k_weights(synthetic_scores["all_negative"], top_k=3)
    assert set(w.keys()) == {"A", "B", "C"}
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(v >= 0 for v in w.values())
    # less negative score gets larger weight
    assert w["A"] > w["B"] > w["C"]


def test_zero_sum_fallback_equal_weights() -> None:
    w = construct_top_k_weights({"A": 0.0, "B": 0.0, "C": 0.0}, top_k=3)
    assert sum(w.values()) == pytest.approx(1.0, abs=1e-12)
    assert all(v == pytest.approx(1.0 / 3.0, abs=1e-12) for v in w.values())


def test_crisis_layer_reduces_exposure_and_top_k() -> None:
    cfg = RegimeExposureConfig(normal_top_k=8, normal_exposure=1.0, crisis_top_k=4, crisis_exposure=0.25)
    scores = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6, "E": 0.5, "F": 0.4}
    out = construct_regime_aware_portfolio(scores, current_regime="Crisis", config=cfg)
    assert out["top_k_used"] == 4
    assert len(out["selected_assets"]) == 4
    assert out["effective_exposure"] == pytest.approx(0.25)
    assert sum(out["adjusted_weights"].values()) == pytest.approx(0.25, abs=1e-12)


def test_non_crisis_uses_normal_exposure_and_top_k() -> None:
    cfg = RegimeExposureConfig(normal_top_k=5, normal_exposure=1.0, crisis_top_k=3, crisis_exposure=0.2)
    scores = {"A": 1.0, "B": 0.9, "C": 0.8, "D": 0.7, "E": 0.6, "F": 0.5}
    out = construct_regime_aware_portfolio(scores, current_regime="Bull", config=cfg)
    assert out["top_k_used"] == 5
    assert len(out["selected_assets"]) == 5
    assert out["effective_exposure"] == pytest.approx(1.0)
    assert sum(out["adjusted_weights"].values()) == pytest.approx(1.0, abs=1e-12)


def test_unknown_regime_defaults_to_normal_path() -> None:
    cfg = RegimeExposureConfig(normal_top_k=6, normal_exposure=1.0, crisis_top_k=3, crisis_exposure=0.2)
    scores = {"A": 0.4, "B": 0.3, "C": 0.2, "D": 0.1}
    out = construct_regime_aware_portfolio(scores, current_regime="RegimeX", config=cfg)
    assert out["top_k_used"] == 6
    assert len(out["selected_assets"]) == 4
    assert out["effective_exposure"] == pytest.approx(1.0)


def test_rank_based_weights_sum_abs_to_one_long_short() -> None:
    df = pd.DataFrame({"adjusted_score": [0.1, 0.2, 0.3, 0.4]})
    out = compute_rank_based_weights(df, long_only=False)
    assert out["weight"].abs().sum() == pytest.approx(1.0, abs=1e-12)
    assert out["weight"].isna().sum() == 0


def test_rank_based_weights_long_only_non_negative_and_normalized() -> None:
    df = pd.DataFrame({"adjusted_score": [0.1, 0.2, 0.3, 0.4]})
    out = compute_rank_based_weights(df, long_only=True)
    assert (out["weight"] >= 0).all()
    assert out["weight"].abs().sum() == pytest.approx(1.0, abs=1e-12)


def test_rank_based_weights_equal_scores_stable_behavior() -> None:
    df = pd.DataFrame({"adjusted_score": [1.0, 1.0, 1.0, 1.0]})
    out_ls = compute_rank_based_weights(df, long_only=False)
    out_lo = compute_rank_based_weights(df, long_only=True)
    # Long/short: no dispersion => all zero weights
    assert (out_ls["weight"] == 0.0).all()
    # Long-only fallback: equal-weight over valid rows
    assert all(v == pytest.approx(0.25) for v in out_lo["weight"].tolist())


def test_rank_based_weights_handles_nans_without_propagation() -> None:
    df = pd.DataFrame({"adjusted_score": [0.5, None, 0.1, float("nan"), 0.9]})
    out = compute_rank_based_weights(df, long_only=True)
    assert out["weight"].isna().sum() == 0
    assert out["rank_pct"].isna().sum() == 0
    assert out["weight"].abs().sum() == pytest.approx(1.0, abs=1e-12)


def test_select_high_conviction_assets_basic_top_k() -> None:
    df = pd.DataFrame(
        {
            "ticker": list("ABCDEFG"),
            "rank_pct": [0.95, 0.90, 0.88, 0.70, 0.65, 0.40, 0.20],
            "adjusted_score": [1.5, 1.1, 1.0, 0.8, 0.7, 0.2, -0.1],
        }
    )
    out = select_high_conviction_assets(df, threshold=0.6, top_k=5)
    assert len(out) == 5
    assert (out["rank_pct"] > 0.6).all()
    assert out["weight"].sum() == pytest.approx(1.0, abs=1e-12)
    assert (out["weight"] >= 0).all()


def test_select_high_conviction_assets_fewer_than_k_uses_all_available() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D"],
            "rank_pct": [0.95, 0.75, 0.55, 0.10],
            "adjusted_score": [1.2, 0.9, 0.2, -0.1],
        }
    )
    out = select_high_conviction_assets(df, threshold=0.6, top_k=5)
    assert len(out) == 2
    assert set(out["ticker"]) == {"A", "B"}
    assert out["weight"].sum() == pytest.approx(1.0, abs=1e-12)


def test_select_high_conviction_assets_none_pass_fallback_to_top_k() -> None:
    df = pd.DataFrame(
        {
            "ticker": ["A", "B", "C", "D", "E", "F"],
            "rank_pct": [0.58, 0.57, 0.56, 0.55, 0.54, 0.53],
            "adjusted_score": [0.9, 0.7, 0.5, 0.3, 0.1, -0.1],
        }
    )
    out = select_high_conviction_assets(df, threshold=0.6, top_k=5)
    assert len(out) == 5
    assert out["weight"].sum() == pytest.approx(1.0, abs=1e-12)
    # top rank should be present, lowest should be dropped
    assert "A" in set(out["ticker"])
    assert "F" not in set(out["ticker"])

