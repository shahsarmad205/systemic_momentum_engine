from __future__ import annotations

import pytest
import numpy as np
import pandas as pd

from backtesting.portfolio_construction import (
    PortfolioConstraints,
    PortfolioConstructor,
    PortfolioInputs,
    RegimeExposureConfig,
    compute_rank_based_weights,
    select_high_conviction_assets,
    construct_regime_aware_portfolio,
    construct_top_k_weights,
)
from backtesting.risk_model import RiskModel
from model_selection.validation import (
    EvaluationConfig,
    MetricIntegrityError,
    annualized_sharpe,
    build_target_weights,
    simulate_executable_portfolio,
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


def _institutional_inputs() -> PortfolioInputs:
    tickers = ["A", "B", "C", "D", "E", "F"]
    return PortfolioInputs(
        date=pd.Timestamp("2022-01-03"),
        tickers=tickers,
        scores=pd.Series([2.0, 1.5, 1.0, -1.0, -1.5, -2.0], index=tickers),
        previous_weights=pd.Series(0.0, index=tickers),
        covariance=np.eye(len(tickers), dtype=float) * 0.04,
        beta=pd.Series([1.4, 1.2, 0.8, -0.8, -1.2, -1.4], index=tickers),
        sectors=pd.Series(["Tech", "Tech", "Health", "Health", "Energy", "Energy"], index=tickers),
        adv_dollar=pd.Series(100_000_000.0, index=tickers),
        squeeze_risk=pd.Series(0.0, index=tickers),
        hard_short_squeeze=pd.Series(0.0, index=tickers),
    )


def _institutional_constraints(**overrides: object) -> PortfolioConstraints:
    values = {
        "path": "long_short_spread",
        "max_positions": 6,
        "min_positions": 2,
        "max_gross": 0.60,
        "max_net": 1e-6,
        "max_name_weight": 0.20,
        "use_optimizer": False,
        "factor_neutral": True,
        "beta_neutral": True,
        "sector_neutral": True,
        "max_beta_abs": 1e-6,
        "max_sector_abs": 1e-6,
        "adv_fraction": 0.05,
        "short_squeeze_filter": True,
        "no_trade_band_weight_diff": 0.0,
        "no_trade_band_total_drift": 0.0,
    }
    values.update(overrides)
    return PortfolioConstraints(**values)


def test_portfolio_constructor_enforces_beta_neutrality() -> None:
    inputs = _institutional_inputs()
    constraints = _institutional_constraints(sector_neutral=False)
    result = PortfolioConstructor.build_weights(inputs, constraints)

    aligned = result.weights.reindex(inputs.tickers).fillna(0.0)
    beta = inputs.beta.reindex(inputs.tickers) - inputs.beta.reindex(inputs.tickers).mean()

    assert not result.violations
    assert abs(float((aligned * beta).sum())) <= constraints.max_beta_abs + 1e-6


def test_risk_model_factor_exposures_align_to_ticker_order() -> None:
    dates = pd.bdate_range("2022-01-03", periods=30)
    beta_by_ticker = {"A": 1.2, "B": -0.4, "C": 0.7}
    price_data = {}
    for idx, ticker in enumerate(["A", "B", "C"]):
        price_data[ticker] = pd.DataFrame(
            {
                "Close": 100.0 + np.arange(len(dates), dtype=float) + idx,
                "Volume": 1_000_000.0 + idx,
                "capm_beta": beta_by_ticker[ticker],
                "market_cap": 1_000_000_000.0 * (idx + 1),
                "momentum": float(idx),
                "book_to_market": 0.5 + idx,
                "quality_score": 1.0 - idx,
            },
            index=dates,
        )

    tickers = ["C", "A", "B"]
    exposures = RiskModel(window=20, min_periods=5).compute_factor_exposures_at_date(
        price_data,
        tickers,
        dates[-1],
        sector_id_map={"A": 1, "B": 2, "C": 1},
        sector_labels={1: "Tech", 2: "Energy"},
    )

    assert list(exposures)[:1] == ["market_beta"]
    assert exposures["market_beta"].tolist() == pytest.approx([0.7, 1.2, -0.4])
    assert exposures["sector:Tech"].tolist() == pytest.approx([1.0, 1.0, 0.0])
    assert exposures["sector:Energy"].tolist() == pytest.approx([0.0, 0.0, 1.0])
    for key in ["size", "momentum", "value", "quality"]:
        assert key in exposures
        assert len(exposures[key]) == len(tickers)


def test_portfolio_constructor_rejects_missing_required_risk_exposures() -> None:
    tickers = ["A", "B", "C"]
    inputs = PortfolioInputs(
        date=pd.Timestamp("2022-01-03"),
        tickers=tickers,
        scores=pd.Series([1.0, 0.0, -1.0], index=tickers),
        adv_dollar=pd.Series(100_000_000.0, index=tickers),
        squeeze_risk=pd.Series(0.0, index=tickers),
    )

    with pytest.raises(ValueError, match="beta exposures"):
        PortfolioConstructor.build_weights(
            inputs,
            _institutional_constraints(beta_neutral=True, sector_neutral=False),
        )

    with pytest.raises(ValueError, match="style exposure 'size'"):
        PortfolioConstructor.build_weights(
            PortfolioInputs(
                date=pd.Timestamp("2022-01-03"),
                tickers=tickers,
                scores=pd.Series([1.0, 0.0, -1.0], index=tickers),
                factor_exposures={"market_beta": pd.Series([1.0, 1.0, 1.0], index=tickers)},
                adv_dollar=pd.Series(100_000_000.0, index=tickers),
                squeeze_risk=pd.Series(0.0, index=tickers),
            ),
            _institutional_constraints(
                beta_neutral=True,
                sector_neutral=False,
                style_exposure_limits={"size": 0.05},
            ),
        )


def test_portfolio_constructor_enforces_risk_model_beta_exposure() -> None:
    inputs = _institutional_inputs()
    risk_exposures = {
        "market_beta": pd.Series([1.4, 1.2, 0.8, -0.8, -1.2, -1.4], index=inputs.tickers),
    }
    inputs = PortfolioInputs(
        date=inputs.date,
        tickers=inputs.tickers,
        scores=inputs.scores,
        previous_weights=inputs.previous_weights,
        covariance=inputs.covariance,
        factor_exposures=risk_exposures,
        adv_dollar=inputs.adv_dollar,
        squeeze_risk=inputs.squeeze_risk,
        hard_short_squeeze=inputs.hard_short_squeeze,
    )
    constraints = _institutional_constraints(sector_neutral=False)
    result = PortfolioConstructor.build_weights(inputs, constraints)

    aligned = result.weights.reindex(inputs.tickers).fillna(0.0)
    beta = risk_exposures["market_beta"] - risk_exposures["market_beta"].mean()

    assert not result.violations
    assert abs(float((aligned * beta).sum())) <= constraints.max_beta_abs + 1e-6


def test_portfolio_constructor_enforces_sector_neutrality() -> None:
    inputs = _institutional_inputs()
    constraints = _institutional_constraints(beta_neutral=False)
    result = PortfolioConstructor.build_weights(inputs, constraints)

    aligned = result.weights.reindex(inputs.tickers).fillna(0.0)
    sectors = inputs.sectors.reindex(inputs.tickers).astype(str)

    assert not result.violations
    for sector in sorted(sectors.unique()):
        exposure = float(aligned[sectors == sector].sum() - aligned.mean() * int((sectors == sector).sum()))
        assert abs(exposure) <= constraints.max_sector_abs + 1e-6


def test_portfolio_constructor_enforces_risk_model_sector_exposure() -> None:
    inputs = _institutional_inputs()
    sector_exposures = {
        "market_beta": pd.Series([1.0] * len(inputs.tickers), index=inputs.tickers),
        "sector:Tech": pd.Series([1.0, 1.0, 0.0, 0.0, 0.0, 0.0], index=inputs.tickers),
        "sector:Health": pd.Series([0.0, 0.0, 1.0, 1.0, 0.0, 0.0], index=inputs.tickers),
        "sector:Energy": pd.Series([0.0, 0.0, 0.0, 0.0, 1.0, 1.0], index=inputs.tickers),
    }
    inputs = PortfolioInputs(
        date=inputs.date,
        tickers=inputs.tickers,
        scores=inputs.scores,
        previous_weights=inputs.previous_weights,
        covariance=inputs.covariance,
        factor_exposures=sector_exposures,
        adv_dollar=inputs.adv_dollar,
        squeeze_risk=inputs.squeeze_risk,
        hard_short_squeeze=inputs.hard_short_squeeze,
    )
    constraints = _institutional_constraints(beta_neutral=False)
    result = PortfolioConstructor.build_weights(inputs, constraints)
    aligned = result.weights.reindex(inputs.tickers).fillna(0.0)

    assert not result.violations
    for name, exposure in sector_exposures.items():
        if not name.startswith("sector:"):
            continue
        centered = exposure - exposure.mean()
        assert abs(float((aligned * centered).sum())) <= constraints.max_sector_abs + 1e-6


def test_portfolio_constructor_respects_gross_net_and_name_limits() -> None:
    inputs = _institutional_inputs()
    constraints = _institutional_constraints(
        beta_neutral=False,
        sector_neutral=False,
        max_gross=0.40,
        max_net=0.05,
        max_name_weight=0.12,
    )
    result = PortfolioConstructor.build_weights(inputs, constraints)

    weights = result.weights
    assert not result.violations
    assert float(weights.abs().sum()) <= constraints.max_gross + 1e-8
    assert abs(float(weights.sum())) <= constraints.max_net + 1e-8
    max_name = float(weights.abs().max()) if not weights.empty else 0.0
    assert max_name <= constraints.max_name_weight + 1e-8


def test_portfolio_constructor_projects_long_short_net_exposure() -> None:
    inputs = _institutional_inputs()
    constraints = _institutional_constraints(
        beta_neutral=False,
        sector_neutral=False,
        max_gross=0.80,
        max_net=0.03,
        max_name_weight=0.30,
        use_optimizer=False,
    )

    raw = pd.Series({"A": 0.30, "B": 0.20, "C": -0.10}, dtype=float)
    projected = PortfolioConstructor()._project_net_exposure(raw, constraints, inputs.tickers)

    assert abs(float(projected.sum())) <= constraints.max_net + 1e-12
    assert float(projected.abs().sum()) <= float(raw.abs().sum()) + 1e-12
    assert float(projected.loc["C"]) == pytest.approx(float(raw.loc["C"]))


def test_short_side_mandate_allows_short_book_without_spread_net_rule() -> None:
    inputs = _institutional_inputs()
    constraints = _institutional_constraints(
        path="short_side",
        beta_neutral=False,
        sector_neutral=False,
        max_gross=0.50,
        max_net=1e-6,
        max_name_weight=0.20,
        use_optimizer=False,
    )

    result = PortfolioConstructor.build_weights(inputs, constraints)

    assert not result.violations
    assert not result.weights.empty
    assert float(result.weights.max()) <= 1e-12
    assert float(result.weights.abs().sum()) <= constraints.max_gross + 1e-8
    assert float(result.weights.sum()) < -0.10


@pytest.mark.parametrize("optimization_type", ["l1", "l2"])
def test_l1_and_non_l1_paths_use_same_constraint_engine(optimization_type: str) -> None:
    inputs = _institutional_inputs()
    constraints = _institutional_constraints(
        use_optimizer=True,
        optimization_type=optimization_type,
        beta_neutral=False,
        sector_neutral=False,
        max_gross=0.50,
        max_net=0.03,
        max_name_weight=0.15,
    )
    result = PortfolioConstructor.build_weights(inputs, constraints)

    weights = result.weights
    assert not result.violations
    assert float(weights.abs().sum()) <= constraints.max_gross + 1e-8
    assert abs(float(weights.sum())) <= constraints.max_net + 1e-8
    max_name = float(weights.abs().max()) if not weights.empty else 0.0
    assert max_name <= constraints.max_name_weight + 1e-8


def test_adaptive_portfolio_control_emits_causal_lambda_gamma_series() -> None:
    dates = pd.bdate_range("2022-01-03", periods=90)
    tickers = ["A", "B", "C", "D"]
    rows = []
    scores = {"A": 1.0, "B": 0.4, "C": -0.4, "D": -1.0}
    for i, dt in enumerate(dates):
        for ticker in tickers:
            score = scores[ticker]
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "score": score + 0.01 * np.sin(i),
                    "daily_return": 0.001 * score + 0.004 * np.sin(i / 3.0),
                    "adv_dollar_20": 100_000_000.0,
                    "realised_vol_20d": 0.025,
                    "capm_beta": 1.0,
                    "sector": "Tech" if ticker in {"A", "B"} else "Health",
                    "short_squeeze_risk": 0.0,
                    "hard_short_squeeze_filter": 0.0,
                }
            )
    scored = pd.DataFrame(rows)
    cfg = EvaluationConfig(
        path="long_short_spread",
        use_optimizer=True,
        optimization_type="l2",
        factor_neutral=False,
        beta_neutral=False,
        sector_neutral=False,
        short_squeeze_filter=False,
        adv_fraction=0.0,
        max_positions=4,
        min_positions=2,
        rebalance_every_days=5,
        lambda_risk=2.0,
        gamma_turnover=2.0,
        adaptive_control_enabled=True,
        adaptive_control_min_history_days=20,
        adaptive_control_lookback_days=60,
        adaptive_control_ema_span=5,
        adaptive_control_target_volatility=0.02,
    )

    targets = build_target_weights(scored, cfg)

    assert not targets.empty
    assert {"_lambda_risk", "_gamma_turnover", "_expected_alpha", "_expected_cost"}.issubset(targets.columns)
    adaptive_rows = targets[targets["_control_status"] == "adaptive"]
    assert not adaptive_rows.empty
    assert adaptive_rows["_lambda_risk"].between(1.0, 8.0).all()
    assert adaptive_rows["_gamma_turnover"].between(0.25, 8.0).all()
    assert float(adaptive_rows["_expected_cost"].median()) > 0.0


def test_portfolio_constructor_de_risks_infeasible_book_to_cash() -> None:
    inputs = _institutional_inputs()
    prior_violation = pd.Series({"A": 0.50}, dtype=float)
    inputs = PortfolioInputs(
        date=inputs.date,
        tickers=inputs.tickers,
        scores=inputs.scores,
        previous_weights=prior_violation,
        covariance=inputs.covariance,
        beta=inputs.beta,
        sectors=inputs.sectors,
        adv_dollar=inputs.adv_dollar,
        squeeze_risk=inputs.squeeze_risk,
        hard_short_squeeze=inputs.hard_short_squeeze,
    )
    constraints = _institutional_constraints(
        use_optimizer=False,
        beta_neutral=False,
        sector_neutral=False,
        max_net=1e-8,
        no_trade_band_total_drift=10.0,
        no_trade_band_weight_diff=10.0,
    )

    result = PortfolioConstructor.build_weights(inputs, constraints)

    assert not result.violations
    assert result.diagnostics["construction_status"] == "de_risked_to_cash"
    assert result.diagnostics["feasibility_repaired"] is True
    assert float(result.weights.abs().sum()) == pytest.approx(0.0, abs=1e-12)


def test_simulator_rejects_post_hoc_weight_neutralization() -> None:
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-03", "2022-01-03", "2022-01-04", "2022-01-04"]),
            "ticker": ["A", "B", "A", "B"],
            "score": [1.0, -1.0, 0.5, -0.5],
            "daily_return": [0.01, -0.01, 0.00, 0.00],
            "capm_beta": [1.0, 1.0, 1.0, 1.0],
            "sector": ["Tech", "Tech", "Tech", "Tech"],
            "adv_dollar_20": [100_000_000.0] * 4,
            "realised_vol_20d": [0.02] * 4,
        }
    )
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-03", "2022-01-03"]),
            "ticker": ["A", "B"],
            "target_weight": [0.40, -0.20],
        }
    )
    cfg = EvaluationConfig(
        path="long_short_spread",
        net_exposure_max=0.01,
        use_optimizer=False,
        factor_neutral=False,
        beta_neutral=False,
        sector_neutral=False,
        adv_fraction=0.0,
        short_squeeze_filter=False,
    )

    with pytest.raises(MetricIntegrityError, match="non-neutral target weights"):
        simulate_executable_portfolio(scored, cfg, target_weights=target_weights)


def test_simulator_preserves_preconstructed_weights() -> None:
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-03", "2022-01-03", "2022-01-04", "2022-01-04"]),
            "ticker": ["A", "B", "A", "B"],
            "score": [1.0, -1.0, 0.5, -0.5],
            "daily_return": [0.01, -0.01, 0.00, 0.00],
            "capm_beta": [2.0, 1.0, 2.0, 1.0],
            "sector": ["Tech", "Tech", "Tech", "Tech"],
            "adv_dollar_20": [100_000_000.0] * 4,
            "realised_vol_20d": [0.02] * 4,
        }
    )
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-03", "2022-01-03"]),
            "ticker": ["A", "B"],
            "target_weight": [0.30, -0.30],
        }
    )
    target_snapshot = target_weights.copy(deep=True)
    cfg = EvaluationConfig(
        path="long_short_spread",
        net_exposure_max=0.01,
        use_optimizer=False,
        factor_neutral=False,
        beta_neutral=False,
        sector_neutral=False,
        adv_fraction=0.0,
        short_squeeze_filter=False,
        costs=EvaluationConfig().costs.__class__(
            commission_bps=0.0,
            spread_bps=0.0,
            borrow_bps=0.0,
            impact_eta=0.0,
        ),
    )

    _, pnl = simulate_executable_portfolio(scored, cfg, target_weights=target_weights)

    pd.testing.assert_frame_equal(target_weights, target_snapshot)
    assert not pnl.empty
    assert pnl.loc[0, "net_exposure"] == pytest.approx(0.0, abs=1e-12)
    assert abs(float(pnl.loc[0, "beta_exposure"])) > 0.0


def test_build_target_weights_canonicalizes_duplicate_scored_rows() -> None:
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2022-01-03", "2022-01-03", "2022-01-03", "2022-01-03", "2022-01-04", "2022-01-04"]
            ),
            "ticker": ["A", "A", "B", "B", "A", "B"],
            "score": [1.0, 1.0, -1.0, -1.0, 0.5, -0.5],
            "daily_return": [0.01, 0.01, -0.01, -0.01, 0.00, 0.00],
            "capm_beta": [1.0] * 6,
            "sector": ["Tech"] * 6,
            "adv_dollar_20": [100_000_000.0] * 6,
            "realised_vol_20d": [0.02] * 6,
        }
    )
    cfg = EvaluationConfig(
        path="long_short_spread",
        net_exposure_max=0.01,
        use_optimizer=False,
        factor_neutral=False,
        beta_neutral=False,
        sector_neutral=False,
        adv_fraction=0.0,
        short_squeeze_filter=False,
    )

    target_weights = build_target_weights(scored, cfg)

    assert not target_weights.duplicated(["date", "ticker"]).any()
    first_day_net = target_weights.loc[target_weights["date"].eq(pd.Timestamp("2022-01-03")), "target_weight"].sum()
    assert first_day_net == pytest.approx(0.0, abs=1e-12)


def test_simulator_rejects_duplicate_preconstructed_target_rows() -> None:
    scored = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-03", "2022-01-03", "2022-01-04", "2022-01-04"]),
            "ticker": ["A", "B", "A", "B"],
            "score": [1.0, -1.0, 0.5, -0.5],
            "daily_return": [0.01, -0.01, 0.00, 0.00],
        }
    )
    target_weights = pd.DataFrame(
        {
            "date": pd.to_datetime(["2022-01-03", "2022-01-03", "2022-01-03"]),
            "ticker": ["A", "A", "B"],
            "target_weight": [0.25, 0.25, -0.50],
        }
    )
    cfg = EvaluationConfig(
        path="long_short_spread",
        net_exposure_max=0.01,
        use_optimizer=False,
        factor_neutral=False,
        beta_neutral=False,
        sector_neutral=False,
        adv_fraction=0.0,
        short_squeeze_filter=False,
    )

    with pytest.raises(MetricIntegrityError, match="unique by date/ticker"):
        simulate_executable_portfolio(scored, cfg, target_weights=target_weights)


def test_zero_book_simulation_returns_finite_zero_ledger() -> None:
    dates = pd.bdate_range("2022-01-03", periods=20)
    scored = pd.DataFrame(
        {
            "date": np.repeat(dates, 2),
            "ticker": ["A", "B"] * len(dates),
            "score": [1.0, -1.0] * len(dates),
            "daily_return": [0.001, -0.001] * len(dates),
            "capm_beta": [1.0, 1.0] * len(dates),
            "sector": ["Tech", "Tech"] * len(dates),
            "adv_dollar_20": [100_000_000.0] * (2 * len(dates)),
            "realised_vol_20d": [0.02] * (2 * len(dates)),
        }
    )
    target_weights = pd.DataFrame(columns=["date", "ticker", "target_weight"])
    cfg = EvaluationConfig(
        path="long_short_spread",
        use_optimizer=False,
        factor_neutral=False,
        beta_neutral=False,
        sector_neutral=False,
    )

    returns, pnl = simulate_executable_portfolio(scored, cfg, target_weights=target_weights)

    assert len(returns) > 0
    assert not pnl.empty
    assert float(pnl["gross_exposure"].sum()) == pytest.approx(0.0)
    assert float(pnl["trade_count"].sum()) == pytest.approx(0.0)
    assert annualized_sharpe(returns) == pytest.approx(0.0)
