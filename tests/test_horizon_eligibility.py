"""
Tests for Horizon Eligibility and Gate — Hardened.
"""

import numpy as np
import pandas as pd
import pytest

from model_selection.horizon_eligibility import (
    HorizonEligibilityContract,
    compute_eligibility,
    compute_all_eligibility,
    estimate_halflife,
    estimate_halflife_from_persistence,
    compute_ic_at_horizon,
    compute_rank_persistence_curve,
    compute_cost_viability,
    compute_effective_signal_count,
    HALFLIFE_TOO_SHORT,
    IC_TOO_WEAK,
    IC_NEGATIVE,
    COST_NOT_VIABLE,
    MISSING_COST_DATA,
)
from model_selection.horizon_gate import (
    HorizonGate,
    HorizonGateConfig,
    HorizonGateResult,
    HorizonIneligibleError,
    filter_eligible_features,
)


def _make_panel(n_dates=200, n_tickers=50, seed=42):
    """Create a synthetic panel for testing."""
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    rows = []
    for date in dates:
        for ticker in tickers:
            rows.append({
                "date": date,
                "ticker": ticker,
                "daily_return": rng.normal(0.0005, 0.02),
                "good_feature": rng.normal(0, 1),
                "bad_feature": rng.normal(0, 1),
                "negative_feature": rng.normal(0, 1),
            })

    df = pd.DataFrame(rows)

    # Make good_feature predictive: add signal to future returns
    for i in range(len(df)):
        date_idx = i // n_tickers
        ticker_idx = i % n_tickers
        if date_idx < n_dates - 5:
            signal = df.iloc[i]["good_feature"] * 0.01
            future_idx = (date_idx + 5) * n_tickers + ticker_idx
            if future_idx < len(df):
                df.iloc[future_idx, df.columns.get_loc("daily_return")] += signal * 0.2

    return df


class TestEstimateHalflife:
    def test_fast_decay(self):
        ic = pd.Series([0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005])
        hl = estimate_halflife(ic)
        assert hl < 3

    def test_slow_decay(self):
        ic_slow = pd.Series([0.1, 0.0995, 0.099, 0.0985, 0.098, 0.0975, 0.097, 0.0965, 0.096, 0.0955,
                             0.095, 0.0945, 0.094, 0.0935, 0.093, 0.0925, 0.092, 0.0915, 0.091, 0.0905])
        ic_fast = pd.Series([0.1, 0.08, 0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001, 0.0005])
        hl_slow = estimate_halflife(ic_slow)
        hl_fast = estimate_halflife(ic_fast)
        assert hl_slow > hl_fast

    def test_insufficient_data(self):
        ic = pd.Series([0.1, 0.08])
        hl = estimate_halflife(ic)
        assert np.isnan(hl)


class TestRankPersistence:
    def test_persistence_curve_computed(self):
        df = _make_panel(n_dates=100, n_tickers=30)
        persistence = compute_rank_persistence_curve(df, "good_feature", max_lag=5)
        assert isinstance(persistence, dict)
        assert len(persistence) > 0

    def test_halflife_from_persistence(self):
        persistence = {1: 0.9, 2: 0.7, 3: 0.55, 4: 0.4, 5: 0.25}
        hl = estimate_halflife_from_persistence(persistence)
        assert 2.5 < hl < 4.0

    def test_halflife_from_persistence_empty(self):
        assert np.isnan(estimate_halflife_from_persistence({}))

    def test_halflife_from_persistence_always_above(self):
        persistence = {1: 0.9, 2: 0.8, 3: 0.7}
        hl = estimate_halflife_from_persistence(persistence)
        assert hl == 3.0

    def test_halflife_from_persistence_always_below(self):
        persistence = {1: 0.3, 2: 0.2, 3: 0.1}
        hl = estimate_halflife_from_persistence(persistence)
        assert hl == 1.0


class TestCostViability:
    def test_viable_strong_ic(self):
        assert compute_cost_viability(0.03, 20, cost_bps=10.0) is True

    def test_not_viable_weak_ic(self):
        assert compute_cost_viability(0.001, 10, cost_bps=10.0) is False

    def test_nan_ic_fails_closed(self):
        assert compute_cost_viability(np.nan, 10) is False

    def test_zero_ic_fails_closed(self):
        assert compute_cost_viability(0.0, 10) is False


class TestEffectiveSignalCount:
    def test_independent_signals(self):
        corr = np.eye(4)
        pr = compute_effective_signal_count(corr)
        assert abs(pr - 4.0) < 0.01

    def test_identical_signals(self):
        corr = np.ones((4, 4))
        pr = compute_effective_signal_count(corr)
        assert abs(pr - 1.0) < 0.01

    def test_empty_matrix(self):
        assert compute_effective_signal_count(None) == 0.0
        assert compute_effective_signal_count(np.array([])) == 0.0


class TestComputeEligibility:
    def test_contract_has_split_admissibility(self):
        df = _make_panel(n_dates=50, n_tickers=30)
        contract = compute_eligibility(df, "good_feature", "test", horizons=[5, 10])
        assert hasattr(contract, "statistically_admissible_horizons")
        assert hasattr(contract, "production_admissible_horizons")
        assert hasattr(contract, "statistical_rejections")
        assert hasattr(contract, "production_rejections")

    def test_missing_feature(self):
        df = _make_panel(n_dates=50, n_tickers=30)
        contract = compute_eligibility(df, "nonexistent", "test", horizons=[5])
        assert contract.feature == "nonexistent"
        assert not contract.statistically_admissible_horizons
        assert not contract.production_admissible_horizons

    def test_production_subset_of_statistical(self):
        df = _make_panel(n_dates=200, n_tickers=50)
        contract = compute_eligibility(df, "good_feature", "test", horizons=[1, 2, 3, 5, 10])
        prod = set(contract.production_admissible_horizons)
        stat = set(contract.statistically_admissible_horizons)
        assert prod.issubset(stat)

    def test_cost_viability_stored(self):
        df = _make_panel(n_dates=200, n_tickers=50)
        contract = compute_eligibility(df, "good_feature", "test", horizons=[5, 10])
        assert isinstance(contract.cost_adjusted_viable, dict)
        assert len(contract.cost_adjusted_viable) > 0


class TestHorizonGate:
    def test_eligible_feature_passes(self):
        contract = HorizonEligibilityContract(
            feature="good", family="test",
            statistically_admissible_horizons=[5, 10],
            production_admissible_horizons=[5, 10],
            production_rejections={},
            statistical_rejections={},
            estimated_halflife=6.0,
        )
        config = HorizonGateConfig(min_production_features=1, min_families=1, min_effective_signals=0.5)
        gate = HorizonGate({"good": contract}, config=config)
        result = gate.evaluate(5)
        assert "good" in result.eligible_features
        assert not result.block_horizon

    def test_ineligible_feature_blocked(self):
        contract = HorizonEligibilityContract(
            feature="bad", family="test",
            statistically_admissible_horizons=[],
            production_admissible_horizons=[],
            production_rejections={10: "HALFLIFE_TOO_SHORT"},
            statistical_rejections={10: "HALFLIFE_TOO_SHORT"},
            estimated_halflife=1.0,
        )
        config = HorizonGateConfig(min_production_features=1, min_families=1, min_effective_signals=0.5)
        gate = HorizonGate({"bad": contract}, config=config)
        result = gate.evaluate(10)
        assert "bad" not in result.eligible_features
        assert "bad" in result.rejected_features

    def test_horizon_blocked_when_zero_eligible(self):
        contracts = {
            "f1": HorizonEligibilityContract(
                feature="f1", family="test",
                statistically_admissible_horizons=[5],
                production_admissible_horizons=[5],
                production_rejections={10: "HALFLIFE_TOO_SHORT"},
                statistical_rejections={10: "HALFLIFE_TOO_SHORT"},
                estimated_halflife=2.0,
            ),
            "f2": HorizonEligibilityContract(
                feature="f2", family="test",
                statistically_admissible_horizons=[5],
                production_admissible_horizons=[5],
                production_rejections={10: "HALFLIFE_TOO_SHORT"},
                statistical_rejections={10: "HALFLIFE_TOO_SHORT"},
                estimated_halflife=1.5,
            ),
        }
        config = HorizonGateConfig(min_production_features=2, min_families=1, min_effective_signals=0.5)
        gate = HorizonGate(contracts, config=config)
        result = gate.evaluate(10)
        assert result.block_horizon
        assert len(result.eligible_features) == 0

    def test_family_concentration_blocks(self):
        contracts = {
            "f1": HorizonEligibilityContract(
                feature="f1", family="momentum",
                statistically_admissible_horizons=[10],
                production_admissible_horizons=[10],
                production_rejections={},
                statistical_rejections={},
                estimated_halflife=12.0,
            ),
            "f2": HorizonEligibilityContract(
                feature="f2", family="momentum",
                statistically_admissible_horizons=[10],
                production_admissible_horizons=[10],
                production_rejections={},
                statistical_rejections={},
                estimated_halflife=12.0,
            ),
            "f3": HorizonEligibilityContract(
                feature="f3", family="momentum",
                statistically_admissible_horizons=[10],
                production_admissible_horizons=[10],
                production_rejections={},
                statistical_rejections={},
                estimated_halflife=12.0,
            ),
        }
        config = HorizonGateConfig(
            min_production_features=3, min_families=2,
            max_family_concentration=0.6, min_effective_signals=0.5
        )
        gate = HorizonGate(contracts, config=config)
        result = gate.evaluate(10)
        assert result.block_horizon
        assert any("FAMILY_CONCENTRATION" in r for r in result.block_reasons)

    def test_insufficient_families_blocks(self):
        contracts = {
            "f1": HorizonEligibilityContract(
                feature="f1", family="momentum",
                statistically_admissible_horizons=[10],
                production_admissible_horizons=[10],
                production_rejections={},
                statistical_rejections={},
                estimated_halflife=12.0,
            ),
        }
        config = HorizonGateConfig(min_production_features=1, min_families=2, min_effective_signals=0.5)
        gate = HorizonGate(contracts, config=config)
        result = gate.evaluate(10)
        assert result.block_horizon
        assert any("INSUFFICIENT_FAMILIES" in r for r in result.block_reasons)

    def test_filter_eligible_features(self):
        contracts = {
            "f1": HorizonEligibilityContract(
                feature="f1", family="test",
                statistically_admissible_horizons=[5, 10],
                production_admissible_horizons=[5, 10],
                production_rejections={},
                statistical_rejections={},
                estimated_halflife=6.0,
            ),
            "f2": HorizonEligibilityContract(
                feature="f2", family="test",
                statistically_admissible_horizons=[5],
                production_admissible_horizons=[5],
                production_rejections={10: "HALFLIFE_TOO_SHORT"},
                statistical_rejections={10: "HALFLIFE_TOO_SHORT"},
                estimated_halflife=2.0,
            ),
        }
        eligible = filter_eligible_features(["f1", "f2", "f3"], contracts, horizon=5)
        assert "f1" in eligible
        assert "f2" in eligible
        assert "f3" not in eligible

        eligible = filter_eligible_features(["f1", "f2", "f3"], contracts, horizon=10)
        assert "f1" in eligible
        assert "f2" not in eligible

    def test_horizon_ineligible_error(self):
        err = HorizonIneligibleError(horizon=10, n_features=0, min_required=2, reasons=["INSUFFICIENT_FEATURES"])
        assert "h10d" in str(err)
        assert "0" in str(err)
        assert "2" in str(err)
        assert "INSUFFICIENT_FEATURES" in str(err)

    def test_gate_uses_config(self):
        contract = HorizonEligibilityContract(
            feature="f1", family="test",
            statistically_admissible_horizons=[5, 10],
            production_admissible_horizons=[5, 10],
            production_rejections={},
            statistical_rejections={},
            estimated_halflife=8.0,
        )
        config = HorizonGateConfig(min_production_features=2, min_families=1, min_effective_signals=0.5)
        gate = HorizonGate({"f1": contract}, config=config)
        result = gate.evaluate(5)
        assert result.block_horizon
        assert result.config.min_production_features == 2

    def test_statistical_vs_production_level(self):
        contract = HorizonEligibilityContract(
            feature="f1", family="test",
            statistically_admissible_horizons=[5, 10],
            production_admissible_horizons=[5],
            production_rejections={10: "COST_NOT_VIABLE"},
            statistical_rejections={},
            estimated_halflife=8.0,
            cost_adjusted_viable={5: True, 10: False},
        )
        config_prod = HorizonGateConfig(
            min_production_features=1, min_families=1,
            min_effective_signals=0.5, use_production_level=True
        )
        config_stat = HorizonGateConfig(
            min_production_features=1, min_families=1,
            min_effective_signals=0.5, use_production_level=False
        )

        gate_prod = HorizonGate({"f1": contract}, config=config_prod)
        gate_stat = HorizonGate({"f1": contract}, config=config_stat)

        result_prod = gate_prod.evaluate(10)
        result_stat = gate_stat.evaluate(10)

        assert result_prod.block_horizon
        assert not result_stat.block_horizon


class TestIntegration:
    def test_ineligible_features_cannot_enter_training(self):
        contracts = {
            "f1": HorizonEligibilityContract(
                feature="f1", family="test",
                statistically_admissible_horizons=[5, 10],
                production_admissible_horizons=[5, 10],
                production_rejections={},
                statistical_rejections={},
                estimated_halflife=8.0,
                ic_by_horizon={5: 0.02, 10: 0.015},
            ),
            "f2": HorizonEligibilityContract(
                feature="f2", family="test",
                statistically_admissible_horizons=[5],
                production_admissible_horizons=[5],
                production_rejections={10: "HALFLIFE_TOO_SHORT"},
                statistical_rejections={10: "HALFLIFE_TOO_SHORT"},
                estimated_halflife=1.0,
                ic_by_horizon={5: 0.02, 10: 0.01},
            ),
            "f3": HorizonEligibilityContract(
                feature="f3", family="test",
                statistically_admissible_horizons=[],
                production_admissible_horizons=[],
                production_rejections={10: "IC_NEGATIVE"},
                statistical_rejections={10: "IC_NEGATIVE"},
                estimated_halflife=3.0,
                ic_by_horizon={5: -0.01, 10: -0.02},
            ),
        }

        config = HorizonGateConfig(min_production_features=1, min_families=1, min_effective_signals=0.5)
        gate = HorizonGate(contracts, config=config)
        result = gate.evaluate(10)

        assert "f1" in result.eligible_features
        assert "f2" not in result.eligible_features
        assert "f3" not in result.eligible_features
        assert result.rejected_features["f2"] == "HALFLIFE_TOO_SHORT"
        assert result.rejected_features["f3"] == "IC_NEGATIVE"
        assert not result.block_horizon

        all_features = ["f1", "f2", "f3"]
        training_features = filter_eligible_features(all_features, contracts, horizon=10)
        assert training_features == ["f1"]

    def test_horizon_blocked_when_no_eligible(self):
        contracts = {
            "f1": HorizonEligibilityContract(
                feature="f1", family="test",
                statistically_admissible_horizons=[5],
                production_admissible_horizons=[5],
                production_rejections={20: "HALFLIFE_TOO_SHORT"},
                statistical_rejections={20: "HALFLIFE_TOO_SHORT"},
                estimated_halflife=2.0,
            ),
            "f2": HorizonEligibilityContract(
                feature="f2", family="test",
                statistically_admissible_horizons=[5],
                production_admissible_horizons=[5],
                production_rejections={20: "HALFLIFE_TOO_SHORT"},
                statistical_rejections={20: "HALFLIFE_TOO_SHORT"},
                estimated_halflife=1.5,
            ),
        }

        config = HorizonGateConfig(min_production_features=2, min_families=1, min_effective_signals=0.5)
        gate = HorizonGate(contracts, config=config)
        result = gate.evaluate(20)

        assert result.block_horizon
        assert len(result.eligible_features) == 0

        training_features = filter_eligible_features(["f1", "f2"], contracts, horizon=20)
        assert training_features == []

        with pytest.raises(HorizonIneligibleError):
            if result.block_horizon:
                raise HorizonIneligibleError(
                    horizon=20,
                    n_features=len(result.eligible_features),
                    min_required=result.config.min_production_features,
                    reasons=result.block_reasons,
                )

    def test_no_hardcoded_thresholds_in_gate(self):
        contract = HorizonEligibilityContract(
            feature="f1", family="test",
            statistically_admissible_horizons=[3, 5],
            production_admissible_horizons=[3, 5],
            production_rejections={10: "HALFLIFE_TOO_SHORT", 20: "HALFLIFE_TOO_SHORT"},
            statistical_rejections={10: "HALFLIFE_TOO_SHORT", 20: "HALFLIFE_TOO_SHORT"},
            estimated_halflife=3.0,
        )

        config = HorizonGateConfig(min_production_features=1, min_families=1, min_effective_signals=0.5)
        gate = HorizonGate({"f1": contract}, config=config)

        result_3 = gate.evaluate(3)
        assert not result_3.block_horizon
        assert "f1" in result_3.eligible_features

        result_5 = gate.evaluate(5)
        assert not result_5.block_horizon
        assert "f1" in result_5.eligible_features

        result_10 = gate.evaluate(10)
        assert result_10.block_horizon
        assert "f1" not in result_10.eligible_features

        result_20 = gate.evaluate(20)
        assert result_20.block_horizon
        assert "f1" not in result_20.eligible_features


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
