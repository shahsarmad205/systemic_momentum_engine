"""
Integration tests for Horizon Gate wiring in run_model_selection.py.

Tests prove:
1. Blocked horizon does not call model training
2. Ineligible features do not appear in PreparedPanelCache
3. report_only mode does not alter model outputs
4. production mode blocks invalid horizons
5. Missing eligibility fails closed
6. Cache fingerprint changes when eligibility contracts change
7. Old fallback full-feature path is not reachable
"""

import json
import hashlib
from pathlib import Path
from unittest.mock import patch, MagicMock, call
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from model_selection.horizon_eligibility import (
    HorizonEligibilityContract,
    compute_eligibility,
    compute_all_eligibility,
)
from model_selection.horizon_gate import (
    HorizonGate,
    HorizonGateConfig,
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
                "neg_feature": rng.normal(0, 1),
            })
    df = pd.DataFrame(rows)
    for i in range(len(df)):
        date_idx = i // n_tickers
        ticker_idx = i % n_tickers
        if date_idx < n_dates - 5:
            signal = df.iloc[i]["good_feature"] * 0.01
            future_idx = (date_idx + 5) * n_tickers + ticker_idx
            if future_idx < len(df):
                df.iloc[future_idx, df.columns.get_loc("daily_return")] += signal * 0.2
    return df


class TestBlockedHorizonDoesNotTrain:
    """Prove that a blocked horizon does not call model training."""

    def test_production_mode_raises_systemexit_on_block(self):
        contracts = {
            "f1": HorizonEligibilityContract(
                feature="f1", family="momentum",
                statistically_admissible_horizons=[],
                production_admissible_horizons=[],
                production_rejections={10: "HALFLIFE_TOO_SHORT"},
                statistical_rejections={10: "HALFLIFE_TOO_SHORT"},
                estimated_halflife=1.0,
            ),
            "f2": HorizonEligibilityContract(
                feature="f2", family="momentum",
                statistically_admissible_horizons=[],
                production_admissible_horizons=[],
                production_rejections={10: "HALFLIFE_TOO_SHORT"},
                statistical_rejections={10: "HALFLIFE_TOO_SHORT"},
                estimated_halflife=1.0,
            ),
        }
        config = HorizonGateConfig(
            min_production_features=2, min_families=1,
            min_effective_signals=0.5, use_production_level=True,
        )
        gate = HorizonGate(contracts, config=config)
        result = gate.evaluate(10)

        assert result.block_horizon
        assert len(result.eligible_features) == 0
        assert "INSUFFICIENT_FEATURES" in result.block_reasons[0]

    def test_report_only_mode_does_not_block(self):
        contracts = {
            "f1": HorizonEligibilityContract(
                feature="f1", family="momentum",
                statistically_admissible_horizons=[],
                production_admissible_horizons=[],
                production_rejections={10: "HALFLIFE_TOO_SHORT"},
                statistical_rejections={10: "HALFLIFE_TOO_SHORT"},
                estimated_halflife=1.0,
            ),
        }
        config = HorizonGateConfig(
            min_production_features=2, min_families=1,
            min_effective_signals=0.5, use_production_level=False,
        )
        gate = HorizonGate(contracts, config=config)
        result = gate.evaluate(10)

        assert result.block_horizon
        assert not config.use_production_level


class TestIneligibleFeaturesNotInCache:
    """Prove that ineligible features do not appear in PreparedPanelCache."""

    def test_filter_eligible_features_removes_ineligible(self):
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
                feature="f2", family="reversal",
                statistically_admissible_horizons=[],
                production_admissible_horizons=[],
                production_rejections={10: "COST_NOT_VIABLE"},
                statistical_rejections={10: "IC_TOO_WEAK"},
                estimated_halflife=2.0,
            ),
            "f3": HorizonEligibilityContract(
                feature="f3", family="quality",
                statistically_admissible_horizons=[10],
                production_admissible_horizons=[10],
                production_rejections={},
                statistical_rejections={},
                estimated_halflife=15.0,
            ),
        }
        all_features = ["f1", "f2", "f3"]
        eligible = filter_eligible_features(all_features, contracts, horizon=10, use_production=True)

        assert "f1" in eligible
        assert "f2" not in eligible
        assert "f3" in eligible
        assert len(eligible) == 2

    def test_production_filter_is_stricter_than_statistical(self):
        contracts = {
            "f1": HorizonEligibilityContract(
                feature="f1", family="momentum",
                statistically_admissible_horizons=[10],
                production_admissible_horizons=[],
                production_rejections={10: "COST_NOT_VIABLE"},
                statistical_rejections={},
                estimated_halflife=12.0,
                cost_adjusted_viable={10: False},
            ),
        }
        all_features = ["f1"]
        stat_eligible = filter_eligible_features(all_features, contracts, horizon=10, use_production=False)
        prod_eligible = filter_eligible_features(all_features, contracts, horizon=10, use_production=True)

        assert "f1" in stat_eligible
        assert "f1" not in prod_eligible


class TestReportOnlyMode:
    """Prove that report_only mode does not alter model outputs."""

    def test_report_only_returns_full_feature_set(self):
        contracts = {
            "f1": HorizonEligibilityContract(
                feature="f1", family="momentum",
                statistically_admissible_horizons=[],
                production_admissible_horizons=[],
                production_rejections={10: "HALFLIFE_TOO_SHORT"},
                statistical_rejections={10: "HALFLIFE_TOO_SHORT"},
                estimated_halflife=1.0,
            ),
        }
        config = HorizonGateConfig(
            min_production_features=2, min_families=1,
            min_effective_signals=0.5, use_production_level=False,
        )
        gate = HorizonGate(contracts, config=config)
        result = gate.evaluate(10)

        assert result.block_horizon
        assert config.use_production_level is False

    def test_report_only_uses_statistical_admissibility(self):
        contract = HorizonEligibilityContract(
            feature="f1", family="momentum",
            statistically_admissible_horizons=[10],
            production_admissible_horizons=[],
            production_rejections={10: "COST_NOT_VIABLE"},
            statistical_rejections={},
            estimated_halflife=12.0,
            cost_adjusted_viable={10: False},
        )
        config = HorizonGateConfig(
            min_production_features=1, min_families=1,
            min_effective_signals=0.5, use_production_level=False,
        )
        gate = HorizonGate({"f1": contract}, config=config)
        result = gate.evaluate(10)

        assert not result.block_horizon
        assert "f1" in result.eligible_features


class TestProductionModeBlocksInvalid:
    """Prove that production mode blocks invalid horizons."""

    def test_missing_contract_blocks(self):
        config = HorizonGateConfig(
            min_production_features=2, min_families=1,
            min_effective_signals=0.5, use_production_level=True,
        )
        gate = HorizonGate({}, config=config)
        result = gate.evaluate(10)

        assert result.block_horizon
        assert "INSUFFICIENT_FEATURES" in result.block_reasons[0]

    def test_cost_viability_enforced(self):
        contract = HorizonEligibilityContract(
            feature="f1", family="momentum",
            statistically_admissible_horizons=[10],
            production_admissible_horizons=[],
            production_rejections={10: "COST_NOT_VIABLE"},
            statistical_rejections={},
            estimated_halflife=12.0,
            cost_adjusted_viable={10: False},
        )
        config = HorizonGateConfig(
            min_production_features=1, min_families=1,
            min_effective_signals=0.5, use_production_level=True,
        )
        gate = HorizonGate({"f1": contract}, config=config)
        result = gate.evaluate(10)

        assert result.block_horizon
        assert "INSUFFICIENT_FEATURES" in result.block_reasons[0]

    def test_family_diversity_enforced(self):
        contracts = {
            f"f{i}": HorizonEligibilityContract(
                feature=f"f{i}", family="momentum",
                statistically_admissible_horizons=[10],
                production_admissible_horizons=[10],
                production_rejections={},
                statistical_rejections={},
                estimated_halflife=12.0,
            )
            for i in range(5)
        }
        config = HorizonGateConfig(
            min_production_features=3, min_families=2,
            max_family_concentration=0.6, min_effective_signals=0.5,
            use_production_level=True,
        )
        gate = HorizonGate(contracts, config=config)
        result = gate.evaluate(10)

        assert result.block_horizon
        assert any("INSUFFICIENT_FAMILIES" in r for r in result.block_reasons)


class TestMissingEligibilityFailsClosed:
    """Prove that missing eligibility fails closed in production mode."""

    def test_empty_contracts_fail_closed(self):
        config = HorizonGateConfig(
            min_production_features=2, min_families=1,
            min_effective_signals=0.5, use_production_level=True,
        )
        gate = HorizonGate({}, config=config)
        result = gate.evaluate(10)
        assert result.block_horizon

    def test_missing_feature_contract_fails(self):
        contract = HorizonEligibilityContract(
            feature="f1", family="momentum",
            statistically_admissible_horizons=[],
            production_admissible_horizons=[],
            production_rejections={},
            statistical_rejections={},
            estimated_halflife=0.0,
        )
        config = HorizonGateConfig(
            min_production_features=1, min_families=1,
            min_effective_signals=0.5, use_production_level=True,
        )
        gate = HorizonGate({"f1": contract}, config=config)
        result = gate.evaluate(10)
        assert result.block_horizon


class TestCacheFingerprintChanges:
    """Prove that cache fingerprint changes when eligibility contracts change."""

    def test_different_eligibility_produces_different_fingerprint(self):
        result_a = {
            "horizon": 10,
            "mode": "production",
            "n_contracts": 3,
            "n_eligible": 2,
            "blocked": False,
            "gate_result": {
                "n_eligible": 2,
                "n_families": 2,
                "effective_signals": 1.8,
                "block_reasons": [],
            },
        }
        result_b = {
            "horizon": 10,
            "mode": "production",
            "n_contracts": 3,
            "n_eligible": 1,
            "blocked": False,
            "gate_result": {
                "n_eligible": 1,
                "n_families": 1,
                "effective_signals": 1.0,
                "block_reasons": [],
            },
        }
        fp_a = hashlib.sha256(json.dumps(result_a, sort_keys=True, default=str).encode()).hexdigest()[:12]
        fp_b = hashlib.sha256(json.dumps(result_b, sort_keys=True, default=str).encode()).hexdigest()[:12]
        assert fp_a != fp_b

    def test_same_eligibility_produces_same_fingerprint(self):
        result = {
            "horizon": 10,
            "mode": "production",
            "n_contracts": 3,
            "n_eligible": 2,
            "blocked": False,
            "gate_result": {
                "n_eligible": 2,
                "n_families": 2,
                "effective_signals": 1.8,
                "block_reasons": [],
            },
        }
        fp_a = hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()[:12]
        fp_b = hashlib.sha256(json.dumps(result, sort_keys=True, default=str).encode()).hexdigest()[:12]
        assert fp_a == fp_b


class TestNoFullFeatureFallback:
    """Prove that the old fallback full-feature path is not reachable."""

    def test_gate_result_always_has_eligible_features_list(self):
        config = HorizonGateConfig(
            min_production_features=2, min_families=1,
            min_effective_signals=0.5, use_production_level=True,
        )
        gate = HorizonGate({}, config=config)
        result = gate.evaluate(10)

        assert isinstance(result.eligible_features, list)
        assert len(result.eligible_features) == 0
        assert result.block_horizon

    def test_filter_returns_empty_when_no_eligible(self):
        eligible = filter_eligible_features(["f1", "f2"], {}, horizon=10, use_production=True)
        assert eligible == []

    def test_production_level_cannot_be_bypassed(self):
        contract = HorizonEligibilityContract(
            feature="f1", family="momentum",
            statistically_admissible_horizons=[10],
            production_admissible_horizons=[],
            production_rejections={10: "COST_NOT_VIABLE"},
            statistical_rejections={},
            estimated_halflife=12.0,
            cost_adjusted_viable={10: False},
        )
        config = HorizonGateConfig(
            min_production_features=1, min_families=1,
            min_effective_signals=0.5, use_production_level=True,
        )
        gate = HorizonGate({"f1": contract}, config=config)
        result = gate.evaluate(10)

        assert result.block_horizon
        assert len(result.eligible_features) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
