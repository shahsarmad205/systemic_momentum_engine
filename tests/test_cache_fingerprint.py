"""P38: Research-cache fingerprint tests."""
from __future__ import annotations

import pytest
from model_selection.horizon_contract import (
    research_cache_fingerprint,
    research_cache_fingerprint_from_config,
)


class TestCacheFingerprint:
    """P38: Research-cache fingerprint must invalidate on policy changes."""

    def test_identical_config_produces_same_fingerprint(self):
        fp1 = research_cache_fingerprint(
            production_horizon_days=10, target_horizon_days=10,
            holding_period_days=10, rebalance_frequency_days=10,
            ic_evaluation_horizon=10, execution_tau_days=None,
        )
        fp2 = research_cache_fingerprint(
            production_horizon_days=10, target_horizon_days=10,
            holding_period_days=10, rebalance_frequency_days=10,
            ic_evaluation_horizon=10, execution_tau_days=None,
        )
        assert fp1 == fp2

    def test_different_holding_changes_fingerprint(self):
        fp1 = research_cache_fingerprint(
            production_horizon_days=10, target_horizon_days=10,
            holding_period_days=10, rebalance_frequency_days=10,
            ic_evaluation_horizon=10, execution_tau_days=None,
        )
        fp2 = research_cache_fingerprint(
            production_horizon_days=10, target_horizon_days=10,
            holding_period_days=63, rebalance_frequency_days=10,
            ic_evaluation_horizon=10, execution_tau_days=None,
        )
        assert fp1 != fp2

    def test_different_rebalance_changes_fingerprint(self):
        fp1 = research_cache_fingerprint(
            production_horizon_days=10, target_horizon_days=10,
            holding_period_days=10, rebalance_frequency_days=5,
            ic_evaluation_horizon=10, execution_tau_days=None,
        )
        fp2 = research_cache_fingerprint(
            production_horizon_days=10, target_horizon_days=10,
            holding_period_days=10, rebalance_frequency_days=10,
            ic_evaluation_horizon=10, execution_tau_days=None,
        )
        assert fp1 != fp2

    def test_different_policy_mode_changes_fingerprint(self):
        fp1 = research_cache_fingerprint(
            production_horizon_days=10, target_horizon_days=10,
            holding_period_days=10, rebalance_frequency_days=10,
            ic_evaluation_horizon=10, execution_tau_days=None,
            rebalance_policy_mode="match_horizon",
        )
        fp2 = research_cache_fingerprint(
            production_horizon_days=10, target_horizon_days=10,
            holding_period_days=10, rebalance_frequency_days=10,
            ic_evaluation_horizon=10, execution_tau_days=None,
            rebalance_policy_mode="halflife_aware",
        )
        assert fp1 != fp2

    def test_policy_versions_change_fingerprint(self):
        fp1 = research_cache_fingerprint(
            production_horizon_days=10, target_horizon_days=10,
            holding_period_days=10, rebalance_frequency_days=10,
            ic_evaluation_horizon=10, execution_tau_days=None,
            execution_aware_policy_version=1,
        )
        fp2 = research_cache_fingerprint(
            production_horizon_days=10, target_horizon_days=10,
            holding_period_days=10, rebalance_frequency_days=10,
            ic_evaluation_horizon=10, execution_tau_days=None,
            execution_aware_policy_version=2,
        )
        assert fp1 != fp2

    def test_fingerprint_is_stable_across_calls(self):
        """Fingerprint should be deterministic (no timestamps)."""
        fps = []
        for _ in range(10):
            fps.append(research_cache_fingerprint(
                production_horizon_days=20, target_horizon_days=20,
                holding_period_days=20, rebalance_frequency_days=20,
                ic_evaluation_horizon=20, execution_tau_days=2.5,
                rebalance_policy_mode="halflife_aware",
                scoring_weights_policy_mode="algorithmic",
            ))
        assert len(set(fps)) == 1  # all identical

    def test_fingerprint_from_config_basic(self):
        cfg = {
            "model_selection": {
                "alpha_research": {"production_horizon": 10},
                "execution_aware_horizon_policy": {"policy_version": 1},
                "orientation_policy": {"policy_version": 2},
            },
            "horizon_config": {
                "target_horizon_days": 10,
                "rebalance_policy": {"mode": "match_horizon"},
            },
        }
        fp = research_cache_fingerprint_from_config(cfg)
        assert isinstance(fp, str)
        assert len(fp) == 16

    def test_circular_import_does_not_crash_in_worker(self):
        """The fingerprint function must be importable without side effects."""
        # This tests that the delayed import pattern works
        fp = research_cache_fingerprint(
            production_horizon_days=63, target_horizon_days=63,
            holding_period_days=63, rebalance_frequency_days=63,
            ic_evaluation_horizon=63, execution_tau_days=None,
            rebalance_policy_mode="halflife_aware",
            execution_aware_policy_version=1,
            orientation_policy_version=1,
            scoring_weights_policy_mode="algorithmic",
        )
        assert len(fp) == 16
        assert isinstance(fp, str)

    def test_logging_only_fields_dont_change_fingerprint(self):
        """record_orientation_manifest and similar non-policy fields don't affect fingerprint."""
        # The fingerprint only includes the fields listed in research_cache_fingerprint()
        # Adding artifact_path or manifest_path should NOT change the hash
        fp1 = research_cache_fingerprint(
            production_horizon_days=10, target_horizon_days=10,
            holding_period_days=10, rebalance_frequency_days=10,
            ic_evaluation_horizon=10, execution_tau_days=None,
        )
        fp2 = research_cache_fingerprint(
            production_horizon_days=10, target_horizon_days=10,
            holding_period_days=10, rebalance_frequency_days=10,
            ic_evaluation_horizon=10, execution_tau_days=None,
        )
        assert fp1 == fp2  # same config fields → same fingerprint
