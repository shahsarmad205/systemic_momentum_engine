"""P35: ExecutionAwareHorizonPolicy tests."""
from __future__ import annotations

import math
import pandas as pd
import pytest
from model_selection.execution_aware_horizon_policy import (
    ExecutionAwareHorizonPolicy,
    ContractCandidate,
    ScoringWeights,
    ScoringWeightsPolicy,
    WeightsMode,
    DataPhase,
    PersistencePolicy,
    CostPolicy,
    FailClosedPolicy,
    TieBreakPolicy,
    ArtifactPolicy,
    FailClosedMode,
    TieBreakMode,
    parse_execution_aware_horizon_policy,
    derive_scoring_weights_from_decay,
    derive_scoring_weights_post_validation,
)
from model_selection.configuration import execution_aware_horizon_policy_config


class TestPolicyDefaults:
    """Policy produces documented defaults when YAML section is absent."""

    def test_default_policy_is_advisory_only(self):
        policy = ExecutionAwareHorizonPolicy()
        assert policy.enabled is True
        assert policy.apply is False  # advisory only
        assert policy.policy_version == 1
        assert len(policy.candidate_contracts) == 4
        assert policy.candidate_contracts[0].target_horizon_days == 5

    def test_default_policy_validates_clean(self):
        policy = ExecutionAwareHorizonPolicy()
        assert policy.validate() == []

    def test_parse_None_returns_defaults(self):
        policy = parse_execution_aware_horizon_policy(None)
        assert policy.apply is False
        assert len(policy.candidate_contracts) == 4

    def test_parse_empty_dict_returns_defaults(self):
        policy = parse_execution_aware_horizon_policy({})
        assert policy.apply is False
        assert policy.enabled is True

    def test_config_parser_empty_config(self):
        policy = execution_aware_horizon_policy_config({})
        assert policy.apply is False


class TestPolicyYamlParsing:
    """Policy parses from YAML-sourced dict."""

    def test_parse_full_config(self):
        raw = {
            "policy_version": 2,
            "enabled": True,
            "apply": False,
            "candidate_contracts": [
                {"target_horizon_days": 10, "holding_period_days": 10, "rebalance_frequency_days": 2},
                {"target_horizon_days": 63, "holding_period_days": 63},
            ],
            "scoring_weights_policy": {
                "mode": "config",
                "explicit": {
                    "ic_strength": 0.30,
                    "ic_consistency": 0.10,
                    "halflife_persistence": 0.30,
                    "cost_adjusted_ic": 0.15,
                    "alpha_capture": 0.10,
                    "execution_sharpe": 0.05,
                },
            },
            "persistence_policy": {
                "enabled": True,
                "formula": "exponential_decay",
                "threshold": 0.40,
                "halflife_source": "ic_decay_table",
            },
            "cost_policy": {
                "enabled": True,
                "max_cost_to_gross_pnl": 0.50,
            },
            "fail_closed_policy": {
                "mode": "fail",
            },
            "tie_break_policy": {
                "mode": "prefer_lower_cost_pnl",
                "tolerance_pct": 3.0,
            },
            "artifact_policy": {
                "horizon_frontier_path": "custom/horizon_frontier.csv",
            },
        }
        policy = parse_execution_aware_horizon_policy(raw)
        assert policy.policy_version == 2
        assert len(policy.candidate_contracts) == 2
        assert policy.candidate_contracts[0].target_horizon_days == 10
        assert policy.candidate_contracts[0].holding_period_days == 10
        assert policy.candidate_contracts[0].rebalance_frequency_days == 2
        assert policy.candidate_contracts[1].holding_period_days == 63
        assert policy.candidate_contracts[1].rebalance_frequency_days is None
        assert policy.scoring_weights_policy.explicit.ic_strength == 0.30
        assert policy.scoring_weights_policy.explicit.halflife_persistence == 0.30
        assert policy.persistence_policy.threshold == 0.40
        assert policy.cost_policy.max_cost_to_gross_pnl == 0.50
        assert policy.fail_closed_policy.mode == FailClosedMode.FAIL
        assert policy.tie_break_policy.mode == TieBreakMode.PREFER_LOWER_COST_PNL
        assert policy.tie_break_policy.tolerance_pct == 3.0
        assert policy.artifact_policy.horizon_frontier_path == "custom/horizon_frontier.csv"

    def test_parse_legacy_candidate_horizons_shorthand(self):
        """candidate_horizons shorthand produces ContractCandidates."""
        raw = {
            "candidate_horizons": [10, 20, 63],
        }
        policy = parse_execution_aware_horizon_policy(raw)
        assert len(policy.candidate_contracts) == 3
        assert policy.candidate_contracts[0].target_horizon_days == 10
        assert policy.candidate_contracts[0].holding_period_days is None  # inherit

    def test_parse_fallback_to_default_contract(self):
        raw = {
            "fail_closed_policy": {
                "mode": "fallback_to_default",
                "default_contract": {
                    "target_horizon_days": 5,
                    "holding_period_days": 5,
                    "rebalance_frequency_days": 2,
                },
            },
        }
        policy = parse_execution_aware_horizon_policy(raw)
        assert policy.fail_closed_policy.mode == FailClosedMode.FALLBACK_TO_DEFAULT
        assert policy.fail_closed_policy.default_contract is not None
        assert policy.fail_closed_policy.default_contract.target_horizon_days == 5


class TestPolicyValidation:
    """Invalid config raises promptly."""

    def test_negative_weight_raises(self):
        raw = {
            "scoring_weights_policy": {
                "mode": "config",
                "explicit": {"ic_strength": -0.1},
            },
        }
        with pytest.raises(ValueError, match="weight_ic_strength"):
            parse_execution_aware_horizon_policy(raw)

    def test_all_zero_weights_raises(self):
        raw = {
            "scoring_weights_policy": {
                "mode": "config",
                "explicit": {
                    "ic_strength": 0.0, "ic_consistency": 0.0,
                    "halflife_persistence": 0.0, "cost_adjusted_ic": 0.0,
                    "alpha_capture": 0.0, "execution_sharpe": 0.0,
                },
            },
        }
        with pytest.raises(ValueError, match="At least one"):
            parse_execution_aware_horizon_policy(raw)

    def test_invalid_persistence_threshold_raises(self):
        raw = {
            "persistence_policy": {"threshold": 1.5},
        }
        with pytest.raises(ValueError, match="Persistence threshold"):
            parse_execution_aware_horizon_policy(raw)

    def test_invalid_persistence_formula_raises(self):
        raw = {
            "persistence_policy": {"formula": "invalid_formula"},
        }
        with pytest.raises(ValueError, match="Unknown persistence formula"):
            parse_execution_aware_horizon_policy(raw)

    def test_empty_candidates_raises(self):
        raw = {
            "candidate_contracts": [],
        }
        with pytest.raises(ValueError, match="At least one"):
            parse_execution_aware_horizon_policy(raw)

    def test_invalid_fail_mode_raises(self):
        raw = {
            "fail_closed_policy": {"mode": "invalid_mode"},
        }
        with pytest.raises(ValueError, match="Invalid fail_closed_policy"):
            parse_execution_aware_horizon_policy(raw)

    def test_invalid_tie_break_mode_raises(self):
        raw = {
            "tie_break_policy": {"mode": "invalid_mode"},
        }
        with pytest.raises(ValueError, match="Invalid tie_break_policy"):
            parse_execution_aware_horizon_policy(raw)

    def test_fallback_without_default_contract_raises(self):
        raw = {
            "fail_closed_policy": {"mode": "fallback_to_default"},
        }
        with pytest.raises(ValueError, match="requires default_contract"):
            parse_execution_aware_horizon_policy(raw)

    def test_config_override_halflife_without_override_raises(self):
        raw = {
            "persistence_policy": {
                "halflife_source": "config_override",
            },
        }
        with pytest.raises(ValueError, match="halflife_override is None"):
            parse_execution_aware_horizon_policy(raw)

    def test_negative_tolerance_raises(self):
        raw = {
            "tie_break_policy": {"tolerance_pct": -1.0},
        }
        with pytest.raises(ValueError, match="tolerance_pct"):
            parse_execution_aware_horizon_policy(raw)


class TestPolicySerialization:
    """Policy serializes to dict and YAML cleanly."""

    def test_to_dict_contains_all_sections(self):
        policy = ExecutionAwareHorizonPolicy()
        d = policy.to_dict()
        assert "policy_version" in d
        assert "candidate_contracts" in d
        assert "scoring_weights_policy" in d
        assert "persistence_policy" in d
        assert "cost_policy" in d
        assert "fail_closed_policy" in d
        assert "tie_break_policy" in d
        assert "artifact_policy" in d

    def test_to_yaml_produces_valid_yaml(self):
        policy = ExecutionAwareHorizonPolicy()
        yaml_str = policy.to_yaml()
        assert "policy_version" in yaml_str
        assert "candidate_contracts" in yaml_str

    def test_to_dict_round_trip_defaults(self):
        """Default policy → dict preserves defaults."""
        policy = ExecutionAwareHorizonPolicy()
        d = policy.to_dict()
        assert d["policy_version"] == 1
        assert d["enabled"] is True
        assert d["apply"] is False

    def test_to_dict_round_trip_custom(self):
        raw = {
            "scoring_weights_policy": {
                "mode": "config",
                "explicit": {"ic_strength": 0.40},
            },
            "candidate_contracts": [
                {"target_horizon_days": 20, "holding_period_days": 10},
            ],
        }
        policy = parse_execution_aware_horizon_policy(raw)
        d = policy.to_dict()
        assert d["scoring_weights_policy"]["explicit"]["ic_strength"] == 0.40
        assert d["candidate_contracts"][0]["target_horizon_days"] == 20
        assert d["candidate_contracts"][0]["holding_period_days"] == 10


class TestSubPolicyValidation:
    """Individual sub-policy validation."""

    def test_scoring_weights_valid_default(self):
        assert ScoringWeights().validate() == []

    def test_scoring_weights_negative(self):
        errors = ScoringWeights(ic_strength=-0.1).validate()
        assert any("ic_strength" in e for e in errors)

    def test_persistence_policy_valid_default(self):
        assert PersistencePolicy().validate() == []

    def test_cost_policy_valid_default(self):
        assert CostPolicy().validate() == []

    def test_fail_closed_policy_valid_default(self):
        assert FailClosedPolicy().validate() == []

    def test_tie_break_policy_valid_default(self):
        assert TieBreakPolicy().validate() == []


class TestContractCandidate:
    """ContractCandidate independently testable."""

    def test_contract_defaults_inherit(self):
        c = ContractCandidate(target_horizon_days=20)
        assert c.holding_period_days is None
        assert c.rebalance_frequency_days is None

    def test_contract_explicit_overrides(self):
        c = ContractCandidate(target_horizon_days=20, holding_period_days=10, rebalance_frequency_days=2)
        assert c.holding_period_days == 10
        assert c.rebalance_frequency_days == 2


def _make_decay_table(ics: list[float], halflives: list[float] | None = None,
                      horizons: list[int] | None = None) -> pd.DataFrame:
    """Build a minimal IC-decay table for weight derivation tests."""
    feat_names = [f"f{i}" for i in range(len(ics))]
    if horizons is None:
        horizons = [5, 10, 20, 63]
    rows = []
    for feat, ic, hl in zip(feat_names, ics,
                             (halflives or [3.0] * len(ics)) * (len(feat_names) // max(len(ics), 1) + 1)):
        for h in horizons:
            rows.append({
                "feature": feat,
                "horizon_days": h,
                "target_type": "net_residual_return",
                "daily_spearman_ic": ic,
                "daily_spearman_ic_std": 0.05,
                "daily_spearman_ic_tstat": ic / 0.05 * math.sqrt(200),
                "signal_halflife_days": hl,
            })
    # Also add raw_return rows for cost gap detection
    for feat, ic in zip(feat_names, ics):
        for h in horizons:
            rows.append({
                "feature": feat,
                "horizon_days": h,
                "target_type": "raw_return",
                "daily_spearman_ic": ic * 1.3,  # raw IC slightly higher → cost gap
                "daily_spearman_ic_std": 0.05,
                "daily_spearman_ic_tstat": (ic * 1.3) / 0.05 * math.sqrt(200),
                "signal_halflife_days": halflives[0] if halflives else 3.0,
            })
    return pd.DataFrame(rows)


class TestAlgorithmicWeightDerivation:
    """P35: Algorithmic scoring weight derivation from IC-decay evidence."""

    def test_config_mode_returns_explicit_weights(self):
        decay = _make_decay_table([0.01, 0.02, 0.015])
        policy = ScoringWeightsPolicy(mode=WeightsMode.CONFIG)
        weights = derive_scoring_weights_from_decay(decay, policy=policy)
        assert weights == policy.explicit

    def test_algorithmic_mode_derives_weights(self):
        decay = _make_decay_table([0.01, 0.02, 0.015, 0.018], halflives=[3.0, 3.0, 3.0, 3.0])
        policy = ScoringWeightsPolicy(mode=WeightsMode.ALGORITHMIC)
        weights = derive_scoring_weights_from_decay(decay, policy=policy)
        total = weights.sum()
        assert abs(total - 1.0) < 0.01, f"Weights sum to {total}, expected ~1.0"
        # All weights should be positive
        for name in ["ic_strength", "ic_consistency", "halflife_persistence",
                     "cost_adjusted_ic", "alpha_capture", "execution_sharpe"]:
            assert getattr(weights, name) > 0, f"{name} should be > 0"

    def test_algorithmic_falls_back_on_insufficient_evidence(self):
        decay = _make_decay_table([0.01])  # only 1 feature, < evidence_min_features=3
        policy = ScoringWeightsPolicy(mode=WeightsMode.ALGORITHMIC, evidence_min_features=3)
        weights = derive_scoring_weights_from_decay(decay, policy=policy)
        assert weights == policy.explicit  # falls back

    def test_algorithmic_with_long_halflife_weights_persistence_higher(self):
        decay_long = _make_decay_table([0.015, 0.016, 0.014], halflives=[30.0, 30.0, 30.0])
        decay_short = _make_decay_table([0.015, 0.016, 0.014], halflives=[1.0, 1.0, 1.0])
        policy = ScoringWeightsPolicy(mode=WeightsMode.ALGORITHMIC)
        w_long = derive_scoring_weights_from_decay(decay_long, policy=policy)
        w_short = derive_scoring_weights_from_decay(decay_short, policy=policy)
        assert w_long.halflife_persistence > w_short.halflife_persistence

    def test_algorithmic_weights_sum_to_one_and_all_positive(self):
        """All derived weights are positive and sum to ~1.0."""
        decay = _make_decay_table([0.001, 0.002, 0.003])
        policy = ScoringWeightsPolicy(mode=WeightsMode.ALGORITHMIC, min_weight=0.05, max_weight=0.40)
        weights = derive_scoring_weights_from_decay(decay, policy=policy)
        total = weights.sum()
        assert abs(total - 1.0) < 0.01
        for name in ["ic_strength", "ic_consistency", "halflife_persistence",
                     "cost_adjusted_ic", "alpha_capture", "execution_sharpe"]:
            assert getattr(weights, name) > 0, f"{name} should be > 0"

    def test_resolve_scoring_weights_config_mode(self):
        policy = ExecutionAwareHorizonPolicy()
        weights = policy.resolve_scoring_weights()
        assert weights == policy.scoring_weights_policy.explicit

    def test_resolve_scoring_weights_algorithmic_mode(self):
        decay = _make_decay_table([0.01, 0.02, 0.015], halflives=[3.0, 3.0, 3.0])
        policy = ExecutionAwareHorizonPolicy(
            scoring_weights_policy=ScoringWeightsPolicy(mode=WeightsMode.ALGORITHMIC),
        )
        weights = policy.resolve_scoring_weights(alpha_decay=decay)
        total = weights.sum()
        assert abs(total - 1.0) < 0.01


class TestPostValidationWeightUpdate:
    """P35: Post-validation weight shifting."""

    def test_config_mode_returns_unchanged(self):
        pre = ScoringWeights(ic_strength=0.30, ic_consistency=0.20)
        policy = ScoringWeightsPolicy(mode=WeightsMode.CONFIG)
        post = derive_scoring_weights_post_validation(
            pre_training_weights=pre,
            alpha_capture_evidence=0.5,
            policy=policy,
        )
        assert post == pre

    def test_no_evidence_returns_unchanged(self):
        pre = ScoringWeights(ic_strength=0.30, ic_consistency=0.20)
        policy = ScoringWeightsPolicy(mode=WeightsMode.ALGORITHMIC)
        post = derive_scoring_weights_post_validation(
            pre_training_weights=pre,
            alpha_capture_evidence=None,
            execution_sharpe_evidence=None,
            policy=policy,
        )
        assert post == pre

    def test_post_validation_shifts_weights_to_evidence(self):
        pre = ScoringWeights(ic_strength=0.40, halflife_persistence=0.30, cost_adjusted_ic=0.20, ic_consistency=0.10)
        policy = ScoringWeightsPolicy(mode=WeightsMode.ALGORITHMIC, min_weight=0.01)
        post = derive_scoring_weights_post_validation(
            pre_training_weights=pre,
            alpha_capture_evidence=0.30,
            execution_sharpe_evidence=0.20,
            policy=policy,
        )
        # alpha_capture and execution_sharpe should have non-trivial weight
        assert post.alpha_capture > 0.01
        assert post.execution_sharpe > 0.01
        # Pre-training weights should be decayed
        assert post.ic_strength < pre.ic_strength
