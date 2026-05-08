"""P37: OrientationPolicy tests."""
from __future__ import annotations

import pytest
from model_selection.orientation_policy import (
    OrientationPolicy,
    OrientationRecord,
    AggregateMode,
    LowConfidenceMode,
    EvidenceSource,
    parse_orientation_policy,
    orientation_manifest_to_dataframe,
)


class TestOrientationPolicyDefaults:
    """Default policy is IC_WEIGHTED, train-only evidence."""

    def test_default_policy_validates_clean(self):
        assert OrientationPolicy().validate() == []

    def test_default_aggregate_mode_is_ic_weighted(self):
        p = OrientationPolicy()
        assert p.aggregate_mode == AggregateMode.IC_WEIGHTED

    def test_default_requires_train_only_evidence(self):
        p = OrientationPolicy()
        assert p.require_train_only_evidence is True

    def test_default_low_confidence_mode(self):
        p = OrientationPolicy()
        assert p.low_confidence_mode == LowConfidenceMode.FALLBACK_TO_AGGREGATE

    def test_default_low_confidence_threshold(self):
        p = OrientationPolicy()
        assert p.low_confidence_ic_abs == 0.001
        assert p.is_low_confidence(0.0005)
        assert not p.is_low_confidence(0.01)


class TestAggregateDirection:
    """Direction aggregation across window-based policies."""

    def _records(self, directions: list[int], ic_raws: list[float] | None = None) -> list[OrientationRecord]:
        if ic_raws is None:
            ic_raws = [abs(d) * 0.02 for d in directions]  # default IC
        return [
            OrientationRecord(
                window_idx=i + 1, direction=d, ic_raw=ic_raws[i], ic_calibrated=ic_raws[i] * d,
            )
            for i, d in enumerate(directions)
        ]

    def test_majority_vote_positive_wins(self):
        p = OrientationPolicy(aggregate_mode=AggregateMode.MAJORITY_VOTE)
        records = self._records([1, 1, -1, 1, -1])
        direction, reason, diag = p.aggregate_direction(records)
        assert direction == 1
        assert diag["n_positive"] == 3
        assert diag["n_negative"] == 2

    def test_majority_vote_negative_wins(self):
        p = OrientationPolicy(aggregate_mode=AggregateMode.MAJORITY_VOTE)
        records = self._records([-1, -1, -1, 1, 1])
        direction, reason, diag = p.aggregate_direction(records)
        assert direction == -1

    def test_majority_vote_tie_goes_positive(self):
        p = OrientationPolicy(aggregate_mode=AggregateMode.MAJORITY_VOTE)
        records = self._records([1, 1, -1, -1])
        direction, reason, diag = p.aggregate_direction(records)
        assert direction == 1  # tie → +1

    def test_ic_weighted_strong_evidence_wins(self):
        """Window with strong IC should dominate regardless of count."""
        p = OrientationPolicy(aggregate_mode=AggregateMode.IC_WEIGHTED)
        # 1 window at -1 with strong IC beats 3 windows at +1 with weak IC
        records = self._records([-1, 1, 1, 1], [0.05, 0.001, 0.001, 0.001])
        direction, reason, diag = p.aggregate_direction(records)
        assert direction == -1
        assert "ic_weighted" in reason

    def test_ic_weighted_equal_weights(self):
        """Equal IC magnitudes → same as majority vote."""
        p = OrientationPolicy(aggregate_mode=AggregateMode.IC_WEIGHTED)
        records = self._records([1, 1, -1], [0.02, 0.02, 0.02])
        direction, reason, diag = p.aggregate_direction(records)
        assert direction == 1  # 2 pos vs 1 neg

    def test_train_only_filters_low_confidence(self):
        """TRAIN_ONLY mode ignores weak-evidence windows."""
        p = OrientationPolicy(
            aggregate_mode=AggregateMode.TRAIN_ONLY,
            low_confidence_ic_abs=0.01,
        )
        # 2 strong negative, 3 weak positive → direction = -1
        records = self._records([-1, -1, 1, 1, 1], [0.05, 0.04, 0.005, 0.002, 0.001])
        direction, reason, diag = p.aggregate_direction(records)
        assert direction == -1

    def test_train_only_all_low_confidence_fallback(self):
        """All windows low-confidence → fallback to +1."""
        p = OrientationPolicy(
            aggregate_mode=AggregateMode.TRAIN_ONLY,
            low_confidence_mode=LowConfidenceMode.FIXED_PLUS_ONE,
            low_confidence_ic_abs=0.10,
        )
        records = self._records([-1, 1, -1], [0.001, 0.002, 0.003])
        direction, reason, diag = p.aggregate_direction(records)
        assert direction == 1
        assert "plus_one" in reason

    def test_empty_records_returns_default(self):
        p = OrientationPolicy()
        direction, reason, diag = p.aggregate_direction([])
        assert direction == 1
        assert diag["n_windows"] == 0


class TestOrientationParseYaml:
    """Policy parses from YAML-sourced dicts."""

    def test_parse_empty_returns_defaults(self):
        p = parse_orientation_policy({})
        assert p.aggregate_mode == AggregateMode.IC_WEIGHTED

    def test_parse_none_returns_defaults(self):
        p = parse_orientation_policy(None)
        assert p.policy_version == 1

    def test_parse_full_config(self):
        raw = {
            "policy_version": 3,
            "aggregate_mode": "majority_vote",
            "low_confidence_mode": "abstain",
            "low_confidence_ic_abs": 0.005,
            "allow_per_window_overrides": True,
            "require_train_only_evidence": False,
            "record_orientation_manifest": False,
        }
        p = parse_orientation_policy(raw)
        assert p.policy_version == 3
        assert p.aggregate_mode == AggregateMode.MAJORITY_VOTE
        assert p.low_confidence_mode == LowConfidenceMode.ABSTAIN
        assert p.low_confidence_ic_abs == 0.005
        assert p.allow_per_window_overrides is True
        assert p.require_train_only_evidence is False
        assert p.record_orientation_manifest is False

    def test_invalid_aggregate_mode_raises(self):
        with pytest.raises(ValueError, match="aggregate_mode"):
            parse_orientation_policy({"aggregate_mode": "invalid"})

    def test_invalid_low_confidence_mode_raises(self):
        with pytest.raises(ValueError, match="low_confidence_mode"):
            parse_orientation_policy({"low_confidence_mode": "invalid"})

    def test_negative_ic_abs_raises(self):
        p = OrientationPolicy(low_confidence_ic_abs=-0.1)
        errors = p.validate()
        assert any("low_confidence_ic_abs" in e for e in errors)

    def test_empty_allowed_sources_raises(self):
        p = OrientationPolicy(allowed_sources=())
        errors = p.validate()
        assert any("allowed_source" in e for e in errors)


class TestOrientationRecord:
    """Per-window orientation record."""

    def test_record_serializes_to_dict(self):
        r = OrientationRecord(
            window_idx=3, direction=-1, ic_raw=-0.02, ic_calibrated=0.02,
            calibration_slope=-1.5, calibration_tstat=-25.0,
            mode="calibrated", reason="long mandate: IC < 0 → flip",
            n_features=6,
        )
        d = r.to_dict()
        assert d["window_idx"] == 3
        assert d["direction"] == -1
        assert d["ic_raw"] == -0.02
        assert d["ic_calibrated"] == 0.02
        assert d["calibration_slope"] == -1.5
        assert d["calibration_tstat"] == -25.0
        assert d["mode"] == "calibrated"


class TestOrientationManifest:
    """Manifest DataFrame export."""

    def test_empty_records_returns_empty_df(self):
        df = orientation_manifest_to_dataframe([])
        assert df is None or df.empty

    def test_records_with_model_name(self):
        records = [
            OrientationRecord(window_idx=1, direction=1, ic_raw=0.02, ic_calibrated=0.02),
            OrientationRecord(window_idx=2, direction=-1, ic_raw=-0.03, ic_calibrated=0.03, mode="calibrated"),
        ]
        df = orientation_manifest_to_dataframe(records, model_name="TestModel", policy_version=2)
        assert df is not None
        assert len(df) == 2
        assert "model_name" in df.columns
        assert "policy_version" in df.columns
        assert df.iloc[0]["model_name"] == "TestModel"
        assert df.iloc[0]["policy_version"] == 2


class TestTrainOnlyEvidence:
    """Policy enforces train-only evidence requirement."""

    def test_require_train_only_is_configurable(self):
        p = OrientationPolicy(require_train_only_evidence=True)
        assert p.require_train_only_evidence is True

        p2 = OrientationPolicy(require_train_only_evidence=False)
        assert p2.require_train_only_evidence is False

    def test_allowed_sources_default_is_validation_split(self):
        p = OrientationPolicy()
        assert EvidenceSource.VALIDATION_SPLIT in p.allowed_sources


class TestPolicyVersionAffectsBehavior:
    """Policy version changes are tracked in manifest and config."""

    def test_different_versions_produce_different_metadata(self):
        p1 = OrientationPolicy(policy_version=1)
        p2 = OrientationPolicy(policy_version=2)
        assert p1.policy_version != p2.policy_version

    def test_manifest_includes_policy_version(self):
        records = [OrientationRecord(window_idx=1, direction=1, ic_raw=0.02, ic_calibrated=0.02)]
        df = orientation_manifest_to_dataframe(records, model_name="M", policy_version=5)
        assert df is not None
        assert df.iloc[0]["policy_version"] == 5
