"""C.4: Tests for nested validation short-circuit and SelectionPlan."""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
import pytest

import run_model_selection as rms
from model_selection.research_state import ResearchStateStore, TimingLedger


def _make_train_df(n_dates=40, n_tickers=3):
    dates = pd.bdate_range("2020-01-01", periods=n_dates)
    rows = []
    for dt in dates:
        for i in range(n_tickers):
            rows.append({
                "date": dt,
                "ticker": f"T{i}",
                "feature": float(i),
                "forward_return": 0.01,
                "target_return": 0.01,
                "target_up": 1,
                "target_rank": 0.5,
                "target_down_decile": 0,
                "daily_return": 0.001,
            })
    return pd.DataFrame(rows)


class DummyPrepared:
    def __init__(self, df):
        self.train_df = df.iloc[:30].copy()
        self.eval_df = df.iloc[30:].copy()
        self.x_train = self.train_df[["feature"]].to_numpy()
        self.x_eval = self.eval_df[["feature"]].to_numpy()


class DummyCache:
    def __init__(self, df):
        self._df = df
        self.fold_calls = 0
        self.state_calls = 0

    def get_prepared_fold(self, **kwargs):
        self.fold_calls += 1
        return DummyPrepared(self._df)

    def get_validation_state(self, **kwargs):
        self.state_calls += 1
        return object()

    def get_training_target(self, **kwargs):
        return self._df["target_return"].to_numpy(dtype=float)

    def stats(self):
        return {}


class DummyTimingLedger:
    def __init__(self):
        self.events = []

    def record(self, event, **fields):
        self.events.append({"event": event, **fields})


# ── 1. Single-candidate short-circuit ───────────────────────────────────────

class TestSingleCandidateShortCircuit:
    def test_short_circuit_selects_sole_candidate(self, monkeypatch):
        """When pool has exactly one candidate, it should be selected without nested validation."""
        train_df = _make_train_df()
        pool = [
            rms.NestedCandidateSpec("Ridge", "regressor", False, object(), ("feature",), "full", 5),
        ]
        monkeypatch.setattr(rms, "_build_nested_candidate_pool", lambda **kwargs: pool)
        monkeypatch.setattr(
            rms, "_nested_inner_windows",
            lambda *args, **kwargs: [
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"),
                 pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-20")),
            ],
        )

        validate_called = {"count": 0}
        def fake_validate(*args, **kwargs):
            validate_called["count"] += 1
            return {"nested_selection_score": 1.0}
        monkeypatch.setattr(rms, "_nested_validate_candidate", fake_validate)

        candidate, metrics, artifacts = rms._select_nested_candidate(
            train_df,
            prepared_cache=DummyCache(train_df),
            research_state=None,
            primary_path="long_only_overlay",
            models=[("Ridge", object(), False, "regressor")],
            cfg={},
            nested_cfg={"enabled": True, "true_selection_enabled": True, "max_windows": 1, "validation_days": 10, "min_train_days": 10},
            feat_cols=["feature"],
            short_feature_subset=[],
            overlay_feature_subset=[],
            default_horizon=5,
            horizon_contract=None,
            max_positions=10,
            min_positions=3,
            embargo_days=0,
            use_risk_adj=False,
            target_cfg=None,
            costs=None,
            max_name_weight=0.1,
            return_artifacts=True,
        )

        assert candidate is not None
        assert candidate.model_name == "Ridge"
        assert validate_called["count"] == 0, "_nested_validate_candidate should NOT be called for single candidate"
        assert metrics["nested_candidate_count"] == 1
        assert metrics.get("nested_short_circuit") == 1.0
        assert len(artifacts) == 1
        assert artifacts[0]["short_circuit"] is True

    def test_short_circuit_emits_selection_plan(self, tmp_path, monkeypatch):
        """Short-circuit should write a selection_plan.json artifact."""
        train_df = _make_train_df()
        pool = [
            rms.NestedCandidateSpec("Ridge", "regressor", False, object(), ("feature",), "full", 5),
        ]
        monkeypatch.setattr(rms, "_build_nested_candidate_pool", lambda **kwargs: pool)
        monkeypatch.setattr(
            rms, "_nested_inner_windows",
            lambda *args, **kwargs: [
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"),
                 pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-20")),
            ],
        )
        monkeypatch.setattr(rms, "_nested_validate_candidate", lambda *args, **kwargs: {"nested_selection_score": 1.0})

        state = ResearchStateStore(root_dir=tmp_path, namespace="c4_test", payload={"k": 1})
        candidate, metrics, artifacts = rms._select_nested_candidate(
            train_df,
            prepared_cache=DummyCache(train_df),
            research_state=state,
            primary_path="long_only_overlay",
            models=[("Ridge", object(), False, "regressor")],
            cfg={},
            nested_cfg={"enabled": True, "true_selection_enabled": True, "max_windows": 1, "validation_days": 10, "min_train_days": 10},
            feat_cols=["feature"],
            short_feature_subset=[],
            overlay_feature_subset=[],
            default_horizon=5,
            horizon_contract=None,
            max_positions=10,
            min_positions=3,
            embargo_days=0,
            use_risk_adj=False,
            target_cfg=None,
            costs=None,
            max_name_weight=0.1,
            return_artifacts=True,
        )

        plan_files = list(state.subdir("selection_plans").glob("plan_*.json"))
        assert len(plan_files) >= 1
        plan_data = json.loads(plan_files[0].read_text())
        assert plan_data["short_circuit_allowed"] is True
        assert plan_data["selection_required"] is False
        assert plan_data["short_circuit_reason"] == "single_candidate_no_selection_decision"

    def test_short_circuit_emits_telemetry(self, monkeypatch):
        """Short-circuit should emit a selection_plan telemetry event."""
        train_df = _make_train_df()
        pool = [
            rms.NestedCandidateSpec("Ridge", "regressor", False, object(), ("feature",), "full", 5),
        ]
        monkeypatch.setattr(rms, "_build_nested_candidate_pool", lambda **kwargs: pool)
        monkeypatch.setattr(
            rms, "_nested_inner_windows",
            lambda *args, **kwargs: [
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"),
                 pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-20")),
            ],
        )
        monkeypatch.setattr(rms, "_nested_validate_candidate", lambda *args, **kwargs: {"nested_selection_score": 1.0})

        ledger = DummyTimingLedger()
        rms._select_nested_candidate(
            train_df,
            prepared_cache=DummyCache(train_df),
            research_state=None,
            primary_path="long_only_overlay",
            models=[("Ridge", object(), False, "regressor")],
            cfg={},
            nested_cfg={"enabled": True, "true_selection_enabled": True, "max_windows": 1, "validation_days": 10, "min_train_days": 10},
            feat_cols=["feature"],
            short_feature_subset=[],
            overlay_feature_subset=[],
            default_horizon=5,
            horizon_contract=None,
            max_positions=10,
            min_positions=3,
            embargo_days=0,
            use_risk_adj=False,
            target_cfg=None,
            costs=None,
            max_name_weight=0.1,
            timing_ledger=ledger,
        )

        plan_events = [e for e in ledger.events if e["event"] == "selection_plan"]
        assert len(plan_events) >= 1
        assert plan_events[0]["short_circuit_allowed"] is True
        assert plan_events[0]["selection_required"] is False


# ── 2. Multiple candidates: nested validation still runs ────────────────────

class TestMultipleCandidates:
    def test_multi_candidate_runs_nested_validation(self, monkeypatch):
        """When pool has >1 candidates, nested validation must run."""
        train_df = _make_train_df()
        pool = [
            rms.NestedCandidateSpec("Ridge", "regressor", False, object(), ("feature",), "full", 5),
            rms.NestedCandidateSpec("Ridge2", "regressor", False, object(), ("feature",), "program", 5),
        ]
        monkeypatch.setattr(rms, "_build_nested_candidate_pool", lambda **kwargs: pool)
        monkeypatch.setattr(
            rms, "_nested_inner_windows",
            lambda *args, **kwargs: [
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"),
                 pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-20")),
            ],
        )

        validate_called = {"count": 0}
        def fake_validate(*args, **kwargs):
            validate_called["count"] += 1
            return {"nested_selection_score": 0.5, "nested_sharpe_mean": 0.3, "nested_ic_mean": 0.01, "nested_windows": 1}
        monkeypatch.setattr(rms, "_nested_validate_candidate", fake_validate)
        monkeypatch.setattr(
            rms, "_nested_proxy_candidate",
            lambda *args, **kwargs: {"proxy_selection_score": 0.3, "proxy_ic_mean": 0.01, "proxy_daily_icir_mean": 0.1, "proxy_turnover_mean": 0.1, "proxy_windows": 1},
        )

        candidate, metrics = rms._select_nested_candidate(
            train_df,
            prepared_cache=DummyCache(train_df),
            research_state=None,
            primary_path="long_only_overlay",
            models=[("Ridge", object(), False, "regressor")],
            cfg={},
            nested_cfg={"enabled": True, "true_selection_enabled": True, "max_windows": 1, "validation_days": 10, "min_train_days": 10},
            feat_cols=["feature"],
            short_feature_subset=[],
            overlay_feature_subset=[],
            default_horizon=5,
            horizon_contract=None,
            max_positions=10,
            min_positions=3,
            embargo_days=0,
            use_risk_adj=False,
            target_cfg=None,
            costs=None,
            max_name_weight=0.1,
        )

        assert validate_called["count"] > 0, "_nested_validate_candidate MUST be called for multiple candidates"
        assert metrics["nested_candidate_count"] == 2


# ── 3. Multiple hyperparameters ─────────────────────────────────────────────

class TestMultipleHyperparameters:
    def test_multi_hyperparam_runs_nested_validation(self, monkeypatch):
        """Same family but different feature views → nested validation runs."""
        train_df = _make_train_df()
        pool = [
            rms.NestedCandidateSpec("Ridge", "regressor", False, object(), ("feature",), "full", 5),
            rms.NestedCandidateSpec("Ridge", "regressor", False, object(), ("feature",), "program", 5),
        ]
        monkeypatch.setattr(rms, "_build_nested_candidate_pool", lambda **kwargs: pool)
        monkeypatch.setattr(
            rms, "_nested_inner_windows",
            lambda *args, **kwargs: [
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"),
                 pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-20")),
            ],
        )

        validate_called = {"count": 0}
        def fake_validate(*args, **kwargs):
            validate_called["count"] += 1
            return {"nested_selection_score": 0.5, "nested_sharpe_mean": 0.3, "nested_ic_mean": 0.01, "nested_windows": 1}
        monkeypatch.setattr(rms, "_nested_validate_candidate", fake_validate)
        monkeypatch.setattr(
            rms, "_nested_proxy_candidate",
            lambda *args, **kwargs: {"proxy_selection_score": 0.3, "proxy_ic_mean": 0.01, "proxy_daily_icir_mean": 0.1, "proxy_turnover_mean": 0.1, "proxy_windows": 1},
        )

        candidate, metrics = rms._select_nested_candidate(
            train_df,
            prepared_cache=DummyCache(train_df),
            research_state=None,
            primary_path="long_only_overlay",
            models=[("Ridge", object(), False, "regressor")],
            cfg={},
            nested_cfg={"enabled": True, "true_selection_enabled": True, "max_windows": 1, "validation_days": 10, "min_train_days": 10},
            feat_cols=["feature"],
            short_feature_subset=[],
            overlay_feature_subset=[],
            default_horizon=5,
            horizon_contract=None,
            max_positions=10,
            min_positions=3,
            embargo_days=0,
            use_risk_adj=False,
            target_cfg=None,
            costs=None,
            max_name_weight=0.1,
        )

        assert validate_called["count"] > 0


# ── 4. Multiple horizons ────────────────────────────────────────────────────

class TestMultipleHorizons:
    def test_multi_horizon_runs_nested_validation(self, monkeypatch):
        """Different horizons → nested validation runs."""
        train_df = _make_train_df()
        pool = [
            rms.NestedCandidateSpec("Ridge", "regressor", False, object(), ("feature",), "full", 5),
            rms.NestedCandidateSpec("Ridge", "regressor", False, object(), ("feature",), "full", 10),
        ]
        monkeypatch.setattr(rms, "_build_nested_candidate_pool", lambda **kwargs: pool)
        monkeypatch.setattr(
            rms, "_nested_inner_windows",
            lambda *args, **kwargs: [
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"),
                 pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-20")),
            ],
        )

        validate_called = {"count": 0}
        def fake_validate(*args, **kwargs):
            validate_called["count"] += 1
            return {"nested_selection_score": 0.5, "nested_sharpe_mean": 0.3, "nested_ic_mean": 0.01, "nested_windows": 1}
        monkeypatch.setattr(rms, "_nested_validate_candidate", fake_validate)
        monkeypatch.setattr(
            rms, "_nested_proxy_candidate",
            lambda *args, **kwargs: {"proxy_selection_score": 0.3, "proxy_ic_mean": 0.01, "proxy_daily_icir_mean": 0.1, "proxy_turnover_mean": 0.1, "proxy_windows": 1},
        )

        candidate, metrics = rms._select_nested_candidate(
            train_df,
            prepared_cache=DummyCache(train_df),
            research_state=None,
            primary_path="long_only_overlay",
            models=[("Ridge", object(), False, "regressor")],
            cfg={},
            nested_cfg={"enabled": True, "true_selection_enabled": True, "max_windows": 1, "validation_days": 10, "min_train_days": 10, "search": {"candidate_horizons": [5, 10]}},
            feat_cols=["feature"],
            short_feature_subset=[],
            overlay_feature_subset=[],
            default_horizon=5,
            horizon_contract=None,
            max_positions=10,
            min_positions=3,
            embargo_days=0,
            use_risk_adj=False,
            target_cfg=None,
            costs=None,
            max_name_weight=0.1,
        )

        assert validate_called["count"] > 0


# ── 5. Cross-family selection ───────────────────────────────────────────────

class TestCrossFamilySelection:
    def test_cross_family_skips_when_single_family(self, monkeypatch):
        """allow_cross_family_selection=true but only 1 candidate from 1 family → short-circuit."""
        train_df = _make_train_df()
        pool = [
            rms.NestedCandidateSpec("Ridge", "regressor", False, object(), ("feature",), "full", 5),
        ]
        monkeypatch.setattr(rms, "_build_nested_candidate_pool", lambda **kwargs: pool)

        validate_called = {"count": 0}
        def fake_validate(*args, **kwargs):
            validate_called["count"] += 1
            return {"nested_selection_score": 0.5, "nested_sharpe_mean": 0.3, "nested_ic_mean": 0.01, "nested_windows": 1}
        monkeypatch.setattr(rms, "_nested_validate_candidate", fake_validate)

        candidate, metrics = rms._select_nested_candidate(
            train_df,
            prepared_cache=DummyCache(train_df),
            research_state=None,
            primary_path="long_only_overlay",
            models=[("Ridge", object(), False, "regressor")],
            cfg={},
            nested_cfg={
                "enabled": True,
                "true_selection_enabled": True,
                "max_windows": 1,
                "validation_days": 10,
                "min_train_days": 10,
                "search": {"allow_cross_family_selection": True},
            },
            feat_cols=["feature"],
            short_feature_subset=[],
            overlay_feature_subset=[],
            default_horizon=5,
            horizon_contract=None,
            max_positions=10,
            min_positions=3,
            embargo_days=0,
            use_risk_adj=False,
            target_cfg=None,
            costs=None,
            max_name_weight=0.1,
        )

        assert validate_called["count"] == 0, (
            "Single-family pool short-circuits even with cross-family enabled"
        )
        assert metrics.get("nested_validation_skipped", 0.0) == 1.0

    def test_cross_family_runs_nested_validation(self, monkeypatch):
        """allow_cross_family_selection=true with 2 families → nested validation runs."""
        train_df = _make_train_df()
        pool = [
            rms.NestedCandidateSpec("Ridge", "regressor", False, object(), ("feature",), "full", 5),
            rms.NestedCandidateSpec("RandomForest", "classifier", False, object(), ("feature",), "full", 5),
        ]
        monkeypatch.setattr(rms, "_build_nested_candidate_pool", lambda **kwargs: pool)
        monkeypatch.setattr(
            rms, "_nested_inner_windows",
            lambda *args, **kwargs: [
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"),
                 pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-20")),
            ],
        )

        validate_called = {"count": 0}
        def fake_validate(*args, **kwargs):
            validate_called["count"] += 1
            return {"nested_selection_score": 0.5, "nested_sharpe_mean": 0.3, "nested_ic_mean": 0.01, "nested_windows": 1}
        monkeypatch.setattr(rms, "_nested_validate_candidate", fake_validate)
        monkeypatch.setattr(
            rms, "_nested_proxy_candidate",
            lambda *args, **kwargs: {"proxy_selection_score": 0.3, "proxy_ic_mean": 0.01, "proxy_daily_icir_mean": 0.1, "proxy_turnover_mean": 0.1, "proxy_windows": 1},
        )

        candidate, metrics = rms._select_nested_candidate(
            train_df,
            prepared_cache=DummyCache(train_df),
            research_state=None,
            primary_path="long_only_overlay",
            models=[("Ridge", object(), False, "regressor")],
            cfg={},
            nested_cfg={
                "enabled": True,
                "true_selection_enabled": True,
                "max_windows": 1,
                "validation_days": 10,
                "min_train_days": 10,
                "search": {"allow_cross_family_selection": True},
            },
            feat_cols=["feature"],
            short_feature_subset=[],
            overlay_feature_subset=[],
            default_horizon=5,
            horizon_contract=None,
            max_positions=10,
            min_positions=3,
            embargo_days=0,
            use_risk_adj=False,
            target_cfg=None,
            costs=None,
            max_name_weight=0.1,
        )

        assert validate_called["count"] > 0, "Multi-family cross-family selection must run nested validation"


# ── 6. Forced nested validation ─────────────────────────────────────────────

class TestForcedNestedValidation:
    def test_forced_nested_runs_even_with_one_candidate(self, monkeypatch):
        """true_selection_enabled=false → nested validation runs even with one candidate."""
        train_df = _make_train_df()
        pool = [
            rms.NestedCandidateSpec("Ridge", "regressor", False, object(), ("feature",), "full", 5),
        ]
        monkeypatch.setattr(rms, "_build_nested_candidate_pool", lambda **kwargs: pool)
        monkeypatch.setattr(
            rms, "_nested_inner_windows",
            lambda *args, **kwargs: [
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"),
                 pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-20")),
            ],
        )

        validate_called = {"count": 0}
        def fake_validate(*args, **kwargs):
            validate_called["count"] += 1
            return {"nested_selection_score": 0.5, "nested_sharpe_mean": 0.3, "nested_ic_mean": 0.01, "nested_windows": 1}
        monkeypatch.setattr(rms, "_nested_validate_candidate", fake_validate)

        candidate, metrics = rms._select_nested_candidate(
            train_df,
            prepared_cache=DummyCache(train_df),
            research_state=None,
            primary_path="long_only_overlay",
            models=[("Ridge", object(), False, "regressor")],
            cfg={},
            nested_cfg={
                "enabled": True,
                "true_selection_enabled": False,
                "max_windows": 1,
                "validation_days": 10,
                "min_train_days": 10,
            },
            feat_cols=["feature"],
            short_feature_subset=[],
            overlay_feature_subset=[],
            default_horizon=5,
            horizon_contract=None,
            max_positions=10,
            min_positions=3,
            embargo_days=0,
            use_risk_adj=False,
            target_cfg=None,
            costs=None,
            max_name_weight=0.1,
        )

        assert validate_called["count"] > 0, "Forced nested validation must run even with one candidate"


# ── 7. Artifact/schema compatibility ────────────────────────────────────────

class TestArtifactCompatibility:
    def test_selection_plan_exists(self, tmp_path, monkeypatch):
        """selection_plan.json must exist after _select_nested_candidate."""
        train_df = _make_train_df()
        pool = [
            rms.NestedCandidateSpec("Ridge", "regressor", False, object(), ("feature",), "full", 5),
        ]
        monkeypatch.setattr(rms, "_build_nested_candidate_pool", lambda **kwargs: pool)
        monkeypatch.setattr(
            rms, "_nested_inner_windows",
            lambda *args, **kwargs: [
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"),
                 pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-20")),
            ],
        )
        monkeypatch.setattr(rms, "_nested_validate_candidate", lambda *args, **kwargs: {"nested_selection_score": 1.0})

        state = ResearchStateStore(root_dir=tmp_path, namespace="c4_artifact", payload={"k": 1})
        rms._select_nested_candidate(
            train_df,
            prepared_cache=DummyCache(train_df),
            research_state=state,
            primary_path="long_only_overlay",
            models=[("Ridge", object(), False, "regressor")],
            cfg={},
            nested_cfg={"enabled": True, "true_selection_enabled": True, "max_windows": 1, "validation_days": 10, "min_train_days": 10},
            feat_cols=["feature"],
            short_feature_subset=[],
            overlay_feature_subset=[],
            default_horizon=5,
            horizon_contract=None,
            max_positions=10,
            min_positions=3,
            embargo_days=0,
            use_risk_adj=False,
            target_cfg=None,
            costs=None,
            max_name_weight=0.1,
        )

        plan_files = list(state.subdir("selection_plans").glob("plan_*.json"))
        assert len(plan_files) >= 1
        data = json.loads(plan_files[0].read_text())
        # Verify all required fields exist
        for key in ["outer_window_id", "model_family", "candidate_count", "selection_required",
                     "short_circuit_allowed", "short_circuit_reason", "selection_disabled_by_config", "horizon_candidates"]:
            assert key in data, f"Missing field: {key}"

    def test_short_circuit_metrics_schema(self, monkeypatch):
        """Short-circuit metrics must have explicit NaN for validation metrics."""
        train_df = _make_train_df()
        pool = [
            rms.NestedCandidateSpec("Ridge", "regressor", False, object(), ("feature",), "full", 5),
        ]
        monkeypatch.setattr(rms, "_build_nested_candidate_pool", lambda **kwargs: pool)
        monkeypatch.setattr(
            rms, "_nested_inner_windows",
            lambda *args, **kwargs: [
                (pd.Timestamp("2020-01-01"), pd.Timestamp("2020-02-01"),
                 pd.Timestamp("2020-02-01"), pd.Timestamp("2020-02-20")),
            ],
        )
        monkeypatch.setattr(rms, "_nested_validate_candidate", lambda *args, **kwargs: {"nested_selection_score": 1.0})

        candidate, metrics = rms._select_nested_candidate(
            train_df,
            prepared_cache=DummyCache(train_df),
            research_state=None,
            primary_path="long_only_overlay",
            models=[("Ridge", object(), False, "regressor")],
            cfg={},
            nested_cfg={"enabled": True, "true_selection_enabled": True, "max_windows": 1, "validation_days": 10, "min_train_days": 10},
            feat_cols=["feature"],
            short_feature_subset=[],
            overlay_feature_subset=[],
            default_horizon=5,
            horizon_contract=None,
            max_positions=10,
            min_positions=3,
            embargo_days=0,
            use_risk_adj=False,
            target_cfg=None,
            costs=None,
            max_name_weight=0.1,
        )

        assert np.isnan(metrics["nested_sharpe_mean"])
        assert np.isnan(metrics["nested_ic_mean"])
        assert metrics["nested_windows"] == 0
        assert np.isnan(metrics["nested_selection_score"])
        assert metrics["nested_candidate_count"] == 1
        assert metrics["nested_validation_skipped"] == 1.0
        assert metrics["nested_metrics_available"] == 0.0
        assert metrics["nested_short_circuit"] == 1.0


# ── 8. SelectionPlan builder ────────────────────────────────────────────────

class TestSelectionPlanBuilder:
    def test_plan_fields_populated(self):
        plan = rms._build_selection_plan(
            outer_window_id="win_001",
            model_family="regressor",
            pool=[rms.NestedCandidateSpec("Ridge", "regressor", False, object(), ("f1",), "full", 5)],
            nested_cfg={"enabled": True, "true_selection_enabled": True},
            search_cfg={},
            windows=[("2020-01-01", "2020-02-01", "2020-02-01", "2020-02-20")],
            train_df=_make_train_df(),
            horizon_contract=None,
            target_cfg=None,
            feat_cols=["f1", "f2"],
        )
        assert plan.outer_window_id == "win_001"
        assert plan.model_family == "regressor"
        assert plan.candidate_count == 1
        assert plan.selection_required is False
        assert plan.short_circuit_allowed is True
        assert plan.short_circuit_reason == "single_candidate_no_selection_decision"
        assert len(plan.candidate_ids) == 1
        assert len(plan.candidate_fingerprints) == 1

    def test_plan_multi_candidate(self):
        plan = rms._build_selection_plan(
            outer_window_id="win_002",
            model_family="regressor",
            pool=[
                rms.NestedCandidateSpec("Ridge", "regressor", False, object(), ("f1",), "full", 5),
                rms.NestedCandidateSpec("Ridge", "regressor", False, object(), ("f1",), "program", 5),
            ],
            nested_cfg={"enabled": True, "true_selection_enabled": True},
            search_cfg={},
            windows=[],
            train_df=_make_train_df(),
            horizon_contract=None,
            target_cfg=None,
            feat_cols=["f1"],
        )
        assert plan.candidate_count == 2
        assert plan.selection_required is True
        assert plan.short_circuit_allowed is False
        # Two candidates with different feature_views → hyperparams differ
        assert "hyperparams" in plan.selection_required_reason or "candidate_count" in plan.selection_required_reason

    def test_plan_to_dict_roundtrip(self):
        plan = rms._build_selection_plan(
            outer_window_id="win_003",
            model_family="regressor",
            pool=[rms.NestedCandidateSpec("Ridge", "regressor", False, object(), ("f1",), "full", 5)],
            nested_cfg={"enabled": True, "true_selection_enabled": True},
            search_cfg={},
            windows=[],
            train_df=_make_train_df(),
            horizon_contract=None,
            target_cfg=None,
            feat_cols=["f1"],
        )
        d = plan.to_dict()
        assert isinstance(d, dict)
        assert d["short_circuit_allowed"] is True
        assert json.dumps(d)  # Must be JSON serializable
