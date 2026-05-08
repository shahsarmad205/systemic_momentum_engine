from __future__ import annotations

import numpy as np
import pandas as pd

import run_model_selection as rms
from run_model_selection import _nested_validation_config
from model_selection.horizon_contract import build_horizon_contract
from model_selection.research_state import ResearchStateStore


def test_nested_validation_config_exposes_true_selection_search_grid() -> None:
    cfg = {
        "model_selection": {
            "nested_validation": {
                "enabled": True,
                "true_selection_enabled": True,
                "search": {
                    "candidate_horizons": [3, 5, 10],
                    "feature_views": ["full", "program"],
                    "turnover_penalties": [0.05, 0.15],
                    "cost_penalties": [1.0, 2.0],
                    "max_candidates": 12,
                    "prefilter_top_k": 5,
                },
            }
        }
    }

    nested = _nested_validation_config(cfg)

    assert nested["true_selection_enabled"] is True
    assert nested["search"]["candidate_horizons"] == [3, 5, 10]
    assert nested["search"]["feature_views"] == ["full", "program"]
    assert nested["search"]["max_candidates"] == 12
    assert nested["search"]["prefilter_top_k"] == 5
    assert nested["search"]["allow_cross_family_selection"] is False


def test_nested_validation_config_canonicalizes_horizon_contract() -> None:
    cfg = {
        "model_selection": {
            "nested_validation": {
                "search": {
                    "candidate_horizons": [10],
                    "max_horizons": 3,
                    "feature_views": ["full"],
                }
            }
        }
    }
    contract = build_horizon_contract(cfg, cli_horizon=20)

    nested = _nested_validation_config(cfg, horizon_contract=contract)

    assert nested["search"]["candidate_horizons"] == [20]
    assert nested["search"]["max_horizons"] == 1


def test_nested_candidate_pool_uses_canonical_nested_config_horizon() -> None:
    pool = rms._build_nested_candidate_pool(
        models=[("Ridge", object(), False, "regressor")],
        primary_path="long_only_overlay",
        cfg={
            "model_selection": {
                "nested_validation": {"search": {"candidate_horizons": [10]}}
            }
        },
        nested_cfg={"search": {"candidate_horizons": [20], "max_horizons": 1, "feature_views": ["full"]}},
        default_horizon=20,
        feat_cols=["feature_a"],
        short_feature_subset=[],
        overlay_feature_subset=[],
    )

    assert {candidate.horizon for candidate in pool} == {20}


def test_optimizer_score_weight_audit_detects_score_suppression() -> None:
    dates = pd.bdate_range("2022-01-03", periods=3)
    rows = []
    for dt in dates:
        for idx in range(10):
            rows.append({
                "date": dt,
                "ticker": f"T{idx}",
                "score": float(idx),
                "daily_return": 0.001 * idx,
                "capm_beta": 1.0,
                "sector": "A" if idx < 5 else "B",
            })
    scored = pd.DataFrame(rows)
    target_weights = pd.DataFrame({
        "date": [dates[0]] * 10,
        "ticker": [f"T{idx}" for idx in range(10)],
        "target_weight": [0.5, 0.5] + [0.0] * 8,
    })

    metrics, detail = rms._optimizer_score_weight_audit(
        scored,
        target_weights,
        model_name="AuditModel",
        window_idx=1,
        horizon_days=5,
    )

    assert not detail.empty
    assert metrics["opt_score_weight_rank_corr_mean"] < 0.0
    assert metrics["opt_top_score_long_capture_mean"] == 0.0
    assert metrics["opt_bottom_score_long_leakage_mean"] > 0.0


def test_nested_model_pool_defaults_to_current_stage_c_family() -> None:
    specs = [
        ("Ridge", object(), False, "regressor"),
        ("LGBMRanker", object(), False, "regressor"),
        ("XGBRegressor", object(), False, "regressor"),
    ]

    pool = rms._nested_model_pool_for_outer(
        specs[0],
        ctx_models=tuple(specs),
        nested_cfg={"search": {}},
    )

    assert pool == [specs[0]]


def test_nested_model_pool_requires_explicit_cross_family_opt_in() -> None:
    specs = [
        ("Ridge", object(), False, "regressor"),
        ("LGBMRanker", object(), False, "regressor"),
    ]

    pool = rms._nested_model_pool_for_outer(
        specs[0],
        ctx_models=tuple(specs),
        nested_cfg={"search": {"allow_cross_family_selection": True}},
    )

    assert pool == specs


def test_nested_selection_precomputes_shared_fold_and_validation_state(monkeypatch) -> None:
    dates = pd.bdate_range("2020-01-01", periods=40)
    train_df = pd.DataFrame(
        {
            "date": dates.repeat(3),
            "ticker": ["A", "B", "C"] * len(dates),
            "feature": 1.0,
            "forward_return": 0.01,
            "target_return": 0.01,
            "target_up": 1,
            "target_rank": 0.5,
            "target_down_decile": 0,
            "daily_return": 0.001,
        }
    )

    class DummyPrepared:
        def __init__(self):
            self.train_df = train_df.iloc[:30].copy()
            self.eval_df = train_df.iloc[30:].copy()
            self.x_train = self.train_df[["feature"]].to_numpy()
            self.x_eval = self.eval_df[["feature"]].to_numpy()

    class DummyCache:
        def __init__(self):
            self.fold_calls = 0
            self.state_calls = 0

        def get_prepared_fold(self, **kwargs):
            self.fold_calls += 1
            return DummyPrepared()

        def get_validation_state(self, **kwargs):
            self.state_calls += 1
            return object()

        def get_training_target(self, **kwargs):
            return np.ones(30, dtype=float)

        def stats(self):
            return {}

    pool = [
        rms.NestedCandidateSpec(
            model_name="M1",
            model_kind="regressor",
            uses_proba=False,
            model_template=object(),
            active_features=("feature",),
            feature_view="full",
            horizon=5,
            turnover_penalty=0.05,
            cost_penalty=1.0,
        ),
        rms.NestedCandidateSpec(
            model_name="M2",
            model_kind="regressor",
            uses_proba=False,
            model_template=object(),
            active_features=("feature",),
            feature_view="full",
            horizon=5,
            turnover_penalty=0.15,
            cost_penalty=1.0,
        ),
    ]

    monkeypatch.setattr(rms, "_build_nested_candidate_pool", lambda **kwargs: pool)
    monkeypatch.setattr(
        rms,
        "_nested_inner_windows",
        lambda *args, **kwargs: [
            (
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-02-01"),
                pd.Timestamp("2020-02-01"),
                pd.Timestamp("2020-02-20"),
            )
        ],
    )
    captured = {"workspace_lengths": []}

    def fake_nested_validate_candidate(train_df, **kwargs):
        captured["workspace_lengths"].append(len(kwargs["nested_workspace"]))
        return {
            "nested_selection_score": 1.0,
            "nested_sharpe_mean": 0.5,
            "nested_ic_mean": 0.02,
            "nested_windows": 1,
        }

    monkeypatch.setattr(rms, "_nested_validate_candidate", fake_nested_validate_candidate)

    cache = DummyCache()
    candidate, metrics = rms._select_nested_candidate(
        train_df,
        prepared_cache=cache,
        research_state=None,
        primary_path="long_short_spread",
        models=[],
        cfg={"model_selection": {"nested_validation": {"enabled": True}}},
        nested_cfg={"enabled": True, "max_windows": 1, "validation_days": 10, "min_train_days": 10},
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

    assert candidate is not None
    assert metrics["nested_candidate_count"] == 2
    assert cache.fold_calls == 1
    assert cache.state_calls == 1
    assert captured["workspace_lengths"] == [1, 1]


def test_nested_selection_prefilters_before_full_executable_validation(monkeypatch) -> None:
    dates = pd.bdate_range("2020-01-01", periods=40)
    train_df = pd.DataFrame(
        {
            "date": dates.repeat(3),
            "ticker": ["A", "B", "C"] * len(dates),
            "feature": 1.0,
            "forward_return": 0.01,
            "target_return": 0.01,
            "target_up": 1,
            "target_rank": 0.5,
            "target_down_decile": 0,
            "daily_return": 0.001,
        }
    )

    class DummyPrepared:
        def __init__(self):
            self.train_df = train_df.iloc[:30].copy()
            self.eval_df = train_df.iloc[30:].copy()
            self.x_train = self.train_df[["feature"]].to_numpy()
            self.x_eval = self.eval_df[["feature"]].to_numpy()

    class DummyCache:
        def get_prepared_fold(self, **kwargs):
            return DummyPrepared()

        def get_validation_state(self, **kwargs):
            return object()

        def stats(self):
            return {}

    pool = [
        rms.NestedCandidateSpec("M1", "long_alpha", False, object(), ("feature",), "full", 5, 0.05, 1.0),
        rms.NestedCandidateSpec("M2", "long_alpha", False, object(), ("feature",), "program", 5, 0.10, 1.0),
        rms.NestedCandidateSpec("M3", "long_alpha", False, object(), ("feature",), "full", 10, 0.15, 1.0),
    ]

    monkeypatch.setattr(rms, "_build_nested_candidate_pool", lambda **kwargs: pool)
    monkeypatch.setattr(
        rms,
        "_nested_inner_windows",
        lambda *args, **kwargs: [
            (
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-02-01"),
                pd.Timestamp("2020-02-01"),
                pd.Timestamp("2020-02-20"),
            )
        ],
    )

    proxy_scores = {"M1": 0.9, "M2": 0.6, "M3": 0.1}
    full_evaluated: list[str] = []

    def fake_proxy(*args, **kwargs):
        name = kwargs["name"]
        return {
            "proxy_ic_mean": proxy_scores[name],
            "proxy_daily_icir_mean": proxy_scores[name],
            "proxy_turnover_mean": 0.1,
            "proxy_selection_score": proxy_scores[name],
            "proxy_windows": 1,
        }

    def fake_full(*args, **kwargs):
        name = kwargs["name"]
        full_evaluated.append(name)
        return {
            "nested_selection_score": proxy_scores[name],
            "nested_sharpe_mean": proxy_scores[name],
            "nested_ic_mean": proxy_scores[name] / 10.0,
            "nested_windows": 1,
        }

    monkeypatch.setattr(rms, "_nested_proxy_candidate", fake_proxy)
    monkeypatch.setattr(rms, "_nested_validate_candidate", fake_full)

    candidate, metrics = rms._select_nested_candidate(
        train_df,
        prepared_cache=DummyCache(),
        research_state=None,
        primary_path="long_short_spread",
        models=[],
        cfg={"model_selection": {"nested_validation": {"enabled": True}}},
        nested_cfg={
            "enabled": True,
            "max_windows": 1,
            "validation_days": 10,
            "min_train_days": 10,
            "search": {"prefilter_top_k": 2},
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
        nested_candidate_workers=1,
    )

    assert candidate is not None
    assert candidate.model_name == "M1"
    assert metrics["nested_candidate_count"] == 3
    assert metrics["nested_prefilter_top_k"] == 2
    assert full_evaluated == ["M1", "M2"]


def test_nested_selection_reuses_persisted_candidate_artifacts(tmp_path, monkeypatch) -> None:
    dates = pd.bdate_range("2020-01-01", periods=30)
    train_df = pd.DataFrame(
        {
            "date": dates.repeat(2),
            "ticker": ["A", "B"] * len(dates),
            "feature": 1.0,
            "forward_return": 0.01,
            "target_return": 0.01,
            "target_up": 1,
            "target_rank": 0.5,
            "target_down_decile": 0,
            "daily_return": 0.001,
        }
    )

    class DummyPrepared:
        def __init__(self):
            self.train_df = train_df.iloc[:20].copy()
            self.eval_df = train_df.iloc[20:].copy()
            self.x_train = self.train_df[["feature"]].to_numpy()
            self.x_eval = self.eval_df[["feature"]].to_numpy()

    class DummyCache:
        def get_prepared_fold(self, **kwargs):
            return DummyPrepared()

        def get_validation_state(self, **kwargs):
            return object()

        def stats(self):
            return {}

    pool = [
        rms.NestedCandidateSpec("M1", "long_alpha", False, object(), ("feature",), "full", 5, 0.05, 1.0),
        rms.NestedCandidateSpec("M2", "long_alpha", False, object(), ("feature",), "program", 5, 0.10, 1.0),
    ]

    monkeypatch.setattr(rms, "_build_nested_candidate_pool", lambda **kwargs: pool)
    monkeypatch.setattr(
        rms,
        "_nested_inner_windows",
        lambda *args, **kwargs: [
            (
                pd.Timestamp("2020-01-01"),
                pd.Timestamp("2020-01-20"),
                pd.Timestamp("2020-01-20"),
                pd.Timestamp("2020-02-10"),
            )
        ],
    )
    monkeypatch.setattr(
        rms,
        "_nested_proxy_candidate",
        lambda *args, **kwargs: {
            "proxy_ic_mean": 0.1 if kwargs["name"] == "M1" else 0.05,
            "proxy_daily_icir_mean": 0.2,
            "proxy_turnover_mean": 0.1,
            "proxy_selection_score": 0.1 if kwargs["name"] == "M1" else 0.05,
            "proxy_windows": 1,
        },
    )
    monkeypatch.setattr(
        rms,
        "_nested_validate_candidate",
        lambda *args, **kwargs: {
            "nested_selection_score": 1.0 if kwargs["name"] == "M1" else 0.5,
            "nested_sharpe_mean": 1.0,
            "nested_ic_mean": 0.02,
            "nested_windows": 1,
            "nested_simulation_mode": "proxy_only",
        },
    )

    state = ResearchStateStore(root_dir=tmp_path, namespace="nested_test", payload={"k": 1})
    first_candidate, first_metrics = rms._select_nested_candidate(
        train_df,
        prepared_cache=DummyCache(),
        research_state=state,
        primary_path="long_short_spread",
        models=[],
        cfg={"model_selection": {"nested_validation": {"enabled": True}}},
        nested_cfg={"enabled": True, "max_windows": 1, "validation_days": 10, "min_train_days": 10, "search": {"prefilter_top_k": 1}},
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
        nested_candidate_workers=1,
    )

    assert first_candidate is not None
    assert first_candidate.model_name == "M1"
    assert first_metrics["nested_cache_hit"] == 0.0
    assert first_metrics["nested_simulation_mode"] == "proxy_only"

    for cache_file in state.subdir("nested_selection").glob("*.json"):
        cache_file.unlink()

    monkeypatch.setattr(rms, "_nested_proxy_candidate", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("proxy should be cached")))
    monkeypatch.setattr(rms, "_nested_validate_candidate", lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("full eval should be cached")))

    second_candidate, second_metrics = rms._select_nested_candidate(
        train_df,
        prepared_cache=DummyCache(),
        research_state=state,
        primary_path="long_short_spread",
        models=[],
        cfg={"model_selection": {"nested_validation": {"enabled": True}}},
        nested_cfg={"enabled": True, "max_windows": 1, "validation_days": 10, "min_train_days": 10, "search": {"prefilter_top_k": 1}},
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
        nested_candidate_workers=1,
    )

    assert second_candidate is not None
    assert second_candidate.model_name == "M1"
    assert second_metrics["nested_cache_hit"] == 0.0
    assert second_metrics["nested_full_candidate_cache_hit_rate"] == 1.0
    assert second_metrics["nested_simulation_mode"] == "proxy_only"
