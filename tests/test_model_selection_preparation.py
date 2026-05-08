from __future__ import annotations

import numpy as np
import pandas as pd

import model_selection.preparation as preparation
from model_selection.preparation import PreparedPanelCache
from model_selection.training import TargetConfig
from model_selection.validation import EvaluationConfig, ExecutionCostConfig


class _DummyPreprocessor:
    def __init__(self, features):
        self.active_features = list(features)

    def transform(self, df):
        return df[self.active_features].to_numpy(dtype=float)


def _base_panel() -> pd.DataFrame:
    dates = pd.bdate_range("2020-01-01", periods=6)
    rows: list[dict[str, object]] = []
    for ticker, offset in [("A", 0.0), ("B", 1.0)]:
        for idx, dt in enumerate(dates):
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "feature": float(idx + offset),
                    "forward_return": 0.01 * (idx + 1),
                    "target_return": 0.01 * (idx + 1),
                    "target_up": int((idx % 2) == 0),
                    "target_rank": float(idx) / 10.0,
                    "target_down_decile": int((idx % 3) == 0),
                    "daily_return": 0.001 * (idx + 1),
                }
            )
    return pd.DataFrame(rows)


def test_prepared_panel_cache_reuses_retarget_and_preprocessing(monkeypatch) -> None:
    retarget_calls: list[tuple[pd.Timestamp, pd.Timestamp, int]] = []
    fit_calls: list[tuple[str, ...]] = []

    def fake_retarget_panel_for_horizon(df, *, horizon_days, target_cfg, costs, max_name_weight):
        retarget_calls.append(
            (
                pd.Timestamp(df["date"].min()),
                pd.Timestamp(df["date"].max()),
                int(horizon_days),
            )
        )
        out = df.copy()
        out["target_return"] = pd.to_numeric(out["forward_return"], errors="coerce").fillna(0.0)
        out["target_up"] = (out["target_return"] > 0.0).astype(int)
        out["target_rank"] = out.groupby("date")["target_return"].rank(pct=True, method="average").fillna(0.0)
        out["target_down_decile"] = 0
        return out

    def fake_fit(cls, df, feature_columns, *, winsor_q=0.01, min_std=1e-6):
        fit_calls.append(tuple(feature_columns))
        return _DummyPreprocessor(feature_columns)

    monkeypatch.setattr(preparation, "retarget_panel_for_horizon", fake_retarget_panel_for_horizon)
    monkeypatch.setattr(preparation.FeaturePreprocessor, "fit", classmethod(fake_fit))

    cache = PreparedPanelCache(
        _base_panel(),
        target_cfg=TargetConfig(),
        costs=ExecutionCostConfig(),
        max_name_weight=0.1,
        winsor_q=0.01,
    )

    first = cache.get_prepared_fold(
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2020-01-06"),
        eval_start=pd.Timestamp("2020-01-06"),
        eval_end=pd.Timestamp("2020-01-09"),
        horizon_days=5,
        active_features=["feature"],
    )
    second = cache.get_prepared_fold(
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2020-01-06"),
        eval_start=pd.Timestamp("2020-01-06"),
        eval_end=pd.Timestamp("2020-01-09"),
        horizon_days=5,
        active_features=["feature"],
    )

    assert first is second
    assert len(retarget_calls) == 1
    assert len(fit_calls) == 1


def test_prepared_panel_cache_reuses_training_targets(monkeypatch) -> None:
    target_calls: list[tuple[str, str, bool]] = []

    def fake_make_training_target(df, *, model_name, model_kind, use_risk_adj):
        target_calls.append((str(model_name), str(model_kind), bool(use_risk_adj)))
        return np.arange(len(df), dtype=float)

    monkeypatch.setattr(preparation, "make_training_target", fake_make_training_target)

    cache = PreparedPanelCache(
        _base_panel(),
        target_cfg=TargetConfig(),
        costs=ExecutionCostConfig(),
        max_name_weight=0.1,
        winsor_q=0.01,
    )

    first = cache.get_training_target(
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-01-06"),
        horizon_days=5,
        model_name="Ridge",
        model_kind="regressor",
        use_risk_adj=False,
    )
    second = cache.get_training_target(
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-01-06"),
        horizon_days=5,
        model_name="Ridge",
        model_kind="regressor",
        use_risk_adj=False,
    )

    assert np.array_equal(first, second)
    assert len(target_calls) == 1


def test_validation_state_cache_accepts_nested_dict_config() -> None:
    cache = PreparedPanelCache(
        _base_panel(),
        target_cfg=TargetConfig(),
        costs=ExecutionCostConfig(),
        max_name_weight=0.1,
        winsor_q=0.01,
    )
    cfg = EvaluationConfig(
        path="long_short_spread",
        style_exposure_limits={"size": 0.10, "momentum": 0.20},
    )

    first = cache.get_validation_state(
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-01-09"),
        horizon_days=5,
        evaluation_cfg=cfg,
    )
    second = cache.get_validation_state(
        start=pd.Timestamp("2020-01-01"),
        end=pd.Timestamp("2020-01-09"),
        horizon_days=5,
        evaluation_cfg=cfg,
    )

    assert first is second


def test_prepared_panel_cache_reports_structurally_unique_fold_topology(monkeypatch) -> None:
    def fake_retarget_panel_for_horizon(df, *, horizon_days, target_cfg, costs, max_name_weight):
        out = df.copy()
        out["target_return"] = pd.to_numeric(out["forward_return"], errors="coerce").fillna(0.0)
        out["target_up"] = (out["target_return"] > 0.0).astype(int)
        out["target_rank"] = out.groupby("date")["target_return"].rank(pct=True, method="average").fillna(0.0)
        out["target_down_decile"] = 0
        return out

    def fake_fit(cls, df, feature_columns, *, winsor_q=0.01, min_std=1e-6):
        return _DummyPreprocessor(feature_columns)

    monkeypatch.setattr(preparation, "retarget_panel_for_horizon", fake_retarget_panel_for_horizon)
    monkeypatch.setattr(preparation.FeaturePreprocessor, "fit", classmethod(fake_fit))

    cache = PreparedPanelCache(
        _base_panel(),
        target_cfg=TargetConfig(),
        costs=ExecutionCostConfig(),
        max_name_weight=0.1,
        winsor_q=0.01,
        max_cache_size=1,
    )
    for offset in range(3):
        cache.get_prepared_fold(
            train_start=pd.Timestamp("2020-01-01"),
            train_end=pd.Timestamp("2020-01-03") + pd.Timedelta(days=offset),
            eval_start=pd.Timestamp("2020-01-03") + pd.Timedelta(days=offset),
            eval_end=pd.Timestamp("2020-01-09"),
            horizon_days=5,
            active_features=["feature"],
        )

    stats = cache.stats()

    assert stats["prepared_fold_lookups"] == 3
    assert stats["prepared_fold_unique_keys"] == 3
    assert stats["prepared_fold_structurally_unique"] is True
    assert stats["prepared_fold_cache_capacity"] == 1


def test_prepared_panel_cache_persists_artifacts_to_disk(tmp_path, monkeypatch) -> None:
    retarget_calls: list[int] = []

    def fake_retarget_panel_for_horizon(df, *, horizon_days, target_cfg, costs, max_name_weight):
        retarget_calls.append(int(horizon_days))
        out = df.copy()
        out["target_return"] = pd.to_numeric(out["forward_return"], errors="coerce").fillna(0.0)
        out["target_up"] = (out["target_return"] > 0.0).astype(int)
        out["target_rank"] = out.groupby("date")["target_return"].rank(pct=True, method="average").fillna(0.0)
        out["target_down_decile"] = 0
        return out

    def fake_fit(cls, df, feature_columns, *, winsor_q=0.01, min_std=1e-6):
        return _DummyPreprocessor(feature_columns)

    monkeypatch.setattr(preparation, "retarget_panel_for_horizon", fake_retarget_panel_for_horizon)
    monkeypatch.setattr(preparation.FeaturePreprocessor, "fit", classmethod(fake_fit))

    first_cache = PreparedPanelCache(
        _base_panel(),
        target_cfg=TargetConfig(),
        costs=ExecutionCostConfig(),
        max_name_weight=0.1,
        winsor_q=0.01,
        artifact_dir=tmp_path / "prepared",
        min_free_space_mb=0.1,
    )
    first_cache.get_prepared_fold(
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2020-01-06"),
        eval_start=pd.Timestamp("2020-01-06"),
        eval_end=pd.Timestamp("2020-01-09"),
        horizon_days=5,
        active_features=["feature"],
    )
    assert len(retarget_calls) == 1

    second_cache = PreparedPanelCache(
        _base_panel(),
        target_cfg=TargetConfig(),
        costs=ExecutionCostConfig(),
        max_name_weight=0.1,
        winsor_q=0.01,
        artifact_dir=tmp_path / "prepared",
        min_free_space_mb=0.1,
    )
    second_cache.get_prepared_fold(
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2020-01-06"),
        eval_start=pd.Timestamp("2020-01-06"),
        eval_end=pd.Timestamp("2020-01-09"),
        horizon_days=5,
        active_features=["feature"],
    )
    second_stats = second_cache.stats()
    assert len(retarget_calls) == 1
    assert second_stats["horizon_artifact_hits"] == 1
    assert second_stats["effective_cache_hit_rate"] > second_stats["prepared_fold_cache_hit_rate"]
    assert list((tmp_path / "prepared").glob("horizon_panel_*.pkl"))
    assert not list((tmp_path / "prepared").glob("prepared_fold_*.pkl"))


def test_prepared_panel_cache_disables_artifact_writes_when_disk_reserve_is_breached(tmp_path, monkeypatch) -> None:
    def fake_retarget_panel_for_horizon(df, *, horizon_days, target_cfg, costs, max_name_weight):
        out = df.copy()
        out["target_return"] = pd.to_numeric(out["forward_return"], errors="coerce").fillna(0.0)
        out["target_up"] = (out["target_return"] > 0.0).astype(int)
        out["target_rank"] = out.groupby("date")["target_return"].rank(pct=True, method="average").fillna(0.0)
        out["target_down_decile"] = 0
        return out

    def fake_fit(cls, df, feature_columns, *, winsor_q=0.01, min_std=1e-6):
        return _DummyPreprocessor(feature_columns)

    monkeypatch.setattr(preparation, "retarget_panel_for_horizon", fake_retarget_panel_for_horizon)
    monkeypatch.setattr(preparation.FeaturePreprocessor, "fit", classmethod(fake_fit))

    usage_type = type(preparation.shutil.disk_usage(tmp_path))
    monkeypatch.setattr(
        preparation.shutil,
        "disk_usage",
        lambda _: usage_type(total=10_000_000, used=9_500_000, free=500_000),
    )

    cache = PreparedPanelCache(
        _base_panel(),
        target_cfg=TargetConfig(),
        costs=ExecutionCostConfig(),
        max_name_weight=0.1,
        winsor_q=0.01,
        artifact_dir=tmp_path / "prepared",
        min_free_space_mb=2.0,
    )
    fold = cache.get_prepared_fold(
        train_start=pd.Timestamp("2020-01-01"),
        train_end=pd.Timestamp("2020-01-06"),
        eval_start=pd.Timestamp("2020-01-06"),
        eval_end=pd.Timestamp("2020-01-09"),
        horizon_days=5,
        active_features=["feature"],
    )

    stats = cache.stats()
    assert len(fold.train_df) > 0
    assert stats["artifact_writes_enabled"] is False
    assert str(stats["artifact_disabled_reason"]).startswith("free_space_below_reserve:")


def test_prepared_panel_cache_runtime_stats_reset_preserves_cached_objects(monkeypatch) -> None:
    retarget_calls: list[int] = []

    def fake_retarget_panel_for_horizon(df, *, horizon_days, target_cfg, costs, max_name_weight):
        retarget_calls.append(int(horizon_days))
        out = df.copy()
        out["target_return"] = pd.to_numeric(out["forward_return"], errors="coerce").fillna(0.0)
        out["target_up"] = (out["target_return"] > 0.0).astype(int)
        out["target_rank"] = out.groupby("date")["target_return"].rank(pct=True, method="average").fillna(0.0)
        out["target_down_decile"] = 0
        return out

    monkeypatch.setattr(preparation, "retarget_panel_for_horizon", fake_retarget_panel_for_horizon)

    cache = PreparedPanelCache(
        _base_panel(),
        target_cfg=TargetConfig(),
        costs=ExecutionCostConfig(),
        max_name_weight=0.1,
        winsor_q=0.01,
        max_cache_size=2,
    )
    cache.get_full_retargeted_panel(5)
    assert cache.stats()["misses"] == 1
    cache.reset_runtime_stats()
    assert cache.stats()["total_lookups"] == 0
    cache.get_full_retargeted_panel(5)
    assert cache.stats()["horizon_memory_hits"] == 1
    assert len(retarget_calls) == 1
