"""Tests for TargetPanelProvider — Phase C.2."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_selection.target_panel_provider import (
    TargetPanelProvider,
    TargetManifest,
    build_target_provider,
    _stable_hash,
    _data_fingerprint,
    _target_config_fingerprint,
    _cost_config_fingerprint,
    TARGET_COLUMNS,
    _PROVIDER_VERSION,
)
from model_selection.training import TargetConfig
from model_selection.validation import ExecutionCostConfig


def _make_panel(n_dates=20, n_tickers=5, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    rows = []
    for dt in dates:
        for i in range(n_tickers):
            rows.append({
                "date": dt,
                "ticker": f"T{i}",
                "forward_return": rng.normal(0.0005, 0.02),
                "capm_beta": 0.8 + i * 0.04,
                "sector": "Tech" if i < n_tickers // 2 else "Industrials",
                "adv_dollar_20": 100_000_000 - i * 1_000_000,
                "realised_vol_20d": 0.02 + i * 0.001,
            })
    return pd.DataFrame(rows)


def _make_panel_with_daily(n_dates=30, n_tickers=5, seed=42):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_dates)
    rows = []
    for dt in dates:
        for i in range(n_tickers):
            rows.append({
                "date": dt,
                "ticker": f"T{i}",
                "daily_return": rng.normal(0.0005, 0.015),
                "capm_beta": 0.8 + i * 0.04,
                "sector": "Tech" if i < n_tickers // 2 else "Industrials",
                "adv_dollar_20": 100_000_000 - i * 1_000_000,
                "realised_vol_20d": 0.02 + i * 0.001,
            })
    return pd.DataFrame(rows)


# ── Fingerprint tests ───────────────────────────────────────────────────────

class TestFingerprints:
    def test_stable_hash_is_deterministic(self):
        h1 = _stable_hash({"a": 1, "b": [2, 3]})
        h2 = _stable_hash({"a": 1, "b": [2, 3]})
        assert h1 == h2

    def test_stable_hash_differs_on_change(self):
        h1 = _stable_hash({"horizon": 5})
        h2 = _stable_hash({"horizon": 10})
        assert h1 != h2

    def test_data_fingerprint_changes_with_values(self):
        df1 = pd.DataFrame({"a": [1.0, 2.0]})
        df2 = pd.DataFrame({"a": [1.0, 3.0]})
        assert _data_fingerprint(df1) != _data_fingerprint(df2)

    def test_data_fingerprint_same_for_same_data(self):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        assert _data_fingerprint(df) == _data_fingerprint(df)

    def test_target_config_fingerprint_differs(self):
        c1 = TargetConfig(horizon_days=5, residualize=True)
        c2 = TargetConfig(horizon_days=10, residualize=True)
        assert _target_config_fingerprint(c1) != _target_config_fingerprint(c2)

    def test_cost_config_fingerprint_differs(self):
        c1 = ExecutionCostConfig(capital=10_000_000)
        c2 = ExecutionCostConfig(capital=20_000_000)
        assert _cost_config_fingerprint(c1) != _cost_config_fingerprint(c2)


# ── Cache key tests ─────────────────────────────────────────────────────────

class TestCacheKey:
    def test_same_input_same_key(self):
        panel = _make_panel()
        cfg = TargetConfig(horizon_days=5)
        costs = ExecutionCostConfig(capital=10_000_000)
        p1 = TargetPanelProvider(panel, target_cfg=cfg, costs=costs, max_name_weight=0.10)
        p2 = TargetPanelProvider(panel.copy(), target_cfg=cfg, costs=costs, max_name_weight=0.10)
        assert p1._cache_key(5) == p2._cache_key(5)

    def test_changed_horizon_changes_key(self):
        panel = _make_panel()
        cfg = TargetConfig(horizon_days=5)
        costs = ExecutionCostConfig(capital=10_000_000)
        provider = TargetPanelProvider(panel, target_cfg=cfg, costs=costs, max_name_weight=0.10)
        k5 = provider._cache_key(5)
        k10 = provider._cache_key(10)
        assert k5 != k10

    def test_changed_target_spec_changes_key(self):
        panel = _make_panel()
        cfg1 = TargetConfig(horizon_days=5, residualize=True)
        cfg2 = TargetConfig(horizon_days=5, residualize=False)
        costs = ExecutionCostConfig(capital=10_000_000)
        p1 = TargetPanelProvider(panel.copy(), target_cfg=cfg1, costs=costs, max_name_weight=0.10)
        p2 = TargetPanelProvider(panel.copy(), target_cfg=cfg2, costs=costs, max_name_weight=0.10)
        assert p1._cache_key(5) != p2._cache_key(5)

    def test_changed_input_values_change_key(self):
        panel1 = _make_panel(seed=42)
        panel2 = _make_panel(seed=99)
        cfg = TargetConfig(horizon_days=5)
        costs = ExecutionCostConfig(capital=10_000_000)
        p1 = TargetPanelProvider(panel1, target_cfg=cfg, costs=costs, max_name_weight=0.10)
        p2 = TargetPanelProvider(panel2, target_cfg=cfg, costs=costs, max_name_weight=0.10)
        assert p1._cache_key(5) != p2._cache_key(5)

    def test_changed_ticker_universe_changes_key(self):
        panel1 = _make_panel(n_tickers=5)
        panel2 = _make_panel(n_tickers=10)
        cfg = TargetConfig(horizon_days=5)
        costs = ExecutionCostConfig(capital=10_000_000)
        p1 = TargetPanelProvider(panel1, target_cfg=cfg, costs=costs, max_name_weight=0.10)
        p2 = TargetPanelProvider(panel2, target_cfg=cfg, costs=costs, max_name_weight=0.10)
        assert p1._cache_key(5) != p2._cache_key(5)

    def test_cache_hit_equals_recompute(self):
        panel = _make_panel()
        cfg = TargetConfig(horizon_days=5)
        costs = ExecutionCostConfig(capital=10_000_000)
        ext_cache: dict = {}
        provider = TargetPanelProvider(
            panel, target_cfg=cfg, costs=costs, max_name_weight=0.10, cache=ext_cache,
        )
        panel1 = provider.get_target_panel(5)
        panel2 = provider.get_target_panel(5)  # Should hit internal cache
        pd.testing.assert_frame_equal(panel1, panel2)


# ── Provider construction tests ─────────────────────────────────────────────

class TestProviderConstruction:
    def test_build_provider_validates_columns(self):
        bad = pd.DataFrame({"foo": [1]})
        cfg = TargetConfig()
        with pytest.raises(ValueError, match="missing required columns"):
            build_target_provider(bad, target_cfg=cfg, costs=ExecutionCostConfig(), max_name_weight=0.10)

    def test_build_provider_validates_return_column(self):
        bad = pd.DataFrame({"date": ["2021-01-01"], "ticker": ["A"]})
        cfg = TargetConfig()
        with pytest.raises(ValueError, match="forward_return.*daily_return"):
            build_target_provider(bad, target_cfg=cfg, costs=ExecutionCostConfig(), max_name_weight=0.10)

    def test_build_provider_succeeds_with_forward_return(self):
        panel = pd.DataFrame({
            "date": ["2021-01-01"],
            "ticker": ["A"],
            "forward_return": [0.01],
        })
        provider = build_target_provider(
            panel, target_cfg=TargetConfig(), costs=ExecutionCostConfig(), max_name_weight=0.10,
        )
        assert isinstance(provider, TargetPanelProvider)

    def test_build_provider_succeeds_with_daily_return(self):
        panel = pd.DataFrame({
            "date": ["2021-01-01"],
            "ticker": ["A"],
            "daily_return": [0.005],
        })
        provider = build_target_provider(
            panel, target_cfg=TargetConfig(), costs=ExecutionCostConfig(), max_name_weight=0.10,
        )
        assert isinstance(provider, TargetPanelProvider)


# ── Target panel tests ──────────────────────────────────────────────────────

class TestTargetPanel:
    def test_panel_has_correct_row_count(self):
        panel = _make_panel(n_dates=20, n_tickers=5)
        cfg = TargetConfig(horizon_days=5)
        costs = ExecutionCostConfig(capital=10_000_000)
        provider = TargetPanelProvider(panel, target_cfg=cfg, costs=costs, max_name_weight=0.10)
        result = provider.get_target_panel(5)
        assert len(result) == len(panel)

    def test_panel_has_target_columns(self):
        panel = _make_panel(n_dates=20, n_tickers=5)
        cfg = TargetConfig(horizon_days=5)
        costs = ExecutionCostConfig(capital=10_000_000)
        provider = TargetPanelProvider(panel, target_cfg=cfg, costs=costs, max_name_weight=0.10)
        result = provider.get_target_panel(5)
        expected = {"target_return", "target_rank", "target_down_decile", "target_up", "target_return_net"}
        assert expected.issubset(set(result.columns))

    def test_panel_is_sorted_by_date_ticker(self):
        panel = _make_panel(n_dates=20, n_tickers=5)
        cfg = TargetConfig(horizon_days=5)
        costs = ExecutionCostConfig(capital=10_000_000)
        provider = TargetPanelProvider(panel, target_cfg=cfg, costs=costs, max_name_weight=0.10)
        result = provider.get_target_panel(5)
        dates = result["date"].values
        tickers = result["ticker"].values
        # Check sorted
        for i in range(1, len(result)):
            assert (dates[i] > dates[i-1]) or (dates[i] == dates[i-1] and tickers[i] >= tickers[i-1])

    def test_multiple_horizons_produce_different_panels(self):
        panel = _make_panel_with_daily(n_dates=30, n_tickers=5)
        cfg = TargetConfig(horizon_days=5, residualize=True)
        costs = ExecutionCostConfig(capital=10_000_000)
        provider = TargetPanelProvider(
            panel, target_cfg=cfg, costs=costs, max_name_weight=0.10, horizons=[5, 10],
        )
        p5 = provider.get_target_panel(5)
        p10 = provider.get_target_panel(10)
        assert len(p5) == len(p10)
        # Different horizons produce different target values
        assert not np.allclose(
            p5["target_return"].values,
            p10["target_return"].values,
            equal_nan=True,
        )

    def test_get_target_columns_returns_subset(self):
        panel = _make_panel(n_dates=20, n_tickers=5)
        cfg = TargetConfig(horizon_days=5)
        costs = ExecutionCostConfig(capital=10_000_000)
        provider = TargetPanelProvider(panel, target_cfg=cfg, costs=costs, max_name_weight=0.10)
        cols_df = provider.get_target_columns(5)
        expected_cols = {"date", "ticker"} | (TARGET_COLUMNS & set(cols_df.columns))
        assert set(cols_df.columns) == expected_cols

    def test_has_horizon(self):
        panel = _make_panel()
        cfg = TargetConfig(horizon_days=5)
        costs = ExecutionCostConfig()
        provider = TargetPanelProvider(panel, target_cfg=cfg, costs=costs, max_name_weight=0.10, horizons=[5, 10, 20])
        assert provider.has_horizon(5)
        assert provider.has_horizon(10)
        assert provider.has_horizon(20)
        assert not provider.has_horizon(3)

    def test_external_cache_hit(self):
        panel = _make_panel()
        cfg = TargetConfig(horizon_days=5)
        costs = ExecutionCostConfig()
        ext_cache: dict = {}
        provider = TargetPanelProvider(
            panel, target_cfg=cfg, costs=costs, max_name_weight=0.10, cache=ext_cache,
        )
        provider.get_target_panel(5)
        assert len(ext_cache) == 1
        # Second provider with same config should hit external cache
        provider2 = TargetPanelProvider(
            panel.copy(), target_cfg=cfg, costs=costs, max_name_weight=0.10, cache=ext_cache,
        )
        p = provider2.get_target_panel(5)
        telemetry = provider2.get_telemetry()
        assert any(t.cache_status == "hit" for t in telemetry)


# ── Manifest tests ──────────────────────────────────────────────────────────

class TestManifest:
    def test_manifest_has_required_fields(self):
        panel = _make_panel(n_dates=20, n_tickers=5)
        cfg = TargetConfig(horizon_days=5)
        costs = ExecutionCostConfig(capital=10_000_000)
        provider = TargetPanelProvider(panel, target_cfg=cfg, costs=costs, max_name_weight=0.10)
        manifest = provider.get_manifest()
        assert isinstance(manifest, TargetManifest)
        assert manifest.provider_version == _PROVIDER_VERSION
        assert manifest.row_count > 0
        assert manifest.date_count > 0
        assert manifest.ticker_count > 0
        assert len(manifest.target_columns) > 0
        assert len(manifest.missingness_report) > 0

    def test_manifest_fingerprints_are_stable(self):
        panel = _make_panel(n_dates=20, n_tickers=5)
        cfg = TargetConfig(horizon_days=5)
        costs = ExecutionCostConfig(capital=10_000_000)
        p1 = TargetPanelProvider(panel.copy(), target_cfg=cfg, costs=costs, max_name_weight=0.10)
        p2 = TargetPanelProvider(panel.copy(), target_cfg=cfg, costs=costs, max_name_weight=0.10)
        m1 = p1.get_manifest()
        m2 = p2.get_manifest()
        assert m1.data_fingerprint == m2.data_fingerprint
        assert m1.target_spec_fingerprints == m2.target_spec_fingerprints


# ── Telemetry tests ─────────────────────────────────────────────────────────

class TestTelemetry:
    def test_telemetry_records_build_events(self):
        panel = _make_panel()
        cfg = TargetConfig(horizon_days=5)
        costs = ExecutionCostConfig()
        provider = TargetPanelProvider(panel, target_cfg=cfg, costs=costs, max_name_weight=0.10)
        provider.get_target_panel(5)
        events = provider.get_telemetry()
        assert len(events) >= 1
        assert events[0].stage == "build_panel"
        assert events[0].cache_status == "miss"
        assert events[0].rows > 0
        assert events[0].elapsed_seconds >= 0

    def test_telemetry_records_cache_hits(self):
        panel = _make_panel()
        cfg = TargetConfig(horizon_days=5)
        costs = ExecutionCostConfig()
        ext_cache: dict = {}
        provider = TargetPanelProvider(
            panel, target_cfg=cfg, costs=costs, max_name_weight=0.10, cache=ext_cache,
        )
        provider.get_target_panel(5)
        provider.get_target_panel(5)  # internal cache hit (no new event)
        events = provider.get_telemetry()
        assert len(events) == 1  # Only one build event


# ── Equivalence with legacy path ────────────────────────────────────────────

class TestLegacyEquivalence:
    def test_panel_equals_direct_retarget(self):
        """TargetPanelProvider output must equal direct retarget_panel_for_horizon."""
        from model_selection.training import retarget_panel_for_horizon

        panel = _make_panel(n_dates=20, n_tickers=5)
        cfg = TargetConfig(horizon_days=5, residualize=True)
        costs = ExecutionCostConfig(capital=10_000_000)

        # Legacy path
        legacy = retarget_panel_for_horizon(
            panel,
            horizon_days=5,
            target_cfg=cfg,
            costs=costs,
            max_name_weight=0.10,
        )

        # Provider path
        provider = TargetPanelProvider(
            panel.copy(), target_cfg=cfg, costs=costs, max_name_weight=0.10,
        )
        from_provider = provider.get_target_panel(5)

        # Compare
        assert len(from_provider) == len(legacy)
        assert set(from_provider.columns) == set(legacy.columns)

        # Sort both for comparison
        legacy = legacy.sort_values(["date", "ticker"]).reset_index(drop=True)
        from_provider = from_provider.sort_values(["date", "ticker"]).reset_index(drop=True)

        for col in TARGET_COLUMNS & set(legacy.columns):
            np.testing.assert_array_almost_equal(
                from_provider[col].values,
                legacy[col].values,
                decimal=12,
                err_msg=f"Column {col} differs between provider and legacy",
            )

    def test_nan_masks_match(self):
        """NaN locations must be identical between provider and legacy."""
        from model_selection.training import retarget_panel_for_horizon

        panel = _make_panel_with_daily(n_dates=30, n_tickers=5)
        cfg = TargetConfig(horizon_days=5, residualize=True)
        costs = ExecutionCostConfig(capital=10_000_000)

        legacy = retarget_panel_for_horizon(
            panel,
            horizon_days=5,
            target_cfg=cfg,
            costs=costs,
            max_name_weight=0.10,
        )

        provider = TargetPanelProvider(
            panel.copy(), target_cfg=cfg, costs=costs, max_name_weight=0.10,
        )
        from_provider = provider.get_target_panel(5)

        legacy = legacy.sort_values(["date", "ticker"]).reset_index(drop=True)
        from_provider = from_provider.sort_values(["date", "ticker"]).reset_index(drop=True)

        for col in TARGET_COLUMNS & set(legacy.columns):
            legacy_nan = legacy[col].isna()
            provider_nan = from_provider[col].isna()
            assert (legacy_nan == provider_nan).all(), f"NaN mask mismatch for {col}"


# ── PreparedPanelCache integration ──────────────────────────────────────────

class TestPreparedPanelCacheIntegration:
    def test_cache_uses_provider_when_passed(self):
        """PreparedPanelCache should use TargetPanelProvider when available."""
        from model_selection.preparation import PreparedPanelCache

        panel = _make_panel(n_dates=20, n_tickers=5)
        cfg = TargetConfig(horizon_days=5, residualize=True)
        costs = ExecutionCostConfig(capital=10_000_000)

        provider = TargetPanelProvider(
            panel.copy(), target_cfg=cfg, costs=costs, max_name_weight=0.10,
        )
        cache = PreparedPanelCache(
            panel,
            target_cfg=cfg,
            costs=costs,
            max_name_weight=0.10,
            winsor_q=0.01,
            target_provider=provider,
        )
        result = cache.get_full_retargeted_panel(5)
        assert len(result) == len(panel)
        assert "target_return" in result.columns
        # Provider should have been used (telemetry should show a build event)
        assert len(provider.get_telemetry()) >= 1

    def test_cache_falls_back_without_provider(self):
        """PreparedPanelCache should work normally without TargetPanelProvider."""
        from model_selection.preparation import PreparedPanelCache

        panel = _make_panel(n_dates=20, n_tickers=5)
        cfg = TargetConfig(horizon_days=5, residualize=True)
        costs = ExecutionCostConfig(capital=10_000_000)

        cache = PreparedPanelCache(
            panel,
            target_cfg=cfg,
            costs=costs,
            max_name_weight=0.10,
            winsor_q=0.01,
        )
        result = cache.get_full_retargeted_panel(5)
        assert len(result) == len(panel)
        assert "target_return" in result.columns

    def test_provider_and_cache_produce_same_output(self):
        """Direct provider output and cache output must be identical."""
        from model_selection.preparation import PreparedPanelCache

        panel = _make_panel(n_dates=20, n_tickers=5)
        cfg = TargetConfig(horizon_days=5, residualize=True)
        costs = ExecutionCostConfig(capital=10_000_000)

        provider = TargetPanelProvider(
            panel.copy(), target_cfg=cfg, costs=costs, max_name_weight=0.10,
        )
        cache = PreparedPanelCache(
            panel.copy(),
            target_cfg=cfg,
            costs=costs,
            max_name_weight=0.10,
            winsor_q=0.01,
            target_provider=provider,
        )

        from_provider = provider.get_target_panel(5)
        from_cache = cache.get_full_retargeted_panel(5)

        from_provider = from_provider.sort_values(["date", "ticker"]).reset_index(drop=True)
        from_cache = from_cache.sort_values(["date", "ticker"]).reset_index(drop=True)

        for col in TARGET_COLUMNS & set(from_provider.columns):
            np.testing.assert_array_almost_equal(
                from_provider[col].values,
                from_cache[col].values,
                decimal=12,
            )
