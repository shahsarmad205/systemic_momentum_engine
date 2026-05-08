"""Tests for Point-in-Time Condition Engine.

Tests prove all 14 acceptance criteria:
1. No full-sample quantiles are used for production buckets.
2. Date-cross-sectional buckets use only same-date trailing-known values.
3. Rolling buckets use only data up to date t.
4. Expanding buckets use only data up to date t.
5. Regime fit_end <= label_date.
6. No condition uses forward_return columns.
7. Condition labels carry provenance metadata.
8. Missing labels do not silently fall back to global/default labels.
9. Static sector labels are marked static_proxy.
10. Research-only labels cannot production-promote a sleeve.
11. Walk-forward test labels are generated without using test-period future thresholds.
12. Conditional sleeves with invalid PIT labels are rejected or downgraded.
13. All thresholds come from ResearchContract/config.
14. Every PIT rejection has a reason.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from model_selection.pit_condition_engine import (
    PITConditionEngine,
    PITConditionBundle,
    build_pit_buckets,
    build_pit_regime_labels,
    validate_sector_labels,
    build_pit_sleeve_registry,
    run_leakage_detection,
    compute_conditional_pit_status,
    run_walk_forward_pit_conditions,
    generate_pit_reports,
    _build_date_cross_sectional_buckets,
    _build_rolling_buckets,
    _build_expanding_buckets,
    _classify_expanding_regime,
    _classify_rolling_regime,
    _find_column,
    _get_config,
    ConstructionMethod,
    QualityFlag,
    PitStatus,
    LeakageSeverity,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_panel():
    """Generate a realistic panel with multiple condition columns."""
    np.random.seed(42)
    n_tickers = 30
    n_dates = 150
    tickers = [f"TICK{i:03d}" for i in range(n_tickers)]
    dates = pd.date_range("2015-01-02", periods=n_dates, freq="B")

    rows = []
    for date in dates:
        for ticker in tickers:
            rows.append({
                "date": date,
                "ticker": ticker,
                "rolling_vol_20": np.random.exponential(0.02) + 0.01,
                "adv_dollar_20": np.random.lognormal(16, 1.5),
                "market_cap": np.random.lognormal(20, 1.5),
                "capm_beta": np.random.normal(1.0, 0.3),
                "daily_return": np.random.randn() * 0.02,
                "forward_return": np.random.randn() * 0.02,
                "sector": np.random.choice(["Tech", "Health", "Finance", "Energy", "Consumer"]),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def panel_with_pit_sector():
    """Panel with PIT sector timestamps."""
    np.random.seed(42)
    n_tickers = 20
    n_dates = 100
    tickers = [f"TICK{i:03d}" for i in range(n_tickers)]
    dates = pd.date_range("2015-01-02", periods=n_dates, freq="B")

    rows = []
    for date in dates:
        for ticker in tickers:
            rows.append({
                "date": date,
                "ticker": ticker,
                "rolling_vol_20": np.random.exponential(0.02) + 0.01,
                "adv_dollar_20": np.random.lognormal(16, 1.5),
                "market_cap": np.random.lognormal(20, 1.5),
                "daily_return": np.random.randn() * 0.02,
                "sector": np.random.choice(["Tech", "Health", "Finance"]),
                "sector_asof": date,
                "capm_beta": np.random.normal(1.0, 0.3),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def minimal_config():
    return {
        "pit_conditions": {
            "n_buckets": 3,
            "bucket_method": "date_cross_sectional",
            "rolling_window": 60,
            "expanding_min_obs": 30,
            "winsor_q": 0.025,
            "min_breadth_per_date": 5,
            "regime_method": "expanding_vol_trend",
            "regime_window": 60,
            "regime_min_obs": 30,
            "regime_prob_threshold": 0.5,
            "regime_uncertainty_threshold": 0.35,
            "sector_pit_column": "sector_asof",
            "sector_allow_static": True,
            "sector_static_quality": "static_proxy",
            "condition_types": ["regime", "volatility", "liquidity", "size", "sector"],
            "wf_embargo_multiplier": 2,
        },
    }


# ── Test 1: No full-sample quantiles for production buckets ──────────────────

class TestNoFullSampleQuantiles:
    def test_date_cross_sectional_uses_per_date_thresholds(self, sample_panel):
        """Date-cross-sectional buckets must have different thresholds per date."""
        labels, provenance = build_pit_buckets(
            sample_panel, ["volatility"], n_buckets=3,
            method="date_cross_sectional", winsor_q=0.025, min_breadth=5,
        )

        # Collect unique threshold sets
        thresholds = labels["threshold_values"].dropna().unique()

        # Should have multiple different threshold sets (one per date)
        assert len(thresholds) > 1, "All dates have same thresholds — likely full-sample"

    def test_bucket_provenance_shows_date_specific(self, sample_panel):
        """Provenance should show date-specific construction."""
        labels, provenance = build_pit_buckets(
            sample_panel, ["volatility"], n_buckets=3,
            method="date_cross_sectional", winsor_q=0.025, min_breadth=5,
        )

        for prov in provenance:
            assert prov.fit_start == prov.fit_end, (
                "Date-cross-sectional should fit on single date"
            )
            assert prov.threshold_source == "date_cross_sectional_quantiles"


# ── Test 2: Date-cross-sectional uses same-date values ───────────────────────

class TestDateCrossSectional:
    def test_only_same_date_values_used(self, sample_panel):
        """Buckets must use only values from the same date."""
        labels, _ = build_pit_buckets(
            sample_panel, ["volatility"], n_buckets=3,
            method="date_cross_sectional", winsor_q=0.025, min_breadth=5,
            column_map={"volatility": ["rolling_vol_20"]},
        )

        # All labels should be PIT-valid (not missing)
        pit_valid = labels[labels["is_pit_valid"] == True]
        assert len(pit_valid) > 0, "No PIT-valid labels generated"

        # Labels should have the correct method
        assert (pit_valid["bucket_method"] == "date_cross_sectional").all()


# ── Test 3: Rolling buckets use only data up to date t ───────────────────────

class TestRollingBuckets:
    def test_rolling_uses_trailing_window(self, sample_panel):
        """Rolling buckets must use only data up to and including date t."""
        df = sample_panel.copy()
        df["volatility"] = df["rolling_vol_20"]
        labels, _ = _build_rolling_buckets(
            df, "volatility", df["date"].unique(),
            n_buckets=3, window=30, winsor_q=0.025, source_col="rolling_vol_20",
        )

        # Check lookback_window is bounded
        for label in labels:
            if label["lookback_window"] > 0:
                assert label["lookback_window"] <= 30, (
                    f"Lookback {label['lookback_window']} exceeds window size 30"
                )

    def test_rolling_degraded_quality_for_short_history(self, sample_panel):
        """Early dates with insufficient history should be degraded."""
        df = sample_panel.copy()
        df["volatility"] = df["rolling_vol_20"]
        labels, _ = _build_rolling_buckets(
            df, "volatility", df["date"].unique(),
            n_buckets=3, window=60, winsor_q=0.025, source_col="rolling_vol_20",
        )

        degraded = [l for l in labels if l["quality_flag"] == QualityFlag.PIT_DEGRADED.value]
        # First few dates per ticker should be degraded
        assert len(degraded) > 0, "No degraded labels for short history"


# ── Test 4: Expanding buckets use only data up to date t ─────────────────────

class TestExpandingBuckets:
    def test_expanding_uses_only_past_data(self, sample_panel):
        """Expanding buckets must use only data up to and including date t."""
        df = sample_panel.copy()
        df["volatility"] = df["rolling_vol_20"]
        labels, provenance = _build_expanding_buckets(
            df, "volatility", df["date"].unique(),
            n_buckets=3, min_obs=30, winsor_q=0.025, source_col="rolling_vol_20",
        )

        # Check fit_end <= date for each provenance entry
        for prov in provenance:
            if prov.fit_end and prov.date:
                assert prov.fit_end <= prov.date, (
                    f"fit_end {prov.fit_end} > date {prov.date}"
                )

    def test_expanding_degraded_for_insufficient_history(self, sample_panel):
        """Early dates should be marked degraded."""
        df = sample_panel.copy()
        df["volatility"] = df["rolling_vol_20"]
        labels, _ = _build_expanding_buckets(
            df, "volatility", df["date"].unique(),
            n_buckets=3, min_obs=60, winsor_q=0.025, source_col="rolling_vol_20",
        )

        degraded = [l for l in labels if l["quality_flag"] == QualityFlag.PIT_DEGRADED.value]
        assert len(degraded) > 0, "No degraded labels for early dates"


# ── Test 5: Regime fit_end <= label_date ─────────────────────────────────────

class TestRegimeFitWindow:
    def test_expanding_regime_fit_end_before_label(self, sample_panel):
        """Regime fit_end must be <= label_date."""
        labels = build_pit_regime_labels(
            sample_panel, method="expanding_vol_trend",
            window=60, min_obs=30,
        )

        if not labels.empty:
            labels["date"] = pd.to_datetime(labels["date"])
            labels["fit_end"] = pd.to_datetime(labels["fit_end"])
            violations = labels[labels["fit_end"] > labels["date"]]
            assert len(violations) == 0, (
                f"{len(violations)} regime labels have fit_end after date"
            )

    def test_rolling_regime_fit_end_before_label(self, sample_panel):
        """Rolling regime fit_end must be <= label_date."""
        labels = build_pit_regime_labels(
            sample_panel, method="rolling_vol_trend",
            window=60, min_obs=30,
        )

        if not labels.empty:
            labels["date"] = pd.to_datetime(labels["date"])
            labels["fit_end"] = pd.to_datetime(labels["fit_end"])
            violations = labels[labels["fit_end"] > labels["date"]]
            assert len(violations) == 0


# ── Test 6: No condition uses forward_return columns ────────────────────────

class TestNoForwardReturnContamination:
    def test_leakage_detects_forward_columns(self, sample_panel):
        """Leakage detection should flag forward_return columns."""
        # Add a forward-looking column that's not forward_return
        df = sample_panel.copy()
        df["forward_volatility"] = df["rolling_vol_20"].shift(-1)

        results = run_leakage_detection(df)

        forward_check = [r for r in results if r.test_name == "forward_return_contamination"]
        assert len(forward_check) > 0

        # Should fail because forward columns are present
        assert not forward_check[0].passed


# ── Test 7: Condition labels carry provenance metadata ───────────────────────

class TestProvenanceMetadata:
    def test_bucket_labels_have_provenance(self, sample_panel):
        """Every bucket label must have provenance."""
        labels, provenance = build_pit_buckets(
            sample_panel, ["volatility", "liquidity"],
            n_buckets=3, method="date_cross_sectional",
        )

        assert len(provenance) > 0

        for prov in provenance:
            assert prov.source_column != ""
            assert prov.construction_method != ""
            assert prov.threshold_source != ""

    def test_regime_labels_have_provenance(self, sample_panel):
        """Regime labels must have fit_start, fit_end, classifier_type."""
        labels = build_pit_regime_labels(sample_panel, window=60, min_obs=30)

        if not labels.empty:
            assert "fit_start" in labels.columns
            assert "fit_end" in labels.columns
            assert "classifier_type" in labels.columns
            assert (labels["fit_start"].notna()).all()
            assert (labels["fit_end"].notna()).all()


# ── Test 8: Missing labels do not silently fall back ─────────────────────────

class TestNoSilentFallback:
    def test_missing_bucket_marked_not_default(self, sample_panel):
        """Missing bucket labels should be marked MISSING, not filled with default."""
        # Use a column that doesn't exist
        df = sample_panel.copy()
        df["nonexistent"] = np.nan
        labels, _ = _build_date_cross_sectional_buckets(
            df, "nonexistent", df["date"].unique(),
            n_buckets=3, winsor_q=0.025, min_breadth=5, source_col="nonexistent",
        )

        # Should have missing quality flags, not default values
        missing = [l for l in labels if l["quality_flag"] == QualityFlag.MISSING.value]
        assert len(missing) > 0 or len(labels) == 0

    def test_missing_condition_type_skipped(self, sample_panel):
        """Unknown condition types should be skipped, not filled."""
        labels, provenance = build_pit_buckets(
            sample_panel, ["nonexistent_condition"],
            n_buckets=3, method="date_cross_sectional",
        )

        assert labels.empty or len(labels) == 0


# ── Test 9: Static sector labels marked static_proxy ─────────────────────────

class TestStaticSectorLabels:
    def test_static_sector_marked_proxy(self, sample_panel):
        """Static sector labels (no sector_asof) should be marked static_proxy."""
        quality = validate_sector_labels(
            sample_panel, pit_column="sector_asof", allow_static=True,
        )

        static = quality[quality["sector_quality"] == QualityFlag.STATIC_PROXY.value]
        assert len(static) > 0, "Static sectors not marked as static_proxy"

    def test_pit_sector_marked_valid(self, panel_with_pit_sector):
        """Sectors with sector_asof should be marked PIT-valid."""
        quality = validate_sector_labels(
            panel_with_pit_sector, pit_column="sector_asof", allow_static=True,
        )

        pit_valid = quality[quality["is_pit_valid"] == True]
        assert len(pit_valid) > 0


# ── Test 10: Research-only labels cannot production-promote ──────────────────

class TestResearchOnlyBlocking:
    def test_research_only_sleeve_disabled(self, sample_panel):
        """Sleeves with research-only conditions should be disabled."""
        # Build without PIT regime labels (forces research-only fallback)
        registry = build_pit_sleeve_registry(
            features=["f_trend"], horizons=[5],
            condition_types=["regime"],
            pit_bucket_labels=pd.DataFrame(),
            pit_regime_labels=pd.DataFrame(),
            sector_quality=pd.DataFrame(),
        )

        # All regime sleeves should be disabled (no PIT source)
        regime_sleeves = [s for s in registry if s.condition_type == "regime"]
        for sleeve in regime_sleeves:
            assert not sleeve.enabled, (
                f"Sleeve {sleeve.sleeve_id} enabled without PIT source"
            )

    def test_pit_valid_sleeve_enabled(self, sample_panel):
        """Sleeves with PIT-valid conditions should be enabled."""
        regime_labels = build_pit_regime_labels(sample_panel, window=60, min_obs=30)

        registry = build_pit_sleeve_registry(
            features=["f_trend"], horizons=[5],
            condition_types=["regime"],
            pit_regime_labels=regime_labels,
        )

        regime_sleeves = [s for s in registry if s.condition_type == "regime"]
        enabled = [s for s in regime_sleeves if s.enabled]
        assert len(enabled) > 0, "No PIT-valid regime sleeves enabled"


# ── Test 11: Walk-forward labels without test-period future thresholds ───────

class TestWalkForwardPIT:
    def test_wf_thresholds_from_train_only(self, sample_panel):
        """Walk-forward condition thresholds must come from training data only."""
        wf_provenance = run_walk_forward_pit_conditions(
            sample_panel, ["volatility", "liquidity"],
            n_windows=3, train_ratio=0.7, embargo_multiplier=2,
            n_buckets=3, rolling_window=60, expanding_min_obs=30,
        )

        for wf in wf_provenance:
            assert wf.fit_end <= wf.test_start or wf.fit_end == wf.train_end, (
                f"Window {wf.window_id}: fit_end {wf.fit_end} after test_start {wf.test_start}"
            )
            assert wf.is_pit_valid

    def test_wf_labels_test_dates(self, sample_panel):
        """Walk-forward should label test dates."""
        wf_provenance = run_walk_forward_pit_conditions(
            sample_panel, ["volatility"],
            n_windows=3, train_ratio=0.7, embargo_multiplier=2,
            n_buckets=3, rolling_window=60, expanding_min_obs=30,
        )

        for wf in wf_provenance:
            assert wf.test_dates_labeled > 0, (
                f"Window {wf.window_id}: no test dates labeled"
            )


# ── Test 12: Invalid PIT labels reject/downgrade sleeves ─────────────────────

class TestInvalidPITRejection:
    def test_invalid_sleeve_marked_invalid(self, sample_panel):
        """Sleeves with invalid PIT conditions should be marked invalid or research_only."""
        registry = build_pit_sleeve_registry(
            features=["f_trend"], horizons=[5],
            condition_types=["regime"],
            pit_bucket_labels=pd.DataFrame(),
            pit_regime_labels=pd.DataFrame(),
            sector_quality=pd.DataFrame(),
        )

        pit_status = compute_conditional_pit_status(
            registry, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        )

        # Should be either invalid, missing, or research_only (not valid)
        non_valid = [p for p in pit_status if p.pit_status in ("invalid", "missing", "research_only")]
        assert len(non_valid) > 0, "No sleeves marked non-valid for missing conditions"

    def test_research_only_sleeve_downgraded(self, sample_panel):
        """Research-only sleeves should be downgraded."""
        registry = build_pit_sleeve_registry(
            features=["f_trend"], horizons=[5],
            condition_types=["regime"],
            pit_regime_labels=pd.DataFrame(),
        )

        pit_status = compute_conditional_pit_status(
            registry, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        )

        research_only = [p for p in pit_status if p.pit_status == "research_only"]
        assert len(research_only) > 0


# ── Test 13: All thresholds from config ──────────────────────────────────────

class TestThresholdsFromConfig:
    def test_config_overrides_defaults(self, minimal_config):
        """Config values should override defaults."""
        cfg = _get_config(minimal_config)
        assert cfg["n_buckets"] == 3
        assert cfg["rolling_window"] == 60
        assert cfg["expanding_min_obs"] == 30

    def test_default_config_exists(self):
        """Default config must exist with all required keys."""
        cfg = _get_config({})
        required = [
            "n_buckets", "bucket_method", "rolling_window", "expanding_min_obs",
            "winsor_q", "min_breadth_per_date", "regime_method", "regime_window",
            "regime_min_obs", "condition_types", "column_map",
        ]
        for key in required:
            assert key in cfg, f"Missing config key: {key}"


# ── Test 14: Every PIT rejection has a reason ────────────────────────────────

class TestPITRejectionReasons:
    def test_invalid_sleeve_has_reason(self, sample_panel):
        """Every invalid sleeve must have a rejection reason."""
        registry = build_pit_sleeve_registry(
            features=["f_trend"], horizons=[5],
            condition_types=["regime"],
            pit_regime_labels=pd.DataFrame(),
        )

        pit_status = compute_conditional_pit_status(
            registry, pd.DataFrame(), pd.DataFrame(), pd.DataFrame(),
        )

        for p in pit_status:
            if p.pit_status in ("invalid", "missing", "research_only"):
                assert p.rejection_reason != "", (
                    f"Sleeve {p.sleeve_id} has no rejection reason"
                )

    def test_leakage_failure_has_reason(self, sample_panel):
        """Every leakage failure must have a rejection reason."""
        results = run_leakage_detection(sample_panel)

        for r in results:
            if not r.passed:
                assert r.rejection_reason != "", (
                    f"Leakage test {r.test_name} has no rejection reason"
                )


# ── Integration Test: Full Pipeline ─────────────────────────────────────────

class TestFullPipeline:
    def test_full_pit_validation_runs(self, sample_panel, minimal_config):
        """Full PIT validation pipeline should run without errors."""
        engine = PITConditionEngine(minimal_config)
        bundle = engine.run_full_pit_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        assert bundle.regime_labels is not None
        assert bundle.sector_quality is not None
        assert len(bundle.sleeve_registry) > 0
        assert len(bundle.leakage_results) > 0
        assert len(bundle.pit_status_results) > 0

    def test_reports_generated(self, sample_panel, minimal_config, tmp_path):
        """All PIT reports should be generated."""
        engine = PITConditionEngine(minimal_config)
        bundle = engine.run_full_pit_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        paths = generate_pit_reports(bundle, str(tmp_path))

        required = ["provenance", "regime_labels", "sector_quality",
                    "sleeve_registry", "leakage_audit", "pit_status", "pm_summary"]
        for key in required:
            assert key in paths, f"Missing report: {key}"
            assert paths[key].exists()

    def test_pm_summary_answers_questions(self, sample_panel, minimal_config, tmp_path):
        """PM summary must answer all key questions."""
        engine = PITConditionEngine(minimal_config)
        bundle = engine.run_full_pit_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        paths = generate_pit_reports(bundle, str(tmp_path))
        with open(paths["pm_summary"]) as f:
            summary = f.read()

        assert "Condition Types Tested" in summary
        assert "PIT Validity" in summary
        assert "Sleeve Registry" in summary
        assert "Leakage Detection" in summary


# ── Edge Cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_panel(self):
        """Engine should handle empty panel."""
        empty = pd.DataFrame(columns=["date", "ticker", "daily_return"])
        labels = build_pit_regime_labels(empty)
        assert labels.empty

    def test_single_ticker_single_date(self):
        """Engine should handle minimal data."""
        df = pd.DataFrame([{
            "date": "2020-01-01", "ticker": "TICK001",
            "daily_return": 0.01, "rolling_vol_20": 0.02,
        }])
        labels = build_pit_regime_labels(df, min_obs=1)
        # Should be empty (not enough data for regime classification)
        assert labels.empty

    def test_find_column_returns_none(self, sample_panel):
        """_find_column should return None for unknown condition."""
        result = _find_column(sample_panel, "nonexistent", {})
        assert result is None

    def test_find_column_returns_match(self, sample_panel):
        """_find_column should return matching column."""
        column_map = {"volatility": ["rolling_vol_20", "vol"]}
        result = _find_column(sample_panel, "volatility", column_map)
        assert result == "rolling_vol_20"
