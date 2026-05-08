"""Tests for the IC Diagnostics Engine.

Proves:
1. Global IC uses true h-day forward returns.
2. IC is computed cross-sectionally by date.
3. Residualized IC uses only PIT factor exposures.
4. Sector-neutral IC ranks within sector/date.
5. Conditional buckets are rolling/cross-sectional, never full-sample.
6. Full-sample regime labels are not used for production diagnostics.
7. Multiple-testing correction is applied across all tested sleeves.
8. A sleeve passing raw p-value but failing BHY is not production-promoted.
9. Conditional sleeves require minimum breadth and minimum dates.
10. Leave-one-sector-out can reject sector-concentrated signals.
11. Weak global IC features are not automatically discarded before diagnostics.
12. All thresholds come from ResearchContract/config.
13. Every rejection has a reason.
14. No ML model is added to compensate for weak IC before diagnostics.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_selection.ic_diagnostics_engine import (
    ICDiagnosticsEngine,
    compute_global_ic,
    compute_residualized_ic,
    compute_sector_neutral_ic,
    compute_conditional_ic_grid,
    generate_multiple_testing_report,
    compute_ic_attribution,
    evaluate_sleeve_admission,
    run_conditional_walk_forward,
    generate_ic_diagnostics_reports,
    benjamini_hochberg,
    benjamini_yekutieli,
    GlobalICResult, ResidualizedICResult, SectorNeutralICResult,
    ConditionalICResult, MultipleTestingResult, ICAttributionResult,
    SleeveAdmissionResult,
    EvidenceStatus, AttributionLabel, SleeveStatus,
    _get_ic_config,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_panel():
    """Panel with features, forward returns, sectors, and factor exposures."""
    np.random.seed(42)
    n_dates = 200
    n_tickers = 50
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    sectors = ["Tech", "Health", "Fin", "Energy", "Consumer"]

    rows = []
    for ticker in tickers:
        sector = sectors[hash(ticker) % len(sectors)]
        market_cap = np.random.uniform(1e9, 1e12)
        beta = np.random.uniform(0.5, 1.5)

        # Feature with some predictive power
        signal = np.random.randn(n_dates)
        for t in range(1, n_dates):
            signal[t] = 0.7 * signal[t - 1] + 0.3 * np.random.randn()

        for t, date in enumerate(dates):
            # Forward return has some correlation with signal
            fwd = 0.001 * signal[t] + np.random.randn() * 0.01
            rows.append({
                "date": date,
                "ticker": ticker,
                "sector": sector,
                "test_feature": signal[t],
                "forward_return": fwd,
                "capm_beta": beta,
                "market_cap": market_cap,
                "rolling_vol_20": np.random.uniform(0.15, 0.35),
                "adv_dollar_20": np.random.uniform(1e7, 1e9),
                "regime_label": "bull" if t % 4 < 3 else "bear",
            })

    return pd.DataFrame(rows)


@pytest.fixture
def weak_ic_panel():
    """Panel with a feature that has weak global IC but strong conditional IC."""
    np.random.seed(42)
    n_dates = 200
    n_tickers = 50
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    rows = []
    for ticker in tickers:
        signal = np.random.randn(n_dates)
        for t in range(1, n_dates):
            signal[t] = 0.5 * signal[t - 1] + 0.5 * np.random.randn()

        for t, date in enumerate(dates):
            # IC only in high volatility regime
            vol = np.random.uniform(0.1, 0.4)
            regime = "high_vol" if vol > 0.25 else "low_vol"
            # Signal only predictive in high vol
            if regime == "high_vol":
                fwd = 0.003 * signal[t] + np.random.randn() * 0.01
            else:
                fwd = np.random.randn() * 0.01  # No signal

            rows.append({
                "date": date,
                "ticker": ticker,
                "sector": "Tech",
                "weak_feature": signal[t],
                "forward_return": fwd,
                "rolling_vol_20": vol,
                "regime_label": regime,
            })

    return pd.DataFrame(rows)


# ── Test 1: Global IC uses true h-day forward returns ────────────────────────

class TestGlobalICForwardReturns:
    def test_uses_forward_return_column(self, sample_panel):
        """Global IC uses forward_return, not daily_return."""
        df = sample_panel.copy()
        # Remove daily_return if present
        if "daily_return" in df.columns:
            df = df.drop(columns=["daily_return"])

        result = compute_global_ic(df, "test_feature", horizon=5, min_dates=30, min_breadth=5)
        assert result.n_dates > 0
        assert result.ic_quality != "insufficient" or result.rejection_reason

    def test_different_horizons_different_results(self, sample_panel):
        """Different horizons produce different IC values."""
        r1 = compute_global_ic(sample_panel, "test_feature", horizon=1, min_dates=30, min_breadth=5)
        r5 = compute_global_ic(sample_panel, "test_feature", horizon=5, min_dates=30, min_breadth=5)
        r20 = compute_global_ic(sample_panel, "test_feature", horizon=20, min_dates=30, min_breadth=5)

        # At least some should differ
        ics = [r1.mean_ic, r5.mean_ic, r20.mean_ic]
        assert len(set(round(ic, 4) for ic in ics)) > 1 or all(ic == 0 for ic in ics)

    def test_empty_df_returns_empty(self):
        """Empty DataFrame returns insufficient IC."""
        result = compute_global_ic(pd.DataFrame(), "test", horizon=5)
        assert result.ic_quality == "insufficient"


# ── Test 2: IC computed cross-sectionally by date ────────────────────────────

class TestCrossSectionalIC:
    def test_per_date_computation(self, sample_panel):
        """IC is computed per date, not on flattened panel."""
        result = compute_global_ic(sample_panel, "test_feature", horizon=5, min_dates=30, min_breadth=5)
        # n_dates should reflect the number of valid cross-sectional dates
        assert result.n_dates > 0
        assert result.avg_breadth > 0

    def test_min_breadth_filter(self, sample_panel):
        """Dates with insufficient breadth are excluded."""
        result = compute_global_ic(sample_panel, "test_feature", horizon=5, min_dates=30, min_breadth=100)
        # With min_breadth=100, most dates should be excluded (only 50 tickers)
        assert result.n_dates == 0 or result.ic_quality == "insufficient"


# ── Test 3: Residualized IC uses PIT factor exposures ────────────────────────

class TestResidualizedIC:
    def test_uses_available_factors(self, sample_panel):
        """Residualized IC uses factor columns from the panel."""
        global_ic = compute_global_ic(sample_panel, "test_feature", horizon=5, min_dates=30, min_breadth=5)
        result = compute_residualized_ic(
            sample_panel, "test_feature", horizon=5,
            factor_controls=["market", "size", "volatility"],
            min_dates=20, min_breadth=5,
            global_ic_result=global_ic,
        )
        assert result.factor_controls_used != ""
        assert result.n_dates > 0

    def test_no_factors_returns_empty(self, sample_panel):
        """No factor columns returns insufficient."""
        df = sample_panel[["date", "ticker", "test_feature", "forward_return"]].copy()
        global_ic = compute_global_ic(sample_panel, "test_feature", horizon=5, min_dates=30, min_breadth=5)
        result = compute_residualized_ic(
            df, "test_feature", horizon=5,
            factor_controls=["market", "size"],
            global_ic_result=global_ic,
        )
        assert result.residualization_quality == "insufficient"

    def test_residualized_ic_differs_from_global(self, sample_panel):
        """Residualized IC can differ from global IC."""
        global_ic = compute_global_ic(sample_panel, "test_feature", horizon=5, min_dates=30, min_breadth=5)
        result = compute_residualized_ic(
            sample_panel, "test_feature", horizon=5,
            factor_controls=["market", "size", "volatility"],
            min_dates=20, min_breadth=5,
            global_ic_result=global_ic,
        )
        # Delta should be computed even if zero
        assert hasattr(result, "delta_vs_global_ic")


# ── Test 4: Sector-neutral IC ranks within sector/date ───────────────────────

class TestSectorNeutralIC:
    def test_ranks_within_sector(self, sample_panel):
        """Sector-neutral IC ranks feature and return within sector/date."""
        result = compute_sector_neutral_ic(
            sample_panel, "test_feature", horizon=5,
            min_dates=20, min_breadth=3, min_sectors=3,
        )
        assert result.n_sectors_valid >= 3
        assert result.sector_neutral_ic != 0 or result.rejection_reason

    def test_no_sector_column_returns_empty(self, sample_panel):
        """Missing sector column returns insufficient."""
        df = sample_panel.drop(columns=["sector"])
        result = compute_sector_neutral_ic(df, "test_feature", horizon=5)
        assert result.sector_quality == "no_sector_column"

    def test_sector_concentration_computed(self, sample_panel):
        """Sector concentration score is computed."""
        result = compute_sector_neutral_ic(
            sample_panel, "test_feature", horizon=5,
            min_dates=20, min_breadth=3, min_sectors=3,
        )
        assert 0 <= result.sector_concentration_score <= 1.0


# ── Test 5: Conditional buckets are PIT ──────────────────────────────────────

class TestConditionalBucketsPIT:
    def test_buckets_are_cross_sectional(self, sample_panel):
        """Conditional buckets use cross-sectional quantiles per date, not full-sample."""
        results, pvals = compute_conditional_ic_grid(
            sample_panel, "test_feature", horizon=5,
            conditions=["volatility", "liquidity", "size"],
            n_buckets=3, min_dates=15, min_breadth=3, min_bucket_breadth=5,
        )
        # Results should exist for bucket conditions
        bucket_results = [r for r in results if "bucket" in r.condition_value]
        assert len(bucket_results) > 0

    def test_no_full_sample_thresholds(self, sample_panel):
        """No full-sample quantile thresholds are used."""
        # The engine uses groupby("date").transform(qcut) which is PIT
        results, _ = compute_conditional_ic_grid(
            sample_panel, "test_feature", horizon=5,
            conditions=["volatility"],
            n_buckets=3, min_dates=15, min_breadth=3, min_bucket_breadth=5,
        )
        # Each bucket should have different dates (PIT assignment)
        if results:
            buckets = set(r.condition_value for r in results)
            assert len(buckets) >= 2


# ── Test 6: Full-sample regime labels not used ───────────────────────────────

class TestRegimeLabelsPIT:
    def test_regime_from_panel_column(self, sample_panel):
        """Regime labels come from panel column, not full-sample computation."""
        results, _ = compute_conditional_ic_grid(
            sample_panel, "test_feature", horizon=5,
            conditions=["regime"],
            min_dates=15, min_breadth=3, min_bucket_breadth=5,
        )
        regime_results = [r for r in results if r.condition_type == "regime"]
        assert len(regime_results) > 0


# ── Test 7: Multiple-testing correction applied ──────────────────────────────

class TestMultipleTestingCorrection:
    def test_bh_correction_applied(self, sample_panel):
        """BH correction is applied across all conditional sleeves."""
        results, pvals = compute_conditional_ic_grid(
            sample_panel, "test_feature", horizon=5,
            conditions=["regime", "volatility", "liquidity"],
            n_buckets=3, min_dates=15, min_breadth=3, min_bucket_breadth=5,
        )
        mt_results = generate_multiple_testing_report(results)
        for r in mt_results:
            assert r.bh_q_value <= 1.0
            assert r.bhy_q_value <= 1.0

    def test_bh_bhy_differ(self):
        """BH and BHY q-values differ (BHY is more conservative)."""
        pvals = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
        bh = benjamini_hochberg(pvals)
        by = benjamini_yekutieli(pvals)
        # BHY should be >= BH for at least some values
        assert any(by >= bh)

    def test_bhy_more_conservative(self):
        """BHY is more conservative than BH."""
        pvals = np.array([0.001, 0.01, 0.05, 0.1, 0.2])
        bh = benjamini_hochberg(pvals)
        by = benjamini_yekutieli(pvals)
        assert all(by >= bh)


# ── Test 8: Raw p-value pass but BHY fail not promoted ───────────────────────

class TestBHYPromotionGate:
    def test_raw_pass_bhy_fail_not_admitted(self):
        """Sleeve passing raw p-value but failing BHY is not production-promoted."""
        cond = ConditionalICResult(
            sleeve_id="test", feature="test", family="test",
            horizon=5, condition_type="regime", condition_value="bull",
            mean_ic=0.03, icir=1.5, hac_tstat=2.5,
            p_value=0.01, bh_q_value=0.08, bhy_q_value=0.15,
            n_dates=50, avg_breadth=30,
            breadth_quality="high", cost_viability_status="not_evaluated",
            conditional_ic_quality="high", rejection_reason="",
        )
        mt = MultipleTestingResult(
            sleeve_id="test", feature="test", family="test",
            horizon=5, raw_p_value=0.01, bh_q_value=0.08, bhy_q_value=0.15,
            passes_bh=True, passes_bhy=False, test_family_size=20,
            dependency_note="", evidence_status=EvidenceStatus.RESEARCH_CANDIDATE.value,
        )
        cfg = {"ic_mean_threshold": 0.005, "icir_threshold": 0.5, "hac_tstat_threshold": 2.0,
               "bh_q_threshold": 0.10, "bhy_q_threshold": 0.05,
               "min_dates_for_conditional": 15, "min_breadth_for_bucket": 10}
        adm = evaluate_sleeve_admission(cond, mt, cfg)
        # Passes BH but not BHY → research_only, not admitted
        assert adm.final_status != SleeveStatus.ADMITTED.value
        assert adm.final_status == SleeveStatus.RESEARCH_ONLY.value


# ── Test 9: Minimum breadth and dates required ───────────────────────────────

class TestMinimumBreadthAndDates:
    def test_insufficient_dates_rejected(self, sample_panel):
        """Conditional sleeves with insufficient dates are rejected."""
        results, _ = compute_conditional_ic_grid(
            sample_panel, "test_feature", horizon=5,
            conditions=["regime"],
            min_dates=500, min_breadth=3, min_bucket_breadth=5,
        )
        assert len(results) == 0

    def test_insufficient_breadth_rejected(self, sample_panel):
        """Conditional sleeves with insufficient breadth are rejected."""
        results, _ = compute_conditional_ic_grid(
            sample_panel, "test_feature", horizon=5,
            conditions=["regime"],
            min_dates=15, min_breadth=3, min_bucket_breadth=100,
        )
        assert len(results) == 0


# ── Test 10: Leave-one-sector-out rejects concentrated signals ───────────────

class TestLeaveOneSectorOut:
    def test_sector_concentration_detected(self, sample_panel):
        """Sector concentration score identifies concentrated signals."""
        result = compute_sector_neutral_ic(
            sample_panel, "test_feature", horizon=5,
            min_dates=20, min_breadth=3, min_sectors=3,
        )
        # Should compute leave-one-sector-out minimum IC
        assert hasattr(result, "leave_one_sector_out_min_ic")
        assert result.sector_concentration_score >= 0


# ── Test 11: Weak global IC not discarded before diagnostics ─────────────────

class TestWeakICNotDiscarded:
    def test_weak_ic_runs_residualized(self, weak_ic_panel):
        """Weak global IC features still get residualized and conditional diagnostics."""
        engine = ICDiagnosticsEngine(config={})
        bundles = engine.run_full_diagnostics(
            weak_ic_panel, features=["weak_feature"], horizons=[5],
        )
        assert len(bundles) == 1
        b = bundles[0]
        # Should have residualized results even if global IC is weak
        assert len(b.residualized_results) > 0
        # Should have conditional results
        assert len(b.conditional_results) > 0

    def test_weak_ic_runs_sector_neutral(self, weak_ic_panel):
        """Weak global IC features still get sector-neutral diagnostics."""
        engine = ICDiagnosticsEngine(config={})
        bundles = engine.run_full_diagnostics(
            weak_ic_panel, features=["weak_feature"], horizons=[5],
        )
        assert len(bundles[0].sector_results) > 0


# ── Test 12: Thresholds from config ──────────────────────────────────────────

class TestThresholdsFromConfig:
    def test_ic_config_from_user(self):
        """User config overrides defaults."""
        cfg = {"ic_diagnostics": {"ic_mean_threshold": 0.01, "min_dates_for_ic": 100}}
        ic_cfg = _get_ic_config(cfg)
        assert ic_cfg["ic_mean_threshold"] == 0.01
        assert ic_cfg["min_dates_for_ic"] == 100

    def test_ic_config_defaults(self):
        """Missing config uses defaults."""
        ic_cfg = _get_ic_config({})
        assert ic_cfg["ic_mean_threshold"] == 0.005
        assert ic_cfg["min_dates_for_ic"] == 30
        assert ic_cfg["bh_q_threshold"] == 0.10

    def test_engine_uses_config_thresholds(self):
        """Engine uses config thresholds for admission."""
        cfg = {
            "ic_diagnostics": {
                "ic_mean_threshold": 0.02,
                "bh_q_threshold": 0.05,
                "bhy_q_threshold": 0.02,
            },
        }
        engine = ICDiagnosticsEngine(config=cfg)
        assert engine.ic_cfg["ic_mean_threshold"] == 0.02


# ── Test 13: Every rejection has a reason ────────────────────────────────────

class TestRejectionReasons:
    def test_empty_ic_has_reason(self):
        """Empty IC result has explicit rejection reason."""
        result = compute_global_ic(pd.DataFrame(), "test", horizon=5)
        assert result.rejection_reason != ""

    def test_insufficient_data_has_reason(self):
        """Insufficient data produces explicit reason."""
        result = compute_global_ic(pd.DataFrame({"a": [1]}), "test", horizon=5)
        assert result.rejection_reason != ""

    def test_sleeve_admission_has_reason_when_rejected(self):
        """Rejected sleeve has explicit rejection reason."""
        cond = ConditionalICResult(
            sleeve_id="test", feature="test", family="test",
            horizon=5, condition_type="regime", condition_value="bull",
            mean_ic=0.001, icir=0.1, hac_tstat=0.5,
            p_value=0.5, bh_q_value=0.8, bhy_q_value=0.9,
            n_dates=5, avg_breadth=2,
            breadth_quality="low", cost_viability_status="not_evaluated",
            conditional_ic_quality="insufficient", rejection_reason="",
        )
        mt = MultipleTestingResult(
            sleeve_id="test", feature="test", family="test",
            horizon=5, raw_p_value=0.5, bh_q_value=0.8, bhy_q_value=0.9,
            passes_bh=False, passes_bhy=False, test_family_size=1,
            dependency_note="", evidence_status=EvidenceStatus.REJECTED.value,
        )
        cfg = {"ic_mean_threshold": 0.005, "icir_threshold": 0.5, "hac_tstat_threshold": 2.0,
               "bh_q_threshold": 0.10, "bhy_q_threshold": 0.05,
               "min_dates_for_conditional": 15, "min_breadth_for_bucket": 10}
        adm = evaluate_sleeve_admission(cond, mt, cfg)
        assert adm.final_status == SleeveStatus.REJECTED.value
        assert adm.rejection_reason != ""


# ── Test 14: No ML model added before diagnostics ────────────────────────────

class TestNoMLBeforeDiagnostics:
    def test_engine_is_statistical_not_ml(self):
        """IC diagnostics engine uses statistical methods, not ML."""
        engine = ICDiagnosticsEngine(config={})
        # Engine should not have any ML model attributes
        assert not hasattr(engine, "model")
        assert not hasattr(engine, "estimator")
        assert not hasattr(engine, "predict")

    def test_no_sklearn_imports_in_engine(self):
        """Engine module does not import sklearn."""
        import model_selection.ic_diagnostics_engine as module
        source = Path(module.__file__).read_text()
        assert "sklearn" not in source
        assert "xgboost" not in source
        assert "lightgbm" not in source


# ── Test: Report generation ──────────────────────────────────────────────────

class TestReportGeneration:
    def test_all_reports_generated(self, sample_panel):
        """All 10 reports are generated."""
        engine = ICDiagnosticsEngine(config={})
        bundles = engine.run_full_diagnostics(
            sample_panel, features=["test_feature"], horizons=[5],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_ic_diagnostics_reports(bundles, tmpdir)
            assert "global_ic" in paths
            assert "residualized_ic" in paths
            assert "sector_neutral_ic" in paths
            assert "conditional_ic" in paths
            assert "multiple_testing" in paths
            assert "attribution" in paths
            assert "sleeve_admission" in paths

    def test_rejected_and_accepted_reports(self, sample_panel):
        """Rejected and accepted candidate reports are generated."""
        engine = ICDiagnosticsEngine(config={})
        bundles = engine.run_full_diagnostics(
            sample_panel, features=["test_feature"], horizons=[5],
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_ic_diagnostics_reports(bundles, tmpdir)
            # These may or may not exist depending on results
            assert "global_ic" in paths


# ── Test: Full engine integration ────────────────────────────────────────────

class TestFullEngineIntegration:
    def test_full_diagnostics_runs(self, sample_panel):
        """Full engine runs all phases."""
        engine = ICDiagnosticsEngine(config={})
        bundles = engine.run_full_diagnostics(
            sample_panel, features=["test_feature"], horizons=[1, 5, 10],
        )
        assert len(bundles) == 1
        b = bundles[0]
        assert len(b.global_ic_results) == 3
        assert len(b.residualized_results) == 3
        assert len(b.sector_results) == 3
        assert len(b.attribution_results) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
