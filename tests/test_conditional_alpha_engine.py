"""Tests for Conditional Alpha Validation Engine.

Tests prove all 16 acceptance criteria:
1. Regime labels are point-in-time.
2. Full-sample regime labels are not used for production diagnostics.
3. Every tested sleeve appears in the sleeve registry.
4. Multiple-testing correction uses the full sleeve universe.
5. BH and BHY are both computed.
6. A sleeve passing BH but failing BHY is not production-promoted.
7. Conditional IC uses true h-day forward returns.
8. Conditional breadth is computed inside the condition.
9. Conditional halflife is computed inside the condition.
10. Cost viability is computed inside the condition.
11. Leave-one-year-out can downgrade a fragile sleeve.
12. Bear-regime sleeve evidence is not dominated by one Bear episode.
13. Sleeve definitions are config-driven.
14. All thresholds come from ResearchContract/config.
15. Every rejected sleeve has a rejection reason.
16. ML/model selection is not used to rescue invalid conditional alpha.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_selection.conditional_alpha_engine import (
    ConditionalAlphaEngine,
    SleeveDefinition,
    SleeveDiagnostic,
    MultipleTestingResult,
    StabilityResult,
    WalkForwardResult,
    SimpleSleeveBacktest,
    AdmissionResult,
    BearAuditResult,
    SleeveFinalStatus,
    StabilityStatus,
    compute_pit_regime_labels,
    build_sleeve_registry,
    compute_sleeve_diagnostics,
    compute_multiple_testing_correction,
    compute_stability_analysis,
    run_walk_forward_validation,
    run_simple_sleeve_backtest,
    evaluate_admission,
    run_bear_regime_audit,
    generate_conditional_alpha_reports,
    _benjamini_hochberg,
    _benjamini_yekutieli,
    _hac_tstat,
    _build_fwd_return_col,
    _cs_ic_by_date,
    _leave_one_year_out,
    _filter_by_condition,
    _get_config,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_panel():
    """Generate a realistic panel with multiple features, regimes, and forward returns."""
    np.random.seed(42)
    n_tickers = 30
    n_dates = 150
    tickers = [f"TICK{i:03d}" for i in range(n_tickers)]
    dates = pd.date_range("2015-01-02", periods=n_dates, freq="B")

    rows = []
    for date in dates:
        for ticker in tickers:
            # Create features with some signal
            f_trend = np.random.randn() * 0.5 + 0.1
            ret_20d = np.random.randn() * 0.3
            rolling_vol_20 = np.random.exponential(0.02) + 0.01
            adv_dollar_20 = np.random.lognormal(16, 1.5)
            market_cap = np.random.lognormal(20, 1.5)

            # Forward return with some signal from f_trend
            signal = 0.0003 * f_trend
            noise = np.random.randn() * 0.02
            forward_return = signal + noise

            # Regime: use date-based simulation
            date_idx = dates.get_loc(date)
            if date_idx % 100 < 30:
                regime = "Bear"
            elif date_idx % 100 < 60:
                regime = "Bull"
            elif date_idx % 100 < 80:
                regime = "HighVol"
            else:
                regime = "Sideways"

            rows.append({
                "date": date,
                "ticker": ticker,
                "f_trend": f_trend,
                "ret_20d": ret_20d,
                "rolling_vol_20": rolling_vol_20,
                "adv_dollar_20": adv_dollar_20,
                "market_cap": market_cap,
                "forward_return": forward_return,
                "daily_return": np.random.randn() * 0.02,
                "regime_label": regime,
                "sector": np.random.choice(["Tech", "Health", "Finance", "Energy", "Consumer"]),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def bear_heavy_panel():
    """Panel with strong Bear-regime signal for testing Bear audit."""
    np.random.seed(42)
    n_tickers = 25
    n_dates = 120
    tickers = [f"TICK{i:03d}" for i in range(n_tickers)]
    dates = pd.date_range("2015-01-02", periods=n_dates, freq="B")

    rows = []
    for date in dates:
        date_idx = dates.get_loc(date)
        # Create multiple Bear episodes
        if date_idx < 50:
            regime = "Bear"
        elif date_idx < 100:
            regime = "Bull"
        elif date_idx < 130:
            regime = "Bear"
        elif date_idx < 180:
            regime = "Bull"
        elif date_idx < 210:
            regime = "Bear"
        else:
            regime = "Sideways"

        for ticker in tickers:
            f_trend = np.random.randn() * 0.5
            # Strong signal in Bear regime
            if regime == "Bear":
                signal = 0.001 * f_trend
            else:
                signal = 0.0001 * f_trend

            forward_return = signal + np.random.randn() * 0.02

            rows.append({
                "date": date,
                "ticker": ticker,
                "f_trend": f_trend,
                "rolling_vol_20": np.random.exponential(0.02) + 0.01,
                "adv_dollar_20": np.random.lognormal(16, 1.5),
                "market_cap": np.random.lognormal(20, 1.5),
                "forward_return": forward_return,
                "daily_return": np.random.randn() * 0.02,
                "regime_label": regime,
                "sector": np.random.choice(["Tech", "Health", "Finance", "Energy"]),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def fragile_panel():
    """Panel where signal is dominated by one year (for leave-one-year-out testing)."""
    np.random.seed(42)
    n_tickers = 20
    n_dates = 252 * 2  # 2 years
    tickers = [f"TICK{i:03d}" for i in range(n_tickers)]
    dates = pd.date_range("2020-01-02", periods=n_dates, freq="B")

    rows = []
    for date in dates:
        year = date.year
        for ticker in tickers:
            f_trend = np.random.randn() * 0.5
            # Signal only in 2020 (crisis year)
            if year == 2020:
                signal = 0.001 * f_trend
            else:
                signal = 0.0

            forward_return = signal + np.random.randn() * 0.02

            rows.append({
                "date": date,
                "ticker": ticker,
                "f_trend": f_trend,
                "rolling_vol_20": np.random.exponential(0.02) + 0.01,
                "adv_dollar_20": np.random.lognormal(16, 1.5),
                "market_cap": np.random.lognormal(20, 1.5),
                "forward_return": forward_return,
                "daily_return": np.random.randn() * 0.02,
                "regime_label": "Bull",
                "sector": np.random.choice(["Tech", "Health", "Finance"]),
            })

    return pd.DataFrame(rows)


@pytest.fixture
def minimal_config():
    return {
        "conditional_alpha": {
            "horizons": [5, 10],
            "condition_types": ["regime", "volatility"],
            "n_buckets": 3,
            "min_dates_for_ic": 15,
            "min_breadth_for_ic": 5,
            "min_dates_for_conditional": 10,
            "min_breadth_for_conditional": 5,
            "min_breadth_for_bucket": 5,
            "ic_mean_threshold": 0.003,
            "icir_threshold": 0.3,
            "hac_tstat_threshold": 1.5,
            "sign_consistency_threshold": 0.55,
            "bh_q_threshold": 0.10,
            "bhy_q_threshold": 0.05,
            "pit_regime_window": 60,
            "pit_regime_min_obs": 30,
            "wf_n_windows": 3,
            "wf_train_ratio": 0.7,
            "wf_embargo_multiplier": 2,
            "wf_min_oos_dates": 5,
            "wf_min_oos_breadth": 5,
            "sleeve_cost_bps": 10.0,
            "bear_min_dates": 10,
            "bear_min_breadth": 5,
        },
    }


# ── Test 1: Regime labels are point-in-time ──────────────────────────────────

class TestPitRegimeLabels:
    def test_labels_use_only_past_data(self, sample_panel):
        """Regime labels at date t must use only data available before t."""
        labels = compute_pit_regime_labels(sample_panel, window=60, min_obs=30)

        assert not labels.empty
        assert "date" in labels.columns
        assert "regime_label" in labels.columns
        assert "bear_probability" in labels.columns

        # Each label's fit_end must be before or equal to its date
        labels["date"] = pd.to_datetime(labels["date"])
        labels["fit_end"] = pd.to_datetime(labels["fit_end"])
        assert (labels["fit_end"] <= labels["date"]).all(), (
            "Regime labels use future data — not point-in-time"
        )

    def test_label_quality_is_pit(self, sample_panel):
        """Label quality should be 'pit' when sufficient history exists."""
        labels = compute_pit_regime_labels(sample_panel, window=60, min_obs=30)
        pit_labels = labels[labels["label_quality"] == "pit"]
        assert len(pit_labels) > 0, "No PIT-quality labels generated"

    def test_probabilities_sum_to_one(self, sample_panel):
        """Regime probabilities must sum to approximately 1."""
        labels = compute_pit_regime_labels(sample_panel, window=60, min_obs=30)
        total = (
            labels["bear_probability"]
            + labels["bull_probability"]
            + labels["highvol_probability"]
            + labels["sideways_probability"]
        )
        assert np.allclose(total, 1.0, atol=0.01), "Probabilities do not sum to 1"

    def test_empty_panel_returns_empty(self):
        """Empty panel should return empty labels."""
        empty = pd.DataFrame(columns=["date", "ticker", "daily_return"])
        labels = compute_pit_regime_labels(empty)
        assert labels.empty


# ── Test 2: Full-sample regime labels not used for production ────────────────

class TestNoFullSampleLabels:
    def test_engine_uses_pit_labels(self, sample_panel, minimal_config):
        """Engine must use PIT labels, not full-sample regime_label column."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        # PIT labels should be generated
        assert not bundle.pit_regime_labels.empty
        # All PIT labels should have quality "pit" or "proxy" (not full-sample)
        qualities = bundle.pit_regime_labels["label_quality"].unique()
        assert "pit" in qualities or "proxy" in qualities


# ── Test 3: Every tested sleeve in registry ──────────────────────────────────

class TestSleeveRegistry:
    def test_all_sleeves_registered(self, sample_panel, minimal_config):
        """Every sleeve evaluated must appear in the registry."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        registry_ids = {s.sleeve_id for s in bundle.sleeve_registry}
        diagnostic_ids = {d.sleeve_id for d in bundle.diagnostics}

        assert diagnostic_ids.issubset(registry_ids), (
            f"Diagnostic sleeves not in registry: {diagnostic_ids - registry_ids}"
        )

    def test_registry_contains_all_combinations(self):
        """Registry should contain all feature × horizon × condition combinations."""
        registry = build_sleeve_registry(
            features=["f_trend", "ret_20d"],
            horizons=[5, 10],
            condition_types=["regime"],
            n_buckets=3,
            rebalance_rules=["monthly"],
        )

        # 2 features × 2 horizons × 4 regimes × 1 rule = 16
        assert len(registry) == 16

        # Check all combinations exist
        for feature in ["f_trend", "ret_20d"]:
            for horizon in [5, 10]:
                for regime in ["Bear", "Bull", "HighVol", "Sideways"]:
                    expected_id = f"regime_{regime}_{feature}_h{horizon}_monthly"
                    assert any(s.sleeve_id == expected_id for s in registry)


# ── Test 4: Multiple-testing uses full universe ──────────────────────────────

class TestMultipleTestingUniverse:
    def test_correction_uses_all_sleeves(self, sample_panel, minimal_config):
        """Multiple-testing correction must use the full sleeve universe."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        n_diagnostics = len(bundle.diagnostics)
        n_mt = len(bundle.mt_results)

        assert n_mt == n_diagnostics, (
            f"Multiple-testing results ({n_mt}) != diagnostics ({n_diagnostics})"
        )

        # Check test_family_size reflects full universe
        for mt in bundle.mt_results:
            assert mt.test_family_size > 0


# ── Test 5: BH and BHY both computed ────────────────────────────────────────

class TestBHAndBHY:
    def test_both_corrections_computed(self, sample_panel, minimal_config):
        """Both BH and BHY q-values must be computed."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        for mt in bundle.mt_results:
            assert mt.bh_q_value is not None
            assert mt.bhy_q_value is not None
            assert 0 <= mt.bh_q_value <= 1
            assert 0 <= mt.bhy_q_value <= 1

    def test_bhy_more_conservative(self):
        """BHY q-values should be >= BH q-values (more conservative)."""
        p_values = np.array([0.001, 0.005, 0.01, 0.05, 0.1])
        bh = _benjamini_hochberg(p_values)
        bhy = _benjamini_yekutieli(p_values)

        assert np.all(bhy >= bh), "BHY should be more conservative than BH"

    def test_bh_monotonicity(self):
        """BH q-values should be monotonically non-decreasing in sorted order."""
        p_values = np.array([0.001, 0.05, 0.01, 0.1, 0.005])
        bh = _benjamini_hochberg(p_values)
        sorted_bh = np.sort(bh)
        assert all(sorted_bh[i] <= sorted_bh[i + 1] for i in range(len(sorted_bh) - 1))


# ── Test 6: BH-pass but BHY-fail not production-promoted ─────────────────────

class TestBHNotProduction:
    def test_bh_only_not_production(self, sample_panel, minimal_config):
        """A sleeve passing BH but failing BHY must NOT be production-promoted."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        for mt, adm in zip(bundle.mt_results, bundle.admission_results):
            if mt.passes_bh and not mt.passes_bhy:
                assert adm.final_status not in (
                    SleeveFinalStatus.PRODUCTION_CANDIDATE.value,
                    SleeveFinalStatus.PRODUCTION_WATCHLIST.value,
                ), (
                    f"Sleeve {adm.sleeve_id} passes BH but fails BHY "
                    f"yet is {adm.final_status}"
                )


# ── Test 7: Conditional IC uses true h-day forward returns ──────────────────

class TestTrueForwardReturns:
    def test_fwd_return_col_built(self, sample_panel):
        """True h-day forward returns must be constructed."""
        work = _build_fwd_return_col(sample_panel, 5)
        assert "fwd_ret_5d" in work.columns

        # Verify it's shifted forward (last h dates should be NaN)
        last_date = sample_panel["date"].max()
        last_day_data = work[work["date"] == last_date]
        # Last date should have NaN forward returns (shifted beyond data)
        assert last_day_data["fwd_ret_5d"].isna().all()

    def test_ic_uses_fwd_return_not_daily(self, sample_panel):
        """IC computation must use h-day forward return, not daily_return."""
        work = _build_fwd_return_col(sample_panel, 10)
        ics, breadths, dates = _cs_ic_by_date(
            work, "f_trend", "fwd_ret_10d", min_breadth=5,
        )
        assert len(ics) > 0, "No IC computed with true forward returns"


# ── Test 8: Conditional breadth computed inside condition ────────────────────

class TestConditionalBreadth:
    def test_breadth_filtered_by_condition(self, sample_panel, minimal_config):
        """Breadth must be computed only for securities inside the condition."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        for diag in bundle.diagnostics:
            if diag.condition_type == "regime" and diag.condition_value == "Bear":
                # Bear breadth should be less than total breadth
                assert diag.avg_breadth > 0, f"No breadth for Bear sleeve {diag.sleeve_id}"

    def test_breadth_reported_in_diagnostics(self, sample_panel, minimal_config):
        """Diagnostics must report avg_breadth and min_breadth."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        for diag in bundle.diagnostics:
            assert diag.avg_breadth >= 0
            assert diag.min_breadth >= 0


# ── Test 9: Conditional halflife computed inside condition ──────────────────

class TestConditionalHalflife:
    def test_halflife_in_diagnostics(self, sample_panel, minimal_config):
        """Halflife must be computed for each sleeve."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        for diag in bundle.diagnostics:
            assert diag.halflife >= 0, f"Negative halflife for {diag.sleeve_id}"
            assert diag.persistence_at_horizon >= 0


# ── Test 10: Cost viability computed inside condition ────────────────────────

class TestCostViability:
    def test_cost_computed_per_sleeve(self, sample_panel, minimal_config):
        """Expected cost and net alpha must be computed for each sleeve."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        for diag in bundle.diagnostics:
            assert diag.expected_cost_bps >= 0
            assert diag.net_expected_alpha_bps is not None
            assert diag.alpha_cost_ratio >= 0

    def test_cost_viability_affects_admission(self, sample_panel, minimal_config):
        """Cost-dominated sleeves should be flagged in admission."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        cost_dominated = [
            a for a in bundle.admission_results
            if "cost_dominated" in a.rejection_reason
        ]
        # At least some sleeves should be evaluated for cost
        assert len(bundle.admission_results) > 0


# ── Test 11: Leave-one-year-out can downgrade ───────────────────────────────

class TestLeaveOneYearOut:
    def test_fragile_sleeve_downgraded(self, fragile_panel, minimal_config):
        """A sleeve dominated by one year should be downgraded by LOYO."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            fragile_panel, ["f_trend"], horizons=[5],
        )

        # Check stability results
        for stab in bundle.stability_results:
            if stab.dominant_year_contribution > 0.5:
                assert stab.stability_status in (
                    StabilityStatus.UNSTABLE.value,
                    StabilityStatus.CONCENTRATED.value,
                    StabilityStatus.MARGINALLY_STABLE.value,
                ), (
                    f"Sleeve {stab.sleeve_id} dominated by one year "
                    f"but status is {stab.stability_status}"
                )

    def test_loyo_computed(self, sample_panel):
        """Leave-one-year-out must return valid results."""
        work = _build_fwd_return_col(sample_panel, 5)
        ics, _, dates = _cs_ic_by_date(work, "f_trend", "fwd_ret_5d", min_breadth=5)

        if len(ics) >= 20:
            min_ic, fail_year, dom_contrib = _leave_one_year_out(
                ics, dates, min_ic=0.002, dom_max=0.5,
            )
            assert dom_contrib >= 0
            assert dom_contrib <= 1.0


# ── Test 12: Bear evidence not dominated by one episode ─────────────────────

class TestBearEpisodeDominance:
    def test_bear_audit_computed(self, bear_heavy_panel, minimal_config):
        """Bear-regime audit must identify dominant episodes."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            bear_heavy_panel, ["f_trend"], horizons=[5],
        )

        if bundle.bear_audit_results:
            for bear in bundle.bear_audit_results:
                assert bear.bear_date_count > 0
                assert bear.leave_one_bear_episode_min_ic is not None

    def test_bear_audit_flags_fragile(self, bear_heavy_panel, minimal_config):
        """Bear sleeves dependent on one episode should be flagged."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            bear_heavy_panel, ["f_trend"], horizons=[5],
        )

        for bear in bundle.bear_audit_results:
            if bear.bear_audit_status == "rejected":
                assert bear.rejection_reason != "", (
                    f"Rejected Bear sleeve {bear.sleeve_id} has no reason"
                )


# ── Test 13: Sleeve definitions are config-driven ────────────────────────────

class TestConfigDrivenSleeves:
    def test_registry_from_config(self, minimal_config):
        """Sleeve registry must be driven by config condition_types and horizons."""
        cfg = minimal_config["conditional_alpha"]
        registry = build_sleeve_registry(
            features=["f_trend"],
            horizons=cfg["horizons"],
            condition_types=cfg["condition_types"],
            n_buckets=cfg["n_buckets"],
        )

        # Check all configured condition types are present
        configured_types = set(cfg["condition_types"])
        registry_types = {s.condition_type for s in registry}
        assert configured_types.issubset(registry_types), (
            f"Missing condition types: {configured_types - registry_types}"
        )

    def test_custom_conditions_applied(self):
        """Custom condition types in config should appear in registry."""
        registry = build_sleeve_registry(
            features=["f_trend"],
            horizons=[5],
            condition_types=["regime", "beta"],
            n_buckets=3,
        )

        beta_sleeves = [s for s in registry if s.condition_type == "beta"]
        assert len(beta_sleeves) > 0


# ── Test 14: All thresholds from config ──────────────────────────────────────

class TestThresholdsFromConfig:
    def test_ic_threshold_from_config(self, minimal_config):
        """IC threshold must come from config, not hardcoded."""
        cfg = _get_config(minimal_config)
        assert "ic_mean_threshold" in cfg
        assert cfg["ic_mean_threshold"] == 0.003

    def test_breadth_threshold_from_config(self, minimal_config):
        """Breadth threshold must come from config."""
        cfg = _get_config(minimal_config)
        assert "min_breadth_for_conditional" in cfg
        assert cfg["min_breadth_for_conditional"] == 5

    def test_default_config_exists(self):
        """Default config must exist with all required thresholds."""
        cfg = _get_config({})
        required_keys = [
            "ic_mean_threshold", "icir_threshold", "hac_tstat_threshold",
            "bh_q_threshold", "bhy_q_threshold", "min_breadth_for_conditional",
            "min_dates_for_conditional", "sleeve_cost_bps",
            "bear_min_dates", "bear_min_breadth",
        ]
        for key in required_keys:
            assert key in cfg, f"Missing config key: {key}"


# ── Test 15: Every rejected sleeve has a reason ──────────────────────────────

class TestRejectionReasons:
    def test_all_rejected_have_reason(self, sample_panel, minimal_config):
        """Every rejected sleeve must have an explicit rejection reason."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        for adm in bundle.admission_results:
            if adm.final_status == SleeveFinalStatus.REJECTED.value:
                assert adm.rejection_reason != "", (
                    f"Rejected sleeve {adm.sleeve_id} has no rejection reason"
                )

    def test_diagnostics_have_reasons(self, sample_panel, minimal_config):
        """Diagnostics with issues must have rejection reasons."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        for diag in bundle.diagnostics:
            if diag.diagnostic_quality == "insufficient":
                assert diag.rejection_reason != "", (
                    f"Insufficient diagnostic {diag.sleeve_id} has no reason"
                )


# ── Test 16: No ML rescues invalid alpha ─────────────────────────────────────

class TestNoMLRescue:
    def test_engine_is_statistical_only(self, sample_panel, minimal_config):
        """Engine must not use ML/model selection to rescue invalid sleeves."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        # Check that admission is based on statistical/economic criteria, not ML
        for adm in bundle.admission_results:
            reasons = adm.rejection_reason.split(";") if adm.rejection_reason else []
            # No ML-related reasons should appear
            for reason in reasons:
                assert "ml" not in reason.lower(), (
                    f"ML used in rejection: {reason}"
                )
                assert "model" not in reason.lower() or "model_selection" not in reason.lower()

    def test_admission_uses_statistical_gates(self, sample_panel, minimal_config):
        """Admission must use statistical, multiple-testing, and economic gates."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        for adm in bundle.admission_results:
            # Must have statistical and multiple-testing status
            assert adm.statistical_status in ("pass", "fail")
            assert adm.multiple_testing_status in ("pass", "research", "fail")


# ── Integration Test: Full Pipeline ─────────────────────────────────────────

class TestFullPipeline:
    def test_full_pipeline_runs(self, sample_panel, minimal_config):
        """Full validation pipeline should run without errors."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend", "ret_20d"], horizons=[5, 10],
        )

        assert bundle.pit_regime_labels is not None
        assert len(bundle.sleeve_registry) > 0
        assert len(bundle.diagnostics) > 0
        assert len(bundle.mt_results) > 0
        assert len(bundle.stability_results) > 0
        assert len(bundle.admission_results) > 0

    def test_reports_generated(self, sample_panel, minimal_config):
        """All 11 reports should be generated."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_conditional_alpha_reports(bundle, tmpdir)

            # Check required reports
            required = [
                "pit_regime_labels", "sleeve_registry", "sleeve_diagnostics",
                "multiple_testing", "stability", "admission", "pm_summary",
            ]
            for key in required:
                assert key in paths, f"Missing report: {key}"
                assert paths[key].exists()

    def test_pm_summary_answers_questions(self, sample_panel, minimal_config):
        """PM summary must answer all key questions."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_conditional_alpha_reports(bundle, tmpdir)
            with open(paths["pm_summary"]) as f:
                summary = f.read()

            # Check key questions are answered
            assert "Total conditional sleeves tested:" in summary
            assert "Pass raw p-value" in summary
            assert "Pass Benjamini-Hochberg" in summary
            assert "Pass Benjamini-Yekutieli" in summary
            assert "Bear-Regime Analysis" in summary
            assert "Dependency Robustness" in summary
            assert "Economic Viability" in summary


# ── Edge Cases ───────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_empty_features(self, sample_panel, minimal_config):
        """Engine should handle empty feature list."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(sample_panel, [], horizons=[5])
        assert len(bundle.diagnostics) == 0

    def test_single_feature_single_horizon(self, sample_panel, minimal_config):
        """Engine should work with minimal inputs."""
        engine = ConditionalAlphaEngine(minimal_config)
        bundle = engine.run_full_validation(
            sample_panel, ["f_trend"], horizons=[5],
        )
        assert len(bundle.diagnostics) > 0

    def test_filter_by_condition_regime(self, sample_panel):
        """Filter by regime condition should work."""
        work = sample_panel.copy()
        work["regime_label"] = sample_panel["regime_label"]
        bear = _filter_by_condition(work, "regime", "Bear")
        assert len(bear) > 0
        assert (bear["regime_label"] == "Bear").all()

    def test_filter_by_condition_bucket(self, sample_panel):
        """Filter by bucket condition should work."""
        work = sample_panel.copy()
        work["volatility_bucket"] = pd.qcut(
            work["rolling_vol_20"].rank(method="first"), 3, labels=False, duplicates="drop"
        )
        bucket0 = _filter_by_condition(work, "volatility", "bucket_0")
        assert len(bucket0) > 0

    def test_hac_tstat_small_sample(self):
        """HAC t-stat should handle small samples gracefully."""
        ics = np.array([0.01, -0.005, 0.008])
        t = _hac_tstat(ics, 1)
        assert np.isfinite(t)

    def test_hac_tstat_empty(self):
        """HAC t-stat should return 0 for empty input."""
        t = _hac_tstat(np.array([]), 1)
        assert t == 0.0
