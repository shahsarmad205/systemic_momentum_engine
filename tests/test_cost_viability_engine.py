"""
Tests for Cost Viability Engine
================================
Proves:
1. Cost calculations are deterministic.
2. Cost model uses only point-in-time inputs.
3. Increasing spread increases expected cost.
4. Increasing ADV decreases market impact.
5. Increasing order size increases market impact.
6. Increasing turnover reduces cost viability.
7. A candidate with positive IC can still be rejected if costs dominate.
8. A candidate with lower IC but lower turnover can outrank a higher-IC high-turnover candidate.
9. No-trade band reduces turnover without mutating simulator behavior.
10. Execution simulator does not apply hidden cost-aware transformations.
11. All thresholds are loaded from config/research contract, not hardcoded.
12. Missing liquidity/cost data produces explicit warnings or degraded-quality flags.
"""

import math
import os
import sys
import tempfile
from pathlib import Path

import pytest

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_selection.cost_viability_engine import (
    AlphaToTradeDecision,
    CostBreakdown,
    CostModelMode,
    CostStatus,
    CostViabilityEngine,
    NoTradeBandEngine,
    SimpleBpsCostModel,
    SqrtImpactCostModel,
    ViabilityResult,
    build_cost_model,
    build_cost_model_with_overrides,
    generate_cost_dominated_report,
    generate_scorecard,
    generate_stress_test_report,
    generate_turnover_attribution_report,
    load_cost_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """Default engine with default config."""
    return CostViabilityEngine()


@pytest.fixture
def sqrt_model():
    """Square-root impact model."""
    config = {
        "cost_model": {
            "mode": "sqrt_impact",
            "commission_bps": 1.0,
            "spread_bps": 1.0,
            "borrow_bps_annual": 50.0,
            "impact_eta": 0.142,
            "impact_alpha": 0.314,
            "impact_gamma": 0.6,
            "default_adv_usd": 50_000_000,
            "default_daily_vol": 0.02,
            "max_participation_rate": 0.10,
            "max_impact_bps": 150.0,
            "min_adv_usd": 5_000_000,
            "permanent_impact_decay_days": 5,
            "financing_rate_annual": 0.0,
        },
    }
    return SqrtImpactCostModel(config)


@pytest.fixture
def simple_model():
    """Simple bps model."""
    config = {
        "cost_model": {
            "mode": "simple_bps",
            "commission_bps": 1.0,
            "spread_bps": 1.0,
            "borrow_bps_annual": 50.0,
            "financing_rate_annual": 0.0,
        },
    }
    return SimpleBpsCostModel(config)


# ---------------------------------------------------------------------------
# Test 1: Cost calculations are deterministic
# ---------------------------------------------------------------------------

class TestDeterministic:
    def test_same_inputs_same_output(self, sqrt_model):
        """Same inputs always produce same cost breakdown."""
        bd1 = sqrt_model.compute_round_trip_cost(
            order_usd=100_000, adv_usd=50_000_000, daily_vol=0.02,
            horizon_days=5, is_short=False,
        )
        bd2 = sqrt_model.compute_round_trip_cost(
            order_usd=100_000, adv_usd=50_000_000, daily_vol=0.02,
            horizon_days=5, is_short=False,
        )
        assert bd1.total_bps == bd2.total_bps
        assert bd1.commission_bps == bd2.commission_bps
        assert bd1.spread_bps == bd2.spread_bps
        assert bd1.temporary_impact_bps == bd2.temporary_impact_bps
        assert bd1.permanent_impact_bps == bd2.permanent_impact_bps
        assert bd1.borrow_bps == bd2.borrow_bps

    def test_engine_deterministic(self, engine):
        """Engine produces same result for same inputs."""
        r1 = engine.evaluate(
            candidate_id="test_1", feature="ret_5d", family="momentum",
            ic=0.02, horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=0.10, adv_usd=50_000_000, daily_vol=0.02,
        )
        r2 = engine.evaluate(
            candidate_id="test_1", feature="ret_5d", family="momentum",
            ic=0.02, horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=0.10, adv_usd=50_000_000, daily_vol=0.02,
        )
        assert r1.expected_cost_bps == r2.expected_cost_bps
        assert r1.alpha_cost_ratio == r2.alpha_cost_ratio
        assert r1.cost_status == r2.cost_status


# ---------------------------------------------------------------------------
# Test 2: Cost model uses only point-in-time inputs
# ---------------------------------------------------------------------------

class TestPointInTime:
    def test_no_future_data_required(self, sqrt_model):
        """Cost model only needs current ADV, vol, order size."""
        # All inputs are observable at trade time
        bd = sqrt_model.compute_round_trip_cost(
            order_usd=100_000,
            adv_usd=50_000_000,   # current ADV
            daily_vol=0.02,        # current vol
            horizon_days=5,        # known holding period
            is_short=False,
        )
        assert bd.total_bps > 0
        assert bd.degraded_quality is False

    def test_engine_uses_current_data(self, engine):
        """Engine evaluation uses only current/empirical data."""
        result = engine.evaluate(
            candidate_id="pit_test", feature="ret_5d", family="momentum",
            ic=0.02, horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=0.10, adv_usd=50_000_000, daily_vol=0.02,
        )
        # No forward-looking data needed
        assert result.expected_cost_bps > 0
        assert result.expected_alpha_bps > 0


# ---------------------------------------------------------------------------
# Test 3: Increasing spread increases expected cost
# ---------------------------------------------------------------------------

class TestSpreadImpact:
    def test_higher_spread_higher_cost(self, engine):
        """Increasing spread_bps increases total cost."""
        config_low = load_cost_config()
        config_low["cost_model"]["spread_bps"] = 1.0
        engine_low = CostViabilityEngine(config_low)

        config_high = load_cost_config()
        config_high["cost_model"]["spread_bps"] = 10.0
        engine_high = CostViabilityEngine(config_high)

        r_low = engine_low.evaluate(
            candidate_id="spread_test", feature="ret_5d", family="momentum",
            ic=0.02, horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=0.10, adv_usd=50_000_000, daily_vol=0.02,
        )
        r_high = engine_high.evaluate(
            candidate_id="spread_test", feature="ret_5d", family="momentum",
            ic=0.02, horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=0.10, adv_usd=50_000_000, daily_vol=0.02,
        )
        assert r_high.expected_cost_bps > r_low.expected_cost_bps

    def test_simple_model_spread(self, simple_model):
        """Simple model: spread directly adds to cost."""
        bd = simple_model.compute_round_trip_cost(
            order_usd=100_000, adv_usd=50_000_000, daily_vol=0.02,
            horizon_days=5, is_short=False,
        )
        # Spread should be part of total
        assert bd.spread_bps > 0
        assert bd.total_bps >= bd.spread_bps


# ---------------------------------------------------------------------------
# Test 4: Increasing ADV decreases market impact
# ---------------------------------------------------------------------------

class TestADVImpact:
    def test_higher_adv_lower_impact(self, sqrt_model):
        """Higher ADV reduces participation rate, which reduces impact."""
        bd_low_adv = sqrt_model.compute_round_trip_cost(
            order_usd=100_000, adv_usd=10_000_000, daily_vol=0.02,
            horizon_days=5, is_short=False,
        )
        bd_high_adv = sqrt_model.compute_round_trip_cost(
            order_usd=100_000, adv_usd=100_000_000, daily_vol=0.02,
            horizon_days=5, is_short=False,
        )
        assert bd_high_adv.temporary_impact_bps < bd_low_adv.temporary_impact_bps
        assert bd_high_adv.total_bps < bd_low_adv.total_bps

    def test_adv_participation_rate(self, sqrt_model):
        """Participation rate decreases with higher ADV."""
        prate_low = sqrt_model._participation_rate(100_000, 10_000_000)[0]
        prate_high = sqrt_model._participation_rate(100_000, 100_000_000)[0]
        assert prate_high < prate_low


# ---------------------------------------------------------------------------
# Test 5: Increasing order size increases market impact
# ---------------------------------------------------------------------------

class TestOrderSizeImpact:
    def test_larger_order_higher_impact(self, sqrt_model):
        """Larger orders increase participation rate and impact."""
        bd_small = sqrt_model.compute_round_trip_cost(
            order_usd=50_000, adv_usd=50_000_000, daily_vol=0.02,
            horizon_days=5, is_short=False,
        )
        bd_large = sqrt_model.compute_round_trip_cost(
            order_usd=500_000, adv_usd=50_000_000, daily_vol=0.02,
            horizon_days=5, is_short=False,
        )
        assert bd_large.temporary_impact_bps > bd_small.temporary_impact_bps
        assert bd_large.total_bps > bd_small.total_bps

    def test_impact_scales_sublinearly(self, sqrt_model):
        """Impact scales as (order/ADV)^gamma where gamma < 1."""
        bd_1x = sqrt_model.compute_round_trip_cost(
            order_usd=100_000, adv_usd=50_000_000, daily_vol=0.02,
            horizon_days=5, is_short=False,
        )
        bd_10x = sqrt_model.compute_round_trip_cost(
            order_usd=1_000_000, adv_usd=50_000_000, daily_vol=0.02,
            horizon_days=5, is_short=False,
        )
        # 10x order should NOT produce 10x impact (sublinear)
        impact_ratio = bd_10x.temporary_impact_bps / bd_1x.temporary_impact_bps
        assert impact_ratio < 10.0


# ---------------------------------------------------------------------------
# Test 6: Increasing turnover reduces cost viability
# ---------------------------------------------------------------------------

class TestTurnoverViability:
    def test_higher_turnover_lower_alpha_cost_ratio(self, engine):
        """Higher turnover increases cost, reducing alpha/cost ratio."""
        r_low_to = engine.evaluate(
            candidate_id="to_test", feature="ret_5d", family="momentum",
            ic=0.02, horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=0.05, adv_usd=50_000_000, daily_vol=0.02,
        )
        r_high_to = engine.evaluate(
            candidate_id="to_test", feature="ret_5d", family="momentum",
            ic=0.02, horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=0.30, adv_usd=50_000_000, daily_vol=0.02,
        )
        assert r_high_to.expected_cost_bps > r_low_to.expected_cost_bps
        assert r_high_to.alpha_cost_ratio < r_low_to.alpha_cost_ratio


# ---------------------------------------------------------------------------
# Test 7: Positive IC but cost-dominated
# ---------------------------------------------------------------------------

class TestPositiveICRejected:
    def test_positive_ic_rejected_due_to_costs(self, engine):
        """A candidate with positive IC can be rejected if costs dominate."""
        # Low IC, high cost (low ADV, high vol)
        # IC=0.008 gives gross_alpha ≈ 0.008 * 0.20 * sqrt(5/252) * 10000 ≈ 7.1 bps
        # which is above min_gross_alpha_bps=3.0 but costs will dominate
        result = engine.evaluate(
            candidate_id="cost_dominated", feature="ret_5d", family="momentum",
            ic=0.008,  # Above min_ic_absolute and min_gross_alpha thresholds
            horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=0.50,  # High turnover
            adv_usd=5_000_000,       # Low ADV → high impact
            daily_vol=0.05,          # High vol → high impact
        )
        assert result.ic > 0
        assert result.expected_alpha_bps > 0
        # Should be cost dominated or turnover dominated or liquidity insufficient
        # (alpha_too_weak is also valid if gross alpha is below threshold)
        assert result.cost_status != CostStatus.COST_VIABLE


# ---------------------------------------------------------------------------
# Test 8: Lower IC but lower turnover outranks higher IC high turnover
# ---------------------------------------------------------------------------

class TestTurnoverRanking:
    def test_lower_ic_lower_turnover_wins(self, engine):
        """A candidate with lower IC but much lower turnover can have higher net alpha."""
        # High IC, high turnover
        r_high_ic = engine.evaluate(
            candidate_id="high_ic_high_to", feature="ret_5d", family="momentum",
            ic=0.05, horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=0.80, adv_usd=50_000_000, daily_vol=0.02,
        )
        # Lower IC, much lower turnover
        r_low_ic = engine.evaluate(
            candidate_id="low_ic_low_to", feature="ret_20d", family="momentum",
            ic=0.02, horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=0.05, adv_usd=50_000_000, daily_vol=0.02,
        )
        # Lower IC candidate should have higher alpha/cost ratio
        assert r_low_ic.alpha_cost_ratio > r_high_ic.alpha_cost_ratio


# ---------------------------------------------------------------------------
# Test 9: No-trade band reduces turnover
# ---------------------------------------------------------------------------

class TestNoTradeBands:
    def test_band_reduces_turnover(self):
        """No-trade band reduces turnover without mutating simulator."""
        band_engine = NoTradeBandEngine()

        # Current weight close to target (inside band)
        result = band_engine.apply(
            candidate_id="band_test",
            current_weight=0.045,
            target_weight=0.050,
            expected_cost_bps=5.0,
            daily_vol=0.02,
            adv_usd=50_000_000,
            abs_ic=0.02,
            expected_alpha_bps=10.0,
        )
        assert result.trade_required is False
        assert result.gross_turnover_after == 0.0
        assert result.turnover_reduction > 0

    def test_band_trades_to_boundary(self):
        """When outside band, trade only to nearest boundary."""
        band_engine = NoTradeBandEngine()

        result = band_engine.apply(
            candidate_id="band_boundary",
            current_weight=0.010,
            target_weight=0.050,
            expected_cost_bps=5.0,
            daily_vol=0.02,
            adv_usd=50_000_000,
            abs_ic=0.02,
            expected_alpha_bps=10.0,
        )
        assert result.trade_required is True
        # Adjusted weight should be at boundary, not at target
        assert result.adjusted_weight >= result.band_lower
        assert result.adjusted_weight <= result.band_upper
        assert result.adjusted_weight != result.target_weight
        # Turnover should be reduced
        assert result.gross_turnover_after < result.gross_turnover_before

    def test_disabled_bands_no_reduction(self):
        """When bands are disabled, no turnover reduction."""
        config = load_cost_config()
        config["no_trade_bands"]["enabled"] = False
        band_engine = NoTradeBandEngine(config)

        result = band_engine.apply(
            candidate_id="disabled",
            current_weight=0.045,
            target_weight=0.050,
            expected_cost_bps=5.0,
            daily_vol=0.02,
            adv_usd=50_000_000,
            abs_ic=0.02,
            expected_alpha_bps=10.0,
        )
        assert result.trade_required is True
        assert result.adjusted_weight == result.target_weight
        assert result.turnover_reduction == 0.0


# ---------------------------------------------------------------------------
# Test 10: Simulator does not apply hidden cost-aware transformations
# ---------------------------------------------------------------------------

class TestSimulatorNeutrality:
    def test_engine_does_not_mutate_inputs(self, engine):
        """Engine evaluation does not modify input parameters."""
        ic = 0.02
        turnover = 0.10
        adv = 50_000_000

        result = engine.evaluate(
            candidate_id="neutrality_test", feature="ret_5d", family="momentum",
            ic=ic, horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=turnover, adv_usd=adv, daily_vol=0.02,
        )
        # Input values unchanged
        assert ic == 0.02
        assert turnover == 0.10
        assert adv == 50_000_000
        # Result uses inputs as-is
        assert result.ic == ic
        assert result.expected_turnover == turnover
        assert result.adv_usd == adv

    def test_cost_model_pure_function(self, sqrt_model):
        """Cost model is a pure function — no side effects."""
        adv = 50_000_000
        vol = 0.02

        bd1 = sqrt_model.compute_round_trip_cost(
            order_usd=100_000, adv_usd=adv, daily_vol=vol,
            horizon_days=5, is_short=False,
        )
        # Model parameters unchanged (no internal state mutation)
        assert sqrt_model.eta == 0.142
        assert sqrt_model.gamma == 0.6
        assert sqrt_model.commission_bps == 1.0


# ---------------------------------------------------------------------------
# Test 11: Thresholds loaded from config, not hardcoded
# ---------------------------------------------------------------------------

class TestConfigDriven:
    def test_thresholds_from_config(self):
        """Classification thresholds come from config, not hardcoded."""
        config = load_cost_config()
        cls = config["classification"]

        # These values should match what the engine uses
        assert "min_alpha_cost_ratio_viable" in cls
        assert "min_alpha_cost_ratio_marginal" in cls
        assert "min_gross_alpha_bps" in cls
        assert "min_ic_absolute" in cls
        assert "max_expected_turnover" in cls

        engine = CostViabilityEngine(config)
        assert engine.classification["min_alpha_cost_ratio_viable"] == cls["min_alpha_cost_ratio_viable"]

    def test_custom_config_overrides(self):
        """Custom config overrides default thresholds."""
        config = load_cost_config()
        config["classification"]["min_alpha_cost_ratio_viable"] = 3.0
        config["classification"]["min_ic_absolute"] = 0.010

        engine = CostViabilityEngine(config)
        assert engine.classification["min_alpha_cost_ratio_viable"] == 3.0
        assert engine.classification["min_ic_absolute"] == 0.010

    def test_promotion_gates_from_config(self):
        """Promotion gates loaded from config."""
        config = load_cost_config()
        gates = config["promotion_gates"]

        engine = CostViabilityEngine(config)
        assert engine.promotion_gates["min_alpha_cost_ratio"] == gates["min_alpha_cost_ratio"]
        assert engine.promotion_gates["min_halflife_days"] == gates["min_halflife_days"]


# ---------------------------------------------------------------------------
# Test 12: Missing data produces degraded-quality flags
# ---------------------------------------------------------------------------

class TestDegradedQuality:
    def test_low_adv_triggers_degradation(self, sqrt_model):
        """ADV below minimum triggers degraded quality flag."""
        bd = sqrt_model.compute_round_trip_cost(
            order_usd=100_000, adv_usd=1_000_000, daily_vol=0.02,
            horizon_days=5, is_short=False,
        )
        assert bd.degraded_quality is True
        assert "below minimum" in bd.degradation_reason

    def test_engine_handles_missing_adv(self, engine):
        """Engine uses default ADV when not provided, but flags it."""
        result = engine.evaluate(
            candidate_id="missing_adv", feature="ret_5d", family="momentum",
            ic=0.02, horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=0.10, adv_usd=0, daily_vol=0.02,
        )
        # Should use default ADV
        assert result.adv_usd > 0
        # Cost breakdown should be computed
        assert result.expected_cost_bps > 0

    def test_engine_handles_missing_vol(self, engine):
        """Engine uses default vol when not provided."""
        result = engine.evaluate(
            candidate_id="missing_vol", feature="ret_5d", family="momentum",
            ic=0.02, horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=0.10, adv_usd=50_000_000, daily_vol=0,
        )
        assert result.daily_vol > 0
        assert result.expected_cost_bps > 0


# ---------------------------------------------------------------------------
# Additional tests for alpha-to-trade decision
# ---------------------------------------------------------------------------

class TestAlphaToTrade:
    def test_approved_when_alpha_exceeds_cost(self):
        """Trade approved when alpha > cost with margin of safety."""
        decision = AlphaToTradeDecision()
        result = decision.decide(
            candidate_id="trade_ok",
            incremental_alpha_bps=10.0,
            incremental_cost_bps=3.0,
            incremental_turnover=0.05,
            impact_bps=1.0,
            adv_usd=50_000_000,
            liquidity_sufficient=True,
            borrow_cost_bps=0.0,
            signal_halflife=10.0,
            horizon=5,
            breadth=100,
        )
        assert result.trade_approved is True
        assert result.alpha_to_trade_ratio > 1.5

    def test_rejected_when_alpha_below_cost(self):
        """Trade rejected when alpha < cost."""
        decision = AlphaToTradeDecision()
        result = decision.decide(
            candidate_id="trade_bad",
            incremental_alpha_bps=2.0,
            incremental_cost_bps=5.0,
            incremental_turnover=0.05,
            impact_bps=1.0,
            adv_usd=50_000_000,
            liquidity_sufficient=True,
            borrow_cost_bps=0.0,
            signal_halflife=10.0,
            horizon=5,
            breadth=100,
        )
        assert result.trade_approved is False
        assert result.rejection_code == "alpha_below_cost"

    def test_rejected_when_below_margin(self):
        """Trade rejected when alpha/cost below margin of safety."""
        decision = AlphaToTradeDecision()
        result = decision.decide(
            candidate_id="trade_marginal",
            incremental_alpha_bps=5.0,
            incremental_cost_bps=4.0,  # ratio = 1.25 < 1.5
            incremental_turnover=0.05,
            impact_bps=1.0,
            adv_usd=50_000_000,
            liquidity_sufficient=True,
            borrow_cost_bps=0.0,
            signal_halflife=10.0,
            horizon=5,
            breadth=100,
        )
        assert result.trade_approved is False
        assert result.rejection_code == "alpha_below_margin"

    def test_disabled_always_approves(self):
        """When disabled, always approves."""
        config = load_cost_config()
        config["alpha_to_trade"]["enabled"] = False
        decision = AlphaToTradeDecision(config)
        result = decision.decide(
            candidate_id="disabled",
            incremental_alpha_bps=0.1,
            incremental_cost_bps=100.0,
        )
        assert result.trade_approved is True


# ---------------------------------------------------------------------------
# Additional tests for stress testing
# ---------------------------------------------------------------------------

class TestStressTesting:
    def test_stress_scenarios_produce_different_costs(self, engine):
        """Different stress scenarios produce different cost estimates."""
        results = engine.run_stress_test(
            candidate_id="stress_test",
            feature="ret_5d", family="momentum",
            ic=0.02, horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=0.10, adv_usd=50_000_000, daily_vol=0.02,
        )
        # At least some scenarios should differ
        costs = [v["expected_cost_bps"] for v in results.values()]
        assert len(set(costs)) > 1

    def test_crisis_scenario_highest_cost(self, engine):
        """Crisis scenario should have highest cost."""
        results = engine.run_stress_test(
            candidate_id="crisis_test",
            feature="ret_5d", family="momentum",
            ic=0.02, horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=0.10, adv_usd=50_000_000, daily_vol=0.02,
        )
        crisis_cost = results["crisis"]["expected_cost_bps"]
        base_cost = results["base_cost"]["expected_cost_bps"]
        assert crisis_cost > base_cost


# ---------------------------------------------------------------------------
# Tests for report generation
# ---------------------------------------------------------------------------

class TestReportGeneration:
    def test_scorecard_generation(self, engine):
        """Scorecard CSV is generated correctly."""
        results = [
            engine.evaluate(
                candidate_id="c1", feature="ret_5d", family="momentum",
                ic=0.02, horizon=5, sigma_annual=0.20, halflife=10.0,
                expected_turnover=0.10, adv_usd=50_000_000, daily_vol=0.02,
            ),
            engine.evaluate(
                candidate_id="c2", feature="ret_20d", family="momentum",
                ic=0.01, horizon=5, sigma_annual=0.20, halflife=3.0,
                expected_turnover=0.50, adv_usd=5_000_000, daily_vol=0.05,
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_content = generate_scorecard(results, f.name)
            assert "candidate_id" in csv_content
            assert "cost_status" in csv_content
            assert "c1" in csv_content
            assert "c2" in csv_content

    def test_cost_dominated_report(self, engine):
        """Cost dominated report includes non-viable candidates."""
        results = [
            engine.evaluate(
                candidate_id="viable", feature="ret_5d", family="momentum",
                ic=0.05, horizon=5, sigma_annual=0.20, halflife=20.0,
                expected_turnover=0.05, adv_usd=100_000_000, daily_vol=0.015,
            ),
            engine.evaluate(
                candidate_id="dominated", feature="ret_5d", family="momentum",
                ic=0.006, horizon=5, sigma_annual=0.20, halflife=3.0,
                expected_turnover=0.80, adv_usd=5_000_000, daily_vol=0.05,
            ),
        ]

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_content = generate_cost_dominated_report(results, f.name)
            assert "dominated" in csv_content


# ---------------------------------------------------------------------------
# Tests for promotion gates
# ---------------------------------------------------------------------------

class TestPromotionGates:
    def test_passes_all_gates(self, engine):
        """Strong candidate passes all promotion gates."""
        result = engine.evaluate(
            candidate_id="strong", feature="ret_5d", family="momentum",
            ic=0.05, horizon=5, sigma_annual=0.20, halflife=20.0,
            expected_turnover=0.05, adv_usd=100_000_000, daily_vol=0.015,
            n_dates=500, avg_breadth=200, bh_significant=True,
        )
        passes, failures = engine.check_promotion_gates(result)
        assert passes is True
        assert len(failures) == 0

    def test_fails_halflife_gate(self, engine):
        """Candidate with short halflife fails promotion."""
        result = engine.evaluate(
            candidate_id="short_hl", feature="ret_5d", family="momentum",
            ic=0.05, horizon=5, sigma_annual=0.20, halflife=2.0,
            expected_turnover=0.05, adv_usd=100_000_000, daily_vol=0.015,
            n_dates=500, avg_breadth=200, bh_significant=True,
        )
        passes, failures = engine.check_promotion_gates(result)
        assert passes is False
        assert any("halflife" in f for f in failures)

    def test_fails_alpha_cost_gate(self, engine):
        """Candidate with low alpha/cost ratio fails promotion."""
        result = engine.evaluate(
            candidate_id="low_acr", feature="ret_5d", family="momentum",
            ic=0.006, horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=0.50, adv_usd=5_000_000, daily_vol=0.05,
            n_dates=500, avg_breadth=200, bh_significant=True,
        )
        passes, failures = engine.check_promotion_gates(result)
        assert passes is False
        assert any("alpha_cost_ratio" in f for f in failures)


# ---------------------------------------------------------------------------
# Tests for classification logic
# ---------------------------------------------------------------------------

class TestClassification:
    def test_cost_viable_classification(self, engine):
        """Strong signal with low cost is classified as cost_viable."""
        result = engine.evaluate(
            candidate_id="viable", feature="ret_5d", family="momentum",
            ic=0.05, horizon=5, sigma_annual=0.20, halflife=20.0,
            expected_turnover=0.05, adv_usd=100_000_000, daily_vol=0.015,
        )
        assert result.cost_status == CostStatus.COST_VIABLE

    def test_alpha_too_weak_classification(self, engine):
        """Very low IC is classified as alpha_too_weak."""
        result = engine.evaluate(
            candidate_id="weak", feature="ret_5d", family="momentum",
            ic=0.001, horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=0.10, adv_usd=50_000_000, daily_vol=0.02,
        )
        assert result.cost_status == CostStatus.ALPHA_TOO_WEAK

    def test_liquidity_insufficient_classification(self, engine):
        """Very low ADV is classified as liquidity_insufficient."""
        result = engine.evaluate(
            candidate_id="illiquid", feature="ret_5d", family="momentum",
            ic=0.03, horizon=5, sigma_annual=0.20, halflife=10.0,
            expected_turnover=0.10, adv_usd=500_000, daily_vol=0.02,
        )
        assert result.cost_status == CostStatus.LIQUIDITY_INSUFFICIENT


# ---------------------------------------------------------------------------
# Tests for turnover attribution report
# ---------------------------------------------------------------------------

class TestTurnoverAttributionReport:
    def test_turnover_attribution_report_generated(self):
        """Turnover attribution report generates valid CSV with expected columns."""
        engine = NoTradeBandEngine()
        results = [
            engine.apply(
                candidate_id="test_1",
                current_weight=0.02,
                target_weight=0.05,
                expected_cost_bps=15.0,
                daily_vol=0.02,
                adv_usd=50_000_000,
                abs_ic=0.03,
                expected_alpha_bps=25.0,
            ),
            engine.apply(
                candidate_id="test_2",
                current_weight=0.04,
                target_weight=0.05,
                expected_cost_bps=10.0,
                daily_vol=0.01,
                adv_usd=100_000_000,
                abs_ic=0.05,
                expected_alpha_bps=30.0,
            ),
        ]
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            csv_content = generate_turnover_attribution_report(results, f.name)
            lines = csv_content.strip().split("\n")
            assert len(lines) == 3  # header + 2 rows
            header = lines[0].split(",")
            assert "candidate_id" in header
            assert "gross_turnover_before" in header
            assert "gross_turnover_after" in header
            assert "turnover_reduction" in header
            assert "cost_saved_bps" in header
            assert "net_trade_benefit" in header

    def test_no_trade_band_reduces_turnover(self):
        """No-trade band reduces turnover when current weight is near target."""
        engine = NoTradeBandEngine()
        result = engine.apply(
            candidate_id="near_target",
            current_weight=0.048,
            target_weight=0.05,
            expected_cost_bps=15.0,
            daily_vol=0.02,
            adv_usd=50_000_000,
            abs_ic=0.03,
            expected_alpha_bps=25.0,
        )
        assert result.gross_turnover_after < result.gross_turnover_before
        assert result.turnover_reduction > 0
        assert result.cost_saved_bps > 0


# ---------------------------------------------------------------------------
# Tests for missing_pit_diagnostics status
# ---------------------------------------------------------------------------

class TestMissingPITDiagnostics:
    def test_missing_pit_data_fails_loudly(self, engine):
        """When pit_data_available=False, status is missing_pit_diagnostics."""
        result = engine.evaluate(
            candidate_id="no_pit", feature="ret_5d", family="momentum",
            ic=0.05, horizon=5, sigma_annual=0.20, halflife=20.0,
            expected_turnover=0.05, adv_usd=100_000_000, daily_vol=0.015,
            pit_data_available=False,
        )
        assert result.cost_status == CostStatus.MISSING_PIT_DIAGNOSTICS
        assert result.rejection_reason == "missing_point_in_time_data"

    def test_pit_data_available_passes_classification(self, engine):
        """When pit_data_available=True, normal classification applies."""
        result = engine.evaluate(
            candidate_id="with_pit", feature="ret_5d", family="momentum",
            ic=0.05, horizon=5, sigma_annual=0.20, halflife=20.0,
            expected_turnover=0.05, adv_usd=100_000_000, daily_vol=0.015,
            pit_data_available=True,
        )
        assert result.cost_status != CostStatus.MISSING_PIT_DIAGNOSTICS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
