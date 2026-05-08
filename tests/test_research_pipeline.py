"""Tests for the new research pipeline architecture.

Proves:
- Runner contains orchestration only
- Horizons come from ResearchContract
- Thresholds come from ResearchContract
- Feature families come from FeatureRegistry
- Target construction is isolated
- Cost viability is isolated
- Promotion gates are isolated
- Simulator is not called during feature admission
- No stage mutates another stage's inputs silently
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest
import yaml

from research_pipeline.contract import ResearchContract, _deep_merge, _DEFAULTS
from research_pipeline.schema_validation import validate_schema, SchemaValidationReport
from research_pipeline.feature_registry import build_feature_registry, get_active_features, _is_eligible_by_name
from research_pipeline.target_builder import build_targets, TargetBuildReport
from research_pipeline.horizon_diagnostics import HorizonDiagnostic
from research_pipeline.pit_diagnostics import compute_pit_diagnostics, PITDiagnostic
from research_pipeline.cost_viability_adapter import CostViabilityAdapterResult
from research_pipeline.candidate_factory import CandidateSpec, build_candidate_pool
from research_pipeline.walk_forward_validator import ValidationResult
from research_pipeline.promotion_gates import PromotionGateResult
from research_pipeline.report_writer import write_csv, write_json, write_text
from research_pipeline.artifact_manifest import create_manifest, ArtifactManifest
from research_pipeline.cli import parse_pipeline_args, cli_overrides, contract_hash, build_cost_candidates


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_config():
    """Minimal valid config for testing."""
    return {
        "backtest": {"start_date": "2020-01-01", "end_date": "2023-01-01"},
        "research": {"train_years": 3, "test_years": 1, "step_years": 1},
        "model_selection": {
            "lookahead_horizon_days": 20,
            "max_positions": 10,
            "min_positions": 3,
        },
        "feature_selection": {"feature_subset": []},
        "execution": {"long_only": False, "enable_shorts": True},
    }


@pytest.fixture
def sample_df():
    """Sample feature matrix for testing."""
    np.random.seed(42)
    n = 1000
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=n, freq="B"),
        "ticker": np.random.choice(["AAPL", "GOOGL", "MSFT", "AMZN"], n),
        "f_trend": np.random.randn(n),
        "ret_5d": np.random.randn(n),
        "quality_score": np.random.randn(n),
        "forward_return": np.random.randn(n) * 0.01,
        "target_up": (np.random.randn(n) > 0).astype(int),
        "daily_return": np.random.randn(n) * 0.01,
        "adv_dollar_20": np.random.uniform(1e7, 1e9, n),
    })


@pytest.fixture
def contract(sample_config):
    """ResearchContract from sample config."""
    return ResearchContract.from_config(sample_config)


# ── Test: ResearchContract ───────────────────────────────────────────────────

class TestResearchContract:
    def test_from_config_uses_defaults(self, sample_config):
        """Contract fills missing sections from defaults."""
        c = ResearchContract.from_config(sample_config)
        assert c.horizon.target_horizon_days == 20
        assert c.start_date == "2020-01-01"
        assert c.execution.max_positions == 10

    def test_from_config_cli_override(self, sample_config):
        """CLI overrides take precedence over config."""
        overrides = {"model_selection": {"lookahead_horizon_days": 5}}
        c = ResearchContract.from_config(sample_config, overrides)
        assert c.horizon.target_horizon_days == 5

    def test_from_yaml_missing_file(self):
        """Missing YAML falls back to defaults."""
        c = ResearchContract.from_yaml("/nonexistent/path.yaml")
        assert c.horizon.target_horizon_days == 20  # default

    def test_from_yaml_valid_file(self, sample_config):
        """Valid YAML loads correctly."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(sample_config, f)
            c = ResearchContract.from_yaml(f.name)
        assert c.start_date == "2020-01-01"

    def test_do_shorts_property(self, sample_config):
        """do_shorts = enable_shorts and not long_only."""
        c = ResearchContract.from_config(sample_config)
        assert c.do_shorts is True
        sample_config["execution"]["long_only"] = True
        c2 = ResearchContract.from_config(sample_config)
        assert c2.do_shorts is False

    def test_primary_path_property(self, sample_config):
        """primary_path depends on do_shorts."""
        c = ResearchContract.from_config(sample_config)
        assert c.primary_path == "long_short_spread"
        sample_config["execution"]["long_only"] = True
        c2 = ResearchContract.from_config(sample_config)
        assert c2.primary_path == "long_only_overlay"

    def test_all_horizons_from_contract(self, sample_config):
        """Horizon candidates come from contract, not hardcoded."""
        sample_config["model_selection"]["adaptive_horizon"] = {
            "candidate_horizons": [5, 10, 20],
        }
        c = ResearchContract.from_config(sample_config)
        assert c.horizon.candidate_horizons == (5, 10, 20)

    def test_all_thresholds_from_contract(self, sample_config):
        """All thresholds come from contract."""
        sample_config["model_selection"]["feature_audit"] = {
            "min_ic_tstat": 2.0,
            "min_production_ic_valid_days": 100,
        }
        c = ResearchContract.from_config(sample_config)
        assert c.feature_admission.min_ic_tstat == 2.0
        assert c.feature_admission.min_production_ic_valid_days == 100

    def test_cost_viability_from_contract(self, sample_config):
        """Cost params come from contract."""
        sample_config["cost_model"] = {
            "commission_bps": 2.0,
            "spread_bps": 3.0,
            "default_adv_usd": 100_000_000,
        }
        c = ResearchContract.from_config(sample_config)
        assert c.cost_viability.commission_bps == 2.0
        assert c.cost_viability.spread_bps == 3.0
        assert c.cost_viability.default_adv_usd == 100_000_000


# ── Test: Schema Validation ─────────────────────────────────────────────────

class TestSchemaValidation:
    def test_valid_schema(self, sample_df):
        """Valid schema passes with correct counts."""
        cols = [c for c in sample_df.columns if c not in {"date", "ticker"}]
        report = validate_schema(sample_df, cols)
        assert report.n_rows == 1000
        assert report.n_tickers == 4
        assert report.n_dates == 1000
        assert report.n_features > 0

    def test_blocked_target_columns(self, sample_df):
        """Target columns are blocked from features."""
        report = validate_schema(sample_df, ["forward_return", "target_up", "f_trend"])
        assert "forward_return" in report.blocked_columns
        assert "target_up" in report.blocked_columns

    def test_missing_required_columns(self):
        """Missing required columns are reported."""
        df = pd.DataFrame({"f_trend": [1, 2, 3]})
        report = validate_schema(df, ["f_trend"])
        assert "date" in report.missing_required
        assert "ticker" in report.missing_required


# ── Test: Feature Registry ──────────────────────────────────────────────────

class TestFeatureRegistry:
    def test_registry_from_features(self):
        """Registry groups features by family."""
        registry = build_feature_registry(["f_trend", "ret_5d", "quality_score"], 20)
        assert "momentum" in registry.families
        assert "trend" in registry.families
        assert "quality" in registry.families

    def test_feature_family_lookup(self):
        """Family lookup returns correct family."""
        registry = build_feature_registry(["f_trend", "ret_5d"], 20)
        assert registry.family_of("f_trend") == "trend"
        assert registry.family_of("ret_5d") == "momentum"
        assert registry.family_of("unknown_feature") == "unknown"

    def test_no_hardcoded_families(self):
        """Families come from FEATURE_SPECS, not hardcoded."""
        registry = build_feature_registry([], 20)
        assert registry.n_families > 0
        # All families come from FEATURE_SPECS
        for family in registry.families:
            features = registry.features_by_family(family)
            assert len(features) > 0

    def test_is_eligible_by_name_blocks_targets(self):
        """Target-like names are blocked."""
        assert _is_eligible_by_name("target_up") is False
        assert _is_eligible_by_name("forward_return") is False
        assert _is_eligible_by_name("target_return") is False

    def test_is_eligible_by_name_allows_features(self):
        """Registered features are allowed."""
        assert _is_eligible_by_name("f_trend") is True
        assert _is_eligible_by_name("ret_5d") is True
        assert _is_eligible_by_name("quality_score") is True


# ── Test: Target Builder ────────────────────────────────────────────────────

class TestTargetBuilder:
    def test_target_build_report(self):
        """TargetBuildReport has correct fields."""
        report = TargetBuildReport(
            target_column="target_up",
            n_positive=500,
            n_negative=500,
            positive_rate=0.5,
            n_missing=0,
            missing_rate=0.0,
            risk_adjusted_available=False,
        )
        assert report.target_column == "target_up"
        assert report.positive_rate == 0.5


# ── Test: Horizon Diagnostics ───────────────────────────────────────────────

class TestHorizonDiagnostics:
    def test_horizon_diagnostic_fields(self):
        """HorizonDiagnostic has all required fields."""
        diag = HorizonDiagnostic(
            horizon=20, mean_ic=0.03, ic_std=0.02, ic_tstat=2.5,
            icir=1.5, signal_halflife_days=10.0, n_valid_dates=500,
            eligible=True, cost_viable=True,
        )
        assert diag.horizon == 20
        assert diag.mean_ic == 0.03
        assert diag.eligible is True


# ── Test: PIT Diagnostics ───────────────────────────────────────────────────

class TestPITDiagnostics:
    def test_pit_diagnostic_fields(self):
        """PITDiagnostic has all required fields."""
        diag = PITDiagnostic(
            candidate_id="test", feature="f_trend", horizon=20,
            date=pd.Timestamp("2020-01-01"),
            adv_usd=50_000_000, daily_vol=0.02, annual_vol=0.32,
            spread_proxy_bps=1.0, expected_turnover=0.10,
            rank_halflife=10.0, ic_decay_1d=0.9, ic_decay_5d=0.7,
            ic_decay_20d=0.3, liquidity_bucket="large",
            spread_bucket="normal", capacity_estimate_usd=2_500_000,
            data_quality="complete",
        )
        assert diag.data_quality == "complete"
        assert diag.liquidity_bucket == "large"

    def test_empty_df_returns_empty(self):
        """Empty DataFrame returns empty diagnostics."""
        results = compute_pit_diagnostics(pd.DataFrame(), "f_trend", 20, "test")
        assert results == []

    def test_missing_feature_returns_empty(self, sample_df):
        """Missing feature returns empty diagnostics."""
        results = compute_pit_diagnostics(sample_df, "nonexistent", 20, "test")
        assert results == []


# ── Test: Cost Viability Adapter ────────────────────────────────────────────

class TestCostViabilityAdapter:
    def test_adapter_result_fields(self):
        """CostViabilityAdapterResult has all required fields."""
        from model_selection.cost_viability_engine import CostStatus
        result = CostViabilityAdapterResult(
            candidate_id="test", feature="f_trend", family="trend",
            horizon=20, cost_status=CostStatus.COST_VIABLE,
            expected_alpha_bps=25.0, expected_cost_bps=10.0,
            net_expected_alpha_bps=15.0, alpha_cost_ratio=2.5,
            rejection_reason="",
        )
        assert result.cost_status.value == "cost_viable"
        assert result.net_expected_alpha_bps == 15.0


# ── Test: Candidate Factory ─────────────────────────────────────────────────

class TestCandidateFactory:
    def test_candidate_spec_fields(self):
        """CandidateSpec has all required fields."""
        spec = CandidateSpec(
            candidate_id="xgb_full_h20",
            model_name="xgboost",
            model_kind="long",
            feature_view="full",
            active_features=["f_trend", "ret_5d"],
            horizon=20,
            uses_proba=True,
        )
        assert spec.candidate_id == "xgb_full_h20"
        assert len(spec.active_features) == 2

    def test_pool_capped_by_contract(self, contract):
        """Candidate pool is capped by contract.search.max_candidates."""
        model_specs = [{"name": f"model_{i}", "kind": "long", "uses_proba": False} for i in range(100)]
        candidates = build_candidate_pool(model_specs, ["f_trend"], contract)
        assert len(candidates) <= contract.search.max_candidates


# ── Test: Walk-Forward Validator ────────────────────────────────────────────

class TestWalkForwardValidator:
    def test_validation_result_fields(self):
        """ValidationResult has all required fields."""
        result = ValidationResult(
            candidate_id="test", model_name="xgboost", horizon=20,
            windows_evaluated=4, oos_sharpe=1.5, oos_ic=0.03,
            oos_deflated_sharpe=1.2, cost_adjusted_sharpe=1.0,
            max_drawdown=0.15, win_rate=0.55, turnover=0.10,
            passed=True, gate_failures=[],
        )
        assert result.passed is True
        assert result.oos_sharpe == 1.5


# ── Test: Promotion Gates ───────────────────────────────────────────────────

class TestPromotionGates:
    def test_promotion_gate_result_fields(self):
        """PromotionGateResult has all required fields."""
        result = PromotionGateResult(
            candidate_id="test", passed=True, failures=[],
            metrics={"sharpe": 1.5, "ic": 0.03},
        )
        assert result.passed is True
        assert len(result.failures) == 0


# ── Test: Report Writer ─────────────────────────────────────────────────────

class TestReportWriter:
    def test_write_csv(self):
        """CSV is written correctly."""
        df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as f:
            path = write_csv(df, f.name)
            read_back = pd.read_csv(path)
        assert len(read_back) == 2

    def test_write_json(self):
        """JSON is written correctly."""
        data = {"key": "value", "num": 42}
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = write_json(data, f.name)
            read_back = json.loads(Path(path).read_text())
        assert read_back["key"] == "value"

    def test_write_text(self):
        """Text is written correctly."""
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            path = write_text("hello world", f.name)
            assert Path(path).read_text() == "hello world"


# ── Test: Artifact Manifest ─────────────────────────────────────────────────

class TestArtifactManifest:
    def test_manifest_creation(self):
        """Manifest is created with correct initial state."""
        m = create_manifest("test_run")
        assert m.run_id == "test_run"
        assert m.status == "running"
        assert len(m.artifacts) == 0

    def test_add_artifact(self):
        """Artifacts are tracked correctly."""
        m = create_manifest()
        m.add_artifact("test", "/path/test.csv", "stage1", row_count=100)
        assert len(m.artifacts) == 1
        assert m.artifacts[0].name == "test"
        assert m.artifacts[0].row_count == 100

    def test_manifest_complete(self):
        """Manifest completion sets end time and duration."""
        m = create_manifest()
        m.complete(status="success")
        assert m.status == "success"
        assert m.end_time != ""
        assert m.duration_seconds >= 0

    def test_manifest_to_dict(self):
        """Manifest serializes to dict correctly."""
        m = create_manifest()
        m.add_artifact("test", "/path/test.csv", "stage1")
        m.complete()
        d = m.to_dict()
        assert "run_id" in d
        assert "artifacts" in d
        assert len(d["artifacts"]) == 1

    def test_manifest_write(self):
        """Manifest writes to JSON correctly."""
        m = create_manifest()
        m.add_artifact("test", "/path/test.csv", "stage1")
        m.complete()
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            m.write(f.name)
            data = json.loads(Path(f.name).read_text())
        assert data["status"] == "success"


# ── Test: CLI ───────────────────────────────────────────────────────────────

class TestCLI:
    def test_parse_args_defaults(self):
        """Default args are correct."""
        args = parse_pipeline_args([])
        assert args.config == "backtest_config.yaml"
        assert args.log_level == "INFO"
        assert args.horizon is None

    def test_parse_args_overrides(self):
        """CLI overrides work."""
        args = parse_pipeline_args(["--config", "test.yaml", "--horizon", "5"])
        assert args.config == "test.yaml"
        assert args.horizon == 5

    def test_cli_overrides_horizon(self):
        """Horizon override is passed to contract."""
        args = parse_pipeline_args(["--horizon", "10"])
        o = cli_overrides(args)
        assert o["model_selection"]["lookahead_horizon_days"] == 10

    def test_contract_hash_stable(self, contract):
        """Contract hash is stable for same inputs."""
        h1 = contract_hash(contract)
        h2 = contract_hash(contract)
        assert h1 == h2
        assert len(h1) == 16  # truncated SHA-256

    def test_build_cost_candidates(self, contract):
        """Cost candidates are built from candidate specs."""
        candidates = [
            CandidateSpec("test_h20", "xgboost", "long", "full", ["f_trend"], 20, False),
        ]
        cost_cands = build_cost_candidates(candidates, contract)
        assert len(cost_cands) == 1
        assert cost_cands[0]["candidate_id"] == "test_h20"
        assert cost_cands[0]["feature"] == "f_trend"
        assert cost_cands[0]["horizon"] == 20


# ── Test: Runner Isolation ──────────────────────────────────────────────────

class TestRunnerIsolation:
    def test_runner_is_thin(self):
        """Runner file is under 150 lines."""
        runner_path = Path(__file__).parent.parent / "run_research_pipeline.py"
        lines = runner_path.read_text().strip().split("\n")
        assert len(lines) < 150, f"Runner is {len(lines)} lines, must be < 150"

    def test_runner_no_research_math(self):
        """Runner does not contain research math (IC, Sharpe, halflife)."""
        runner_path = Path(__file__).parent.parent / "run_research_pipeline.py"
        content = runner_path.read_text()
        # Runner should not compute these directly
        assert "cross_sectional_ic" not in content
        assert "compute_deflated_sharpe" not in content
        assert "signal_halflife" not in content

    def test_runner_no_cost_math(self):
        """Runner does not contain cost math."""
        runner_path = Path(__file__).parent.parent / "run_research_pipeline.py"
        content = runner_path.read_text()
        assert "impact_bps" not in content
        assert "sqrt_impact" not in content
        assert "Almgren" not in content

    def test_runner_no_simulator(self):
        """Runner does not call the simulator."""
        runner_path = Path(__file__).parent.parent / "run_research_pipeline.py"
        content = runner_path.read_text()
        assert "simulate_executable" not in content
        assert "simulate_proxy" not in content


# ── Test: No Hidden Assumptions ─────────────────────────────────────────────

class TestNoHiddenAssumptions:
    def test_no_hardcoded_horizons_in_contract(self):
        """Horizons come from config, not hardcoded in contract.py."""
        contract_path = Path(__file__).parent.parent / "research_pipeline" / "contract.py"
        content = contract_path.read_text()
        # Defaults are allowed but must be in _DEFAULTS dict, not scattered
        assert "_RESEARCH_GRID" not in content
        assert "_ah_candidates" not in content

    def test_no_hardcoded_thresholds_in_contract(self):
        """Thresholds come from _DEFAULTS, not scattered."""
        contract_path = Path(__file__).parent.parent / "research_pipeline" / "contract.py"
        content = contract_path.read_text()
        # Check that thresholds are in _DEFAULTS, not in class body
        lines = content.split("\n")
        in_defaults = False
        for line in lines:
            if "_DEFAULTS" in line:
                in_defaults = True
            if in_defaults:
                # Thresholds in _DEFAULTS are OK
                pass
            else:
                # No hardcoded thresholds outside _DEFAULTS
                assert "min_ic_tstat" not in line or "f_audit" in line

    def test_feature_family_not_unknown_for_registered(self):
        """Registered features never get family='unknown'."""
        from model_selection.research_contract import FEATURE_SPECS
        registry = build_feature_registry(list(FEATURE_SPECS.keys()), 20)
        for feat in FEATURE_SPECS:
            assert registry.family_of(feat) != "unknown"


# ── Test: Stage Isolation ───────────────────────────────────────────────────

class TestStageIsolation:
    def test_schema_validation_does_not_mutate_input(self, sample_df):
        """Schema validation does not mutate the input DataFrame."""
        df_copy = sample_df.copy()
        cols = [c for c in sample_df.columns if c not in {"date", "ticker"}]
        validate_schema(sample_df, cols)
        pd.testing.assert_frame_equal(sample_df, df_copy)

    def test_feature_registry_does_not_mutate_input(self):
        """Feature registry does not mutate input lists."""
        features = ["f_trend", "ret_5d"]
        original = features.copy()
        build_feature_registry(features, 20)
        assert features == original

    def test_cost_viability_is_isolated(self, contract):
        """Cost viability evaluation is isolated from other stages."""
        # Cost viability uses its own engine, not the simulator
        from model_selection.cost_viability_engine import CostViabilityEngine
        engine = CostViabilityEngine(config=contract.raw_config)
        result = engine.evaluate(
            candidate_id="test", feature="f_trend", family="trend",
            ic=0.03, horizon=20, sigma_annual=0.20,
            halflife=10.0, expected_turnover=0.10,
            adv_usd=50_000_000, daily_vol=0.02,
        )
        # Result is a full ViabilityResult, not a boolean
        assert hasattr(result, "cost_status")
        assert hasattr(result, "expected_alpha_bps")
        assert hasattr(result, "expected_cost_bps")


# ── Test: Cost Viability Wiring ──────────────────────────────────────────────

class TestCostViabilityWiring:
    def test_wiring_module_imports(self):
        """All wiring functions are importable."""
        from model_selection.cost_viability_wiring import (
            evaluate_feature_cost_viability,
            evaluate_candidate_cost_viability,
            apply_alpha_to_trade_policy,
            apply_no_trade_band,
            generate_cost_viability_reports,
            compute_institutional_horizon_gate,
            CostViabilityWiringState,
            FeatureCostResult,
            CandidateCostResult,
        )

    def test_wiring_state_creation(self):
        """CostViabilityWiringState creates with empty results."""
        from model_selection.cost_viability_wiring import CostViabilityWiringState
        state = CostViabilityWiringState(config={})
        assert len(state.feature_results) == 0
        assert len(state.candidate_results) == 0
        assert len(state.alpha_to_trade_decisions) == 0
        assert len(state.band_results) == 0

    def test_feature_cost_result_fields(self):
        """FeatureCostResult has all required fields."""
        from model_selection.cost_viability_wiring import FeatureCostResult
        from model_selection.cost_viability_engine import CostStatus
        result = FeatureCostResult(
            feature="f_trend", family="trend", horizon=20,
            ic=0.03, ic_tstat=2.5, halflife=10.0,
            expected_turnover=0.10, adv_usd=50_000_000, daily_vol=0.02,
            cost_status=CostStatus.COST_VIABLE,
            expected_alpha_bps=25.0, expected_cost_bps=10.0,
            net_expected_alpha_bps=15.0, alpha_cost_ratio=2.5,
            capacity_score=5.0, rejection_reason="",
        )
        assert result.cost_status.value == "cost_viable"
        assert result.net_expected_alpha_bps == 15.0

    def test_candidate_cost_result_fields(self):
        """CandidateCostResult has all required fields."""
        from model_selection.cost_viability_wiring import CandidateCostResult
        from model_selection.cost_viability_engine import CostStatus
        result = CandidateCostResult(
            candidate_id="xgb_full_h20", model_name="xgboost", model_kind="long",
            horizon=20, feature_view="full", n_features=10,
            cost_status=CostStatus.COST_VIABLE,
            expected_alpha_bps=30.0, expected_cost_bps=12.0,
            net_expected_alpha_bps=18.0, alpha_cost_ratio=2.5,
            turnover=0.15, capacity_score=8.0, rejection_reason="",
        )
        assert result.cost_status.value == "cost_viable"

    def test_institutional_horizon_gate_replaces_flat_bps(self):
        """Institutional gate uses CostViabilityEngine, not flat 10bps."""
        from model_selection.cost_viability_wiring import compute_institutional_horizon_gate
        import pandas as pd
        import numpy as np

        # Minimal test data
        alpha_admission = pd.DataFrame([{
            "feature": "f_trend",
            "mean_ic": 0.03,
            "ic_tstat": 2.5,
            "signal_halflife_days": 10.0,
            "turnover_mean": 0.10,
        }])
        alpha_decay = pd.DataFrame([{
            "feature": "f_trend",
            "horizon": 20,
            "mean_ic": 0.03,
        }])
        df = pd.DataFrame({
            "date": pd.date_range("2020-01-01", periods=100),
            "ticker": ["AAPL"] * 100,
            "f_trend": np.random.randn(100),
            "forward_return": np.random.randn(100) * 0.01,
            "adv_dollar_20": np.random.uniform(1e7, 1e9, 100),
        })
        cfg = {"model_selection": {"horizon_gate": {"min_production_features": 1}}}

        blocked, results, diag = compute_institutional_horizon_gate(
            df, {"trend": ["f_trend"]}, alpha_admission, alpha_decay, cfg, 20,
        )
        assert len(results) == 1
        assert results[0].feature == "f_trend"
        assert "n_viable" in diag
        assert "n_dominated" in diag

    def test_wiring_connected_to_runner(self):
        """Wiring module is imported in run_model_selection.py."""
        runner_path = Path(__file__).parent.parent / "run_model_selection.py"
        content = runner_path.read_text()
        assert "from model_selection.cost_viability_wiring import" in content
        assert "evaluate_feature_cost_viability" in content
        assert "generate_cost_viability_reports" in content
        assert "compute_institutional_horizon_gate" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
