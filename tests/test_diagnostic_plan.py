from __future__ import annotations

from model_selection.diagnostic_plan import (
    DIAGNOSTIC_REGISTRY,
    DiagnosticExecutionPlan,
    resolve_diagnostics_config,
)


class TestDiagnosticExecutionPlan:
    def test_defaults_off(self) -> None:
        plan = DiagnosticExecutionPlan()
        assert not plan.research_diagnostics
        assert not plan.empirical_baselines
        assert not plan.any_enabled

    def test_defaults_off_from_empty_config(self) -> None:
        plan = DiagnosticExecutionPlan.from_config({})
        assert not plan.research_diagnostics
        assert not plan.empirical_baselines
        assert not plan.any_enabled

    def test_defaults_off_from_nested_empty(self) -> None:
        cfg = {"model_selection": {"validation": {"diagnostics": {}}}}
        plan = DiagnosticExecutionPlan.from_config(cfg)
        assert not plan.research_diagnostics
        assert not plan.empirical_baselines
        assert not plan.any_enabled

    def test_from_config_enables_research_diagnostics(self) -> None:
        cfg = {
            "model_selection": {
                "validation": {
                    "diagnostics": {"research_diagnostics": True},
                }
            }
        }
        plan = DiagnosticExecutionPlan.from_config(cfg)
        assert plan.research_diagnostics
        assert not plan.empirical_baselines
        assert plan.any_enabled

    def test_from_config_enables_empirical_baselines(self) -> None:
        cfg = {
            "model_selection": {
                "validation": {
                    "diagnostics": {"empirical_baselines": True},
                }
            }
        }
        plan = DiagnosticExecutionPlan.from_config(cfg)
        assert not plan.research_diagnostics
        assert plan.empirical_baselines
        assert plan.any_enabled

    def test_from_config_enables_both(self) -> None:
        cfg = {
            "model_selection": {
                "validation": {
                    "diagnostics": {
                        "research_diagnostics": True,
                        "empirical_baselines": True,
                    },
                }
            }
        }
        plan = DiagnosticExecutionPlan.from_config(cfg)
        assert plan.research_diagnostics
        assert plan.empirical_baselines
        assert plan.any_enabled

    def test_record_tracks_executed_candidates(self) -> None:
        plan = DiagnosticExecutionPlan(research_diagnostics=True)
        plan.record("research_diagnostics", "model_a")
        plan.record("research_diagnostics", "model_b")
        plan.record("empirical_baselines", "model_a")
        assert plan.executed["research_diagnostics"] == ["model_a", "model_b"]
        assert plan.executed["empirical_baselines"] == ["model_a"]

    def test_summary(self) -> None:
        plan = DiagnosticExecutionPlan(research_diagnostics=True)
        plan.record("research_diagnostics", "model_a")
        s = plan.summary
        assert "research_diagnostics=ON" in s
        assert "empirical_baselines=OFF" in s
        assert "research_diagnostics=1" in s

    def test_summary_none_executed(self) -> None:
        plan = DiagnosticExecutionPlan()
        s = plan.summary
        assert "research_diagnostics=OFF" in s
        assert "empirical_baselines=OFF" in s
        assert "executed" not in s

    def test_resolve_diagnostics_config_empty(self) -> None:
        resolved = resolve_diagnostics_config({})
        assert resolved["research_diagnostics"] is False
        assert resolved["empirical_baselines"] is False

    def test_resolve_diagnostics_config_none(self) -> None:
        resolved = resolve_diagnostics_config(None)
        assert resolved["research_diagnostics"] is False
        assert resolved["empirical_baselines"] is False

    def test_resolve_diagnostics_config_bool_values(self) -> None:
        resolved = resolve_diagnostics_config({"research_diagnostics": True, "empirical_baselines": False})
        assert resolved["research_diagnostics"] is True
        assert resolved["empirical_baselines"] is False

    def test_resolve_diagnostics_config_dict_value(self) -> None:
        resolved = resolve_diagnostics_config({"research_diagnostics": {"enabled": True}})
        assert resolved["research_diagnostics"] is True

    def test_resolve_diagnostics_config_dict_disabled(self) -> None:
        resolved = resolve_diagnostics_config({"research_diagnostics": {"enabled": False}})
        assert resolved["research_diagnostics"] is False

    def test_registry_all_entries_have_required_keys(self) -> None:
        required = {"description", "affects_gates", "module", "cost_profile", "default_enabled"}
        for name, spec in DIAGNOSTIC_REGISTRY.items():
            missing = required - set(spec.keys())
            assert not missing, f"{name} missing: {missing}"

    def test_registry_entries_states_valid(self) -> None:
        for name, spec in DIAGNOSTIC_REGISTRY.items():
            assert isinstance(spec["affects_gates"], bool), f"{name} affects_gates must be bool"

    def test_registry_none_affect_gates(self) -> None:
        for name, spec in DIAGNOSTIC_REGISTRY.items():
            assert not spec["affects_gates"], (
                f"{name} claims to affect gates — must stay in main path"
            )

    def test_research_diagnostics_default_off_in_registry(self) -> None:
        spec = DIAGNOSTIC_REGISTRY["research_diagnostics"]
        assert spec["default_enabled"] is False

    def test_empirical_baselines_default_off_in_registry(self) -> None:
        spec = DIAGNOSTIC_REGISTRY["empirical_baselines"]
        assert spec["default_enabled"] is False

    def test_config_section_path(self) -> None:
        cfg = {
            "model_selection": {
                "validation": {
                    "diagnostics": {
                        "research_diagnostics": True,
                        "empirical_baselines": True,
                    },
                }
            }
        }
        plan = DiagnosticExecutionPlan.from_config(cfg)
        assert plan.research_diagnostics
        assert plan.empirical_baselines

    def test_any_enabled_true_when_research_on(self) -> None:
        assert DiagnosticExecutionPlan(research_diagnostics=True).any_enabled

    def test_any_enabled_true_when_empirical_on(self) -> None:
        assert DiagnosticExecutionPlan(empirical_baselines=True).any_enabled

    def test_any_enabled_false_when_both_off(self) -> None:
        assert not DiagnosticExecutionPlan().any_enabled
