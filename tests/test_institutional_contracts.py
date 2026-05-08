from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pandas as pd

from model_selection.configuration import execution_cost_config, target_config
from model_selection.horizon_contract import build_horizon_contract
from model_selection.institutional_contracts import (
    RunTelemetryContract,
    build_cache_key_spec,
    build_cost_assumption_set,
    build_feature_manifest,
    build_horizon_run_contract,
    build_pit_audit_ledger,
    build_pit_transform_specs,
    build_promotion_gate_specs,
    build_target_manifest,
    build_target_specs,
    emit_telemetry_event,
    stable_fingerprint,
    write_institutional_run_manifest,
)
from model_selection.research_state import ResearchStateStore
from model_selection.validation import PromotionGateConfig
from model_selection.validation import evaluate_promotion_gates


ROOT = Path(__file__).resolve().parents[1]


def _base_cfg() -> dict:
    return {
        "model_selection": {
            "alpha_research": {"production_horizon": 10, "horizons": [1, 5, 10]},
            "target": {"residualize": True, "net_of_costs": False, "winsor_q": 0.01},
            "validation": {"capital": 1_000_000, "max_name_weight": 0.05},
            "promotion": {"gates": {"min_sharpe": 0.5}},
        },
        "horizon_config": {
            "target_horizon_days": 10,
            "holding_period_days": 10,
            "rebalance_frequency_days": 10,
        },
        "cost_model": {"commission_bps": 1.0, "spread_bps": 2.0, "borrow_bps": 25.0},
        "market_impact": {"eta": 0.142, "alpha": 0.314, "gamma": 0.6},
    }


def _contract_stack(cfg: dict | None = None):
    cfg = cfg or _base_cfg()
    hc = build_horizon_contract(cfg)
    hrc = build_horizon_run_contract(hc, cfg, config_path="cfg.yaml")
    tc = target_config(cfg, horizon_contract=hc)
    target_specs = build_target_specs(tc, cfg=cfg, horizon_contract=hrc)
    costs = build_cost_assumption_set(execution_cost_config(cfg), cfg=cfg)
    return cfg, hrc, tc, target_specs, costs


def test_horizon_run_contract_exposes_separated_horizon_fields() -> None:
    cfg, hrc, *_ = _contract_stack()
    assert hrc.target_horizon_days == 10
    assert hrc.ic_evaluation_horizon_days == 10
    assert hrc.holding_horizon_days == 10
    assert hrc.rebalance_horizon_days == 10
    assert hrc.decay_horizons == (1, 5, 10)
    assert hrc.sources["target_horizon_days"] == "horizon_config.target_horizon_days"
    assert hrc.fingerprint() == build_horizon_run_contract(build_horizon_contract(cfg), cfg, config_path="cfg.yaml").fingerprint()


def test_target_spec_fingerprint_changes_with_horizon_and_cost_policy() -> None:
    cfg, hrc, _, target_specs, _ = _contract_stack()
    first_fp = target_specs["target_return"].fingerprint()

    cfg_h20 = _base_cfg()
    cfg_h20["model_selection"]["alpha_research"]["production_horizon"] = 20
    cfg_h20["horizon_config"]["target_horizon_days"] = 20
    cfg_h20["horizon_config"]["holding_period_days"] = 20
    cfg_h20["horizon_config"]["rebalance_frequency_days"] = 20
    _, _, _, target_specs_h20, _ = _contract_stack(cfg_h20)
    assert first_fp != target_specs_h20["target_return"].fingerprint()

    cfg_cost = _base_cfg()
    cfg_cost["model_selection"]["strict_alpha_separation"] = False
    cfg_cost["model_selection"]["target"]["net_of_costs"] = True
    hc_cost = build_horizon_contract(cfg_cost)
    hrc_cost = build_horizon_run_contract(hc_cost, cfg_cost)
    specs_cost = build_target_specs(target_config(cfg_cost, horizon_contract=hc_cost), cfg=cfg_cost, horizon_contract=hrc_cost)
    assert first_fp != specs_cost["target_return"].fingerprint()
    assert hrc.target_horizon_days == 10


def test_feature_manifest_registers_known_features_and_reports_unknowns() -> None:
    manifest = build_feature_manifest(["ret_5d", "not_a_registered_feature"], cfg=_base_cfg())
    assert manifest["feature_fingerprints"]["ret_5d"]
    assert manifest["unknown_features"] == ["not_a_registered_feature"]


def test_pit_specs_and_ledger_are_machine_readable() -> None:
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "ticker": ["A", "A"],
            "regime_label": ["Bull", "Bear"],
            "sector": ["Tech", "Tech"],
        }
    )
    specs = build_pit_transform_specs(df, cfg=_base_cfg())
    assert set(specs) == {"regime_label", "sector"}
    rows = build_pit_audit_ledger(df, specs)
    assert {row["column_name"] for row in rows} == {"regime_label", "sector"}
    assert all(row["eligible_for_promotion_gates"] for row in rows)


def test_cost_assumption_fingerprint_changes_when_cost_policy_changes() -> None:
    cfg1 = _base_cfg()
    cfg2 = _base_cfg()
    cfg2["cost_model"]["spread_bps"] = 7.0
    cost1 = build_cost_assumption_set(execution_cost_config(cfg1), cfg=cfg1)
    cost2 = build_cost_assumption_set(execution_cost_config(cfg2), cfg=cfg2)
    assert cost1.fingerprint() != cost2.fingerprint()


def test_cache_key_spec_is_deterministic_across_processes() -> None:
    code = (
        "from model_selection.institutional_contracts import build_cache_key_spec\n"
        "cfg={'model_selection': {'alpha_research': {'production_horizon': 10}}}\n"
        "spec=build_cache_key_spec(cache_name='x', cfg=cfg, tickers=['MSFT','AAPL'], "
        "start_date='2020-01-01', end_date='2020-02-01', horizon_contract_fingerprint='h')\n"
        "print(spec.fingerprint())\n"
    )
    first = subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, text=True).strip()
    second = subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, text=True).strip()
    assert first == second


def test_research_state_accepts_explicit_contract_cache_fingerprint(tmp_path: Path) -> None:
    store = ResearchStateStore(
        root_dir=tmp_path,
        namespace="manual",
        payload={"cache_fingerprint": "fp_a"},
    )
    assert store.payload["cache_fingerprint"] == "fp_a"


def test_promotion_gate_specs_are_deterministic_and_severity_explicit() -> None:
    cfg, hrc, _, target_specs, costs = _contract_stack()
    specs = build_promotion_gate_specs(
        PromotionGateConfig(execution_robustness_affect_selection=False),
        horizon_contract_fingerprint=hrc.fingerprint(),
        target_spec_fingerprint=target_specs["target_return"].fingerprint(),
        cost_assumption_fingerprint=costs.fingerprint(),
    )
    by_name = {s.gate_name: s for s in specs}
    assert by_name["min_ic_tstat"].severity == "blocking"
    assert by_name["robust_turnover"].severity == "diagnostic"
    assert stable_fingerprint([s.to_dict() for s in specs]) == stable_fingerprint([s.to_dict() for s in specs])
    assert cfg


def test_diagnostic_promotion_gates_report_without_blocking() -> None:
    row = {
        "n_windows": 8,
        "oos_sharpe_chained": 1.0,
        "exec_sharpe": 0.8,
        "horizon_adj_ic_tstat": 3.0,
        "horizon_adj_ic_ir": 1.2,
        "oos_beat_rate": 0.8,
        "exec_max_dd": -0.1,
        "oos_psr": 0.9,
        "exec_beta_abs_mean": 0.01,
        "exec_max_sector_abs_mean": 0.01,
        "exec_cost_to_gross_pnl": 0.1,
        "decile_spread": 0.01,
        "decile_monotonicity": 0.8,
        "exec_long_leg_sharpe": 0.5,
        "exec_short_leg_sharpe": 0.5,
        "subsumption_alpha_ann": 0.1,
        "subsumption_alpha_tstat": 3.0,
        "subsumption_r2": 0.1,
        "subsumption_max_abs_loading": 0.5,
    }
    result = evaluate_promotion_gates(
        row,
        PromotionGateConfig(
            execution_robustness_enabled=True,
            execution_robustness_affect_selection=False,
            execution_robustness_fail_on_missing=True,
            dynamic_thresholds_enabled=False,
        ),
    )
    assert result["promotion_pass"] is True
    assert "diagnostic:robust_halflife" in result["promotion_failures"]


def test_telemetry_and_integrated_manifest(tmp_path: Path) -> None:
    cfg, hrc, _, target_specs, costs = _contract_stack()
    df = pd.DataFrame({"date": pd.to_datetime(["2020-01-01"]), "ticker": ["A"], "target_return": [0.1]})
    target_manifest = build_target_manifest(df, target_specs)
    pit_specs = build_pit_transform_specs(df, cfg=cfg)
    feature_manifest = build_feature_manifest(["ret_5d"], cfg=cfg, pit_specs=pit_specs)
    gate_specs = build_promotion_gate_specs(
        PromotionGateConfig(),
        horizon_contract_fingerprint=hrc.fingerprint(),
        target_spec_fingerprint=target_specs["target_return"].fingerprint(),
        cost_assumption_fingerprint=costs.fingerprint(),
    )
    cache_spec = build_cache_key_spec(
        cache_name="prepared_panel_cache",
        cfg=cfg,
        tickers=["A"],
        horizon_contract_fingerprint=hrc.fingerprint(),
        target_spec_fingerprints=list(target_manifest["target_spec_fingerprints"].values()),
        feature_spec_fingerprints=list(feature_manifest["feature_fingerprints"].values()),
        cost_assumption_fingerprint=costs.fingerprint(),
    )
    telemetry = RunTelemetryContract(run_id="run1", telemetry_dir=str(tmp_path))
    emit_telemetry_event(tmp_path, run_id="run1", stage="test", event_type="cache_miss", message="miss", ledger="cache_events.jsonl")
    manifest = write_institutional_run_manifest(
        tmp_path,
        run_id="run1",
        config_path="cfg.yaml",
        cfg=cfg,
        horizon_contract=hrc,
        target_manifest=target_manifest,
        pit_specs=pit_specs,
        feature_manifest=feature_manifest,
        cost_assumption_set=costs,
        promotion_gate_specs=gate_specs,
        cache_key_specs=[cache_spec],
        artifacts={"target_manifest": str(tmp_path / "target_manifest.json")},
        telemetry_contract=telemetry,
    )
    assert (tmp_path / "cache_events.jsonl").exists()
    path = tmp_path / "institutional_run_manifest.json"
    assert path.exists()
    loaded = json.loads(path.read_text())
    assert loaded["horizon_run_contract_fingerprint"] == hrc.fingerprint()
    assert manifest["cache_key_specs"]["prepared_panel_cache"] == cache_spec.fingerprint()
