from __future__ import annotations

import pytest
from model_selection.configuration import (
    alpha_admission_config,
    evaluation_config,
    horizon_contract_config,
    warn_deprecated_config_duplicates,
    target_config,
)
from model_selection.horizon_contract import (
    build_horizon_contract,
    HorizonConfigurationError,
    RebalanceMode,
)


def _base_cfg() -> dict:
    return {
        "horizon_config": {
            "target_horizon_days": 10,
            "holding_period_days": 10,
            "rebalance_frequency_days": 5,
            "ic_evaluation_horizon": 10,
            "execution_tau_days": None,
            "allow_cross_horizon_evaluation": True,
        },
        "backtest": {
            "lookahead_horizon_days": 5,
            "holding_period_days": 5,
            "optimization_config": {
                "execution": {
                    "horizon_days": 3,
                    "tau_exec": None,
                }
            },
        },
        "signals": {"weights": {"ic_horizon_days": 5}},
        "model_selection": {
            "lookahead_horizon_days": 10,
            "production_horizons": [10],
            "alpha_research": {"production_horizon": 10},
            "horizon_alignment": {"alignment_multiplier": 2.0},
            "nested_validation": {"search": {"candidate_horizons": [10]}},
            "validation": {},
            "target": {"residualize": True, "net_of_costs": False},
        },
    }


def test_horizon_contract_canonicalizes_legacy_fields_without_value_tuning() -> None:
    contract = build_horizon_contract(_base_cfg())

    assert contract.config.target_horizon_days == 10
    assert contract.config.holding_period_days == 10
    assert contract.config.rebalance_frequency_days == 5
    assert contract.config.ic_evaluation_horizon == 10
    assert contract.config.execution_tau_days is None
    assert "model_selection.lookahead_horizon_days" in contract.replaced_fields
    assert any("backtest.holding_period_days" in warning for warning in contract.warnings)
    assert any("backtest.optimization_config.execution.horizon_days" in warning for warning in contract.warnings)


def test_cli_horizon_sets_production_horizon_dims_inherit() -> None:
    """CLI --horizon sets production; other dims use explicit config or inherit."""
    contract = build_horizon_contract(_base_cfg(), cli_horizon=20)

    assert contract.config.production_horizon_days == 20
    assert contract.config.target_horizon_days == 10       # explicit in horizon_config
    assert contract.config.holding_period_days == 10       # explicit in horizon_config
    assert contract.config.ic_evaluation_horizon == 10     # explicit in horizon_config
    # rebalance uses horizon_config.rebalance_frequency_days=5 (legacy, explicit)
    assert contract.config.rebalance_frequency_days == 5
    assert contract.source_map["production_horizon_days"] == "cli.--horizon"
    assert contract.source_map["target_horizon_days"] == "horizon_config.target_horizon_days"
    assert contract.source_map["holding_period_days"] == "horizon_config.holding_period_days"
    assert contract.source_map["ic_evaluation_horizon"] == "horizon_config.ic_evaluation_horizon"


def test_configuration_helpers_derive_from_horizon_contract() -> None:
    cfg = _base_cfg()
    contract = horizon_contract_config(cfg)

    tgt = target_config(cfg, horizon_contract=contract)
    admission = alpha_admission_config(cfg, horizon_contract=contract)
    eval_cfg = evaluation_config(
        cfg,
        path="long_short_spread",
        max_positions=10,
        min_positions=3,
        horizon_contract=contract,
    )

    assert tgt.horizon_days == contract.config.target_horizon_days
    assert admission.production_horizon == contract.config.ic_evaluation_horizon
    assert admission.minimum_admitted_features == 0
    assert admission.fail_if_below_minimum is False
    assert admission.enforce_horizon_alignment is True
    assert admission.horizon_alignment_multiplier == 2.0
    assert eval_cfg.horizon_days == contract.config.target_horizon_days
    assert eval_cfg.rebalance_every_days == contract.config.rebalance_frequency_days


def test_deprecated_duplicate_config_warns_but_canonical_wins(caplog) -> None:
    cfg = _base_cfg()
    cfg["exposure_limits"] = {"lambda_risk": 2.0, "gamma_turnover": 2.0}
    cfg["model_selection"]["validation"]["lambda_risk"] = 9.0
    cfg["backtest"]["optimization_config"]["optimizer"] = {
        "lambda_risk": 7.0,
        "gamma_turnover": 8.0,
    }

    warn_deprecated_config_duplicates(cfg)
    eval_cfg = evaluation_config(
        cfg,
        path="long_short_spread",
        max_positions=10,
        min_positions=3,
    )

    assert eval_cfg.lambda_risk == 2.0
    assert eval_cfg.gamma_turnover == 2.0
    assert "Deprecated duplicate config field" in caplog.text


def test_horizon_contract_audit_rows_map_all_legacy_fields() -> None:
    contract = build_horizon_contract(_base_cfg())
    rows = contract.audit_rows()

    legacy_fields = {row["legacy_field"] for row in rows}
    assert "model_selection.alpha_research.production_horizon" in legacy_fields
    assert "backtest.rebalance_every_trading_days" in legacy_fields
    assert all(row["canonical_field"] for row in rows)


# ═══════════════════════════════════════════════════════════════════════════════
# P32: RebalancePolicy wiring tests
# ═══════════════════════════════════════════════════════════════════════════════


def _reb_base_cfg() -> dict:
    return {
        "horizon_config": {
            "target_horizon_days": 20,
            "holding_period_days": 20,
        },
        "backtest": {
            "rebalance_every_trading_days": 5,
        },
        "model_selection": {
            "alpha_research": {"production_horizon": 20},
            "horizon_alignment": {"alignment_multiplier": 2.0},
            "nested_validation": {"search": {"candidate_horizons": [20]}},
            "validation": {},
            "target": {"residualize": True, "net_of_costs": False},
        },
    }


def test_rebalance_policy_match_horizon_with_cli() -> None:
    """Explicit match_horizon policy: rebalance follows production (CLI or config)."""
    cfg = _reb_base_cfg()
    cfg["horizon_config"]["rebalance_policy"] = {"mode": "match_horizon"}
    contract = build_horizon_contract(cfg, cli_horizon=10)
    assert contract.config.rebalance_frequency_days == 10
    assert contract.source_map["rebalance_policy_mode"] == "match_horizon"
    assert contract.source_map["rebalance_frequency_days"] == "production_horizon_days"


def test_rebalance_policy_match_horizon_without_cli_uses_legacy() -> None:
    """No rebalance_policy configured: legacy config field controls rebalance."""
    cfg = _reb_base_cfg()
    # No rebalance_policy → defaults to match_horizon, falls back to legacy config
    contract = build_horizon_contract(cfg)
    assert contract.config.rebalance_frequency_days == 5  # from backtest.rebalance_every_trading_days
    assert contract.source_map["rebalance_policy_mode"] == "match_horizon"
    assert "backtest.rebalance_every_trading_days" in contract.source_map["rebalance_frequency_days"]


def test_rebalance_policy_halflife_aware_resolves_from_halflife_evidence() -> None:
    """HALFLIFE_AWARE with halflife=3d: rebalance = floor(3) = 3."""
    cfg = _reb_base_cfg()
    cfg["horizon_config"]["rebalance_policy"] = {
        "mode": "halflife_aware",
        "halflife_fallback_mode": "fail_closed",
    }
    contract = build_horizon_contract(cfg, halflife_days=3.7)
    assert contract.config.rebalance_frequency_days == 3  # floor(3.7)
    assert contract.source_map["rebalance_policy_mode"] == "halflife_aware"
    assert contract.source_map["rebalance_frequency_days"] == "horizon_config.rebalance_policy.halflife_aware"
    assert contract.source_map["rebalance_policy_halflife_days_supplied"] == "3.7"


def test_rebalance_policy_halflife_aware_capped_by_holding() -> None:
    """HALFLIFE_AWARE with halflife > holding: capped at holding_period."""
    cfg = _reb_base_cfg()
    cfg["horizon_config"]["rebalance_policy"] = {
        "mode": "halflife_aware",
    }
    contract = build_horizon_contract(cfg, halflife_days=100.0)
    # halflife=100d → floor=100, but capped at holding=20d
    assert contract.config.rebalance_frequency_days == 20
    assert contract.source_map["rebalance_policy_mode"] == "halflife_aware"


def test_rebalance_policy_halflife_aware_missing_halflife_fail_closed() -> None:
    """HALFLIFE_AWARE without halflife evidence: raises HorizonConfigurationError."""
    cfg = _reb_base_cfg()
    cfg["horizon_config"]["rebalance_policy"] = {
        "mode": "halflife_aware",
        "halflife_fallback_mode": "fail_closed",
    }
    with pytest.raises(HorizonConfigurationError, match="halflife_aware requires halflife_days"):
        build_horizon_contract(cfg)


def test_rebalance_policy_halflife_aware_missing_halflife_fallback_match_production() -> None:
    """HALFLIFE_AWARE without halflife: falls back to production_horizon."""
    cfg = _reb_base_cfg()
    cfg["horizon_config"]["rebalance_policy"] = {
        "mode": "halflife_aware",
        "halflife_fallback_mode": "match_production",
    }
    contract = build_horizon_contract(cfg)
    assert contract.config.rebalance_frequency_days == 20  # production_horizon
    assert "match_production" in contract.source_map["rebalance_frequency_days"]
    assert any("halflife_aware" in w and "match_production" in w for w in contract.warnings)


def test_rebalance_policy_halflife_aware_missing_halflife_fallback_legacy_config() -> None:
    """HALFLIFE_AWARE without halflife: falls back to legacy config."""
    cfg = _reb_base_cfg()
    cfg["horizon_config"]["rebalance_policy"] = {
        "mode": "halflife_aware",
        "halflife_fallback_mode": "legacy_config",
    }
    contract = build_horizon_contract(cfg)
    assert contract.config.rebalance_frequency_days == 5  # backtest.rebalance_every_trading_days
    assert "legacy_config" in contract.source_map["rebalance_frequency_days"]
    assert any("halflife_aware" in w and "legacy_config" in w for w in contract.warnings)


def test_rebalance_policy_overlap_mode() -> None:
    """OVERLAP mode: rebalance = min(frequency_days, holding - 1)."""
    cfg = _reb_base_cfg()
    cfg["horizon_config"]["rebalance_policy"] = {
        "mode": "overlap",
        "frequency_days": 10,
    }
    contract = build_horizon_contract(cfg)
    # holding=20d, frequency_days=10 → min(10, 19) = 10
    assert contract.config.rebalance_frequency_days == 10
    assert "overlap" in contract.source_map["rebalance_frequency_days"]


def test_rebalance_policy_fixed_mode() -> None:
    """FIXED mode: rebalance = frequency_days from policy."""
    cfg = _reb_base_cfg()
    cfg["horizon_config"]["rebalance_policy"] = {
        "mode": "fixed",
        "frequency_days": 7,
    }
    contract = build_horizon_contract(cfg)
    assert contract.config.rebalance_frequency_days == 7
    assert "fixed" in contract.source_map["rebalance_frequency_days"]


def test_rebalance_policy_unknown_mode_raises() -> None:
    """Unknown mode string raises HorizonConfigurationError."""
    cfg = _reb_base_cfg()
    cfg["horizon_config"]["rebalance_policy"] = {"mode": "invalid_mode"}
    with pytest.raises(HorizonConfigurationError, match="Unknown rebalance_policy.mode"):
        build_horizon_contract(cfg)


def test_rebalance_policy_audit_metadata_present() -> None:
    """All audit fields are present in source_map."""
    cfg = _reb_base_cfg()
    cfg["horizon_config"]["rebalance_policy"] = {
        "mode": "match_horizon",
        "frequency_days": None,
    }
    contract = build_horizon_contract(cfg)
    sm = contract.source_map
    assert "rebalance_policy_mode" in sm
    assert "rebalance_policy_source" in sm
    assert "rebalance_policy_frequency_days" in sm
    assert "rebalance_policy_halflife_days_supplied" in sm
    assert sm["rebalance_policy_mode"] == "match_horizon"
    assert sm["rebalance_policy_frequency_days"] == "none"
    assert sm["rebalance_policy_halflife_days_supplied"] == "none"


# ═══════════════════════════════════════════════════════════════════════════════
# P33: Dimension separation tests
# ═══════════════════════════════════════════════════════════════════════════════

def _dim_base_cfg() -> dict:
    return {
        "horizon_config": {},
        "backtest": {},
        "model_selection": {
            "alpha_research": {"production_horizon": 20},
            "horizon_alignment": {"alignment_multiplier": 2.0},
            "nested_validation": {"search": {"candidate_horizons": [20]}},
            "validation": {},
            "target": {"residualize": True, "net_of_costs": False},
        },
    }


def test_dimensions_all_inherit_production_by_default() -> None:
    """Without explicit horizon_config, all dims inherit production_horizon_days."""
    c = build_horizon_contract(_dim_base_cfg()).config
    assert c.target_horizon_days == 20
    assert c.holding_period_days == 20
    assert c.rebalance_frequency_days == 20
    assert c.ic_evaluation_horizon == 20


def test_dimensions_all_inherit_cli_production() -> None:
    """CLI --horizon changes production; all dims inherit if no explicit overrides."""
    c = build_horizon_contract(_dim_base_cfg(), cli_horizon=63).config
    assert c.production_horizon_days == 63
    assert c.target_horizon_days == 63
    assert c.holding_period_days == 63
    assert c.rebalance_frequency_days == 63
    assert c.ic_evaluation_horizon == 63


def test_holding_period_independent_of_cli() -> None:
    """Explicit holding_period_days survives CLI override."""
    cfg = _dim_base_cfg()
    cfg["horizon_config"]["holding_period_days"] = 10
    cfg["horizon_config"]["rebalance_policy"] = {"mode": "fixed", "frequency_days": 10}
    c = build_horizon_contract(cfg, cli_horizon=63).config
    assert c.production_horizon_days == 63
    assert c.target_horizon_days == 63       # inherits production
    assert c.holding_period_days == 10        # explicit override
    assert c.rebalance_frequency_days == 10   # from rebalance_policy.fixed


def test_target_horizon_independent_of_cli() -> None:
    """Explicit target_horizon_days survives CLI override."""
    cfg = _dim_base_cfg()
    cfg["horizon_config"]["target_horizon_days"] = 5
    c = build_horizon_contract(cfg, cli_horizon=63).config
    assert c.production_horizon_days == 63
    assert c.target_horizon_days == 5          # explicit override
    assert c.holding_period_days == 63         # inherits production


def test_dimensions_separated_source_tracking() -> None:
    """Each dimension records its independent resolution source."""
    cfg = _dim_base_cfg()
    cfg["horizon_config"]["holding_period_days"] = 10
    cfg["horizon_config"]["ic_evaluation_horizon"] = 20
    cfg["horizon_config"]["allow_cross_horizon_evaluation"] = True
    cfg["horizon_config"]["rebalance_policy"] = {"mode": "fixed", "frequency_days": 10}
    contract = build_horizon_contract(cfg, cli_horizon=63)
    sm = contract.source_map
    assert sm["production_horizon_days"] == "cli.--horizon"
    assert sm["target_horizon_days"] == "production_horizon_days"
    assert sm["holding_period_days"] == "horizon_config.holding_period_days"
    assert "rebalance_policy" in sm["rebalance_frequency_days"]
    assert sm["ic_evaluation_horizon"] == "horizon_config.ic_evaluation_horizon"


def test_dimensions_independent_warnings_on_divergence() -> None:
    """When a dimension differs from production, a warning is emitted."""
    cfg = _dim_base_cfg()
    cfg["horizon_config"]["holding_period_days"] = 5
    cfg["horizon_config"]["rebalance_policy"] = {"mode": "fixed", "frequency_days": 5}
    contract = build_horizon_contract(cfg)
    assert any("holding_period_days=5" in w for w in contract.warnings)


def test_dimensions_ic_eval_guard_unchanged() -> None:
    """ic_eval cross-horizon guard still raises if allow_cross_horizon_evaluation=false."""
    cfg = _dim_base_cfg()
    cfg["horizon_config"]["ic_evaluation_horizon"] = 10
    # allow_cross_horizon_evaluation defaults to false
    with pytest.raises(HorizonConfigurationError, match="ic_evaluation_horizon"):
        build_horizon_contract(cfg)


def test_dimensions_execution_tau_strictly_independent() -> None:
    """execution_tau_days is never collapsed — already independent."""
    cfg = _dim_base_cfg()
    cfg["horizon_config"]["execution_tau_days"] = 2.5
    c = build_horizon_contract(cfg, cli_horizon=63).config
    assert c.execution_tau_days == 2.5
