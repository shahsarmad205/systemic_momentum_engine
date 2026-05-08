from __future__ import annotations

import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


CONTRACT_SCHEMA_VERSION = "phase_b.1"


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in sorted(value.items(), key=lambda item: str(item[0]))}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return str(value)
    return value


def stable_fingerprint(payload: Any, *, length: int = 24) -> str:
    encoded = json.dumps(
        _jsonable(payload),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[: int(length)]


def config_fingerprint(cfg: dict[str, Any]) -> str:
    return stable_fingerprint({"schema": "config", "payload": cfg})


def dataframe_fingerprint(df: pd.DataFrame, *, columns: list[str] | None = None) -> str:
    if df is None or df.empty:
        return "empty"
    keep = [c for c in (columns or list(df.columns)) if c in df.columns]
    work = df.loc[:, keep].copy()
    if "date" in work.columns:
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
    hashed = pd.util.hash_pandas_object(work, index=False, categorize=True)
    digest = hashlib.sha256()
    digest.update(json.dumps(list(work.columns), sort_keys=True).encode("utf-8"))
    digest.update(str(work.shape).encode("utf-8"))
    digest.update(hashed.to_numpy(dtype="uint64", copy=False).tobytes())
    return digest.hexdigest()[:24]


@dataclass(frozen=True)
class FingerprintedContract:
    version: str = CONTRACT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["fingerprint"] = self.fingerprint()
        return _jsonable(payload)

    def fingerprint(self) -> str:
        payload = asdict(self)
        payload.pop("fingerprint", None)
        return stable_fingerprint(payload)


@dataclass(frozen=True)
class HorizonRunContract(FingerprintedContract):
    target_horizon_days: int = 0
    ic_evaluation_horizon_days: int = 0
    holding_horizon_days: int = 0
    rebalance_horizon_days: int = 0
    decay_horizons: tuple[int, ...] = ()
    cost_evaluation_horizon_days: int = 0
    promotion_horizon_days: int = 0
    sources: dict[str, str] = field(default_factory=dict)
    config_path: str = ""
    config_fingerprint: str = ""
    validation_status: str = "unknown"
    validation_failures: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    base_contract: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetSpec(FingerprintedContract):
    target_column: str = ""
    horizon_days: int = 0
    return_column: str = "forward_return"
    price_source: str = "feature_panel.forward_return"
    forward_return_method: str = "prealigned_feature_builder_forward_return"
    residualization_method: str = "none"
    factor_model_version: str = ""
    sector_neutralization_policy: str = ""
    cost_adjustment_policy: str = "none"
    winsorization_policy: str = ""
    missing_data_policy: str = "fill_target_outputs_zero_after_nan_safe_processing"
    pit_boundary_policy: str = "feature_panel_forward_return_already_shifted_by_holding_period"
    config_fingerprint: str = ""
    target_config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PITTransformSpec(FingerprintedContract):
    transform_name: str = ""
    fit_window_type: str = "external"
    fit_start: str = ""
    fit_end_policy: str = "as_of_or_source_defined"
    transform_date_policy: str = "same_date_panel_join"
    min_history: int = 0
    lag_days: int = 0
    threshold_method: str = ""
    grouping_keys: tuple[str, ...] = ()
    missing_data_policy: str = "preserve_missing"
    diagnostic_only: bool = False
    eligible_for_promotion_gates: bool = True
    config_fingerprint: str = ""


@dataclass(frozen=True)
class InstitutionalFeatureSpec(FingerprintedContract):
    feature_name: str = ""
    family: str = "unknown"
    subfamily: str = ""
    source_columns: tuple[str, ...] = ()
    transformation_method: str = "registered_feature_builder_output"
    required_lags: tuple[int, ...] = ()
    pit_eligible: bool = True
    pit_transform_fingerprint: str = ""
    expected_frequency: str = "daily_panel"
    missing_data_policy: str = "model_preprocessor_train_medians"
    winsorization_policy: str = "model_preprocessor_train_quantiles"
    owner_module_path: str = "model_selection.research_contract.FEATURE_SPECS"
    deprecation_status: str = "active"
    config_fingerprint: str = ""
    legacy_contract: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CostAssumptionSet(FingerprintedContract):
    spread_model: dict[str, Any] = field(default_factory=dict)
    slippage_model: dict[str, Any] = field(default_factory=dict)
    impact_model: dict[str, Any] = field(default_factory=dict)
    commission_model: dict[str, Any] = field(default_factory=dict)
    borrow_model: dict[str, Any] = field(default_factory=dict)
    liquidity_adv_source: str = "adv_dollar_20_or_default_adv_usd"
    participation_cap_policy: dict[str, Any] = field(default_factory=dict)
    capacity_policy: dict[str, Any] = field(default_factory=dict)
    stress_scenarios: dict[str, Any] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    config_fingerprint: str = ""


@dataclass(frozen=True)
class CacheKeySpec(FingerprintedContract):
    cache_name: str = ""
    code_version: str = CONTRACT_SCHEMA_VERSION
    data_source_fingerprint: str = ""
    universe_ticker_fingerprint: str = ""
    date_range: dict[str, str] = field(default_factory=dict)
    horizon_contract_fingerprint: str = ""
    target_spec_fingerprints: tuple[str, ...] = ()
    feature_spec_fingerprints: tuple[str, ...] = ()
    cost_assumption_fingerprint: str = ""
    pit_transform_fingerprints: tuple[str, ...] = ()
    model_spec_fingerprint: str = ""
    split_fingerprint: str = ""
    score_vector_fingerprint: str = ""
    config_fingerprint: str = ""
    random_seed: str = ""
    library_versions: dict[str, str] = field(default_factory=dict)
    safety_classification: str = "production_deterministic"


@dataclass(frozen=True)
class PromotionGateSpec(FingerprintedContract):
    gate_name: str = ""
    severity: str = "blocking"
    metric_name: str = ""
    threshold_source: str = "PromotionGateConfig"
    horizon_source: str = "HorizonRunContract"
    required_input_contract_fingerprints: dict[str, str] = field(default_factory=dict)
    missing_data_behavior: str = "fail_closed_or_gate_specific_default"
    pass_fail_rule: str = ""
    reason_code: str = ""


@dataclass(frozen=True)
class PromotionDecision(FingerprintedContract):
    overall_status: str = "unknown"
    passed_gates: tuple[str, ...] = ()
    failed_blocking_gates: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    diagnostics: tuple[str, ...] = ()
    deterministic_reason_codes: tuple[str, ...] = ()
    metric_values_used: dict[str, Any] = field(default_factory=dict)
    threshold_values_used: dict[str, Any] = field(default_factory=dict)
    horizon_contract_fingerprint: str = ""
    target_spec_fingerprint: str = ""
    cost_assumption_fingerprint: str = ""
    cache_key_or_run_id: str = ""


@dataclass(frozen=True)
class RunTelemetryContract(FingerprintedContract):
    run_id: str = ""
    telemetry_dir: str = ""
    timing_events: str = "run_telemetry.jsonl"
    cache_events: str = "cache_events.jsonl"
    warning_events: str = "failure_ledger.jsonl"
    error_events: str = "failure_ledger.jsonl"
    diagnostic_failure_events: str = "failure_ledger.jsonl"
    promotion_decision_events: str = "run_telemetry.jsonl"
    artifact_write_events: str = "artifact_events.jsonl"


def build_horizon_run_contract(
    horizon_contract: Any,
    cfg: dict[str, Any],
    *,
    config_path: str = "",
) -> HorizonRunContract:
    c = horizon_contract.config
    ms = cfg.get("model_selection", {}) or {}
    alpha = (ms.get("alpha_research", {}) or {}) if isinstance(ms, dict) else {}
    raw_horizons = alpha.get("horizons", [1, 2, 3, 5, 10, 20])
    decay_horizons = tuple(sorted({max(1, int(h)) for h in raw_horizons if str(h).strip()}))
    failures = tuple(str(x) for x in horizon_contract.validate_alignment())
    status = "valid" if not failures else "warnings"
    return HorizonRunContract(
        target_horizon_days=int(c.target_horizon_days),
        ic_evaluation_horizon_days=int(c.ic_evaluation_horizon),
        holding_horizon_days=int(c.holding_period_days),
        rebalance_horizon_days=int(c.rebalance_frequency_days),
        decay_horizons=decay_horizons,
        cost_evaluation_horizon_days=int(c.target_horizon_days),
        promotion_horizon_days=int(c.production_horizon_days),
        sources=dict(horizon_contract.source_map),
        config_path=str(config_path),
        config_fingerprint=config_fingerprint(cfg),
        validation_status=status,
        validation_failures=failures,
        warnings=tuple(str(w) for w in horizon_contract.warnings),
        base_contract=horizon_contract.to_dict(),
    )


def build_target_specs(
    target_cfg: Any,
    *,
    cfg: dict[str, Any],
    horizon_contract: HorizonRunContract,
) -> dict[str, TargetSpec]:
    target_dict = _jsonable(asdict(target_cfg) if is_dataclass(target_cfg) else vars(target_cfg))
    residual = "datewise_factor_residualization" if bool(getattr(target_cfg, "residualize", False)) else "datewise_cross_sectional_demean"
    cost_policy = "subtract_expected_round_trip_cost" if bool(getattr(target_cfg, "net_of_costs", False)) else "pure_alpha_no_cost_subtraction"
    winsor = f"datewise_quantile_clip_q={float(getattr(target_cfg, 'winsor_q', 0.0)):.6g};max_abs_return={float(getattr(target_cfg, 'max_abs_return', 0.0)):.6g}"
    common = {
        "horizon_days": int(getattr(target_cfg, "horizon_days", horizon_contract.target_horizon_days)),
        "residualization_method": residual,
        "cost_adjustment_policy": cost_policy,
        "winsorization_policy": winsor,
        "config_fingerprint": config_fingerprint(cfg),
        "target_config": target_dict,
    }
    specs: dict[str, TargetSpec] = {}
    for col in (
        "forward_return",
        "target_return_net",
        "target_return",
        "target_rank",
        "target_down_decile",
        "target_up",
        "target_expected_cost",
        "target_expected_participation",
        "target_expected_fixed_cost",
        "target_expected_temporary_impact",
        "target_expected_permanent_impact",
        "target_expected_borrow_cost",
    ):
        specs[col] = TargetSpec(target_column=col, **common)
    return specs


def build_target_manifest(df: pd.DataFrame, target_specs: dict[str, TargetSpec]) -> dict[str, Any]:
    columns = []
    for name, spec in sorted(target_specs.items()):
        if df is not None and name in df.columns:
            series = df[name]
            nan_count = int(pd.to_numeric(series, errors="coerce").isna().sum()) if series.dtype != object else int(series.isna().sum())
            non_null = int(series.notna().sum())
        else:
            nan_count = None
            non_null = 0
        columns.append({
            "target_column": name,
            "fingerprint": spec.fingerprint(),
            "horizon_days": spec.horizon_days,
            "formula_path": "model_selection.training.add_institutional_targets",
            "non_null_count": non_null,
            "nan_count": nan_count,
            "spec": spec.to_dict(),
        })
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "target_columns": columns,
        "target_spec_fingerprints": {name: spec.fingerprint() for name, spec in sorted(target_specs.items())},
    }


def build_pit_transform_specs(df: pd.DataFrame, *, cfg: dict[str, Any]) -> dict[str, PITTransformSpec]:
    cfg_fp = config_fingerprint(cfg)
    specs: dict[str, PITTransformSpec] = {}
    regime_cols = [c for c in ("regime_label", "regime_score", "regime_proba_bull", "regime_proba_bear", "regime_proba_crisis") if df is not None and c in df.columns]
    for col in regime_cols:
        specs[col] = PITTransformSpec(
            transform_name=f"{col}.market_regime_panel_join",
            fit_window_type="external",
            fit_end_policy="source_defined_no_later_than_transform_date",
            transform_date_policy="date_keyed_panel_join",
            threshold_method="source_model_posterior" if "proba" in col or "score" in col else "source_model_label",
            grouping_keys=("date",),
            config_fingerprint=cfg_fp,
        )
    for col in [c for c in ("sector", "sector_asof") if df is not None and c in df.columns]:
        specs[col] = PITTransformSpec(
            transform_name=f"{col}.security_master_classification",
            fit_window_type="external",
            fit_end_policy="security_master_effective_date",
            transform_date_policy="asof_join",
            threshold_method="not_applicable",
            grouping_keys=("ticker", "date"),
            config_fingerprint=cfg_fp,
        )
    bucket_like = [c for c in (df.columns if df is not None else []) if str(c).endswith(("_bucket", "_decile", "_quantile"))]
    for col in bucket_like:
        specs.setdefault(str(col), PITTransformSpec(
            transform_name=f"{col}.panel_bucket",
            fit_window_type="external",
            fit_end_policy="source_defined_no_later_than_transform_date",
            transform_date_policy="same_date_panel_join",
            threshold_method="source_defined",
            grouping_keys=("date",),
            config_fingerprint=cfg_fp,
        ))
    return specs


def build_pit_audit_ledger(df: pd.DataFrame, specs: dict[str, PITTransformSpec]) -> list[dict[str, Any]]:
    if df is None or df.empty or "date" not in df.columns:
        date_min = date_max = ""
    else:
        dates = pd.to_datetime(df["date"], errors="coerce")
        date_min = dates.min().isoformat() if dates.notna().any() else ""
        date_max = dates.max().isoformat() if dates.notna().any() else ""
    rows = []
    for col, spec in sorted(specs.items()):
        rows.append({
            "column_name": col,
            "transform_spec_fingerprint": spec.fingerprint(),
            "as_of_date": "per_row_date",
            "fit_start": spec.fit_start,
            "fit_end": spec.fit_end_policy,
            "transform_start": date_min,
            "transform_end": date_max,
            "rows_transformed": int(df[col].notna().sum()) if df is not None and col in df.columns else 0,
            "diagnostic_only": bool(spec.diagnostic_only),
            "eligible_for_promotion_gates": bool(spec.eligible_for_promotion_gates),
            "spec": spec.to_dict(),
        })
    return rows


def build_feature_manifest(feature_columns: list[str], *, cfg: dict[str, Any], pit_specs: dict[str, PITTransformSpec] | None = None) -> dict[str, Any]:
    from model_selection.research_contract import FEATURE_SPECS

    cfg_fp = config_fingerprint(cfg)
    pit_specs = pit_specs or {}
    features = []
    unknown = []
    for name in sorted(str(f) for f in feature_columns):
        legacy = FEATURE_SPECS.get(name)
        if legacy is None:
            unknown.append(name)
            family = "unknown"
            source = "unregistered"
            horizon = 0
            timestamp = ""
            expected_sign = 0
            decay = "unknown"
        else:
            family = legacy.family
            source = legacy.source
            horizon = int(legacy.horizon_days)
            timestamp = legacy.timestamp
            expected_sign = int(legacy.expected_sign)
            decay = legacy.decay_profile
        pit_fp = pit_specs[name].fingerprint() if name in pit_specs else ""
        spec = InstitutionalFeatureSpec(
            feature_name=name,
            family=family,
            subfamily=decay,
            required_lags=(horizon,) if horizon else (),
            pit_eligible=True,
            pit_transform_fingerprint=pit_fp,
            config_fingerprint=cfg_fp,
            legacy_contract={
                "source": source,
                "timestamp": timestamp,
                "expected_sign": expected_sign,
                "horizon_days": horizon,
            },
        )
        features.append(spec.to_dict())
    return {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "feature_count": len(features),
        "unknown_features": unknown,
        "features": features,
        "feature_fingerprints": {item["feature_name"]: item["fingerprint"] for item in features},
        "registry_fingerprint": stable_fingerprint(features),
    }


def build_cost_assumption_set(costs: Any, *, cfg: dict[str, Any]) -> CostAssumptionSet:
    cost_dict = _jsonable(asdict(costs) if is_dataclass(costs) else vars(costs))
    return CostAssumptionSet(
        spread_model={"spread_bps": cost_dict.get("spread_bps"), "round_trip_policy": "half_spread_one_way"},
        slippage_model={"slippage_bps": (cfg.get("cost_model", {}) or {}).get("slippage_bps")},
        impact_model={
            "impact_eta": cost_dict.get("impact_eta"),
            "impact_alpha": cost_dict.get("impact_alpha"),
            "impact_gamma": cost_dict.get("impact_gamma"),
            "permanent_impact_decay_days": cost_dict.get("permanent_impact_decay_days"),
        },
        commission_model={"commission_bps": cost_dict.get("commission_bps")},
        borrow_model={"borrow_bps": cost_dict.get("borrow_bps"), "annualization_basis": 252},
        participation_cap_policy={"max_participation_rate": cost_dict.get("max_participation_rate")},
        capacity_policy={"capital": cost_dict.get("capital"), "default_adv_usd": cost_dict.get("default_adv_usd")},
        stress_scenarios=(cfg.get("cost_stress_scenarios", {}) or {}),
        units={
            "bps_fields": "basis_points",
            "cost_outputs": "return_fraction",
            "capital": "dollars",
            "borrow": "annualized_bps_prorated_by_horizon",
        },
        config_fingerprint=config_fingerprint(cfg),
    )


def build_promotion_gate_specs(
    gate_cfg: Any,
    *,
    horizon_contract_fingerprint: str,
    target_spec_fingerprint: str = "",
    cost_assumption_fingerprint: str = "",
) -> list[PromotionGateSpec]:
    gate_map = {
        "min_windows": ("n_windows", ">="),
        "min_sharpe": ("oos_sharpe_chained", ">="),
        "min_cost_aware_sharpe": ("exec_sharpe", ">="),
        "min_ic_tstat": ("horizon_adj_ic_tstat|oos_ic_tstat|cs_ic_spearman_tstat", ">="),
        "min_ic_ir": ("horizon_adj_ic_ir|oos_ic_ir|cs_ic_spearman_annualized_icir|cs_ic_spearman_ir", ">="),
        "min_beat_rate": ("oos_beat_rate", ">="),
        "max_drawdown": ("exec_max_dd|oos_max_dd", ">="),
        "min_psr": ("oos_psr", ">="),
        "max_beta_abs_mean": ("exec_beta_abs_mean", "<="),
        "max_sector_abs_mean": ("exec_max_sector_abs_mean", "<="),
        "max_cost_to_gross_pnl": ("exec_cost_to_gross_pnl", "<="),
        "min_decile_spread": ("decile_spread", ">="),
        "min_tail_monotonicity": ("decile_monotonicity", ">="),
        "min_long_leg_sharpe": ("exec_long_leg_sharpe", ">="),
        "min_short_leg_sharpe": ("exec_short_leg_sharpe", ">="),
        "min_subsumption_alpha_ann": ("subsumption_alpha_ann", ">="),
        "min_subsumption_alpha_tstat": ("subsumption_alpha_tstat", ">="),
        "max_subsumption_r2": ("subsumption_r2", "<="),
        "max_subsumption_loading_abs": ("subsumption_max_abs_loading", "<="),
        "robust_halflife": ("diag_robust_signal_halflife", ">="),
        "robust_caic_ratio": ("diag_caic_to_raw_ic_ratio", ">="),
        "robust_turnover": ("diag_robust_turnover_mean", "<="),
    }
    fps = {
        "horizon": horizon_contract_fingerprint,
        "target": target_spec_fingerprint,
        "cost": cost_assumption_fingerprint,
    }
    specs: list[PromotionGateSpec] = []
    robustness_blocking = bool(getattr(gate_cfg, "execution_robustness_affect_selection", False))
    for gate, (metric, op) in gate_map.items():
        is_robust = gate.startswith("robust_")
        threshold_field = {
            "robust_halflife": "min_signal_halflife_buffer",
            "robust_caic_ratio": "min_caic_to_ic_ratio",
            "robust_turnover": "max_avg_turnover",
        }.get(gate, gate)
        severity = "blocking" if (not is_robust or robustness_blocking) else "diagnostic"
        specs.append(PromotionGateSpec(
            gate_name=gate,
            severity=severity,
            metric_name=metric,
            required_input_contract_fingerprints=fps,
            pass_fail_rule=f"{metric} {op} PromotionGateConfig.{threshold_field}",
            reason_code=gate.upper(),
        ))
    return specs


def promotion_decision_from_row(
    row: dict[str, Any],
    *,
    horizon_contract_fingerprint: str,
    target_spec_fingerprint: str = "",
    cost_assumption_fingerprint: str = "",
    run_id: str = "",
) -> PromotionDecision:
    failures = tuple(
        f.strip() for f in str(row.get("promotion_failures", "") or "").split(",") if f.strip()
    )
    diagnostics = tuple(f for f in failures if f.startswith("diagnostic:"))
    blocking = tuple(f for f in failures if not f.startswith("diagnostic:"))
    gate_cols = {k: bool(v) for k, v in row.items() if str(k).startswith("gate_") and not str(k).startswith("gate_threshold_")}
    passed = tuple(sorted(k.replace("gate_", "", 1) for k, ok in gate_cols.items() if ok))
    status = "passed" if bool(row.get("promotion_pass", False)) else "failed"
    return PromotionDecision(
        overall_status=status,
        passed_gates=passed,
        failed_blocking_gates=blocking,
        diagnostics=diagnostics,
        deterministic_reason_codes=tuple(f.upper().replace("DIAGNOSTIC:", "DIAGNOSTIC_") for f in failures),
        metric_values_used={k: _jsonable(v) for k, v in row.items() if not str(k).startswith("gate_threshold_")},
        threshold_values_used={k: _jsonable(v) for k, v in row.items() if str(k).startswith("gate_threshold_")},
        horizon_contract_fingerprint=horizon_contract_fingerprint,
        target_spec_fingerprint=target_spec_fingerprint,
        cost_assumption_fingerprint=cost_assumption_fingerprint,
        cache_key_or_run_id=str(run_id),
    )


def build_cache_key_spec(
    *,
    cache_name: str,
    cfg: dict[str, Any],
    tickers: list[str] | None = None,
    start_date: str = "",
    end_date: str = "",
    horizon_contract_fingerprint: str = "",
    target_spec_fingerprints: list[str] | tuple[str, ...] = (),
    feature_spec_fingerprints: list[str] | tuple[str, ...] = (),
    cost_assumption_fingerprint: str = "",
    pit_transform_fingerprints: list[str] | tuple[str, ...] = (),
    data_source_fingerprint: str = "",
    model_spec_fingerprint: str = "",
    split_fingerprint: str = "",
    score_vector_fingerprint: str = "",
    random_seed: str = "",
) -> CacheKeySpec:
    return CacheKeySpec(
        cache_name=str(cache_name),
        data_source_fingerprint=str(data_source_fingerprint),
        universe_ticker_fingerprint=stable_fingerprint(sorted(str(t).upper() for t in (tickers or []))),
        date_range={"start": str(start_date), "end": str(end_date)},
        horizon_contract_fingerprint=str(horizon_contract_fingerprint),
        target_spec_fingerprints=tuple(sorted(str(x) for x in target_spec_fingerprints)),
        feature_spec_fingerprints=tuple(sorted(str(x) for x in feature_spec_fingerprints)),
        cost_assumption_fingerprint=str(cost_assumption_fingerprint),
        pit_transform_fingerprints=tuple(sorted(str(x) for x in pit_transform_fingerprints)),
        model_spec_fingerprint=str(model_spec_fingerprint),
        split_fingerprint=str(split_fingerprint),
        score_vector_fingerprint=str(score_vector_fingerprint),
        config_fingerprint=config_fingerprint(cfg),
        random_seed=str(random_seed),
        library_versions={
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
    )


def write_json_artifact(path: str | Path, payload: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    return out


def append_jsonl(path: str | Path, event: dict[str, Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(event), sort_keys=True) + "\n")
    return out


def emit_telemetry_event(
    out_dir: str | Path,
    *,
    run_id: str,
    stage: str,
    event_type: str,
    message: str,
    severity: str = "info",
    ledger: str = "run_telemetry.jsonl",
    module_function: str = "",
    recoverable: bool = True,
    contract_fingerprints: dict[str, str] | None = None,
    artifact_path: str | Path | None = None,
    exception: BaseException | None = None,
) -> Path:
    event = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "run_id": str(run_id),
        "stage": str(stage),
        "module_function": str(module_function),
        "severity": str(severity),
        "event_type": str(event_type),
        "message": str(message),
        "exception_type": type(exception).__name__ if exception is not None else "",
        "exception": str(exception) if exception is not None else "",
        "contract_fingerprints": contract_fingerprints or {},
        "artifact_path": str(artifact_path) if artifact_path is not None else "",
        "recoverable": bool(recoverable),
    }
    return append_jsonl(Path(out_dir) / ledger, event)


def write_institutional_run_manifest(
    out_dir: str | Path,
    *,
    run_id: str,
    config_path: str,
    cfg: dict[str, Any],
    horizon_contract: HorizonRunContract,
    target_manifest: dict[str, Any],
    pit_specs: dict[str, PITTransformSpec],
    feature_manifest: dict[str, Any],
    cost_assumption_set: CostAssumptionSet,
    promotion_gate_specs: list[PromotionGateSpec],
    cache_key_specs: list[CacheKeySpec],
    artifacts: dict[str, str],
    telemetry_contract: RunTelemetryContract,
    warnings: list[str] | tuple[str, ...] = (),
    errors: list[str] | tuple[str, ...] = (),
) -> dict[str, Any]:
    git_version = ""
    try:
        git_version = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        git_version = "unavailable"
    manifest = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "run_id": str(run_id),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "git_code_version": git_version,
        "config_path": str(config_path),
        "config_fingerprint": config_fingerprint(cfg),
        "horizon_run_contract_fingerprint": horizon_contract.fingerprint(),
        "target_spec_fingerprints": target_manifest.get("target_spec_fingerprints", {}),
        "pit_transform_fingerprints": {k: v.fingerprint() for k, v in sorted(pit_specs.items())},
        "feature_spec_fingerprints": feature_manifest.get("feature_fingerprints", {}),
        "cost_assumption_set_fingerprint": cost_assumption_set.fingerprint(),
        "promotion_gate_spec_fingerprint": stable_fingerprint([s.to_dict() for s in promotion_gate_specs]),
        "cache_key_specs": {spec.cache_name: spec.fingerprint() for spec in cache_key_specs},
        "output_artifacts": artifacts,
        "telemetry_artifacts": telemetry_contract.to_dict(),
        "warnings_summary": list(warnings),
        "errors_summary": list(errors),
        "production_eligibility_summary": {
            "horizon_contract_status": horizon_contract.validation_status,
            "unknown_feature_count": len(feature_manifest.get("unknown_features", [])),
            "pit_transform_count": len(pit_specs),
        },
    }
    write_json_artifact(Path(out_dir) / "institutional_run_manifest.json", manifest)
    return manifest
