from __future__ import annotations

import logging
import os
from typing import Any

from model_selection.alpha_research import AlphaAdmissionConfig
from model_selection.execution_aware_horizon_policy import (
    ExecutionAwareHorizonPolicy,
    parse_execution_aware_horizon_policy,
)
from model_selection.horizon_contract import HorizonContract, build_horizon_contract
from model_selection.orientation_policy import OrientationPolicy, parse_orientation_policy
from model_selection.training import TargetConfig
from model_selection.validation import EvaluationConfig, ExecutionCostConfig, LongAlphaCandidateConfig, PromotionGateConfig

logger = logging.getLogger(__name__)


def _warn_deprecated_if_present(
    cfg: dict[str, Any],
    *,
    canonical_section: str,
    canonical_key: str,
    deprecated_paths: tuple[tuple[str, ...], ...],
) -> list[tuple[str, str, str]]:
    """Warn when a deprecated duplicate config key is present.

    Resolution remains canonical-first; this warning is a governance guardrail
    so duplicate fields cannot silently become alternative sources of truth.
    """
    canonical = cfg.get(canonical_section, {}) if isinstance(cfg, dict) else {}
    has_canonical = isinstance(canonical, dict) and canonical_key in canonical
    findings: list[tuple[str, str, str]] = []
    for path in deprecated_paths:
        cur: Any = cfg
        for part in path:
            if not isinstance(cur, dict) or part not in cur:
                cur = None
                break
            cur = cur[part]
        if cur is None:
            continue
        dotted = ".".join(path)
        target = f"{canonical_section}.{canonical_key}"
        if has_canonical:
            findings.append(("ignored", dotted, target))
        else:
            findings.append(("fallback", dotted, target))
    return findings


def warn_deprecated_config_duplicates(cfg: dict[str, Any]) -> None:
    """Emit warnings for duplicate optimizer/risk/cost/liquidity parameters."""
    findings: list[tuple[str, str, str]] = []
    findings.extend(_warn_deprecated_if_present(
        cfg,
        canonical_section="exposure_limits",
        canonical_key="lambda_risk",
        deprecated_paths=(("backtest", "optimization_config", "optimizer", "lambda_risk"), ("model_selection", "validation", "lambda_risk")),
    ))
    findings.extend(_warn_deprecated_if_present(
        cfg,
        canonical_section="exposure_limits",
        canonical_key="gamma_turnover",
        deprecated_paths=(("backtest", "optimization_config", "optimizer", "gamma_turnover"), ("model_selection", "validation", "gamma_turnover")),
    ))
    findings.extend(_warn_deprecated_if_present(
        cfg,
        canonical_section="portfolio_constraints",
        canonical_key="max_gross_exposure",
        deprecated_paths=(("backtest", "optimization_config", "optimizer", "gross_cap"), ("model_selection", "validation", "max_gross")),
    ))
    findings.extend(_warn_deprecated_if_present(
        cfg,
        canonical_section="portfolio_constraints",
        canonical_key="net_exposure_max",
        deprecated_paths=(("backtest", "optimization_config", "optimizer", "net_exposure_max"), ("model_selection", "validation", "net_exposure_max")),
    ))
    findings.extend(_warn_deprecated_if_present(
        cfg,
        canonical_section="portfolio_constraints",
        canonical_key="min_position_weight",
        deprecated_paths=(("backtest", "optimization_config", "optimizer", "min_position_weight"), ("model_selection", "validation", "min_position_weight")),
    ))
    findings.extend(_warn_deprecated_if_present(
        cfg,
        canonical_section="liquidity_limits",
        canonical_key="max_adv_fraction",
        deprecated_paths=(("backtest", "optimization_config", "liquidity", "max_adv_fraction"), ("model_selection", "validation", "max_adv_fraction"), ("risk_factors", "liquidity", "max_adv_pct")),
    ))
    for key in ("commission_bps", "spread_bps", "slippage_bps"):
        findings.extend(_warn_deprecated_if_present(
            cfg,
            canonical_section="cost_model",
            canonical_key=key,
            deprecated_paths=(("execution_costs", key),),
        ))

    ignored = [(dotted, target) for action, dotted, target in findings if action == "ignored"]
    fallback = [(dotted, target) for action, dotted, target in findings if action == "fallback"]
    if ignored:
        examples = ", ".join(f"{d}->{t}" for d, t in ignored[:6])
        suffix = f"; +{len(ignored) - 6} more" if len(ignored) > 6 else ""
        logger.warning(
            "Deprecated duplicate config field summary: %d ignored in favor of canonical sections (%s%s)",
            len(ignored),
            examples,
            suffix,
        )
    if fallback:
        examples = ", ".join(f"{d}->{t}" for d, t in fallback[:6])
        suffix = f"; +{len(fallback) - 6} more" if len(fallback) > 6 else ""
        logger.warning(
            "Deprecated config fallback summary: %d legacy field(s) used because canonical values are missing (%s%s)",
            len(fallback),
            examples,
            suffix,
        )


def _parse_path_overrides(raw: Any) -> dict[str, dict[str, float]]:
    """Parse the path_overrides YAML block into {model_kind: {field: float}}.

    Accepts:
      path_overrides:
        short_side:
          min_ic_tstat: 1.0
          min_ic_ir: 0.30
        short_classifier:
          min_beat_rate: 0.50
    """
    if not isinstance(raw, dict):
        return {}
    result: dict[str, dict[str, float]] = {}
    for path_key, threshold_map in raw.items():
        if not isinstance(threshold_map, dict):
            continue
        parsed: dict[str, float] = {}
        for field_name, value in threshold_map.items():
            try:
                parsed[str(field_name)] = float(value)
            except (TypeError, ValueError):
                pass
        if parsed:
            result[str(path_key)] = parsed
    return result


def feature_builder_data_kwargs(cfg: dict[str, Any]) -> dict[str, Any]:
    """Translate YAML data/universe config into explicit feature-builder inputs."""
    data_cfg = cfg.get("data", {}) or {}
    provider = str(data_cfg.get("provider", "") or "").strip().lower() or None
    raw_cache_dir = str(data_cfg.get("cache_dir", "") or "").strip()
    cache_dir = raw_cache_dir or None
    if provider == "wrds":
        from pathlib import Path

        cache_root = Path(raw_cache_dir or "data/cache")
        cache_dir = str(cache_root if cache_root.name == "wrds" else cache_root / "wrds")

    raw_cache_ttl = data_cfg.get("cache_ttl_days", 1)
    kwargs: dict[str, Any] = {
        "data_provider": provider,
        "cache_dir": cache_dir,
        "cache_ttl_days": 1 if raw_cache_ttl is None else int(raw_cache_ttl),
    }
    if provider == "wrds":
        kwargs["wrds_username"] = os.environ.get("WRDS_USERNAME")
        kwargs["wrds_ticker_to_permno"] = cfg.get("wrds_ticker_to_permno", {}) or {}
        kwargs["strict_fundamentals"] = bool(data_cfg.get("strict_fundamentals", False))
    return kwargs


def execution_cost_config(cfg: dict[str, Any]) -> ExecutionCostConfig:
    ec = cfg.get("execution_costs", {}) or {}
    cm = cfg.get("cost_model", {}) or {}
    mi = cfg.get("market_impact", {}) or {}
    ms = cfg.get("model_selection", {}) or {}
    val = (ms.get("validation", {}) or {}) if isinstance(ms, dict) else {}
    return ExecutionCostConfig(
        capital=float(val.get("capital", cfg.get("initial_capital", 10_000_000.0)) or 10_000_000.0),
        commission_bps=float(cm.get("commission_bps", ec.get("commission_bps", 1.0)) or 1.0),
        spread_bps=float(cm.get("spread_bps", ec.get("spread_bps", 1.0)) or 1.0),
        borrow_bps=float(cm.get("borrow_bps", ec.get("short_borrow_bps", ec.get("borrow_bps", 50.0))) or 50.0),
        impact_eta=float(mi.get("eta", 0.142) or 0.142),
        impact_alpha=float(mi.get("alpha", 0.314) or 0.314),
        impact_gamma=float(mi.get("gamma", 0.6) or 0.6),
        default_adv_usd=float(mi.get("adv_usd_default", 50_000_000.0) or 50_000_000.0),
        default_daily_vol=float(mi.get("daily_vol_default", 0.02) or 0.02),
        max_participation_rate=float(mi.get("max_participation_rate", val.get("max_participation_rate", 0.10)) or 0.10),
        permanent_impact_decay_days=int(mi.get("permanent_impact_decay_days", 5) or 5),
    )


def horizon_contract_config(cfg: dict[str, Any], *, cli_horizon: int | None = None) -> HorizonContract:
    """Return the canonical run-level horizon contract."""

    return build_horizon_contract(cfg, cli_horizon=cli_horizon)


def evaluation_config(
    cfg: dict[str, Any],
    *,
    path: str,
    max_positions: int,
    min_positions: int,
    horizon: int | None = None,
    horizon_contract: HorizonContract | None = None,
    signal_halflife_days: float | None = None,  # P8: decay-aware execution
) -> EvaluationConfig:
    contract = horizon_contract or build_horizon_contract(cfg, cli_horizon=horizon)
    horizon = int(contract.config.target_horizon_days)
    bt = cfg.get("backtest", {}) or {}
    opt = (bt.get("optimization_config", {}) or {}) if isinstance(bt, dict) else {}
    factor = (opt.get("factor_model", {}) or {}) if isinstance(opt, dict) else {}
    liq = (opt.get("liquidity", {}) or {}) if isinstance(opt, dict) else {}
    op = (opt.get("optimizer", {}) or {}) if isinstance(opt, dict) else {}
    ms = cfg.get("model_selection", {}) or {}
    val = (ms.get("validation", {}) or {}) if isinstance(ms, dict) else {}
    factor_max = factor.get("max_exposures", {}) or {}

    # Canonical sections — single source of truth; override deprecated sub-sections.
    pc = cfg.get("portfolio_constraints") or {}
    ll = cfg.get("liquidity_limits") or {}
    el = cfg.get("exposure_limits") or {}
    apc = (el.get("adaptive_portfolio_control", {}) or {}) if isinstance(el, dict) else {}

    # Resolution order: canonical → ms.validation → deprecated fallbacks
    _max_name_weight = float(
        pc.get("max_name_weight",
               val.get("max_name_weight",
                       (opt.get("optimizer") or {}).get("max_weight", 0.10))) or 0.10
    )
    _max_gross = float(
        pc.get("max_gross_exposure",
               val.get("max_gross",
                       (opt.get("optimizer") or {}).get("gross_cap", 1.0))) or 1.0
    )
    _net_exposure_max = float(
        pc.get("net_exposure_max",
               val.get("net_exposure_max",
                       op.get("net_exposure_max", 0.10))) or 0.10
    )
    _min_position_weight = float(
        pc.get("min_position_weight",
               val.get("min_position_weight",
                       op.get("min_position_weight", 0.0))) or 0.0
    )
    _adv_fraction = float(
        ll.get("max_adv_fraction",
               val.get("max_adv_fraction",
                       liq.get("max_adv_fraction", 0.05))) or 0.05
    )
    _lambda_risk = float(
        el.get("lambda_risk",
               val.get("lambda_risk",
                       op.get("lambda_risk", 2.0))) or 2.0
    )
    _gamma_turnover = float(
        el.get("gamma_turnover",
               val.get("gamma_turnover",
                       op.get("gamma_turnover", 4.0))) or 4.0
    )
    _covariance_window = int(
        el.get("covariance_window",
               val.get("optimizer_lookback_days", 60)) or 60
    )
    _max_beta_abs = float(
        el.get("max_market_beta",
               val.get("max_beta_abs",
                       factor_max.get("market_beta", 0.15))) or 0.15
    )
    _max_sector_abs = float(
        el.get("max_sector_net_exposure",
               val.get("max_sector_abs",
                       factor.get("sector_net_exposure_max", 0.12))) or 0.12
    )
    _optimization_type = str(val.get("optimization_type", "l1")).lower()

    return EvaluationConfig(
        max_positions=int(max_positions),
        min_positions=int(min_positions),
        horizon_days=int(horizon),
        path=str(path),
        optimization_type=_optimization_type,
        rebalance_every_days=int(contract.config.rebalance_frequency_days),
        factor_neutral=bool(val.get("factor_neutral", factor.get("enabled", True))),
        beta_neutral=bool(val.get("beta_neutral", True)),
        sector_neutral=bool(val.get("sector_neutral", True)),
        max_beta_abs=_max_beta_abs,
        max_sector_abs=_max_sector_abs,
        adv_fraction=_adv_fraction,
        max_gross=_max_gross,
        max_name_weight=_max_name_weight,
        constraint_passes=int(val.get("constraint_passes", 3) or 3),
        use_optimizer=bool(val.get("use_optimizer", True)),
        lambda_risk=_lambda_risk,
        gamma_turnover=_gamma_turnover,
        net_exposure_max=_net_exposure_max,
        min_position_weight=_min_position_weight,
        no_trade_band_weight_diff=float(val.get("no_trade_band_weight_diff", 0.015) or 0.015),
        no_trade_band_total_drift=float(val.get("no_trade_band_total_drift", 0.05) or 0.05),
        optimizer_lookback_days=_covariance_window,
        optimizer_alpha_scale=float(val.get("optimizer_alpha_scale", 1.0) or 1.0),
        short_squeeze_filter=bool(val.get("short_squeeze_filter", True)),
        short_squeeze_max_risk=float(val.get("short_squeeze_max_risk", 0.75) or 0.75),
        market_neutral_shorts=bool(val.get("market_neutral_shorts", True)),
        costs=execution_cost_config(cfg),
        adaptive_control_enabled=bool(apc.get("enabled", False)),
        adaptive_control_lookback_days=int(apc.get("lookback_days", 252) or 252),
        adaptive_control_min_history_days=int(apc.get("min_history_days", 60) or 60),
        adaptive_control_ema_span=int(apc.get("ema_span", 20) or 20),
        adaptive_control_target_volatility=float(apc.get("target_volatility", 0.15) or 0.15),
        adaptive_lambda_floor_factor=float(apc.get("lambda_floor_factor", 0.50) or 0.50),
        adaptive_lambda_ceil_factor=float(apc.get("lambda_ceil_factor", 4.00) or 4.00),
        adaptive_gamma_floor_factor=float(apc.get("gamma_floor_factor", 0.125) or 0.125),
        adaptive_gamma_ceil_factor=float(apc.get("gamma_ceil_factor", 4.00) or 4.00),
        adaptive_min_expected_alpha=float(apc.get("min_expected_alpha", 1e-4) or 1e-4),
        signal_halflife_days=float(signal_halflife_days) if signal_halflife_days is not None else float("nan"),
    )


def promotion_gate_config(cfg: dict[str, Any]) -> PromotionGateConfig:
    ms = cfg.get("model_selection", {}) or {}
    gates = ((ms.get("promotion", {}) or {}).get("gates", {}) or {}) if isinstance(ms, dict) else {}
    robust = (gates.get("execution_robustness") or {}) if isinstance(gates, dict) else {}
    return PromotionGateConfig(
        min_sharpe=float(gates.get("min_sharpe", 0.50) or 0.50),
        min_ic_tstat=float(gates.get("min_ic_tstat", 2.0) or 2.0),
        min_ic_ir=float(gates.get("min_ic_ir", 0.75) or 0.75),
        min_beat_rate=float(gates.get("min_beat_rate", 0.625) or 0.625),
        max_drawdown=float(gates.get("max_drawdown", -0.25) or -0.25),
        min_cost_aware_sharpe=float(gates.get("min_cost_aware_sharpe", 0.25) or 0.25),
        min_windows=int(gates.get("min_windows", 6) or 6),
        min_psr=float(gates.get("min_psr", 0.60) or 0.60),
        max_beta_abs_mean=float(gates.get("max_beta_abs_mean", 0.15) or 0.15),
        max_sector_abs_mean=float(gates.get("max_sector_abs_mean", 0.12) or 0.12),
        max_cost_to_gross_pnl=float(gates.get("max_cost_to_gross_pnl", 0.50) or 0.50),
        min_decile_spread=float(gates.get("min_decile_spread", 0.0) or 0.0),
        min_tail_monotonicity=float(gates.get("min_tail_monotonicity", 0.50) or 0.50),
        min_long_leg_sharpe=float(gates.get("min_long_leg_sharpe", 0.0) or 0.0),
        min_short_leg_sharpe=float(gates.get("min_short_leg_sharpe", 0.0) or 0.0),
        min_subsumption_alpha_ann=float(gates.get("min_subsumption_alpha_ann", 0.0) or 0.0),
        min_subsumption_alpha_tstat=float(gates.get("min_subsumption_alpha_tstat", 1.0) or 1.0),
        max_subsumption_r2=float(gates.get("max_subsumption_r2", 0.80) or 0.80),
        max_subsumption_loading_abs=float(gates.get("max_subsumption_loading_abs", 1.50) or 1.50),
        
        # Execution Robustness Gates
        execution_robustness_enabled=bool(robust.get("enabled", gates.get("execution_robustness_enabled", True))),
        execution_robustness_affect_selection=bool(robust.get("affect_selection", gates.get("execution_robustness_affect_selection", True))),
        execution_robustness_fail_on_missing=bool(robust.get("fail_on_missing_metrics", gates.get("execution_robustness_fail_on_missing", True))),
        min_signal_halflife_buffer=float(robust.get("min_signal_halflife_buffer", gates.get("min_signal_halflife_buffer", 1.0))),
        min_caic_to_ic_ratio=float(robust.get("min_caic_to_ic_ratio", gates.get("min_caic_to_ic_ratio", 0.30))),
        max_avg_turnover=float(robust.get("max_avg_turnover", gates.get("max_avg_turnover", 0.80))),
        dynamic_thresholds_enabled=bool((gates.get("dynamic_thresholds") or {}).get("enabled", True)),
        dynamic_threshold_confidence=float((gates.get("dynamic_thresholds") or {}).get("confidence", 0.95)),
        dynamic_threshold_min_effective_obs=int((gates.get("dynamic_thresholds") or {}).get("min_effective_obs", 12)),
        dynamic_threshold_reference_ic_std=float((gates.get("dynamic_thresholds") or {}).get("reference_ic_std", 0.05)),
        dynamic_threshold_reference_turnover=float((gates.get("dynamic_thresholds") or {}).get("reference_turnover", 0.35)),
        
        path_overrides=_parse_path_overrides(gates.get("path_overrides") or {}),
    )


def long_alpha_candidate_config(cfg: dict[str, Any]) -> LongAlphaCandidateConfig:
    """Parse long_alpha_candidate tier gates from YAML.

    Located at: model_selection.promotion.long_alpha_candidate_gates
    All fields are optional and fall back to LongAlphaCandidateConfig defaults.
    """
    ms = cfg.get("model_selection", {}) or {}
    promo = (ms.get("promotion", {}) or {}) if isinstance(ms, dict) else {}
    g = (promo.get("long_alpha_candidate_gates", {}) or {}) if isinstance(promo, dict) else {}
    return LongAlphaCandidateConfig(
        min_long_leg_sharpe=float(g.get("min_long_leg_sharpe", 0.0) or 0.0),
        min_ic_tstat=float(g.get("min_ic_tstat", 1.0) or 1.0),
        min_beat_rate=float(g.get("min_beat_rate", 0.50) or 0.50),
        max_drawdown=float(g.get("max_drawdown", -0.35) or -0.35),
        min_psr=float(g.get("min_psr", 0.50) or 0.50),
        max_cost_to_gross_pnl=float(g.get("max_cost_to_gross_pnl", 0.75) or 0.75),
        min_decile_spread=float(g.get("min_decile_spread", 0.0) or 0.0),
        min_windows=int(g.get("min_windows", 4) or 4),
    )


def target_config(
    cfg: dict[str, Any],
    *,
    horizon: int | None = None,
    horizon_contract: HorizonContract | None = None,
) -> TargetConfig:
    contract = horizon_contract or build_horizon_contract(cfg, cli_horizon=horizon)
    horizon = int(contract.config.target_horizon_days)
    ms = cfg.get("model_selection", {}) or {}
    target = (ms.get("target", {}) or {}) if isinstance(ms, dict) else {}
    strict_alpha = bool(ms.get("strict_alpha_separation", True)) if isinstance(ms, dict) else True
    return TargetConfig(
        horizon_days=int(horizon),
        residualize=bool(target.get("residualize", True)),
        net_of_costs=False if strict_alpha else bool(target.get("net_of_costs", False)),
        residual_ridge=float(target.get("residual_ridge", 1e-4) or 1e-4),
        winsor_q=float(target.get("winsor_q", 0.01) or 0.0),
        max_abs_return=float(target.get("max_abs_return", 5.0) or 5.0),
    )


def alpha_admission_config(
    cfg: dict[str, Any],
    *,
    horizon: int | None = None,
    horizon_contract: HorizonContract | None = None,
) -> AlphaAdmissionConfig:
    contract = horizon_contract or build_horizon_contract(cfg, cli_horizon=horizon)
    horizon = int(contract.config.target_horizon_days)
    ms = cfg.get("model_selection", {}) or {}
    raw = (ms.get("alpha_research", {}) or {}) if isinstance(ms, dict) else {}
    horizon_alignment = horizon_alignment_config(cfg)
    horizons_raw = raw.get("horizons", [1, 2, 3, 5, 10, 20])
    horizons = tuple(int(h) for h in horizons_raw if str(h).strip()) or (1, 2, 3, 5, 10, 20)
    base_target = target_config(cfg, horizon_contract=contract)
    stability = (raw.get("stability", {}) or {}) if isinstance(raw, dict) else {}
    # P15: Minimum valid-day thresholds scale with horizon to ensure at least
    # 4 non-overlapping IC observations regardless of the configured horizon.
    # Default: max(20, 4 * horizon_days).  At 10d: 40 days.  At 63d: 252 days.
    _h_min_obs = max(20, int(horizon) * 4)
    return AlphaAdmissionConfig(
        enabled=bool(raw.get("enabled", True)),
        enforce=bool(raw.get("enforce_admission", True)),
        horizons=tuple(sorted({max(1, int(h)) for h in horizons})),
        production_horizon=int(contract.config.production_horizon_days),
        min_coverage=float(raw.get("min_coverage", 0.80) or 0.80),
        min_abs_ic=float(raw.get("min_abs_ic", 0.001) or 0.001),
        min_ic_tstat=float(raw.get("min_ic_tstat", 0.50) or 0.50),
        min_monotonicity=float(raw.get("min_monotonicity", 0.50) or 0.50),
        min_regime_stability=float(raw.get("min_regime_stability", 0.50) or 0.50),
        min_ic_valid_days=int(raw.get("min_ic_valid_days", _h_min_obs) or _h_min_obs),
        min_regime_valid_days=int(raw.get("min_regime_valid_days", _h_min_obs) or _h_min_obs),
        min_spread_valid_days=int(raw.get("min_spread_valid_days", _h_min_obs) or _h_min_obs),
        min_monotonicity_valid_days=int(raw.get("min_monotonicity_valid_days", _h_min_obs) or _h_min_obs),
        min_marginal_abs_ic=float(raw.get("min_marginal_abs_ic", 0.00025) or 0.00025),
        min_marginal_residual_variance_ratio=float(raw.get("min_marginal_residual_variance_ratio", 0.05) or 0.05),
        allow_inversion=bool(raw.get("allow_inversion", True)),
        minimum_admitted_features=int(raw.get("minimum_admitted_features", 0) or 0),
        fail_if_below_minimum=bool(raw.get("fail_if_below_minimum", False)),
        enforce_horizon_alignment=bool(raw.get("enforce_horizon_alignment", True)),
        cross_horizon_admission=bool(raw.get("cross_horizon_admission", True)),
        horizon_alignment_multiplier=float(
            raw.get("horizon_alignment_multiplier", horizon_alignment["multiplier"]) or horizon_alignment["multiplier"]
        ),
        residual_ridge=float(raw.get("residual_ridge", base_target.residual_ridge) or 1e-4),
        winsor_q=float(raw.get("winsor_q", base_target.winsor_q) or 0.0),
        max_abs_return=float(raw.get("max_abs_return", base_target.max_abs_return) or 5.0),
        bhy_alpha=float(raw.get("bhy_alpha", 0.05) or 0.05),
        apply_bhy_correction=bool(raw.get("apply_bhy_correction", True)),
        multi_horizon_admission=bool(raw.get("multi_horizon_admission", False)),
        fallback_mode=str(raw.get("fallback_mode", "percentile") or "percentile"),
        min_admission_percentile=float(raw.get("min_admission_percentile", 0.60) or 0.60),
        max_features_by_rank=int(raw.get("max_features_by_rank", 12) or 12),
        min_admitted_for_fallback=int(raw.get("min_admitted_for_fallback", 3) or 3),
        ic_cv_max=float(stability.get("ic_cv_max", 2.0) or 2.0),
        ic_sign_flip_max=int(stability.get("ic_sign_flip_max", 2) or 2),
        min_halflife_days=float(stability.get("min_halflife_days", 1.0) or 1.0),
    )


def embargo_days_config(cfg: dict[str, Any], *, default_horizon: int = 5) -> int:
    """Return the configured embargo (calendar days) between train-end and test-start.

    Resolution order (first match wins):
      1. model_selection.embargo_days  — canonical new field
      2. research.embargo_days         — legacy field
      3. dynamic_embargo               — computed from FEATURE_SPECS max lookback (new)
      4. 2 × default_horizon           — old inline default (kept for backward compat)

    The dynamic embargo computes the maximum lookback across all features and applies
    a safety multiplier to prevent information leakage from long-horizon momentum,
    fundamental, and reversal features.
    """
    ms = cfg.get("model_selection", {}) or {}
    if "embargo_days" in ms:
        return max(1, int(ms["embargo_days"]))
    research = cfg.get("research", {}) or {}
    if "embargo_days" in research:
        return max(1, int(research["embargo_days"]))

    # Dynamic embargo: derived from FEATURE_SPECS max horizon.
    use_dynamic = ms.get("dynamic_embargo", research.get("dynamic_embargo", True))
    if use_dynamic:
        from model_selection.research_contract import FEATURE_SPECS
        max_horizon = max((spec.horizon_days for spec in FEATURE_SPECS.values()), default=5)
        # P15: Embargo floor scales with holding period — at least 2× the
        # holding period to prevent any forward-return leakage through the
        # embargo window.  The 1.5× safety multiplier on the max feature
        # horizon covers feature computation lookback windows.
        _horizon_floor = max(5, 2 * int(default_horizon))
        dynamic = max(_horizon_floor, int(max_horizon * 1.5))
        return dynamic

    return max(5, 2 * int(default_horizon))


def horizon_alignment_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Parse feature-horizon alignment config from model_selection.horizon_alignment.

    Returns:
      enabled   — bool: run the report at all
      enforce   — bool: block misaligned features (false = report only)
      multiplier — float: feature.horizon_days <= target * multiplier → aligned
      report_path — str: CSV path for the alignment ledger
      verbose — bool: print per-feature rows during pre-admission audit
    """
    ms = cfg.get("model_selection", {}) or {}
    raw = (ms.get("horizon_alignment", {}) or {}) if isinstance(ms, dict) else {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "enforce": bool(raw.get("enforce", False)),
        "multiplier": float(raw.get("alignment_multiplier", 2.0) or 2.0),
        "report_path": str(raw.get("report_path", "output/horizon_alignment_report.csv") or ""),
        "verbose": bool(raw.get("verbose", False)),
    }


def preprocess_winsor_q(cfg: dict[str, Any]) -> float:
    ms = cfg.get("model_selection", {}) or {}
    prep = (ms.get("preprocessing", {}) or {}) if isinstance(ms, dict) else {}
    return float(prep.get("winsor_q", 0.01) or 0.0)


_VALID_SIM_MODES = frozenset({"executable", "proxy_only"})


def _parse_simulation_policy(nested: dict[str, Any]) -> dict[str, str]:
    """Parse ``simulation_policy`` from nested validation config.

    Defaults:
    - ``nested_full``: ``"executable"`` (backward-compatible; set to ``"proxy_only"``
      to use rank-based weights instead of QP optimizer for inner candidate ranking).
    - ``final_validation``: always ``"executable"`` — not configurable, forced here
      as a safety invariant so proxy metrics can never leak into promotion gates.
    """
    raw = (nested.get("simulation_policy", {}) or {}) if isinstance(nested, dict) else {}
    nested_full = str(raw.get("nested_full", "executable")).strip().lower()
    if nested_full not in _VALID_SIM_MODES:
        nested_full = "executable"
    return {
        "nested_full": nested_full,
        "final_validation": "executable",  # invariant — never proxy
    }


def nested_validation_config(
    cfg: dict[str, Any],
    *,
    horizon_contract: HorizonContract | None = None,
) -> dict[str, Any]:
    ms = cfg.get("model_selection", {}) or {}
    nested = (ms.get("nested_validation", {}) or {}) if isinstance(ms, dict) else {}
    search = (nested.get("search", {}) or {}) if isinstance(nested, dict) else {}
    raw_horizons = search.get("candidate_horizons", nested.get("candidate_horizons", []))
    max_horizons = int(search.get("max_horizons", nested.get("max_horizons", 3)) or 3)
    if horizon_contract is not None:
        raw_horizons = [int(horizon_contract.config.target_horizon_days)]
        max_horizons = 1
    return {
        "enabled": bool(nested.get("enabled", True)),
        "true_selection_enabled": bool(nested.get("true_selection_enabled", True)),
        "max_windows": int(nested.get("max_windows", 1) or 1),
        "validation_days": int(nested.get("validation_days", 126) or 126),
        "min_train_days": int(nested.get("min_train_days", 504) or 504),
        "min_sharpe": float(nested.get("min_sharpe", 0.0) or 0.0),
        "min_ic": float(nested.get("min_ic", 0.0) or 0.0),
        "search": {
            "candidate_horizons": raw_horizons,
            "max_horizons": max_horizons,
            "feature_views": search.get("feature_views", nested.get("feature_views", ["full", "program"])),
            "max_candidates": int(search.get("max_candidates", nested.get("max_candidates", 24)) or 24),
            "prefilter_top_k": int(search.get("prefilter_top_k", nested.get("prefilter_top_k", 6)) or 6),
            "prefilter_windows": int(search.get("prefilter_windows", nested.get("prefilter_windows", 1)) or 1),
            "proxy_max_iter": int(search.get("proxy_max_iter", nested.get("proxy_max_iter", 30)) or 30),
            "ic_qp_floor": float(search.get("ic_qp_floor", 0.01)),
            "allow_cross_family_selection": bool(search.get("allow_cross_family_selection", False)),
        },
        "simulation_policy": _parse_simulation_policy(nested),
    }


def parallel_research_config(cfg: dict[str, Any], *, n_models: int = 1) -> dict[str, int | bool]:
    ms = cfg.get("model_selection", {}) or {}
    raw = (ms.get("parallel_research", {}) or {}) if isinstance(ms, dict) else {}
    cpu = max(1, int(os.cpu_count() or 1))
    default_model_workers = min(max(1, cpu // 2), max(1, int(n_models)), 4)
    model_workers = int(raw.get("model_workers", default_model_workers) or default_model_workers)
    model_workers = max(1, min(model_workers, max(1, int(n_models))))
    nested_default = 1 if model_workers > 1 else min(max(1, cpu // 2), 4)
    nested_workers = int(raw.get("nested_candidate_workers", nested_default) or nested_default)
    nested_workers = max(1, nested_workers)
    economic_model_workers = int(raw.get("economic_model_workers", 1) or 1)
    economic_model_workers = max(1, min(economic_model_workers, max(1, int(n_models))))
    return {
        "enabled": bool(raw.get("enabled", True)),
        "model_workers": int(model_workers),
        "economic_model_workers": int(economic_model_workers),
        "nested_candidate_workers": int(nested_workers),
    }


def screening_config(cfg: dict[str, Any]) -> dict[str, int | bool]:
    ms = cfg.get("model_selection", {}) or {}
    raw = (ms.get("screening", {}) or {}) if isinstance(ms, dict) else {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "shortlist_top_k_per_path": int(raw.get("shortlist_top_k_per_path", 4) or 4),
        "min_keep_per_path": int(raw.get("min_keep_per_path", 2) or 2),
        "adaptive_execution_budget": bool(raw.get("adaptive_execution_budget", False)),
        "require_positive_feasibility": bool(raw.get("require_positive_feasibility", False)),
    }


def execution_aware_horizon_policy_config(cfg: dict[str, Any]) -> ExecutionAwareHorizonPolicy:
    """P35: Parse execution_aware_horizon_policy from YAML config.

    Located at: model_selection.execution_aware_horizon_policy
    Returns a validated ExecutionAwareHorizonPolicy with documented defaults
    when the section is absent.
    """
    ms = cfg.get("model_selection", {}) or {}
    raw = (ms.get("execution_aware_horizon_policy", {}) or {}) if isinstance(ms, dict) else {}
    return parse_execution_aware_horizon_policy(raw)


def orientation_policy_config(cfg: dict[str, Any]) -> OrientationPolicy:
    """P37: Parse orientation_policy from YAML config.

    Located at: model_selection.orientation_policy
    """
    ms = cfg.get("model_selection", {}) or {}
    raw = (ms.get("orientation_policy", {}) or {}) if isinstance(ms, dict) else {}
    return parse_orientation_policy(raw)
