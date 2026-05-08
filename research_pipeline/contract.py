"""ResearchContract — single source of truth for all research assumptions.

No stage should read raw YAML keys. Stages read from this contract.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


# ── Defaults (only used when config is completely missing) ───────────────────

_DEFAULTS: dict[str, Any] = {
    "backtest": {
        "start_date": "2018-01-01",
        "end_date": "2024-01-01",
    },
    "research": {
        "train_years": 5.0,
        "test_years": 1.0,
        "step_years": 1.0,
        "walk_forward_windows": False,
        "walk_forward_train_ratio": 0.70,
    },
    "model_selection": {
        "lookahead_horizon_days": 20,
        "max_positions": 10,
        "min_positions": 3,
        "production_horizons": [],
        "validation": {
            "n_windows": 4,
            "train_ratio": 0.70,
            "validation_days": 126,
            "min_train_days": 504,
            "embargo_days": None,  # computed from horizon
        },
        "search": {
            "max_candidates": 24,
            "prefilter_top_k": 6,
            "shortlist_top_k_per_path": 4,
            "min_keep_per_path": 2,
            "proxy_max_iter": 30,
        },
        "adaptive_horizon": {
            "apply": False,
            "candidate_horizons": [5, 10, 20, 40, 60, 63],
            "candidate_rebalance_frequencies": [2, 3, 5, 10, 20, 63],
        },
        "horizon_gate": {
            "enabled": True,
            "min_production_features": 3,
            "min_families": 2,
            "max_family_concentration": 0.6,
            "min_effective_signals": 1.5,
            "cost_bps": 10.0,
        },
        "horizon_alignment": {
            "multiplier": 2.0,
            "enforce": False,
        },
        "feature_audit": {
            "min_ic_tstat": 1.5,
            "min_production_ic_valid_days": 50,
            "fail_if_below_minimum": False,
            "minimum_admitted_features": 3,
        },
        "execution_aware_horizon": {
            "apply": False,
        },
        "ensemble_size": 3,
        "economic_policy": {
            "max_cost_pnl": 0.50,
            "max_impact_fraction": 0.70,
        },
    },
    "feature_selection": {
        "feature_subset": [],
        "short_feature_subset": [],
        "overlay_feature_subset": [],
    },
    "execution": {
        "long_only": False,
        "enable_shorts": True,
    },
    "horizon_config": {
        "persistence_filter": {},
    },
    "regime": {
        "enabled": False,
        "confirmation_days": 5,
    },
    "data": {
        "provider": "polygon",
        "cache_dir": ".cache",
        "cache_ttl_days": 7,
    },
    "cost_model": {
        "commission_bps": 1.0,
        "spread_bps": 1.0,
        "borrow_bps_annual": 50.0,
        "impact_eta": 0.142,
        "impact_alpha": 0.314,
        "impact_gamma": 0.6,
        "default_adv_usd": 50_000_000,
        "default_daily_vol": 0.02,
    },
}


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge override into base recursively."""
    result = dict(base)
    for key, val in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(val, dict):
            result[key] = _deep_merge(result[key], val)
        else:
            result[key] = val
    return result


def _get_nested(cfg: dict, *keys: str, default: Any = None) -> Any:
    """Safely traverse nested dict keys."""
    cur = cfg
    for k in keys:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(k, default)
        if cur is default:
            return default
    return cur


@dataclass(frozen=True)
class HorizonSpec:
    """Horizon configuration for a single run."""
    target_horizon_days: int
    production_horizon_days: int
    holding_period_days: int
    embargo_days: int
    candidate_horizons: tuple[int, ...]
    candidate_rebalance_frequencies: tuple[int, ...]
    adaptive_horizon_enabled: bool
    horizon_gate_enabled: bool
    horizon_alignment_multiplier: float
    horizon_alignment_enforce: bool


@dataclass(frozen=True)
class ValidationSpec:
    """Walk-forward validation configuration."""
    train_years: float
    test_years: float
    step_years: float
    n_windows: int
    train_ratio: float
    validation_days: int
    min_train_days: int
    embargo_days: int
    walk_forward_windows_mode: bool


@dataclass(frozen=True)
class FeatureAdmissionSpec:
    """Feature admission thresholds."""
    min_ic_tstat: float
    min_production_ic_valid_days: int
    fail_if_below_minimum: bool
    minimum_admitted_features: int


@dataclass(frozen=True)
class PromotionGateSpec:
    """Promotion gate thresholds."""
    min_ic_tstat: float
    min_ic_ir: float
    min_sharpe: float
    min_cost_aware_sharpe: float
    max_beta_abs_mean: float
    max_sector_abs_mean: float
    min_beat_rate: float
    min_psr: float
    min_long_leg_sharpe: float
    min_short_leg_sharpe: float
    max_drawdown: float
    min_decile_spread: float
    min_tail_monotonicity: float
    max_cost_to_gross_pnl: float
    min_windows: int
    nested_min_ic: float
    nested_min_sharpe: float
    nested_min_windows: int
    robust_halflife: bool
    robust_caic_ratio: float
    robust_turnover: float
    diagnostic_only: bool


@dataclass(frozen=True)
class CostViabilitySpec:
    """Cost viability thresholds."""
    commission_bps: float
    spread_bps: float
    borrow_bps_annual: float
    impact_eta: float
    impact_alpha: float
    impact_gamma: float
    default_adv_usd: float
    default_daily_vol: float
    max_cost_pnl: float
    max_impact_fraction: float


@dataclass(frozen=True)
class ExecutionSpec:
    """Portfolio execution configuration."""
    long_only: bool
    enable_shorts: bool
    max_positions: int
    min_positions: int


@dataclass(frozen=True)
class SearchSpec:
    """Nested search configuration."""
    max_candidates: int
    prefilter_top_k: int
    shortlist_top_k_per_path: int
    min_keep_per_path: int
    proxy_max_iter: int


@dataclass(frozen=True)
class ReportSpec:
    """Report output configuration."""
    output_dir: str
    model_comparison_path: str
    model_diagnostics_path: str
    research_report_path: str
    feature_admission_path: str
    ic_decay_path: str
    horizon_contract_audit_path: str
    feature_research_ledger_path: str
    halflife_persistence_audit_path: str
    baseline_diagnostics_path: str
    alpha_execution_decomposition_path: str
    alpha_capture_summary_path: str
    economic_selection_audit_path: str
    optimizer_score_weight_audit_path: str
    economic_policy_report_path: str
    simulation_runtime_report_path: str
    horizon_governance_path: str
    horizon_sweep_manifest_path: str
    long_alpha_candidate_path: str


@dataclass
class ResearchContract:
    """Central contract owning all research assumptions.

    Built from YAML config + CLI overrides. Immutable after construction.
    All pipeline stages read from this object — never from raw config dicts.
    """
    # Raw config preserved for backward compatibility with legacy modules
    raw_config: dict[str, Any]

    # Date range
    start_date: str
    end_date: str

    # Horizon
    horizon: HorizonSpec

    # Validation
    validation: ValidationSpec

    # Feature admission
    feature_admission: FeatureAdmissionSpec

    # Cost viability
    cost_viability: CostViabilitySpec

    # Execution
    execution: ExecutionSpec

    # Search
    search: SearchSpec

    # Promotion gates (loaded from validation.py PromotionGateConfig)
    promotion_gates: PromotionGateSpec | None = None

    # Feature subsets
    feature_subset: tuple[str, ...] = ()
    short_feature_subset: tuple[str, ...] = ()
    overlay_feature_subset: tuple[str, ...] = ()

    # Regime
    regime_enabled: bool = False
    regime_confirmation_days: int = 5

    # Reports
    reports: ReportSpec | None = None

    # Churn filter
    persistence_filter: dict[str, Any] = field(default_factory=dict)

    # Data
    data_provider: str = "polygon"
    cache_dir: str = ".cache"
    cache_ttl_days: int = 7

    # Ensemble
    ensemble_size: int = 3

    # Risk-adjusted target
    risk_adj_target: bool = False

    # Debug
    debug_validation: bool = False

    # Discard suspicious models
    discard_suspicious_models: bool = False

    # Min test days
    min_test_days: int = 30

    # Min OOS days
    min_oos_days: int | None = None

    # Select metric
    select_metric: str = "oos_deflated_sharpe"

    # Limit tickers
    limit_tickers: int = 0

    # Save all models
    save_all_models: bool = False

    # Simplified mode
    simplified: bool = False

    # Viability check
    viability_check: bool = False

    # Research pillars (C2-C21)
    regime_conditioning: bool = False
    feature_redundancy: bool = False
    ensemble_weighting: bool = False
    meta_labeling: bool = False
    meta_model_name: str | None = None
    meta_model_type: str = "ridge"
    short_modeling: bool = False
    horizon_optimization: bool = False
    confidence_weighting: bool = False
    regime_gating: bool = False
    asymmetry_correction: bool = False
    capacity_analysis: bool = False
    marginal_value: bool = False
    cost_sensitivity: bool = False
    joint_optimization: bool = False
    deployability_ranking: bool = False

    @property
    def do_shorts(self) -> bool:
        return self.execution.enable_shorts and not self.execution.long_only

    @property
    def primary_path(self) -> str:
        return "long_short_spread" if self.do_shorts else "long_only_overlay"

    @classmethod
    def from_config(
        cls,
        config: dict[str, Any],
        cli_overrides: dict[str, Any] | None = None,
    ) -> "ResearchContract":
        """Build ResearchContract from YAML config + CLI overrides.

        CLI overrides take precedence over config, which takes precedence over defaults.
        """
        cfg = _deep_merge(_DEFAULTS, config)
        if cli_overrides:
            cfg = _deep_merge(cfg, cli_overrides)

        bt = cfg.get("backtest", {})
        research = cfg.get("research", {})
        ms = cfg.get("model_selection", {})
        feat_sel = cfg.get("feature_selection", {})
        exe = cfg.get("execution", {})
        regime = cfg.get("regime", {})
        data = cfg.get("data", {})
        cost = cfg.get("cost_model", {})
        h_cfg = cfg.get("horizon_config", {})
        val = ms.get("validation", {})
        h_gate = ms.get("horizon_gate", {})
        h_align = ms.get("horizon_alignment", {})
        f_audit = ms.get("feature_audit", {})
        eah = ms.get("execution_aware_horizon", {})
        econ = ms.get("economic_policy", {})
        search = ms.get("search", {})

        # Horizon
        lookahead = int(ms.get("lookahead_horizon_days", 20))
        holding = lookahead
        embargo_raw = val.get("embargo_days")
        embargo = int(embargo_raw) if embargo_raw else 2 * holding
        ah = ms.get("adaptive_horizon", {})

        horizon = HorizonSpec(
            target_horizon_days=lookahead,
            production_horizon_days=lookahead,
            holding_period_days=holding,
            embargo_days=embargo,
            candidate_horizons=tuple(ah.get("candidate_horizons", [5, 10, 20, 40, 60, 63])),
            candidate_rebalance_frequencies=tuple(
                ah.get("candidate_rebalance_frequencies", [2, 3, 5, 10, 20, 63])
            ),
            adaptive_horizon_enabled=bool(ah.get("apply", False)),
            horizon_gate_enabled=bool(h_gate.get("enabled", True)),
            horizon_alignment_multiplier=float(h_align.get("multiplier", 2.0)),
            horizon_alignment_enforce=bool(h_align.get("enforce", False)),
        )

        # Validation
        validation = ValidationSpec(
            train_years=float(research.get("train_years", 5)),
            test_years=float(research.get("test_years", 1)),
            step_years=float(research.get("step_years", 1)),
            n_windows=int(val.get("n_windows", 4)),
            train_ratio=float(val.get("train_ratio", 0.70)),
            validation_days=int(val.get("validation_days", 126)),
            min_train_days=int(val.get("min_train_days", 504)),
            embargo_days=embargo,
            walk_forward_windows_mode=bool(research.get("walk_forward_windows", False)),
        )

        # Feature admission
        feature_admission = FeatureAdmissionSpec(
            min_ic_tstat=float(f_audit.get("min_ic_tstat", 1.5)),
            min_production_ic_valid_days=int(f_audit.get("min_production_ic_valid_days", 50)),
            fail_if_below_minimum=bool(f_audit.get("fail_if_below_minimum", False)),
            minimum_admitted_features=int(f_audit.get("minimum_admitted_features", 3)),
        )

        # Cost viability
        cost_viability = CostViabilitySpec(
            commission_bps=float(cost.get("commission_bps", 1.0)),
            spread_bps=float(cost.get("spread_bps", 1.0)),
            borrow_bps_annual=float(cost.get("borrow_bps_annual", 50.0)),
            impact_eta=float(cost.get("impact_eta", 0.142)),
            impact_alpha=float(cost.get("impact_alpha", 0.314)),
            impact_gamma=float(cost.get("impact_gamma", 0.6)),
            default_adv_usd=float(cost.get("default_adv_usd", 50_000_000)),
            default_daily_vol=float(cost.get("default_daily_vol", 0.02)),
            max_cost_pnl=float(econ.get("max_cost_pnl", 0.50)),
            max_impact_fraction=float(econ.get("max_impact_fraction", 0.70)),
        )

        # Execution
        execution = ExecutionSpec(
            long_only=bool(exe.get("long_only", False)),
            enable_shorts=bool(exe.get("enable_shorts", True)),
            max_positions=int(ms.get("max_positions", 10)),
            min_positions=int(ms.get("min_positions", 3)),
        )

        # Search
        search_spec = SearchSpec(
            max_candidates=int(search.get("max_candidates", 24)),
            prefilter_top_k=int(search.get("prefilter_top_k", 6)),
            shortlist_top_k_per_path=int(search.get("shortlist_top_k_per_path", 4)),
            min_keep_per_path=int(search.get("min_keep_per_path", 2)),
            proxy_max_iter=int(search.get("proxy_max_iter", 30)),
        )

        # Reports
        reports = ReportSpec(
            output_dir="output/models",
            model_comparison_path="output/models/model_comparison.csv",
            model_diagnostics_path="output/models/model_diagnostics.csv",
            research_report_path="output/models/research_report.txt",
            feature_admission_path="output/models/feature_admission.csv",
            ic_decay_path="output/models/alpha_ic_decay.csv",
            horizon_contract_audit_path="output/models/horizon_contract_audit.csv",
            feature_research_ledger_path="output/models/feature_research_ledger.csv",
            halflife_persistence_audit_path="output/models/halflife_persistence_audit.csv",
            baseline_diagnostics_path="output/models/baseline_diagnostics.parquet",
            alpha_execution_decomposition_path="output/models/alpha_execution_decomposition.parquet",
            alpha_capture_summary_path="output/models/alpha_capture_summary.csv",
            economic_selection_audit_path="output/models/economic_selection_audit.parquet",
            optimizer_score_weight_audit_path="output/models/optimizer_score_weight_audit.parquet",
            economic_policy_report_path="output/models/economic_policy_report.txt",
            simulation_runtime_report_path="output/models/simulation_runtime_report.json",
            horizon_governance_path="output/models/horizon_governance.json",
            horizon_sweep_manifest_path="output/models/horizon_sweep_manifest.json",
            long_alpha_candidate_path="output/models/long_alpha_candidate.csv",
        )

        return cls(
            raw_config=cfg,
            start_date=str(bt.get("start_date", "2018-01-01")),
            end_date=str(bt.get("end_date", "2024-01-01")),
            horizon=horizon,
            validation=validation,
            feature_admission=feature_admission,
            cost_viability=cost_viability,
            execution=execution,
            search=search_spec,
            feature_subset=tuple(str(c).strip() for c in feat_sel.get("feature_subset", []) if str(c).strip()),
            short_feature_subset=tuple(str(c).strip() for c in feat_sel.get("short_feature_subset", []) if str(c).strip()),
            overlay_feature_subset=tuple(str(c).strip() for c in feat_sel.get("overlay_feature_subset", []) if str(c).strip()),
            regime_enabled=bool(regime.get("enabled", False)),
            regime_confirmation_days=int(regime.get("confirmation_days", 5)),
            reports=reports,
            persistence_filter=h_cfg.get("persistence_filter", {}),
            data_provider=str(data.get("provider", "polygon")),
            cache_dir=str(data.get("cache_dir", ".cache")),
            cache_ttl_days=int(data.get("cache_ttl_days", 7)),
            ensemble_size=int(ms.get("ensemble_size", 3)),
        )

    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        cli_overrides: dict[str, Any] | None = None,
    ) -> "ResearchContract":
        """Load from YAML file."""
        path = Path(path)
        if not path.exists():
            logger.warning("Config not found at %s, using defaults", path)
            cfg = {}
        else:
            with open(path) as f:
                cfg = yaml.safe_load(f) or {}
        return cls.from_config(cfg, cli_overrides)
