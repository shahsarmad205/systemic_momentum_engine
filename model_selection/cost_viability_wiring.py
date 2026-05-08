"""Institutional cost viability wiring for run_model_selection.py.

Bridges the CostViabilityEngine into the existing model selection pipeline:
1. Feature-level cost viability scoring (replaces flat 10bps in horizon gate)
2. Candidate-level cost viability after model evaluation
3. Alpha-to-trade policy before simulator calls
4. Report generation (scorecard, stress test, turnover attribution, dominated candidates)

Usage:
    from model_selection.cost_viability_wiring import (
        wire_cost_viability_into_pipeline,
        evaluate_feature_cost_viability,
        evaluate_candidate_cost_viability,
        apply_alpha_to_trade_policy,
        generate_cost_viability_reports,
    )

Insertion points in run_model_selection.py:
    - After run_alpha_research (L6691): evaluate_feature_cost_viability()
    - After _evaluate_model_family (L5350): evaluate_candidate_cost_viability()
    - Before simulate_executable_portfolio (L4737): apply_alpha_to_trade_policy()
    - At end of main(): generate_cost_viability_reports()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_selection.cost_viability_engine import (
    CostViabilityEngine,
    CostStatus,
    ViabilityResult,
    AlphaToTradeDecision,
    BandResult,
    generate_scorecard,
    generate_stress_test_report,
    generate_cost_dominated_report,
    generate_turnover_attribution_report,
)
from model_selection.research_contract import FEATURE_SPECS
from model_selection.horizon_eligibility import HorizonEligibilityContract

logger = logging.getLogger(__name__)

_PREFERRED_ALPHA_TARGET_TYPE = "net_residual_return"


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class FeatureCostResult:
    """Cost viability result for a single feature at a single horizon."""
    feature: str
    family: str
    horizon: int
    ic: float
    ic_tstat: float
    halflife: float
    expected_turnover: float
    adv_usd: float
    daily_vol: float
    cost_status: CostStatus
    expected_alpha_bps: float
    expected_cost_bps: float
    net_expected_alpha_bps: float
    alpha_cost_ratio: float
    capacity_score: float
    rejection_reason: str


@dataclass
class CandidateCostResult:
    """Cost viability result for a model candidate."""
    candidate_id: str
    model_name: str
    model_kind: str
    horizon: int
    feature_view: str
    n_features: int
    cost_status: CostStatus
    expected_alpha_bps: float
    expected_cost_bps: float
    net_expected_alpha_bps: float
    alpha_cost_ratio: float
    turnover: float
    capacity_score: float
    rejection_reason: str


@dataclass
class CostViabilityWiringState:
    """Accumulates all cost viability results during a pipeline run."""
    feature_results: list[FeatureCostResult] = field(default_factory=list)
    production_feature_results: list[FeatureCostResult] = field(default_factory=list)
    candidate_results: list[CandidateCostResult] = field(default_factory=list)
    alpha_to_trade_decisions: list[AlphaToTradeDecision] = field(default_factory=list)
    band_results: list[BandResult] = field(default_factory=list)
    stress_results: dict[str, Any] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def engine(self) -> CostViabilityEngine:
        return CostViabilityEngine(config=_cost_viability_engine_config(self.config))


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _cost_viability_engine_config(cfg: dict[str, Any]) -> dict[str, Any]:
    """Normalize full run config into the CostViabilityEngine contract."""
    cfg = cfg or {}
    out: dict[str, Any] = {}
    explicit = cfg.get("cost_viability", {}) if isinstance(cfg.get("cost_viability"), dict) else {}
    out = _deep_merge(out, explicit)
    for section in ("cost_model", "alpha_estimation", "classification", "promotion_gates", "stress_scenarios", "no_trade_band"):
        if section in cfg and section not in out:
            out[section] = cfg[section]
    ms = cfg.get("model_selection", {}) if isinstance(cfg.get("model_selection"), dict) else {}
    ms_cv = ms.get("cost_viability", {}) if isinstance(ms.get("cost_viability"), dict) else {}
    out = _deep_merge(out, ms_cv)
    return out or cfg


def _cost_rejection_code(result: FeatureCostResult) -> str:
    status = str(result.cost_status.value if hasattr(result.cost_status, "value") else result.cost_status)
    reason = str(result.rejection_reason or status).strip()
    if reason and reason != status:
        return f"{status}:{reason}"
    return status


def feature_cost_results_to_horizon_contracts(
    feature_results: list[FeatureCostResult],
    horizon: int,
    *,
    production_statuses: tuple[str, ...] = ("cost_viable",),
) -> dict[str, HorizonEligibilityContract]:
    """Convert institutional cost results into full horizon gate contracts.

    The horizon gate/reporting stack expects HorizonEligibilityContract objects.
    Keeping this conversion explicit prevents lightweight compatibility shims
    from bypassing required diagnostics such as ic_by_horizon.
    """
    h = int(horizon)
    eligible_statuses = {str(s) for s in production_statuses}
    contracts: dict[str, HorizonEligibilityContract] = {}
    for result in feature_results:
        status = str(result.cost_status.value if hasattr(result.cost_status, "value") else result.cost_status)
        prod_ok = status in eligible_statuses
        ic = _finite_float(result.ic, np.nan)
        halflife = _finite_float(result.halflife, np.nan)
        stat_ok = np.isfinite(ic) and ic != 0.0
        rejection = _cost_rejection_code(result)
        contracts[result.feature] = HorizonEligibilityContract(
            feature=result.feature,
            family=result.family,
            ic_by_horizon={h: ic} if np.isfinite(ic) else {},
            ic_decay_curve={h: ic} if np.isfinite(ic) else {},
            estimated_halflife=halflife if np.isfinite(halflife) else 0.0,
            statistically_admissible_horizons=[h] if stat_ok else [],
            statistical_rejections={} if stat_ok else {h: "IC_NOT_MEASURABLE"},
            production_admissible_horizons=[h] if prod_ok else [],
            production_rejections={} if prod_ok else {h: rejection},
            cost_adjusted_viable={h: prod_ok},
            cost_proxy_bps=float(result.expected_cost_bps),
            n_observations=0,
        )
    return contracts


def filter_feature_cost_results(
    feature_results: list[FeatureCostResult],
    features: list[str] | set[str] | tuple[str, ...],
) -> list[FeatureCostResult]:
    """Return cost results for an explicit production feature subset."""
    allowed = {str(f) for f in features}
    if not allowed:
        return []
    return [r for r in feature_results if str(r.feature) in allowed]


# ── Feature-level cost viability ─────────────────────────────────────────────

def evaluate_feature_cost_viability(
    alpha_admission: pd.DataFrame,
    alpha_decay: pd.DataFrame,
    df: pd.DataFrame,
    cfg: dict[str, Any],
    horizon: int,
) -> list[FeatureCostResult]:
    """Evaluate cost viability for each alpha-admission row.

    Called after run_alpha_research() in run_model_selection.py.
    Replaces the flat 10bps cost assumption in the horizon gate.

    Args:
        alpha_admission: DataFrame from run_alpha_research with per-feature IC stats
        alpha_decay: DataFrame with per-feature per-horizon IC decay
        df: Feature matrix panel (for ADV, vol lookups)
        cfg: Full config dict
        horizon: Current prediction horizon in days

    Returns:
        List of FeatureCostResult for each feature.
    """
    engine = CostViabilityEngine(config=_cost_viability_engine_config(cfg))
    results = []

    # Fail-fast: verify CostBreakdown constructor contract before per-feature work
    try:
        from model_selection.cost_viability_engine import CostBreakdown as _Cb
        _Cb(total_bps=0.0, commission_bps=0.0, spread_bps=0.0,
            temporary_impact_bps=0.0, permanent_impact_bps=0.0,
            borrow_bps=0.0, financing_bps=0.0, participation_rate=0.0,
            adv_usd=50_000_000.0, daily_vol=0.02)
    except Exception as _cb_exc:
        raise RuntimeError(
            f"CostBreakdown constructor contract validation failed: {_cb_exc}. "
            "Fix the required fields or disable cost viability."
        ) from _cb_exc

    # Extract per-feature stats from alpha_admission
    for _, row in alpha_admission.iterrows():
        feat = str(row.get("feature", ""))
        if not feat:
            continue

        spec = FEATURE_SPECS.get(feat)
        family = spec.family if spec else "unknown"

        ic = _finite_float(row.get("production_ic"), np.nan)
        if not np.isfinite(ic):
            ic = _lookup_decay_stat(alpha_decay, feat, horizon, "daily_spearman_ic", default=0.0)
        ic_tstat = _finite_float(row.get("production_ic_tstat"), np.nan)
        if not np.isfinite(ic_tstat):
            ic_tstat = _lookup_decay_stat(alpha_decay, feat, horizon, "daily_spearman_ic_tstat", default=0.0)
        halflife = _lookup_feature_halflife(alpha_decay, feat, horizon)
        if not np.isfinite(halflife):
            halflife = _finite_float(row.get("signal_halflife_days"), 0.0)
        turnover = _finite_float(row.get("turnover_mean"), np.nan)
        if not np.isfinite(turnover):
            turnover = _configured_feature_turnover(cfg)

        # Look up ADV and vol from panel
        adv_usd = _lookup_adv(df, feat)
        daily_vol = _lookup_daily_vol(df, feat)

        # IC decay at this horizon
        ic_decay = _lookup_ic_decay(alpha_decay, feat, horizon)

        result = engine.evaluate(
            candidate_id=f"feature_{feat}_h{horizon}",
            feature=feat,
            family=family,
            ic=ic,
            horizon=horizon,
            sigma_annual=0.20,
            halflife=halflife,
            expected_turnover=turnover,
            adv_usd=adv_usd,
            daily_vol=daily_vol,
            icir=ic / max(turnover, 0.001),
            t_stat=ic_tstat,
            ic_decay_curve=ic_decay,
        )

        results.append(FeatureCostResult(
            feature=feat,
            family=family,
            horizon=horizon,
            ic=ic,
            ic_tstat=ic_tstat,
            halflife=halflife,
            expected_turnover=turnover,
            adv_usd=adv_usd,
            daily_vol=daily_vol,
            cost_status=result.cost_status,
            expected_alpha_bps=result.expected_alpha_bps,
            expected_cost_bps=result.expected_cost_bps,
            net_expected_alpha_bps=result.net_expected_alpha_bps,
            alpha_cost_ratio=result.alpha_cost_ratio,
            capacity_score=result.capacity_score,
            rejection_reason=result.rejection_reason,
        ))

    logger.info("Feature cost viability: %d features evaluated at h%dd", len(results), horizon)
    return results


def _lookup_adv(df: pd.DataFrame, feature: str) -> float:
    """Look up median ADV for a feature's non-null dates."""
    adv_col = None
    for c in ["adv_dollar_20", "adv_dollar", "adv_usd"]:
        if c in df.columns:
            adv_col = c
            break
    if adv_col is None:
        return 50_000_000  # default from config
    adv = float(df[adv_col].median())
    return adv if np.isfinite(adv) and adv > 0 else 50_000_000


def _lookup_daily_vol(df: pd.DataFrame, feature: str) -> float:
    """Look up median daily vol for the panel."""
    vol_col = None
    for c in ["daily_return", "forward_return"]:
        if c in df.columns:
            vol_col = c
            break
    if vol_col is None:
        return 0.02  # default from config
    vol = float(df[vol_col].std())
    return vol if np.isfinite(vol) and vol > 0 else 0.02


def _finite_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def _configured_feature_turnover(cfg: dict[str, Any]) -> float:
    """Read expected feature turnover from config, falling back to legacy default."""
    for path in (
        ("cost_viability", "feature_expected_turnover"),
        ("model_selection", "cost_viability", "feature_expected_turnover"),
        ("model_selection", "horizon_gate", "feature_expected_turnover"),
    ):
        cur: Any = cfg
        for key in path:
            if not isinstance(cur, dict) or key not in cur:
                cur = None
                break
            cur = cur[key]
        if cur is not None:
            return max(0.0, _finite_float(cur, 0.10))
    return 0.10


def _feature_decay_rows(alpha_decay: pd.DataFrame, feature: str, horizon: int | None = None) -> pd.DataFrame:
    if alpha_decay is None or alpha_decay.empty or "feature" not in alpha_decay.columns:
        return pd.DataFrame()
    rows = alpha_decay[alpha_decay["feature"].astype(str) == str(feature)].copy()
    if rows.empty:
        return rows
    if "target_type" in rows.columns:
        preferred = rows[rows["target_type"].astype(str) == _PREFERRED_ALPHA_TARGET_TYPE]
        if not preferred.empty:
            rows = preferred
    if horizon is not None and "horizon_days" in rows.columns:
        same_h = rows[pd.to_numeric(rows["horizon_days"], errors="coerce") == int(horizon)]
        if not same_h.empty:
            rows = same_h
    return rows


def _lookup_decay_stat(
    alpha_decay: pd.DataFrame,
    feature: str,
    horizon: int,
    column: str,
    *,
    default: float,
) -> float:
    rows = _feature_decay_rows(alpha_decay, feature, horizon)
    if rows.empty or column not in rows.columns:
        return float(default)
    vals = pd.to_numeric(rows[column], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return float(default)
    return float(vals.mean())


def _lookup_feature_halflife(alpha_decay: pd.DataFrame, feature: str, horizon: int) -> float:
    rows = _feature_decay_rows(alpha_decay, feature, horizon)
    if rows.empty or "signal_halflife_days" not in rows.columns:
        return np.nan
    vals = pd.to_numeric(rows["signal_halflife_days"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if vals.empty:
        return np.nan
    return float(vals.median())


def _lookup_ic_decay(
    alpha_decay: pd.DataFrame,
    feature: str,
    horizon: int,
) -> dict[int, float] | None:
    """Look up IC decay curve for a feature."""
    if alpha_decay is None or alpha_decay.empty:
        return None
    feat_rows = _feature_decay_rows(alpha_decay, feature)
    if feat_rows.empty:
        return None
    horizon_col = "horizon_days" if "horizon_days" in feat_rows.columns else "horizon"
    ic_col = "daily_spearman_ic" if "daily_spearman_ic" in feat_rows.columns else "mean_ic"
    if horizon_col not in feat_rows.columns or ic_col not in feat_rows.columns:
        return None
    decay = {}
    grouped = feat_rows.assign(
        _h=pd.to_numeric(feat_rows[horizon_col], errors="coerce"),
        _ic=pd.to_numeric(feat_rows[ic_col], errors="coerce"),
    ).dropna(subset=["_h", "_ic"])
    for h, grp in grouped.groupby("_h", sort=True):
        h_int = int(h)
        ic = float(grp["_ic"].mean())
        if h_int > 0 and np.isfinite(ic):
            decay[h_int] = ic
    return decay if decay else None


# ── Candidate-level cost viability ───────────────────────────────────────────

def evaluate_candidate_cost_viability(
    candidate_id: str,
    model_name: str,
    model_kind: str,
    horizon: int,
    feature_view: str,
    active_features: list[str],
    oos_metrics: dict[str, float],
    cfg: dict[str, Any],
) -> CandidateCostResult:
    """Evaluate cost viability for a model candidate after evaluation.

    Called after _evaluate_model_family() in run_model_selection.py.

    Args:
        candidate_id: Unique candidate identifier
        model_name: Model name (e.g., "xgboost")
        model_kind: "long", "short", or "overlay"
        horizon: Prediction horizon in days
        feature_view: "full" or "program"
        active_features: List of features used by this candidate
        oos_metrics: Dict of OOS metrics from evaluation (sharpe, ic, turnover, etc.)
        cfg: Full config dict

    Returns:
        CandidateCostResult with full cost viability assessment.
    """
    engine = CostViabilityEngine(config=_cost_viability_engine_config(cfg))

    ic = float(oos_metrics.get("oos_ic", 0.0))
    turnover = float(oos_metrics.get("exec_turnover_mean", 0.10))
    sharpe = float(oos_metrics.get("oos_deflated_sharpe", 0.0))

    # Primary feature for cost lookup
    primary_feature = active_features[0] if active_features else "unknown"
    spec = FEATURE_SPECS.get(primary_feature)
    family = spec.family if spec else "unknown"

    # Use median halflife across features
    halflife = float(oos_metrics.get("signal_halflife_days", 0.0))

    result = engine.evaluate(
        candidate_id=candidate_id,
        feature=primary_feature,
        family=family,
        ic=ic,
        horizon=horizon,
        sigma_annual=0.20,
        halflife=halflife,
        expected_turnover=turnover,
        adv_usd=50_000_000,  # From config in production
        daily_vol=0.02,
        icir=ic / max(turnover, 0.001),
        t_stat=float(oos_metrics.get("ic_tstat", 0.0)),
        is_short=(model_kind == "short"),
        position_weight=1.0 / max(len(active_features), 1),
        capital=10_000_000.0,
    )

    return CandidateCostResult(
        candidate_id=candidate_id,
        model_name=model_name,
        model_kind=model_kind,
        horizon=horizon,
        feature_view=feature_view,
        n_features=len(active_features),
        cost_status=result.cost_status,
        expected_alpha_bps=result.expected_alpha_bps,
        expected_cost_bps=result.expected_cost_bps,
        net_expected_alpha_bps=result.net_expected_alpha_bps,
        alpha_cost_ratio=result.alpha_cost_ratio,
        turnover=turnover,
        capacity_score=result.capacity_score,
        rejection_reason=result.rejection_reason,
    )


# ── Alpha-to-trade policy ────────────────────────────────────────────────────

def apply_alpha_to_trade_policy(
    candidate_id: str,
    current_weight: float,
    target_weight: float,
    expected_alpha_bps: float,
    expected_cost_bps: float,
    sigma_annual: float,
    adv_usd: float,
    ic: float,
    cfg: dict[str, Any],
) -> AlphaToTradeDecision:
    """Apply alpha-to-trade policy before simulator call.

    Called before simulate_executable_portfolio() in run_model_selection.py.
    Approves a trade only when expected incremental alpha exceeds expected
    incremental cost by a configurable margin of safety.

    Args:
        candidate_id: Unique candidate identifier
        current_weight: Current portfolio weight
        target_weight: Target portfolio weight
        expected_alpha_bps: Expected alpha in basis points
        expected_cost_bps: Expected cost in basis points
        sigma_annual: Annualized volatility
        adv_usd: Average daily volume in USD
        ic: Out-of-sample IC
        cfg: Full config dict

    Returns:
        AlphaToTradeDecision with trade_approved flag and diagnostics.
    """
    engine = CostViabilityEngine(config=_cost_viability_engine_config(cfg))

    decision = engine.alpha_to_trade_decision(
        candidate_id=candidate_id,
        current_weight=current_weight,
        target_weight=target_weight,
        expected_alpha_bps=expected_alpha_bps,
        expected_cost_bps=expected_cost_bps,
        sigma_annual=sigma_annual,
        adv_usd=adv_usd,
        ic=ic,
    )

    return decision


# ── No-trade band policy ─────────────────────────────────────────────────────

def apply_no_trade_band(
    candidate_id: str,
    current_weight: float,
    target_weight: float,
    expected_cost_bps: float,
    daily_vol: float,
    adv_usd: float,
    abs_ic: float,
    expected_alpha_bps: float,
    cfg: dict[str, Any],
) -> BandResult:
    """Apply no-trade band to reduce unnecessary turnover.

    Called before simulator to determine if a trade should be executed.

    Args:
        candidate_id: Unique candidate identifier
        current_weight: Current portfolio weight
        target_weight: Target portfolio weight
        expected_cost_bps: Expected cost in basis points
        daily_vol: Daily volatility
        adv_usd: Average daily volume in USD
        abs_ic: Absolute IC
        expected_alpha_bps: Expected alpha in basis points
        cfg: Full config dict

    Returns:
        BandResult with turnover and cost diagnostics.
    """
    from model_selection.cost_viability_engine import NoTradeBandEngine

    band_engine = NoTradeBandEngine(config=_cost_viability_engine_config(cfg))

    return band_engine.apply(
        candidate_id=candidate_id,
        current_weight=current_weight,
        target_weight=target_weight,
        expected_cost_bps=expected_cost_bps,
        daily_vol=daily_vol,
        adv_usd=adv_usd,
        abs_ic=abs_ic,
        expected_alpha_bps=expected_alpha_bps,
    )


# ── Report generation ────────────────────────────────────────────────────────

def generate_cost_viability_reports(
    feature_results: list[FeatureCostResult],
    candidate_results: list[CandidateCostResult],
    band_results: list[BandResult],
    cfg: dict[str, Any],
    output_dir: str | Path = "output/models/cost_viability",
) -> dict[str, Path]:
    """Generate all cost viability reports.

    Called at the end of main() in run_model_selection.py.

    Args:
        feature_results: Per-feature cost viability results
        candidate_results: Per-candidate cost viability results
        band_results: No-trade band results
        cfg: Full config dict
        output_dir: Output directory for reports

    Returns:
        Dict mapping report name to file path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert to ViabilityResult format for report generators
    viability_results = _feature_results_to_viability(feature_results)

    paths = {}

    # 1. Cost viability scorecard
    scorecard_path = output_dir / "cost_viability_scorecard.csv"
    generate_scorecard(viability_results, scorecard_path)
    paths["scorecard"] = scorecard_path

    # 2. Cost stress test
    engine = CostViabilityEngine(config=_cost_viability_engine_config(cfg))
    stress_results = {}
    for fr in feature_results:
        stress = engine.run_stress_test(
            candidate_id=f"feature_{fr.feature}_h{fr.horizon}",
            feature=fr.feature,
            family=fr.family,
            ic=fr.ic,
            horizon=fr.horizon,
            sigma_annual=0.20,
            halflife=fr.halflife,
            expected_turnover=fr.expected_turnover,
            adv_usd=fr.adv_usd,
            daily_vol=fr.daily_vol,
        )
        stress_results[fr.feature] = stress

    stress_path = output_dir / "cost_stress_test.csv"
    generate_stress_test_report(stress_results, stress_path)
    paths["stress_test"] = stress_path

    # 3. Turnover attribution
    turnover_path = output_dir / "turnover_attribution.csv"
    generate_turnover_attribution_report(band_results, turnover_path)
    paths["turnover_attribution"] = turnover_path

    # 4. Cost dominated candidates
    dominated_path = output_dir / "cost_dominated_candidates.csv"
    generate_cost_dominated_report(viability_results, dominated_path)
    paths["cost_dominated"] = dominated_path

    logger.info("Cost viability reports generated: %s", paths)
    return paths


def _feature_results_to_viability(
    results: list[FeatureCostResult],
) -> list[ViabilityResult]:
    """Convert FeatureCostResult to ViabilityResult for report generators."""
    from model_selection.cost_viability_engine import ViabilityResult, CostBreakdown

    out = []
    for r in results:
        out.append(ViabilityResult(
            candidate_id=f"feature_{r.feature}_h{r.horizon}",
            feature=r.feature,
            family=r.family,
            sleeve="",
            horizon=r.horizon,
            regime="",
            ic=r.ic,
            icir=r.ic / max(r.expected_turnover, 0.001),
            t_stat=r.ic_tstat,
            halflife=r.halflife,
            sigma_annual=0.20,
            daily_vol=r.daily_vol,
            adv_usd=r.adv_usd,
            expected_turnover=r.expected_turnover,
            expected_alpha_bps=r.expected_alpha_bps,
            expected_cost_bps=r.expected_cost_bps,
            net_expected_alpha_bps=r.net_expected_alpha_bps,
            alpha_cost_ratio=r.alpha_cost_ratio,
            capacity_score=r.capacity_score,
            cost_status=r.cost_status,
            rejection_reason=r.rejection_reason,
            cost_breakdown=CostBreakdown(
                commission_bps=1.0,
                spread_bps=1.0,
                temporary_impact_bps=r.expected_cost_bps * 0.5,
                permanent_impact_bps=r.expected_cost_bps * 0.3,
                borrow_bps=0.0,
                financing_bps=0.0,
                participation_rate=0.0,
                adv_usd=r.adv_usd,
                daily_vol=r.daily_vol,
                total_bps=r.expected_cost_bps,
            ),
        ))
    return out


# ── Horizon gate integration ─────────────────────────────────────────────────

def summarize_feature_cost_gate(
    feature_results: list[FeatureCostResult],
    cfg: dict[str, Any],
    horizon: int,
) -> tuple[bool, dict[str, Any]]:
    """Summarize already-computed feature cost results for the horizon gate."""
    viable = [r for r in feature_results if r.cost_status == CostStatus.COST_VIABLE]
    marginal = [r for r in feature_results if r.cost_status == CostStatus.MARGINAL]
    dominated = [
        r for r in feature_results
        if r.cost_status not in (CostStatus.COST_VIABLE, CostStatus.MARGINAL)
    ]

    hg_cfg = cfg.get("model_selection", {}).get("horizon_gate", {})
    min_features = int(hg_cfg.get("min_production_features", 3))
    min_families = int(hg_cfg.get("min_families", 2))

    viable_families = {r.family for r in viable if r.family != "unknown"}
    blocked = len(viable) < min_features or len(viable_families) < min_families

    diagnostics = {
        "horizon": int(horizon),
        "n_viable": len(viable),
        "n_marginal": len(marginal),
        "n_dominated": len(dominated),
        "n_total": len(feature_results),
        "viable_families": sorted(viable_families),
        "n_viable_families": len(viable_families),
        "blocked": blocked,
        "block_reasons": [],
    }

    if len(viable) < min_features:
        diagnostics["block_reasons"].append(
            f"insufficient_viable_features: {len(viable)} < {min_features}"
        )
    if len(viable_families) < min_families:
        diagnostics["block_reasons"].append(
            f"insufficient_viable_families: {len(viable_families)} < {min_families}"
        )

    reason_counts: dict[str, int] = {}
    for r in dominated:
        reason = r.rejection_reason or r.cost_status.value
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    diagnostics["dominant_rejection_reasons"] = sorted(
        reason_counts.items(), key=lambda x: -x[1]
    )

    logger.info(
        "Institutional horizon gate h%dd: %d viable, %d marginal, %d dominated, blocked=%s",
        horizon, len(viable), len(marginal), len(dominated), blocked,
    )

    return blocked, diagnostics


def compute_institutional_horizon_gate(
    df: pd.DataFrame,
    feature_families: dict[str, list[str]],
    alpha_admission: pd.DataFrame,
    alpha_decay: pd.DataFrame,
    cfg: dict[str, Any],
    horizon: int,
) -> tuple[bool, list[FeatureCostResult], dict[str, Any]]:
    """Institutional horizon gate replacing flat 10bps cost assumption.

    Use this instead of compute_all_eligibility() with flat cost_bps.

    Args:
        df: Feature matrix panel
        feature_families: Dict mapping family name to list of features
        alpha_admission: DataFrame from run_alpha_research
        alpha_decay: DataFrame with IC decay curves
        cfg: Full config dict
        horizon: Current prediction horizon

    Returns:
        (horizon_blocked, feature_results, gate_diagnostics)
    """
    feature_results = evaluate_feature_cost_viability(
        alpha_admission, alpha_decay, df, cfg, horizon
    )
    blocked, diagnostics = summarize_feature_cost_gate(feature_results, cfg, horizon)
    return blocked, feature_results, diagnostics


# ── Convenience: full pipeline wiring ────────────────────────────────────────

def wire_cost_viability_into_pipeline(
    alpha_admission: pd.DataFrame,
    alpha_decay: pd.DataFrame,
    df: pd.DataFrame,
    cfg: dict[str, Any],
    horizon: int,
    output_dir: str | Path = "output/models/cost_viability",
) -> CostViabilityWiringState:
    """Full institutional cost viability wiring for a pipeline run.

    Call this after run_alpha_research() in run_model_selection.py.
    Returns a state object that accumulates results for later report generation.

    Args:
        alpha_admission: DataFrame from run_alpha_research
        alpha_decay: DataFrame with IC decay curves
        df: Feature matrix panel
        cfg: Full config dict
        horizon: Current prediction horizon
        output_dir: Output directory for reports

    Returns:
        CostViabilityWiringState with all results accumulated.
    """
    state = CostViabilityWiringState(config=cfg)

    # 1. Feature-level cost viability
    state.feature_results = evaluate_feature_cost_viability(
        alpha_admission, alpha_decay, df, cfg, horizon
    )

    # 2. Institutional horizon gate
    blocked, gate_diag = summarize_feature_cost_gate(
        state.feature_results, cfg, horizon
    )

    if blocked:
        logger.warning(
            "Horizon h%dd blocked by institutional cost gate: %s",
            horizon, gate_diag.get("block_reasons", []),
        )

    # 3. Generate reports
    generate_cost_viability_reports(
        state.feature_results,
        state.candidate_results,
        state.band_results,
        cfg,
        output_dir,
    )

    return state
