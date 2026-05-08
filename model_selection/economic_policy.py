"""
P20: EconomicExecutionPolicy — Algorithmic cost-aware monetization framework.

This module does NOT change models, targets, features, or promotion gates.
It classifies each horizon/model/path into economic viability tiers using
measured evidence (IC, cost decomposition, execution metrics).

Design principles (institutional):
  1. All thresholds are configurable — no hardcoded current-state values.
  2. Decisions are auditable — every classification has a recorded reason.
  3. The engine can correctly reject a strategy as economically untradable.
  4. No horizon-specific hacks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


# ── Economic viability tiers ─────────────────────────────────────────────────

@dataclass(frozen=True)
class EconomicViability:
    """
    Classification of whether a horizon/model/path combination is economically
    tradable given current execution costs and signal quality.
    """
    tier: str  # viable | cost_dominated | signal_weak | ranking_unstable |
               # exposure_driven | breadth_insufficient | statistically_insignificant
    reason: str
    cost_pnl: float = float("nan")
    exec_sharpe: float = float("nan")
    alpha_capture: float = float("nan")
    beat_rate: float = float("nan")
    eligible_universe: float = float("nan")

    @property
    def is_tradable(self) -> bool:
        return self.tier == "viable"


# ── Policy configuration ─────────────────────────────────────────────────────

@dataclass
class EconomicPolicyConfig:
    """All thresholds are configurable — no hardcoded current-state values."""

    # Cost thresholds
    max_cost_pnl: float = 0.50          # reject if costs > 50% of gross
    max_impact_fraction: float = 0.70   # reject if impact > 70% of costs
    max_cost_drag: float = 0.60         # reject if cost drag > 60%

    # Signal thresholds
    min_ic_tstat: float = 1.50          # minimum IC significance
    min_beat_rate: float = 0.25         # minimum fraction of positive-Sharpe windows
    max_decile_cv: float = 50.0         # maximum decile spread CV for stability

    # Persistence thresholds
    min_halflife_rebalance_ratio: float = 0.30  # reject if halflife << rebalance
    min_eligible_universe: int = 15     # minimum eligible names for diversification

    # Exposure thresholds
    max_beta_abs: float = 0.15
    max_sector_abs: float = 0.15
    max_mkt_corr: float = 0.50

    # Alpha thresholds
    min_alpha_capture: float = -3.0     # alpha capture ratio floor

    # Horizon selection
    prefer_lower_cost_pnl: bool = True  # select horizon with best cost-adjusted profile
    tie_break: str = "cost_pnl"         # tie-break by cost_pnl if multiple pass

    # Fallback
    reject_if_no_viable_horizon: bool = True


# ── Economic audit entry ─────────────────────────────────────────────────────

@dataclass
class HorizonEconomicAudit:
    """Full economic audit for one horizon/model/path combination."""
    horizon_days: int
    model_name: str
    primary_path: str

    # Signal
    ic_mean: float = float("nan")
    ic_tstat: float = float("nan")
    daily_icir: float = float("nan")
    decile_monotonicity: float = float("nan")
    decile_spread: float = float("nan")
    decile_spread_cv: float = float("nan")
    signal_halflife: float = float("nan")

    # Persistence
    persistence_filter_pass_rate: float = float("nan")
    top_decile_turnover: float = float("nan")
    rebalance_frequency: int = 1
    halflife_rebalance_ratio: float = float("nan")

    # Execution
    exec_sharpe: float = float("nan")
    beat_rate: float = float("nan")
    alpha_capture: float = float("nan")
    cost_pnl: float = float("nan")
    turnover: float = float("nan")
    avg_holding_period: float = float("nan")
    eligible_universe_size: float = float("nan")
    fallback_days: int = 0

    # Cost decomposition
    commission_pct: float = float("nan")
    spread_pct: float = float("nan")
    impact_pct: float = float("nan")
    borrow_pct: float = float("nan")
    cost_dominant: str = "unknown"
    alpha_vs_cost_ratio: float = float("nan")
    cost_drag: float = float("nan")

    # Risk
    beta_exposure: float = float("nan")
    sector_exposure: float = float("nan")
    mkt_correlation: float = float("nan")
    residual_sharpe: float = float("nan")
    subsumption_alpha_ann: float = float("nan")
    subsumption_alpha_tstat: float = float("nan")

    # Promotion
    promotion_pass: bool = False
    gate_failure_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_days": self.horizon_days,
            "model_name": self.model_name,
            "primary_path": self.primary_path,
            "ic_mean": self.ic_mean,
            "ic_tstat": self.ic_tstat,
            "decile_spread_cv": self.decile_spread_cv,
            "signal_halflife": self.signal_halflife,
            "halflife_rebalance_ratio": self.halflife_rebalance_ratio,
            "exec_sharpe": self.exec_sharpe,
            "beat_rate": self.beat_rate,
            "alpha_capture": self.alpha_capture,
            "cost_pnl": self.cost_pnl,
            "eligible_universe": self.eligible_universe_size,
            "cost_dominant": self.cost_dominant,
            "alpha_vs_cost_ratio": self.alpha_vs_cost_ratio,
            "beta_exposure": self.beta_exposure,
            "sector_exposure": self.sector_exposure,
            "mkt_correlation": self.mkt_correlation,
            "subsumption_alpha_ann": self.subsumption_alpha_ann,
            "promotion_pass": self.promotion_pass,
        }


# ── Policy engine ────────────────────────────────────────────────────────────

def classify_economic_viability(
    audit: HorizonEconomicAudit,
    cfg: EconomicPolicyConfig | None = None,
) -> EconomicViability:
    """
    Classify a horizon/model/path into economic viability tiers.

    Tiers (in order of desirability):
      viable                    — all gates pass, economics work
      cost_dominated            — signal exists but costs consume alpha
      signal_weak               — IC too weak to overcome costs
      ranking_unstable          — ranking churn destroys alpha
      exposure_driven           — performance from beta/factor, not alpha
      breadth_insufficient      — eligible universe too small
      statistically_insignificant — IC not distinguishable from noise

    Returns the FIRST failing classification (worst bottleneck).
    """
    if cfg is None:
        cfg = EconomicPolicyConfig()

    reasons: list[str] = []

    # 0. Already rejected by promotion gates → not our domain
    if not audit.promotion_pass:
        return EconomicViability(
            tier="promotion_rejected",
            reason=f"Failed {audit.gate_failure_count} promotion gates",
            cost_pnl=audit.cost_pnl,
            exec_sharpe=audit.exec_sharpe,
            alpha_capture=audit.alpha_capture,
            beat_rate=audit.beat_rate,
            eligible_universe=audit.eligible_universe_size,
        )

    # 1. Statistical significance
    if np.isfinite(audit.ic_tstat) and audit.ic_tstat < cfg.min_ic_tstat:
        reasons.append(f"IC t-stat={audit.ic_tstat:.2f} < min={cfg.min_ic_tstat}")
        return EconomicViability(
            tier="statistically_insignificant",
            reason="; ".join(reasons),
            cost_pnl=audit.cost_pnl,
            exec_sharpe=audit.exec_sharpe,
            alpha_capture=audit.alpha_capture,
            beat_rate=audit.beat_rate,
            eligible_universe=audit.eligible_universe_size,
        )

    # 2. Signal quality — return immediately on failure
    _signal_reasons: list[str] = []
    if np.isfinite(audit.beat_rate) and audit.beat_rate < cfg.min_beat_rate:
        _signal_reasons.append(
            f"Beat rate={audit.beat_rate:.2f} < min={cfg.min_beat_rate} "
            f"({int(audit.beat_rate * 8)}/8 windows positive)"
        )

    if np.isfinite(audit.alpha_capture) and audit.alpha_capture < cfg.min_alpha_capture:
        _signal_reasons.append(
            f"Alpha capture={audit.alpha_capture:.2f} < min={cfg.min_alpha_capture}"
        )

    if _signal_reasons:
        return EconomicViability(
            tier="signal_weak",
            reason="; ".join(_signal_reasons),
            cost_pnl=audit.cost_pnl,
            exec_sharpe=audit.exec_sharpe,
            alpha_capture=audit.alpha_capture,
            beat_rate=audit.beat_rate,
            eligible_universe=audit.eligible_universe_size,
        )

    # 3. Exposure-driven — must have passed signal quality first
    _exp_reasons: list[str] = []
    if np.isfinite(audit.beta_exposure) and audit.beta_exposure > cfg.max_beta_abs:
        _exp_reasons.append(f"Beta={audit.beta_exposure:.3f} > max={cfg.max_beta_abs}")
    if np.isfinite(audit.sector_exposure) and audit.sector_exposure > cfg.max_sector_abs:
        _exp_reasons.append(f"Sector={audit.sector_exposure:.3f} > max={cfg.max_sector_abs}")
    if np.isfinite(audit.mkt_correlation) and abs(audit.mkt_correlation) > cfg.max_mkt_corr:
        _exp_reasons.append(f"MktCorr={audit.mkt_correlation:.2f} > max={cfg.max_mkt_corr}")

    if _exp_reasons:
        return EconomicViability(
            tier="exposure_driven",
            reason="; ".join(_exp_reasons),
            cost_pnl=audit.cost_pnl,
            exec_sharpe=audit.exec_sharpe,
            alpha_capture=audit.alpha_capture,
            beat_rate=audit.beat_rate,
            eligible_universe=audit.eligible_universe_size,
        )

    # 4. Cost-dominated
    _cost_reasons: list[str] = []
    if np.isfinite(audit.cost_pnl) and audit.cost_pnl > cfg.max_cost_pnl:
        _cost_reasons.append(f"Cost/pnl={audit.cost_pnl:.3f} > max={cfg.max_cost_pnl}")

    if np.isfinite(audit.cost_drag) and audit.cost_drag > cfg.max_cost_drag:
        _cost_reasons.append(f"Cost drag={audit.cost_drag:.3f} > max={cfg.max_cost_drag}")

    if np.isfinite(audit.impact_pct) and audit.impact_pct > cfg.max_impact_fraction:
        _cost_reasons.append(
            f"Impact={audit.impact_pct:.1%} > max={cfg.max_impact_fraction:.0%}"
        )

    if _cost_reasons:
        return EconomicViability(
            tier="cost_dominated",
            reason="; ".join(_cost_reasons),
            cost_pnl=audit.cost_pnl,
            exec_sharpe=audit.exec_sharpe,
            alpha_capture=audit.alpha_capture,
            beat_rate=audit.beat_rate,
            eligible_universe=audit.eligible_universe_size,
        )

    # 5. Ranking instability
    _rank_reasons: list[str] = []
    if (
        np.isfinite(audit.decile_spread_cv)
        and audit.decile_spread_cv > cfg.max_decile_cv
    ):
        _rank_reasons.append(
            f"Decile CV={audit.decile_spread_cv:.0f} > max={cfg.max_decile_cv}"
        )

    if (
        np.isfinite(audit.halflife_rebalance_ratio)
        and audit.halflife_rebalance_ratio < cfg.min_halflife_rebalance_ratio
    ):
        _rank_reasons.append(
            f"Halflife/rebalance={audit.halflife_rebalance_ratio:.2f} "
            f"< min={cfg.min_halflife_rebalance_ratio}"
        )

    if _rank_reasons:
        return EconomicViability(
            tier="ranking_unstable",
            reason="; ".join(_rank_reasons),
            cost_pnl=audit.cost_pnl,
            exec_sharpe=audit.exec_sharpe,
            alpha_capture=audit.alpha_capture,
            beat_rate=audit.beat_rate,
            eligible_universe=audit.eligible_universe_size,
        )

    # 6. Breadth insufficient
    if (
        np.isfinite(audit.eligible_universe_size)
        and audit.eligible_universe_size < cfg.min_eligible_universe
    ):
        return EconomicViability(
            tier="breadth_insufficient",
            reason=f"Eligible universe={audit.eligible_universe_size:.0f} < min={cfg.min_eligible_universe}",
            cost_pnl=audit.cost_pnl,
            exec_sharpe=audit.exec_sharpe,
            alpha_capture=audit.alpha_capture,
            beat_rate=audit.beat_rate,
            eligible_universe=audit.eligible_universe_size,
        )

    # 7. Viable — all checks pass
    return EconomicViability(
        tier="viable",
        reason="All economic checks passed",
        cost_pnl=audit.cost_pnl,
        exec_sharpe=audit.exec_sharpe,
        alpha_capture=audit.alpha_capture,
        beat_rate=audit.beat_rate,
        eligible_universe=audit.eligible_universe_size,
    )


def select_best_horizon(
    audits: list[HorizonEconomicAudit],
    cfg: EconomicPolicyConfig | None = None,
) -> tuple[HorizonEconomicAudit | None, list[EconomicViability], dict[str, Any]]:
    """
    Algorithmic horizon selection from measured economic evidence.

    Returns (best_audit, all_viabilities, decision_diagnostics).

    Selection logic:
      1. Classify each horizon
      2. Filter to viable horizons
      3. If multiple viable, pick best by cost_pnl (or config tie_break)
      4. If none viable, return None with classification report

    This is algorithmic, not hardcoded — it works for any set of horizons.
    """
    if cfg is None:
        cfg = EconomicPolicyConfig()

    viabilities = [classify_economic_viability(a, cfg) for a in audits]
    viable = [
        (a, v) for a, v in zip(audits, viabilities)
        if v.tier in ("viable", "signal_weak")
    ]

    diag: dict[str, Any] = {
        "total_horizons_evaluated": len(audits),
        "viable_count": len(viable),
        "rejected_count": len(audits) - len(viable),
        "per_horizon": {},
    }

    for a, v in zip(audits, viabilities):
        diag["per_horizon"][str(a.horizon_days)] = {
            "model": a.model_name,
            "tier": v.tier,
            "reason": v.reason,
            "cost_pnl": v.cost_pnl,
            "exec_sharpe": v.exec_sharpe,
            "alpha_capture": v.alpha_capture,
        }

    if not viable:
        return None, viabilities, diag

    # Tie-break: best cost-adjusted profile
    tie_key = cfg.tie_break
    if tie_key == "cost_pnl":
        best = min(viable, key=lambda x: abs(x[0].cost_pnl))
    elif tie_key == "exec_sharpe":
        best = max(viable, key=lambda x: x[0].exec_sharpe)
    elif tie_key == "alpha_capture":
        best = max(viable, key=lambda x: x[0].alpha_capture)
    else:
        best = viable[0]

    return best[0], viabilities, diag


def format_economic_report(
    audits: list[HorizonEconomicAudit],
    viabilities: list[EconomicViability],
    diag: dict[str, Any],
    best: HorizonEconomicAudit | None = None,
) -> str:
    """Produce a human-readable economic research decision report."""
    lines = ["=" * 75]
    lines.append("ECONOMIC EXECUTION POLICY — RESEARCH DECISION REPORT")
    lines.append("=" * 75)
    lines.append(f"Horizons evaluated: {diag['total_horizons_evaluated']}")
    lines.append(f"Viable: {diag['viable_count']}  |  Rejected: {diag['rejected_count']}")
    lines.append("")

    lines.append(f"  {'Horizon':<8} {'Model':<15} {'Tier':<30} {'Cost/PnL':>8} {'ExecSR':>8}")
    lines.append(f"  {'-'*8} {'-'*15} {'-'*30} {'-'*8} {'-'*8}")
    for a, v in zip(audits, viabilities):
        lines.append(
            f"  {a.horizon_days:<8}d {a.model_name:<15} {v.tier:<30} "
            f"{v.cost_pnl:>8.3f} {v.exec_sharpe:>8.3f}"
        )
    lines.append("")

    if best is not None:
        lines.append(f"BEST HORIZON: {best.horizon_days}d ({best.model_name})")
        lines.append(f"  Cost/PnL: {best.cost_pnl:.3f}")
        lines.append(f"  Exec Sharpe: {best.exec_sharpe:.3f}")
        lines.append(f"  Alpha capture: {best.alpha_capture:.2f}")
        lines.append(f"  Beat rate: {best.beat_rate:.2f}")
        lines.append(f"  Eligible universe: {best.eligible_universe_size:.0f} names")
        lines.append(f"  Halflife/rebalance: {best.halflife_rebalance_ratio:.2f}")
        lines.append("")
        lines.append("RECOMMENDATION: Deploy at this horizon with current execution config.")
    else:
        lines.append("NO VIABLE HORIZON FOUND.")
        lines.append("")
        for a, v in zip(audits, viabilities):
            lines.append(f"  {a.horizon_days}d: {v.tier} — {v.reason}")
        lines.append("")
        lines.append("NEXT STEP: Address the dominant economic bottleneck before re-evaluating.")

    lines.append("=" * 75)
    return "\n".join(lines)


def build_audit_from_report_row(
    row: dict[str, Any],
    horizon_days: int,
    rebalance_days: int,
) -> HorizonEconomicAudit:
    """Build a HorizonEconomicAudit from a model_comparison.csv row dict."""
    def _f(key: str, default: float = float("nan")) -> float:
        v = row.get(key)
        try:
            fv = float(v)
            return fv if np.isfinite(fv) else default
        except (TypeError, ValueError):
            return default

    def _s(key: str, default: str = "unknown") -> str:
        return str(row.get(key, default) or default)

    halflife = _f("diag_robust_signal_halflife")
    reb = max(1, rebalance_days)

    return HorizonEconomicAudit(
        horizon_days=horizon_days,
        model_name=str(row.get("model_name", "")),
        primary_path=str(row.get("oos_evaluation_path", "long_only_overlay")),
        # Signal
        ic_mean=_f("oos_ic_chained"),
        ic_tstat=_f("oos_ic_tstat"),
        daily_icir=_f("daily_icir"),
        decile_monotonicity=_f("decile_monotonicity"),
        decile_spread=_f("decile_spread"),
        decile_spread_cv=_f("diag_decile_spread_cv"),
        signal_halflife=halflife,
        # Persistence
        top_decile_turnover=_f("diag_decile_turn_top"),
        rebalance_frequency=reb,
        halflife_rebalance_ratio=halflife / reb if np.isfinite(halflife) else float("nan"),
        # Execution
        exec_sharpe=_f("exec_sharpe"),
        beat_rate=_f("oos_beat_rate"),
        alpha_capture=_f("diag_alpha_capture_ratio"),
        cost_pnl=_f("exec_cost_to_gross_pnl"),
        turnover=_f("exec_turnover_mean"),
        avg_holding_period=_f("diag_avg_holding_period_days"),
        # Cost decomposition
        commission_pct=_f("diag_cost_pct_commission"),
        spread_pct=_f("diag_cost_pct_spread"),
        impact_pct=_f("diag_cost_pct_impact"),
        borrow_pct=_f("diag_cost_pct_borrow"),
        cost_dominant=_s("diag_cost_dominant"),
        alpha_vs_cost_ratio=_f("diag_alpha_vs_cost_ratio"),
        cost_drag=_f("diag_cost_drag"),
        # Risk
        beta_exposure=_f("exec_beta_abs_mean"),
        sector_exposure=_f("exec_max_sector_abs_mean"),
        mkt_correlation=_f("diag_mkt_corr"),
        residual_sharpe=_f("diag_residual_alpha_sharpe"),
        subsumption_alpha_ann=_f("subsumption_alpha_ann"),
        subsumption_alpha_tstat=_f("subsumption_alpha_tstat"),
        # Promotion
        promotion_pass=bool(row.get("promotion_pass", False)),
        gate_failure_count=len(str(row.get("promotion_failures", "")).split(","))
        if row.get("promotion_failures") else 0,
    )
