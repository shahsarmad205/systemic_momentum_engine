from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


# ── P28: Run regime classification ────────────────────────────────────────────

class SweepMode(str, enum.Enum):
    """Classifies the purpose of a horizon contract build."""
    SINGLE_PRODUCTION = "single_production"
    MULTI_HORIZON_SWEEP = "multi_horizon_sweep"
    CROSS_HORIZON_DIAGNOSTIC = "cross_horizon_diagnostic"
    INVALID_CONFLICT = "invalid_conflict"


class RebalanceMode(str, enum.Enum):
    """P32: How rebalance frequency relates to production horizon."""
    MATCH_HORIZON = "match_horizon"       # rebalance = production_horizon
    FIXED = "fixed"                       # rebalance from config, independent
    OVERLAP = "overlap"                   # rebalance < holding, >1 cycle per hold
    HALFLIFE_AWARE = "halflife_aware"     # rebalance ≤ halflife, auto-calibrated


@dataclass(frozen=True)
class RebalancePolicy:
    """
    P32: Explicit rebalance governance — no longer forced to match horizon.

    match_horizon  : rebalance = production_horizon_days (legacy default)
    fixed          : rebalance = frequency_days from config
    overlap        : rebalance < holding_period, explicit cycles per hold
    halflife_aware : rebalance ≤ floor(halflife), calibrated at runtime
    """
    mode: RebalanceMode = RebalanceMode.MATCH_HORIZON
    frequency_days: int | None = None        # for fixed/overlap
    max_turnover_budget: float = 1.0         # max daily turnover (1.0 = unconstrained)
    min_score_persistence_at_rebalance: float = 0.30  # require 30% ranking survival
    cost_budget: float = 0.50                # max cost/pnl ratio

    def resolve_rebalance(self, production_horizon: int, holding_period: int,
                          halflife_days: float = float("nan")) -> int:
        """Resolve the concrete rebalance frequency from policy and constraints."""
        if self.mode == RebalanceMode.FIXED and self.frequency_days is not None:
            return max(1, int(self.frequency_days))
        if self.mode == RebalanceMode.OVERLAP and self.frequency_days is not None:
            return max(1, min(int(self.frequency_days), holding_period - 1))
        if self.mode == RebalanceMode.HALFLIFE_AWARE:
            if np.isfinite(halflife_days) and halflife_days > 0:
                # Rebalance at floor(halflife) to ensure most rankings survive
                reb = max(1, int(halflife_days))
                # But never exceed holding period
                return min(reb, max(1, holding_period))
            # Fall back to match_horizon if halflife unavailable
        return max(1, int(production_horizon))


@dataclass(frozen=True)
class HorizonConfig:
    """Canonical run-level horizon contract.

    P24: production_horizon_days is the AUTHORITATIVE source.  All other fields
    default to it unless explicitly overridden in horizon_config.  The old
    behaviour where ic_evaluation_horizon silently overrode production_horizon
    is removed — mismatches now raise HorizonConfigurationError.

    CLI ``--horizon`` overrides production_horizon_days directly.
    """

    production_horizon_days: int
    target_horizon_days: int
    holding_period_days: int
    rebalance_frequency_days: int
    ic_evaluation_horizon: int
    execution_tau_days: float | None


class HorizonConfigurationError(RuntimeError):
    """Raised when horizon fields irreconcilably conflict."""

    pass


@dataclass(frozen=True)
class HorizonContract:
    config: HorizonConfig
    source_map: dict[str, str]
    legacy_values: dict[str, Any]
    warnings: tuple[str, ...]
    replaced_fields: tuple[str, ...]
    sweep_mode: SweepMode = SweepMode.SINGLE_PRODUCTION
    resolved_horizon_regime: SweepMode = SweepMode.SINGLE_PRODUCTION

    def validate_alignment(self) -> list[str]:
        """
        P15: Fail-fast checks that prevent silent horizon misalignment.

        Returns a list of failure messages.  An empty list means all checks pass.
        Callers should raise RuntimeError if any failures are returned when
        running in production (non-research) mode.
        """
        failures: list[str] = []
        c = self.config
        p, t, h, r, i = c.production_horizon_days, c.target_horizon_days, c.holding_period_days, c.rebalance_frequency_days, c.ic_evaluation_horizon

        # 0. Production horizon must be positive
        if p < 1:
            failures.append(f"production_horizon_days={p} — must be >= 1 day.")

        # 1. Holding period must match target horizon unless explicitly split
        if h != t:
            _src_h = self.source_map.get("holding_period_days", "unknown")
            if _src_h == "mirror_target":
                pass
            else:
                failures.append(
                    f"Holding period ({h}d) differs from target horizon ({t}d). "
                    f"Source: {_src_h}. This is valid if intentional (multi-horizon)."
                )

        # 2. IC evaluation horizon must match target unless explicitly research mode
        if i != t:
            _src_i = self.source_map.get("ic_evaluation_horizon", "unknown")
            failures.append(
                f"IC evaluation horizon ({i}d) differs from target horizon ({t}d). "
                f"Source: {_src_i}. Run with multi_horizon_admission=true if intentional."
            )

        # 3. Rebalance frequency must not exceed holding period
        if r > h:
            failures.append(
                f"Rebalance frequency ({r}d) exceeds holding period ({h}d). "
                "Positions would be churned before reaching target hold."
            )

        # 4. All fields must be positive
        for name, val in [("target", t), ("holding", h), ("rebalance", r), ("ic_eval", i)]:
            if val < 1:
                failures.append(f"Horizon field '{name}' is {val} — must be >= 1 day.")

        # 5. Execution tau, if set, must be reasonable
        tau = c.execution_tau_days
        if tau is not None and tau <= 0:
            failures.append(f"execution_tau_days={tau} must be null or positive.")

        return failures

    def to_dict(self) -> dict[str, Any]:
        return {
            "horizon_config": asdict(self.config),
            "source_map": dict(self.source_map),
            "legacy_values": dict(self.legacy_values),
            "warnings": list(self.warnings),
            "replaced_fields": list(self.replaced_fields),
            "sweep_mode": self.sweep_mode.value,
            "resolved_horizon_regime": self.resolved_horizon_regime.value,
        }

    def audit_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        canonical = asdict(self.config)
        for legacy_field, canonical_field in _FIELD_MAP.items():
            rows.append(
                {
                    "legacy_field": legacy_field,
                    "legacy_value": self.legacy_values.get(legacy_field),
                    "canonical_field": canonical_field,
                    "canonical_value": canonical.get(canonical_field),
                    "source": self.source_map.get(canonical_field, ""),
                    "status": "replaced" if legacy_field in self.replaced_fields else "audited",
                }
            )
        return rows


# ── P28: HorizonSweepContract — parent for multi-horizon dispatch ─────────────

@dataclass(frozen=True)
class HorizonSweepContract:
    """Parent contract for multi-horizon sweep runs.

    When production_horizons=[5,10,20,63], the parent sweep contract
    owns the sweep list.  Each child HorizonContract is built for the
    active_horizon.  The child does NOT warn about production_horizons
    because the list is the parent's sweep list, not a configuration
    conflict.

    sweep_mode classifies the run:
      SINGLE_PRODUCTION   — one horizon, strict alignment
      MULTI_HORIZON_SWEEP — parent dispatches children, sweep is intentional
      CROSS_HORIZON_DIAGNOSTIC — diagnostic mode, mismatches permitted
      INVALID_CONFLICT    — mismatch detected, run rejected
    """

    sweep_horizons: tuple[int, ...]
    active_horizon: int
    sweep_mode: SweepMode
    children: dict[int, "HorizonContract"] = field(default_factory=dict)
    parent_sources: dict[str, str] = field(default_factory=dict)

    @property
    def active_child(self) -> "HorizonContract | None":
        return self.children.get(self.active_horizon)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sweep_horizons": list(self.sweep_horizons),
            "active_horizon": self.active_horizon,
            "sweep_mode": self.sweep_mode.value,
            "parent_sources": dict(self.parent_sources),
            "child_count": len(self.children),
        }


def build_horizon_contracts_for_sweep(
    cfg: dict[str, Any],
    sweep_horizons: list[int],
    *,
    cli_horizon: int | None = None,
    halflife_days: float = float("nan"),
) -> HorizonSweepContract:
    """
    P28: Build a parent sweep contract with one child per horizon.

    Each child is built via build_horizon_contract with the active_horizon
    set to the sweep horizon.  The parent sweep_horizons list does NOT
    create child-level warnings — it is intentional sweep behavior.
    """
    children: dict[int, HorizonContract] = {}
    parent_sources = {"sweep_mode": "multi_horizon_sweep",
                       "sweep_horizons": str(sweep_horizons)}

    for h in sweep_horizons:
        child = build_horizon_contract(
            cfg,
            active_horizon=h,   # P29: sweep-assigned, not CLI
            cli_horizon=cli_horizon,
            sweep_mode=SweepMode.MULTI_HORIZON_SWEEP,
            halflife_days=halflife_days,
        )
        children[h] = child

    active = cli_horizon if cli_horizon is not None else sweep_horizons[0]

    return HorizonSweepContract(
        sweep_horizons=tuple(sweep_horizons),
        active_horizon=active,
        sweep_mode=SweepMode.MULTI_HORIZON_SWEEP,
        children=children,
        parent_sources=parent_sources,
    )


_FIELD_MAP: dict[str, str] = {
    "model_selection.lookahead_horizon_days": "target_horizon_days",
    "model_selection.production_horizons": "target_horizon_days",
    "model_selection.nested_validation.search.candidate_horizons": "target_horizon_days",
    # P29: alpha_research.production_horizon maps to production_horizon_days
    # (was incorrectly mapped to ic_evaluation_horizon in P24)
    "model_selection.alpha_research.production_horizon": "production_horizon_days",
    "backtest.lookahead_horizon_days": "target_horizon_days",
    "backtest.holding_period_days": "holding_period_days",
    "backtest.rebalance_every_trading_days": "rebalance_frequency_days",
    "backtest.optimization_config.execution.horizon_days": "target_horizon_days",
    "signals.weights.ic_horizon_days": "ic_evaluation_horizon",
}


def _as_int(value: Any, default: int) -> int:
    try:
        if value is None or value == "":
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _as_float_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_rebalance_mode(raw: str) -> RebalanceMode:
    """Parse a rebalance mode string into RebalanceMode enum, with validation."""
    try:
        return RebalanceMode(raw.strip().lower())
    except ValueError:
        raise HorizonConfigurationError(
            f"Unknown rebalance_policy.mode: '{raw}'. "
            f"Valid modes: {[m.value for m in RebalanceMode]}."
        )


def _get_nested(raw: dict[str, Any], path: str) -> Any:
    cur: Any = raw
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def build_horizon_contract(
    cfg: dict[str, Any],
    *,
    cli_horizon: int | None = None,
    active_horizon: int | None = None,
    sweep_mode: SweepMode = SweepMode.SINGLE_PRODUCTION,
    halflife_days: float = float("nan"),
) -> HorizonContract:
    """Build the canonical horizon contract for a single model-selection run.

    P28/P29: sweep_mode controls how conflicts are classified.
    active_horizon is the horizon assigned by a parent sweep (not by CLI).
    cli_horizon is the actual CLI --horizon override.

    In SINGLE_PRODUCTION, mismatched fields raise HorizonConfigurationError.
    In MULTI_HORIZON_SWEEP, active_horizon sets production_horizon and the
    production_horizons list is never warned about.
    In CROSS_HORIZON_DIAGNOSTIC, ic_eval ≠ production is permitted (flagged).

    P32: halflife_days is optional evidence for RebalancePolicy.HALFLIFE_AWARE.
    When mode=halflife_aware and halflife_days is provided, the rebalance
    frequency is auto-calibrated to ≤ floor(halflife_days).  When halflife
    evidence is missing, the policy's halflife_fallback_mode controls whether
    the engine fails closed or falls back to production_horizon_days.
    """

    cfg = cfg or {}
    hc_raw = cfg.get("horizon_config", {}) or {}
    ms = cfg.get("model_selection", {}) or {}
    bt = cfg.get("backtest", {}) or {}
    bt_exec = ((bt.get("optimization_config", {}) or {}).get("execution", {}) or {}) if isinstance(bt, dict) else {}

    legacy_values = {field: _get_nested(cfg, field) for field in _FIELD_MAP}
    warnings: list[str] = []

    # ── P32: Parse rebalance_policy from YAML into a live dataclass ────────────
    _reb_raw = (hc_raw.get("rebalance_policy", {}) or {})
    _reb_mode_str = str(_reb_raw.get("mode", "match_horizon")).strip().lower()
    _reb_mode = _parse_rebalance_mode(_reb_mode_str)
    _reb_freq = _as_int_or_none(_reb_raw.get("frequency_days"))
    _reb_halflife_fallback = str(_reb_raw.get("halflife_fallback_mode", "fail_closed")).strip().lower()
    _reb = RebalancePolicy(
        mode=_reb_mode,
        frequency_days=_reb_freq,
        max_turnover_budget=float(_reb_raw.get("max_turnover_budget", 1.0) or 1.0),
        min_score_persistence_at_rebalance=float(_reb_raw.get("min_score_persistence_at_rebalance", 0.30) or 0.30),
        cost_budget=float(_reb_raw.get("cost_budget", 0.50) or 0.50),
    )
    rebalance_policy_source = (
        "horizon_config.rebalance_policy"
        if _reb_raw
        else "rebalance_policy_defaults"
    )

    # ── P29: production_horizon_days is AUTHORITATIVE ──────────────────────────
    # Resolution precedence:
    #   1. CLI --horizon (user override)
    #   2. active_horizon (parent sweep assignment, not CLI)
    #   3. alpha_research.production_horizon (config)
    #   4. lookahead_horizon_days (legacy fallback)
    alpha_raw = (ms.get("alpha_research", {}) or {}) if isinstance(ms, dict) else {}
    _default_prod = _as_int(ms.get("lookahead_horizon_days", bt.get("lookahead_horizon_days", 10)), 10)
    _prod_from_alpha = _as_int(alpha_raw.get("production_horizon", _default_prod), _default_prod)

    if cli_horizon is not None:
        production = _as_int(cli_horizon, _prod_from_alpha)
        prod_source = "cli.--horizon"
    elif active_horizon is not None:
        # P29: Sweep-assigned horizon, not CLI
        production = _as_int(active_horizon, _prod_from_alpha)
        prod_source = "sweep.active_horizon"
    else:
        production = _prod_from_alpha
        prod_source = "model_selection.alpha_research.production_horizon"

    # ── P33 + Sweep Governance: Dimension resolution ───────────────────────────
    # In sweep mode (MULTI_HORIZON_SWEEP), the sweep horizon collapses ALL
    # dimensions unless horizon_config.sweep_dimension_separation is explicitly
    # true.  This prevents 5d/20d/63d children from inheriting stale 10d
    # config values and writing into the wrong output namespace.
    #
    # In single-production mode, explicit horizon_config fields take precedence
    # over inherited defaults (P33 dimension separation).
    _sweep_collapse = (
        sweep_mode == SweepMode.MULTI_HORIZON_SWEEP
        and not bool(hc_raw.get("sweep_dimension_separation", False))
    )

    # ── target_horizon ───────────────────────────────────────────────────────────
    if _sweep_collapse:
        target = int(production)
        target_source = "sweep_horizon_collapsed"
    elif "target_horizon_days" in hc_raw:
        target = _as_int(hc_raw.get("target_horizon_days"), production)
        target_source = "horizon_config.target_horizon_days"
        if target != production:
            warnings.append(
                f"target_horizon_days={target}d differs from production_horizon_days={production}d. "
                "Target horizon resolved independently per horizon_config."
            )
    else:
        target = int(production)
        target_source = "production_horizon_days"

    # ── holding_period ──────────────────────────────────────────────────────────
    if _sweep_collapse:
        holding = int(production)
        holding_source = "sweep_horizon_collapsed"
    elif "holding_period_days" in hc_raw:
        holding = _as_int(hc_raw.get("holding_period_days"), production)
        holding_source = "horizon_config.holding_period_days"
        if holding != production:
            warnings.append(
                f"holding_period_days={holding}d differs from production_horizon_days={production}d. "
                "Holding period resolved independently per horizon_config."
            )
    else:
        holding = int(production)
        holding_source = "production_horizon_days"

    # ── P32: rebalance — delegated to RebalancePolicy.resolve_rebalance() ──────
    _legacy_rebalance = _as_int(bt.get("rebalance_every_trading_days", production), production)
    _legacy_rebalance = _as_int(hc_raw.get("rebalance_frequency_days"), _legacy_rebalance)
    _legacy_reb_source = (
        "horizon_config.rebalance_frequency_days"
        if "rebalance_frequency_days" in hc_raw
        else "backtest.rebalance_every_trading_days"
    )

    if _sweep_collapse:
        # Sweep mode: rebalance collapses to production horizon
        rebalance = int(production)
        rebalance_source = "sweep_horizon_collapsed"
    elif _reb_mode == RebalanceMode.MATCH_HORIZON and _reb_raw:
        # Explicit rebalance_policy with match_horizon → production is authoritative
        rebalance = int(production)
        rebalance_source = "production_horizon_days"
    elif _reb_mode == RebalanceMode.FIXED and _reb_freq is not None:
        rebalance = _reb.resolve_rebalance(production, holding)
        rebalance_source = "horizon_config.rebalance_policy.fixed"
    elif _reb_mode == RebalanceMode.OVERLAP and _reb_freq is not None:
        rebalance = _reb.resolve_rebalance(production, holding)
        rebalance_source = "horizon_config.rebalance_policy.overlap"
    elif _reb_mode == RebalanceMode.HALFLIFE_AWARE:
        if np.isfinite(halflife_days) and halflife_days > 0:
            rebalance = _reb.resolve_rebalance(production, holding, halflife_days=halflife_days)
            rebalance_source = "horizon_config.rebalance_policy.halflife_aware"
        elif _reb_halflife_fallback == "match_production":
            rebalance = max(1, int(production))
            rebalance_source = "rebalance_policy.halflife_aware.fallback.match_production"
            warnings.append(
                f"rebalance_policy.mode=halflife_aware but no halflife_days evidence available; "
                f"fell back to production_horizon_days={production}d per halflife_fallback_mode=match_production."
            )
        elif _reb_halflife_fallback == "legacy_config":
            rebalance = _legacy_rebalance
            rebalance_source = "rebalance_policy.halflife_aware.fallback.legacy_config"
            warnings.append(
                f"rebalance_policy.mode=halflife_aware but no halflife_days evidence available; "
                f"fell back to {_legacy_reb_source}={rebalance}d per halflife_fallback_mode=legacy_config."
            )
        else:
            raise HorizonConfigurationError(
                f"rebalance_policy.mode=halflife_aware requires halflife_days evidence but none was provided. "
                f"Set halflife_fallback_mode='match_production' or 'legacy_config' to allow fallback, "
                f"or provide halflife evidence at build time."
            )
    else:
        # match_horizon without explicit config → use legacy config
        rebalance = _legacy_rebalance
        rebalance_source = _legacy_reb_source

    # ── ic_evaluation_horizon ───────────────────────────────────────────────────
    if _sweep_collapse:
        ic_eval = int(production)
        ic_source = "sweep_horizon_collapsed"
    elif "ic_evaluation_horizon" in hc_raw:
        ic_eval = _as_int(hc_raw.get("ic_evaluation_horizon"), production)
        ic_source = "horizon_config.ic_evaluation_horizon"
        if ic_eval != production:
            _allow_cross = bool(hc_raw.get("allow_cross_horizon_evaluation", False))
            if not _allow_cross:
                raise HorizonConfigurationError(
                    f"ic_evaluation_horizon ({ic_eval}d) ≠ production_horizon ({production}d). "
                    f"Set horizon_config.allow_cross_horizon_evaluation: true to permit this "
                    f"or remove ic_evaluation_horizon from horizon_config to inherit production_horizon."
                )
            warnings.append(
                f"ic_evaluation_horizon={ic_eval}d differs from production_horizon={production}d. "
                "Cross-horizon evaluation is active. Feature admission will evaluate at "
                f"{ic_eval}d while models train at {target}d."
            )
    else:
        ic_eval = int(production)
        ic_source = "production_horizon_days"

    # ── execution_tau ──────────────────────────────────────────────────────────
    tau = _as_float_or_none(hc_raw.get("execution_tau_days", bt_exec.get("tau_exec")))
    tau_source = (
        "horizon_config.execution_tau_days"
        if "execution_tau_days" in hc_raw
        else "backtest.optimization_config.execution.tau_exec"
    )

    # ── P24: Legacy field audit — warn about stale values ─────────────────────
    if holding != target and holding != production:
        warnings.append(
            f"horizon_config mismatch: holding_period_days={holding} differs from "
            f"target_horizon_days={target}. This is allowed only as an explicit override."
        )
    for legacy_field, canonical_field in _FIELD_MAP.items():
        legacy = legacy_values.get(legacy_field)
        if legacy is None:
            continue
        canonical_value = getattr(
            HorizonConfig(production, target, holding, rebalance, ic_eval, tau),
            canonical_field,
        )
        comparable = legacy
        if isinstance(legacy, list):
            comparable = legacy[0] if len(legacy) == 1 else legacy
        try:
            legacy_num: Any = float(comparable) if comparable is not None and comparable != "" else comparable
            canon_num: Any = float(canonical_value) if canonical_value is not None else canonical_value
        except (TypeError, ValueError):
            legacy_num = comparable
            canon_num = canonical_value
        if isinstance(legacy_num, float) and isinstance(canon_num, float):
            differs = abs(legacy_num - canon_num) > 1e-12
        else:
            differs = legacy_num != canon_num
        if differs:
            # P28: In sweep mode, production_horizons is the parent's sweep
            # list — NOT a conflict with the child's active horizon.
            if sweep_mode == SweepMode.MULTI_HORIZON_SWEEP and legacy_field == "model_selection.production_horizons":
                continue
            warnings.append(
                f"{legacy_field}={legacy!r} differs from canonical "
                f"{canonical_field}={canonical_value!r}; canonical value is used for model selection."
            )

    # P29: Determine resolved_horizon_regime
    _allow_cross = bool(hc_raw.get("allow_cross_horizon_evaluation", False))
    if sweep_mode == SweepMode.MULTI_HORIZON_SWEEP:
        resolved_regime = SweepMode.MULTI_HORIZON_SWEEP
    elif ic_eval != production and _allow_cross:
        resolved_regime = SweepMode.CROSS_HORIZON_DIAGNOSTIC
    elif ic_eval != production:
        resolved_regime = SweepMode.INVALID_CONFLICT
    else:
        resolved_regime = SweepMode.SINGLE_PRODUCTION

    contract = HorizonContract(
        config=HorizonConfig(
            production_horizon_days=max(1, int(production)),
            target_horizon_days=max(1, int(target)),
            holding_period_days=max(1, int(holding)),
            rebalance_frequency_days=max(1, int(rebalance)),
            ic_evaluation_horizon=max(1, int(ic_eval)),
            execution_tau_days=tau,
        ),
        source_map={
            "production_horizon_days": prod_source,
            "target_horizon_days": target_source,
            "holding_period_days": holding_source,
            "rebalance_frequency_days": rebalance_source,
            "ic_evaluation_horizon": ic_source,
            "execution_tau_days": tau_source,
            # P32: RebalancePolicy audit metadata
            "rebalance_policy_mode": _reb_mode.value,
            "rebalance_policy_source": rebalance_policy_source,
            "rebalance_policy_frequency_days": str(_reb_freq) if _reb_freq is not None else "none",
            "rebalance_policy_halflife_days_supplied": str(round(halflife_days, 2)) if np.isfinite(halflife_days) else "none",
            # Sweep governance audit fields
            "sweep_horizon": str(int(production)) if sweep_mode == SweepMode.MULTI_HORIZON_SWEEP else "none",
            "dimension_override_policy": "sweep_collapsed" if _sweep_collapse else "independent",
        },
        legacy_values=legacy_values,
        warnings=tuple(dict.fromkeys(warnings)),
        replaced_fields=tuple(_FIELD_MAP.keys()),
        sweep_mode=sweep_mode,
        resolved_horizon_regime=resolved_regime,
    )

    # P15: Fail-fast horizon alignment validation
    # Raise on structural misalignment (not advisory warnings).
    _align_failures = contract.validate_alignment()
    if _align_failures:
        _fail_msg = (
            "HORIZON CONTRACT ALIGNMENT FAILURE — The following fields disagree:\n  "
            + "\n  ".join(_align_failures)
            + "\n\nFix backtest_config.yaml horizon_config section or pass matching --horizon."
        )
        # Rebalance > holding is a hard structural error
        _structural_errors = [
            f for f in _align_failures
            if "exceeds holding period" in f or "must be >= 1 day" in f
        ]
        if _structural_errors:
            raise RuntimeError(_fail_msg)
        # Non-structural mismatches are logged as warnings for research review
        import logging
        _logger = logging.getLogger(__name__)
        _logger.warning(_fail_msg)

    return contract


# ═══════════════════════════════════════════════════════════════════════════════
# P38: Research-Cache Fingerprint — Stable policy hash for cache invalidation
# ═══════════════════════════════════════════════════════════════════════════════
#
# Changing research-sensitive horizon/rebalance policy MUST invalidate downstream
# artifacts (prepared panels, targets, research state).  This function produces a
# short deterministic hash from the resolved HorizonContract dimensions and any
# active policy metadata.  It is ADDED to existing cache keys — it does NOT
# replace them — so unrelated config changes do not cause collateral invalidation.

import hashlib as _hashlib


def research_cache_fingerprint(
    *,
    production_horizon_days: int,
    target_horizon_days: int,
    holding_period_days: int,
    rebalance_frequency_days: int,
    ic_evaluation_horizon: int,
    execution_tau_days: float | None,
    rebalance_policy_mode: str = "",
    execution_aware_policy_version: int = 0,
    orientation_policy_version: int = 0,
    scoring_weights_policy_mode: str = "",
) -> str:
    """
    P38: Produce a short, stable fingerprint of research-sensitive config.

    Changing any input field invalidates downstream caches.  The fingerprint is
    designed to be appended to existing cache keys — it does NOT replace the
    content-based signature of the cached data itself.

    Fields included:
      - Every resolved dimension from HorizonConfig
      - RebalancePolicy mode (governs how rebalance is resolved)
      - ExecutionAwareHorizonPolicy version (governs selection rules)
      - OrientationPolicy version (governs score-direction aggregation)
      - Scoring weights policy mode (governs weight derivation)

    Fields intentionally EXCLUDED:
      - Timestamps, PIDs, hostnames (not stable)
      - Feature subsets (content-hashed separately via frame_fingerprint)
      - Cost model parameters (already in EvaluationConfig → validation caches)
      - Promotion gate thresholds (already in PromotionGateConfig → gate caches)
    """
    payload = {
        "production_horizon_days": int(production_horizon_days),
        "target_horizon_days": int(target_horizon_days),
        "holding_period_days": int(holding_period_days),
        "rebalance_frequency_days": int(rebalance_frequency_days),
        "ic_evaluation_horizon": int(ic_evaluation_horizon),
        "execution_tau_days": (
            round(float(execution_tau_days), 6)
            if execution_tau_days is not None
            else "auto"
        ),
        "rebalance_policy_mode": str(rebalance_policy_mode),
        "execution_aware_policy_version": int(execution_aware_policy_version),
        "orientation_policy_version": int(orientation_policy_version),
        "scoring_weights_policy_mode": str(scoring_weights_policy_mode),
    }
    encoded = repr(payload).encode("utf-8")
    return _hashlib.sha1(encoded).hexdigest()[:16]


def research_cache_fingerprint_from_config(cfg: dict[str, Any]) -> str:
    """
    P38: Compute research cache fingerprint from a full config dict.

    Convenience wrapper that extracts horizon contract, rebalance policy,
    and policy versions from the config dict without requiring callers to
    have pre-built HorizonContract objects.
    """
    from model_selection.horizon_contract import build_horizon_contract

    contract = build_horizon_contract(cfg)
    c = contract.config
    rebalance_mode = contract.source_map.get("rebalance_policy_mode", "")

    ms = cfg.get("model_selection", {}) or {}
    eaw = (ms.get("execution_aware_horizon_policy", {}) or {}) if isinstance(ms, dict) else {}
    ori = (ms.get("orientation_policy", {}) or {}) if isinstance(ms, dict) else {}

    return research_cache_fingerprint(
        production_horizon_days=c.production_horizon_days,
        target_horizon_days=c.target_horizon_days,
        holding_period_days=c.holding_period_days,
        rebalance_frequency_days=c.rebalance_frequency_days,
        ic_evaluation_horizon=c.ic_evaluation_horizon,
        execution_tau_days=c.execution_tau_days,
        rebalance_policy_mode=rebalance_mode,
        execution_aware_policy_version=int(eaw.get("policy_version", 0)),
        orientation_policy_version=int(ori.get("policy_version", 0)),
        scoring_weights_policy_mode=str(
            (eaw.get("scoring_weights_policy", {}) or {}).get("mode", "")
        ),
    )
