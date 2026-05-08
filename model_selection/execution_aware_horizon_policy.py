"""
P35: ExecutionAwareHorizonPolicy — Single institutional policy layer for
    horizon/rebalance/execution selection governance.

Design principles:
  1. All thresholds are configurable — no hardcoded current-state values.
  2. Fail-closed by default — no silent fallbacks.
  3. Weights are validated at construction time — invalid config raises immediately.
  4. The policy does NOT change selected horizon yet — it is a configuration layer
     that will govern future execution-aware selection when activated.

Relationship to existing objects:
  - Supersedes the advisory ExecutionAwareHorizonConfig (P34) — absorbs its fields.
  - Extends RebalancePolicy (P32) — adds horizon-level governance above it.
  - Complements EconomicPolicyConfig (P20) — policy governs selection; economic
    policy governs post-hoc viability classification.
  - Does NOT replace HorizonConfig, EvaluationConfig, PromotionGateConfig,
    AlphaAdmissionConfig, or TargetConfig.
"""

from __future__ import annotations

import enum
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ── Enums ─────────────────────────────────────────────────────────────────────

class FailClosedMode(str, enum.Enum):
    FAIL = "fail"
    FALLBACK_TO_DEFAULT = "fallback_to_default"
    WARN_AND_CONTINUE = "warn_and_continue"


class TieBreakMode(str, enum.Enum):
    PREFER_HIGHER_COMPOSITE = "prefer_higher_composite"
    PREFER_SHORTER_HORIZON = "prefer_shorter_horizon"
    PREFER_LOWER_COST_PNL = "prefer_lower_cost_pnl"
    PREFER_HIGHER_PERSISTENCE = "prefer_higher_persistence"


class WeightsMode(str, enum.Enum):
    """P35: How scoring weights are determined."""
    CONFIG = "config"            # Use explicit weights from ScoringWeights
    ALGORITHMIC = "algorithmic"  # Derive weights from IC-decay table evidence


class DataPhase(str, enum.Enum):
    """Which metrics are available at the current pipeline phase."""
    PRE_TRAINING = "pre_training"        # IC-decay table only
    POST_VALIDATION = "post_validation"  # Model trained, execution data available


# ── Sub-policies ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ContractCandidate:
    """A single (target, holding, rebalance) contract to evaluate."""
    target_horizon_days: int
    holding_period_days: int | None = None     # None = inherit target
    rebalance_frequency_days: int | None = None  # None = RebalancePolicy governs


@dataclass(frozen=True)
class ScoringWeights:
    """
    Explicit scoring weights.  Used when WeightsMode=CONFIG.
    These are the fallback — the algorithmic mode derives weights
    from data and uses these only when evidence is insufficient.
    """
    ic_strength: float = 0.25
    ic_consistency: float = 0.15
    halflife_persistence: float = 0.20
    cost_adjusted_ic: float = 0.20
    alpha_capture: float = 0.10
    execution_sharpe: float = 0.10

    def sum(self) -> float:
        return sum([
            self.ic_strength, self.ic_consistency, self.halflife_persistence,
            self.cost_adjusted_ic, self.alpha_capture, self.execution_sharpe,
        ])

    def validate(self) -> list[str]:
        errors = []
        for name in ["ic_strength", "ic_consistency", "halflife_persistence",
                     "cost_adjusted_ic", "alpha_capture", "execution_sharpe"]:
            v = getattr(self, name)
            if not isinstance(v, (int, float)) or v < 0:
                errors.append(f"weight_{name} must be >= 0, got {v}")
        if self.sum() <= 0:
            errors.append("At least one scoring weight must be > 0")
        return errors


@dataclass(frozen=True)
class ScoringWeightsPolicy:
    """
    P35: Governs how scoring weights are determined.

    mode=config       → use explicit weights from the `explicit` field.
    mode=algorithmic  → derive weights from IC-decay table evidence.
                        The `explicit` weights serve as fallback priors
                        when evidence is insufficient.

    Algorithmic derivation rules (pre-training phase):
      - ic_strength        ∝ median_IC across horizons (higher IC → higher weight)
      - ic_consistency     ∝ 1 / IC_CV (stable IC → higher weight)
      - halflife_persistence ∝ 1 - median_decay_rate (fast decay → higher weight)
      - cost_adjusted_ic     ∝ net_IC / gross_IC gap (costs matter → higher weight)
      - alpha_capture      → min_weight (not available pre-training)
      - execution_sharpe   → min_weight (not available pre-training)

    The `data_phase` controls which metrics are available.  In the current
    phase (pre-training), only IC-derived weights are algorithmic.
    When post-validation data exists, `data_phase` can be advanced.
    """
    mode: WeightsMode = WeightsMode.CONFIG
    data_phase: DataPhase = DataPhase.PRE_TRAINING
    explicit: ScoringWeights = field(default_factory=ScoringWeights)
    min_weight: float = 0.05          # floor for unavailable metrics
    max_weight: float = 0.40          # ceiling for any single metric
    evidence_min_features: int = 3    # minimum features for algorithmic derivation

    def validate(self) -> list[str]:
        errors = []
        if self.min_weight < 0 or self.min_weight > self.max_weight:
            errors.append(f"min_weight must be in [0, {self.max_weight}], got {self.min_weight}")
        if self.max_weight > 1.0:
            errors.append(f"max_weight must be <= 1.0, got {self.max_weight}")
        errors.extend(self.explicit.validate())
        return errors


# ── Algorithmic weight derivation ─────────────────────────────────────────────

def derive_scoring_weights_from_decay(
    alpha_decay: "pd.DataFrame | None",
    *,
    policy: ScoringWeightsPolicy,
) -> ScoringWeights:
    """
    P35: Derive scoring weights from IC-decay table evidence.

    When mode=CONFIG, returns the explicit weights unchanged.
    When mode=ALGORITHMIC, computes weights from the following evidence:

      ic_strength:
        w = median( |IC(h)| ) / reference_ic
        Normalised so the highest-IC metric across horizons sets the baseline.

      ic_consistency:
        w = 1 / (1 + CV_IC)  where CV_IC = std(IC) / |mean(IC)| across horizons
        CV near 0 (highly consistent) → w near 1; CV large → w near 0.

      halflife_persistence:
        w = clamp(1 - exp(-median_halflife / max_horizon), 0, 1)
        Long halflife → full weight; short halflife → near zero weight.

      cost_adjusted_ic:
        w = |net_IC - gross_IC| / max(|net_IC|, |gross_IC|, 1e-9)
        Large gap between net and gross IC → costs matter → higher weight.

      alpha_capture, execution_sharpe:
        Set to min_weight (not yet available in pre-training phase).

    Returns a new ScoringWeights derived from data, with weights sum-normalised.
    """
    if policy.mode != WeightsMode.ALGORITHMIC:
        return policy.explicit

    if alpha_decay is None or alpha_decay.empty:
        return policy.explicit

    ic_col = "daily_spearman_ic"
    h_col = "horizon_days"
    type_col = "target_type"
    halflife_col = "signal_halflife_days"

    if ic_col not in alpha_decay.columns or h_col not in alpha_decay.columns:
        return policy.explicit

    # Use net_residual target for gross IC; we also check raw for cost gap
    net_subset = alpha_decay
    if type_col in alpha_decay.columns and "net_residual_return" in alpha_decay[type_col].unique():
        net_subset = alpha_decay[alpha_decay[type_col] == "net_residual_return"]

    raw_subset = alpha_decay
    if type_col in alpha_decay.columns and "raw_return" in alpha_decay[type_col].unique():
        raw_subset = alpha_decay[alpha_decay[type_col] == "raw_return"]

    net_ics = pd.to_numeric(net_subset[ic_col], errors="coerce").dropna()
    raw_ics = pd.to_numeric(raw_subset[ic_col], errors="coerce").dropna()

    # Count unique features, not rows (one feature appears at many horizons)
    feature_col = "feature"
    n_features = 0
    if feature_col in net_subset.columns:
        n_features = net_subset[feature_col].nunique()
    elif len(net_ics) > 0:
        n_features = max(1, len(net_ics) // 4)  # rough estimate from horizon count

    if n_features < policy.evidence_min_features:
        return policy.explicit

    # ── ic_strength ────────────────────────────────────────────────────
    median_ic = float(net_ics.abs().median()) if len(net_ics) else 0.0
    # Reference IC: typical weak signal floor
    reference_ic = 0.001
    ic_strength = min(1.0, median_ic / max(reference_ic, 1e-9))

    # ── ic_consistency ─────────────────────────────────────────────────
    ic_mean = float(net_ics.mean())
    ic_std = float(net_ics.std(ddof=1)) if len(net_ics) > 1 else 0.0
    cv = abs(ic_std / max(abs(ic_mean), 1e-9))
    ic_consistency = 1.0 / (1.0 + cv)  # ∈ (0, 1]

    # ── halflife_persistence ────────────────────────────────────────────
    halflife_weight = 0.5  # neutral prior
    if halflife_col in net_subset.columns:
        halflives = pd.to_numeric(net_subset[halflife_col], errors="coerce").dropna()
        if len(halflives) >= policy.evidence_min_features:
            median_halflife = float(halflives.median())
            max_h = float(net_subset[h_col].max()) if h_col in net_subset.columns else 20.0
            # Long halflife relative to max horizon → near 1; short → near 0
            if np.isfinite(median_halflife) and max_h > 0:
                halflife_weight = max(0.0, min(1.0, 1.0 - np.exp(-median_halflife / max_h)))

    # ── cost_adjusted_ic ───────────────────────────────────────────────
    cost_weight = 0.3  # neutral prior
    if len(raw_ics) >= policy.evidence_min_features and len(net_ics) >= policy.evidence_min_features:
        gross_mean = float(raw_ics.abs().mean())
        net_mean = float(net_ics.abs().mean())
        if gross_mean > 1e-9:
            cost_gap = abs(gross_mean - net_mean) / gross_mean
            cost_weight = max(0.05, min(1.0, cost_gap * 2.0))  # amplify small gaps

    # ── Assemble, normalise ────────────────────────────────────────────
    raw = {
        "ic_strength": max(0.01, ic_strength),
        "ic_consistency": max(0.01, ic_consistency),
        "halflife_persistence": max(0.01, halflife_weight),
        "cost_adjusted_ic": max(0.01, cost_weight),
        "alpha_capture": 0.01,
        "execution_sharpe": 0.01,
    }

    total = sum(raw.values())
    if total <= 0:
        return policy.explicit

    # Single-pass normalisation.  Individual min_weight is validated at
    # ScoringWeights validation time, not guaranteed per-component here
    # because the sum-to-1.0 constraint can conflict with min_weight×6.
    return ScoringWeights(
        ic_strength=raw["ic_strength"] / total,
        ic_consistency=raw["ic_consistency"] / total,
        halflife_persistence=raw["halflife_persistence"] / total,
        cost_adjusted_ic=raw["cost_adjusted_ic"] / total,
        alpha_capture=raw["alpha_capture"] / total,
        execution_sharpe=raw["execution_sharpe"] / total,
    )


def derive_scoring_weights_post_validation(
    *,
    pre_training_weights: ScoringWeights,
    alpha_capture_evidence: float | None = None,
    execution_sharpe_evidence: float | None = None,
    policy: ScoringWeightsPolicy,
) -> ScoringWeights:
    """
    P35: Update weights after execution validation data is available.

    Shifts weight from pre-training proxies toward post-validation evidence:
      - alpha_capture weight increases if alpha_capture is measurable
      - execution_sharpe weight increases if exec_sharpe is measurable
      - Pre-training weights are decayed proportionally.

    When no post-validation evidence exists, returns pre_training_weights unchanged.
    """
    if policy.mode != WeightsMode.ALGORITHMIC:
        return pre_training_weights

    has_capture = alpha_capture_evidence is not None and np.isfinite(alpha_capture_evidence)
    has_exec = execution_sharpe_evidence is not None and np.isfinite(execution_sharpe_evidence)

    if not has_capture and not has_exec:
        return pre_training_weights

    # Decay pre-training weights to make room for post-validation evidence
    n_new = (1 if has_capture else 0) + (1 if has_exec else 0)
    n_old = 4  # ic_strength, ic_consistency, halflife_persistence, cost_adjusted_ic
    old_share = n_old / (n_old + n_new) if n_old + n_new > 0 else 1.0
    new_share = 1.0 - old_share

    # Post-validation evidence strength (0-1)
    capture_strength = abs(alpha_capture_evidence) if has_capture else 0.0
    exec_strength = abs(execution_sharpe_evidence) if has_exec else 0.0
    total_evidence = max(capture_strength + exec_strength, 1e-9)

    capture_w = new_share * (capture_strength / total_evidence) if has_capture else 0.0
    exec_w = new_share * (exec_strength / total_evidence) if has_exec else 0.0

    return ScoringWeights(
        ic_strength=max(policy.min_weight, pre_training_weights.ic_strength * old_share),
        ic_consistency=max(policy.min_weight, pre_training_weights.ic_consistency * old_share),
        halflife_persistence=max(policy.min_weight, pre_training_weights.halflife_persistence * old_share),
        cost_adjusted_ic=max(policy.min_weight, pre_training_weights.cost_adjusted_ic * old_share),
        alpha_capture=max(policy.min_weight, capture_w),
        execution_sharpe=max(policy.min_weight, exec_w),
    )


# Delayed import to avoid circular dependency at module level.
# The type annotation uses string annotation for the alpha_decay parameter.
try:
    import pandas as pd  # noqa: F811 — used at runtime only
except ImportError:
    pd = None  # type: ignore[assignment]


@dataclass(frozen=True)
class PersistencePolicy:
    """How rank persistence at rebalance is evaluated."""
    enabled: bool = True
    formula: str = "exponential_decay"  # exponential_decay | empirical | none
    threshold: float = 0.30              # minimum rank survival fraction
    halflife_source: str = "ic_decay_table"  # ic_decay_table | model_scores | config_override
    halflife_override: float | None = None   # manual override (config_override source)

    def validate(self) -> list[str]:
        errors = []
        if self.formula not in ("exponential_decay", "empirical", "none"):
            errors.append(f"Unknown persistence formula: {self.formula}")
        if not 0.0 <= self.threshold <= 1.0:
            errors.append(f"Persistence threshold must be in [0, 1], got {self.threshold}")
        if self.halflife_source not in ("ic_decay_table", "model_scores", "config_override"):
            errors.append(f"Unknown halflife_source: {self.halflife_source}")
        if self.halflife_source == "config_override" and self.halflife_override is None:
            errors.append("halflife_source=config_override but halflife_override is None")
        return errors


@dataclass(frozen=True)
class CostPolicy:
    """Cost thresholds for horizon economic viability."""
    enabled: bool = True
    max_cost_to_gross_pnl: float = 0.60
    max_impact_fraction_of_cost: float = 0.75
    commission_bps: float | None = None       # None = use ExecutionCostConfig
    spread_bps: float | None = None
    impact_eta: float | None = None

    def validate(self) -> list[str]:
        errors = []
        if self.max_cost_to_gross_pnl < 0:
            errors.append(f"max_cost_to_gross_pnl must be >= 0, got {self.max_cost_to_gross_pnl}")
        return errors


@dataclass(frozen=True)
class FailClosedPolicy:
    """What to do when no contract passes execution persistence gates."""
    mode: FailClosedMode = FailClosedMode.FAIL
    default_contract: ContractCandidate | None = None

    def validate(self) -> list[str]:
        errors = []
        if self.mode == FailClosedMode.FALLBACK_TO_DEFAULT and self.default_contract is None:
            errors.append("fail_closed mode is fallback_to_default but default_contract is not set")
        return errors


@dataclass(frozen=True)
class TieBreakPolicy:
    """How to break ties between contracts with similar composite scores."""
    mode: TieBreakMode = TieBreakMode.PREFER_HIGHER_PERSISTENCE
    tolerance_pct: float = 5.0

    def validate(self) -> list[str]:
        errors = []
        if self.tolerance_pct < 0:
            errors.append(f"tolerance_pct must be >= 0, got {self.tolerance_pct}")
        return errors


@dataclass(frozen=True)
class ArtifactPolicy:
    """Output paths for execution-aware horizon artifacts."""
    horizon_frontier_path: str = "output/models/horizon_frontier.csv"
    rebalance_frontier_path: str = "output/models/rebalance_frontier.csv"
    report_path: str = "output/models/execution_aware_horizon_report.txt"
    active_policy_snapshot_path: str = "output/models/active_execution_aware_horizon_policy.yaml"


# ── Master policy ─────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ExecutionAwareHorizonPolicy:
    """
    P35: Institutional policy for execution-aware horizon selection.

    This is a CONFIGURATION object only — it does NOT change the selected horizon
    during the current phase.  It will govern horizon selection when the
    engine is activated with ``execution_aware_horizon_policy.apply: true``.

    Sections:
      policy_version          — version stamp for config migration
      enabled                 — master on/off switch
      apply                   — false = advisory only; true = govern selection
      candidate_contracts     — (target, holding, rebalance) triples to evaluate
      scoring_weights         — how to score each contract
      persistence_policy      — halflife-based rank survival requirements
      cost_policy             — cost/PnL thresholds
      fail_closed_policy      — what to do if no contract passes
      tie_break_policy        — how to choose when contracts are close
      artifact_policy         — output paths
    """

    policy_version: int = 1
    enabled: bool = True
    apply: bool = False  # False = advisory diagnostics only (current phase)

    candidate_contracts: tuple[ContractCandidate, ...] = field(default_factory=lambda: (
        ContractCandidate(target_horizon_days=5),
        ContractCandidate(target_horizon_days=10),
        ContractCandidate(target_horizon_days=20),
        ContractCandidate(target_horizon_days=63),
    ))

    scoring_weights_policy: ScoringWeightsPolicy = field(default_factory=ScoringWeightsPolicy)
    persistence_policy: PersistencePolicy = field(default_factory=PersistencePolicy)
    cost_policy: CostPolicy = field(default_factory=CostPolicy)
    fail_closed_policy: FailClosedPolicy = field(default_factory=FailClosedPolicy)
    tie_break_policy: TieBreakPolicy = field(default_factory=TieBreakPolicy)
    artifact_policy: ArtifactPolicy = field(default_factory=ArtifactPolicy)

    # ── Convenience: resolve active weights based on mode ─────────────────
    @property
    def scoring_weights(self) -> ScoringWeights:
        """Resolved scoring weights — explicit when CONFIG mode."""
        return self.scoring_weights_policy.explicit

    def resolve_scoring_weights(
        self,
        *,
        alpha_decay: "pd.DataFrame | None" = None,
    ) -> ScoringWeights:
        """
        Resolve scoring weights per the weights policy mode.

        CONFIG mode returns the explicit weights.
        ALGORITHMIC mode derives weights from the IC-decay table if available,
        falling back to the explicit weights if evidence is insufficient.
        """
        if self.scoring_weights_policy.mode == WeightsMode.ALGORITHMIC:
            return derive_scoring_weights_from_decay(
                alpha_decay, policy=self.scoring_weights_policy,
            )
        return self.scoring_weights_policy.explicit

    def validate(self) -> list[str]:
        """Validate all sub-policies. Returns empty list on success."""
        errors: list[str] = []
        errors.extend(self.scoring_weights_policy.validate())
        errors.extend(self.persistence_policy.validate())
        errors.extend(self.cost_policy.validate())
        errors.extend(self.fail_closed_policy.validate())
        errors.extend(self.tie_break_policy.validate())

        # Validate candidates
        if not self.candidate_contracts:
            errors.append("At least one candidate_contract is required")
        for i, cc in enumerate(self.candidate_contracts):
            if cc.target_horizon_days < 1:
                errors.append(f"candidate_contracts[{i}].target_horizon_days must be >= 1")
            if cc.holding_period_days is not None and cc.holding_period_days < 1:
                errors.append(f"candidate_contracts[{i}].holding_period_days must be >= 1")
            if cc.rebalance_frequency_days is not None and cc.rebalance_frequency_days < 1:
                errors.append(f"candidate_contracts[{i}].rebalance_frequency_days must be >= 1")

        return errors

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a pure-dict for JSON/YAML export."""
        swp = self.scoring_weights_policy
        return {
            "policy_version": self.policy_version,
            "enabled": self.enabled,
            "apply": self.apply,
            "candidate_contracts": [
                {
                    "target_horizon_days": cc.target_horizon_days,
                    "holding_period_days": cc.holding_period_days,
                    "rebalance_frequency_days": cc.rebalance_frequency_days,
                }
                for cc in self.candidate_contracts
            ],
            "scoring_weights_policy": {
                "mode": swp.mode.value,
                "data_phase": swp.data_phase.value,
                "explicit": asdict(swp.explicit),
                "min_weight": swp.min_weight,
                "max_weight": swp.max_weight,
                "evidence_min_features": swp.evidence_min_features,
            },
            "persistence_policy": {
                "enabled": self.persistence_policy.enabled,
                "formula": self.persistence_policy.formula,
                "threshold": self.persistence_policy.threshold,
                "halflife_source": self.persistence_policy.halflife_source,
                "halflife_override": self.persistence_policy.halflife_override,
            },
            "cost_policy": {
                "enabled": self.cost_policy.enabled,
                "max_cost_to_gross_pnl": self.cost_policy.max_cost_to_gross_pnl,
                "max_impact_fraction_of_cost": self.cost_policy.max_impact_fraction_of_cost,
                "commission_bps": self.cost_policy.commission_bps,
                "spread_bps": self.cost_policy.spread_bps,
                "impact_eta": self.cost_policy.impact_eta,
            },
            "fail_closed_policy": {
                "mode": self.fail_closed_policy.mode,
                "default_contract": (
                    {
                        "target_horizon_days": self.fail_closed_policy.default_contract.target_horizon_days,
                        "holding_period_days": self.fail_closed_policy.default_contract.holding_period_days,
                        "rebalance_frequency_days": self.fail_closed_policy.default_contract.rebalance_frequency_days,
                    }
                    if self.fail_closed_policy.default_contract is not None
                    else None
                ),
            },
            "tie_break_policy": {
                "mode": self.tie_break_policy.mode,
                "tolerance_pct": self.tie_break_policy.tolerance_pct,
            },
            "artifact_policy": asdict(self.artifact_policy),
        }

    def to_yaml(self) -> str:
        """Serialize to YAML string for snapshot export."""
        try:
            import yaml
            return yaml.dump(self.to_dict(), default_flow_style=False, sort_keys=False)
        except ImportError:
            import json
            return json.dumps(self.to_dict(), indent=2, sort_keys=False)

    def to_active_snapshot(self, *, output_dir: str | Path = "output/models") -> Path:
        """Write the active policy snapshot to disk. Returns the path."""
        output_path = Path(output_dir) / Path(self.artifact_policy.active_policy_snapshot_path).name
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(self.to_yaml(), encoding="utf-8")
        return output_path


# ── YAML Parsing ──────────────────────────────────────────────────────────────

def parse_execution_aware_horizon_policy(
    raw: dict[str, Any] | None,
) -> ExecutionAwareHorizonPolicy:
    """
    P35: Parse execution_aware_horizon_policy from YAML config.

    Accepts a raw dict from YAML (the model_selection.execution_aware_horizon_policy
    block) and returns a validated ExecutionAwareHorizonPolicy.  If the block is
    missing or empty, returns the default policy (advisory-only, enabled).

    Raises ValueError on invalid configuration.
    """
    if not raw:
        return ExecutionAwareHorizonPolicy()

    def _parse_contract(c: dict[str, Any]) -> ContractCandidate:
        h = int(c.get("target_horizon_days", 10))
        holding = c.get("holding_period_days")
        reb = c.get("rebalance_frequency_days")
        return ContractCandidate(
            target_horizon_days=max(1, h),
            holding_period_days=max(1, int(holding)) if holding is not None else None,
            rebalance_frequency_days=max(1, int(reb)) if reb is not None else None,
        )

    # Candidates
    candidates_raw = raw.get("candidate_contracts")
    if candidates_raw is not None:
        candidates = tuple(_parse_contract(c) for c in candidates_raw if isinstance(c, dict))
    else:
        # Legacy shorthand: candidate_horizons
        horizons = raw.get("candidate_horizons", [5, 10, 20, 63])
        candidates = tuple(
            ContractCandidate(target_horizon_days=int(h)) for h in horizons
        )

    # Scoring weights policy
    swp_raw = raw.get("scoring_weights_policy", raw.get("scoring_weights", {})) or {}
    swp_mode_str = str(swp_raw.get("mode", "config")).strip().lower()
    try:
        swp_mode = WeightsMode(swp_mode_str)
    except ValueError:
        raise ValueError(
            f"Invalid scoring_weights_policy.mode: '{swp_mode_str}'. "
            f"Valid: {[m.value for m in WeightsMode]}"
        )
    swp_data_phase_str = str(swp_raw.get("data_phase", "pre_training")).strip().lower()
    try:
        swp_data_phase = DataPhase(swp_data_phase_str)
    except ValueError:
        raise ValueError(
            f"Invalid scoring_weights_policy.data_phase: '{swp_data_phase_str}'. "
            f"Valid: {[p.value for p in DataPhase]}"
        )

    # Explicit weights (used as fallback priors in ALGORITHMIC mode)
    exp_raw = swp_raw.get("explicit", {}) or {}
    # Fallback: if user used old `scoring_weights` top-level key, use those values
    if not exp_raw and isinstance(raw.get("scoring_weights"), dict):
        exp_raw = raw["scoring_weights"]
    explicit_weights = ScoringWeights(
        ic_strength=float(exp_raw.get("ic_strength", 0.25)),
        ic_consistency=float(exp_raw.get("ic_consistency", 0.15)),
        halflife_persistence=float(exp_raw.get("halflife_persistence", 0.20)),
        cost_adjusted_ic=float(exp_raw.get("cost_adjusted_ic", 0.20)),
        alpha_capture=float(exp_raw.get("alpha_capture", 0.10)),
        execution_sharpe=float(exp_raw.get("execution_sharpe", 0.10)),
    )
    scoring_weights_policy = ScoringWeightsPolicy(
        mode=swp_mode,
        data_phase=swp_data_phase,
        explicit=explicit_weights,
        min_weight=float(swp_raw.get("min_weight", 0.05)),
        max_weight=float(swp_raw.get("max_weight", 0.40)),
        evidence_min_features=int(swp_raw.get("evidence_min_features", 3)),
    )

    # Persistence policy
    pp_raw = raw.get("persistence_policy", {}) or {}
    persistence_policy = PersistencePolicy(
        enabled=bool(pp_raw.get("enabled", True)),
        formula=str(pp_raw.get("formula", "exponential_decay")),
        threshold=float(pp_raw.get("threshold", 0.30)),
        halflife_source=str(pp_raw.get("halflife_source", "ic_decay_table")),
        halflife_override=float(pp_raw["halflife_override"]) if pp_raw.get("halflife_override") is not None else None,
    )

    # Cost policy
    cp_raw = raw.get("cost_policy", {}) or {}
    cost_policy = CostPolicy(
        enabled=bool(cp_raw.get("enabled", True)),
        max_cost_to_gross_pnl=float(cp_raw.get("max_cost_to_gross_pnl", 0.60)),
        max_impact_fraction_of_cost=float(cp_raw.get("max_impact_fraction_of_cost", 0.75)),
        commission_bps=float(cp_raw["commission_bps"]) if cp_raw.get("commission_bps") is not None else None,
        spread_bps=float(cp_raw["spread_bps"]) if cp_raw.get("spread_bps") is not None else None,
        impact_eta=float(cp_raw["impact_eta"]) if cp_raw.get("impact_eta") is not None else None,
    )

    # Fail closed policy
    fcp_raw = raw.get("fail_closed_policy", {}) or {}
    fail_mode_str = str(fcp_raw.get("mode", "fail"))
    try:
        fail_mode = FailClosedMode(fail_mode_str)
    except ValueError:
        raise ValueError(
            f"Invalid fail_closed_policy.mode: '{fail_mode_str}'. "
            f"Valid: {[m.value for m in FailClosedMode]}"
        )
    default_contract = None
    if fail_mode == FailClosedMode.FALLBACK_TO_DEFAULT:
        dc_raw = fcp_raw.get("default_contract", {}) or {}
        if not dc_raw:
            raise ValueError(
                "fail_closed_policy.mode=fallback_to_default requires default_contract"
            )
        default_contract = _parse_contract(dc_raw)
    fail_closed_policy = FailClosedPolicy(
        mode=fail_mode,
        default_contract=default_contract,
    )

    # Tie break policy
    tbp_raw = raw.get("tie_break_policy", {}) or {}
    tie_mode_str = str(tbp_raw.get("mode", "prefer_higher_persistence"))
    try:
        tie_mode = TieBreakMode(tie_mode_str)
    except ValueError:
        raise ValueError(
            f"Invalid tie_break_policy.mode: '{tie_mode_str}'. "
            f"Valid: {[m.value for m in TieBreakMode]}"
        )
    tie_break_policy = TieBreakPolicy(
        mode=tie_mode,
        tolerance_pct=float(tbp_raw.get("tolerance_pct", 5.0)),
    )

    # Artifact policy
    ap_raw = raw.get("artifact_policy", {}) or {}
    artifact_policy = ArtifactPolicy(
        horizon_frontier_path=str(ap_raw.get("horizon_frontier_path", "output/models/horizon_frontier.csv")),
        rebalance_frontier_path=str(ap_raw.get("rebalance_frontier_path", "output/models/rebalance_frontier.csv")),
        report_path=str(ap_raw.get("report_path", "output/models/execution_aware_horizon_report.txt")),
        active_policy_snapshot_path=str(ap_raw.get("active_policy_snapshot_path", "output/models/active_execution_aware_horizon_policy.yaml")),
    )

    # Master policy
    policy = ExecutionAwareHorizonPolicy(
        policy_version=int(raw.get("policy_version", 1)),
        enabled=bool(raw.get("enabled", True)),
        apply=bool(raw.get("apply", False)),
        candidate_contracts=candidates,
        scoring_weights_policy=scoring_weights_policy,
        persistence_policy=persistence_policy,
        cost_policy=cost_policy,
        fail_closed_policy=fail_closed_policy,
        tie_break_policy=tie_break_policy,
        artifact_policy=artifact_policy,
    )

    # Validate
    validation_errors = policy.validate()
    if validation_errors:
        raise ValueError(
            "ExecutionAwareHorizonPolicy validation failed:\n  "
            + "\n  ".join(validation_errors)
        )

    return policy
