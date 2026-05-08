"""Feature diversity wiring for run_model_selection.py.

Bridges the FeatureDiversityEngine into the existing model selection pipeline:
1. Feature registry construction
2. Redundancy diagnostics across the admitted feature set
3. Effective signal count and family concentration
4. Marginal IC and cluster representative selection
5. Feature admission decisions
6. Walk-forward diversity stability
7. Report generation (11 CSVs + PM summary)

Usage:
    from model_selection.feature_diversity_wiring import (
        wire_feature_diversity_into_pipeline,
        evaluate_feature_diversity,
        generate_feature_diversity_reports,
    )

Insertion point in run_model_selection.py:
    - After PIT condition engine (L6888): evaluate_feature_diversity()
    - At end of main(): generate_feature_diversity_reports()
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_selection.feature_diversity_engine import (
    FeatureDiversityEngine,
    FeatureDiversityBundle,
    FeatureRegistryEntry,
    RedundancyPair,
    FeatureCluster,
    EffectiveSignalResult,
    MarginalValueResult,
    FamilyConcentrationResult,
    RepresentativeSelection,
    FeatureDiversityAdmission,
    DiversityWalkForwardResult,
    FeatureFinalStatus,
    DiversityStatus,
    generate_diversity_reports,
)
from model_selection.research_contract import FEATURE_SPECS

logger = logging.getLogger(__name__)


# ── Data structures ──────────────────────────────────────────────────────────

@dataclass
class FeatureDiversityWiringState:
    """Accumulates all feature diversity results during a pipeline run."""
    bundle: FeatureDiversityBundle | None = None
    report_paths: dict[str, Path] = field(default_factory=dict)
    config: dict[str, Any] = field(default_factory=dict)

    @property
    def n_raw_features(self) -> int:
        return len(self.bundle.registry) if self.bundle else 0

    @property
    def n_effective_signals(self) -> float:
        if not self.bundle:
            return 0.0
        for e in self.bundle.effective_signals:
            if e.scope == "full_universe":
                return e.n_effective_signals
        return 0.0

    @property
    def n_admitted(self) -> int:
        if not self.bundle:
            return 0
        return sum(1 for a in self.bundle.admissions if "admitted" in a.final_status)

    @property
    def n_rejected(self) -> int:
        if not self.bundle:
            return 0
        return sum(1 for a in self.bundle.admissions if "rejected" in a.final_status)

    @property
    def n_clusters(self) -> int:
        return len(self.bundle.cluster_membership) if self.bundle else 0

    @property
    def diversity_status(self) -> str:
        if not self.bundle:
            return "unknown"
        for e in self.bundle.effective_signals:
            if e.scope == "full_universe":
                return e.diversity_status
        return "unknown"


# ── Feature-level diversity evaluation ───────────────────────────────────────

def evaluate_feature_diversity(
    df: pd.DataFrame,
    features: list[str],
    cfg: dict[str, Any],
    horizons: list[int] | None = None,
) -> FeatureDiversityBundle:
    """Evaluate feature diversity for the admitted feature set.

    Called after alpha research and PIT condition validation in
    run_model_selection.py.

    Args:
        df: Feature matrix panel with date/ticker index
        features: List of admitted feature names
        cfg: Full config dict
        horizons: List of prediction horizons (optional)

    Returns:
        FeatureDiversityBundle with full analysis results.
    """
    engine = FeatureDiversityEngine(config=cfg)

    bundle = engine.run_full_diversity_analysis(
        df, features, horizons=horizons,
    )

    logger.info(
        "Feature diversity: %d raw features, %.1f effective signals, "
        "%d clusters, %d admitted, %d rejected",
        len(bundle.registry),
        next((e.n_effective_signals for e in bundle.effective_signals if e.scope == "full_universe"), 0),
        len(bundle.cluster_membership),
        sum(1 for a in bundle.admissions if "admitted" in a.final_status),
        sum(1 for a in bundle.admissions if "rejected" in a.final_status),
    )

    return bundle


# ── Report generation ────────────────────────────────────────────────────────

def generate_feature_diversity_reports(
    bundle: FeatureDiversityBundle,
    output_dir: str | Path = "output/models/feature_diversity",
) -> dict[str, Path]:
    """Generate all feature diversity reports.

    Called at the end of main() in run_model_selection.py.

    Args:
        bundle: FeatureDiversityBundle from evaluate_feature_diversity()
        output_dir: Output directory for reports

    Returns:
        Dict mapping report name to file path.
    """
    return generate_diversity_reports(bundle, output_dir)


# ── Convenience: full pipeline wiring ────────────────────────────────────────

def wire_feature_diversity_into_pipeline(
    df: pd.DataFrame,
    features: list[str],
    cfg: dict[str, Any],
    horizons: list[int] | None = None,
    output_dir: str | Path = "output/models/feature_diversity",
) -> FeatureDiversityWiringState:
    """Full feature diversity wiring for a pipeline run.

    Call this after alpha research and PIT condition validation in
    run_model_selection.py.

    Args:
        df: Feature matrix panel
        features: List of admitted feature names
        cfg: Full config dict
        horizons: List of prediction horizons (optional)
        output_dir: Output directory for reports

    Returns:
        FeatureDiversityWiringState with all results accumulated.
    """
    state = FeatureDiversityWiringState(config=cfg)

    # Run full diversity analysis
    state.bundle = evaluate_feature_diversity(df, features, cfg, horizons)

    # Generate reports
    state.report_paths = generate_feature_diversity_reports(
        state.bundle, output_dir,
    )

    return state


# ── Feature pruning suggestions ──────────────────────────────────────────────

def get_feature_pruning_suggestions(
    bundle: FeatureDiversityBundle,
) -> list[dict[str, Any]]:
    """Generate feature pruning suggestions based on diversity analysis.

    Returns a list of dicts with feature, reason, and suggested action.
    """
    suggestions = []

    # Build lookups
    marginal_lookup: dict[str, MarginalValueResult] = {}
    for m in bundle.marginal_values:
        marginal_lookup[m.feature] = m

    admission_lookup: dict[str, FeatureDiversityAdmission] = {}
    for a in bundle.admissions:
        admission_lookup[a.feature] = a

    # Rejected redundant features
    for a in bundle.admissions:
        if a.final_status == FeatureFinalStatus.REJECTED_REDUNDANT.value:
            suggestions.append({
                "feature": a.feature,
                "family": a.family,
                "reason": a.rejection_reason,
                "action": "remove",
                "priority": "high",
            })

    # Zero marginal value features
    for m in bundle.marginal_values:
        if m.marginal_value_status in ("redundant_low_value", "negative_marginal_value"):
            if m.feature not in [s["feature"] for s in suggestions]:
                suggestions.append({
                    "feature": m.feature,
                    "family": m.family,
                    "reason": m.rejection_reason,
                    "action": "remove_or_research",
                    "priority": "medium",
                })

    # Family concentration warnings
    for fc in bundle.family_concentration:
        if fc.family_concentration_status == "concentrated":
            suggestions.append({
                "feature": f"[{fc.family} family]",
                "family": fc.family,
                "reason": f"family_share_{fc.family_share:.2f}_exceeds_threshold",
                "action": "prune_family",
                "priority": "high",
            })

    # Low effective breadth
    for e in bundle.effective_signals:
        if e.scope == "full_universe" and e.diversity_status in (
            DiversityStatus.DUPLICATE_STACK.value,
            DiversityStatus.LOW_EFFECTIVE_BREADTH.value,
        ):
            suggestions.append({
                "feature": "[universe]",
                "family": "all",
                "reason": f"effective_ratio_{e.effective_ratio:.2f}_too_low",
                "action": "major_pruning_needed",
                "priority": "critical",
            })

    return suggestions
