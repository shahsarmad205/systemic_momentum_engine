"""Pipeline stage: feature registry.

Responsibility: Provide a clean interface to FEATURE_SPECS for the pipeline.
Wraps the existing FeatureFamilyRegistry from research_contract.py.
"""
from __future__ import annotations

import logging

from model_selection.research_contract import (
    FEATURE_SPECS,
    FeatureFamilyRegistry,
    FeatureSpec,
    filter_horizon_aligned_features,
    get_horizon_alignment_report,
)

logger = logging.getLogger(__name__)


def build_feature_registry(
    feature_columns: list[str],
    prediction_horizon_days: int,
    alignment_multiplier: float = 2.0,
) -> FeatureFamilyRegistry:
    """Build feature registry from available columns.

    Returns registry filtered to columns that exist in the data.
    """
    registry = FeatureFamilyRegistry(FEATURE_SPECS)

    # Log alignment report
    report = get_horizon_alignment_report(
        feature_columns,
        prediction_horizon_days,
        alignment_multiplier=alignment_multiplier,
    )
    logger.info("Feature horizon alignment: %d aligned, %d misaligned, %d unknown",
                report["n_aligned"], report["n_misaligned"], report["n_unknown"])

    return registry


def get_active_features(
    df_columns: list[str],
    registry: FeatureFamilyRegistry,
    prediction_horizon_days: int,
    alignment_multiplier: float = 2.0,
    enforce_alignment: bool = False,
) -> list[str]:
    """Get active feature columns for model training.

    Returns leakage-safe, horizon-aligned features that exist in the data.
    """
    from model_selection.research_contract import is_model_feature_column
    import pandas as pd

    # Step 1: Filter to model-eligible columns
    eligible = []
    for col in df_columns:
        if col in {"date", "ticker"}:
            continue
        series = pd.Series(dtype=float)  # dummy for type check
        if col in df_columns:
            # We check eligibility by name patterns, not dtype (dtype needs actual series)
            if _is_eligible_by_name(col):
                eligible.append(col)

    # Step 2: Horizon alignment
    if enforce_alignment:
        aligned = filter_horizon_aligned_features(
            eligible,
            prediction_horizon_days,
            alignment_multiplier=alignment_multiplier,
        )
    else:
        aligned = eligible

    # Step 3: Filter to columns actually present
    active = [c for c in aligned if c in df_columns]

    logger.info("Active features: %d (from %d eligible, horizon=%dd)",
                len(active), len(eligible), prediction_horizon_days)

    return active


def _is_eligible_by_name(col: str) -> bool:
    """Quick name-based eligibility check (without needing Series dtype)."""
    from model_selection.research_contract import (
        ALPHA_METADATA_COLUMNS, TARGET_COLUMNS, RISK_EXECUTION_COLUMNS,
        SHORT_TARGET_COLUMNS, FEATURE_SPECS,
    )
    if col in ALPHA_METADATA_COLUMNS:
        return False
    if col in TARGET_COLUMNS:
        return False
    if col in RISK_EXECUTION_COLUMNS:
        return False
    if col in SHORT_TARGET_COLUMNS:
        return False
    if col.startswith("target_") or "forward" in col.lower() or "direction" in col.lower():
        return False
    if col.startswith("short_") and col not in FEATURE_SPECS:
        return False
    if col.endswith(("_timestamp", "__timestamp", "_asof", "__asof", "_available_at", "__available_at")):
        return False
    return True


def get_feature_family(feature: str, registry: FeatureFamilyRegistry) -> str:
    """Get family for a feature. Returns 'unknown' if not registered."""
    return registry.family_of(feature)


def get_feature_spec(feature: str) -> FeatureSpec | None:
    """Get FeatureSpec for a registered feature."""
    return FEATURE_SPECS.get(feature)
