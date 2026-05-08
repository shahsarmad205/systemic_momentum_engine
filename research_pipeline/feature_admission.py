"""Pipeline stage: feature admission.

Responsibility: Decide which features pass admission gates.
Delegates to existing alpha_research.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from research_pipeline.contract import ResearchContract

logger = logging.getLogger(__name__)


@dataclass
class FeatureAdmissionResult:
    feature: str
    family: str
    admitted: bool
    ic: float
    ic_tstat: float
    icir: float
    halflife: float
    rejection_reason: str = ""


def run_feature_admission(
    df: pd.DataFrame,
    feature_columns: list[str],
    contract: ResearchContract,
) -> tuple[list[str], list[FeatureAdmissionResult]]:
    """Run feature admission pipeline.

    Delegates to existing run_alpha_research for IC computation,
    BHY correction, marginal contribution, and redundancy checks.

    Returns:
        (admitted_features, admission_results)
    """
    from model_selection.alpha_research import (
        run_alpha_research,
        AlphaAdmissionConfig,
    )

    alpha_cfg = AlphaAdmissionConfig(
        min_ic_tstat=contract.feature_admission.min_ic_tstat,
        min_production_ic_valid_days=contract.feature_admission.min_production_ic_valid_days,
        fail_if_below_minimum=contract.feature_admission.fail_if_below_minimum,
        minimum_admitted_features=contract.feature_admission.minimum_admitted_features,
    )

    horizon = contract.horizon.target_horizon_days

    result = run_alpha_research(
        df=df,
        feature_columns=feature_columns,
        horizon=horizon,
        config=alpha_cfg,
    )

    admitted = result.get("admitted_features", [])

    # Build admission report
    results = []
    for feat in feature_columns:
        spec = _get_feature_spec(feat)
        feat_result = result.get("feature_results", {}).get(feat, {})
        is_admitted = feat in admitted

        results.append(FeatureAdmissionResult(
            feature=feat,
            family=spec.family if spec else "unknown",
            admitted=is_admitted,
            ic=feat_result.get("mean_ic", 0.0),
            ic_tstat=feat_result.get("ic_tstat", 0.0),
            icir=feat_result.get("icir", 0.0),
            halflife=feat_result.get("halflife", 0.0),
            rejection_reason=feat_result.get("rejection_reason", ""),
        ))

    logger.info("Feature admission: %d/%d admitted", len(admitted), len(feature_columns))

    return admitted, results


def _get_feature_spec(feature: str):
    """Get feature spec from research contract."""
    from model_selection.research_contract import FEATURE_SPECS
    return FEATURE_SPECS.get(feature)
