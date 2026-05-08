"""Pipeline stage: candidate factory.

Responsibility: Create model candidate specs from models, features, and horizons.
No model fitting — just spec construction.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from research_pipeline.contract import ResearchContract

logger = logging.getLogger(__name__)


@dataclass
class CandidateSpec:
    """Specification for a single model candidate."""
    candidate_id: str
    model_name: str
    model_kind: str  # "long", "short", "overlay"
    feature_view: str  # "full", "program"
    active_features: list[str]
    horizon: int
    uses_proba: bool


def build_candidate_pool(
    model_specs: list[dict[str, Any]],
    active_features: list[str],
    contract: ResearchContract,
) -> list[CandidateSpec]:
    """Build candidate pool from model specs × horizons × feature views.

    Preserves existing behavior from _build_nested_candidate_pool.
    """
    candidates = []
    horizons = [contract.horizon.target_horizon_days]
    feature_views = ["full"]

    for model in model_specs:
        for h in horizons:
            for view in feature_views:
                spec = CandidateSpec(
                    candidate_id=f"{model['name']}_{view}_h{h}",
                    model_name=model["name"],
                    model_kind=model.get("kind", "long"),
                    feature_view=view,
                    active_features=active_features,
                    horizon=h,
                    uses_proba=model.get("uses_proba", False),
                )
                candidates.append(spec)

    # Cap pool size per contract
    max_candidates = contract.search.max_candidates
    if len(candidates) > max_candidates:
        candidates = candidates[:max_candidates]
        logger.info("Candidate pool capped to %d", max_candidates)

    logger.info("Candidate pool built: %d candidates", len(candidates))
    return candidates


def build_model_specs(contract: ResearchContract) -> list[dict[str, Any]]:
    """Build model specs from the model registry.

    Delegates to existing model_registry.build_models.
    """
    from model_selection.model_registry import build_models

    models = build_models(contract.raw_config)
    return models
