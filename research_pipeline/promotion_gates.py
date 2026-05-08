"""Pipeline stage: promotion gates.

Responsibility: Evaluate promotion gates for a candidate.
Delegates to existing validation.py evaluate_promotion_gates.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from research_pipeline.contract import ResearchContract

logger = logging.getLogger(__name__)


@dataclass
class PromotionGateResult:
    candidate_id: str
    passed: bool
    failures: list[str]
    metrics: dict[str, float]


def evaluate_promotion_gates(
    candidate_id: str,
    metrics: dict[str, float],
    contract: ResearchContract,
) -> PromotionGateResult:
    """Evaluate promotion gates for a candidate.

    Delegates to existing validation.py evaluate_promotion_gates.
    All thresholds come from contract.
    """
    from model_selection.validation import evaluate_promotion_gates as _eval_gates

    passes, failures = _eval_gates(metrics, contract.raw_config)

    return PromotionGateResult(
        candidate_id=candidate_id,
        passed=passes,
        failures=failures,
        metrics=metrics,
    )
