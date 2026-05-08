"""Pipeline stage: walk-forward validator.

Responsibility: Execute walk-forward validation for a candidate.
Delegates to existing validation.py and the nested search logic.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import pandas as pd

from research_pipeline.contract import ResearchContract

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    candidate_id: str
    model_name: str
    horizon: int
    windows_evaluated: int
    oos_sharpe: float
    oos_ic: float
    oos_deflated_sharpe: float
    cost_adjusted_sharpe: float
    max_drawdown: float
    win_rate: float
    turnover: float
    passed: bool
    gate_failures: list[str]


def run_walk_forward_validation(
    df: pd.DataFrame,
    candidate: Any,  # CandidateSpec or nested spec
    contract: ResearchContract,
) -> ValidationResult:
    """Run walk-forward validation for a single candidate.

    Delegates to existing _evaluate_model_family and nested validation.
    This is a thin wrapper — the heavy lifting is in validation.py.
    """
    # The existing pipeline uses _evaluate_model_family which handles:
    # 1. Walk-forward window splitting
    # 2. Per-window model fitting
    # 3. Score calibration
    # 4. Executable portfolio simulation
    # 5. Metric aggregation
    # 6. Promotion gate evaluation

    # For the new pipeline, we delegate to the existing logic
    # but wrap it with proper input/output contracts.

    logger.info("Walk-forward validation: %s, horizon=%d",
                candidate.candidate_id if hasattr(candidate, 'candidate_id') else str(candidate),
                candidate.horizon if hasattr(candidate, 'horizon') else 20)

    # Placeholder — the actual implementation delegates to the existing
    # _evaluate_model_family in run_model_selection.py during extraction.
    # This module will be populated during Phase 5 extraction.

    return ValidationResult(
        candidate_id=candidate.candidate_id if hasattr(candidate, 'candidate_id') else "unknown",
        model_name=candidate.model_name if hasattr(candidate, 'model_name') else "unknown",
        horizon=candidate.horizon if hasattr(candidate, 'horizon') else 20,
        windows_evaluated=0,
        oos_sharpe=0.0,
        oos_ic=0.0,
        oos_deflated_sharpe=0.0,
        cost_adjusted_sharpe=0.0,
        max_drawdown=0.0,
        win_rate=0.0,
        turnover=0.0,
        passed=False,
        gate_failures=["not_yet_extracted"],
    )
