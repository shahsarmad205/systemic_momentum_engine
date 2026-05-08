"""Pipeline stage: cost viability adapter.

Responsibility: Bridge between the pipeline and the CostViabilityEngine.
No cost math here — just delegation and result mapping.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from model_selection.cost_viability_engine import (
    CostViabilityEngine,
    ViabilityResult,
    CostStatus,
)
from research_pipeline.contract import ResearchContract

logger = logging.getLogger(__name__)


@dataclass
class CostViabilityAdapterResult:
    candidate_id: str
    feature: str
    family: str
    horizon: int
    cost_status: CostStatus
    expected_alpha_bps: float
    expected_cost_bps: float
    net_expected_alpha_bps: float
    alpha_cost_ratio: float
    rejection_reason: str


def evaluate_cost_viability(
    candidate_id: str,
    feature: str,
    family: str,
    ic: float,
    horizon: int,
    sigma_annual: float,
    halflife: float,
    expected_turnover: float,
    adv_usd: float,
    daily_vol: float,
    contract: ResearchContract,
    icir: float = 0.0,
    t_stat: float = 0.0,
    is_short: bool = False,
    position_weight: float = 0.0,
    capital: float = 10_000_000.0,
) -> CostViabilityAdapterResult:
    """Evaluate cost viability for a single candidate.

    Delegates to CostViabilityEngine. All thresholds come from contract.
    """
    engine = CostViabilityEngine(config=contract.raw_config)

    result = engine.evaluate(
        candidate_id=candidate_id,
        feature=feature,
        family=family,
        ic=ic,
        horizon=horizon,
        sigma_annual=sigma_annual,
        halflife=halflife,
        expected_turnover=expected_turnover,
        adv_usd=adv_usd,
        daily_vol=daily_vol,
        icir=icir,
        t_stat=t_stat,
        is_short=is_short,
        position_weight=position_weight,
        capital=capital,
    )

    return CostViabilityAdapterResult(
        candidate_id=result.candidate_id,
        feature=result.feature,
        family=result.family,
        horizon=result.horizon,
        cost_status=result.cost_status,
        expected_alpha_bps=result.expected_alpha_bps,
        expected_cost_bps=result.expected_cost_bps,
        net_expected_alpha_bps=result.net_expected_alpha_bps,
        alpha_cost_ratio=result.alpha_cost_ratio,
        rejection_reason=result.rejection_reason,
    )


def batch_evaluate_cost_viability(
    candidates: list[dict],
    contract: ResearchContract,
) -> list[CostViabilityAdapterResult]:
    """Evaluate cost viability for multiple candidates."""
    results = []
    for c in candidates:
        r = evaluate_cost_viability(
            candidate_id=c["candidate_id"],
            feature=c["feature"],
            family=c.get("family", "unknown"),
            ic=c.get("ic", 0.0),
            horizon=c.get("horizon", 20),
            sigma_annual=c.get("sigma_annual", 0.20),
            halflife=c.get("halflife", 0.0),
            expected_turnover=c.get("expected_turnover", 0.10),
            adv_usd=c.get("adv_usd", contract.cost_viability.default_adv_usd),
            daily_vol=c.get("daily_vol", contract.cost_viability.default_daily_vol),
            contract=contract,
            icir=c.get("icir", 0.0),
            t_stat=c.get("t_stat", 0.0),
            is_short=c.get("is_short", False),
        )
        results.append(r)
    return results
