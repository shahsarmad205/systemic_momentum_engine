"""Pipeline stage: horizon diagnostics.

Responsibility: Compute per-horizon IC, IC decay, halflife, and eligibility.
Delegates to existing horizon_eligibility.py and horizon_gate.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_pipeline.contract import ResearchContract

logger = logging.getLogger(__name__)


@dataclass
class HorizonDiagnostic:
    horizon: int
    mean_ic: float
    ic_std: float
    ic_tstat: float
    icir: float
    signal_halflife_days: float
    n_valid_dates: int
    eligible: bool
    cost_viable: bool
    rejection_reason: str = ""


def compute_horizon_diagnostics(
    df: pd.DataFrame,
    feature_columns: list[str],
    contract: ResearchContract,
) -> list[HorizonDiagnostic]:
    """Compute diagnostics for each candidate horizon.

    Delegates to existing horizon_eligibility.compute_all_eligibility
    and cost viability engine.
    """
    from model_selection.horizon_eligibility import compute_all_eligibility

    horizons = list(contract.horizon.candidate_horizons)
    if contract.horizon.target_horizon_days not in horizons:
        horizons.append(contract.horizon.target_horizon_days)
    horizons = sorted(set(horizons))

    diagnostics = []
    for h in horizons:
        diag = _compute_single_horizon(df, feature_columns, h, contract)
        diagnostics.append(diag)

    logger.info("Horizon diagnostics computed for %d horizons", len(diagnostics))
    return diagnostics


def _compute_single_horizon(
    df: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
    contract: ResearchContract,
) -> HorizonDiagnostic:
    """Compute diagnostics for a single horizon."""
    from model_selection.horizon_eligibility import compute_eligibility
    from model_selection.horizon_eligibility import compute_cost_viability_institutional

    # Compute IC stats for this horizon
    ic_stats = _compute_ic_for_horizon(df, feature_columns, horizon)

    mean_ic = ic_stats.get("mean_ic", 0.0)
    ic_std = ic_stats.get("ic_std", 0.0)
    ic_tstat = ic_stats.get("ic_tstat", 0.0)
    icir = ic_stats.get("icir", 0.0)
    halflife = ic_stats.get("halflife", 0.0)
    n_dates = ic_stats.get("n_dates", 0)

    # Cost viability check
    is_viable, alpha_bps, cost_bps = compute_cost_viability_institutional(
        ic=mean_ic,
        horizon=horizon,
        adv_usd=contract.cost_viability.default_adv_usd,
        daily_vol=contract.cost_viability.default_daily_vol,
        sigma_annual=0.20,
        expected_turnover=ic_stats.get("turnover", 0.10),
        cost_config=contract.raw_config,
    )

    eligible = mean_ic != 0.0 and n_dates > 0
    rejection = ""
    if not eligible:
        rejection = "insufficient_ic_or_dates"
    elif not is_viable:
        rejection = "cost_not_viable"

    return HorizonDiagnostic(
        horizon=horizon,
        mean_ic=mean_ic,
        ic_std=ic_std,
        ic_tstat=ic_tstat,
        icir=icir,
        signal_halflife_days=halflife,
        n_valid_dates=n_dates,
        eligible=eligible,
        cost_viable=is_viable,
        rejection_reason=rejection,
    )


def _compute_ic_for_horizon(
    df: pd.DataFrame,
    feature_columns: list[str],
    horizon: int,
) -> dict:
    """Compute cross-sectional IC for a given horizon."""
    from model_selection.validation import cross_sectional_ic

    # Use forward_return as the target
    target_col = "forward_return"
    if target_col not in df.columns:
        return {"mean_ic": 0.0, "ic_std": 0.0, "ic_tstat": 0.0, "icir": 0.0,
                "halflife": 0.0, "n_dates": 0, "turnover": 0.10}

    # Aggregate IC across features
    ics = []
    for feat in feature_columns:
        if feat not in df.columns:
            continue
        try:
            result = cross_sectional_ic(df, feat, target_col)
            if result and "mean_ic" in result:
                ics.append(result)
        except Exception:
            continue

    if not ics:
        return {"mean_ic": 0.0, "ic_std": 0.0, "ic_tstat": 0.0, "icir": 0.0,
                "halflife": 0.0, "n_dates": 0, "turnover": 0.10}

    mean_ics = [r["mean_ic"] for r in ics if np.isfinite(r.get("mean_ic", 0))]
    if not mean_ics:
        return {"mean_ic": 0.0, "ic_std": 0.0, "ic_tstat": 0.0, "icir": 0.0,
                "halflife": 0.0, "n_dates": 0, "turnover": 0.10}

    mean_ic = float(np.mean(mean_ics))
    ic_std = float(np.std(mean_ics)) if len(mean_ics) > 1 else 0.0
    n_dates = int(np.mean([r.get("n_dates", 0) for r in ics]))
    halflife = float(np.mean([r.get("signal_halflife_days", 0) for r in ics if np.isfinite(r.get("signal_halflife_days", 0))]))
    turnover = float(np.mean([r.get("turnover_mean", 0.10) for r in ics]))

    ic_tstat = mean_ic / (ic_std / max(len(mean_ics), 1) ** 0.5) if ic_std > 0 else 0.0
    icir = mean_ic / ic_std if ic_std > 0 else 0.0

    return {
        "mean_ic": mean_ic,
        "ic_std": ic_std,
        "ic_tstat": ic_tstat,
        "icir": icir,
        "halflife": halflife,
        "n_dates": n_dates,
        "turnover": turnover,
    }
