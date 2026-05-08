"""Pipeline stage: target construction.

Responsibility: Build target columns from the feature matrix.
Delegates to existing training.py and target_construction.py.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd

from research_pipeline.contract import ResearchContract

logger = logging.getLogger(__name__)


@dataclass
class TargetBuildReport:
    target_column: str
    n_positive: int
    n_negative: int
    positive_rate: float
    n_missing: int
    missing_rate: float
    risk_adjusted_available: bool


def build_targets(
    df: pd.DataFrame,
    contract: ResearchContract,
) -> tuple[pd.DataFrame, TargetBuildReport]:
    """Build target columns on the feature matrix.

    Preserves existing behavior:
    - forward_return from feature_builder
    - target_up = forward_return > 0
    - y_bin = target_up.astype(int)
    - Optional: risk-adjusted target
    """
    from model_selection.training import add_institutional_targets, TargetConfig

    # Build base targets
    target_cfg = _build_target_config(contract)
    df = add_institutional_targets(df, target_cfg)

    # Ensure binary target exists
    if "forward_return" in df.columns:
        df["target_up"] = (df["forward_return"] > 0).astype(int)
        target_col = "target_up"
    elif "target_return" in df.columns:
        df["target_up"] = (df["target_return"] > 0).astype(int)
        target_col = "target_up"
    else:
        logger.warning("No forward_return or target_return column found")
        target_col = "target_up"
        df["target_up"] = 0

    # Risk-adjusted target if requested
    risk_adj_available = False
    if contract.risk_adj_target:
        risk_adj_available = _apply_risk_adjusted_target(df)

    report = _build_target_report(df, target_col, risk_adj_available)

    logger.info("Targets built: %s, positive_rate=%.2f, missing_rate=%.2f",
                report.target_column, report.positive_rate, report.missing_rate)

    return df, report


def _build_target_config(contract: ResearchContract) -> TargetConfig:
    """Build TargetConfig from contract."""
    from model_selection.training import TargetConfig
    raw = contract.raw_config.get("model_selection", {}).get("target", {})
    return TargetConfig(
        residualize=raw.get("residualize", False),
        net_of_costs=raw.get("net_of_costs", False),
        winsor_q=raw.get("winsor_q", None),
    )


def _apply_risk_adjusted_target(df: pd.DataFrame) -> bool:
    """Apply risk-adjusted target: forward_return / realized_vol."""
    if "forward_return" not in df.columns:
        return False

    vol_col = None
    for candidate in ["realised_vol_20d", "vol_20_simple", "rolling_vol_20"]:
        if candidate in df.columns:
            vol_col = candidate
            break

    if vol_col is None:
        return False

    vol = pd.to_numeric(df[vol_col], errors="coerce")
    fwd = pd.to_numeric(df["forward_return"], errors="coerce")
    valid = vol.notna() & (vol > 0) & fwd.notna()

    df["forward_return_risk_adj"] = fwd
    df.loc[valid, "forward_return_risk_adj"] = fwd[valid] / vol[valid]

    return True


def _build_target_report(
    df: pd.DataFrame,
    target_col: str,
    risk_adj_available: bool,
) -> TargetBuildReport:
    """Build target construction report."""
    if target_col in df.columns:
        s = pd.to_numeric(df[target_col], errors="coerce")
        n_pos = int((s > 0).sum())
        n_neg = int((s <= 0).sum())
        n_missing = int(s.isna().sum())
        total = len(s)
    else:
        n_pos = n_neg = n_missing = 0
        total = 1

    return TargetBuildReport(
        target_column=target_col,
        n_positive=n_pos,
        n_negative=n_neg,
        positive_rate=n_pos / max(total, 1),
        n_missing=n_missing,
        missing_rate=n_missing / max(total, 1),
        risk_adjusted_available=risk_adj_available,
    )
