"""Pipeline stage: data loading.

Responsibility: Load feature matrix from the feature builder.
No research math — just I/O and basic validation.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

from research_pipeline.contract import ResearchContract

logger = logging.getLogger(__name__)


def load_feature_matrix(
    contract: ResearchContract,
    tickers: list[str],
) -> pd.DataFrame:
    """Load feature matrix from the feature builder cache.

    Delegates to the existing FeatureBuilder to preserve behavior.
    Returns the raw panel DataFrame.
    """
    from agents.weight_learning_agent.feature_builder import build_feature_matrix

    logger.info("Loading feature matrix: %s to %s, %d tickers",
                contract.start_date, contract.end_date, len(tickers))

    df = build_feature_matrix(
        tickers=tickers,
        start_date=contract.start_date,
        end_date=contract.end_date,
        **_feature_builder_kwargs(contract),
    )

    logger.info("Feature matrix loaded: %d rows, %d columns, %d tickers, %d dates",
                len(df), len(df.columns),
                df["ticker"].nunique() if "ticker" in df.columns else 0,
                df["date"].nunique() if "date" in df.columns else 0)

    return df


def _feature_builder_kwargs(contract: ResearchContract) -> dict[str, Any]:
    """Build kwargs for feature_builder from contract."""
    kwargs = {
        "cache_dir": contract.cache_dir,
        "cache_ttl_days": contract.cache_ttl_days,
    }
    if contract.persistence_filter:
        kwargs["persistence_filter"] = contract.persistence_filter
    return kwargs


def apply_feature_subset(
    df: pd.DataFrame,
    contract: ResearchContract,
    model_kind: str = "long",
) -> pd.DataFrame:
    """Filter feature matrix to the subset for a given model kind.

    If no subset is configured, returns all features.
    """
    if model_kind == "short" and contract.short_feature_subset:
        keep = set(contract.short_feature_subset)
    elif model_kind == "overlay" and contract.overlay_feature_subset:
        keep = set(contract.overlay_feature_subset)
    elif contract.feature_subset:
        keep = set(contract.feature_subset)
    else:
        return df

    meta_cols = {"date", "ticker", "sector", "regime_label"}
    target_cols = {c for c in df.columns if c.startswith("target_") or "forward" in c.lower()}
    risk_cols = {"daily_return", "adv_dollar_20", "realised_vol_20d"}

    cols_to_keep = meta_cols | target_cols | risk_cols | keep
    available = [c for c in df.columns if c in cols_to_keep]

    logger.info("Feature subset applied: %d of %d columns retained for %s",
                len(available), len(df.columns), model_kind)

    return df[available]
