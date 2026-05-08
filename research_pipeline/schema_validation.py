"""Pipeline stage: schema validation.

Responsibility: Validate the feature matrix has required columns,
correct dtypes, and no leakage columns in the feature set.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from model_selection.research_contract import (
    ALPHA_METADATA_COLUMNS,
    TARGET_COLUMNS,
    RISK_EXECUTION_COLUMNS,
    SHORT_TARGET_COLUMNS,
    is_model_feature_column,
)

logger = logging.getLogger(__name__)


@dataclass
class SchemaValidationReport:
    n_rows: int
    n_columns: int
    n_tickers: int
    n_dates: int
    n_features: int
    n_targets: int
    n_blocked_columns: int
    blocked_columns: tuple[str, ...]
    missing_required: tuple[str, ...]
    non_numeric_features: tuple[str, ...]
    null_rate_by_feature: dict[str, float]


REQUIRED_COLUMNS = {"date", "ticker"}


def validate_schema(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
) -> SchemaValidationReport:
    """Validate feature matrix schema and identify blocked columns."""
    blocked = _find_blocked_columns(df, feature_columns)
    missing = _find_missing_required(df)
    non_numeric = _find_non_numeric(df, feature_columns)
    null_rates = _compute_null_rates(df, feature_columns)

    feature_cols = [c for c in feature_columns if c in df.columns and c not in blocked]

    report = SchemaValidationReport(
        n_rows=len(df),
        n_columns=len(df.columns),
        n_tickers=df["ticker"].nunique() if "ticker" in df.columns else 0,
        n_dates=df["date"].nunique() if "date" in df.columns else 0,
        n_features=len(feature_cols),
        n_targets=len([c for c in df.columns if c in TARGET_COLUMNS]),
        n_blocked_columns=len(blocked),
        blocked_columns=tuple(blocked),
        missing_required=tuple(missing),
        non_numeric_features=tuple(non_numeric),
        null_rate_by_feature=null_rates,
    )

    if report.missing_required:
        logger.error("Missing required columns: %s", report.missing_required)
    if report.n_blocked_columns > 0:
        logger.info("Blocked %d leakage columns: %s",
                     report.n_blocked_columns, report.blocked_columns[:10])

    return report


def _find_blocked_columns(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
) -> list[str]:
    """Find columns that should not be model features (leakage protection)."""
    blocked = []
    for col in feature_columns:
        if col not in df.columns:
            continue
        if not is_model_feature_column(col, df[col]):
            blocked.append(col)
    return blocked


def _find_missing_required(df: pd.DataFrame) -> list[str]:
    """Find required columns that are missing."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def _find_non_numeric(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
) -> list[str]:
    """Find feature columns that are not numeric."""
    non_numeric = []
    for col in feature_columns:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            non_numeric.append(col)
    return non_numeric


def _compute_null_rates(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
) -> dict[str, float]:
    """Compute null rate for each feature column."""
    rates = {}
    for col in feature_columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            rates[col] = float(df[col].isna().mean())
    return rates
