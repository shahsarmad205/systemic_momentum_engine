"""
Validator — validate OHLCV structure and value constraints.
"""

from __future__ import annotations

import pandas as pd

REQUIRED_OHLCV = ["open", "high", "low", "close", "volume"]


class OHLCVValidationError(Exception):
    """Raised when OHLCV validation fails."""

    pass


def validate_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate OHLCV structure and constraints.

    - Ensures required columns exist: open, high, low, close, volume.
    - Ensures prices (open, high, low, close) > 0.
    - Ensures volume >= 0.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with lowercase OHLCV columns (e.g. from basic_clean).

    Returns
    -------
    pd.DataFrame
        The same DataFrame if valid.

    Raises
    ------
    OHLCVValidationError
        If required columns are missing or value constraints are violated.
    """
    if df is None:
        raise OHLCVValidationError("DataFrame is None")

    if df.empty:
        return df.copy()

    missing = [c for c in REQUIRED_OHLCV if c not in df.columns]
    if missing:
        raise OHLCVValidationError(
            f"Missing required OHLCV columns: {missing}. "
            f"Required: {REQUIRED_OHLCV}. Found: {list(df.columns)}."
        )

    price_cols = ["open", "high", "low", "close"]
    for col in price_cols:
        if col not in df.columns:
            continue
        invalid = (df[col] <= 0).any()
        if invalid:
            bad = (df[col] <= 0).sum()
            raise OHLCVValidationError(
                f"Column '{col}' must have values > 0. Found {bad} non-positive value(s)."
            )

    if "volume" in df.columns:
        invalid = (df["volume"] < 0).any()
        if invalid:
            bad = (df["volume"] < 0).sum()
            raise OHLCVValidationError(
                f"Column 'volume' must have values >= 0. Found {bad} negative value(s)."
            )

    return df
