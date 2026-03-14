"""
Pipeline — run clean, deduplicate, validate; return canonical OHLCV DataFrame.
"""

from __future__ import annotations

import pandas as pd

from .cleaner import basic_clean
from .deduplicator import remove_duplicates
from .validator import validate_ohlcv

# Canonical column names expected by the rest of the project
CANONICAL_OHLCV = ["Open", "High", "Low", "Close", "Volume"]


def process_market_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Run the full cleaning pipeline on raw market OHLCV data.

    Order: basic_clean -> remove_duplicates -> validate_ohlcv.
    Output columns are renamed to Open, High, Low, Close, Volume for
    compatibility with the rest of the codebase.

    Parameters
    ----------
    df : pd.DataFrame
        Raw OHLCV data (e.g. from get_ohlcv or MarketDataLoader).

    Returns
    -------
    pd.DataFrame
        Cleaned, deduplicated, validated DataFrame with DatetimeIndex
        and columns Open, High, Low, Close, Volume.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    # 1. Standardize names, datetime index, sort, numeric
    cleaned = basic_clean(df)
    if cleaned.empty:
        return cleaned

    # 2. Remove duplicate rows and duplicate timestamps (keep last)
    cleaned = remove_duplicates(cleaned)
    if cleaned.empty:
        return cleaned

    # 3. Validate OHLCV structure and value constraints
    cleaned = validate_ohlcv(cleaned)

    # Restore canonical column names for project compatibility
    lower_to_canonical = {
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    }
    rename = {c: lower_to_canonical[c] for c in lower_to_canonical if c in cleaned.columns}
    if rename:
        cleaned = cleaned.rename(columns=rename)

    return cleaned
