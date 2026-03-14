"""
Cleaner — standardize column names, datetime index, and numeric types.
"""

from __future__ import annotations

import pandas as pd


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize a market data DataFrame for downstream processing.

    - Standardize column names: lowercase, strip leading/trailing spaces.
    - Convert date-like column to datetime index if present.
    - Sort by datetime index.
    - Coerce OHLCV columns to numeric.

    Parameters
    ----------
    df : pd.DataFrame
        Raw market data (e.g. OHLCV from a provider).

    Returns
    -------
    pd.DataFrame
        Cleaned copy with standardized column names and index.
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()

    # Standardize column names: lowercase, strip spaces
    out.columns = [str(c).strip().lower() for c in out.columns]

    # Ensure datetime index: use 'date' column, or convert existing index
    if isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index, errors="coerce")
    elif out.index.name and str(out.index.name).strip().lower() == "date":
        out.index = pd.to_datetime(out.index, errors="coerce")
    elif "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
        out = out.dropna(subset=["date"])
        out = out.set_index("date")
    else:
        out.index = pd.to_datetime(out.index, errors="coerce")

    # Drop rows with invalid index
    out = out[out.index.notna()]
    if out.empty:
        return out

    # Sort by datetime index
    out = out.sort_index()

    # Ensure numeric columns are numeric (OHLCV + any other numeric)
    ohlcv_lower = ["open", "high", "low", "close", "volume"]
    for col in out.columns:
        if col in ohlcv_lower or out[col].dtype.kind not in ("i", "u", "f"):
            out[col] = pd.to_numeric(out[col], errors="coerce")

    return out
