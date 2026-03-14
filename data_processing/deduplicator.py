"""
Deduplicator — remove duplicate rows and duplicate timestamps.
"""

from __future__ import annotations

import pandas as pd


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove duplicate rows and duplicate timestamps, keeping last occurrence per timestamp.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a datetime index (e.g. from basic_clean).

    Returns
    -------
    pd.DataFrame
        Copy with duplicate rows removed and one row per index value (last kept).
    """
    if df is None or df.empty:
        return df.copy() if df is not None else pd.DataFrame()

    out = df.copy()

    # Drop full duplicate rows
    out = out.drop_duplicates()

    # For duplicate timestamps, keep last occurrence
    if out.index.duplicated(keep="last").any():
        out = out[~out.index.duplicated(keep="last")]

    return out
