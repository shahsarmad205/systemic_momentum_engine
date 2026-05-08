# region imports
from __future__ import annotations
try:
    from AlgorithmImports import *
except ImportError:
    pass
# endregion
"""
Alternative and Fundamental Features
====================================
This module provides placeholders and initial implementations for features 
orthogonal to price momentum (SUE, Short Interest, Analyst Revisions).
"""

import pandas as pd

def compute_sue_score(df: pd.DataFrame, earnings_data: pd.DataFrame = None) -> pd.Series:
    """
    Standardized Unexpected Earnings (SUE) score.
    SUE = (Actual - Consensus) / StdDev(Surprises)
    
    Placeholder: returns zero if no data provided.
    """
    if earnings_data is None or earnings_data.empty:
        return pd.Series(0.0, index=df.index, name="f_sue_score")
    
    # Implementation logic:
    # 1. Align earnings dates with OHLCV index.
    # 2. Forward-fill the last known SUE score until the next announcement.
    return pd.Series(0.0, index=df.index, name="f_sue_score")

def compute_short_interest_features(df: pd.DataFrame, short_data: pd.DataFrame = None) -> pd.DataFrame:
    """
    Features derived from short interest (Short Float %, Short Interest Change).
    
    Placeholder: returns zeros if no data provided.
    """
    results = pd.DataFrame(index=df.index)
    results["f_short_float_pct"] = 0.0
    results["f_short_interest_change"] = 0.0
    
    if short_data is None or short_data.empty:
        return results
    
    return results

def compute_analyst_revision_momentum(df: pd.DataFrame, revision_data: pd.DataFrame = None) -> pd.Series:
    """
    Analyst revision momentum (change in consensus EPS estimates).
    
    Placeholder: returns zero if no data provided.
    """
    if revision_data is None or revision_data.empty:
        return pd.Series(0.0, index=df.index, name="f_analyst_revisions")
    
    return pd.Series(0.0, index=df.index, name="f_analyst_revisions")
