"""Shared feature lookup utilities for the model_selection research stack.

Extracted from duplicated _get_family and _find_condition_column helpers across:
  - ic_diagnostics_engine.py:1660 (_get_family)
  - conditional_alpha_engine.py:212 (_get_family)
  - signal_decay_engine.py:1138 (_get_family)
  - pit_condition_engine.py:913 (_get_family)
  - feature_diversity_engine.py:169 (_get_family)
  - ic_diagnostics_engine.py:925 (_find_condition_column)
  - conditional_alpha_engine.py:817 (_find_condition_column)
"""
from __future__ import annotations

from typing import Any

import pandas as pd


# ── Feature family lookup ────────────────────────────────────────────────────
# Source: ic_diagnostics_engine.py:1660, conditional_alpha_engine.py:212,
#         signal_decay_engine.py:1138, pit_condition_engine.py:913,
#         feature_diversity_engine.py:169
#
# Two variants exist:
#   A. No try/except (ic_diagnostics, signal_decay) — import is at module level
#      anyway so failure would happen on import, not at call time.
#   B. With try/except returning "unknown" (conditional_alpha, pit_condition,
#      feature_diversity) — defensive wrapper.
#
# We expose the safer variant (B) so all callers get the same behavior.

def get_family(feature: str) -> str:
    """Look up a feature's family from FEATURE_SPECS.

    Args:
        feature: Feature name (e.g. "f_trend", "ret_20d").

    Returns:
        Family string from FEATURE_SPECS, or "unknown" if not found.
    """
    try:
        from model_selection.research_contract import FEATURE_SPECS
        spec = FEATURE_SPECS.get(feature)
        return spec.family if spec else "unknown"
    except Exception:
        return "unknown"


# ── Condition column discovery ───────────────────────────────────────────────
# Source: ic_diagnostics_engine.py:925, conditional_alpha_engine.py:817
#
# BEHAVIORAL DIFFERENCE: ic_diagnostics_engine has a larger col_map with 8
# entries (regime, volatility, liquidity, size, sector, spread, trend, beta).
# conditional_alpha_engine has only 3 entries (volatility, liquidity, size).
#
# We expose the FULL col_map from ic_diagnostics_engine. Callers that only
# need a subset will still work correctly since the function returns None
# for unknown conditions.

_CONDITION_COL_MAP: dict[str, list[str]] = {
    "regime": ["regime_label", "regime", "market_regime"],
    "volatility": ["rolling_vol_20", "vol_20_simple", "realised_vol_20d", "volatility"],
    "liquidity": ["adv_dollar_20", "adv_dollar", "turnover_pct_rank", "liquidity"],
    "size": ["market_cap", "log_market_cap", "size", "cap_size"],
    "sector": ["sector", "industry", "gics_sector"],
    "spread": ["spread_bps", "bid_ask_spread"],
    "trend": ["f_trend", "trend_signal"],
    "beta": ["capm_beta", "beta"],
}


def find_condition_column(df: pd.DataFrame, condition: str) -> str | None:
    """Find the column name for a given condition type.

    Searches df.columns for candidate column names associated with the
    condition type. Returns the first match, or None if no match found.

    Args:
        df: DataFrame to search.
        condition: Condition type (e.g. "volatility", "liquidity", "size").

    Returns:
        Column name if found, None otherwise.
    """
    for candidate in _CONDITION_COL_MAP.get(condition, [condition]):
        if candidate in df.columns:
            return candidate
    return None
