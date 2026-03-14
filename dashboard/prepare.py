"""
Prepare structured outputs for dashboards — thin adapters over DataFrames.
"""

from __future__ import annotations

import pandas as pd


def prepare_equity_series(daily_equity: pd.DataFrame) -> pd.DataFrame:
    """Return equity dataframe with date column if index-only."""
    if daily_equity.empty:
        return daily_equity
    out = daily_equity.copy()
    if "date" not in out.columns and out.index.name == "date":
        out = out.reset_index()
    return out


def prepare_trades_summary(trades: pd.DataFrame) -> dict:
    """Aggregate trade stats for dashboard binding."""
    if trades.empty:
        return {"n_trades": 0}
    return {
        "n_trades": len(trades),
        "win_rate": float((trades["return"] > 0).mean()) if "return" in trades else 0.0,
        "total_pnl": float(trades["pnl"].sum()) if "pnl" in trades else 0.0,
    }
