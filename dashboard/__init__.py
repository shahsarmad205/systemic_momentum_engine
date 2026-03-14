"""
Dashboard layer — prepare data for visualization (no UI here).
"""

from .prepare import prepare_equity_series, prepare_trades_summary

__all__ = ["prepare_equity_series", "prepare_trades_summary"]
