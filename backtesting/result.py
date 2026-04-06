"""
BacktestResult — Lightweight container for a completed backtest run.

Extracted from backtester.py to keep the result data-structure independently
importable (e.g. for testing, analytics scripts, and Monte Carlo runners
that only need to consume results, not run the full backtester).
"""

from __future__ import annotations

import pandas as pd

from .config import BacktestConfig


class BacktestResult:
    """Holds everything produced by a single backtest run."""

    __slots__ = ("trades", "daily_equity", "metrics", "config",
                 "price_data", "signal_data", "regime_data", "position_sizing_comparison")

    def __init__(
        self,
        trades: pd.DataFrame,
        daily_equity: pd.DataFrame,
        metrics: dict,
        config: BacktestConfig,
    ):
        self.trades = trades
        self.daily_equity = daily_equity
        self.metrics = metrics
        self.config = config
        self.price_data: dict[str, pd.DataFrame] = {}
        self.signal_data: dict[str, pd.DataFrame] = {}
        self.regime_data: dict[pd.Timestamp, str] = {}
        self.position_sizing_comparison: dict | None = None
