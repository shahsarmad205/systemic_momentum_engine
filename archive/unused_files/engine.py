"""
PortfolioEngine — wraps Portfolio + execution-aware open/close.
"""

from __future__ import annotations

import pandas as pd

from backtesting.portfolio import Portfolio, Position


class PortfolioEngine:
    """
    Exposes open_position / close_position / update_portfolio (record_equity).
    Delegates to existing Portfolio; execution slippage stays in Backtester for now
    to avoid duplicating apply_entry_slippage logic.
    """

    def __init__(self, initial_capital: float, max_positions: int):
        self._portfolio = Portfolio(initial_capital, max_positions)

    @property
    def portfolio(self) -> Portfolio:
        return self._portfolio

    def open_position(self, **kwargs) -> Position | None:
        return self._portfolio.open_position(**kwargs)

    def close_position(self, pos: Position, exit_date: pd.Timestamp, exit_price: float, exit_cost: float) -> dict:
        return self._portfolio.close_position(pos, exit_date, exit_price, exit_cost)

    def update_portfolio(self, date: pd.Timestamp, regime: str = "") -> None:
        self._portfolio.record_equity(date, regime)
