"""
StrategyEngine — cross-sectional ranking and daily signal selection.
"""

from __future__ import annotations

import pandas as pd

from .candidates import build_ranked_candidates


class StrategyEngine:
    """
    Converts per-ticker signal rows into ranked trade candidates for a day.
    """

    def generate_daily_signals(
        self,
        date: pd.Timestamp,
        all_ticker_signals: list,
        config,
        trading_days: list,
        day_index: int,
        regime: str,
    ) -> list[dict]:
        """
        all_ticker_signals: list of (ticker, sig_row) for this date (same shape as daily_signals[date]).
        Returns ranked candidate dicts ready for portfolio layer.
        """
        return build_ranked_candidates(
            date, all_ticker_signals, config, trading_days, day_index, regime
        )
