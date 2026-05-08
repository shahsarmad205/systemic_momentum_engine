"""
PerformanceAnalyzer — metrics and reporting (delegates to backtesting.metrics + analytics).
"""

from __future__ import annotations

import os

import pandas as pd

from backtesting.analytics import compute_ic_decay as _compute_ic_decay
from backtesting.analytics import compute_rank_ic_decay as _compute_rank_ic_decay
from backtesting.metrics import compute_all_metrics


class PerformanceAnalyzer:
    """
    compute_performance_metrics(trades, daily_equity, config)
    generate_reports(...) optional CSV/plot hooks
    """

    @staticmethod
    def compute_performance_metrics(
        trades: pd.DataFrame,
        daily_equity: pd.DataFrame,
        config,
    ) -> dict:
        return compute_all_metrics(trades, daily_equity, config)

    @staticmethod
    def ic_decay(price_data: dict, signal_data: dict, lags: list[int]):
        return _compute_ic_decay(price_data, signal_data, lags)

    @staticmethod
    def rank_ic_decay(price_data: dict, signal_data: dict, lags: list[int]):
        return _compute_rank_ic_decay(price_data, signal_data, lags)

    @staticmethod
    def generate_reports(
        result_metrics: dict,
        output_dir: str = "output/experiments",
        basename: str = "report",
    ) -> str:
        """Write metrics to a text file; returns path."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"{basename}.txt")
        lines = [f"{k}: {v}" for k, v in sorted(result_metrics.items())]
        with open(path, "w") as fh:
            fh.write("\n".join(lines))
        return path
