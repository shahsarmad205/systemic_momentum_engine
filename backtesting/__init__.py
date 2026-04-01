"""
Backtesting package initialization.
Exports core engine, signal, configuration, and analysis utilities.
"""

from __future__ import annotations

# Core Engine and Signals
from .backtester import Backtester, BacktestResult
from .signals import SignalEngine
from .engine import StrategyEngine

# Configuration
from .config import BacktestConfig, load_config

# Analytics and Metrics
from .analytics import (
    best_ic_horizon,
    compute_ic_decay,
    compute_rank_ic_decay,
    run_execution_costs_sensitivity,
    run_transaction_cost_sensitivity,
    run_walk_forward
)
from .metrics import (
    compute_all_metrics,
    compute_win_rate,
    compute_average_return,
    compute_profit_factor,
    compute_sharpe_ratio
)

# Plotting
from .plotting import (
    plot_equity_curve,
    plot_ic_decay,
    plot_position_count,
    plot_regime_returns
)

# Strategy candidates (Flattened from strategy/)
from .candidates import build_ranked_candidates
from .cross_sectional import build_cross_sectional_candidates
from .regime_multipliers import get_multiplier, load_regime_multipliers

__all__ = [
    "Backtester",
    "BacktestResult",
    "SignalEngine",
    "StrategyEngine",
    "BacktestConfig",
    "load_config",
    "best_ic_horizon",
    "compute_ic_decay",
    "compute_rank_ic_decay",
    "run_execution_costs_sensitivity",
    "run_transaction_cost_sensitivity",
    "run_walk_forward",
    "compute_all_metrics",
    "compute_win_rate",
    "compute_average_return",
    "compute_profit_factor",
    "compute_sharpe_ratio",
    "plot_equity_curve",
    "plot_ic_decay",
    "plot_position_count",
    "plot_regime_returns",
    "build_ranked_candidates",
    "build_cross_sectional_candidates",
    "get_multiplier",
    "load_regime_multipliers",
]
