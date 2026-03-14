"""
backtesting — Institutional-grade Quant Research Backtesting Engine
=====================================================================
"""

from .config import BacktestConfig, load_config
from .portfolio import Portfolio, Position
from .execution import ExecutionEngine
from .regime import MarketRegimeAgent
from .signals import SignalEngine
from .backtester import Backtester, BacktestResult
from utils.sectors import SECTOR_MAP

from .metrics import (
    compute_all_metrics,
    compute_win_rate,
    compute_average_return,
    compute_profit_factor,
    compute_sharpe_ratio,
    compute_sortino_ratio,
    compute_calmar_ratio,
    compute_max_drawdown,
    compute_signal_accuracy,
    compute_information_coefficient,
    compute_rank_ic,
    compute_trade_duration_stats,
)

from .analytics import (
    compute_ic_decay,
    compute_rank_ic_decay,
    best_ic_horizon,
    walk_forward_splits,
    run_walk_forward,
    parameter_grid,
    run_parameter_sweep,
    run_transaction_cost_sensitivity,
    run_execution_costs_sensitivity,
    DEFAULT_COST_SCENARIOS,
)

from .plotting import (
    plot_equity_curve,
    plot_ic_decay,
    plot_regime_returns,
    plot_position_count,
)
