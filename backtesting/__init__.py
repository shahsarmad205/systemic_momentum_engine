"""
backtesting — Institutional-grade Quant Research Backtesting Engine
=====================================================================
"""

from utils.sectors import SECTOR_MAP

from .analytics import (
    DEFAULT_COST_SCENARIOS,
    best_ic_horizon,
    compute_ic_decay,
    compute_rank_ic_decay,
    parameter_grid,
    run_execution_costs_sensitivity,
    run_parameter_sweep,
    run_transaction_cost_sensitivity,
    run_walk_forward,
    walk_forward_splits,
)
from .backtester import Backtester, BacktestResult
from .config import BacktestConfig, load_config
from .execution import ExecutionEngine
from .metrics import (
    compute_all_metrics,
    compute_average_return,
    compute_calmar_ratio,
    compute_information_coefficient,
    compute_max_drawdown,
    compute_profit_factor,
    compute_rank_ic,
    compute_sharpe_ratio,
    compute_signal_accuracy,
    compute_sortino_ratio,
    compute_trade_duration_stats,
    compute_win_rate,
)
from .plotting import (
    plot_equity_curve,
    plot_ic_decay,
    plot_position_count,
    plot_regime_returns,
)
from .portfolio import Portfolio, Position
from .regime import MarketRegimeAgent
from .signals import SignalEngine
