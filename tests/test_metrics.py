import math

import numpy as np
import pandas as pd

from backtesting.metrics import (
    compute_win_rate,
    compute_profit_factor,
    compute_max_drawdown,
    compute_drawdown_stats,
    bootstrap_performance_cis,
    compute_turnover,
)


def test_basic_trade_metrics():
    trades = pd.DataFrame(
        {
            "return": [0.10, -0.05, 0.0],
            "pnl": [10.0, -5.0, 0.0],
        }
    )

    win_rate = compute_win_rate(trades)
    pf = compute_profit_factor(trades)

    assert math.isclose(win_rate, 1.0 / 3.0, rel_tol=1e-6)
    assert math.isclose(pf, 2.0, rel_tol=1e-6)


def test_drawdown_depth_and_duration():
    # Simple equity curve: 100 → 120 → 90 → 95 → 130
    equity = pd.DataFrame(
        {
            "date": pd.date_range("2020-01-01", periods=5, freq="D"),
            "equity": [100.0, 120.0, 90.0, 95.0, 130.0],
        }
    )

    max_dd = compute_max_drawdown(equity)
    dd_stats = compute_drawdown_stats(equity)

    # Max drawdown is from 120 → 90 : (90-120)/120 = -0.25
    assert math.isclose(max_dd, -0.25, rel_tol=1e-6)
    assert dd_stats["max_drawdown_duration"] > 0


def test_bootstrap_performance_cis_basic():
    # Mildly positive, low-vol daily returns
    rng = np.random.default_rng(42)
    rets = pd.Series(0.001 + 0.0005 * rng.standard_normal(252))

    ci = bootstrap_performance_cis(rets, n_bootstrap=200)

    for key in (
        "sharpe_ci_low",
        "sharpe_ci_high",
        "sortino_ci_low",
        "sortino_ci_high",
        "calmar_ci_low",
        "calmar_ci_high",
    ):
        assert key in ci
        assert np.isfinite(ci[key])

    assert ci["sharpe_ci_high"] >= ci["sharpe_ci_low"]


def test_turnover_computation():
    trades = pd.DataFrame(
        {
            "entry_date": pd.to_datetime(["2020-01-01", "2020-01-02"]),
            "exit_date": pd.to_datetime(["2020-01-03", "2020-01-04"]),
            "position_size": [10_000.0, 20_000.0],
        }
    )
    initial_capital = 100_000.0
    avg_cost_bps = 10.0

    metrics = compute_turnover(trades, initial_capital, avg_cost_bps)

    assert metrics["avg_daily_turnover"] > 0.0
    assert metrics["annualised_turnover"] > 0.0
    assert metrics["turnover_cost_drag_bps"] >= 0.0

