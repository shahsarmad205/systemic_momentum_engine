"""
Monte Carlo portfolio simulation: bootstrap resample historical trades
to simulate distribution of portfolio outcomes.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.getLogger("matplotlib").setLevel(logging.WARNING)


def simulate_portfolio(
    trades_df: pd.DataFrame,
    n_simulations: int = 1000,
    resample_method: str = "bootstrap",
    seed: int | None = None,
) -> pd.DataFrame:
    """
    Bootstrap resample historical trades to simulate distribution of portfolio outcomes.

    Each simulation draws (with replacement) a random sample of trades of the same
    size as the original, then computes total_return, sharpe (annualised from
    trade returns), and max_drawdown (from cumulative PnL path).
    Returns DataFrame with columns: total_return, sharpe, max_drawdown (one row per simulation).
    """
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=["total_return", "sharpe", "max_drawdown"])
    n = len(trades_df)
    if n < 2:
        return pd.DataFrame(columns=["total_return", "sharpe", "max_drawdown"])
    rng = np.random.default_rng(seed)
    ret_col = "return" if "return" in trades_df.columns else "net_return"
    if ret_col not in trades_df.columns:
        if "pnl" in trades_df.columns and "position_size" in trades_df.columns:
            trades_df = trades_df.copy()
            trades_df["_ret"] = trades_df["pnl"] / trades_df["position_size"].replace(0, np.nan)
            ret_col = "_ret"
        else:
            return pd.DataFrame(columns=["total_return", "sharpe", "max_drawdown"])
    returns = trades_df[ret_col].astype(float).values
    results = []
    for _ in range(n_simulations):
        idx = rng.integers(0, n, size=n)
        sample_ret = returns[idx]
        total_return = float(np.prod(1.0 + sample_ret) - 1.0)
        mean_r = np.mean(sample_ret)
        std_r = np.std(sample_ret)
        sharpe = (mean_r / std_r * np.sqrt(252)) if std_r > 1e-12 else 0.0
        cum = np.cumprod(1.0 + sample_ret)
        peak = np.maximum.accumulate(cum)
        dd = (cum - peak) / peak
        max_drawdown = float(np.min(dd)) if len(dd) else 0.0
        results.append({"total_return": total_return, "sharpe": sharpe, "max_drawdown": max_drawdown})
    return pd.DataFrame(results)


def plot_simulation_results(
    simulation_results: pd.DataFrame,
    actual_metrics: dict,
    save_path: str = "output/research/monte_carlo_portfolio.png",
) -> str:
    """
    Plot distribution of simulated outcomes vs actual.

    Shows: histogram of Sharpe ratios, return distribution, drawdown distribution
    with actual result marked. Returns save_path.
    """
    if simulation_results is None or simulation_results.empty:
        return save_path
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return save_path
    import os
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    actual_sharpe = actual_metrics.get("sharpe_ratio")
    actual_return = actual_metrics.get("total_return")
    actual_dd = actual_metrics.get("max_drawdown")

    axes[0].hist(simulation_results["sharpe"], bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    if actual_sharpe is not None and np.isfinite(actual_sharpe):
        axes[0].axvline(actual_sharpe, color="red", linestyle="--", linewidth=2, label="Actual")
    axes[0].set_xlabel("Sharpe ratio")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Sharpe distribution")
    axes[0].legend()

    axes[1].hist(simulation_results["total_return"], bins=50, alpha=0.7, color="green", edgecolor="white")
    if actual_return is not None and np.isfinite(actual_return):
        axes[1].axvline(actual_return, color="red", linestyle="--", linewidth=2, label="Actual")
    axes[1].set_xlabel("Total return")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Return distribution")
    axes[1].legend()

    axes[2].hist(simulation_results["max_drawdown"], bins=50, alpha=0.7, color="coral", edgecolor="white")
    if actual_dd is not None and np.isfinite(actual_dd):
        axes[2].axvline(actual_dd, color="red", linestyle="--", linewidth=2, label="Actual")
    axes[2].set_xlabel("Max drawdown")
    axes[2].set_ylabel("Count")
    axes[2].set_title("Drawdown distribution")
    axes[2].legend()

    fig.suptitle("Monte Carlo portfolio simulation vs actual", fontsize=12)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    return save_path
