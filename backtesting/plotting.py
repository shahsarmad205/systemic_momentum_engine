"""
Backtest Visualization
========================
Equity curve, drawdown, regime bands, IC decay, and regime-return plots.
Decoupled from the backtester — accepts DataFrames and dicts.
"""

import os
from itertools import groupby
import logging

import numpy as np
import pandas as pd
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


BACKTESTS_DIR = os.path.join("output", "backtests")

REGIME_COLORS = {
    "Bull": "#2ecc71",
    "Bear": "#e74c3c",
    "Sideways": "#f39c12",
    "Crisis": "#8e44ad",
}


# ------------------------------------------------------------------
# Equity Curve + Drawdown (with regime shading)
# ------------------------------------------------------------------

def plot_equity_curve(
    daily_equity: pd.DataFrame,
    initial_capital: float = 100_000,
    save_path: str | None = None,
) -> str:
    if save_path is None:
        os.makedirs(BACKTESTS_DIR, exist_ok=True)
        save_path = os.path.join(BACKTESTS_DIR, "equity_curve.png")
    else:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)

    dates = daily_equity["date"]
    equity = daily_equity["equity"]
    peak = equity.expanding().max()
    drawdown = (equity - peak) / peak

    fig, (ax_eq, ax_dd) = plt.subplots(
        2, 1, figsize=(15, 9),
        gridspec_kw={"height_ratios": [3, 1]},
        sharex=True,
    )
    fig.suptitle(
        f"Portfolio Equity Curve  —  start ${initial_capital:,.0f}",
        fontsize=14, fontweight="bold", color="#2c3e50",
    )

    # --- regime background bands ---
    if "regime" in daily_equity.columns:
        _draw_regime_bands(ax_eq, daily_equity, equity.min() * 0.97, equity.max() * 1.03)

    # --- equity line ---
    ax_eq.plot(dates, equity, lw=1.4, color="#2c3e50", label="Portfolio")
    ax_eq.plot(dates, peak, lw=0.7, color="#3498db", ls="--", alpha=0.5, label="Peak")
    ax_eq.fill_between(dates, equity, peak, where=(equity < peak),
                       color="#e74c3c", alpha=0.12, label="Drawdown zone")
    ax_eq.axhline(initial_capital, color="#7f8c8d", lw=0.5, ls=":")
    ax_eq.set_ylabel("Portfolio Value ($)")
    ax_eq.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"${x:,.0f}"))
    ax_eq.legend(loc="upper left", fontsize=8)
    ax_eq.grid(True, alpha=0.25)

    # --- drawdown panel ---
    ax_dd.fill_between(dates, drawdown, 0, color="#e74c3c", alpha=0.4)
    ax_dd.set_ylabel("Drawdown")
    ax_dd.set_xlabel("Date")
    ax_dd.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    ax_dd.grid(True, alpha=0.25)

    fig.autofmt_xdate(rotation=30)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


def _draw_regime_bands(ax, daily_equity, ymin, ymax):
    """Shade background by market regime (Bull/Bear/Sideways/Crisis)."""
    dates = daily_equity["date"].tolist()
    regimes = daily_equity["regime"].tolist()

    for regime, grp in groupby(zip(dates, regimes), key=lambda x: x[1]):
        grp_list = list(grp)
        color = REGIME_COLORS.get(regime, "#bdc3c7")
        ax.axvspan(grp_list[0][0], grp_list[-1][0], alpha=0.06, color=color)


# ------------------------------------------------------------------
# IC Decay bar chart
# ------------------------------------------------------------------

def plot_ic_decay(
    ic_values: list[float],
    lags: list[int],
    title: str = "IC Decay Analysis",
    save_path: str | None = None,
) -> str:
    if save_path is None:
        os.makedirs(BACKTESTS_DIR, exist_ok=True)
        save_path = os.path.join(BACKTESTS_DIR, "ic_decay.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in ic_values]
    ax.bar(range(len(lags)), ic_values, tick_label=[str(l) for l in lags],
           color=colors, alpha=0.75, edgecolor="#2c3e50", linewidth=0.5)
    ax.axhline(0, color="#7f8c8d", lw=0.6, ls=":")
    ax.set_xlabel("Forward Horizon (trading days)")
    ax.set_ylabel("Information Coefficient")
    ax.set_title(title, fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------
# Average return by market regime
# ------------------------------------------------------------------

def plot_regime_returns(
    trades: pd.DataFrame,
    save_path: str | None = None,
) -> str | None:
    if trades.empty or "regime" not in trades.columns:
        return None
    if save_path is None:
        os.makedirs(BACKTESTS_DIR, exist_ok=True)
        save_path = os.path.join(BACKTESTS_DIR, "regime_returns.png")

    grouped = trades.groupby("regime")["return"].agg(["mean", "count", "std"])

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(
        grouped.index,
        grouped["mean"],
        color=[REGIME_COLORS.get(r, "#95a5a6") for r in grouped.index],
        alpha=0.8, edgecolor="#2c3e50", linewidth=0.5,
    )
    ax.axhline(0, color="#7f8c8d", lw=0.6, ls=":")
    ax.set_ylabel("Average Return per Trade")
    ax.set_title("Performance by Market Regime", fontweight="bold")
    ax.grid(True, alpha=0.25, axis="y")

    for bar, (_, row) in zip(bars, grouped.iterrows()):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.0005,
            f"n={int(row['count'])}",
            ha="center", fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path


# ------------------------------------------------------------------
# Position-count over time
# ------------------------------------------------------------------

def plot_position_count(
    daily_equity: pd.DataFrame,
    save_path: str | None = None,
) -> str | None:
    if daily_equity.empty or "n_positions" not in daily_equity.columns:
        return None
    if save_path is None:
        os.makedirs(BACKTESTS_DIR, exist_ok=True)
        save_path = os.path.join(BACKTESTS_DIR, "position_count.png")

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(daily_equity["date"], daily_equity["n_positions"],
                    color="#3498db", alpha=0.35)
    ax.plot(daily_equity["date"], daily_equity["n_positions"],
            lw=0.7, color="#2c3e50")
    ax.set_ylabel("Open Positions")
    ax.set_xlabel("Date")
    ax.set_title("Concurrent Positions Over Time", fontweight="bold")
    ax.grid(True, alpha=0.25)

    fig.autofmt_xdate(rotation=30)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return save_path
