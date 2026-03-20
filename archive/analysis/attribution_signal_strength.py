"""
Attribution Analysis by Signal Strength
======================================

This script analyzes how trade performance varies with signal strength.

Steps:
  - Load backtest trades from trades.csv
  - Use the `adjusted_score` column at entry as signal strength
  - Bin trades into quintiles by adjusted_score
  - For each bin, compute:
        * average return
        * win rate
        * profit factor
  - Plot a bar chart of average return by signal-strength bin
  - Save the plot to output/backtests/attribution_signal_strength.png

Usage (from project root):
  python analysis/attribution_signal_strength.py \
      --trades output/backtests/trades.csv \
      --output output/backtests/attribution_signal_strength.png
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import logging
import numpy as np
import pandas as pd
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Attribution analysis by signal strength (adjusted_score).")
    p.add_argument(
        "--trades",
        default="output/backtests/trades.csv",
        help="Trades CSV produced by backtest (default: output/backtests/trades.csv)",
    )
    p.add_argument(
        "--output",
        default="output/backtests/attribution_signal_strength.png",
        help="Path to save the attribution plot (default: output/backtests/attribution_signal_strength.png)",
    )
    p.add_argument(
        "--bins",
        type=int,
        default=5,
        help="Number of quantile bins for signal strength (default: 5 = quintiles)",
    )
    return p.parse_args()


def _load_trades(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Trades CSV not found: {path}")
    trades = pd.read_csv(path)
    if trades.empty:
        raise RuntimeError(f"Trades CSV is empty: {path}")
    if "adjusted_score" not in trades.columns:
        raise RuntimeError("Trades CSV must contain an 'adjusted_score' column for attribution.")
    if "return" not in trades.columns:
        raise RuntimeError("Trades CSV must contain a 'return' column.")
    return trades


def _bin_by_signal_strength(trades: pd.DataFrame, n_bins: int) -> pd.DataFrame:
    """
    Add a 'signal_bin' column to trades using quantile bins on adjusted_score.
    Falls back to fewer bins if there are not enough unique values.
    """
    df = trades.copy()
    scores = df["adjusted_score"].astype(float)

    # Ensure we have enough unique scores for the requested bins
    unique_scores = np.unique(scores[~np.isnan(scores)])
    bins = min(n_bins, max(1, len(unique_scores)))
    if bins <= 1:
        df["signal_bin"] = "all"
        return df

    try:
        df["signal_bin"] = pd.qcut(scores, q=bins, duplicates="drop")
    except ValueError:
        # Fallback: treat all trades as one bin
        df["signal_bin"] = "all"
    return df


def _profit_factor(group: pd.DataFrame) -> float:
    """
    Dollar-weighted profit factor for a group of trades.
    """
    if group.empty:
        return 0.0
    col = "pnl" if "pnl" in group.columns else "return"
    profits = group.loc[group[col] > 0, col].sum()
    losses = group.loc[group[col] < 0, col].abs().sum()
    if losses == 0:
        return float("inf") if profits > 0 else 0.0
    return float(profits / losses)


def compute_attribution(trades: pd.DataFrame, n_bins: int = 5) -> pd.DataFrame:
    """
    Return a summary table with metrics per signal-strength bin.
    """
    df = _bin_by_signal_strength(trades, n_bins=n_bins)
    if "signal_bin" not in df.columns:
        df["signal_bin"] = "all"

    groups = df.groupby("signal_bin", observed=True)
    rows = []
    for bin_label, g in groups:
        if g.empty:
            continue
        avg_ret = float(g["return"].mean())
        win_rate = float((g["return"] > 0).sum() / len(g))
        pf = _profit_factor(g)
        rows.append(
            {
                "signal_bin": str(bin_label),
                "n_trades": int(len(g)),
                "avg_return": avg_ret,
                "win_rate": win_rate,
                "profit_factor": pf,
            }
        )

    summary = pd.DataFrame(rows)
    # Order bins by increasing average adjusted_score if we used intervals,
    # otherwise keep as-is.
    try:
        # If labels are Interval-like, sort by left bound
        intervals = summary["signal_bin"].apply(lambda x: x.left if hasattr(x, "left") else np.nan)
        if intervals.notna().all():
            summary = summary.assign(_left=intervals).sort_values("_left").drop(columns="_left")
    except Exception:
        pass

    return summary


def plot_attribution(summary: pd.DataFrame, output_path: Path) -> None:
    """
    Plot bar chart of average return by signal-strength bin.
    """
    if summary.empty:
        raise RuntimeError("Attribution summary is empty; nothing to plot.")

    os.makedirs(output_path.parent, exist_ok=True)

    x = summary["signal_bin"].astype(str)
    y = summary["avg_return"]

    plt.figure(figsize=(8, 4))
    bars = plt.bar(x, y, color="tab:blue", alpha=0.8)
    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.title("Average Trade Return by Signal-Strength Bin")
    plt.xlabel("Signal-strength bin (adjusted_score quintiles)")
    plt.ylabel("Average return per trade")

    # Annotate bar tops with basic stats (win rate)
    win_rates = summary["win_rate"]
    for bar, wr in zip(bars, win_rates):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{wr:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()
    trades_path = (ROOT / args.trades).resolve()
    output_path = (ROOT / args.output).resolve()

    print(f"Loading trades from {trades_path} …")
    trades = _load_trades(trades_path)

    print("Computing attribution by signal-strength bin …")
    summary = compute_attribution(trades, n_bins=args.bins)

    if summary.empty:
        print("No attribution data to plot (summary is empty).")
        return

    print(summary.to_string(index=False, formatters={
        "avg_return": lambda v: f"{v:.4f}",
        "win_rate": lambda v: f"{v:.2%}",
        "profit_factor": lambda v: f"{v:.2f}" if np.isfinite(v) else "inf",
    }))

    print(f"\nPlotting attribution to {output_path} …")
    plot_attribution(summary, output_path)
    print("Done.")


if __name__ == "__main__":
    main()

