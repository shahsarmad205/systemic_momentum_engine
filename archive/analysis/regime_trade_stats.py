from __future__ import annotations

"""
Regime Trade Statistics
=======================

Utility script to analyse walk-forward backtest trades by market regime.

For each trade it:
- Tags the market regime at entry using the existing `regime` column.
- Optionally maps finer regimes (Crisis/HighVol) into coarse buckets:
  'Bull', 'Bear', 'Sideways'.
- Computes, per regime:
  - Win rate
  - Average P&L
  - Number of long vs short trades
  - Loss rate

Usage (from project root):

    python -m analysis.regime_trade_stats

By default it reads `output/backtests/trades.csv`. You can override the path
by setting the TRADES_CSV environment variable.
"""

import os
from typing import Literal

import pandas as pd

RegimeBucket = Literal["Bull", "Bear", "Sideways", "Other"]


def _map_regime(raw: str | float) -> RegimeBucket:
    """Map raw regime label into coarse bucket."""
    if not isinstance(raw, str):
        return "Other"
    lab = raw.strip()
    if lab == "Bull":
        return "Bull"
    if lab in {"Bear", "Crisis", "HighVol"}:
        return "Bear"
    if lab == "Sideways":
        return "Sideways"
    return "Other"


def compute_regime_stats(trades: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-regime statistics:
    - win_rate
    - loss_rate
    - avg_pnl
    - n_trades
    - n_long
    - n_short
    """
    df = trades.copy()
    if "regime" not in df.columns:
        raise ValueError("trades.csv is missing required 'regime' column.")

    df["regime_bucket"] = df["regime"].map(_map_regime)

    if "pnl" not in df.columns:
        raise ValueError("trades.csv is missing required 'pnl' column.")
    if "direction" not in df.columns:
        raise ValueError("trades.csv is missing required 'direction' column.")

    df["is_win"] = df["pnl"] > 0
    df["is_loss"] = df["pnl"] < 0
    df["is_long"] = df["direction"] > 0
    df["is_short"] = df["direction"] < 0

    grouped = df.groupby("regime_bucket")

    stats = pd.DataFrame(
        {
            "n_trades": grouped.size(),
            "win_rate": grouped["is_win"].mean().fillna(0.0),
            "loss_rate": grouped["is_loss"].mean().fillna(0.0),
            "avg_pnl": grouped["pnl"].mean().fillna(0.0),
            "n_long": grouped["is_long"].sum(),
            "n_short": grouped["is_short"].sum(),
        }
    )

    # Ensure nice ordering of regimes
    order = ["Bull", "Bear", "Sideways", "Other"]
    stats = stats.reindex([r for r in order if r in stats.index])

    return stats


def main() -> None:
    trades_path = os.environ.get("TRADES_CSV", "output/backtests/trades.csv")
    if not os.path.isfile(trades_path):
        raise SystemExit(f"Trades CSV not found at '{trades_path}'. Run a backtest first.")

    trades = pd.read_csv(trades_path, parse_dates=["signal_date", "entry_date", "exit_date"])

    stats = compute_regime_stats(trades)

    if stats.empty:
        print("No trades found in trades CSV.")
        return

    # Pretty-print summary table
    df_disp = stats.copy()
    df_disp["win_rate"] = (df_disp["win_rate"] * 100).map("{:.1f}%".format)
    df_disp["loss_rate"] = (df_disp["loss_rate"] * 100).map("{:.1f}%".format)
    df_disp["avg_pnl"] = df_disp["avg_pnl"].map("{:.2f}".format)

    print("\nRegime performance summary:")
    print(df_disp.to_string())
    print()

    # Flag regimes with high loss rate
    bad = stats[stats["loss_rate"] > 0.60]
    if not bad.empty:
        print("⚠ Regimes with loss rate > 60%:")
        for regime, row in bad.iterrows():
            print(
                f"  - {regime}: loss_rate={row['loss_rate']:.1%}, "
                f"n_trades={int(row['n_trades'])}, avg_pnl={row['avg_pnl']:.2f}"
            )
    else:
        print("No regimes with loss rate > 60%.")


if __name__ == "__main__":
    main()

