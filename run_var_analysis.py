"""
Standalone VaR analysis: load daily_equity.csv, compute Historical / Parametric / CVaR,
output comparison table and plot VaR over time vs actual daily P&L.

Usage:
    python run_var_analysis.py
    python run_var_analysis.py --equity output/backtests/daily_equity.csv --output output/research/var_report.csv
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import numpy as np
import pandas as pd

logging.getLogger("matplotlib").setLevel(logging.WARNING)

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from risk.var import conditional_var, historical_var, parametric_var


def _breach_count(returns: pd.Series, var_series: pd.Series) -> int:
    """Number of days where actual loss exceeded VaR (return < -VaR)."""
    common = returns.index.intersection(var_series.dropna().index)
    if len(common) == 0:
        return 0
    r = returns.loc[common]
    v = var_series.loc[common]
    return int((r < -v).sum())


def run(
    equity_path: str = "output/backtests/daily_equity.csv",
    report_path: str = "output/research/var_report.csv",
    plot_path: str | None = "output/research/var_plot.png",
    window: int = 252,
) -> None:
    if not os.path.isfile(equity_path):
        print(f"File not found: {equity_path}")
        return
    df = pd.read_csv(equity_path)
    if df.empty or "equity" not in df.columns:
        print("daily_equity must have an 'equity' column.")
        return
    if "date" in df.columns:
        df = df.sort_values("date")
        equity_series = df.set_index("date")["equity"]
    else:
        equity_series = df["equity"]
    returns = equity_series.pct_change().dropna()
    if len(returns) < 2:
        print("Not enough data for VaR.")
        return

    # VaR series (95% and 99%)
    hvar_95 = historical_var(returns, confidence=0.95, window=window)
    hvar_99 = historical_var(returns, confidence=0.99, window=window)
    pvar_95 = parametric_var(returns, confidence=0.95, window=window)
    pvar_99 = parametric_var(returns, confidence=0.99, window=window)
    cvar_95 = conditional_var(returns, confidence=0.95, window=window)

    # Last (current) values for table
    def last(s: pd.Series) -> float:
        d = s.dropna()
        return float(d.iloc[-1]) if len(d) > 0 else np.nan

    # Breach counts (95% VaR)
    b_hist = _breach_count(returns, hvar_95)
    b_param = _breach_count(returns, pvar_95)

    table = [
        {
            "Method": "Historical",
            "VaR_95_pct": last(hvar_95) * 100,
            "VaR_99_pct": last(hvar_99) * 100,
            "CVaR_95_pct": last(cvar_95) * 100,
            "Breaches_95": b_hist,
        },
        {
            "Method": "Parametric",
            "VaR_95_pct": last(pvar_95) * 100,
            "VaR_99_pct": last(pvar_99) * 100,
            "CVaR_95_pct": np.nan,
            "Breaches_95": b_param,
        },
    ]
    report_df = pd.DataFrame(table)

    os.makedirs(os.path.dirname(report_path) or ".", exist_ok=True)
    report_df.to_csv(report_path, index=False)
    print(f"VaR report saved to {report_path}")
    print()
    print("Method        | VaR 95%  | VaR 99%  | CVaR 95% | Breaches")
    print("-" * 55)
    for _, row in report_df.iterrows():
        v95 = f"{row['VaR_95_pct']:.2f}%" if pd.notna(row["VaR_95_pct"]) else " — "
        v99 = f"{row['VaR_99_pct']:.2f}%" if pd.notna(row["VaR_99_pct"]) else " — "
        cv = f"{row['CVaR_95_pct']:.2f}%" if pd.notna(row["CVaR_95_pct"]) else " — "
        print(f"{row['Method']:<13} | {v95:>8} | {v99:>8} | {cv:>8} | {int(row['Breaches_95'])}")

    if plot_path:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            os.makedirs(os.path.dirname(plot_path) or ".", exist_ok=True)
            common_idx = returns.index.intersection(hvar_95.dropna().index)
            if len(common_idx) > 0:
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(common_idx, -returns.loc[common_idx].values, label="Daily loss (actual)", alpha=0.7, color="gray")
                ax.plot(common_idx, hvar_95.loc[common_idx].values, label="Historical VaR (95%)", color="C0")
                ax.plot(common_idx, pvar_95.loc[common_idx].values, label="Parametric VaR (95%)", color="C1")
                ax.plot(common_idx, cvar_95.loc[common_idx].values, label="CVaR (95%)", color="C2")
                ax.set_xlabel("Date")
                ax.set_ylabel("Loss (%)")
                ax.set_title("VaR over time vs actual daily P&L (loss)")
                ax.legend(loc="upper right")
                ax.grid(True, alpha=0.3)
                fig.tight_layout()
                fig.savefig(plot_path, dpi=150)
                plt.close(fig)
                print(f"Plot saved to {plot_path}")
            else:
                print("No common index for plot; skipping.")
        except Exception as e:
            print(f"Plot skipped: {e}")


def main():
    p = argparse.ArgumentParser(description="VaR analysis from daily_equity.csv")
    p.add_argument("--equity", default="output/backtests/daily_equity.csv", help="Path to daily_equity CSV")
    p.add_argument("--output", default="output/research/var_report.csv", help="Path for var_report.csv")
    p.add_argument("--plot", default="output/research/var_plot.png", help="Path for VaR plot (use '' to skip)")
    p.add_argument("--window", type=int, default=252, help="Rolling window for VaR")
    args = p.parse_args()
    run(
        equity_path=args.equity,
        report_path=args.output,
        plot_path=args.plot or None,
        window=args.window,
    )


if __name__ == "__main__":
    main()
