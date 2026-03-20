"""
Feature Stability Analysis
===========================

This script recomputes the full feature matrix used for weight learning
and analyzes how each feature's distribution changes over time.

For each numeric feature it:
  - Computes yearly cross-sectional mean and standard deviation
  - Plots these statistics over time (one PNG per feature)
  - Summarizes features whose moments are most unstable across years

Usage (from project root):
  python analysis/feature_stability.py

Options:
  --start, --end          : date range (default: 2018-01-01 → 2024-01-01)
  --holding-period        : forward return holding period (default: 20)
  --tickers ...           : optional explicit ticker list
  --output-dir            : where to write plots and summary (default: output/analysis)
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure project root is on path when run as python analysis/feature_stability.py
_root = Path(__file__).resolve().parents[1]
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

import numpy as np
import pandas as pd
import logging
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import matplotlib.pyplot as plt
import seaborn as sns

from agents.weight_learning_agent import build_feature_matrix


ROOT = Path(__file__).resolve().parents[1]


def _resolve_tickers(cli_tickers: list[str] | None) -> list[str]:
    """
    Resolve tickers similarly to run_weight_learning:
      - If CLI tickers are provided, use them.
      - Else, try main.TICKERS.
      - Else, fall back to a hard-coded large-cap + ETF list.
    """
    if cli_tickers:
        return cli_tickers
    try:
        from config import get_effective_tickers  # type: ignore
    except ImportError:
        get_effective_tickers = None

    fallback: list[str]
    try:
        from main import TICKERS  # type: ignore

        fallback = list(TICKERS)
    except Exception:
        fallback = [
            "AAPL",
            "MSFT",
            "NVDA",
            "AMZN",
            "GOOG",
            "META",
            "TSLA",
            "JPM",
            "V",
            "JNJ",
            "WMT",
            "PG",
            "XOM",
            "UNH",
            "BAC",
            "ABBV",
            "PFE",
            "KO",
            "PEP",
            "MRK",
            "AVGO",
            "COST",
            "TMO",
            "CSCO",
            "MCD",
            "NKE",
            "ADBE",
            "CRM",
            "AMD",
            "INTC",
            "ORCL",
            "IBM",
            "GS",
            "CAT",
            "SPY",
            "QQQ",
            "IWM",
            "DIA",
            "XLK",
            "VTI",
        ]

    if get_effective_tickers is not None:
        return get_effective_tickers([], fallback)
    return fallback


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Feature stability analysis for weight learning.")
    p.add_argument("--start", default="2018-01-01", help="Training start date (default: 2018-01-01)")
    p.add_argument("--end", default="2024-01-01", help="Training end date (default: 2024-01-01)")
    p.add_argument(
        "--holding-period",
        type=int,
        default=20,
        help="Forward return holding period in trading days (default: 20)",
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Optional explicit ticker list (space-separated)",
    )
    p.add_argument(
        "--output-dir",
        default="output/analysis",
        help="Directory to write plots and summary table (default: output/analysis)",
    )
    return p.parse_args()


def _load_feature_matrix(args: argparse.Namespace) -> pd.DataFrame:
    tickers = _resolve_tickers(args.tickers)
    print(f"Building feature matrix for {len(tickers)} tickers "
          f"from {args.start} to {args.end} (holding={args.holding_period}d)…")
    df = build_feature_matrix(
        tickers=tickers,
        start_date=args.start,
        end_date=args.end,
        holding_period=args.holding_period,
    )
    if df.empty:
        raise RuntimeError("Feature matrix is empty — cannot run stability analysis.")
    if "date" not in df.columns:
        raise RuntimeError("Feature matrix must contain a 'date' column.")
    df["date"] = pd.to_datetime(df["date"])
    return df


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
    # Numeric columns only, excluding obvious labels/targets
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    exclude = {"direction", "forward_return"}
    feature_cols = [c for c in numeric_cols if c not in exclude]
    return feature_cols


def compute_yearly_moments(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """
    Compute yearly cross-sectional mean and std for each feature.

    Returns a DataFrame indexed by year with MultiIndex columns:
      (feature, 'mean') and (feature, 'std').
    """
    df = df.copy()
    df["year"] = df["date"].dt.year
    grp = df.groupby("year")[feature_cols]

    means = grp.mean()
    stds = grp.std(ddof=0)

    # Stack into a single wide table with (feature, stat) columns
    moments = {}
    for col in feature_cols:
        moments[(col, "mean")] = means[col]
        moments[(col, "std")] = stds[col]
    out = pd.DataFrame(moments)
    out.columns = pd.MultiIndex.from_tuples(out.columns, names=["feature", "stat"])
    return out


def plot_feature_moments(moments: pd.DataFrame, out_dir: Path) -> None:
    """
    For each feature, plot its yearly mean and std over time.
    Writes one PNG per feature into out_dir.
    """
    sns.set_style("whitegrid")
    years = moments.index.values

    for feature in sorted(set(moments.columns.get_level_values("feature"))):
        sub = moments[feature]
        plt.figure(figsize=(8, 4))
        if "mean" in sub.columns:
            sns.lineplot(x=years, y=sub["mean"], marker="o", label="mean")
        if "std" in sub.columns:
            sns.lineplot(x=years, y=sub["std"], marker="o", label="std")
        plt.title(f"Feature stability: {feature}")
        plt.xlabel("Year")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        path = out_dir / f"feature_stability_{feature}.png"
        plt.savefig(path)
        plt.close()


def summarize_unstable_features(moments: pd.DataFrame) -> pd.DataFrame:
    """
    Compute variance of yearly means and stds for each feature.

    Returns a table with:
      feature, var_mean, var_std, combined_var
    sorted by combined_var descending.
    """
    features = sorted(set(moments.columns.get_level_values("feature")))
    rows = []
    for feat in features:
        sub = moments[feat]
        m = sub.get("mean")
        s = sub.get("std")
        var_mean = float(m.var(ddof=0)) if m is not None else 0.0
        var_std = float(s.var(ddof=0)) if s is not None else 0.0
        rows.append(
            {
                "feature": feat,
                "var_mean": var_mean,
                "var_std": var_std,
                "combined_var": var_mean + var_std,
            }
        )
    summary = pd.DataFrame(rows).sort_values("combined_var", ascending=False)
    return summary


def main() -> None:
    args = parse_args()
    out_dir = (ROOT / args.output_dir).resolve()
    os.makedirs(out_dir, exist_ok=True)

    df = _load_feature_matrix(args)
    feature_cols = _select_feature_columns(df)
    print(f"Using {len(feature_cols)} numeric features for stability analysis.")

    moments = compute_yearly_moments(df, feature_cols)

    print("Plotting feature moments (mean/std by year)…")
    plot_feature_moments(moments, out_dir)

    summary = summarize_unstable_features(moments)
    summary_path = out_dir / "feature_stability_summary.csv"
    summary.to_csv(summary_path, index=False)

    print("\nTop 10 most unstable features (by variance of moments):")
    print(summary.head(10).to_string(index=False))
    print(f"\nFull summary saved to: {summary_path}")
    print(f"Per-feature plots written under: {out_dir}")


if __name__ == "__main__":
    main()

