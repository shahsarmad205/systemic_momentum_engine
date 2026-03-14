from __future__ import annotations

"""
Feature IC Decay Analysis
=========================

For each feature in the weight-learning feature matrix, this script:

1. Computes Information Coefficient (Spearman rank correlation) between
   the feature and forward returns at lags:
      [1, 2, 3, 5, 10, 15, 20] days.
2. Plots IC vs lag for the top 10 features ranked by |IC| at lag=1.
3. Flags short-horizon features: IC(1) > 0.05 and |IC(5)| ~ 0.

Usage (from project root):

    python -m analysis.feature_ic_decay

It will:
- Build a feature matrix via weight-learning's `build_feature_matrix`
  (default 5-day holding window, 2018-01-01 → 2024-01-01).
- Download OHLCV data for the same tickers to compute forward returns.
"""

import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from agents.weight_learning_agent import build_feature_matrix
from config import get_effective_tickers
from utils.market_data import get_ohlcv


START_DATE = "2013-01-01"
END_DATE = "2024-01-01"
DEFAULT_HOLDING_DAYS = 5
LAGS = [1, 2, 3, 5, 10, 15, 20]


def _resolve_tickers() -> List[str]:
    """Mirror run_weight_learning's ticker resolution."""
    try:
        from main import TICKERS

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
    return get_effective_tickers([], fallback)


def _attach_forward_returns_by_lag(
    features_df: pd.DataFrame,
    lags: List[int],
) -> pd.DataFrame:
    """
    For each (ticker, date) row in features_df, attach forward returns
    at the requested lags, computed from OHLCV Close prices.
    """
    out_chunks: List[pd.DataFrame] = []
    max_lag = max(lags)

    for ticker in sorted(features_df["ticker"].unique()):
        sub = features_df[features_df["ticker"] == ticker].copy()
        if sub.empty:
            continue
        dates = pd.to_datetime(sub["date"].unique())
        dl_start = dates.min()
        # add some buffer for forward lags
        dl_end = dates.max() + pd.Timedelta(days=max_lag * 2)

        try:
            ohlcv = get_ohlcv(
                ticker,
                dl_start.strftime("%Y-%m-%d"),
                dl_end.strftime("%Y-%m-%d"),
                provider="yahoo",
                cache_dir="data/cache/ohlcv",
                use_cache=True,
                cache_ttl_days=0,
            )
        except Exception:
            ohlcv = None

        if ohlcv is None or ohlcv.empty or "Close" not in ohlcv.columns:
            continue

        close = ohlcv["Close"]

        # Align sub by date index
        sub_dates = pd.to_datetime(sub["date"])
        for lag in lags:
            fwd = close.shift(-lag) / close - 1.0
            aligned = fwd.reindex(sub_dates)
            sub[f"forward_{lag}d"] = aligned.values

        out_chunks.append(sub)

    if not out_chunks:
        return pd.DataFrame()

    return pd.concat(out_chunks, ignore_index=True)


def compute_feature_ic_decay(df: pd.DataFrame, lags: List[int]) -> pd.DataFrame:
    """
    Compute IC (Spearman rank correlation) for each feature vs
    forward returns at given lags.

    Returns:
        DataFrame indexed by feature with columns ic_1d, ic_2d, ...
    """
    # Identify numeric feature columns; exclude meta/target columns.
    exclude_prefixes = ("forward_",)
    exclude_cols = {
        "ticker",
        "date",
        "direction",
        "sector",
        "regime_label",
    }
    num_cols = []
    for col in df.columns:
        if col in exclude_cols:
            continue
        if any(col.startswith(p) for p in exclude_prefixes):
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            num_cols.append(col)

    records = []
    for feat in num_cols:
        row = {"feature": feat}
        x = df[feat].values
        for lag in lags:
            y_col = f"forward_{lag}d"
            if y_col not in df.columns:
                row[f"ic_{lag}d"] = np.nan
                continue
            y = df[y_col].values
            mask = np.isfinite(x) & np.isfinite(y)
            if mask.sum() <= 2:
                row[f"ic_{lag}d"] = np.nan
                continue
            try:
                rho, _ = spearmanr(x[mask], y[mask])
                val = float(rho) if np.isfinite(rho) else np.nan
            except Exception:
                val = np.nan
            row[f"ic_{lag}d"] = val
        records.append(row)

    ic_df = pd.DataFrame(records).set_index("feature")
    return ic_df


def plot_top_feature_ic_decay(ic_df: pd.DataFrame, lags: List[int], out_path: str, top_n: int = 10) -> None:
    """Plot IC vs lag for top-N features by |IC(1d)|."""
    lag1_col = f"ic_{lags[0]}d"
    if lag1_col not in ic_df.columns:
        return

    ic_df = ic_df.copy()
    ic_df["abs_ic_lag1"] = ic_df[lag1_col].abs()
    ic_df = ic_df.sort_values("abs_ic_lag1", ascending=False)
    top = ic_df.head(top_n)

    plt.figure(figsize=(8, 5))
    for feat, row in top.iterrows():
        vals = [row.get(f"ic_{lag}d", np.nan) for lag in lags]
        plt.plot(lags, vals, marker="o", label=feat)

    plt.axhline(0.0, color="black", linewidth=0.8)
    plt.xlabel("Forward lag (days)")
    plt.ylabel("Information Coefficient (Spearman vs forward return)")
    plt.title("IC vs Forward Lag — Top Features by |IC(1d)|")
    plt.legend(fontsize=8, loc="best")
    plt.grid(True, linestyle="--", alpha=0.3)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main() -> None:
    tickers = _resolve_tickers()
    print(f"Using {len(tickers)} tickers for IC decay analysis.")

    # 1. Build base feature matrix (uses default holding window 5d but that's not critical for IC by lag).
    print("Building feature matrix…")
    features_df = build_feature_matrix(
        tickers=tickers,
        start_date=START_DATE,
        end_date=END_DATE,
        holding_period=DEFAULT_HOLDING_DAYS,
    )
    if features_df.empty:
        print("[ERROR] Feature matrix is empty; aborting.")
        return

    print(
        f"  features: {len(features_df):,} rows × {features_df.shape[1]} cols "
        f"({features_df['ticker'].nunique()} tickers)"
    )

    # 2. Attach forward returns for multiple lags.
    print(f"Attaching forward returns for lags {LAGS}…")
    df_with_targets = _attach_forward_returns_by_lag(features_df, LAGS)
    if df_with_targets.empty:
        print("[ERROR] Could not attach forward returns; aborting.")
        return

    # 3. Compute feature IC decay.
    print("Computing IC decay per feature…")
    ic_df = compute_feature_ic_decay(df_with_targets, LAGS)

    out_dir = "output/research"
    os.makedirs(out_dir, exist_ok=True)
    ic_csv_path = os.path.join(out_dir, "feature_ic_decay.csv")
    ic_df.to_csv(ic_csv_path)
    print(f"  IC decay table written to {ic_csv_path}")

    # 4. Plot IC vs lag for top 10 features by |IC(1d)|.
    plot_path = os.path.join(out_dir, "feature_ic_decay_top10.png")
    plot_top_feature_ic_decay(ic_df, LAGS, plot_path, top_n=10)
    print(f"  IC decay plot (top 10 features) written to {plot_path}")

    # 5. Flag short-horizon features: IC(1d) > 0.05 and |IC(5d)| < 0.01.
    lag1_col = "ic_1d"
    lag5_col = "ic_5d"
    short_horizon = []
    if lag1_col in ic_df.columns and lag5_col in ic_df.columns:
        for feat, row in ic_df.iterrows():
            ic1 = row.get(lag1_col)
            ic5 = row.get(lag5_col)
            if ic1 is None or ic5 is None or np.isnan(ic1) or np.isnan(ic5):
                continue
            if ic1 > 0.05 and abs(ic5) < 0.01:
                short_horizon.append((feat, ic1, ic5))

    if short_horizon:
        print("\nShort-horizon features (IC(1d) > 0.05 and |IC(5d)| < 0.01):")
        for feat, ic1, ic5 in sorted(short_horizon, key=lambda x: -x[1]):
            print(f"  {feat:<30} IC(1d)={ic1:+.4f}  IC(5d)={ic5:+.4f}")
    else:
        print("\nNo features met the short-horizon criteria.")


if __name__ == "__main__":
    main()

