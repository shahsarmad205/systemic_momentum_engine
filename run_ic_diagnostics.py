#!/usr/bin/env python3
"""
IC Decay Diagnostic — runs in < 30 seconds using cached data.

Reports:
  • IC at 1, 3, 5, 10, 20-day forward horizons for the current best_long_model
  • Rank-IC (Spearman) at each horizon — more robust to outliers
  • Identifies the horizon where predictive power peaks

Usage:
  python run_ic_diagnostics.py
  python run_ic_diagnostics.py --tickers AAPL MSFT NVDA AMZN GOOGL --start 2020-01-01
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

HORIZONS = [1, 3, 5, 10, 20]
DEFAULT_MODEL = "output/models/best_long_model.pkl"
DEFAULT_TICKERS_FILE = "config/sp500_tickers.txt"
SAMPLE_N = 50   # use 50 tickers for speed (< 30s)


def _load_config(path: str = "backtest_config.yaml") -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _load_close(ticker: str, start: str, end: str, cache_dir: str) -> pd.Series | None:
    path = Path(cache_dir) / f"{ticker}.parquet"
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
        df.index = pd.to_datetime(df.index)
        df = df.sort_index()
        col = next((c for c in ["Close", "close", "Adj Close"] if c in df.columns), None)
        if col is None:
            return None
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        return s.loc[(s.index >= pd.Timestamp(start)) & (s.index <= pd.Timestamp(end))]
    except Exception:
        return None


def _build_feature_rows(tickers: list[str], start: str, end: str, cache_dir: str) -> pd.DataFrame:
    """Build a simple feature matrix from cached parquet files."""
    from agents.weight_learning_agent.feature_builder import build_feature_matrix

    cfg = _load_config()
    feat_sel = cfg.get("feature_selection", {}) or {}
    subset = feat_sel.get("feature_subset") or None

    df = build_feature_matrix(
        tickers,
        start_date=start,
        end_date=end,
        holding_period=10,
        feature_subset=subset,
    )
    return df if df is not None else pd.DataFrame()


def _compute_ic(scores: np.ndarray, fwd_rets: np.ndarray) -> tuple[float, float]:
    """Returns (Pearson IC, Spearman rank-IC)."""
    mask = np.isfinite(scores) & np.isfinite(fwd_rets)
    if mask.sum() < 10:
        return float("nan"), float("nan")
    s, r = scores[mask], fwd_rets[mask]
    pearson = float(pd.Series(s).corr(pd.Series(r)))
    spearman = float(pd.Series(s).corr(pd.Series(r), method="spearman"))
    return pearson, spearman


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", nargs="*", default=None)
    parser.add_argument("--start", default="2019-01-01")
    parser.add_argument("--end", default="2022-01-01")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--n", type=int, default=SAMPLE_N, help="Number of tickers to sample")
    args = parser.parse_args()

    cfg = _load_config()
    cache_dir = cfg.get("data", {}).get("cache_dir", "data/cache")

    # Select tickers
    if args.tickers:
        tickers = [t.upper() for t in args.tickers]
    else:
        txt = Path(DEFAULT_TICKERS_FILE)
        if txt.exists():
            all_tickers = [t.strip() for t in txt.read_text().splitlines() if t.strip()]
        else:
            all_tickers = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "JPM", "V", "UNH"]
        rng = np.random.default_rng(42)
        tickers = sorted(rng.choice(all_tickers, size=min(args.n, len(all_tickers)), replace=False).tolist())

    # Load model first so we know which features to build
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        return
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    if isinstance(obj, dict):
        model = obj.get("estimator") or obj.get("model")
        feature_cols_pkl = (obj.get("feature_columns")
                            or obj.get("features")
                            or obj.get("feature_cols")
                            or [])
        train_horizon = obj.get("horizon_days", 1)
        print(f"  Trained at horizon: {train_horizon}d")
    else:
        model = obj
        feature_cols_pkl = []
        train_horizon = 1
    if model is None:
        print("ERROR: could not load model from pickle.")
        return
    print(f"  Model: {type(model).__name__}")

    print(f"Building feature matrix: {len(tickers)} tickers, {args.start} → {args.end}")
    # Build matrix with model's feature set (not config subset) to ensure correct columns
    from agents.weight_learning_agent.feature_builder import build_feature_matrix
    df = build_feature_matrix(
        tickers,
        start_date=args.start,
        end_date=args.end,
        holding_period=10,
        feature_subset=feature_cols_pkl if feature_cols_pkl else None,
    )
    if df is None or df.empty:
        print("ERROR: feature matrix is empty — check that data cache is populated.")
        return

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    print(f"  {len(df):,} rows, {df['date'].nunique()} dates, {df['ticker'].nunique()} tickers")

    # model already loaded above — use feature_cols_pkl as the definitive list
    feature_cols = feature_cols_pkl

    # Determine feature columns
    feat_exclude = {"date", "ticker", "sector", "direction", "regime_label", "y_bin", "forward_return",
                    "forward_return_excess"}
    if feature_cols:
        feat_cols = [c for c in feature_cols if c in df.columns]
    else:
        feat_cols = [c for c in df.columns
                     if c not in feat_exclude
                     and "forward" not in c.lower()
                     and "direction" not in c.lower()
                     and pd.api.types.is_numeric_dtype(df[c])]
    if not feat_cols:
        print("ERROR: no usable feature columns.")
        return
    print(f"  Features: {len(feat_cols)}")

    # Score all rows
    X = df[feat_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-10.0, 10.0).values
    X = np.nan_to_num(X, nan=0.0).copy()
    with np.errstate(all="ignore"):
        if hasattr(model, "predict_proba"):
            scores = model.predict_proba(X)[:, 1].astype(float) - 0.5
        else:
            scores = model.predict(X).astype(float)
    df["score"] = scores

    # Compute IC at each horizon using cached close prices
    print()
    print(f"{'Horizon':>8}  {'Pearson IC':>10}  {'Rank IC':>10}  {'N pairs':>8}  {'Interpretation'}")
    print("-" * 65)

    results = []
    for horizon in HORIZONS:
        fwd_rets = []
        score_vals = []
        for ticker in tickers:
            close = _load_close(ticker, args.start, args.end, cache_dir)
            if close is None or len(close) < horizon + 5:
                continue
            ticker_rows = df[df["ticker"] == ticker].copy()
            if ticker_rows.empty:
                continue
            for _, row in ticker_rows.iterrows():
                date = pd.Timestamp(row["date"])
                if date not in close.index:
                    continue
                idx = close.index.get_loc(date)
                future_idx = idx + horizon
                if future_idx >= len(close):
                    continue
                cur, fut = float(close.iloc[idx]), float(close.iloc[future_idx])
                if cur <= 0:
                    continue
                fwd_rets.append((fut - cur) / cur)
                score_vals.append(float(row["score"]))

        s_arr = np.array(score_vals, dtype=float)
        r_arr = np.array(fwd_rets, dtype=float)
        pearson, spearman = _compute_ic(s_arr, r_arr)
        n = int(np.isfinite(s_arr).sum())

        # Grinold-Kahn implied Sharpe: IC * sqrt(252/horizon) * sqrt(n_stocks_per_day)
        # Use 12 longs (typical from config)
        n_stocks = 12
        impl_sharpe = (abs(pearson) * np.sqrt(252.0 / horizon) * np.sqrt(n_stocks)
                       if np.isfinite(pearson) else float("nan"))

        flag = ""
        if np.isfinite(pearson):
            if abs(pearson) >= 0.04:
                flag = "★ Strong"
            elif abs(pearson) >= 0.02:
                flag = "✓ Usable"
            else:
                flag = "✗ Weak"

        results.append({"horizon": horizon, "pearson": pearson, "spearman": spearman,
                        "n": n, "impl_sharpe": impl_sharpe})
        p_str = f"{pearson:+.4f}" if np.isfinite(pearson) else "   nan"
        sp_str = f"{spearman:+.4f}" if np.isfinite(spearman) else "   nan"
        is_str = f"  →  impl Sharpe≈{impl_sharpe:.2f}" if np.isfinite(impl_sharpe) else ""
        print(f"{horizon:>6}d  {p_str:>10}  {sp_str:>10}  {n:>8,}  {flag}{is_str}")

    # Best horizon
    valid = [(r["horizon"], abs(r["pearson"])) for r in results if np.isfinite(r["pearson"])]
    if valid:
        best_h, best_ic = max(valid, key=lambda x: x[1])
        print()
        print(f"  Best horizon: {best_h}d  (|IC|={best_ic:.4f})")
        model_train_h = train_horizon
        if model_train_h != best_h:
            print(f"  NOTE: model trained at {model_train_h}d horizon — consider retraining at {best_h}d")
        else:
            print(f"  Model horizon matches ({model_train_h}d) ✓")

    print()
    print("Run `python run_model_selection.py` to retrain models with the updated 10d horizon.")


if __name__ == "__main__":
    main()
