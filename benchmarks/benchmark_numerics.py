"""Runtime benchmarks for research_numerics_core.py.

Compares old loop-based implementations against new vectorized functions
on a representative synthetic panel.

Usage:
    python benchmarks/benchmark_numerics.py [--n-tickers 100] [--n-dates 500] [--n-features 10]
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from model_selection.research_numerics_core import (
    compute_forward_returns,
    vectorized_spearman_ic_from_panel,
    compute_rank_persistence,
    compute_ic_decay,
    batch_hac_tstat,
    compute_feature_redundancy,
)
from model_selection.signal_decay_engine import (
    compute_ic_decay_curve as _old_compute_ic_decay,
    compute_rank_persistence_curve as _old_compute_rank_persistence,
)
from model_selection.ic_diagnostics_engine import (
    compute_global_ic as _old_compute_global_ic,
)


def make_panel(n_tickers: int, n_dates: int, n_features: int) -> pd.DataFrame:
    """Create a synthetic panel for benchmarking."""
    np.random.seed(42)
    tickers = [f"T{i:03d}" for i in range(n_tickers)]
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")

    rows = []
    for ticker in tickers:
        base = np.random.randn()
        for t, date in enumerate(dates):
            fwd = np.random.randn() * 0.02
            row = {
                "date": date,
                "ticker": ticker,
                "forward_return": fwd,
                "daily_return": fwd,
                "sector": np.random.choice(["Tech", "Health", "Fin", "Energy", "Consumer"]),
                "market_cap": np.random.uniform(1e9, 1e12),
                "rolling_vol_20": np.random.uniform(0.15, 0.35),
                "adv_dollar_20": np.random.uniform(1e7, 1e9),
                "capm_beta": np.random.uniform(0.5, 1.5),
            }
            for j in range(n_features):
                row[f"feat_{j}"] = base + np.random.randn() * 0.5
            rows.append(row)

    return pd.DataFrame(rows)


def benchmark_forward_returns(panel: pd.DataFrame, horizons: list[int]) -> tuple[float, float]:
    """Benchmark forward return computation."""
    # Old: per-ticker loop
    t0 = time.perf_counter()
    for h in horizons:
        work = panel[["date", "ticker", "forward_return"]].copy()
        col = f"fwd_ret_{h}d"
        work[col] = work.groupby("ticker", sort=False)["forward_return"].transform(
            lambda x: x.rolling(h).sum() if h > 1 else x
        )
        work[col] = work.groupby("ticker", sort=False)[col].shift(-h)
    t_old = time.perf_counter() - t0

    # New: vectorized
    t0 = time.perf_counter()
    compute_forward_returns(panel, horizons, compound=False)
    t_new = time.perf_counter() - t0

    return t_old, t_new


def benchmark_ic(panel: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    """Benchmark IC computation."""
    # Old: per-date groupby + spearmanr loop
    t0 = time.perf_counter()
    for feat in features:
        for date, grp in panel.groupby("date", sort=False):
            if len(grp) < 8:
                continue
            feat_vals = grp[feat].values
            fwd_vals = grp["forward_return"].values
            if np.nanstd(feat_vals) < 1e-15 or np.nanstd(fwd_vals) < 1e-15:
                continue
            from scipy import stats as scipy_stats
            scipy_stats.spearmanr(feat_vals, fwd_vals)
    t_old = time.perf_counter() - t0

    # New: vectorized
    t0 = time.perf_counter()
    vectorized_spearman_ic_from_panel(panel, features, "forward_return", min_breadth=8)
    t_new = time.perf_counter() - t0

    return t_old, t_new


def benchmark_rank_persistence(panel: pd.DataFrame, features: list[str], lags: list[int]) -> tuple[float, float]:
    """Benchmark rank persistence computation."""
    t0 = time.perf_counter()
    for feat in features:
        _old_compute_rank_persistence(panel, feat, lags, min_dates=30, min_breadth=8)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    compute_rank_persistence(panel, features, lags, min_dates=30, min_breadth=8)
    t_new = time.perf_counter() - t0

    return t_old, t_new


def benchmark_ic_decay(panel: pd.DataFrame, features: list[str], horizons: list[int]) -> tuple[float, float]:
    """Benchmark IC decay computation."""
    t0 = time.perf_counter()
    for feat in features:
        _old_compute_ic_decay(panel, feat, horizons, min_dates=30, min_breadth=8)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    compute_ic_decay(panel, features, horizons, min_dates=30, min_breadth=8)
    t_new = time.perf_counter() - t0

    return t_old, t_new


def benchmark_redundancy(panel: pd.DataFrame, features: list[str]) -> tuple[float, float]:
    """Benchmark feature redundancy computation."""
    # Old: O(F²×D) pair loop with per-date filtering
    from scipy import stats as scipy_stats
    t0 = time.perf_counter()
    n = len(features)
    for i in range(n):
        for j in range(i + 1, n):
            for date, grp in panel.groupby("date", sort=False):
                if len(grp) < 8:
                    continue
                a = grp[features[i]].values
                b = grp[features[j]].values
                valid = np.isfinite(a) & np.isfinite(b)
                if valid.sum() < 8:
                    continue
                scipy_stats.spearmanr(a[valid], b[valid])
    t_old = time.perf_counter() - t0

    # New: vectorized corr matrix
    t0 = time.perf_counter()
    compute_feature_redundancy(panel, features, method="spearman")
    t_new = time.perf_counter() - t0

    return t_old, t_new


def benchmark_hac(n_dates: int, n_features: int) -> tuple[float, float]:
    """Benchmark HAC t-stat computation."""
    np.random.seed(42)
    ics = np.random.randn(n_dates, n_features) * 0.01

    # Old: scalar loop
    t0 = time.perf_counter()
    for j in range(n_features):
        from model_selection.ic_diagnostics_engine import _hac_tstat
        _hac_tstat(ics[:, j], max(1, n_features - 1))
    t_old = time.perf_counter() - t0

    # New: batch
    t0 = time.perf_counter()
    batch_hac_tstat(ics, lags=max(1, n_features - 1))
    t_new = time.perf_counter() - t0

    return t_old, t_new


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-tickers", type=int, default=100)
    parser.add_argument("--n-dates", type=int, default=500)
    parser.add_argument("--n-features", type=int, default=10)
    args = parser.parse_args()

    n_tickers = args.n_tickers
    n_dates = args.n_dates
    n_features = args.n_features

    print(f"Benchmarking: {n_tickers} tickers, {n_dates} dates, {n_features} features")
    print(f"Panel size: {n_tickers * n_dates:,} rows")
    print("=" * 70)

    panel = make_panel(n_tickers, n_dates, n_features)
    features = [f"feat_{j}" for j in range(n_features)]
    horizons = [1, 5, 10, 20]
    lags = [1, 5, 10, 20]

    benchmarks = [
        ("Forward Returns", lambda: benchmark_forward_returns(panel, horizons)),
        ("Spearman IC", lambda: benchmark_ic(panel, features)),
        ("Rank Persistence", lambda: benchmark_rank_persistence(panel, features, lags)),
        ("IC Decay", lambda: benchmark_ic_decay(panel, features, horizons)),
        ("Feature Redundancy", lambda: benchmark_redundancy(panel, features)),
        ("HAC t-stat", lambda: benchmark_hac(n_dates, n_features)),
    ]

    results = []
    for name, fn in benchmarks:
        try:
            t_old, t_new = fn()
            speedup = t_old / t_new if t_new > 0 else float("inf")
            results.append((name, t_old, t_new, speedup))
            print(f"{name:25s} | Old: {t_old:8.3f}s | New: {t_new:8.3f}s | Speedup: {speedup:6.1f}x")
        except Exception as e:
            print(f"{name:25s} | ERROR: {e}")
            results.append((name, float("nan"), float("nan"), float("nan")))

    print("=" * 70)
    print("\nSummary:")
    for name, t_old, t_new, speedup in results:
        if np.isfinite(speedup):
            print(f"  {name}: {speedup:.1f}x faster")


if __name__ == "__main__":
    main()
