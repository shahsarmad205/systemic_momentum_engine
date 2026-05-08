"""Phase 2 benchmarks for research_numerics_core.py.

Compares legacy per-date loop vs new tensor/chunked implementations
across multiple panel sizes with memory tracking.

Usage:
    python benchmarks/benchmark_phase2.py [--quick]

With --quick: only 50K rows
Without --quick: 50K, 250K, 1.7M rows
"""
from __future__ import annotations

import argparse
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from model_selection.research_numerics_core import (
    compute_daily_ic_series,
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
            }
            for j in range(n_features):
                row[f"feat_{j}"] = base + np.random.randn() * 0.5
            rows.append(row)

    return pd.DataFrame(rows)


def get_peak_rss_mb() -> float:
    """Get peak RSS in MB."""
    ru = resource.getrusage(resource.RUSAGE_SELF)
    return ru.ru_maxrss / 1024  # macOS: KB → MB


def benchmark_ic(
    panel: pd.DataFrame,
    features: list[str],
    mode: str,
) -> tuple[float, float, dict]:
    """Benchmark IC computation with memory tracking.

    Returns: (elapsed_seconds, peak_rss_mb, memory_report)
    """
    rss_before = get_peak_rss_mb()
    t0 = time.perf_counter()

    ic_df, _, _ = compute_daily_ic_series(
        panel, features, "forward_return", min_breadth=8, mode=mode,
    )

    elapsed = time.perf_counter() - t0
    rss_after = get_peak_rss_mb()
    peak_rss = max(rss_after - rss_before, 0)

    report = ic_df.attrs.get("memory_report", {})
    return elapsed, peak_rss, report


def run_benchmark(n_tickers: int, n_dates: int, n_features: int) -> None:
    """Run benchmarks for a given panel size."""
    n_rows = n_tickers * n_dates
    label = f"{n_rows // 1000}K rows" if n_rows >= 1000 else f"{n_rows} rows"
    print(f"\n{'=' * 80}")
    print(f"Panel: {n_tickers} tickers × {n_dates} dates × {n_features} features = {n_rows:,} rows")
    print(f"{'=' * 80}")

    panel = make_panel(n_tickers, n_dates, n_features)
    features = [f"feat_{j}" for j in range(n_features)]

    modes = ["legacy", "full_tensor", "chunked"]
    results = {}

    for mode in modes:
        try:
            elapsed, peak_rss, report = benchmark_ic(panel, features, mode)
            results[mode] = {
                "elapsed": elapsed,
                "peak_rss_mb": peak_rss,
                "report": report,
            }
            exec_mode = report.execution_mode if hasattr(report, "execution_mode") else mode
            batch = report.batch_size if hasattr(report, "batch_size") else "N/A"
            print(f"  {mode:15s} | {elapsed:8.3f}s | RSS: {peak_rss:8.1f}MB | "
                  f"Mode: {exec_mode} | Batch: {batch}")
        except Exception as e:
            print(f"  {mode:15s} | ERROR: {e}")
            results[mode] = None

    # Compute speedups
    if results.get("legacy") and results.get("full_tensor"):
        legacy_t = results["legacy"]["elapsed"]
        tensor_t = results["full_tensor"]["elapsed"]
        speedup = legacy_t / tensor_t if tensor_t > 0 else float("inf")
        print(f"\n  Tensor vs Legacy speedup: {speedup:.1f}x")

    if results.get("legacy") and results.get("chunked"):
        legacy_t = results["legacy"]["elapsed"]
        chunked_t = results["chunked"]["elapsed"]
        speedup = legacy_t / chunked_t if chunked_t > 0 else float("inf")
        print(f"  Chunked vs Legacy speedup: {speedup:.1f}x")

    # Memory comparison
    if results.get("legacy") and results.get("full_tensor"):
        legacy_rss = results["legacy"]["peak_rss_mb"]
        tensor_rss = results["full_tensor"]["peak_rss_mb"]
        print(f"\n  Legacy RSS: {legacy_rss:.1f}MB | Tensor RSS: {tensor_rss:.1f}MB")

    # Numerical parity: legacy vs tensor
    if results.get("legacy") and results.get("full_tensor"):
        ic_legacy, _, _ = compute_daily_ic_series(
            panel, features, "forward_return", min_breadth=8, mode="legacy",
        )
        ic_tensor, _, _ = compute_daily_ic_series(
            panel, features, "forward_return", min_breadth=8, mode="full_tensor",
        )

        max_diff = 0.0
        nan_mismatches = 0
        for feat in features:
            l_vals = ic_legacy[feat].values
            t_vals = ic_tensor[feat].values
            both_finite = np.isfinite(l_vals) & np.isfinite(t_vals)
            if both_finite.any():
                diff = np.max(np.abs(l_vals[both_finite] - t_vals[both_finite]))
                max_diff = max(max_diff, diff)
            nan_legacy = np.isnan(l_vals) & np.isfinite(t_vals)
            nan_tensor = np.isfinite(l_vals) & np.isnan(t_vals)
            nan_mismatches += nan_legacy.sum() + nan_tensor.sum()

        print(f"\n  Max IC difference (legacy vs tensor): {max_diff:.2e}")
        print(f"  NaN pattern mismatches: {nan_mismatches}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Only run 50K rows")
    args = parser.parse_args()

    print("Phase 2 Benchmarks: Tensor-First IC Computation")
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"Pandas: {pd.__version__}")

    configs = [
        (50, 1000, 10),    # 50K rows
        (50, 5000, 10),    # 250K rows
    ]

    if not args.quick:
        configs.append((500, 3400, 50))  # 1.7M rows

    for n_tickers, n_dates, n_features in configs:
        run_benchmark(n_tickers, n_dates, n_features)


if __name__ == "__main__":
    main()
