"""
GBM analysis: estimate parameters, simulate forward, validate calibration per ticker.

Usage:
    python run_gbm_analysis.py
    python run_gbm_analysis.py --config backtest_config.yaml --horizon 20 --tickers AAPL MSFT
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)


def run(
    config_path: str = "backtest_config.yaml",
    tickers: list[str] | None = None,
    horizon_days: int = 20,
    n_paths: int = 1000,
    seed: int = 42,
) -> None:
    from backtesting.config import load_config
    from config import get_effective_tickers
    from simulation.gbm import (
        estimate_gbm_params,
        simulate_gbm,
        backtest_gbm_accuracy,
    )
    from utils.market_data import get_ohlcv

    config = load_config(config_path)
    fallback = config.tickers or ["AAPL", "MSFT", "GOOG", "NVDA", "AMZN"]
    tickers = tickers or get_effective_tickers(config.tickers or [], fallback)
    if not tickers:
        print("No tickers.")
        return

    start = getattr(config, "start_date", "2018-01-01")
    end = getattr(config, "end_date", "2024-01-01")
    T = horizon_days / 252.0
    n_steps = max(1, horizon_days)

    rows = []
    for ticker in tickers:
        try:
            df = get_ohlcv(ticker, start, end, use_cache=True, cache_ttl_days=0)
        except Exception:
            continue
        if df is None or df.empty or "Close" not in df.columns:
            continue
        prices = df["Close"].dropna()
        if len(prices) < 252 + horizon_days:
            continue
        mu, sigma = estimate_gbm_params(prices, window=252)
        S0 = float(prices.iloc[-1])
        paths = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths=n_paths, seed=seed)
        final = paths[:, -1]
        rets = (final - S0) / S0
        p_gain_2 = float(np.mean(rets > 0.02))
        p_gain_5 = float(np.mean(rets > 0.05))
        acc = backtest_gbm_accuracy(prices, horizon_days=horizon_days, n_paths=500, seed=seed)
        coverage_95 = acc.get("coverage_95", np.nan)
        rows.append({
            "ticker": ticker,
            "mu_ann": mu,
            "sigma_ann": sigma,
            "p_gain_2": p_gain_2,
            "p_gain_5": p_gain_5,
            "coverage_95": coverage_95,
        })
    if not rows:
        print("No data for any ticker.")
        return
    table = pd.DataFrame(rows)
    h = horizon_days
    print()
    print(f"  Ticker | mu (ann) | sigma (ann) | P(>2%, {h}d) | P(>5%, {h}d) | Coverage 95%")
    print("  " + "-" * 72)
    for _, r in table.iterrows():
        mu_s = f"{r['mu_ann']:.1%}" if np.isfinite(r["mu_ann"]) else " — "
        sig_s = f"{r['sigma_ann']:.1%}" if np.isfinite(r["sigma_ann"]) else " — "
        p2 = f"{r['p_gain_2']:.1%}" if np.isfinite(r["p_gain_2"]) else " — "
        p5 = f"{r['p_gain_5']:.1%}" if np.isfinite(r["p_gain_5"]) else " — "
        cov = f"{r['coverage_95']:.1%}" if np.isfinite(r["coverage_95"]) else " — "
        print(f"  {r['ticker']:<6} | {mu_s:>8} | {sig_s:>11} | {p2:>11} | {p5:>11} | {cov:>12}")
    out_dir = "output/research"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gbm_analysis.csv")
    table.rename(columns={"p_gain_2": f"P(>2%, {horizon_days}d)", "p_gain_5": f"P(>5%, {horizon_days}d)", "coverage_95": "Coverage_95"}).to_csv(out_path, index=False)
    print(f"\n  Saved → {out_path}")


def main():
    p = argparse.ArgumentParser(description="GBM analysis per ticker")
    p.add_argument("--config", default="backtest_config.yaml", help="Config YAML")
    p.add_argument("--tickers", nargs="+", default=None, help="Tickers")
    p.add_argument("--horizon", type=int, default=20, help="Forward horizon in days")
    p.add_argument("--n-paths", type=int, default=1000, help="Paths per simulation")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    args = p.parse_args()
    run(
        config_path=args.config,
        tickers=args.tickers,
        horizon_days=args.horizon,
        n_paths=args.n_paths,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
