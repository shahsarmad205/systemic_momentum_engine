"""
Run three backtests: Equal-weight, MV Max-Sharpe, MV Min-Variance.
Print comparison table and generate efficient frontier plot.

Usage:
    python run_mv_comparison.py
    python run_mv_comparison.py --config backtest_config.yaml --tickers AAPL MSFT GOOG
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


def _vol_from_equity(daily_equity: pd.DataFrame) -> float:
    """Annualized volatility of portfolio returns (%)."""
    if daily_equity.empty or "equity" not in daily_equity.columns:
        return 0.0
    eq = daily_equity["equity"]
    ret = eq.pct_change().dropna()
    if len(ret) < 2:
        return 0.0
    return float(ret.std() * np.sqrt(252) * 100)


def run_comparison(
    config_path: str = "backtest_config.yaml",
    tickers: list[str] | None = None,
    frontier_path: str = "output/research/efficient_frontier.png",
) -> None:
    from backtesting.config import load_config
    from backtest.engine import BacktestEngine
    from config import get_effective_tickers

    config = load_config(config_path)
    if tickers:
        config.tickers = tickers
    fallback = config.tickers or ["AAPL", "MSFT", "GOOG", "AMZN", "NVDA"]
    tickers = get_effective_tickers(config.tickers or [], fallback)
    if not tickers:
        print("No tickers to run.")
        return

    results = []
    # 1) Equal-weight
    cfg_eq = load_config(config_path)
    if tickers:
        cfg_eq.tickers = tickers
    cfg_eq.position_sizing = "equal"
    cfg_eq.position_sizing_method = "equal"
    cfg_eq.mean_variance_enabled = False
    engine_eq = BacktestEngine(config=cfg_eq, config_path=config_path)
    print("Running backtest: Equal-weight…")
    r_eq = engine_eq.run_backtest(tickers)
    m_eq = r_eq.metrics or {}
    vol_eq = _vol_from_equity(r_eq.daily_equity)
    results.append({
        "Method": "Equal-weight",
        "Sharpe": m_eq.get("sharpe_ratio", np.nan),
        "Return": m_eq.get("total_return", np.nan),
        "MaxDD": m_eq.get("max_drawdown", np.nan),
        "Vol": vol_eq,
    })

    # 2) MV Max-Sharpe
    cfg_ms = load_config(config_path)
    if tickers:
        cfg_ms.tickers = tickers
    cfg_ms.position_sizing = "mv_max_sharpe"
    cfg_ms.position_sizing_method = "mv_max_sharpe"
    cfg_ms.mean_variance_enabled = True
    cfg_ms.mean_variance_method = "max_sharpe"
    cfg_ms.mean_variance_lookback_days = 60
    cfg_ms.mean_variance_rebalance_days = 20
    cfg_ms.mean_variance_max_single_weight = 0.25
    engine_ms = BacktestEngine(config=cfg_ms, config_path=config_path)
    print("Running backtest: MV Max-Sharpe…")
    r_ms = engine_ms.run_backtest(tickers)
    m_ms = r_ms.metrics or {}
    vol_ms = _vol_from_equity(r_ms.daily_equity)
    results.append({
        "Method": "MV Max-Sharpe",
        "Sharpe": m_ms.get("sharpe_ratio", np.nan),
        "Return": m_ms.get("total_return", np.nan),
        "MaxDD": m_ms.get("max_drawdown", np.nan),
        "Vol": vol_ms,
    })

    # 3) MV Min-Variance
    cfg_mv = load_config(config_path)
    if tickers:
        cfg_mv.tickers = tickers
    cfg_mv.position_sizing = "mv_min_variance"
    cfg_mv.position_sizing_method = "mv_min_variance"
    cfg_mv.mean_variance_enabled = True
    cfg_mv.mean_variance_method = "min_variance"
    cfg_mv.mean_variance_lookback_days = 60
    cfg_mv.mean_variance_rebalance_days = 20
    cfg_mv.mean_variance_max_single_weight = 0.25
    engine_mv = BacktestEngine(config=cfg_mv, config_path=config_path)
    print("Running backtest: MV Min-Variance…")
    r_mv = engine_mv.run_backtest(tickers)
    m_mv = r_mv.metrics or {}
    vol_mv = _vol_from_equity(r_mv.daily_equity)
    results.append({
        "Method": "MV Min-Variance",
        "Sharpe": m_mv.get("sharpe_ratio", np.nan),
        "Return": m_mv.get("total_return", np.nan),
        "MaxDD": m_mv.get("max_drawdown", np.nan),
        "Vol": vol_mv,
    })

    # Print table
    print()
    print("  Method          | Sharpe | Return | MaxDD  | Vol")
    print("  " + "-" * 50)
    for row in results:
        sh = row["Sharpe"]
        ret = row["Return"]
        dd = row["MaxDD"]
        vol = row["Vol"]
        sh_s = f"{sh:.3f}" if np.isfinite(sh) else " — "
        ret_s = f"{ret:.2%}" if np.isfinite(ret) else " — "
        dd_s = f"{dd:.2%}" if np.isfinite(dd) else " — "
        vol_s = f"{vol:.2f}%" if np.isfinite(vol) else " — "
        print(f"  {row['Method']:<16} | {sh_s:>6} | {ret_s:>6} | {dd_s:>6} | {vol_s:>6}")

    # Efficient frontier plot from first run's price data
    if r_eq.price_data and getattr(r_eq, "price_data", None):
        try:
            from portfolio.mean_variance import (
                compute_efficient_frontier,
                max_sharpe_weights,
                min_variance_weights,
            )
            price_data = r_eq.price_data
            all_dates = sorted(set().union(*(set(d.index) for d in price_data.values())))
            returns_list = []
            for tk, df in price_data.items():
                if df is None or df.empty or "Close" not in df.columns:
                    continue
                close = df["Close"].reindex(all_dates).ffill().bfill()
                ret = close.pct_change().dropna()
                ret.name = tk
                returns_list.append(ret)
            if returns_list:
                returns_df = pd.concat(returns_list, axis=1, join="inner").dropna(how="all")
                if returns_df.shape[0] >= 20 and returns_df.shape[1] >= 1:
                    frontier = compute_efficient_frontier(
                        returns_df, n_portfolios=1000, risk_free_rate=0.04
                    )
                    ms_w = max_sharpe_weights(returns_df, risk_free_rate=0.04, constraints={"max_weight": 0.25})
                    mv_w = min_variance_weights(returns_df, constraints={"max_weight": 0.25})
                    n = len(returns_df.columns)
                    eq_w = {t: 1.0 / n for t in returns_df.columns}
                    mu = returns_df.mean().values * 252
                    cov = returns_df.cov().values * 252
                    def port_ret(w_dict):
                        w = np.array([w_dict.get(t, 0) for t in returns_df.columns])
                        return float(w @ mu)
                    def port_vol(w_dict):
                        w = np.array([w_dict.get(t, 0) for t in returns_df.columns])
                        return float(np.sqrt(w @ cov @ w))
                    ret_ms = port_ret(ms_w)
                    vol_ms_p = port_vol(ms_w)
                    ret_mv = port_ret(mv_w)
                    vol_mv_p = port_vol(mv_w)
                    ret_eq = port_ret(eq_w)
                    vol_eq_p = port_vol(eq_w)
                    import matplotlib
                    matplotlib.use("Agg")
                    import matplotlib.pyplot as plt
                    os.makedirs(os.path.dirname(frontier_path) or ".", exist_ok=True)
                    fig, ax = plt.subplots(figsize=(8, 5))
                    ax.scatter(
                        frontier["volatility"],
                        frontier["expected_return"],
                        c=frontier["sharpe_ratio"],
                        alpha=0.4,
                        s=10,
                        cmap="viridis",
                    )
                    # Efficient frontier: sort by vol and plot upper envelope
                    ef = frontier.sort_values("volatility")
                    ef = ef.assign(max_ret=ef["expected_return"].cummax())
                    ax.plot(ef["volatility"].values, ef["max_ret"].values, "b-", alpha=0.8, linewidth=2, label="Efficient frontier")
                    ax.scatter([vol_ms_p], [ret_ms], marker="*", s=400, color="gold", edgecolors="black", label="Max Sharpe")
                    ax.scatter([vol_mv_p], [ret_mv], marker="D", s=200, color="green", edgecolors="black", label="Min Variance")
                    ax.scatter([vol_eq_p], [ret_eq], marker="o", s=150, color="gray", edgecolors="black", label="Equal-weight")
                    ax.set_xlabel("Volatility (annual)")
                    ax.set_ylabel("Expected Return (annual)")
                    ax.set_title("Efficient Frontier")
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    fig.tight_layout()
                    fig.savefig(frontier_path, dpi=150)
                    plt.close(fig)
                    print(f"\n  Efficient frontier plot → {frontier_path}")
        except Exception as e:
            print(f"\n  Efficient frontier plot skipped: {e}")


def main():
    p = argparse.ArgumentParser(description="Compare Equal-weight vs MV Max-Sharpe vs MV Min-Variance")
    p.add_argument("--config", default="backtest_config.yaml", help="Backtest config YAML")
    p.add_argument("--tickers", nargs="+", default=None, help="Override tickers")
    p.add_argument("--frontier", default="output/research/efficient_frontier.png", help="Path for frontier plot")
    args = p.parse_args()
    run_comparison(
        config_path=args.config,
        tickers=args.tickers,
        frontier_path=args.frontier,
    )


if __name__ == "__main__":
    main()
