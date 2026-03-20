"""
Correlation & Diversification Risk Analysis
===========================================

This script analyzes correlation risk and diversification quality
of a backtest by:
  - Reconstructing the daily portfolio weights from trades.csv
  - Computing rolling correlation matrices of underlying asset returns
  - Computing the diversification ratio over time:
        DR_t = (w_t^T sigma_t) / sqrt(w_t^T Sigma_t w_t)
    where:
        w_t      = vector of portfolio weights by asset on day t
        sigma_t  = vector of asset vols over a rolling window
        Sigma_t  = covariance matrix over the same window
  - Plotting DR_t over time and flagging low-diversification regimes.

Usage (from project root):
  python analysis/correlation_risk.py \
      --config backtest_config.yaml \
      --trades output/backtests/trades.csv \
      --equity output/backtests/daily_equity.csv \
      --lookback 60 \
      --threshold 1.1
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

import logging
import numpy as np
import pandas as pd
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import matplotlib.pyplot as plt

from backtesting import load_config
from utils.market_data import get_ohlcv


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Correlation and diversification risk analysis.")
    p.add_argument(
        "--config",
        default="backtest_config.yaml",
        help="Backtest config YAML (default: backtest_config.yaml)",
    )
    p.add_argument(
        "--trades",
        default="output/backtests/trades.csv",
        help="Trades CSV from backtest (default: output/backtests/trades.csv)",
    )
    p.add_argument(
        "--equity",
        default="output/backtests/daily_equity.csv",
        help="Daily equity CSV from backtest (default: output/backtests/daily_equity.csv)",
    )
    p.add_argument(
        "--lookback",
        type=int,
        default=60,
        help="Lookback window in trading days for correlations/vols (default: 60)",
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=1.1,
        help="Diversification ratio threshold for flagging low-diversification periods (default: 1.1)",
    )
    p.add_argument(
        "--output",
        default="output/backtests/diversification_ratio.png",
        help="Path to write diversification ratio plot (default: output/backtests/diversification_ratio.png)",
    )
    return p.parse_args()


def _load_trades(trades_path: Path) -> pd.DataFrame:
    if not trades_path.exists():
        raise FileNotFoundError(f"Trades CSV not found: {trades_path}")
    trades = pd.read_csv(trades_path)
    if trades.empty:
        raise RuntimeError(f"Trades CSV is empty: {trades_path}")
    trades["entry_date"] = pd.to_datetime(trades["entry_date"])
    trades["exit_date"] = pd.to_datetime(trades["exit_date"])
    if "holding_days" not in trades.columns:
        trades["holding_days"] = (trades["exit_date"] - trades["entry_date"]).dt.days
    return trades


def _load_equity(equity_path: Path) -> pd.DataFrame:
    if not equity_path.exists():
        raise FileNotFoundError(f"Equity CSV not found: {equity_path}")
    eq = pd.read_csv(equity_path)
    if eq.empty:
        raise RuntimeError(f"Equity CSV is empty: {equity_path}")
    eq["date"] = pd.to_datetime(eq["date"])
    eq = eq.sort_values("date").reset_index(drop=True)
    return eq


def _infer_tickers_from_trades(trades: pd.DataFrame) -> List[str]:
    return sorted(trades["ticker"].unique().tolist())


def _load_price_data(
    tickers: List[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cfg,
) -> Dict[str, pd.DataFrame]:
    """
    Reload OHLCV data for the tickers over [start, end] with a small buffer.
    """
    from utils.market_data import get_ohlcv  # local import to avoid heavy import at module load

    price_data: Dict[str, pd.DataFrame] = {}
    start_str = (start - pd.Timedelta(days=5 * cfg.vol_lookback_days)).strftime("%Y-%m-%d")
    end_str = end.strftime("%Y-%m-%d")
    for tk in tickers:
        df = get_ohlcv(
            tk,
            start_date=start_str,
            end_date=end_str,
            provider=cfg.data_provider,
            cache_dir=cfg.cache_dir,
            cache_ttl_days=cfg.cache_ttl_days,
        )
        if df is None or df.empty:
            continue
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        price_col = "AdjClose" if "AdjClose" in df.columns else "Close"
        df = df[[price_col]].rename(columns={price_col: "price"})
        price_data[tk] = df.sort_index()
    return price_data


def _build_position_weights(
    trades: pd.DataFrame,
    equity: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Reconstruct daily position weights for each ticker from trades and daily equity.

    Returns:
        weights: DataFrame indexed by date with columns = tickers (weights sum to <= 1)
        tickers: list of tickers in the universe
    """
    tickers = sorted(trades["ticker"].unique().tolist())
    dates = equity["date"]

    # Initialize size per position as constant from trades["position_size"]
    # Approximate: assume position_size is roughly constant over holding period.
    size_matrix = pd.DataFrame(0.0, index=dates, columns=tickers)

    for _, row in trades.iterrows():
        tk = row["ticker"]
        entry = row["entry_date"]
        exit_dt = row["exit_date"]
        mask = (dates >= entry) & (dates <= exit_dt)
        size_matrix.loc[mask, tk] += float(row["position_size"])

    # Convert to weights using daily equity
    equity_series = equity.set_index("date")["equity"]
    weights = size_matrix.copy()
    for dt in dates:
        eq_val = float(equity_series.get(dt, np.nan))
        if not np.isfinite(eq_val) or eq_val <= 0:
            weights.loc[dt, :] = 0.0
        else:
            weights.loc[dt, :] = size_matrix.loc[dt, :] / eq_val

    return weights, tickers


def _compute_returns(price_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Build a wide returns DataFrame indexed by date with columns = tickers.
    """
    rets = []
    for tk, df in price_data.items():
        s = df["price"].pct_change().rename(tk)
        rets.append(s)
    if not rets:
        raise RuntimeError("No price data available to compute returns.")
    wide = pd.concat(rets, axis=1).sort_index()
    return wide


def compute_diversification_ratio_series(
    weights: pd.DataFrame,
    returns: pd.DataFrame,
    lookback: int,
) -> pd.Series:
    """
    Compute diversification ratio time-series:
        DR_t = (w_t^T sigma_t) / sqrt(w_t^T Sigma_t w_t)
    Only considers assets with non-zero weights and valid data in the window.
    """
    all_dates = weights.index.intersection(returns.index)
    dr_values = []
    dr_index = []

    for dt in all_dates:
        window = returns.loc[:dt].tail(lookback)
        if len(window) < lookback // 2:
            continue

        w = weights.loc[dt]
        active = w[w.abs() > 1e-6].index.tolist()
        if not active:
            continue

        win = window[active].dropna(axis=1, how="all")
        active = [c for c in active if c in win.columns]
        if len(active) < 2:
            continue

        win = win[active].dropna()
        if len(win) < lookback // 2:
            continue

        sigma = win.std(ddof=0).values  # individual vols
        cov = win.cov().values          # covariance matrix
        w_vec = w[active].values

        if np.allclose(w_vec, 0.0) or np.any(~np.isfinite(w_vec)):
            continue

        port_vol = float(np.sqrt(np.dot(w_vec, cov @ w_vec)))
        if port_vol <= 0 or not np.isfinite(port_vol):
            continue

        num = float(np.dot(np.abs(w_vec), sigma))
        dr = num / port_vol if port_vol > 0 else np.nan
        if not np.isfinite(dr):
            continue

        dr_index.append(dt)
        dr_values.append(dr)

    return pd.Series(dr_values, index=pd.to_datetime(dr_index), name="diversification_ratio")


def plot_diversification_ratio(
    dr: pd.Series,
    threshold: float,
    output_path: Path,
) -> None:
    """
    Plot diversification ratio over time and shade periods below threshold.
    """
    if dr.empty:
        raise RuntimeError("Diversification ratio series is empty; nothing to plot.")

    os.makedirs(output_path.parent, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(dr.index, dr.values, label="Diversification ratio", color="tab:blue")
    plt.axhline(threshold, color="red", linestyle="--", label=f"Threshold = {threshold:.2f}")

    # Highlight periods where DR falls below threshold
    low = dr[dr < threshold]
    if not low.empty:
        plt.scatter(low.index, low.values, color="red", s=10, label="Below threshold")

    plt.title("Portfolio Diversification Ratio Over Time")
    plt.xlabel("Date")
    plt.ylabel("Diversification ratio")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = parse_args()

    cfg = load_config(args.config)
    trades_path = (ROOT / args.trades).resolve()
    equity_path = (ROOT / args.equity).resolve()
    output_path = (ROOT / args.output).resolve()

    print(f"Loading trades from {trades_path} …")
    trades = _load_trades(trades_path)

    print(f"Loading daily equity from {equity_path} …")
    equity = _load_equity(equity_path)

    start = equity["date"].min()
    end = equity["date"].max()

    tickers = _infer_tickers_from_trades(trades)
    print(f"Found {len(tickers)} tickers in trades.")

    print(f"Loading price data for returns from {start.date()} to {end.date()} …")
    price_data = _load_price_data(tickers, start, end, cfg)
    returns = _compute_returns(price_data)

    # Align weights and returns to common date range
    print("Reconstructing daily portfolio weights …")
    weights, tickers_used = _build_position_weights(trades, equity)
    weights = weights.reindex(returns.index).fillna(0.0)

    print(f"Computing diversification ratio with lookback={args.lookback} days …")
    dr = compute_diversification_ratio_series(weights, returns, lookback=args.lookback)

    print("Plotting diversification ratio and flagging low-diversification periods …")
    plot_diversification_ratio(dr, threshold=args.threshold, output_path=output_path)

    low_periods = dr[dr < args.threshold]
    if not low_periods.empty:
        print("\nPeriods with diversification ratio below threshold:")
        print(low_periods.to_frame().head(20).to_string())
        print(f"\nTotal days below threshold: {len(low_periods)}")
    else:
        print("\nNo periods found with diversification ratio below threshold.")

    print(f"\nDiversification ratio plot saved to: {output_path}")


if __name__ == "__main__":
    main()

