"""
Options analysis report from backtest trades.

Loads trades.csv and produces a report with:
- Average implied vol at entry by regime (when available from options columns or computed)
- Average delta exposure
- Theoretical cost of hedging entire portfolio with puts
- Comparison: stock P&L vs equivalent ATM call P&L

Usage:
    python run_options_analysis.py
    python run_options_analysis.py --trades output/backtests/trades.csv --output output/backtests/options_report.txt
"""

from __future__ import annotations

import argparse
import os
import sys

import pandas as pd

# Ensure project root on path
_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from options.black_scholes import bs_price, implied_vol_from_historical


def _ensure_options_columns(trades: pd.DataFrame, risk_free_rate: float = 0.04, expiry_days: int = 30) -> pd.DataFrame:
    """If trades lack options columns, compute them from price data (load OHLCV per ticker)."""
    need = {"atm_call_price", "atm_call_delta", "breakeven_pct", "put_hedge_cost"}
    if need.issubset(trades.columns):
        return trades
    try:
        from utils.market_data import get_ohlcv
    except ImportError:
        return trades
    if "entry_date" not in trades.columns or "entry_price" not in trades.columns or "ticker" not in trades.columns:
        return trades
    trades = trades.copy()
    entry_dates = pd.to_datetime(trades["entry_date"])
    start = entry_dates.min() - pd.Timedelta(days=60)
    end = entry_dates.max() + pd.Timedelta(days=5)
    T = expiry_days / 365.0
    atm_call_price_list = []
    atm_call_delta_list = []
    breakeven_pct_list = []
    put_hedge_cost_list = []
    for _, row in trades.iterrows():
        tk = row["ticker"]
        S = float(row["entry_price"])
        ed = pd.Timestamp(row["entry_date"])
        sigma = 0.20
        try:
            df = get_ohlcv(tk, start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d"), use_cache=True, cache_ttl_days=0)
            if df is not None and not df.empty and "Close" in df.columns:
                series = df["Close"].loc[df.index <= ed]
                if len(series) >= 2:
                    from options.black_scholes import bs_greeks
                    iv_series = implied_vol_from_historical(series, window=30)
                    last = iv_series.dropna()
                    if not last.empty:
                        sigma = float(last.iloc[-1])
        except Exception:
            pass
        from options.black_scholes import bs_greeks
        atm_call = bs_price(S, S, T, risk_free_rate, sigma, "call")
        g = bs_greeks(S, S, T, risk_free_rate, sigma, "call")
        put_hedge = bs_price(S, S, T, risk_free_rate, sigma, "put")
        atm_call_price_list.append(round(atm_call, 4))
        atm_call_delta_list.append(round(g["delta"], 6))
        breakeven_pct_list.append(round(100.0 * atm_call / S, 4) if S > 0 else None)
        put_hedge_cost_list.append(round(put_hedge, 4))
    trades["atm_call_price"] = atm_call_price_list
    trades["atm_call_delta"] = atm_call_delta_list
    trades["breakeven_pct"] = breakeven_pct_list
    trades["put_hedge_cost"] = put_hedge_cost_list
    return trades


def run_report(trades_path: str, output_path: str | None, risk_free_rate: float, expiry_days: int) -> str:
    if not os.path.isfile(trades_path):
        return f"Trades file not found: {trades_path}"
    trades = pd.read_csv(trades_path)
    if trades.empty:
        return "No trades in file."
    for col in ["entry_date", "entry_price", "ticker", "regime", "pnl", "position_size", "exit_price", "shares"]:
        if col not in trades.columns:
            return f"Trades CSV missing column: {col}"
    trades = _ensure_options_columns(trades, risk_free_rate=risk_free_rate, expiry_days=expiry_days)
    lines = []
    lines.append("=" * 60)
    lines.append("OPTIONS ANALYSIS REPORT")
    lines.append("=" * 60)
    lines.append(f"Trades file: {trades_path}")
    lines.append(f"Total trades: {len(trades)}")
    lines.append("")

    if "atm_call_delta" in trades.columns:
        agg_dict = {"count": ("pnl", "count"), "avg_delta": ("atm_call_delta", "mean"), "avg_breakeven_pct": ("breakeven_pct", "mean")}
        if "implied_vol" in trades.columns:
            agg_dict["avg_implied_vol"] = ("implied_vol", "mean")
        by_regime = trades.groupby("regime", dropna=False).agg(**agg_dict).round(4)
        lines.append("Metrics at entry by regime:")
        lines.append(by_regime.to_string())
        lines.append("")
        lines.append(f"Overall average delta: {trades['atm_call_delta'].mean():.4f}")
        if "implied_vol" in trades.columns:
            lines.append(f"Overall average implied vol: {trades['implied_vol'].mean():.2%}")
        lines.append("")

    if "put_hedge_cost" in trades.columns and "shares" in trades.columns:
        total_put_hedge = (trades["put_hedge_cost"] * trades["shares"]).sum()
        lines.append("Theoretical cost of hedging entire portfolio with ATM puts (at entry):")
        lines.append(f"  Total put hedge cost (sum over all trades at entry): ${total_put_hedge:,.2f}")
        lines.append("")

    # Stock P&L vs equivalent ATM call P&L
    if "pnl" in trades.columns and "atm_call_price" in trades.columns and "exit_price" in trades.columns and "entry_price" in trades.columns and "shares" in trades.columns:
        stock_pnl = trades["pnl"].sum()
        # Per trade: call P&L = (max(0, exit - strike) - atm_call_price) * shares (simplified: payoff at expiry style)
        def call_pnl_row(r):
            S = float(r["entry_price"])
            E = float(r["exit_price"])
            c = float(r["atm_call_price"])
            sh = float(r["shares"])
            payoff = max(0.0, E - S)
            return (payoff - c) * sh
        trades["_call_pnl"] = trades.apply(call_pnl_row, axis=1)
        call_pnl = trades["_call_pnl"].sum()
        lines.append("Comparison: Stock P&L vs equivalent ATM call P&L (theoretical at-expiry payoff):")
        lines.append(f"  Total stock P&L:     ${stock_pnl:,.2f}")
        lines.append(f"  Total ATM call P&L:   ${call_pnl:,.2f}")
        lines.append("")

    lines.append("=" * 60)
    report = "\n".join(lines)
    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w") as f:
            f.write(report)
    return report


def main():
    p = argparse.ArgumentParser(description="Options analysis report from trades.csv")
    p.add_argument("--trades", default="output/backtests/trades.csv", help="Path to trades CSV")
    p.add_argument("--output", default=None, help="Write report to this file (default: print only)")
    p.add_argument("--risk-free-rate", type=float, default=0.04, help="Risk-free rate for BS")
    p.add_argument("--expiry-days", type=int, default=30, help="Option expiry in days for BS")
    args = p.parse_args()
    report = run_report(args.trades, args.output, args.risk_free_rate, args.expiry_days)
    print(report)
    if args.output:
        print(f"Report written to {args.output}")


if __name__ == "__main__":
    main()
