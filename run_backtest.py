"""
Backtest Runner
================
Entry point for the Trend Signal Engine backtesting system.

Usage:
    python run_backtest.py                             # default config
    python run_backtest.py --config my_config.yaml     # custom config
    python run_backtest.py --tickers AAPL MSFT NVDA    # arbitrary tickers (config or CLI)
    python run_backtest.py --walk-forward              # walk-forward test
    python run_backtest.py --ic-decay                  # IC decay analysis
    python run_backtest.py --cost-sensitivity          # transaction cost sensitivity

Output:
    - Console performance summary
    - output/backtests/equity_curve.png
    - output/backtests/trades.csv
    - output/backtests/daily_equity.csv
    - output/backtests/regime_returns.png
    - output/backtests/ic_decay.png  (with --ic-decay)
    - output/backtests/cost_sensitivity_report.csv  (with --cost-sensitivity)
"""

import argparse
import math
import os
from datetime import datetime

from backtesting import (
    load_config,
    compute_ic_decay,
    compute_rank_ic_decay,
    best_ic_horizon,
    run_walk_forward,
    run_transaction_cost_sensitivity,
    run_execution_costs_sensitivity,
    plot_equity_curve,
    plot_ic_decay,
    plot_regime_returns,
    plot_position_count,
)

from backtest.engine import BacktestEngine
from config import DEV_MODE, setup_logging, get_effective_tickers, apply_dev_mode


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Trend Signal Engine — Quant Research Backtester",
    )
    p.add_argument("--config", default="backtest_config.yaml",
                   help="Path to YAML config (default: backtest_config.yaml)")
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Override tickers (space-separated)")
    p.add_argument("--mode", choices=["price", "full", "learned"], default=None,
                   help="Signal mode: 'price' | 'full' | 'learned' (dynamic weights)")
    p.add_argument("--holding-period", type=int, default=None,
                   help="Override holding_period_days in backtest config")
    p.add_argument("--long-only", action="store_true",
                   help="Disable shorts: negative scores ⇒ no position (long-only)")
    p.add_argument("--walk-forward", action="store_true",
                   help="Run walk-forward analysis instead of single backtest")
    p.add_argument("--ic-decay", action="store_true",
                   help="Run IC decay analysis after backtest")
    p.add_argument("--cost-sensitivity", action="store_true",
                   help="Run transaction cost sensitivity analysis (multiple cost scenarios)")
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose (DEBUG-level) logging")
    return p.parse_args()


# ------------------------------------------------------------------
# Pretty-print helpers
# ------------------------------------------------------------------

SEP = "=" * 65

def _print_header(config, tickers):
    shorts_lbl = "ON" if config.enable_shorts else "OFF"
    regime_lbl = "ON" if config.regime_enabled else "OFF"
    sector_lbl = "ON" if config.sector_enabled else "OFF"
    mode_labels = {
        "price": "Trend + Volatility",
        "full": "All agents",
        "learned": "Learned weights",
    }
    mode_lbl = mode_labels.get(config.signal_mode, config.signal_mode)

    print(f"\n{SEP}")
    print("  Trend Signal Engine — Quant Research Backtester")
    print(SEP)
    print(f"  Period        : {config.start_date}  →  {config.end_date}")
    print(f"  Tickers       : {len(tickers)}")
    print(f"  Capital       : ${config.initial_capital:,.0f}")
    print(f"  Max Positions : {config.max_positions}")
    hold_lbl = f"{config.holding_period_days} days"
    if getattr(config, "dynamic_holding_enabled", False):
        hold_lbl += " (dynamic by signal/strength)"
    print(f"  Holding       : {hold_lbl}")
    print(f"  Slippage      : {config.slippage_bps} bps")
    print(f"  Commission    : ${config.commission_per_trade:.2f} / trade")
    print(f"  Min |Signal|  : {config.min_signal_strength}")
    print(f"  Signal Mode   : {mode_lbl}")
    print(f"  Shorts        : {shorts_lbl}")
    print(f"  Regime        : {regime_lbl}")
    print(f"  Sector Cap    : {sector_lbl}")
    if getattr(config, "cross_sectional_ranking", False):
        print(f"  Cross-section : TOP_LONGS={getattr(config, 'top_longs', 5)}  TOP_SHORTS={getattr(config, 'top_shorts', 5)}  "
              f"market_neutral={getattr(config, 'market_neutral', True)}")
    risk_sizing = getattr(config, "position_sizing", "equal")
    risk_max = getattr(config, "max_position_pct_of_equity", 0.25)
    risk_stop = getattr(config, "stop_loss_pct", 0.0)
    risk_take = getattr(config, "take_profit_pct", 0.0)
    risk_parts = [f"sizing={risk_sizing}", f"max_pos={risk_max:.0%}"]
    if risk_stop > 0:
        risk_parts.append(f"stop={-risk_stop:.1%}")
    if risk_take > 0:
        risk_parts.append(f"take={risk_take:.1%}")
    print(f"  Risk          : {', '.join(risk_parts)}")
    print(SEP)
    print()


def _print_metrics(m):
    dur = m.get("duration_stats", {})
    max_dd_dur = m.get("max_drawdown_duration", 0)
    avg_dd = m.get("avg_drawdown", 0.0)
    avg_dd_dur = m.get("avg_drawdown_duration", 0.0)
    avg_turn = m.get("avg_daily_turnover", 0.0)
    ann_turn = m.get("annualised_turnover", 0.0)
    turn_cost_bps = m.get("turnover_cost_drag_bps", 0.0)
    sharpe_ci_low = m.get("sharpe_ci_low", None)
    sharpe_ci_high = m.get("sharpe_ci_high", None)
    sortino_ci_low = m.get("sortino_ci_low", None)
    sortino_ci_high = m.get("sortino_ci_high", None)
    calmar_ci_low = m.get("calmar_ci_low", None)
    calmar_ci_high = m.get("calmar_ci_high", None)

    print(f"\n{SEP}")
    print("  Backtest Results")
    print(SEP)
    print()
    print(f"  Total Trades            : {m['total_trades']:,}")
    print(f"    Bullish                : {m['bullish_trades']:,}")
    print(f"    Bearish                : {m['bearish_trades']:,}")
    print()
    print(f"  Win Rate                : {m['win_rate']:.1%}")
    print(f"  Average Return          : {m['average_return']:.2%}")
    print(f"  Profit Factor           : {m['profit_factor']:.2f}")
    print(f"  Sharpe Ratio            : {m['sharpe_ratio']:.2f}")
    if sharpe_ci_low is not None and sharpe_ci_high is not None:
        print(f"    95% CI                : [{sharpe_ci_low:.2f}, {sharpe_ci_high:.2f}]")
    print(f"  Sortino Ratio           : {m['sortino_ratio']:.2f}")
    if sortino_ci_low is not None and sortino_ci_high is not None:
        print(f"    95% CI                : [{sortino_ci_low:.2f}, {sortino_ci_high:.2f}]")
    print(f"  Calmar Ratio            : {m['calmar_ratio']:.2f}")
    if calmar_ci_low is not None and calmar_ci_high is not None:
        print(f"    95% CI                : [{calmar_ci_low:.2f}, {calmar_ci_high:.2f}]")
    print(f"  Max Drawdown            : {m['max_drawdown']:.2%}")
    if max_dd_dur or avg_dd or avg_dd_dur:
        print(f"  Max DD Duration (days)  : {max_dd_dur}")
        print(f"  Avg Drawdown            : {avg_dd:.2%}")
        print(f"  Avg DD Duration (days)  : {avg_dd_dur:.1f}")
    if avg_turn or ann_turn:
        print()
        print(f"  Avg Daily Turnover      : {avg_turn:.2f}")
        print(f"  Annual Turnover         : {ann_turn:.2f}")
        if turn_cost_bps:
            print(f"  Turnover Cost Drag (bps): {turn_cost_bps:.1f}")
    print(f"  Signal Accuracy         : {m['signal_accuracy']:.1%}")
    print(f"  Information Coefficient : {m['information_coefficient']:.4f}")
    print(f"  Rank IC (Spearman)      : {m['rank_ic']:.4f}")
    print()
    print(f"  Avg Holding (days)      : {dur.get('mean', 0)}")
    print(f"  Med Holding (days)      : {dur.get('median', 0)}")
    print(f"  Min / Max Holding       : {dur.get('min', 0)} / {dur.get('max', 0)}")
    print()
    print(f"  Starting Capital        : ${m['starting_capital']:>14,.2f}")
    print(f"  Final Capital           : ${m['final_capital']:>14,.2f}")
    print(f"  Total PnL               : ${m['total_pnl']:>14,.2f}")
    print(f"  Total Return            : {m['total_return']:.2%}")
    if "gross_return" in m and "net_return" in m:
        print(f"  Gross Return (pre-cost)  : {m['gross_return']:.2%}")
        print(f"  Net Return (after cost)  : {m['net_return']:.2%}")
    if "total_transaction_costs" in m and m.get("total_transaction_costs", 0) != 0:
        print(f"  Total Transaction Costs  : ${m['total_transaction_costs']:>14,.2f}")
        print(f"  Average Cost per Trade   : ${m.get('average_cost_per_trade', 0):>14,.2f}")

    # Risk Metrics (VaR)
    v95 = m.get("var_95_1d")
    v99 = m.get("var_99_1d")
    cvar = m.get("cvar_95")
    v95_5d = m.get("var_95_5d")
    breach_count = m.get("var_breach_count_95", 0)
    breach_rate = m.get("var_breach_rate_95")

    def _valid(x):
        return x is not None and not (isinstance(x, float) and math.isnan(x))

    if _valid(v95):
        print()
        print("  Risk Metrics (VaR):")
        print(f"    Historical VaR (95%, 1d) : {v95:.2%}")
        if _valid(v99):
            print(f"    Historical VaR (99%, 1d) : {v99:.2%}")
        if _valid(cvar):
            print(f"    CVaR / Expected Shortfall  : {cvar:.2%}")
        if _valid(v95_5d):
            print(f"    VaR (95%, 5d scaled)      : {v95_5d:.2%}")
        print(f"    VaR breaches (95%)        : {breach_count} days (expected ~5% of trading days)")
        if _valid(breach_rate):
            print(f"    VaR breach rate           : {breach_rate:.1%} (should be ~5% if model correct)")

    print()


def _print_signal_breakdown(trades):
    from backtesting import compute_win_rate, compute_average_return

    bullish = trades[trades["signal"] == "Bullish"]
    bearish = trades[trades["signal"] == "Bearish"]

    if not bullish.empty:
        print(f"  Bullish Win Rate        : {compute_win_rate(bullish):.1%}")
        print(f"  Bullish Avg Return      : {compute_average_return(bullish):.2%}")
    if not bearish.empty:
        print(f"  Bearish Win Rate        : {compute_win_rate(bearish):.1%}")
        print(f"  Bearish Avg Return      : {compute_average_return(bearish):.2%}")
    if not bullish.empty or not bearish.empty:
        print()


# ------------------------------------------------------------------
# IC decay runner
# ------------------------------------------------------------------

def _run_ic_decay(result, config):
    print("  Running IC decay analysis…")
    lags = config.ic_decay_lags

    pearson_ic = compute_ic_decay(result.price_data, result.signal_data, lags)
    spearman_ic = compute_rank_ic_decay(result.price_data, result.signal_data, lags)

    # IC decay curve (logged as table)
    print(f"\n  IC decay (horizons: {lags})")
    print(f"  {'Lag':>5}  {'Pearson IC':>12}  {'Rank IC':>12}")
    print(f"  {'─'*5}  {'─'*12}  {'─'*12}")
    for lag, pic, ric in zip(lags, pearson_ic, spearman_ic):
        print(f"  {lag:>5}  {pic:>12.4f}  {ric:>12.4f}")
    print()

    best_lag_p, best_ic_p = best_ic_horizon(lags, pearson_ic)
    best_lag_r, best_ic_r = best_ic_horizon(lags, spearman_ic)
    print(f"  Best horizon (Pearson): {best_lag_p}d  (IC = {best_ic_p:.4f})")
    print(f"  Best horizon (Rank IC): {best_lag_r}d  (IC = {best_ic_r:.4f})")
    print()

    p1 = plot_ic_decay(pearson_ic, lags, title="Pearson IC Decay")
    print(f"  Saved → {p1}")

    p2 = plot_ic_decay(
        spearman_ic, lags,
        title="Rank IC Decay",
        save_path="output/backtests/rank_ic_decay.png",
    )
    print(f"  Saved → {p2}")


# ------------------------------------------------------------------
# Transaction cost sensitivity runner
# ------------------------------------------------------------------

def _run_cost_sensitivity(config, tickers):
    """Run backtests across cost scenarios and output a report (console + CSV)."""
    print("\n  Transaction cost sensitivity analysis")
    print("  (Strategy performance across different slippage & commission assumptions)\n")

    scenarios = getattr(config, "cost_sensitivity_scenarios", None)
    df = run_transaction_cost_sensitivity(config, tickers, scenarios=scenarios, verbose=True)

    # Console report
    print(f"\n{SEP}")
    print("  Transaction Cost Sensitivity Report")
    print(SEP)
    print()
    # Format key columns for display
    report = df.copy()
    report["total_return"] = report["total_return"].apply(lambda x: f"{x:.2%}")
    report["max_drawdown"] = report["max_drawdown"].apply(lambda x: f"{x:.2%}")
    report["win_rate"] = report["win_rate"].apply(lambda x: f"{x:.1%}")
    report["total_pnl"] = report["total_pnl"].apply(lambda x: f"${x:,.0f}")
    report["sharpe_ratio"] = report["sharpe_ratio"].round(2)
    report["profit_factor"] = report["profit_factor"].round(2)
    print(report.to_string(index=False))
    print()

    # Save full numeric report
    out_path = getattr(config, "cost_sensitivity_report_path", "output/backtests/cost_sensitivity_report.csv")
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"  Report saved → {out_path}")
    print()

    # Summary: profitability survives?
    if not df.empty:
        baseline = df.iloc[0]
        print(f"  Baseline (zero cost) : Return {baseline['total_return']:.2%}  Sharpe {baseline['sharpe_ratio']:.2f}")
        worst_idx = df["total_return"].idxmin()
        worst = df.loc[worst_idx]
        print(f"  Worst scenario      : Return {worst['total_return']:.2%}  (slippage={worst['slippage_bps']} bps, commission=${worst['commission_per_trade']:.2f})")
        profitable = (df["total_return"] > 0).sum()
        print(f"  Profitable scenarios: {profitable} / {len(df)}")
    print(SEP)


# ------------------------------------------------------------------
# Walk-forward runner
# ------------------------------------------------------------------

def _run_walk_forward(config, tickers):
    results, summary_df = run_walk_forward(config, tickers)

    print(f"\n{SEP}")
    print(f"  Walk-Forward Validation — OOS summary  ({len(results)} windows)")
    print(SEP)

    if not summary_df.empty:
        disp = summary_df.copy()
        disp["oos_max_drawdown"] = disp["oos_max_drawdown"].apply(lambda x: f"{x:.2%}")
        disp["oos_directional_accuracy"] = disp["oos_directional_accuracy"].apply(lambda x: f"{x:.1%}")
        disp["oos_total_return"] = disp["oos_total_return"].apply(lambda x: f"{x:.2%}")
        disp["oos_sharpe"] = disp["oos_sharpe"].round(2)
        disp["oos_information_coefficient"] = disp["oos_information_coefficient"].round(4)
        disp["oos_rank_ic"] = disp["oos_rank_ic"].round(4)
        cols = [
            "window", "test_start", "test_end", "oos_sharpe", "oos_max_drawdown",
            "oos_directional_accuracy", "oos_information_coefficient", "oos_rank_ic",
            "oos_total_return", "oos_total_trades",
        ]
        cols = [c for c in cols if c in disp.columns]
        print()
        print(disp[cols].to_string(index=False))
        print()
        if "oos_sharpe" in summary_df.columns:
            print(f"  Mean OOS Sharpe     : {summary_df['oos_sharpe'].mean():.2f}")
            print(f"  Mean OOS Max DD     : {summary_df['oos_max_drawdown'].mean():.2%}")
            print(f"  Mean OOS Dir. Acc.  : {summary_df['oos_directional_accuracy'].mean():.1%}")
            print(f"  Mean OOS IC         : {summary_df['oos_information_coefficient'].mean():.4f}")
        print()
    else:
        for i, res in enumerate(results, 1):
            m = res.metrics
            print(f"\n  Window {i}: {m['total_trades']} trades | "
                  f"Sharpe {m['sharpe_ratio']:.2f} | "
                  f"Return {m['total_return']:.2%} | "
                  f"Max DD {m['max_drawdown']:.2%}")
    print()


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def _resolve_tickers(config):
    """Return tickers from config or CLI; fallback to main.TICKERS. In DEV_MODE, capped to 10."""
    try:
        from main import TICKERS
        fallback = list(TICKERS)
    except Exception:
        fallback = [
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META", "TSLA",
            "JPM", "V", "JNJ", "WMT", "PG", "XOM", "UNH", "HD",
            "BAC", "MA", "ABBV", "PFE", "KO", "PEP", "MRK", "AVGO",
            "COST", "TMO", "CSCO", "ACN", "MCD", "NKE", "ADBE",
            "CRM", "AMD", "INTC", "ORCL", "IBM", "GS", "CAT", "BA",
            "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "VTI",
        ]
    return get_effective_tickers(config.tickers or [], fallback)


def _print_outputs_produced(config, result, outputs_written: list) -> None:
    """Print a clear 'what was produced' block at the end of every backtest run."""
    print(f"\n{SEP}")
    print("  OUTPUTS PRODUCED")
    print(SEP)
    if not outputs_written:
        print("  No trades generated — no files written.")
        print(f"  Period run: {config.start_date}  →  {config.end_date}")
        print("  Check data, signal thresholds, or config (e.g. min_signal_strength).")
    else:
        n_trades = len(result.trades) if result.trades is not None and not result.trades.empty else 0
        m = result.metrics if getattr(result, "metrics", None) else {}
        sharpe = m.get("sharpe_ratio", 0.0)
        total_ret = m.get("total_return", 0.0)
        print(f"  Result    : {n_trades} trades  |  Sharpe {sharpe:.3f}  |  Return {total_ret:.2%}")
        print()
        for path in outputs_written:
            if path and path.strip():
                print(f"  → {path}")
    print(SEP)
    print("  Backtest complete.")
    print(SEP)


def main():
    args = parse_args()
    # Verbose flag overrides DEV_MODE; otherwise use DEV_MODE for convenience.
    setup_logging(verbose=args.verbose or DEV_MODE)
    config = load_config(args.config)

    if args.tickers:
        config.tickers = args.tickers
    if args.mode:
        config.signal_mode = args.mode
    if args.holding_period is not None:
        config.holding_period_days = args.holding_period
    if args.long_only:
        config.long_only = True
        # Ensure shorts are disabled in long-only mode.
        config.allow_shorts = False
        config.enable_shorts = False

    apply_dev_mode(config)
    tickers = _resolve_tickers(config)

    # --- Walk-forward mode ---
    if args.walk_forward:
        _print_header(config, tickers)
        _run_walk_forward(config, tickers)
        print(f"{SEP}\n  Walk-forward complete.\n{SEP}")
        return

    # --- Standard single-window backtest ---
    _print_header(config, tickers)

    engine = BacktestEngine(config=config, config_path=args.config)
    result = engine.run_backtest(tickers)

    if result.trades.empty:
        _print_outputs_produced(config, result, outputs_written=[])
        return

    _print_metrics(result.metrics)
    _print_signal_breakdown(result.trades)

    # ------------------------------------------------------------------
    # Prominent summary block + write to latest_summary.txt
    # ------------------------------------------------------------------
    m = result.metrics
    start_dt = datetime.strptime(config.start_date, "%Y-%m-%d")
    end_dt = datetime.strptime(config.end_date, "%Y-%m-%d")
    years = max((end_dt - start_dt).days / 365.25, 1e-6)

    sharpe = m.get("sharpe_ratio", 0.0)
    sortino = m.get("sortino_ratio", 0.0)
    total_ret = m.get("total_return", 0.0)
    cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0 if total_ret > -1.0 else -1.0
    max_dd = m.get("max_drawdown", 0.0)
    win_rate = m.get("win_rate", 0.0)
    trades_per_year = m.get("total_trades", 0) / years
    turnover = m.get("annualised_turnover", 0.0)
    cost_drag_bps = m.get("turnover_cost_drag_bps", 0.0)
    cost_drag = cost_drag_bps / 10_000.0

    trades_df = result.trades
    long_pnl = short_pnl = 0.0
    if trades_df is not None and not trades_df.empty:
        long_pnl = float(trades_df.loc[trades_df["direction"] > 0, "pnl"].sum())
        short_pnl = float(trades_df.loc[trades_df["direction"] < 0, "pnl"].sum())
    start_cap = m.get("starting_capital", config.initial_capital)
    long_pnl_frac = long_pnl / start_cap if start_cap else 0.0
    short_pnl_frac = short_pnl / start_cap if start_cap else 0.0

    summary_block = (
        "\n" + "=" * 60 + "\n"
        "  BACKTEST RESULTS SUMMARY\n"
        + "=" * 60 + "\n"
        f"  Sharpe Ratio         : {sharpe:.3f}\n"
        f"  Sortino Ratio        : {sortino:.3f}\n"
        f"  CAGR                 : {cagr:.2%}\n"
        f"  Max Drawdown         : {max_dd:.2%}\n"
        f"  Win Rate             : {win_rate:.1%}\n"
        f"  Trades per Year      : {trades_per_year:.0f}\n"
        f"  Annual Turnover      : {turnover:.0%}\n"
        f"  Est. Cost Drag (ann) : {cost_drag:.2%}\n"
        f"  Long P&L contrib     : {long_pnl_frac:.2%}\n"
        f"  Short P&L contrib    : {short_pnl_frac:.2%}\n"
        + "=" * 60 + "\n"
    )

    print(summary_block)

    out_dir = os.path.join("output", "backtests")
    os.makedirs(out_dir, exist_ok=True)
    latest_summary_path = os.path.join(out_dir, "latest_summary.txt")
    with open(latest_summary_path, "w") as fh:
        fh.write(summary_block)

    # --- Save CSVs ---
    out_dir = os.path.join("output", "backtests")
    os.makedirs(out_dir, exist_ok=True)

    if config.save_trades_csv:
        result.trades.to_csv(config.trades_csv_path, index=False)
        print(f"  Trades CSV  → {config.trades_csv_path}")

    if config.save_equity_csv:
        result.daily_equity.to_csv(config.equity_csv_path, index=False)
        print(f"  Equity CSV  → {config.equity_csv_path}")

    # --- Plots ---
    if not result.daily_equity.empty:
        p = plot_equity_curve(
            result.daily_equity, config.initial_capital, config.equity_curve_path,
        )
        print(f"  Equity plot → {p}")

    if not result.trades.empty:
        p = plot_regime_returns(result.trades)
        if p:
            print(f"  Regime plot → {p}")

    if not result.daily_equity.empty:
        p = plot_position_count(result.daily_equity)
        if p:
            print(f"  Pos. count  → {p}")

    # --- Optional IC decay ---
    if args.ic_decay and result.price_data:
        print()
        _run_ic_decay(result, config)

    # --- Optional transaction cost sensitivity ---
    if args.cost_sensitivity:
        print()
        _run_cost_sensitivity(config, tickers)

    # --- Execution costs sensitivity (config: execution_costs.sensitivity_test) ---
    if getattr(config, "execution_costs_sensitivity_test", False):
        print()
        print("  Execution cost sensitivity (total bps scenarios)…")
        run_execution_costs_sensitivity(config, tickers)

    # Experiment snapshot (timestamped run folder)
    exp_dir = engine.save_experiment_snapshot(result)
    print(f"  Experiment snapshot → {exp_dir}")

    # Always end with a clear "what was produced" block
    outputs_list = [latest_summary_path, exp_dir]
    if config.save_trades_csv:
        outputs_list.append(config.trades_csv_path)
    if config.save_equity_csv:
        outputs_list.append(config.equity_csv_path)
    if not result.daily_equity.empty:
        outputs_list.append(config.equity_curve_path)
    outputs_list.append(os.path.join("output", "backtests", "regime_returns.png"))
    outputs_list.append(os.path.join("output", "backtests", "position_count.png"))
    _print_outputs_produced(config, result, outputs_written=outputs_list)


if __name__ == "__main__":
    main()
