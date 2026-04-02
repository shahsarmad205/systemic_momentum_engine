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

from backtest.engine import BacktestEngine
from backtesting import (
    best_ic_horizon,
    compute_ic_decay,
    compute_rank_ic_decay,
    load_config,
    plot_equity_curve,
    plot_ic_decay,
    plot_position_count,
    plot_regime_returns,
    run_execution_costs_sensitivity,
    run_transaction_cost_sensitivity,
    run_walk_forward,
)
from config import DEV_MODE, apply_dev_mode, get_effective_tickers, setup_logging

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Trend Signal Engine — Quant Research Backtester",
    )
    p.add_argument("--config", default="backtest_config.yaml",
                   help="Path to YAML config (default: backtest_config.yaml)")
    p.add_argument(
        "--learned-weights",
        default=None,
        metavar="PATH",
        help="Override learned_weights_path (JSON). Use when comparing saved weight files.",
    )
    p.add_argument(
        "--signal-confidence-mult",
        type=float,
        default=None,
        metavar="X",
        help="Override signal_confidence_multiplier (and bull/bear/crisis variants) e.g. 0.5 vs 0.8 so "
        "learned-score magnitude affects trade selection.",
    )
    p.add_argument("--tickers", nargs="+", default=None,
                   help="Override tickers (space-separated)")
    p.add_argument("--mode", choices=["price", "full", "learned", "ml", "ensemble"], default=None,
                   help="Signal mode: 'price' | 'full' | 'learned' | 'ml' | 'ensemble'")
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
    p.add_argument(
        "--scan-signal-thresholds",
        action="store_true",
        help="Scan std-based abs(adjusted_score) thresholds (0.1σ,0.2σ,0.3σ,0.5σ) and print a comparison table",
    )
    p.add_argument(
        "--scan-signal-confidence",
        action="store_true",
        help="Scan signal_confidence_multiplier values (0.1,0.2,0.3,0.5) vs win rate / Sharpe / CAGR",
    )
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
        "ml": "Single ML model",
        "ensemble": "ML ensemble",
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
    ann_turn_old = m.get("annualised_turnover_old", None)
    if ann_turn_old is None:
        ann_turn_old = m.get("annualised_turnover_old", 0.0)
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
    if "net_sharpe_ratio" in m:
        print(f"  Net Sharpe Ratio        : {m['net_sharpe_ratio']:.2f}")
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
        if ann_turn_old is not None:
            print(f"  Annual Turnover (old)  : {ann_turn_old:.2f}")
        print(f"  Avg Daily Turnover      : {avg_turn:.2f}")
        print(f"  Annual Turnover         : {ann_turn:.2f}")
        if turn_cost_bps:
            print(f"  Turnover Cost Drag (bps): {turn_cost_bps:.1f}")

        # Detailed turnover cross-checks
        if "turnover_shares_based" in m:
            print(f"  Turnover (shares-based): {m.get('turnover_shares_based', 0.0):.2f}")
        if "turnover_changes_over_avg_positions" in m:
            print(
                f"  Position changes / avg positions held: "
                f"{m.get('turnover_changes_over_avg_positions', 0.0):.2f}"
            )
        if "turnover_trades_per_year_x_avg_position_size_based" in m:
            print(
                "  Turnover (trades_per_year*avg_position_size): "
                f"{m.get('turnover_trades_per_year_x_avg_position_size_based', 0.0):.2f}"
            )
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
    from backtesting import compute_average_return, compute_win_rate

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
    for lag, pic, ric in zip(lags, pearson_ic, spearman_ic, strict=False):
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

def _run_cost_sensitivity(config, tickers, price_data=None, signal_data=None):
    """Run backtests across cost scenarios and output a report (console + CSV)."""
    print("\n  Transaction cost sensitivity analysis")
    print("  (Strategy performance across different slippage & commission assumptions)\n")

    scenarios = getattr(config, "cost_sensitivity_scenarios", None)
    df = run_transaction_cost_sensitivity(
        config, tickers, scenarios=scenarios, verbose=True,
        price_data=price_data, signal_data=signal_data
    )

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

    if args.learned_weights:
        config.learned_weights_path = args.learned_weights

    if args.signal_confidence_mult is not None:
        m = float(args.signal_confidence_mult)
        config.signal_confidence_multiplier = m
        for attr in (
            "signal_confidence_multiplier_bull",
            "signal_confidence_multiplier_bear",
            "signal_confidence_multiplier_crisis",
        ):
            if hasattr(config, attr):
                setattr(config, attr, m)

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

    if args.scan_signal_thresholds:
        multipliers = [0.1, 0.2, 0.3, 0.5]
        orig_std_mult = getattr(config, "signal_threshold_std_multiplier", None)

        # Use the configured window to compute CAGR and trades/year.
        start_dt = datetime.strptime(config.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(config.end_date, "%Y-%m-%d")
        years = max((end_dt - start_dt).days / 365.25, 1e-6)

        print(f"\n{SEP}")
        print("  Signal Threshold Scan (std-based abs(adjusted_score))")
        print(SEP)
        print("  Signal-score std used per run comes from the backtester.")
        print()

        # Phase 1: Prepare data and signals ONCE
        engine_prep = BacktestEngine(config=config, config_path=args.config)
        price_data, signal_data = engine_prep.prepare_universe_data(tickers)

        rows = []
        for mult in multipliers:
            config.signal_threshold_std_multiplier = float(mult)
            engine = BacktestEngine(config=config, config_path=args.config)
            result = engine.run_backtest(tickers, price_data=price_data, signal_data=signal_data)
            m = result.metrics

            total_ret = float(m.get("total_return", 0.0) or 0.0)
            cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0 if total_ret > -1.0 else -1.0
            win_rate = float(m.get("win_rate", 0.0) or 0.0)
            trades_per_year = float(m.get("total_trades", 0) or 0) / years
            sharpe = float(m.get("sharpe_ratio", 0.0) or 0.0)
            max_dd = float(m.get("max_drawdown", 0.0) or 0.0)

            rows.append({
                "sigma_mult": mult,
                "win_rate": win_rate,
                "trades_per_year": trades_per_year,
                "sharpe": sharpe,
                "cagr": cagr,
                "max_drawdown": max_dd,
            })

        # Restore config
        config.signal_threshold_std_multiplier = orig_std_mult

        # Print comparison table
        print("  Comparison Table")
        print("  " + "-" * 90)
        header = (
            f"{'σ (mult)':>9} | {'win_rate':>9} | {'trades/yr':>10} | {'Sharpe':>7} | "
            f"{'CAGR':>8} | {'max_drawdown':>14}"
        )
        print(header)
        print("  " + "-" * 90)
        for r in rows:
            print(
                f"  {r['sigma_mult']:>9.1f} | {r['win_rate']*100:>8.2f}% | {r['trades_per_year']:>10.1f} | "
                f"{r['sharpe']:>7.2f} | {r['cagr']*100:>7.2f}% | {r['max_drawdown']*100:>13.2f}%"
            )
        print("  " + "-" * 90)
        return

    if args.scan_signal_confidence:
        multipliers = [0.1, 0.2, 0.3, 0.5, 0.65, 0.8]
        orig_scm = getattr(config, "signal_confidence_multiplier", None)

        start_dt = datetime.strptime(config.start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(config.end_date, "%Y-%m-%d")
        years = max((end_dt - start_dt).days / 365.25, 1e-6)

        print(f"\n{SEP}")
        print("  Signal confidence scan (abs(raw_score) vs mult × rolling 60d std)")
        print(SEP)

        # Phase 1: Prepare data and signals ONCE
        engine_prep = BacktestEngine(config=config, config_path=args.config)
        price_data, signal_data = engine_prep.prepare_universe_data(tickers)

        rows = []
        for mult in multipliers:
            config.signal_confidence_multiplier = float(mult)
            engine = BacktestEngine(config=config, config_path=args.config)
            result = engine.run_backtest(tickers, price_data=price_data, signal_data=signal_data)
            m = result.metrics

            total_ret = float(m.get("total_return", 0.0) or 0.0)
            cagr = (1.0 + total_ret) ** (1.0 / years) - 1.0 if total_ret > -1.0 else -1.0
            win_rate = float(m.get("win_rate", 0.0) or 0.0)
            trades_per_year = float(m.get("total_trades", 0) or 0) / years
            sharpe = float(m.get("sharpe_ratio", 0.0) or 0.0)
            net_sharpe = float(m.get("net_sharpe_ratio", sharpe) or 0.0)
            max_dd = float(m.get("max_drawdown", 0.0) or 0.0)

            rows.append({
                "mult": mult,
                "win_rate": win_rate,
                "trades_per_year": trades_per_year,
                "sharpe": sharpe,
                "net_sharpe": net_sharpe,
                "cagr": cagr,
                "max_drawdown": max_dd,
            })

        config.signal_confidence_multiplier = orig_scm

        print("  mult | win_rate | trades/yr | Sharpe | NetSharpe |   CAGR |   maxDD")
        print("  " + "-" * 72)
        for r in rows:
            print(
                f"  {r['mult']:.1f} | {r['win_rate']*100:7.2f}% | {r['trades_per_year']:9.1f} | "
                f"{r['sharpe']:6.2f} | {r['net_sharpe']:9.2f} | {r['cagr']*100:6.2f}% | {r['max_drawdown']*100:7.2f}%"
            )
        print("  " + "-" * 72)
        return

    engine = BacktestEngine(config=config, config_path=args.config)
    
    # Phase 1: Prepare data and signals ONCE for the entire universe (if not already done by a scan above)
    # Note: If a scan was run above, price_data/signal_data will already be in local scope.
    if 'price_data' not in locals() or 'signal_data' not in locals():
        price_data, signal_data = engine.prepare_universe_data(tickers)
    
    # Run the primary backtest with pre-loaded data
    result = engine.run_backtest(tickers, price_data=price_data, signal_data=signal_data)

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
    net_sharpe = m.get("net_sharpe_ratio", sharpe)
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
        f"  Net Sharpe Ratio     : {net_sharpe:.3f}\n"
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

    # --- Institutional Alpha Analysis (Held-to-Expiry) ---
    if "expiry_win_rate" in m:
        wr_realized = m.get("realized_expiry_win_rate", 0.0)
        avg_realized = m.get("realized_expiry_avg_return", 0.0)
        n_realized = m.get("exit_reason_counts", {}).get("expiry", 0)
        n_total = m.get("expiry_sample_size", 0)
        
        p50 = m.get("expiry_p_value_50", 1.0)
        p60 = m.get("expiry_p_value_60", 1.0)
        p70 = m.get("expiry_p_value_70", 1.0)
        ci_l = m.get("expiry_ci_low", 0.0)
        ci_h = m.get("expiry_ci_high", 1.0)
        
        summary_block += (
            "\n"
            "============================================================\n"
            "  INSTITUTIONAL ALPHA ANALYSIS (HELD-TO-EXPIRY)\n"
            "============================================================\n"
            f"  Expiry Win Rate:  {wr_realized:.1%}  (exit_reason == 'expiry' only)\n"
            f"  Avg Expiry Return: {avg_realized:+.3%}\n"
            f"  Sample Size: {n_realized:,} expiry trades (of {n_total:,} total)\n\n"
            
            "  EXIT REASON BREAKDOWN\n"
        )
        
        reasons = m.get("exit_reason_counts", {})
        for r, count in reasons.items():
            summary_block += f"  {r:<20} : {count}\n"
            
        summary_block += (
            "\n"
            "  STOP LOSS / EXPIRY REALIZED STATS\n"
            f"  Stop loss pct        : {m.get('stop_loss_pct', 0.0):.1%}\n"
            f"  Stopped avg return   : {m.get('stopped_avg_return', 0.0):+.3%}\n"
            f"  Expiry avg return    : {m.get('realized_expiry_avg_return', 0.0):+.3%}\n"
            f"  Expiry win rate      : {m.get('realized_expiry_win_rate', 0.0):.1%}\n\n"
            
            "  CLEAN PRE-2024 ANALYSIS (REALIZED EXPIRY)\n"
            f"  Sample Size (n)      : {m.get('clean_pre2024_n', 0)}\n"
            f"  Win Rate             : {m.get('clean_pre2024_wr', 0.0):.1%}\n"
            f"  95% Conf. Interval   : [{m.get('clean_pre2024_ci_low', 0.0):.1%}, {m.get('clean_pre2024_ci_high', 1.0):.1%}]\n"
            f"  p-value vs 50% null  : {m.get('clean_pre2024_p50', 1.0):.2e}\n"
            f"  p-value vs 70% null  : {m.get('clean_pre2024_p70', 1.0):.2e}\n"
            f"  p-value vs 80% null  : {m.get('clean_pre2024_p80', 1.0):.2e}\n\n"
            
            "  PAPER ALPHA (ALL SIGNALS - ILLUSTRATIVE)\n"
            f"  p-value vs 50% null  : {p50:.2e}\n"
            f"  p-value vs 60% null  : {p60:.2e}\n"
            f"  p-value vs 70% null  : {p70:.2e}\n"
            f"  95% Conf. Interval   : [{ci_l:.1%}, {ci_h:.1%}]\n\n"
            
            "  REGIME ROBUSTNESS (EXPIRY WIN RATE)\n"
        )
        
        rob = m.get("expiry_regime_robustness", {})
        for r_name in ["Bull", "Bear", "Sideways", "Crisis"]:
            r_data = rob.get(r_name, {"win_rate": 0.0, "count": 0})
            summary_block += f"  {r_name:<20} : {r_data['win_rate']:.1%} (n={r_data['count']})\n"
            
        summary_block += "\n  YEAR-BY-YEAR PERSISTENCE\n  "
        years_p = m.get("expiry_yearly_persistence", {})
        sorted_years = sorted(years_p.keys())
        y_parts = [f"{y}: {years_p[y]:.1%}" for y in sorted_years]
        summary_block += " | ".join(y_parts) + "\n"
        summary_block += "============================================================\n"

    # --- ADVANCED STATISTICAL AUDIT ---
    summary_block += (
        "\n"
        "============================================================\n"
        "  ADVANCED STATISTICAL AUDIT & RIGOR DIAGNOSTICS\n"
        "============================================================\n"
    )
    
    # 1. Grinold IR Decomposition
    g = m.get("grinold_ir", {})
    if g:
        summary_block += (
            "  INFORMATION RATIO DECOMPOSITION (Grinold)\n"
            f"  IC (Daily mean)      : {g.get('ic_mean', 0.0):.4f}\n"
            f"  Effective Breadth    : {g.get('effective_breadth', 0.0):.1f} independent bets/yr\n"
            f"  Predicted IR         : {g.get('predicted_ir_grinold', 0.0):.3f}\n"
            f"  Realized IR (t-stat) : {g.get('realized_ir', 0.0):.3f}\n\n"
        )
        
    # 2. Sharpe Significance (Lo 2002 HAC-adjusted)
    ls = m.get("sharpe_significance", {})
    if ls:
        summary_block += (
            "  SHARPE SIGNIFICANCE (Lo 2002 Correction)\n"
            f"  T-statistic          : {ls.get('t_stat', 0.0):.3f}\n"
            f"  Significant (>1.96)  : {'YES' if ls.get('is_significant') else 'NO'}\n"
            f"  Probabilistic Sharpe : {m.get('psr', 0.0):.1%}\n\n"
        )
        
    # 3. Win Rate MLE (Beta-Binomial)
    wm = m.get("win_rate_mle", {})
    if wm:
        ci = wm.get("ci_95", (0,0))
        summary_block += (
            "  WIN RATE MLE (BETA-BINOMIAL POSTERIOR)\n"
            f"  MAP Estimate         : {wm.get('map_win_rate', 0.0):.1%}\n"
            f"  95% Credible Int.    : [{ci[0]:.1%}, {ci[1]:.1%}]\n"
            f"  P(True WR > 70%)     : {wm.get('prob_gt_70', 0.0):.1%}\n\n"
        )
        
    # 4. Newey-West HAC Adjusted Sharpe
    nw = m.get("newey_west_sharpe_stats", {})
    if nw:
        summary_block += (
            "  AUTOCORRELATION ADJUSTMENT (Newey-West)\n"
            f"  Standard Sharpe      : {nw.get('std_sharpe', 0.0):.3f}\n"
            f"  HAC-Adjusted Sharpe  : {nw.get('nw_sharpe', 0.0):.3f}\n"
            f"  Inflation Factor     : {nw.get('inflation_factor', 1.0):.2f}x\n\n"
        )
        
    # 5. Kelly with Estimation Uncertainty
    ku = m.get("kelly_uncertainty", {})
    if ku:
        summary_block += (
            "  CONSERVATIVE KELLY (Bootstrap)\n"
            f"  Kelly Mean           : {ku.get('kelly_mean', 0.0):.1%}\n"
            f"  Kelly 25th Ptile     : {ku.get('kelly_25th', 0.0):.1%} (Recommended)\n"
            f"  Kelly 5th Ptile      : {ku.get('kelly_5th', 0.0):.1%}\n\n"
        )
        
    # 6. Fama-French Attribution
    ff = m.get("ff_attribution", {})
    if ff:
        summary_block += (
            "  FAMA-FRENCH 5-FACTOR ATTRIBUTION\n"
            f"  Annualized Alpha     : {ff.get('alpha_ann', 0.0):+.2%}\n"
            f"  Alpha t-stat         : {ff.get('alpha_t_stat', 0.0):.3f}\n"
            f"  R-Squared            : {ff.get('r_squared', 0.0):.1%}\n"
        )
    else:
        summary_block += "  FAMA-FRENCH ATTRIBUTION : [Skipped/No Internet]\n"
        
    summary_block += "============================================================\n"

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
        _run_cost_sensitivity(config, tickers, price_data=price_data, signal_data=signal_data)

    # --- Execution costs sensitivity (config: execution_costs.sensitivity_test) ---
    if getattr(config, "execution_costs_sensitivity_test", False):
        print()
        print("  Execution cost sensitivity (total bps scenarios)…")
        run_execution_costs_sensitivity(config, tickers, price_data=price_data, signal_data=signal_data)

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
