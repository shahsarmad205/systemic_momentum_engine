#!/usr/bin/env python3
"""
Live Paper Trading Runner
Run at 4:15pm ET after market close.
Generates signals from learned weights, then reconciles targets vs Alpaca and
places next-session orders (market / day — broker defaults).

Usage (from trend_signal_engine/):
  python run_live_trading.py              # dry run (safe; no orders)
  python run_live_trading.py --execute    # place orders (paper only per broker)
  python run_live_trading.py --close-all  # emergency close (interactive)

From Quant-project/: ../trend_signal_engine/run_live_trading.py or use repo run_live_trading.py launcher.

Pre-flight (before orders): non-empty signals (else prior snapshot), score dispersion,
then net cash flow: sum(target_value for to_open) − sum(market_value for to_close).
Abort only if net_cash_required > 0 and net_cash_required > buying_power + cash.
Otherwise print ``Net cash required: … – proceeding.`` and continue.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _chdir_root() -> None:
    os.chdir(_ROOT)


def _append_execution_skip_log(entry: dict[str, Any]) -> Path:
    log_dir = _ROOT / "output" / "live"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "execution_log.jsonl"
    with open(log_file, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, default=str) + "\n")
    return log_file


def generate_signals(as_of_date: str) -> pd.DataFrame | None:
    """
    Build features through as_of_date, score all tickers with LearnedWeights,
    return ranked universe (execution engine applies sizing / threshold).
    """
    from agents.weight_learning_agent.feature_builder import build_feature_matrix
    from agents.weight_learning_agent.weight_model import LearnedWeights

    from run_daily_signals import _detect_regime, compute_score

    config_path = _ROOT / "backtest_config.yaml"
    with open(config_path, encoding="utf-8") as fh:
        config = yaml.safe_load(fh) or {}
    tickers = config.get("tickers", [])
    if not tickers:
        print("  No tickers in backtest_config.yaml")
        return None

    weights_path = _ROOT / "output" / "learned_weights.json"
    if not weights_path.is_file():
        print(f"  Missing {weights_path}")
        return None

    lw = LearnedWeights.load(str(weights_path))

    lookback_start = (pd.Timestamp(as_of_date) - timedelta(days=400)).strftime("%Y-%m-%d")

    try:
        df = build_feature_matrix(
            tickers,
            start_date=lookback_start,
            end_date=as_of_date,
        )
    except Exception as e:  # noqa: BLE001
        print(f"  Feature build failed: {e}")
        return None

    if df.empty:
        print("  Feature matrix is empty")
        return None

    df["date"] = pd.to_datetime(df["date"])
    elts = pd.Timestamp(as_of_date)
    eligible = df[df["date"] <= elts].copy()
    if eligible.empty:
        print(f"  No feature rows at or before {as_of_date}")
        return None

    last_date = eligible["date"].max()
    latest = eligible[eligible["date"] == last_date].copy()
    print(f"  Feature end date: {last_date.date()}  (tickers: {len(latest)})")

    scores: dict[str, float] = {}
    for _, row in latest.iterrows():
        ticker = str(row["ticker"])
        scores[ticker] = compute_score(row, lw)

    ranked = (
        pd.DataFrame([{"ticker": t, "score": s} for t, s in scores.items()])
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    regime = _detect_regime(as_of_date)
    print(f"  Market regime: {regime}")
    if regime in ("Crisis",):
        ranked = ranked[ranked["score"] > ranked["score"].quantile(0.95)]
        print("  Crisis filter: top 5% signals only")

    if ranked.empty:
        print("  No signals left after regime filter")
        return None

    return ranked


def load_fallback_signals(as_of: str) -> tuple[pd.DataFrame | None, str]:
    """
    If today's signal grid is empty, use the most recent prior snapshot:
    output/live/signal_history.csv, else output/signals/YYYY-MM-DD_rankings.csv.
    """
    as_ts = pd.Timestamp(as_of)
    hist = _ROOT / "output" / "live" / "signal_history.csv"
    if hist.is_file():
        try:
            h = pd.read_csv(hist)
            if not h.empty and "date" in h.columns and "ticker" in h.columns and "score" in h.columns:
                h["date"] = pd.to_datetime(h["date"], errors="coerce")
                h = h.dropna(subset=["date"])
                prior = h[h["date"] < as_ts.normalize()]
                if not prior.empty:
                    last_d = prior["date"].max()
                    snap = prior.loc[prior["date"] == last_d, ["ticker", "score"]].copy()
                    snap["ticker"] = snap["ticker"].astype(str).str.strip()
                    snap["score"] = pd.to_numeric(snap["score"], errors="coerce")
                    snap = snap.dropna(subset=["ticker", "score"])
                    if not snap.empty:
                        snap = snap.sort_values("score", ascending=False).reset_index(drop=True)
                        return snap, f"signal_history:{last_d.strftime('%Y-%m-%d')}"
        except Exception:
            pass

    signals_dir = _ROOT / "output" / "signals"
    for i in range(1, 21):
        prev = (as_ts - timedelta(days=i)).strftime("%Y-%m-%d")
        p = signals_dir / f"{prev}_rankings.csv"
        if p.is_file():
            try:
                snap = pd.read_csv(p)
                if "ticker" in snap.columns and "score" in snap.columns:
                    snap = snap[["ticker", "score"]].copy()
                    snap["score"] = pd.to_numeric(snap["score"], errors="coerce")
                    snap = snap.dropna(subset=["ticker", "score"])
                    if not snap.empty:
                        snap = snap.sort_values("score", ascending=False).reset_index(drop=True)
                        return snap, f"rankings_csv:{prev}"
            except Exception:
                continue
    return None, ""


def check_score_dispersion(signals: pd.DataFrame) -> tuple[bool, str]:
    """Fail if all (finite) scores are identical — degenerate / stuck model."""
    if signals.empty or "score" not in signals.columns:
        return False, "empty_or_no_score_column"
    s = pd.to_numeric(signals["score"], errors="coerce").dropna()
    if len(s) < 2:
        return False, "fewer_than_2_finite_scores"
    std = float(s.std(ddof=0))
    rng = float(s.max() - s.min())
    if not (std == std) or std < 1e-10:
        return False, "score_std_near_zero"
    if rng < 1e-10:
        return False, "score_range_near_zero"
    return True, ""


def save_signal_history(
    signals: pd.DataFrame,
    date: str,
    execution_result: dict,
) -> None:
    output_dir = _ROOT / "output" / "live"
    output_dir.mkdir(parents=True, exist_ok=True)

    placed = execution_result.get("orders_placed") or []
    tickers_placed: set[str] = set()
    for o in placed:
        if isinstance(o, dict) and o.get("ticker"):
            tickers_placed.add(str(o["ticker"]))

    out = signals.copy()
    out["date"] = date
    out["executed"] = out["ticker"].astype(str).isin(tickers_placed)

    history_file = output_dir / "signal_history.csv"
    if history_file.exists():
        history = pd.read_csv(history_file)
        history = pd.concat([history, out], ignore_index=True)
        history = history.drop_duplicates(subset=["date", "ticker"], keep="last")
    else:
        history = out

    history.to_csv(history_file, index=False)
    print(f"  Signal history: {history_file}")


def main() -> None:
    _chdir_root()

    parser = argparse.ArgumentParser(description="Live paper trading (signals + Alpaca execution)")
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Submit orders (default: dry run only)",
    )
    parser.add_argument(
        "--close-all",
        action="store_true",
        help="Emergency: close all open positions (interactive confirm)",
    )
    parser.add_argument(
        "--date",
        type=str,
        default=None,
        help="As-of date for features/signals YYYY-MM-DD (default: today local)",
    )
    args = parser.parse_args()

    as_of = args.date or datetime.now().strftime("%Y-%m-%d")
    dry_run = not args.execute

    print()
    print(f"{'=' * 55}")
    print(f"  LIVE PAPER TRADING — {as_of}")
    print(f"  Mode: {'DRY RUN' if dry_run else 'EXECUTE (live orders)'}")
    print(f"{'=' * 55}")

    from brokers.alpaca_broker import AlpacaBroker
    from brokers.execution_engine import ExecutionEngine

    if args.close_all:
        broker = AlpacaBroker()
        print("  [!] EMERGENCY CLOSE ALL POSITIONS")
        confirm = input("  Type CONFIRM to proceed: ")
        if confirm == "CONFIRM":
            results = broker.close_all_positions()
            print(f"  Close submissions: {len(results)}")
        return

    broker: AlpacaBroker | None = None
    if not dry_run:
        broker = AlpacaBroker()
        if broker.is_market_open():
            print(
                "  [i] Market is open — closes can fill before buys; "
                "full rebalance in one run is supported."
            )
        else:
            print("  [!] Market is closed — new orders are usually queued for the next session.")
            print(
                "      Close sells may not fill until the open, so buying power can stay at $0 "
                "until then. For same-day rotate, run --execute during regular hours (or re-run "
                "opens after closes have filled)."
            )

    print()
    print("  Step 1: Generating signals...")
    fallback_label = ""
    signals = generate_signals(as_of)
    if signals is None or signals.empty:
        fb, fallback_label = load_fallback_signals(as_of)
        if fb is not None and not fb.empty:
            signals = fb
            print(
                f"  [!] Same-day signals empty — using fallback: {fallback_label} ({len(signals)} rows)"
            )
        else:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "dry_run": dry_run,
                "skipped": True,
                "reason": "preflight_no_signals",
                "detail": "generate_signals empty and no fallback",
                "as_of": as_of,
                "orders_placed": [],
                "orders_skipped": [],
            }
            print("  ABORT: No signals generated and no previous-day snapshot found.")
            print("         Check data, output/learned_weights.json, and output/live/signal_history.csv")
            _append_execution_skip_log(entry)
            return

    print(f"  Generated {len(signals)} ranked signals")

    ok_disp, disp_reason = check_score_dispersion(signals)
    if not ok_disp:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            "skipped": True,
            "reason": "preflight_scores_degenerate",
            "detail": disp_reason,
            "as_of": as_of,
            "fallback_source": fallback_label,
            "orders_placed": [],
            "orders_skipped": [],
        }
        print(f"  ABORT: Signal scores failed dispersion check ({disp_reason}) — model may be stuck.")
        _append_execution_skip_log(entry)
        signals_dir = _ROOT / "output" / "signals"
        signals_dir.mkdir(parents=True, exist_ok=True)
        signals.to_csv(signals_dir / f"{as_of}_rankings.csv", index=False)
        save_signal_history(signals, as_of, entry)
        return

    signals_dir = _ROOT / "output" / "signals"
    signals_dir.mkdir(parents=True, exist_ok=True)
    rankings_file = signals_dir / f"{as_of}_rankings.csv"
    signals.to_csv(rankings_file, index=False)
    print(f"  Saved: {rankings_file}")

    print()
    print("  Top signals:")
    print(f"  {'Rank':<5} {'Ticker':<8} {'Score':>10}")
    print(f"  {'-' * 28}")
    for i, (_, row) in enumerate(signals.head(8).iterrows(), 1):
        print(f"  {i:<5} {row['ticker']:<8} {row['score']:>10.4f}")

    if broker is None:
        try:
            broker = AlpacaBroker()
        except (FileNotFoundError, ValueError) as exc:
            if dry_run:
                print()
                print("  Step 2: Skipped — Alpaca not configured (dry run only).")
                print(f"  Reason: {exc}")
                print(
                    "  Fix: set ALPACA_API_KEY + ALPACA_SECRET_KEY, or fill "
                    "config/alpaca_config.yaml (see config/alpaca_config.example.yaml)."
                )
                print("  Rankings and signal files above were still written.")
                stub: dict[str, Any] = {
                    "timestamp": datetime.now().isoformat(),
                    "dry_run": True,
                    "skipped": True,
                    "reason": "alpaca_unavailable",
                    "detail": str(exc),
                    "n_open": 0,
                    "n_close": 0,
                    "n_hold": 0,
                    "orders_placed": [],
                    "orders_skipped": [],
                }
                log_path = _append_execution_skip_log(stub)
                print(f"  Logged skip: {log_path}")
                save_signal_history(signals, as_of, stub)
                print()
                print(f"{'=' * 55}")
                print("  Done (signals only). execution_log: output/live/execution_log.jsonl")
                print(f"{'=' * 55}")
                return
            raise

    engine = ExecutionEngine(broker)

    try:
        account = broker.get_account()
        engine._dd_current_equity = float(account["equity"])
        dd_mult = engine._get_drawdown_multiplier()
        current_positions = broker.get_positions()
        target = engine.compute_target_portfolio(signals, account, verbose=False)
        if len(target) > 0 and dd_mult != 1.0:
            target = target.copy()
            target["target_value"] = pd.to_numeric(
                target["target_value"], errors="coerce"
            ).fillna(0.0) * float(dd_mult)

        print()
        if len(target) == 0:
            print("  Pre-flight (net cash): no target book — skipping liquidity check.")
        else:
            changes_pf = engine.reconcile(target, current_positions, verbose=False)
            to_close = changes_pf["to_close"]
            to_open = changes_pf["to_open"]

            open_set = {str(t).strip().upper() for t in to_open}
            tgt = target.copy()
            tgt["ticker_u"] = tgt["ticker"].astype(str).str.strip().str.upper()
            new_buy_notional = float(
                pd.to_numeric(
                    tgt.loc[tgt["ticker_u"].isin(open_set), "target_value"],
                    errors="coerce",
                )
                .fillna(0.0)
                .sum()
            )

            cash_from_sales = 0.0
            if current_positions is not None and not current_positions.empty and to_close:
                pos = current_positions.copy()
                pos["ticker"] = pos["ticker"].astype(str).str.strip().str.upper()
                close_set = {str(t).strip().upper() for t in to_close}
                sel = pos["ticker"].isin(close_set)
                cash_from_sales = float(
                    pd.to_numeric(pos.loc[sel, "market_value"], errors="coerce")
                    .fillna(0.0)
                    .sum()
                )

            net_cash_required = new_buy_notional - cash_from_sales
            cash = float(account["cash"])
            buying_power = float(account.get("buying_power", cash))
            available = buying_power + cash

            print("  Pre-flight (net cash flow)")
            print(f"    New-buy notional (to_open, sum target_value): ${new_buy_notional:,.2f}")
            print(f"    Est. proceeds from closes (to_close MV):     ${cash_from_sales:,.2f}")
            print(f"    Net cash required (opens − sales):           ${net_cash_required:,.2f}")
            print(f"    Cash:                                        ${cash:,.2f}")
            print(f"    Buying power:                                ${buying_power:,.2f}")
            print(f"    Available (buying_power + cash):             ${available:,.2f}")

            if net_cash_required > 0 and net_cash_required > available:
                entry = {
                    "timestamp": datetime.now().isoformat(),
                    "dry_run": dry_run,
                    "skipped": True,
                    "reason": "preflight_net_cash_exceeds_available",
                    "detail": (
                        f"net_cash_required={net_cash_required:.2f} > available={available:.2f} "
                        f"(new_buy_notional={new_buy_notional:.2f}, cash_from_sales={cash_from_sales:.2f})"
                    ),
                    "as_of": as_of,
                    "new_buy_notional": new_buy_notional,
                    "cash_from_sales": cash_from_sales,
                    "net_cash_required": net_cash_required,
                    "cash": cash,
                    "buying_power": buying_power,
                    "available_liquidity": available,
                    "fallback_source": fallback_label,
                    "orders_placed": [],
                    "orders_skipped": [],
                }
                print()
                print(
                    f"  ABORT: Net cash required {net_cash_required:,.2f} > available funds {available:,.2f} "
                    "— not executing."
                )
                _append_execution_skip_log(entry)
                save_signal_history(signals, as_of, entry)
                print()
                print(f"{'=' * 55}")
                print("  Stopped before orders. execution_log: output/live/execution_log.jsonl")
                print(f"{'=' * 55}")
                return

            print(f"  Net cash required: {net_cash_required:,.2f} – proceeding.")

        print()
        print(f"  Step 2: Execution ({'DRY RUN' if dry_run else 'LIVE'})...")
        result = engine.execute(signals, dry_run=dry_run)

        save_signal_history(signals, as_of, result)

        print()
        account = broker.get_account()
        positions = broker.get_positions()
        print("  Account summary:")
        print(f"  Equity:        ${account['equity']:>12,.2f}")
        print(f"  Cash:          ${account['cash']:>12,.2f}")
        print(f"  Buying power:  ${account.get('buying_power', account['cash']):>12,.2f}")
        print(f"  Positions: {len(positions)}")
        if not positions.empty:
            print()
            print(f"  {'Ticker':<8} {'Value':>10} {'P&L':>10} {'P&L %':>8}")
            print(f"  {'-' * 40}")
            for _, pos in positions.iterrows():
                print(
                    f"  {pos['ticker']:<8} "
                    f"${pos['market_value']:>9,.0f} "
                    f"${pos['unrealized_pnl']:>9,.0f} "
                    f"{float(pos['unrealized_pnl_pct']) * 100:>7.1f}%"
                )

        print()
        print(f"{'=' * 55}")
        print("  Done. execution_log: output/live/execution_log.jsonl")
        print(f"{'=' * 55}")

    except (
        requests.exceptions.Timeout,
        requests.exceptions.ConnectionError,
    ) as exc:
        stub_net: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            "skipped": True,
            "reason": "alpaca_network_error",
            "detail": str(exc),
            "as_of": as_of,
            "fallback_source": fallback_label,
            "n_open": 0,
            "n_close": 0,
            "n_hold": 0,
            "orders_placed": [],
            "orders_skipped": [],
        }
        print()
        print("  [!] Alpaca API unreachable (timeout or connection error).")
        print(f"      {exc}")
        print("  Rankings were saved; skip execution and retry when the network is up.")
        log_path = _append_execution_skip_log(stub_net)
        save_signal_history(signals, as_of, stub_net)
        print(f"  Logged: {log_path}")
        print()
        print(f"{'=' * 55}")
        print("  Done (signals only; no broker calls succeeded).")
        print(f"{'=' * 55}")


if __name__ == "__main__":
    main()
