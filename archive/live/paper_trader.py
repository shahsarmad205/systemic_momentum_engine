"""
Paper Trader
============

Lightweight paper-trading runner that reuses the same signal generation,
weight-learning model, execution, and cost assumptions as the backtester.

Intended usage:
  - Run with a conservative capital allocation (e.g. $5k–$10k)
  - Use the same YAML config and learned weights as research backtests
  - Log all trades and daily equity to CSVs under output/live/
  - Optionally serve a simple web dashboard (static file server) to inspect results

This is not a live broker connector; it simulates trades locally using
historical OHLCV data fetched on demand, with the same slippage and
transaction cost model as the offline backtests.
"""

from __future__ import annotations

import argparse
import http.server
import os
import socketserver
from datetime import datetime
from pathlib import Path

from backtest.engine import BacktestEngine
from backtesting import load_config
from config import DEV_MODE, apply_dev_mode, setup_logging

ROOT = Path(__file__).resolve().parents[1]
LIVE_OUTPUT_DIR = ROOT / "output" / "live"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trend Signal Engine — Paper Trader")
    p.add_argument(
        "--config",
        default="backtest_config.yaml",
        help="YAML config for signals and backtest settings (default: backtest_config.yaml)",
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Optional explicit ticker list (space-separated)",
    )
    p.add_argument(
        "--weights",
        default=None,
        help="Optional learned_weights.json to enable signal_mode='learned'",
    )
    p.add_argument(
        "--capital",
        type=float,
        default=None,
        help="Initial capital for paper trading (overrides config.initial_capital)",
    )
    p.add_argument(
        "--log-dir",
        default=str(LIVE_OUTPUT_DIR),
        help="Directory to write paper-trade logs (default: output/live)",
    )
    p.add_argument(
        "--dashboard",
        action="store_true",
        help="Serve a simple static dashboard for the log directory on localhost",
    )
    p.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for the optional dashboard server (default: 8000)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging",
    )
    return p.parse_args()


def run_dashboard(log_dir: Path, port: int) -> None:
    """
    Serve the contents of `log_dir` via a simple HTTP server.
    This is a minimal "dashboard": open CSVs/PNGs in a browser.
    """
    os.chdir(str(log_dir))
    handler = http.server.SimpleHTTPRequestHandler
    with socketserver.TCPServer(("", port), handler) as httpd:
        print(f"\nServing paper-trading logs from {log_dir} at http://localhost:{port}/")
        print("Press Ctrl+C to stop.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping dashboard server.")


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose or DEV_MODE)

    cfg_path = str(args.config)
    config = load_config(cfg_path)

    # Override tickers and capital if requested
    if args.tickers:
        config.tickers = args.tickers
    if args.capital is not None:
        config.initial_capital = float(args.capital)

    # Enable learned weights if provided
    if args.weights:
        config.signal_mode = "learned"
        config.learned_weights_path = args.weights

    # Apply dev-mode shortening if enabled (optional)
    apply_dev_mode(config)

    # Timestamp for this paper-trading run
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir).expanduser().resolve() / f"session_{ts}"
    os.makedirs(log_dir, exist_ok=True)

    print("\nPaper Trader session starting:")
    print(f"  Config     : {cfg_path}")
    print(f"  Tickers    : {len(config.tickers or [])}")
    print(f"  Capital    : ${config.initial_capital:,.2f}")
    print(f"  Log dir    : {log_dir}")
    if args.weights:
        print(f"  Weights    : {args.weights} (signal_mode='learned')")
    print()

    # Run a single backtest-style simulation with the given config & capital.
    engine = BacktestEngine(config=config, config_path=cfg_path)
    result = engine.run_backtest(config.tickers or None)

    # Persist trades and equity for this paper session.
    if not result.trades.empty:
        trades_path = log_dir / "paper_trades.csv"
        result.trades.to_csv(trades_path, index=False)
        print(f"  Paper trades CSV → {trades_path}")
    else:
        print("  [WARN] No trades generated in this paper-trading run.")

    if not result.daily_equity.empty:
        equity_path = log_dir / "paper_equity.csv"
        result.daily_equity.to_csv(equity_path, index=False)
        print(f"  Paper equity CSV → {equity_path}")

    # Also write a short metrics summary for this session
    if result.metrics:
        import json

        metrics_path = log_dir / "paper_metrics.json"
        with metrics_path.open("w") as fh:
            json.dump(
                {k: v for k, v in result.metrics.items()},
                fh,
                indent=2,
                default=str,
            )
        print(f"  Paper metrics JSON → {metrics_path}")

    print("\nPaper Trader run complete.")

    # Optional simple dashboard
    if args.dashboard:
        run_dashboard(log_dir, args.port)


if __name__ == "__main__":
    main()

