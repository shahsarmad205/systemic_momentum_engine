"""
Walk-Forward Validation Runner
================================
Runs the walk-forward validation framework on the configured backtest universe.

Usage:
    python run_walk_forward.py
    python run_walk_forward.py --config backtest_config.yaml
    python run_walk_forward.py --tickers AAPL MSFT NVDA
    python run_walk_forward.py --train-weights      # retrain weights per window
    python run_walk_forward.py --verbose

Output:
    - output/backtests/walk_forward_validation_report.csv
    - Console summary of per-window OOS Sharpe, drawdown, IC
"""

from __future__ import annotations

import argparse
from pathlib import Path

from backtesting import load_config, run_walk_forward
from config import DEV_MODE, apply_dev_mode, get_effective_tickers, setup_logging

ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output" / "backtests"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trend Signal Engine — Walk-Forward Validation")
    p.add_argument(
        "--config",
        default="backtest_config.yaml",
        help="Backtest YAML config file (default: backtest_config.yaml)",
    )
    p.add_argument(
        "--tickers",
        nargs="+",
        default=None,
        help="Optional explicit ticker list (space-separated)",
    )
    p.add_argument(
        "--train-weights",
        action="store_true",
        default=False,
        help="Retrain weight model on each training window before OOS evaluation",
    )
    p.add_argument(
        "--report-path",
        default=str(OUTPUT_DIR / "walk_forward_validation_report.csv"),
        help="Output CSV path for the walk-forward summary report",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging",
    )
    return p.parse_args()


def _resolve_tickers(cfg, cli_tickers: list[str] | None) -> list[str]:
    from utils.universe import load_universe

    tickers = cli_tickers or cfg.tickers or []
    if not tickers:
        try:
            from utils.default_universe import DEFAULT_TICKERS
            tickers = list(DEFAULT_TICKERS)
        except Exception:
            tickers = []

    universe_cfg = getattr(cfg, "universe", {}) or {}
    if universe_cfg.get("mode") == "sp500":
        try:
            tickers = load_universe(universe_cfg)
        except Exception:
            pass

    return get_effective_tickers(tickers, tickers)


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose or DEV_MODE)

    cfg = load_config(args.config)
    apply_dev_mode(cfg)

    tickers = _resolve_tickers(cfg, args.tickers)
    if not tickers:
        print("[ERROR] No tickers resolved; aborting.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=========================================================")
    print("  Walk-Forward Validation")
    print("=========================================================")
    print(f"  Config      : {args.config}")
    print(f"  Tickers     : {len(tickers)}")
    print(f"  Period      : {cfg.start_date} → {cfg.end_date}")
    print(f"  Windows     : {cfg.walk_forward_windows}")
    print(f"  Train ratio : {cfg.walk_forward_train_ratio}")
    print(f"  Retrain wts : {args.train_weights}")
    print(f"  Report      : {args.report_path}")
    print("=========================================================\n")

    results, summary = run_walk_forward(
        cfg,
        tickers,
        train_weights=args.train_weights,
        report_path=args.report_path,
    )

    if summary.empty:
        print("[WARN] Walk-forward summary is empty — no OOS windows completed.")
        return

    print("\n--- Walk-Forward Summary ---")
    display_cols = [c for c in ("window", "train_start", "train_end", "test_start", "test_end",
                                "oos_sharpe", "oos_cagr", "oos_max_dd", "oos_ic") if c in summary.columns]
    print(summary[display_cols].to_string(index=False))

    mean_sharpe = float(summary["oos_sharpe"].mean()) if "oos_sharpe" in summary.columns else float("nan")
    print(f"\n  Mean OOS Sharpe : {mean_sharpe:.3f}")
    print(f"  Report saved    : {args.report_path}\n")


if __name__ == "__main__":
    main()
