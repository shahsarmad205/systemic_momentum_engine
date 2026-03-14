"""
Auto-Retrain Runner
====================

Periodically retrains the weight-learning model on expanding data,
compares the new model's walk-forward performance vs. the current
production model, and updates production weights when the new model
outperforms by Sharpe on validation.

Intended usage:
  - Schedule via cron or a task scheduler, e.g. weekly or monthly:

      # Every Monday at 3am
      0 3 * * MON  /usr/bin/python3 /path/to/run_auto_retrain.py --config backtest_config.yaml >> auto_retrain.log 2>&1

  - Keep `config.learned_weights_path` pointing to the current
    "production" weights JSON.
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd

from backtesting import load_config, run_walk_forward
from config import apply_dev_mode, setup_logging, DEV_MODE


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "output"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trend Signal Engine — Auto-Retrain Runner")
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
        "--min-sharpe-improvement",
        type=float,
        default=0.05,
        help="Minimum Sharpe improvement on OOS validation required to accept new model (default: 0.05).",
    )
    p.add_argument(
        "--archive-dir",
        default=str(OUTPUT_DIR / "weights_archive"),
        help="Directory to archive old production weights (default: output/weights_archive)",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose (DEBUG-level) logging",
    )
    return p.parse_args()


def _resolve_tickers(cfg, cli_tickers: list[str] | None) -> list[str]:
    from config import get_effective_tickers

    tickers = cli_tickers or cfg.tickers or []
    # Fall back to main.TICKERS if config.tickers is empty
    if not tickers:
        try:
            from main import TICKERS  # type: ignore

            tickers = list(TICKERS)
        except Exception:
            tickers = []
    return get_effective_tickers(tickers, tickers)


def main() -> None:
    args = parse_args()
    setup_logging(verbose=args.verbose or DEV_MODE)

    cfg_path = str(args.config)
    config = load_config(cfg_path)
    apply_dev_mode(config)

    tickers = _resolve_tickers(config, args.tickers)
    if not tickers:
        print("[ERROR] No tickers available for auto-retrain; aborting.")
        return

    print("\n=========================================================")
    print("  Auto-Retrain — Weight-Learning Model")
    print("=========================================================")
    print(f"  Config file     : {cfg_path}")
    print(f"  Tickers         : {len(tickers)}")
    print(f"  Period          : {config.start_date} → {config.end_date}")
    print(f"  Current weights : {getattr(config, 'learned_weights_path', '') or 'N/A'}")
    print("=========================================================\n")

    # Ensure walk-forward uses dynamic weight training on each train window
    config.signal_mode = "learned"
    config.walk_forward_train_weights = True

    # --- 1) Baseline: current production model walk-forward (if any) ---
    prod_weights_path = getattr(config, "learned_weights_path", "") or ""
    baseline_sharpe = None

    if prod_weights_path:
        print("Phase 1: Baseline walk-forward with current production weights…")
        # Use existing weights without retraining inside walk-forward
        base_cfg = load_config(cfg_path)
        apply_dev_mode(base_cfg)
        base_cfg.learned_weights_path = prod_weights_path
        base_cfg.signal_mode = "learned"
        base_cfg.walk_forward_train_weights = False

        _, baseline_summary = run_walk_forward(
            base_cfg,
            tickers,
            train_weights=False,
            report_path="output/backtests/walk_forward_baseline.csv",
        )
        if not baseline_summary.empty:
            baseline_sharpe = float(baseline_summary["oos_sharpe"].mean())
            print(f"  Baseline OOS Sharpe (mean over windows): {baseline_sharpe:.3f}")
        else:
            print("  [WARN] Baseline walk-forward summary empty; treating baseline Sharpe as 0.")
            baseline_sharpe = 0.0
    else:
        print("Phase 1: No existing production weights; baseline Sharpe set to 0.")
        baseline_sharpe = 0.0

    # --- 2) Candidate: walk-forward with retrained weights on each train window ---
    print("\nPhase 2: Candidate walk-forward with auto-trained weights…")
    # Use config with train_weights=True; run_walk_forward will train weights per window
    _, cand_summary = run_walk_forward(
        config,
        tickers,
        train_weights=True,
        report_path="output/backtests/walk_forward_candidate.csv",
    )

    if cand_summary.empty:
        print("  [ERROR] Candidate walk-forward summary empty; not updating production weights.")
        return

    cand_sharpe = float(cand_summary["oos_sharpe"].mean())
    print(f"  Candidate OOS Sharpe (mean over windows): {cand_sharpe:.3f}")

    improvement = cand_sharpe - (baseline_sharpe or 0.0)
    print(f"\n  Sharpe improvement vs baseline: {improvement:+.3f}")

    if improvement < args.min_sharpe_improvement:
        print(
            f"  [INFO] Improvement {improvement:+.3f} < threshold {args.min_sharpe_improvement:.3f}; "
            "keeping existing production weights."
        )
        return

    # --- 3) Promote best candidate weights from last walk-forward window ---
    print("\nPhase 3: Promoting new production weights…")
    # Take the last non-empty weights_path from candidate summary
    non_empty = cand_summary[cand_summary["weights_path"] != ""]
    if non_empty.empty:
        print("  [ERROR] No candidate weights_path entries found; cannot promote.")
        return

    best_row = non_empty.iloc[-1]
    new_weights_path = Path(best_row["weights_path"]).resolve()

    if not new_weights_path.is_file():
        print(f"  [ERROR] Candidate weights file not found: {new_weights_path}")
        return

    archive_dir = Path(args.archive_dir).expanduser().resolve()
    archive_dir.mkdir(parents=True, exist_ok=True)

    # Archive old weights if present
    if prod_weights_path:
        old_path = Path(prod_weights_path).resolve()
        if old_path.is_file():
            ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            archived_name = f"{old_path.stem}_archived_{ts}{old_path.suffix}"
            archived_path = archive_dir / archived_name
            shutil.copy2(str(old_path), str(archived_path))
            print(f"  Archived old weights → {archived_path}")

    # Promote candidate weights into the production path used by config
    # If config has no learned_weights_path yet, set a sensible default.
    if not prod_weights_path:
        prod_weights_path = str(OUTPUT_DIR / "learned_weights.json")

    prod_dest = Path(prod_weights_path).resolve()
    prod_dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(new_weights_path), str(prod_dest))
    print(f"  Updated production weights → {prod_dest}")

    print("\nAuto-retrain finished successfully.")


if __name__ == "__main__":
    main()

