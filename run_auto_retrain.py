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
import shutil
from datetime import datetime
from pathlib import Path

from backtesting import load_config, run_walk_forward
from config import DEV_MODE, apply_dev_mode, setup_logging

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

    # B3: Respect the configured signal_mode instead of always forcing "learned".
    # When signal_mode is "ml" or "ensemble", auto-retrain evaluates the existing
    # ML model ensemble (best_long_model.pkl + best_short_model.pkl) via walk-forward
    # and promotes it only if its OOS Sharpe beats the current production model.
    # The time-decayed ensemble blending (B3) only applies to "learned" mode.
    active_mode = str(getattr(config, "signal_mode", "learned") or "learned").lower()
    if active_mode in ("ml", "ensemble"):
        config.walk_forward_train_weights = False  # no per-window weight training in ML mode
        print(f"  signal_mode={active_mode!r}: running walk-forward evaluation of ML models (no weight retraining).")
    else:
        # Ensure walk-forward uses dynamic weight training on each train window (legacy learned mode)
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

    # --- 3) Promote production model ---
    print("\nPhase 3: Promoting new production model…")

    if active_mode in ("ml", "ensemble"):
        # B3 (ML mode): the ML models are already on disk (best_long_model.pkl etc.).
        # Walk-forward confirmed they're still competitive. No file copy needed —
        # the backtester reads the model paths directly from config.
        print(
            f"  signal_mode={active_mode!r}: ML models already at configured paths "
            f"(ml_long_model_path={getattr(config, 'ml_long_model_path', 'N/A')}). "
            "No file promotion required."
        )
        print("\nAuto-retrain (ML mode) finished successfully.")
        return

    # Legacy "learned" mode: time-decayed ensemble of walk-forward LearnedWeights
    non_empty = cand_summary[cand_summary["weights_path"] != ""].copy()
    if non_empty.empty:
        print("  [ERROR] No candidate weights_path entries found; cannot promote.")
        return

    # Load all per-window LearnedWeights and blend with exponential time-decay.
    # Windows are ordered oldest→newest; most-recent window gets weight λ^0=1.0.
    _DECAY = 0.7  # λ: each older window discounted by 30%
    from agents.weight_learning_agent.weight_model import LearnedWeights
    import dataclasses

    window_weights: list[LearnedWeights] = []
    decay_factors: list[float] = []
    n_windows = len(non_empty)
    for i, (_, row) in enumerate(non_empty.iterrows()):
        p = Path(row["weights_path"]).resolve()
        if not p.is_file():
            print(f"  [WARN] Window {i} weights file not found, skipping: {p}")
            continue
        try:
            lw = LearnedWeights.load(str(p))
        except Exception as exc:
            print(f"  [WARN] Failed to load window {i} weights ({p}): {exc}")
            continue
        # Exponent: most-recent window (last) → power 0; oldest → power (n-1)
        power = (n_windows - 1) - i
        window_weights.append(lw)
        decay_factors.append(_DECAY ** power)
        print(f"  Window {i}: decay={_DECAY**power:.3f}  path={p.name}")

    if not window_weights:
        print("  [ERROR] No valid window weights loaded; cannot promote.")
        return

    # Weighted average of all numeric w_* fields
    total_decay = sum(decay_factors)
    base = dataclasses.asdict(window_weights[0])
    blended: dict = {}
    _NUMERIC_SKIP = {"n_samples", "score_direction"}
    for key, val in base.items():
        if isinstance(val, (int, float)) and key not in _NUMERIC_SKIP:
            blended[key] = sum(
                dataclasses.asdict(lw)[key] * d
                for lw, d in zip(window_weights, decay_factors)
            ) / total_decay
        else:
            # Keep metadata from the most-recent window
            blended[key] = dataclasses.asdict(window_weights[-1])[key]

    ensemble_lw = LearnedWeights.from_dict(blended)
    print(f"  Ensemble blended from {len(window_weights)} windows (λ={_DECAY})")

    # Write ensemble to a temp file so we have a real path to copy from
    import tempfile, os
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=".json")
    os.close(tmp_fd)
    ensemble_lw.save(tmp_path)
    new_weights_path = Path(tmp_path).resolve()

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

    # Clean up temp file created for ensemble blend (if applicable)
    if str(new_weights_path).endswith(".json") and "tmp" in str(new_weights_path).lower():
        try:
            new_weights_path.unlink(missing_ok=True)
        except Exception:
            pass

    print("\nAuto-retrain finished successfully.")


if __name__ == "__main__":
    main()

