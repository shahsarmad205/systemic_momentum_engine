#!/usr/bin/env python3
"""
Monthly (or on-demand) full-history retrain of the LearnedWeights ridge baseline.

- Loads ``backtest_config.yaml`` the same way as ``run_model_selection.py`` (tickers,
  ``model_selection.lookahead_horizon_days``, optional ``feature_selection.feature_subset``).
- Builds the feature matrix via ``build_feature_matrix`` up to **yesterday** (calendar).
- Trains ``WeightLearner`` with ``model_type="ridge"`` and ``target_type="regression"``
  (same pipeline that produces ``output/learned_weights.json`` + scaler).
- Overwrites ``output/learned_weights.json`` and ``output/learned_weights_scaler.json``.

Run from the ``trend_signal_engine`` directory (same as other runners):

  python run_retrain_baseline.py
  python run_retrain_baseline.py --quick-backtest
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

# Ensure imports resolve when run from repo root or elsewhere.
_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _read_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _setup_logging(log_file: Path | None, verbose: bool) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def _quick_backtest_compare(
    *,
    config_path: Path,
    train_end: str,
    previous_weights: Path | None,
    new_weights: Path,
    calendar_span_days: int,
    warn_ratio: float,
) -> None:
    from backtesting.backtester import Backtester
    from backtesting.config import load_config

    log = logging.getLogger(__name__)
    cfg_base = load_config(str(config_path))
    end_ts = pd.Timestamp(train_end)
    start_guess = end_ts - pd.Timedelta(days=int(calendar_span_days))
    cfg_start = pd.Timestamp(cfg_base.start_date)
    cfg_base.start_date = max(cfg_start, start_guess).strftime("%Y-%m-%d")
    cfg_base.end_date = train_end
    cfg_base.signal_mode = "learned"

    def _one_run(label: str, weights_path: Path) -> dict[str, Any] | None:
        if not weights_path.is_file():
            return None
        cfg = deepcopy(cfg_base)
        cfg.learned_weights_path = str(weights_path)
        log.info("Quick backtest [%s] weights=%s window %s → %s", label, weights_path, cfg.start_date, cfg.end_date)
        bt = Backtester(cfg)
        res = bt.run()
        return dict(res.metrics or {})

    old_m = _one_run("previous", previous_weights) if previous_weights else None
    new_m = _one_run("new", new_weights)
    if not new_m:
        log.warning("Quick backtest: no metrics for new weights.")
        return

    def _sharpe(m: dict[str, Any] | None) -> float:
        if not m:
            return float("nan")
        for k in ("net_sharpe_ratio", "sharpe_ratio"):
            v = m.get(k)
            if v is not None:
                try:
                    return float(v)
                except (TypeError, ValueError):
                    pass
        return float("nan")

    s_new = _sharpe(new_m)
    log.info("Quick backtest NEW net_sharpe≈%.4f  total_return=%s", s_new, new_m.get("total_return"))
    if old_m:
        s_old = _sharpe(old_m)
        log.info("Quick backtest PREVIOUS net_sharpe≈%.4f  total_return=%s", s_old, old_m.get("total_return"))
        if s_old == s_old and s_new == s_new and s_old > 1e-6 and s_new < float(warn_ratio) * s_old:
            log.warning(
                "NEW Sharpe (%.4f) is below %.0f%% of PREVIOUS (%.4f) on this window — review before relying on refresh.",
                s_new,
                float(warn_ratio) * 100.0,
                s_old,
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain LearnedWeights baseline on full history through yesterday.")
    parser.add_argument("--config", type=str, default="backtest_config.yaml")
    parser.add_argument(
        "--horizon",
        type=int,
        default=None,
        help="Forward return horizon in trading days (default: model_selection.lookahead_horizon_days or 5)",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default="",
        help="Train through this date inclusive (YYYY-MM-DD). Default: yesterday (local).",
    )
    parser.add_argument(
        "--use-config-feature-subset",
        action="store_true",
        help="If set, pass feature_selection.feature_subset like run_model_selection; default trains on full feature columns (matches typical learned_weights.json).",
    )
    parser.add_argument(
        "--limit-tickers",
        type=int,
        default=0,
        help="Optional cap on universe size for debugging.",
    )
    parser.add_argument(
        "--out-weights",
        type=str,
        default="output/learned_weights.json",
    )
    parser.add_argument(
        "--out-scaler",
        type=str,
        default="output/learned_weights_scaler.json",
    )
    parser.add_argument(
        "--backup-suffix",
        type=str,
        default=".before_retrain",
        help="If previous weights exist, copy to learned_weights.json{suffix} before overwrite.",
    )
    parser.add_argument(
        "--quick-backtest",
        action="store_true",
        help="After save, run Backtester on a short recent window with learned mode; compare to backup if present.",
    )
    parser.add_argument(
        "--quick-backtest-days",
        type=int,
        default=380,
        help="Approximate calendar span for quick backtest window ending at train end.",
    )
    parser.add_argument(
        "--quick-backtest-warn-ratio",
        type=float,
        default=0.9,
        help="If NEW net_sharpe < ratio * OLD net_sharpe, emit a warning.",
    )
    parser.add_argument("--log-file", type=str, default="output/logs/retrain_baseline.log")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    os.chdir(_ROOT)
    cfg_path = Path(args.config)
    if not cfg_path.is_file():
        raise SystemExit(f"Config not found: {cfg_path.resolve()} (cwd={Path.cwd()})")

    log_path = Path(args.log_file) if args.log_file else None
    _setup_logging(log_path, args.verbose)
    log = logging.getLogger(__name__)

    train_end = str(args.as_of).strip()
    if not train_end:
        train_end = (datetime.now().date() - timedelta(days=1)).isoformat()
    train_end_ts = pd.Timestamp(train_end)

    cfg = _read_config(cfg_path)
    tickers = list(cfg.get("tickers", []) or [])
    if int(args.limit_tickers or 0) > 0:
        tickers = tickers[: int(args.limit_tickers)]
    if not tickers:
        raise SystemExit("No tickers in config.")

    bt = cfg.get("backtest", {}) or {}
    ms_cfg = cfg.get("model_selection", {}) or {}
    feature_sel = cfg.get("feature_selection", {}) or {}
    feature_subset_cfg = feature_sel.get("feature_subset", []) or []
    feature_subset_cfg = [str(c).strip() for c in feature_subset_cfg if str(c).strip()]

    horizon = int(args.horizon) if args.horizon is not None else int(ms_cfg.get("lookahead_horizon_days", 5) or 5)

    start_date = str(bt.get("start_date", "2018-01-01"))
    end_date = train_end  # full history through train end

    if pd.Timestamp(start_date) > train_end_ts:
        raise SystemExit(f"backtest.start_date {start_date} is after train end {train_end}")

    if args.use_config_feature_subset and feature_subset_cfg:
        feature_subset = feature_subset_cfg
        log.info("Using config feature_selection.feature_subset (%d columns).", len(feature_subset))
    else:
        feature_subset = None
        log.info("Using full feature matrix (feature_subset=None).")

    log.info("Retrain baseline: %d tickers | horizon=%d | %s → %s", len(tickers), horizon, start_date, end_date)

    from agents.weight_learning_agent.feature_builder import build_feature_matrix
    from agents.weight_learning_agent.weight_model import WeightLearner

    df = build_feature_matrix(
        tickers,
        start_date=start_date,
        end_date=end_date,
        holding_period=int(horizon),
        feature_subset=feature_subset,
    )
    if df is None or df.empty:
        raise SystemExit("build_feature_matrix returned empty; aborting.")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])
    df = df[df["date"] <= train_end_ts]
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)

    if "forward_return" not in df.columns:
        raise SystemExit("Feature matrix missing forward_return.")
    df["forward_return"] = pd.to_numeric(df["forward_return"], errors="coerce")
    df = df.dropna(subset=["forward_return"])
    if df.empty:
        raise SystemExit("No rows after dropping NaN forward_return.")

    out_weights = Path(args.out_weights)
    out_scaler = Path(args.out_scaler)
    backup_path = out_weights.with_name(out_weights.name + str(args.backup_suffix))

    prev_backup: Path | None = None
    if out_weights.is_file():
        shutil.copy2(out_weights, backup_path)
        prev_backup = backup_path
        log.info("Backed up existing weights to %s", backup_path)

    learner = WeightLearner(
        model_type="ridge",
        alpha=0.01,
        time_decay_lambda=0.001,
        target_type="regression",
        return_target_type="raw",
    )
    log.info("Fitting WeightLearner (ridge / regression) on %d rows…", len(df))
    lw = learner.fit(df, feature_cols=None)
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    lw.save(str(out_weights))
    learner.save_scaler(str(out_scaler))
    log.info("Wrote %s and %s", out_weights, out_scaler)
    log.info(
        "Train stats: samples=%s IC=%s R2=%s start=%s end=%s",
        getattr(lw, "n_samples", None),
        getattr(lw, "ic", None),
        getattr(lw, "r2", None),
        getattr(lw, "train_start", None),
        getattr(lw, "train_end", None),
    )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "train_end": train_end,
        "start_date": start_date,
        "horizon": horizon,
        "n_rows": int(len(df)),
        "n_tickers": len(tickers),
        "feature_subset_mode": "config" if args.use_config_feature_subset and feature_subset_cfg else "full",
        "weights_path": str(out_weights),
        "scaler_path": str(out_scaler),
        "backup_path": str(prev_backup) if prev_backup else None,
    }
    meta_path = out_weights.parent / "retrain_baseline_last.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    log.info("Wrote metadata %s", meta_path)

    if args.quick_backtest:
        _quick_backtest_compare(
            config_path=cfg_path,
            train_end=train_end,
            previous_weights=prev_backup,
            new_weights=out_weights,
            calendar_span_days=int(args.quick_backtest_days),
            warn_ratio=float(args.quick_backtest_warn_ratio),
        )

    log.info("Done.")
    raise SystemExit(0)


if __name__ == "__main__":
    main()
