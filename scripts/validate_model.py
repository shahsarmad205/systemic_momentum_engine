#!/usr/bin/env python3
"""
Quick model artifact validation using recent walk-forward windows.

This script validates the selected model artifact against recent OOS windows
using the same leakage-safe split logic as run_model_selection.py.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.weight_learning_agent.feature_builder import build_feature_matrix
from run_model_selection import (  # noqa: E402
    _concat_window_daily_returns,
    _max_drawdown_from_daily_returns,
    _read_config,
    _sharpe_from_series,
    _strategy_daily_returns,
    _win_rate_from_daily_returns,
    _walk_forward_windows,
    _walk_forward_windows_by_count,
)


def _score_model(model: object, model_kind: str, x_test: np.ndarray) -> np.ndarray:
    if model_kind == "regressor":
        return model.predict(x_test).astype(float)
    if model_kind == "short_classifier":
        if hasattr(model, "predict_proba"):
            p_down = model.predict_proba(x_test)[:, 1].astype(float)
            return -(p_down - 0.5)
        return -model.predict(x_test).astype(float) + 0.5
    if hasattr(model, "predict_proba"):
        p_up = model.predict_proba(x_test)[:, 1].astype(float)
        return p_up - 0.5
    if hasattr(model, "decision_function"):
        return model.decision_function(x_test).astype(float)
    pred = model.predict(x_test).astype(int)
    return pred.astype(float) - 0.5


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate best model artifact on recent walk-forward windows")
    parser.add_argument("--config", type=str, default="backtest_config.yaml")
    parser.add_argument("--model-path", type=str, default="output/models/best_model.pkl")
    parser.add_argument("--windows", type=int, default=2, help="How many most-recent windows to validate")
    parser.add_argument("--min-test-days", type=int, default=20)
    parser.add_argument("--embargo-days", type=int, default=0, help="0 = auto (2*horizon, min 5)")
    parser.add_argument("--min-sharpe-mean", type=float, default=None)
    parser.add_argument("--max-sharpe-std", type=float, default=None)
    parser.add_argument("--min-window-sharpe", type=float, default=None)
    parser.add_argument("--min-chained-sharpe", type=float, default=None)
    parser.add_argument("--min-validation-windows", type=int, default=None)
    parser.add_argument("--max-drawdown-abs", type=float, default=None)
    parser.add_argument("--min-win-rate", type=float, default=None)
    args = parser.parse_args()

    cfg = _read_config(args.config)
    bt = cfg.get("backtest", {}) or {}
    research = cfg.get("research", {}) or {}
    ms_cfg = cfg.get("model_selection", {}) or {}
    stab_cfg = (ms_cfg.get("stability_metrics", {}) or {})
    gate_cfg = (ms_cfg.get("validator_gates", {}) or {})
    gate_enabled = bool(gate_cfg.get("enabled", False))
    max_positions = int(ms_cfg.get("max_positions", 10) or 10)
    min_positions = int(ms_cfg.get("min_positions", 3) or 3)
    max_positions = int(max(1, max_positions))
    min_positions = int(max(1, min_positions))
    if min_positions > max_positions:
        min_positions = max_positions

    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"ERROR: model not found at {model_path}")
        return 1

    with open(model_path, "rb") as fh:
        artifact = pickle.load(fh)

    estimator = artifact.get("estimator")
    model_name = str(artifact.get("model_name", ""))
    model_kind = str(artifact.get("model_type", "classifier"))
    feature_columns = [str(c) for c in (artifact.get("feature_columns", []) or [])]
    horizon = int(artifact.get("horizon_days", ms_cfg.get("lookahead_horizon_days", 1)) or 1)

    if estimator is None:
        print("ERROR: artifact estimator is empty; cannot validate this model type")
        return 1
    if not feature_columns:
        print("ERROR: artifact feature_columns missing")
        return 1

    if model_name == "ConsensusRidge":
        consensus_path = model_path.parent / "consensus_features.json"
        if not consensus_path.exists():
            print(f"ERROR: consensus feature file missing at {consensus_path}")
            return 1
        try:
            consensus_payload = json.loads(consensus_path.read_text(encoding="utf-8"))
        except Exception as exc:
            print(f"ERROR: failed to parse consensus feature file: {exc}")
            return 1
        consensus_features = [str(x) for x in (consensus_payload.get("selected_features", []) or [])]
        if not consensus_features:
            print("ERROR: consensus feature file has empty selected_features")
            return 1
        if set(consensus_features) != set(feature_columns):
            print("ERROR: artifact feature_columns do not match consensus_features.json")
            return 1

    tickers = list(cfg.get("tickers", []) or [])
    if not tickers:
        print("ERROR: no tickers configured")
        return 1

    start_date = str(bt.get("start_date", "2018-01-01"))
    end_date = str(bt.get("end_date", "2024-01-01"))

    df = build_feature_matrix(
        tickers,
        start_date=start_date,
        end_date=end_date,
        holding_period=horizon,
        feature_subset=None,
    )
    if df is None or df.empty:
        print("ERROR: feature matrix is empty")
        return 1

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values(["ticker", "date"]).reset_index(drop=True)
    df["forward_return"] = pd.to_numeric(df["forward_return"], errors="coerce")
    df = df.dropna(subset=["forward_return"])

    missing_cols = [c for c in feature_columns if c not in df.columns]
    if missing_cols:
        print(f"ERROR: missing feature columns in matrix (sample): {missing_cols[:10]}")
        return 1

    windows = _walk_forward_windows(
        start_date,
        end_date,
        float(research.get("train_years", 2.0) or 2.0),
        float(research.get("test_years", 0.5) or 0.5),
        float(research.get("step_years", 0.5) or 0.5),
    )
    if len(windows) <= 1:
        windows = _walk_forward_windows_by_count(
            df["date"],
            n_windows=int(research.get("walk_forward_windows", 4) or 4),
            train_ratio=float(research.get("walk_forward_train_ratio", 0.7) or 0.7),
        )
    if len(windows) < 1:
        print("ERROR: no walk-forward windows available for validation")
        return 1

    take_n = int(max(1, min(args.windows, len(windows))))
    selected_windows = windows[-take_n:]
    embargo_days = int(args.embargo_days) if int(args.embargo_days) > 0 else int(max(5, 2 * horizon))

    daily_parts: list[pd.Series] = []
    window_sharpes: list[float] = []

    from sklearn.base import clone

    for idx, (tr_s, tr_e, te_s, te_e) in enumerate(selected_windows, 1):
        purge_cutoff = te_s - pd.Timedelta(days=embargo_days)
        tr = df[(df["date"] >= tr_s) & (df["date"] < min(tr_e, purge_cutoff))].copy()
        te = df[(df["date"] >= te_s) & (df["date"] < te_e)].copy()

        if tr.empty or te.empty:
            print(f"WARN: skip window {idx}/{take_n} (empty train/test)")
            continue
        if int(te["date"].nunique()) < int(args.min_test_days):
            print(f"WARN: skip window {idx}/{take_n} (test days < {args.min_test_days})")
            continue

        x_tr = tr[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        x_te = te[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        if model_kind == "regressor":
            y_tr = tr["forward_return"].to_numpy(dtype=float)
        elif model_kind == "short_classifier":
            y_tr = (tr["forward_return"].to_numpy(dtype=float) < 0).astype(int)
        else:
            y_tr = (tr["forward_return"].to_numpy(dtype=float) > 0).astype(int)

        fitted = clone(estimator)
        fitted.fit(x_tr, y_tr)
        score = _score_model(fitted, model_kind, x_te)

        te_scored = te.assign(score=score)
        dr = _strategy_daily_returns(
            te_scored,
            max_positions=max_positions,
            min_positions=min_positions,
        )
        if dr.empty:
            print(f"WARN: skip window {idx}/{take_n} (no invested days)")
            continue

        daily_parts.append(dr)
        sh = _sharpe_from_series(dr.to_numpy(dtype=float))
        window_sharpes.append(float(sh) if np.isfinite(sh) else float("nan"))
        print(
            f"window {idx}/{take_n}: test={te_s.date()}->{(te_e - pd.Timedelta(days=1)).date()} "
            f"days={len(dr)} sharpe={sh:.4f}"
        )

    if not window_sharpes:
        print("ERROR: no valid validation windows")
        return 1

    sharpe_arr = np.asarray(window_sharpes, dtype=float)
    sharpe_mean = float(np.nanmean(sharpe_arr))
    sharpe_std = float(np.nanstd(sharpe_arr, ddof=1)) if len(sharpe_arr) > 1 else 0.0
    sharpe_min = float(np.nanmin(sharpe_arr))
    chained_daily = _concat_window_daily_returns(daily_parts)
    chained_sharpe = _sharpe_from_series(chained_daily)
    max_drawdown = _max_drawdown_from_daily_returns(chained_daily)
    win_rate = _win_rate_from_daily_returns(chained_daily)

    min_sharpe_mean = (
        float(args.min_sharpe_mean)
        if args.min_sharpe_mean is not None
        else float(stab_cfg.get("min_sharpe_mean", 0.0) or 0.0)
    )
    max_sharpe_std = (
        float(args.max_sharpe_std)
        if args.max_sharpe_std is not None
        else float(stab_cfg.get("max_sharpe_std", 1.2) or 1.2)
    )
    min_window_sharpe = (
        float(args.min_window_sharpe)
        if args.min_window_sharpe is not None
        else float(stab_cfg.get("min_window_sharpe", -0.5) or -0.5)
    )
    min_chained_sharpe = (
        float(args.min_chained_sharpe)
        if args.min_chained_sharpe is not None
        else (float(gate_cfg.get("min_chained_sharpe", 0.0) or 0.0) if gate_enabled else None)
    )
    min_validation_windows = (
        int(args.min_validation_windows)
        if args.min_validation_windows is not None
        else (int(gate_cfg.get("min_validation_windows", 1) or 1) if gate_enabled else None)
    )
    max_drawdown_abs = (
        float(args.max_drawdown_abs)
        if args.max_drawdown_abs is not None
        else (float(gate_cfg.get("max_drawdown_abs", 1.0) or 1.0) if gate_enabled else None)
    )
    min_win_rate = (
        float(args.min_win_rate)
        if args.min_win_rate is not None
        else (float(gate_cfg.get("min_win_rate", 0.0) or 0.0) if gate_enabled else None)
    )

    print("\nValidation summary")
    print(f"  oos_sharpe_chained: {chained_sharpe:.4f}")
    print(f"  oos_sharpe_mean:    {sharpe_mean:.4f}")
    print(f"  oos_sharpe_std:     {sharpe_std:.4f}")
    print(f"  oos_sharpe_min:     {sharpe_min:.4f}")
    print(f"  max_drawdown:       {max_drawdown:.4f}")
    print(f"  win_rate:           {win_rate:.4f}")
    print(f"  n_windows_used:     {len(window_sharpes)}")

    failed: list[str] = []
    if not np.isfinite(sharpe_mean):
        failed.append("sharpe_mean_non_finite")
    if not np.isfinite(sharpe_std):
        failed.append("sharpe_std_non_finite")
    if not np.isfinite(sharpe_min):
        failed.append("window_min_non_finite")
    if np.isfinite(sharpe_mean) and sharpe_mean < min_sharpe_mean:
        failed.append(f"sharpe_mean<{min_sharpe_mean:.3f}")
    if np.isfinite(sharpe_std) and sharpe_std > max_sharpe_std:
        failed.append(f"sharpe_std>{max_sharpe_std:.3f}")
    if np.isfinite(sharpe_min) and sharpe_min < min_window_sharpe:
        failed.append(f"window_min<{min_window_sharpe:.3f}")
    if min_validation_windows is not None and len(window_sharpes) < int(min_validation_windows):
        failed.append(f"n_windows<{int(min_validation_windows)}")
    if min_chained_sharpe is not None:
        if not np.isfinite(chained_sharpe):
            failed.append("chained_sharpe_non_finite")
        elif chained_sharpe < min_chained_sharpe:
            failed.append(f"chained_sharpe<{min_chained_sharpe:.3f}")
    if max_drawdown_abs is not None:
        if not np.isfinite(max_drawdown):
            failed.append("max_drawdown_non_finite")
        elif abs(max_drawdown) > max_drawdown_abs:
            failed.append(f"abs(max_drawdown)>{max_drawdown_abs:.3f}")
    if min_win_rate is not None:
        if not np.isfinite(win_rate):
            failed.append("win_rate_non_finite")
        elif win_rate < min_win_rate:
            failed.append(f"win_rate<{min_win_rate:.3f}")

    if failed:
        print(f"FAIL: {', '.join(failed)}")
        return 1

    print("PASS: validation thresholds satisfied")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
