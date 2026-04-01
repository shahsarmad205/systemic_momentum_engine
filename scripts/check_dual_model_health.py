#!/usr/bin/env python3
"""Dual-model health checks for split long/short stack governance."""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.weight_learning_agent.feature_builder import build_feature_matrix


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"config not found: {path}")
    try:
        import yaml

        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        raise RuntimeError(f"failed to parse config {path}: {exc}") from exc


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


def _safe_float(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def _load_artifact(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, dict):
        raise RuntimeError(f"invalid model artifact: {path}")
    if obj.get("estimator") is None:
        raise RuntimeError(f"artifact missing estimator: {path}")
    return obj


def _resolve_path(root: Path, raw_path: str) -> Path:
    p = Path(str(raw_path or "").strip())
    if p.is_absolute():
        return p
    return (root / p).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check split long/short model health")
    parser.add_argument("--config", type=str, default="backtest_config.yaml")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when health status is not PASS")
    args = parser.parse_args()

    cfg_path = (ROOT / args.config).resolve()
    try:
        cfg = _read_config(cfg_path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    ms = cfg.get("model_selection", {}) or {}
    sig = cfg.get("signals", {}) or {}
    split_train = bool((ms.get("split_models", {}) or {}).get("enabled", False))
    split_infer_cfg = (sig.get("split_models", {}) or {})
    split_infer = bool(split_infer_cfg.get("enabled", False))

    gate_cfg = ((cfg.get("governance", {}) or {}).get("dual_model_health", {}) or {})
    gate_enabled = bool(gate_cfg.get("enabled", False))

    now_utc = datetime.now(timezone.utc)
    failures: list[str] = []
    metrics: dict[str, Any] = {}
    incidents = {"long": [], "short": []}

    if not gate_enabled:
        payload = {
            "run_at_utc": now_utc.isoformat(),
            "status": "PASS",
            "reason": "gate_disabled",
            "metrics": {},
            "failures": [],
            "incidents": incidents,
        }
        out_dir = ROOT / "output" / "live" / "dual_model_health"
        out_dir.mkdir(parents=True, exist_ok=True)
        latest = out_dir / "dual_model_health_latest.json"
        latest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("Dual-model health gate disabled")
        return 0

    if not split_train and not split_infer:
        payload = {
            "run_at_utc": now_utc.isoformat(),
            "status": "PASS",
            "reason": "split_models_disabled",
            "metrics": {},
            "failures": [],
            "incidents": incidents,
        }
        out_dir = ROOT / "output" / "live" / "dual_model_health"
        out_dir.mkdir(parents=True, exist_ok=True)
        latest = out_dir / "dual_model_health_latest.json"
        latest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("Split models disabled; dual-model health pass")
        return 0

    long_model_path = _resolve_path(ROOT, str(split_infer_cfg.get("long_model_path", "output/models/best_model_long.pkl")))
    short_model_path = _resolve_path(ROOT, str(split_infer_cfg.get("short_model_path", "output/models/best_model_short.pkl")))

    max_artifact_age_hours = float(gate_cfg.get("max_artifact_age_hours", 72.0) or 72.0)
    min_score_std = float(gate_cfg.get("min_score_std", 1e-4) or 1e-4)
    min_abs_ratio = float(gate_cfg.get("min_abs_score_ratio", 0.25) or 0.25)
    max_abs_ratio = float(gate_cfg.get("max_abs_score_ratio", 4.0) or 4.0)

    for side, path in (("long", long_model_path), ("short", short_model_path)):
        if not path.exists():
            failures.append(f"{side}_artifact_missing")
            incidents[side].append({"code": "artifact_missing", "path": str(path)})
            continue
        age_hours = (now_utc.timestamp() - path.stat().st_mtime) / 3600.0
        metrics[f"{side}_artifact_age_hours"] = _safe_float(age_hours)
        if age_hours > max_artifact_age_hours:
            failures.append(f"{side}_artifact_stale")
            incidents[side].append({"code": "artifact_stale", "age_hours": age_hours, "path": str(path)})

    long_art: dict[str, Any] | None = None
    short_art: dict[str, Any] | None = None
    if long_model_path.exists():
        try:
            long_art = _load_artifact(long_model_path)
        except Exception as exc:  # noqa: BLE001
            failures.append("long_artifact_invalid")
            incidents["long"].append({"code": "artifact_invalid", "detail": str(exc)})
    if short_model_path.exists():
        try:
            short_art = _load_artifact(short_model_path)
        except Exception as exc:  # noqa: BLE001
            failures.append("short_artifact_invalid")
            incidents["short"].append({"code": "artifact_invalid", "detail": str(exc)})

    long_cols = [str(c) for c in ((long_art or {}).get("feature_columns", []) or [])]
    short_cols = [str(c) for c in ((short_art or {}).get("feature_columns", []) or [])]
    if long_art is not None and not long_cols:
        failures.append("long_feature_schema_missing")
        incidents["long"].append({"code": "feature_schema_missing"})
    if short_art is not None and not short_cols:
        failures.append("short_feature_schema_missing")
        incidents["short"].append({"code": "feature_schema_missing"})

    target_long = str((long_art or {}).get("target", "") or "")
    target_short = str((short_art or {}).get("target", "") or "")
    if long_art is not None and target_long and target_long != "y_long":
        failures.append("long_target_mismatch")
        incidents["long"].append({"code": "target_mismatch", "value": target_long})
    if short_art is not None and target_short and target_short != "y_short":
        failures.append("short_target_mismatch")
        incidents["short"].append({"code": "target_mismatch", "value": target_short})

    if long_art is not None and short_art is not None and long_cols and short_cols:
        bt = cfg.get("backtest", {}) or {}
        end_date = str(bt.get("end_date", datetime.now().date().isoformat()))
        lookback_days = int(gate_cfg.get("feature_lookback_days", 120) or 120)
        start_date = (pd.Timestamp(end_date) - pd.Timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        tickers = [str(t).strip() for t in (cfg.get("tickers", []) or []) if str(t).strip()]

        if not tickers:
            failures.append("feature_matrix_no_tickers")
        else:
            horizon = int(max(int((long_art or {}).get("horizon_days", 1) or 1), int((short_art or {}).get("horizon_days", 1) or 1)))
            df = build_feature_matrix(
                tickers,
                start_date=start_date,
                end_date=end_date,
                holding_period=horizon,
                feature_subset=None,
            )
            if df is None or df.empty:
                failures.append("feature_matrix_empty")
            else:
                miss_long = [c for c in long_cols if c not in df.columns]
                miss_short = [c for c in short_cols if c not in df.columns]
                metrics["long_missing_features"] = int(len(miss_long))
                metrics["short_missing_features"] = int(len(miss_short))
                if miss_long:
                    failures.append("long_feature_schema_mismatch")
                    incidents["long"].append({"code": "feature_schema_mismatch", "missing_sample": miss_long[:10]})
                if miss_short:
                    failures.append("short_feature_schema_mismatch")
                    incidents["short"].append({"code": "feature_schema_mismatch", "missing_sample": miss_short[:10]})

                if not miss_long and not miss_short:
                    x_long = df[long_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
                    x_short = df[short_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
                    score_long = _score_model(long_art["estimator"], str(long_art.get("model_type", "classifier")), x_long)
                    score_short = _score_model(short_art["estimator"], str(short_art.get("model_type", "short_classifier")), x_short)

                    long_std = float(np.std(score_long)) if len(score_long) > 0 else float("nan")
                    short_std = float(np.std(score_short)) if len(score_short) > 0 else float("nan")
                    long_abs_mean = float(np.mean(np.abs(score_long))) if len(score_long) > 0 else float("nan")
                    short_abs_mean = float(np.mean(np.abs(score_short))) if len(score_short) > 0 else float("nan")

                    metrics["long_score_std"] = _safe_float(long_std)
                    metrics["short_score_std"] = _safe_float(short_std)
                    metrics["long_abs_score_mean"] = _safe_float(long_abs_mean)
                    metrics["short_abs_score_mean"] = _safe_float(short_abs_mean)

                    if not np.isfinite(long_std) or long_std < min_score_std:
                        failures.append("long_directional_drift")
                        incidents["long"].append({"code": "low_score_std", "value": _safe_float(long_std)})
                    if not np.isfinite(short_std) or short_std < min_score_std:
                        failures.append("short_directional_drift")
                        incidents["short"].append({"code": "low_score_std", "value": _safe_float(short_std)})

                    if np.isfinite(long_abs_mean) and np.isfinite(short_abs_mean) and short_abs_mean > 1e-12:
                        ratio = float(long_abs_mean / short_abs_mean)
                        metrics["long_short_abs_score_ratio"] = ratio
                        if ratio < min_abs_ratio or ratio > max_abs_ratio:
                            failures.append("directional_score_imbalance")
                            incidents["long"].append({"code": "abs_score_ratio_out_of_bounds", "ratio": ratio})
                            incidents["short"].append({"code": "abs_score_ratio_out_of_bounds", "ratio": ratio})

    status = "PASS" if not failures else "FAIL"
    payload = {
        "run_at_utc": now_utc.isoformat(),
        "status": status,
        "failures": failures,
        "metrics": metrics,
        "incidents": incidents,
        "paths": {
            "long_model_path": str(long_model_path),
            "short_model_path": str(short_model_path),
        },
        "thresholds": {
            "max_artifact_age_hours": max_artifact_age_hours,
            "min_score_std": min_score_std,
            "min_abs_score_ratio": min_abs_ratio,
            "max_abs_score_ratio": max_abs_ratio,
        },
    }

    out_dir = ROOT / "output" / "live" / "dual_model_health"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = now_utc.strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"dual_model_health_{ts}.json"
    latest_path = out_dir / "dual_model_health_latest.json"
    encoded = json.dumps(payload, indent=2, allow_nan=False) + "\n"
    report_path.write_text(encoded, encoding="utf-8")
    latest_path.write_text(encoded, encoding="utf-8")

    print(f"Dual-model health report: {report_path}")
    print(json.dumps({"status": status, "n_failures": len(failures)}, indent=2))

    if args.strict and status != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
