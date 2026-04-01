#!/usr/bin/env python3
"""
Shadow monitor for promoted model lifecycle.

Compares the current shadow model against the production model and emits
score drift + signal overlap diagnostics for recent data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from agents.weight_learning_agent.feature_builder import build_feature_matrix
from run_model_selection import _sharpe_from_series, _strategy_daily_returns


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"config not found: {path}")
    try:
        import yaml

        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        raise RuntimeError(f"failed to parse config {path}: {exc}") from exc


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"failed to parse json {path}: {exc}") from exc


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _resolve_models_path(models_dir: Path, rel_path: str) -> Path | None:
    rel = str(rel_path or "").strip()
    if not rel:
        return None
    p = Path(rel)
    if p.is_absolute():
        return None
    try:
        resolved = (ROOT / p).resolve()
        if not resolved.is_relative_to(models_dir.resolve()):
            return None
    except Exception:
        return None
    return resolved


def _load_artifact(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, dict):
        raise RuntimeError(f"Invalid model artifact at {path}")
    return obj


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


def _daily_topk_overlap(df: pd.DataFrame, k: int) -> float:
    if df.empty:
        return float("nan")
    overlaps: list[float] = []
    for _, g in df.groupby("date", sort=True):
        g = g.sort_values("ticker")
        top_shadow = set(g.nlargest(k, "score_shadow")["ticker"].astype(str).tolist())
        top_prod = set(g.nlargest(k, "score_prod")["ticker"].astype(str).tolist())
        if not top_shadow or not top_prod:
            continue
        denom = float(max(len(top_shadow), len(top_prod)))
        overlaps.append(len(top_shadow.intersection(top_prod)) / denom)
    if not overlaps:
        return float("nan")
    return float(np.mean(overlaps))


def _latest_shadow_entry(registry: dict[str, Any]) -> dict[str, Any] | None:
    entries = registry.get("entries", []) if isinstance(registry, dict) else []
    if not isinstance(entries, list):
        return None
    shadow_entries = [e for e in entries if str(e.get("current_state", "")) == "shadow"]
    if not shadow_entries:
        return None

    def _entry_ts(e: dict[str, Any]) -> str:
        hist = e.get("state_history", [])
        if isinstance(hist, list) and hist:
            for h in reversed(hist):
                if str(h.get("state", "")) == "shadow":
                    return str(h.get("at_utc", ""))
        return ""

    shadow_entries.sort(key=_entry_ts, reverse=True)
    return shadow_entries[0]


def _entry_by_run_id(registry: dict[str, Any], run_id: str) -> dict[str, Any] | None:
    entries = registry.get("entries", []) if isinstance(registry, dict) else []
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if str(entry.get("run_id", "")) == str(run_id):
            return entry
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Run shadow-vs-production drift monitor")
    parser.add_argument("--config", type=str, default="backtest_config.yaml")
    parser.add_argument("--lookback-days", type=int, default=0)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on monitor gate failure")
    args = parser.parse_args()

    cfg_path = ROOT / args.config
    try:
        cfg = _read_config(cfg_path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1
    ms_cfg = cfg.get("model_selection", {}) or {}
    sm_cfg = ms_cfg.get("shadow_monitor", {}) or {}

    if not bool(sm_cfg.get("enabled", True)):
        print("Shadow monitor disabled in config")
        return 0

    models_dir = ROOT / "output" / "models"
    try:
        registry = _read_json(models_dir / "model_registry.json", {})
        pointer = _read_json(models_dir / "production_pointer.json", {})
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    shadow_entry = _latest_shadow_entry(registry)
    if shadow_entry is None:
        print("ERROR: no shadow model found in registry")
        return 1
    if not pointer:
        print("ERROR: no production_pointer.json found; cannot compare shadow vs production")
        return 1

    shadow_model_path = _resolve_models_path(models_dir, str(shadow_entry.get("best_model_path", "")))
    prod_model_path = _resolve_models_path(models_dir, str(pointer.get("best_model_path", "")))
    if shadow_model_path is None:
        print("ERROR: unsafe or missing shadow model path in registry")
        return 1
    if prod_model_path is None:
        print("ERROR: unsafe or missing production model path in pointer")
        return 1
    if not shadow_model_path.exists():
        print(f"ERROR: shadow model not found at {shadow_model_path}")
        return 1
    if not prod_model_path.exists():
        print(f"ERROR: production model not found at {prod_model_path}")
        return 1

    shadow_hash_expected = str((shadow_entry.get("artifact_hashes", {}) or {}).get("best_model_sha256", "") or "")
    if not shadow_hash_expected:
        print("ERROR: missing shadow artifact hash in registry")
        return 1
    shadow_hash_got = _sha256_file(shadow_model_path)
    if shadow_hash_got != shadow_hash_expected:
        print("ERROR: shadow artifact hash mismatch")
        return 1

    prod_run_id = str(pointer.get("run_id", "") or "")
    prod_entry = _entry_by_run_id(registry, prod_run_id)
    if prod_entry is None:
        print("ERROR: production run_id not found in registry; cannot verify production artifact integrity")
        return 1
    prod_hash_expected = str((prod_entry.get("artifact_hashes", {}) or {}).get("best_model_sha256", "") or "")
    if not prod_hash_expected:
        print("ERROR: missing production artifact hash in registry")
        return 1
    prod_hash_got = _sha256_file(prod_model_path)
    if prod_hash_got != prod_hash_expected:
        print("ERROR: production artifact hash mismatch")
        return 1

    shadow_art = _load_artifact(shadow_model_path)
    prod_art = _load_artifact(prod_model_path)

    shadow_model = shadow_art.get("estimator")
    prod_model = prod_art.get("estimator")
    if shadow_model is None or prod_model is None:
        print("ERROR: missing estimator in one or both artifacts")
        return 1

    shadow_kind = str(shadow_art.get("model_type", "classifier"))
    prod_kind = str(prod_art.get("model_type", "classifier"))
    shadow_cols = [str(c) for c in (shadow_art.get("feature_columns", []) or [])]
    prod_cols = [str(c) for c in (prod_art.get("feature_columns", []) or [])]
    horizon = int(shadow_art.get("horizon_days", ms_cfg.get("lookahead_horizon_days", 1)) or 1)

    if not shadow_cols or not prod_cols:
        print("ERROR: missing feature_columns in artifact(s)")
        return 1

    bt = cfg.get("backtest", {}) or {}
    end_date = pd.Timestamp(str(bt.get("end_date", "2024-01-01")))
    lb_days = int(args.lookback_days) if int(args.lookback_days) > 0 else int(sm_cfg.get("lookback_days", 90) or 90)
    start_date = (end_date - timedelta(days=lb_days)).date().isoformat()

    tickers = list(cfg.get("tickers", []) or [])
    if not tickers:
        print("ERROR: no tickers configured")
        return 1

    df = build_feature_matrix(
        tickers,
        start_date=start_date,
        end_date=end_date.date().isoformat(),
        holding_period=horizon,
        feature_subset=None,
    )
    if df is None or df.empty:
        print("ERROR: feature matrix empty in lookback window")
        return 1

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "forward_return"]).sort_values(["date", "ticker"]).reset_index(drop=True)

    missing_shadow = [c for c in shadow_cols if c not in df.columns]
    missing_prod = [c for c in prod_cols if c not in df.columns]
    if missing_shadow:
        print(f"ERROR: missing shadow features (sample): {missing_shadow[:8]}")
        return 1
    if missing_prod:
        print(f"ERROR: missing production features (sample): {missing_prod[:8]}")
        return 1

    x_shadow = df[shadow_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    x_prod = df[prod_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)

    score_shadow = _score_model(shadow_model, shadow_kind, x_shadow)
    score_prod = _score_model(prod_model, prod_kind, x_prod)

    scored = df[["date", "ticker", "forward_return"]].copy()
    scored["score_shadow"] = score_shadow
    scored["score_prod"] = score_prod

    corr = float(np.corrcoef(scored["score_shadow"], scored["score_prod"])[0, 1]) if len(scored) > 2 else float("nan")
    mean_abs_delta = float(np.mean(np.abs(scored["score_shadow"] - scored["score_prod"]))) if len(scored) else float("nan")
    max_positions = int(ms_cfg.get("max_positions", 10) or 10)
    min_positions = int(ms_cfg.get("min_positions", 3) or 3)
    overlap = _daily_topk_overlap(scored, max_positions)

    shadow_dr = _strategy_daily_returns(
        scored.rename(columns={"score_shadow": "score"}),
        max_positions=max_positions,
        min_positions=min_positions,
    )
    prod_dr = _strategy_daily_returns(
        scored.rename(columns={"score_prod": "score"}),
        max_positions=max_positions,
        min_positions=min_positions,
    )

    shadow_sharpe = _sharpe_from_series(shadow_dr.to_numpy(dtype=float)) if len(shadow_dr) else float("nan")
    prod_sharpe = _sharpe_from_series(prod_dr.to_numpy(dtype=float)) if len(prod_dr) else float("nan")

    min_score_corr = float(sm_cfg.get("min_score_corr", 0.85) or 0.85)
    min_topk_overlap = float(sm_cfg.get("min_topk_overlap", 0.50) or 0.50)
    max_abs_score_delta = float(sm_cfg.get("max_abs_score_delta", 0.25) or 0.25)

    failures: list[str] = []
    if not np.isfinite(corr) or corr < min_score_corr:
        failures.append(f"score_corr<{min_score_corr:.3f}")
    if not np.isfinite(overlap) or overlap < min_topk_overlap:
        failures.append(f"topk_overlap<{min_topk_overlap:.3f}")
    if not np.isfinite(mean_abs_delta) or mean_abs_delta > max_abs_score_delta:
        failures.append(f"abs_score_delta>{max_abs_score_delta:.3f}")

    ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")
    report = {
        "run_at_utc": pd.Timestamp.utcnow().isoformat(),
        "lookback_days": int(lb_days),
        "window": {
            "start_date": start_date,
            "end_date": end_date.date().isoformat(),
        },
        "shadow": {
            "run_id": str(shadow_entry.get("run_id", "")),
            "model_name": str(shadow_entry.get("model_name", "")),
            "model_path": str(shadow_model_path),
            "sharpe": float(shadow_sharpe) if np.isfinite(shadow_sharpe) else None,
        },
        "production": {
            "run_id": str(pointer.get("run_id", "")),
            "model_name": str(pointer.get("model_name", "")),
            "model_path": str(prod_model_path),
            "sharpe": float(prod_sharpe) if np.isfinite(prod_sharpe) else None,
        },
        "metrics": {
            "score_corr": float(corr) if np.isfinite(corr) else None,
            "mean_abs_score_delta": float(mean_abs_delta) if np.isfinite(mean_abs_delta) else None,
            "daily_topk_overlap": float(overlap) if np.isfinite(overlap) else None,
            "n_rows": int(len(scored)),
            "n_days": int(scored["date"].nunique()),
        },
        "gates": {
            "min_score_corr": min_score_corr,
            "min_topk_overlap": min_topk_overlap,
            "max_abs_score_delta": max_abs_score_delta,
        },
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }

    out_dir = ROOT / "output" / "models"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / f"shadow_monitor_{ts}.json"
    latest_path = out_dir / "shadow_monitor_latest.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    latest_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    print(f"Shadow monitor report: {report_path}")
    print(f"status={report['status']} score_corr={report['metrics']['score_corr']} topk_overlap={report['metrics']['daily_topk_overlap']}")

    if failures and args.strict:
        print("FAIL: " + ", ".join(failures))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
