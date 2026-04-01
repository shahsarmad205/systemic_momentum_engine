#!/usr/bin/env python3
"""
Compare production and shadow model outputs on recent data.

Writes JSON reports to output/models/shadow_reports/.
"""

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
from run_model_selection import _read_config

MODELS_DIR = ROOT / "output" / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
POINTER_PATH = MODELS_DIR / "production_pointer.json"


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


def _load_model_artifact(path: Path) -> dict[str, Any]:
    with open(path, "rb") as fh:
        obj = pickle.load(fh)
    if not isinstance(obj, dict):
        raise RuntimeError(f"invalid artifact structure: {path}")
    if obj.get("estimator") is None:
        raise RuntimeError(f"artifact missing estimator: {path}")
    return obj


def _resolve_model_path(path_str: str) -> Path | None:
    raw = Path(str(path_str or "").strip())
    if not str(raw):
        return None
    if raw.is_absolute():
        resolved = raw.resolve()
    else:
        resolved = (ROOT / raw).resolve()
    try:
        if not resolved.is_relative_to(MODELS_DIR.resolve()):
            return None
    except Exception:
        return None
    if resolved.suffix.lower() != ".pkl":
        return None
    return resolved


def _safe_float(x: float) -> float | None:
    return float(x) if np.isfinite(x) else None


def _latest_shadow_entry(registry: dict[str, Any]) -> dict[str, Any] | None:
    entries = registry.get("entries", []) if isinstance(registry, dict) else []
    if not isinstance(entries, list):
        return None
    shadows = [e for e in entries if str(e.get("current_state", "")) == "shadow"]
    if not shadows:
        return None

    def _ts(e: dict[str, Any]) -> str:
        hist = e.get("state_history", [])
        if isinstance(hist, list) and hist:
            last = hist[-1]
            if isinstance(last, dict):
                return str(last.get("at_utc", ""))
        return ""

    shadows.sort(key=_ts, reverse=True)
    return shadows[0]


def _top_k_overlap_by_day(df: pd.DataFrame, top_k: int) -> float:
    overlaps: list[float] = []
    for _, day_df in df.groupby("date"):
        k_eff = int(max(1, min(int(top_k), int(len(day_df)))))
        prod_top = set(day_df.nlargest(k_eff, "score_prod")["ticker"].astype(str).tolist())
        shad_top = set(day_df.nlargest(k_eff, "score_shadow")["ticker"].astype(str).tolist())
        if not prod_top and not shad_top:
            continue
        denom = float(max(1, k_eff))
        overlaps.append(float(len(prod_top & shad_top) / denom))
    return float(np.mean(overlaps)) if overlaps else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare production and shadow model behavior")
    parser.add_argument("--config", type=str, default="backtest_config.yaml")
    parser.add_argument("--production-model-path", type=str, default="")
    parser.add_argument("--shadow-model-path", type=str, default="")
    parser.add_argument("--shadow-run-id", type=str, default="")
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when threshold gates fail")
    parser.add_argument("--min-score-correlation", type=float, default=None)
    parser.add_argument("--min-score-sign-agreement", type=float, default=None)
    parser.add_argument("--max-score-mae", type=float, default=None)
    parser.add_argument("--min-top-k-overlap-mean", type=float, default=None)
    args = parser.parse_args()

    cfg = _read_config(args.config)
    bt_cfg = cfg.get("backtest", {}) or {}
    ms_cfg = cfg.get("model_selection", {}) or {}
    sh_cfg = (ms_cfg.get("shadow_monitoring", {}) or ms_cfg.get("shadow_monitor", {}) or {})
    gov_cfg = (cfg.get("governance", {}) or {})
    cmp_cfg = (gov_cfg.get("shadow_comparison", {}) or {})

    cmp_enabled = bool(cmp_cfg.get("enabled", True))
    if not cmp_enabled:
        print("Shadow comparison disabled in config")
        return 0

    lookback_days = int(max(5, args.lookback_days if args.lookback_days is not None else sh_cfg.get("lookback_days", 60) or 60))
    top_k = int(max(1, args.top_k if args.top_k is not None else sh_cfg.get("top_k_overlap", 10) or 10))

    cfg_min_score_correlation = cmp_cfg.get("min_score_correlation", 0.85)
    if cfg_min_score_correlation is None:
        cfg_min_score_correlation = 0.85
    cfg_min_score_sign_agreement = cmp_cfg.get("min_score_sign_agreement", 0.55)
    if cfg_min_score_sign_agreement is None:
        cfg_min_score_sign_agreement = 0.55
    cfg_max_score_mae = cmp_cfg.get("max_score_mae", 0.25)
    if cfg_max_score_mae is None:
        cfg_max_score_mae = 0.25
    cfg_min_top_k_overlap_mean = cmp_cfg.get("min_top_k_overlap_mean", 0.50)
    if cfg_min_top_k_overlap_mean is None:
        cfg_min_top_k_overlap_mean = 0.50

    min_score_correlation = float(args.min_score_correlation) if args.min_score_correlation is not None else float(cfg_min_score_correlation)
    min_score_sign_agreement = float(args.min_score_sign_agreement) if args.min_score_sign_agreement is not None else float(cfg_min_score_sign_agreement)
    max_score_mae = float(args.max_score_mae) if args.max_score_mae is not None else float(cfg_max_score_mae)
    min_top_k_overlap_mean = float(args.min_top_k_overlap_mean) if args.min_top_k_overlap_mean is not None else float(cfg_min_top_k_overlap_mean)

    production_path: Path | None = None
    if args.production_model_path:
        production_path = _resolve_model_path(args.production_model_path)
    else:
        pointer = json.loads(POINTER_PATH.read_text(encoding="utf-8")) if POINTER_PATH.exists() else {}
        prod_rel = str(pointer.get("best_model_path", "") or "")
        if prod_rel:
            production_path = _resolve_model_path(prod_rel)

    if production_path is None or not production_path.exists():
        print("ERROR: production model path not found; provide --production-model-path or create production pointer")
        return 1

    shadow_path: Path | None = None
    if args.shadow_model_path:
        shadow_path = _resolve_model_path(args.shadow_model_path)
    else:
        registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8")) if REGISTRY_PATH.exists() else {}
        shadow_entry: dict[str, Any] | None = None
        if args.shadow_run_id:
            entries = registry.get("entries", []) if isinstance(registry, dict) else []
            if isinstance(entries, list):
                for e in entries:
                    if str(e.get("run_id", "")) == str(args.shadow_run_id):
                        shadow_entry = e
                        break
        else:
            shadow_entry = _latest_shadow_entry(registry)

        if shadow_entry:
            shadow_rel = str(shadow_entry.get("best_model_path", "") or "")
            if shadow_rel:
                shadow_path = _resolve_model_path(shadow_rel)

    if shadow_path is None or not shadow_path.exists():
        print("ERROR: shadow model path not found; provide --shadow-model-path or create shadow registry entry")
        return 1

    prod_art = _load_model_artifact(production_path)
    shad_art = _load_model_artifact(shadow_path)

    prod_feats = [str(x) for x in (prod_art.get("feature_columns", []) or [])]
    shad_feats = [str(x) for x in (shad_art.get("feature_columns", []) or [])]
    if not prod_feats or not shad_feats:
        print("ERROR: one or both model artifacts are missing feature_columns")
        return 1

    horizon = int(max(int(prod_art.get("horizon_days", 1) or 1), int(shad_art.get("horizon_days", 1) or 1)))

    tickers = list(cfg.get("tickers", []) or [])
    if not tickers:
        print("ERROR: no tickers configured")
        return 1

    start_date = str(bt_cfg.get("start_date", "2018-01-01"))
    end_date = str(bt_cfg.get("end_date", "2024-01-01"))

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
    df = df.dropna(subset=["date"]).sort_values(["date", "ticker"]).reset_index(drop=True)

    required_cols = sorted(set(prod_feats + shad_feats))
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"ERROR: feature columns missing from matrix (sample): {missing[:10]}")
        return 1

    cutoff = df["date"].max() - pd.Timedelta(days=lookback_days)
    work = df[df["date"] >= cutoff].copy()
    if work.empty:
        print("ERROR: no rows in selected lookback window")
        return 1

    x_prod = work[prod_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
    x_shad = work[shad_feats].replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)

    score_prod = _score_model(prod_art["estimator"], str(prod_art.get("model_type", "classifier")), x_prod)
    score_shadow = _score_model(shad_art["estimator"], str(shad_art.get("model_type", "classifier")), x_shad)

    work = work.assign(score_prod=score_prod, score_shadow=score_shadow)

    prod_vals = work["score_prod"].to_numpy(dtype=float)
    shad_vals = work["score_shadow"].to_numpy(dtype=float)
    corr = float(np.corrcoef(prod_vals, shad_vals)[0, 1]) if len(work) > 1 else float("nan")
    sign_agree = float(np.mean(np.sign(prod_vals) == np.sign(shad_vals)))
    mae = float(np.mean(np.abs(work["score_prod"].to_numpy(dtype=float) - work["score_shadow"].to_numpy(dtype=float))))
    topk_overlap = _top_k_overlap_by_day(work[["date", "ticker", "score_prod", "score_shadow"]], top_k=top_k)

    failures: list[str] = []
    if not np.isfinite(corr) or corr < min_score_correlation:
        failures.append(f"score_correlation<{min_score_correlation:.3f}")
    if not np.isfinite(sign_agree) or sign_agree < min_score_sign_agreement:
        failures.append(f"score_sign_agreement<{min_score_sign_agreement:.3f}")
    if not np.isfinite(mae) or mae > max_score_mae:
        failures.append(f"score_mae>{max_score_mae:.3f}")
    if not np.isfinite(topk_overlap) or topk_overlap < min_top_k_overlap_mean:
        failures.append(f"top_k_overlap_mean<{min_top_k_overlap_mean:.3f}")

    now_utc = datetime.now(timezone.utc)

    report = {
        "generated_at_utc": now_utc.isoformat(),
        "run_at_utc": now_utc.isoformat(),
        "lookback_days": int(lookback_days),
        "top_k": int(top_k),
        "n_rows": int(len(work)),
        "n_days": int(work["date"].nunique()),
        "production": {
            "path": str(production_path),
            "model_name": str(prod_art.get("model_name", "")),
            "model_type": str(prod_art.get("model_type", "")),
            "horizon_days": int(prod_art.get("horizon_days", 1) or 1),
        },
        "shadow": {
            "path": str(shadow_path),
            "model_name": str(shad_art.get("model_name", "")),
            "model_type": str(shad_art.get("model_type", "")),
            "horizon_days": int(shad_art.get("horizon_days", 1) or 1),
        },
        "metrics": {
            "score_correlation": _safe_float(corr),
            "score_sign_agreement": _safe_float(sign_agree),
            "score_mae": _safe_float(mae),
            "top_k_overlap_mean": _safe_float(topk_overlap),
        },
        "gates": {
            "min_score_correlation": min_score_correlation,
            "min_score_sign_agreement": min_score_sign_agreement,
            "max_score_mae": max_score_mae,
            "min_top_k_overlap_mean": min_top_k_overlap_mean,
        },
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
    }

    out_dir = MODELS_DIR / "shadow_reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = out_dir / f"shadow_compare_{ts}.json"
    latest_path = out_dir / "latest_shadow_compare.json"
    payload = json.dumps(report, indent=2, allow_nan=False) + "\n"
    out_path.write_text(payload, encoding="utf-8")
    latest_path.write_text(payload, encoding="utf-8")

    print(f"Saved shadow comparison report: {out_path}")
    print(
        "Summary | "
        f"corr={corr:.4f} sign_agree={sign_agree:.4f} mae={mae:.6f} top{top_k}_overlap={topk_overlap:.4f}"
    )
    if failures:
        print("FAIL: " + ", ".join(failures))
        if args.strict:
            return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
