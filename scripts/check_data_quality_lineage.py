#!/usr/bin/env python3
"""Phase 10: Data Quality Check (Lineage + Audit Trail)."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import utils.data_governance as dqc
import utils.data_governance as dql

ROOT = Path(__file__).resolve().parents[1]


def _read_config(path: Path) -> dict[str, Any]:
    """Read YAML config file."""
    if not path.exists():
        raise RuntimeError(f"config not found: {path}")
    try:
        import yaml

        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as exc:
        raise RuntimeError(f"failed to parse config {path}: {exc}") from exc
    return cfg if isinstance(cfg, dict) else {}


def _read_json(path: Path) -> dict[str, Any] | None:
    """Read JSON file safely."""
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _parse_iso8601_utc(raw: str) -> datetime | None:
    """Parse ISO8601 datetime string."""
    s = str(raw or "").strip()
    if not s:
        return None
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(s)
    except Exception:
        return None
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def main() -> int:
    """Check data quality lineage for OHLCV cache.

    Exit codes:
      0: Pass (checks passed or disabled)
      1: Configuration/runtime error
      2: Data quality check failed (strict FAIL)
    """
    parser = argparse.ArgumentParser(description="Check data quality and build lineage")
    parser.add_argument("--config", type=str, default="backtest_config.yaml")
    parser.add_argument("--strict", action="store_true", help="Exit 2 on any quality failure")
    args = parser.parse_args()

    cfg_path = (ROOT / args.config).resolve()
    try:
        cfg = _read_config(cfg_path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    gov_cfg = cfg.get("governance") or {}
    if not isinstance(gov_cfg, dict):
        gov_cfg = {}

    dq_cfg = gov_cfg.get("data_quality") or {}
    if not isinstance(dq_cfg, dict):
        dq_cfg = {}

    # Check if Phase 10 is enabled
    enabled = bool(dq_cfg.get("enabled", True))
    if not enabled:
        return 0

    # Output configuration
    out_rel = str(dq_cfg.get("output_dir", "output/live/data_quality") or "output/live/data_quality")
    out_dir = (cfg_path.parent / out_rel).resolve() if not Path(out_rel).is_absolute() else Path(out_rel)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load thresholds
    null_threshold = float(dq_cfg.get("strict_null_threshold", 0.05))
    drift_threshold = float(dq_cfg.get("drift_threshold", 2.0))
    check_schema = bool(dq_cfg.get("check_schema", True))
    check_nulls = bool(dq_cfg.get("check_nulls", True))
    check_drift = bool(dq_cfg.get("check_drift", True))
    fail_on_missing_slo = bool(dq_cfg.get("fail_on_missing_slo", False))
    fail_on_slo_not_pass = bool(dq_cfg.get("fail_on_slo_not_pass", False))

    # Check upstream SLO dependency
    daily_cfg = gov_cfg.get("daily_summary") or {}
    if not isinstance(daily_cfg, dict):
        daily_cfg = {}
    default_summary_rel = str(daily_cfg.get("output_dir", "output/live/governance") or "output/live/governance")
    summary_path = (cfg_path.parent / default_summary_rel).resolve() / "governance_daily_summary_latest.json"
    slo_summary = _read_json(summary_path)

    now_utc = datetime.now(timezone.utc)

    # Validate SLO if required
    slo_status = "UNKNOWN"
    slo_failures: list[str] = []
    if slo_summary is None:
        slo_status = "MISSING"
        if fail_on_missing_slo:
            return 1
    else:
        slo_status = str(slo_summary.get("overall_status", "")).strip().upper() or "UNKNOWN"
        slo_failures = slo_summary.get("failures", [])
        if not isinstance(slo_failures, list):
            slo_failures = []

        # If SLO is BLOCKED or not PASS, handle based on config
        if slo_status == "BLOCKED":
            if fail_on_slo_not_pass:
                return 2
        elif slo_status != "PASS" and fail_on_slo_not_pass:
            return 2

    # Load OHLCV cache
    cache_rel = "output/live/ohlcv_cache"
    cache_dir = (cfg_path.parent / cache_rel).resolve() if not Path(cache_rel).is_absolute() else Path(cache_rel)

    if not cache_dir.exists() or not list(cache_dir.glob("*.parquet")):
        # No cache: PASS (nothing to invalidate)
        output_ts = now_utc.strftime("%Y%m%dT%H%M%SZ")
        output_payload = {
            "run_at_utc": now_utc.isoformat(),
            "overall_status": "PASS",
            "reason": "no_ohlcv_cache_found",
            "checks": [],
            "tickers": [],
        }
        output_path = out_dir / f"data_quality_lineage_{output_ts}.json"
        latest_path = out_dir / "data_quality_lineage_latest.json"
        output_path.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
        latest_path.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
        return 0

    # Process each ticker's OHLCV file
    all_lineages: list[dict[str, Any]] = []
    any_failed = False

    for parquet_file in sorted(cache_dir.glob("*.parquet")):
        ticker = parquet_file.stem

        try:
            df = pd.read_parquet(parquet_file)
        except Exception as exc:
            print(f"WARNING: Failed to load {ticker} parquet: {exc}")
            continue

        tracker = dql.LineageTracker(ticker=ticker)
        tracker.add_data_source(path=str(parquet_file), rows=len(df))
        tracker.set_upstream_slo(status=slo_status, failures=slo_failures)

        # Run configured checks
        passed = True

        if check_schema:
            checker = dqc.DataQualityChecker(df, null_threshold=null_threshold, drift_threshold=drift_threshold)
            schema_result = checker.validate_schema()
            tracker.add_check(name="schema_validation", status=schema_result["status"], details=schema_result)
            if schema_result["status"] != "PASS":
                passed = False

        if check_nulls and passed:
            checker = dqc.DataQualityChecker(df, null_threshold=null_threshold, drift_threshold=drift_threshold)
            nulls_result = checker.detect_nulls()
            tracker.add_check(name="null_detection", status=nulls_result["status"], details=nulls_result)
            if nulls_result["status"] != "PASS":
                passed = False

        if check_drift and passed and len(df) > 10:
            # For drift check, we'd need a baseline. For now, skip or use simple checks
            checker = dqc.DataQualityChecker(df, null_threshold=null_threshold, drift_threshold=drift_threshold)
            drift_result = checker.detect_drift(baseline=None)
            tracker.add_check(name="drift_detection", status=drift_result["status"], details=drift_result)

        lineage = tracker.build_lineage()
        all_lineages.append(lineage)

        if not passed:
            any_failed = True

    # Aggregate results
    overall_status = "FAIL" if any_failed else "PASS"

    output_ts = now_utc.strftime("%Y%m%dT%H%M%SZ")
    output_payload = {
        "run_at_utc": now_utc.isoformat(),
        "overall_status": overall_status,
        "upstream_slo": {
            "status": slo_status,
            "failures": slo_failures,
        },
        "tickers_checked": len(all_lineages),
        "lineages": all_lineages,
    }

    output_path = out_dir / f"data_quality_lineage_{output_ts}.json"
    latest_path = out_dir / "data_quality_lineage_latest.json"
    output_path.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")
    latest_path.write_text(json.dumps(output_payload, indent=2) + "\n", encoding="utf-8")

    print(f"Data quality lineage: {latest_path}")
    print(json.dumps({"overall_status": overall_status, "tickers_checked": len(all_lineages)}, indent=2))

    # Exit codes
    if overall_status == "FAIL" or (args.strict and overall_status != "PASS"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
