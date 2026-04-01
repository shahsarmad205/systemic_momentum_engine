#!/usr/bin/env python3
"""Evaluate governance SLO health from the latest governance summary."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"config not found: {path}")
    try:
        import yaml

        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as exc:
        raise RuntimeError(f"failed to parse config {path}: {exc}") from exc
    return cfg if isinstance(cfg, dict) else {}


def _read_json(path: Path) -> tuple[str, dict[str, Any] | None]:
    if not path.exists():
        return ("missing", None)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return ("invalid", None)
    if not isinstance(payload, dict):
        return ("invalid", None)
    return ("ok", payload)


def _parse_iso8601_utc(raw: str) -> datetime | None:
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
    parser = argparse.ArgumentParser(description="Evaluate governance SLO health")
    parser.add_argument("--config", type=str, default="backtest_config.yaml")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when SLO status is not PASS")
    parser.add_argument("--summary", type=str, default="", help="Optional explicit governance summary path")
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
    daily_cfg = gov_cfg.get("daily_summary") or {}
    if not isinstance(daily_cfg, dict):
        daily_cfg = {}
    slo_cfg = gov_cfg.get("slo") or {}
    if not isinstance(slo_cfg, dict):
        slo_cfg = {}

    if not bool(slo_cfg.get("enabled", True)):
        print("SLO check disabled in config")
        return 0

    default_summary_rel = str(daily_cfg.get("output_dir", "output/live/governance") or "output/live/governance")
    default_summary_path = (cfg_path.parent / default_summary_rel).resolve() / "governance_daily_summary_latest.json"
    summary_path = Path(args.summary).resolve() if args.summary else default_summary_path
    summary_state, summary = _read_json(summary_path)

    now_utc = datetime.now(timezone.utc)
    max_age_raw = slo_cfg.get("max_summary_age_hours", 36.0)
    if max_age_raw is None:
        max_age_raw = 36.0
    max_summary_age_hours = float(max_age_raw)

    allowed_statuses = slo_cfg.get("allowed_overall_statuses", ["PASS"])
    if not isinstance(allowed_statuses, list) or not allowed_statuses:
        allowed_statuses = ["PASS"]
    allowed_statuses = [str(x).upper() for x in allowed_statuses]

    max_failures_raw = slo_cfg.get("max_governance_failures", 0)
    if max_failures_raw is None:
        max_failures_raw = 0
    max_governance_failures = int(max_failures_raw)

    fail_on_missing_summary = bool(slo_cfg.get("fail_on_missing_summary", True))
    fail_on_invalid_summary = bool(slo_cfg.get("fail_on_invalid_summary", True))
    blocked_statuses_cfg = slo_cfg.get("blocked_statuses", ["BLOCKED"])
    if not isinstance(blocked_statuses_cfg, list):
        blocked_statuses_cfg = ["BLOCKED"]
    blocked_statuses = [str(x).upper() for x in blocked_statuses_cfg]

    checks: list[dict[str, Any]] = []
    reasons: list[str] = []

    if summary_state == "missing":
        checks.append(
            {
                "name": "governance_summary",
                "status": "MISSING",
                "reason": "summary_missing",
                "path": str(summary_path),
            }
        )
        if fail_on_missing_summary:
            reasons.append("summary_missing")
        summary_status = ""
        summary_failures = []
    elif summary_state == "invalid":
        checks.append(
            {
                "name": "governance_summary",
                "status": "FAIL",
                "reason": "summary_invalid",
                "path": str(summary_path),
            }
        )
        if fail_on_invalid_summary:
            reasons.append("summary_invalid")
        summary_status = ""
        summary_failures = []
    else:
        summary_status = str(summary.get("overall_status", "")).strip().upper()
        summary_failures = summary.get("failures", [])
        if not isinstance(summary_failures, list):
            summary_failures = []

        run_at = _parse_iso8601_utc(str(summary.get("run_at_utc", "") or ""))
        if run_at is None:
            checks.append(
                {
                    "name": "summary_freshness",
                    "status": "FAIL",
                    "reason": "summary_run_at_missing",
                }
            )
            reasons.append("summary_run_at_missing" if fail_on_invalid_summary else "")
        else:
            age_hours = (now_utc - run_at).total_seconds() / 3600.0
            if age_hours < 0:
                checks.append(
                    {
                        "name": "summary_freshness",
                        "status": "FAIL",
                        "reason": "summary_run_at_in_future",
                        "age_hours": age_hours,
                    }
                )
                reasons.append("summary_run_at_in_future")
            elif age_hours > max_summary_age_hours:
                checks.append(
                    {
                        "name": "summary_freshness",
                        "status": "STALE",
                        "reason": "summary_stale",
                        "age_hours": age_hours,
                    }
                )
                reasons.append("summary_stale")
            else:
                checks.append(
                    {
                        "name": "summary_freshness",
                        "status": "PASS",
                        "reason": "",
                        "age_hours": age_hours,
                    }
                )

        if not summary_status:
            checks.append(
                {
                    "name": "overall_status",
                    "status": "FAIL",
                    "reason": "summary_status_missing",
                }
            )
            if fail_on_invalid_summary:
                reasons.append("summary_status_missing")
        elif summary_status in blocked_statuses:
            checks.append(
                {
                    "name": "overall_status",
                    "status": "FAIL",
                    "reason": "governance_blocked",
                    "value": summary_status,
                }
            )
            reasons.append("governance_blocked")
        elif summary_status not in allowed_statuses:
            checks.append(
                {
                    "name": "overall_status",
                    "status": "FAIL",
                    "reason": "governance_status_not_allowed",
                    "value": summary_status,
                }
            )
            reasons.append("governance_status_not_allowed")
        else:
            checks.append(
                {
                    "name": "overall_status",
                    "status": "PASS",
                    "reason": "",
                    "value": summary_status,
                }
            )

        n_failures = len(summary_failures)
        if n_failures > max_governance_failures:
            checks.append(
                {
                    "name": "governance_failures_budget",
                    "status": "FAIL",
                    "reason": "governance_failures_exceeded",
                    "value": n_failures,
                }
            )
            reasons.append("governance_failures_exceeded")
        else:
            checks.append(
                {
                    "name": "governance_failures_budget",
                    "status": "PASS",
                    "reason": "",
                    "value": n_failures,
                }
            )

    reasons = [r for r in reasons if r]
    if summary_status in blocked_statuses:
        status = "BLOCKED"
    elif reasons:
        status = "FAIL"
    else:
        status = "PASS"

    out_dir_rel = str(slo_cfg.get("output_dir", "output/live/slo") or "output/live/slo")
    out_dir = (cfg_path.parent / out_dir_rel).resolve() if not Path(out_dir_rel).is_absolute() else Path(out_dir_rel)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "run_at_utc": now_utc.isoformat(),
        "status": status,
        "reasons": reasons,
        "checks": checks,
        "source": {
            "governance_summary_path": str(summary_path),
            "governance_summary_run_at_utc": (summary or {}).get("run_at_utc") if isinstance(summary, dict) else None,
            "governance_overall_status": summary_status or None,
        },
        "thresholds": {
            "max_summary_age_hours": max_summary_age_hours,
            "allowed_overall_statuses": allowed_statuses,
            "max_governance_failures": max_governance_failures,
            "blocked_statuses": blocked_statuses,
            "fail_on_missing_summary": fail_on_missing_summary,
            "fail_on_invalid_summary": fail_on_invalid_summary,
        },
    }

    ts = now_utc.strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"slo_health_{ts}.json"
    latest_path = out_dir / "slo_health_latest.json"
    encoded = json.dumps(payload, indent=2) + "\n"
    report_path.write_text(encoded, encoding="utf-8")
    latest_path.write_text(encoded, encoding="utf-8")

    print(f"SLO health report: {report_path}")
    print(json.dumps({"status": status, "n_reasons": len(reasons)}, indent=2))

    if args.strict and status != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
