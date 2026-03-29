#!/usr/bin/env python3
"""Generate standalone daily governance health summary report."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"config not found: {path}")
    try:
        import yaml

        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception as exc:
        raise RuntimeError(f"failed to parse config {path}: {exc}") from exc
    if not isinstance(cfg, dict):
        return {}
    return cfg


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


def _report_status(
    *,
    payload: dict[str, Any] | None,
    now_utc: datetime,
    max_age_hours: float,
) -> tuple[str, str, float | None]:
    if payload is None:
        return "MISSING", "report_missing", None

    status = str(payload.get("status", "")).strip().upper()
    run_at = _parse_iso8601_utc(str(payload.get("run_at_utc", "") or ""))
    if not status:
        return "FAIL", "status_missing", None
    if status != "PASS":
        return "FAIL", f"status_{status.lower()}", None
    if run_at is None:
        return "FAIL", "run_at_utc_missing", None

    age_hours = (now_utc - run_at).total_seconds() / 3600.0
    if age_hours < 0:
        return "FAIL", "run_at_in_future", age_hours
    if age_hours > max_age_hours:
        return "FAIL", "report_stale", age_hours
    return "PASS", "", age_hours


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate daily governance health summary")
    parser.add_argument("--config", type=str, default="backtest_config.yaml")
    parser.add_argument("--max-age-hours", type=float, default=None)
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when overall status is not PASS")
    args = parser.parse_args()

    cfg_path = (ROOT / args.config).resolve()
    try:
        cfg = _read_config(cfg_path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    now_utc = datetime.now(timezone.utc)
    readiness_cfg = (((cfg.get("model_selection") or {}).get("promotion") or {}).get("production_readiness") or {})
    gov_cfg = ((cfg.get("governance") or {}).get("daily_summary") or {})

    default_age = float(readiness_cfg.get("max_report_age_hours", 36.0) or 36.0)
    cfg_age = float(gov_cfg.get("max_report_age_hours", default_age) or default_age)
    max_age_hours = float(args.max_age_hours) if args.max_age_hours is not None else cfg_age

    out_rel = str(gov_cfg.get("output_dir", "output/live/governance") or "output/live/governance")
    out_dir = (cfg_path.parent / out_rel).resolve() if not Path(out_rel).is_absolute() else Path(out_rel)
    out_dir.mkdir(parents=True, exist_ok=True)

    shadow_path = ROOT / "output" / "models" / "shadow_monitor_latest.json"
    risk_path = ROOT / "output" / "live" / "risk_gate" / "risk_gate_latest.json"
    tca_path = ROOT / "output" / "live" / "tca" / "tca_health_latest.json"
    pointer_path = ROOT / "output" / "models" / "production_pointer.json"

    live_cfg = cfg.get("live") or {}
    if not isinstance(live_cfg, dict):
        live_cfg = {}
    latch_rel = str(live_cfg.get("trading_halt_latch_path", "output/live/trading_halt_latch.json") or "output/live/trading_halt_latch.json")
    latch_path = (cfg_path.parent / latch_rel).resolve() if not Path(latch_rel).is_absolute() else Path(latch_rel)

    shadow = _read_json(shadow_path)
    risk = _read_json(risk_path)
    tca = _read_json(tca_path)
    latch = _read_json(latch_path)
    pointer = _read_json(pointer_path)

    gates: dict[str, Any] = {}
    failures: list[str] = []

    for gate_name, payload in (("shadow_monitor", shadow), ("risk_gate", risk), ("tca_health", tca)):
        status, reason, age_hours = _report_status(payload=payload, now_utc=now_utc, max_age_hours=max_age_hours)
        gates[gate_name] = {
            "status": status,
            "reason": reason,
            "age_hours": age_hours,
            "run_at_utc": payload.get("run_at_utc") if isinstance(payload, dict) else None,
            "failures": payload.get("failures", []) if isinstance(payload, dict) else [],
            "metrics": payload.get("metrics", {}) if isinstance(payload, dict) else {},
            "path": str({"shadow_monitor": shadow_path, "risk_gate": risk_path, "tca_health": tca_path}[gate_name]),
        }
        if status != "PASS":
            failures.append(f"{gate_name}:{reason}")

    halt_active = bool((latch or {}).get("halt_active", False)) if isinstance(latch, dict) else False
    halt_reason = str((latch or {}).get("reason", "") or "") if isinstance(latch, dict) else ""

    if halt_active:
        overall_status = "BLOCKED"
        failures.append("trading_halt_latch_active")
    elif all(gates[g]["status"] == "PASS" for g in ("shadow_monitor", "risk_gate", "tca_health")):
        overall_status = "PASS"
    else:
        overall_status = "FAIL"

    summary = {
        "run_at_utc": now_utc.isoformat(),
        "overall_status": overall_status,
        "max_report_age_hours": max_age_hours,
        "gates": gates,
        "trading_halt_latch": {
            "path": str(latch_path),
            "present": isinstance(latch, dict),
            "halt_active": halt_active,
            "reason": halt_reason,
            "updated_at_utc": (latch or {}).get("updated_at_utc") if isinstance(latch, dict) else None,
        },
        "production": {
            "path": str(pointer_path),
            "present": isinstance(pointer, dict),
            "run_id": (pointer or {}).get("run_id") if isinstance(pointer, dict) else None,
            "model_name": (pointer or {}).get("model_name") if isinstance(pointer, dict) else None,
            "promoted_at_utc": (pointer or {}).get("promoted_at_utc") if isinstance(pointer, dict) else None,
        },
        "failures": failures,
    }

    ts = now_utc.strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"governance_daily_summary_{ts}.json"
    latest_path = out_dir / "governance_daily_summary_latest.json"
    report_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    latest_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print(f"Governance summary: {report_path}")
    print(json.dumps({"overall_status": overall_status, "n_failures": len(failures)}, indent=2))

    if args.strict and overall_status != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
