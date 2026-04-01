#!/usr/bin/env python3
"""Rollout guard for split-stack deployment rings with optional auto-rollback."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from utils.trading_control import set_trading_halt_latch


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"config not found: {path}")
    try:
        import yaml

        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        raise RuntimeError(f"failed to parse config {path}: {exc}") from exc


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _latest_previous_production_run_id(log_path: Path, current_run_id: str) -> str | None:
    if not log_path.exists():
        return None
    seq: list[str] = []
    try:
        for line in log_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                ev = json.loads(line)
            except Exception:
                continue
            if not isinstance(ev, dict) or not bool(ev.get("success", False)):
                continue
            action = str(ev.get("action", "") or "").strip().lower()
            if action == "promote" and str(ev.get("to_state", "") or "").strip().lower() == "production":
                rid = str(ev.get("run_id", "") or "").strip()
                if rid:
                    seq.append(rid)
            elif action == "rollback":
                rid = str(ev.get("to_run_id", "") or "").strip()
                if rid:
                    seq.append(rid)
    except Exception:
        return None

    for rid in reversed(seq):
        if rid and rid != current_run_id:
            return rid
    return None


def _run_rollback(*, to_run_id: str, actor: str, reason: str, dry_run: bool) -> tuple[int, str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "rollback_model.py"),
        "--to-run-id",
        to_run_id,
        "--actor",
        actor,
        "--reason",
        reason,
    ]
    if dry_run:
        cmd.append("--dry-run")
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, check=False)
    out = ((proc.stdout or "") + "\n" + (proc.stderr or "")).strip()
    return int(proc.returncode), out[-4000:]


def main() -> int:
    parser = argparse.ArgumentParser(description="Run rollout guard checks and optional rollback")
    parser.add_argument("--config", type=str, default="backtest_config.yaml")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when rollout status is not PASS")
    args = parser.parse_args()

    cfg_path = (ROOT / args.config).resolve()
    try:
        cfg = _read_config(cfg_path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    rg = ((cfg.get("governance") or {}).get("rollout_guard") or {})
    enabled = bool(rg.get("enabled", False))
    out_dir = ROOT / "output" / "live" / "rollout_guard"
    out_dir.mkdir(parents=True, exist_ok=True)

    now_utc = datetime.now(timezone.utc)
    checks: dict[str, Any] = {}
    failures: list[str] = []
    rollback: dict[str, Any] = {"attempted": False, "executed": False}

    if not enabled:
        payload = {
            "run_at_utc": now_utc.isoformat(),
            "status": "PASS",
            "reason": "guard_disabled",
            "checks": {},
            "failures": [],
            "rollback": rollback,
        }
        latest = out_dir / "rollout_guard_latest.json"
        latest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print("Rollout guard disabled")
        return 0

    require_gov = bool(rg.get("require_governance_pass", True))
    require_split = bool(rg.get("require_split_validation_pass", True))
    require_dual = bool(rg.get("require_dual_model_health_pass", True))

    gov_path = ROOT / "output" / "live" / "governance" / "governance_daily_summary_latest.json"
    split_path = ROOT / "output" / "models" / "split_validation_latest.json"
    dual_path = ROOT / "output" / "live" / "dual_model_health" / "dual_model_health_latest.json"

    if require_gov:
        gov = _read_json(gov_path)
        st = str((gov or {}).get("overall_status", "MISSING") or "MISSING").upper()
        checks["governance_summary"] = {"path": str(gov_path), "status": st}
        if st != "PASS":
            failures.append("governance_summary_not_pass")

    if require_split:
        split = _read_json(split_path)
        st = str((split or {}).get("status", "MISSING") or "MISSING").upper()
        checks["split_validation"] = {"path": str(split_path), "status": st}
        if st != "PASS":
            failures.append("split_validation_not_pass")

    if require_dual:
        dual = _read_json(dual_path)
        st = str((dual or {}).get("status", "MISSING") or "MISSING").upper()
        checks["dual_model_health"] = {"path": str(dual_path), "status": st}
        if st != "PASS":
            failures.append("dual_model_health_not_pass")

    status = "PASS" if not failures else "FAIL"

    if failures and bool(rg.get("auto_halt_on_fail", True)):
        latch_path = set_trading_halt_latch(
            cfg,
            active=True,
            reason=f"rollout_guard failed: {', '.join(failures)}",
            source="scripts.run_rollout_guard",
            details={"failures": failures},
        )
        checks["halt_latch"] = {"path": str(latch_path), "status": "SET"}

    if failures and bool(rg.get("auto_rollback_on_fail", False)):
        rollback["attempted"] = True
        pointer = _read_json(ROOT / "output" / "models" / "production_pointer.json") or {}
        current_run = str(pointer.get("run_id", "") or "").strip()
        prev_run = _latest_previous_production_run_id(
            ROOT / "output" / "models" / "promotion_log.jsonl",
            current_run,
        )
        rollback["current_run_id"] = current_run
        rollback["target_run_id"] = prev_run
        if prev_run:
            rc, tail = _run_rollback(
                to_run_id=prev_run,
                actor=str(rg.get("rollback_actor", "rollout_guard") or "rollout_guard"),
                reason=str(rg.get("rollback_reason", "rollout_guard_gate_breach") or "rollout_guard_gate_breach"),
                dry_run=bool(rg.get("rollback_dry_run", False)),
            )
            rollback["return_code"] = rc
            rollback["output_tail"] = tail
            rollback["executed"] = rc == 0
            if rc != 0:
                failures.append("rollback_failed")
        else:
            rollback["executed"] = False
            rollback["reason"] = "no_previous_production_run"
            failures.append("rollback_target_missing")

    ts = now_utc.strftime("%Y%m%dT%H%M%SZ")
    report_path = out_dir / f"rollout_guard_{ts}.json"
    latest_path = out_dir / "rollout_guard_latest.json"
    payload = {
        "run_at_utc": now_utc.isoformat(),
        "status": "PASS" if not failures else "FAIL",
        "checks": checks,
        "failures": failures,
        "rollback": rollback,
    }
    encoded = json.dumps(payload, indent=2) + "\n"
    report_path.write_text(encoded, encoding="utf-8")
    latest_path.write_text(encoded, encoding="utf-8")

    print(f"Rollout guard report: {report_path}")
    print(json.dumps({"status": payload["status"], "n_failures": len(failures)}, indent=2))

    if args.strict and payload["status"] != "PASS":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
