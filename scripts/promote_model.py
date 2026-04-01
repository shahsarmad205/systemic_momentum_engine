#!/usr/bin/env python3
"""
Promote model artifacts through lifecycle states:
research -> candidate -> shadow -> production.

This script enforces manifest integrity and validator gates before promotion,
and records immutable promotion events.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "output" / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
POINTER_PATH = MODELS_DIR / "production_pointer.json"
LOG_PATH = MODELS_DIR / "promotion_log.jsonl"

ALLOWED_TRANSITIONS: dict[str, set[str]] = {
    "research": {"candidate"},
    "candidate": {"shadow"},
    "shadow": {"production"},
    "production": set(),
}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            chunk = fh.read(8192)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def _log_promotion_event(payload: dict[str, Any]) -> None:
    _append_jsonl(LOG_PATH, payload)


def _resolve_manifest(run_id: str | None, manifest_path: str | None) -> Path:
    if manifest_path:
        path = (ROOT / manifest_path).resolve() if not Path(manifest_path).is_absolute() else Path(manifest_path)
        return path
    if run_id:
        return MODELS_DIR / f"run_manifest_{run_id}.json"
    return MODELS_DIR / "latest_run_manifest.json"


def _resolve_models_artifact(rel_path: str) -> Path | None:
    rel = str(rel_path or "").strip()
    if not rel:
        return None
    raw = Path(rel)
    if raw.is_absolute():
        return None
    try:
        resolved = (ROOT / raw).resolve()
        if not resolved.is_relative_to(MODELS_DIR.resolve()):
            return None
    except Exception:
        return None
    return resolved


def _verify_manifest_artifacts(manifest: dict[str, Any], manifest_file: Path) -> tuple[bool, str, dict[str, Path]]:
    artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
    resolved: dict[str, Path] = {}
    checks = [
        ("best_model_path", "best_model_sha256"),
        ("best_meta_path", "best_meta_sha256"),
        ("report_path", "report_sha256"),
    ]
    for path_key, hash_key in checks:
        rel = str(artifacts.get(path_key, "") or "")
        expected = str(artifacts.get(hash_key, "") or "")
        if not rel or not expected:
            return False, f"manifest missing {path_key}/{hash_key}", {}
        p = _resolve_models_artifact(rel)
        if p is None:
            return False, f"unsafe artifact path in manifest: {rel}", {}
        if not p.exists():
            return False, f"artifact not found: {p}", {}
        got = _sha256_file(p)
        if got != expected:
            return False, f"sha mismatch for {p.name}", {}
        resolved[path_key] = p

    cfg = manifest.get("config", {}) if isinstance(manifest, dict) else {}
    cfg_rel = str(cfg.get("path", "") or "")
    cfg_hash = str(cfg.get("sha256", "") or "")
    if cfg_rel and cfg_hash:
        cfg_path = ROOT / cfg_rel
        if cfg_path.exists() and _sha256_file(cfg_path) != cfg_hash:
            return False, f"config sha mismatch for {cfg_path}", {}

    if not manifest.get("run_id"):
        return False, f"missing run_id in manifest {manifest_file}", {}
    return True, "", resolved


def _run_validator(config_path: str, model_path: str) -> tuple[bool, int, str]:
    cmd = [
        sys.executable,
        str(ROOT / "scripts" / "validate_model.py"),
        "--config",
        config_path,
        "--model-path",
        model_path,
    ]
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
    output = (proc.stdout or "") + ("\n" + proc.stderr if proc.stderr else "")
    return proc.returncode == 0, proc.returncode, output[-4000:]


def _git_head_and_dirty(cwd: Path) -> tuple[str, bool]:
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(cwd), text=True).strip()
    except Exception:
        return "", False
    try:
        dirty_out = subprocess.check_output(["git", "status", "--porcelain"], cwd=str(cwd), text=True)
        dirty = bool(str(dirty_out).strip())
    except Exception:
        dirty = False
    return head, dirty


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


def _resolve_halt_latch_path(cfg: dict[str, Any]) -> Path:
    live = cfg.get("live") or {}
    if not isinstance(live, dict):
        live = {}
    rel = str(live.get("trading_halt_latch_path", "output/live/trading_halt_latch.json") or "").strip()
    p = Path(rel)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def _check_status_report(
    *,
    name: str,
    path: Path,
    max_age_hours: float,
    now_utc: datetime,
) -> tuple[bool, str, dict[str, Any]]:
    if not path.exists():
        return False, f"missing_{name}_report", {"path": str(path), "status": None}
    payload = _read_json(path, None)
    if not isinstance(payload, dict):
        return False, f"invalid_{name}_report_json", {"path": str(path), "status": None}

    status = str(payload.get("status", "")).strip().upper()
    run_at = _parse_iso8601_utc(str(payload.get("run_at_utc", "") or ""))
    details: dict[str, Any] = {
        "path": str(path),
        "status": status,
        "run_at_utc": payload.get("run_at_utc"),
    }
    if status != "PASS":
        return False, f"{name}_status_not_pass", details
    if run_at is None:
        return False, f"{name}_missing_run_at_utc", details

    age_hours = (now_utc - run_at).total_seconds() / 3600.0
    details["age_hours"] = age_hours
    if age_hours < 0:
        return False, f"{name}_run_at_in_future", details
    if age_hours > float(max_age_hours):
        return False, f"{name}_report_stale", details

    return True, "", details


def _check_production_readiness(cfg: dict[str, Any]) -> tuple[bool, str, dict[str, Any]]:
    ms_cfg = (cfg.get("model_selection", {}) or {})
    promo_cfg = (ms_cfg.get("promotion", {}) or {})
    readiness = (promo_cfg.get("production_readiness", {}) or {})
    if not bool(readiness.get("enabled", True)):
        return True, "", {"enabled": False}

    raw_max_age_hours = readiness.get("max_report_age_hours", 36.0)
    max_age_hours = 36.0 if raw_max_age_hours is None else float(raw_max_age_hours)
    now_utc = datetime.now(timezone.utc)
    details: dict[str, Any] = {
        "enabled": True,
        "max_report_age_hours": max_age_hours,
    }

    if bool(readiness.get("require_halt_latch_clear", True)):
        latch_path = _resolve_halt_latch_path(cfg)
        details["halt_latch_path"] = str(latch_path)
        if not latch_path.exists():
            return False, "missing_trading_halt_latch", details
        latch = _read_json(latch_path, None)
        if not isinstance(latch, dict):
            return False, "invalid_trading_halt_latch", details
        halt_active = bool(latch.get("halt_active", False))
        details["halt_latch_active"] = halt_active
        details["halt_latch_reason"] = str(latch.get("reason", "") or "")
        if halt_active:
            return False, "trading_halt_latch_active", details

    if bool(readiness.get("require_shadow_monitor_pass", True)):
        ok, reason, rep_details = _check_status_report(
            name="shadow_monitor",
            path=(ROOT / "output" / "models" / "shadow_monitor_latest.json"),
            max_age_hours=max_age_hours,
            now_utc=now_utc,
        )
        details["shadow_monitor"] = rep_details
        if not ok:
            return False, reason, details

    if bool(readiness.get("require_risk_gate_pass", True)):
        ok, reason, rep_details = _check_status_report(
            name="risk_gate",
            path=(ROOT / "output" / "live" / "risk_gate" / "risk_gate_latest.json"),
            max_age_hours=max_age_hours,
            now_utc=now_utc,
        )
        details["risk_gate"] = rep_details
        if not ok:
            return False, reason, details

    if bool(readiness.get("require_tca_health_pass", True)):
        ok, reason, rep_details = _check_status_report(
            name="tca_health",
            path=(ROOT / "output" / "live" / "tca" / "tca_health_latest.json"),
            max_age_hours=max_age_hours,
            now_utc=now_utc,
        )
        details["tca_health"] = rep_details
        if not ok:
            return False, reason, details

    if bool(readiness.get("require_split_validation_pass", False)):
        ok, reason, rep_details = _check_status_report(
            name="split_validation",
            path=(ROOT / "output" / "models" / "split_validation_latest.json"),
            max_age_hours=max_age_hours,
            now_utc=now_utc,
        )
        details["split_validation"] = rep_details
        if not ok:
            return False, reason, details

    return True, "", details


def main() -> int:
    parser = argparse.ArgumentParser(description="Promote a model through lifecycle states")
    parser.add_argument("--config", type=str, default="backtest_config.yaml")
    parser.add_argument("--to-state", type=str, required=True, choices=["candidate", "shadow", "production"])
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--actor", type=str, default="manual")
    parser.add_argument("--reason", type=str, default="")
    parser.add_argument("--skip-validator", action="store_true")
    args = parser.parse_args()

    cfg = _read_json(Path("/dev/null"), {})
    try:
        import yaml

        with open(ROOT / args.config, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
    except Exception:
        cfg = {}

    ms_cfg = (cfg.get("model_selection", {}) or {})
    promo_cfg = (ms_cfg.get("promotion", {}) or {})
    require_validator_pass = bool(promo_cfg.get("require_validator_pass", True))
    block_dirty_git_for_production = bool(promo_cfg.get("block_dirty_git_for_production", True))

    if require_validator_pass and args.skip_validator:
        print("ERROR: --skip-validator is not allowed when promotion.require_validator_pass=true")
        _log_promotion_event(
            {
                "at_utc": _utc_now_iso(),
                "success": False,
                "action": "promote",
                "to_state": args.to_state,
                "manifest": args.manifest or "latest",
                "error": "skip_validator_not_allowed",
                "actor": args.actor,
                "reason": args.reason,
            }
        )
        return 1

    manifest_file = _resolve_manifest(args.run_id or None, args.manifest or None)
    try:
        if not manifest_file.resolve().is_relative_to(MODELS_DIR.resolve()):
            print(f"ERROR: manifest path must be inside {MODELS_DIR}")
            _log_promotion_event(
                {
                    "at_utc": _utc_now_iso(),
                    "success": False,
                    "action": "promote",
                    "to_state": args.to_state,
                    "manifest": str(manifest_file),
                    "error": "unsafe_manifest_path",
                    "actor": args.actor,
                    "reason": args.reason,
                }
            )
            return 1
    except Exception:
        print(f"ERROR: invalid manifest path: {manifest_file}")
        _log_promotion_event(
            {
                "at_utc": _utc_now_iso(),
                "success": False,
                "action": "promote",
                "to_state": args.to_state,
                "manifest": str(manifest_file),
                "error": "invalid_manifest_path",
                "actor": args.actor,
                "reason": args.reason,
            }
        )
        return 1
    if not manifest_file.exists():
        print(f"ERROR: manifest not found: {manifest_file}")
        _log_promotion_event(
            {
                "at_utc": _utc_now_iso(),
                "success": False,
                "action": "promote",
                "to_state": args.to_state,
                "manifest": str(manifest_file),
                "error": "manifest_not_found",
                "actor": args.actor,
                "reason": args.reason,
            }
        )
        return 1

    manifest = _read_json(manifest_file, {})
    ok, reason, resolved_artifacts = _verify_manifest_artifacts(manifest, manifest_file)
    if not ok:
        print(f"ERROR: manifest integrity check failed: {reason}")
        _log_promotion_event(
            {
                "at_utc": _utc_now_iso(),
                "success": False,
                "action": "promote",
                "to_state": args.to_state,
                "manifest": str(manifest_file),
                "error": reason,
                "actor": args.actor,
                "reason": args.reason,
            },
        )
        return 1

    run_id = str(manifest.get("run_id"))
    selected_model = str(manifest.get("selected_model", ""))
    artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
    best_model_rel = str(artifacts.get("best_model_path", "") or "")
    best_meta_rel = str(artifacts.get("best_meta_path", "") or "")
    best_model_path = resolved_artifacts.get("best_model_path")
    if best_model_path is None:
        print("ERROR: missing resolved best_model_path")
        _log_promotion_event(
            {
                "at_utc": _utc_now_iso(),
                "success": False,
                "action": "promote",
                "to_state": args.to_state,
                "run_id": run_id,
                "manifest": str(manifest_file),
                "error": "missing_resolved_best_model_path",
                "actor": args.actor,
                "reason": args.reason,
            }
        )
        return 1

    registry = _read_json(REGISTRY_PATH, {"version": 1, "updated_at_utc": "", "entries": []})
    entries = registry.get("entries", []) if isinstance(registry, dict) else []
    if not isinstance(entries, list):
        entries = []

    entry: dict[str, Any] | None = None
    for e in entries:
        if str(e.get("run_id", "")) == run_id:
            entry = e
            break

    if entry is None:
        entry = {
            "run_id": run_id,
            "model_name": selected_model,
            "selected_by": str(manifest.get("selected_by", "")),
            "manifest_path": str(manifest_file.relative_to(ROOT)) if manifest_file.is_relative_to(ROOT) else str(manifest_file),
            "best_model_path": best_model_rel,
            "best_meta_path": best_meta_rel,
            "artifact_hashes": {
                "best_model_sha256": str(artifacts.get("best_model_sha256", "")),
                "best_meta_sha256": str(artifacts.get("best_meta_sha256", "")),
                "report_sha256": str(artifacts.get("report_sha256", "")),
            },
            "config": manifest.get("config", {}),
            "git": manifest.get("git", {}),
            "current_state": "research",
            "state_history": [
                {
                    "state": "research",
                    "at_utc": _utc_now_iso(),
                    "actor": "system",
                    "reason": "registered_from_manifest",
                }
            ],
        }
        entries.append(entry)

    current_state = str(entry.get("current_state", "research"))
    target_state = str(args.to_state)

    if current_state == target_state:
        print(f"No-op: run {run_id} already in state '{target_state}'")
        return 0

    allowed = ALLOWED_TRANSITIONS.get(current_state, set())
    if target_state not in allowed:
        print(f"ERROR: invalid transition {current_state} -> {target_state}")
        _log_promotion_event(
            {
                "at_utc": _utc_now_iso(),
                "success": False,
                "action": "promote",
                "run_id": run_id,
                "from_state": current_state,
                "to_state": target_state,
                "manifest": str(manifest_file),
                "error": "invalid_transition",
                "actor": args.actor,
                "reason": args.reason,
            }
        )
        return 1

    if args.to_state == "production" and block_dirty_git_for_production:
        head, dirty = _git_head_and_dirty(ROOT)
        git_meta = manifest.get("git", {}) if isinstance(manifest, dict) else {}
        expected_commit = str(git_meta.get("commit", "") or "").strip()
        if not head:
            print("ERROR: refusing production promotion because live git state is unavailable")
            _log_promotion_event(
                {
                    "at_utc": _utc_now_iso(),
                    "success": False,
                    "action": "promote",
                    "run_id": run_id,
                    "from_state": current_state,
                    "to_state": target_state,
                    "manifest": str(manifest_file),
                    "error": "git_state_unavailable",
                    "actor": args.actor,
                    "reason": args.reason,
                }
            )
            return 1
        if not expected_commit:
            print("ERROR: refusing production promotion because manifest commit is missing")
            _log_promotion_event(
                {
                    "at_utc": _utc_now_iso(),
                    "success": False,
                    "action": "promote",
                    "run_id": run_id,
                    "from_state": current_state,
                    "to_state": target_state,
                    "manifest": str(manifest_file),
                    "error": "manifest_commit_missing",
                    "actor": args.actor,
                    "reason": args.reason,
                }
            )
            return 1
        if dirty:
            print("ERROR: refusing production promotion because manifest git state is dirty")
            _log_promotion_event(
                {
                    "at_utc": _utc_now_iso(),
                    "success": False,
                    "action": "promote",
                    "run_id": run_id,
                    "from_state": current_state,
                    "to_state": target_state,
                    "manifest": str(manifest_file),
                    "error": "dirty_git_blocked_production",
                    "actor": args.actor,
                    "reason": args.reason,
                }
            )
            return 1
        if expected_commit != head:
            print(
                "ERROR: refusing production promotion because manifest commit does not match current HEAD"
            )
            _log_promotion_event(
                {
                    "at_utc": _utc_now_iso(),
                    "success": False,
                    "action": "promote",
                    "run_id": run_id,
                    "from_state": current_state,
                    "to_state": target_state,
                    "manifest": str(manifest_file),
                    "error": "manifest_commit_mismatch",
                    "actor": args.actor,
                    "reason": args.reason,
                    "head": head,
                    "manifest_commit": expected_commit,
                }
            )
            return 1

    if require_validator_pass and not args.skip_validator:
        passed, code, tail = _run_validator(args.config, best_model_rel)
        if not passed:
            print("ERROR: validator failed; promotion blocked")
            print(tail)
            _log_promotion_event(
                {
                    "at_utc": _utc_now_iso(),
                    "success": False,
                    "action": "promote",
                    "to_state": args.to_state,
                    "run_id": run_id,
                    "actor": args.actor,
                    "reason": args.reason,
                    "validator_exit_code": code,
                },
            )
            return 1

    if args.to_state == "production":
        ready_ok, ready_reason, ready_details = _check_production_readiness(cfg)
        if not ready_ok:
            print(f"ERROR: production readiness check failed: {ready_reason}")
            _log_promotion_event(
                {
                    "at_utc": _utc_now_iso(),
                    "success": False,
                    "action": "promote",
                    "to_state": args.to_state,
                    "run_id": run_id,
                    "actor": args.actor,
                    "reason": args.reason,
                    "error": ready_reason,
                    "production_readiness": ready_details,
                },
            )
            return 1

    entry["current_state"] = target_state
    history = entry.get("state_history", [])
    if not isinstance(history, list):
        history = []
    history.append(
        {
            "state": target_state,
            "at_utc": _utc_now_iso(),
            "actor": args.actor,
            "reason": args.reason or "manual_promotion",
        }
    )
    entry["state_history"] = history

    registry["version"] = 1
    registry["updated_at_utc"] = _utc_now_iso()
    registry["entries"] = entries

    previous_pointer = _read_json(POINTER_PATH, {})
    if target_state == "production":
        pointer = {
            "run_id": run_id,
            "model_name": selected_model,
            "best_model_path": best_model_rel,
            "best_meta_path": best_meta_rel,
            "manifest_path": entry.get("manifest_path", ""),
            "promoted_at_utc": _utc_now_iso(),
            "promoted_by": args.actor,
            "reason": args.reason or "manual_promotion",
            "config": manifest.get("config", {}),
            "git": manifest.get("git", {}),
        }
        _atomic_write_json(POINTER_PATH, pointer)
        try:
            _atomic_write_json(REGISTRY_PATH, registry)
        except Exception:
            if previous_pointer:
                _atomic_write_json(POINTER_PATH, previous_pointer)
            elif POINTER_PATH.exists():
                POINTER_PATH.unlink()
            raise
    else:
        _atomic_write_json(REGISTRY_PATH, registry)

    _log_promotion_event(
        {
            "at_utc": _utc_now_iso(),
            "success": True,
            "action": "promote",
            "run_id": run_id,
            "from_state": current_state,
            "to_state": target_state,
            "model_name": selected_model,
            "manifest": str(manifest_file),
            "actor": args.actor,
            "reason": args.reason,
            "validator_required": require_validator_pass and not args.skip_validator,
        },
    )

    print(f"Promoted run {run_id}: {current_state} -> {target_state}")
    print(f"Registry: {REGISTRY_PATH}")
    if target_state == "production":
        print(f"Production pointer: {POINTER_PATH}")
    print(f"Promotion log: {LOG_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
