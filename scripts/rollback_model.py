#!/usr/bin/env python3
"""Rollback production model pointer to an explicit run_id with audit logging."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = ROOT / "output" / "models"
REGISTRY_PATH = MODELS_DIR / "model_registry.json"
POINTER_PATH = MODELS_DIR / "production_pointer.json"
LOG_PATH = MODELS_DIR / "promotion_log.jsonl"


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


def _find_registry_entry(registry: dict[str, Any], run_id: str) -> dict[str, Any] | None:
    entries = registry.get("entries", []) if isinstance(registry, dict) else []
    if not isinstance(entries, list):
        return None
    for entry in entries:
        if str(entry.get("run_id", "")) == run_id:
            return entry
    return None


def _validate_target_entry(entry: dict[str, Any]) -> tuple[bool, str]:
    hashes = entry.get("artifact_hashes", {}) if isinstance(entry, dict) else {}
    if not isinstance(hashes, dict):
        return False, "invalid_artifact_hashes"

    checks = [
        ("best_model_path", "best_model_sha256"),
        ("best_meta_path", "best_meta_sha256"),
    ]
    for path_key, hash_key in checks:
        rel = str(entry.get(path_key, "") or "")
        expected = str(hashes.get(hash_key, "") or "")
        if not rel or not expected:
            return False, f"missing_{path_key}_or_{hash_key}"

        artifact_path = _resolve_models_artifact(rel)
        if artifact_path is None:
            return False, f"unsafe_{path_key}"
        if not artifact_path.exists():
            return False, f"missing_{path_key}"

        got = _sha256_file(artifact_path)
        if got != expected:
            return False, f"sha_mismatch_{path_key}"

    return True, ""


def main() -> int:
    parser = argparse.ArgumentParser(description="Rollback production model pointer to a prior run_id")
    parser.add_argument("--to-run-id", type=str, required=True)
    parser.add_argument("--actor", type=str, default="manual")
    parser.add_argument("--reason", type=str, default="manual_rollback")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target_run_id = str(args.to_run_id or "").strip()
    event_base = {
        "at_utc": _utc_now_iso(),
        "action": "rollback",
        "to_run_id": target_run_id,
        "actor": args.actor,
        "reason": args.reason,
        "dry_run": bool(args.dry_run),
    }

    pointer = _read_json(POINTER_PATH, None)
    if not isinstance(pointer, dict):
        print(f"ERROR: missing or invalid production pointer: {POINTER_PATH}")
        _append_jsonl(LOG_PATH, {**event_base, "success": False, "error": "invalid_production_pointer"})
        return 1

    current_run_id = str(pointer.get("run_id", "") or "")
    if not current_run_id:
        print("ERROR: production pointer missing run_id")
        _append_jsonl(LOG_PATH, {**event_base, "success": False, "error": "pointer_missing_run_id"})
        return 1

    registry = _read_json(REGISTRY_PATH, None)
    if not isinstance(registry, dict):
        print(f"ERROR: missing or invalid registry: {REGISTRY_PATH}")
        _append_jsonl(LOG_PATH, {**event_base, "success": False, "error": "invalid_registry"})
        return 1

    target_entry = _find_registry_entry(registry, target_run_id)
    if target_entry is None:
        print(f"ERROR: target run_id not found in registry: {target_run_id}")
        _append_jsonl(LOG_PATH, {**event_base, "success": False, "error": "target_run_id_not_found"})
        return 1

    target_state = str(target_entry.get("current_state", "") or "")
    if target_state not in {"candidate", "shadow", "production"}:
        print(f"ERROR: target run_id state not rollback-eligible: {target_state}")
        _append_jsonl(
            LOG_PATH,
            {**event_base, "success": False, "error": "target_state_not_eligible", "target_state": target_state},
        )
        return 1

    ok, reason = _validate_target_entry(target_entry)
    if not ok:
        print(f"ERROR: target artifact validation failed: {reason}")
        _append_jsonl(LOG_PATH, {**event_base, "success": False, "error": reason})
        return 1

    target_pointer = {
        "run_id": str(target_entry.get("run_id", "") or ""),
        "model_name": str(target_entry.get("model_name", "") or ""),
        "best_model_path": str(target_entry.get("best_model_path", "") or ""),
        "best_meta_path": str(target_entry.get("best_meta_path", "") or ""),
        "manifest_path": str(target_entry.get("manifest_path", "") or ""),
        "promoted_at_utc": _utc_now_iso(),
        "promoted_by": args.actor,
        "reason": args.reason,
        "config": target_entry.get("config", {}),
        "git": target_entry.get("git", {}),
        "rollback": {
            "from_run_id": current_run_id,
            "to_run_id": target_run_id,
            "at_utc": _utc_now_iso(),
        },
    }

    print("Rollback plan:")
    print(json.dumps({"from_run_id": current_run_id, "to_run_id": target_run_id}, indent=2))

    if args.dry_run:
        print("Dry-run only: no files written")
        return 0

    try:
        _atomic_write_json(POINTER_PATH, target_pointer)
    except Exception as exc:
        print(f"ERROR: failed to update production pointer atomically: {exc}")
        _append_jsonl(
            LOG_PATH,
            {**event_base, "success": False, "error": "pointer_write_failed", "from_run_id": current_run_id},
        )
        return 1

    rollback_event = {
        **event_base,
        "success": True,
        "from_run_id": current_run_id,
        "to_run_id": target_run_id,
        "from_model_name": str(pointer.get("model_name", "") or ""),
        "to_model_name": str(target_entry.get("model_name", "") or ""),
        "artifact_validation": "pass",
    }
    _append_jsonl(LOG_PATH, rollback_event)

    print(f"Rollback complete: production pointer now targets run_id={target_run_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
