#!/usr/bin/env python3
"""Reconcile model registry and production pointer from a validated run manifest."""

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


def _read_json_required(path: Path, label: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"{label}_not_found")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise RuntimeError(f"invalid_{label}_json") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"invalid_{label}_json")
    return payload


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


def _restore_file(path: Path, data: bytes | None) -> None:
    if data is None:
        if path.exists():
            path.unlink()
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "wb") as fh:
        fh.write(data)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp_path, path)


def _resolve_models_artifact(rel_path: str) -> Path | None:
    rel = str(rel_path or "").strip()
    if not rel:
        return None
    p = Path(rel)
    if p.is_absolute():
        return None
    try:
        resolved = (ROOT / p).resolve()
        if not resolved.is_relative_to(MODELS_DIR.resolve()):
            return None
    except Exception:
        return None
    return resolved


def _resolve_manifest(run_id: str | None, manifest_path: str | None) -> Path:
    if manifest_path:
        p = Path(manifest_path)
        return (ROOT / p).resolve() if not p.is_absolute() else p
    if run_id:
        return MODELS_DIR / f"run_manifest_{run_id}.json"
    return MODELS_DIR / "latest_run_manifest.json"


def _verify_manifest(manifest: dict[str, Any], manifest_path: Path) -> tuple[bool, str]:
    if not isinstance(manifest, dict):
        return False, "invalid_manifest_json"
    if not str(manifest.get("run_id", "") or ""):
        return False, "manifest_missing_run_id"

    artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
    checks = [
        ("best_model_path", "best_model_sha256"),
        ("best_meta_path", "best_meta_sha256"),
        ("report_path", "report_sha256"),
    ]
    for path_key, hash_key in checks:
        rel = str(artifacts.get(path_key, "") or "")
        expected = str(artifacts.get(hash_key, "") or "")
        if not rel or not expected:
            return False, f"manifest_missing_{path_key}_or_{hash_key}"
        fp = _resolve_models_artifact(rel)
        if fp is None:
            return False, f"unsafe_{path_key}"
        if not fp.exists():
            return False, f"missing_{path_key}"
        got = _sha256_file(fp)
        if got != expected:
            return False, f"sha_mismatch_{path_key}"

    try:
        if not manifest_path.resolve().is_relative_to(MODELS_DIR.resolve()):
            return False, "unsafe_manifest_path"
    except Exception:
        return False, "invalid_manifest_path"

    return True, ""


def _build_entry_from_manifest(manifest: dict[str, Any], manifest_path: Path) -> dict[str, Any]:
    artifacts = manifest.get("artifacts", {}) if isinstance(manifest, dict) else {}
    return {
        "run_id": str(manifest.get("run_id", "") or ""),
        "model_name": str(manifest.get("selected_model", "") or ""),
        "selected_by": str(manifest.get("selected_by", "") or ""),
        "manifest_path": str(manifest_path.relative_to(ROOT)) if manifest_path.is_relative_to(ROOT) else str(manifest_path),
        "best_model_path": str(artifacts.get("best_model_path", "") or ""),
        "best_meta_path": str(artifacts.get("best_meta_path", "") or ""),
        "artifact_hashes": {
            "best_model_sha256": str(artifacts.get("best_model_sha256", "") or ""),
            "best_meta_sha256": str(artifacts.get("best_meta_sha256", "") or ""),
            "report_sha256": str(artifacts.get("report_sha256", "") or ""),
        },
        "config": manifest.get("config", {}),
        "git": manifest.get("git", {}),
    }


def _reconcile_registry(manifest_entry: dict[str, Any], actor: str, reason: str) -> dict[str, Any]:
    if REGISTRY_PATH.exists():
        registry = _read_json_required(REGISTRY_PATH, "registry")
    else:
        registry = {"version": 1, "updated_at_utc": "", "entries": []}
    entries = registry.get("entries", []) if isinstance(registry, dict) else []
    if not isinstance(entries, list):
        raise RuntimeError("invalid_registry_entries")

    run_id = str(manifest_entry.get("run_id", "") or "")
    existing: dict[str, Any] | None = None
    for e in entries:
        if str(e.get("run_id", "") or "") == run_id:
            existing = e
            break

    if existing is None:
        current_state = "research"
        state_history: list[dict[str, Any]] = [
            {
                "state": "research",
                "at_utc": _utc_now_iso(),
                "actor": actor,
                "reason": "reconcile_registered_from_manifest",
            }
        ]
        existing = {
            **manifest_entry,
            "current_state": current_state,
            "state_history": state_history,
        }
        entries.append(existing)
    else:
        current_state = str(existing.get("current_state", "research") or "research")
        state_history = existing.get("state_history", [])
        if not isinstance(state_history, list):
            state_history = []
        state_history.append(
            {
                "state": current_state,
                "at_utc": _utc_now_iso(),
                "actor": actor,
                "reason": reason,
            }
        )
        existing.update(manifest_entry)
        existing["current_state"] = current_state
        existing["state_history"] = state_history

    registry["version"] = 1
    registry["updated_at_utc"] = _utc_now_iso()
    registry["entries"] = entries
    return registry


def _build_pointer(manifest_entry: dict[str, Any], actor: str, reason: str) -> dict[str, Any]:
    return {
        "run_id": str(manifest_entry.get("run_id", "") or ""),
        "model_name": str(manifest_entry.get("model_name", "") or ""),
        "best_model_path": str(manifest_entry.get("best_model_path", "") or ""),
        "best_meta_path": str(manifest_entry.get("best_meta_path", "") or ""),
        "manifest_path": str(manifest_entry.get("manifest_path", "") or ""),
        "promoted_at_utc": _utc_now_iso(),
        "promoted_by": actor,
        "reason": reason,
        "config": manifest_entry.get("config", {}),
        "git": manifest_entry.get("git", {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile model registry/pointer from manifest")
    parser.add_argument("--run-id", type=str, default="")
    parser.add_argument("--manifest", type=str, default="")
    parser.add_argument("--actor", type=str, default="manual")
    parser.add_argument("--reason", type=str, default="reconcile_registry_pointer")
    parser.add_argument("--skip-pointer-update", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    manifest_path = _resolve_manifest(args.run_id or None, args.manifest or None)
    event = {
        "at_utc": _utc_now_iso(),
        "action": "reconcile",
        "actor": args.actor,
        "reason": args.reason,
        "manifest": str(manifest_path),
        "dry_run": bool(args.dry_run),
    }

    try:
        if not manifest_path.resolve().is_relative_to(MODELS_DIR.resolve()):
            print(f"ERROR: manifest path must be inside {MODELS_DIR}")
            _append_jsonl(LOG_PATH, {**event, "success": False, "error": "unsafe_manifest_path"})
            return 1
    except Exception:
        print(f"ERROR: invalid manifest path: {manifest_path}")
        _append_jsonl(LOG_PATH, {**event, "success": False, "error": "invalid_manifest_path"})
        return 1

    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}")
        _append_jsonl(LOG_PATH, {**event, "success": False, "error": "manifest_not_found"})
        return 1

    manifest = _read_json(manifest_path, None)
    ok, err = _verify_manifest(manifest, manifest_path)
    if not ok:
        print(f"ERROR: manifest integrity verification failed: {err}")
        _append_jsonl(LOG_PATH, {**event, "success": False, "error": err})
        return 1

    try:
        entry = _build_entry_from_manifest(manifest, manifest_path)
        registry = _reconcile_registry(entry, args.actor, args.reason)
        pointer = _build_pointer(entry, args.actor, args.reason)
    except RuntimeError as exc:
        print(f"ERROR: reconcile preparation failed: {exc}")
        _append_jsonl(LOG_PATH, {**event, "success": False, "error": str(exc)})
        return 1

    print("Reconcile plan:")
    print(json.dumps({"run_id": entry.get("run_id"), "model_name": entry.get("model_name")}, indent=2))

    if args.dry_run:
        print("Dry-run only: no files written")
        return 0

    prev_registry = REGISTRY_PATH.read_bytes() if REGISTRY_PATH.exists() else None
    prev_pointer = POINTER_PATH.read_bytes() if POINTER_PATH.exists() else None
    try:
        _atomic_write_json(REGISTRY_PATH, registry)
        if not args.skip_pointer_update:
            _atomic_write_json(POINTER_PATH, pointer)
    except Exception as exc:
        # Compensation to avoid partially-updated governance state.
        try:
            _restore_file(REGISTRY_PATH, prev_registry)
            if not args.skip_pointer_update:
                _restore_file(POINTER_PATH, prev_pointer)
        except Exception:
            pass
        print(f"ERROR: reconcile write failed: {exc}")
        _append_jsonl(LOG_PATH, {**event, "success": False, "error": "reconcile_write_failed"})
        return 1

    try:
        _append_jsonl(
            LOG_PATH,
            {
                **event,
                "success": True,
                "run_id": entry.get("run_id"),
                "model_name": entry.get("model_name"),
                "pointer_updated": not bool(args.skip_pointer_update),
            },
        )
    except Exception as exc:
        # Governance files are already committed at this point; do not fail ambiguous.
        print(f"WARNING: reconcile committed but audit log append failed: {exc}")

    print(f"Reconciled registry at {REGISTRY_PATH}")
    if not args.skip_pointer_update:
        print(f"Reconciled pointer at {POINTER_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
