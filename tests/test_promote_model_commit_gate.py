from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import scripts.promote_model as pm


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def test_production_promotion_fails_when_manifest_commit_missing(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path
    models = root / "output" / "models"
    models.mkdir(parents=True, exist_ok=True)

    best_model_rel = "output/models/best_model.pkl"
    best_meta_rel = "output/models/best_model.meta.json"
    report_rel = "output/models/model_comparison.csv"

    (root / best_model_rel).write_bytes(b"model-bytes")
    (root / best_meta_rel).write_text("{}\n", encoding="utf-8")
    (root / report_rel).write_text("col\n1\n", encoding="utf-8")

    run_id = "run_missing_commit"
    manifest_path = models / f"run_manifest_{run_id}.json"
    manifest = {
        "run_id": run_id,
        "selected_model": "xgb",
        "selected_by": "test",
        "artifacts": {
            "best_model_path": best_model_rel,
            "best_model_sha256": _sha(root / best_model_rel),
            "best_meta_path": best_meta_rel,
            "best_meta_sha256": _sha(root / best_meta_rel),
            "report_path": report_rel,
            "report_sha256": _sha(root / report_rel),
        },
        "config": {},
        "git": {},
    }
    _write_json(manifest_path, manifest)

    registry = {
        "version": 1,
        "updated_at_utc": "",
        "entries": [
            {
                "run_id": run_id,
                "model_name": "xgb",
                "selected_by": "test",
                "manifest_path": str(manifest_path.relative_to(root)),
                "best_model_path": best_model_rel,
                "best_meta_path": best_meta_rel,
                "artifact_hashes": {
                    "best_model_sha256": _sha(root / best_model_rel),
                    "best_meta_sha256": _sha(root / best_meta_rel),
                    "report_sha256": _sha(root / report_rel),
                },
                "config": {},
                "git": {},
                "current_state": "shadow",
                "state_history": [],
            }
        ],
    }
    _write_json(models / "model_registry.json", registry)

    cfg = {
        "model_selection": {
            "promotion": {
                "require_validator_pass": False,
                "block_dirty_git_for_production": True,
                "production_readiness": {"enabled": False},
            }
        }
    }
    _write_json(root / "cfg.yaml", cfg)

    monkeypatch.setattr(pm, "ROOT", root)
    monkeypatch.setattr(pm, "MODELS_DIR", models)
    monkeypatch.setattr(pm, "REGISTRY_PATH", models / "model_registry.json")
    monkeypatch.setattr(pm, "POINTER_PATH", models / "production_pointer.json")
    monkeypatch.setattr(pm, "LOG_PATH", models / "promotion_log.jsonl")
    monkeypatch.setattr(pm, "_git_head_and_dirty", lambda _cwd: ("abc123", False))

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_model.py",
            "--config",
            "cfg.yaml",
            "--to-state",
            "production",
            "--run-id",
            run_id,
        ],
    )

    rc = pm.main()
    assert rc == 1

    events = (models / "promotion_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    payload = json.loads(events[-1])
    assert payload.get("error") == "manifest_commit_missing"
