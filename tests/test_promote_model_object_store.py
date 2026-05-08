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


def _write_manifest_fixture(root: Path, run_id: str) -> Path:
    models = root / "output" / "models"
    models.mkdir(parents=True, exist_ok=True)

    best_model_rel = "output/models/best_model.pkl"
    best_meta_rel = "output/models/best_model.meta.json"
    report_rel = "output/models/model_comparison.csv"
    manifest_rel = f"output/models/run_manifest_{run_id}.json"

    (root / best_model_rel).write_bytes(b"model-bytes")
    (root / best_meta_rel).write_text('{"kind": "meta"}\n', encoding="utf-8")
    (root / report_rel).write_text("model,score\nxgb,1.0\n", encoding="utf-8")

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
        "git": {"commit": "abc123", "dirty": False},
    }
    _write_json(root / manifest_rel, manifest)
    return root / manifest_rel


def test_candidate_promotion_publishes_artifacts_to_file_object_store(
    monkeypatch,
    tmp_path: Path,
) -> None:
    root = tmp_path
    models = root / "output" / "models"
    run_id = "run_object_store"
    _write_manifest_fixture(root, run_id)

    object_store_root = root / "registry"
    cfg = {
        "model_selection": {
            "promotion": {
                "require_validator_pass": False,
                "object_store": {
                    "enabled": True,
                    "uri": object_store_root.as_uri(),
                    "family": "ensemble",
                    "version": "v-test",
                },
            }
        }
    }
    _write_json(root / "cfg.yaml", cfg)

    monkeypatch.setattr(pm, "ROOT", root)
    monkeypatch.setattr(pm, "MODELS_DIR", models)
    monkeypatch.setattr(pm, "REGISTRY_PATH", models / "model_registry.json")
    monkeypatch.setattr(pm, "POINTER_PATH", models / "production_pointer.json")
    monkeypatch.setattr(pm, "LOG_PATH", models / "promotion_log.jsonl")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_model.py",
            "--config",
            "cfg.yaml",
            "--to-state",
            "candidate",
            "--run-id",
            run_id,
        ],
    )

    rc = pm.main()

    assert rc == 0
    prefix = object_store_root / "ensemble" / "v-test"
    assert (prefix / "model.pkl").read_bytes() == b"model-bytes"
    assert (prefix / "model_metadata.json").exists()
    assert (prefix / "manifest.json").exists()
    assert (prefix / "model_comparison.csv").exists()

    local_metadata = models / "artifact_registry" / f"{run_id}_candidate_artifact_metadata.json"
    metadata = json.loads(local_metadata.read_text(encoding="utf-8"))
    assert metadata["prefix_uri"] == prefix.as_uri()
    assert metadata["objects"]["model"]["sha256"] == _sha(prefix / "model.pkl")
    assert metadata["objects"]["artifact_metadata"]["uri"] == (prefix / "artifact_metadata.json").as_uri()

    registry = json.loads((models / "model_registry.json").read_text(encoding="utf-8"))
    entry = registry["entries"][0]
    assert entry["current_state"] == "candidate"
    assert entry["object_store"]["artifact_version"] == "v-test"


def test_object_store_enabled_without_uri_fails_fast(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path
    models = root / "output" / "models"
    run_id = "run_missing_registry"
    _write_manifest_fixture(root, run_id)

    cfg = {
        "model_selection": {
            "promotion": {
                "require_validator_pass": False,
                "object_store": {"enabled": True, "uri_env": "MISSING_MODEL_REGISTRY_URI"},
            }
        }
    }
    _write_json(root / "cfg.yaml", cfg)

    monkeypatch.setattr(pm, "ROOT", root)
    monkeypatch.setattr(pm, "MODELS_DIR", models)
    monkeypatch.setattr(pm, "REGISTRY_PATH", models / "model_registry.json")
    monkeypatch.setattr(pm, "POINTER_PATH", models / "production_pointer.json")
    monkeypatch.setattr(pm, "LOG_PATH", models / "promotion_log.jsonl")
    monkeypatch.delenv("MISSING_MODEL_REGISTRY_URI", raising=False)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "promote_model.py",
            "--config",
            "cfg.yaml",
            "--to-state",
            "candidate",
            "--run-id",
            run_id,
        ],
    )

    rc = pm.main()

    assert rc == 1
    events = (models / "promotion_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert json.loads(events[-1])["error"] == "object_store_uri_missing"
