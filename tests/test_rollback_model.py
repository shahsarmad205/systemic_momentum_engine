from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.rollback_model as rm


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _setup_paths(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rm, "ROOT", tmp_path)
    monkeypatch.setattr(rm, "MODELS_DIR", tmp_path / "output" / "models")
    monkeypatch.setattr(rm, "REGISTRY_PATH", tmp_path / "output" / "models" / "model_registry.json")
    monkeypatch.setattr(rm, "POINTER_PATH", tmp_path / "output" / "models" / "production_pointer.json")
    monkeypatch.setattr(rm, "LOG_PATH", tmp_path / "output" / "models" / "promotion_log.jsonl")


def _seed_registry_with_target(tmp_path: Path, *, target_run_id: str = "run_target") -> None:
    model_path = tmp_path / "output" / "models" / "best_model_target.pkl"
    meta_path = tmp_path / "output" / "models" / "best_model_target.meta.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(b"model-bytes")
    meta_path.write_text('{"x":1}\n', encoding="utf-8")

    registry = {
        "version": 1,
        "entries": [
            {
                "run_id": target_run_id,
                "model_name": "TargetModel",
                "best_model_path": "output/models/best_model_target.pkl",
                "best_meta_path": "output/models/best_model_target.meta.json",
                "manifest_path": "output/models/run_manifest_target.json",
                "artifact_hashes": {
                    "best_model_sha256": rm._sha256_file(model_path),
                    "best_meta_sha256": rm._sha256_file(meta_path),
                },
                "config": {"path": "backtest_config.yaml", "sha256": "abc"},
                "git": {"commit": "deadbeef", "dirty": False},
                "current_state": "shadow",
            }
        ],
    }
    _write_json(tmp_path / "output" / "models" / "model_registry.json", registry)


def test_rollback_success_updates_pointer_and_logs(monkeypatch, tmp_path: Path) -> None:
    _setup_paths(monkeypatch, tmp_path)
    _seed_registry_with_target(tmp_path)

    _write_json(
        tmp_path / "output" / "models" / "production_pointer.json",
        {
            "run_id": "run_current",
            "model_name": "CurrentModel",
            "best_model_path": "output/models/current.pkl",
            "best_meta_path": "output/models/current.meta.json",
        },
    )

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "rollback_model.py",
            "--to-run-id",
            "run_target",
            "--actor",
            "ops",
            "--reason",
            "incident",
        ],
    )
    rc = rm.main()
    assert rc == 0

    pointer = json.loads((tmp_path / "output" / "models" / "production_pointer.json").read_text(encoding="utf-8"))
    assert pointer["run_id"] == "run_target"
    assert pointer["rollback"]["from_run_id"] == "run_current"

    lines = (tmp_path / "output" / "models" / "promotion_log.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    event = json.loads(lines[0])
    assert event["action"] == "rollback"
    assert event["success"] is True


def test_rollback_missing_run_id_fails(monkeypatch, tmp_path: Path) -> None:
    _setup_paths(monkeypatch, tmp_path)
    _seed_registry_with_target(tmp_path)
    _write_json(tmp_path / "output" / "models" / "production_pointer.json", {"run_id": "run_current"})

    monkeypatch.setattr(sys, "argv", ["rollback_model.py", "--to-run-id", "run_unknown"])
    rc = rm.main()
    assert rc == 1


def test_rollback_hash_mismatch_fails(monkeypatch, tmp_path: Path) -> None:
    _setup_paths(monkeypatch, tmp_path)
    _seed_registry_with_target(tmp_path)
    _write_json(tmp_path / "output" / "models" / "production_pointer.json", {"run_id": "run_current"})

    registry_path = tmp_path / "output" / "models" / "model_registry.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["entries"][0]["artifact_hashes"]["best_model_sha256"] = "bad"
    _write_json(registry_path, registry)

    monkeypatch.setattr(sys, "argv", ["rollback_model.py", "--to-run-id", "run_target"])
    rc = rm.main()
    assert rc == 1


def test_rollback_dry_run_has_no_side_effects(monkeypatch, tmp_path: Path) -> None:
    _setup_paths(monkeypatch, tmp_path)
    _seed_registry_with_target(tmp_path)
    pointer_path = tmp_path / "output" / "models" / "production_pointer.json"
    original = {
        "run_id": "run_current",
        "model_name": "CurrentModel",
    }
    _write_json(pointer_path, original)

    monkeypatch.setattr(sys, "argv", ["rollback_model.py", "--to-run-id", "run_target", "--dry-run"])
    rc = rm.main()
    assert rc == 0

    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
    assert pointer == original
    assert not (tmp_path / "output" / "models" / "promotion_log.jsonl").exists()
