from __future__ import annotations

import hashlib
import json
import pickle
import sys
from datetime import datetime, timezone
from pathlib import Path

import scripts.governance_daily_summary as gds
import scripts.reconcile_model_governance as rmg
import scripts.rollback_model as rb
import scripts.run_shadow_monitor as rsm


class IdentityEstimator:
    def predict(self, x):
        return x[:, 0]


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_config(path: Path) -> None:
    path.write_text(
        """
backtest:
  end_date: "2024-01-01"
model_selection:
  max_positions: 2
  min_positions: 1
  lookahead_horizon_days: 1
  shadow_monitor:
    enabled: true
    lookback_days: 30
    min_score_corr: 0.5
    min_topk_overlap: 0.5
    max_abs_score_delta: 1.0
  promotion:
    production_readiness:
      max_report_age_hours: 36
live:
  trading_halt_latch_path: output/live/trading_halt_latch.json
governance:
  daily_summary:
    output_dir: output/live/governance
tickers:
  - AAA
  - BBB
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _build_feature_df(*args, **kwargs):
    del args, kwargs
    import pandas as pd

    rows = [
        {"date": "2023-12-28", "ticker": "AAA", "forward_return": 0.01, "f1": 0.1},
        {"date": "2023-12-28", "ticker": "BBB", "forward_return": -0.01, "f1": -0.1},
        {"date": "2023-12-29", "ticker": "AAA", "forward_return": 0.02, "f1": 0.2},
        {"date": "2023-12-29", "ticker": "BBB", "forward_return": -0.02, "f1": -0.2},
    ]
    return pd.DataFrame(rows)


def _seed_artifacts(tmp_path: Path) -> tuple[str, Path, Path, Path]:
    models = tmp_path / "output" / "models"
    models.mkdir(parents=True, exist_ok=True)

    model_path = models / "best_model.pkl"
    meta_path = models / "best_model.meta.json"
    report_path = models / "model_comparison.csv"

    art = {
        "estimator": IdentityEstimator(),
        "model_type": "regressor",
        "feature_columns": ["f1"],
        "horizon_days": 1,
    }
    with open(model_path, "wb") as fh:
        pickle.dump(art, fh)
    meta_path.write_text('{"model":"identity"}\n', encoding="utf-8")
    report_path.write_text("a,b\n1,2\n", encoding="utf-8")

    run_id = "20260329T200000Z"
    manifest = {
        "run_id": run_id,
        "selected_model": "IdentityModel",
        "selected_by": "test",
        "git": {"commit": "abc", "dirty": True},
        "config": {"path": "backtest_config.yaml", "sha256": "x"},
        "artifacts": {
            "best_model_path": "output/models/best_model.pkl",
            "best_model_sha256": _sha(model_path),
            "best_meta_path": "output/models/best_model.meta.json",
            "best_meta_sha256": _sha(meta_path),
            "report_path": "output/models/model_comparison.csv",
            "report_sha256": _sha(report_path),
        },
    }
    _write_json(models / "latest_run_manifest.json", manifest)

    registry = {
        "version": 1,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "entries": [
            {
                "run_id": run_id,
                "model_name": "IdentityModel",
                "selected_by": "test",
                "manifest_path": "output/models/latest_run_manifest.json",
                "best_model_path": "output/models/best_model.pkl",
                "best_meta_path": "output/models/best_model.meta.json",
                "artifact_hashes": {
                    "best_model_sha256": manifest["artifacts"]["best_model_sha256"],
                    "best_meta_sha256": manifest["artifacts"]["best_meta_sha256"],
                    "report_sha256": manifest["artifacts"]["report_sha256"],
                },
                "config": manifest["config"],
                "git": manifest["git"],
                "current_state": "shadow",
                "state_history": [{"state": "shadow", "at_utc": datetime.now(timezone.utc).isoformat(), "actor": "test", "reason": "seed"}],
            }
        ],
    }
    _write_json(models / "model_registry.json", registry)

    pointer = {
        "run_id": run_id,
        "model_name": "IdentityModel",
        "best_model_path": "output/models/best_model.pkl",
        "best_meta_path": "output/models/best_model.meta.json",
        "manifest_path": "output/models/latest_run_manifest.json",
        "promoted_at_utc": datetime.now(timezone.utc).isoformat(),
        "promoted_by": "test",
        "reason": "seed",
        "config": manifest["config"],
        "git": manifest["git"],
    }
    _write_json(models / "production_pointer.json", pointer)

    _write_json(tmp_path / "output" / "live" / "risk_gate" / "risk_gate_latest.json", {"status": "PASS", "run_at_utc": datetime.now(timezone.utc).isoformat(), "metrics": {}, "failures": []})
    _write_json(tmp_path / "output" / "live" / "tca" / "tca_health_latest.json", {"status": "PASS", "run_at_utc": datetime.now(timezone.utc).isoformat(), "metrics": {}, "failures": []})
    _write_json(tmp_path / "output" / "live" / "trading_halt_latch.json", {"halt_active": False, "reason": "ok", "updated_at_utc": datetime.now(timezone.utc).isoformat()})

    return run_id, model_path, meta_path, report_path


def test_drift_fail_then_reconcile_and_recover(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml")
    run_id, model_path, meta_path, report_path = _seed_artifacts(tmp_path)

    # Monkeypatch roots for scripts under test.
    for mod in (rb, rmg, gds, rsm):
        monkeypatch.setattr(mod, "ROOT", tmp_path)
    monkeypatch.setattr(rb, "MODELS_DIR", tmp_path / "output" / "models")
    monkeypatch.setattr(rb, "REGISTRY_PATH", tmp_path / "output" / "models" / "model_registry.json")
    monkeypatch.setattr(rb, "POINTER_PATH", tmp_path / "output" / "models" / "production_pointer.json")
    monkeypatch.setattr(rb, "LOG_PATH", tmp_path / "output" / "models" / "promotion_log.jsonl")

    monkeypatch.setattr(rmg, "MODELS_DIR", tmp_path / "output" / "models")
    monkeypatch.setattr(rmg, "REGISTRY_PATH", tmp_path / "output" / "models" / "model_registry.json")
    monkeypatch.setattr(rmg, "POINTER_PATH", tmp_path / "output" / "models" / "production_pointer.json")
    monkeypatch.setattr(rmg, "LOG_PATH", tmp_path / "output" / "models" / "promotion_log.jsonl")

    monkeypatch.chdir(tmp_path)

    # Drift: mutate model artifact after registry hash was recorded.
    drift_art = {
        "estimator": IdentityEstimator(),
        "model_type": "regressor",
        "feature_columns": ["f1"],
        "horizon_days": 1,
        "drift": True,
    }
    with open(model_path, "wb") as fh:
        pickle.dump(drift_art, fh)

    # 1) rollback dry-run fails because hash check fails.
    monkeypatch.setattr(sys, "argv", ["rollback_model.py", "--to-run-id", run_id, "--dry-run"])
    assert rb.main() == 1

    # 2) shadow monitor fails early on artifact hash mismatch.
    monkeypatch.setattr(sys, "argv", ["run_shadow_monitor.py", "--config", "backtest_config.yaml", "--strict"])
    assert rsm.main() == 1

    # 3) strict governance summary fails due missing shadow_monitor_latest report.
    monkeypatch.setattr(sys, "argv", ["governance_daily_summary.py", "--config", "backtest_config.yaml", "--strict"])
    assert gds.main() == 2

    # Refresh manifest hashes to represent newly accepted artifact set.
    manifest_path = tmp_path / "output" / "models" / "latest_run_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["artifacts"]["best_model_sha256"] = _sha(model_path)
    manifest["artifacts"]["best_meta_sha256"] = _sha(meta_path)
    manifest["artifacts"]["report_sha256"] = _sha(report_path)
    _write_json(manifest_path, manifest)

    # Reconcile registry + pointer from validated manifest.
    monkeypatch.setattr(
        sys,
        "argv",
        ["reconcile_model_governance.py", "--manifest", "output/models/latest_run_manifest.json", "--actor", "test"],
    )
    assert rmg.main() == 0

    # rollback now passes in dry-run.
    monkeypatch.setattr(sys, "argv", ["rollback_model.py", "--to-run-id", run_id, "--dry-run"])
    assert rb.main() == 0

    # shadow monitor now passes with lightweight mocked feature matrix.
    monkeypatch.setattr(rsm, "build_feature_matrix", _build_feature_df)
    monkeypatch.setattr(sys, "argv", ["run_shadow_monitor.py", "--config", "backtest_config.yaml", "--strict"])
    assert rsm.main() == 0

    # strict governance summary now passes.
    monkeypatch.setattr(sys, "argv", ["governance_daily_summary.py", "--config", "backtest_config.yaml", "--strict"])
    assert gds.main() == 0


def test_reconcile_fails_closed_on_invalid_registry(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml")
    _seed_artifacts(tmp_path)

    # Corrupt registry and ensure reconcile aborts without destructive rewrite.
    registry_path = tmp_path / "output" / "models" / "model_registry.json"
    registry_path.write_text("{bad json", encoding="utf-8")

    monkeypatch.setattr(rmg, "ROOT", tmp_path)
    monkeypatch.setattr(rmg, "MODELS_DIR", tmp_path / "output" / "models")
    monkeypatch.setattr(rmg, "REGISTRY_PATH", registry_path)
    monkeypatch.setattr(rmg, "POINTER_PATH", tmp_path / "output" / "models" / "production_pointer.json")
    monkeypatch.setattr(rmg, "LOG_PATH", tmp_path / "output" / "models" / "promotion_log.jsonl")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(sys, "argv", ["reconcile_model_governance.py", "--manifest", "output/models/latest_run_manifest.json"])
    assert rmg.main() == 1
    assert registry_path.read_text(encoding="utf-8") == "{bad json"


def test_reconcile_write_failure_restores_previous_files(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml")
    _seed_artifacts(tmp_path)

    monkeypatch.setattr(rmg, "ROOT", tmp_path)
    monkeypatch.setattr(rmg, "MODELS_DIR", tmp_path / "output" / "models")
    monkeypatch.setattr(rmg, "REGISTRY_PATH", tmp_path / "output" / "models" / "model_registry.json")
    monkeypatch.setattr(rmg, "POINTER_PATH", tmp_path / "output" / "models" / "production_pointer.json")
    monkeypatch.setattr(rmg, "LOG_PATH", tmp_path / "output" / "models" / "promotion_log.jsonl")
    monkeypatch.chdir(tmp_path)

    reg_before = rmg.REGISTRY_PATH.read_text(encoding="utf-8")
    ptr_before = rmg.POINTER_PATH.read_text(encoding="utf-8")

    original_atomic = rmg._atomic_write_json

    def _fail_on_pointer(path: Path, payload: dict) -> None:
        if path == rmg.POINTER_PATH:
            raise OSError("simulated pointer write failure")
        original_atomic(path, payload)

    monkeypatch.setattr(rmg, "_atomic_write_json", _fail_on_pointer)

    monkeypatch.setattr(sys, "argv", ["reconcile_model_governance.py", "--manifest", "output/models/latest_run_manifest.json"])
    assert rmg.main() == 1

    assert rmg.REGISTRY_PATH.read_text(encoding="utf-8") == reg_before
    assert rmg.POINTER_PATH.read_text(encoding="utf-8") == ptr_before


def test_reconcile_rejects_manifest_outside_models_dir(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml")
    _seed_artifacts(tmp_path)

    outside_manifest = tmp_path / "outside_manifest.json"
    outside_manifest.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(rmg, "ROOT", tmp_path)
    monkeypatch.setattr(rmg, "MODELS_DIR", tmp_path / "output" / "models")
    monkeypatch.setattr(rmg, "REGISTRY_PATH", tmp_path / "output" / "models" / "model_registry.json")
    monkeypatch.setattr(rmg, "POINTER_PATH", tmp_path / "output" / "models" / "production_pointer.json")
    monkeypatch.setattr(rmg, "LOG_PATH", tmp_path / "output" / "models" / "promotion_log.jsonl")
    monkeypatch.chdir(tmp_path)

    monkeypatch.setattr(sys, "argv", ["reconcile_model_governance.py", "--manifest", str(outside_manifest)])
    assert rmg.main() == 1


def test_reconcile_success_when_log_append_fails(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml")
    run_id, _, _, _ = _seed_artifacts(tmp_path)

    monkeypatch.setattr(rmg, "ROOT", tmp_path)
    monkeypatch.setattr(rmg, "MODELS_DIR", tmp_path / "output" / "models")
    monkeypatch.setattr(rmg, "REGISTRY_PATH", tmp_path / "output" / "models" / "model_registry.json")
    monkeypatch.setattr(rmg, "POINTER_PATH", tmp_path / "output" / "models" / "production_pointer.json")
    monkeypatch.setattr(rmg, "LOG_PATH", tmp_path / "output" / "models" / "promotion_log.jsonl")
    monkeypatch.chdir(tmp_path)

    original_append = rmg._append_jsonl

    def _append_with_success_failure(path: Path, payload: dict) -> None:
        if payload.get("success") is True:
            raise OSError("simulated log fsync failure")
        original_append(path, payload)

    monkeypatch.setattr(rmg, "_append_jsonl", _append_with_success_failure)
    monkeypatch.setattr(sys, "argv", ["reconcile_model_governance.py", "--manifest", "output/models/latest_run_manifest.json"])
    assert rmg.main() == 0

    pointer = json.loads((tmp_path / "output" / "models" / "production_pointer.json").read_text(encoding="utf-8"))
    assert pointer["run_id"] == run_id
