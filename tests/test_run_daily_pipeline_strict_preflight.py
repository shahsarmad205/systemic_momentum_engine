from __future__ import annotations

import io
import json
import subprocess
from pathlib import Path

import run_daily_pipeline as rdp


def test_strict_preflight_success_clears_latch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    models = tmp_path / "output" / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "production_pointer.json").write_text("{}\n", encoding="utf-8")
    target = tmp_path / "output" / "live" / "target_executable_latest.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("ticker,target_weight\nAAPL,0.1\n", encoding="utf-8")

    calls: list[str] = []

    def _ok_run(argv: list[str], *, cwd: Path, logf: io.StringIO, label: str) -> None:
        del argv, cwd, logf
        calls.append(label)

    monkeypatch.setattr(rdp, "run_subprocess", _ok_run)

    cfg = {
        "live": {
            "trading_enabled": True,
            "trading_halt_latch_path": "output/live/gov_halt_latch.json",
        }
    }
    ok = rdp.run_governance_preflight(
        cfg=cfg,
        cfg_path=tmp_path / "cfg.yaml",
        cwd=tmp_path,
        logf=io.StringIO(),
    )

    assert ok is True
    assert calls == [
        "strict_preflight:shadow_monitor",
        "strict_preflight:hard_limit_check",
        "strict_preflight:tca_health_check",
    ]

    latch = tmp_path / "output" / "live" / "gov_halt_latch.json"
    payload = json.loads(latch.read_text(encoding="utf-8"))
    assert payload["halt_active"] is False
    assert payload["source"] == "run_daily_pipeline.strict_preflight"


def test_strict_preflight_failure_sets_latch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    models = tmp_path / "output" / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "production_pointer.json").write_text("{}\n", encoding="utf-8")
    target = tmp_path / "output" / "live" / "target_executable_latest.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("ticker,target_weight\nAAPL,0.1\n", encoding="utf-8")

    def _fail_run(argv: list[str], *, cwd: Path, logf: io.StringIO, label: str) -> None:
        del argv, cwd, logf
        if label == "strict_preflight:hard_limit_check":
            raise subprocess.CalledProcessError(returncode=2, cmd=["python", "check_risk_limits.py"])

    monkeypatch.setattr(rdp, "run_subprocess", _fail_run)

    cfg = {
        "live": {
            "trading_enabled": True,
            "trading_halt_latch_path": "output/live/gov_halt_latch.json",
        }
    }
    ok = rdp.run_governance_preflight(
        cfg=cfg,
        cfg_path=tmp_path / "cfg.yaml",
        cwd=tmp_path,
        logf=io.StringIO(),
    )

    assert ok is False
    latch = tmp_path / "output" / "live" / "gov_halt_latch.json"
    payload = json.loads(latch.read_text(encoding="utf-8"))
    assert payload["halt_active"] is True
    assert "strict preflight failed" in payload["reason"]
    assert payload["source"] == "run_daily_pipeline.strict_preflight"


def test_strict_preflight_failure_writes_incident_artifact(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RUN_ID", "testincident123")

    models = tmp_path / "output" / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "production_pointer.json").write_text("{}\n", encoding="utf-8")
    target = tmp_path / "output" / "live" / "target_executable_latest.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("ticker,target_weight\nAAPL,0.1\n", encoding="utf-8")

    def _fail_run(argv: list[str], *, cwd: Path, logf: io.StringIO, label: str) -> None:
        del argv, cwd, logf, label
        raise RuntimeError("risk check failed")

    monkeypatch.setattr(rdp, "run_subprocess", _fail_run)

    cfg = {
        "live": {
            "trading_enabled": True,
            "trading_halt_latch_path": "output/live/gov_halt_latch.json",
        }
    }
    ok = rdp.run_governance_preflight(
        cfg=cfg,
        cfg_path=tmp_path / "cfg.yaml",
        cwd=tmp_path,
        logf=io.StringIO(),
    )

    assert ok is False
    incident = (
        tmp_path
        / "output"
        / "runs"
        / "testincident123"
        / "strict_preflight_incident.json"
    )
    payload = json.loads(incident.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["run_id"] == "testincident123"
    assert payload["failed_check_label"].startswith("strict_preflight:")
    assert payload["exception_type"] == "RuntimeError"


def test_strict_preflight_incident_write_failure_still_sets_latch(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    models = tmp_path / "output" / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "production_pointer.json").write_text("{}\n", encoding="utf-8")
    target = tmp_path / "output" / "live" / "target_executable_latest.csv"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("ticker,target_weight\nAAPL,0.1\n", encoding="utf-8")

    def _fail_run(argv: list[str], *, cwd: Path, logf: io.StringIO, label: str) -> None:
        del argv, cwd, logf, label
        raise RuntimeError("risk check failed")

    def _fail_incident(**kwargs: object) -> Path:
        del kwargs
        raise OSError("disk full")

    monkeypatch.setattr(rdp, "run_subprocess", _fail_run)
    monkeypatch.setattr(rdp, "_write_preflight_incident_artifact", _fail_incident)

    cfg = {
        "live": {
            "trading_enabled": True,
            "trading_halt_latch_path": "output/live/gov_halt_latch.json",
        }
    }
    ok = rdp.run_governance_preflight(
        cfg=cfg,
        cfg_path=tmp_path / "cfg.yaml",
        cwd=tmp_path,
        logf=io.StringIO(),
    )

    assert ok is False
    latch = tmp_path / "output" / "live" / "gov_halt_latch.json"
    payload = json.loads(latch.read_text(encoding="utf-8"))
    assert payload["halt_active"] is True


def test_strict_preflight_missing_target_fails_closed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)

    models = tmp_path / "output" / "models"
    models.mkdir(parents=True, exist_ok=True)
    (models / "production_pointer.json").write_text("{}\n", encoding="utf-8")

    called = {"count": 0}

    def _noop_run(argv: list[str], *, cwd: Path, logf: io.StringIO, label: str) -> None:
        del argv, cwd, logf, label
        called["count"] += 1

    monkeypatch.setattr(rdp, "run_subprocess", _noop_run)

    cfg = {
        "live": {
            "trading_enabled": True,
            "trading_halt_latch_path": "output/live/gov_halt_latch.json",
        }
    }
    ok = rdp.run_governance_preflight(
        cfg=cfg,
        cfg_path=tmp_path / "cfg.yaml",
        cwd=tmp_path,
        logf=io.StringIO(),
    )

    assert ok is False
    assert called["count"] == 0
    latch = tmp_path / "output" / "live" / "gov_halt_latch.json"
    payload = json.loads(latch.read_text(encoding="utf-8"))
    assert payload["halt_active"] is True
