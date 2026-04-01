from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import scripts.promote_model as pm


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _cfg() -> dict:
    return {
        "model_selection": {
            "promotion": {
                "production_readiness": {
                    "enabled": True,
                    "max_report_age_hours": 36,
                    "require_halt_latch_clear": True,
                    "require_shadow_monitor_pass": True,
                    "require_risk_gate_pass": True,
                    "require_tca_health_pass": True,
                    "require_split_validation_pass": True,
                }
            }
        },
        "live": {
            "trading_halt_latch_path": "output/live/trading_halt_latch.json",
        },
    }


def test_production_readiness_passes_with_fresh_reports(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pm, "ROOT", tmp_path)
    cfg = _cfg()
    now = datetime.now(timezone.utc)

    _write_json(tmp_path / "output" / "live" / "trading_halt_latch.json", {"halt_active": False, "reason": "ok"})
    _write_json(tmp_path / "output" / "models" / "shadow_monitor_latest.json", {"status": "PASS", "run_at_utc": now.isoformat()})
    _write_json(tmp_path / "output" / "live" / "risk_gate" / "risk_gate_latest.json", {"status": "PASS", "run_at_utc": now.isoformat()})
    _write_json(tmp_path / "output" / "live" / "tca" / "tca_health_latest.json", {"status": "PASS", "run_at_utc": now.isoformat()})
    _write_json(tmp_path / "output" / "models" / "split_validation_latest.json", {"status": "PASS", "run_at_utc": now.isoformat()})

    ok, reason, details = pm._check_production_readiness(cfg)
    assert ok is True
    assert reason == ""
    assert details["enabled"] is True


def test_production_readiness_fails_when_halt_active(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pm, "ROOT", tmp_path)
    cfg = _cfg()

    _write_json(tmp_path / "output" / "live" / "trading_halt_latch.json", {"halt_active": True, "reason": "strict preflight failed"})

    ok, reason, _ = pm._check_production_readiness(cfg)
    assert ok is False
    assert reason == "trading_halt_latch_active"


def test_production_readiness_fails_when_report_stale(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pm, "ROOT", tmp_path)
    cfg = _cfg()
    old = datetime.now(timezone.utc) - timedelta(hours=100)

    _write_json(tmp_path / "output" / "live" / "trading_halt_latch.json", {"halt_active": False, "reason": "ok"})
    _write_json(tmp_path / "output" / "models" / "shadow_monitor_latest.json", {"status": "PASS", "run_at_utc": old.isoformat()})
    _write_json(tmp_path / "output" / "live" / "risk_gate" / "risk_gate_latest.json", {"status": "PASS", "run_at_utc": old.isoformat()})
    _write_json(tmp_path / "output" / "live" / "tca" / "tca_health_latest.json", {"status": "PASS", "run_at_utc": old.isoformat()})
    _write_json(tmp_path / "output" / "models" / "split_validation_latest.json", {"status": "PASS", "run_at_utc": old.isoformat()})

    ok, reason, _ = pm._check_production_readiness(cfg)
    assert ok is False
    assert reason in {"shadow_monitor_report_stale", "risk_gate_report_stale", "tca_health_report_stale", "split_validation_report_stale"}


def test_production_readiness_fails_when_split_validation_missing(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pm, "ROOT", tmp_path)
    cfg = _cfg()
    now = datetime.now(timezone.utc)

    _write_json(tmp_path / "output" / "live" / "trading_halt_latch.json", {"halt_active": False, "reason": "ok"})
    _write_json(tmp_path / "output" / "models" / "shadow_monitor_latest.json", {"status": "PASS", "run_at_utc": now.isoformat()})
    _write_json(tmp_path / "output" / "live" / "risk_gate" / "risk_gate_latest.json", {"status": "PASS", "run_at_utc": now.isoformat()})
    _write_json(tmp_path / "output" / "live" / "tca" / "tca_health_latest.json", {"status": "PASS", "run_at_utc": now.isoformat()})

    ok, reason, details = pm._check_production_readiness(cfg)
    assert ok is False
    assert reason == "missing_split_validation_report"
    assert "split_validation" in details


def test_production_readiness_honors_zero_max_age_hours(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(pm, "ROOT", tmp_path)
    cfg = _cfg()
    cfg["model_selection"]["promotion"]["production_readiness"]["max_report_age_hours"] = 0

    old = datetime.now(timezone.utc) - timedelta(seconds=2)
    _write_json(tmp_path / "output" / "live" / "trading_halt_latch.json", {"halt_active": False, "reason": "ok"})
    _write_json(tmp_path / "output" / "models" / "shadow_monitor_latest.json", {"status": "PASS", "run_at_utc": old.isoformat()})
    _write_json(tmp_path / "output" / "live" / "risk_gate" / "risk_gate_latest.json", {"status": "PASS", "run_at_utc": old.isoformat()})
    _write_json(tmp_path / "output" / "live" / "tca" / "tca_health_latest.json", {"status": "PASS", "run_at_utc": old.isoformat()})
    _write_json(tmp_path / "output" / "models" / "split_validation_latest.json", {"status": "PASS", "run_at_utc": old.isoformat()})

    ok, reason, _ = pm._check_production_readiness(cfg)
    assert ok is False
    assert reason.endswith("_report_stale")
