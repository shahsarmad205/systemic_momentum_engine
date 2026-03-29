from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import scripts.governance_daily_summary as gds


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_config(path: Path) -> None:
    path.write_text(
        """
model_selection:
  promotion:
    production_readiness:
      max_report_age_hours: 36
live:
  trading_halt_latch_path: output/live/trading_halt_latch.json
governance:
  daily_summary:
    output_dir: output/live/governance
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _seed_reports(root: Path, *, status: str = "PASS", run_at: datetime | None = None) -> None:
    now = run_at or datetime.now(timezone.utc)
    payload = {"status": status, "run_at_utc": now.isoformat(), "metrics": {}, "failures": []}
    _write_json(root / "output" / "models" / "shadow_monitor_latest.json", payload)
    _write_json(root / "output" / "live" / "risk_gate" / "risk_gate_latest.json", payload)
    _write_json(root / "output" / "live" / "tca" / "tca_health_latest.json", payload)
    _write_json(root / "output" / "models" / "production_pointer.json", {"run_id": "r1", "model_name": "M1"})
    _write_json(root / "output" / "live" / "trading_halt_latch.json", {"halt_active": False, "reason": "ok"})


def _run(monkeypatch, tmp_path: Path, *extra: str) -> int:
    monkeypatch.setattr(gds, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    argv = ["governance_daily_summary.py", "--config", "backtest_config.yaml", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return gds.main()


def test_governance_summary_pass(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml")
    _seed_reports(tmp_path)

    rc = _run(monkeypatch, tmp_path)
    assert rc == 0

    latest = tmp_path / "output" / "live" / "governance" / "governance_daily_summary_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "PASS"


def test_governance_summary_missing_report_fails(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml")
    _seed_reports(tmp_path)
    (tmp_path / "output" / "live" / "risk_gate" / "risk_gate_latest.json").unlink()

    rc = _run(monkeypatch, tmp_path)
    assert rc == 0

    latest = tmp_path / "output" / "live" / "governance" / "governance_daily_summary_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "FAIL"


def test_governance_summary_stale_report_fails(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml")
    old = datetime.now(timezone.utc) - timedelta(hours=100)
    _seed_reports(tmp_path, run_at=old)

    rc = _run(monkeypatch, tmp_path)
    assert rc == 0

    latest = tmp_path / "output" / "live" / "governance" / "governance_daily_summary_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "FAIL"


def test_governance_summary_halt_active_is_blocked(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml")
    _seed_reports(tmp_path)
    _write_json(
        tmp_path / "output" / "live" / "trading_halt_latch.json",
        {"halt_active": True, "reason": "strict preflight failed"},
    )

    rc = _run(monkeypatch, tmp_path)
    assert rc == 0

    latest = tmp_path / "output" / "live" / "governance" / "governance_daily_summary_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "BLOCKED"


def test_governance_summary_max_age_zero_is_honored(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml")
    slightly_old = datetime.now(timezone.utc) - timedelta(seconds=5)
    _seed_reports(tmp_path, run_at=slightly_old)

    rc = _run(monkeypatch, tmp_path, "--max-age-hours", "0")
    assert rc == 0

    latest = tmp_path / "output" / "live" / "governance" / "governance_daily_summary_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["overall_status"] == "FAIL"
