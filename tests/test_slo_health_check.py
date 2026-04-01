from __future__ import annotations

import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import scripts.slo_health_check as shc


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_config(path: Path) -> None:
    path.write_text(
        """
governance:
  daily_summary:
    output_dir: output/live/governance
  slo:
    enabled: true
    output_dir: output/live/slo
    max_summary_age_hours: 36
    allowed_overall_statuses: [PASS]
    blocked_statuses: [BLOCKED]
    max_governance_failures: 0
    fail_on_missing_summary: true
    fail_on_invalid_summary: true
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _seed_summary(root: Path, *, status: str, run_at: datetime | None = None, failures: list[str] | None = None) -> None:
    now = run_at or datetime.now(timezone.utc)
    payload = {
        "run_at_utc": now.isoformat(),
        "overall_status": status,
        "failures": failures or [],
    }
    _write_json(root / "output" / "live" / "governance" / "governance_daily_summary_latest.json", payload)


def _run(monkeypatch, tmp_path: Path, *extra: str) -> int:
    monkeypatch.setattr(shc, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    argv = ["slo_health_check.py", "--config", "backtest_config.yaml", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return shc.main()


def test_slo_health_check_pass_strict_returns_zero(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml")
    _seed_summary(tmp_path, status="PASS", failures=[])

    rc = _run(monkeypatch, tmp_path, "--strict")
    assert rc == 0

    latest = tmp_path / "output" / "live" / "slo" / "slo_health_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"


def test_slo_health_check_missing_summary_fails_strict(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml")

    rc = _run(monkeypatch, tmp_path, "--strict")
    assert rc == 2

    latest = tmp_path / "output" / "live" / "slo" / "slo_health_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert "summary_missing" in payload["reasons"]


def test_slo_health_check_blocked_is_strict_nonzero(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml")
    _seed_summary(tmp_path, status="BLOCKED", failures=["trading_halt_latch_active"])

    rc = _run(monkeypatch, tmp_path, "--strict")
    assert rc == 2

    latest = tmp_path / "output" / "live" / "slo" / "slo_health_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "BLOCKED"
    assert "governance_blocked" in payload["reasons"]


def test_slo_health_check_stale_summary_fails(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml")
    old = datetime.now(timezone.utc) - timedelta(hours=80)
    _seed_summary(tmp_path, status="PASS", run_at=old)

    rc = _run(monkeypatch, tmp_path, "--strict")
    assert rc == 2

    latest = tmp_path / "output" / "live" / "slo" / "slo_health_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert "summary_stale" in payload["reasons"]


def test_slo_health_check_non_strict_returns_zero(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml")
    _seed_summary(tmp_path, status="FAIL", failures=["risk_gate:status_fail"])

    rc = _run(monkeypatch, tmp_path)
    assert rc == 0

    latest = tmp_path / "output" / "live" / "slo" / "slo_health_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"


def test_slo_health_check_invalid_summary_respects_fail_on_invalid(monkeypatch, tmp_path: Path) -> None:
        cfg_path = tmp_path / "backtest_config.yaml"
        cfg_path.write_text(
                """
governance:
    daily_summary:
        output_dir: output/live/governance
    slo:
        enabled: true
        output_dir: output/live/slo
        max_summary_age_hours: 36
        allowed_overall_statuses: [PASS]
        blocked_statuses: [BLOCKED]
        max_governance_failures: 0
        fail_on_missing_summary: false
        fail_on_invalid_summary: true
""".strip()
                + "\n",
                encoding="utf-8",
        )
        bad_path = tmp_path / "output" / "live" / "governance" / "governance_daily_summary_latest.json"
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        bad_path.write_text("{not-json}\n", encoding="utf-8")

        rc = _run(monkeypatch, tmp_path, "--strict")
        assert rc == 2

        latest = tmp_path / "output" / "live" / "slo" / "slo_health_latest.json"
        payload = json.loads(latest.read_text(encoding="utf-8"))
        assert payload["status"] == "FAIL"
        assert "summary_invalid" in payload["reasons"]
