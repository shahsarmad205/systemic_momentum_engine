from __future__ import annotations

import json
import sys
from pathlib import Path

import scripts.run_rollout_guard as rrg


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _write_cfg(path: Path, *, auto_rollback: bool = False) -> None:
    path.write_text(
        "\n".join(
            [
                "live:",
                "  trading_halt_latch_path: output/live/trading_halt_latch.json",
                "governance:",
                "  rollout_guard:",
                "    enabled: true",
                "    require_governance_pass: true",
                "    require_split_validation_pass: true",
                "    require_dual_model_health_pass: true",
                "    auto_halt_on_fail: true",
                f"    auto_rollback_on_fail: {'true' if auto_rollback else 'false'}",
                "    rollback_dry_run: true",
                "    rollback_actor: rollout_guard",
                "    rollback_reason: rollout_guard_gate_breach",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _seed_pass_reports(root: Path) -> None:
    _write_json(root / "output" / "live" / "governance" / "governance_daily_summary_latest.json", {"overall_status": "PASS"})
    _write_json(root / "output" / "models" / "split_validation_latest.json", {"status": "PASS"})
    _write_json(root / "output" / "live" / "dual_model_health" / "dual_model_health_latest.json", {"status": "PASS"})


def _run(monkeypatch, tmp_path: Path, *extra: str) -> int:
    monkeypatch.setattr(rrg, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    argv = ["run_rollout_guard.py", "--config", "backtest_config.yaml", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return rrg.main()


def test_rollout_guard_pass(monkeypatch, tmp_path: Path) -> None:
    _write_cfg(tmp_path / "backtest_config.yaml")
    _seed_pass_reports(tmp_path)

    rc = _run(monkeypatch, tmp_path, "--strict")
    assert rc == 0

    latest = tmp_path / "output" / "live" / "rollout_guard" / "rollout_guard_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"


def test_rollout_guard_fail_sets_halt_latch(monkeypatch, tmp_path: Path) -> None:
    _write_cfg(tmp_path / "backtest_config.yaml")
    _write_json(tmp_path / "output" / "live" / "governance" / "governance_daily_summary_latest.json", {"overall_status": "FAIL"})

    rc = _run(monkeypatch, tmp_path, "--strict")
    assert rc == 2

    latch = tmp_path / "output" / "live" / "trading_halt_latch.json"
    latch_payload = json.loads(latch.read_text(encoding="utf-8"))
    assert latch_payload["halt_active"] is True

    latest = tmp_path / "output" / "live" / "rollout_guard" / "rollout_guard_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert "governance_summary_not_pass" in payload["failures"]


def test_rollout_guard_attempts_rollback_when_enabled(monkeypatch, tmp_path: Path) -> None:
    _write_cfg(tmp_path / "backtest_config.yaml", auto_rollback=True)
    _write_json(tmp_path / "output" / "live" / "governance" / "governance_daily_summary_latest.json", {"overall_status": "FAIL"})
    _write_json(tmp_path / "output" / "models" / "split_validation_latest.json", {"status": "PASS"})
    _write_json(tmp_path / "output" / "live" / "dual_model_health" / "dual_model_health_latest.json", {"status": "PASS"})
    _write_json(tmp_path / "output" / "models" / "production_pointer.json", {"run_id": "run_current"})

    promo_log = tmp_path / "output" / "models" / "promotion_log.jsonl"
    promo_log.parent.mkdir(parents=True, exist_ok=True)
    promo_log.write_text(
        "\n".join(
            [
                json.dumps({"success": True, "action": "promote", "to_state": "production", "run_id": "run_prev"}),
                json.dumps({"success": True, "action": "promote", "to_state": "production", "run_id": "run_current"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(rrg, "_run_rollback", lambda **kwargs: (0, "ok"))

    rc = _run(monkeypatch, tmp_path, "--strict")
    assert rc == 2

    latest = tmp_path / "output" / "live" / "rollout_guard" / "rollout_guard_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["rollback"]["attempted"] is True
    assert payload["rollback"]["executed"] is True
    assert payload["rollback"]["target_run_id"] == "run_prev"
