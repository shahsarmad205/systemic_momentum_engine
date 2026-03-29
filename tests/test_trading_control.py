from __future__ import annotations

import json
import os
from pathlib import Path

from utils.trading_control import is_live_trading_allowed, trading_halt_reason


def test_trading_halt_env_blocks() -> None:
    cfg = {"live": {"trading_enabled": True, "trading_halt_env": "MY_HALT"}}
    os.environ["MY_HALT"] = "1"
    try:
        assert is_live_trading_allowed(cfg, halt_env_var="MY_HALT") is False
        assert "MY_HALT" in (trading_halt_reason(cfg, halt_env_var="MY_HALT") or "")
    finally:
        del os.environ["MY_HALT"]


def test_yaml_disabled() -> None:
    cfg = {"live": {"trading_enabled": False}}
    assert is_live_trading_allowed(cfg) is False
    assert trading_halt_reason(cfg) == "live.trading_enabled is false"


def test_latch_file_active_blocks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = {"live": {"trading_enabled": True, "trading_halt_latch_path": "output/live/halt_latch.json"}}
    latch = tmp_path / "output" / "live" / "halt_latch.json"
    latch.parent.mkdir(parents=True, exist_ok=True)
    latch.write_text(
        json.dumps({"halt_active": True, "reason": "strict preflight failed"}),
        encoding="utf-8",
    )

    assert is_live_trading_allowed(cfg) is False
    assert "strict preflight failed" in (trading_halt_reason(cfg) or "")


def test_latch_file_inactive_allows(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = {"live": {"trading_enabled": True, "trading_halt_latch_path": "output/live/halt_latch.json"}}
    latch = tmp_path / "output" / "live" / "halt_latch.json"
    latch.parent.mkdir(parents=True, exist_ok=True)
    latch.write_text(json.dumps({"halt_active": False}), encoding="utf-8")

    assert is_live_trading_allowed(cfg) is True
    assert trading_halt_reason(cfg) is None


def test_latch_file_invalid_payload_blocks(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    cfg = {"live": {"trading_enabled": True, "trading_halt_latch_path": "output/live/halt_latch.json"}}
    latch = tmp_path / "output" / "live" / "halt_latch.json"
    latch.parent.mkdir(parents=True, exist_ok=True)
    latch.write_text("{invalid json", encoding="utf-8")

    assert is_live_trading_allowed(cfg) is False
    assert "invalid_halt_latch_payload" in (trading_halt_reason(cfg) or "")
