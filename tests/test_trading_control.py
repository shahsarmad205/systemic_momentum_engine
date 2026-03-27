from __future__ import annotations

import os

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
