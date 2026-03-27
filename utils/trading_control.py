"""
Global live trading gates (kill switch) for execution.

- ``live.trading_enabled`` in YAML (default True)
- Environment variable (default ``TRADING_HALTED``): if set to 1/true/yes/on, live broker orders are blocked.
"""

from __future__ import annotations

import os
from typing import Any


def is_live_trading_allowed(
    config: dict[str, Any],
    *,
    halt_env_var: str = "TRADING_HALTED",
) -> bool:
    live = config.get("live") or {}
    if not isinstance(live, dict):
        live = {}
    if not bool(live.get("trading_enabled", True)):
        return False
    key = str(live.get("trading_halt_env", halt_env_var) or halt_env_var).strip()
    v = (os.environ.get(key) or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return False
    return True


def trading_halt_reason(config: dict[str, Any], *, halt_env_var: str = "TRADING_HALTED") -> str | None:
    if is_live_trading_allowed(config, halt_env_var=halt_env_var):
        return None
    live = config.get("live") or {}
    if not isinstance(live, dict):
        live = {}
    if not bool(live.get("trading_enabled", True)):
        return "live.trading_enabled is false"
    key = str(live.get("trading_halt_env", halt_env_var) or halt_env_var).strip()
    return f"environment {key} is set (trading halt)"
