# region imports
from __future__ import annotations
try:
    from AlgorithmImports import *
except ImportError:
    pass
# endregion
"""
Global live trading gates (kill switch) for execution.

- ``live.trading_enabled`` in YAML (default True)
- Environment variable (default ``TRADING_HALTED``): if set to 1/true/yes/on, live broker orders are blocked.
"""


import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _resolve_halt_latch_path(
    config: dict[str, Any],
    *,
    default_path: str = "output/live/trading_halt_latch.json",
) -> Path:
    live = config.get("live") or {}
    if not isinstance(live, dict):
        live = {}
    rel = str(live.get("trading_halt_latch_path", default_path) or default_path).strip()
    return Path(rel)


def _coerce_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in {"1", "true", "yes", "on", "y"}
    return False


def _read_halt_latch_payload(config: dict[str, Any]) -> tuple[bool, str | None]:
    path = _resolve_halt_latch_path(config)
    if not path.is_file():
        return False, None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return True, f"invalid_halt_latch_payload in {path}"
    if not isinstance(raw, dict):
        return True, f"invalid_halt_latch_payload in {path}"

    active = _coerce_bool(
        raw.get("halt_active", raw.get("active", raw.get("halted", raw.get("trading_halted", False))))
    )
    if not active:
        return False, None
    reason = str(raw.get("reason", "") or "").strip()
    if reason:
        return True, reason
    return True, f"halt latch active in {path}"


def set_trading_halt_latch(
    config: dict[str, Any],
    *,
    active: bool,
    reason: str,
    source: str,
    details: dict[str, Any] | None = None,
) -> Path:
    path = _resolve_halt_latch_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "halt_active": bool(active),
        "reason": str(reason),
        "source": str(source),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if details:
        payload["details"] = dict(details)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


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
    latch_active, _ = _read_halt_latch_payload(config)
    if latch_active:
        return False
    return True


def trading_halt_reason(config: dict[str, Any], *, halt_env_var: str = "TRADING_HALTED") -> str | None:
    live = config.get("live") or {}
    if not isinstance(live, dict):
        live = {}
    if not bool(live.get("trading_enabled", True)):
        return "live.trading_enabled is false"
    key = str(live.get("trading_halt_env", halt_env_var) or halt_env_var).strip()
    v = (os.environ.get(key) or "").strip().lower()
    if v in ("1", "true", "yes", "on"):
        return f"environment {key} is set (trading halt)"
    latch_active, latch_reason = _read_halt_latch_payload(config)
    if latch_active:
        return latch_reason or "halt latch active"
    return None
