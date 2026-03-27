"""
Run context for production observability: correlation IDs, git SHA, dry-run flags.

Set env ``RUN_ID`` for cron; otherwise a short UUID is generated per process.
"""

from __future__ import annotations

import os
import subprocess
import uuid
from datetime import datetime, timezone
from typing import Any


def get_run_id() -> str:
    rid = (os.environ.get("RUN_ID") or os.environ.get("CORRELATION_ID") or "").strip()
    if rid:
        return rid
    return uuid.uuid4().hex[:12]


def ensure_run_id_in_env() -> str:
    """Ensure ``RUN_ID`` is set for this process and child processes (stable for one run)."""
    rid = (os.environ.get("RUN_ID") or os.environ.get("CORRELATION_ID") or "").strip()
    if rid:
        os.environ.setdefault("RUN_ID", rid)
        return rid
    rid = uuid.uuid4().hex[:12]
    os.environ["RUN_ID"] = rid
    return rid


def get_git_sha_short(fallback: str = "") -> str:
    sha = (os.environ.get("GIT_SHA") or os.environ.get("SOURCE_VERSION") or "").strip()
    if sha:
        return sha[:12]
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
            cwd=os.path.dirname(os.path.dirname(__file__)),
        )
        if out.returncode == 0 and out.stdout.strip():
            return out.stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        pass
    return fallback


def context_dict(*, dry_run: bool | None = None, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    """Structured fields to merge into log messages or JSON logs."""
    ctx: dict[str, Any] = {
        "run_id": get_run_id(),
        "git_sha": get_git_sha_short(),
        "ts_utc": datetime.now(timezone.utc).isoformat(),
    }
    if dry_run is not None:
        ctx["dry_run"] = dry_run
    if extra:
        ctx.update(extra)
    return ctx


def format_ctx(
    prefix: str = "",
    *,
    dry_run: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> str:
    """Human-readable suffix for print/log lines, e.g. ``[run_id=abc git=def]``."""
    c = context_dict(dry_run=dry_run, extra=extra)
    parts = [f"{k}={v}" for k, v in c.items() if k != "ts_utc"]
    inner = " ".join(parts)
    return f"{prefix}[{inner}]" if inner else prefix
