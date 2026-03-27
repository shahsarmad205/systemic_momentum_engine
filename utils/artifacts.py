from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from utils.hashes import sha256_file, sha256_yaml_file


def run_dir(run_id: str, *, root: Path | None = None) -> Path:
    base = (root or Path.cwd()) / "output" / "runs" / str(run_id).strip()
    base.mkdir(parents=True, exist_ok=True)
    return base


def write_run_metadata(run_dir_path: Path, metadata: dict[str, Any]) -> Path:
    p = run_dir_path / "run_metadata.json"
    meta = {"ts_utc": datetime.now(timezone.utc).isoformat(), **metadata}
    p.write_text(json.dumps(meta, indent=2, default=str) + "\n", encoding="utf-8")
    return p


def snapshot_file(src: Path, dst_dir: Path, *, dst_name: str | None = None) -> dict[str, Any]:
    out: dict[str, Any] = {"src": str(src)}
    if not src.exists():
        out["present"] = False
        return out
    dst = dst_dir / (dst_name or src.name)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    out.update({"present": True, "dst": str(dst), "sha256": sha256_file(dst)})
    return out


def snapshot_config(config_path: Path, dst_dir: Path) -> dict[str, Any]:
    out = snapshot_file(config_path, dst_dir, dst_name="backtest_config.yaml")
    if out.get("present"):
        out["config_sha256"] = sha256_yaml_file(config_path)
    return out

