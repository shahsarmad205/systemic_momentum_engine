from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import yaml


def sha256_file(path: str | Path) -> str:
    p = Path(path)
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_yaml_obj(obj: Any) -> str:
    payload = yaml.safe_dump(obj, sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def sha256_yaml_file(path: str | Path) -> str:
    p = Path(path)
    with open(p, encoding="utf-8") as fh:
        obj = yaml.safe_load(fh) or {}
    return sha256_yaml_obj(obj)

