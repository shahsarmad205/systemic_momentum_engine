"""
Data contract helpers: OHLCV cache coverage and freshness vs the last US session.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


def max_ohlcv_cache_bar_date(cache_dir: Path) -> pd.Timestamp | None:
    """Latest *trading* date found across ``*.parquet`` in the OHLCV cache directory."""
    if not cache_dir.is_dir():
        return None
    mx: pd.Timestamp | None = None
    for p in sorted(cache_dir.glob("*.parquet")):
        try:
            df = pd.read_parquet(p)
        except Exception:
            continue
        if df.empty:
            continue
        idx = df.index
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx, errors="coerce")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
        try:
            m = pd.Timestamp(idx.max())
        except Exception:
            continue
        if mx is None or m > mx:
            mx = m
    return mx


def required_latest_cache_date() -> pd.Timestamp:
    """Last completed US business day (relative to *now*)."""
    today = pd.Timestamp.now().normalize()
    return (today - pd.tseries.offsets.BDay(1)).normalize()


def cache_covers_session(mx: pd.Timestamp | None, need: pd.Timestamp) -> bool:
    """True when cache latest bar is on or after ``need`` (same calendar as pipeline checks)."""
    if mx is None:
        return False
    mx_date = mx.normalize()
    return mx_date >= need.normalize()


def ohlcv_cache_dir(cfg: dict[str, Any], root: Path) -> Path:
    bt = cfg.get("backtest", cfg)
    rel = bt.get("cache_dir", "data/cache/ohlcv")
    return (root / rel).resolve()
