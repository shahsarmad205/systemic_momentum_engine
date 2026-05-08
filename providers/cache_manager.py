"""Shared OHLCV cache helpers for provider adapters."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import pandas as pd

DEFAULT_CACHE_DIR = "data/cache"
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def safe_filename(ticker: str) -> str:
    """Sanitize ticker for use in provider cache paths."""
    return re.sub(r"[^\w\-.]", "_", str(ticker).upper())


def cache_path(cache_dir: str, ticker: str, provider: str = "yahoo") -> Path:
    """Return the provider-namespaced per-ticker parquet cache path."""
    prov = str(provider or "").strip().lower()
    if not prov:
        raise ValueError("cache provider namespace must be explicit")
    suffix = "" if prov == "yahoo" else f"_{prov}"
    return Path(cache_dir) / f"{safe_filename(ticker)}{suffix}.parquet"


def load_cached(path: Path, cache_ttl_days: int) -> pd.DataFrame | None:
    """Load a cached OHLCV parquet if present, fresh, and structurally valid."""
    if not path.exists():
        return None
    if cache_ttl_days > 0:
        mtime = path.stat().st_mtime
        if (datetime.now().timestamp() - mtime) > cache_ttl_days * 86400:
            return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if not all(col in df.columns for col in OHLCV_COLUMNS):
        return None
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    return df.sort_index()


def save_cache(path: Path, df: pd.DataFrame) -> None:
    """Persist an OHLCV frame to parquet, preserving provider-specific columns."""
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)


def df_overlaps_window(df: pd.DataFrame | None, start: pd.Timestamp, end: pd.Timestamp) -> bool:
    """True when a frame's DatetimeIndex intersects the requested inclusive window."""
    if df is None or df.empty:
        return False
    mn = pd.Timestamp(df.index.min())
    mx = pd.Timestamp(df.index.max())
    return not (mx < start or mn > end)


def clear_provider_cache(
    *,
    ticker: str | None = None,
    cache_dir: str | None = None,
    provider: str = "yahoo",
) -> None:
    """Clear one provider cache namespace, or one ticker within that namespace."""
    base = Path(cache_dir or DEFAULT_CACHE_DIR)
    if not base.exists():
        return
    if ticker is None:
        pattern = "*.parquet" if provider == "yahoo" else f"*_{provider}.parquet"
        candidates = base.glob(pattern)
    else:
        candidates = [cache_path(str(base), ticker, provider)]
    for path in candidates:
        try:
            path.unlink()
        except OSError:
            pass
