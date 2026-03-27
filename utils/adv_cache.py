"""
Average daily volume (ADV) in **shares** for liquidity checks.

Reads/writes ``output/adv_cache.csv`` and falls back to OHLCV parquet under
``backtest.cache_dir``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from utils.market_data import _cache_path

logger = logging.getLogger(__name__)

DEFAULT_ADV_SHARES = 1_000_000.0


def load_adv_cache(path: Path) -> dict[str, float]:
    """``{ticker.upper(): adv_shares}`` from CSV (columns ticker/symbol, adv)."""
    if not path.is_file():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty:
        return {}
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("ticker") or cols.get("symbol")
    acol = cols.get("adv")
    if not tcol or not acol:
        return {}
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        t = str(row[tcol]).strip().upper()
        try:
            v = float(row[acol])
        except Exception:
            continue
        if t and v == v and v > 0:
            out[t] = v
    return out


def append_adv_cache(path: Path, ticker: str, adv: float) -> None:
    """Merge one ticker into cache file on disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    m = load_adv_cache(path)
    m[str(ticker).strip().upper()] = float(adv)
    df = pd.DataFrame([{"ticker": k, "adv": v} for k, v in sorted(m.items())])
    try:
        df.to_csv(path, index=False)
    except Exception:
        pass


def mean_volume_from_ohlcv_parquet(
    ticker: str,
    cache_dir: Path,
    lookback: int,
) -> float | None:
    """Mean daily Volume over last ``lookback`` rows from cached OHLCV parquet."""
    p = _cache_path(str(cache_dir), ticker)
    if not p.is_file():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    if df.empty or "Volume" not in df.columns:
        return None
    vol = pd.to_numeric(df["Volume"], errors="coerce").dropna()
    if len(vol) == 0:
        return None
    tail = vol.tail(max(1, int(lookback)))
    m = float(tail.mean())
    return m if m == m and m > 0 else None


def latest_open_from_ohlcv_parquet(ticker: str, cache_dir: Path) -> float | None:
    p = _cache_path(str(cache_dir), ticker)
    if not p.is_file():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    if df.empty or "Open" not in df.columns:
        return None
    op = pd.to_numeric(df["Open"], errors="coerce").dropna()
    if op.empty:
        return None
    last = float(op.iloc[-1])
    return last if last == last and last > 0 else None


def latest_close_from_ohlcv_parquet(ticker: str, cache_dir: Path) -> float | None:
    p = _cache_path(str(cache_dir), ticker)
    if not p.is_file():
        return None
    try:
        df = pd.read_parquet(p)
    except Exception:
        return None
    if df.empty or "Close" not in df.columns:
        return None
    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    if close.empty:
        return None
    last = float(close.iloc[-1])
    return last if last == last and last > 0 else None


def get_adv_shares(
    ticker: str,
    *,
    cache_dir: Path,
    adv_cache_path: Path,
    lookback: int,
    refresh: bool,
    default: float = DEFAULT_ADV_SHARES,
) -> tuple[float, str]:
    """
    Return (adv_shares, source_tag).

    * ``cache`` — from CSV
    * ``parquet`` — computed from OHLCV and written to CSV
    * ``default`` — no data
    """
    t = str(ticker).strip().upper()
    if not refresh:
        m = load_adv_cache(adv_cache_path)
        if t in m:
            return float(m[t]), "cache"

    adv = mean_volume_from_ohlcv_parquet(t, cache_dir, lookback)
    if adv is not None:
        append_adv_cache(adv_cache_path, t, adv)
        return adv, "parquet"

    logger.warning(
        "ADV fallback for %s: using default %.0f shares (no parquet/cache)",
        t,
        default,
    )
    return float(default), "default"
