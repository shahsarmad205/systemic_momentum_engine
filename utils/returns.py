"""Load aligned daily simple returns from local OHLCV parquet cache."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.market_data import _cache_path


def _close_returns_series(cache_dir: Path, ticker: str) -> pd.Series | None:
    path = _cache_path(str(cache_dir), ticker)
    if not path.is_file():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df.empty or "Close" not in df.columns:
        return None
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, errors="coerce")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
    close = pd.to_numeric(df["Close"], errors="coerce")
    close.index = idx
    ret = close.sort_index().pct_change()
    return ret.dropna()


def load_aligned_returns(
    tickers: list[str],
    cache_dir: Path,
    lookback: int,
    *,
    end_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Build a DataFrame of daily simple returns (rows=dates), inner-joined across tickers.

    Returns the last ``lookback`` rows (after dropna join). Tickers with no file or
    insufficient history are omitted.
    """
    lookback = int(max(5, lookback))
    series_map: dict[str, pd.Series] = {}
    for raw in tickers:
        t = str(raw).strip()
        if not t:
            continue
        s = _close_returns_series(cache_dir, t)
        if s is None or len(s) < lookback:
            continue
        series_map[t.upper()] = s

    if not series_map:
        return pd.DataFrame(), []

    df = pd.DataFrame(series_map).dropna(how="any")
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        df = df.loc[df.index <= end_ts]
    if len(df) < lookback:
        return pd.DataFrame(), list(series_map.keys())
    df = df.iloc[-lookback:]
    return df, list(df.columns)
