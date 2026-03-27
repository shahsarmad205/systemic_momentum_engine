"""
Compatibility module for backtesting/signal imports.

`trend_signal_engine/main.py` imports `from data_loader import download_stock_data`.
During unit tests, `trend_signal_engine/` is added to `sys.path`, so Python
expects `trend_signal_engine/data_loader.py` to exist.

This wrapper re-exports `download_stock_data` implemented via
`utils.market_data.get_ohlcv`.
"""

from __future__ import annotations

import re

import pandas as pd

from utils.market_data import get_ohlcv


def _parse_period_to_days(period: str) -> int:
    s = str(period).strip().lower()
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([a-z]+)", s)
    if not m:
        raise ValueError(f"Unsupported period format: {period!r}")

    val = float(m.group(1))
    unit = m.group(2)

    if unit in ("d", "day", "days"):
        return int(round(val))
    if unit in ("m", "mo", "mos", "month", "months"):
        return int(round(val * 30.0))
    if unit in ("y", "yr", "yrs", "year", "years"):
        return int(round(val * 365.0))

    raise ValueError(f"Unsupported period unit in: {period!r}")


def download_stock_data(
    ticker: str,
    *,
    period: str = "2y",
    start: str | None = None,
    end: str | None = None,
    provider: str = "yahoo",
    use_cache: bool = True,
):
    end_dt = pd.Timestamp(end) if end is not None else pd.Timestamp.today().normalize()

    if start is None:
        days = _parse_period_to_days(period)
        start_dt = end_dt - pd.Timedelta(days=days)
    else:
        start_dt = pd.Timestamp(start)

    return get_ohlcv(
        ticker,
        start_dt.strftime("%Y-%m-%d"),
        end_dt.strftime("%Y-%m-%d"),
        provider=provider,
        use_cache=use_cache,
    )

