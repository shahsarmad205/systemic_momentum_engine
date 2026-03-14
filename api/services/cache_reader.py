"""
CacheReader — read signals cache from output/cache/signals_YYYY-MM-DD.json.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from fastapi import HTTPException

# Default: project root / output / cache (so it works when run from api/ or project root)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_CACHE = _PROJECT_ROOT / "output" / "cache"
CACHE_DIR = os.environ.get("SIGNALS_CACHE_DIR", str(_DEFAULT_CACHE))
DATE_PATTERN = re.compile(r"signals_(\d{4}-\d{2}-\d{2})\.json")


def _list_cache_dates(cache_dir: str) -> list[str]:
    """Return sorted list of date strings (YYYY-MM-DD) that have a cache file, newest first."""
    if not os.path.isdir(cache_dir):
        return []
    dates = []
    for name in os.listdir(cache_dir):
        m = DATE_PATTERN.fullmatch(name)
        if m:
            dates.append(m.group(1))
    return sorted(dates, reverse=True)


def _read_cache_file(cache_dir: str, date_str: str, detail: str | None = None) -> dict[str, Any]:
    path = os.path.join(cache_dir, f"signals_{date_str}.json")
    if not os.path.isfile(path):
        raise HTTPException(
            status_code=503,
            detail=detail or "Signals not yet generated for today",
        )
    with open(path) as f:
        return json.load(f)


class CacheReader:
    """
    Read daily signals cache from output/cache/signals_YYYY-MM-DD.json.
    Raises HTTPException(503, "Signals not yet generated for today") when no cache exists.
    """

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir or CACHE_DIR

    def list_dates(self) -> list[str]:
        """Return cache dates (YYYY-MM-DD) newest first."""
        return _list_cache_dates(self.cache_dir)

    def load_latest(self) -> dict[str, Any]:
        """Read the most recent signals_{date}.json from the cache dir; return parsed dict."""
        dates = _list_cache_dates(self.cache_dir)
        if not dates:
            raise HTTPException(status_code=503, detail="Signals not yet generated for today")
        return _read_cache_file(self.cache_dir, dates[0], detail="Signals not yet generated for today")

    def load_date(self, date_str: str) -> dict[str, Any]:
        """Read cache for the given date (YYYY-MM-DD). Raises HTTPException 503 if missing."""
        return _read_cache_file(
            self.cache_dir,
            date_str,
            detail=f"Signals not yet generated for {date_str}",
        )

    def get_signal(self, ticker: str) -> dict[str, Any] | None:
        """Load latest cache and return the signal dict for the ticker, or None if not found."""
        data = self.load_latest()
        signals = data.get("signals") or {}
        ticker = ticker.upper()
        return signals.get(ticker)

    def get_top_signals(
        self,
        n_longs: int = 5,
        n_shorts: int = 5,
    ) -> dict[str, list[dict[str, Any]]]:
        """
        Return top long and short signals with full signal data.
        Keys: "top_longs", "top_shorts". Each value is a list of dicts
        with "ticker" plus the full signal payload for that ticker.
        """
        data = self.load_latest()
        signals = data.get("signals") or {}
        top_longs_tickers = (data.get("top_longs") or [])[:n_longs]
        top_shorts_tickers = (data.get("top_shorts") or [])[:n_shorts]

        top_longs = []
        for t in top_longs_tickers:
            sig = signals.get(t.upper() if isinstance(t, str) else t)
            if sig is not None:
                top_longs.append({"ticker": t.upper() if isinstance(t, str) else t, **sig})

        top_shorts = []
        for t in top_shorts_tickers:
            sig = signals.get(t.upper() if isinstance(t, str) else t)
            if sig is not None:
                top_shorts.append({"ticker": t.upper() if isinstance(t, str) else t, **sig})

        return {"top_longs": top_longs, "top_shorts": top_shorts}
