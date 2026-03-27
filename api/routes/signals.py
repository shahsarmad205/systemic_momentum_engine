"""
Signals API: full payload, single-ticker raw, single-ticker with explanation, and history for sparklines.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from api.services.cache_reader import CacheReader
from api.services.explainer import SignalExplainer

router = APIRouter()
CACHE_DIR = os.environ.get("SIGNALS_CACHE_DIR", "output/cache")
_cache_reader = CacheReader()
_explainer = SignalExplainer()


def _load_signals_for_date(date: str) -> dict[str, Any]:
    path = os.path.join(CACHE_DIR, f"signals_{date}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


def _signal_data_for_ticker(signal: dict[str, Any], ticker: str) -> dict[str, Any]:
    """Build dict for SignalExplainer.explain() (ticker + signal fields)."""
    return {"ticker": ticker, **signal}


@router.get("/signal/{ticker}")
def get_signal(ticker: str):
    """
    Load signal from cache; add explanation (from cache or via SignalExplainer).
    Returns ticker, signal_score, direction, confidence, quant_score, news_sentiment,
    composite_score, top_headlines (max 3), bullets, risk, summary, rank, last_updated.
    404 if ticker not in cache.
    """
    ticker = ticker.upper()
    try:
        data = _cache_reader.load_latest()
    except HTTPException:
        raise
    signals_map = data.get("signals") or {}
    if ticker not in signals_map:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not found in cache")
    signal = signals_map[ticker]
    signal_data = _signal_data_for_ticker(signal, ticker)
    explanation = _explainer.explain(signal_data)
    top_headlines = (signal.get("top_headlines") or [])[:3]
    return {
        "ticker": ticker,
        "signal_score": signal.get("signal_score"),
        "direction": signal.get("direction"),
        "confidence": signal.get("confidence"),
        "quant_score": signal.get("quant_score"),
        "news_sentiment": signal.get("news_sentiment"),
        "composite_score": signal.get("composite_score"),
        "top_headlines": top_headlines,
        "bullets": explanation.get("bullets", []),
        "risk": explanation.get("risk", ""),
        "summary": explanation.get("summary", ""),
        "rank": signal.get("rank"),
        "last_updated": data.get("generated_at"),
    }


@router.get("/signal/{ticker}/history")
def get_signal_history(
    ticker: str,
    days: int = Query(7, ge=1, le=90, description="Number of days of history"),
):
    """
    Load last N days of cache files; return list of {date, signal_score, direction, news_sentiment}
    for sparkline charts.
    """
    ticker = ticker.upper()
    dates = _cache_reader.list_dates()
    if not dates:
        raise HTTPException(status_code=503, detail="No signals cache available")
    dates = dates[:days]
    out = []
    for date_str in dates:
        try:
            data = _cache_reader.load_date(date_str)
        except HTTPException:
            continue
        signals_map = data.get("signals") or {}
        if ticker not in signals_map:
            continue
        sig = signals_map[ticker]
        out.append({
            "date": date_str,
            "signal_score": sig.get("signal_score"),
            "direction": sig.get("direction"),
            "news_sentiment": sig.get("news_sentiment"),
        })
    return out


@router.get("")
def get_signals(date: str | None = Query(None, description="Date YYYY-MM-DD; default today")):
    """Return full signals payload for the given date (or today)."""
    if not date:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        return _load_signals_for_date(date)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Signals not found for {date}. Run daily_runner to generate."
        ) from None


@router.get("/{ticker}")
def get_signal_for_ticker(
    ticker: str,
    date: str | None = Query(None, description="Date YYYY-MM-DD; default today"),
):
    """Return raw signal for a single ticker (no explanation)."""
    if not date:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        data = _load_signals_for_date(date)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Signals not found for {date}.") from None
    ticker = ticker.upper()
    signals_map = data.get("signals") or {}
    if ticker not in signals_map:
        raise HTTPException(status_code=404, detail=f"Ticker {ticker} not in signals for {date}.")
    return {
        "date": data.get("date"),
        "generated_at": data.get("generated_at"),
        "ticker": ticker,
        "signal": signals_map[ticker],
    }
