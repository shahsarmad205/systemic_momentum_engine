"""
Market API: overview (top longs/shorts), search by ticker, and price history.
"""

from __future__ import annotations

import pandas as pd
from datetime import datetime, timedelta, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query

from api.services.cache_reader import CacheReader

router = APIRouter()
_cache_reader = CacheReader()


@router.get("/overview")
def get_market_overview():
    """
    Top longs/shorts from cache and market sentiment.
    Returns: top_longs (full signal dicts), top_shorts, market_sentiment, last_updated.
    """
    try:
        data = _cache_reader.load_latest()
    except HTTPException:
        raise
    top_signals = _cache_reader.get_top_signals(5, 5)
    return {
        "top_longs": top_signals["top_longs"],
        "top_shorts": top_signals["top_shorts"],
        "market_sentiment": data.get("market_sentiment", 0.0),
        "last_updated": data.get("generated_at"),
    }


@router.get("/search")
def search_tickers(q: str = Query(..., min_length=1, description="Search query (ticker substring)")):
    """
    Filter tickers from latest cache by query (case insensitive).
    Returns matching tickers with signal_score and direction.
    """
    try:
        data = _cache_reader.load_latest()
    except HTTPException:
        raise
    signals = data.get("signals") or {}
    query = (q or "").strip().lower()
    if not query:
        return []
    out = []
    for ticker, sig in signals.items():
        if query in (ticker or "").lower():
            out.append({
                "ticker": ticker,
                "signal_score": sig.get("signal_score"),
                "direction": sig.get("direction"),
            })
    return sorted(out, key=lambda x: (x["ticker"]))


def _get_loader():
    from data import MarketDataLoader
    from config import load_config
    cfg = load_config("backtest_config.yaml")
    return MarketDataLoader(
        provider=getattr(cfg, "data_provider", "yahoo"),
        cache_dir=getattr(cfg, "cache_dir", "data/cache/ohlcv"),
        use_cache=getattr(cfg, "cache_ohlcv", True),
        cache_ttl_days=int(getattr(cfg, "cache_ttl_days", 0)),
    )


@router.get("/history/{ticker}")
def get_price_history(
    ticker: str,
    start: str | None = Query(None, description="Start date YYYY-MM-DD"),
    end: str | None = Query(None, description="End date YYYY-MM-DD"),
):
    """Return OHLCV history for the ticker. Defaults to last 365 days."""
    now = datetime.now(timezone.utc)
    end_date = (end or now.strftime("%Y-%m-%d"))
    if not start:
        start_date = (now - timedelta(days=365)).strftime("%Y-%m-%d")
    else:
        start_date = start
    try:
        loader = _get_loader()
        df = loader.load_price_history(ticker.upper(), start_date, end_date)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    if df is None or df.empty:
        raise HTTPException(status_code=404, detail=f"No price data for {ticker}")
    # Normalize: flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.reset_index()
    df["date"] = df.iloc[:, 0].astype(str)
    rows = []
    for _, row in df.iterrows():
        rows.append({
            "date": row["date"],
            "open": float(row.get("Open", 0)),
            "high": float(row.get("High", 0)),
            "low": float(row.get("Low", 0)),
            "close": float(row.get("Close", 0)),
            "volume": int(row.get("Volume", 0)),
        })
    return {"ticker": ticker.upper(), "start": start_date, "end": end_date, "rows": rows}
