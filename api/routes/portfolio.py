"""
Portfolio API: top_longs, top_shorts, market_sentiment; POST /portfolio/analyze for holdings.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from api.services.cache_reader import CacheReader
from api.services.explainer import suggest_portfolio

router = APIRouter()
CACHE_DIR = os.environ.get("SIGNALS_CACHE_DIR", "output/cache")
_cache_reader = CacheReader()

# Concentration thresholds for risk_flags
CONC_MAX_SINGLE = 0.25   # single position > 25%
CONC_TOP2 = 0.50         # top 2 > 50%
TECH_HIGH = 0.60         # tech > 60%


class HoldingItem(BaseModel):
    ticker: str
    weight: float


class AnalyzeBody(BaseModel):
    holdings: list[HoldingItem]


def _load_signals_for_date(date: str) -> dict[str, Any]:
    path = os.path.join(CACHE_DIR, f"signals_{date}.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        return json.load(f)


def _get_sector(ticker: str) -> str:
    try:
        from utils.sectors import get_sector as _gs
        return _gs(ticker)
    except Exception:
        return "Other"


@router.post("/analyze")
def post_portfolio_analyze(body: AnalyzeBody):
    """
    Analyze holdings: weighted_avg_signal, tech_exposure, avg_news_sentiment,
    market_beta_estimate, risk_flags, suggestion, holdings_analysis.
    """
    try:
        data = _cache_reader.load_latest()
    except HTTPException:
        raise
    signals_map = data.get("signals") or {}
    holdings = body.holdings or []
    if not holdings:
        return {
            "weighted_avg_signal": 0.0,
            "tech_exposure": 0.0,
            "avg_news_sentiment": 0.0,
            "market_beta_estimate": 1.0,
            "risk_flags": [],
            "suggestion": "Add holdings to analyze.",
            "holdings_analysis": [],
        }
    total_weight = sum(h.weight for h in holdings)
    weights_norm = [h.weight / total_weight if total_weight else 0.0 for h in holdings]
    holdings_analysis = []
    weighted_signal_sum = 0.0
    weighted_news_sum = 0.0
    weighted_beta_sum = 0.0
    tech_count = 0
    for h, w in zip(holdings, weights_norm, strict=False):
        ticker = (h.ticker or "").strip().upper()
        sig = signals_map.get(ticker) or {}
        signal_score = float(sig.get("signal_score", 0))
        direction = sig.get("direction", "Neutral")
        news_sentiment = float(sig.get("news_sentiment", 0))
        composite = float(sig.get("composite_score", 0))
        beta_proxy = 1.0 + composite
        weighted_signal_sum += w * signal_score
        weighted_news_sum += w * news_sentiment
        weighted_beta_sum += w * beta_proxy
        if _get_sector(ticker) == "Technology":
            tech_count += 1
        holdings_analysis.append({
            "ticker": ticker,
            "weight": round(w, 4),
            "signal_score": signal_score,
            "direction": direction,
        })
    tech_exposure = (tech_count / len(holdings)) * 100.0 if holdings else 0.0
    sorted_weights = sorted(weights_norm, reverse=True)
    risk_flags = []
    if sorted_weights and sorted_weights[0] > CONC_MAX_SINGLE:
        risk_flags.append(f"Single position > {CONC_MAX_SINGLE * 100:.0f}%")
    if len(sorted_weights) >= 2 and (sorted_weights[0] + sorted_weights[1]) > CONC_TOP2:
        risk_flags.append(f"Top 2 positions > {CONC_TOP2 * 100:.0f}%")
    if tech_exposure >= TECH_HIGH * 100:
        risk_flags.append(f"High tech concentration ({tech_exposure:.0f}%)")
    holdings_summary = "; ".join(
        f"{a['ticker']} {a['weight']*100:.1f}% {a['direction']}" for a in holdings_analysis[:10]
    )
    if len(holdings_analysis) > 10:
        holdings_summary += " ..."
    suggestion = suggest_portfolio(
        weighted_avg_signal=weighted_signal_sum,
        tech_exposure=tech_exposure,
        avg_news_sentiment=weighted_news_sum,
        market_beta_estimate=weighted_beta_sum,
        risk_flags=risk_flags,
        holdings_summary=holdings_summary or "none",
    )
    return {
        "weighted_avg_signal": round(weighted_signal_sum, 4),
        "tech_exposure": round(tech_exposure, 2),
        "avg_news_sentiment": round(weighted_news_sum, 4),
        "market_beta_estimate": round(weighted_beta_sum, 4),
        "risk_flags": risk_flags,
        "suggestion": suggestion,
        "holdings_analysis": holdings_analysis,
    }


@router.get("")
def get_portfolio(date: str | None = Query(None, description="Date YYYY-MM-DD; default today")):
    """Return top_longs, top_shorts, and market_sentiment for the date."""
    if not date:
        date = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    try:
        data = _load_signals_for_date(date)
    except FileNotFoundError:
        raise HTTPException(
            status_code=404, detail=f"Portfolio/signals not found for {date}. Run daily_runner."
        ) from None
    return {
        "date": data.get("date"),
        "generated_at": data.get("generated_at"),
        "top_longs": data.get("top_longs", []),
        "top_shorts": data.get("top_shorts", []),
        "market_sentiment": data.get("market_sentiment", 0.0),
    }


@router.get("/longs")
def get_top_longs(date: str | None = Query(None, description="Date YYYY-MM-DD; default today")):
    """Return list of top long tickers."""
    resp = get_portfolio(date=date)
    return {"date": resp["date"], "tickers": resp["top_longs"]}


@router.get("/shorts")
def get_top_shorts(date: str | None = Query(None, description="Date YYYY-MM-DD; default today")):
    """Return list of top short tickers."""
    resp = get_portfolio(date=date)
    return {"date": resp["date"], "tickers": resp["top_shorts"]}
