"""
News scraper for a given ticker.

Fetches headlines with timestamps from:
  - yfinance: Ticker(ticker).news (or get_news)
  - RSS: Reuters (business/markets), Seeking Alpha (ticker-specific)

Returns a list of items with headline and timestamp for the last 7 days.
Used by the "full" signals pipeline (regional_news, global_news, social sentiment).
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Any

import feedparser
import requests
import yfinance as yf

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = 15
USER_AGENT = "TrendSignalEngine/1.0 (News Scraper)"
DAYS_BACK = 7

# Reuters: business/markets RSS (Reuters Best general feed)
REUTERS_RSS = "https://reutersbest.com/feed/"

# Seeking Alpha: ticker-specific combined feed
SEEKING_ALPHA_RSS_TEMPLATE = "https://seekingalpha.com/api/sa/combined/{ticker}.xml"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts_utc(dt: datetime) -> datetime:
    """Ensure datetime is timezone-aware UTC."""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _strip_html(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<[^>]+>", "", text).strip()


def _in_last_7_days(ts: datetime) -> bool:
    cutoff = _utcnow() - timedelta(days=DAYS_BACK)
    return _parse_ts_utc(ts) >= cutoff


# ---------------------------------------------------------------------------
# Source 1: yfinance
# ---------------------------------------------------------------------------

def _fetch_yfinance_news(ticker: str) -> list[dict[str, Any]]:
    """
    Fetch news for ticker via yfinance.Ticker(ticker).news (or get_news).
    Returns list of dicts with keys: headline, timestamp, source, url.
    """
    out: list[dict[str, Any]] = []
    try:
        t = yf.Ticker(ticker)
        # Prefer get_news for larger count; fallback to .news
        raw = getattr(t, "get_news", None)
        if callable(raw):
            items = raw(count=100, tab="news")
        else:
            items = getattr(t, "news", None) or []
        if not isinstance(items, list):
            items = []
    except Exception:
        items = []

    for item in items:
        if not isinstance(item, dict):
            continue
        headline = item.get("title") or item.get("headline") or ""
        if not headline:
            continue
        # providerPublishTime is Unix seconds
        pub_ts = item.get("providerPublishTime")
        if pub_ts is not None:
            try:
                ts = datetime.fromtimestamp(int(pub_ts), tz=timezone.utc)
            except (TypeError, ValueError, OSError):
                ts = _utcnow()
        else:
            ts = _utcnow()
        if not _in_last_7_days(ts):
            continue
        out.append({
            "headline": headline.strip(),
            "timestamp": ts,
            "source": "yfinance",
            "url": item.get("link") or item.get("url") or "",
        })
    return out


# ---------------------------------------------------------------------------
# Source 2: Reuters RSS
# ---------------------------------------------------------------------------

def _fetch_reuters_rss() -> list[dict[str, Any]]:
    """
    Fetch Reuters (business/markets) RSS. Returns list of dicts with
    headline, timestamp, source, url. Not ticker-specific; general finance.
    """
    out: list[dict[str, Any]] = []
    try:
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(REUTERS_RSS, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception:
        return out

    for entry in feed.entries[:50]:
        headline = (entry.get("title") or "").strip()
        if not headline:
            continue
        if entry.get("published_parsed"):
            try:
                # feedparser: time.struct_time in UTC
                ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                ts = _utcnow()
        else:
            ts = _utcnow()
        if not _in_last_7_days(ts):
            continue
        out.append({
            "headline": headline,
            "timestamp": ts,
            "source": "reuters",
            "url": entry.get("link") or "",
        })
    return out


# ---------------------------------------------------------------------------
# Source 3: Seeking Alpha RSS (ticker-specific)
# ---------------------------------------------------------------------------

def _fetch_seeking_alpha_rss(ticker: str) -> list[dict[str, Any]]:
    """
    Fetch Seeking Alpha combined feed for the given ticker.
    Returns list of dicts with headline, timestamp, source, url.
    """
    out: list[dict[str, Any]] = []
    url = SEEKING_ALPHA_RSS_TEMPLATE.format(ticker=ticker.upper())
    try:
        headers = {"User-Agent": USER_AGENT}
        resp = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        feed = feedparser.parse(resp.content)
    except Exception:
        return out

    for entry in feed.entries[:50]:
        headline = (entry.get("title") or "").strip()
        if not headline:
            continue
        if entry.get("published_parsed"):
            try:
                ts = datetime(*entry.published_parsed[:6], tzinfo=timezone.utc)
            except (TypeError, ValueError):
                ts = _utcnow()
        else:
            ts = _utcnow()
        if not _in_last_7_days(ts):
            continue
        out.append({
            "headline": headline,
            "timestamp": ts,
            "source": "seeking_alpha",
            "url": entry.get("link") or "",
        })
    return out


# ---------------------------------------------------------------------------
# Deduplication and public API
# ---------------------------------------------------------------------------

def _dedupe_by_headline(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep first occurrence of each headline (normalised for comparison)."""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for x in items:
        key = (x.get("headline") or "").strip().lower()[:200]
        if not key or key in seen:
            continue
        seen.add(key)
        result.append(x)
    return result


def fetch_news_for_ticker(
    ticker: str,
    *,
    days_back: int = DAYS_BACK,
) -> list[dict[str, Any]]:
    """
    Fetch news for the given ticker from yfinance, Reuters RSS, and Seeking Alpha RSS.

    Returns a list of dicts, each with:
      - headline: str
      - timestamp: datetime (timezone-aware UTC)
      - source: "yfinance" | "reuters" | "seeking_alpha"
      - url: str (optional, may be empty)

    Only items within the last `days_back` days are included. Duplicates by
    headline are removed. Results are sorted by timestamp descending (newest first).
    """
    if not ticker or not str(ticker).strip():
        return []

    ticker = str(ticker).strip().upper()
    cutoff = _utcnow() - timedelta(days=days_back)

    def in_range(item: dict) -> bool:
        ts = item.get("timestamp")
        if ts is None:
            return False
        return _parse_ts_utc(ts) >= cutoff

    all_items: list[dict[str, Any]] = []
    all_items.extend(_fetch_yfinance_news(ticker))
    all_items.extend(_fetch_reuters_rss())
    all_items.extend(_fetch_seeking_alpha_rss(ticker))

    all_items = [x for x in all_items if in_range(x)]
    all_items = _dedupe_by_headline(all_items)
    all_items.sort(key=lambda x: x.get("timestamp") or datetime.min.replace(tzinfo=timezone.utc), reverse=True)

    return all_items
