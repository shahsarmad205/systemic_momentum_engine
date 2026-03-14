"""
Global News Agent — Live API Version
======================================
Pulls recent macroeconomic and market-wide news from live free APIs
and produces a sentiment score using VADER.

Data sources (tried in order, failures are silently skipped):

    1. GDELT Project         — free, no key required
    2. CNBC Markets RSS      — free, no key required
    3. Finnhub General News  — requires FINNHUB_KEY env var
    4. NewsAPI.org Headlines  — requires NEWSAPI_KEY env var

At minimum, GDELT + CNBC RSS work without any API keys.
Set environment variables for richer coverage:

    export FINNHUB_KEY="your_finnhub_api_key"
    export NEWSAPI_KEY="your_newsapi_api_key"
"""

import os
import re
from datetime import datetime, timedelta

import numpy as np
import requests
import feedparser
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

from utils.sentiment_decay import apply_time_decay, compute_hours_since, NEWS_DECAY_RATE


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

FINNHUB_KEY = os.environ.get("FINNHUB_KEY", "")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")

REQUEST_TIMEOUT = 10
USER_AGENT = "TrendSignalEngine/1.0"

RECENCY_DECAY = 0.1

_analyser = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def strip_html_tags(text: str) -> str:
    """Remove HTML markup, leaving only plain text."""
    clean = re.sub(r"<[^>]+>", "", text)
    return clean.strip()


# ---------------------------------------------------------------------------
# Source 1: GDELT Project (free, no key)
# ---------------------------------------------------------------------------

def fetch_gdelt_global_news() -> list[dict]:
    """
    Fetch recent macro/market headlines from the GDELT DOC 2.0 API.

    GDELT is completely free and requires no API key.
    The query targets economy, markets, trade, and central-bank topics.
    """
    query = "economy OR markets OR trade OR inflation OR interest rates OR federal reserve"

    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": query,
        "mode": "artlist",
        "maxrecords": 30,
        "timespan": "14d",
        "format": "json",
        "sourcelang": "english",
    }

    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    articles = []
    for item in data.get("articles", []):
        # GDELT seendate format: "YYYYMMDDTHHMMSSZ"
        date_str = item.get("seendate", "")
        try:
            published = datetime.strptime(date_str, "%Y%m%dT%H%M%SZ")
        except (ValueError, TypeError):
            published = datetime.now()

        # GDELT articles only have titles, no descriptions
        articles.append({
            "headline": item.get("title", ""),
            "description": "",
            "published_date": published,
            "source": "gdelt",
        })

    return articles


# ---------------------------------------------------------------------------
# Source 2: CNBC Markets RSS (free, no key)
# ---------------------------------------------------------------------------

def fetch_cnbc_markets_rss() -> list[dict]:
    """
    Fetch the CNBC Markets RSS feed.

    Provides broad financial and market news headlines.
    """
    url = "https://www.cnbc.com/id/20910258/device/rss/rss.html"

    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    feed = feedparser.parse(response.content)

    articles = []
    for entry in feed.entries[:20]:
        if entry.get("published_parsed"):
            published = datetime(*entry.published_parsed[:6])
        else:
            published = datetime.now()

        headline = entry.get("title", "")
        description = strip_html_tags(entry.get("summary", ""))

        articles.append({
            "headline": headline,
            "description": description,
            "published_date": published,
            "source": "cnbc_rss",
        })

    return articles


# ---------------------------------------------------------------------------
# Source 3: Finnhub General News
# ---------------------------------------------------------------------------

def fetch_finnhub_general_news() -> list[dict]:
    """
    Fetch general market news from Finnhub.

    Skipped entirely if FINNHUB_KEY is not set.
    """
    if not FINNHUB_KEY:
        return []

    url = "https://finnhub.io/api/v1/news"
    params = {
        "category": "general",
        "token": FINNHUB_KEY,
    }

    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    raw_articles = response.json()

    articles = []
    for item in raw_articles[:20]:
        unix_ts = item.get("datetime", 0)
        published = datetime.fromtimestamp(unix_ts) if unix_ts else datetime.now()

        articles.append({
            "headline": item.get("headline", ""),
            "description": item.get("summary", ""),
            "published_date": published,
            "source": "finnhub",
        })

    return articles


# ---------------------------------------------------------------------------
# Source 4: NewsAPI.org Top Business Headlines
# ---------------------------------------------------------------------------

def fetch_newsapi_global_headlines() -> list[dict]:
    """
    Fetch top business headlines from NewsAPI.

    Skipped entirely if NEWSAPI_KEY is not set.
    Free tier: 100 requests/day.
    """
    if not NEWSAPI_KEY:
        return []

    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "category": "business",
        "language": "en",
        "pageSize": 20,
        "apiKey": NEWSAPI_KEY,
    }

    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    articles = []
    for item in data.get("articles", []):
        date_str = item.get("publishedAt", "")
        try:
            published = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            published = datetime.now()

        articles.append({
            "headline": item.get("title", "") or "",
            "description": item.get("description", "") or "",
            "published_date": published,
            "source": "newsapi",
        })

    return articles


# ---------------------------------------------------------------------------
# Sentiment scoring (VADER)
# ---------------------------------------------------------------------------

def score_sentiment(headline: str, description: str) -> float:
    """
    Run VADER on the combined headline + description text.

    Returns the compound score in the range [-1, +1].
    """
    text = f"{headline}. {description}"
    scores = _analyser.polarity_scores(text)
    return scores["compound"]


# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------

def compute_recency_weight(article_date: datetime, reference_date: datetime) -> float:
    """
    Exponential decay weight based on how many days old the article is.

    weight = exp(-decay * days_old)
    """
    days_old = (reference_date - article_date).days

    if days_old < 0:
        days_old = 0

    weight = np.exp(-RECENCY_DECAY * days_old)
    return float(weight)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_global_news_model() -> dict:
    """
    Full pipeline: fetch from live APIs → score → weight → aggregate.

    Tries each source independently and combines all articles.
    If every source fails, returns neutral.

    This is called once per batch run (global news is the same for all stocks).

    Returns:
        global_sentiment_score : float in [-1, +1]
        impact_factor          : float in [0, 1]
    """
    all_articles = []

    # --- Collect articles from every available source ---

    # Source 1: GDELT (always available, no key)
    try:
        gdelt_articles = fetch_gdelt_global_news()
        all_articles.extend(gdelt_articles)
    except Exception:
        pass

    # Source 2: CNBC Markets RSS (always available, no key)
    try:
        cnbc_articles = fetch_cnbc_markets_rss()
        all_articles.extend(cnbc_articles)
    except Exception:
        pass

    # Source 3: Finnhub general news
    try:
        finnhub_articles = fetch_finnhub_general_news()
        all_articles.extend(finnhub_articles)
    except Exception:
        pass

    # Source 4: NewsAPI business headlines
    try:
        newsapi_articles = fetch_newsapi_global_headlines()
        all_articles.extend(newsapi_articles)
    except Exception:
        pass

    # If no articles from any source, return neutral
    if not all_articles:
        return {
            "global_sentiment_score": 0.0,
            "impact_factor": 0.0,
        }

    # --- Score each article and apply hour-based time-decay ---
    now = datetime.now()

    decayed_sentiments = []
    weights = []

    for article in all_articles:
        raw_sentiment = score_sentiment(article["headline"], article["description"])

        hours_old = compute_hours_since(article["published_date"], now)
        decayed = apply_time_decay(raw_sentiment, hours_old, NEWS_DECAY_RATE)
        decayed_sentiments.append(decayed)

        weight = compute_recency_weight(article["published_date"], now)
        weights.append(weight)

    decayed_sentiments = np.array(decayed_sentiments)
    weights = np.array(weights)

    # Average of time-decayed sentiments
    if len(decayed_sentiments) > 0:
        weighted_sentiment = float(np.mean(decayed_sentiments))
    else:
        weighted_sentiment = 0.0

    # Impact factor: normalised by reference max (15 perfectly recent articles)
    total_weight = weights.sum()
    impact_factor = min(total_weight / 15.0, 1.0)

    return {
        "global_sentiment_score": round(weighted_sentiment, 4),
        "impact_factor": round(impact_factor, 4),
    }
