"""
Regional News Agent — Live API Version
========================================
Pulls recent news for a specific stock from live free APIs
and produces a sentiment score using VADER.

Data sources (tried in order, failures are silently skipped):

    1. Finnhub Company News  — requires FINNHUB_KEY env var
    2. Google News RSS        — free, no key required
    3. NewsAPI.org            — requires NEWSAPI_KEY env var

At minimum, Google News RSS works without any API keys.
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

# API keys are read from environment variables.
# If a key is not set, that source is simply skipped.
FINNHUB_KEY = os.environ.get("FINNHUB_KEY", "")
NEWSAPI_KEY = os.environ.get("NEWSAPI_KEY", "")

# HTTP request settings
REQUEST_TIMEOUT = 10
USER_AGENT = "TrendSignalEngine/1.0"

# Exponential decay rate for recency weighting.
# 0.1 means a 7-day-old article retains ~50% influence.
RECENCY_DECAY = 0.1

# VADER analyser (created once, reused for every call)
_analyser = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Utility: strip HTML tags from RSS descriptions
# ---------------------------------------------------------------------------

def strip_html_tags(text: str) -> str:
    """Remove HTML markup, leaving only plain text."""
    clean = re.sub(r"<[^>]+>", "", text)
    return clean.strip()


# ---------------------------------------------------------------------------
# Source 1: Finnhub Company News
# ---------------------------------------------------------------------------

def fetch_finnhub_company_news(ticker: str) -> list[dict]:
    """
    Fetch recent company-specific news from Finnhub.

    Returns a list of normalised article dicts.
    Skipped entirely if FINNHUB_KEY is not set.
    """
    if not FINNHUB_KEY:
        return []

    today = datetime.now().strftime("%Y-%m-%d")
    thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    url = "https://finnhub.io/api/v1/company-news"
    params = {
        "symbol": ticker,
        "from": thirty_days_ago,
        "to": today,
        "token": FINNHUB_KEY,
    }

    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    raw_articles = response.json()

    # Normalise into our standard format (take the 20 most recent)
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
# Source 2: Google News RSS
# ---------------------------------------------------------------------------

def fetch_google_news_rss(ticker: str) -> list[dict]:
    """
    Fetch recent stock news from Google News via RSS.

    Free and requires no API key.  Uses the requests library for the
    HTTP call (so we can set a User-Agent) and feedparser for XML parsing.
    """
    query = f"{ticker} stock"
    url = (
        f"https://news.google.com/rss/search"
        f"?q={query}&hl=en-US&gl=US&ceid=US:en"
    )

    headers = {"User-Agent": USER_AGENT}
    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()

    feed = feedparser.parse(response.content)

    articles = []
    for entry in feed.entries[:15]:
        # Parse publication date from the feed entry
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
            "source": "google_news_rss",
        })

    return articles


# ---------------------------------------------------------------------------
# Source 3: NewsAPI.org
# ---------------------------------------------------------------------------

def fetch_newsapi_articles(ticker: str) -> list[dict]:
    """
    Fetch recent articles mentioning the ticker from NewsAPI.

    Skipped entirely if NEWSAPI_KEY is not set.
    Free tier: 100 requests/day, searches back ~30 days.
    """
    if not NEWSAPI_KEY:
        return []

    thirty_days_ago = (datetime.now() - timedelta(days=30)).strftime("%Y-%m-%d")

    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{ticker} stock",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": 20,
        "from": thirty_days_ago,
        "apiKey": NEWSAPI_KEY,
    }

    response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    articles = []
    for item in data.get("articles", []):
        # NewsAPI dates are ISO 8601: "2026-03-08T12:00:00Z"
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

    Recent articles get weight ≈ 1.0, older articles fade toward 0.
    """
    days_old = (reference_date - article_date).days

    if days_old < 0:
        days_old = 0

    weight = np.exp(-RECENCY_DECAY * days_old)
    return float(weight)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_regional_news_model(ticker: str) -> dict:
    """
    Full pipeline: fetch from live APIs → score → weight → aggregate.

    Tries each source independently and combines all articles that were
    successfully retrieved.  If every source fails, returns neutral.

    Returns:
        regional_sentiment_score : float in [-1, +1]
        impact_factor            : float in [0, 1]
    """
    all_articles = []

    # --- Collect articles from every available source ---

    # Source 1: Finnhub
    try:
        finnhub_articles = fetch_finnhub_company_news(ticker)
        all_articles.extend(finnhub_articles)
    except Exception:
        pass

    # Source 2: Google News RSS (always available, no key)
    try:
        rss_articles = fetch_google_news_rss(ticker)
        all_articles.extend(rss_articles)
    except Exception:
        pass

    # Source 3: NewsAPI
    try:
        newsapi_articles = fetch_newsapi_articles(ticker)
        all_articles.extend(newsapi_articles)
    except Exception:
        pass

    # If no articles from any source, return neutral
    if not all_articles:
        return {
            "regional_sentiment_score": 0.0,
            "impact_factor": 0.0,
        }

    # --- Score each article and apply hour-based time-decay ---
    now = datetime.now()

    decayed_sentiments = []
    weights = []

    for article in all_articles:
        raw_sentiment = score_sentiment(article["headline"], article["description"])

        # Compute hours since publication and decay the sentiment
        hours_old = compute_hours_since(article["published_date"], now)
        decayed = apply_time_decay(raw_sentiment, hours_old, NEWS_DECAY_RATE)
        decayed_sentiments.append(decayed)

        # Recency weight is still used for the impact factor calculation
        weight = compute_recency_weight(article["published_date"], now)
        weights.append(weight)

    decayed_sentiments = np.array(decayed_sentiments)
    weights = np.array(weights)

    # Average of time-decayed sentiments (decay already encodes recency)
    if len(decayed_sentiments) > 0:
        weighted_sentiment = float(np.mean(decayed_sentiments))
    else:
        weighted_sentiment = 0.0

    # Impact factor: normalised by reference max (10 perfectly recent articles)
    total_weight = weights.sum()
    impact_factor = min(total_weight / 10.0, 1.0)

    return {
        "regional_sentiment_score": round(weighted_sentiment, 4),
        "impact_factor": round(impact_factor, 4),
    }
