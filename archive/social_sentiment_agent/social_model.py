"""
Social Sentiment Agent
=======================
Monitors social media platforms for stock-specific sentiment
and produces a score using VADER.

Data sources (tried in order, failures are silently skipped):

    1. Reddit RSS  — r/wallstreetbets search (free, no key)
    2. Reddit RSS  — r/stocks search (free, no key)
    3. Stocktwits public stream (free, best-effort — may be rate-limited)

At minimum the Reddit RSS feeds work without any API keys.

For future expansion, set environment variables:
    export REDDIT_CLIENT_ID="..."
    export REDDIT_CLIENT_SECRET="..."
    (enables higher rate limits via PRAW, not implemented in this prototype)
"""

import re
import time
from datetime import datetime

import feedparser
import numpy as np
import requests
from utils.sentiment_decay import SOCIAL_DECAY_RATE, apply_time_decay, compute_hours_since
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REQUEST_TIMEOUT = 10
USER_AGENT = "TrendSignalEngine/1.0"

# Social posts lose relevance faster than traditional news articles
RECENCY_DECAY = 0.15

# Subreddits to search (most active retail-investor communities)
SUBREDDITS = ["wallstreetbets", "stocks"]

# VADER analyser
_analyser = SentimentIntensityAnalyzer()


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def strip_html_tags(text: str) -> str:
    """Remove HTML markup, leaving only plain text."""
    clean = re.sub(r"<[^>]+>", "", text)
    return clean.strip()


# ---------------------------------------------------------------------------
# Source 1 & 2: Reddit RSS (free, no key)
# ---------------------------------------------------------------------------

def fetch_reddit_rss(ticker: str, subreddit: str) -> list[dict]:
    """
    Search a subreddit for recent posts mentioning the ticker via RSS.

    Reddit exposes a public RSS endpoint for search results.
    No authentication required.
    """
    url = f"https://www.reddit.com/r/{subreddit}/search.rss"
    params = {
        "q": ticker,
        "sort": "new",
        "restrict_sr": "on",
        "t": "week",
    }
    headers = {"User-Agent": USER_AGENT}

    response = requests.get(
        url, params=params, headers=headers, timeout=REQUEST_TIMEOUT
    )
    response.raise_for_status()

    feed = feedparser.parse(response.content)

    posts = []
    for entry in feed.entries[:10]:
        # Reddit RSS uses 'updated_parsed' for post timestamps
        if entry.get("updated_parsed"):
            published = datetime(*entry.updated_parsed[:6])
        elif entry.get("published_parsed"):
            published = datetime(*entry.published_parsed[:6])
        else:
            published = datetime.now()

        title = strip_html_tags(entry.get("title", ""))
        body = strip_html_tags(entry.get("summary", ""))

        posts.append({
            "headline": title,
            "description": body,
            "published_date": published,
            "source": f"reddit_r/{subreddit}",
        })

    return posts


# ---------------------------------------------------------------------------
# Source 3: Stocktwits public stream (best-effort)
# ---------------------------------------------------------------------------

def fetch_stocktwits(ticker: str) -> list[dict]:
    """
    Fetch the public message stream for a ticker from Stocktwits.

    This endpoint may be rate-limited or unavailable.
    Failures are caught by the caller's try/except.
    """
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    headers = {"User-Agent": USER_AGENT}

    response = requests.get(url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()

    posts = []
    for msg in data.get("messages", [])[:15]:
        date_str = msg.get("created_at", "")

        # Stocktwits timestamps come in varying ISO formats
        try:
            published = datetime.strptime(date_str, "%Y-%m-%dT%H:%M:%SZ")
        except (ValueError, TypeError):
            published = datetime.now()

        posts.append({
            "headline": msg.get("body", ""),
            "description": "",
            "published_date": published,
            "source": "stocktwits",
        })

    return posts


# ---------------------------------------------------------------------------
# Sentiment scoring (VADER)
# ---------------------------------------------------------------------------

def score_sentiment(headline: str, description: str) -> float:
    """
    Run VADER on headline + description.

    Returns the compound score in [-1, +1].
    VADER handles internet slang, emojis, and capitalisation well,
    which makes it suitable for social media text.
    """
    text = f"{headline}. {description}"
    scores = _analyser.polarity_scores(text)
    return scores["compound"]


# ---------------------------------------------------------------------------
# Recency weighting
# ---------------------------------------------------------------------------

def compute_recency_weight(post_date: datetime, reference_date: datetime) -> float:
    """
    Exponential decay weight.

    Social posts use a steeper decay (0.15) than news articles (0.1)
    because social sentiment shifts faster.
    """
    days_old = max(0, (reference_date - post_date).days)
    weight = np.exp(-RECENCY_DECAY * days_old)
    return float(weight)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_social_sentiment_model(ticker: str) -> dict:
    """
    Full pipeline: fetch from Reddit + Stocktwits → score → weight → aggregate.

    Returns:
        social_sentiment_score : float in [-1, +1]
        impact_factor          : float in [0, 1]
    """
    all_posts = []

    # --- Reddit RSS for each subreddit ---
    for subreddit in SUBREDDITS:
        try:
            posts = fetch_reddit_rss(ticker, subreddit)
            all_posts.extend(posts)
        except Exception:
            pass
        # Small delay between Reddit requests to stay polite
        time.sleep(0.3)

    # --- Stocktwits (best-effort) ---
    try:
        st_posts = fetch_stocktwits(ticker)
        all_posts.extend(st_posts)
    except Exception:
        pass

    # If no social data at all, return neutral
    if not all_posts:
        return {
            "social_sentiment_score": 0.0,
            "impact_factor": 0.0,
        }

    # --- Score each post and apply hour-based time-decay ---
    now = datetime.now()

    decayed_sentiments = []
    weights = []

    for post in all_posts:
        raw_sentiment = score_sentiment(post["headline"], post["description"])

        hours_old = compute_hours_since(post["published_date"], now)
        decayed = apply_time_decay(raw_sentiment, hours_old, SOCIAL_DECAY_RATE)
        decayed_sentiments.append(decayed)

        weight = compute_recency_weight(post["published_date"], now)
        weights.append(weight)

    decayed_sentiments = np.array(decayed_sentiments)
    weights = np.array(weights)

    # Average of time-decayed sentiments (steeper decay for social data)
    if len(decayed_sentiments) > 0:
        weighted_sentiment = float(np.mean(decayed_sentiments))
    else:
        weighted_sentiment = 0.0

    # Impact factor: normalised by 8 perfectly recent posts
    total_weight = weights.sum()
    impact_factor = min(total_weight / 8.0, 1.0)

    return {
        "social_sentiment_score": round(weighted_sentiment, 4),
        "impact_factor": round(impact_factor, 4),
    }
