"""
Sentiment scoring for news headlines using ProsusAI/finbert.

Integrates with pipeline.news_scraper output (fetch_news_for_ticker).
Used by the full signals pipeline for regional_news / global_news / social.
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any

from transformers import pipeline as hf_pipeline

# ---------------------------------------------------------------------------
# FinBERT pipeline (loaded once, shared for caching)
# ---------------------------------------------------------------------------

_PIPELINE: Any = None


def _get_pipeline():
    global _PIPELINE
    if _PIPELINE is None:
        _PIPELINE = hf_pipeline(
            "sentiment-analysis",
            model="ProsusAI/finbert",
        )
    return _PIPELINE


# ---------------------------------------------------------------------------
# Per-headline cache to avoid reprocessing identical headlines
# ---------------------------------------------------------------------------

@lru_cache(maxsize=2048)
def _cached_single_headline(headline: str) -> tuple[str, float]:
    """
    Run FinBERT on a single headline. Returns (label, score).
    Cached by headline string.
    """
    headline = (headline or "").strip()
    if not headline:
        return ("neutral", 0.0)
    pipe = _get_pipeline()
    # Truncate to model max length if needed (e.g. 512 tokens)
    text = headline[:4000]
    out = pipe(text, truncation=True, max_length=512)
    if not out:
        return ("neutral", 0.0)
    item = out[0]
    label = (item.get("label") or "neutral").lower()
    score = float(item.get("score") or 0.0)
    return (label, score)


def _label_to_value(label: str) -> float:
    """Map FinBERT label to numeric: positive=+1, negative=-1, neutral=0."""
    label = (label or "").lower()
    if label == "positive":
        return 1.0
    if label == "negative":
        return -1.0
    return 0.0


def _aggregate_score_to_label(score: float) -> str:
    """Map aggregate score in [-1, 1] to display label."""
    if score <= -0.5:
        return "Very Negative"
    if score <= -0.1:
        return "Negative"
    if score < 0.1:
        return "Neutral"
    if score < 0.5:
        return "Positive"
    return "Very Positive"


# ---------------------------------------------------------------------------
# SentimentScorer
# ---------------------------------------------------------------------------

class SentimentScorer:
    """
    Scores financial text sentiment using ProsusAI/finbert.
    Loads the model once; uses per-headline cache to avoid reprocessing.
    """

    def __init__(self) -> None:
        """Load the FinBERT pipeline once (model=ProsusAI/finbert)."""
        _get_pipeline()

    def score_headlines(self, headlines: list[str]) -> float:
        """
        Run FinBERT on each headline; map positive=+1, negative=-1, neutral=0;
        weight by confidence; return aggregate in [-1, 1].
        Empty list returns 0.0. Uses cache for identical headlines.
        """
        if not headlines:
            return 0.0
        weighted_sum = 0.0
        weight_total = 0.0
        for h in headlines:
            label, conf = _cached_single_headline((h or "").strip())
            if not (h or "").strip():
                continue
            val = _label_to_value(label)
            weighted_sum += val * conf
            weight_total += conf
        if weight_total <= 0:
            return 0.0
        raw = weighted_sum / weight_total
        return max(-1.0, min(1.0, raw))

    def score_ticker_news(self, news_items: list[dict]) -> dict[str, Any]:
        """
        Take output from fetch_news_for_ticker() and return aggregate score,
        label, article_count, daily_scores (grouped by date), and source_breakdown.
        Handles empty list gracefully (score 0.0, label Neutral, zeros).
        """
        empty = {
            "aggregate_score": 0.0,
            "label": "Neutral",
            "article_count": 0,
            "daily_scores": [],
            "source_breakdown": {"yfinance": 0.0, "seeking_alpha": 0.0, "reuters": 0.0},
        }
        if not news_items:
            return empty

        # Per-article sentiment (headline -> score for weighting)
        headlines_list = []
        by_date: dict[str, list[dict]] = defaultdict(list)
        by_source: dict[str, list[float]] = defaultdict(list)

        for item in news_items:
            headline = (item.get("headline") or "").strip()
            if not headline:
                continue
            ts = item.get("timestamp")
            source = (item.get("source") or "").strip().lower()
            if source == "seeking alpha":
                source = "seeking_alpha"
            # Date key YYYY-MM-DD (use UTC)
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                date_key = ts.astimezone(timezone.utc).strftime("%Y-%m-%d")
            else:
                date_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")

            label, conf = _cached_single_headline(headline)
            val = _label_to_value(label) * conf
            weight = conf
            # Store for daily aggregate: we need weighted avg per day
            by_date[date_key].append({
                "headline": headline,
                "weighted_val": val,
                "weight": weight,
            })
            headlines_list.append(headline)
            if source:
                by_source[source].append(val / weight if weight else 0.0)

        if not headlines_list:
            return empty

        aggregate_score = self.score_headlines(headlines_list)
        label = _aggregate_score_to_label(aggregate_score)
        article_count = len(headlines_list)

        # daily_scores: list of {date, avg_sentiment, article_count, headlines}
        daily_scores = []
        for date_key in sorted(by_date.keys(), reverse=True):
            entries = by_date[date_key]
            total_w = sum(e["weight"] for e in entries)
            if total_w <= 0:
                avg_sentiment = 0.0
            else:
                avg_sentiment = sum(e["weighted_val"] for e in entries) / total_w
            avg_sentiment = max(-1.0, min(1.0, avg_sentiment))
            daily_scores.append({
                "date": date_key,
                "avg_sentiment": round(avg_sentiment, 4),
                "article_count": len(entries),
                "headlines": [e["headline"] for e in entries],
            })

        # source_breakdown: average sentiment per source (yfinance, seeking_alpha, reuters)
        source_breakdown = {"yfinance": 0.0, "seeking_alpha": 0.0, "reuters": 0.0}
        for src, vals in by_source.items():
            if src not in source_breakdown or not vals:
                continue
            avg = sum(vals) / len(vals)
            source_breakdown[src] = round(max(-1.0, min(1.0, avg)), 4)

        return {
            "aggregate_score": round(aggregate_score, 4),
            "label": label,
            "article_count": article_count,
            "daily_scores": daily_scores,
            "source_breakdown": source_breakdown,
        }
