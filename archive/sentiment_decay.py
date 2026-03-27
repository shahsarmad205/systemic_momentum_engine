"""
Sentiment Time-Decay
=====================
Applies exponential time-decay to sentiment scores so that older
articles / posts have progressively less influence.

Formula:
    effective_sentiment = sentiment_score * exp(-decay_rate * hours_old)

Decay rates:
    NEWS_DECAY_RATE   = 0.10  (news articles)
    SOCIAL_DECAY_RATE = 0.15  (social media posts — faster decay)
"""

import numpy as np

# Decay rate per hour for traditional news (regional + global)
NEWS_DECAY_RATE = 0.10

# Decay rate per hour for social media (Reddit, Stocktwits)
SOCIAL_DECAY_RATE = 0.15


def apply_time_decay(sentiment_score: float, hours_old: float, decay_rate: float) -> float:
    """
    Reduce a sentiment score based on how many hours have elapsed.

    Args:
        sentiment_score : raw VADER compound score in [-1, +1]
        hours_old       : hours since the article / post was published
        decay_rate      : decay constant (higher = faster fade)

    Returns:
        Decayed sentiment value.  Approaches 0 as hours_old grows.
    """
    decay_factor = np.exp(-decay_rate * hours_old)
    effective_sentiment = sentiment_score * decay_factor
    return float(effective_sentiment)


def compute_hours_since(published_date, reference_date) -> float:
    """
    Compute the number of hours between two datetimes.

    Returns 0 if the published date is in the future relative to reference.
    """
    delta = reference_date - published_date
    total_seconds = delta.total_seconds()

    if total_seconds < 0:
        return 0.0

    hours = total_seconds / 3600.0
    return hours
