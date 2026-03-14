import numpy as np
import pandas as pd


def sigmoid(x: float) -> float:
    """Map any real number to the (0, 1) range."""
    return 1.0 / (1.0 + np.exp(-x))


def compute_trend_score(latest_features: pd.Series) -> float:
    """
    Combine momentum features into a single trend score.

    The score is a weighted sum of normalised signals:
        - 3-month momentum  (weight 0.30)
        - 6-month momentum  (weight 0.25)
        - MA crossover       (weight 0.25)
        - Daily return        (weight 0.20)

    Momentum values are multiplied by a scaling factor so they
    sit in a range the sigmoid can work with sensibly.
    """
    # Scaling factor converts small decimal returns into a wider numeric range.
    # Without this, momentum values like 0.05 would all map near sigmoid(0) ≈ 0.5.
    scale = 10.0

    weighted_score = (
        0.30 * latest_features["momentum_3m"] * scale
        + 0.25 * latest_features["momentum_6m"] * scale
        + 0.25 * latest_features["ma_crossover_signal"]
        + 0.20 * latest_features["daily_return"] * scale
    )

    return float(weighted_score)


def classify_signal(probability: float) -> str:
    """
    Convert a probability into a human-readable signal.

    Thresholds:
        probability > 0.60  -> Bullish
        probability < 0.40  -> Bearish
        otherwise           -> Neutral
    """
    if probability > 0.60:
        return "Bullish"
    elif probability < 0.40:
        return "Bearish"
    else:
        return "Neutral"


def run_trend_model(features: pd.DataFrame) -> dict:
    """
    Run the trend model on the most recent row of computed features.

    Returns a dict with:
        trend_score   - raw weighted score
        probability_up - sigmoid-transformed probability
        signal         - Bullish / Bearish / Neutral
    """
    # Use the last available row (most recent trading day)
    latest = features.iloc[-1]

    trend_score = compute_trend_score(latest)
    probability_up = sigmoid(trend_score)
    signal = classify_signal(probability_up)

    results = {
        "trend_score": round(trend_score, 4),
        "probability_up": round(probability_up, 4),
        "signal": signal,
    }

    return results
