"""
News Impact Detection
======================
Identifies trading days where news likely caused abnormal price movement,
using return magnitude and volume spikes as confirmation signals.
"""

import numpy as np
import pandas as pd


def detect_news_impact(price_df: pd.DataFrame) -> pd.DataFrame:
    """
    Enrich a price DataFrame with news-impact indicators.

    Inputs:
        price_df — DataFrame with at least: Close, Volume
                   (as produced by data_loader / feature_engineering)

    Steps:
        1. Daily return                  = Close.pct_change()
        2. Rolling 20-day volatility     = return.rolling(20).std()
        3. Abnormal return               = |return| / volatility
        4. Volume spike ratio            = Volume / rolling-20-day mean(Volume)
        5. News event flag               = abnormal_return > 2  OR  volume_ratio > 2
        6. Impact score                  = abnormal_return + log(volume_ratio)

    Returns:
        A copy of price_df with added columns:
            daily_return, volatility, abnormal_return,
            volume_ratio, news_event_flag, impact_score
    """
    df = price_df.copy()

    # Step 1: daily return
    df["daily_return"] = df["Close"].pct_change()

    # Step 2: rolling 20-day volatility of returns
    df["volatility"] = df["daily_return"].rolling(window=20).std()

    # Step 3: abnormal return (guard against division by zero)
    safe_volatility = df["volatility"].replace(0, np.nan)
    df["abnormal_return"] = df["daily_return"].abs() / safe_volatility

    # Step 4: volume spike ratio
    avg_volume = df["Volume"].rolling(window=20).mean()
    safe_avg_volume = avg_volume.replace(0, np.nan)
    df["volume_ratio"] = df["Volume"] / safe_avg_volume

    # Step 5: a day is a "news-impact candidate" if either threshold is breached
    df["news_event_flag"] = (
        (df["abnormal_return"] > 2.0) | (df["volume_ratio"] > 2.0)
    )

    # Step 6: impact score = abnormal_return + log(volume_ratio)
    # Clip volume_ratio to avoid log(0)
    safe_ratio = df["volume_ratio"].clip(lower=0.01)
    df["impact_score"] = df["abnormal_return"].fillna(0) + np.log(safe_ratio).fillna(0)

    return df
