from __future__ import annotations

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Annualised volatility thresholds used to classify confidence.
#
# Daily standard deviation is multiplied by sqrt(252) to annualise.
#   < 20 %  annualised  →  Low volatility   →  High confidence
#   20–40 % annualised  →  Medium volatility →  Medium confidence
#   > 40 %  annualised  →  High volatility   →  Low confidence
# ---------------------------------------------------------------------------

ANNUALISE_FACTOR = np.sqrt(252)

LOW_VOL_THRESHOLD = 0.20       # 20 % annualised
HIGH_VOL_THRESHOLD = 0.40      # 40 % annualised


def compute_daily_returns(close_prices: pd.Series) -> pd.Series:
    """Percentage change from one trading day to the next."""
    daily_returns = close_prices.pct_change()
    return daily_returns


def compute_rolling_volatility(daily_returns: pd.Series, window: int) -> pd.Series:
    """
    Rolling standard deviation of daily returns over the given window.

    This measures how much the stock price fluctuates day-to-day
    within a recent lookback period.
    """
    rolling_vol = daily_returns.rolling(window=window).std()
    return rolling_vol


def compute_vol_term_structure(daily_returns: pd.Series) -> pd.DataFrame:
    """
    Compute a simple realised volatility term-structure and vol-of-vol.

    Horizons:
        - 5d, 10d, 20d, 60d realised volatility (daily std of returns)
        - vol_of_vol_20: 20-day rolling std of the 20d realised volatility
        - jump_indicator: |return| > 2 * 20d realised vol (1.0 else 0.0)

    Returns
    -------
    pd.DataFrame
        Columns: vol_5d, vol_10d, vol_20d, vol_60d, vol_of_vol_20, jump_indicator
        Index: same as daily_returns.
    """
    vol_5 = compute_rolling_volatility(daily_returns, window=5)
    vol_10 = compute_rolling_volatility(daily_returns, window=10)
    vol_20 = compute_rolling_volatility(daily_returns, window=20)
    vol_60 = compute_rolling_volatility(daily_returns, window=60)

    vol_of_vol_20 = vol_20.rolling(20).std()
    jump_indicator = (
        (daily_returns.abs() > (2.0 * vol_20.replace(0.0, np.nan)))
        .astype(float)
        .fillna(0.0)
    )

    out = pd.DataFrame(
        {
            "vol_5d": vol_5,
            "vol_10d": vol_10,
            "vol_20d": vol_20,
            "vol_60d": vol_60,
            "vol_of_vol_20": vol_of_vol_20,
            "jump_indicator": jump_indicator,
        }
    )
    return out


def classify_confidence(annualised_vol: float) -> str:
    """
    Map annualised volatility to a confidence level.

    Low volatility means the trend signal is more trustworthy (High confidence).
    High volatility means the trend signal is less reliable (Low confidence).
    """
    if annualised_vol < LOW_VOL_THRESHOLD:
        return "High"
    elif annualised_vol <= HIGH_VOL_THRESHOLD:
        return "Medium"
    else:
        return "Low"


def compute_rolling_confidence(daily_returns: pd.Series, window: int = 20) -> pd.Series:
    """
    Compute a confidence label for every row in the time series.

    Uses the rolling volatility (annualised) at each point in time.
    This is used for plotting confidence shading across the full chart.
    """
    rolling_vol = compute_rolling_volatility(daily_returns, window=window)

    # Annualise so the thresholds are in familiar percentage terms
    annualised_vol = rolling_vol * ANNUALISE_FACTOR

    # Map each day's volatility to a confidence label
    confidence_series = annualised_vol.apply(classify_confidence)

    return confidence_series


def run_volatility_model(stock_data: pd.DataFrame) -> dict | None:
    """
    Analyse volatility and produce a confidence metric for the most recent day.

    Steps:
        1. Compute daily returns from closing prices
        2. Compute 20-day and 50-day rolling volatility
        3. Annualise the 20-day volatility
        4. Map to a confidence level (High / Medium / Low)

    Returns a dict with:
        volatility_20   - latest 20-day rolling std of daily returns
        volatility_50   - latest 50-day rolling std of daily returns
        confidence      - High / Medium / Low
    """
    close_prices = stock_data["Close"]

    # Step 1: daily percentage returns
    daily_returns = compute_daily_returns(close_prices)

    # Step 2: rolling volatility over two windows
    vol_20 = compute_rolling_volatility(daily_returns, window=20)
    vol_50 = compute_rolling_volatility(daily_returns, window=50)

    # We need at least 50 rows of returns to have both windows filled
    if vol_50.dropna().empty:
        return None

    # Step 3: grab the most recent values
    latest_vol_20 = float(vol_20.iloc[-1])
    latest_vol_50 = float(vol_50.iloc[-1])

    # Step 4: annualise the short-window vol for confidence classification
    annualised_vol_20 = latest_vol_20 * ANNUALISE_FACTOR
    confidence = classify_confidence(annualised_vol_20)

    results = {
        "volatility_20": round(latest_vol_20, 6),
        "volatility_50": round(latest_vol_50, 6),
        "confidence": confidence,
    }

    return results
