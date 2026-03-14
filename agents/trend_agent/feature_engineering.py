import pandas as pd
import numpy as np


def compute_daily_returns(close_prices: pd.Series) -> pd.Series:
    """Percentage change from one trading day to the next."""
    daily_returns = close_prices.pct_change()
    return daily_returns


def compute_momentum(close_prices: pd.Series, lookback_days: int) -> pd.Series:
    """
    Momentum = (current price / price N days ago) - 1

    Measures cumulative return over the lookback window.
    min_periods=1 allows partial windows when data is shorter than lookback.
    """
    momentum = close_prices / close_prices.shift(lookback_days) - 1
    return momentum


def compute_moving_average(close_prices: pd.Series, window: int) -> pd.Series:
    """Simple moving average over the given window. min_periods=1 for partial windows."""
    moving_avg = close_prices.rolling(window=window, min_periods=1).mean()
    return moving_avg


def compute_ma_crossover_signal(ma_short: pd.Series, ma_long: pd.Series) -> pd.Series:
    """
    Moving average crossover signal.

    Returns +1 when the short MA is above the long MA (bullish),
    and -1 when the short MA is below the long MA (bearish).
    """
    signal = pd.Series(np.where(ma_short > ma_long, 1, -1), index=ma_short.index)
    return signal


def build_features(stock_data: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all momentum and trend features from raw stock data.

    Returns a new DataFrame with the original price data plus:
        - daily_return
        - momentum_3m  (63 trading days)
        - momentum_6m  (126 trading days)
        - ma_50
        - ma_200
        - ma_crossover_signal

    Uses adjusted close ('AdjClose') when available to avoid spurious
    jumps from splits/dividends; falls back to 'Close'.
    """
    close = stock_data["AdjClose"] if "AdjClose" in stock_data.columns else stock_data["Close"]

    features = stock_data.copy()

    # Daily percentage returns
    features["daily_return"] = compute_daily_returns(close)

    # Momentum over 3 months and 6 months
    features["momentum_3m"] = compute_momentum(close, lookback_days=63)
    features["momentum_6m"] = compute_momentum(close, lookback_days=126)

    # Moving averages
    features["ma_50"] = compute_moving_average(close, window=50)
    features["ma_200"] = compute_moving_average(close, window=200)

    # Crossover signal: 50-day MA vs 200-day MA
    features["ma_crossover_signal"] = compute_ma_crossover_signal(
        features["ma_50"], features["ma_200"]
    )

    # Core feature columns (used for fillna and for empty fallback).
    core_cols = [
        "daily_return",
        "momentum_3m",
        "momentum_6m",
        "ma_50",
        "ma_200",
        "ma_crossover_signal",
    ]
    # Trim only minimal warmup: drop row 0 where daily_return is NaN from pct_change().
    # Do not dropna on momentum_3m/momentum_6m (they need 63/126 rows) so we keep most rows.
    features.dropna(subset=["daily_return"], inplace=True)
    # Fill NaNs in other core cols (warmup periods for momentum/MA) so we never collapse to one row.
    features[core_cols] = features[core_cols].fillna(0)
    # If dropna removed everything (e.g. single-row input), rebuild full-frame and fill so we return all rows.
    if features.empty and len(stock_data) > 0:
        features = stock_data.copy()
        features["daily_return"] = compute_daily_returns(close)
        features["momentum_3m"] = compute_momentum(close, lookback_days=63)
        features["momentum_6m"] = compute_momentum(close, lookback_days=126)
        features["ma_50"] = close.rolling(50, min_periods=1).mean()
        features["ma_200"] = close.rolling(200, min_periods=1).mean()
        features["ma_crossover_signal"] = compute_ma_crossover_signal(
            features["ma_50"], features["ma_200"]
        )
        features[core_cols] = features[core_cols].fillna(0)
    return features
