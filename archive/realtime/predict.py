"""
Stateless Prediction for Real-Time Updates
===========================================
Single-ticker, latest-bar signal only. No global state; fast path for streaming.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


def predict_latest_signal(
    ticker: str,
    ohlcv_df: pd.DataFrame,
    signal_engine,
    *,
    min_bars: int = 210,
) -> dict[str, Any] | None:
    """
    Compute the latest signal for one ticker from OHLCV history.
    Stateless: only ohlcv_df and signal_engine (with its weights) are used.

    Intended for real-time: call when new bar arrives; keep ohlcv_df
    as a rolling window (e.g. 400+ bars for trend/vol features).

    Parameters
    ----------
    ticker : str
        Symbol (for attachment to result).
    ohlcv_df : pd.DataFrame
        OHLCV with DatetimeIndex; must have Open, High, Low, Close, Volume.
    signal_engine : SignalEngine
        Configured engine (weights/learned_weights set at init).
    min_bars : int
        Minimum rows required; return None if insufficient.

    Returns
    -------
    dict with keys: ticker, date, signal, adjusted_score, confidence, trend_score,
    or None if data insufficient or no signal row.
    """
    if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < min_bars:
        return None
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col not in ohlcv_df.columns:
            return None

    signals_df = signal_engine.generate_signals(ohlcv_df)
    if signals_df.empty:
        return None

    last = signals_df.iloc[-1]
    date = signals_df.index[-1]
    return {
        "ticker": ticker,
        "date": pd.Timestamp(date).isoformat(),
        "signal": str(last.get("signal", "Neutral")),
        "adjusted_score": float(last.get("adjusted_score", 0.0)),
        "confidence": str(last.get("confidence", "")),
        "trend_score": float(last.get("trend_score", 0.0)),
    }
