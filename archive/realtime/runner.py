"""
Streaming Runner — Wire data hooks to prediction and signal stream
===================================================================
Example: register a data-ready callback that runs predict_latest_signal
and publishes to SignalStream. Suitable for polling or async data feeds.
"""

from __future__ import annotations

from .hooks import DataIngestionHooks
from .signal_stream import SignalStream
from .predict import predict_latest_signal


def make_data_ready_handler(signal_engine, stream: SignalStream, min_bars: int = 210):
    """
    Return a callback suitable for DataIngestionHooks.register_data_ready:
    when data is ready for a ticker, compute latest signal and publish to stream.

    Usage:
        hooks = DataIngestionHooks()
        stream = SignalStream()
        handler = make_data_ready_handler(engine, stream)
        hooks.register_data_ready(handler)
        # ... feeder calls hooks.notify_data_ready(ticker, ohlcv_df)
    """
    def handler(ticker: str, ohlcv_df):
        out = predict_latest_signal(ticker, ohlcv_df, signal_engine, min_bars=min_bars)
        if out is not None:
            stream.publish(ticker, out)
    return handler


def run_streaming_poll(
    tickers: list[str],
    signal_engine,
    stream: SignalStream,
    get_ohlcv_fn,
    *,
    min_bars: int = 210,
):
    """
    One-shot poll: for each ticker, fetch OHLCV via get_ohlcv_fn(ticker),
    compute latest signal, publish to stream. Use with cron or scheduler.

    get_ohlcv_fn(ticker) -> pd.DataFrame with OHLCV columns.
    """
    hooks = DataIngestionHooks()
    handler = make_data_ready_handler(signal_engine, stream, min_bars=min_bars)
    hooks.register_data_ready(handler)
    for ticker in tickers:
        try:
            df = get_ohlcv_fn(ticker)
            if df is not None and not df.empty:
                hooks.notify_data_ready(ticker, df)
        except Exception:
            pass
