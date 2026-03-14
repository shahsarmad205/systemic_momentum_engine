"""
Real-Time Integration Layer
============================
Prepares the system for live trading with:

- Asynchronous data ingestion hooks (register callbacks for new bars/data).
- Lightweight signal streaming (publish signals to dashboard or message queue).
- Stateless, fast prediction for single-ticker updates.

Usage:
    from realtime import DataIngestionHooks, SignalStream, predict_latest_signal

    hooks = DataIngestionHooks()
    hooks.on_bar(lambda ticker, bar: stream.publish(ticker, bar))

    stream = SignalStream()
    stream.subscribe(lambda ticker, signal: print(ticker, signal))
"""

from .hooks import DataIngestionHooks
from .signal_stream import SignalStream
from .predict import predict_latest_signal
from .runner import make_data_ready_handler, run_streaming_poll

__all__ = [
    "DataIngestionHooks",
    "SignalStream",
    "predict_latest_signal",
    "make_data_ready_handler",
    "run_streaming_poll",
]
