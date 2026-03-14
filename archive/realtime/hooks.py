"""
Asynchronous Data Ingestion Hooks
==================================
Register callbacks to be notified when new market data is available.
Designed for async feeders (e.g. websocket bars, scheduled fetches) to push
data without blocking; the engine can then run prediction and stream signals.

All callbacks are invoked synchronously by default; callers can schedule
work on a thread pool or event loop for true async.
"""

from __future__ import annotations

import threading
from typing import Callable

import pandas as pd


Callback = Callable[[str, pd.DataFrame], None]


class DataIngestionHooks:
    """
    Lightweight hook registry for data ingestion events.

    - on_bar(ticker, bar_df): one or more new OHLCV rows for a ticker.
    - on_data_ready(ticker, ohlcv_df): full or incremental OHLCV dataset ready.

    Callbacks are invoked in registration order. Exceptions in one callback
    do not stop others. Thread-safe for register/notify.
    """

    def __init__(self):
        self._on_bar: list[Callback] = []
        self._on_data_ready: list[Callback] = []
        self._lock = threading.Lock()

    def register_bar(self, callback: Callback) -> None:
        """Register a callback for new bar(s): callback(ticker, bar_df)."""
        with self._lock:
            self._on_bar.append(callback)

    def register_data_ready(self, callback: Callback) -> None:
        """Register a callback for data ready: callback(ticker, ohlcv_df)."""
        with self._lock:
            self._on_data_ready.append(callback)

    def unregister_bar(self, callback: Callback) -> None:
        with self._lock:
            if callback in self._on_bar:
                self._on_bar.remove(callback)

    def unregister_data_ready(self, callback: Callback) -> None:
        with self._lock:
            if callback in self._on_data_ready:
                self._on_data_ready.remove(callback)

    def notify_bar(self, ticker: str, bar_df: pd.DataFrame) -> None:
        """
        Notify all bar callbacks. bar_df should have OHLCV columns and a DatetimeIndex.
        Safe to call from an async thread; callbacks run in caller's thread.
        """
        with self._lock:
            callbacks = list(self._on_bar)
        for cb in callbacks:
            try:
                cb(ticker, bar_df)
            except Exception:
                pass  # log in production

    def notify_data_ready(self, ticker: str, ohlcv_df: pd.DataFrame) -> None:
        """
        Notify all data-ready callbacks (e.g. full history or chunk updated).
        Safe to call from an async thread.
        """
        with self._lock:
            callbacks = list(self._on_data_ready)
        for cb in callbacks:
            try:
                cb(ticker, ohlcv_df)
            except Exception:
                pass
