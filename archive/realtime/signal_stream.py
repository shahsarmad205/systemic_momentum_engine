"""
Signal Streaming Interface
===========================
Publish signals to one or more consumers: dashboard, message queue, or logging.
Lightweight in-process pub/sub; for MQ/HTTP, add an adapter that subscribes and forwards.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

SignalCallback = Callable[[str, dict[str, Any]], None]


class SignalStream:
    """
    Lightweight signal streaming: subscribe with callbacks, publish(ticker, signal_dict).

    - subscribe(callback): callback(ticker, signal_dict) is invoked on each publish.
    - publish(ticker, signal_dict): deliver to all subscribers. signal_dict should
      include at least: signal, adjusted_score, confidence, and optionally trend_score, timestamp_utc.
    - get_recent(n): return last n (ticker, signal_dict) for dashboard snapshot (optional).
    """

    def __init__(self, max_recent: int = 100):
        self._subscribers: list[SignalCallback] = []
        self._lock = threading.Lock()
        self._recent: deque[tuple[str, dict[str, Any], float]] = deque(maxlen=max_recent)

    def subscribe(self, callback: SignalCallback) -> None:
        with self._lock:
            self._subscribers.append(callback)

    def unsubscribe(self, callback: SignalCallback) -> None:
        with self._lock:
            if callback in self._subscribers:
                self._subscribers.remove(callback)

    def publish(self, ticker: str, signal_dict: dict[str, Any]) -> None:
        """
        Publish one signal. Adds timestamp_utc if missing.
        Callbacks run in caller's thread; use a queue in consumer for async.
        """
        payload = dict(signal_dict)
        if "timestamp_utc" not in payload:
            payload["timestamp_utc"] = time.time()
        with self._lock:
            subs = list(self._subscribers)
            self._recent.append((ticker, payload, payload["timestamp_utc"]))
        for cb in subs:
            try:
                cb(ticker, payload)
            except Exception:
                pass

    def get_recent(self, n: int = 20) -> list[tuple[str, dict[str, Any]]]:
        """Return the n most recent (ticker, signal_dict) for dashboards."""
        with self._lock:
            items = list(self._recent)[-n:]
        return [(t, d) for t, d, _ in items]
