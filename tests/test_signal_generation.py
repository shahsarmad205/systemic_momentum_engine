import numpy as np
import pandas as pd

from backtesting.signals import SignalEngine


def _make_dummy_ohlcv(n: int = 50) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = np.linspace(100.0, 120.0, n)
    df = pd.DataFrame(
        {
            "Open": prices * 0.99,
            "High": prices * 1.01,
            "Low": prices * 0.98,
            "Close": prices,
            "AdjClose": prices,
            "Volume": np.full(n, 1_000_000, dtype=float),
        },
        index=dates,
    )
    return df


def test_signal_engine_generates_signals_basic():
    stock_data = _make_dummy_ohlcv()
    engine = SignalEngine()

    signals = engine.generate_signals(stock_data)

    # Should produce a non-empty DataFrame with core columns.
    assert not signals.empty
    for col in ("trend_score", "confidence", "adjusted_score", "signal"):
        assert col in signals.columns

    # Signals should be in the allowed set.
    allowed = {"Bullish", "Bearish", "Neutral"}
    assert signals["signal"].isin(allowed).all()

