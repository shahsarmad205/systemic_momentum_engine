from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.backtester import Backtester
from backtesting.config import BacktestConfig


def _make_price_frame(start: str, periods: int, base: float) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=periods)
    close = base + np.arange(periods, dtype=float) * 0.5
    return pd.DataFrame(
        {
            "Open": close * 0.999,
            "High": close * 1.001,
            "Low": close * 0.998,
            "Close": close,
            "AdjClose": close,
            "Volume": np.full(periods, 1_000_000.0),
        },
        index=idx,
    )


def _make_signal_frame(start: str, periods: int, score: float) -> pd.DataFrame:
    idx = pd.bdate_range(start=start, periods=periods)
    return pd.DataFrame(
        {
            "signal": ["Bullish"] * periods,
            "adjusted_score": [score] * periods,
            "confidence": ["High"] * periods,
            "smoothed_score": [score] * periods,
        },
        index=idx,
    )


def test_no_future_leakage_and_numeric_safety() -> None:
    cfg = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-01-31",
        initial_capital=100_000,
        max_positions=1,
        holding_period_days=1,
        min_holding_period_days=0,
        rebalance_every_trading_days=1,
        signal_mode="price",
        regime_enabled=False,
        long_only=True,
        position_sizing="equal_weight",
        position_sizing_method="equal_weight",
    )
    bt = Backtester(cfg)
    periods = 12
    price_data = {"AAPL": _make_price_frame("2024-01-01", periods, 100.0)}
    signal_data = {"AAPL": _make_signal_frame("2024-01-01", periods, 0.9)}
    regime_data = {d: "Bull" for d in price_data["AAPL"].index}

    res = bt.run_with_custom_signals(price_data, signal_data, regime_data)
    trades = res.trades
    assert not trades.empty

    # No lookahead: entries execute at next day open, so entry_date must be strictly after signal_date.
    s = pd.to_datetime(trades["signal_date"])
    e = pd.to_datetime(trades["entry_date"])
    assert (e > s).all()

    # Numeric safety
    assert np.isfinite(pd.to_numeric(trades["position_size"], errors="coerce")).all()
    assert np.isfinite(pd.to_numeric(trades["return"], errors="coerce")).all()

    # ~12% of equity cap; allow small slack when equity drifts up intra-window
    assert float(trades["position_size"].max()) <= 100_000.0 * 0.12 * 1.02 + 1e-6


def test_ranking_then_selection_top_name_preferred() -> None:
    cfg = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-01-31",
        initial_capital=100_000,
        max_positions=1,
        holding_period_days=1,
        min_holding_period_days=0,
        rebalance_every_trading_days=1,
        signal_mode="price",
        regime_enabled=False,
        long_only=True,
        position_sizing="equal_weight",
        position_sizing_method="equal_weight",
    )
    bt = Backtester(cfg)
    periods = 12
    price_data = {
        "AAPL": _make_price_frame("2024-01-01", periods, 100.0),
        "MSFT": _make_price_frame("2024-01-01", periods, 110.0),
    }
    signal_data = {
        "AAPL": _make_signal_frame("2024-01-01", periods, 1.0),
        "MSFT": _make_signal_frame("2024-01-01", periods, 0.2),
    }
    regime_data = {d: "Bull" for d in price_data["AAPL"].index}

    res = bt.run_with_custom_signals(price_data, signal_data, regime_data)
    trades = res.trades
    assert not trades.empty
    # With one slot and persistently higher score, higher-ranked name should dominate selections.
    counts = trades["ticker"].value_counts()
    assert counts.index[0] == "AAPL"
    assert counts.get("AAPL", 0) >= counts.get("MSFT", 0)

