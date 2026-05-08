from __future__ import annotations

import numpy as np
import pandas as pd

from backtesting.regime import MarketRegimeAgent


def test_market_regime_agent_wrds_avoids_market_data_downloads(monkeypatch):
    index = pd.bdate_range("2019-01-01", periods=320)
    close = pd.Series(np.linspace(100.0, 140.0, len(index)), index=index)
    spy = pd.DataFrame(
        {
            "Open": close,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": 1_000_000.0,
        },
        index=index,
    )

    monkeypatch.setattr(
        "backtesting.regime.load_wrds_price_panel",
        lambda tickers, **kwargs: {"SPY": spy},
    )
    monkeypatch.setattr(
        "backtesting.regime.get_ohlcv",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("get_ohlcv should not be called in WRDS mode")),
    )

    agent = MarketRegimeAgent(data_provider="wrds", wrds_username="tester")
    regimes = agent.detect_regimes("2020-01-02", "2020-12-31")

    assert regimes
    assert set(regimes.values()).issubset({"Bull", "Bear", "Cautious", "Sideways", "Crisis"})

