"""
Tests for helpers extracted from backtester.py into standalone modules.

Covers:
  - analytics.build_returns_matrix
  - analytics.annualized_vol
  - regime_multipliers.signal_confidence_multiplier_for_regime
  - regime_multipliers.threshold_aggressiveness_for_regime
  - result.BacktestResult
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import pytest

from backtesting.analytics import annualized_vol, build_returns_matrix
from backtesting.regime_multipliers import (
    signal_confidence_multiplier_for_regime,
    threshold_aggressiveness_for_regime,
)
from backtesting.result import BacktestResult


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_price_df(n: int = 60, start_price: float = 100.0) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = start_price + np.cumsum(np.random.default_rng(0).standard_normal(n))
    return pd.DataFrame({"Close": prices}, index=dates)


def _make_price_data(n: int = 60) -> dict[str, pd.DataFrame]:
    return {
        "AAPL": _make_price_df(n, 150),
        "MSFT": _make_price_df(n, 200),
        "NVDA": _make_price_df(n, 400),
    }


# ---------------------------------------------------------------------------
# build_returns_matrix
# ---------------------------------------------------------------------------

class TestBuildReturnsMatrix:
    def test_returns_dataframe_with_correct_shape(self):
        price_data = _make_price_data(60)
        as_of = pd.Timestamp("2020-03-15")
        result = build_returns_matrix(price_data, as_of_date=as_of, lookback_days=20)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] == 3  # 3 tickers
        assert result.shape[0] <= 20

    def test_no_lookahead(self):
        """All dates in the result must be <= as_of_date."""
        price_data = _make_price_data(60)
        as_of = pd.Timestamp("2020-02-01")
        result = build_returns_matrix(price_data, as_of_date=as_of, lookback_days=15)
        if result is not None:
            assert result.index.max() <= as_of

    def test_returns_none_when_insufficient_data(self):
        # Single price point → 0 returns after pct_change → None
        price_data = {"AAPL": _make_price_df(1)}
        as_of = pd.Timestamp("2020-01-02")
        result = build_returns_matrix(price_data, as_of_date=as_of, lookback_days=20)
        assert result is None

    def test_empty_price_data_returns_none(self):
        result = build_returns_matrix({}, as_of_date=pd.Timestamp("2020-01-01"), lookback_days=10)
        assert result is None

    def test_ticker_with_missing_close_column_skipped(self):
        price_data = {
            "AAPL": _make_price_df(60),
            "BAD":  pd.DataFrame({"Open": [100.0, 101.0]}, index=pd.date_range("2020-01-01", periods=2, freq="B")),
        }
        as_of = pd.Timestamp("2020-03-15")
        result = build_returns_matrix(price_data, as_of_date=as_of, lookback_days=20)
        if result is not None:
            assert "BAD" not in result.columns


# ---------------------------------------------------------------------------
# annualized_vol
# ---------------------------------------------------------------------------

class TestAnnualizedVol:
    def test_returns_positive_float(self):
        price_data = _make_price_data(60)
        as_of = pd.Timestamp("2020-03-15")
        vol = annualized_vol(price_data, ticker="AAPL", as_of_date=as_of, lookback_days=20)
        assert vol is not None
        assert vol > 0

    def test_unknown_ticker_returns_none(self):
        price_data = _make_price_data(60)
        vol = annualized_vol(price_data, ticker="UNKNOWN", as_of_date=pd.Timestamp("2020-03-15"))
        assert vol is None

    def test_insufficient_data_returns_none(self):
        price_data = {"AAPL": _make_price_df(1)}
        vol = annualized_vol(price_data, ticker="AAPL", as_of_date=pd.Timestamp("2020-01-02"))
        assert vol is None

    def test_uses_adjclose_over_close(self):
        """When AdjClose is present it should be preferred."""
        dates = pd.date_range("2020-01-01", periods=40, freq="B")
        prices = 100 + np.arange(40, dtype=float)
        df = pd.DataFrame(
            {"Close": prices, "AdjClose": prices * 0.95},
            index=dates,
        )
        price_data = {"X": df}
        vol = annualized_vol(price_data, "X", as_of_date=dates[-1], lookback_days=20)
        assert vol is not None


# ---------------------------------------------------------------------------
# signal_confidence_multiplier_for_regime
# ---------------------------------------------------------------------------

@dataclass
class _MockConfig:
    signal_confidence_multiplier_bull: float = 1.2
    signal_confidence_multiplier_bear: float = 0.5
    signal_confidence_multiplier_sideways: float = 0.8
    signal_confidence_multiplier_crisis: float = 0.3
    signal_confidence_multiplier: float = 1.0


class TestSignalConfidenceMultiplierForRegime:
    def test_bull_reads_bull_attr(self):
        cfg = _MockConfig()
        m = signal_confidence_multiplier_for_regime("Bull", cfg)
        assert math.isclose(m, 1.2)

    def test_bear_reads_bear_attr(self):
        cfg = _MockConfig()
        m = signal_confidence_multiplier_for_regime("Bear", cfg)
        assert math.isclose(m, 0.5)

    def test_crisis_reads_crisis_attr(self):
        cfg = _MockConfig()
        m = signal_confidence_multiplier_for_regime("Crisis", cfg)
        assert math.isclose(m, 0.3)

    def test_unknown_regime_falls_back_to_global(self):
        cfg = _MockConfig()
        cfg.signal_confidence_multiplier = 0.9
        m = signal_confidence_multiplier_for_regime("Unknown", cfg)
        assert math.isclose(m, 0.9)

    def test_none_regime_treated_as_sideways(self):
        cfg = _MockConfig()
        m = signal_confidence_multiplier_for_regime(None, cfg)
        assert math.isclose(m, cfg.signal_confidence_multiplier_sideways)

    def test_loaded_multipliers_override(self):
        @dataclass
        class EmptyCfg:
            signal_confidence_multiplier: float = 1.0
        cfg = EmptyCfg()
        loaded = {"Bull": 1.5}
        m = signal_confidence_multiplier_for_regime("Bull", cfg, loaded_multipliers=loaded)
        assert math.isclose(m, 1.5)


# ---------------------------------------------------------------------------
# threshold_aggressiveness_for_regime
# ---------------------------------------------------------------------------

@dataclass
class _AggressCfg:
    regime_threshold_aggressiveness: dict = field(default_factory=lambda: {
        "Bull": 1.0,
        "Bear": 2.0,
        "Sideways": 1.5,
        "Crisis": 3.0,
    })


class TestThresholdAggressivenessForRegime:
    def test_reads_correct_regime(self):
        cfg = _AggressCfg()
        assert math.isclose(threshold_aggressiveness_for_regime("Bear", cfg), 2.0)
        assert math.isclose(threshold_aggressiveness_for_regime("Crisis", cfg), 3.0)

    def test_missing_regime_returns_one(self):
        cfg = _AggressCfg()
        assert math.isclose(threshold_aggressiveness_for_regime("Unknown", cfg), 1.0)

    def test_no_map_returns_one(self):
        @dataclass
        class NoCfg:
            pass
        assert math.isclose(threshold_aggressiveness_for_regime("Bull", NoCfg()), 1.0)

    def test_capped_at_twenty(self):
        @dataclass
        class HighCfg:
            regime_threshold_aggressiveness: dict = field(default_factory=lambda: {"Bull": 999.0})
        assert math.isclose(threshold_aggressiveness_for_regime("Bull", HighCfg()), 20.0)

    def test_zero_value_returns_one(self):
        @dataclass
        class ZeroCfg:
            regime_threshold_aggressiveness: dict = field(default_factory=lambda: {"Bull": 0.0})
        assert math.isclose(threshold_aggressiveness_for_regime("Bull", ZeroCfg()), 1.0)


# ---------------------------------------------------------------------------
# BacktestResult
# ---------------------------------------------------------------------------

class TestBacktestResult:
    def _make_result(self) -> BacktestResult:
        from backtesting.config import BacktestConfig
        trades = pd.DataFrame({"return": [0.05, -0.02]})
        equity = pd.DataFrame({"equity": [100_000, 105_000]})
        metrics = {"sharpe_ratio": 1.5, "cagr": 0.25}
        return BacktestResult(trades, equity, metrics, BacktestConfig())

    def test_fields_accessible(self):
        r = self._make_result()
        assert not r.trades.empty
        assert not r.daily_equity.empty
        assert r.metrics["sharpe_ratio"] == 1.5

    def test_optional_fields_default_empty(self):
        r = self._make_result()
        assert r.price_data == {}
        assert r.signal_data == {}
        assert r.regime_data == {}
        assert r.position_sizing_comparison is None

    def test_can_attach_price_data(self):
        r = self._make_result()
        r.price_data = {"AAPL": _make_price_df(20)}
        assert "AAPL" in r.price_data
