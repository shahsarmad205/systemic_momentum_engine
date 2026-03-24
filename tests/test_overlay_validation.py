from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from backtesting.backtester import Backtester
from backtesting.config import BacktestConfig
from utils.vol_sizing import apply_vol_kill_switch


def _mk_backtester() -> Backtester:
    cfg = BacktestConfig(
        start_date="2024-01-01",
        end_date="2024-01-31",
        max_positions=10,
        rank_transform_enabled=True,
        high_conviction_threshold=0.6,
        portfolio_top_k=5,
        crisis_portfolio_top_k=4,
        normal_exposure=1.0,
        crisis_exposure=0.25,
        long_only=True,
    )
    return Backtester(cfg)


def _entries(scores: list[float]) -> list[dict]:
    out = []
    for i, s in enumerate(scores):
        out.append(
            {
                "ticker": f"T{i}",
                "adjusted_score": float(s),
                "signal": "Bullish",
                "confidence": "High",
                "signal_date": pd.Timestamp("2024-01-02"),
                "exit_date": pd.Timestamp("2024-01-10"),
                "position_scale": 1.0,
            }
        )
    return out


def test_crisis_exposure_and_breadth_reduction() -> None:
    bt = _mk_backtester()
    base = _entries([0.95, 0.9, 0.8, 0.7, 0.62, 0.5, 0.4])
    normal = bt._apply_portfolio_improvements(base, regime_today="Bull")
    crisis = bt._apply_portfolio_improvements(base, regime_today="Crisis")

    assert len(normal) <= 5
    assert len(crisis) <= 4
    normal_exp = sum(float(e.get("portfolio_weight", 0.0)) for e in normal)
    crisis_exp = sum(float(e.get("portfolio_weight", 0.0)) for e in crisis)
    assert normal_exp == 1.0
    assert crisis_exp == 0.25
    assert crisis_exp < normal_exp


def test_volatility_spike_kill_switch_reduces_exposure() -> None:
    positions = pd.Series([10000.0, 10000.0, 10000.0, 10000.0])
    realized_vol = pd.Series([0.15, 0.18, 0.30, 0.35])
    out = apply_vol_kill_switch(positions, realized_vol, threshold_annual=0.25, cut_factor=0.5)
    assert out.iloc[0] == 10000.0
    assert out.iloc[1] == 10000.0
    assert out.iloc[2] == 5000.0
    assert out.iloc[3] == 5000.0


def test_flat_signals_stable_and_threshold_logic() -> None:
    bt = _mk_backtester()
    flat = _entries([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    out = bt._apply_portfolio_improvements(flat, regime_today="Bull")
    # all equal ranks (0.5) fail >0.6 threshold; fallback should still produce a stable top-k basket
    assert len(out) == 5
    w = [float(e.get("portfolio_weight", 0.0)) for e in out]
    assert all(v >= 0 for v in w)
    assert abs(sum(w) - 1.0) < 1e-12

