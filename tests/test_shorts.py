import pandas as pd

from backtesting.portfolio import Portfolio
from backtesting.config import BacktestConfig
from strategy.candidates import build_ranked_candidates


def test_ranked_candidates_respects_max_longs_max_shorts():
    cfg = BacktestConfig()
    cfg.enable_shorts = True
    cfg.long_only = False
    cfg.max_positions = 6
    cfg.max_longs = 2
    cfg.max_shorts = 4

    date = pd.Timestamp("2024-01-10")
    trading_days = [date + pd.Timedelta(days=i) for i in range(10)]

    # Build fake daily_signals_at_date as list[(ticker, sig_row)] matching candidates.py expectation
    rows = []
    # 5 longs
    for i in range(5):
        r = pd.Series({"adjusted_score": 1.0 + i, "signal": "Bullish", "confidence": "HIGH"})
        rows.append((f"L{i}", r))
    # 5 shorts
    for i in range(5):
        r = pd.Series({"adjusted_score": -(1.0 + i), "signal": "Bearish", "confidence": "HIGH"})
        rows.append((f"S{i}", r))

    out = build_ranked_candidates(
        date=date,
        daily_signals_at_date=rows,
        config=cfg,
        trading_days=trading_days,
        day_index=0,
        regime="Sideways",
        ticker_trade_counts={},
    )

    assert len(out) <= cfg.max_positions
    n_longs = sum(1 for c in out if c["signal"] == "Bullish")
    n_shorts = sum(1 for c in out if c["signal"] == "Bearish")
    assert n_longs <= cfg.max_longs
    assert n_shorts <= cfg.max_shorts


def test_short_borrow_cost_applies_daily_drag():
    p = Portfolio(initial_capital=100_000, max_positions=10)
    # Open a short position (Bearish => direction=-1). Use entry_cost=0 to isolate borrow drag.
    pos = p.open_position(
        ticker="TEST",
        signal="Bearish",
        signal_date=pd.Timestamp("2024-01-02"),
        entry_date=pd.Timestamp("2024-01-03"),
        planned_exit_date=pd.Timestamp("2024-01-10"),
        entry_price=100.0,
        adjusted_score=-1.0,
        confidence="HIGH",
        regime="Sideways",
        entry_cost=0.0,
        size_dollars=10_000.0,
    )
    assert pos is not None
    # Mark-to-market unchanged.
    pos.current_price = 100.0

    equity_before = p.equity
    # Apply one day of borrow cost at 50 bps annualized.
    borrow_bps = 50.0
    daily_rate = (borrow_bps / 10_000.0) / 252.0
    short_notional = abs(float(pos.market_value))
    borrow_cost = short_notional * daily_rate
    p.cash -= borrow_cost
    p.record_equity(pd.Timestamp("2024-01-04"), short_borrow_cost=borrow_cost)

    assert p.equity < equity_before
    assert p.equity_history[-1]["short_borrow_cost"] == borrow_cost

