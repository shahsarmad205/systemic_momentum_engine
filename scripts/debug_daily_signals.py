"""
Debug script: patch _simulate to print daily_signals vs trading_days key match,
then run backtest with correct orig_simulate call (4 args, no extra write_exposure_diagnostics).
Run from project root: python scripts/debug_daily_signals.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import backtesting.backtester as bmod
from backtesting.backtester import Backtester
from backtesting.config import load_config
from config import get_effective_tickers
from main import TICKERS

cfg = load_config("backtest_config.yaml")
tickers = get_effective_tickers(cfg.tickers or [], list(TICKERS))

def _to_calendar_key(ts):
    import datetime
    if isinstance(ts, datetime.date) and not isinstance(ts, datetime.datetime):
        return ts.isoformat()
    return pd.Timestamp(ts).strftime("%Y-%m-%d")

orig_simulate = bmod.Backtester._simulate

def patched_simulate(self, price_data, signal_data, regime_data):
    # Build same daily_signals and trading_days as real _simulate to inspect keys
    from collections import defaultdict
    start_ts = pd.Timestamp(self.config.start_date)
    end_ts = pd.Timestamp(self.config.end_date)
    from datetime import timedelta

    from backtesting.signals import EXIT_BUFFER_DAYS
    sim_end = end_ts + timedelta(days=EXIT_BUFFER_DAYS)
    cross_sectional = getattr(self.config, "cross_sectional_ranking", False)
    daily_signals = defaultdict(list)
    for ticker, sig_df in signal_data.items():
        mask = (sig_df.index >= start_ts) & (sig_df.index <= end_ts)
        for date, row in sig_df[mask].iterrows():
            d = _to_calendar_key(date)
            if cross_sectional:
                daily_signals[d].append((ticker, row))
            elif row["signal"] != "Neutral":
                daily_signals[d].append((ticker, row))
    print(f"daily_signals keys: {len(daily_signals)}")
    sample = list(daily_signals.keys())[:3]
    print(f"Sample keys: {sample} type={type(sample[0])}")
    print(f"Total entries: {sum(len(v) for v in daily_signals.values())}")
    all_dates = set()
    for df in price_data.values():
        all_dates.update(df.index)
    trading_days = sorted(d for d in all_dates if start_ts <= d <= sim_end)
    sample_td = [_to_calendar_key(d) for d in trading_days[:3]]
    print(f"Sample trading day keys: {sample_td}")
    print(f"First trading day in daily_signals: {sample_td[0] in daily_signals if sample_td else False}")
    return orig_simulate(self, price_data, signal_data, regime_data)

bmod.Backtester._simulate = patched_simulate
bt = Backtester(cfg)
result = bt.run(tickers)
trades = result.trades
eq = result.daily_equity
print(f"Total trades: {len(trades)}")
print(f"Max positions: {eq.n_positions.max()}")
print(f"Days with 0 positions: {(eq.n_positions==0).sum()} / {len(eq)}")
print(f"Tickers traded: {trades.ticker.nunique()}")
