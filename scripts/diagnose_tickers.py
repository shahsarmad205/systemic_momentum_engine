"""
Quick diagnostic: how many tickers have signals and data in the backtest.
Run from project root: python scripts/diagnose_tickers.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backtesting.config import load_config
from main import TICKERS
from config import get_effective_tickers
from utils.market_data import get_ohlcv
from backtesting.signals import SignalEngine
import pandas as pd
from collections import defaultdict

def _to_calendar_key(ts) -> str:
    return pd.Timestamp(ts).strftime("%Y-%m-%d")

cfg = load_config("backtest_config.yaml")
tickers = get_effective_tickers(cfg.tickers or [], list(TICKERS))
print(f"Config tickers (effective): {len(tickers)} → {tickers[:5]}...")

# Load price data (same as backtester)
start_ts = pd.Timestamp(cfg.start_date)
end_ts = pd.Timestamp(cfg.end_date)
start_str = start_ts.strftime("%Y-%m-%d")
end_str = end_ts.strftime("%Y-%m-%d")
provider = getattr(cfg, "data_provider", "yahoo")
cache_dir = getattr(cfg, "cache_dir", "data/cache/ohlcv")
cache_ttl = int(getattr(cfg, "cache_ttl_days", 0) or 0)
use_cache = getattr(cfg, "cache_ohlcv", True)
price_data = {}
for tk in tickers:
    df = get_ohlcv(
        tk,
        start_str,
        end_str,
        provider=provider,
        cache_dir=cache_dir,
        use_cache=use_cache,
        cache_ttl_days=cache_ttl,
    )
    if df is not None and not df.empty:
        price_data[tk] = df
print(f"Tickers with price data: {len(price_data)} → {list(price_data.keys())}")

# Build signals (same as backtester: load learned weights if needed, then generate per ticker)
learned_weights = None
regime_weights = None
regime_series = None
if getattr(cfg, "signal_mode", "price") == "learned" and getattr(cfg, "learned_weights_path", ""):
    try:
        from agents.weight_learning_agent.weight_model import load_regime_weights
        regime_weights, _ = load_regime_weights(cfg.learned_weights_path)
        if regime_weights is None:
            from agents.weight_learning_agent import LearnedWeights
            learned_weights = LearnedWeights.load(cfg.learned_weights_path)
    except Exception:
        pass
signal_engine = SignalEngine(
    weights=getattr(cfg, "signal_weights", None),
    learned_weights=learned_weights,
    regime_weights=regime_weights,
    regime_series=regime_series,
    signal_smoothing_enabled=getattr(cfg, "signal_smoothing_enabled", True),
    signal_smoothing_span=int(getattr(cfg, "signal_smoothing_span", 5)),
)
signal_engine.config = cfg
signal_data = {}
for tk, data in price_data.items():
    sig_df = signal_engine.generate_signals(data)
    if sig_df is not None and not sig_df.empty:
        signal_data[tk] = sig_df
print(f"Tickers with signal data: {len(signal_data)} → {list(signal_data.keys())}")

# Count non-Neutral signals per ticker (daily_signals style)
daily_signals = defaultdict(list)
for ticker, sig_df in signal_data.items():
    mask = (sig_df.index >= start_ts) & (sig_df.index <= end_ts)
    for date, row in sig_df[mask].iterrows():
        if row["signal"] != "Neutral":
            daily_signals[_to_calendar_key(date)].append((ticker, row))

tickers_with_signals = set()
for day_entries in daily_signals.values():
    for tk, _ in day_entries:
        tickers_with_signals.add(tk)
print(f"Tickers with at least one non-Neutral signal: {len(tickers_with_signals)} → {sorted(tickers_with_signals)}")
print(f"Days with at least one signal: {len(daily_signals)}")
print(f"Total signal entries: {sum(len(v) for v in daily_signals.values())}")
