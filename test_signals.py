from backtesting.config import load_config
from backtesting.signals import SignalEngine
from utils.market_data import get_ohlcv

cfg = load_config("backtest_config.yaml")
engine = SignalEngine(weights=cfg.signal_weights, config=cfg)

df = get_ohlcv("AAPL", "2020-01-01", "2023-01-01", use_cache=True, cache_ttl_days=0)
sigs = engine.generate_signals(df)

counts = sigs["signal"].value_counts()
print(counts)
