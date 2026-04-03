
import pandas as pd
import numpy as np
from backtesting.config import load_config
from backtesting.signals import SignalEngine

def test_inversion():
    config = load_config("backtest_config.yaml")
    # Force direction = 1.0 first
    config.score_direction = 1.0
    
    # Mock some stock data
    dates = pd.date_range("2020-01-01", periods=100)
    data = pd.DataFrame({
        "Open": np.random.randn(100) + 100,
        "High": np.random.randn(100) + 101,
        "Low": np.random.randn(100) + 99,
        "Close": np.random.randn(100) + 100,
        "Volume": np.random.randn(100) * 1000 + 10000
    }, index=dates)
    data["AdjClose"] = data["Close"]
    
    engine = SignalEngine(config=config)
    signals1 = engine.generate_signals(data)
    score1 = signals1["adjusted_score"].iloc[-1]
    
    # Now flip direction
    config.score_direction = -1.0
    # Clear cache to be sure
    engine._signal_cache = {}
    signals2 = engine.generate_signals(data)
    score2 = signals2["adjusted_score"].iloc[-1]
    
    print(f"Score (dir=1.0): {score1:.4f}")
    print(f"Score (dir=-1.0): {score2:.4f}")
    
    if score1 == -score2:
        print("SUCCESS: Inversion is working in SignalEngine.")
    else:
        print("FAILURE: Inversion is NOT working.")

if __name__ == "__main__":
    test_inversion()
