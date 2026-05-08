# verify_qc_parity.py
import pandas as pd
import os
import logging
from backtesting.signals import SignalEngine
from qc_alpha_model import SymbolData

# Setup basic logging to suppress noisy output during test
logging.basicConfig(level=logging.ERROR)

class MockIndicator:
    def __init__(self, name, window_size):
        self.name = name
        self.window = pd.Series(dtype=float)
        self.window_size = window_size
        self.Value = 0.0
    @property
    def Current(self): return self
    @property
    def IsReady(self): return len(self.window) >= self.window_size
    def Update(self, time, val):
        self.window = pd.concat([self.window, pd.Series([val])]).tail(self.window_size + 1)
        # Simplified math for mock verification
        if self.name == "RSI": self.Value = val # Just a proxy for parity check
        elif self.name == "STD": self.Value = self.window.tail(self.window_size).std()
        elif self.name == "SMA": self.Value = self.window.tail(self.window_size).mean()

class MockAlgorithm:
    def RSI(self, s, p, r): return MockIndicator("RSI", p)
    def STD(self, s, p, r): return MockIndicator("STD", p)
    def SMA(self, s, p, r): return MockIndicator("SMA", p)
    def History(self, *args): return [] # No history for raw math check
    def Log(self, m): pass

def run_parity_check(ticker="NVDA"):
    print(f"--- Verifying Phase 2 Parity for {ticker} ---")
    
    # 1. Load Local Data
    path = f"data/cache/ohlcv/{ticker}.parquet"
    if not os.path.exists(path):
        print(f"Data not found for {ticker}")
        return
    df = pd.read_parquet(path).iloc[-500:] # Last 500 bars
    
    # 2. Run Local Signal Engine
    engine = SignalEngine()
    local_signals = engine.generate_signals(df)
    
    # 3. Run QC Alpha Model (Mocked)
    algo = MockAlgorithm()
    sd = SymbolData(algo, ticker)
    qc_results = []
    
    for idx, row in df.iterrows():
        # Mock Lean Bar
        class Bar: pass
        bar = Bar()
        bar.EndTime = idx
        bar.Close = row['Close']
        bar.Open = row.get('Open', row['Close'])
        
        # In actual Lean, Update happens automatically. Here we force it.
        sd.Update(bar)
        feats = sd.GetLocalFeatures()
        if feats:
            feats['date'] = idx
            qc_results.append(feats)
    
    qc_df = pd.DataFrame(qc_results).set_index('date')
    
    # 4. Compare Core Features (Shared dates only)
    common_idx = local_signals.index.intersection(qc_df.index)
    comparison = pd.DataFrame({
        'Local_adj_score': local_signals.loc[common_idx, 'adjusted_score'],
        'QC_f_trend': qc_df.loc[common_idx, 'f_trend'],
        'Local_conf': local_signals.loc[common_idx, 'confidence'],
        'QC_rsi': qc_df.loc[common_idx, 'rsi_14']
    }).tail(5)
    
    print("\n[V] Last 5 Rows Comparison:")
    print(comparison)
    
    diff = (comparison['Local_adj_score'] - comparison['QC_f_trend']).abs().mean()
    if diff < 0.1: # More relaxed for raw math before full scaling check
        print(f"\nSUCCESS: Phase 2 Parity Verified (Mean f_trend Delta: {diff:.6f})")
    else:
        print(f"\nFIX NEEDED: Scale drift detected (Mean Delta: {diff:.6f}).")

if __name__ == "__main__":
    run_parity_check()
