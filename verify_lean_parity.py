"""
Lean vs. Yahoo Parity Test
==========================
Verifies that the new Lean Data Provider correctly rescales and aligns
data compared to the legacy Yahoo Finance provider.
"""

import pandas as pd
import numpy as np
from utils.market_data import get_ohlcv

def verify_parity(ticker="AAPL", days=100):
    print(f"--- Verifying Lean vs. Yahoo Parity for {ticker} (Last {days} days) ---")
    
    end_date = pd.Timestamp.now().strftime("%Y-%m-%d")
    start_date = (pd.Timestamp.now() - pd.Timedelta(days=days+200)).strftime("%Y-%m-%d") # extra buffer
    
    # 1. Fetch from Lean
    print(f"Fetching from Lean...")
    df_lean = get_ohlcv(ticker, start_date, end_date, provider="lean", use_cache=False)
    
    # 2. Fetch from Yahoo
    print(f"Fetching from Yahoo...")
    df_yahoo = get_ohlcv(ticker, start_date, end_date, provider="yahoo", use_cache=False)
    
    if df_lean.empty or df_yahoo.empty:
        print(f"[ERROR] One or both dataframes are empty. Lean: {df_lean.shape}, Yahoo: {df_yahoo.shape}")
        return

    # Align indices
    common_idx = df_lean.index.intersection(df_yahoo.index)
    if len(common_idx) == 0:
        print("[ERROR] No overlapping dates found.")
        return
        
    lean = df_lean.loc[common_idx]
    yahoo = df_yahoo.loc[common_idx]
    
    print(f"Comparing {len(common_idx)} overlapping dates...")
    
    # Compare Close Prices
    delta = (lean["Close"] - yahoo["Close"]).abs()
    mean_delta = delta.mean()
    max_delta = delta.max()
    
    # Tolerances: Yahoo adjusted close vs Lean standard close might vary slightly 
    # due to precision or corporate actions handling, but should be near-zero for non-adjusted.
    print(f"\n[RESULTS] Close Price Delta:")
    print(f"  Mean Delta: {mean_delta:.6f}")
    print(f"  Max Delta:  {max_delta:.6f}")
    
    if mean_delta < 0.05:
        print("\nSUCCESS: Lean and Yahoo data are synchronized (within tolerance).")
    else:
        print("\nWARNING: Significant divergence detected. Check for split or adjustment differences.")
        
    print("\nSample Comparison (Last 5 days):")
    comp = pd.DataFrame({
        "Lean_Close": lean["Close"],
        "Yahoo_Close": yahoo["Close"],
        "Delta": delta
    }).tail(5)
    print(comp.to_string())

if __name__ == "__main__":
    verify_parity("AAPL", days=500)
