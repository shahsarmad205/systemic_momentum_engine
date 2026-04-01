#!/usr/bin/env python3
"""Diagnose regime score std and Rank IC issues."""

import pandas as pd
import numpy as np
from backtesting.config import load_config
from backtesting.analytics import compute_rank_ic_decay

cfg = load_config('backtest_config.yaml')

# Check ensemble config
print("="*60)
print("ENSEMBLE CONFIG")
print("="*60)
if hasattr(cfg, 'signals'):
    sig_cfg = cfg.signals
    if isinstance(sig_cfg, dict):
        ens_cfg = sig_cfg.get('ensemble', {})
    else:
        ens_cfg = getattr(sig_cfg, 'ensemble', {})
    
    if isinstance(ens_cfg, dict):
        print(f"normalize: {ens_cfg.get('normalize', True)}")
        print(f"standardize: {ens_cfg.get('standardize', False)}")
        print(f"clip: {ens_cfg.get('clip', False)}")
    else:
        print(f"normalize: {getattr(ens_cfg, 'normalize', True)}")
        print(f"standardize: {getattr(ens_cfg, 'standardize', False)}")
        print(f"clip: {getattr(ens_cfg, 'clip', False)}")

# Load latest trades and daily equity
trades = pd.read_csv('output/backtests/trades.csv')
daily = pd.read_csv('output/backtests/daily_equity.csv')

print("\n" + "="*60)
print("REGIME SCORE STATISTICS")
print("="*60)

if 'adjusted_score' in trades.columns and 'regime' in trades.columns:
    regime_stats = trades.groupby('regime')['adjusted_score'].agg(['count', 'mean', 'std', 'min', 'max'])
    print("\nRegime Score Stats from Trades:")
    print(regime_stats.to_string())
    
    print("\nDetailed Regime Score Std:")
    for reg in ['Bull', 'Bear', 'Sideways', 'Crisis']:
        reg_scores = trades[trades['regime'] == reg]['adjusted_score']
        if len(reg_scores) > 0:
            print(f"  {reg:10s}: std={reg_scores.std():.6f}, mean={reg_scores.mean():.6f}, count={len(reg_scores)}")

print("\n" + "="*60)
print("DAILY EQUITY STATISTICS")
print("="*60)

daily['date'] = pd.to_datetime(daily['date'])
daily['ret'] = daily['equity'].pct_change()
print(f"\nDaily returns: mean={daily['ret'].mean():.6f}, std={daily['ret'].std():.6f}")
print(f"Date range: {daily['date'].min()} to {daily['date'].max()}")

if 'regime' in daily.columns:
    regime_daily_stats = daily.groupby('regime')['ret'].agg(['count', 'mean', 'std'])
    print("\nDaily Returns by Regime:")
    print(regime_daily_stats.to_string())

print("\n" + "="*60)
print("TRADE STATISTICS")
print("="*60)
print(f"Total trades: {len(trades)}")
print(f"Columns: {trades.columns.tolist()}")

if 'pnl' in trades.columns:
    by_regime = trades.groupby('regime')['pnl'].agg(['count', 'sum', 'mean', 'std'])
    print("\nPnL by Regime:")
    print(by_regime.to_string())

print("\n" + "="*60)
print("RANK IC DIAGNOSTIC")
print("="*60)
print("\nWarning: Rank IC typically computed from fwd returns vs signal scores.")
print("Need price_data (OHLCV) aligned with signal_data.")

# Show score distribution
if 'adjusted_score' in trades.columns:
    print(f"\nScore Distribution:")
    print(f"  Min: {trades['adjusted_score'].min():.6f}")
    print(f"  Q1:  {trades['adjusted_score'].quantile(0.25):.6f}")
    print(f"  Med: {trades['adjusted_score'].median():.6f}")
    print(f"  Q3:  {trades['adjusted_score'].quantile(0.75):.6f}")
    print(f"  Max: {trades['adjusted_score'].max():.6f}")
    print(f"  Std: {trades['adjusted_score'].std():.6f}")

# Check for signal inversion
if 'pnl' in trades.columns and 'adjusted_score' in trades.columns:
    corr = trades['adjusted_score'].corr(trades['pnl'])
    print(f"\nScore vs PnL Correlation: {corr:.6f}")
    if corr < -0.05:
        print("  ⚠️  INVERTED: Higher scores associated with LOWER pnl!")
    elif corr < 0.05:
        print("  ⚠️  WEAK: Score has almost no directional power.")
    else:
        print("  ✓ Score directionally aligned with pnl.")
