# Trend Signal Engine

A production-grade quantitative research and live-trading system for systematic momentum signal generation, ML ensemble scoring, portfolio optimization, and QuantConnect deployment.

---

## Overview

The Trend Signal Engine is an end‑to‑end framework designed to generate and backtest systematic equity signals using machine learning. It incorporates walk‑forward validation, cross‑sectional feature engineering, regime‑aware risk management, and realistic transaction costs. The engine is built for reproducibility and scales from local research to live trading on QuantConnect.

## Key Features

- **ML Ensemble Scoring** – Combines LGBMRanker, Ridge, and XGBoost models to produce long/short signals on S&P 500 constituents.
- **Robust Feature Pipeline** – 30+ technical and volatility indicators (RSI, rolling returns, volatility ratios, lagged features) normalised using cross‑sectional z‑scores to eliminate lookahead bias.
- **Walk‑Forward Validation** – 6 independent out‑of‑sample windows (2009‑2022) with mean OOS Sharpe **1.19**, 5/6 positive windows, and mean IC **+0.077** – statistically credible (Deflated Sharpe Ratio 1.00).
- **Regime‑Conditional Models** – Dedicated XGBoost classifiers for Bull, Bear, HighVol, and Normal regimes to adapt signal strength to market conditions.
- **Risk Management** – VIX deleveraging, factor neutralisation (sector, beta, size), dynamic position sizing, and circuit breakers to cap drawdowns.
- **Realistic Costs** – 1.0 bps slippage, $1 per trade commission, and a market impact model (Almgren‑Chriss).

---

## Current Production Baseline



| Metric | Value |
|--------|-------|
| **Sharpe Ratio** | **1.252** |
| Net Sharpe (HAC-Adjusted) | 0.944 |
| Sortino Ratio | 0.905 |
| **CAGR** | **11.55%** |
| **Max Drawdown** | **-17.88%** |
| Total Return | +230.03% |
| Win Rate (held-to-expiry) | **58.7%** |
| Avg Expiry Return | +2.28% |
| Trades | 2,399 |
| Trades per Year | 160 |
| T-statistic (Lo 2002) | 38.18 (p < 0.001) |
| Probabilistic Sharpe | 100% |
| Backtest Period | 2008–2022 (15 years) |
| Universe | S&P 500 (~493 tickers) |

**Year-by-Year Win Rates:**
2008: 47.5% | 2009: 66.5% | 2010: 66.7% | 2011: 50.8% | 2012: 58.6% | 2013: 69.7% | 2014: 61.9% | 2015: 43.0% | 2016: 67.3% | 2017: 61.1% | 2018: 52.7% | 2019: 59.0% | 2020: 72.6% | 2021: 61.5% | 2022: 73.1%

**Regime breakdown:**

| Regime | Win Rate | Sample |
|--------|----------|--------|
| Bull | 65.6% | n=1019 |
| Bear | 40.9%  | n=44 |
| Sideways | 68.8% | n=77 |
| Crisis | 56.4% | n=165 |



## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run full backtest (2008–2022, S&P 500 universe)
python run_backtest.py

# Run walk-forward OOS validation
python run_backtest.py --walk-forward

# Run ML model selection (trains VotingRegressor ensemble)
python run_model_selection.py

# Run ML model selection with risk-adjusted target (C3)
python run_model_selection.py --risk-adj-target

# Run ML model selection + regime-specific models (C2)
python run_model_selection.py --regime-models

# Auto-retrain (respects signal_mode from config — ml or learned)
python run_auto_retrain.py
```

---

## Architecture

### Signal Pipeline

```
Market Data (Yahoo/Tiingo)
    ↓
Feature Builder (agents/weight_learning_agent/feature_builder.py)
    23 features: momentum, volatility, CAPM, value, quality
    + forward_return_risk_adj (C3)
    ↓
Cross-Sectional Z-Score Panel (Pillar 29 / A1)
    Joint panel normalization — matches training distribution exactly
    ↓
ML Ensemble Scorer (utils/ensemble_scoring.py)
    VotingRegressor: XGBRegressor + LGBMRanker + Ridge
    + C2 Regime Routing: Bear → xgb_regime_bear.pkl
                         Crisis/HighVol → xgb_regime_highvol.pkl
    ↓
Signal Engine (backtesting/signals.py)
    Adjusted score + short_score_raw (dedicated short model)
    ↓
Cross-Sectional Ranking (backtesting/cross_sectional.py)
    Top-N longs by adjusted_score
    Shorts ranked by short_score_raw (dedicated short model — A2)
    ↓
Backtester (backtesting/backtester.py)
    Regime-aware sizing, factor neutralization, dynamic holding
```

### Regime Detection (backtesting/regime.py)

- **Hard labels**: Bull / Bear / Crisis / Sideways (SPY SMA-50/200 + VIX ≥ 30)
- **Continuous score** (D1): sigmoid on VIX + SMA gap → [0=Bull, 1=Crisis]
- Linear gross exposure interpolation: 100% in Bull → 25% in Crisis (no cliff edges)
- Confirmation window: 1-day (configurable hysteresis)

### Risk Controls

| Control | Mechanism |
|---------|-----------|
| D1 Continuous Regime Score | Linear gross cap interpolation vs hard label switches |
| D3 Sharpe Circuit Breaker | 60d rolling Sharpe < 0.0 → halve gross cap; recover at > 0.3 |
| Factor Neutralization (B1) | Market beta, sector, size neutralization per rebalance day |
| Bear Regime | Liquidate longs on entry, no new longs, 7-day max hold |
| Crisis Regime | Accelerated exit for losers, selective entries top-15% score only |
| Dynamic Holding (B2) | Hold longer for high-conviction signals, exit earlier for weak |
| Short Stop-Loss | 5% adverse move cap on all short positions |

---

## ML Model

### Production Ensemble (`output/models/best_long_model.pkl`)

- **Type**: `VotingRegressor` (sklearn)
- **Members**: XGBRegressor (60% weight) + LGBMRankerWrapper (cross-sectional lambdarank) + Ridge (40% weight)
- **Target**: `forward_return` over 10-day horizon (or `forward_return_risk_adj` with `--risk-adj-target`)
- **Features** (23):

```
vol_expansion, f_trend, short_term_reversal, ret_20d, ret_1d,
vol_ratio_5_20, nearness_52w_low, sector_relative_60d,
nearness_52w_high, cs_momentum_percentile,
dist_from_52w_high, low_vol_score, momentum_acceleration,
ret_5d, capm_residual_vol, down_up_vol_ratio, rsi_14,
capm_alpha, ret_10d, sector_relative_20d, rsi_overbought,
rolling_vol_20, quality_score
```

### Regime-Conditional Models (C2)

| Model | Regime | Architecture |
|-------|--------|--------------|
| `xgb_regime_bear.pkl` | Bear | XGBClassifier |
| `xgb_regime_highvol.pkl` | Crisis / HighVol | XGBClassifier |
| `best_long_model.pkl` | Bull / Sideways / Normal | VotingRegressor |

Routing only activates for Bear/Crisis/HighVol. Bull (82% of days) always uses the superior general ensemble.

### Short Model (`output/models/best_short_model.pkl`)

Dedicated short candidate scorer. Output (`short_score_raw`) is stored per ticker in the signal DataFrame and consumed by `cross_sectional.py` to rank short candidates independently from the long score.

---

## Key Implementation Details

### A1 — Training/Inference Feature Parity
Training uses cross-sectional z-scores across the daily panel (`_apply_cross_sectional_zscore_columns`). Inference now applies the same transformation via `Pillar 29` joint-panel vectorization in `backtester.py:844–888` before calling `generate_signals`.

### A2 — Short Model Wiring
`cross_sectional.py` checks for non-zero `short_score_raw` values in the signal DataFrame. If present, shorts are ranked by the dedicated short model score rather than the bottom of the long-rank list.

### B3 — Walk-Forward Ensemble
`run_auto_retrain.py` respects `signal_mode` from config. In `ml` mode it runs OOS evaluation without overwriting model files. In `learned` mode it applies exponential time-decay blending (λ=0.7) across walk-forward windows.

### C3 — Risk-Adjusted Target
`forward_return_risk_adj = forward_return / (vol_20d × √holding_period)` — rewards high-return, low-volatility stocks. Enabled via `python run_model_selection.py --risk-adj-target`.

---

## Configuration

Key settings in `backtest_config.yaml`:

```yaml
signals:
  mode: ml
  ml_long_model_path: output/models/best_long_model.pkl
  ml_short_model_path: output/models/best_short_model.pkl
  ml_regime_models_dir: output/models   # C2: regime routing

regime:
  confirmation_days: 1
  continuous_score_enabled: true        # D1: linear gross cap
  score_bull_gross_cap: 1.0
  score_crisis_gross_cap: 0.25

risk:
  max_drawdown_pct: 0.20                # hard circuit breaker
  sharpe_circuit_breaker:               # D3: rolling Sharpe CB
    enabled: true
    window_days: 60
    threshold: 0.0
    recovery_threshold: 0.3
    exposure_scale: 0.5
```

---

## QuantConnect Deployment

The strategy is deployed to QuantConnect via `LeanCloud/BinaryEdge/`:

```
LeanCloud/BinaryEdge/
├── main.py                 # QC algorithm (regime detection, D3 Sharpe CB)
├── qc_alpha_model.py       # Alpha model (24-feature inference, C2 routing, D1 scaling)
├── best_long_model.pkl     # VotingRegressor ensemble (13MB)
├── xgb_regime_bear.pkl     # C2: Bear regime specialist
├── xgb_regime_highvol.pkl  # C2: Crisis/HighVol specialist
└── config.json
```

**Sync local → QC:**
```bash
lean cloud push
lean backtest "LeanCloud/BinaryEdge"
```

**Key QC implementation notes:**
- `LGBMRankerWrapper` is defined in `qc_alpha_model.py` and injected into `__main__` before unpickling (required for VotingRegressor deserialization)
- `sector_relative_20d/60d` approximated as stock return minus universe mean return (no sector feed in QC)
- `earnings_surprise` removed from model (was always 0.0 in QC — training/inference gap eliminated)
- D3 Sharpe CB implemented in `OnEndOfDay` — reduces `top_n` when 60d Sharpe < 0

---

## Phases Implemented

| Phase | Item | Status | Impact |
|-------|------|--------|--------|
| A | A1 CS z-score at inference | ✅ Wired | Closes training/inference gap |
| A | A2 Short model wiring | ✅ Wired | Dedicated short ranking |
| A | A3 Sector mapping (500 tickers) | ✅ Wired | sector_relative features valid |
| B | B1 Factor neutralization | ✅ Wired | Reduces regime-driven drawdowns |
| B | B2 Dynamic holding period | ✅ Wired | Higher conviction → longer hold |
| B | B3 Walk-forward ensemble | ✅ Wired | ML mode: OOS eval; Learned mode: decay blend |
| C | C1 Earnings surprise | ✅ Training + Inference | Yfinance per-ticker, cached |
| C | C2 Regime-conditional models | ✅ Wired | Bear/Crisis specialist routing |
| C | C3 Risk-adjusted target | ✅ Wired | `--risk-adj-target` flag |
| D | D1 Continuous regime score | ✅ Wired | Linear gross cap interpolation |
| D | D2 Short book sizing (0.7×) | ✅ Wired | Snap profit raised to 5% |
| D | D3 Sharpe circuit breaker | ✅ Wired | 60d Sharpe → halve exposure |

---

## Next Steps

1. **Retrain model** — `earnings_surprise` removed from all layers; retrain to get a clean 23-feature model:
   ```bash
   python run_model_selection.py --risk-adj-target
   ```
2. **Run local backtest** — confirm Sharpe ≥ 1.0 with shorts disabled + new model
3. **Sync + QC backtest** — `lean cloud push && lean backtest "LeanCloud/BinaryEdge"`
4. **Walk-forward OOS validation** — confirm Sharpe ≥ 0.7 across all OOS windows
5. **Retrain regime models as regressors** — current `xgb_regime_*.pkl` are classifiers; matching the general VotingRegressor architecture would improve C2 quality
6. **Live paper trading** — deploy after QC validation

---

## Project Structure

```
trend_signal_engine/
├── backtesting/
│   ├── backtester.py           # Main simulation loop
│   ├── signals.py              # Signal engine + ML inference
│   ├── cross_sectional.py      # CS ranking, short model routing (A2)
│   ├── regime.py               # Regime detection + continuous score (D1)
│   └── config.py               # BacktestConfig dataclass
├── agents/weight_learning_agent/
│   ├── feature_builder.py      # 24-feature panel builder, C1/C3
│   └── weight_model.py         # LearnedWeights (legacy learned mode)
├── utils/
│   └── ensemble_scoring.py     # Model loading + ensemble inference
├── research/
│   └── factor_neutralization.py
├── run_model_selection.py       # ML training: VotingRegressor, C2, C3
├── run_auto_retrain.py          # Auto-retrain respecting signal_mode
├── run_backtest.py              # Backtest entry point
├── backtest_config.yaml         # All configuration
├── LeanCloud/BinaryEdge/        # QuantConnect deployment
│   ├── main.py
│   ├── qc_alpha_model.py
│   ├── best_long_model.pkl
│   ├── xgb_regime_bear.pkl
│   └── xgb_regime_highvol.pkl
└── output/
    ├── models/                  # Trained model artifacts
    ├── backtests/               # Equity curves, trades, summaries
    └── experiments/             # Timestamped experiment snapshots
```

---

## Disclaimer

This system is a research tool. Nothing here constitutes investment advice. Past backtest performance does not guarantee future results.
