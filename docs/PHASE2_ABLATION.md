# Phase 2: VIX term scale + feature ablation

## 1. `vix_term_zscore` (training & live)

Raw VIX/VIX3M ratio (~0.8–1.3) is replaced in the feature matrix by a **252d rolling, lagged z-score** (aligned with other scaled features). The learned weight key is **`w_vix_term_zscore`**. Old JSON with `w_vix_term` is migrated on load.

## 2. Feature ablation (`TSE_ABLATION_STEP`)

| Step | Set includes |
|------|----------------|
| 0 | Phase 1 baseline (trend, ret, core vol, volume, corr, regime flag, sentiment slots) |
| 1 | + RSI + BB |
| 2 | + dist_high, dist_low |
| 3 | + overnight_gap, intraday_rev |
| 4 | + sector_relative_20d/60d |
| 5 | + vix_zscore, vol_spike, vix_term_zscore |
| 6 | + rolling_vol_5/60, vol_of_vol_20, jump, cs_momentum_percentile (full COMPOUND) |

Unset `TSE_ABLATION_STEP` → **no masking** (full features, default).

## 3. Automated ladder

From `trend_signal_engine/`:

```bash
python scripts/run_phase2_ablation.py
python scripts/run_phase2_ablation.py --compare-models   # Ridge vs GBR on best Sharpe step
```

Output: `output/learning/phase2_ablation_table.md`.

## 4. Differentiate models in backtest (confidence gate)

If `signal_confidence_multiplier` is too high (e.g. 0.8), small changes in learned weights may not change which names pass the gate. Temporarily lower it:

```bash
python run_backtest.py --signal-confidence-mult 0.5
```

## 5. Retrain after code changes

```bash
unset TSE_ABLATION_STEP   # full feature set
python run_weight_learning.py --model ridge --target-type regression
python run_backtest.py
```
