# Model Selection Stability Guide

## What changed

`run_model_selection.py` now supports stability-aware selection, optional hyperparameter tuning, optional feature selection, and consensus feature voting with no-lookahead constraints.

- Walk-forward ranking still uses chained OOS Sharpe (`oos_sharpe_chained`) computed from concatenated OOS daily returns.
- Additional window-level stability metrics are logged:
  - `oos_sharpe_mean`
  - `oos_sharpe_std`
  - `oos_sharpe_min`
  - `oos_sharpe_max`
- If `model_selection.stability_metrics.enabled` is true, deployment prefers models that satisfy stability thresholds (with explicit fallback behavior documented below).
- When `model_selection.consensus_features.enabled: true`, the final artifact is `ConsensusRidge` trained on vote-aggregated consensus features.
  - If `ConsensusRidge` fails stability thresholds, it is kept in the report for diagnostics but not auto-selected for deployment.

## Config knobs

```yaml
model_selection:
  min_windows: 6
  stability_metrics:
    enabled: true
    min_sharpe_mean: 0.0
    max_sharpe_std: 1.6
    min_window_sharpe: -0.5
  hyperparameter_tuning:
    enabled: true
    method: "random"   # grid | random
    n_trials: 10
    cv: 3
  consensus_features:
    enabled: true
    top_k_per_model_window: 20
    vote_threshold_ratio: 0.5
    weighting: "sharpe_weighted"       # uniform | sharpe_weighted
    fallback_min_features: 12
  validator_gates:
    enabled: true
    min_validation_windows: 2
    min_chained_sharpe: 0.5
    max_drawdown_abs: 0.45
    min_win_rate: 0.5

feature_selection:
  enabled: true
  method: "importance" # importance | recursive
  n_features: 20
```

## How Consensus Voting Works

- For each model-window fit, top-K features are extracted using model-native importance:
  - linear models: `abs(coef_)`
  - tree/boosting models: `feature_importances_`
- Votes are aggregated across vote-period model-window events (windows before the consensus evaluation period).
  - When `weighting: sharpe_weighted`, model weights are computed from vote-period windows only.
- A feature enters consensus if its vote mass is at least:
  - `vote_threshold_ratio * total_vote_mass`
- If no feature clears threshold, fallback uses top `fallback_min_features` by vote score.
- `output/models/consensus_features.json` is written with:
  - selected features
  - vote scores
  - threshold mass / total mass
  - weighting mode used

The final `best_model.pkl` is then a Ridge regressor trained on consensus features.

## Immutable Run Manifest

Each run writes immutable metadata for reproducibility:

- `output/models/run_manifest_<UTC>.json`
- `output/models/latest_run_manifest.json`

Manifest includes:

- selected model and selection metric
- git commit and dirty state
- config path and SHA-256 hash
- core data snapshot stats (dates, rows, tickers)
- SHA-256 hashes for report/model/meta artifacts

## Promotion Workflow

Lifecycle states:

- `research` -> `candidate` -> `shadow` -> `production`

Promotion command:

```bash
python scripts/promote_model.py --config backtest_config.yaml --to-state candidate --actor <name> --reason "initial promotion"
python scripts/promote_model.py --config backtest_config.yaml --to-state shadow --actor <name> --reason "shadow rollout"
python scripts/promote_model.py --config backtest_config.yaml --to-state production --actor <name> --reason "approved for production"
```

Artifacts written:

- `output/models/model_registry.json`
- `output/models/production_pointer.json` (only on production promotion)
- `output/models/promotion_log.jsonl`

Promotion gates:

- manifest integrity (artifact hashes must match)
- validator pass (unless explicitly skipped)
- transition validity (no skipping states)
- optional block on dirty git state for production
- production readiness checks for `to-state production`:
  - halt latch must be clear
  - latest `shadow_monitor` report status must be PASS and fresh
  - latest `risk_gate` report status must be PASS and fresh
  - latest `tca_health` report status must be PASS and fresh

## Shadow Monitoring

Run shadow drift checks against the current production pointer:

```bash
python scripts/run_shadow_monitor.py --config backtest_config.yaml
python scripts/run_shadow_monitor.py --config backtest_config.yaml --strict
```

Artifacts written:

- `output/models/shadow_monitor_<UTC>.json`
- `output/models/shadow_monitor_latest.json`

Default shadow monitor gates (from `model_selection.shadow_monitor`):

- `min_score_corr`
- `min_topk_overlap`
- `max_abs_score_delta`

If no production pointer exists, monitoring exits with an error so rollout steps remain explicit.

## Hard-Limit Risk Gate (Phase 4)

Before orders are sent in `run_live_trading.py`, a fail-closed hard-limit gate now runs on the final executable target book (after execution-side factor/liquidity filtering).

Default checks:

- max gross exposure
- max absolute net exposure
- max absolute single-name exposure
- max absolute short single-name exposure
- max sector exposure (from sector mapping)

Config source: `risk.hard_limit_gate` (with fallback to existing `risk` / `risk_factors` keys).

Artifacts written on each run:

- `output/live/target_pretrade_<DATE>.csv`
- `output/live/target_pretrade_latest.csv`
- `output/live/target_executable_<DATE>.csv`
- `output/live/target_executable_latest.csv`
- `output/live/risk_gate/risk_gate_<UTC>.json`
- `output/live/risk_gate/risk_gate_latest.json`

Standalone check command:

```bash
python scripts/check_risk_limits.py --config backtest_config.yaml --target-csv output/live/target_executable_latest.csv --strict
```

If hard-limit checks fail, live execution aborts before broker order placement.

## TCA Health Gate (Phase 5)

Strict governance preflight now includes a transaction-cost analysis (TCA) gate based on live fill slippage.

Run standalone:

```bash
python scripts/check_tca_health.py --config backtest_config.yaml --strict
```

Artifacts written:

- `output/live/tca/tca_health_<UTC>.json`
- `output/live/tca/tca_health_latest.json`

Default TCA gates come from `governance.tca_health`:

- `max_avg_slippage_bps` (rolling mean adverse bps)
- `max_p95_slippage_bps` (tail slippage)
- `min_fills` with optional `fail_on_no_data`

When strict preflight is enabled in `run_daily_pipeline.py`, TCA health is checked alongside shadow monitor and hard-limit risk checks, and failures activate the live trading halt latch.

## Production Readiness (Phase 6)

`promote_model.py --to-state production` now enforces governance-readiness policy from `model_selection.promotion.production_readiness`.

Default required artifacts:

- `output/models/shadow_monitor_latest.json` (PASS)
- `output/live/risk_gate/risk_gate_latest.json` (PASS)
- `output/live/tca/tca_health_latest.json` (PASS)
- `output/live/trading_halt_latch.json` (`halt_active: false`)

Reports are freshness-checked by `max_report_age_hours`.

## Ops Governance Automation (Phase 8)

`run_ops_suite.py` now enforces governance checks as a first-class operational gate:

- Forces `run_daily_pipeline.py --strict-preflight` by default.
- Runs `scripts/governance_daily_summary.py --strict` after pipeline completion.
- Aborts the suite if governance summary is not `PASS`.

Use:

```bash
python run_ops_suite.py
```

To disable strict governance enforcement for controlled debugging only:

```bash
python run_ops_suite.py --no-strict-governance
```

## How tuning works

- Tuning is run on the first valid training window for each model.
- The best parameters are reused for subsequent windows for consistency.
- CV inside tuning is date-based and purged with an embargo gap, so validation folds do not leak adjacent label horizons.
- Search methods:
  - `random` -> `RandomizedSearchCV`
  - `grid` -> `GridSearchCV`
- For `XGBRegressor`, default settings are regularized (`max_depth=3`, `learning_rate=0.05`, `n_estimators=100`, `subsample=0.8`, `colsample_bytree=0.8`, `reg_alpha=0.1`, `reg_lambda=1.0`).

## How feature selection works

- Feature selection is fit only on training data (first valid training window per model).
- Selected columns are reused for that model across all windows and final training.
- Methods:
  - `importance`: random-forest feature importances
  - `recursive`: RFE with a linear estimator

## Interpreting stability metrics

- Prefer higher `oos_sharpe_chained` among models that pass stability filters.
- Watch `oos_sharpe_std`:
  - lower is more consistent across regimes/windows
- Watch `oos_sharpe_min`:
  - very negative values indicate brittle behavior in bad windows
- Deployment rule when stability is enabled:
  - If any non-consensus model passes stability, deployment selection is restricted to those stable models.
  - If none pass, the script warns and falls back to the best non-consensus model by the selected metric.
  - `ConsensusRidge` must also satisfy `min_windows` to be deployment-eligible.

A practical target profile for live candidates is:

- `oos_sharpe_mean >= 0`
- `oos_sharpe_std <= 1.6`
- `oos_sharpe_min >= -0.5`

## Quick validation before live use

Use:

```bash
python scripts/validate_model.py --config backtest_config.yaml
```

This runs a quick walk-forward validation on recent windows and fails fast if stability thresholds are not met.

By default it enforces stability thresholds and, when `model_selection.validator_gates.enabled: true`, enforces validator gates from `model_selection.validator_gates`.
CLI arguments still override config thresholds.
