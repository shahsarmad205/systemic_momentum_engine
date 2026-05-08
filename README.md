# Trend Signal Engine

Production-oriented quantitative research and trading stack for point-in-time equity universe construction, momentum and ML alpha scoring, portfolio optimization, risk overlays, execution checks, and QuantConnect deployment.

This repository is currently organized as a research-to-production platform, not a single notebook strategy. The important distinction is that alpha research, beta-sensitive deployment views, backtest simulation, and live execution governance are separate stages with different failure modes.

## System Flow

```text
WRDS / cached market data
  -> point-in-time universe and delisting-aware price panels
  -> feature matrix and cross-sectional normalization
  -> ML / multi-alpha signal generation
  -> factor neutralization and liquidity-aware portfolio construction
  -> drawdown, VaR, hard-limit, and TCA gates
  -> backtest reports, live paper execution, or QuantConnect deployment
```

## Main Entry Points

Authoritative paths are separated by mandate:
- Research selection: `run_model_selection.py`
- Historical portfolio simulation: `run_backtest.py`
- Daily live/paper orchestration: `run_daily_pipeline.py`
- QuantConnect deployment package: `LeanCloud/BinaryEdge/`

Compatibility-only facades:
- `main.py` is a legacy batch runner and default-universe holder for older scripts.
- Root `qc_main.py` and `qc_alpha_model.py` now re-export the LEAN package and are not deployment sources of truth.

| Command | Purpose |
| --- | --- |
| `python run_backtest.py` | Run the local historical backtest from `backtest_config.yaml`. |
| `python run_backtest.py --walk-forward` | Run walk-forward backtest validation. |
| `python run_model_selection.py` | Run ML walk-forward model selection. Default selection uses formal `oos_deflated_sharpe`. |
| `python run_model_selection.py --run_sim_test` | Fast self-test for portfolio simulation semantics. |
| `python run_daily_signals.py` | Generate daily signal snapshots. |
| `python run_live_trading.py` | Dry-run broker target generation and preflight checks. |
| `python run_live_trading.py --execute` | Submit paper/live orders after governance gates pass. |
| `python run_daily_pipeline.py --strict-preflight` | Operational pipeline with strict governance preflight. |
| `python run_ops_suite.py` | End-to-end daily ops suite with governance summary. |

## Architecture

| Area | Key paths | Role |
| --- | --- | --- |
| Configuration | `backtest_config.yaml`, `backtesting/config.py`, `config.py` | YAML parsing, defaults, dev-mode controls, universe and risk settings. |
| Universe and data | `utils/wrds_universe.py`, `utils/wrds_data.py`, `utils/wrds_loader.py`, `utils/market_data.py`, `utils/universe.py` | WRDS point-in-time universe, CRSP delisting-aware prices, cache management, legacy provider fallback. |
| Feature generation | `agents/weight_learning_agent/feature_builder.py`, `features/cross_sectional.py`, `features/feature_pipeline.py`, `features/wrds_fundamental_builder.py` | Technical, sector-relative, panel-normalization, liquidity, and fundamental features used by research and live scoring. |
| Signal generation | `backtesting/signals.py`, `backtesting/multi_alpha.py`, `utils/ensemble_scoring.py` | Learned weights, ML model scoring, short-model scoring, and multi-alpha combination. |
| Portfolio construction | `backtesting/optimizer.py`, `backtesting/risk_model.py`, `backtesting/cross_sectional.py`, `backtesting/position_sizing.py`, `backtesting/trade_scheduler.py` | Rank selection, continuous optimization, factor constraints, liquidity caps, sizing, and rebalance scheduling. |
| Backtesting | `backtesting/backtester.py`, `backtest/engine.py`, `backtesting/analytics.py`, `backtesting/metrics.py` | Historical simulation, mark-to-market analytics, trade logs, walk-forward reports, and plots. |
| Risk overlays | `risk/drawdown_overlay.py`, `risk/hard_limits.py`, `risk/var.py`, `backtesting/regime.py` | Unified drawdown scaling, exposure limits, VaR checks, regime detection, and crisis controls. |
| Execution and governance | `brokers/execution_engine.py`, `brokers/alpaca_broker.py`, `scripts/check_risk_limits.py`, `scripts/check_tca_health.py`, `scripts/run_shadow_monitor.py` | Target-to-order translation, broker reconciliation, hard-limit gates, TCA health, and model drift checks. |
| Deployment | `LeanCloud/BinaryEdge/`, `verify_qc_parity.py`, `verify_lean_parity.py` | QuantConnect/LEAN algorithm package and local parity checks. Root QC files are compatibility facades only. |
| Research utilities | `analysis/`, `research/`, `scripts/validate_*` | IC analysis, ablations, robustness checks, calibration, and operational diagnostics. |
| Tests | `tests/` | Unit and integration tests for model selection, risk gates, WRDS context, execution controls, and signal generation. |

## Model Selection Semantics

The model-selection control plane is being decomposed into dedicated modules:
- `model_selection/configuration.py` for YAML-to-runtime config translation
- `model_selection/model_registry.py` for model families and prefit ensembles
- `model_selection/statistics.py` for formal model-ranking metrics
- `run_model_selection.py` as orchestration only

This split is intentional: institutional research governance should not live in one giant script.

`run_model_selection.py` now separates research and deployment views:

| Path | Use | Construction |
| --- | --- | --- |
| `long_short_spread` | Alpha research and long-model ranking | Long top-ranked names and short bottom-ranked names. |
| `long_only_overlay` | Beta-sensitive deployment diagnostics | Long top-ranked names only. Reported separately as `overlay_oos_*`. |
| `short_side` | Dedicated short-model validation | Short bottom-ranked names with PnL sign-flipped. |

Selection is rank-based rather than `score > 0` gated. Chained OOS metrics use the same portfolio rules as per-window runs, and forward returns are horizon-normalized before daily performance metrics are computed.

## Data Notes

- The institutional research path should use WRDS point-in-time universe membership and CRSP delisting-aware returns.
- `universe.mode: "wrds"` in `backtest_config.yaml` controls point-in-time universe construction.
- `data/cache/` and `data/cache/wrds/` are generated caches, not source code.
- Yahoo/yfinance remains present as a legacy or fallback provider in parts of the codebase. For production-grade research, prefer WRDS and treat Yahoo-derived results as non-authoritative.

Required WRDS environment:

```bash
export WRDS_USERNAME=<your_wrds_username>
```

## Configuration

Most behavior is controlled in `backtest_config.yaml`:

- `backtest.use_continuous_optimization`: enables the continuous optimizer path.
- `backtest.factor_model`: market, sector, size, and momentum exposure constraints.
- `backtest.liquidity`: ADV-based universe and position constraints.
- `risk.drawdown_overlay`: shared drawdown scaling policy.
- `risk.var_check`: live/backtest VaR preflight behavior.
- `signals.mode`: signal source, usually `ml` for model artifacts.
- `signals.ml_long_model_path` and `signals.ml_short_model_path`: production model artifacts.
- `model_selection`: walk-forward horizon, position counts, and ensemble selection settings.

## Outputs and Artifacts

| Path | Contents |
| --- | --- |
| `output/backtests/` | Latest backtest equity, trades, plots, and summary. |
| `output/models/` | Model comparison reports, selected model artifacts, manifests, and promotion state. |
| `output/live/` | Live signal, target, risk gate, TCA, and execution logs. |
| `output/experiments/` | Timestamped research run snapshots. |
| `data/cache/` | Provider price caches and WRDS extracts. |
| `graphify-out/` | Local code knowledge graph. Regenerate after code edits when required by `AGENTS.md`. |

Generated artifacts can become large. Do not commit local `.venv`, provider caches, `__pycache__`, `.ruff_cache`, or ad hoc output snapshots unless a specific artifact is intentionally versioned.

## Validation

Useful fast checks:

```bash
python -m py_compile run_model_selection.py
python run_model_selection.py --run_sim_test
python -m pytest tests/test_model_selection_split_phase2.py -q
python -m pytest tests/test_optimizer_constraints.py tests/test_drawdown_overlay.py -q
```

Broader checks:

```bash
python -m pytest
python scripts/validate_wrds_migration.py --config backtest_config.yaml
python scripts/check_risk_limits.py --config backtest_config.yaml --strict
python scripts/check_tca_health.py --config backtest_config.yaml --strict
```

## QuantConnect Deployment

The LEAN package lives in `LeanCloud/BinaryEdge/` and is the deployment source of truth. Validate local parity before promoting a model into QC:

```bash
python verify_qc_parity.py
python verify_lean_parity.py
```

Then use the LEAN CLI from the `LeanCloud/` workspace as appropriate for cloud backtests and deployment.

## Operating Principles

- Research metrics must be out-of-sample and aligned with the traded portfolio construction.
- Live deployment metrics must be separated from alpha research metrics.
- WRDS/CRSP point-in-time and delisting-aware data should be the source of truth for institutional research.
- Risk overlays must be shared across research and live paths where possible.
- Generated data and reports are reproducibility artifacts, not core source code.
