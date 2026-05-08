# Codebase Audit - 2026-04-22

Scope: static architecture review, import/reference scan, lint scan for unused symbols, artifact size audit, and README refresh. This audit does not delete files. Removal decisions below are classified by confidence.

## Architecture Summary

The repository has four main layers:

1. Research and data layer: `utils/wrds_*`, `utils/market_data.py`, `agents/weight_learning_agent/feature_builder.py`, `features/`, `analysis/`, `research/`.
2. Simulation layer: `backtesting/` owns the main engine, signals, optimizer, risk model, scheduler, analytics, and metrics. `backtest/engine.py` is a facade used by runners.
3. Production operations layer: `run_daily_pipeline.py`, `run_live_trading.py`, `brokers/`, `risk/`, and governance scripts under `scripts/`.
4. Deployment layer: `LeanCloud/BinaryEdge/` and QC parity files.

Primary flow:

```text
WRDS/Yahoo cache -> universe -> features -> signals -> optimizer/risk overlays -> backtest or live targets -> broker/QC
```

Graphify confirms the most connected abstractions are `TransactionCostModel`, `BacktestConfig`, `Backtester`, `SignalEngine`, `BacktestResult`, `MultiAlphaEngine`, `MarketImpactModel`, `FactorNeutralizer`, `MarketRegimeAgent`, and `PortfolioOptimizer`.

## Classification by Area

| Path | Classification | Rationale |
| --- | --- | --- |
| `backtesting/` | Core | Main simulation, signals, optimizer, risk, analytics, metrics, and portfolio construction. |
| `backtest/` | Supporting | Facade around `backtesting.Backtester`; used by `run_backtest.py` and validation flows. |
| `utils/wrds_data.py`, `utils/wrds_loader.py`, `utils/wrds_universe.py` | Core | Point-in-time universe, CRSP data, delisting-aware loader. |
| `utils/market_data.py` | Core but legacy-contaminated | Still high inbound usage. Contains Yahoo fallback and provider cache logic. |
| `utils/ensemble_scoring.py`, `utils/adv_cache.py`, `utils/data_governance.py`, `utils/live_trades.py`, `utils/trading_control.py` | Core | Used in ML scoring, liquidity, daily pipeline, execution logs, and halt controls. |
| `agents/weight_learning_agent/` | Core | Feature builder, weight model, ensemble model, and regime detection used by training/backtest/live. |
| `features/feature_pipeline.py`, `features/wrds_fundamental_builder.py`, `features/fundamental_router.py` | Core | Feature construction and WRDS fundamental routing. |
| `features/alternative_features.py`, `features/breadth_features.py`, `features/capm_features.py`, `features/latent_factor_features.py` | Supporting | Research or optional feature modules; not all are on the default live path. |
| `risk/` | Core | Drawdown overlay, hard limits, and VaR used by research/live governance. |
| `execution/` | Core | Transaction cost and market impact models. |
| `brokers/` | Core for live | Alpaca adapter and execution engine. |
| `scripts/check_*`, `scripts/run_*guard*`, `scripts/promote_model.py`, `scripts/rollback_model.py` | Core for ops | Governance and promotion controls. |
| `scripts/validate_*`, `analysis/`, `tools/` | Supporting | Research diagnostics and validation utilities; generally CLI entry points, not imported libraries. |
| `simulation/`, `options/` | Supporting | Scenario analysis and option analytics. `options` is imported by backtester and option runner. |
| `portfolio/` | Supporting | Mean-variance utilities and thin package facade; used by comparison and optional backtester path. |
| `LeanCloud/BinaryEdge/` | Core for QC deployment | LEAN algorithm, QC alpha model, serialized artifacts, config. |
| `qc_alpha_model.py`, `qc_main.py` | Supporting/legacy QC | Local QC parity or older QC entry point. Keep until parity workflow is consolidated. |
| `data_processing/`, `data/loader.py`, `data_loader.py` | Redundant-risky | Older data pipeline wrappers. `data_loader.py` is referenced by `main.py` compatibility notes; not safe to delete without checking launcher usage. |
| `main.py` | Redundant-risky | Legacy/simple launcher; static inbound exists. Keep until CLI consolidation is complete. |
| `config/sp500_tickers.txt` | Supporting | Useful fallback, but today's membership is survivorship-biased for research. |
| `output/`, `data/cache/`, `.ruff_cache/`, `.pytest_cache/`, `__pycache__/` | Generated | Not source. Safe to exclude from commits; deletion depends on reproducibility needs. |

## Redundant or Dead-Code Candidates

High-confidence cleanup:

- Python bytecode caches: `__pycache__/`, `*/__pycache__/`.
- Tool caches: `.ruff_cache/`, `.pytest_cache/`.
- Local virtualenv: `.venv/` if dependencies can be rebuilt from requirements.
- Generated Graphify cache files under `graphify-out/cache/`; keep `graphify-out/GRAPH_REPORT.md` and `graphify-out/graph.json` if this repo intentionally tracks graph state.

Medium-confidence code cleanup:

- `data_processing/*`, `data/loader.py`, and `data_loader.py` appear to be older ingestion paths. Keep for now because `main.py` and compatibility docs still reference them.
- `qc_main.py` and root `qc_alpha_model.py` may be older QC entry points because the active package is `LeanCloud/BinaryEdge/`. Keep until QC parity tests and deployment docs are consolidated.
- `analysis/*` and many `tools/*` scripts are zero-inbound by import graph, but most are valid CLI research tools. Do not delete solely because they are not imported.
- `backtesting/engine.py`, `backtesting/factor_normalization.py`, `backtesting/oos_forward_validation.py`, `backtesting/universe_selector.py`, and `research/ic_engine.py` are zero-inbound in static imports. Treat as supporting or experimental until their CLI/config usage is audited.

Do not remove:

- `utils/market_data.py`: although Yahoo remains, it is still widely imported and provides cache/provider plumbing.
- `config/sp500_tickers.txt`: survivorship-biased for research, but useful fallback for development and WRDS outage cases.
- `output/models/*.pkl` if they are the current promoted artifacts. Move to an artifact store only after promotion metadata points to the replacement location.
- `LeanCloud/BinaryEdge/*.pkl` if QuantConnect deployment depends on embedded model files.

## Function and Import Findings

Static lint scan: `ruff check --select F401,F841` reported 60 findings.

Safe low-risk cleanup candidates:

- `audit_calibration.py`: unused `pandas`, `Path`, and `matplotlib.pyplot` imports.
- `backtesting/analytics.py`: unused `os` import.
- `backtesting/trade_scheduler.py`: unused `numpy` import.
- `data/loader.py`: unused `OHLCV_COLUMNS` import.
- `execution/market_impact.py`: unused `math` and `field` imports.
- `features/alternative_features.py`: unused `numpy` import.
- `features/fundamental_router.py`: unused `numpy` import.
- `utils/ensemble_scoring.py`: unused `warnings` import.
- `utils/market_data.py`: unused `io` and `numpy` imports.
- `run_model_selection.py`: unused `RidgeClassifier` and local `warnings` import were removed in this pass.

Assigned-but-unused locals worth reviewing:

- `backtesting/backtester.py`: `sharpe_cb_window`, `sharpe_cb_thr`, `sharpe_cb_recovery`, `sharpe_cb_enabled`, `n_pos`.
- `backtesting/optimizer.py`: `w_prev_iter`.
- `backtesting/portfolio_construction.py`: `rc_i`.
- `backtesting/regime.py`: `last_refit_idx`.
- `features/feature_pipeline.py`: `shrout_proxy`.
- `scripts/run_rollout_guard.py`: `status`.
- `scripts/validate_wrds_migration.py`: `universe`.

Environment-sensitive cleanup candidates:

- `LeanCloud/BinaryEdge/main.py`: unused `System.Decimal`. Remove only after QC compile check.
- `LeanCloud/BinaryEdge/qc_alpha_model.py`: unused `lightgbm` import may be a dependency availability sentinel. Do not remove without QC unpickle test.
- `qc_alpha_model.py`: unused `io` and `old_reducer_override`; likely safe locally, but QC parity should gate removal.

## Repository Size Findings

Observed size profile:

| Path | Approx size | Classification |
| --- | ---: | --- |
| Repository total | 5.4 GB | Too large for clean source repo. |
| `data/cache/` | 2.6 GB | Generated market-data cache. |
| `.venv/` | 529 MB | Local environment, should not be versioned. |
| `LeanCloud/` | 239 MB | Includes QC code plus sample tick/minute zip data and model artifacts. |
| `output/` | 72 MB | Generated reports, models, experiments, live logs. |
| `data/cache/wrds/` | 68 MB | WRDS cache, generated but valuable for reproducibility. |
| `graphify-out/` | 5.8 MB | Generated knowledge graph. |

## Suggested Deletions

Safe to delete locally if not needed for the current run:

- `__pycache__/` and nested `*/__pycache__/`.
- `.ruff_cache/`.
- `.pytest_cache/`.
- Old `output/experiments/<timestamp>/` runs after archiving selected benchmark runs.
- Stale `output/backtests/*.png`, `output/backtests/*.csv`, and diagnostics generated by exploratory runs.

Risky or confirm-first deletions:

- `data/cache/`: large, but deleting forces expensive refetch/rebuild and may remove delisting/PIT cache context.
- `data/cache/wrds/`: valuable WRDS snapshots; archive rather than delete if reproducibility matters.
- `output/models/`: contains selected model artifacts and governance state.
- `LeanCloud/data/`: sample LEAN data can be regenerated, but may be needed for local LEAN smoke tests.
- `LeanCloud/BinaryEdge/*.pkl`: deployment artifacts; delete only after replacing with Object Store or updated package sync.
- Legacy-looking code paths (`main.py`, `data_processing/`, `qc_main.py`) until entrypoint consolidation is complete.

## Refactoring Performed

- Removed two unused imports from `run_model_selection.py`.
- Rewrote `README.md` to describe the current WRDS-aware, model-selection-aware, governance-aware architecture.

## Recommended Next Refactors

1. Add a first-class `providers.data_provider: wrds` configuration key and make Yahoo fallback explicit rather than implicit.
2. Split `utils/market_data.py` into provider adapters (`wrds`, `yahoo`, `alpaca`) plus shared cache utilities.
3. Consolidate QC files so only one local QC alpha model path exists, then delete the old root-level QC entry if parity passes.
4. Move generated artifacts to a retention policy: keep latest, keep promoted model releases, archive benchmark experiments, delete scratch outputs.
5. Run `ruff check --select F401,F841 --fix` on non-QC files after tests are green.
6. Add an import-boundary test to prevent live code from accidentally using research-only or Yahoo-only paths.
