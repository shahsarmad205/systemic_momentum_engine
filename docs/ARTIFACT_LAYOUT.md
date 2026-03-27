# Artifact layout and retention

All paths are relative to `trend_signal_engine/` unless noted.

## Run logs (append-only)

| Path | Producer | Suggested retention |
|------|----------|---------------------|
| `output/live/pipeline.log` | `run_daily_pipeline.py` | 30–90 days (rotate / compress) |
| `output/live/retrain.log` | `run_retrain_model.py` | 90 days |
| `output/live/execution_log.jsonl` | `brokers/execution_engine.py`, `run_live_trading.py` | 180 days |
| `output/live/order_intents.jsonl` | `brokers/execution_engine.py` | 180 days |
| `output/live/trades.csv`, `trades_pending.csv` | Slippage tracking | 1 year |
| `output/live/slippage_metrics.csv` | Slippage rollups | 1 year |
| `output/live/monitoring_metrics.csv`, `ic_tracker.csv`, `daily_pnl.csv` | Performance tracker | 1–2 years |
| `output/live/dashboard.html` | Monitoring | Replace in place |

## Per-run snapshots (audit-friendly)

| Path | Producer | Contents |
|------|----------|----------|
| `output/runs/<RUN_ID>/run_metadata.json` | `run_daily_pipeline.py` | CLI args, git SHA, and file snapshots |
| `output/runs/<RUN_ID>/backtest_config.yaml` | `run_daily_pipeline.py` | Config snapshot used for the run |
| `output/runs/<RUN_ID>/requirements-lock.txt` | `run_daily_pipeline.py` | Dependency lock snapshot (if present) |
| `output/runs/<RUN_ID>/learned_weights.json` | `run_daily_pipeline.py` | Weights snapshot (if present) |
| `output/runs/<RUN_ID>/learned_weights_scaler.json` | `run_daily_pipeline.py` | Scaler snapshot (if present) |

## Model artifacts

| Path | Purpose |
|------|---------|
| `output/learned_weights.json`, `learned_weights_scaler.json` | **Live** scoring weights |
| `output/*.before_retrain` | Pre-retrain backup (suffix configurable) |
| `output/retrain_model_last.json` | Last retrain outcome / validation stats |
| `output/retraining_last_accepted_oos.json` | Last accepted OOS baseline |
| `output/models/releases/<YYYYMMDD>/` | Versioned snapshot after successful accept (`retraining.release_snapshot_on_success`) |

**Restore:** Copy the desired release folder’s `learned_weights.json` (and scaler) into `output/`, then rerun pipeline dry-run before enabling execution.

## Correlation ID

Set `RUN_ID` in the environment for cron runs so `pipeline.log`, `execution_log.jsonl`, and execution entries share the same id (see `utils/run_context.py`).
