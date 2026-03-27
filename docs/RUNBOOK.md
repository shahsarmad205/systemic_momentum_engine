# Operations runbook (Trend Signal Engine)

Short recovery paths for batch jobs and live execution. Log paths: see [ARTIFACT_LAYOUT.md](ARTIFACT_LAYOUT.md).

## Cron / `run_daily_pipeline.py` failed

1. Open `output/live/pipeline.log` (or your `pipeline.pipeline_log` path); search for `FAIL`, `ABORT`, `Traceback`.
2. **Subprocess retrain failed:** See *Retrain rejected or subprocess error* below.
3. **`run_live_trading` failed:** Check `output/live/execution_log.jsonl` last line for `skipped`, `reason`, `orders_placed`.
4. Re-run after fix: `python run_daily_pipeline.py --dry-run` then full run if clean.

## Alpaca 401 / unauthorized

1. Confirm `ALPACA_API_KEY` and `ALPACA_SECRET_KEY` are **different** strings (Key ID vs Secret). See `brokers/alpaca_broker.py` validation.
2. Verify paper vs live URLs in `config/alpaca_config.yaml` match the key type.
3. `./scripts/reconcile_positions.py` after auth works to validate API reads.

## Retrain rejected or `retrain_model_last.json` status ≠ ok

1. Read `output/retrain_model_last.json`: `reject_reason`, `validation_mean_new_sharpe`, windows.
2. Check OHLCV cache dates under `backtest.cache_dir` — OOS validation needs fresh bars.
3. If rejection is expected, no deploy: live weights remain (`*.before_retrain` restores on reject).

## `fetch_fills` / slippage CSV empty or stale

1. Confirm `slippage_tracking.enabled: true` and paths in `backtest_config.yaml`.
2. Ensure `scripts/fetch_fills.py` (or your scheduled job) runs after fills appear in Alpaca.
3. Review `output/live/trades_pending.csv` vs `trades.csv` for stuck rows.

## Execution skipped liquidity / risk

1. Last `execution_log.jsonl` entry: `liquidity_skips`, `risk_skips`.
2. Tune `risk_factors.liquidity` (ADV caps) or rankings if skips are systematic.

## Kill switch

- Set env `TRADING_HALTED=1` (or `live.trading_halt_env` in YAML) **or** `live.trading_enabled: false`.
- With `--execute`, the engine **prints** planned closes/buys but does **not** call the broker when halted.

## Broker-side limits and idempotency

- Set Alpaca account **buying power**, max position count, and notional limits from their dashboard; code limits (e.g. `risk.max_position_pct_of_equity`, `max_positions`) are additive, not a substitute.
- **Re-running the same day:** Safe for read-only steps; live execution may replay intended orders — prefer deduplication via Alpaca order history or running once after close with a clear `RUN_ID` in logs.
- **Reconciliation:** `python scripts/reconcile_positions.py` — exits 1 if Alpaca tickers ≠ latest `paper_positions.csv` snapshot.

## Drift / monitoring

- `monitoring.ic_alert_threshold` and `sharpe_alert_threshold` drive alerts from `run_performance_tracker.py`.
- On breach: verify data freshness, weights file age, and compare rolling metrics in `output/live/monitoring_metrics.csv`; escalate per team process.
