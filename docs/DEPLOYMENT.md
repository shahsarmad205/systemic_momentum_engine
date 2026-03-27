# Deployment model

Choose **one** primary runtime; both options assume `trend_signal_engine/` as the working directory.

## A. VM or bare metal + cron (simplest)

1. Install Python 3.11+, clone repo, create venv, `pip install -r requirements-lock.txt` (see [DEPENDENCIES.md](DEPENDENCIES.md)).
2. Set env: `RUN_ID` (optional, cron can use date), `TRADING_HALTED` if needed, Alpaca keys.
3. Crontab (example, after US close):  
   `15 16 * * 1-5 cd /app/trend_signal_engine && /app/venv/bin/python run_daily_pipeline.py`
4. Optional API: `uvicorn api.main:app --host 0.0.0.0 --port 8000` under systemd, or `gunicorn` + reverse proxy.

**Rollback:** Restore previous `output/learned_weights.json` and `learned_weights_scaler.json` from `output/models/releases/<date>/` or git; verify with `python run_live_trading.py` (dry run) and monitoring CSVs.

**systemd templates:** see [`deploy/`](../deploy/README.md) for timer/service unit templates and a suggested `/opt/trend_signal_engine` layout.

## B. Container + scheduler

1. Build image with `WORKDIR /app/trend_signal_engine`, `COPY` project, `pip install -r requirements-lock.txt`.
2. **Batch:** Scheduled Job (K8s CronJob, ECS Scheduled Task, Cloud Run Job) running `python run_daily_pipeline.py` with secrets as env.
3. **API:** Deployment/Service exposing port with `uvicorn api.main:app`; use `/health` and `/ready` for probes.

**Rollback:** Revert to previous image **or** restore weight files from release directory as in (A).

## Paper → live checklist (before removing paper-only gates)

1. Kill switch tested (`TRADING_HALTED=1` with `--execute` places no orders).
2. Reconciliation clean on paper for multiple weeks (positions vs broker, fills vs intended orders).
3. Alpaca account limits aligned with `risk` / `backtest.max_positions` and deployment notional caps.
4. **Shorts (if used):** paper validates borrow/short availability, margin usage, and that `execution.enable_shorts` / broker path matches backtest assumptions (see [STRATEGY_SHORTS_UNIVERSE_LIVE.md](STRATEGY_SHORTS_UNIVERSE_LIVE.md)).
5. **Runbook:** operators know [RUNBOOK.md](RUNBOOK.md) kill switch, log locations, and who to call on failure.
6. Explicit sign-off documented outside this repo (compliance / risk); live API keys in a dedicated secret store, not developer laptops by default.
7. **Size:** start at the smallest capital that still exercises the full stack; scale only after a defined minimum track-record window.
