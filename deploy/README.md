# Deployment (Hybrid_DevProd)

This folder contains **templates** for running Trend Signal Engine on a dedicated VM using **systemd timers** (preferred) or cron.

## Directory layout (recommended)

- `/opt/trend_signal_engine/` (repo clone)
- `/opt/trend_signal_engine/venv/` (Python 3.11 venv)
- `/opt/trend_signal_engine/env/prod.env` (env file, owned by root, `chmod 600`)

## Install (VM runtime)

From `trend_signal_engine/`:

```bash
python3.11 -m venv /opt/trend_signal_engine/venv
/opt/trend_signal_engine/venv/bin/pip install -U pip
/opt/trend_signal_engine/venv/bin/pip install -r requirements-lock.txt
```

## systemd unit installation

Copy the templates:

```bash
sudo mkdir -p /etc/systemd/system
sudo cp deploy/systemd/trend-signal-engine*.service /etc/systemd/system/
sudo cp deploy/systemd/trend-signal-engine*.timer /etc/systemd/system/
sudo systemctl daemon-reload
```

Enable daily pipeline timer:

```bash
sudo systemctl enable --now trend-signal-engine-daily.timer
sudo systemctl status trend-signal-engine-daily.timer
```

Optional API:

```bash
sudo systemctl enable --now trend-signal-engine.service
sudo systemctl status trend-signal-engine.service
```

## Environment file (`/opt/trend_signal_engine/env/prod.env`)

At minimum:

- `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`
- `TRADING_HALTED=1` during rollout / incidents
- Optional `RUN_ID` (or let each run generate one)

## Rollback

- Restore prior model snapshot under `output/models/releases/<YYYYMMDD>/`
- Re-run `python run_live_trading.py` in **dry run** mode, then re-enable execution.

