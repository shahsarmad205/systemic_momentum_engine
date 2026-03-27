# Logs + alerts (free): Grafana Cloud + Grafana Alloy

Goal: ship **systemd/journald** logs from the production VM to a hosted free tier and create alerts.

## 1) Create Grafana Cloud Free account

- Create a Grafana Cloud stack (Free).
- Enable Logs (Loki) and Alerting.

## 2) Install Grafana Alloy on the VM

Grafana Alloy is the recommended successor to Promtail (Promtail is EOL in 2026).

Follow Grafana’s official install instructions for your OS, then place the config template below.

## 3) Configure Alloy to scrape journald

Use `deploy/grafana/alloy.config.example` as a starting point.\n\nYou must fill in:\n- Loki endpoint URL\n- A token/credentials from your Grafana Cloud stack\n\n## 4) Alerts to configure (minimum)\n\n- Pipeline failed (systemd unit exit != 0)\n- “Trading halted” detection (log pattern)\n- Reconciliation drift (when you schedule `scripts/reconcile_positions.py`)\n\n## 5) Verify\n\n- Search logs for `run_id=`\n- Force a test alert (Grafana UI)\n\n*** End Patch"})}]}�
