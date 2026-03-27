# Oracle Cloud (Always Free) production VM setup

This is a **free-first** path to a real production VM suitable for student budgets.

## 1) Create an OCI Always Free VM

- Create an Oracle Cloud account (Free Tier).
- Create a **Compute Instance** (Always Free eligible).
- Prefer **Ubuntu LTS**.

## 2) SSH hardening (minimum)

- Disable password auth; use SSH keys only.
- Create a dedicated user `tse` (no sudo by default; use controlled sudo when needed).
- Restrict inbound ports (allow only SSH and optional API port 8000 if you run the API).

## 3) Install dependencies

On the VM:

```bash
sudo apt update
sudo apt install -y git python3.11 python3.11-venv
```

## 4) Deploy

```bash
sudo mkdir -p /opt/trend_signal_engine
sudo chown -R "$USER":"$USER" /opt/trend_signal_engine
cd /opt/trend_signal_engine
git clone <your-repo-url> .
cd trend_signal_engine

python3.11 -m venv /opt/trend_signal_engine/venv
/opt/trend_signal_engine/venv/bin/pip install -U pip
/opt/trend_signal_engine/venv/bin/pip install -r requirements-lock.txt
```

## 5) Configure environment file

Create `/opt/trend_signal_engine/env/prod.env` (root owned, `chmod 600`). Use the template in
[`deploy/templates/prod.env.example`](../templates/prod.env.example).

## 6) systemd timers/services

Follow [`deploy/README.md`](../README.md) to install/enable the systemd unit templates.

