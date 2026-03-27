# Secrets and credentials

## Principles

- **Never commit** real API keys. Use `config/alpaca_config.example.yaml` as a template; keep `config/alpaca_config.yaml` out of version control (see `.gitignore`).
- **Prefer environment variables** in deployment: `ALPACA_API_KEY`, `ALPACA_SECRET_KEY`, optional `ALPACA_CONFIG` for a non-default YAML path.
- **Rotation:** Generate new keys in Alpaca dashboard, update secret manager / env, deploy, then revoke old keys. Smoke-test with `scripts/reconcile_positions.py` or a read-only API call.
- **SMTP / Slack:** If using `monitoring.alert_method` email or Slack, store `SMTP_*` and webhook URLs in the environment or secret manager, not in committed YAML (placeholders only).

## Fresh clone checklist

1. Copy `config/alpaca_config.example.yaml` → `config/alpaca_config.yaml` (local only) **or** export env vars.
2. **Production / reproducible:** `pip install -r requirements-lock.txt` (see [DEPENDENCIES.md](DEPENDENCIES.md)). **Local dev:** `pip install -r requirements-dev.txt`.
3. After any change to `requirements.txt`, regenerate `requirements-lock.txt` on **Python 3.11** (matches CI) and commit both.

## Supply chain

- CI runs `pip-audit` on `requirements.txt`; review failures and pin upgrades.
- Regenerate lockfile after changing direct dependencies (see comment in `requirements-lock.txt`).
