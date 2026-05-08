# Governance (regulated target)

This repository supports a regulated operating model, but **governance is enforced by process** (approvals, access controls, audit trails). This document defines the minimum bar.

## Roles (minimum)

- **Developer**: can propose code/config changes via PR; cannot access live keys by default.
- **Operator**: runs deployments and monitors daily runs; can enable/disable the kill switch; no ability to merge code unilaterally.
- **Approver**: approves risk/config/model promotion changes; reviews audit artifacts.

## Change classes and required approvals

| Change | Examples | Required approvals |
|--------|----------|--------------------|
| **Risk / trading controls** | `risk.*`, `backtest.max_positions`, sizing logic, kill-switch semantics | 2-person review (Operator + Approver) |
| **Model promotion** | `output/learned_weights.json` updates, retrain acceptance thresholds | 2-person review + audit pack |
| **Secrets / live keys** | secret manager entries, switching paper→live | Approver sign-off + ticket outside repo |
| **Infra / deploy** | systemd timers, cron schedules, log shipping | Operator review + change window |

## Promotion controls

`scripts/promote_model.py` enforces manifest checksums, validator gates, production
readiness reports, and clean-git checks before production promotion. Promoted `.pkl`
artifacts are published to the configured object store and represented locally by metadata
JSON only. The default production configuration requires `MODEL_REGISTRY_URI` to be set to
an approved `s3://` or `gs://` prefix.

Live paths (`brokers/`, `risk/`, and `run_daily_pipeline.py`) must not import from
`research/` or `analysis/`. CI runs `tests/test_import_boundaries.py` plus
`ruff --select F401,F841` so unused-code regressions and research/live boundary leaks fail
before merge.

## Required audit artifacts per trading day

From `output/runs/<RUN_ID>/` and `output/live/`:

- `run_metadata.json` (args + snapshots)
- `execution_log.jsonl` entry for the run (`run_id`, `as_of`, halt status, placed orders)
- `order_intents.jsonl` entries for any submitted orders
- Reconciliation output (see `scripts/reconcile_positions.py`)

## Incident response (minimum)

All incidents must have:

1. **Immediate containment**: set `TRADING_HALTED=1`.
2. **Triage**: identify affected run_id(s), verify orders/positions, preserve artifacts off-host.
3. **Postmortem**: timeline, root cause, corrective actions, and tests added.

Tabletop drill cadence: **monthly** until stable; then quarterly.
