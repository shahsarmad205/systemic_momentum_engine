# Incident runbook (regulated ops)

This is the “first 15 minutes” playbook. For general ops issues see [RUNBOOK.md](RUNBOOK.md).

## 1) Immediate containment

- Set `TRADING_HALTED=1` in the production environment.
- If running under systemd, verify the environment file was reloaded (or restart the daily service).

## 2) Collect evidence (do not overwrite)

- Identify the `RUN_ID` for the affected run:
  - `output/live/pipeline.log`
  - `output/live/execution_log.jsonl`
- Preserve:
  - `output/runs/<RUN_ID>/` (snapshots + metadata)
  - `output/live/execution_log.jsonl`
  - `output/live/order_intents.jsonl`
  - any reconciliation output

## 3) Classify

- **Data issue**: cache stale, missing bars, feature build failures.
- **Broker issue**: HTTP 401/403, timeouts, rejected orders.
- **Logic issue**: unexpected exposure, risk limits bypass, duplicate execution.

## 4) Resolution patterns

### Unexpected orders placed

1. Confirm `TRADING_HALTED=1` is set.
2. Pull latest Alpaca positions and compare with `output/portfolio/paper_positions.csv` (or live equivalent).
3. Run reconciliation and record results.
4. If exposure is unacceptable: execute manual close via broker UI or `run_live_trading.py --close-all` (paper first).

### Duplicate run / replay

1. Check `output/live/order_intents.jsonl` for repeated intents with same `as_of|ticker|side`.
2. Identify the second run’s trigger (timer replay, manual rerun, clock skew).
3. Fix scheduler and add a guard if needed.

### Alpaca unauthorized

1. Validate key pair (Key ID vs Secret).
2. Validate base URL matches paper/live.
3. Rotate keys if compromise suspected.

## 5) Post-incident requirements

- Write a short incident report outside the repo and attach the run_id(s).
- Add/strengthen tests (kill switch, idempotency, reconciliation gating) before re-enabling trading.

