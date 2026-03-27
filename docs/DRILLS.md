# Incident drills (monthly)

This is a lightweight drill program designed for a student budget but aligned with regulated-style expectations: **repeatable drills + evidence**.

## Evidence standard

For each drill:

- Record `RUN_ID`
- Save an audit pack: `python scripts/build_audit_pack.py --run-id <RUN_ID>`
- Save a short note (outside repo or in a private ops repo): what you did, what you observed, what you changed.

## Drill 1: Kill switch

1. Set `TRADING_HALTED=1`.
2. Run `run_daily_pipeline.py` and ensure no broker calls happen even if execution is enabled.
3. Confirm the logs show halted + planned orders only.

## Drill 2: Data staleness / missing cache

1. Temporarily point cache dir to an empty folder (or remove one ticker cache file in a safe environment).\n2. Run pipeline and confirm it refuses to trade / alerts.\n3. Restore cache settings.\n\n## Drill 3: Reconciliation drift\n\n1. Introduce a controlled mismatch between broker state and expected snapshot (paper only).\n2. Run `scripts/reconcile_positions.py`.\n3. Confirm drift is detected and escalated.\n\n## Drill cadence\n\n- Monthly until stable.\n- Quarterly afterwards.\n\n*** End Patch}]}$
