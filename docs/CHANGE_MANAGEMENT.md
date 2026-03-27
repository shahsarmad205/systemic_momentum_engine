# Change management (regulated-style)

This repo supports a regulated operating model via **process + enforcement** (GitHub protections, approvals, audit packs).

## Branch protection (required)

Enable on your default branch:

- Require PRs
- Require status checks:
  - CI (ruff, pytest, pip-audit)
- Require **CODEOWNERS** review
- Restrict who can push to the branch

## Approval rules (minimum)

- Changes to trading/risk execution paths require **2-person review**:
  - `trend_signal_engine/brokers/`
  - `trend_signal_engine/run_live_trading.py`
  - `trend_signal_engine/backtest_config.yaml`
- Dependency changes require review:
  - `trend_signal_engine/requirements*.txt`

## Release discipline

For any change that could affect live exposure:

1. Run `pytest` and `ruff check .`
2. Run the pipeline in halted mode (`TRADING_HALTED=1`)
3. Produce an audit pack for the run id:
   - `python scripts/build_audit_pack.py --run-id <RUN_ID>`

