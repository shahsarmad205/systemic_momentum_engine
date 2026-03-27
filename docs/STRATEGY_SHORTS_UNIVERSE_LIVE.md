# Strategy scale: shorts, universe, live promotion

## Short selling (research-gated)

**Default:** long-biased only (`execution.long_only: true`, `execution.enable_shorts: false` in `backtest_config.yaml`).

**Enable shorts only after evidence:**

1. **Config**
   - Set `execution.enable_shorts: true` and `execution.long_only: false` when the backtest compares favourably.
   - Optional cross-sectional book: `backtest.cross_sectional.enabled: true` and `market_neutral: true` forces a short leg when that section is wired for top/bottom names (`top_longs` / `top_shorts` in YAML comments).

2. **Backtest protocol**
   - Same universe slice, same cost model (`execution_costs`), same walk-forward / OOS splits as long-only baseline.
   - Compare: Sharpe, max drawdown, turnover, worst loss days, and capacity assumptions.
   - Record decision and hash of config in research notes or version control comments.

3. **Paper trading**
   - Confirm Alpaca shortable universe, borrow/locate behaviour, and margin usage.
   - No live short risk until paper P&L and reconciliation match expectations for several weeks.

## Universe scale (~500 names)

**Current:** ~150 liquid US names in `backtest_config.yaml` under `tickers`.

**Staged rollout (recommended):**

| Stage | Target count | Operations checklist |
|-------|--------------|----------------------|
| 1 | ~150 (baseline) | `build_feature_matrix` wall time, disk under `data/cache/`, failed download count. |
| 2 | ~300 | Batch downloads, optional concurrency limits, monitor rate limits. |
| 3 | ~500 | Liquidity / ADV filters from `risk` section; alert on pipeline duration SLO (set a target, e.g. under 45 minutes on your runner). |

**Before each stage:** snapshot cache size, median `run_daily_pipeline.py` runtime, and count of skipped tickers.

## Live money (complements `DEPLOYMENT.md`)

- Complete **paper** track record and reconciliation (`scripts/reconcile_positions.py` where applicable).
- Separate live API keys, account-level notional and position limits; smallest feasible capital first.
- Org approvals, monitoring, and formal risk sign-off stay outside the repo.
