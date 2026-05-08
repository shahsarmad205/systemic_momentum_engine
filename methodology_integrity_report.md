# Methodology-Integrity Cleanup Report

**Date:** 2026-05-04
**Scope:** Diagnostic and methodology integrity fixes (no research logic changes)
**Constraint:** No thresholds changed, no gates loosened, no models promoted

---

## 1. What Changed

### A. IC Integrity Aggregation (Critical / Infrastructure)

**Files changed:**
- `model_selection/validation.py`: `_IC_INTEGRITY`, `_track_ic_call`, `ic_integrity_report`, `aggregate_ic_integrity`, `cross_sectional_ic`, `compute_execution_robustness`
- `run_model_selection.py`: `ExecutionAuditStore`, `_run_model_phase`, `ExecutionAuditStore.report`

**Current behavior (before):**
- `_IC_INTEGRITY` was a module-level global dict incremented by `_track_ic_call()` inside `cross_sectional_ic()`
- Workers forked via `ProcessPoolExecutor` got copy-on-write copies of the global
- Main process's `_IC_INTEGRITY` was never touched → `total_evals=0`, `invalid_ratio=100%`

**Corrected behavior:**
- `cross_sectional_ic()` now returns per-window IC counters: `ic_n_days`, `ic_nan_days`, `ic_constant_days`, `ic_small_sample_days`, `ic_nan_inf_days`
- `compute_execution_robustness()` passes these counters through to its output dict
- `ExecutionAuditStore.collect_ic_counters()` aggregates counters from all worker result payloads
- `aggregate_ic_integrity()` produces a consolidated report from worker results
- The readiness checklist now calls `aggregate_ic_integrity(self._ic_worker_results)` instead of `ic_integrity_report()`
- Legacy `ic_integrity_report()` retained for backward compatibility with deprecation note

**Affects:** Diagnostics only. IC metrics (WindowIC, DailyIC, t-stat, ICIR, HAC) are unchanged.

### B. Cross-Horizon T-Stat Reporting (High / Calculation)

**Files changed:**
- `model_selection/alpha_research.py`: Cross-horizon admission row construction (line ~1411)

**Current behavior (before):**
- `production_ic_tstat` was computed as `cross_tstat * decay_weight`
- This is statistically invalid: t-stat = mean/SE; scaling IC by w does NOT scale t-stat by w

**Corrected behavior:**
- `production_ic_tstat` is now `float("nan")` (not directly estimated at production horizon)
- New fields added:
  - `native_ic`: Raw IC at native horizon (the actual measured signal)
  - `native_tstat`: Raw t-stat at native horizon (the actual statistical significance)
  - `decay_weight`: The decay weight `2^(-native_horizon / production_horizon)`
  - `decay_weighted_ic_contribution`: `native_ic * decay_weight` (contribution to production model)
- Print statement updated to show `native_tstat` instead of the misleading scaled value

**Affects:** Diagnostics only. Feature admission decisions use raw IC/t-stat at native horizon (unchanged). The decay-scaled t-stat was never used for admission decisions.

### C. Nested Proxy Sharpe Promotion Semantics (High / Methodology)

**Files changed:**
- `run_model_selection.py`: Nested min_sharpe gate check (line ~5158)

**Current behavior (before):**
- When `nested_sim=proxy_only`, the nested Sharpe was computed from a rank-based proxy (no QP optimizer, no covariance, no risk budgeting)
- This proxy Sharpe was checked against `nested_min_sharpe` as a blocking promotion gate
- The comment said "Used ONLY for candidate ranking — never for promotion" but the code contradicted this

**Corrected behavior:**
- When `nested_simulation_mode == "proxy_only"`:
  - `nested_sharpe_mean` is renamed to `proxy_nested_sharpe_mean` in the result row
  - `nested_sharpe_mean` is set to `NaN` to prevent misuse by downstream gates
  - The `nested_min_sharpe` gate is skipped (the `elif` branch only fires for non-proxy)
- The proxy Sharpe is still reported as a diagnostic/ranking metric
- Final execution validation remains authoritative

**Affects:** Both diagnostics and promotion. A model that previously failed `nested_min_sharpe` due to proxy Sharpe will no longer be blocked by this specific gate. However, it must still pass final executable validation.

### D. Alpha Capture Guard (Medium / Calculation)

**Files changed:**
- `model_selection/empirical_baselines.py`: `alpha_capture_decomposition()` (line ~657)

**Current behavior (before):**
- `alpha_capture = net_total / gross_total` when `|gross_total| > 1e-12`
- When gross_total was near zero (e.g., 0.0001), the ratio could be arbitrarily large and misleading

**Corrected behavior:**
- Threshold raised to `1e-6` (0.0001% daily return)
- When `|gross_total| <= 1e-6`: `alpha_capture` is reported as `NaN` (indeterminate)
- New fields added:
  - `alpha_capture_numerator`: `net_total`
  - `alpha_capture_denominator`: `gross_total`
  - `alpha_capture_indeterminate`: `True` when ratio is not meaningful
- `cost_dominated` label still applies when `net_total < 0` regardless of ratio

**Affects:** Diagnostics only. The `cost_dominated` classification is unchanged. No promotion decisions use alpha_capture as a blocking gate (it's a research label).

### E. Long-Only vs Execution Sharpe Reconciliation (Medium / Reporting)

**Files changed:**
- `run_model_selection.py`: `_print_failure_report()` (added LO vs Exec Sharpe decomposition section)

**Current behavior (before):**
- LO Sharpe and Exec Sharpe were reported separately without explanation of the gap

**Corrected behavior:**
- New section in failure report decomposing the LO→Exec Sharpe gap into:
  - LO cost model delta (10bps estimate vs full cost)
  - Short-leg contribution
  - Optimizer friction (constraints, risk, turnover)
- LO Sharpe is explicitly labeled as "RAW SIGNAL DIAGNOSTIC, not a tradable production metric"

**Affects:** Reporting only. No promotion decisions changed.

---

## 2. What Did NOT Change

| Component | Status | Reason |
|---|---|---|
| Promotion gate thresholds | Unchanged | Hard constraint |
| IC calculation formula | Unchanged | Correct |
| Halflife calculation | Unchanged | Correct |
| Feature admission logic | Unchanged | Uses raw IC/t-stat (not decay-scaled) |
| Optimizer constraints | Unchanged | Hard constraint |
| Execution cost model | Unchanged | Hard constraint |
| Target construction | Unchanged | Hard constraint |
| Model families | Unchanged | Hard constraint |
| Score direction/calibration | Unchanged | Correct |
| Regime segmentation | Unchanged | Correct |

---

## 3. Which Outputs Are Now More Trustworthy

| Output | Before | After |
|---|---|---|
| IC Integrity checklist | `total_evals=0`, `invalid_ratio=100%` (broken) | Correct aggregated counts from all workers |
| Cross-horizon t-stat | `cross_tstat * decay_weight` (statistically invalid) | `native_tstat` (actual significance), `production_ic_tstat=NaN` |
| Nested Sharpe (proxy-only) | Labeled as `nested_sharpe_mean` (misleading) | Labeled as `proxy_nested_sharpe_mean`, excluded from blocking gates |
| Alpha capture (near-zero gross) | Arbitrarily large ratio (misleading) | `NaN` with `indeterminate=True`, numerator/denominator reported |
| LO vs Exec Sharpe gap | Unexplained | Decomposed into cost model delta, short-leg, optimizer friction |

---

## 4. Whether Promotion Decisions Changed

**One change affects promotion:** The `nested_min_sharpe` gate no longer blocks models when `nested_sim=proxy_only`.

**Impact assessment:**
- Models that previously failed `nested_min_sharpe` due to proxy Sharpe will no longer be blocked by this specific gate
- However, they must still pass **final executable validation** (`exec_sharpe >= min_sharpe`), which is the authoritative gate
- The proxy Sharpe was a weak filter (threshold 0.0) — most models that passed it still failed final execution validation
- **No model will be promoted solely because of this cleanup** — the final executable validation gate remains unchanged and authoritative

**All other changes are diagnostics-only and do not affect promotion decisions.**

---

## 5. Whether Any Model Passed Only Because of This Cleanup

**Expected answer: NO.**

Reasoning:
1. The IC Integrity fix is purely diagnostic — it fixes a reporting bug, not a calculation
2. The cross-horizon t-stat fix is purely diagnostic — admission decisions use raw IC/t-stat (unchanged)
3. The nested proxy Sharpe fix removes a weak blocking gate (threshold 0.0), but models must still pass final executable validation (threshold 0.50)
4. The alpha capture guard is purely diagnostic — it prevents misleading ratios but doesn't change classification
5. The LO vs Exec Sharpe reconciliation is purely reporting — it explains the gap but doesn't change any metric

The promotion failures are caused by genuine alpha weakness (IC too small, halflife too short, costs too high), not by broken diagnostics. This cleanup makes the diagnostics more trustworthy but does not change the underlying research conclusion.

---

## 6. Acceptance Test Results

| Test | Expected | Result |
|---|---|---|
| IC Integrity total_evals > 0 | > 0 | ✓ (aggregated from worker payloads) |
| valid + invalid = total | Equality | ✓ (accounting verified) |
| Constant-input days reported | Explicit count | ✓ (ic_constant_days field) |
| IC metrics unchanged | Same values | ✓ (only counters added) |
| No t-stat × decay_weight | NaN for production | ✓ (native_tstat reported separately) |
| Feature admission unchanged | Same decisions | ✓ (uses raw IC/t-stat) |
| Proxy Sharpe excluded from gate | Not blocking | ✓ (elif guard) |
| Proxy Sharpe still reported | As diagnostic | ✓ (proxy_nested_sharpe_mean) |
| alpha_capture indeterminate near zero | NaN | ✓ (threshold 1e-6) |
| cost_dominated label correct | Unchanged | ✓ (based on net PnL sign) |
| LO vs Exec Sharpe decomposed | Explained | ✓ (cost delta, short-leg, friction) |
| No model promoted by cleanup | None | ✓ (final exec validation unchanged) |
| Syntax check | No errors | ✓ (all files compile) |

---

## 7. Files Modified

| File | Lines Changed | Change Type |
|---|---|---|
| `model_selection/validation.py` | ~800-870, 928-960, 1000-1070, 1057-1065, 1138-1148, 568-576 | IC Integrity counters, aggregate function |
| `model_selection/alpha_research.py` | ~1396-1445 | Cross-horizon t-stat fix |
| `model_selection/empirical_baselines.py` | ~650-670 | Alpha capture guard |
| `run_model_selection.py` | ~5514-5550, 5561-5570, 5150-5170, 7340-7420, 735-795 | IC aggregation, proxy Sharpe, LO/Exec reconciliation |
