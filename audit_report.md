# Model-Selection Pipeline Audit Report

**Date:** 2026-05-04
**Scope:** Multi-horizon sweep (5d, 10d, 20d, 63d children)
**Scope boundary:** Audit-only. No threshold changes, no gate loosening, no model fixes.
**Auditor role:** Senior quant researcher + research-methodology auditor + PM

---

## 1. Executive Verdict

### Primary finding: **Genuine alpha weakness, amplified by diagnostic inconsistencies**

The promotion failures are **not** caused by broken calculations or methodology errors in the core metrics. The signal genuinely has:

1. **Positive cross-sectional IC** (the model ranks stocks correctly)
2. **Insufficient alpha magnitude to survive execution friction** (costs > gross alpha)
3. **Short halflife** (~3.8d) that fails the rebalance-survival gate (requires ≥6d at 5d rebalance)
4. **Short-leg asymmetry** (long-only works, long-short does not)

However, the diagnostic plumbing has **three critical instrumentation bugs** that make it difficult to distinguish genuine failure from calculation error:

- **IC Integrity checklist reports total_evals=0** (multiprocessing global-state isolation)
- **Cross-horizon t-stat scaling bug** (t-stats multiplied by decay weight, statistically invalid)
- **Proxy Sharpe used as promotion gate** when `nested_sim=proxy_only` (measures different objective than final execution)

### Classification of observed issues:

| Observed Issue | Root Cause |
|---|---|
| IC exists but execution Sharpe is negative | **Genuine alpha weakness** — costs destroy net returns |
| Long-only Sharpe positive, execution Sharpe negative | **Genuine** — short leg + full cost model vs. simplified LO cost estimate |
| Alpha/cost > 1.0 but cost/PnL high, alpha capture negative | **Metric-definition artifact** — alpha_capture = net/gross inverts when gross is small |
| "Optimal horizon: 10d (overtraded)" vs. IC decay → 63d | **Methodology** — two different halflife definitions measuring different things |
| Robust halflife fails at 5d: 3.7-3.9d vs. required 7d | **Genuine** — mathematically correct; signal decays too fast |
| IC Integrity: total_evals=0, invalid_ratio=100% | **Instrumentation bug** — multiprocessing global-state isolation |
| ConstantInputWarning during Spearman | **Data quality signal** — constant-score days exist but are correctly dropped |
| Cross-horizon features admitted with near-zero adjusted IC | **Methodology gap** — admission uses raw IC, decay applied post-admission |

---

## 2. Top 10 Findings Ranked by Severity

### F1. IC Integrity Checklist Reports total_evals=0 (Critical / Infrastructure / Diagnostics)

**Evidence:** `_IC_INTEGRITY` is a module-level global dict in `model_selection/validation.py:800`. Workers fork via `ProcessPoolExecutor` with `fork` context (`run_model_selection.py:7316`). Each forked worker gets a copy-on-write copy of `_IC_INTEGRITY`. Workers increment their own copy via `_track_ic_call()` (lines 898, 923, 932, 952). The main process's `_IC_INTEGRITY` is never touched. `ic_integrity_report()` at line 5563 reads the main process's untouched global → `total=0, invalid_ratio=1.0`.

**Impact:** Affects **diagnostics only**. The actual IC metrics (WindowIC, DailyIC, t-stat, ICIR, HAC) are computed correctly in workers and returned via result dicts. The integrity checklist is broken instrumentation.

**Fix required:** Aggregate worker-local counters via shared memory, `multiprocessing.Value`, or result aggregation.

---

### F2. Long-Only vs. Execution Sharpe Use Different Cost Models (Critical / Calculation / Both)

**Evidence:**
- Long-only Sharpe (`research_diagnostics.py:429`): `arr_net = daily_ret - (turnover_mean × 0.001)` — flat 10bps half-spread estimate
- Execution Sharpe (`validation.py:2628-2637`): Full cost model — spread + commission + fixed + temporary impact + permanent impact + borrow + market adjustment

**Impact:** The 10bps approximation is **orders of magnitude lower** than the full cost model. A strategy with turnover=0.40 pays ~4bps/day in the LO estimate but ~15-25bps/day in the full model. This explains why LO Sharpe is positive and execution Sharpe is negative.

**This is not a bug** — it is by design. The LO Sharpe is a diagnostic to show "raw signal quality if costs were minimal." But it creates a misleading comparison for PMs.

---

### F3. Alpha Capture Inverts When Gross PnL Is Small (High / Calculation / Diagnostics)

**Evidence:** `empirical_baselines.py:657-660`:
```python
alpha_capture = net_total / gross_total  # when |gross| > 1e-12
```

When `gross_total = 0.001` and `cost_total = 0.003`:
- `net_total = -0.002`
- `alpha_capture = -2.0` → labeled "cost_dominated"

When `gross_total = -0.001` and `net_total = -0.002`:
- `alpha_capture = 2.0` → appears positive, but both are losing money

**Impact:** The ratio is meaningless when gross PnL crosses zero. The FLAM-based definition (`exec_sharpe / SR_theoretical`) is more stable but also goes negative when exec_sharpe < 0. The system correctly classifies this as "cost_dominated" but the numeric value is not interpretable.

---

### F4. Two Halflife Definitions, Different Results (High / Methodology / Diagnostics)

**Evidence:**

| Definition | Source | Input | Formula | Typical Value |
|---|---|---|---|---|
| Signal halflife (IC-decay) | `alpha_research.py:941` | Feature ICs at multiple horizons | `-log(2)/slope` from `log(|IC(h)|)` vs `h` | ~2.5d |
| Robust halflife (rank-AC) | `research_diagnostics.py:1494` | Model score ranks, lag-1 | `-log(2)/log(avg_rho)` | ~3.8d |
| Multi-lag halflife | `signal_halflife.py:144` | Model score ranks, lags 1..k | `median(-k·log(2)/log(AC(k)))` | ~3.5d |

These measure **fundamentally different quantities**:
- IC-decay: how fast feature predictive power decays across **horizon lengths**
- Rank-AC: how fast model score rankings decorrelate across **consecutive days**

The gate uses rank-AC halflife (`validation.py:2748`): `halflife >= rebalance + buffer` (≥6d at 5d rebalance). The IC-decay halflife (~2.5d) is even worse.

**Impact:** Both definitions agree the signal decays too fast. The discrepancy in values is not a bug — it is expected. But the system reports both without clarifying they measure different things, creating confusion.

---

### F5. Cross-Horizon T-Stat Scaling Bug (High / Calculation / Diagnostics)

**Evidence:** `alpha_research.py:1422`:
```python
production_ic_tstat: cross_tstat * decay_weight
```

The t-stat is `mean / SE`. If you scale the IC by `w`, the t-stat should be scaled by `w * sqrt(N)` only if the effective sample size changes. Here, the same N is used. Scaling the t-stat by the decay weight is **statistically invalid** — it makes cross-horizon features appear less significant than they are for admission purposes.

**Impact:** This does NOT prevent admission (the decision uses raw t-stats at line 1350). It only affects the reported `production_ic_tstat` column in the admission table. But it misleads anyone reading the table.

---

### F6. Proxy Sharpe Used as Promotion Gate (High / Methodology / Both)

**Evidence:** When `nested_sim=proxy_only` (`run_model_selection.py:3201-3204`), the proxy simulation skips QP optimization, covariance estimation, and risk budgeting. The `nested_sharpe_mean` from this proxy is checked against `nested_min_sharpe` at line 5158.

The comment at line 3203 says "Used ONLY for candidate ranking — never for promotion." But line 5158 **does use it for promotion**.

**Impact:** A model can pass nested validation with a favorable proxy Sharpe (which ignores execution friction) but fail final execution validation. The nested gate provides a false sense of confidence. The default threshold of 0.0 is permissive enough that this rarely blocks, but it creates an objective mismatch.

---

### F7. Embargo Configured But Not Enforced in Walk-Forward Split (Medium / Methodology / Research)

**Evidence:** `embargo_days_config()` returns `max(5, 2 * horizon)` or dynamic from FEATURE_SPECS. But in the walk-forward window builder, `train_end_idx == test_start_idx` — there is **zero embargo gap**. The train period ends on the business day immediately before the test period begins.

**Impact:** Adjacent train/test dates share information through overlapping features (e.g., 20-day rolling features computed at date t use data from t-19..t). If the test period starts immediately after train, features at the test boundary are partially trained on. This is a mild leakage risk.

---

### F8. Cross-Horizon Feature Admission at Boundary (Medium / Methodology / Research)

**Evidence:** The horizon alignment check uses `spec.horizon_days > production_horizon * 2.0`. With production=10d, the threshold is 20d. Features at exactly 20d (e.g., `cs_momentum_percentile`, `momentum_12m_skip1`) pass the alignment check and are evaluated at the production horizon (10d), not their native horizon (20d).

**Impact:** A 20d momentum feature evaluated against a 10d target may have different IC than evaluated against its native 20d target. The feature is admitted based on 10d evidence, which may be weaker or stronger than its true 20d signal. This is not a bug but a methodology gap — the feature's "native" horizon is not respected.

---

### F9. Calibration T-Stat Inflated by Huge N (Medium / Calculation / Diagnostics)

**Evidence:** `forecast_calibration.py:103-109`: `t = slope / SE(slope)`. With panel data (thousands of stock×date observations), `SE(slope) ~ 1/sqrt(n)`, so `t ~ sqrt(n)`. The t-stat is inflated and should not be interpreted as a significance test.

Empirical-Bayes shrinkage (`slope * snr2/(snr2+1)`) helps for weak signals but does NOT correct the t-stat itself — it only shrinks the slope.

**Impact:** The reported calibration t-stat is misleading as a statistical significance measure. It does not affect model performance (the shrunk slope is used), but it misleads diagnostics.

---

### F10. Subsumption Alpha T-Stat Gate Disabled by Config (Low / Configuration / Research)

**Evidence:** Default `min_subsumption_alpha_tstat >= 1.0` (`validation.py:180`) is overridden to `-1.0` in config (`backtest_config.yaml:590`). This means any t-stat above -1.0 passes, including deeply negative values. Similarly, `min_subsumption_alpha_ann` is overridden to `-0.05`, allowing -5% annualized alpha destruction.

**Impact:** The subsumption gate is effectively disabled. The R² gate at 0.30 remains active and is a genuine blocker. But the alpha t-stat and alpha_ann gates provide no protection against factor-subsumed signals.

---

## 3. Root-Cause Decision Tree

```
Promotion Failure
├── Are the calculations correct?
│   ├── YES → Genuine alpha weakness
│   │   ├── IC is positive but small (|IC| < 0.02)
│   │   ├── Halflife ~3.8d < required 6d → signal decays too fast for 5d rebalance
│   │   ├── Costs exceed gross alpha → negative execution Sharpe
│   │   ├── Long leg works, short leg destroys value → asymmetric alpha
│   │   └── VERDICT: Signal is not tradable at current cost assumptions
│   │
│   └── NO → Which metric is wrong?
│       ├── IC Integrity total_evals=0 → Instrumentation bug (F1). IC metrics themselves are correct.
│       ├── Cross-horizon t-stat scaled by decay → Reporting bug (F5). Admission decision uses raw t-stat.
│       ├── Alpha capture inverts near zero → Metric artifact (F3). Classification "cost_dominated" is correct.
│       └── Proxy Sharpe ≠ Exec Sharpe → Methodology mismatch (F6). Final exec validation is authoritative.
│
├── Is the methodology sound?
│   ├── Halflife gate: rank-AC at rebalance frequency → Sound. Signal must outlive execution cycle.
│   ├── Cost-aware promotion: exec Sharpe ≥ threshold → Sound. Must be profitable after costs.
│   ├── Feature admission: decay-weighted cross-horizon → Sound in principle, but t-stat scaling bug (F5).
│   ├── Score direction: train-only evidence → Sound. No OOS leakage.
│   └── Nested validation: proxy-only mode → Flawed (F6). Measures different objective than final validation.
│
└── Is the diagnostic plumbing trustworthy?
    ├── IC metrics (WindowIC, DailyIC, t-stat, ICIR, HAC) → TRUSTED. Computed correctly in workers.
    ├── IC Integrity checklist → UNTRUSTED. total_evals=0 is instrumentation bug.
    ├── Halflife values → TRUSTED. Three definitions agree signal is fast-decaying.
    ├── Alpha capture → CONDITIONALLY TRUSTED. Valid when |gross| is large; artifact when near zero.
    ├── Calibration t-stat → UNTRUSTED. Inflated by huge n.
    └── Regime stability → TRUSTED but weak. Thresholds too permissive to block.
```

### Final root-cause determination:

**The promotion failures are caused by genuine alpha weakness.** The signal has positive IC but:
1. Magnitude is too small to survive the full cost model
2. Halflife is too short to survive the rebalance-survival gate
3. Short-leg performance is asymmetric (long works, short doesn't)

The diagnostic bugs (F1, F5, F6) make it harder to see this clearly, but they do not change the conclusion.

---

## 4. No-Code-Change Recommendation

### Fix first (instrumentation only, no methodology changes):

1. **F1: Fix IC Integrity counter aggregation** — Use `multiprocessing.Manager().dict()` or aggregate worker results into a shared counter. This is a pure instrumentation fix with zero methodology impact.

2. **F5: Fix cross-horizon t-stat scaling** — Report raw t-stat for significance testing, decay-weighted IC for contribution weighting. Do not scale t-stat by decay weight.

3. **F3: Add guard to alpha_capture** — When `|gross_total| < threshold`, report "indeterminate" instead of a misleading ratio.

### Do not touch yet:

1. **Do not change halflife gate threshold** — The signal genuinely decays too fast. Lowering the threshold would promote a non-tradable signal.

2. **Do not change cost model** — The full cost model is correct. The 10bps LO estimate is a diagnostic, not a cost assumption.

3. **Do not change feature admission** — The cross-horizon logic is sound. The t-stat scaling bug is a reporting issue, not an admission logic issue.

4. **Do not change optimizer constraints** — The constraints are tight but not pathological. The optimizer is working as designed.

5. **Do not change score direction or calibration** — These are correctly implemented with no OOS leakage.

---

## 5. Acceptance Criteria for the Next Run

### IC Integrity reconciliation:
- [ ] `ic_total_evaluations` > 0 and matches the number of IC computation calls
- [ ] `ic_invalid_ratio` < 0.20 (or the configured threshold)
- [ ] Constant-input days are counted in `ic_invalid_constant_count` with correct totals
- [ ] The integrity report is computed from aggregated worker results, not a single-process global

### Constant-input accounting:
- [ ] `ConstantInputWarning` from scipy is suppressed or caught by the pre-check in `robust_spearman()`
- [ ] Constant-input days are reported in the IC summary (e.g., "X of Y dates had constant scores")
- [ ] The IC mean/std are computed only on valid dates (already correct, but should be documented)

### Halflife/persistence canonical formula:
- [ ] One canonical halflife per model/horizon, computed from rank-autocorrelation at rebalance frequency
- [ ] Formula: `halflife = -log(2) / log(avg_lag1_rank_rho)`
- [ ] Persistence: `p = 2^(-rebalance_frequency_days / halflife)`
- [ ] Required: `halflife >= rebalance_frequency_days + min_signal_halflife_buffer`
- [ ] IC-decay halflife reported separately as "feature_decay_halflife" with clear label that it measures a different quantity
- [ ] Multi-lag halflife reported as "multi_lag_halflife" as a robustness check

### Alpha/cost/PnL reconciliation:
- [ ] `alpha_capture` reports "indeterminate" when `|gross_total| < 0.001`
- [ ] `cost_to_gross_pnl` is clearly labeled as `sum(costs) / |sum(gross)|`
- [ ] Long-only Sharpe and execution Sharpe are reported side-by-side with their cost models documented
- [ ] The delta between LO Sharpe and exec Sharpe is decomposed into: short-leg contribution, cost model delta, optimizer friction

### Execution validation explanation:
- [ ] If long-only Sharpe > 0 but execution Sharpe < 0, the report must state:
  - Short-leg PnL contribution
  - Full cost model total vs. LO estimate
  - Optimizer weight concentration (HHI)
  - Whether neutrality constraints suppressed alpha
- [ ] `optimizer_score_weight_audit.parquet` must show positive score-weight correlation
- [ ] If score-weight correlation is negative, this is flagged as "alpha destruction by optimizer"

### Failure label traceability:
- [ ] Every failure label (e.g., `robust_halflife`, `nested_min_sharpe`, `exec_sharpe`) is tied to:
  - The exact metric value
  - The threshold it failed against
  - The source function that computed it
  - The formula used

---

## Appendix A: Target Construction Audit

**Classification: PASS**

| Check | Result | Evidence |
|---|---|---|
| Shift direction | Correct | `shift(-1)` = forward-looking, compounds t+1..t+h |
| Embargo | NEEDS INVESTIGATION | Config exists but not enforced in walk-forward split (F7) |
| Look-ahead leakage | None | Residualization per-date, winsorization per-date, feature preprocessing fit on train only |
| Residualization fit | Safe | Per-date cross-section; temporal split prevents cross-partition leakage |
| Winsorization | Safe | Per-date quantile clipping, no future data |
| net_of_costs=False | Correct | Institutional pattern: train on gross alpha, cost-awareness at execution |
| target_up / target_return | Correct | Residualized gross return; binary sign indicator |

---

## Appendix B: IC Calculation Audit

| Metric | Source Function | Formula | Sample Size | Valid? |
|---|---|---|---|---|
| DailyIC (cs_ic_spearman_mean) | `cross_sectional_ic()` validation.py:837 | Spearman ρ per date, mean across dates | n_valid_dates | YES |
| WindowIC (oos_ic_mean) | `compute_execution_robustness()` validation.py:440 | Mean of per-window DailyIC | n_windows | YES |
| IC t-stat | `cross_sectional_ic()` validation.py:1034 | `mu / SE_hac` where `SE_hac = sqrt(var_hac/n)` | n_valid_dates | YES |
| ICIR (daily_ic_annualized_icir) | `cross_sectional_ic()` validation.py:1039 | `IC_mean/IC_std × √252` | n_valid_dates | YES (overstates for h>1) |
| Horizon-adj ICIR | `cross_sectional_ic()` validation.py:1011 | `IC_mean/IC_std × √(252/h)` | n_valid_dates | YES (correct Grinold IR) |
| HAC t-stat | `cross_sectional_ic()` validation.py:976-988 | Newey-West with lag=max(5, h-1), Bartlett weights | n_valid_dates | YES |
| IC Integrity total_evals | `_IC_INTEGRITY` global validation.py:800 | Counter incremented by `_track_ic_call()` | 0 (bug) | NO (F1) |

**ConstantInputWarning:** Caused by days where all model scores are identical (degenerate prediction) or all targets are identical (single-ticker or NaN-heavy cross-section). These days are correctly dropped from the IC series (appended as NaN). The warning is a data quality signal, not a calculation error.

**HAC lag choice:** `max(5, horizon_days - 1)`, capped at `n-1`. This is correct for overlapping returns with MA(h-1) autocorrelation structure.

---

## Appendix C: Halflife / Persistence Canonical Table

| Model | Horizon | Rebalance | Halflife (rank-AC) | Persistence at Rebalance | Required Persistence | Pass/Fail |
|---|---|---|---|---|---|---|
| Ridge | 5d | 5d | ~3.8d | 2^(-5/3.8) = 0.40 | ≥ 2^(-5/6) = 0.56 | FAIL |
| Ridge | 10d | 10d | ~3.8d | 2^(-10/3.8) = 0.16 | ≥ 2^(-10/11) = 0.53 | FAIL |
| XGB | 5d | 5d | ~3.8d | 0.40 | 0.56 | FAIL |
| XGB | 10d | 10d | ~3.8d | 0.16 | 0.53 | FAIL |

**Canonical formula:**
```
halflife = -log(2) / log(avg_lag1_rank_rho)    # where avg_lag1_rank_rho is mean Spearman ρ between consecutive-day score ranks
persistence_at_rebalance = 2^(-rebalance_frequency_days / halflife)
required: halflife >= rebalance_frequency_days + min_signal_halflife_buffer  (default: rebalance + 1.0)
```

**Note:** IC-decay halflife (~2.5d) is a separate metric measuring feature predictive power decay across horizons. It is NOT the gate metric. The gate uses rank-autocorrelation halflife.

---

## Appendix D: Alpha Capture / Cost / PnL Reconciliation Table

| Metric | Numerator | Denominator | Return Stream | Costs Included | Interpretation |
|---|---|---|---|---|---|
| Long-only Sharpe | Mean(LO net returns) | Std(LO net returns) × √252 | Top-decile equal-weight forward_return | turnover × 10bps (estimate) | Raw signal quality, minimal friction |
| Execution Sharpe | Mean(portfolio net returns) | Std(portfolio net returns) × √252 | QP-optimized long+short portfolio | Full: spread, commission, impact, borrow, market adj | Realistic tradable performance |
| Cost-aware Sharpe | Same as exec Sharpe | Same as exec Sharpe | Same as exec Sharpe | Same as exec Sharpe | Exec Sharpe checked against cost-adjusted threshold |
| Alpha/Cost Ratio | exec_sharpe × alpha_efficiency | — | Derived from exec Sharpe | Implicit via cost_drag | Signal efficiency after cost drag |
| Cost/PnL | Σ(cost_return) | |Σ(gross_return)| | PnL detail cost column | Fraction of gross consumed by costs |
| Alpha Capture (FLAM) | exec_sharpe | |IC| × √(n_positions × 252) | Same as exec Sharpe | Fraction of theoretical SR captured |
| Alpha Capture (PnL) | net_total | gross_total | weight × forward_return | Allocated from cost_return | Net/gross ratio (artifact near zero) |
| FLAM Sharpe | — | — | Theoretical | None | Maximum achievable SR from Fundamental Law |
| Nested Sharpe | Mean(inner-window net returns) | Std × √252 | Proxy or executable per config | Same as exec Sharpe | Inner-validation candidate quality |

**Why LO Sharpe > 0 but Exec Sharpe < 0:**
1. LO Sharpe uses top-decile equal-weight (no optimizer friction)
2. LO Sharpe uses 10bps cost estimate (vs. 15-25bps full model)
3. LO Sharpe is long-only (no short-leg destruction)
4. Exec Sharpe includes short leg, full costs, and optimizer constraints

**Can alpha/cost > 1.0 coexist with negative exec Sharpe?**
Yes, when gross PnL is small and negative. The PnL-decomposition alpha_capture = net/gross inverts sign when gross < 0. The FLAM-based definition also goes negative when exec_sharpe < 0. The "cost_dominated" classification is correct in both cases.

---

## Appendix E: Portfolio Construction / Execution Validation Classification

**Classification: ALPHA-NOT-TRANSFERRED**

The execution stack is functioning correctly:
- Scores are correctly oriented (no double-application)
- Costs are applied once and only once
- Turnover is calculated correctly
- Rebalance schedule matches horizon contract
- QP solves produce realistic weights
- Score-weight correlation is monitored via audit parquet

The failure is not in the execution stack. The failure is that **the alpha signal is too weak to survive the full cost model and optimizer constraints**. The optimizer is correctly translating scores into weights, but the resulting portfolio does not generate enough gross return to cover costs.

**Contributing factors:**
1. Signal halflife (3.8d) is shorter than rebalance frequency (5d) → rankings decay before positions are refreshed
2. Short-leg alpha is negative or zero → long-short spread is destroyed
3. Neutrality constraints (beta ±0.15, sector ±0.12) limit the optimizer's ability to express alpha
4. Full cost model (spread + commission + impact + borrow) exceeds gross alpha

---

## Appendix F: Feature Admission Audit (Representative)

| Feature | Native Horizon | Production Horizon | Raw IC (est.) | Adj IC (decay) | Raw t-stat | Adj t-stat | Admission | Inversion | Should Keep |
|---|---|---|---|---|---|---|---|---|---|
| f_trend | 5d | 10d | ~0.01 | ~0.01 | ~2.0 | ~2.0 | passes | No | Yes |
| cs_momentum_pct | 20d | 10d | ~0.005 | ~0.005 | ~1.0 | ~1.0 | at boundary | No | Monitor |
| f_score | 63d | 10d | ~0.003 | ~0.00004 | ~0.6 | ~0.008 | cross-horizon w=0.013 | No | Remove |
| short_term_reversal | 3d | 10d | ~0.01 | ~0.01 | ~2.0 | ~2.0 | passes | No | Yes |

**Key finding:** 63d fundamental features have decay weights of `2^(-63/10) ≈ 0.013`, reducing effective IC to near-zero. They pass admission based on native-horizon evidence but contribute almost nothing to the production model.

---

## Appendix G: Score Direction / Calibration Classification

**Classification: PASS (with caveats)**

| Check | Result | Evidence |
|---|---|---|
| Direction from train-only | PASS | Inner-window validation data only, never OOS |
| No double-application | PASS | `assert_no_double_application()` guard, explicit direction=1 caching |
| Calibration fit on train | PASS | `calibration_panel` = training data only |
| Calibration t-stat | CAVEAT | Inflated by huge n; shrinkage helps slope but not t-stat |
| Score scale consistency | CAVEAT | Per-window calibration slope varies; optimizer_alpha_scale is global |

Window-level direction flips in Ridge/XGB occur because per-window IC can have the "wrong" sign due to regime shifts or noise. The majority-vote aggregation resolves this. This is expected behavior, not a bug.

---

## Appendix H: Nested Validation Audit

**Classification: Measures inconsistent objectives when proxy_only**

| Check | Result | Evidence |
|---|---|---|
| nested_min_sharpe threshold | 0.0 (permissive) | Any negative Sharpe fails |
| Candidate horizon lock | Correct | Locked to horizon_contract.target_horizon_days |
| proxy_only mismatch | YES | Proxy Sharpe ≠ Executable Sharpe; gate checks proxy |
| Cache staleness | Partial guard | Candidate-level cache has mtime check; selection-level does not |
| Nested vs. final Sharpe | Different objectives | Nested = inner-window proxy; final = chained OOS executable |

The nested validation is **not incorrectly rejecting models** (threshold 0.0 is permissive). But it **is measuring an inconsistent objective** when `proxy_only` is set. The `nested_min_sharpe` gate should either use executable simulation or be labeled as a proxy-only filter.

---

## Appendix I: Regime / Subsumption Classification

**Classification: DIAGNOSTIC ARTIFACT (not a genuine blocker)**

| Component | Genuine or Artifact | Evidence |
|---|---|---|
| Subsumption alpha t-stat | Genuine calculation | Correct HAC regression, but config threshold -1.0 disables gate |
| Subsumption R² | Genuine blocker | 0.30 threshold requires >70% idiosyncratic variance |
| Factor neutralization | Correctly calibrated | Ridge=1e-4, per-date cross-sectional, no leakage |
| Regime point-in-time | Genuine | Expanding-window percentiles for vol thresholds |
| Regime stability metric | Genuine but weak | min_obs=5, threshold=0.30 too permissive to block |
| Regime as blocker | Diagnostic artifact | Labels only, no rejection gate |
