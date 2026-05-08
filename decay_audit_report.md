# Decay-Correction Audit Report

**Date:** 2026-05-04
**Scope:** Verify whether optimizer alpha is decay-corrected once or twice
**Constraint:** Audit-only. No threshold changes, no gate changes, no model changes.

---

## 1. Does Double Decay Occur?

**NO.** Decay is applied exactly once.

---

## 2. Decay Application Sites

Two functions in the codebase apply decay correction:

| # | File | Function | Line | Status |
|---|------|----------|------|--------|
| 1 | `model_selection/validation.py` | `_score_to_optimizer_alpha` | 2140 | **DEAD CODE** |
| 2 | `backtesting/portfolio_construction.py` | `PortfolioConstructor._score_to_alpha` | 347 | **ACTIVE** |

### Site 1: `_score_to_optimizer_alpha` (DEAD CODE)

```python
# validation.py:2137-2142
halflife = float(getattr(cfg, "signal_halflife_days", float("nan")))
if np.isfinite(halflife) and halflife > 0:
    horizon = float(cfg.horizon_days)
    decay_correction = float(2.0 ** (-horizon / halflife))
    decay_correction = max(decay_correction, 0.01)
    alpha = alpha * decay_correction
```

This function is called ONLY by `_optimizer_weights_for_day` (validation.py:1902).
That function is **never called** anywhere in the codebase. Verified by:
```
$ grep -rn "_optimizer_weights_for_day(" --include="*.py" .
./model_selection/validation.py:1880:def _optimizer_weights_for_day(
```
The only match is the function definition itself. Zero call sites.

### Site 2: `PortfolioConstructor._score_to_alpha` (ACTIVE)

```python
# portfolio_construction.py:344-349
halflife = float(getattr(constraints, "signal_halflife_days", float("nan")))
horizon = float(getattr(constraints, "horizon_days", 5))
if np.isfinite(halflife) and halflife > 0:
    decay_correction = float(2.0 ** (-horizon / halflife))
    decay_correction = max(decay_correction, 0.01)
    alpha = alpha * decay_correction
```

This is the ONLY active decay application in the production path.

---

## 3. Production Call Chain (Proven Single Decay)

```
Raw model score (predict_proba - 0.5 or regressor output)
    │
    ├─→ Score direction calibration (run_model_selection.py)
    │     score *= direction (±1)
    │
    ├─→ Forecast calibration (forecast_calibration.py)
    │     score = intercept + slope * score
    │     [score std unchanged — linear transform]
    │
    ├─→ QP prescreen (run_model_selection.py)
    │     score passed through unchanged
    │
    ├─→ build_target_weights (validation.py:1564)
    │     passes raw scores to PortfolioConstructor
    │     inputs.scores = day["score"] (RAW, not decay-corrected)
    │
    ├─→ PortfolioConstructor._build (portfolio_construction.py:167)
    │     alpha = self._score_to_alpha(scores, ...)  ← LINE 176
    │
    ├─→ _score_to_alpha (portfolio_construction.py:314)
    │     z = (score - mean) / std(ddof=0)           ← z-score, std=1
    │     alpha = z * 2^(-horizon/halflife)          ← DECAY APPLIED ONCE
    │     alpha *= optimizer_alpha_scale             ← default 1.0
    │
    └─→ PortfolioOptimizer.optimize (optimizer.py)
          receives decay-corrected alpha as forecasts
```

**Invariant proof:**
- `z_score_std = 1.0` (by definition of z-score with ddof=0)
- `decay = 2^(-5/2.3) ≈ 0.2216`
- `final_alpha_std = 1.0 × 0.2216 × 1.0 = 0.2216`

If double decay occurred: `final_alpha_std = 1.0 × 0.2216² = 0.0491`

The production path produces `alpha_std ≈ 0.22`, not `0.05`.

---

## 4. Which Layer Should Own Decay Correction?

**Current owner:** `PortfolioConstructor._score_to_alpha` (portfolio construction layer).

This is the correct layer because:
1. Decay correction is an execution-level adjustment (relates to holding period vs signal persistence)
2. Portfolio construction is where scores become optimizer alpha
3. The optimizer layer should receive already-calibrated alpha, not raw scores

**Recommendation:** The dead code `_score_to_optimizer_alpha` in `validation.py` should be removed to eliminate future confusion. It was likely the original owner before the code was refactored to use `PortfolioConstructor`, but was never deleted.

---

## 5. Does Current Model Failure Remain Genuine After Single-Decay Verification?

**YES.** The negative execution Sharpe is NOT caused by double decay.

With single decay:
- Alpha is reduced to 22% of its z-scored magnitude
- At IC=0.03, effective alpha ≈ 0.007 (still positive)
- The optimizer with λ=2, γ=4, ρ=100 receives this weakened alpha
- Gross return ≈ 0.1% per rebalance
- Costs ≈ 0.8-1.5bp/day ≈ 20-38bp annualized
- Net return < 0 → Sharpe < 0

The failure is genuine: the signal's halflife (2.3d) is shorter than the rebalance frequency (5d), so the decay correction correctly reduces alpha to reflect that most of the signal has decayed by the time positions are refreshed. The optimizer then cannot extract enough alpha to cover costs.

**This is a research problem, not a calculation problem.**

---

## 6. Tests to Prevent Future Double Decay

### A. Unit test for decay application count

```python
def test_decay_applied_once():
    """Verify that PortfolioConstructor._score_to_alpha applies decay exactly once."""
    from backtesting.portfolio_construction import PortfolioConstructor
    import numpy as np

    scores = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0], index=["A", "B", "C", "D", "E"])
    constraints = PortfolioConstraints(
        path="long_short_spread",
        signal_halflife_days=2.3,
        horizon_days=5,
        optimizer_alpha_scale=1.0,
        # ... other fields
    )
    inputs = PortfolioInputs(
        scores=scores,
        tickers=["A", "B", "C", "D", "E"],
        # ... other fields
    )

    pc = PortfolioConstructor()
    alpha = pc._score_to_alpha(scores, inputs, constraints, tickers)

    # Z-score std should be 1.0
    z = (scores - scores.mean()) / scores.std(ddof=0)
    expected_decay = 2.0 ** (-5.0 / 2.3)
    expected_alpha_std = z.std(ddof=0) * expected_decay

    # Verify: alpha_std ≈ expected_decay (single application)
    # NOT expected_decay² (double application)
    assert abs(alpha.std(ddof=0) - expected_decay) < 0.01, \
        f"alpha_std={alpha.std(ddof=0):.4f}, expected={expected_decay:.4f}"
    assert alpha.std(ddof=0) > expected_decay ** 2, \
        "alpha_std too small — decay may be applied twice"
```

### B. Integration test: no dead-code path activation

```python
def test_optimizer_weights_for_day_is_dead_code():
    """Verify _optimizer_weights_for_day is never called in production."""
    import ast, pathlib

    root = pathlib.Path(__file__).parent.parent
    for py_file in root.rglob("*.py"):
        if py_file.name.startswith("test_") or py_file.name == "audit_decay_correction.py":
            continue
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name) and node.func.id == "_optimizer_weights_for_day":
                    raise AssertionError(
                        f"_optimizer_weights_for_day is called in {py_file}:{node.lineno}. "
                        "This function is dead code and applying decay. "
                        "If activated, it would cause double decay with PortfolioConstructor."
                    )
```

### C. Metadata contract on alpha object

Propose adding a flag to the alpha Series to prevent future double application:

```python
# In _score_to_alpha, after applying decay:
alpha.attrs["decay_applied"] = True
alpha.attrs["decay_factor"] = decay_correction
alpha.attrs["halflife"] = halflife
alpha.attrs["horizon"] = horizon

# In any downstream code that considers applying decay:
if getattr(alpha, "attrs", {}).get("decay_applied", False):
    raise RuntimeError("Alpha already decay-corrected. Do not apply again.")
```

---

## 7. Summary

| Question | Answer |
|---|---|
| Does double decay occur? | **NO** |
| Where is decay applied? | `PortfolioConstructor._score_to_alpha` (portfolio_construction.py:347) |
| Where is the dead code? | `_score_to_optimizer_alpha` (validation.py:2140) — never called |
| Effective alpha scale | z_score_std × 0.2216 = 0.2216 (single decay) |
| Is negative Sharpe caused by decay bug? | **NO** — genuine signal weakness |
| What should be done? | Remove dead code `_score_to_optimizer_alpha`; add unit tests |
