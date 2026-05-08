"""Shared statistical utilities for the model_selection research stack.

Extracted from duplicated implementations across:
  - ic_diagnostics_engine.py
  - conditional_alpha_engine.py
  - signal_decay_engine.py
  - research_numerics_core.py
  - research_diagnostics.py
  - alpha_research.py

All functions here are behaviorally identical to their source implementations.
No research math has been changed.
"""
from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as scipy_stats


# ── Newey-West HAC t-stat (standard form) ────────────────────────────────────
# Source: ic_diagnostics_engine.py:281, conditional_alpha_engine.py:161,
#         research_numerics_core.py:1423
# These three copies are byte-identical (modulo explicit float() casts).
#
# NOT merged (behaviorally different):
#   - signal_decay_engine.py:463  → different guard (n < nw_lag*2+5), different
#     se floor (1e-12 vs 1e-15), no np.isfinite check on gamma_k
#   - alpha_research.py:239       → horizon-aware lag selection, returns nan
#   - research_diagnostics.py:143 → takes pd.Series, different guard (n < 4)

def hac_tstat(ics: np.ndarray, nw_lag: int) -> float:
    """Newey-West HAC t-stat for an IC (or any) series.

    Args:
        ics: 1-D array of values (e.g. daily ICs).
        nw_lag: Number of Newey-West lags for autocorrelation correction.

    Returns:
        HAC-adjusted t-statistic. Returns 0.0 if series too short or variance
        too small.
    """
    n = len(ics)
    if n < 5:
        return 0.0
    mean_ic = float(np.mean(ics))
    var = float(np.var(ics, ddof=1))
    if var < 1e-15:
        return 0.0
    for k in range(1, min(nw_lag + 1, n)):
        if n - k < 2:
            continue
        gamma_k = float(np.cov(ics[k:], ics[:-k])[0, 1])
        if np.isfinite(gamma_k):
            var += 2.0 * (1.0 - k / (nw_lag + 1)) * gamma_k
    se = float(np.sqrt(max(var / n, 1e-15)))
    return mean_ic / se


# ── p-value from t-stat ──────────────────────────────────────────────────────
# Source: ic_diagnostics_engine.py:301, conditional_alpha_engine.py:179
# Byte-identical.

def p_from_tstat(t: float, n: int) -> float:
    """Two-sided p-value from a t-statistic.

    Args:
        t: t-statistic value.
        n: Number of observations used to compute the t-stat.

    Returns:
        Two-sided p-value. Returns 1.0 if t is non-finite or n < 3.
    """
    if not np.isfinite(t) or n < 3:
        return 1.0
    return float(2.0 * (1.0 - scipy_stats.t.cdf(abs(t), df=max(n - 2, 1))))


# ── Benjamini-Hochberg q-values ──────────────────────────────────────────────
# Source: ic_diagnostics_engine.py:310 (public name),
#         conditional_alpha_engine.py:185 (private name _benjamini_hochberg)
# Byte-identical.

def benjamini_hochberg(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg q-values (step-up FDR control).

    Args:
        p_values: 1-D array of raw p-values.

    Returns:
        Array of q-values, same length as input.
    """
    m = len(p_values)
    if m == 0:
        return np.array([])
    order = np.argsort(p_values)
    q = np.ones(m)
    for i, idx in enumerate(order):
        q[idx] = min(p_values[idx] * m / (i + 1), 1.0)
    # Ensure monotonicity
    for i in range(m - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i + 1]])
    return q


# ── Benjamini-Yekutieli q-values ─────────────────────────────────────────────
# Source: ic_diagnostics_engine.py:325 (public name),
#         conditional_alpha_engine.py:198 (private name _benjamini_yekutieli)
# Byte-identical.

def benjamini_yekutieli(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Yekutieli q-values (FDR control under arbitrary dependency).

    Args:
        p_values: 1-D array of raw p-values.

    Returns:
        Array of q-values, same length as input.
    """
    m = len(p_values)
    if m == 0:
        return np.array([])
    c_m = sum(1.0 / j for j in range(1, m + 1))
    order = np.argsort(p_values)
    q = np.ones(m)
    for i, idx in enumerate(order):
        q[idx] = min(p_values[idx] * m * c_m / (i + 1), 1.0)
    for i in range(m - 2, -1, -1):
        q[order[i]] = min(q[order[i]], q[order[i + 1]])
    return q


# ── IC quality classification ────────────────────────────────────────────────
# Source: ic_diagnostics_engine.py:435, signal_decay_engine.py:481,
#         research_numerics_core.py:1497
# All three copies are byte-identical.

def ic_quality(
    n_dates: int,
    avg_breadth: int,
    mean_ic: float,
    t_stat: float,
    stability: float,
) -> str:
    """Classify IC estimate quality using standard thresholds.

    Args:
        n_dates: Number of dates in the IC series.
        avg_breadth: Average cross-sectional breadth.
        mean_ic: Mean IC value.
        t_stat: HAC-adjusted t-statistic.
        stability: Sign consistency (fraction of same-sign ICs).

    Returns:
        One of: "insufficient", "low", "medium", "high".
    """
    if n_dates < 20 or avg_breadth < 5:
        return "insufficient"
    if abs(t_stat) < 1.0 or stability < 0.4:
        return "low"
    if abs(t_stat) >= 2.0 and stability >= 0.7 and n_dates >= 50:
        return "high"
    return "medium"


# ── Winsorization (numpy array form) ─────────────────────────────────────────
# Source: ic_diagnostics_engine.py:624, conditional_alpha_engine.py:221
# Byte-identical (conditional_alpha_engine has default q=0.025, ic_diagnostics
# requires q explicitly). We expose q with a default to cover both call sites.
#
# NOT merged (behaviorally different):
#   - short_modeling.py:118 → takes pd.Series, uses s.quantile not np.nanpercentile

def winsorize(x: np.ndarray, q: float = 0.025) -> np.ndarray:
    """Winsorize a numpy array at quantile q on both tails.

    Args:
        x: 1-D numpy array.
        q: Quantile to clip (e.g. 0.025 clips bottom 2.5% and top 2.5%).

    Returns:
        Winsorized array. Returns input unchanged if q <= 0 or len(x) < 10.
    """
    if q <= 0 or len(x) < 10:
        return x
    lo, hi = np.nanpercentile(x, [q * 100, (1 - q) * 100])
    return np.clip(x, lo, hi)


# ── Standardization ──────────────────────────────────────────────────────────
# Source: ic_diagnostics_engine.py:632
# Single implementation, no duplicates. Included here so engines can import
# from one place.

def standardize(x: np.ndarray | float) -> np.ndarray:
    """Standardize to zero mean, unit variance.

    Args:
        x: numpy array or scalar.

    Returns:
        Standardized array. If std < 1e-15, returns centered (demeaned) array
        without scaling.
    """
    if isinstance(x, float):
        return np.array([x])
    x = np.asarray(x, dtype=float)
    std = np.nanstd(x)
    if std < 1e-15:
        return x - np.nanmean(x)
    return (x - np.nanmean(x)) / std


# ── Robust Spearman correlation ──────────────────────────────────────────────
# Suppresses ConstantInputWarning from scipy when either input is degenerate.
# Used by conditional_alpha_engine to avoid per-sleeve stderr pollution.

import warnings as _warnings


def robust_spearmanr(
    a: np.ndarray,
    b: np.ndarray,
    min_length: int = 3,
) -> tuple[float, float]:
    """Spearman correlation that handles constant/degenerate input gracefully.

    Returns (nan, nan) instead of raising ConstantInputWarning when:
      - Either input has zero variance
      - Either input contains all NaN
      - Length < min_length

    Args:
        a: First array.
        b: Second array.
        min_length: Minimum valid observation count (default 3).

    Returns:
        (correlation, p_value). (nan, nan) if degenerate.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape:
        return np.nan, np.nan
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < min_length:
        return np.nan, np.nan
    a_clean = a[valid]
    b_clean = b[valid]
    if np.nanstd(a_clean) < 1e-15 or np.nanstd(b_clean) < 1e-15:
        return np.nan, np.nan
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", scipy_stats.ConstantInputWarning)
        c, p = scipy_stats.spearmanr(a_clean, b_clean)
    return float(c) if np.isfinite(c) else np.nan, float(p) if np.isfinite(p) else np.nan
