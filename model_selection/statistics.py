from __future__ import annotations

import numpy as np


def weighted_recency_mean(vals: np.ndarray, decay_base: float = 0.95) -> float:
    mask = np.isfinite(vals)
    if not np.any(mask):
        return float("nan")
    n = len(vals)
    weights = np.array([decay_base ** (n - 1 - i) for i in range(n)], dtype=float)
    m_vals = vals[mask]
    m_weights = weights[mask]
    return float(np.sum(m_vals * m_weights) / np.sum(m_weights))


def compute_institutional_metrics(
    ic_vals: np.ndarray,
    sharpe_vals: np.ndarray,
    oos_cagr: float,
    oos_max_dd: float,
) -> dict[str, float]:
    ic_finite = ic_vals[np.isfinite(ic_vals)]
    sp_finite = sharpe_vals[np.isfinite(sharpe_vals)]
    n = len(ic_finite)

    ic_mean = float(np.nanmean(ic_finite)) if n > 0 else float("nan")
    ic_std = float(np.nanstd(ic_finite, ddof=1)) if n > 1 else float("nan")

    if np.isfinite(ic_std) and ic_std > 1e-9:
        ic_ir = ic_mean / ic_std
        ic_tstat = ic_mean * np.sqrt(n) / ic_std
    else:
        ic_ir = float("nan")
        ic_tstat = float("nan")

    abs_dd = abs(oos_max_dd) if np.isfinite(oos_max_dd) else float("nan")
    if np.isfinite(oos_cagr) and np.isfinite(abs_dd) and abs_dd > 1e-4:
        calmar = float(np.clip(oos_cagr / abs_dd, -10.0, 10.0))
    else:
        calmar = float("nan")

    # Beat rate = fraction of windows with positive Sharpe.
    # The previous definition AND'd (sharpe>0, ic>0), which silently halved the
    # count for models where IC fluctuates around zero.  That AND is now a
    # separate diagnostic (oos_dual_beat_rate) so it can inform without gating.
    if len(sp_finite) > 0:
        beat_rate = float(np.sum(sp_finite > 0)) / float(len(sp_finite))
    else:
        beat_rate = float("nan")
    dual_beat_rate = (
        float(np.sum((sp_finite > 0) & (ic_finite > 0))) / float(n)
        if (n > 0 and len(sp_finite) == n)
        else float("nan")
    )

    _ic_ir_safe = ic_ir if np.isfinite(ic_ir) else 0.0
    _calmar_safe = calmar if np.isfinite(calmar) else 0.0
    _beat_safe = beat_rate if np.isfinite(beat_rate) else 0.5
    composite = float(_ic_ir_safe * (1.0 + max(_calmar_safe, 0.0)) * _beat_safe)

    return {
        "window_ic_mean": float(ic_mean) if np.isfinite(ic_mean) else float("nan"),
        "window_ic_std": float(ic_std) if np.isfinite(ic_std) else float("nan"),
        "window_ic_ir": float(ic_ir) if np.isfinite(ic_ir) else float("nan"),
        "window_ic_tstat": float(ic_tstat) if np.isfinite(ic_tstat) else float("nan"),
        "oos_ic_ir": float(ic_ir) if np.isfinite(ic_ir) else float("nan"),
        "oos_ic_tstat": float(ic_tstat) if np.isfinite(ic_tstat) else float("nan"),
        "oos_calmar": float(calmar) if np.isfinite(calmar) else float("nan"),
        "oos_beat_rate": float(beat_rate) if np.isfinite(beat_rate) else float("nan"),
        "oos_dual_beat_rate": float(dual_beat_rate) if np.isfinite(dual_beat_rate) else float("nan"),
        "oos_composite": composite,
    }


def compute_psr(daily_returns: np.ndarray, benchmark_sr: float = 0.0) -> float:
    try:
        from scipy.stats import kurtosis as _kurt, norm as _snorm, skew as _skew
    except ImportError:
        return 0.5

    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    t = len(r)
    if t < 10:
        return 0.5
    mu = float(r.mean())
    sigma = float(r.std(ddof=1))
    if sigma < 1e-10:
        return 0.5
    sr_hat = mu / sigma
    sr_benchmark = float(benchmark_sr) / np.sqrt(252.0)
    gamma3 = float(_skew(r))
    gamma4 = float(_kurt(r, fisher=True)) + 3.0
    denom_sq = 1.0 - gamma3 * sr_hat + (gamma4 - 1.0) / 4.0 * sr_hat**2
    if denom_sq <= 0:
        return 0.5
    z = (sr_hat - sr_benchmark) * np.sqrt(t - 1) / np.sqrt(denom_sq)
    return float(_snorm.cdf(z))


def compute_deflated_sharpe(daily_returns: np.ndarray, *, n_trials: int, benchmark_sr: float = 0.0) -> float:
    try:
        from scipy.stats import kurtosis as _kurt, norm as _snorm, skew as _skew
    except ImportError:
        return 0.5

    r = np.asarray(daily_returns, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10:
        return 0.5
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    if sigma <= 1e-12:
        return 0.5
    sr_hat = mu / sigma
    skew = float(_skew(r))
    full_kurt = float(_kurt(r, fisher=True)) + 3.0
    sr_var = (1.0 - skew * sr_hat + ((full_kurt - 1.0) / 4.0) * sr_hat**2) / max(n - 1, 1)
    if not np.isfinite(sr_var) or sr_var <= 1e-12:
        return 1.0 if sr_hat > 0.0 else 0.0

    trials = max(1, int(n_trials))
    if trials > 1:
        euler_gamma = 0.5772156649
        expected_max_z = (
            (1.0 - euler_gamma) * _snorm.ppf(1.0 - 1.0 / trials)
            + euler_gamma * _snorm.ppf(1.0 - 1.0 / (trials * np.e))
        )
        sr_star = max(float(benchmark_sr) / np.sqrt(252.0), float(np.sqrt(sr_var) * expected_max_z))
    else:
        sr_star = float(benchmark_sr) / np.sqrt(252.0)
    return float(_snorm.cdf((sr_hat - sr_star) / np.sqrt(sr_var)))


def _bhy_critical_value(p_values: np.ndarray, alpha: float = 0.05) -> float:
    """
    Benjamini-Hochberg-Yekutieli (BHY) critical value for FDR control under arbitrary dependence.

    Implements the BH-Y procedure: reject H_i if p_(i) <= (i/m) * alpha / c(m)
    where c(m) = sum(1/j, j=1..m) ensures FDR control under any dependence structure.

    Returns the largest p-value threshold that survives correction.
    """
    p = np.asarray(p_values, dtype=float)
    p = p[np.isfinite(p) & (p > 0)]
    m = len(p)
    if m == 0:
        return alpha
    c_m = float(sum(1.0 / j for j in range(1, m + 1)))
    p_sorted = np.sort(p)
    thresholds = (np.arange(1, m + 1, dtype=float) / m) * (alpha / c_m)
    mask = p_sorted <= thresholds
    if not np.any(mask):
        return float("nan")
    idx = np.where(mask)[0][-1]
    return float(p_sorted[idx])


def bhy_adjust_pvalues(p_values: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """
    Return BHY-adjusted p-values (q-values) using the step-up procedure.

    For each p-value, returns the smallest FDR level at which it would be rejected.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    if m == 0:
        return p
    c_m = float(sum(1.0 / j for j in range(1, m + 1)))
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=float)
    adjusted_sorted = np.minimum(1.0, p_sorted * m * c_m / ranks)
    # Enforce monotonicity in sorted-p order, then restore original order.
    adjusted_sorted = np.minimum.accumulate(adjusted_sorted[::-1])[::-1]
    out = np.empty_like(adjusted_sorted)
    out[order] = adjusted_sorted
    return out


def compute_effective_trials_from_search_space(
    *,
    n_models: int,
    feature_families: int,
    horizon_choices: int,
    feature_subsets: int,
    path_types: int,
) -> int:
    """
    Compute the effective number of independent tests from the full search tree.

    Uses the harmonic mean approximation to account for overlapping tests:
    T_effective = (n_models * families * horizons * subsets * paths) / c
    where c = sqrt(correlation_factor) accounts for non-independence.

    For highly correlated tests (features from the same family, overlapping horizons),
    we apply a correlation penalty based on the geometric mean of the branching factors.
    """
    raw = n_models * feature_families * horizon_choices * feature_subsets * path_types
    if raw <= 1:
        return 1
    # Correlation adjustment: when tests share data, the effective number is lower.
    # Conservative estimate uses the square root of the product of branching factors.
    branches = [n_models, feature_families, horizon_choices, feature_subsets, path_types]
    active = [b for b in branches if b > 1]
    if not active:
        return 1
    geo_mean = float(np.prod([b**0.2 for b in active]))
    correlation_penalty = max(1.0, geo_mean * 0.4)
    effective = max(1, int(raw / correlation_penalty))
    return effective


def compute_ic_bhy_threshold(
    ic_tstats: np.ndarray,
    n_days: int,
    alpha: float = 0.05,
) -> float:
    """
    Convert BHY p-value threshold to a minimum absolute IC required for admission.

    Uses the asymptotic normal approximation for the supplied IC t-statistics.
    """
    try:
        from scipy.stats import norm as _snorm
    except ImportError:
        return 0.02
    t = np.asarray(ic_tstats, dtype=float)
    t = t[np.isfinite(t)]
    if len(t) == 0:
        return 0.02
    two_sided_p = np.clip(2.0 * (1.0 - _snorm.cdf(np.abs(t))), np.finfo(float).tiny, 1.0)
    cv = _bhy_critical_value(two_sided_p, alpha=alpha)
    if not np.isfinite(cv):
        return 0.02
    t_critical = _snorm.ppf(1.0 - cv / 2.0)
    # IC threshold: t = ic * sqrt(n) / std_ic, assuming std_ic ≈ 1/sqrt(n) under null
    # So ic ≈ t / sqrt(n) * 1/sqrt(n) = t / n
    return float(t_critical / max(n_days, 1))
