from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Iterable

import numpy as np
import pandas as pd

from model_selection.research_contract import FEATURE_SPECS
from model_selection.residualization import residualize_against_controls, safe_center
from model_selection.statistics import bhy_adjust_pvalues, compute_ic_bhy_threshold
from model_selection.training import TargetConfig, add_institutional_targets
from model_selection.validation import ExecutionCostConfig


DEFAULT_DECAY_HORIZONS: tuple[int, ...] = (1, 2, 3, 5, 10, 20)
ALPHA_RESEARCH_SCHEMA_VERSION = "alpha_research_bhy_fail_closed_v2"
TARGET_RAW = "raw_return"
TARGET_RESIDUAL = "residual_return"
TARGET_NET_RESIDUAL = "net_residual_return"
_TARGET_HORIZON_RE = re.compile(r"(?:^|_)(?P<horizon>\d+)d$")


@dataclass(frozen=True)
class AlphaAdmissionConfig:
    """Feature admission policy for production model selection.

    P11: Stability screening gates — institutional-grade feature quality control.
    Features must pass not only IC magnitude gates but also consistency gates:
      - ic_cv_max: max coefficient of variation of IC across horizons (lower = more stable)
      - ic_sign_flip_max: max allowed sign flips of rolling IC (0 = never flips)
      - min_halflife_days: minimum signal decay halflife (reject features that decay too fast)

    P12: Cross-horizon feature integration (institutional fix).
    Features at different native horizons are NOT rejected — instead they are
    weighted by exp(-h_diff / production_horizon) so that a 20d feature
    contributes 2^(-10/20) ≈ 0.71 of its IC to a 10d prediction. This is
    the Two Sigma / AQR approach: ensemble across horizons, don't discard.
    """

    enabled: bool = True
    enforce: bool = True
    horizons: tuple[int, ...] = DEFAULT_DECAY_HORIZONS
    production_horizon: int = 5
    min_coverage: float = 0.80
    min_abs_ic: float = 0.001
    min_ic_tstat: float = 0.50
    min_monotonicity: float = 0.50
    min_regime_stability: float = 0.50
    min_ic_valid_days: int = 20
    min_regime_valid_days: int = 20
    min_spread_valid_days: int = 20
    min_monotonicity_valid_days: int = 20
    min_marginal_abs_ic: float = 0.00025
    min_marginal_residual_variance_ratio: float = 0.05
    allow_inversion: bool = True
    minimum_admitted_features: int = 0
    fail_if_below_minimum: bool = False
    enforce_horizon_alignment: bool = True
    horizon_alignment_multiplier: float = 2.0
    residual_ridge: float = 1e-4
    winsor_q: float = 0.01
    max_abs_return: float = 5.0
    bhy_alpha: float = 0.05
    ic_tstat_hac_max_lag: int = 5
    apply_bhy_correction: bool = True
    multi_horizon_admission: bool = False
    fallback_mode: str = "percentile"
    min_admission_percentile: float = 0.60
    max_features_by_rank: int = 12
    min_admitted_for_fallback: int = 3
    # P11 stability gates — configurable via backtest_config.yaml:
    #   model_selection.alpha_research.stability.ic_cv_max
    #   model_selection.alpha_research.stability.ic_sign_flip_max
    #   model_selection.alpha_research.stability.min_halflife_days
    ic_cv_max: float = 2.0  # max IC coefficient of variation across horizons
    ic_sign_flip_max: int = 2  # max allowed IC sign flips across horizons
    min_halflife_days: float = 1.0  # minimum signal decay halflife
    # P12: Cross-horizon feature integration
    # When enabled, features at non-production horizons are admitted with
    # decay-weighted IC rather than being rejected. The weight is:
    #   w(h_feat) = 2^(-h_feat / production_horizon) for h_feat > production_horizon
    #   w(h_feat) = 1.0 for h_feat <= production_horizon
    cross_horizon_admission: bool = True
    # P27: Enable verbose stability gate logging
    log_stability_gates: bool = False


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _two_sided_p_from_tstat(tstats: np.ndarray) -> np.ndarray:
    """Return asymptotic two-sided p-values for already-standardized t-statistics."""
    arr = np.asarray(tstats, dtype=float)
    try:
        from scipy.stats import norm as _snorm
    except ImportError:
        p = 2.0 * (1.0 - np.minimum(1.0, np.abs(arr) / 3.0))
    else:
        p = 2.0 * (1.0 - _snorm.cdf(np.abs(arr)))
    return np.clip(p, np.finfo(float).tiny, 1.0)


def _forward_return_from_daily(
    df: pd.DataFrame,
    horizon: int,
    *,
    halflife_days: float | None = None,
) -> pd.Series:
    """
    Compute t+1 through t+h cumulative return from daily returns.

    When ``halflife_days`` is provided, returns a decay-weighted cumulative
    return instead of an equal-weighted compound.  Each day t+1 through t+h
    is weighted by w_d = 2^(-d/halflife), so near-term returns dominate the
    target.  This is the institutional fix for the signal-halflife vs
    prediction-horizon mismatch: when the signal decays in 2-3 days but the
    model predicts a 10-day return, 80-95% of the target variance is noise.
    Decay-weighted targets reduce the noise contribution from later days.
    """
    if "daily_return" not in df.columns:
        if horizon == 5 and "forward_return" in df.columns:
            return _safe_numeric(df["forward_return"])
        return pd.Series(np.nan, index=df.index, dtype=float)

    work = df[["ticker", "date", "daily_return"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["_r"] = _safe_numeric(work["daily_return"])
    out = pd.Series(np.nan, index=df.index, dtype=float)
    h = max(1, int(horizon))

    if halflife_days is not None and np.isfinite(halflife_days) and halflife_days > 0:
        # Decay-weighted: each day d from 1..h gets weight 2^(-d/halflife)
        weights = np.array([2.0 ** (-d / halflife_days) for d in range(1, h + 1)])
        z = float(weights.sum())
        if z <= 0:
            weights = np.ones(h) / float(h)
            z = 1.0
        weights = weights / z
        for _, g in work.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
            rets = g["_r"].shift(-1).values
            # Compute decay-weighted sum of future daily returns
            n = len(rets)
            fwd = np.full(n, np.nan)
            for i in range(n - h):
                window = rets[i : i + h]
                if not np.isnan(window).all():
                    fwd[i] = float(np.nansum(window * weights[:len(window)]))
            out.loc[g.index] = fwd
    else:
        # Standard compound return (equal-weighted)
        for _, g in work.sort_values(["ticker", "date"]).groupby("ticker", sort=False):
            one_plus_next = (1.0 + g["_r"]).shift(-1)
            fwd = one_plus_next.rolling(h, min_periods=h).apply(np.prod, raw=True).shift(-(h - 1)) - 1.0
            out.loc[g.index] = fwd.to_numpy(dtype=float)

    if horizon == 5 and "forward_return" in df.columns:
        fallback = _safe_numeric(df["forward_return"])
        out = out.fillna(fallback)
    return out.replace([np.inf, -np.inf], np.nan)


def build_alpha_decay_targets(
    df: pd.DataFrame,
    *,
    horizons: Iterable[int],
    base_target_cfg: TargetConfig,
    costs: ExecutionCostConfig,
    max_name_weight: float,
) -> pd.DataFrame:
    """Attach raw, residual, and net-of-cost residual targets for each horizon."""
    out = df.copy()
    for raw_h in horizons:
        h = max(1, int(raw_h))
        raw_col = f"{TARGET_RAW}_{h}d"
        resid_col = f"{TARGET_RESIDUAL}_{h}d"
        net_col = f"{TARGET_NET_RESIDUAL}_{h}d"

        raw = _forward_return_from_daily(out, h)
        out[raw_col] = raw

        tmp = out.copy()
        tmp["forward_return"] = raw
        resid = add_institutional_targets(
            tmp,
            cfg=TargetConfig(
                horizon_days=h,
                residualize=True,
                net_of_costs=False,
                residual_ridge=float(base_target_cfg.residual_ridge),
                winsor_q=float(base_target_cfg.winsor_q),
                max_abs_return=float(base_target_cfg.max_abs_return),
            ),
            costs=costs,
            max_name_weight=max_name_weight,
        )
        out[resid_col] = _safe_numeric(resid["target_return"])

        net = add_institutional_targets(
            tmp,
            cfg=TargetConfig(
                horizon_days=h,
                residualize=True,
                net_of_costs=True,
                residual_ridge=float(base_target_cfg.residual_ridge),
                winsor_q=float(base_target_cfg.winsor_q),
                max_abs_return=float(base_target_cfg.max_abs_return),
            ),
            costs=costs,
            max_name_weight=max_name_weight,
        )
        out[net_col] = _safe_numeric(net["target_return"])
    return out


def _daily_ic_values(df: pd.DataFrame, feature: str, target: str, *, regime: str | None = None) -> np.ndarray:
    cols = ["date", feature, target]
    if regime is not None:
        cols.append("regime_label")
    work = df[cols].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work[feature] = _safe_numeric(work[feature])
    work[target] = _safe_numeric(work[target])
    work = work.dropna(subset=["date", feature, target])
    if regime is not None and "regime_label" in work.columns:
        work = work[work["regime_label"].astype(str).eq(str(regime))]
    vals: list[float] = []
    for _, g in work.groupby("date", sort=True):
        if len(g) < 8 or g[feature].nunique() < 2 or g[target].nunique() < 2:
            continue
        corr = g[feature].corr(g[target], method="spearman")
        if np.isfinite(corr):
            vals.append(float(corr))
    return np.asarray(vals, dtype=float)


def _hac_tstat(values: np.ndarray, *, max_lag: int = 5, horizon_days: int = 1) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3:
        return float("nan")
    mu = float(arr.mean())
    demeaned = arr - mu
    # P26: Horizon-aware HAC lag selection.
    # The naive rule  lag = max(5, h-1)  over-penalises long-horizon features
    # because it assumes h-1 days of overlap always contribute to autocorrelation.
    # For slow-decay features (fundamentals, quality), the IC series does NOT
    # experience daily overlap autocorrelation — the feature changes quarterly.
    # The effective lag is bounded by the number of non-overlapping observations:
    #   n_nonoverlap = n_raw × holding_period / horizon_days
    # and the decay profile of the feature (fast/medium/slow).
    #
    # P26 policy:
    #   base_lag = min(max_lag, max(0, int(horizon_days) - 1))
    #   Then clamp to: min(base_lag, n_nonoverlap)
    # This prevents a 63d feature with 8 quarterly observations from getting
    # 62-lag HAC correction that destroys tstat significance.
    n_raw = len(arr)
    n_nonoverlap = max(2, int(n_raw / max(1, int(horizon_days))))
    base_lag = min(int(max_lag), max(0, int(horizon_days) - 1))
    effective_lag = min(base_lag, n_nonoverlap)
    lag_n = min(effective_lag, n_raw - 1)
    var = float(np.dot(demeaned, demeaned) / n_raw)
    for lag in range(1, lag_n + 1):
        cov = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n_raw)
        var += 2.0 * (1.0 - lag / (lag_n + 1.0)) * cov
    se = np.sqrt(max(var, 0.0) / n_raw)
    return mu / se if np.isfinite(se) and se > 1e-12 else float("nan")


def _decile_shape(df: pd.DataFrame, feature: str, target: str, *, sign: int) -> tuple[float, float]:
    spreads: list[float] = []
    mono: list[float] = []
    work = df[["date", feature, target]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work[feature] = _safe_numeric(work[feature])
    work[target] = _safe_numeric(work[target])
    work = work.dropna()
    for _, g in work.groupby("date", sort=True):
        if len(g) < 30 or g[feature].nunique() < 10:
            continue
        try:
            bucket = pd.qcut(g[feature].rank(method="first"), 10, labels=False, duplicates="drop")
        except ValueError:
            continue
        by_bucket = g.groupby(bucket, observed=True)[target].mean().sort_index()
        if len(by_bucket) < 3:
            continue
        raw_spread = float(by_bucket.iloc[-1] - by_bucket.iloc[0])
        diffs = np.diff(by_bucket.to_numpy(dtype=float))
        spreads.append(float(sign) * raw_spread)
        if len(diffs):
            mono.append(float(((float(sign) * diffs) >= 0.0).mean()))
    return (
        float(np.nanmean(spreads)) if spreads else float("nan"),
        float(np.nanmean(mono)) if mono else float("nan"),
    )


# ── P26: Horizon buckets for statistically-comparable FDR calibration ────────

_HORIZON_BUCKETS: dict[str, tuple[int, int]] = {
    "short": (1, 5),
    "medium": (6, 21),
    "long": (22, 63),
    "ultra_long": (64, 252),
}


def _horizon_bucket_label(horizon_days: int) -> str:
    for name, (lo, hi) in _HORIZON_BUCKETS.items():
        if lo <= int(horizon_days) <= hi:
            return name
    return "unknown"


def _compute_bucket_bhy_thresholds(
    decay: pd.DataFrame,
    *,
    alpha: float = 0.05,
    hac_max_lag: int = 5,
) -> dict[str, float]:
    """
    P26: Compute BHY-adjusted t-stat threshold PER HORIZON BUCKET.

    Features at 1-5d are calibrated together (short bucket).
    Features at 6-21d are calibrated together (medium bucket).
    Features at 22-63d are calibrated together (long bucket).

    This prevents short-horizon features from inflating the FDR threshold
    for long-horizon features (and vice versa).  Each bucket's features
    share statistically-comparable HAC correction and observation counts.
    """
    from model_selection.statistics import bhy_adjust_pvalues

    thresholds: dict[str, float] = {"unknown": 0.02}

    decay = decay.copy()
    decay["_bucket"] = decay["horizon_days"].apply(lambda h: _horizon_bucket_label(int(h)))

    for bucket_name in _HORIZON_BUCKETS:
        bucket_rows = decay[
            decay["_bucket"].eq(bucket_name)
            & decay["target_type"].eq(TARGET_NET_RESIDUAL)
        ]
        if bucket_rows.empty or len(bucket_rows) < 3:
            thresholds[bucket_name] = 0.02
            continue

        tstats_raw = []
        n_days_raw = []
        for _, row in bucket_rows.iterrows():
            t = float(row.get("daily_spearman_ic_tstat", np.nan))
            n = int(row.get("ic_n_days", 0))
            if np.isfinite(t) and n >= 3:
                tstats_raw.append(abs(t))
                n_days_raw.append(n)

        if not tstats_raw:
            thresholds[bucket_name] = 0.02
            continue

        avg_n = int(np.mean(n_days_raw))
        try:
            from scipy.stats import norm as _snorm
        except ImportError:
            p_values = 2.0 * (1.0 - np.minimum(1.0, np.asarray(tstats_raw) / 3.0))
        else:
            p_values = 2.0 * (
                1.0 - _snorm.cdf(np.asarray(tstats_raw) * np.sqrt(max(avg_n - 1, 1)))
            )

        adj_p = bhy_adjust_pvalues(p_values, alpha=alpha)
        passing = adj_p <= alpha
        if not np.any(passing):
            thresholds[bucket_name] = 0.02
        else:
            best_p = float(np.max(adj_p[passing]))
            try:
                from scipy.stats import norm as _snorm
            except ImportError:
                t_crit = 3.0 / np.sqrt(max(avg_n - 1, 1))
            else:
                t_crit = _snorm.ppf(1.0 - best_p / 2.0) / np.sqrt(max(avg_n - 1, 1))
            thresholds[bucket_name] = float(max(t_crit, 0.0))

    return thresholds


def _compute_bhy_tstat_threshold(
    decay: pd.DataFrame,
    *,
    alpha: float = 0.05,
    production_horizon: int = 5,
    hac_max_lag: int = 5,
) -> float:
    """
    Compute the BHY-adjusted minimum |t-stat| for feature admission.

    Extracts all raw IC t-stats at the production horizon, converts them to
    p-values using HAC-robust standard errors, and applies the Benjamini-Hochberg-Yekutieli
    procedure to find the FDR-controlled threshold.
    """
    prod_target = f"{TARGET_NET_RESIDUAL}_{int(production_horizon)}d"
    rows = decay[
        decay["target_type"].eq(TARGET_NET_RESIDUAL)
        & decay["horizon_days"].eq(int(production_horizon))
    ].copy()
    if rows.empty:
        return 0.02
    tstats_raw = []
    ic_n_days_raw = []
    for _, row in rows.iterrows():
        t = float(row.get("daily_spearman_ic_tstat", np.nan))
        n = int(row.get("ic_n_days", 0))
        if np.isfinite(t) and n >= 3:
            tstats_raw.append(t)
            ic_n_days_raw.append(n)
    if not tstats_raw:
        return 0.02
    tstats_arr = np.asarray(tstats_raw, dtype=float)
    p_values = _two_sided_p_from_tstat(tstats_arr)
    adjusted_p = bhy_adjust_pvalues(p_values, alpha=alpha)
    # Find the p-value threshold and convert back to t-stat
    threshold_p = float(alpha)
    passing = adjusted_p <= threshold_p
    if not np.any(passing):
        # If none pass, return the minimum p-value's equivalent t-stat
        best_p = float(np.min(adjusted_p))
    else:
        best_p = float(np.max(adjusted_p[passing]))
    try:
        from scipy.stats import norm as _snorm
    except ImportError:
        t_crit = 3.0
    else:
        t_crit = _snorm.ppf(1.0 - best_p / 2.0)
    return float(max(t_crit, 0.0))


def _target_columns_for_horizon(horizon: int) -> dict[str, str]:
    h = max(1, int(horizon))
    return {
        TARGET_RAW: f"{TARGET_RAW}_{h}d",
        TARGET_RESIDUAL: f"{TARGET_RESIDUAL}_{h}d",
        TARGET_NET_RESIDUAL: f"{TARGET_NET_RESIDUAL}_{h}d",
    }


def _target_horizon_from_column(target_col: str) -> int:
    """Extract the forward horizon from canonical alpha target columns.

    Target names are constructed as ``<target_type>_<N>d``.  Parse this as a
    contract rather than by splitting on underscores, because tokens such as
    ``"1d"`` are horizon labels, not base-10 integers.
    """

    match = _TARGET_HORIZON_RE.search(str(target_col))
    if match is None:
        return 1
    return max(1, int(match.group("horizon")))


def _vectorized_spearman_corr(feature_ranks: np.ndarray, target_ranks: np.ndarray, *, min_obs: int = 8) -> np.ndarray:
    y = np.asarray(target_ranks, dtype=float).reshape(-1, 1)
    x = np.asarray(feature_ranks, dtype=float)
    if x.ndim != 2 or y.shape[0] != x.shape[0]:
        return np.full(x.shape[1] if x.ndim == 2 else 0, np.nan, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    count = mask.sum(axis=0)
    x_masked = np.where(mask, x, 0.0)
    y_masked = np.where(mask, y, 0.0)
    mean_x = np.divide(
        x_masked.sum(axis=0),
        count,
        out=np.full(x.shape[1], np.nan, dtype=float),
        where=count > 0,
    )
    mean_y = np.divide(
        y_masked.sum(axis=0),
        count,
        out=np.full(x.shape[1], np.nan, dtype=float),
        where=count > 0,
    )
    x_centered = np.where(mask, x - mean_x, 0.0)
    y_centered = np.where(mask, y - mean_y, 0.0)
    cov = (x_centered * y_centered).sum(axis=0)
    var_x = (x_centered * x_centered).sum(axis=0)
    var_y = (y_centered * y_centered).sum(axis=0)
    corr = np.full(x.shape[1], np.nan, dtype=float)
    good = (count >= int(min_obs)) & (var_x > 1e-12) & (var_y > 1e-12)
    corr[good] = cov[good] / np.sqrt(var_x[good] * var_y[good])
    return corr


def _vectorized_bucket_shape(
    feature_ranks: np.ndarray,
    target_values: np.ndarray,
    *,
    bins: int = 10,
    min_obs: int = 30,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.asarray(feature_ranks, dtype=float)
    y = np.asarray(target_values, dtype=float).reshape(-1, 1)
    if x.ndim != 2 or y.shape[0] != x.shape[0]:
        m = x.shape[1] if x.ndim == 2 else 0
        nan = np.full(m, np.nan, dtype=float)
        return nan, nan, nan
    mask = np.isfinite(x) & np.isfinite(y)
    count = mask.sum(axis=0)
    valid = mask & (count[np.newaxis, :] >= int(min_obs))
    denom = np.where(count > 0, count, 1)
    scaled = np.floor(((x - 1.0) * int(bins)) / denom[np.newaxis, :]).astype(float)
    bucket_idx = np.clip(np.nan_to_num(scaled, nan=-1.0), -1, int(bins) - 1).astype(int)
    bucket_idx[~valid] = -1
    bucket_means = np.full((int(bins), x.shape[1]), np.nan, dtype=float)
    for b in range(int(bins)):
        member = valid & (bucket_idx == b)
        member_count = member.sum(axis=0)
        sums = np.where(member, y, 0.0).sum(axis=0)
        bucket_means[b] = np.divide(
            sums,
            member_count,
            out=np.full(x.shape[1], np.nan, dtype=float),
            where=member_count > 0,
        )
    spread = bucket_means[-1] - bucket_means[0]
    diffs = np.diff(bucket_means, axis=0)
    valid_diffs = np.isfinite(diffs)
    monotonicity = np.divide(
        (diffs >= 0.0).sum(axis=0),
        valid_diffs.sum(axis=0),
        out=np.full(x.shape[1], np.nan, dtype=float),
        where=valid_diffs.sum(axis=0) > 0,
    )
    diffs_inv = np.diff(bucket_means[::-1], axis=0)
    valid_diffs_inv = np.isfinite(diffs_inv)
    inverted_monotonicity = np.divide(
        (diffs_inv >= 0.0).sum(axis=0),
        valid_diffs_inv.sum(axis=0),
        out=np.full(x.shape[1], np.nan, dtype=float),
        where=valid_diffs_inv.sum(axis=0) > 0,
    )
    return spread, monotonicity, inverted_monotonicity


def _guarded_column_mean(values: np.ndarray, *, min_count: int = 1) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        return np.full(0, np.nan, dtype=float), np.zeros(0, dtype=int)
    valid = np.isfinite(arr)
    counts = valid.sum(axis=0).astype(int)
    sums = np.where(valid, arr, 0.0).sum(axis=0)
    out = np.full(arr.shape[1], np.nan, dtype=float)
    eligible = counts >= int(min_count)
    out[eligible] = sums[eligible] / counts[eligible]
    return out, counts


def _guarded_column_std(values: np.ndarray, *, ddof: int = 1, min_count: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(values, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        return np.full(0, np.nan, dtype=float), np.zeros(0, dtype=int)
    valid = np.isfinite(arr)
    counts = valid.sum(axis=0).astype(int)
    means, _ = _guarded_column_mean(arr, min_count=1)
    centered = np.where(valid, arr - means[np.newaxis, :], 0.0)
    denom = counts - int(ddof)
    need = max(int(ddof) + 1, int(min_count) if min_count is not None else int(ddof) + 1)
    eligible = counts >= need
    var = np.full(arr.shape[1], np.nan, dtype=float)
    var[eligible] = (centered[:, eligible] * centered[:, eligible]).sum(axis=0) / denom[eligible]
    return np.sqrt(np.clip(var, a_min=0.0, a_max=None)), counts


def _support_failures(
    *,
    ic_valid_days: int,
    regime_valid_days: int,
    spread_valid_days: int,
    monotonicity_valid_days: int,
    cfg: AlphaAdmissionConfig,
    require_regime: bool,
) -> list[str]:
    failures: list[str] = []
    if int(ic_valid_days) < int(cfg.min_ic_valid_days):
        failures.append("insufficient_ic_support")
    if require_regime and int(regime_valid_days) < int(cfg.min_regime_valid_days):
        failures.append("insufficient_regime_support")
    if int(spread_valid_days) < int(cfg.min_spread_valid_days):
        failures.append("insufficient_spread_support")
    if int(monotonicity_valid_days) < int(cfg.min_monotonicity_valid_days):
        failures.append("insufficient_monotonicity_support")
    return failures


def _support_status(
    *,
    ic_valid_days: int,
    regime_valid_days: int,
    spread_valid_days: int,
    monotonicity_valid_days: int,
    cfg: AlphaAdmissionConfig,
    require_regime: bool,
) -> tuple[bool, str]:
    failures = _support_failures(
        ic_valid_days=ic_valid_days,
        regime_valid_days=regime_valid_days,
        spread_valid_days=spread_valid_days,
        monotonicity_valid_days=monotonicity_valid_days,
        cfg=cfg,
        require_regime=require_regime,
    )
    return (not failures, "available" if not failures else ",".join(failures))


def _coerce_int(value: object, default: int = 0) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return int(default)
    if not np.isfinite(numeric):
        return int(default)
    return int(numeric)


def _adaptive_thresholds(
    decay: pd.DataFrame,
    cfg: AlphaAdmissionConfig,
    *,
    target_col: str = "daily_spearman_ic",
    tstat_col: str = "daily_spearman_ic_tstat",
    feature_specs: dict | None = None,
) -> dict[str, float]:
    """Compute dynamic admission baselines from the cross-section of candidate features."""
    if bool(cfg.multi_horizon_admission):
        candidates = decay[decay["target_type"].eq(TARGET_NET_RESIDUAL)].copy()
    else:
        candidates = decay[
            decay["target_type"].eq(TARGET_NET_RESIDUAL)
            & decay["horizon_days"].eq(int(cfg.production_horizon))
        ].copy()
    if candidates.empty or len(candidates) < cfg.min_admitted_for_fallback:
        return {}

    ics = _safe_numeric(candidates[target_col]).abs().dropna()
    tstats = _safe_numeric(candidates[tstat_col]).abs().dropna()
    coverages = _safe_numeric(candidates.get("coverage", pd.Series(np.nan, index=candidates.index))).dropna()
    monos = _safe_numeric(candidates.get("positive_monotonicity", pd.Series(np.nan, index=candidates.index))).dropna()
    regimes = _safe_numeric(candidates.get("regime_positive_rate", pd.Series(np.nan, index=candidates.index))).dropna()

    pct = float(cfg.min_admission_percentile)

    adaptive: dict[str, float] = {}
    if len(ics) >= 2:
        adaptive["adaptive_min_abs_ic"] = float(np.quantile(ics.values, max(0.0, pct - 0.10)))
    if len(tstats) >= 2:
        adaptive["adaptive_min_tstat"] = float(np.quantile(tstats.values, max(0.0, pct - 0.10)))
    if len(coverages) >= 2:
        adaptive["adaptive_min_coverage"] = float(np.quantile(coverages.values, max(0.0, pct - 0.15)))
    if len(monos) >= 2:
        adaptive["adaptive_min_monotonicity"] = float(np.quantile(monos.values, max(0.0, pct - 0.10)))
    if len(regimes) >= 2:
        adaptive["adaptive_min_regime_stability"] = float(np.quantile(regimes.values, max(0.0, pct - 0.10)))

    return adaptive


def _matrix_decay_stats(
    df: pd.DataFrame,
    feature_columns: list[str],
    *,
    target_col: str,
    cfg: AlphaAdmissionConfig,
) -> dict[str, np.ndarray]:
    work_cols = ["date", target_col, *feature_columns]
    if "regime_label" in df.columns:
        work_cols.append("regime_label")
    work = df.loc[:, [c for c in work_cols if c in df.columns]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work[target_col] = _safe_numeric(work[target_col])
    for feature in feature_columns:
        work[feature] = _safe_numeric(work[feature])
    work = work.dropna(subset=["date"]).sort_values("date")

    daily_ic_rows: list[np.ndarray] = []
    spread_pos_rows: list[np.ndarray] = []
    mono_pos_rows: list[np.ndarray] = []
    spread_neg_rows: list[np.ndarray] = []
    mono_neg_rows: list[np.ndarray] = []
    regime_by_day: list[str] = []

    for _, g in work.groupby("date", sort=True):
        if len(g) < 8:
            continue
        x = g[feature_columns].to_numpy(dtype=float, copy=False)
        y = g[target_col].to_numpy(dtype=float, copy=False)
        x_rank = pd.DataFrame(x, columns=feature_columns).rank(axis=0, method="average", na_option="keep").to_numpy(dtype=float)
        y_rank = pd.Series(y).rank(method="average", na_option="keep").to_numpy(dtype=float)
        daily_ic_rows.append(_vectorized_spearman_corr(x_rank, y_rank, min_obs=8))
        spread_pos, mono_pos, mono_neg = _vectorized_bucket_shape(x_rank, y, bins=10, min_obs=30)
        spread_pos_rows.append(spread_pos)
        mono_pos_rows.append(mono_pos)
        spread_neg_rows.append(-spread_pos)
        mono_neg_rows.append(mono_neg)
        regime_by_day.append(str(g["regime_label"].iloc[0]) if "regime_label" in g.columns else "")

    m = len(feature_columns)
    if not daily_ic_rows:
        nan = np.full(m, np.nan, dtype=float)
        zeros = np.zeros(m, dtype=int)
        return {
            "daily_ic_mean": nan,
            "daily_ic_std": nan,
            "daily_ic_tstat": nan,
            "ic_n_days": zeros,
            "regime_valid_days": zeros,
            "spread_valid_days": zeros,
            "monotonicity_valid_days": zeros,
            "positive_spread": nan,
            "positive_monotonicity": nan,
            "inverted_spread": nan,
            "inverted_monotonicity": nan,
            "regime_positive_rate": np.ones(m, dtype=float),
            "regime_inverted_positive_rate": np.ones(m, dtype=float),
            "evidence_available": np.zeros(m, dtype=bool),
            "evidence_status": np.asarray(["insufficient_ic_support,insufficient_spread_support,insufficient_monotonicity_support"] * m, dtype=object),
            "require_regime_support": np.zeros(m, dtype=bool),
        }

    daily_ic = np.vstack(daily_ic_rows)
    spread_pos_arr = np.vstack(spread_pos_rows) if spread_pos_rows else np.full((0, m), np.nan, dtype=float)
    mono_pos_arr = np.vstack(mono_pos_rows) if mono_pos_rows else np.full((0, m), np.nan, dtype=float)
    spread_neg_arr = np.vstack(spread_neg_rows) if spread_neg_rows else np.full((0, m), np.nan, dtype=float)
    mono_neg_arr = np.vstack(mono_neg_rows) if mono_neg_rows else np.full((0, m), np.nan, dtype=float)

    regime_positive = np.ones(m, dtype=float)
    regime_inverted = np.ones(m, dtype=float)
    regime_valid_days = np.zeros(m, dtype=int)
    if any(regime_by_day):
        regime_arr = np.asarray(regime_by_day, dtype=object)
        regimes = sorted({str(r) for r in regime_arr if str(r)})
        tested = np.zeros(m, dtype=float)
        positives = np.zeros(m, dtype=float)
        inverted = np.zeros(m, dtype=float)
        for regime in regimes:
            idx = regime_arr == regime
            subset = daily_ic[idx]
            means, counts = _guarded_column_mean(subset, min_count=1)
            tested_mask = counts >= 5
            tested += tested_mask.astype(float)
            positives += (tested_mask & (means > 0.0)).astype(float)
            inverted += (tested_mask & (means < 0.0)).astype(float)
            regime_valid_days += np.where(tested_mask, counts, 0).astype(int)
        regime_positive = np.divide(
            positives,
            tested,
            out=np.ones(m, dtype=float),
            where=tested > 0,
        )
        regime_inverted = np.divide(
            inverted,
            tested,
            out=np.ones(m, dtype=float),
            where=tested > 0,
        )
    else:
        regime_valid_days = np.isfinite(daily_ic).sum(axis=0).astype(int)

    daily_ic_mean, ic_n_days = _guarded_column_mean(daily_ic, min_count=1)
    daily_ic_std, _ = _guarded_column_std(daily_ic, ddof=1, min_count=2)
    _h_days = _target_horizon_from_column(target_col)
    daily_ic_tstat = np.array(
        [_hac_tstat(daily_ic[:, j], horizon_days=_h_days) for j in range(m)], dtype=float
    )
    positive_spread, spread_valid_days = _guarded_column_mean(spread_pos_arr, min_count=1)
    positive_monotonicity, monotonicity_valid_days = _guarded_column_mean(mono_pos_arr, min_count=1)
    inverted_spread, _ = _guarded_column_mean(spread_neg_arr, min_count=1)
    inverted_monotonicity, _ = _guarded_column_mean(mono_neg_arr, min_count=1)
    evidence_available = np.zeros(m, dtype=bool)
    evidence_status = np.empty(m, dtype=object)
    require_regime = any(regime_by_day)
    for j in range(m):
        evidence_available[j], evidence_status[j] = _support_status(
            ic_valid_days=int(ic_n_days[j]),
            regime_valid_days=int(regime_valid_days[j]),
            spread_valid_days=int(spread_valid_days[j]),
            monotonicity_valid_days=int(monotonicity_valid_days[j]),
            cfg=cfg,
            require_regime=require_regime,
        )
    return {
        "daily_ic_mean": daily_ic_mean,
        "daily_ic_std": daily_ic_std,
        "daily_ic_tstat": daily_ic_tstat,
        "ic_n_days": ic_n_days,
        "regime_valid_days": regime_valid_days,
        "spread_valid_days": spread_valid_days,
        "monotonicity_valid_days": monotonicity_valid_days,
        "positive_spread": positive_spread,
        "positive_monotonicity": positive_monotonicity,
        "inverted_spread": inverted_spread,
        "inverted_monotonicity": inverted_monotonicity,
        "regime_positive_rate": regime_positive,
        "regime_inverted_positive_rate": regime_inverted,
        "evidence_available": evidence_available,
        "evidence_status": evidence_status,
        "require_regime_support": np.full(m, require_regime, dtype=bool),
    }


def compute_ic_decay_table(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    horizons: Iterable[int],
    cfg: AlphaAdmissionConfig,
) -> pd.DataFrame:
    """Compute per-feature IC decay across raw, residual, and production targets."""
    feature_list = [str(feature) for feature in feature_columns if str(feature) in df.columns]
    if not feature_list:
        return pd.DataFrame()
    feature_frame = df.loc[:, feature_list].apply(_safe_numeric)
    coverage = feature_frame.notna().mean(axis=0)
    nonzero_rate = (feature_frame.fillna(0.0).abs() > 1e-12).mean(axis=0)
    rows: list[dict[str, object]] = []
    for h in horizons:
        for target_type, target_col in _target_columns_for_horizon(int(h)).items():
            if target_col not in df.columns:
                continue
            stats = _matrix_decay_stats(df, feature_list, target_col=target_col, cfg=cfg)
            for idx, feature in enumerate(feature_list):
                spec = FEATURE_SPECS.get(feature)
                rows.append(
                    {
                        "feature": feature,
                        "family": spec.family if spec else "unknown",
                        "expected_horizon_days": spec.horizon_days if spec else np.nan,
                        "expected_sign": int(spec.expected_sign) if spec else 0,
                        "target_type": target_type,
                        "horizon_days": int(h),
                        "coverage": float(coverage.get(feature, np.nan)),
                        "nonzero_rate": float(nonzero_rate.get(feature, np.nan)),
                        "daily_spearman_ic": float(stats["daily_ic_mean"][idx]),
                        "daily_spearman_ic_std": float(stats["daily_ic_std"][idx]),
                        "daily_spearman_ic_tstat": float(stats["daily_ic_tstat"][idx]),
                        "ic_n_days": int(stats["ic_n_days"][idx]),
                        "regime_valid_days": int(stats["regime_valid_days"][idx]),
                        "spread_valid_days": int(stats["spread_valid_days"][idx]),
                        "monotonicity_valid_days": int(stats["monotonicity_valid_days"][idx]),
                        "positive_spread": float(stats["positive_spread"][idx]),
                        "positive_monotonicity": float(stats["positive_monotonicity"][idx]),
                        "inverted_spread": float(stats["inverted_spread"][idx]),
                        "inverted_monotonicity": float(stats["inverted_monotonicity"][idx]),
                        "regime_positive_rate": float(stats["regime_positive_rate"][idx]),
                        "regime_inverted_positive_rate": float(stats["regime_inverted_positive_rate"][idx]),
                        "evidence_available": bool(stats["evidence_available"][idx]),
                        "evidence_status": str(stats["evidence_status"][idx]),
                        "require_regime_support": bool(stats["require_regime_support"][idx]),
                    }
                )
    return pd.DataFrame(rows)


def _compute_signal_halflife_from_decay(decay: pd.DataFrame) -> pd.DataFrame:
    """
    Compute signal halflife from multi-horizon IC decay pattern.

    Institutional approach: fit an exponential decay model |IC(h)| = |IC(0)| * 2^(-h/halflife)
    across all available horizons for each (feature, target_type) pair. This is equivalent
    to a linear regression of log(|IC|) vs horizon.

    When empirical data is insufficient (single horizon, noisy IC, or signal strengthening),
    falls back to a structural prior from FEATURE_SPECS.decay_profile:

        "fast"   →  3.0d  (momentum, reversal — typical halflife 1-5 days)
        "medium" → 12.0d  (volume, regime, volatility — typical halflife 5-20 days)
        "slow"   → 45.0d  (quality, accruals, margins — typical halflife 20-90 days)

    The prior is conservative (understates halflife → over-penalizes in stability gates,
    which is the safe direction). Empirical data always overrides the prior.

    Handles edge cases:
      - Single horizon: uses structural prior if decay_profile is set
      - IC near zero: clip |IC| at 1e-6 to avoid log(-inf)
      - IC increases with horizon: uses structural prior
      - Fewer than 2 valid horizons: uses structural prior
    """
    if decay.empty or "horizon_days" not in decay.columns:
        return decay

    # Prior halflife mapping from FEATURE_SPECS.decay_profile
    _DECAY_PROFILE_HALFLIFE_PRIOR: dict[str, float] = {
        "fast": 3.0,
        "medium": 12.0,
        "slow": 45.0,
    }

    halflife_map: dict[tuple[str, str], float] = {}
    _log2 = np.log(2.0)

    for (feature, target_type), group in decay.groupby(["feature", "target_type"], sort=False):
        group = group.sort_values("horizon_days")
        ics = pd.to_numeric(group["daily_spearman_ic"], errors="coerce")
        horizons = pd.to_numeric(group["horizon_days"], errors="coerce")

        valid = ics.notna() & horizons.notna()
        valid_ics = ics[valid].abs().clip(lower=1e-6)
        valid_horizons = horizons[valid]

        if len(valid_ics) < 2:
            continue

        y = np.log(valid_ics.values)
        x = valid_horizons.values

        slope, intercept = np.polyfit(x, y, 1)

        # Slope must be negative (decaying) and meaningful
        if slope >= 0:
            continue

        halflife = -_log2 / slope
        halflife_map[(feature, target_type)] = float(halflife)

    # P10: Apply structural halflife prior from FEATURE_SPECS.decay_profile for
    # features where empirical estimation failed (single horizon, strengthening IC).
    # The prior ensures P11 stability gates have data to work with even for features
    # with sparse horizon coverage.
    features_with_empirical = {key[0] for key in halflife_map}
    for feature_name, spec in FEATURE_SPECS.items():
        if feature_name in features_with_empirical:
            continue
        prior = _DECAY_PROFILE_HALFLIFE_PRIOR.get(spec.decay_profile)
        if prior is None:
            continue
        for target_type in decay.loc[decay["feature"].eq(feature_name), "target_type"].unique():
            halflife_map.setdefault((feature_name, str(target_type)), prior)

    if halflife_map:
        decay = decay.copy()
        decay["signal_halflife_days"] = decay.apply(
            lambda r: halflife_map.get((str(r.get("feature", "")), str(r.get("target_type", ""))), float("nan")),
            axis=1,
        )
        # P10: also attach decay_profile for downstream consumers
        decay["decay_profile"] = decay["feature"].map(
            lambda f: getattr(FEATURE_SPECS.get(f), "decay_profile", "medium")
        )
    else:
        decay = decay.copy()
        decay["signal_halflife_days"] = float("nan")

    return decay


# P10 Family-Aware Admission Relaxation
# ============================================================================
# Fundamental features have quarterly cadence, sparse cross-sections, and
# naturally weaker IC at high frequency.  Evaluating them with the same
# statistical-power thresholds as daily price features (coverage≥0.70,
# ic_valid_days≥20) is a category error — it rejects features that are
# structurally sound but naturally sparse.
#
# This function returns a relaxed copy of the AlphaAdmissionConfig for
# fundamental families, lowering coverage and valid-day thresholds to
# levels appropriate for quarterly data.  IC magnitude and t-stat thresholds
# remain unchanged — these measure signal quality, not data frequency.
_FAMILIES_REQUIRING_RELAXATION = frozenset({
    "fundamental_quality", "fundamental_deterioration", "fundamental_leverage",
    "dilution", "reporting_quality",
})


def _family_aware_admission_cfg(
    cfg: AlphaAdmissionConfig,
    feature: str,
) -> AlphaAdmissionConfig:
    """
    Return a (possibly relaxed) copy of cfg appropriate for the feature's family.

    Fundamental-family features receive relaxed coverage and valid-day thresholds
    reflecting the quarterly cadence and smaller non-overlapping CS observation
    count of Compustat data.  All other families use the original cfg unchanged.
    """
    spec = FEATURE_SPECS.get(feature)
    if spec is None:
        return cfg
    if spec.family not in _FAMILIES_REQUIRING_RELAXATION:
        return cfg

    return replace(
        cfg,
        min_coverage=min(float(cfg.min_coverage), 0.25),
        min_ic_valid_days=min(int(cfg.min_ic_valid_days), 8),
        min_regime_valid_days=min(int(cfg.min_regime_valid_days), 8),
        min_spread_valid_days=min(int(cfg.min_spread_valid_days), 8),
        min_monotonicity_valid_days=min(int(cfg.min_monotonicity_valid_days), 8),
        min_regime_stability=min(float(cfg.min_regime_stability), 0.25),
    )


def _passes_orientation(
    row: pd.Series,
    cfg: AlphaAdmissionConfig,
    *,
    sign: int,
    bhy_min_tstat: float | None = None,
    adaptive: dict[str, float] | None = None,
) -> bool:
    ic = float(row.get("daily_spearman_ic", np.nan))
    tstat = float(row.get("daily_spearman_ic_tstat", np.nan))
    coverage = float(row.get("coverage", np.nan))
    use_adaptive = adaptive is not None and len(adaptive) > 0
    min_tstat = bhy_min_tstat if bhy_min_tstat is not None else float(cfg.min_ic_tstat)
    if use_adaptive and "adaptive_min_tstat" in adaptive:
        min_tstat = max(min_tstat, adaptive["adaptive_min_tstat"])
    if sign > 0:
        mono = float(row.get("positive_monotonicity", np.nan))
        regime = float(row.get("regime_positive_rate", np.nan))
    else:
        mono = float(row.get("inverted_monotonicity", np.nan))
        regime = float(row.get("regime_inverted_positive_rate", np.nan))
    min_coverage = float(cfg.min_coverage)
    min_abs_ic = float(cfg.min_abs_ic)
    min_monotonicity = float(cfg.min_monotonicity)
    min_regime = float(cfg.min_regime_stability)
    if use_adaptive:
        if "adaptive_min_coverage" in adaptive:
            min_coverage = min(min_coverage, adaptive["adaptive_min_coverage"])
        if "adaptive_min_abs_ic" in adaptive:
            min_abs_ic = min(min_abs_ic, adaptive["adaptive_min_abs_ic"])
        if "adaptive_min_monotonicity" in adaptive:
            min_monotonicity = min(min_monotonicity, adaptive["adaptive_min_monotonicity"])
        if "adaptive_min_regime_stability" in adaptive:
            min_regime = min(min_regime, adaptive["adaptive_min_regime_stability"])
    support_failures = _support_failures(
        ic_valid_days=_coerce_int(row.get("ic_n_days", 0)),
        regime_valid_days=_coerce_int(row.get("regime_valid_days", row.get("ic_n_days", 0))),
        spread_valid_days=_coerce_int(row.get("spread_valid_days", 0)),
        monotonicity_valid_days=_coerce_int(row.get("monotonicity_valid_days", 0)),
        cfg=cfg,
        require_regime=bool(row.get("require_regime_support", False)),
    )
    return bool(
        not support_failures
        and np.isfinite(ic)
        and np.isfinite(tstat)
        and np.isfinite(coverage)
        and coverage >= min_coverage
        and float(sign) * ic >= min_abs_ic
        and float(sign) * tstat >= min_tstat
        and (not np.isfinite(mono) or mono >= min_monotonicity)
        and (not np.isfinite(regime) or regime >= min_regime)
    )


def _best_other_horizon(decay: pd.DataFrame, feature: str, cfg: AlphaAdmissionConfig, *, bhy_min_tstat: float | None = None) -> tuple[int | None, int]:
    relaxed_cfg = _family_aware_admission_cfg(cfg, feature)
    rows = decay[
        decay["feature"].eq(feature)
        & decay["target_type"].eq(TARGET_NET_RESIDUAL)
        & ~decay["horizon_days"].eq(int(cfg.production_horizon))
    ].copy()
    if rows.empty:
        return None, 1
    best_h: int | None = None
    best_sign = 1
    best_score = -np.inf
    for _, row in rows.iterrows():
        for sign in (1, -1) if cfg.allow_inversion else (1,):
            if not _passes_orientation(row, relaxed_cfg, sign=sign, bhy_min_tstat=bhy_min_tstat):
                continue
            score = float(sign) * float(row.get("daily_spearman_ic", np.nan))
            if np.isfinite(score) and score > best_score:
                best_score = score
                best_h = int(row["horizon_days"])
                best_sign = int(sign)
    return best_h, best_sign


def _marginal_daily_ic(
    df: pd.DataFrame,
    feature: str,
    target: str,
    accepted_features: list[str],
    *,
    sign: int,
    ridge: float,
    min_residual_variance_ratio: float,
) -> float:
    if not accepted_features:
        vals = _daily_ic_values(df, feature, target)
        return float(sign) * float(np.nanmean(vals)) if len(vals) else float("nan")
    cols = ["date", feature, target, *accepted_features]
    work = df[[c for c in cols if c in df.columns]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for col in [feature, target, *accepted_features]:
        if col in work.columns:
            work[col] = _safe_numeric(work[col])
    vals: list[float] = []
    for _, g in work.dropna(subset=["date", feature, target]).groupby("date", sort=True):
        peers = [c for c in accepted_features if c in g.columns]
        if len(g) < max(10, len(peers) + 5) or g[feature].nunique() < 2 or g[target].nunique() < 2:
            continue
        x = safe_center(g[feature].to_numpy(dtype=float), cap=10.0)[:, 0]
        y = safe_center(g[target].to_numpy(dtype=float), cap=10.0)[:, 0]
        x_base_std = float(np.nanstd(x))
        if peers:
            z = g[peers].to_numpy(dtype=float)
            residualized = residualize_against_controls(
                np.column_stack([x, y]),
                z,
                ridge=float(ridge),
                value_cap=10.0,
                control_cap=10.0,
            )
            xy_resid = np.asarray(residualized.residual, dtype=float)
            if xy_resid.ndim != 2 or xy_resid.shape[1] != 2:
                continue
            x = xy_resid[:, 0]
            y = xy_resid[:, 1]
            x_resid_std = float(np.nanstd(x))
            if x_base_std > 1e-12 and (x_resid_std / x_base_std) < float(min_residual_variance_ratio):
                continue
        sx = float(np.nanstd(x))
        sy = float(np.nanstd(y))
        if sx <= 1e-12 or sy <= 1e-12:
            continue
        corr = pd.Series(x).corr(pd.Series(y), method="spearman")
        if np.isfinite(corr):
            vals.append(float(sign) * float(corr))
    return float(np.nanmean(vals)) if vals else float("nan")


def _redundancy_diagnostics(
    df: pd.DataFrame,
    feature: str,
    accepted_features: list[str],
) -> tuple[float, str]:
    """Return strongest same-date rank correlation to already admitted features."""

    if not accepted_features or "date" not in df.columns or feature not in df.columns:
        return float("nan"), ""
    peers = [c for c in accepted_features if c in df.columns]
    if not peers:
        return float("nan"), ""
    work = df[["date", feature, *peers]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    for col in [feature, *peers]:
        work[col] = _safe_numeric(work[col])
    by_peer: dict[str, list[float]] = {p: [] for p in peers}
    for _, g in work.dropna(subset=["date", feature]).groupby("date", sort=True):
        if len(g) < 10 or g[feature].nunique() < 2:
            continue
        for peer in peers:
            pair = g[[feature, peer]].dropna()
            if len(pair) < 10 or pair[peer].nunique() < 2:
                continue
            corr = pair[feature].corr(pair[peer], method="spearman")
            if np.isfinite(corr):
                by_peer[peer].append(abs(float(corr)))
    scored = [
        (peer, float(np.nanmean(vals)))
        for peer, vals in by_peer.items()
        if vals and np.isfinite(np.nanmean(vals))
    ]
    if not scored:
        return float("nan"), ""
    scored.sort(key=lambda item: item[1], reverse=True)
    return float(scored[0][1]), str(scored[0][0])


def build_feature_admission(
    decay: pd.DataFrame,
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    cfg: AlphaAdmissionConfig,
) -> pd.DataFrame:
    """Produce a production feature manifest from IC decay and admission gates."""
    feature_list = list(feature_columns)
    # P26: Per-bucket BHY thresholds — each horizon bucket has its own
    # FDR-calibrated tstat threshold instead of a single global one.
    # This prevents short-horizon features from inflating thresholds
    # for long-horizon features (and vice versa).
    bhy_min_tstat_default = float(cfg.min_ic_tstat)
    bucket_bhy_thresholds: dict[str, float] = {"unknown": bhy_min_tstat_default}
    bhy_qvalues: dict[str, float] = {}
    if cfg.apply_bhy_correction and len(feature_list) > 1:
        bucket_bhy_thresholds = _compute_bucket_bhy_thresholds(
            decay,
            alpha=cfg.bhy_alpha,
            hac_max_lag=cfg.ic_tstat_hac_max_lag,
        )
        # Also compute per-feature q-values at the production horizon for reporting
        prod_target = f"{TARGET_NET_RESIDUAL}_{int(cfg.production_horizon)}d"
        all_tstats = []
        all_n_days = []
        for feature in feature_list:
            prod = decay[
                decay["feature"].eq(feature)
                & decay["target_type"].eq(TARGET_NET_RESIDUAL)
                & decay["horizon_days"].eq(int(cfg.production_horizon))
            ]
            if not prod.empty:
                t = float(prod.iloc[0].get("daily_spearman_ic_tstat", np.nan))
                n = int(prod.iloc[0].get("ic_n_days", 0))
                if np.isfinite(t) and n >= 3:
                    all_tstats.append(abs(t))
                    all_n_days.append(n)
        if all_tstats:
            tstat_lookup = {}
            for feature in feature_list:
                prod = decay[
                    decay["feature"].eq(feature)
                    & decay["target_type"].eq(TARGET_NET_RESIDUAL)
                    & decay["horizon_days"].eq(int(cfg.production_horizon))
                ]
                if not prod.empty:
                    t = float(prod.iloc[0].get("daily_spearman_ic_tstat", np.nan))
                    n = int(prod.iloc[0].get("ic_n_days", 0))
                    if np.isfinite(t) and n >= 3:
                        tstat_lookup[feature] = (t, n)
            ordered_feats = list(tstat_lookup.keys())
            if ordered_feats:
                p_vals = _two_sided_p_from_tstat(
                    np.asarray([tstat_lookup[f][0] for f in ordered_feats], dtype=float)
                )
                adj_p2 = bhy_adjust_pvalues(p_vals, alpha=cfg.bhy_alpha)
                for i, feat in enumerate(ordered_feats):
                    bhy_qvalues[feat] = float(adj_p2[i])

    rows: list[dict[str, object]] = []
    prod_target = f"{TARGET_NET_RESIDUAL}_{int(cfg.production_horizon)}d"
    use_multi_horizon = bool(cfg.multi_horizon_admission)
    for feature in feature_columns:
        spec = FEATURE_SPECS.get(feature)
        eval_horizon = int(spec.horizon_days) if (use_multi_horizon and spec is not None) else int(cfg.production_horizon)
        # P26: Per-bucket BHY threshold for this feature's evaluation horizon
        _bucket = _horizon_bucket_label(eval_horizon)
        _feat_bhy = bucket_bhy_thresholds.get(_bucket, bhy_min_tstat_default)
        eval_target = f"{TARGET_NET_RESIDUAL}_{eval_horizon}d"
        prod = decay[
            decay["feature"].eq(feature)
            & decay["target_type"].eq(TARGET_NET_RESIDUAL)
            & decay["horizon_days"].eq(eval_horizon)
        ]
        if prod.empty:
            if not use_multi_horizon or (spec is not None and eval_horizon != int(cfg.production_horizon)):
                prod = decay[
                    decay["feature"].eq(feature)
                    & decay["target_type"].eq(TARGET_NET_RESIDUAL)
                    & decay["horizon_days"].eq(int(cfg.production_horizon))
                ]
            if prod.empty:
                rows.append(
                    {
                        "feature": feature,
                        "family": spec.family if spec else "unknown",
                        "admitted": False,
                        "recommended_action": "remove",
                        "transform_sign": 1,
                        "selected_horizon_days": np.nan,
                        "reason": "missing_production_horizon",
                        "bhy_qvalue": float("nan"),
                        "bhy_min_tstat": _feat_bhy,
                        "eval_horizon_days": int(cfg.production_horizon),
                    }
                )
                continue
        row = prod.iloc[0]
        evidence_available = bool(row.get("evidence_available", False))
        evidence_status = str(row.get("evidence_status", "available"))
        if (
            bool(cfg.enforce_horizon_alignment)
            and spec is not None
            and int(spec.horizon_days) > int(cfg.production_horizon) * float(cfg.horizon_alignment_multiplier)
        ):
            # P25: Cross-horizon feature integration — NO LONGER gated behind
            # `not use_multi_horizon`.  The decay-weighting path is a COMPLEMENT
            # to multi-horizon admission, not an alternative.  When both are
            # enabled, a feature evaluated at its native 63d horizon with
            # insufficient IC may still be admitted at the production horizon
            # via decay weight w = 2^(-h_native / h_production).
            enable_cross_horizon = bool(getattr(cfg, "cross_horizon_admission", True))
            threshold = int(cfg.production_horizon) * float(cfg.horizon_alignment_multiplier)
            if not enable_cross_horizon:
                rows.append(
                    {
                        "feature": feature,
                        "family": spec.family,
                        "admitted": False,
                        "recommended_action": "move_horizon",
                        "transform_sign": 1,
                        "selected_horizon_days": int(spec.horizon_days),
                        "reason": f"horizon_misaligned:{int(spec.horizon_days)}d>{threshold:g}d",
                        "production_ic": float(row.get("daily_spearman_ic", np.nan)),
                        "production_ic_tstat": float(row.get("daily_spearman_ic_tstat", np.nan)),
                        "production_ic_valid_days": _coerce_int(row.get("ic_n_days", 0)),
                        "production_regime_valid_days": _coerce_int(row.get("regime_valid_days", 0)),
                        "production_spread_valid_days": _coerce_int(row.get("spread_valid_days", 0)),
                        "production_monotonicity_valid_days": _coerce_int(row.get("monotonicity_valid_days", 0)),
                        "production_evidence_available": evidence_available,
                        "production_evidence_status": evidence_status,
                        "production_monotonicity": float(row.get("positive_monotonicity", np.nan)),
                        "production_regime_stability": float(row.get("regime_positive_rate", np.nan)),
                        "marginal_ic": float("nan"),
                        "eval_horizon_days": eval_horizon,
                    }
                )
                continue
            # Check if feature passes gates at its native horizon
            native_horizon = int(spec.horizon_days)
            native_target = f"{TARGET_NET_RESIDUAL}_{native_horizon}d"
            native_row = decay[
                decay["feature"].eq(feature)
                & decay["target_type"].eq(TARGET_NET_RESIDUAL)
                & decay["horizon_days"].eq(native_horizon)
            ]
            cross_pass = False
            cross_sign = 1
            cross_ic = float("nan")
            cross_tstat = float("nan")
            if not native_row.empty:
                cross_pass = _passes_orientation(
                    native_row.iloc[0],
                    _family_aware_admission_cfg(cfg, feature),
                    sign=1,
                    bhy_min_tstat=_feat_bhy,
                )
                cross_sign = 1
                cross_ic = float(native_row.iloc[0].get("daily_spearman_ic", np.nan))
                cross_tstat = float(native_row.iloc[0].get("daily_spearman_ic_tstat", np.nan))
                if not cross_pass and bool(cfg.allow_inversion):
                    cross_pass = _passes_orientation(
                        native_row.iloc[0],
                        _family_aware_admission_cfg(cfg, feature),
                        sign=-1,
                        bhy_min_tstat=_feat_bhy,
                    )
                    if cross_pass:
                        cross_sign = -1
                        cross_ic = -float(native_row.iloc[0].get("daily_spearman_ic", np.nan))
                        cross_tstat = abs(float(native_row.iloc[0].get("daily_spearman_ic_tstat", np.nan)))
            if not cross_pass:
                # Fall through to normal rejection below
                rows.append(
                    {
                        "feature": feature,
                        "family": spec.family if spec else "unknown",
                        "admitted": False,
                        "recommended_action": "move_horizon",
                        "transform_sign": 1,
                        "selected_horizon_days": int(spec.horizon_days),
                        "reason": f"horizon_misaligned:{int(spec.horizon_days)}d>{threshold:g}d",
                        "production_ic": float(row.get("daily_spearman_ic", np.nan)),
                        "production_ic_tstat": float(row.get("daily_spearman_ic_tstat", np.nan)),
                        "production_ic_valid_days": _coerce_int(row.get("ic_n_days", 0)),
                        "production_regime_valid_days": _coerce_int(row.get("regime_valid_days", 0)),
                        "production_spread_valid_days": _coerce_int(row.get("spread_valid_days", 0)),
                        "production_monotonicity_valid_days": _coerce_int(row.get("monotonicity_valid_days", 0)),
                        "production_evidence_available": evidence_available,
                        "production_evidence_status": evidence_status,
                        "production_monotonicity": float(row.get("positive_monotonicity", np.nan)),
                        "production_regime_stability": float(row.get("regime_positive_rate", np.nan)),
                        "marginal_ic": float("nan"),
                        "eval_horizon_days": eval_horizon,
                    }
                )
                continue
            # Compute decay weight: w = 2^(-h_native / h_production)
            # A 20d feature with 10d prod horizon: w = 2^(-2) = 0.25
            # A 60d feature with 10d prod horizon: w = 2^(-6) = 0.016
            decay_weight = 2.0 ** (-float(native_horizon) / float(cfg.production_horizon))
            weighted_ic = cross_ic * decay_weight
            # P40: Do NOT scale t-stat by decay_weight.  The t-stat measures
            # statistical significance at the native horizon; scaling it by the
            # decay weight is statistically invalid.  Report native_tstat for
            # significance, production_ic for contribution weighting, and
            # production_ic_tstat as NaN (not directly estimated).
            # P25: Log cross-horizon admission with decay weight
            if enable_cross_horizon:
                _side = "INVERTED" if cross_sign < 0 else "direct"
                print(
                    f"  [CrossHorizon] {feature:<30} family={spec.family:<25} "
                    f"native={native_horizon}d prod={cfg.production_horizon}d "
                    f"w={decay_weight:.3f} IC={cross_ic:.4f}→{weighted_ic:.4f} "
                    f"native_tstat={cross_tstat:.2f} (production_tstat=NA)"
                    f" admitted:{_side}"
                )
            rows.append(
                {
                    "feature": feature,
                    "family": spec.family if spec else "unknown",
                    "admitted": True,
                    "recommended_action": "admit_cross_horizon" if cross_sign > 0 else "admit_cross_horizon_inverted",
                    "transform_sign": cross_sign,
                    "selected_horizon_days": int(spec.horizon_days),
                    "eval_horizon_days": int(spec.horizon_days),
                    "reason": f"cross_horizon_admitted:w={decay_weight:.3f}_ic={cross_ic:.4f}",
                    # Native-horizon evidence (statistical significance belongs here)
                    "native_ic": cross_ic,
                    "native_tstat": cross_tstat,
                    # Production-horizon contribution (decay-weighted, NOT significance)
                    "production_ic": weighted_ic,
                    "production_ic_tstat": float("nan"),  # Not directly estimated at production horizon
                    "decay_weight": decay_weight,
                    "decay_weighted_ic_contribution": weighted_ic,
                    "production_ic_valid_days": _coerce_int(native_row.iloc[0].get("ic_n_days", 0)) if not native_row.empty else 0,
                    "production_regime_valid_days": _coerce_int(native_row.iloc[0].get("regime_valid_days", 0)) if not native_row.empty else 0,
                    "production_spread_valid_days": _coerce_int(native_row.iloc[0].get("spread_valid_days", 0)) if not native_row.empty else 0,
                    "production_monotonicity_valid_days": _coerce_int(native_row.iloc[0].get("monotonicity_valid_days", 0)) if not native_row.empty else 0,
                    "production_evidence_available": False if native_row.empty else bool(native_row.iloc[0].get("evidence_available", False)),
                    "production_evidence_status": "available" if native_row.empty else str(native_row.iloc[0].get("evidence_status", "available")),
                    "production_monotonicity": float("nan"),
                    "production_regime_stability": float("nan"),
                    "marginal_ic": float("nan"),
                    "cross_horizon_weight": decay_weight,
                    "cross_horizon_native_h": int(spec.horizon_days),
                }
            )
            continue
        original_pass = _passes_orientation(row, _family_aware_admission_cfg(cfg, feature), sign=1, bhy_min_tstat=_feat_bhy)
        inverted_pass = bool(cfg.allow_inversion) and _passes_orientation(row, _family_aware_admission_cfg(cfg, feature), sign=-1, bhy_min_tstat=_feat_bhy)
        if original_pass or inverted_pass:
            sign = 1 if original_pass else -1
            action = "admit" if sign > 0 else "invert"
            reason = "passes_current_horizon"
        else:
            other_h, other_sign = _best_other_horizon(decay, feature, cfg, bhy_min_tstat=_feat_bhy)
            sign = 1
            if other_h is not None:
                action = "move_horizon" if other_sign > 0 else "invert_and_move_horizon"
                reason = f"best_evidence_at_{other_h}d"
            else:
                # P10: Production-horizon fallback for fundamental families.
                # Fundamentals evaluated at their native 63d may fail due to
                # sparse CS obs (not weak IC).  When no alternative horizon
                # passes, try the production horizon with relaxed thresholds.
                production_h = int(cfg.production_horizon)
                relaxed_cfg = _family_aware_admission_cfg(cfg, feature)
                needs_fallback = (
                    spec is not None
                    and spec.family in _FAMILIES_REQUIRING_RELAXATION
                )
                if needs_fallback:
                    prod_row = decay[
                        decay["feature"].eq(feature)
                        & decay["target_type"].eq(TARGET_NET_RESIDUAL)
                        & decay["horizon_days"].eq(production_h)
                    ]
                else:
                    prod_row = decay.iloc[0:0]  # empty

                prod_pass = (
                    needs_fallback
                    and not prod_row.empty
                    and _passes_orientation(prod_row.iloc[0], relaxed_cfg, sign=1, bhy_min_tstat=_feat_bhy)
                )
                prod_inv = (
                    needs_fallback
                    and not prod_pass
                    and bool(cfg.allow_inversion)
                    and not prod_row.empty
                    and _passes_orientation(prod_row.iloc[0], relaxed_cfg, sign=-1, bhy_min_tstat=_feat_bhy)
                )
                if prod_pass or prod_inv:
                    sign = 1 if prod_pass else -1
                    action = "admit_production_fallback" if sign > 0 else "invert_production_fallback"
                    reason = f"admitted_at_production_horizon_{production_h}d"
                else:
                    regime_pos = float(row.get("regime_positive_rate", np.nan))
                    regime_inv = float(row.get("regime_inverted_positive_rate", np.nan))
                    if not evidence_available:
                        action = "remove"
                        reason = f"insufficient_production_evidence:{evidence_status}"
                    else:
                        action = "condition_by_regime" if max(regime_pos, regime_inv) >= cfg.min_regime_stability else "remove"
                        reason = "fails_current_horizon_gates"
            rows.append(
                {
                    "feature": feature,
                    "family": spec.family if spec else "unknown",
                    "admitted": False,
                    "recommended_action": action,
                    "transform_sign": sign,
                    "selected_horizon_days": other_h if other_h is not None else int(cfg.production_horizon),
                    "reason": reason,
                    "production_ic": float(row.get("daily_spearman_ic", np.nan)),
                    "production_ic_tstat": float(row.get("daily_spearman_ic_tstat", np.nan)),
                    "production_ic_valid_days": _coerce_int(row.get("ic_n_days", 0)),
                    "production_regime_valid_days": _coerce_int(row.get("regime_valid_days", 0)),
                    "production_spread_valid_days": _coerce_int(row.get("spread_valid_days", 0)),
                    "production_monotonicity_valid_days": _coerce_int(row.get("monotonicity_valid_days", 0)),
                    "production_evidence_available": evidence_available,
                    "production_evidence_status": evidence_status,
                    "production_monotonicity": float(row.get("positive_monotonicity", np.nan)),
                    "production_regime_stability": float(row.get("regime_positive_rate", np.nan)),
                    "marginal_ic": float("nan"),
                    "eval_horizon_days": eval_horizon,
                }
            )
            continue

        mono_col = "positive_monotonicity" if sign > 0 else "inverted_monotonicity"
        regime_col = "regime_positive_rate" if sign > 0 else "regime_inverted_positive_rate"
        sel_horizon = eval_horizon if use_multi_horizon else int(cfg.production_horizon)
        rows.append(
            {
                "feature": feature,
                "family": spec.family if spec else "unknown",
                "admitted": True,
                "recommended_action": action,
                "transform_sign": sign,
                "selected_horizon_days": sel_horizon,
                "eval_horizon_days": eval_horizon,
                "reason": reason,
                "production_ic": float(row.get("daily_spearman_ic", np.nan)),
                "production_ic_tstat": float(row.get("daily_spearman_ic_tstat", np.nan)),
                "production_ic_valid_days": _coerce_int(row.get("ic_n_days", 0)),
                "production_regime_valid_days": _coerce_int(row.get("regime_valid_days", 0)),
                "production_spread_valid_days": _coerce_int(row.get("spread_valid_days", 0)),
                "production_monotonicity_valid_days": _coerce_int(row.get("monotonicity_valid_days", 0)),
                "production_evidence_available": evidence_available,
                "production_evidence_status": evidence_status,
                "production_monotonicity": float(row.get(mono_col, np.nan)),
                "production_regime_stability": float(row.get(regime_col, np.nan)),
                "marginal_ic": float("nan"),
            }
        )

    admission = pd.DataFrame(rows)
    if admission.empty:
        return admission

    # ── P27: Admissible stability horizon sets ───────────────────────────────────
    # Each decay profile and family has a minimum horizon for stability evaluation.
    # This prevents short-horizon noise (1d, 2d) from penalising slow-decay features
    # whose signal only manifests at their native horizon (21d+).

    _ADMISSIBLE_STABILITY_MIN_HORIZON: dict[str, int] = {
        "fast": 1, "medium": 5, "slow": 21,
    }
    _FAMILY_STABILITY_MIN_HORIZON: dict[str, int] = {
        "fundamental_quality": 21, "fundamental_deterioration": 21,
        "fundamental_leverage": 21, "dilution": 21, "reporting_quality": 21,
        "quality": 21, "quality_lowvol": 10,
        "short_momentum": 5, "reversal": 3, "momentum": 3, "sector_relative": 5,
    }

    # P11: Stability screening gates — institutional-grade feature quality control.
    # Features must pass IC consistency checks across horizons before admission.
    # This prevents features with erratic IC patterns from entering the model.
    if cfg.ic_cv_max is not None or cfg.ic_sign_flip_max is not None or cfg.min_halflife_days is not None:
        for idx in admission.index:
            if not admission.loc[idx, "admitted"]:
                continue
            feat = str(admission.loc[idx, "feature"])
            feat_decay = decay[
                decay["feature"].eq(feat)
                & decay["target_type"].eq(TARGET_NET_RESIDUAL)
            ]
            if feat_decay.empty or len(feat_decay) < 2:
                continue

            # P27: Filter to admissible stability horizons only
            spec = FEATURE_SPECS.get(feat)
            profile = spec.decay_profile if spec else "medium"
            family = spec.family if spec else "unknown"
            native_h = int(spec.horizon_days) if spec else int(cfg.production_horizon)

            min_h = _FAMILY_STABILITY_MIN_HORIZON.get(family)
            if min_h is None:
                min_h = _ADMISSIBLE_STABILITY_MIN_HORIZON.get(profile, 1)
            min_h = max(min_h, native_h // 4)

            full_horizons = sorted(int(h) for h in feat_decay["horizon_days"].unique())
            included_h = [h for h in full_horizons if h >= min_h]
            excluded_h = [h for h in full_horizons if h < min_h]

            stability_decay = feat_decay[
                feat_decay["horizon_days"].isin(included_h)
            ]
            if stability_decay.empty or len(stability_decay) < 2:
                continue

            ics = pd.to_numeric(stability_decay["daily_spearman_ic"], errors="coerce").dropna()
            tstats = pd.to_numeric(stability_decay["daily_spearman_ic_tstat"], errors="coerce").dropna()

            # Gate 1: IC Coefficient of Variation across horizons
            # High CV means the feature's predictive power is horizon-dependent → unstable
            if cfg.ic_cv_max is not None and len(ics) >= 2:
                ic_mean = ics.mean()
                ic_std = ics.std()
                ic_cv = ic_std / abs(ic_mean) if abs(ic_mean) > 1e-8 else float("inf")
                admission.loc[idx, "ic_cv"] = ic_cv
                if ic_cv > float(cfg.ic_cv_max):
                    admission.loc[idx, "admitted"] = False
                    admission.loc[idx, "recommended_action"] = "remove"
                    admission.loc[idx, "reason"] = f"ic_cv_too_high:{ic_cv:.2f}>max:{cfg.ic_cv_max}"
                    continue

            # Gate 2: IC sign flips across ADMISSIBLE horizons only
            if cfg.ic_sign_flip_max is not None and len(ics) >= 2:
                sorted_decay = stability_decay.sort_values("horizon_days")
                sorted_ics = pd.to_numeric(sorted_decay["daily_spearman_ic"], errors="coerce").dropna()
                ic_signs = np.sign(sorted_ics.values)
                sign_flips = int(np.sum(np.abs(np.diff(ic_signs)) > 0))
                admission.loc[idx, "ic_sign_flips"] = sign_flips
                if sign_flips > int(cfg.ic_sign_flip_max):
                    admission.loc[idx, "admitted"] = False
                    admission.loc[idx, "recommended_action"] = "remove"
                    admission.loc[idx, "reason"] = (
                        f"ic_sign_flips:{sign_flips}>max:{cfg.ic_sign_flip_max}"
                    )
                    continue

            # P27: Log stability evaluation details
            if bool(getattr(cfg, "log_stability_gates", False)):
                _cv_str = (
                    "{:.2f}".format(admission.loc[idx, "ic_cv"])
                    if "ic_cv" in admission.columns and pd.notna(admission.loc[idx, "ic_cv"])
                    else "n/a"
                )
                _sf_str = (
                    str(admission.loc[idx, "ic_sign_flips"])
                    if "ic_sign_flips" in admission.columns
                    else "n/a"
                )
                print(
                    f"  [P27Stability] {feat:<30} fam={family:<25} native={native_h}d "
                    f"min_h={min_h}d included={included_h} excluded={excluded_h} "
                    f"cv={_cv_str} flips={_sf_str}"
                )

            # Gate 3: Minimum halflife — reject features that decay too fast
            # Fast-decaying features (halflife < 1 day) are essentially noise
            if cfg.min_halflife_days is not None:
                halflifes = pd.to_numeric(feat_decay.get("signal_halflife_days", pd.Series(dtype=float)), errors="coerce").dropna()
                if len(halflifes) > 0:
                    min_halflife = halflifes.min()
                    admission.loc[idx, "min_halflife_days"] = min_halflife
                    if min_halflife < float(cfg.min_halflife_days):
                        admission.loc[idx, "admitted"] = False
                        admission.loc[idx, "recommended_action"] = "remove"
                        admission.loc[idx, "reason"] = f"halflife_too_short:{min_halflife:.1f}d<min:{cfg.min_halflife_days}d"
                        continue

    n_admitted = int(admission["admitted"].eq(True).sum())
    def _fallback_eligible(row: pd.Series) -> bool:
        action = str(row.get("recommended_action", ""))
        reason = str(row.get("reason", ""))
        if "move_horizon" in action:
            return False
        if reason.startswith(("horizon_misaligned", "missing_production_horizon", "insufficient_production_evidence")):
            return False
        return True

    allow_fallback = cfg.fallback_mode != "disabled" and not bool(cfg.fail_if_below_minimum)
    if n_admitted == 0 and allow_fallback:
        adaptive = _adaptive_thresholds(decay, cfg)
        if adaptive:
            for i, row in admission.iterrows():
                if row.get("admitted", False):
                    continue
                if not _fallback_eligible(row):
                    continue
                feat = str(row.get("feature", ""))
                prod = decay[
                    decay["feature"].eq(feat)
                    & decay["target_type"].eq(TARGET_NET_RESIDUAL)
                    & decay["horizon_days"].eq(int(cfg.production_horizon))
                ]
                if prod.empty:
                    continue
                p_row = prod.iloc[0]
                orig = _passes_orientation(p_row, cfg, sign=1, bhy_min_tstat=_feat_bhy)
                inv = bool(cfg.allow_inversion) and _passes_orientation(p_row, cfg, sign=-1, bhy_min_tstat=_feat_bhy)
                if orig or inv:
                    continue
                orig_a = _passes_orientation(p_row, cfg, sign=1, bhy_min_tstat=_feat_bhy, adaptive=adaptive)
                inv_a = bool(cfg.allow_inversion) and _passes_orientation(p_row, cfg, sign=-1, bhy_min_tstat=_feat_bhy, adaptive=adaptive)
                if orig_a or inv_a:
                    sign = 1 if orig_a else -1
                    action = "admit_adaptive" if sign > 0 else "invert_adaptive"
                    admission.loc[i, "admitted"] = True
                    admission.loc[i, "recommended_action"] = action
                    admission.loc[i, "transform_sign"] = sign
                    admission.loc[i, "selected_horizon_days"] = int(cfg.production_horizon)
                    admission.loc[i, "reason"] = "passes_adaptive_thresholds"

        n_admitted = int(admission["admitted"].eq(True).sum())
        if n_admitted == 0:
            fallback_candidates = admission.loc[
                admission.apply(_fallback_eligible, axis=1)
            ].copy()
            fallback_candidates["_abs_ic"] = pd.to_numeric(fallback_candidates.get("production_ic", pd.Series(np.nan, index=fallback_candidates.index)), errors="coerce").abs()
            fallback_candidates["_abs_tstat"] = pd.to_numeric(fallback_candidates.get("production_ic_tstat", pd.Series(np.nan, index=fallback_candidates.index)), errors="coerce").abs()
            fallback_candidates["_coverage_score"] = pd.to_numeric(fallback_candidates.get("production_regime_stability", pd.Series(0.5, index=fallback_candidates.index)), errors="coerce").fillna(0.5)
            fallback_candidates["_quality"] = fallback_candidates["_abs_ic"] * fallback_candidates["_abs_tstat"].clip(lower=0.1) * fallback_candidates["_coverage_score"]
            fallback_candidates = fallback_candidates.sort_values("_quality", ascending=False).head(int(cfg.max_features_by_rank))
            for i, fc in fallback_candidates.iterrows():
                ic = float(fc.get("_abs_ic", 0))
                tstat = float(fc.get("_abs_tstat", 0))
                if np.isfinite(ic) and np.isfinite(tstat) and ic > 0 and tstat > 0:
                    orig_ic = float(fc.get("production_ic", 0))
                    sign = 1 if orig_ic >= 0 else -1
                    admission.loc[i, "admitted"] = True
                    admission.loc[i, "recommended_action"] = "admit_fallback_rank" if sign > 0 else "invert_fallback_rank"
                    admission.loc[i, "transform_sign"] = sign
                    admission.loc[i, "selected_horizon_days"] = int(cfg.production_horizon)
                    admission.loc[i, "reason"] = "top_n_fallback_admission"
            for col in ["_abs_ic", "_abs_tstat", "_coverage_score", "_quality"]:
                if col in admission.columns:
                    admission = admission.drop(columns=[col])

    if "bhy_qvalue" not in admission.columns:
        admission["bhy_qvalue"] = [bhy_qvalues.get(f, float("nan")) for f in admission["feature"]]
    if "bhy_min_tstat" not in admission.columns:
        admission["bhy_min_tstat"] = bhy_min_tstat_default
    if "redundancy_max_abs_corr" not in admission.columns:
        admission["redundancy_max_abs_corr"] = np.nan
    if "redundant_with" not in admission.columns:
        admission["redundant_with"] = ""

    candidates = admission[admission["admitted"].eq(True)].copy()
    candidates["_quality"] = (
        pd.to_numeric(candidates["production_ic"], errors="coerce").abs().fillna(0.0)
        * pd.to_numeric(candidates["production_ic_tstat"], errors="coerce").abs().fillna(0.0).clip(lower=0.1)
        * pd.to_numeric(candidates["production_regime_stability"], errors="coerce").fillna(1.0).clip(lower=0.1)
    )
    accepted: list[str] = []
    accepted_rows: set[int] = set()
    for idx, cand in candidates.sort_values("_quality", ascending=False).iterrows():
        feature = str(cand["feature"])
        sign = int(cand.get("transform_sign", 1) or 1)
        marginal_ic = _marginal_daily_ic(
            df,
            feature,
            prod_target,
            accepted,
            sign=sign,
            ridge=float(cfg.residual_ridge),
            min_residual_variance_ratio=float(cfg.min_marginal_residual_variance_ratio),
        )
        admission.loc[idx, "marginal_ic"] = marginal_ic
        redundancy_corr, redundant_with = _redundancy_diagnostics(df, feature, accepted)
        admission.loc[idx, "redundancy_max_abs_corr"] = redundancy_corr
        admission.loc[idx, "redundant_with"] = redundant_with
        if np.isfinite(marginal_ic) and marginal_ic >= float(cfg.min_marginal_abs_ic):
            accepted.append(feature)
            accepted_rows.add(int(idx))
        else:
            admission.loc[idx, "admitted"] = False
            admission.loc[idx, "recommended_action"] = "remove"
            admission.loc[idx, "reason"] = "fails_marginal_contribution"

    return admission.drop(columns=[c for c in ["_quality"] if c in admission.columns])


def run_alpha_research(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    cfg: AlphaAdmissionConfig,
    target_cfg: TargetConfig,
    costs: ExecutionCostConfig,
    max_name_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return enriched dataframe, full IC-decay table, and feature admission table."""
    horizons = tuple(sorted({max(1, int(h)) for h in cfg.horizons}))
    enriched = build_alpha_decay_targets(
        df,
        horizons=horizons,
        base_target_cfg=TargetConfig(
            horizon_days=int(target_cfg.horizon_days),
            residualize=bool(target_cfg.residualize),
            net_of_costs=bool(target_cfg.net_of_costs),
            residual_ridge=float(cfg.residual_ridge),
            winsor_q=float(cfg.winsor_q),
            max_abs_return=float(cfg.max_abs_return),
        ),
        costs=costs,
        max_name_weight=max_name_weight,
    )
    decay = compute_ic_decay_table(enriched, feature_columns, horizons=horizons, cfg=cfg)
    decay = _compute_signal_halflife_from_decay(decay)
    admission = build_feature_admission(decay, enriched, feature_columns, cfg=cfg)
    return enriched, decay, admission


def apply_admitted_feature_transforms(df: pd.DataFrame, admission: pd.DataFrame) -> pd.DataFrame:
    """Apply feature inversions selected by admission before train-only preprocessing."""
    out = df.copy()
    if admission is None or admission.empty:
        return out
    for _, row in admission[admission["admitted"].eq(True)].iterrows():
        feature = str(row.get("feature", ""))
        sign = int(row.get("transform_sign", 1) or 1)
        if feature in out.columns and sign < 0:
            out[feature] = -_safe_numeric(out[feature])
    return out


def summarize_admission(admission: pd.DataFrame) -> dict[str, float]:
    if admission is None or admission.empty:
        return {
            "alpha_features_admitted": 0.0,
            "alpha_features_inverted": 0.0,
            "alpha_features_removed": 0.0,
            "alpha_features_move_horizon": 0.0,
        }
    admitted = admission["admitted"].eq(True)
    actions = admission["recommended_action"].astype(str)
    return {
        "alpha_features_admitted": float(admitted.sum()),
        "alpha_features_inverted": float((admitted & actions.str.contains("invert", regex=False)).sum()),
        "alpha_features_removed": float((~admitted & actions.eq("remove")).sum()),
        "alpha_features_move_horizon": float((~admitted & actions.str.contains("move_horizon", regex=False)).sum()),
    }
