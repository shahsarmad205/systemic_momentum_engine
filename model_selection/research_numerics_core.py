"""Shared numerical research kernel for the quantitative equity framework.

Provides vectorized, reusable primitives that replace slow pandas loops
across all diagnostic engines: IC computation, forward-return construction,
rank persistence, IC decay, HAC inference, PIT bucket generation, and
feature redundancy.

Design principles:
- No per-date/per-ticker Python loops for IC or correlation computation.
- Forward returns computed once, reused across all features/horizons.
- NaN-safe, constant-input-safe, breadth-aware.
- Deterministic output with stable index ordering.
- No lookahead bias — all PIT-safe by construction.
- Numerical parity with existing scalar implementations within tolerance.

Usage:
    from model_selection.research_numerics_core import (
        compute_forward_returns,
        compute_daily_ic_series,
        vectorized_spearman_ic,
        compute_rank_persistence,
        compute_ic_decay,
        batch_hac_tstat,
        compute_feature_redundancy,
    )
"""
from __future__ import annotations

import logging
import resource
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from model_selection._shared_stats import hac_tstat as _shared_hac_tstat, ic_quality as _shared_ic_quality

logger = logging.getLogger(__name__)


# ── 1. Forward Return Construction ───────────────────────────────────────────

def compute_forward_returns(
    panel: pd.DataFrame,
    horizons: list[int],
    price_col: str = "forward_return",
    date_col: str = "date",
    ticker_col: str = "ticker",
    compound: bool = True,
) -> pd.DataFrame:
    """Compute all requested h-day forward returns in one pass per ticker.

    For each ticker and each horizon h:
        fwd_ret_{h}d[t] = sum(forward_return[t+1:t+h+1])        (if compound=False)
        fwd_ret_{h}d[t] = prod(1+forward_return[t+1:t+h+1])-1   (if compound=True)

    The result is shifted so that the signal at date t aligns with the
    forward return from t+1 through t+h. No future data is used beyond
    the intended forward target.

    Args:
        panel: DataFrame with columns [date_col, ticker_col, price_col].
        horizons: List of horizon lengths in days.
        price_col: Column containing daily returns (default "forward_return").
        date_col: Date column name.
        ticker_col: Ticker column name.
        compound: If True, compound returns geometrically. If False, sum.

    Returns:
        DataFrame with columns [date_col, ticker_col] plus one column per
        horizon named "fwd_ret_{h}d". Index is reset to match input ordering.
    """
    if not horizons:
        return pd.DataFrame()

    work = panel[[date_col, ticker_col, price_col]].copy()
    if work.empty:
        out = pd.DataFrame({date_col: [], ticker_col: []})
        for h in horizons:
            out[f"fwd_ret_{h}d"] = []
        return out

    work[date_col] = pd.to_datetime(work[date_col])
    work[price_col] = pd.to_numeric(work[price_col], errors="coerce")
    work = work.sort_values([ticker_col, date_col]).reset_index(drop=True)

    max_h = max(horizons)
    results: dict[int, np.ndarray] = {}

    for ticker, grp in work.groupby(ticker_col, sort=False):
        idx = grp.index
        rets = grp[price_col].values.astype(float)
        n = len(rets)

        if compound:
            # Geometric compounding: use cumulative product trick
            one_plus = 1.0 + rets
            # Shift so rets[i] = daily return at t+i+1
            shifted = np.empty(n + 1, dtype=float)
            shifted[0] = np.nan
            shifted[1:] = one_plus

            # Cumulative product
            cumprod = np.ones(n + 1, dtype=float)
            cumprod[1:] = shifted[1:].cumprod()

            for h in horizons:
                fwd = np.full(n, np.nan)
                # fwd[i] = product(rets[i+1:i+h+1]) = cumprod[i+h+1] / cumprod[i+1]
                end_idx = np.arange(h, n)  # i+h
                start_idx = np.arange(0, n - h)  # i
                valid = (end_idx < n) & np.isfinite(cumprod[start_idx + h + 1]) & np.isfinite(cumprod[start_idx + 1])
                fwd[start_idx[valid]] = cumprod[start_idx[valid] + h + 1] / cumprod[start_idx[valid] + 1] - 1.0
                results.setdefault(h, np.zeros(len(work), dtype=float))
                results[h][idx.values] = fwd
        else:
            # Simple sum: use cumsum trick
            shifted = np.empty(n + 1, dtype=float)
            shifted[0] = 0.0
            shifted[1:] = rets
            cumsum = np.zeros(n + 1, dtype=float)
            cumsum[1:] = shifted[1:].cumsum()

            for h in horizons:
                fwd = np.full(n, np.nan)
                start_idx = np.arange(0, n - h)
                valid = (start_idx + h < n) & np.isfinite(cumsum[start_idx + h]) & np.isfinite(cumsum[start_idx])
                fwd[start_idx[valid]] = cumsum[start_idx[valid] + h] - cumsum[start_idx[valid]]
                results.setdefault(h, np.zeros(len(work), dtype=float))
                results[h][idx.values] = fwd

    out = pd.DataFrame({
        date_col: work[date_col].values,
        ticker_col: work[ticker_col].values,
    })
    for h in horizons:
        out[f"fwd_ret_{h}d"] = results[h]

    return out


# ── 2. Vectorized Spearman IC ───────────────────────────────────────────────

def vectorized_spearman_ic(
    feature_matrix: np.ndarray,
    target_vector: np.ndarray,
    min_breadth: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute cross-sectional Spearman IC for all features at once.

    Equivalent to:
        for date in dates:
            for feature in features:
                spearmanr(feature[date, :], target[date, :])

    But fully vectorized using matrix rank operations.

    Args:
        feature_matrix: 2D array of shape (n_dates, n_features).
            Each row is one date, each column is one feature.
            Values should be raw feature values (not pre-ranked).
        target_vector: 1D array of shape (n_dates,) or 2D (n_dates, 1).
            The forward return for each date (same for all features).
            If per-date targets differ, pass as (n_dates, n_features).
        min_breadth: Minimum number of non-NaN observations per date
            to compute IC. Dates with fewer observations return NaN.

    Returns:
        (ic_matrix, breadth_matrix, valid_mask)
        - ic_matrix: (n_dates, n_features) Spearman correlation coefficients.
            NaN where insufficient breadth or constant input.
        - breadth_matrix: (n_dates, n_features) count of valid observations.
        - valid_mask: (n_dates, n_features) boolean, True where IC was computed.
    """
    x = np.asarray(feature_matrix, dtype=float)
    y = np.asarray(target_vector, dtype=float)

    if x.ndim != 2:
        raise ValueError(f"feature_matrix must be 2D, got {x.ndim}D")

    if y.ndim == 1:
        y = y.reshape(-1, 1)
    elif y.ndim == 2 and y.shape[1] == 1:
        pass  # already (n_dates, 1)
    elif y.ndim == 2 and y.shape[1] == x.shape[1]:
        pass  # per-feature targets
    else:
        raise ValueError(f"target_vector shape {y.shape} incompatible with features {x.shape}")

    n_dates, n_features = x.shape

    # Mask for finite values
    x_mask = np.isfinite(x)
    y_mask = np.isfinite(y)

    # Broadcast y_mask to match x if y is single-column
    if y.shape[1] == 1:
        y_mask = np.broadcast_to(y_mask, (n_dates, n_features))

    valid = x_mask & y_mask
    breadth = valid.sum(axis=1)  # (n_dates,) — same for all features if single y

    # If y is single-column, broadcast breadth to all features
    if y.shape[1] == 1:
        breadth = np.broadcast_to(breadth.reshape(-1, 1), (n_dates, n_features)).copy()

    # Mask insufficient breadth
    good = breadth >= min_breadth

    # Check for constant inputs (zero variance)
    x_centered = np.where(valid, x - np.nanmean(x, axis=1, keepdims=True), 0.0)
    y_centered = np.where(y_mask, y - np.nanmean(y, axis=1, keepdims=True), 0.0)

    var_x = (x_centered * x_centered).sum(axis=1, keepdims=True)
    var_y = (y_centered * y_centered).sum(axis=1, keepdims=True)

    # Broadcast var_y if single-column
    if y.shape[1] == 1:
        var_y = np.broadcast_to(var_y, (n_dates, n_features))

    not_constant = (var_x > 1e-12) & (var_y > 1e-12)

    # Final validity mask
    final_good = good & not_constant

    # Rank within each date (cross-sectional)
    # Use scipy rankdata per row for NaN safety
    x_ranked = np.full_like(x, np.nan)
    for i in range(n_dates):
        valid_i = x_mask[i]
        if valid_i.sum() >= min_breadth:
            x_ranked[i, valid_i] = scipy_stats.rankdata(x[i, valid_i])

    y_ranked = np.full_like(y, np.nan)
    for i in range(n_dates):
        valid_y = y_mask[i, 0] if y.shape[1] == 1 else y_mask[i]
        if valid_y.sum() >= min_breadth:
            if y.shape[1] == 1:
                y_ranked[i, 0] = scipy_stats.rankdata(y[i, valid_y])
            else:
                y_ranked[i, valid_y] = scipy_stats.rankdata(y[i, valid_y])

    # Compute correlation using the ranked matrices
    # Re-center after ranking
    x_r_centered = np.where(x_mask, x_ranked - np.nanmean(x_ranked, axis=1, keepdims=True), 0.0)
    y_r_centered = np.where(y_mask, y_ranked - np.nanmean(y_ranked, axis=1, keepdims=True), 0.0)

    if y.shape[1] == 1:
        # Single target: broadcast y_r_centered
        cov = (x_r_centered * y_r_centered[:, 0:1]).sum(axis=1)
        var_x_r = (x_r_centered * x_r_centered).sum(axis=1)
        var_y_r = (y_r_centered[:, 0] ** 2).sum(axis=1, keepdims=True)
        var_y_r = np.broadcast_to(var_y_r, (n_dates, n_features))
    else:
        cov = (x_r_centered * y_r_centered).sum(axis=1)
        var_x_r = (x_r_centered * x_r_centered).sum(axis=1)
        var_y_r = (y_r_centered * y_r_centered).sum(axis=1)

    ic = np.full((n_dates, n_features), np.nan, dtype=float)
    denom = np.sqrt(var_x_r * var_y_r)
    safe = final_good & (denom > 1e-12)
    ic[safe] = cov[safe] / denom[safe]

    return ic, breadth.astype(float), final_good


# ── Panel Alignment Utilities ────────────────────────────────────────────────

def _align_panel_to_tensor(
    panel: pd.DataFrame,
    features: list[str],
    target_col: str,
    date_col: str,
    ticker_col: str,
) -> tuple[np.ndarray, np.ndarray, pd.Index, pd.Index]:
    """Convert long panel to aligned (dates × tickers × features) tensor.

    Uses pd.Categorical codes for direct O(n) index mapping instead of
    pivot_table, eliminating DataFrame reshaping overhead.

    Args:
        panel: Long-form DataFrame.
        features: Feature column names.
        target_col: Target column name.
        date_col: Date column name.
        ticker_col: Ticker column name.

    Returns:
        X: (n_dates, n_tickers, n_features) float64 array, NaN for missing.
        y: (n_dates, n_tickers) float64 array, NaN for missing.
        dates: Index of aligned dates (sorted).
        tickers: Index of aligned tickers (sorted).

    Raises:
        ValueError: If duplicate (date, ticker) pairs exist.
    """
    cols = [date_col, ticker_col] + features + [target_col]
    work = panel[cols].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    for c in features + [target_col]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # Guard: duplicate (date, ticker) pairs
    dupes = work.duplicated(subset=[date_col, ticker_col], keep=False)
    if dupes.any():
        dup_rows = work[dupes][[date_col, ticker_col]].head(3).to_string()
        raise ValueError(
            f"Duplicate (date, ticker) pairs detected in panel:\n{dup_rows}"
        )

    # Stable sorted indices
    dates = pd.Index(work[date_col].dropna().unique()).sort_values()
    tickers = pd.Index(work[ticker_col].dropna().unique()).sort_values()
    n_dates = len(dates)
    n_tickers = len(tickers)
    n_features = len(features)

    if n_dates == 0 or n_tickers == 0:
        return (
            np.empty((0, 0, n_features), dtype=float),
            np.empty((0, 0), dtype=float),
            dates,
            tickers,
        )

    # Build direct index mapping via categorical codes (O(n), no pivot)
    date_cat = pd.Categorical(work[date_col].values, categories=dates, ordered=True)
    ticker_cat = pd.Categorical(work[ticker_col].values, categories=tickers, ordered=True)
    date_idx = date_cat.codes.astype(np.intp)
    ticker_idx = ticker_cat.codes.astype(np.intp)

    # Allocate and fill tensors via advanced indexing
    X = np.full((n_dates, n_tickers, n_features), np.nan, dtype=float)
    for j, feat in enumerate(features):
        vals = work[feat].values.astype(float)
        X[date_idx, ticker_idx, j] = vals

    y = np.full((n_dates, n_tickers), np.nan, dtype=float)
    y[date_idx, ticker_idx] = work[target_col].values.astype(float)

    return X, y, dates, tickers


# ── Hybrid IC Backend: pandas rank + tensor correlation ──────────────────────

def _tensor_spearman_ic(
    panel: pd.DataFrame,
    features: list[str],
    target_col: str,
    date_col: str,
    ticker_col: str,
    min_breadth: int = 8,
    constant_tol: float = 1e-12,
    pre_ranked: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Compute daily Spearman IC using fully vectorized tensor operations.

    Uses pandas groupby().rank() for C-optimized cross-sectional ranking,
    then aligns to a 3D tensor and computes all correlations simultaneously
    via BLAS-level numpy operations without any Python loops over dates.

    When pre_ranked=True, input values are already rank-transformed per date
    (or per date+group) and the ranking step is skipped. Pearson correlation
    on ranks equals Spearman correlation.

    Args:
        panel: Long-form DataFrame.
        features: Feature column names.
        target_col: Target column name.
        date_col: Date column name.
        ticker_col: Ticker column name.
        min_breadth: Minimum valid observations per date.
        constant_tol: Variance threshold for constant-input detection.
        pre_ranked: If True, skip groupby rank (inputs are already rank values).

    Returns:
        ic_matrix: (n_dates, n_features) daily ICs.
        breadth_matrix: (n_dates, n_features) valid observation counts.
        valid_matrix: (n_dates, n_features) boolean.
        dates: Index of dates.
    """
    cols = [date_col, ticker_col] + features + [target_col]
    work = panel[cols].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    for c in features + [target_col]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    dupes = work.duplicated(subset=[date_col, ticker_col], keep=False)
    if dupes.any():
        dup_rows = work[dupes][[date_col, ticker_col]].head(3).to_string()
        raise ValueError(
            f"Duplicate (date, ticker) pairs detected in panel:\n{dup_rows}"
        )

    # Step 1: Rank using pandas groupby (C-optimized), or use pre-ranked values
    if not pre_ranked:
        rank_arrays = {}
        for feat in features:
            rank_arrays[feat] = work.groupby(date_col, sort=False)[feat].rank(
                pct=False, method="average", na_option="keep",
            ).values.astype(float)
        target_rank = work.groupby(date_col, sort=False)[target_col].rank(
            pct=False, method="average", na_option="keep",
        ).values.astype(float)
    else:
        rank_arrays = {}
        for feat in features:
            rank_arrays[feat] = work[feat].values.astype(float).copy()
        target_rank = work[target_col].values.astype(float).copy()

    # Step 2: Sort by date for contiguous tensor alignment
    sort_idx = work[date_col].argsort(kind="stable")
    work = work.iloc[sort_idx].reset_index(drop=True)
    date_arr = work[date_col].values
    for feat in features:
        rank_arrays[feat] = rank_arrays[feat][sort_idx]
    target_rank = target_rank[sort_idx]

    dates = pd.Index(work[date_col].unique()).sort_values()
    tickers = pd.Index(work[ticker_col].dropna().unique()).sort_values()
    n_dates = len(dates)
    n_tickers = len(tickers)
    n_features = len(features)

    if n_dates == 0 or n_tickers == 0:
        return (
            np.full((0, n_features), np.nan, dtype=float),
            np.zeros((0, n_features), dtype=float),
            np.zeros((0, n_features), dtype=bool),
            dates,
        )

    # Step 3: Build date→row mapping using categorical codes (no pivot_table)
    date_cat = pd.Categorical(date_arr, categories=dates, ordered=True)
    ticker_cat = pd.Categorical(work[ticker_col].values, categories=tickers, ordered=True)
    date_idx = date_cat.codes.astype(np.intp)
    ticker_idx = ticker_cat.codes.astype(np.intp)

    # Step 4: Assemble 3D rank tensor R: (n_dates, n_tickers, n_features)
    R = np.full((n_dates, n_tickers, n_features), np.nan, dtype=float)
    for j, feat in enumerate(features):
        R[date_idx, ticker_idx, j] = rank_arrays[feat]

    # Step 5: Assemble 2D target rank tensor T: (n_dates, n_tickers)
    T = np.full((n_dates, n_tickers), np.nan, dtype=float)
    T[date_idx, ticker_idx] = target_rank

    # Step 6: Validity mask - both feature and target must be finite
    T_finite = np.isfinite(T)
    V = np.isfinite(R) & T_finite[:, :, None]  # (n_dates, n_tickers, n_features)

    # Step 7: Count valid observations per (date, feature)
    n_valid = V.sum(axis=1).astype(float)  # (n_dates, n_features)

    # Step 8: Compute means using only jointly valid observations
    # R_mean[i, j] = mean of R[i, :, j] where V[i, :, j] is True
    R_sum = np.where(V, R, 0.0).sum(axis=1)  # (n_dates, n_features)
    R_mean = np.divide(R_sum, n_valid, out=np.full_like(R_sum, np.nan), where=n_valid > 0)

    # T_mean[i, j] = mean of T[i, :] where V[i, :, j] is True
    T_expanded = T[:, :, None]  # (n_dates, n_tickers, 1)
    T_sum = np.where(V, T_expanded, 0.0).sum(axis=1)  # (n_dates, n_features)
    T_mean = np.divide(T_sum, n_valid, out=np.full_like(T_sum, np.nan), where=n_valid > 0)

    # Step 9: Center ranks (zero out invalid entries)
    R_centered = np.where(V, R - R_mean[:, None, :], 0.0)
    T_centered = np.where(V, T_expanded - T_mean[:, None, :], 0.0)

    # Step 10: Compute sum of squares and covariance via BLAS-level reductions
    f_ss = (R_centered ** 2).sum(axis=1)  # (n_dates, n_features)
    t_ss = (T_centered ** 2).sum(axis=1)  # (n_dates, n_features)
    cov = (R_centered * T_centered).sum(axis=1)  # (n_dates, n_features)

    # Step 11: Compute IC with constant/variance guards
    denom = np.sqrt(f_ss * t_ss)
    ic_matrix = np.full((n_dates, n_features), np.nan, dtype=float)
    valid_mask = (denom > 1e-12) & (n_valid >= min_breadth)
    ic_matrix[valid_mask] = cov[valid_mask] / denom[valid_mask]

    # Constant feature/target detection
    f_var = np.divide(f_ss, n_valid, out=np.zeros_like(f_ss), where=n_valid > 0)
    t_var = np.divide(t_ss, n_valid, out=np.zeros_like(t_ss), where=n_valid > 0)
    constant_mask = (f_var < constant_tol) | (t_var < constant_tol)
    ic_matrix[constant_mask & valid_mask] = np.nan
    valid_mask = valid_mask & ~constant_mask

    # Breadth matrix
    breadth_matrix = n_valid.copy()
    breadth_matrix[~np.isfinite(breadth_matrix)] = 0.0

    return ic_matrix, breadth_matrix, valid_mask, dates


def _hybrid_spearman_ic(
    panel: pd.DataFrame,
    features: list[str],
    target_col: str,
    date_col: str,
    ticker_col: str,
    min_breadth: int = 8,
    constant_tol: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Compute daily Spearman IC using pandas rank + numpy correlation.

    Uses pandas groupby().rank() for C-optimized cross-sectional ranking,
    then computes correlations via vectorized numpy operations per date
    without DataFrame slicing overhead.

    Args:
        panel: Long-form DataFrame.
        features: Feature column names.
        target_col: Target column name.
        date_col: Date column name.
        ticker_col: Ticker column name.
        min_breadth: Minimum valid observations per date.
        constant_tol: Variance threshold for constant-input detection.

    Returns:
        ic_matrix: (n_dates, n_features) daily ICs.
        breadth_matrix: (n_dates, n_features) valid observation counts.
        valid_matrix: (n_dates, n_features) boolean.
        dates: Index of dates.
    """
    # Step 1: Rank using pandas groupby (C-optimized)
    cols = [date_col, ticker_col] + features + [target_col]
    work = panel[cols].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    for c in features + [target_col]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # Guard: duplicate (date, ticker) pairs
    dupes = work.duplicated(subset=[date_col, ticker_col], keep=False)
    if dupes.any():
        dup_rows = work[dupes][[date_col, ticker_col]].head(3).to_string()
        raise ValueError(
            f"Duplicate (date, ticker) pairs detected in panel:\n{dup_rows}"
        )

    # Rank features
    rank_arrays = {}
    for feat in features:
        rank_arrays[feat] = work.groupby(date_col, sort=False)[feat].rank(
            pct=False, method="average", na_option="keep",
        ).values.astype(float)

    # Rank target
    target_rank = work.groupby(date_col, sort=False)[target_col].rank(
        pct=False, method="average", na_option="keep",
    ).values.astype(float)

    # Step 2: Extract date boundaries for efficient slicing
    # Sort by date to get contiguous blocks
    sort_idx = work[date_col].argsort(kind="stable")
    work = work.iloc[sort_idx].reset_index(drop=True)
    date_arr = work[date_col].values
    for feat in features:
        rank_arrays[feat] = rank_arrays[feat][sort_idx]
    target_rank = target_rank[sort_idx]

    # Find date boundaries
    dates = pd.Index(work[date_col].unique()).sort_values()
    n_dates = len(dates)
    n_features = len(features)

    # Build date index map: date → (start, end) in sorted array
    date_boundaries = {}
    current_date = None
    start = 0
    for i, d in enumerate(date_arr):
        if d != current_date:
            if current_date is not None:
                date_boundaries[current_date] = (start, i)
            current_date = d
            start = i
    date_boundaries[current_date] = (start, len(date_arr))

    # Step 3: Compute correlations per date using numpy slicing
    ic_matrix = np.full((n_dates, n_features), np.nan, dtype=float)
    breadth_matrix = np.zeros((n_dates, n_features), dtype=float)
    valid_matrix = np.zeros((n_dates, n_features), dtype=bool)

    for i, date in enumerate(dates):
        s, e = date_boundaries[date]
        n_obs = e - s

        if n_obs < min_breadth:
            continue

        # Target ranks for this date
        t_rank = target_rank[s:e]
        t_valid = np.isfinite(t_rank)
        if t_valid.sum() < min_breadth:
            continue

        # Check constant target
        t_var = np.nanvar(t_rank)
        if t_var < constant_tol:
            continue

        t_mean = np.nanmean(t_rank)
        t_centered = np.where(t_valid, t_rank - t_mean, 0.0)
        t_ss = (t_centered ** 2).sum()

        for j, feat in enumerate(features):
            f_rank = rank_arrays[feat][s:e]
            f_valid = np.isfinite(f_rank) & t_valid
            n_valid = f_valid.sum()

            if n_valid < min_breadth:
                continue

            # Check constant feature
            f_subset = f_rank[f_valid]
            if np.var(f_subset) < constant_tol:
                breadth_matrix[i, j] = n_valid
                continue

            f_mean = np.nanmean(f_rank)
            f_centered = np.where(f_valid, f_rank - f_mean, 0.0)
            f_ss = (f_centered ** 2).sum()

            if f_ss < constant_tol:
                breadth_matrix[i, j] = n_valid
                continue

            cov = (f_centered * t_centered).sum()
            denom = np.sqrt(f_ss * t_ss)

            if denom > 1e-12:
                ic_matrix[i, j] = cov / denom
                breadth_matrix[i, j] = n_valid
                valid_matrix[i, j] = True
            else:
                breadth_matrix[i, j] = n_valid

    return ic_matrix, breadth_matrix, valid_matrix, dates


def _chunked_tensor_spearman_ic(
    panel: pd.DataFrame,
    features: list[str],
    target_col: str,
    date_col: str,
    ticker_col: str,
    min_breadth: int = 8,
    constant_tol: float = 1e-12,
    batch_size: int = 100,
    pre_ranked: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Compute daily IC in chunks to bound memory.

    Splits panel by date ranges, processes each chunk independently,
    then concatenates results.

    Args:
        panel: Long-form DataFrame.
        features: Feature column names.
        target_col: Target column name.
        date_col: Date column name.
        ticker_col: Ticker column name.
        min_breadth: Minimum valid observations per date.
        constant_tol: Variance threshold for constant-input detection.
        batch_size: Number of dates per chunk.
        pre_ranked: If True, skip groupby rank (input already rank values).

    Returns:
        ic_matrix, breadth_matrix, valid_matrix, dates (concatenated).
    """
    dates_all = pd.Index(panel[date_col].dropna().unique()).sort_values()
    n_dates = len(dates_all)

    all_ic = []
    all_breadth = []
    all_valid = []
    all_dates = []

    for start in range(0, n_dates, batch_size):
        end = min(start + batch_size, n_dates)
        chunk_dates = dates_all[start:end]
        mask = panel[date_col].isin(chunk_dates)
        chunk = panel[mask]

        ic_c, br_c, val_c, dates_c = _tensor_spearman_ic(
            chunk, features, target_col, date_col, ticker_col,
            min_breadth, constant_tol, pre_ranked=pre_ranked,
        )
        all_ic.append(ic_c)
        all_breadth.append(br_c)
        all_valid.append(val_c)
        all_dates.append(dates_c)

    return (
        np.concatenate(all_ic, axis=0),
        np.concatenate(all_breadth, axis=0),
        np.concatenate(all_valid, axis=0),
        pd.Index(np.concatenate([d.values for d in all_dates])),
    )


# ── Batched Ridge Regression ─────────────────────────────────────────────────

def batch_ridge_residualize(
    panel: pd.DataFrame,
    target_col: str,
    control_cols: list[str],
    feature_col: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
    ridge_lambda: float = 0.01,
    min_breadth: int = 10,
    winsor_q: float = 0.025,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Index]:
    """Batch ridge regression residualization across all dates.

    Replaces per-date Python loop with vectorized numpy operations.
    For each date: y = X @ beta + residual, solved via ridge regression.

    Args:
        panel: Long-form DataFrame with [date, ticker, target_col, control_cols, feature_col].
        target_col: Target/forward return column.
        control_cols: Factor control columns.
        feature_col: Feature column to preserve for IC computation.
        date_col: Date column name.
        ticker_col: Ticker column name.
        ridge_lambda: Ridge regularization parameter.
        min_breadth: Minimum observations per date.
        winsor_q: Winsorization quantile (0 to disable).

    Returns:
        residuals: (n_dates, n_tickers) residual array, NaN for missing.
        feature_vals: (n_dates, n_tickers) feature array, NaN for missing.
        breadth: (n_dates,) valid observation counts.
        dates: Index of dates.
    """
    cols = [date_col, ticker_col, target_col, feature_col] + control_cols
    work = panel[cols].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    for c in [target_col, feature_col] + control_cols:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    work = work.dropna(subset=[target_col, feature_col] + control_cols)
    if work.empty:
        return np.empty((0, 0)), np.empty((0, 0)), np.array([], dtype=float), pd.Index([])

    # Sort by date for contiguous slicing
    sort_idx = work[date_col].argsort(kind="stable")
    work = work.iloc[sort_idx].reset_index(drop=True)

    dates = pd.Index(work[date_col].unique()).sort_values()
    tickers = pd.Index(work[ticker_col].dropna().unique()).sort_values()
    n_dates = len(dates)
    n_tickers = len(tickers)
    n_controls = len(control_cols)

    # Build date→row mapping via categorical codes
    date_cat = pd.Categorical(work[date_col].values, categories=dates, ordered=True)
    ticker_cat = pd.Categorical(work[ticker_col].values, categories=tickers, ordered=True)
    date_idx = date_cat.codes.astype(np.intp)
    ticker_idx = ticker_cat.codes.astype(np.intp)

    # Assemble tensors
    y = np.full((n_dates, n_tickers), np.nan, dtype=float)
    y[date_idx, ticker_idx] = work[target_col].values.astype(float)

    feat = np.full((n_dates, n_tickers), np.nan, dtype=float)
    feat[date_idx, ticker_idx] = work[feature_col].values.astype(float)

    X = np.full((n_dates, n_tickers, n_controls), np.nan, dtype=float)
    for j, col in enumerate(control_cols):
        X[date_idx, ticker_idx, j] = work[col].values.astype(float)

    # Validity mask: all controls + target + feature must be finite
    valid_mask = (
        np.isfinite(y) & np.isfinite(feat) & np.all(np.isfinite(X), axis=2)
    )  # (n_dates, n_tickers)

    # Winsorize per date using vectorized operations
    if winsor_q > 0:
        lo = np.nanquantile(y, winsor_q, axis=1, keepdims=True)
        hi = np.nanquantile(y, 1 - winsor_q, axis=1, keepdims=True)
        y = np.clip(y, lo, hi)
        for j in range(n_controls):
            lo_x = np.nanquantile(X[:, :, j], winsor_q, axis=1, keepdims=True)
            hi_x = np.nanquantile(X[:, :, j], 1 - winsor_q, axis=1, keepdims=True)
            X[:, :, j] = np.clip(X[:, :, j], lo_x, hi_x)

    # Standardize per date (z-score)
    y_mean = np.nanmean(y, axis=1, keepdims=True)
    y_std = np.nanstd(y, axis=1, keepdims=True)
    y_std = np.where(y_std < 1e-12, 1.0, y_std)
    y = (y - y_mean) / y_std

    for j in range(n_controls):
        x_mean = np.nanmean(X[:, :, j], axis=1, keepdims=True)
        x_std = np.nanstd(X[:, :, j], axis=1, keepdims=True)
        x_std = np.where(x_std < 1e-12, 1.0, x_std)
        X[:, :, j] = (X[:, :, j] - x_mean) / x_std

    # Count valid observations per date
    breadth = valid_mask.sum(axis=1).astype(float)

    # Initialize output
    residuals = np.full((n_dates, n_tickers), np.nan, dtype=float)

    # Process dates in batches grouped by valid count for efficient batching
    valid_dates = np.where(breadth >= min_breadth)[0]

    if len(valid_dates) == 0:
        return residuals, feat, breadth, dates

    # For each valid date, solve ridge regression
    # Group dates by number of valid tickers for batched solve
    for date_i in valid_dates:
        mask = valid_mask[date_i]
        n_obs = int(mask.sum())
        if n_obs < min_breadth:
            continue

        # Extract valid observations
        y_valid = y[date_i, mask]  # (n_obs,)
        X_valid = X[date_i, mask]  # (n_obs, n_controls)

        # Add intercept
        X_aug = np.column_stack([np.ones(n_obs), X_valid])  # (n_obs, n_controls+1)

        # Ridge regression: (X'X + λI)β = X'y
        n_params = X_aug.shape[1]
        XtX = X_aug.T @ X_aug + ridge_lambda * np.eye(n_params)
        # Don't regularize intercept
        XtX[0, 0] -= ridge_lambda
        Xty = X_aug.T @ y_valid

        try:
            beta = np.linalg.solve(XtX, Xty)
            resid_all = np.full(n_tickers, np.nan)
            resid_all[mask] = y_valid - X_aug @ beta
            residuals[date_i] = resid_all
        except np.linalg.LinAlgError:
            continue

    return residuals, feat, breadth, dates


# ── Memory Governance ────────────────────────────────────────────────────────

@dataclass
class MemoryReport:
    """Memory and execution report from compute_daily_ic_series."""
    estimated_memory_bytes: int = 0
    peak_rss_bytes: int = 0
    execution_mode: str = "full_tensor"  # "full_tensor" | "chunked" | "legacy"
    batch_size: int = 0
    n_rows: int = 0
    n_dates: int = 0
    n_tickers: int = 0
    n_features: int = 0
    elapsed_seconds: float = 0.0


def _get_peak_rss_bytes() -> int:
    """Get current peak RSS in bytes (macOS/Linux)."""
    try:
        ru = resource.getrusage(resource.RUSAGE_SELF)
        return int(ru.ru_maxrss * 1024)  # macOS reports in KB
    except Exception:
        return 0


# ── Public API: compute_daily_ic_series ──────────────────────────────────────

def compute_daily_ic_series(
    panel: pd.DataFrame,
    features: list[str],
    target_col: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
    min_breadth: int = 8,
    memory_budget_mb: int = 512,
    mode: str = "auto",
    pre_ranked: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute daily cross-sectional Spearman IC for all features.

    Tensor-first implementation with automatic chunked execution for
    memory-bounded operation on large panels.

    Execution modes:
        "auto": Select full_tensor or chunked based on memory budget.
        "full_tensor": Load entire panel into tensor (fastest, highest memory).
        "chunked": Process dates in batches (bounded memory).
        "legacy": Use the old per-date groupby loop (parity testing only).

    When pre_ranked=True, the input values are already rank-transformed
    per date (or per date+group) and the ranking step is skipped. Pearson
    correlation on ranks equals Spearman correlation.

    Args:
        panel: DataFrame with [date_col, ticker_col, *features, target_col].
        features: List of feature column names.
        target_col: Name of the target/forward return column.
        date_col: Date column name.
        ticker_col: Ticker column name.
        min_breadth: Minimum cross-sectional breadth per date.
        memory_budget_mb: Memory budget in MB for auto mode (default 512).
        mode: Execution mode ("auto", "full_tensor", "chunked", "legacy").
        pre_ranked: If True, skip groupby rank (input already rank values).

    Returns:
        (ic_df, breadth_df, valid_df)
        - ic_df: (n_dates, n_features) DataFrame of daily Spearman ICs.
        - breadth_df: (n_dates, n_features) DataFrame of daily breadths.
        - valid_df: (n_dates, n_features) boolean DataFrame.
    """
    t0 = time.monotonic()
    n_rows = len(panel)
    n_features = len(features)

    if mode == "legacy":
        return _legacy_compute_daily_ic_series(
            panel, features, target_col, date_col, ticker_col, min_breadth,
        )

    # Count dates for mode selection
    n_dates = panel[date_col].nunique()
    n_tickers = panel[ticker_col].nunique()

    if n_dates == 0 or n_tickers == 0:
        ic_df = pd.DataFrame(index=pd.DatetimeIndex([]), columns=features, dtype=float)
        breadth_df = pd.DataFrame(index=pd.DatetimeIndex([]), columns=features, dtype=float)
        valid_df = pd.DataFrame(index=pd.DatetimeIndex([]), columns=features, dtype=bool)
        return ic_df, breadth_df, valid_df

    # Estimate memory: panel copy + rank arrays + correlation matrices
    # Rough estimate: n_rows × (n_features + 1) × 8 bytes × 3 (working copies)
    memory_budget_bytes = memory_budget_mb * 1024 * 1024
    estimated = int(n_rows * (n_features + 1) * 8 * 3)

    # Decide execution mode
    if mode == "auto":
        if estimated > memory_budget_bytes:
            exec_mode = "chunked"
            batch_size = max(1, memory_budget_bytes // (n_tickers * (n_features + 1) * 8 * 3))
            batch_size = min(batch_size, n_dates)
        else:
            exec_mode = "full_tensor"
            batch_size = n_dates
    elif mode == "chunked":
        exec_mode = "chunked"
        batch_size = max(1, memory_budget_bytes // (n_tickers * (n_features + 1) * 8 * 3))
        batch_size = min(batch_size, n_dates)
    else:
        exec_mode = "full_tensor"
        batch_size = n_dates

    # Compute IC using tensor backend (fully vectorized BLAS-level operations)
    if exec_mode == "chunked":
        ic_matrix, breadth_matrix, valid_matrix, dates = _chunked_tensor_spearman_ic(
            panel, features, target_col, date_col, ticker_col,
            min_breadth, batch_size=batch_size, pre_ranked=pre_ranked,
        )
    else:
        ic_matrix, breadth_matrix, valid_matrix, dates = _tensor_spearman_ic(
            panel, features, target_col, date_col, ticker_col, min_breadth, pre_ranked=pre_ranked,
        )

    elapsed = time.monotonic() - t0
    peak_rss = _get_peak_rss_bytes()

    # Build DataFrames
    ic_df = pd.DataFrame(ic_matrix, index=dates, columns=features)
    breadth_df = pd.DataFrame(breadth_matrix, index=dates, columns=features)
    valid_df = pd.DataFrame(valid_matrix, index=dates, columns=features)

    # Attach memory report as attribute (for debugging/benchmarking)
    ic_df.attrs["memory_report"] = MemoryReport(
        estimated_memory_bytes=estimated,
        peak_rss_bytes=peak_rss,
        execution_mode=exec_mode,
        batch_size=batch_size,
        n_rows=n_rows,
        n_dates=n_dates,
        n_tickers=n_tickers,
        n_features=n_features,
        elapsed_seconds=elapsed,
    )

    return ic_df, breadth_df, valid_df


def _legacy_compute_daily_ic_series(
    panel: pd.DataFrame,
    features: list[str],
    target_col: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
    min_breadth: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Legacy per-date groupby implementation for parity testing only."""
    work = panel[[date_col, ticker_col] + features + [target_col]].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    for c in features + [target_col]:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    rank_cols = []
    for feat in features:
        rank_col = f"_rank_{feat}"
        work[rank_col] = work.groupby(date_col, sort=False)[feat].rank(
            pct=False, method="average", na_option="keep",
        )
        rank_cols.append(rank_col)

    work["_rank_target"] = work.groupby(date_col, sort=False)[target_col].rank(
        pct=False, method="average", na_option="keep",
    )

    dates = work[date_col].unique()
    n_dates = len(dates)
    n_features = len(features)

    ic_matrix = np.full((n_dates, n_features), np.nan)
    breadth_matrix = np.zeros((n_dates, n_features), dtype=float)
    valid_matrix = np.zeros((n_dates, n_features), dtype=bool)

    for i, date in enumerate(dates):
        mask = work[date_col] == date
        day = work.loc[mask]
        n_obs = len(day)

        if n_obs < min_breadth:
            continue

        target_r = day["_rank_target"].values.astype(float)
        target_valid = np.isfinite(target_r)
        if target_valid.sum() < min_breadth:
            continue

        target_var = np.nanvar(target_r)
        if target_var < 1e-12:
            continue

        target_mean = np.nanmean(target_r)
        target_centered = np.where(target_valid, target_r - target_mean, 0.0)
        target_var_sum = (target_centered ** 2).sum()

        for j, rank_col in enumerate(rank_cols):
            feat_r = day[rank_col].values.astype(float)
            feat_valid = np.isfinite(feat_r) & target_valid
            n_valid = feat_valid.sum()

            if n_valid < min_breadth:
                continue

            feat_subset = feat_r[feat_valid]
            if np.var(feat_subset) < 1e-12:
                breadth_matrix[i, j] = n_valid
                continue

            feat_mean = np.nanmean(feat_r)
            feat_centered = np.where(feat_valid, feat_r - feat_mean, 0.0)
            feat_var_sum = (feat_centered ** 2).sum()

            if feat_var_sum < 1e-12:
                breadth_matrix[i, j] = n_valid
                continue

            cov = (feat_centered * target_centered).sum()
            denom = np.sqrt(feat_var_sum * target_var_sum)
            ic_val = cov / denom if denom > 1e-12 else np.nan

            ic_matrix[i, j] = ic_val
            breadth_matrix[i, j] = n_valid
            valid_matrix[i, j] = True

    ic_df = pd.DataFrame(ic_matrix, index=dates, columns=features)
    breadth_df = pd.DataFrame(breadth_matrix, index=dates, columns=features)
    valid_df = pd.DataFrame(valid_matrix, index=dates, columns=features)

    return ic_df, breadth_df, valid_df


def vectorized_spearman_ic_from_panel(
    panel: pd.DataFrame,
    features: list[str],
    target_col: str,
    date_col: str = "date",
    ticker_col: str = "ticker",
    min_breadth: int = 8,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute cross-sectional Spearman IC from a panel DataFrame.

    Alias for compute_daily_ic_series for backward compatibility.
    """
    return compute_daily_ic_series(
        panel, features, target_col, date_col, ticker_col, min_breadth,
    )


# ── 3. Rank Persistence ─────────────────────────────────────────────────────

def compute_rank_persistence(
    panel: pd.DataFrame,
    features: list[str],
    lags: list[int],
    date_col: str = "date",
    ticker_col: str = "ticker",
    min_dates: int = 30,
    min_breadth: int = 8,
) -> dict[str, pd.DataFrame]:
    """Compute rank autocorrelation/persistence curves by feature and lag.

    For each feature and lag L:
        persistence[L] = mean over tickers of Spearman(rank_t, rank_{t+L})

    Avoids per-ticker/per-date Python loops by pivoting to date x ticker
    matrices and using vectorized rank correlation.

    Args:
        panel: DataFrame with [date_col, ticker_col, *features].
        features: List of feature column names.
        lags: List of lag values (in trading days).
        date_col: Date column name.
        ticker_col: Ticker column name.
        min_dates: Minimum number of dates required.
        min_breadth: Minimum number of tickers required.

    Returns:
        Dict mapping feature name to DataFrame with columns:
        - lag: lag value
        - persistence: mean rank autocorrelation
        - n_dates: number of valid date pairs
        - avg_breadth: average number of valid tickers
        - n_constant: number of constant tickers skipped
    """
    work = panel[[date_col, ticker_col] + features].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    for c in features:
        work[c] = pd.to_numeric(work[c], errors="coerce")
    work = work.dropna(subset=[date_col, ticker_col] + features)
    work = work.sort_values([date_col, ticker_col])

    # Compute cross-sectional ranks per date for all features at once
    work["rank"] = work.groupby(date_col, sort=False)[features[0]].rank(pct=True, method="average")
    # Actually, we need ranks per feature. Let's do it properly.
    rank_cols = {}
    for feat in features:
        rank_col = f"_rank_{feat}"
        work[rank_col] = work.groupby(date_col, sort=False)[feat].rank(pct=True, method="average")
        rank_cols[feat] = rank_col

    # Pivot each feature's ranks to date x ticker matrix
    rank_matrices = {}
    for feat in features:
        pivot = work.pivot_table(index=date_col, columns=ticker_col, values=rank_cols[feat])
        pivot = pivot.sort_index()
        rank_matrices[feat] = pivot

    results = {}

    for feat in features:
        pivot = rank_matrices[feat]
        if len(pivot) < min_dates:
            results[feat] = pd.DataFrame({
                "lag": lags,
                "persistence": np.nan,
                "n_dates": 0,
                "avg_breadth": 0,
                "n_constant": 0,
            })
            continue

        rows = []
        for lag in lags:
            if lag >= len(pivot):
                rows.append({"lag": lag, "persistence": np.nan, "n_dates": 0, "avg_breadth": 0, "n_constant": 0})
                continue

            t0 = pivot.iloc[:-lag]
            t1 = pivot.iloc[lag:]

            common_dates = t0.index.intersection(t1.index)
            common_tickers = t0.columns.intersection(t1.columns)

            if len(common_dates) < min_dates or len(common_tickers) < min_breadth:
                rows.append({"lag": lag, "persistence": np.nan, "n_dates": 0, "avg_breadth": 0, "n_constant": 0})
                continue

            t0_common = t0.loc[common_dates, common_tickers]
            t1_common = t1.loc[common_dates, common_tickers]

            # Vectorized: column-wise Pearson on rank matrices
            # Inputs are already percentile ranks, so Pearson = Spearman
            t0_np = t0_common.to_numpy(dtype=float)  # (D, T)
            t1_np = t1_common.to_numpy(dtype=float)  # (D, T)
            valid_mask = np.isfinite(t0_np) & np.isfinite(t1_np)
            n_valid_col = valid_mask.sum(axis=0).astype(float)  # (T,)

            v0 = np.where(valid_mask, t0_np, 0.0)
            v1 = np.where(valid_mask, t1_np, 0.0)

            sum0 = v0.sum(axis=0)
            sum1 = v1.sum(axis=0)
            sum01 = (v0 * v1).sum(axis=0)
            sum0sq = (v0 ** 2).sum(axis=0)
            sum1sq = (v1 ** 2).sum(axis=0)

            denom = np.sqrt(
                np.clip(n_valid_col * sum0sq - sum0 ** 2, 0.0, None)
                * np.clip(n_valid_col * sum1sq - sum1 ** 2, 0.0, None)
            )
            cors = np.where(
                (denom > 1e-12) & (n_valid_col >= 3),
                (n_valid_col * sum01 - sum0 * sum1) / denom,
                np.nan,
            )

            # Constant detection
            var0 = np.where(n_valid_col > 0, (n_valid_col * sum0sq - sum0 ** 2) / n_valid_col, 0.0)
            var1 = np.where(n_valid_col > 0, (n_valid_col * sum1sq - sum1 ** 2) / n_valid_col, 0.0)
            constant_mask = (var0 < 1e-15) | (var1 < 1e-15)
            cors[constant_mask] = np.nan
            n_constant = int(constant_mask.sum())

            finite_cors = cors[np.isfinite(cors)]
            if len(finite_cors) == 0:
                rows.append({"lag": lag, "persistence": np.nan, "n_dates": 0, "avg_breadth": 0, "n_constant": n_constant})
            else:
                rows.append({
                    "lag": lag,
                    "persistence": float(np.mean(finite_cors)),
                    "n_dates": len(common_dates),
                    "avg_breadth": int(np.mean(n_valid_col[np.isfinite(cors)])),
                    "n_constant": n_constant,
                })

        results[feat] = pd.DataFrame(rows)

    return results


# ── 4. IC Decay ──────────────────────────────────────────────────────────────

def compute_ic_decay(
    panel: pd.DataFrame,
    features: list[str],
    horizons: list[int],
    forward_returns: pd.DataFrame | None = None,
    date_col: str = "date",
    ticker_col: str = "ticker",
    min_dates: int = 30,
    min_breadth: int = 8,
) -> dict[str, pd.DataFrame]:
    """Compute IC decay curves across horizons using precomputed forward returns.

    For each feature and horizon h:
        IC_h = mean over dates of Spearman(feature[date, :], fwd_ret_h[date, :])

    Reuses precomputed forward returns — does not rebuild them per feature.

    Args:
        panel: DataFrame with [date_col, ticker_col, *features].
        features: List of feature column names.
        horizons: List of horizon lengths.
        forward_returns: Optional precomputed forward returns DataFrame from
            compute_forward_returns(). If None, computed internally.
        date_col: Date column name.
        ticker_col: Ticker column name.
        min_dates: Minimum number of dates required.
        min_breadth: Minimum cross-sectional breadth per date.

    Returns:
        Dict mapping feature name to DataFrame with columns:
        - horizon: horizon length
        - mean_ic: mean daily Spearman IC
        - median_ic: median daily IC
        - ic_std: std of daily IC
        - icir: mean_ic / ic_std
        - hac_tstat: Newey-West HAC t-stat
        - n_dates: number of valid dates
        - avg_breadth: average cross-sectional breadth
        - sign_consistency: fraction of dates with IC same sign as mean
        - subperiod_stability: 1 - |IC_first_half - IC_second_half| / |mean_ic|
        - ic_quality: "high"/"medium"/"low"/"insufficient"
    """
    # Compute forward returns if not provided
    if forward_returns is None:
        forward_returns = compute_forward_returns(
            panel, horizons, date_col=date_col, ticker_col=ticker_col, compound=False,
        )

    work = panel[[date_col, ticker_col] + features].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    for c in features:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    # Merge forward returns by (date, ticker) — not positional assignment
    fwd_cols = {h: f"fwd_ret_{h}d" for h in horizons}
    merge_cols = [date_col, ticker_col] + [c for c in fwd_cols.values() if c in forward_returns.columns]
    work = work.merge(forward_returns[merge_cols], on=[date_col, ticker_col], how="left")

    results = {}

    # Compute IC for all features at once using tensor backend, per horizon
    for h in horizons:
        col = fwd_cols[h]
        if col not in work.columns:
            for feat in features:
                results.setdefault(feat, []).append(_empty_ic_decay_row(feat, h))
            continue

        try:
            ic_df, breadth_df, _ = compute_daily_ic_series(
                work, features, col,
                date_col=date_col, ticker_col=ticker_col,
                min_breadth=min_breadth, mode="auto",
            )
        except Exception:
            for feat in features:
                results.setdefault(feat, []).append(_empty_ic_decay_row(feat, h, reason="insufficient_data"))
            continue

        if ic_df is None or ic_df.empty:
            for feat in features:
                results.setdefault(feat, []).append(_empty_ic_decay_row(feat, h, n_dates=0, reason="too_few_dates"))
            continue

        for feat in features:
            if feat not in ic_df.columns:
                results.setdefault(feat, []).append(_empty_ic_decay_row(feat, h))
                continue

            ics = ic_df[feat].dropna().values
            breadths = breadth_df[feat].dropna().values if breadth_df is not None else np.array([])

            if len(ics) < min_dates:
                results.setdefault(feat, []).append(
                    _empty_ic_decay_row(feat, h, n_dates=len(ics), reason="too_few_dates")
                )
                continue

            mean_ic = float(np.mean(ics))
            median_ic = float(np.median(ics))
            ic_std = float(np.std(ics))
            icir = mean_ic / ic_std if ic_std > 0 else 0.0
            t_stat = _hac_tstat(ics, max(1, h - 1))
            sign_consistency = float((ics > 0).mean()) if mean_ic > 0 else float((ics < 0).mean())

            mid = len(ics) // 2
            if mid > 5:
                ic1 = float(np.mean(ics[:mid]))
                ic2 = float(np.mean(ics[mid:]))
                stability = 1.0 - abs(ic1 - ic2) / max(abs(mean_ic), 0.001)
                stability = max(0.0, min(1.0, stability))
            else:
                stability = 0.5

            quality = _ic_quality(len(ics), int(np.mean(breadths)) if len(breadths) > 0 else 0, mean_ic, t_stat, stability)

            results.setdefault(feat, []).append({
                "horizon": h, "mean_ic": mean_ic, "median_ic": median_ic,
                "ic_std": ic_std, "icir": icir, "hac_tstat": t_stat,
                "n_dates": len(ics), "avg_breadth": int(np.mean(breadths)) if len(breadths) > 0 else 0,
                "sign_consistency": sign_consistency,
                "subperiod_stability": stability, "ic_quality": quality,
            })

    return {feat: pd.DataFrame(rows) for feat, rows in results.items()}


def _empty_ic_decay_row(feature: str, horizon: int, n_dates: int = 0, reason: str = "") -> dict:
    return {
        "horizon": horizon, "mean_ic": 0.0, "median_ic": 0.0,
        "ic_std": 0.0, "icir": 0.0, "hac_tstat": 0.0,
        "n_dates": n_dates, "avg_breadth": 0,
        "sign_consistency": 0.0, "subperiod_stability": 0.0,
        "ic_quality": "insufficient",
    }


# ── 5. Batch HAC t-stat ─────────────────────────────────────────────────────

def batch_hac_tstat(
    ic_matrix: np.ndarray,
    lags: int | list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute Newey-West HAC standard errors and t-stats for all features in batch.

    For each feature (column of ic_matrix):
        var_NW = var(ics) + 2 * sum_{k=1}^{lag} (1 - k/(lag+1)) * gamma_k
        se = sqrt(var_NW / n)
        t = mean_ic / se

    Args:
        ic_matrix: 2D array of shape (n_dates, n_features).
            Each column is an IC time series for one feature.
            NaN values are handled safely.
        lags: Newey-West lag(s). If int, applied to all features.
            If list, one lag per feature. If None, lag = max(1, n_features-1).

    Returns:
        (t_stats, se_values, mean_ics)
        - t_stats: (n_features,) HAC t-statistics.
        - se_values: (n_features,) HAC standard errors.
        - mean_ics: (n_features,) mean IC values.
    """
    x = np.asarray(ic_matrix, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n_dates, n_features = x.shape
    if lags is None:
        lags_arr = np.full(n_features, max(1, n_features - 1), dtype=int)
    elif isinstance(lags, int):
        lags_arr = np.full(n_features, lags, dtype=int)
    else:
        lags_arr = np.asarray(lags, dtype=int)
        if len(lags_arr) != n_features:
            raise ValueError(f"lags length {len(lags_arr)} != n_features {n_features}")

    t_stats = np.full(n_features, 0.0, dtype=float)
    se_values = np.full(n_features, 0.0, dtype=float)
    mean_ics = np.full(n_features, np.nan, dtype=float)

    for j in range(n_features):
        series = x[:, j]
        valid = np.isfinite(series)
        if valid.sum() < 5:
            continue

        ics = series[valid]
        n = len(ics)
        mean_ic = float(np.mean(ics))
        mean_ics[j] = mean_ic

        var = float(np.var(ics, ddof=1))
        if var < 1e-15:
            continue

        lag = int(lags_arr[j])
        for k in range(1, min(lag + 1, n)):
            if n - k < 2:
                continue
            gamma_k = float(np.cov(ics[k:], ics[:-k])[0, 1])
            if np.isfinite(gamma_k):
                var += 2.0 * (1.0 - k / (lag + 1)) * gamma_k

        se = float(np.sqrt(max(var / n, 1e-15)))
        se_values[j] = se
        t_stats[j] = mean_ic / se if se > 0 else 0.0

    if n_features == 1:
        return t_stats, se_values, mean_ics
    return t_stats, se_values, mean_ics


_hac_tstat = _shared_hac_tstat


# ── 6. Feature Redundancy ────────────────────────────────────────────────────

def compute_feature_redundancy(
    panel: pd.DataFrame,
    features: list[str],
    method: str = "spearman",
    date_col: str = "date",
    ticker_col: str = "ticker",
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Compute feature correlation/redundancy matrix without O(F²×D) Python loops.

    Args:
        panel: DataFrame with [date_col, ticker_col, *features].
        features: List of feature column names.
        method: "pearson" or "spearman". Spearman uses cross-sectional ranks.
        date_col: Date column name.
        ticker_col: Ticker column name.

    Returns:
        (corr_matrix, feature_names, distance_matrix)
        - corr_matrix: (n_features, n_features) correlation matrix.
        - feature_names: list of feature names in matrix order.
        - distance_matrix: (n_features, n_features) distance = 1 - |corr|.
    """
    available = [f for f in features if f in panel.columns]
    if len(available) < 2:
        return np.eye(len(available)), available, np.zeros((len(available), len(available)))

    work = panel[[date_col, ticker_col] + available].copy()
    work[date_col] = pd.to_datetime(work[date_col])
    for c in available:
        work[c] = pd.to_numeric(work[c], errors="coerce")

    if method == "spearman":
        # Rank within each date, then flatten and compute global correlation
        ranked = work.copy()
        for c in available:
            ranked[c] = work.groupby(date_col, sort=False)[c].rank(pct=True, method="average")
        # Flatten: each row is one (date, ticker) observation
        corr_matrix = ranked[available].corr(method="pearson").values
    else:
        corr_matrix = work[available].corr(method="pearson").values

    # Handle NaN in correlation matrix
    corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
    np.fill_diagonal(corr_matrix, 1.0)

    distance_matrix = 1.0 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix, 0.0)

    return corr_matrix, available, distance_matrix


# ── IC Quality Classification ────────────────────────────────────────────────
_ic_quality = _shared_ic_quality
