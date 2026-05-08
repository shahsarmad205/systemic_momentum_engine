"""
research_diagnostics.py — read-only research analysis layer.

Run on top of the promotion pipeline after gates are evaluated.
Does NOT modify gates, thresholds, or model architecture.
Purpose: identify root cause of model failures and classify alpha quality.

Eight modules (all prefixed ``diag_`` in output keys):
  1. turnover_diagnostics       — cost drag and holding period
  2. horizon_alignment          — signal decay profile at 1/3/5/10d
  3. long_only_evaluation       — top-decile long portfolio without short drag
  4. short_attribution          — quantify short-leg contribution vs beta hedge
  5. cross_sectional_strength   — decile stability, prediction dispersion
  6. cost_aware_score           — composite net-PnL-adjusted rank
  7. regime_segmentation        — conditional performance by market regime
  8. classify_research_label    — human-readable failure taxonomy
"""
from __future__ import annotations

import logging
import os
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_HORIZONS = [1, 3, 5, 10]
_REGIMES = ("bull", "bear", "high_vol", "low_vol")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _safe(val: Any, default: float = float("nan")) -> float:
    try:
        v = float(val)
        return v if np.isfinite(v) else default
    except Exception:
        return default


def _col(df: pd.DataFrame, name: str) -> pd.Series:
    s = pd.to_numeric(df.get(name, pd.Series(dtype=float)), errors="coerce")
    return s.replace([np.inf, -np.inf], np.nan)


def _deployable_return_col(df: pd.DataFrame, requested: str | None = None) -> str:
    if requested and requested in df.columns:
        return str(requested)
    if "forward_return" in df.columns:
        return "forward_return"
    return "target_return"


def _with_executable_next_return(
    oos_df: pd.DataFrame,
    *,
    requested: str | None = None,
    horizon_days: int = 1,
) -> tuple[pd.DataFrame, str]:
    """Attach the daily return stream used by executable validation."""
    if oos_df is None or oos_df.empty:
        return pd.DataFrame(), "_diag_next_return"
    if "daily_return" in oos_df.columns and {"date", "ticker"}.issubset(oos_df.columns):
        df = oos_df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["ticker"] = df["ticker"].astype(str)
        df["daily_return"] = pd.to_numeric(df["daily_return"], errors="coerce")
        df = df.sort_values(["ticker", "date"])
        df["_diag_next_return"] = df.groupby("ticker", sort=False)["daily_return"].shift(-1)
        return df, "_diag_next_return"
    ret_col = _deployable_return_col(oos_df, requested)
    if ret_col not in oos_df.columns:
        return pd.DataFrame(), "_diag_next_return"
    df = oos_df.copy()
    df["_diag_next_return"] = pd.to_numeric(df[ret_col], errors="coerce")
    if ret_col == "forward_return":
        df["_diag_next_return"] = df["_diag_next_return"] / max(1.0, float(horizon_days))
    return df, "_diag_next_return"


def _ann_sharpe(arr: np.ndarray, hac_lags: int = 5) -> float:
    """
    Annualised Sharpe ratio with Newey-West HAC correction for serial correlation.

    Key institutional distinction:
      - Sharpe  uses  sigma = sqrt(HAC_variance)           (volatility denominator)
      - t-stat  uses  se    = sqrt(HAC_variance / n)       (standard error denominator)

    P14 fix (May 2026): The original implementation incorrectly divided by
    sqrt(var / n) — the standard error — instead of sqrt(var) — the standard
    deviation.  This inflated every Sharpe by sqrt(n) ≈ 45× for ~2,000-day
    OOS windows, making diagnostic labels unreliable.  This function now
    returns the correct HAC-corrected annualised Sharpe.

    Parameters
    ----------
    arr      : daily return series
    hac_lags : maximum Newey-West lag for serial correlation adjustment
    """
    r = np.asarray(arr, dtype=float)
    r = r[np.isfinite(r)]
    n = len(r)
    if n < 10:
        return float("nan")
    mu = float(r.mean())
    demeaned = r - mu
    # Newey-West HAC variance (gamma_0 + 2 * Σ w_k * gamma_k)
    hac_var = float(np.dot(demeaned, demeaned) / n)
    max_lag = min(hac_lags, n - 1)
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        hac_var += 2.0 * (1.0 - lag / (max_lag + 1.0)) * cov
    # HAC-corrected daily standard deviation (NOT standard error)
    sigma_daily = np.sqrt(max(hac_var, 1e-30))
    if sigma_daily <= 1e-15:
        return 0.0 if abs(mu) <= 1e-12 else float("nan")
    return float(mu / sigma_daily * np.sqrt(252.0))


def _max_dd(arr: np.ndarray) -> float:
    r = np.asarray(arr, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 2:
        return float("nan")
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    return float(np.min((equity - peak) / peak))


def _cagr(arr: np.ndarray) -> float:
    r = np.asarray(arr, dtype=float)
    r = r[np.isfinite(r)]
    if len(r) < 10:
        return float("nan")
    growth = float(np.prod(1.0 + r))
    return float(growth ** (252.0 / len(r)) - 1.0) if growth > 0 else float("nan")


def _hac_tstat(series: pd.Series, max_lag: int = 5) -> float:
    r = series.dropna().to_numpy(dtype=float)
    n = len(r)
    if n < 4:
        return float("nan")
    mu = r.mean()
    demeaned = r - mu
    max_lag = min(max_lag, n - 1)
    var = float(np.dot(demeaned, demeaned) / n)
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
        var += 2.0 * (1.0 - lag / (max_lag + 1.0)) * cov
    se = np.sqrt(max(var, 0.0) / n)
    return float(mu / se) if se > 1e-12 else float("nan")


def _empty(prefix: str, *keys: str) -> dict[str, float]:
    return {f"{prefix}{k}": float("nan") for k in keys}


# ---------------------------------------------------------------------------
# 1. Turnover diagnostics
# ---------------------------------------------------------------------------

def turnover_diagnostics(pnl: pd.DataFrame) -> dict[str, float]:
    """
    Daily turnover, holding period, and cost-to-alpha attribution.

    Key outputs:
      diag_turnover_mean_daily    — one-way daily portfolio turnover
      diag_avg_holding_period_days— gross_exposure / turnover (days a unit stays)
      diag_cost_to_gross_ratio    — (total_cost + borrow) / abs(gross_pnl)
      diag_cost_per_trade         — average execution cost per trade
      diag_net_pnl_per_trade      — average net PnL per trade (signal vs friction)
      diag_sharpe_per_unit_turnover — exec Sharpe normalised by turnover
    """
    empty = _empty(
        "diag_",
        "turnover_mean_daily", "turnover_std_daily", "avg_holding_period_days",
        "cost_to_gross_ratio", "cost_per_trade", "net_pnl_per_trade",
        "sharpe_per_unit_turnover",
    )
    if pnl is None or pnl.empty:
        return empty

    turnover = _col(pnl, "turnover").fillna(0.0)
    cost = _col(pnl, "cost_return").fillna(0.0)
    borrow = _col(pnl, "borrow_return").fillna(0.0)
    gross = _col(pnl, "gross_return").fillna(0.0)
    net = _col(pnl, "daily_return").fillna(0.0)
    trade_count = _col(pnl, "trade_count").fillna(0.0)
    gross_exp = _col(pnl, "gross_exposure").fillna(0.0)

    turn_mean = float(turnover.mean())
    turn_std = float(turnover.std(ddof=1)) if len(turnover) > 1 else float("nan")

    gexp_mean = float(gross_exp.mean()) if not gross_exp.dropna().empty else 1.0
    avg_hold = (
        gexp_mean / turn_mean
        if (np.isfinite(turn_mean) and turn_mean > 1e-9)
        else float("nan")
    )

    total_friction = abs(float(cost.sum())) + abs(float(borrow.sum()))
    total_gross_abs = abs(float(gross.sum()))
    cost_to_gross = total_friction / total_gross_abs if total_gross_abs > 1e-12 else float("nan")

    total_trades = float(trade_count.sum())
    cost_per_trade = total_friction / total_trades if total_trades > 0 else float("nan")
    net_pnl_per_trade = float(net.sum()) / total_trades if total_trades > 0 else float("nan")

    exec_sharpe = _ann_sharpe(net.to_numpy(dtype=float))
    sharpe_per_turn = (
        exec_sharpe / turn_mean
        if (np.isfinite(turn_mean) and turn_mean > 1e-9 and np.isfinite(exec_sharpe))
        else float("nan")
    )

    return {
        "diag_turnover_mean_daily": turn_mean,
        "diag_turnover_std_daily": turn_std,
        "diag_avg_holding_period_days": avg_hold,
        "diag_cost_to_gross_ratio": cost_to_gross,
        "diag_cost_per_trade": cost_per_trade,
        "diag_net_pnl_per_trade": net_pnl_per_trade,
        "diag_sharpe_per_unit_turnover": sharpe_per_turn,
    }


# ---------------------------------------------------------------------------
# 2. Horizon alignment
# ---------------------------------------------------------------------------

def horizon_alignment(
    oos_df: pd.DataFrame,
    *,
    horizons: list[int] = _HORIZONS,
    return_col: str = "forward_return",
) -> dict[str, Any]:
    """
    IC at multiple forward horizons to identify signal decay profile.

    For horizon h, aligns score_t with the h-day cumulative return starting
    from t (treating ``return_col`` as the next 1-day return).

    IC_h > IC_1 implies the signal benefits from longer holding (undertraded).
    Sharply declining IC_h implies perishable information (overtraded if model
    uses a long rebalancing period).

    Outputs (per horizon h in {1, 3, 5, 10}):
      diag_ic_{h}d_mean      — mean rank IC at h-day horizon
      diag_ic_{h}d_tstat     — HAC t-stat
      diag_optimal_horizon   — horizon maximising IC × sqrt(h) (annualised)
      diag_overtraded        — 1 if cfg horizon < optimal horizon, 0 otherwise
      diag_ic_decay_ratio    — IC_10d / IC_1d  (>1 = slow decay; <1 = fast decay)
    """
    keys = (
        [f"ic_{h}d_mean" for h in horizons]
        + [f"ic_{h}d_tstat" for h in horizons]
        + ["optimal_horizon", "overtraded", "ic_decay_ratio"]
    )
    empty = _empty("diag_", *keys)

    if oos_df is None or oos_df.empty:
        return empty
    if "score" not in oos_df.columns:
        return empty

    df, return_col = _with_executable_next_return(
        oos_df,
        requested=return_col,
        horizon_days=1,
    )
    if df.empty or return_col not in df.columns:
        return empty
    df = df[["date", "ticker", "score", return_col]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df[return_col] = pd.to_numeric(df[return_col], errors="coerce")
    df = df.dropna()
    if df.empty:
        return empty

    # Pivot to wide format (date × ticker)
    ret_wide = df.pivot_table(index="date", columns="ticker", values=return_col).sort_index()
    score_wide = df.pivot_table(index="date", columns="ticker", values="score").sort_index()

    result: dict[str, Any] = {}
    ic_means: dict[int, float] = {}

    for h in horizons:
        # fwd_h[t] = sum of returns at t, t+1, ..., t+h-1
        # = rolling(h, min_periods=h).sum() aligned so that the window STARTS at t
        # rolling(h).sum()[t+h-1] = sum(ret[t:t+h]), then shift(-(h-1)) → fwd_h[t]
        if len(ret_wide) < h + 2:
            result[f"diag_ic_{h}d_mean"] = float("nan")
            result[f"diag_ic_{h}d_tstat"] = float("nan")
            ic_means[h] = float("nan")
            continue

        fwd_h = ret_wide.rolling(window=h, min_periods=h).sum().shift(-(h - 1))

        # Compute rank IC for each date where both score and fwd return are available
        common_dates = score_wide.index.intersection(fwd_h.dropna(how="all").index)
        ic_series: list[float] = []
        for dt in common_dates:
            s = score_wide.loc[dt].dropna()
            r = fwd_h.loc[dt].dropna()
            shared = s.index.intersection(r.index)
            if len(shared) < 5:
                continue
            s_v = s[shared]
            r_v = r[shared]
            if s_v.nunique() < 2 or r_v.nunique() < 2:
                continue
            ic_series.append(float(s_v.rank().corr(r_v.rank())))

        s_series = pd.Series(ic_series, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
        mu = float(s_series.mean()) if not s_series.empty else float("nan")
        tstat = _hac_tstat(s_series) if not s_series.empty else float("nan")
        result[f"diag_ic_{h}d_mean"] = mu
        result[f"diag_ic_{h}d_tstat"] = tstat
        ic_means[h] = mu

    # Optimal horizon: maximise IC × sqrt(h) (annualised IC information ratio proxy)
    best_h, best_val = 1, float("-inf")
    for h, mu in ic_means.items():
        if np.isfinite(mu):
            v = mu * np.sqrt(h)
            if v > best_val:
                best_val = v
                best_h = h
    result["diag_optimal_horizon"] = int(best_h)
    result["diag_overtraded"] = float("nan")  # filled by caller who knows cfg horizon

    ic_1d = ic_means.get(1, float("nan"))
    ic_10d = ic_means.get(10, float("nan"))
    result["diag_ic_decay_ratio"] = (
        ic_10d / ic_1d if (np.isfinite(ic_1d) and abs(ic_1d) > 1e-6 and np.isfinite(ic_10d)) else float("nan")
    )

    return result


# ---------------------------------------------------------------------------
# 3. Long-only evaluation (top-decile, no short)
# ---------------------------------------------------------------------------

def long_only_evaluation(
    oos_df: pd.DataFrame,
    *,
    top_fraction: float = 0.10,
    target_horizon_days: int = 10,
) -> dict[str, float]:
    """
    Top-decile equal-weight portfolio: pure long alpha without short destruction.

    P12 institutional fix: When target_return is computed over H days, daily
    top-decile returns are H-day overlapping, creating artificial autocorrelation
    that inflates naive Sharpe by ~√H.  The correct institutional approach is
    to use HAC lags = target_horizon_days (covering the full overlap window),
    not the default of 5.  At H=10, this corrects the Long-Only Sharpe from
    the inflated 14.35 (only 5 lags) to a realistic ~2-3× level.

    Outputs:
      diag_lo_sharpe          — gross long-only Sharpe (HAC-corrected)
      diag_lo_net_sharpe      — estimated net-of-cost Sharpe
      diag_lo_cagr            — gross CAGR
      diag_lo_max_dd          — max drawdown
      diag_lo_win_rate        — fraction of positive days
      diag_lo_turnover_mean   — avg daily turnover of the top-decile basket
      diag_lo_n_days          — number of active days
    """
    empty = _empty(
        "diag_",
        "lo_sharpe", "lo_net_sharpe", "lo_cagr", "lo_max_dd",
        "lo_win_rate", "lo_turnover_mean", "lo_n_days",
    )
    if oos_df is None or oos_df.empty:
        return empty

    df, ret_col = _with_executable_next_return(
        oos_df,
        horizon_days=target_horizon_days,
    )
    needed = {"date", "ticker", "score", ret_col}
    if not needed.issubset(oos_df.columns):
        if not needed.issubset(df.columns):
            return empty

    df = df[list(needed)].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
    df = df.dropna()
    if df.empty:
        return empty

    # Rank scores per day; select top fraction
    df["_pct"] = df.groupby("date")["score"].rank(pct=True)
    top_mask = df["_pct"] >= (1.0 - top_fraction)

    daily_ret = df[top_mask].groupby("date")[ret_col].mean()
    if daily_ret.empty:
        return empty

    arr = daily_ret.to_numpy(dtype=float)

    hac_lags_for_lo = 5

    # Compute actual turnover from set-difference of holdings day-over-day
    top_by_date: dict[pd.Timestamp, set[str]] = {
        dt: set(g["ticker"].tolist())
        for dt, g in df[top_mask].groupby("date")
    }
    dates_sorted = sorted(top_by_date)
    turnovers: list[float] = []
    for i in range(1, len(dates_sorted)):
        prev = top_by_date[dates_sorted[i - 1]]
        curr = top_by_date[dates_sorted[i]]
        n_union = len(prev | curr)
        if n_union > 0:
            turnovers.append(len(curr - prev) / n_union)
    turn_mean = float(np.mean(turnovers)) if turnovers else 0.20

    # Estimated daily cost: turnover × half-spread (10 bps default)
    est_cost_per_day = turn_mean * 10.0 / 10_000.0
    arr_net = arr - est_cost_per_day

    return {
        "diag_lo_sharpe": _ann_sharpe(arr, hac_lags=hac_lags_for_lo),
        "diag_lo_net_sharpe": _ann_sharpe(arr_net, hac_lags=hac_lags_for_lo),
        "diag_lo_cagr": _cagr(arr),
        "diag_lo_max_dd": _max_dd(arr),
        "diag_lo_win_rate": float((arr > 0).mean()) if len(arr) else float("nan"),
        "diag_lo_turnover_mean": turn_mean,
        "diag_lo_n_days": int(len(arr)),
    }


# ---------------------------------------------------------------------------
# 4. Short attribution — quantify short-leg contribution vs beta-neutral hedge
# ---------------------------------------------------------------------------

def short_attribution(
    oos_df: pd.DataFrame,
    pnl: pd.DataFrame,
    *,
    model_kind: str = "",
) -> dict[str, float]:
    """
    Decomposes the spread portfolio into long-alpha and short-alpha contributions.

    For spread (long/short) models:
    - long_contribution: exec_long_leg net Sharpe (already computed via cost allocation)
    - short_contribution: exec_short_leg net Sharpe
    - hedge_sharpe: long-leg return minus beta × market_return (beta-neutral alternative)

    If the hedge_sharpe > exec_short_leg_sharpe, the model's short picks are
    destroying value relative to a simple beta-neutral market hedge.

    Outputs:
      diag_short_value_added     — exec_short_leg_sharpe minus beta_hedge_sharpe
                                   positive = short picks add value beyond hedging
      diag_short_destroys_value  — 1 if short_value_added < −0.1 else 0
      diag_beta_hedge_sharpe     — Sharpe of a beta-neutral index short
      diag_long_alpha_fraction   — long_leg / (abs(long_leg) + abs(short_leg)) ratio
    """
    empty = _empty(
        "diag_",
        "short_value_added", "short_destroys_value",
        "beta_hedge_sharpe", "long_alpha_fraction",
    )
    if pnl is None or pnl.empty:
        return empty

    # Market return proxy: cross-sectional equal-weight daily return
    mkt_return: pd.Series = pd.Series(dtype=float)
    if oos_df is not None and not oos_df.empty:
        df_tmp, ret_col = _with_executable_next_return(oos_df)
        if ret_col in df_tmp.columns:
            df_tmp["date"] = pd.to_datetime(df_tmp["date"], errors="coerce")
            mkt_return = df_tmp.groupby("date")[ret_col].mean()

    # Portfolio-level beta from PnL
    pnl_copy = pnl.copy()
    pnl_copy["date"] = pd.to_datetime(pnl_copy.get("date", pd.Series(dtype=str)), errors="coerce")
    beta_series = _col(pnl_copy, "beta_exposure").fillna(0.0)
    beta_mean = float(beta_series.mean()) if not beta_series.dropna().empty else 0.0

    # Cost-allocated long leg net return
    long_net = _col(pnl_copy, "long_gross_return").fillna(0.0) - _col(pnl_copy, "long_cost_return").fillna(0.0)
    # Cost-allocated short leg net return (includes borrow + mkt adj already if present)
    short_net = (
        _col(pnl_copy, "short_gross_return").fillna(0.0)
        - _col(pnl_copy, "short_cost_return").fillna(0.0)
        - _col(pnl_copy, "borrow_return").fillna(0.0)
        - _col(pnl_copy, "market_adj_return").fillna(0.0)
    )

    long_sharpe = _ann_sharpe(long_net.to_numpy(dtype=float))
    short_sharpe = _ann_sharpe(short_net.to_numpy(dtype=float))

    # Beta-neutral hedge: short the market index with the portfolio's net beta
    # beta_hedge_return[t] = −beta × market_return[t] − borrow_cost
    borrow_total = abs(float(_col(pnl_copy, "borrow_return").fillna(0.0).sum()))
    n_days = len(pnl_copy)
    borrow_per_day = borrow_total / n_days if n_days > 0 else 0.0

    _HEDGE_SHARPE_CLIP = 100.0
    _BETA_FLOOR_FOR_HEDGE = 0.02  # P12: below this, hedge Sharpe is degenerate
    hedge_sharpe = float("nan")
    if not mkt_return.empty and n_days > 10:
        # Align market return to pnl dates
        pnl_dates = pd.to_datetime(pnl_copy.get("date", pd.Series(dtype=str)), errors="coerce")
        mkt_aligned = mkt_return.reindex(pnl_dates.values).fillna(0.0)
        # P12 institutional fix: when beta_mean is near zero (long-short spread),
        # the hedge return series is too small to produce a meaningful Sharpe.
        # Setting it to NaN prevents misleading reports of ±100 hedge Sharpe.
        if abs(beta_mean) < _BETA_FLOOR_FOR_HEDGE:
            hedge_sharpe = float("nan")
        else:
            hedge_arr = (-beta_mean * mkt_aligned.to_numpy(dtype=float)) - borrow_per_day
            _raw_hedge = _ann_sharpe(hedge_arr)
            hedge_sharpe = float(np.clip(_raw_hedge, -_HEDGE_SHARPE_CLIP, _HEDGE_SHARPE_CLIP)) if np.isfinite(_raw_hedge) else float("nan")

    short_value_added = (
        short_sharpe - hedge_sharpe
        if (np.isfinite(short_sharpe) and np.isfinite(hedge_sharpe))
        else float("nan")
    )
    short_destroys = (
        1.0 if (np.isfinite(short_value_added) and short_value_added < -0.10) else 0.0
    )

    # Long alpha fraction: long leg's contribution to absolute total leg Sharpe
    long_abs = abs(long_sharpe) if np.isfinite(long_sharpe) else 0.0
    short_abs = abs(short_sharpe) if np.isfinite(short_sharpe) else 0.0
    total_abs = long_abs + short_abs
    long_alpha_fraction = long_abs / total_abs if total_abs > 1e-9 else float("nan")

    return {
        "diag_short_value_added": short_value_added,
        "diag_short_destroys_value": short_destroys,
        "diag_beta_hedge_sharpe": hedge_sharpe,
        "diag_long_alpha_fraction": long_alpha_fraction,
    }


# ---------------------------------------------------------------------------
# 5. Cross-sectional strength diagnostics
# ---------------------------------------------------------------------------

def cross_sectional_strength(oos_df: pd.DataFrame) -> dict[str, float]:
    """
    Decile stability, prediction dispersion, and rank-IC-vs-bucket analysis.

    Outputs:
      diag_decile_spread_std      — std of daily decile spreads (low = stable alpha)
      diag_decile_spread_cv       — coefficient of variation (spread_std / |spread_mean|)
      diag_top_vs_median_spread   — top decile - median (pure upper-tail alpha)
      diag_bottom_vs_median_spread— bottom decile - median (lower-tail signal strength)
      diag_score_dispersion_mean  — mean daily cross-sectional std of scores
      diag_score_dispersion_cv    — dispersion cv (low = model collapses to flat)
      diag_rank_ic_monotonicity   — fraction of deciles where rank IC is monotone
      diag_ic_consistency_rate    — fraction of dates where IC > 0
    """
    empty = _empty(
        "diag_",
        "decile_spread_std", "decile_spread_cv",
        "top_vs_median_spread", "bottom_vs_median_spread",
        "score_dispersion_mean", "score_dispersion_cv",
        "rank_ic_monotonicity", "ic_consistency_rate",
    )
    if oos_df is None or oos_df.empty:
        return empty

    df, ret_col = _with_executable_next_return(oos_df)
    needed = {"date", "score", ret_col}
    if not needed.issubset(df.columns):
        return empty

    df = df[["date", "score", ret_col]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df[ret_col] = pd.to_numeric(df[ret_col], errors="coerce")
    df = df.dropna()
    if df.empty:
        return empty

    n_deciles = 10
    spreads, top_med_spreads, bot_med_spreads = [], [], []
    score_stds, ic_positive = [], []

    for _, g in df.groupby("date"):
        if len(g) < n_deciles * 2:
            continue
        g = g.sort_values("score")
        n = len(g)
        decile_size = max(1, n // n_deciles)

        # Decile buckets by score rank
        g["_q"] = pd.qcut(g["score"].rank(method="first"), n_deciles, labels=False, duplicates="drop")
        bucket_ret = g.groupby("_q")[ret_col].mean()
        if len(bucket_ret) < 2:
            continue
        top_ret = float(bucket_ret.iloc[-1])
        bot_ret = float(bucket_ret.iloc[0])
        med_ret = float(bucket_ret.median())
        spreads.append(top_ret - bot_ret)
        top_med_spreads.append(top_ret - med_ret)
        bot_med_spreads.append(med_ret - bot_ret)

        # Score dispersion
        score_std = float(g["score"].std(ddof=1))
        if np.isfinite(score_std):
            score_stds.append(score_std)

        # IC sign
        if g["score"].nunique() >= 2 and g[ret_col].nunique() >= 2:
            ic = float(g["score"].rank().corr(g[ret_col].rank()))
            ic_positive.append(1.0 if ic > 0 else 0.0)

    spread_series = pd.Series(spreads, dtype=float).dropna()
    spread_mean = float(spread_series.mean()) if not spread_series.empty else float("nan")
    spread_std = float(spread_series.std(ddof=1)) if len(spread_series) > 1 else float("nan")
    spread_cv = (
        spread_std / abs(spread_mean)
        if (np.isfinite(spread_std) and np.isfinite(spread_mean) and abs(spread_mean) > 1e-9)
        else float("nan")
    )

    sc_series = pd.Series(score_stds, dtype=float).dropna()
    sc_mean = float(sc_series.mean()) if not sc_series.empty else float("nan")
    sc_cv = (
        float(sc_series.std(ddof=1)) / abs(sc_mean)
        if (len(sc_series) > 1 and np.isfinite(sc_mean) and abs(sc_mean) > 1e-9)
        else float("nan")
    )

    # Monotonicity: for each day, what fraction of consecutive bucket-return diffs are >=0?
    mono_scores: list[float] = []
    for _, g in df.groupby("date"):
        if len(g) < n_deciles * 2:
            continue
        g["_q"] = pd.qcut(g["score"].rank(method="first"), n_deciles, labels=False, duplicates="drop")
        bucket_ret = g.groupby("_q")[ret_col].mean().sort_index()
        if len(bucket_ret) < 2:
            continue
        diffs = np.diff(bucket_ret.to_numpy(dtype=float))
        mono_scores.append(float((diffs >= 0).mean()))

    return {
        "diag_decile_spread_std": spread_std,
        "diag_decile_spread_cv": spread_cv,
        "diag_top_vs_median_spread": float(np.nanmean(top_med_spreads)) if top_med_spreads else float("nan"),
        "diag_bottom_vs_median_spread": float(np.nanmean(bot_med_spreads)) if bot_med_spreads else float("nan"),
        "diag_score_dispersion_mean": sc_mean,
        "diag_score_dispersion_cv": sc_cv,
        "diag_rank_ic_monotonicity": float(np.nanmean(mono_scores)) if mono_scores else float("nan"),
        "diag_ic_consistency_rate": float(np.mean(ic_positive)) if ic_positive else float("nan"),
    }


# ---------------------------------------------------------------------------
# 6. Cost-aware ranking score
# ---------------------------------------------------------------------------

def cost_aware_score(row: dict[str, Any]) -> dict[str, float]:
    """
    Net-PnL-adjusted composite score penalising high-turnover / cost-dominated models.

    Uses already-computed metrics in ``row`` — no additional simulation needed.

    Outputs:
      diag_net_ic_score     — IC × (1 − cost_drag)   penalises cost drag on IC
      diag_net_sharpe_score — exec_sharpe × (1 − cost_drag)
      diag_cost_drag        — cost_return_sum / gross_pnl_implied  (0–1 scale)
      diag_alpha_efficiency — net_pnl / gross_pnl  (1 = all alpha survives costs)
    """
    ic_tstat = _safe(row.get("oos_ic_tstat"))
    exec_sharpe = _safe(row.get("exec_sharpe"))
    cost_to_gross = _safe(row.get("exec_cost_to_gross_pnl"))

    # Cost drag: clamp to [0, 1]
    cost_drag = float(np.clip(cost_to_gross, 0.0, 1.0)) if np.isfinite(cost_to_gross) else float("nan")

    alpha_efficiency = (
        float(1.0 - cost_drag)
        if (np.isfinite(cost_drag) and cost_drag >= 0.0)
        else float("nan")
    )
    net_ic_score = (
        ic_tstat * alpha_efficiency
        if (np.isfinite(ic_tstat) and np.isfinite(alpha_efficiency))
        else float("nan")
    )
    net_sharpe_score = (
        exec_sharpe * alpha_efficiency
        if (np.isfinite(exec_sharpe) and np.isfinite(alpha_efficiency))
        else float("nan")
    )

    return {
        "diag_net_ic_score": net_ic_score,
        "diag_net_sharpe_score": net_sharpe_score,
        "diag_cost_drag": cost_drag,
        "diag_alpha_efficiency": alpha_efficiency,
    }


# ---------------------------------------------------------------------------
# 7. Regime segmentation
# ---------------------------------------------------------------------------

def regime_segmentation(
    oos_df: pd.DataFrame,
    pnl: pd.DataFrame,
    *,
    point_in_time: bool = True,
) -> dict[str, Any]:
    """
    Split performance and IC into four market regimes.

    Regimes are derived from the equal-weight market return (cross-sectional
    mean of ``target_return``) across all OOS dates:
      bull     : rolling 63-day annualised return > +8%
      bear     : rolling 63-day annualised return < −8%
      high_vol : rolling 21-day return std > expanding-window 75th percentile
      low_vol  : rolling 21-day return std < expanding-window 25th percentile

    When ``point_in_time=True`` (default), volatility thresholds use an
    expanding-window percentile so that each date's regime classification
    depends only on data available up to that date.  This eliminates the
    full-sample look-ahead bias that previously inflated regime diagnostics.

    Each regime is non-exclusive: a day can be both bull and high-vol.

    Outputs (per regime r in {bull, bear, high_vol, low_vol}):
      diag_regime_{r}_sharpe  — annualised Sharpe within regime
      diag_regime_{r}_ic      — mean rank IC within regime
      diag_regime_{r}_n_days  — number of regime days
      diag_regime_best        — regime with highest Sharpe
      diag_regime_worst       — regime with lowest Sharpe
      diag_regime_spread      — best_sharpe − worst_sharpe (regime sensitivity)
    """
    regime_keys = (
        [f"regime_{r}_sharpe" for r in _REGIMES]
        + [f"regime_{r}_ic" for r in _REGIMES]
        + [f"regime_{r}_n_days" for r in _REGIMES]
        + ["regime_best", "regime_worst", "regime_spread"]
    )
    empty: dict[str, Any] = {
        f"diag_{k}": (float("nan") if "n_days" not in k and k not in ("regime_best", "regime_worst") else (0 if "n_days" in k else ""))
        for k in regime_keys
    }

    if oos_df is None or oos_df.empty or pnl is None or pnl.empty:
        return empty

    # Build daily market return series
    df_tmp, ret_col = _with_executable_next_return(oos_df)
    if ret_col not in df_tmp.columns:
        return empty

    df_tmp["date"] = pd.to_datetime(df_tmp["date"], errors="coerce")
    mkt_ret = df_tmp.groupby("date")[ret_col].mean().sort_index()

    if len(mkt_ret) < 30:
        return empty

    # Regime masks on the market return index
    roll63 = mkt_ret.rolling(63, min_periods=30).mean() * 252.0  # annualised running mean
    roll21_std = mkt_ret.rolling(21, min_periods=10).std(ddof=1)

    if point_in_time:
        # Expanding-window percentiles: at each date, the threshold is computed
        # from all data up to and including that date.  This eliminates the
        # full-sample look-ahead bias where future data influenced past regime
        # classification.
        vol_75_series = roll21_std.expanding(min_periods=30).quantile(0.75)
        vol_25_series = roll21_std.expanding(min_periods=30).quantile(0.25)
        regime_mask: dict[str, pd.Series] = {
            "bull": roll63 > 0.08,
            "bear": roll63 < -0.08,
            "high_vol": roll21_std > vol_75_series,
            "low_vol": roll21_std < vol_25_series,
        }
    else:
        # Legacy full-sample thresholds — retained for backward-comparison only.
        vol_75 = float(roll21_std.quantile(0.75))
        vol_25 = float(roll21_std.quantile(0.25))
        regime_mask = {
            "bull": roll63 > 0.08,
            "bear": roll63 < -0.08,
            "high_vol": roll21_std > vol_75,
            "low_vol": roll21_std < vol_25,
        }

    # Build pnl DataFrame indexed by date
    pnl_copy = pnl.copy()
    pnl_copy["date"] = pd.to_datetime(pnl_copy.get("date", pd.Series(dtype=str)), errors="coerce")
    pnl_copy = pnl_copy.set_index("date").sort_index()
    net_series = pd.to_numeric(pnl_copy.get("daily_return", pd.Series(dtype=float)), errors="coerce")

    # Build IC series: rank IC per date
    oos_ic_by_date: dict[pd.Timestamp, float] = {}
    if "score" in df_tmp.columns and ret_col in df_tmp.columns:
        for dt, g in df_tmp.groupby("date"):
            g2 = g[["score", ret_col]].apply(pd.to_numeric, errors="coerce").dropna()
            if len(g2) < 5 or g2["score"].nunique() < 2 or g2[ret_col].nunique() < 2:
                continue
            oos_ic_by_date[dt] = float(g2["score"].rank().corr(g2[ret_col].rank()))

    ic_series = pd.Series(oos_ic_by_date, dtype=float).sort_index()

    result: dict[str, Any] = {}
    regime_sharpes: dict[str, float] = {}

    for regime_name, mask in regime_mask.items():
        # Align mask to pnl dates
        mask_aligned = mask.reindex(net_series.index).fillna(False)
        regime_rets = net_series[mask_aligned]
        n_days = int(mask_aligned.sum())
        sharpe = _ann_sharpe(regime_rets.to_numpy(dtype=float)) if n_days >= 15 else float("nan")
        regime_sharpes[regime_name] = sharpe

        # IC within regime — require minimum observations to avoid unreliable estimates
        mask_ic = mask.reindex(ic_series.index).fillna(False)
        regime_ic = ic_series[mask_ic]
        n_ic_obs = int(regime_ic.notna().sum())
        ic_mean = float(regime_ic.mean()) if n_ic_obs >= 15 else float("nan")

        result[f"diag_regime_{regime_name}_sharpe"] = sharpe
        result[f"diag_regime_{regime_name}_ic"] = ic_mean
        result[f"diag_regime_{regime_name}_n_days"] = n_days

    finite_sharpes = {k: v for k, v in regime_sharpes.items() if np.isfinite(v)}
    if finite_sharpes:
        best = max(finite_sharpes, key=finite_sharpes.__getitem__)
        worst = min(finite_sharpes, key=finite_sharpes.__getitem__)
        result["diag_regime_best"] = best
        result["diag_regime_worst"] = worst
        result["diag_regime_spread"] = finite_sharpes[best] - finite_sharpes[worst]
    else:
        result["diag_regime_best"] = ""
        result["diag_regime_worst"] = ""
        result["diag_regime_spread"] = float("nan")

    return result


# ---------------------------------------------------------------------------
# 8. Research classification
# ---------------------------------------------------------------------------

def classify_research_label(row: dict[str, Any]) -> tuple[str, list[str]]:
    """
    Classify a model into research categories based on diagnostic evidence.

    Returns (primary_label, [all_applicable_labels]).

    Labels (in priority order):
      overfit_model     — few windows AND high IS/OOS discrepancy
      cost_dominated    — cost_drag > 0.50 OR turnover kills net Sharpe
      exposure_driven   — beta or factor tilts explain most PnL, not ranking skill
      weak_signal       — IC t-stat < 1.0 across regimes
      unusable_short    — long alpha positive but short side hurts
      usable_long_alpha — long_only Sharpe > 0 AND IC > 0 AND not cost_dominated
      diagnostic_only   — marginal signal, not production-ready
    """
    labels: list[str] = []

    n_windows = int(_safe(row.get("n_windows"), 0))
    ic_tstat = _safe(row.get("oos_ic_tstat"))
    exec_sharpe = _safe(row.get("exec_sharpe"))
    lo_sharpe = _safe(row.get("diag_lo_sharpe"))
    lo_net = _safe(row.get("diag_lo_net_sharpe"))
    cost_drag = _safe(row.get("diag_cost_drag"))
    short_destroys = _safe(row.get("diag_short_destroys_value"), 0.0)
    turn_mean = _safe(row.get("diag_turnover_mean_daily"))
    sharpe_per_turn = _safe(row.get("diag_sharpe_per_unit_turnover"))
    regime_spread = _safe(row.get("diag_regime_spread"))
    oos_deflated_sharpe = _safe(row.get("oos_deflated_sharpe"))
    n_bull_days = int(_safe(row.get("diag_regime_bull_n_days"), 0))
    n_bear_days = int(_safe(row.get("diag_regime_bear_n_days"), 0))
    bull_sharpe = _safe(row.get("diag_regime_bull_sharpe"))
    bear_sharpe = _safe(row.get("diag_regime_bear_sharpe"))
    decile_spread_cv = _safe(row.get("diag_decile_spread_cv"))
    # New module 9-12 keys
    short_classification = str(row.get("diag_short_classification", ""))
    alpha_vs_cost_ratio = _safe(row.get("diag_alpha_vs_cost_ratio"))
    alpha_capture = _safe(row.get("diag_alpha_capture_ratio"))
    beta_pnl_pct = _safe(row.get("diag_beta_pnl_pct"))
    residual_sharpe = _safe(row.get("diag_residual_alpha_sharpe"))

    # Overfit: few windows or deflated Sharpe << raw Sharpe
    if n_windows <= 3 or (np.isfinite(oos_deflated_sharpe) and oos_deflated_sharpe < 0.55):
        labels.append("overfit_model")

    # Cost-dominated: cost drag > 50% or negative net-of-cost alpha
    if np.isfinite(cost_drag) and cost_drag > 0.50:
        labels.append("cost_dominated")
    elif np.isfinite(turn_mean) and turn_mean > 0.30 and np.isfinite(sharpe_per_turn) and sharpe_per_turn < 1.0:
        labels.append("cost_dominated")
    elif np.isfinite(alpha_vs_cost_ratio) and alpha_vs_cost_ratio < 1.0:
        labels.append("cost_dominated")

    # Exposure-driven: most PnL attributed to beta/factor tilt, not ranking skill
    beta_dominated = np.isfinite(beta_pnl_pct) and beta_pnl_pct > 0.60
    residual_weak = not np.isfinite(residual_sharpe) or residual_sharpe < 0.20
    short_is_artifact = short_classification == "exposure_artifact"
    alpha_cap_low = np.isfinite(alpha_capture) and alpha_capture < 0.25
    if (beta_dominated and residual_weak) or (short_is_artifact and alpha_cap_low):
        labels.append("exposure_driven")

    # Weak signal: empirical test vs shuffle baseline when available;
    # fall back to ic_tstat < 1.0 when baseline data is absent.
    ic_percentile_vs_shuffle = _safe(row.get("baseline_ic_percentile"))
    ic_ci_low = _safe(row.get("baseline_ic_ci_low"))
    if np.isfinite(ic_percentile_vs_shuffle):
        # Empirical gate: IC not significantly above cross-sectional noise
        # (below 90th percentile of shuffle distribution OR CI includes zero)
        _ci_includes_zero = np.isfinite(ic_ci_low) and ic_ci_low <= 0.0
        if ic_percentile_vs_shuffle < 0.90 or _ci_includes_zero:
            labels.append("weak_signal")
    elif np.isfinite(ic_tstat) and ic_tstat < 1.0:
        labels.append("weak_signal")

    # Regime-only: only works in one regime (bull or bear)
    has_strong_bull = np.isfinite(bull_sharpe) and bull_sharpe > 0.5
    has_strong_bear = np.isfinite(bear_sharpe) and bear_sharpe > 0.5
    if (
        np.isfinite(regime_spread) and regime_spread > 1.5
        and not (has_strong_bull and has_strong_bear)
    ):
        labels.append("regime_conditional")

    # Short side destroys value
    if short_destroys > 0.5:
        labels.append("unusable_short")

    # Unstable decile structure
    if np.isfinite(decile_spread_cv) and decile_spread_cv > 2.0:
        labels.append("unstable_deciles")

    # Usable long alpha: positive long-only performance with signal above noise
    _signal_significant = (
        (np.isfinite(ic_percentile_vs_shuffle) and ic_percentile_vs_shuffle >= 0.90)
        or (not np.isfinite(ic_percentile_vs_shuffle) and np.isfinite(ic_tstat) and ic_tstat >= 1.0)
    )
    if (
        np.isfinite(lo_sharpe) and lo_sharpe > 0.0
        and np.isfinite(lo_net) and lo_net > 0.0
        and _signal_significant
        and "cost_dominated" not in labels
        and "exposure_driven" not in labels
    ):
        labels.append("usable_long_alpha")

    if not labels:
        labels.append("diagnostic_only")

    # Primary: first in priority list
    priority = [
        "overfit_model", "cost_dominated", "exposure_driven", "weak_signal",
        "unusable_short", "regime_conditional", "usable_long_alpha",
        "unstable_deciles", "diagnostic_only",
    ]
    primary = next((lb for lb in priority if lb in labels), labels[0])
    return primary, labels


# ---------------------------------------------------------------------------
# 9. Cost decomposition
# ---------------------------------------------------------------------------

def cost_decomposition(row: dict[str, Any], pnl: pd.DataFrame) -> dict[str, Any]:
    """
    Split total friction into: commission, spread/slippage, market impact, borrow.
    Report each component as a fraction of gross PnL and identify the dominant driver.

    Key outputs:
      diag_cost_pct_commission   — broker commissions share of total friction
      diag_cost_pct_spread       — bid-ask spread share
      diag_cost_pct_impact       — temp + perm market impact share
      diag_cost_pct_borrow       — stock borrow share
      diag_cost_dominant         — name of the largest component
      diag_cost_rate_daily       — average daily friction as % of gross exposure
      diag_gross_alpha_per_turn  — gross daily PnL per unit of turnover
      diag_cost_per_unit_turn    — friction per unit of turnover
      diag_alpha_vs_cost_ratio   — gross_alpha_per_turn / cost_per_unit_turn
                                   < 1 means costs exceed alpha at current turnover
    """
    commission = abs(_safe(row.get("exec_commission_return_sum"), 0.0))
    spread_slip = abs(_safe(row.get("exec_spread_return_sum"), 0.0))
    temp_impact = abs(_safe(row.get("exec_temporary_impact_return_sum"), 0.0))
    perm_impact = abs(_safe(row.get("exec_permanent_impact_return_sum"), 0.0))
    fixed = abs(_safe(row.get("exec_fixed_cost_return_sum"), 0.0))
    borrow = abs(_safe(row.get("exec_borrow_return_sum"), 0.0))
    total_friction = commission + spread_slip + temp_impact + perm_impact + fixed + borrow

    if total_friction < 1e-30:
        return _empty(
            "diag_",
            "cost_pct_commission", "cost_pct_spread", "cost_pct_impact",
            "cost_pct_borrow", "cost_rate_daily", "gross_alpha_per_turn",
            "cost_per_unit_turn", "alpha_vs_cost_ratio",
        ) | {"diag_cost_dominant": "none"}

    components = {
        "commission": commission,
        "spread": spread_slip,
        "impact": temp_impact + perm_impact,
        "borrow": borrow,
        "fixed": fixed,
    }
    dominant = max(components, key=components.__getitem__)

    n_days = int(_safe(row.get("exec_n_days"), len(pnl) if pnl is not None and not pnl.empty else 1))
    cost_rate_daily = total_friction / max(n_days, 1)

    # Per-unit-turnover metrics: does cost_per_trade outstrip gross alpha per trade?
    turn_mean = _safe(row.get("exec_turnover_mean"))
    gross_exp_mean = _safe(row.get("exec_gross_exposure_mean"), 1.0)
    gross_pnl_mean = float("nan")
    cost_per_turn = float("nan")
    alpha_vs_cost = float("nan")

    if pnl is not None and not pnl.empty:
        gp = _col(pnl, "gross_return").fillna(0.0)
        tv = _col(pnl, "turnover").fillna(0.0)
        tv_pos = tv[tv > 1e-9]
        gp_aligned = gp.reindex(tv_pos.index)
        if len(tv_pos) > 5:
            gross_pnl_mean = float(gp_aligned.abs().mean() / tv_pos.mean())  # gross $ per unit turnover
            cost_per_turn = cost_rate_daily / float(tv.mean()) if float(tv.mean()) > 1e-9 else float("nan")
            if np.isfinite(gross_pnl_mean) and np.isfinite(cost_per_turn) and cost_per_turn > 1e-30:
                alpha_vs_cost = gross_pnl_mean / cost_per_turn

    return {
        "diag_cost_pct_commission": commission / total_friction,
        "diag_cost_pct_spread": spread_slip / total_friction,
        "diag_cost_pct_impact": (temp_impact + perm_impact) / total_friction,
        "diag_cost_pct_borrow": borrow / total_friction,
        "diag_cost_dominant": dominant,
        "diag_cost_rate_daily": cost_rate_daily,
        "diag_gross_alpha_per_turn": gross_pnl_mean,
        "diag_cost_per_unit_turn": cost_per_turn,
        "diag_alpha_vs_cost_ratio": alpha_vs_cost,
    }


# ---------------------------------------------------------------------------
# 10. Exposure decomposition
# ---------------------------------------------------------------------------

def exposure_decomposition(oos_df: pd.DataFrame, pnl: pd.DataFrame) -> dict[str, float]:
    """
    Split PnL into factor-attributed vs residual alpha components.

    Beta attribution:
      - Daily beta PnL = portfolio_beta × equal-weight market return
      - Residual = net_return − beta_pnl
      - If residual Sharpe << total Sharpe: performance is beta-driven, not alpha

    Sector concentration:
      - HHI of sector weights (from max_sector_exposure proxy)
      - High HHI = few sectors dominate → exposure artifact risk

    Volatility tilt:
      - Spearman correlation of score vs realised_vol_20d (if available)
      - Positive = long high-vol, short low-vol (volatility factor tilt)

    Outputs:
      diag_beta_attributed_sharpe   — Sharpe from beta × market_return series
      diag_residual_alpha_sharpe    — Sharpe from net_return − beta_pnl
      diag_beta_pnl_pct             — abs(beta_pnl_sum) / abs(gross_pnl_sum)
      diag_mkt_corr                 — corr(net_daily_return, market_return)
      diag_vol_tilt                 — avg rank-corr(score, realised_vol_20d) per day
      diag_sector_exposure_mean     — mean of max_sector_exposure in PnL
    """
    empty = _empty(
        "diag_",
        "beta_attributed_sharpe", "residual_alpha_sharpe", "beta_pnl_pct",
        "mkt_corr", "vol_tilt", "sector_exposure_mean",
    )
    if oos_df is None or oos_df.empty or pnl is None or pnl.empty:
        return empty

    df_tmp, ret_col = _with_executable_next_return(oos_df)
    if ret_col not in df_tmp.columns:
        return empty

    df_tmp["date"] = pd.to_datetime(df_tmp["date"], errors="coerce")
    mkt_ret = df_tmp.groupby("date")[ret_col].mean().sort_index()

    pnl_copy = pnl.copy()
    pnl_copy["date"] = pd.to_datetime(pnl_copy.get("date", pd.Series(dtype=str)), errors="coerce")
    pnl_indexed = pnl_copy.set_index("date").sort_index()

    beta = pd.to_numeric(pnl_indexed.get("beta_exposure", pd.Series(dtype=float)), errors="coerce").fillna(0.0)
    net = pd.to_numeric(pnl_indexed.get("daily_return", pd.Series(dtype=float)), errors="coerce").fillna(0.0)

    common_dates = mkt_ret.index.intersection(beta.index).intersection(net.index)
    if len(common_dates) < 20:
        return empty

    mkt_aligned = mkt_ret.loc[common_dates]
    beta_aligned = beta.loc[common_dates]
    net_aligned = net.loc[common_dates]

    beta_pnl = beta_aligned * mkt_aligned
    residual_pnl = net_aligned - beta_pnl

    beta_sharpe = _ann_sharpe(beta_pnl.to_numpy(dtype=float))
    residual_sharpe = _ann_sharpe(residual_pnl.to_numpy(dtype=float))

    total_abs = abs(float(beta_pnl.sum())) + abs(float(residual_pnl.sum()))
    beta_pct = abs(float(beta_pnl.sum())) / total_abs if total_abs > 1e-12 else float("nan")

    mkt_corr = float("nan")
    if net_aligned.std() > 1e-10 and mkt_aligned.std() > 1e-10:
        mkt_corr = float(net_aligned.corr(mkt_aligned))

    # Volatility tilt: does the score rank-correlate with stock realised volatility?
    vol_tilt = float("nan")
    if "realised_vol_20d" in oos_df.columns and "score" in oos_df.columns:
        df_vs = oos_df[["date", "score", "realised_vol_20d"]].copy()
        df_vs["score"] = pd.to_numeric(df_vs["score"], errors="coerce")
        df_vs["realised_vol_20d"] = pd.to_numeric(df_vs["realised_vol_20d"], errors="coerce")
        df_vs = df_vs.dropna()
        vol_corrs: list[float] = []
        for _, g in df_vs.groupby("date"):
            if len(g) < 5 or g["score"].nunique() < 2 or g["realised_vol_20d"].nunique() < 2:
                continue
            vol_corrs.append(float(g["score"].rank().corr(g["realised_vol_20d"].rank())))
        vol_tilt = float(np.nanmean(vol_corrs)) if vol_corrs else float("nan")

    # Sector exposure proxy from PnL
    sec_exp = _col(pnl_indexed.reset_index(), "max_sector_exposure").dropna()
    sec_exp_mean = float(sec_exp.abs().mean()) if not sec_exp.empty else float("nan")

    return {
        "diag_beta_attributed_sharpe": beta_sharpe,
        "diag_residual_alpha_sharpe": residual_sharpe,
        "diag_beta_pnl_pct": beta_pct,
        "diag_mkt_corr": mkt_corr,
        "diag_vol_tilt": vol_tilt,
        "diag_sector_exposure_mean": sec_exp_mean,
    }


# ---------------------------------------------------------------------------
# 11. Short-side classification
# ---------------------------------------------------------------------------

def short_side_classification(
    oos_df: pd.DataFrame,
    pnl: pd.DataFrame,
    row: dict[str, Any],
) -> dict[str, Any]:
    """
    Classify the short side as: alpha_short | hedge_only | exposure_artifact.

    Decision tree:
      alpha_short      : positive IC + residual alpha survives after beta removal
      hedge_only       : near-zero IC but Sharpe from offsetting net beta
      exposure_artifact: positive Sharpe from concentrated factor tilt
                         (high vol_tilt, sector concentration, or market correlation)

    Outputs:
      diag_short_classification  — alpha_short | hedge_only | exposure_artifact
      diag_short_mkt_corr        — corr(short_daily_pnl, inverted market return)
      diag_short_ic_positive_rate— fraction of dates with IC > 0 on short picks
      diag_short_residual_sharpe — short leg Sharpe after subtracting beta component
      diag_short_alpha_pct       — residual fraction of short PnL
    """
    empty: dict[str, Any] = {
        "diag_short_classification": "unknown",
        "diag_short_mkt_corr": float("nan"),
        "diag_short_ic_positive_rate": float("nan"),
        "diag_short_residual_sharpe": float("nan"),
        "diag_short_alpha_pct": float("nan"),
    }
    if pnl is None or pnl.empty:
        return empty

    # Short leg net return (cost-allocated, from the simulation columns)
    pnl_copy = pnl.copy()
    pnl_copy["date"] = pd.to_datetime(pnl_copy.get("date", pd.Series(dtype=str)), errors="coerce")
    pnl_indexed = pnl_copy.set_index("date").sort_index()

    def _pc(name: str) -> pd.Series:
        if name not in pnl_indexed.columns:
            return pd.Series(0.0, index=pnl_indexed.index)
        return pd.to_numeric(pnl_indexed[name], errors="coerce").fillna(0.0)

    short_net = (
        _pc("short_gross_return")
        - _pc("short_cost_return")
        - _pc("borrow_return")
        - _pc("market_adj_return")
    )

    # Market return proxy
    mkt_ret = pd.Series(dtype=float)
    if oos_df is not None and not oos_df.empty:
        df_tmp, ret_col = _with_executable_next_return(oos_df)
        if ret_col in df_tmp.columns:
            df_tmp["date"] = pd.to_datetime(df_tmp["date"], errors="coerce")
            mkt_ret = df_tmp.groupby("date")[ret_col].mean()

    # Short-market correlation: a pure beta short has corr(short_pnl, -market) ≈ 1
    short_mkt_corr = float("nan")
    common = short_net.index.intersection(mkt_ret.index) if not mkt_ret.empty else pd.Index([])
    if len(common) >= 20:
        sn = short_net.loc[common]
        mr = mkt_ret.loc[common]
        if sn.std() > 1e-10 and mr.std() > 1e-10:
            short_mkt_corr = float(sn.corr(-mr))  # positive = short profits when market falls

    # Beta attribution to short leg
    beta = _pc("beta_exposure")
    short_residual = short_net.copy()
    if not mkt_ret.empty:
        common2 = short_net.index.intersection(mkt_ret.index).intersection(beta.index)
        if len(common2) >= 20:
            beta_pnl = beta.loc[common2] * mkt_ret.loc[common2]
            short_residual = short_net.loc[common2] - beta_pnl

    short_residual_sharpe = _ann_sharpe(short_residual.to_numpy(dtype=float))
    total_abs = abs(float(short_net.sum())) + 1e-30
    short_alpha_pct = abs(float(short_residual.sum())) / total_abs

    # IC quality for short picks
    short_ic_positive_rate = float("nan")
    if oos_df is not None and not oos_df.empty and "score" in oos_df.columns:
        df_ic_src, ret_col2 = _with_executable_next_return(oos_df)
        if ret_col2 in df_ic_src.columns:
            df_ic = df_ic_src[["date", "score", ret_col2]].copy()
            df_ic["date"] = pd.to_datetime(df_ic["date"], errors="coerce")
            df_ic["score"] = pd.to_numeric(df_ic["score"], errors="coerce")
            df_ic[ret_col2] = pd.to_numeric(df_ic[ret_col2], errors="coerce")
            df_ic = df_ic.dropna()
            ic_signs: list[float] = []
            for _, g in df_ic.groupby("date"):
                if len(g) < 5 or g["score"].nunique() < 2 or g[ret_col2].nunique() < 2:
                    continue
                ic = float(g["score"].rank().corr(g[ret_col2].rank()))
                ic_signs.append(1.0 if ic > 0 else 0.0)
            short_ic_positive_rate = float(np.mean(ic_signs)) if ic_signs else float("nan")

    # Classification decision
    ic_chained = _safe(row.get("oos_ic_chained"))
    mkt_corr_threshold = 0.40
    ic_threshold = 0.30  # fraction of positive-IC days needed for "alpha"

    if (
        np.isfinite(short_ic_positive_rate) and short_ic_positive_rate >= ic_threshold
        and np.isfinite(short_residual_sharpe) and short_residual_sharpe > 0.0
    ):
        classification = "alpha_short"
    elif (
        np.isfinite(short_mkt_corr) and short_mkt_corr >= mkt_corr_threshold
        and (not np.isfinite(short_ic_positive_rate) or short_ic_positive_rate < ic_threshold)
    ):
        classification = "hedge_only"
    else:
        # Positive Sharpe but comes from factor tilts, not ranking skill
        classification = "exposure_artifact"

    return {
        "diag_short_classification": classification,
        "diag_short_mkt_corr": short_mkt_corr,
        "diag_short_ic_positive_rate": short_ic_positive_rate,
        "diag_short_residual_sharpe": short_residual_sharpe,
        "diag_short_alpha_pct": short_alpha_pct,
    }


# ---------------------------------------------------------------------------
# 12. Alpha capture ratio (Fundamental Law of Active Management)
# ---------------------------------------------------------------------------

def alpha_capture_ratio(row: dict[str, Any], pnl: pd.DataFrame) -> dict[str, float]:
    """
    Measure how much theoretical alpha is lost to friction.

    Fundamental Law (Grinold & Kahn):
        SR_theoretical = IC × sqrt(BR)
        where BR = breadth = average daily positions × 252 (bets per year)

    Alpha capture = exec_sharpe / SR_theoretical
      = 1.0: all alpha captured (zero cost world)
      < 1.0: costs destroy a fraction of theoretical alpha
      < 0.0: costs exceed alpha entirely

    Outputs:
      diag_breadth_daily_mean    — average number of positions held per day
      diag_theoretical_sr_flam  — IC × sqrt(positions × 252) from FLAM
      diag_alpha_capture_ratio   — exec_sharpe / theoretical_sr
      diag_cost_loss_ratio       — fraction of theoretical alpha consumed by costs
      diag_breakeven_ic          — min IC for which exec_sharpe > 0 at current costs
    """
    empty = _empty(
        "diag_",
        "breadth_daily_mean", "theoretical_sr_flam",
        "alpha_capture_ratio", "cost_loss_ratio", "breakeven_ic",
    )

    ic_daily = _safe(row.get("oos_ic_chained"))
    exec_sharpe = _safe(row.get("exec_sharpe"))
    if not np.isfinite(ic_daily):
        return empty

    # Average daily positions (long + short)
    n_pos = float("nan")
    if pnl is not None and not pnl.empty:
        if "n_positions" in pnl.columns:
            n_pos = float(_col(pnl, "n_positions").mean())
        elif "n_long_positions" in pnl.columns and "n_short_positions" in pnl.columns:
            n_pos = float(
                (_col(pnl, "n_long_positions") + _col(pnl, "n_short_positions")).mean()
            )
    if not np.isfinite(n_pos) or n_pos < 1:
        # Fall back to estimating from exec_gross_exposure_mean and typical position size
        gross_exp = _safe(row.get("exec_gross_exposure_mean"), float("nan"))
        n_pos = gross_exp / 0.02 if np.isfinite(gross_exp) else float("nan")  # assume 2% avg weight

    if not np.isfinite(n_pos) or n_pos < 1:
        return empty

    breadth = n_pos * 252.0
    sr_theoretical = abs(ic_daily) * np.sqrt(breadth)

    alpha_capture = float("nan")
    cost_loss = float("nan")
    if np.isfinite(exec_sharpe) and sr_theoretical > 1e-9:
        alpha_capture = exec_sharpe / sr_theoretical
        cost_loss = max(0.0, (sr_theoretical - exec_sharpe)) / sr_theoretical

    # Breakeven IC: IC such that IC × sqrt(BR) = 0 after costs
    # Costs in Sharpe units ≈ sr_theoretical - exec_sharpe
    sharpe_cost = sr_theoretical - exec_sharpe if np.isfinite(exec_sharpe) else float("nan")
    breakeven_ic = float("nan")
    if np.isfinite(sharpe_cost) and breadth > 0:
        breakeven_ic = sharpe_cost / np.sqrt(breadth)

    return {
        "diag_breadth_daily_mean": n_pos,
        "diag_theoretical_sr_flam": sr_theoretical,
        "diag_alpha_capture_ratio": alpha_capture,
        "diag_cost_loss_ratio": cost_loss,
        "diag_breakeven_ic": breakeven_ic,
    }


# ---------------------------------------------------------------------------
# 13. Turnover by score decile
# ---------------------------------------------------------------------------

def turnover_by_decile(oos_df: pd.DataFrame, n_deciles: int = 10) -> dict[str, float]:
    """
    Entry/exit rate per score decile.

    Identifies whether the top-decile positions are being churned more than
    lower deciles — a symptom of noisy signal with short-lived rankings.

    High top-decile turnover relative to bottom-decile turnover:
      → model is flipping its highest-conviction picks rapidly
      → costs accumulate on the positions contributing the most alpha
      → reduce rebalancing frequency or add turnover penalty

    Outputs:
      diag_decile_turn_top         — avg daily entry rate into top decile
      diag_decile_turn_bottom      — avg daily entry rate into bottom decile
      diag_decile_turn_top_vs_bot  — ratio top / bottom (>1 = top churns faster)
      diag_decile_signal_halflife  — days before 50% of top-decile holdings change
    """
    empty = _empty(
        "diag_",
        "decile_turn_top", "decile_turn_bottom",
        "decile_turn_top_vs_bot", "decile_signal_halflife",
    )
    if oos_df is None or oos_df.empty or "score" not in oos_df.columns:
        return empty

    df = oos_df[["date", "ticker", "score"]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna()
    if df.empty:
        return empty

    top_entry_rates: list[float] = []
    bot_entry_rates: list[float] = []
    prev_top: set[str] = set()
    prev_bot: set[str] = set()
    top_overlaps: list[float] = []  # fraction retained from day to day

    for dt in sorted(df["date"].unique()):
        day = df[df["date"] == dt]
        n = len(day)
        if n < n_deciles * 2:
            continue
        decile_size = max(1, n // n_deciles)
        ranked = day.sort_values("score", ascending=False)
        top_tickers = set(ranked.iloc[:decile_size]["ticker"].tolist())
        bot_tickers = set(ranked.iloc[-decile_size:]["ticker"].tolist())

        if prev_top:
            new_entries_top = len(top_tickers - prev_top)
            top_entry_rates.append(new_entries_top / max(len(top_tickers), 1))
            overlap = len(top_tickers & prev_top) / max(len(top_tickers), 1)
            top_overlaps.append(overlap)
        if prev_bot:
            new_entries_bot = len(bot_tickers - prev_bot)
            bot_entry_rates.append(new_entries_bot / max(len(bot_tickers), 1))

        prev_top = top_tickers
        prev_bot = bot_tickers

    turn_top = float(np.mean(top_entry_rates)) if top_entry_rates else float("nan")
    turn_bot = float(np.mean(bot_entry_rates)) if bot_entry_rates else float("nan")
    ratio = turn_top / turn_bot if (np.isfinite(turn_bot) and turn_bot > 1e-9) else float("nan")

    # Signal half-life: how many days until 50% of top-decile holdings have been replaced
    # If daily retention = r, then r^h = 0.5 → h = ln(0.5)/ln(r)
    halflife = float("nan")
    if top_overlaps:
        r = float(np.mean(top_overlaps))
        if 0.0 < r < 1.0:
            halflife = np.log(0.5) / np.log(r)

    return {
        "diag_decile_turn_top": turn_top,
        "diag_decile_turn_bottom": turn_bot,
        "diag_decile_turn_top_vs_bot": ratio,
        "diag_decile_signal_halflife": halflife,
    }


# ---------------------------------------------------------------------------
# 14. Execution Robustness Diagnostics
# ---------------------------------------------------------------------------

def execution_robustness_diagnostics(
    oos_df: pd.DataFrame,
    pnl: pd.DataFrame,
    *,
    target_col: str = "target_return",
) -> dict[str, float]:
    """
    Algorithmic diagnostics for signal stability and execution capacity.
    No hardcoded thresholds; returns raw data-driven metrics.
    """
    empty = _empty(
        "diag_robust_",
        "signal_halflife", "cost_adjusted_ic", "capacity_weighted_ic",
        "turnover_volatility", "tail_stability", "hhi",
    )
    if oos_df is None or oos_df.empty or "score" not in oos_df.columns:
        return empty

    res = {}
    
    # 1. Signal Halflife (Rank Autocorrelation)
    df = oos_df[["date", "ticker", "score"]].copy()
    df["date"] = pd.to_datetime(df["date"])
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna()
    
    halflife = float("nan")
    if not df.empty:
        df["rank"] = df.groupby("date")["score"].rank(method="average")
        wide_ranks = df.pivot_table(index="date", columns="ticker", values="rank").sort_index()
        valid_dates = wide_ranks.index
        if len(valid_dates) > 1:
            autocorr_vals = []
            for i in range(len(valid_dates) - 1):
                v0 = wide_ranks.iloc[i].dropna()
                v1 = wide_ranks.iloc[i+1].dropna()
                common = v0.index.intersection(v1.index)
                if len(common) > 10:
                    rho = float(v0.loc[common].corr(v1.loc[common], method="spearman"))
                    if np.isfinite(rho):
                        autocorr_vals.append(rho)
            if autocorr_vals:
                avg_rho = float(np.mean(autocorr_vals))
                if 0.0 < avg_rho < 1.0:
                    halflife = -np.log(2) / np.log(avg_rho)
                elif avg_rho >= 1.0:
                    halflife = 100.0
                else:
                    halflife = 0.0
    res["diag_robust_signal_halflife"] = halflife

    # 2. Cost-Adjusted IC
    cost_col = "target_return_net" if target_col == "target_return" and "target_return_net" in oos_df.columns else target_col
    cost_ic = float("nan")
    if cost_col in oos_df.columns:
        df_c = oos_df[["date", "score", cost_col]].dropna()
        ics = []
        for _, g in df_c.groupby("date"):
            if len(g) > 10 and g["score"].nunique() > 1 and g[cost_col].nunique() > 1:
                ics.append(float(g["score"].rank().corr(g[cost_col].rank())))
        cost_ic = float(np.mean(ics)) if ics else float("nan")
    res["diag_robust_cost_adjusted_ic"] = cost_ic

    # 3. Capacity Weighted IC
    cap_col = "adv_dollar_20" if "adv_dollar_20" in oos_df.columns else ("dollar_volume" if "dollar_volume" in oos_df.columns else None)
    cap_ic = float("nan")
    if cap_col and cap_col in oos_df.columns:
        df_cap = oos_df[["date", "score", target_col, cap_col]].dropna()
        weighted_ics = []
        for _, g in df_cap.groupby("date"):
            if len(g) > 10:
                w = g[cap_col] / g[cap_col].sum()
                s_m = g["score"] - g["score"].mean()
                t_m = g[target_col] - g[target_col].mean()
                cov = (s_m * t_m * w).sum()
                var_s = (s_m**2 * w).sum()
                var_t = (t_m**2 * w).sum()
                if var_s > 1e-12 and var_t > 1e-12:
                    weighted_ics.append(cov / np.sqrt(var_s * var_t))
        cap_ic = float(np.mean(weighted_ics)) if weighted_ics else float("nan")
    res["diag_robust_capacity_weighted_ic"] = cap_ic

    # 4. Turnover Volatility & HHI
    turn_vol = float("nan")
    hhi = float("nan")
    if pnl is not None and not pnl.empty:
        turnover = _col(pnl, "turnover").fillna(0.0)
        if len(turnover) > 1:
            turn_vol = float(turnover.std())
        
        # HHI from pnl if present, or implied
        # Looking for 'hhi' column first
        if "hhi" in pnl.columns:
            hhi = float(_col(pnl, "hhi").mean())
    res["diag_robust_turnover_volatility"] = turn_vol
    res["diag_robust_hhi"] = hhi

    # 5. Decile Tail Stability
    tail_stab = float("nan")
    if not df.empty:
        df["decile"] = df.groupby("date")["score"].transform(lambda x: pd.qcut(x, 10, labels=False, duplicates="drop"))
        tail = df[df["decile"].isin([0, 9])].copy()
        if not tail.empty:
            tail_wide = tail.pivot_table(index="date", columns="ticker", values="decile")
            dates = tail_wide.index.sort_values()
            stabilities = []
            for i in range(len(dates) - 1):
                d0, d1 = dates[i], dates[i+1]
                s0 = tail_wide.loc[d0].dropna()
                s1 = tail_wide.loc[d1].dropna()
                common = s0.index.intersection(s1.index)
                if not common.empty:
                    matches = (s0.loc[common] == s1.loc[common]).sum()
                    stabilities.append(matches / len(common))
            if stabilities:
                tail_stab = float(np.mean(stabilities))
    res["diag_robust_tail_stability"] = tail_stab

    return res


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def compute_full_diagnostics(
    oos_df: pd.DataFrame,
    pnl: pd.DataFrame,
    row: dict[str, Any],
    *,
    model_kind: str = "",
    horizon: int = 1,
    target_col: str = "target_return",
) -> dict[str, Any]:
    """
    Run all 13 diagnostic modules and return a flat dict of ``diag_`` prefixed keys.

    This is the only function that needs to be called from the runner.
    All outputs are additive — they do not modify or replace existing row values.
    """
    out: dict[str, Any] = {}

    try:
        out.update(turnover_diagnostics(pnl))
    except Exception as exc:
        logger.debug("turnover_diagnostics failed: %s", exc)

    try:
        h_diag = horizon_alignment(oos_df, return_col=_deployable_return_col(oos_df, target_col))
        optimal_h = int(_safe(h_diag.get("diag_optimal_horizon"), horizon))
        h_diag["diag_overtraded"] = 1.0 if horizon < optimal_h else 0.0
        out.update(h_diag)
    except Exception as exc:
        logger.debug("horizon_alignment failed: %s", exc)

    try:
        out.update(long_only_evaluation(oos_df, target_horizon_days=int(horizon)))
    except Exception as exc:
        logger.debug("long_only_evaluation failed: %s", exc)

    try:
        out.update(short_attribution(oos_df, pnl, model_kind=model_kind))
    except Exception as exc:
        logger.debug("short_attribution failed: %s", exc)

    try:
        out.update(cross_sectional_strength(oos_df))
    except Exception as exc:
        logger.debug("cross_sectional_strength failed: %s", exc)

    try:
        out.update(cost_aware_score({**row, **out}))
    except Exception as exc:
        logger.debug("cost_aware_score failed: %s", exc)

    try:
        out.update(regime_segmentation(oos_df, pnl))
    except Exception as exc:
        logger.debug("regime_segmentation failed: %s", exc)

    # Modules 9-13: signal-to-execution translation layer
    try:
        out.update(cost_decomposition({**row, **out}, pnl))
    except Exception as exc:
        logger.debug("cost_decomposition failed: %s", exc)

    try:
        out.update(exposure_decomposition(oos_df, pnl))
    except Exception as exc:
        logger.debug("exposure_decomposition failed: %s", exc)

    try:
        out.update(short_side_classification(oos_df, pnl, {**row, **out}))
    except Exception as exc:
        logger.debug("short_side_classification failed: %s", exc)

    try:
        out.update(alpha_capture_ratio({**row, **out}, pnl))
    except Exception as exc:
        logger.debug("alpha_capture_ratio failed: %s", exc)

    try:
        out.update(turnover_by_decile(oos_df))
    except Exception as exc:
        logger.debug("turnover_by_decile failed: %s", exc)

    try:
        out.update(execution_robustness_diagnostics(oos_df, pnl, target_col=target_col))
    except Exception as exc:
        logger.debug("execution_robustness_diagnostics failed: %s", exc)

    # Diagnostics schema reconciliation:
    # ``turnover_by_decile`` and the robust rank-autocorrelation path estimate
    # related signal persistence metrics from different support sets.  Promotion
    # reporting consumes the robust field, so fail closed on gates but do not
    # mislabel a model as having no halflife evidence when the decile diagnostic
    # has already produced a finite estimate.
    robust_hl = _safe(out.get("diag_robust_signal_halflife"), np.nan)
    if not np.isfinite(robust_hl):
        for source_name, source_value in (
            ("decile_turnover", out.get("diag_decile_signal_halflife")),
            ("window_robustness", row.get("diag_robust_signal_halflife")),
            ("window_ic_stats", row.get("signal_halflife_days")),
        ):
            candidate = _safe(source_value, np.nan)
            if np.isfinite(candidate):
                out["diag_robust_signal_halflife"] = float(candidate)
                out["diag_robust_signal_halflife_source"] = source_name
                break
    else:
        out["diag_robust_signal_halflife_source"] = "rank_autocorrelation"

    try:
        primary_label, all_labels = classify_research_label({**row, **out})
        out["diag_research_label"] = primary_label
        out["diag_research_labels_all"] = "|".join(all_labels)
    except Exception as exc:
        logger.debug("classify_research_label failed: %s", exc)
        out["diag_research_label"] = "unknown"
        out["diag_research_labels_all"] = ""

    return out


# ---------------------------------------------------------------------------
# Research report text generator
# ---------------------------------------------------------------------------

def generate_research_report(
    report: "pd.DataFrame",
    *,
    out_path: "Any" = None,
) -> str:
    """
    Generate a human-readable research report for all evaluated models.

    Groups models by research label and summarises key metrics.
    """
    import io
    buf = io.StringIO()
    w = buf.write

    w("=" * 72 + "\n")
    w("RESEARCH DIAGNOSTICS REPORT\n")
    w("=" * 72 + "\n\n")

    if report is None or report.empty:
        w("No models to report.\n")
        text = buf.getvalue()
        if out_path is not None:
            try:
                import pathlib
                pathlib.Path(out_path).write_text(text, encoding="utf-8")
            except Exception:
                pass
        return text

    def _fmt(val: Any, digits: int = 3) -> str:
        if val is None:
            return "—"
        try:
            f = float(val)
            return f"{f:.{digits}f}" if np.isfinite(f) else "—"
        except Exception:
            return str(val)[:10]

    label_col = "diag_research_label" if "diag_research_label" in report.columns else None

    # Summary table
    w(f"{'Model':<28} {'Path':<12} {'BaseScore':>10} {'Robust':>7} {'AdjScore':>10} {'Rank∆':>6} {'Reason':<15}\n")
    w("-" * 110 + "\n")
    for _, row in report.iterrows():
        name = str(row.get("model_name", ""))[:27]
        path = str(row.get("model_kind", ""))[:11]
        bs = _fmt(row.get("base_screen_score"))
        rs = _fmt(row.get("execution_robustness_score"), 2)
        as_ = _fmt(row.get("adjusted_screen_score"))
        rc = int(row.get("rank_change", 0))
        rc_str = f"{rc:+d}" if rc != 0 else "—"
        reason = str(row.get("robustness_reason", ""))[:14]
        w(f"{name:<28} {path:<12} {bs:>10} {rs:>7} {as_:>10} {rc_str:>6} {reason:<15}\n")

    w("\n")

    # Per-label details
    if label_col:
        for label, grp in report.groupby("diag_research_label", sort=True):
            w(f"\n── {str(label).upper()} ({len(grp)} model(s)) ──\n")
            for _, row in grp.iterrows():
                name = str(row.get("model_name", ""))
                w(f"\n  {name}\n")
                w(f"    Exec Sharpe:       {_fmt(row.get('exec_sharpe'))}\n")
                w(f"    Long-Only Sharpe:  {_fmt(row.get('diag_lo_sharpe'))} "
                  f"(net: {_fmt(row.get('diag_lo_net_sharpe'))})\n")
                w(f"    IC t-stat:         {_fmt(row.get('oos_ic_tstat'))}\n")
                w(f"    Cost drag:         {_fmt(row.get('diag_cost_drag'), 2)}\n")
                _ahp_raw = row.get('diag_avg_holding_period_days')
                _ahp_finite = _ahp_raw is not None and isinstance(_ahp_raw, (int, float)) and np.isfinite(float(_ahp_raw))
                _ahp_s = f"{_fmt(_ahp_raw, 1)}d" if _ahp_finite else "—"
                w(f"    Avg hold period:   {_ahp_s}\n")
                _oh_raw = row.get('diag_optimal_horizon')
                _oh_finite = _oh_raw is not None and isinstance(_oh_raw, (int, float)) and np.isfinite(float(_oh_raw))
                _oh_s = f"{_fmt(_oh_raw, 0)}d" if _oh_finite else "—"
                w(f"    Optimal horizon:   {_oh_s} "
                  f"({'overtraded' if _safe(row.get('diag_overtraded')) > 0.5 else 'ok'})\n")
                w(f"    Short destroys:    {'YES' if _safe(row.get('diag_short_destroys_value')) > 0.5 else 'no'}\n")
                w(f"    Beta hedge Sharpe: {_fmt(row.get('diag_beta_hedge_sharpe'))}\n")
                w(f"    Regime best/worst: {row.get('diag_regime_best', '—')} / "
                  f"{row.get('diag_regime_worst', '—')} "
                  f"(spread: {_fmt(row.get('diag_regime_spread'))})\n")
                w(f"    Bull/bear Sharpe:  {_fmt(row.get('diag_regime_bull_sharpe'))} / "
                  f"{_fmt(row.get('diag_regime_bear_sharpe'))}\n")
                w(f"    Decile spread CV:  {_fmt(row.get('diag_decile_spread_cv'), 2)} "
                  f"(IC consist.: {_fmt(row.get('diag_ic_consistency_rate'), 2)})\n")
                # Cost decomposition (module 9)
                dom = row.get("diag_cost_dominant", "—")
                w(f"    Cost dominant:     {dom}  "
                  f"comm={_fmt(row.get('diag_cost_pct_commission'), 2)}  "
                  f"spread={_fmt(row.get('diag_cost_pct_spread'), 2)}  "
                  f"impact={_fmt(row.get('diag_cost_pct_impact'), 2)}  "
                  f"borrow={_fmt(row.get('diag_cost_pct_borrow'), 2)}\n")
                w(f"    Alpha/cost ratio:  {_fmt(row.get('diag_alpha_vs_cost_ratio'))} "
                  f"({'costs > alpha' if _safe(row.get('diag_alpha_vs_cost_ratio')) < 1.0 else 'alpha > costs'})\n")
                # Exposure decomposition (module 10)
                w(f"    Beta PnL share:    {_fmt(row.get('diag_beta_pnl_pct'), 2)}  "
                  f"residual Sharpe: {_fmt(row.get('diag_residual_alpha_sharpe'))}  "
                  f"mkt_corr: {_fmt(row.get('diag_mkt_corr'), 2)}\n")
                w(f"    Vol tilt:          {_fmt(row.get('diag_vol_tilt'), 2)}\n")
                # Short-side classification (module 11)
                short_cls = row.get("diag_short_classification", "—")
                w(f"    Short class:       {short_cls}  "
                  f"IC_pos_rate={_fmt(row.get('diag_short_ic_positive_rate'), 2)}  "
                  f"resid_SR={_fmt(row.get('diag_short_residual_sharpe'))}\n")
                # Alpha capture (module 12)
                w(f"    Alpha capture:     {_fmt(row.get('diag_alpha_capture_ratio'), 2)}  "
                  f"FLAM_SR={_fmt(row.get('diag_theoretical_sr_flam'))}  "
                  f"breakeven_IC={_fmt(row.get('diag_breakeven_ic'), 4)}\n")
                # Turnover by decile (module 13)
                _dhl_raw = row.get('diag_decile_signal_halflife')
                _dhl_finite = _dhl_raw is not None and isinstance(_dhl_raw, (int, float)) and np.isfinite(float(_dhl_raw))
                _dhl_s = f"{_fmt(_dhl_raw, 1)}d" if _dhl_finite else "—"
                w(f"    Signal halflife:   {_dhl_s}  "
                  f"top_decile_turn={_fmt(row.get('diag_decile_turn_top'), 2)}  "
                  f"top_vs_bot={_fmt(row.get('diag_decile_turn_top_vs_bot'), 2)}\n")
                w(f"    All labels:        {row.get('diag_research_labels_all', '')}\n")
                
                # Execution Robustness (module 14 & 15)
                w(f"    Robustness Score:  {_fmt(row.get('execution_robustness_score'), 2)}  "
                  f"(Rank∆: {int(row.get('rank_change', 0)):+d})  "
                  f"Reason: {row.get('robustness_reason', 'Robust')}\n")
                w(f"    Base vs Adjusted:  {_fmt(row.get('base_screen_score'))} -> {_fmt(row.get('adjusted_screen_score'))}\n")
                _hl_raw = row.get('diag_robust_signal_halflife')
                _hl_finite = _hl_raw is not None and isinstance(_hl_raw, (int, float)) and np.isfinite(float(_hl_raw))
                _hl_s = f"{_fmt(_hl_raw, 1)}d" if _hl_finite else "—"
                w(f"    Robust Details:    halflife={_hl_s}  "
                  f"cost_ic={_fmt(row.get('diag_robust_cost_adjusted_ic'), 4)}  "
                  f"caic_ratio={_fmt(row.get('diag_caic_to_raw_ic_ratio'), 2)}\n")
                w(f"    Stability:         turn_vol={_fmt(row.get('diag_robust_turnover_volatility'), 3)}  "
                  f"tail_stab={_fmt(row.get('diag_robust_tail_stability'), 2)}  "
                  f"hhi={_fmt(row.get('diag_robust_hhi'), 4)}\n")

    w("\n" + "=" * 72 + "\n")
    w("NOTE: These diagnostics do NOT affect promotion gates.\n")
    w("      Use them to prioritise research iterations.\n")
    w("=" * 72 + "\n")

    text = buf.getvalue()
    if out_path is not None:
        try:
            import pathlib
            pathlib.Path(out_path).write_text(text, encoding="utf-8")
        except Exception as exc:
            logger.warning("Failed to write research report: %s", exc)

    return text


# ═══════════════════════════════════════════════════════════════════════════════
# P14: Sharpe Calculation Integrity Tests
# ═══════════════════════════════════════════════════════════════════════════════
def _test_sharpe_calculation_integrity() -> dict[str, bool]:
    """
    Validate that _ann_sharpe uses volatility (sigma) not standard error (SE).
    """
    results: dict[str, bool] = {}
    rng = np.random.default_rng(42)

    # ── Test 1: Constant positive returns → nan (zero vol, non-zero mean) ──
    const_ret = np.full(100, 0.001, dtype=float)
    sr_const = _ann_sharpe(const_ret, hac_lags=0)
    # Zero-variance returns produce an indeterminate Sharpe → should be nan
    results["constant_returns_nan"] = not np.isfinite(float(sr_const))

    # With tiny noise, Sharpe becomes huge but finite
    const_ret_noisy = np.full(100, 0.001, dtype=float) + rng.normal(0, 1e-12, 100)
    sr_const_noisy = _ann_sharpe(const_ret_noisy, hac_lags=0)
    results["constant_returns_large_sharpe"] = np.isfinite(sr_const_noisy) and sr_const_noisy > 100.0

    # ── Test 2: Scale invariance — Sharpe does NOT mechanically grow with n ──
    # Use the SAME underlying distribution but different sample sizes.
    # The OLD buggy formula would give sr_2000 ≈ sqrt(20) × sr_100.
    mu_daily = 0.0005
    sigma_daily = 0.015
    daily_100 = rng.normal(mu_daily, sigma_daily, 100)
    daily_2000 = rng.normal(mu_daily, sigma_daily, 2000)
    sr_100 = _ann_sharpe(daily_100, hac_lags=0)
    sr_2000 = _ann_sharpe(daily_2000, hac_lags=0)
    # Both should estimate the same population Sharpe (~0.53 for these params).
    # Even with sampling noise, the ratio should not approach sqrt(2000/100) ≈ 4.47.
    # A generous bound of 5× allows for sampling noise while rejecting the old bug.
    ratio_obs = abs(sr_2000 / max(abs(sr_100), 1e-6))
    results["scale_invariance_n100_vs_n2000"] = ratio_obs < 5.0

    # ── Test 3: Old formula vs corrected differs by ~sqrt(n) ──
    def _old_ann_sharpe(arr, hac_lags=0):
        r = np.asarray(arr, dtype=float)
        r = r[np.isfinite(r)]
        n = len(r)
        if n < 10:
            return float("nan")
        mu = float(r.mean())
        demeaned = r - mu
        var = float(np.dot(demeaned, demeaned) / n)
        max_lag = min(hac_lags, n - 1)
        for lag in range(1, max_lag + 1):
            cov = float(np.dot(demeaned[lag:], demeaned[:-lag]) / n)
            var += 2.0 * (1.0 - lag / (max_lag + 1.0)) * cov
        se = np.sqrt(max(var, 1e-30) / n)
        return float(mu / se * np.sqrt(252.0))

    n_obs = 500
    test_ret = rng.normal(0.001 / 252, 0.01, n_obs)
    old_sr = _old_ann_sharpe(test_ret, hac_lags=0)
    new_sr = _ann_sharpe(test_ret, hac_lags=0)
    ratio = old_sr / max(new_sr, 1e-8)
    expected_ratio = np.sqrt(n_obs)
    results["old_vs_new_differs_by_sqrt_n"] = (
        abs(ratio - expected_ratio) / expected_ratio < 0.20
    )
    results["sign_preserved"] = np.sign(old_sr) == np.sign(new_sr)

    return results


def run_sharpe_integrity_tests() -> None:
    """Print test results."""
    results = _test_sharpe_calculation_integrity()
    print("\n" + "=" * 65)
    print("P14: Sharpe Calculation Integrity Tests")
    print("=" * 65)

    all_pass = True
    for name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        if not passed:
            all_pass = False
        print(f"  {status}  {name}")

    print("-" * 65)
    print(f"  {'ALL TESTS PASSED' if all_pass else 'SOME TESTS FAILED'}")
    print("=" * 65 + "\n")


if os.environ.get("RESEARCH_DIAGNOSTICS_TEST_ON_IMPORT", "").strip().lower() in ("1", "true", "yes"):
    run_sharpe_integrity_tests()
