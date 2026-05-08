"""
Signal Stability
================
Cross-sectional normalization, stability diagnostics, and data-driven shrinkage
for ranking signals.

Stability problem
-----------------
A raw model score S_t can be unstable across dates for reasons unrelated to
alpha quality: changing feature scale, retraining artifacts, or erratic model
confidence.  Unstable scores lead to high turnover and poor realized Sharpe
even when point-in-time IC is acceptable.

Transformations
---------------
1. Cross-sectional z-score:  S_norm_t = (S_t − μ_t) / σ_t
2. Optional rank transform:  S_rank_t = rank(S_norm_t) / N_t  ∈ (0, 1)

Stability metrics
-----------------
rank_autocorr    : mean Spearman ρ(rank_t, rank_{t-1}) across consecutive dates
                   High (→ 1.0) = stable relative ordering
spread_cv        : std(daily_decile_spread) / |mean(daily_decile_spread)|
                   Low (→ 0.0) = consistent cross-sectional separation
monotonicity     : fraction of consecutive decile pairs with correct sign on target
                   High (→ 1.0) = linear decile structure

Data-driven shrinkage (temporal EMA)
-------------------------------------
Shrinkage via simple cross-sectional scaling (λ × S_norm) is Spearman-invariant
and produces a flat objective.  Instead we use per-ticker temporal EMA:

    S_ema_t = λ × S_norm_t + (1 − λ) × S_ema_{t-1}

This genuinely changes cross-sectional rankings because it blends the current
z-score with prior-period z-scores.  The tradeoff is real:

    λ = 1.0  →  no smoothing  (maximum freshness, noisier rankings)
    λ → 0.0  →  heavy smoothing  (stale IC, stable rankings)
    λ*       →  optimal point on the IC × rank_autocorr frontier

Collapse guard
--------------
If mean daily cross-sectional std of S_ema falls below
min_std_fraction × mean std of S_norm, the normalized score (λ=1) is used.

Public API
----------
cross_section_normalize : z-score (+ optional rank transform) per date
compute_stability_metrics : rank_autocorr, spread_cv, monotonicity
learn_shrinkage_lambda  : data-driven λ from temporal EMA grid search
apply_signal_shrinkage  : per-ticker EMA smoothing with collapse guard
compute_stability_diagnostics : before/after comparison
format_stability_report : fixed-width text summary
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StabilityConfig:
    rank_transform: bool = False          # apply rank transform after z-score
    lambda_grid_steps: int = 21          # λ ∈ linspace(0, 1, steps)
    min_std_fraction: float = 0.05       # collapse guard: fraction of original std
    min_obs_per_date: int = 10           # skip dates with fewer observations
    target_col_for_ic: str = "target_return"   # target column for IC in λ search
    # Columns for decile spread computation
    decile_n: int = 10


# ---------------------------------------------------------------------------
# Step 1: cross-sectional normalization
# ---------------------------------------------------------------------------


def cross_section_normalize(
    df: pd.DataFrame,
    score_col: str,
    *,
    cfg: StabilityConfig | None = None,
    out_col: str = "score_norm",
) -> pd.DataFrame:
    """
    Z-score normalize ``score_col`` within each date's cross-section.

    If cfg.rank_transform is True, additionally transform to uniform ranks
    in (0, 1) per date after z-scoring.

    Returns a copy of df with ``out_col`` added.
    """
    if cfg is None:
        cfg = StabilityConfig()

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[out_col] = np.nan

    for dt, g in out.groupby("date", sort=True):
        s = pd.to_numeric(g[score_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(s)
        if mask.sum() < cfg.min_obs_per_date:
            out.loc[g.index, out_col] = 0.0
            continue

        mu = s[mask].mean()
        sigma = s[mask].std(ddof=1)
        if sigma < 1e-10:
            out.loc[g.index, out_col] = 0.0
            continue

        s_norm = (s - mu) / sigma
        s_norm = np.where(mask, s_norm, np.nan)

        if cfg.rank_transform:
            valid_idx = np.where(mask)[0]
            ranks = scipy_stats.rankdata(s_norm[mask], method="average")
            s_norm[valid_idx] = (ranks - 0.5) / mask.sum()

        out.loc[g.index, out_col] = s_norm

    return out


# ---------------------------------------------------------------------------
# Step 2: stability metrics
# ---------------------------------------------------------------------------


def compute_stability_metrics(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    *,
    cfg: StabilityConfig | None = None,
) -> dict[str, float]:
    """
    Compute three complementary stability metrics for a ranking signal.

    rank_autocorr
    -------------
    For each pair of consecutive dates (t, t+1), compute Spearman ρ of
    score ranks across tickers that appear in both dates.  Average over all
    consecutive pairs.

    spread_cv
    ---------
    Compute per-date decile spread (mean top-decile target − mean bottom-decile
    target using score ranks).  CV = std / |mean| across dates.

    monotonicity
    ------------
    For each date, rank scores into deciles and check whether mean target in
    decile k > mean target in decile k-1 for k = 2..10.  Average hit rate.

    Returns
    -------
    {rank_autocorr, spread_cv, monotonicity, n_consecutive_pairs, n_dates}
    """
    if cfg is None:
        cfg = StabilityConfig()

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    # ── rank autocorrelation ─────────────────────────────────────────────────
    rank_ac_vals: list[float] = []
    dates = sorted(work["date"].unique())

    # Build per-date {ticker: rank} maps — index by ticker for join
    date_ranks: dict = {}
    for dt in dates:
        g = work[work["date"] == dt]
        s = pd.to_numeric(g[score_col], errors="coerce")
        valid_mask = s.notna()
        if valid_mask.sum() < cfg.min_obs_per_date:
            continue
        tickers_dt = g.loc[valid_mask.index[valid_mask], "ticker"].to_numpy()
        ranks = scipy_stats.rankdata(s[valid_mask].to_numpy(dtype=float), method="average")
        date_ranks[dt] = pd.Series(ranks, index=tickers_dt, name="rank")

    for i in range(len(dates) - 1):
        t0, t1 = dates[i], dates[i + 1]
        if t0 not in date_ranks or t1 not in date_ranks:
            continue
        r0 = date_ranks[t0].rename("r0")
        r1 = date_ranks[t1].rename("r1")

        # Join on ticker (index is ticker name)
        joined = (
            pd.concat([r0, r1], axis=1)
            .dropna()
        )
        if len(joined) < 5:
            continue
        rho, _ = scipy_stats.spearmanr(joined["r0"], joined["r1"])
        if math.isfinite(rho):
            rank_ac_vals.append(float(rho))

    rank_autocorr = float(np.mean(rank_ac_vals)) if rank_ac_vals else float("nan")

    # ── decile spread and monotonicity ───────────────────────────────────────
    spread_vals: list[float] = []
    mono_vals: list[float] = []

    for dt, g in work.groupby("date", sort=True):
        s = pd.to_numeric(g[score_col], errors="coerce")
        t = pd.to_numeric(g[target_col], errors="coerce")
        mask = s.notna() & t.notna()
        if mask.sum() < cfg.decile_n * 2:
            continue

        s_valid = s[mask]
        t_valid = t[mask]

        decile_bins = pd.qcut(s_valid.rank(method="first"), cfg.decile_n, labels=False)
        means = t_valid.groupby(decile_bins).mean()
        if len(means) < 2:
            continue

        ordered = means.sort_index()
        spread_vals.append(float(ordered.iloc[-1] - ordered.iloc[0]))

        # Monotonicity: fraction of consecutive pairs with positive slope
        diffs = np.diff(ordered.values)
        mono_vals.append(float((diffs > 0).mean()))

    spread_cv = float("nan")
    if len(spread_vals) > 1:
        arr = np.array(spread_vals, dtype=float)
        mu = float(arr.mean())
        sigma = float(arr.std(ddof=1))
        spread_cv = sigma / abs(mu) if abs(mu) > 1e-10 else float("nan")

    return {
        "rank_autocorr": rank_autocorr,
        "spread_cv": spread_cv,
        "monotonicity": float(np.mean(mono_vals)) if mono_vals else float("nan"),
        "n_consecutive_pairs": len(rank_ac_vals),
        "n_dates": len(spread_vals),
    }


# ---------------------------------------------------------------------------
# Internal EMA helpers (temporal shrinkage)
# ---------------------------------------------------------------------------


def _temporal_ema_wide(wide: pd.DataFrame, lambda_val: float) -> pd.DataFrame:
    """
    Per-ticker temporal EMA of a (date × ticker) DataFrame.

    S_ema_t = λ × S_norm_t + (1 − λ) × S_ema_{t-1}

    Equivalent to pandas ewm(alpha=λ, adjust=False).mean() along the date axis.
    λ=1.0 returns an unchanged copy. λ=0.0 returns the first row propagated
    forward (fully smoothed / stale signal).
    """
    lam = float(np.clip(lambda_val, 0.0, 1.0))
    if lam >= 1.0 - 1e-9:
        return wide.copy()
    if lam < 1e-6:
        # α=0 not supported by pandas; propagate first valid value for each ticker
        return wide.ffill().bfill().iloc[[0]].reindex(wide.index, method="ffill")
    return wide.ewm(alpha=lam, adjust=False).mean()


def _wide_mean_ic(
    ema_wide: pd.DataFrame,
    target_wide: pd.DataFrame,
    min_obs: int,
) -> float:
    """Mean daily Spearman IC between ema_wide and target_wide (date × ticker)."""
    common_dates = ema_wide.index.intersection(target_wide.index)
    ic_vals: list[float] = []
    for dt in common_dates:
        s = ema_wide.loc[dt].dropna()
        t = target_wide.loc[dt].dropna()
        common_tickers = s.index.intersection(t.index)
        if len(common_tickers) < min_obs or s[common_tickers].std() == 0.0:
            continue
        r, _ = scipy_stats.spearmanr(s[common_tickers].values, t[common_tickers].values)
        if math.isfinite(r):
            ic_vals.append(float(r))
    return float(np.mean(ic_vals)) if ic_vals else float("nan")


def _wide_rank_autocorr(ema_wide: pd.DataFrame, min_obs: int) -> float:
    """Mean Spearman rank autocorrelation across consecutive dates."""
    dates = list(ema_wide.index)
    rac_vals: list[float] = []
    for i in range(len(dates) - 1):
        s0 = ema_wide.loc[dates[i]].dropna()
        s1 = ema_wide.loc[dates[i + 1]].dropna()
        common = s0.index.intersection(s1.index)
        if len(common) < min_obs:
            continue
        r, _ = scipy_stats.spearmanr(s0[common].values, s1[common].values)
        if math.isfinite(r):
            rac_vals.append(float(r))
    return float(np.mean(rac_vals)) if rac_vals else float("nan")


def _pivot_score(df: pd.DataFrame, score_col: str) -> pd.DataFrame:
    """Pivot long (date, ticker, score) to wide (date × ticker), sorted by date."""
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["ticker"] = work["ticker"].astype(str)
    wide = work.pivot_table(index="date", columns="ticker", values=score_col, aggfunc="first")
    return wide.sort_index()


def _ema_wide_to_long(ema_wide: pd.DataFrame, out_col: str, ref_df: pd.DataFrame) -> pd.Series:
    """
    Reindex (date × ticker) EMA wide DataFrame back to the rows of ref_df.

    Returns a Series aligned to ref_df.index.
    """
    ref = ref_df.copy()
    ref["date"] = pd.to_datetime(ref["date"], errors="coerce")
    ref["ticker"] = ref["ticker"].astype(str)
    lookup_idx = pd.MultiIndex.from_arrays([ref["date"], ref["ticker"]])

    # Stack wide to MultiIndex Series (date, ticker) → value
    try:
        stacked = ema_wide.stack(future_stack=True)
    except TypeError:
        stacked = ema_wide.stack(dropna=False)

    stacked.name = out_col
    return stacked.reindex(lookup_idx).values


# ---------------------------------------------------------------------------
# Step 3: data-driven shrinkage λ via temporal EMA
# ---------------------------------------------------------------------------


def learn_shrinkage_lambda(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    *,
    cfg: StabilityConfig | None = None,
) -> tuple[float, dict[str, float]]:
    """
    Search for the optimal temporal EMA smoothing parameter λ ∈ [0, 1].

    For each λ, compute per-ticker EMA of the cross-sectionally normalized score:
        S_ema_t = λ × S_norm_t + (1 − λ) × S_ema_{t-1}

    Objective: f(λ) = |mean_IC(S_ema, target)| × rank_autocorr(S_ema)

    The tradeoff is genuine because EMA blends time-periods:
      λ=1.0 → pure current z-score (max IC, noisier rankings)
      λ→0.0 → dominated by past z-scores (stale IC, very stable rankings)
      λ*    → sits at the efficient frontier of IC × stability

    Returns
    -------
    lambda_star  : optimal λ (1.0 = no smoothing)
    diagnostics  : {lambda_star, objective_star, ic_star, rank_autocorr_star}
    """
    if cfg is None:
        cfg = StabilityConfig()

    # 1. Cross-sectional normalize once
    normed = cross_section_normalize(df, score_col, cfg=cfg, out_col="_snorm")

    # 2. Pivot normalized score and target to wide (date × ticker)
    wide = _pivot_score(normed, "_snorm")

    if target_col not in df.columns:
        return 1.0, {
            "lambda_star": 1.0, "objective_star": float("nan"),
            "ic_star": float("nan"), "rank_autocorr_star": float("nan"),
        }
    target_wide = _pivot_score(df, target_col).reindex(index=wide.index)

    # 3. Grid search over λ
    grid = np.linspace(0.0, 1.0, cfg.lambda_grid_steps)
    # Guarantee λ=1.0 (no-smoothing baseline) is always evaluated
    if not np.any(grid >= 1.0 - 1e-9):
        grid = np.append(grid, 1.0)

    best_lambda = 1.0
    best_obj = -np.inf
    best_ic = float("nan")
    best_rac = float("nan")

    for lam in grid:
        ema_wide = _temporal_ema_wide(wide, float(lam))
        ic_mean = _wide_mean_ic(ema_wide, target_wide, cfg.min_obs_per_date)
        if not math.isfinite(ic_mean):
            continue
        rac = _wide_rank_autocorr(ema_wide, cfg.min_obs_per_date)
        if not math.isfinite(rac):
            rac = 0.0
        obj = abs(ic_mean) * max(rac, 0.0)
        if obj > best_obj:
            best_obj = obj
            best_lambda = float(lam)
            best_ic = ic_mean
            best_rac = rac

    return best_lambda, {
        "lambda_star": best_lambda,
        "objective_star": best_obj if math.isfinite(best_obj) else float("nan"),
        "ic_star": best_ic,
        "rank_autocorr_star": best_rac,
    }


# ---------------------------------------------------------------------------
# Step 4: apply shrinkage (temporal EMA)
# ---------------------------------------------------------------------------


def apply_signal_shrinkage(
    df: pd.DataFrame,
    score_col: str,
    lambda_val: float,
    *,
    cfg: StabilityConfig | None = None,
    out_col: str = "score_shrunk",
) -> pd.DataFrame:
    """
    Apply temporal EMA smoothing with parameter λ to the ranking signal.

    Per ticker, across consecutive dates:
        S_ema_t = λ × S_norm_t + (1 − λ) × S_ema_{t-1}

    λ=1.0 → no smoothing (output = cross-sectional z-score per date)
    λ<1.0 → EMA blends current and past normalized scores, increasing
              rank stability at the cost of IC freshness.

    Collapse guard: if mean daily cross-sectional std of S_ema falls below
    min_std_fraction × mean std of S_norm, fall back to S_norm (λ=1).

    Returns copy of df with out_col appended.
    """
    if cfg is None:
        cfg = StabilityConfig()

    lam = float(np.clip(lambda_val, 0.0, 1.0))

    # 1. Cross-sectional normalize
    normed = cross_section_normalize(df, score_col, cfg=cfg, out_col="_snorm_tmp")

    # 2. Pivot to wide (date × ticker), sorted by date for EMA
    wide = _pivot_score(normed, "_snorm_tmp")

    # 3. Apply temporal EMA
    ema_wide = _temporal_ema_wide(wide, lam)

    # 4. Collapse guard: compare mean daily cross-sectional std
    orig_std_mean = float(wide.std(axis=1, ddof=1).mean())
    ema_std_mean = float(ema_wide.std(axis=1, ddof=1).mean())
    if (
        math.isfinite(orig_std_mean)
        and orig_std_mean > 1e-10
        and math.isfinite(ema_std_mean)
        and ema_std_mean < cfg.min_std_fraction * orig_std_mean
    ):
        ema_wide = wide  # fallback to normalized score

    # 5. Reindex back to original df rows
    out = df.copy()
    out[out_col] = _ema_wide_to_long(ema_wide, out_col, df)
    return out


# ---------------------------------------------------------------------------
# Full diagnostic pipeline
# ---------------------------------------------------------------------------


def compute_stability_diagnostics(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    *,
    cfg: StabilityConfig | None = None,
    raw_return_col: str = "forward_return",
    holding_days: int = 5,
) -> dict[str, float | str]:
    """
    End-to-end diagnostics: metrics before and after optimal temporal EMA shrinkage.

    Returns
    -------
    lambda_star         : learned smoothing parameter (1.0 = no smoothing)
    rank_autocorr_before / after
    spread_cv_before / after
    monotonicity_before / after
    ic_before / after   : mean daily IC (Spearman)
    sharpe_before / after : long-quintile Sharpe
    """
    if cfg is None:
        cfg = StabilityConfig()

    # Metrics before
    before = compute_stability_metrics(df, score_col, target_col, cfg=cfg)

    # Learn λ
    lam, lam_diag = learn_shrinkage_lambda(df, score_col, target_col, cfg=cfg)

    # Apply shrinkage
    shrunk = apply_signal_shrinkage(df, score_col, lam, cfg=cfg, out_col="_score_shrunk")

    # Metrics after
    after = compute_stability_metrics(shrunk, "_score_shrunk", target_col, cfg=cfg)

    def _mean_ic(frame: pd.DataFrame, col: str) -> float:
        ic_vals: list[float] = []
        for _, g in frame.groupby("date", sort=True):
            s = pd.to_numeric(g[col], errors="coerce").to_numpy(dtype=float)
            t = pd.to_numeric(g[target_col], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(s) & np.isfinite(t)
            if mask.sum() < 5 or s[mask].std() == 0:
                continue
            r, _ = scipy_stats.spearmanr(s[mask], t[mask])
            if math.isfinite(r):
                ic_vals.append(float(r))
        return float(np.mean(ic_vals)) if ic_vals else float("nan")

    def _q_sharpe(frame: pd.DataFrame, col: str) -> float:
        if raw_return_col not in frame.columns:
            return float("nan")
        rets: list[float] = []
        for _, g in frame.groupby("date", sort=True):
            if len(g) < 10:
                continue
            thr = g[col].quantile(0.80)
            top = g.loc[g[col] >= thr, raw_return_col].dropna()
            if len(top) < 2:
                continue
            rets.append(float(top.mean()))
        if len(rets) < 10:
            return float("nan")
        arr = np.array(rets, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < 10 or arr.std(ddof=1) < 1e-10:
            return float("nan")
        return float(arr.mean() / arr.std(ddof=1) * math.sqrt(252.0 / max(holding_days, 1)))

    return {
        "lambda_star": lam,
        "rank_autocorr_before": before.get("rank_autocorr", float("nan")),
        "rank_autocorr_after": after.get("rank_autocorr", float("nan")),
        "spread_cv_before": before.get("spread_cv", float("nan")),
        "spread_cv_after": after.get("spread_cv", float("nan")),
        "monotonicity_before": before.get("monotonicity", float("nan")),
        "monotonicity_after": after.get("monotonicity", float("nan")),
        "ic_before": _mean_ic(df, score_col),
        "ic_after": _mean_ic(shrunk, "_score_shrunk"),
        "sharpe_before": _q_sharpe(df, score_col),
        "sharpe_after": _q_sharpe(shrunk, "_score_shrunk"),
    }


def format_stability_report(diag: dict[str, float | str]) -> str:
    """Render stability diagnostics as a fixed-width text block."""
    def _f(v: object) -> str:
        if isinstance(v, float) and math.isfinite(v):
            return f"{v:8.4f}"
        return "     NaN"

    lines = [
        "Signal Stability Diagnostics",
        "=" * 60,
        f"  Optimal shrinkage λ       : {_f(diag.get('lambda_star'))}",
        "  (λ=1.0 → no EMA smoothing | λ<1.0 → blend with past z-scores)",
        "",
        f"  {'Metric':<24} {'Before':>10} {'After':>10}",
        "  " + "-" * 46,
        f"  {'Rank autocorrelation':<24} {_f(diag.get('rank_autocorr_before')):>10} "
        f"{_f(diag.get('rank_autocorr_after')):>10}",
        f"  {'Spread CV':<24} {_f(diag.get('spread_cv_before')):>10} "
        f"{_f(diag.get('spread_cv_after')):>10}",
        f"  {'Monotonicity':<24} {_f(diag.get('monotonicity_before')):>10} "
        f"{_f(diag.get('monotonicity_after')):>10}",
        f"  {'Mean daily IC':<24} {_f(diag.get('ic_before')):>10} "
        f"{_f(diag.get('ic_after')):>10}",
        f"  {'Long-quintile Sharpe':<24} {_f(diag.get('sharpe_before')):>10} "
        f"{_f(diag.get('sharpe_after')):>10}",
        "=" * 60,
        "  Rank autocorr ↑ = more stable rankings across dates",
        "  Spread CV ↓ = more consistent cross-sectional separation",
    ]
    return "\n".join(lines)
