"""Empirical baseline comparisons for model diagnostics.

All functions operate on an existing OOS panel DataFrame with columns:
    date, ticker, score, forward_return (target_return optional)

No new data is loaded. All randomness is seeded for reproducibility.
Results are prefixed with ``baseline_`` in the returned dict.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats


def _spearman_ic_per_date(df: pd.DataFrame, score_col: str, ret_col: str) -> np.ndarray:
    """Return per-date Spearman IC values as a 1-D float array."""
    ic_vals: list[float] = []
    for _, grp in df.groupby("date", sort=False):
        s = grp[score_col].to_numpy(dtype=float)
        r = grp[ret_col].to_numpy(dtype=float)
        mask = np.isfinite(s) & np.isfinite(r)
        if mask.sum() < 3:
            continue
        sm, rm = s[mask], r[mask]
        if sm.std() == 0.0 or rm.std() == 0.0:
            continue
        ic, _ = stats.spearmanr(sm, rm)
        if np.isfinite(ic):
            ic_vals.append(float(ic))
    return np.array(ic_vals, dtype=float)


def score_shuffle_baseline(
    oos_df: pd.DataFrame,
    *,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """Bootstrap cross-sectional score shuffle to build a null-IC distribution.

    For each bootstrap iteration: within every date, randomly permute scores
    across tickers (preserving the cross-sectional distribution), then compute
    the mean daily Spearman IC of the shuffled panel.

    Comparing the actual model IC to this null distribution gives an empirical
    significance measure that requires no distributional assumption.
    """
    nan = float("nan")
    empty: dict[str, float] = {
        "baseline_shuffle_ic_mean": nan,
        "baseline_shuffle_ic_std": nan,
        "baseline_ic_percentile": nan,
        "baseline_ic_tstat_vs_shuffle": nan,
    }

    ret_col = "target_return" if "target_return" in oos_df.columns else "forward_return"
    needed = {"date", "score", ret_col}
    if not needed.issubset(oos_df.columns):
        return empty

    df = oos_df[["date", "score", ret_col]].dropna()
    if df.empty or df["date"].nunique() < 5:
        return empty

    actual_ics = _spearman_ic_per_date(df, "score", ret_col)
    if len(actual_ics) < 3:
        return empty
    actual_ic_mean = float(actual_ics.mean())

    rng = np.random.default_rng(seed)
    dates = df["date"].to_numpy()
    scores = df["score"].to_numpy(dtype=float)
    rets = df[ret_col].to_numpy(dtype=float)

    date_labels, date_idx = np.unique(dates, return_inverse=True)
    n_dates = len(date_labels)
    shuffle_means: list[float] = []

    for _ in range(n_bootstrap):
        shuffled_scores = scores.copy()
        for d in range(n_dates):
            mask = date_idx == d
            idx_d = np.where(mask)[0]
            shuffled_scores[idx_d] = rng.permuted(shuffled_scores[idx_d])

        tmp = pd.DataFrame({"date": dates, "score": shuffled_scores, ret_col: rets})
        ics = _spearman_ic_per_date(tmp, "score", ret_col)
        if len(ics) > 0:
            shuffle_means.append(float(ics.mean()))

    if len(shuffle_means) < 10:
        return empty

    null = np.array(shuffle_means, dtype=float)
    null_mean = float(null.mean())
    null_std = float(null.std(ddof=1))
    percentile = float(np.mean(null < actual_ic_mean))
    tstat_vs_shuffle = (
        (actual_ic_mean - null_mean) / null_std if null_std > 1e-12 else nan
    )

    # Block bootstrap 95% CI on the actual IC (resample days with replacement)
    n_days = len(actual_ics)
    boot_ic_means: list[float] = []
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n_days, size=n_days)
        boot_ic_means.append(float(actual_ics[idx].mean()))
    boot_arr = np.array(boot_ic_means, dtype=float)
    ic_ci_low = float(np.percentile(boot_arr, 2.5))
    ic_ci_high = float(np.percentile(boot_arr, 97.5))

    return {
        "baseline_shuffle_ic_mean": null_mean,
        "baseline_shuffle_ic_std": null_std,
        "baseline_ic_percentile": percentile,
        "baseline_ic_tstat_vs_shuffle": tstat_vs_shuffle,
        "baseline_ic_ci_low": ic_ci_low,
        "baseline_ic_ci_high": ic_ci_high,
    }


def sign_flip_baseline(oos_df: pd.DataFrame) -> dict[str, float]:
    """Compare model IC to sign-flipped score IC.

    A valid signal should satisfy: IC(score) > 0 > IC(-score).
    If IC(-score) > IC(score) the sign convention is inverted (e.g. a
    short_classifier that was not properly negated before execution).

    Also computes the ratio of sign-flipped daily PnL to actual daily PnL using
    equal-weight top-k long portfolios, to quantify how much value comes from
    signal direction vs noise.
    """
    nan = float("nan")
    empty: dict[str, float] = {
        "baseline_flip_ic_mean": nan,
        "baseline_flip_ic_vs_actual_ratio": nan,
        "baseline_sign_inversion_detected": False,
    }

    ret_col = "target_return" if "target_return" in oos_df.columns else "forward_return"
    needed = {"date", "score", ret_col}
    if not needed.issubset(oos_df.columns):
        return empty

    df = oos_df[["date", "score", ret_col]].dropna()
    if df.empty or df["date"].nunique() < 5:
        return empty

    actual_ics = _spearman_ic_per_date(df, "score", ret_col)
    if len(actual_ics) < 3:
        return empty
    actual_ic_mean = float(actual_ics.mean())

    df_flip = df.copy()
    df_flip["score"] = -df_flip["score"]
    flip_ics = _spearman_ic_per_date(df_flip, "score", ret_col)
    flip_ic_mean = float(flip_ics.mean()) if len(flip_ics) >= 3 else nan

    ic_ratio = (flip_ic_mean / actual_ic_mean) if (
        np.isfinite(flip_ic_mean) and np.isfinite(actual_ic_mean) and abs(actual_ic_mean) > 1e-9
    ) else nan
    sign_inversion = bool(
        np.isfinite(flip_ic_mean) and np.isfinite(actual_ic_mean) and flip_ic_mean > actual_ic_mean
    )

    return {
        "baseline_flip_ic_mean": flip_ic_mean,
        "baseline_flip_ic_vs_actual_ratio": ic_ratio,
        "baseline_sign_inversion_detected": sign_inversion,
    }


def equal_weight_decile_baseline(
    oos_df: pd.DataFrame,
    *,
    n_deciles: int = 10,
) -> dict[str, float]:
    """Equal-weight decile portfolio diagnostics vs the model's own signal.

    Within each date, assign stocks to n_deciles buckets by score rank.
    Compute equal-weight mean forward_return per decile per day, then:
      - top-minus-bottom spread (D10 - D1)
      - t-stat of that daily spread across dates
      - monotonicity: fraction of adjacent decile pairs where return increases

    This separates cross-sectional ordering ability from portfolio-level noise.
    """
    nan = float("nan")
    empty: dict[str, float] = {
        "baseline_ew_decile_spread_mean": nan,
        "baseline_ew_decile_spread_tstat": nan,
        "baseline_ew_decile_monotonicity": nan,
        "baseline_ew_decile_spread_sharpe": nan,
    }

    ret_col = "target_return" if "target_return" in oos_df.columns else "forward_return"
    needed = {"date", "score", ret_col}
    if not needed.issubset(oos_df.columns):
        return empty

    df = oos_df[["date", "score", ret_col]].dropna()
    if df.empty or df["date"].nunique() < 5:
        return empty

    df = df.copy()
    pct_rank = df.groupby("date")["score"].rank(pct=True, method="first")
    n_per_date = df.groupby("date")["score"].transform("count")
    valid = n_per_date >= n_deciles
    df["decile"] = np.where(
        valid,
        np.floor(pct_rank.clip(0, 1 - 1e-12) * n_deciles).astype(int),
        np.nan,
    )
    df = df.dropna(subset=["decile"])
    df["decile"] = df["decile"].astype(int)

    if df.empty:
        return empty

    daily_decile = (
        df.groupby(["date", "decile"])[ret_col]
        .mean()
        .unstack("decile")
    )

    top_d = n_deciles - 1
    bot_d = 0
    if top_d not in daily_decile.columns or bot_d not in daily_decile.columns:
        return empty

    spread = daily_decile[top_d] - daily_decile[bot_d]
    spread = spread.dropna()
    if len(spread) < 5:
        return empty

    spread_mean = float(spread.mean())
    spread_std = float(spread.std(ddof=1))
    spread_tstat = (
        float(stats.ttest_1samp(spread.to_numpy(), 0.0).statistic)
        if len(spread) >= 5 else nan
    )
    spread_sharpe = (
        spread_mean / spread_std * np.sqrt(252) if spread_std > 1e-12 else nan
    )

    all_deciles = sorted(d for d in daily_decile.columns if isinstance(d, (int, np.integer)))
    if len(all_deciles) >= 2:
        decile_means = np.array([
            daily_decile[d].mean() for d in all_deciles if d in daily_decile.columns
        ])
        n_pairs = len(decile_means) - 1
        monotone_pairs = int(np.sum(np.diff(decile_means) > 0))
        monotonicity = monotone_pairs / n_pairs if n_pairs > 0 else nan
    else:
        monotonicity = nan

    return {
        "baseline_ew_decile_spread_mean": spread_mean,
        "baseline_ew_decile_spread_tstat": spread_tstat,
        "baseline_ew_decile_monotonicity": monotonicity,
        "baseline_ew_decile_spread_sharpe": spread_sharpe,
    }


def alpha_execution_decomposition(
    oos_df: pd.DataFrame,
    pnl_df: pd.DataFrame,
    *,
    n_deciles: int = 10,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """Decompose PnL into raw signal alpha vs implementation shortfall.

    raw_alpha     = top-decile EW return - bottom-decile EW return (score-ranked,
                    forward_return as outcome). This is the alpha available to a
                    perfect frictionless executor.
    implemented   = daily gross_return from the executable portfolio simulation.
    execution_drag= raw_alpha - implemented (captures slippage, capacity, timing).

    Also computes shuffle and sign-flip raw_alpha for context.
    """
    nan = float("nan")
    empty: dict[str, float] = {
        "decomp_raw_alpha_mean": nan,
        "decomp_raw_alpha_tstat": nan,
        "decomp_implemented_pnl_mean": nan,
        "decomp_execution_drag_mean": nan,
        "decomp_execution_drag_pct": nan,
        "decomp_raw_alpha_vs_shuffle_percentile": nan,
        "decomp_flip_raw_alpha_mean": nan,
        "decomp_alpha_capture_ratio": nan,
    }

    ret_col = "target_return" if "target_return" in oos_df.columns else "forward_return"
    if not {"date", "score", ret_col}.issubset(oos_df.columns):
        return empty

    df = oos_df[["date", "score", ret_col]].dropna()
    if df.empty or df["date"].nunique() < 5:
        return empty

    # ── Raw alpha: equal-weight top vs bottom decile per date ──────────────
    pct_rank = df.groupby("date")["score"].rank(pct=True, method="first")
    n_per_date = df.groupby("date")["score"].transform("count")
    valid = n_per_date >= n_deciles
    df = df.copy()
    df["decile"] = np.where(
        valid,
        np.floor(pct_rank.clip(0, 1 - 1e-12) * n_deciles).astype(int),
        np.nan,
    )
    df = df.dropna(subset=["decile"])
    if df.empty:
        return empty

    daily_decile = df.groupby(["date", "decile"])[ret_col].mean().unstack("decile")
    top_d, bot_d = n_deciles - 1, 0
    if top_d not in daily_decile.columns or bot_d not in daily_decile.columns:
        return empty

    raw_spread = (daily_decile[top_d] - daily_decile[bot_d]).dropna()
    if len(raw_spread) < 5:
        return empty

    raw_alpha_mean = float(raw_spread.mean())
    raw_alpha_tstat = float(
        stats.ttest_1samp(raw_spread.to_numpy(), 0.0).statistic
    ) if len(raw_spread) >= 5 else nan

    # ── Implemented PnL: actual gross portfolio return from simulation ──────
    implemented_mean = nan
    execution_drag = nan
    execution_drag_pct = nan
    alpha_capture_ratio = nan

    if pnl_df is not None and not pnl_df.empty and "gross_return" in pnl_df.columns:
        gross = pd.to_numeric(pnl_df["gross_return"], errors="coerce").dropna()
        if len(gross) >= 5:
            implemented_mean = float(gross.mean())
            execution_drag = raw_alpha_mean - implemented_mean
            if abs(raw_alpha_mean) > 1e-12:
                execution_drag_pct = execution_drag / abs(raw_alpha_mean)
                alpha_capture_ratio = implemented_mean / raw_alpha_mean

    # ── Shuffle baseline for raw_alpha ─────────────────────────────────────
    rng = np.random.default_rng(seed)
    dates_arr = df["date"].to_numpy()
    scores_arr = df["score"].to_numpy(dtype=float)
    rets_arr = df[ret_col].to_numpy(dtype=float)
    date_labels, date_idx = np.unique(dates_arr, return_inverse=True)
    n_dates = len(date_labels)

    shuffle_spreads: list[float] = []
    for _ in range(n_bootstrap):
        shuf_scores = scores_arr.copy()
        for d in range(n_dates):
            idx_d = np.where(date_idx == d)[0]
            shuf_scores[idx_d] = rng.permuted(shuf_scores[idx_d])
        tmp = pd.DataFrame({"date": dates_arr, "score": shuf_scores, ret_col: rets_arr})
        pr = tmp.groupby("date")["score"].rank(pct=True, method="first")
        nc = tmp.groupby("date")["score"].transform("count")
        tmp["decile"] = np.where(nc >= n_deciles, np.floor(pr.clip(0, 1 - 1e-12) * n_deciles).astype(int), np.nan)
        tmp = tmp.dropna(subset=["decile"])
        if tmp.empty:
            continue
        dd = tmp.groupby(["date", "decile"])[ret_col].mean().unstack("decile")
        if top_d in dd.columns and bot_d in dd.columns:
            s = (dd[top_d] - dd[bot_d]).dropna()
            if len(s) > 0:
                shuffle_spreads.append(float(s.mean()))

    raw_alpha_vs_shuffle_pct = nan
    if len(shuffle_spreads) >= 10:
        null_arr = np.array(shuffle_spreads, dtype=float)
        raw_alpha_vs_shuffle_pct = float(np.mean(null_arr < raw_alpha_mean))

    # ── Sign-flip raw alpha ─────────────────────────────────────────────────
    df_flip = df.copy()
    df_flip["score"] = -df_flip["score"]
    pr_flip = df_flip.groupby("date")["score"].rank(pct=True, method="first")
    nc_flip = df_flip.groupby("date")["score"].transform("count")
    df_flip["decile"] = np.where(nc_flip >= n_deciles, np.floor(pr_flip.clip(0, 1 - 1e-12) * n_deciles).astype(int), np.nan)
    df_flip = df_flip.dropna(subset=["decile"])
    flip_raw_alpha_mean = nan
    if not df_flip.empty:
        dd_flip = df_flip.groupby(["date", "decile"])[ret_col].mean().unstack("decile")
        if top_d in dd_flip.columns and bot_d in dd_flip.columns:
            flip_spread = (dd_flip[top_d] - dd_flip[bot_d]).dropna()
            if len(flip_spread) > 0:
                flip_raw_alpha_mean = float(flip_spread.mean())

    return {
        "decomp_raw_alpha_mean": raw_alpha_mean,
        "decomp_raw_alpha_tstat": raw_alpha_tstat,
        "decomp_implemented_pnl_mean": implemented_mean,
        "decomp_execution_drag_mean": execution_drag,
        "decomp_execution_drag_pct": execution_drag_pct,
        "decomp_raw_alpha_vs_shuffle_percentile": raw_alpha_vs_shuffle_pct,
        "decomp_flip_raw_alpha_mean": flip_raw_alpha_mean,
        "decomp_alpha_capture_ratio": alpha_capture_ratio,
    }


def compute_empirical_baselines(
    oos_df: pd.DataFrame,
    *,
    model_kind: str = "",
    pnl_df: pd.DataFrame | None = None,
    n_bootstrap: int = 200,
    seed: int = 42,
) -> dict[str, float]:
    """Run all empirical baseline comparisons and return merged results.

    Designed for minimal footprint: operates solely on the OOS panel and PnL
    detail already in memory, no new data loading.
    """
    if oos_df is None or oos_df.empty:
        return {}

    results: dict[str, float] = {}
    results["baseline_model_kind"] = str(model_kind)

    results.update(score_shuffle_baseline(oos_df, n_bootstrap=n_bootstrap, seed=seed))
    results.update(sign_flip_baseline(oos_df))
    results.update(equal_weight_decile_baseline(oos_df))
    results.update(alpha_execution_decomposition(
        oos_df, pnl_df if pnl_df is not None else pd.DataFrame(),
        n_bootstrap=n_bootstrap, seed=seed,
    ))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# P36: Alpha-capture decomposition — per-date-per-ticker score-to-PnL attribution
# ═══════════════════════════════════════════════════════════════════════════════
#
# Explains positive-IC / negative-execution-Sharpe by tracing each ticker from
# forecast score → optimizer weight → realized return → cost → net PnL.
# Read-only diagnostic — does NOT change any model, weight, or gate.


def alpha_capture_decomposition(
    scored_df: pd.DataFrame,
    target_weights_df: pd.DataFrame | None = None,
    pnl_detail_df: pd.DataFrame | None = None,
    *,
    model_name: str = "unknown",
    window_idx: int | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    P36: Per-date-per-ticker score-to-PnL attribution.

    Joins forecast scores, optimizer target weights, and realized execution
    PnL to produce a row-level decomposition of where alpha is created vs where
    costs destroy it.

    Parameters:
        scored_df        — per-ticker-per-date scores + forward returns
        target_weights_df — optimizer weights (date, ticker, target_weight)
        pnl_detail_df    — per-date execution PnL (from simulate_executable_portfolio)

    Returns:
        decomposition_df — per-date-per-ticker attribution rows
        summary          — aggregate diagnostics (selected IC, weighted IC, etc.)
    """
    empty_summary: dict[str, Any] = {
        "model_name": model_name,
        "window_idx": window_idx,
        "decomp_status": "empty",
    }

    if scored_df is None or scored_df.empty:
        return pd.DataFrame(), empty_summary

    required = {"date", "ticker", "score"}
    if not required.issubset(scored_df.columns):
        return pd.DataFrame(), {**empty_summary, "decomp_status": "missing_columns"}

    ret_col = "target_return" if "target_return" in scored_df.columns else "forward_return"
    if ret_col not in scored_df.columns:
        return pd.DataFrame(), {**empty_summary, "decomp_status": "missing_return_col"}

    work = scored_df[["date", "ticker", "score"]].copy()

    ret_col = "target_return" if "target_return" in scored_df.columns else "forward_return"
    if ret_col not in scored_df.columns:
        return pd.DataFrame(), {**empty_summary, "decomp_status": "missing_return_col"}

    work = scored_df[["date", "ticker", "score"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work[ret_col] = pd.to_numeric(scored_df[ret_col], errors="coerce")
    work = work.dropna(subset=["date", "ticker", "score", ret_col])
    if work.empty:
        return pd.DataFrame(), {**empty_summary, "decomp_status": "empty_after_dropna"}

    # Score rank percentile (cross-sectional per date)
    work["score_rank_pct"] = work.groupby("date")["score"].rank(pct=True, method="average")

    # Selected-for-trade: top decile of score rank
    work["selected_for_trade"] = work["score_rank_pct"] >= 0.90

    # ── Join target weights if available ──────────────────────────────────
    has_weights = target_weights_df is not None and not target_weights_df.empty and "target_weight" in target_weights_df.columns
    if has_weights:
        tw = target_weights_df[["date", "ticker", "target_weight"]].copy()
        tw["date"] = pd.to_datetime(tw["date"], errors="coerce")
        tw["ticker"] = tw["ticker"].astype(str)
        tw["target_weight"] = pd.to_numeric(tw["target_weight"], errors="coerce").fillna(0.0)
        work = work.merge(tw, on=["date", "ticker"], how="left")
        work["target_weight"] = work["target_weight"].fillna(0.0)
        work["has_weight"] = work["target_weight"].abs() > 1e-12
    else:
        work["target_weight"] = 0.0
        work["has_weight"] = False

    # ── Gross PnL contribution ───────────────────────────────────────────
    # Per-ticker: weight × forward_return (daily-equivalent)
    horizon_days = 5  # default; override if detectible
    work["gross_pnl_per_ticker"] = work["target_weight"] * work[ret_col]

    # ── Cost allocation ───────────────────────────────────────────────────
    has_pnl = pnl_detail_df is not None and not pnl_detail_df.empty
    daily_cost: pd.Series | None = None
    if has_pnl and "cost_return" in pnl_detail_df.columns:
        pnl = pnl_detail_df.copy()
        pnl["date"] = pd.to_datetime(pnl["date"], errors="coerce")
        daily_cost = pnl.groupby("date")["cost_return"].sum()
        daily_cost = daily_cost.reindex(work["date"].unique()).fillna(0.0)

    if daily_cost is not None and not daily_cost.empty:
        # Allocate cost proportional to absolute weight
        work["abs_weight"] = work["target_weight"].abs()
        date_abs_sum = work.groupby("date")["abs_weight"].transform("sum")
        work["cost_share"] = np.where(
            date_abs_sum > 1e-12,
            work["abs_weight"] / date_abs_sum,
            0.0,
        )
        date_cost_map = daily_cost.to_dict()
        work["allocated_cost"] = work.apply(
            lambda r: float(date_cost_map.get(pd.Timestamp(r["date"]), 0.0)) * float(r["cost_share"]),
            axis=1,
        )
    else:
        work["cost_share"] = 0.0
        work["allocated_cost"] = 0.0

    # ── Net PnL contribution ─────────────────────────────────────────────
    work["net_pnl_per_ticker"] = work["gross_pnl_per_ticker"] - work["allocated_cost"]

    # ── Aggregate diagnostics ────────────────────────────────────────────
    summary: dict[str, Any] = {
        "model_name": model_name,
        "window_idx": window_idx,
        "decomp_status": "ok",
        "n_dates": int(work["date"].nunique()),
        "n_tickers": int(work["ticker"].nunique()),
        "n_rows": len(work),
        "n_selected_stock_days": int(work["selected_for_trade"].sum()),
    }

    # Full-universe IC (Spearman per date)
    ic_vals = []
    for _, grp in work.groupby("date", sort=False):
        if len(grp) < 5 or grp["score"].nunique() < 2 or grp[ret_col].nunique() < 2:
            continue
        try:
            rho = stats.spearmanr(grp["score"], grp[ret_col]).statistic
            if np.isfinite(rho):
                ic_vals.append(float(rho))
        except Exception:
            pass
    if ic_vals:
        ic_arr = np.array(ic_vals)
        summary["full_universe_ic_mean"] = float(np.mean(ic_arr))
        summary["full_universe_ic_tstat"] = float(np.mean(ic_arr) / np.std(ic_arr, ddof=1) * np.sqrt(len(ic_arr))) if len(ic_arr) > 1 else float("nan")
        summary["full_universe_ic_n_dates"] = len(ic_arr)
    else:
        summary["full_universe_ic_mean"] = float("nan")

    # Selected-universe IC (only stocks with |weight| > 0)
    if has_weights:
        selected = work[work["has_weight"]]
        sel_ic_vals = []
        for _, grp in selected.groupby("date", sort=False):
            if len(grp) < 3 or grp["score"].nunique() < 2 or grp[ret_col].nunique() < 2:
                continue
            try:
                rho = stats.spearmanr(grp["score"], grp[ret_col]).statistic
                if np.isfinite(rho):
                    sel_ic_vals.append(float(rho))
            except Exception:
                pass
        if sel_ic_vals:
            sel_arr = np.array(sel_ic_vals)
            summary["selected_universe_ic_mean"] = float(np.mean(sel_arr))
        else:
            summary["selected_universe_ic_mean"] = float("nan")
    else:
        summary["selected_universe_ic_mean"] = float("nan")

    # Weighted IC (score × |weight| correlation with return)
    if has_weights:
        w_ic_vals = []
        for _, grp in selected.groupby("date", sort=False):
            if len(grp) < 3:
                continue
            w = grp["target_weight"].abs().to_numpy(dtype=float)
            if w.sum() < 1e-12:
                continue
            scores = grp["score"].to_numpy(dtype=float)
            rets = grp[ret_col].to_numpy(dtype=float)
            w_score = np.average(scores, weights=w)
            w_ret = np.average(rets, weights=w)
            w_demean_s = scores - w_score
            w_demean_r = rets - w_ret
            num = np.average(w_demean_s * w_demean_r, weights=w)
            denom = np.sqrt(np.average(w_demean_s ** 2, weights=w) * np.average(w_demean_r ** 2, weights=w))
            if denom > 1e-12:
                w_ic_vals.append(float(num / denom))
        if w_ic_vals:
            summary["weighted_ic_mean"] = float(np.mean(w_ic_vals))
        else:
            summary["weighted_ic_mean"] = float("nan")
    else:
        summary["weighted_ic_mean"] = float("nan")

    # Score-weight correlation (does optimizer weight align with score?)
    if has_weights:
        corr_vals = []
        for _, grp in work[work["has_weight"]].groupby("date", sort=False):
            if len(grp) < 3 or grp["target_weight"].nunique() < 2 or grp["score"].nunique() < 2:
                continue
            try:
                rho = stats.spearmanr(grp["target_weight"], grp["score"]).statistic
                if np.isfinite(rho):
                    corr_vals.append(float(rho))
            except Exception:
                pass
        summary["score_weight_corr_mean"] = float(np.mean(corr_vals)) if corr_vals else float("nan")
    else:
        summary["score_weight_corr_mean"] = float("nan")

    # Aggregate PnL
    gross_total = float(work["gross_pnl_per_ticker"].sum())
    cost_total = float(work["allocated_cost"].sum())
    net_total = gross_total - cost_total
    summary["gross_pnl_total"] = gross_total
    summary["cost_total"] = cost_total
    summary["net_pnl_total"] = net_total
    summary["cost_to_gross_pnl"] = abs(cost_total) / max(abs(gross_total), 1e-12) if abs(gross_total) > 1e-12 else float("nan")

    # P40: Guard against misleading alpha_capture ratios when gross_total is near zero.
    # When |gross| is tiny, net/gross can produce arbitrarily large positive or negative
    # values that are numerically meaningless.  Report as indeterminate and keep the
    # cost_dominated classification based on net PnL sign.
    _gross_threshold = 1e-6  # 0.0001% daily return — below this the ratio is noise
    alpha_cap = float("nan")
    if abs(gross_total) > _gross_threshold:
        alpha_cap = net_total / gross_total
    elif abs(cost_total) > 1e-12:
        # Fallback: gross/cost ratio (inverse) for diagnostic only
        alpha_cap = gross_total / abs(cost_total)
    summary["alpha_capture"] = alpha_cap
    summary["alpha_capture_label"] = (
        "cost_dominated" if (np.isfinite(alpha_cap) and alpha_cap < 0)
        else ("positive" if (np.isfinite(alpha_cap) and alpha_cap > 0) else "unknown")
    )
    # P40: Report numerator and denominator alongside the ratio for auditability
    summary["alpha_capture_numerator"] = net_total
    summary["alpha_capture_denominator"] = gross_total
    summary["alpha_capture_indeterminate"] = abs(gross_total) <= _gross_threshold

    return work, summary


def alpha_capture_summary_from_per_model(
    per_model_parts: list[dict[str, Any]],
) -> pd.DataFrame:
    """
    P36: Combine per-window decomposition summaries into one model-level table.

    Parameters:
        per_model_parts — list of dicts from alpha_capture_decomposition() summaries

    Returns:
        DataFrame with one row per model/window
    """
    rows = []
    for part in per_model_parts:
        if not isinstance(part, dict):
            continue
        rows.append({
            "model_name": str(part.get("model_name", "?")),
            "window_idx": part.get("window_idx"),
            "full_universe_ic_mean": part.get("full_universe_ic_mean", float("nan")),
            "selected_universe_ic_mean": part.get("selected_universe_ic_mean", float("nan")),
            "weighted_ic_mean": part.get("weighted_ic_mean", float("nan")),
            "score_weight_corr_mean": part.get("score_weight_corr_mean", float("nan")),
            "gross_pnl_total": part.get("gross_pnl_total", float("nan")),
            "cost_total": part.get("cost_total", float("nan")),
            "net_pnl_total": part.get("net_pnl_total", float("nan")),
            "cost_to_gross_pnl": part.get("cost_to_gross_pnl", float("nan")),
            "alpha_capture": part.get("alpha_capture", float("nan")),
            "alpha_capture_label": str(part.get("alpha_capture_label", "unknown")),
            "decomp_status": str(part.get("decomp_status", "unknown")),
        })
    return pd.DataFrame(rows)
