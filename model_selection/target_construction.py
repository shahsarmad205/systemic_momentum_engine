"""
Target Construction
===================
Seven forward-return target encodings plus cross-sectional evaluation diagnostics.

Target types
------------
raw_5d           : 5-day compounded forward return (baseline)
raw_10d          : 10-day compounded forward return
market_residual  : raw_5d after removing cross-sectional market-beta component
sector_residual  : raw_5d after removing cross-sectional sector means
factor_residual  : raw_5d after removing market + sector + size (log-ADV) + vol
vol_scaled       : raw_5d / (realized_vol * sqrt(5)) — Sharpe-like, demeaned per date
cost_adjusted    : raw_5d - expected_round_trip_cost, demeaned per date

All residualization is performed cross-sectionally per date using ridge OLS.

Evaluation metrics (per target, per proxy feature)
---------------------------------------------------
ic_mean         : Mean daily cross-sectional rank IC(proxy, target)
ic_tstat        : Newey–West HAC t-stat on the daily IC series
ic_stability    : |IC mean| / IC std  (IC IR — higher = more consistent)
decile_spread   : Mean (top-decile target) − (bottom-decile target) using proxy ranks
monotonicity    : Fraction of adjacent proxy deciles with correct sign on target
long_sharpe     : Annualized Sharpe of top-quintile portfolio (by proxy) on raw_5d
composite_score : Weighted combination of the above metrics for final ranking
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from model_selection.alpha_research import (
    _daily_ic_values,
    _decile_shape,
    _forward_return_from_daily,
    _hac_tstat,
)
from model_selection.residualization import residualize_against_controls, safe_center
from model_selection.training import _expected_round_trip_cost
from model_selection.validation import ExecutionCostConfig

_EPS = 1e-12
_RIDGE = 1e-4
_MAX_ABS = 5.0
_DEFAULT_ADV = 50_000_000.0
_DEFAULT_VOL = 0.015

# ── Target column names ───────────────────────────────────────────────────────
RAW_5D = "raw_5d"
RAW_10D = "raw_10d"
RAW_20D = "raw_20d"
RAW_63D = "raw_63d"
MARKET_RESIDUAL = "market_residual_5d"
SECTOR_RESIDUAL = "sector_residual_5d"
FACTOR_RESIDUAL = "factor_residual_5d"
VOL_SCALED = "vol_scaled_5d"
COST_ADJUSTED = "cost_adjusted_5d"
MARKET_RESIDUAL_10D = "market_residual_10d"
SECTOR_RESIDUAL_10D = "sector_residual_10d"
FACTOR_RESIDUAL_10D = "factor_residual_10d"
MARKET_RESIDUAL_20D = "market_residual_20d"
SECTOR_RESIDUAL_20D = "sector_residual_20d"
FACTOR_RESIDUAL_20D = "factor_residual_20d"
MARKET_RESIDUAL_63D = "market_residual_63d"
SECTOR_RESIDUAL_63D = "sector_residual_63d"
FACTOR_RESIDUAL_63D = "factor_residual_63d"

TARGET_MENU: tuple[str, ...] = (
    RAW_5D,
    RAW_10D,
    RAW_20D,
    RAW_63D,
    MARKET_RESIDUAL,
    SECTOR_RESIDUAL,
    FACTOR_RESIDUAL,
    VOL_SCALED,
    COST_ADJUSTED,
    MARKET_RESIDUAL_10D,
    SECTOR_RESIDUAL_10D,
    FACTOR_RESIDUAL_10D,
    MARKET_RESIDUAL_20D,
    SECTOR_RESIDUAL_20D,
    FACTOR_RESIDUAL_20D,
    MARKET_RESIDUAL_63D,
    SECTOR_RESIDUAL_63D,
    FACTOR_RESIDUAL_63D,
)


def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _cs_winsorize(s: pd.Series, date_col: pd.Series, q: float) -> pd.Series:
    if q <= 0.0:
        return s
    lo = s.groupby(date_col).transform(lambda x: x.quantile(q))
    hi = s.groupby(date_col).transform(lambda x: x.quantile(1.0 - q))
    return s.clip(lower=lo, upper=hi)


def _residualize_panel_by_date(
    df: pd.DataFrame,
    target: pd.Series,
    *,
    use_market: bool = False,
    use_sector: bool = False,
    use_size: bool = False,
    use_vol: bool = False,
    ridge: float = _RIDGE,
    max_abs: float = _MAX_ABS,
) -> pd.Series:
    """
    Per-date cross-sectional ridge residualization.

    Builds a control matrix from the selected components (market beta, sector
    dummies, log-ADV, realized vol), then calls residualize_against_controls
    which solves the augmented least-squares system with ridge regularization.
    Falls back to centered y when no valid controls are available.
    """
    out = pd.Series(np.nan, index=df.index, dtype=float)
    work = df.copy()
    work["_target"] = _safe_numeric(target)
    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    for _, g in work.groupby("date", sort=True):
        y = pd.to_numeric(g["_target"], errors="coerce").to_numpy(dtype=float)
        if not np.isfinite(y).any():
            out.loc[g.index] = 0.0
            continue

        factors: list[pd.Series] = []

        if use_market:
            if "capm_beta" in g.columns:
                beta = _safe_numeric(g["capm_beta"]).fillna(1.0).clip(-3.0, 3.0)
                factors.append(beta - float(beta.mean()))
            elif "spy_forward_5d" in g.columns:
                spy = _safe_numeric(g["spy_forward_5d"]).fillna(0.0).clip(-0.5, 0.5)
                factors.append(spy - float(spy.mean()))

        if use_sector and "sector" in g.columns:
            dummies = pd.get_dummies(g["sector"].fillna("Unknown").astype(str), dtype=float)
            for col in dummies.columns:
                factors.append(dummies[col] - float(dummies[col].mean()))

        if use_size and "adv_dollar_20" in g.columns:
            adv = _safe_numeric(g["adv_dollar_20"]).clip(lower=1.0).fillna(_DEFAULT_ADV)
            log_adv = np.log(adv)
            factors.append(log_adv - float(log_adv.mean()))

        if use_vol and "realised_vol_20d" in g.columns:
            vol = _safe_numeric(g["realised_vol_20d"]).fillna(_DEFAULT_VOL).clip(0.0, 1.0)
            factors.append(vol - float(vol.mean()))

        y_centered = safe_center(y, cap=max_abs)[:, 0]

        if not factors:
            out.loc[g.index] = y_centered
            continue

        x = pd.concat(factors, axis=1).to_numpy(dtype=float)
        x = np.nan_to_num(x, nan=0.0)
        result = residualize_against_controls(y_centered, x, ridge=ridge, value_cap=max_abs)
        out.loc[g.index] = np.asarray(result.residual, dtype=float)

    return out.fillna(0.0)


def build_target_menu(
    df: pd.DataFrame,
    *,
    costs: ExecutionCostConfig,
    max_name_weight: float,
    ridge: float = _RIDGE,
    max_abs_return: float = _MAX_ABS,
    winsor_q: float = 0.01,
) -> pd.DataFrame:
    """
    Append all seven target columns to df.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with date, ticker.  Needs daily_return (preferred) or forward_return
        for raw_5d/raw_10d.  Optional columns: capm_beta, sector, adv_dollar_20,
        realised_vol_20d, spy_forward_5d.
    costs : ExecutionCostConfig
        Transaction cost model for cost_adjusted target.
    max_name_weight : float
        Max position weight, used in cost estimation.

    Returns
    -------
    pd.DataFrame with added columns:
        raw_5d, raw_10d, market_residual_5d, sector_residual_5d,
        factor_residual_5d, vol_scaled_5d, cost_adjusted_5d
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # ── Raw forward returns ───────────────────────────────────────────────────
    raw5 = _forward_return_from_daily(out, 5).clip(-max_abs_return, max_abs_return)
    raw10 = _forward_return_from_daily(out, 10).clip(-max_abs_return, max_abs_return)
    raw20 = _forward_return_from_daily(out, 20).clip(-max_abs_return, max_abs_return)
    raw63 = _forward_return_from_daily(out, 63).clip(-max_abs_return, max_abs_return)
    out[RAW_5D] = raw5
    out[RAW_10D] = raw10
    out[RAW_20D] = raw20
    out[RAW_63D] = raw63

    raw5_w = _cs_winsorize(raw5, out["date"], winsor_q)
    raw10_w = _cs_winsorize(raw10, out["date"], winsor_q)
    raw20_w = _cs_winsorize(raw20, out["date"], winsor_q)
    raw63_w = _cs_winsorize(raw63, out["date"], winsor_q)

    # ── Market-residual: remove CAPM beta only ────────────────────────────────
    out[MARKET_RESIDUAL] = _residualize_panel_by_date(
        out, raw5_w,
        use_market=True,
        ridge=ridge, max_abs=max_abs_return,
    )

    # ── Sector-residual: remove sector means only ─────────────────────────────
    out[SECTOR_RESIDUAL] = _residualize_panel_by_date(
        out, raw5_w,
        use_sector=True,
        ridge=ridge, max_abs=max_abs_return,
    )

    # ── Factor-residual: market + sector + size + vol ─────────────────────────
    out[FACTOR_RESIDUAL] = _residualize_panel_by_date(
        out, raw5_w,
        use_market=True, use_sector=True, use_size=True, use_vol=True,
        ridge=ridge, max_abs=max_abs_return,
    )

    # ── Vol-scaled: raw_5d / (σ × √5), demeaned per date ─────────────────────
    if "realised_vol_20d" in out.columns:
        vol = _safe_numeric(out["realised_vol_20d"]).clip(lower=1e-4).fillna(_DEFAULT_VOL)
        vol_sc = (raw5 / (vol * math.sqrt(5))).replace([np.inf, -np.inf], np.nan).clip(-5.0, 5.0)
    else:
        vol_sc = raw5_w.copy()
    vol_sc = vol_sc.groupby(out["date"]).transform(lambda x: x - x.mean())
    out[VOL_SCALED] = vol_sc.fillna(0.0)

    # ── Cost-adjusted: raw_5d − round-trip cost, demeaned ────────────────────
    rtc = _expected_round_trip_cost(
        out, costs=costs, horizon_days=5, max_name_weight=float(max_name_weight)
    )
    cost_adj = (raw5 - rtc).clip(-max_abs_return, max_abs_return)
    cost_adj = cost_adj.groupby(out["date"]).transform(lambda x: x - x.mean())
    out[COST_ADJUSTED] = cost_adj.fillna(0.0)

    # ── 10-day residualized targets (medium-term momentum features) ──────────
    # Mathematical basis: diag_optimal_horizon=10 across all models. Features
    # (ret_5d, ret_10d, cs_momentum_percentile, nearness_52w_high) have signal
    # halflives of 5-12 days. 10d provides ~500 non-overlapping obs (vs ~100 at 20d)
    # while still capturing the medium-term reversal-to-continuation transition.
    out[MARKET_RESIDUAL_10D] = _residualize_panel_by_date(
        out, raw10_w,
        use_market=True,
        ridge=ridge, max_abs=max_abs_return,
    )
    out[SECTOR_RESIDUAL_10D] = _residualize_panel_by_date(
        out, raw10_w,
        use_sector=True,
        ridge=ridge, max_abs=max_abs_return,
    )
    out[FACTOR_RESIDUAL_10D] = _residualize_panel_by_date(
        out, raw10_w,
        use_market=True, use_sector=True, use_size=True, use_vol=True,
        ridge=ridge, max_abs=max_abs_return,
    )

    # ── 20-day residualized targets (momentum / intermediate features) ────────
    # Mathematical basis: momentum_12m_skip1, sector_relative_60d, capm_alpha
    # all have economic signal horizons of 20-63 days. Evaluating them against
    # a 5-day target creates signal horizon mismatch — IC underestimates true
    # predictive power by sqrt(20/5) ≈ 2× due to overlapping return noise.
    out[MARKET_RESIDUAL_20D] = _residualize_panel_by_date(
        out, raw20_w,
        use_market=True,
        ridge=ridge, max_abs=max_abs_return,
    )
    out[SECTOR_RESIDUAL_20D] = _residualize_panel_by_date(
        out, raw20_w,
        use_sector=True,
        ridge=ridge, max_abs=max_abs_return,
    )
    out[FACTOR_RESIDUAL_20D] = _residualize_panel_by_date(
        out, raw20_w,
        use_market=True, use_sector=True, use_size=True, use_vol=True,
        ridge=ridge, max_abs=max_abs_return,
    )

    # ── 63-day residualized targets (fundamental features, quarterly cycle) ───
    # Mathematical basis: fundamental features (accruals, leverage, margins)
    # are published quarterly (≈63 trading days). Their IC versus a 5-day target
    # is near zero because the signal requires a full earnings cycle to realise.
    # IC against 63d target ≈ IC_5d × sqrt(63/5) ≈ 3.5× larger (Grinold-Kahn).
    out[MARKET_RESIDUAL_63D] = _residualize_panel_by_date(
        out, raw63_w,
        use_market=True,
        ridge=ridge, max_abs=max_abs_return,
    )
    out[SECTOR_RESIDUAL_63D] = _residualize_panel_by_date(
        out, raw63_w,
        use_sector=True,
        ridge=ridge, max_abs=max_abs_return,
    )
    out[FACTOR_RESIDUAL_63D] = _residualize_panel_by_date(
        out, raw63_w,
        use_market=True, use_sector=True, use_size=True, use_vol=True,
        ridge=ridge, max_abs=max_abs_return,
    )

    return out


def _long_quintile_sharpe(
    df: pd.DataFrame,
    proxy_col: str,
    raw_return_col: str,
    *,
    holding_days: int = 5,
    min_obs: int = 20,
) -> float:
    """
    Annualized Sharpe of going long top quintile by proxy, realizing raw_return_col.

    One observation per date: average return of tickers in the top 20% by proxy.
    Annualizes assuming holding_days-day holding periods (non-overlapping).
    """
    work = df[["date", proxy_col, raw_return_col]].dropna()
    if work.empty:
        return float("nan")

    period_rets: list[float] = []
    for _, g in work.groupby("date", sort=True):
        if len(g) < 10:
            continue
        threshold = g[proxy_col].quantile(0.80)
        top = g.loc[g[proxy_col] >= threshold, raw_return_col]
        if len(top) < 2:
            continue
        period_rets.append(float(top.mean()))

    if len(period_rets) < min_obs:
        return float("nan")

    arr = np.asarray(period_rets, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < min_obs:
        return float("nan")

    mu = float(arr.mean())
    sigma = float(arr.std(ddof=1))
    if sigma < _EPS:
        return float("nan")

    annual_periods = 252.0 / max(holding_days, 1)
    return float((mu / sigma) * math.sqrt(annual_periods))


def _evaluate_one_target(
    df: pd.DataFrame,
    target_col: str,
    proxy_features: list[str],
    *,
    raw_return_col: str = RAW_5D,
    sign: int = 1,
    min_days: int = 20,
) -> dict[str, float]:
    """Compute all evaluation metrics for one target column."""
    if target_col not in df.columns:
        return {k: float("nan") for k in ("ic_mean", "ic_tstat", "ic_stability",
                                            "decile_spread", "monotonicity", "long_sharpe")}

    available = [f for f in proxy_features if f in df.columns]
    if not available:
        return {k: float("nan") for k in ("ic_mean", "ic_tstat", "ic_stability",
                                            "decile_spread", "monotonicity", "long_sharpe")}

    ic_means: list[float] = []
    ic_tstats: list[float] = []
    ic_stabilities: list[float] = []
    decile_spreads: list[float] = []
    monotonicity_vals: list[float] = []
    long_sharpes: list[float] = []

    for proxy in available:
        ic_vals = _daily_ic_values(df, proxy, target_col)
        if len(ic_vals) >= min_days:
            mu = float(np.nanmean(ic_vals))
            std = float(np.nanstd(ic_vals, ddof=1))
            ic_means.append(mu)
            ic_tstats.append(_hac_tstat(ic_vals))
            ic_stabilities.append(abs(mu) / std if std > _EPS else float("nan"))

        spread, mono = _decile_shape(df, proxy, target_col, sign=sign)
        if math.isfinite(spread):
            decile_spreads.append(spread)
        if math.isfinite(mono):
            monotonicity_vals.append(mono)

        if raw_return_col in df.columns:
            sharpe = _long_quintile_sharpe(df, proxy, raw_return_col)
            if math.isfinite(sharpe):
                long_sharpes.append(sharpe)

    def _nanmean_list(vals: list[float]) -> float:
        finite = [v for v in vals if math.isfinite(v)]
        return float(np.mean(finite)) if finite else float("nan")

    return {
        "ic_mean": _nanmean_list(ic_means),
        "ic_tstat": _nanmean_list(ic_tstats),
        "ic_stability": _nanmean_list(ic_stabilities),
        "decile_spread": _nanmean_list(decile_spreads),
        "monotonicity": _nanmean_list(monotonicity_vals),
        "long_sharpe": _nanmean_list(long_sharpes),
    }


def _composite_score(row: pd.Series) -> float:
    """
    Weighted composite across four independent metrics.

    Weights reflect relative importance for alpha capture:
      - IC t-stat (0.35):  statistical reliability of signal prediction
      - long_sharpe (0.30): realized return quality — most policy-relevant
      - monotonicity (0.20): cross-sectional structure (spread linearity)
      - ic_stability (0.15): consistency over time (ICIR)
    """
    weights = {
        "ic_tstat": 0.35,
        "long_sharpe": 0.30,
        "monotonicity": 0.20,
        "ic_stability": 0.15,
    }
    score = 0.0
    total_w = 0.0
    for key, w in weights.items():
        v = float(row.get(key, float("nan")))
        if math.isfinite(v):
            score += w * v
            total_w += w
    return score / total_w if total_w > 0.5 else float("nan")


def evaluate_targets(
    df: pd.DataFrame,
    proxy_features: list[str],
    *,
    costs: ExecutionCostConfig,
    max_name_weight: float,
    ridge: float = _RIDGE,
    max_abs_return: float = _MAX_ABS,
    winsor_q: float = 0.01,
    min_days: int = 20,
) -> pd.DataFrame:
    """
    Build the full target menu and return a diagnostics table sorted by composite score.

    Parameters
    ----------
    df : pd.DataFrame
        Panel with date, ticker, and ideally: daily_return, capm_beta, sector,
        adv_dollar_20, realised_vol_20d, spy_forward_5d.
    proxy_features : list[str]
        Existing alpha signals to use as predictors in IC/decile evaluation.
        Typical: ["f_trend", "ret_5d", "cs_momentum_percentile"].
        All features must already be columns in df.
    costs : ExecutionCostConfig
        Cost model for cost_adjusted target.
    max_name_weight : float
        Max position weight for cost estimation.

    Returns
    -------
    pd.DataFrame with columns:
        rank, target_type, ic_mean, ic_tstat, ic_stability,
        decile_spread, monotonicity, long_sharpe, composite_score
    Sorted by composite_score descending (rank 1 = best).
    """
    panel = build_target_menu(
        df,
        costs=costs,
        max_name_weight=max_name_weight,
        ridge=ridge,
        max_abs_return=max_abs_return,
        winsor_q=winsor_q,
    )

    rows: list[dict] = []
    for target_type in TARGET_MENU:
        metrics = _evaluate_one_target(
            panel,
            target_type,
            proxy_features,
            raw_return_col=RAW_5D,
            sign=1,
            min_days=min_days,
        )
        metrics["target_type"] = target_type
        rows.append(metrics)

    col_order = ["target_type", "ic_mean", "ic_tstat", "ic_stability",
                 "decile_spread", "monotonicity", "long_sharpe"]
    result = pd.DataFrame(rows)[col_order]
    result["composite_score"] = result.apply(_composite_score, axis=1)
    result = result.sort_values("composite_score", ascending=False).reset_index(drop=True)
    result.insert(0, "rank", range(1, len(result) + 1))
    return result


def format_diagnostics_table(table: pd.DataFrame) -> str:
    """Render the diagnostics table as a fixed-width text report."""
    if table.empty:
        return "No target diagnostics available."

    sep = "=" * 88
    lines = [
        "Target Construction Diagnostics",
        sep,
        f"{'Rk':>2}  {'Target Type':<25}  {'IC Mean':>8}  {'IC t-stat':>9}  "
        f"{'IC IR':>7}  {'D10 Sprd':>8}  {'Mono':>6}  {'L/O Sharpe':>10}  {'Score':>7}",
        "-" * 88,
    ]

    def _f(v: object) -> str:
        if isinstance(v, float) and math.isfinite(v):
            return f"{v:8.4f}"
        return "     NaN"

    for _, row in table.iterrows():
        lines.append(
            f"{int(row.get('rank', 0)):>2}  {str(row['target_type']):<25}  "
            f"{_f(row.get('ic_mean'))}  "
            f"{_f(row.get('ic_tstat'))}  "
            f"{_f(row.get('ic_stability'))}  "
            f"{_f(row.get('decile_spread'))}  "
            f"{_f(row.get('monotonicity'))}  "
            f"{_f(row.get('long_sharpe'))}  "
            f"{_f(row.get('composite_score'))}"
        )

    lines += [
        sep,
        "IC Mean     = mean daily rank-IC(proxy_feature, target)",
        "IC t-stat   = Newey-West HAC t-stat on IC series",
        "IC IR       = |IC mean| / IC std  (Information Ratio — higher = more consistent)",
        "D10 Spread  = mean(top decile target) − mean(bottom decile target) using proxy ranks",
        "Mono        = fraction of adjacent proxy deciles with correct sign on target",
        "L/O Sharpe  = annualized Sharpe of top-quintile portfolio (by proxy) on raw_5d",
        "Score       = 0.35×IC-tstat + 0.30×L/O-Sharpe + 0.20×Mono + 0.15×IC-IR",
    ]
    return "\n".join(lines)
