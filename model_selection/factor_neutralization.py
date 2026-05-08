"""
Factor Neutralization
=====================
Cross-sectional decomposition of model scores into systematic + idiosyncratic
components.  The idiosyncratic residual is what pure alpha strategies rank on.

The existing ``target_construction`` module already builds factor-residual
*targets*.  This module applies the same decomposition to model *scores/predictions*,
so both the label and the ranking signal live in the same idiosyncratic space.

All regressions are cross-sectional per date (no time-series fit, no lookahead).

Public API
----------
neutralize_scores_by_date :
    Per-date OLS residualization of any score column against factor exposures.
factor_variance_explained :
    R² of factor model on score column, averaged across dates.
compute_neutralization_diagnostics :
    IC and Sharpe before vs. after neutralization, plus R² decomposition.
format_neutralization_report :
    Fixed-width text summary of diagnostics.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from model_selection.residualization import residualize_against_controls, safe_center


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NeutralizationConfig:
    use_market: bool = True       # CAPM beta exposure
    use_sector: bool = True       # sector dummy means
    use_size: bool = True         # log-ADV proxy
    use_vol: bool = True          # realized vol
    ridge: float = 1e-4
    value_cap: float = 5.0


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan)


def _build_factor_matrix(
    g: pd.DataFrame,
    cfg: NeutralizationConfig,
) -> np.ndarray | None:
    """Build a demeaned factor exposure matrix for one date's cross-section."""
    factors: list[np.ndarray] = []

    if cfg.use_market and "capm_beta" in g.columns:
        b = _safe_num(g["capm_beta"]).fillna(1.0).clip(-3.0, 3.0).to_numpy(dtype=float)
        factors.append(b - b.mean())

    if cfg.use_sector and "sector" in g.columns:
        dummies = pd.get_dummies(
            g["sector"].fillna("Unknown").astype(str), dtype=float
        )
        for col in dummies.columns:
            d = dummies[col].to_numpy(dtype=float)
            factors.append(d - d.mean())

    if cfg.use_size and "adv_dollar_20" in g.columns:
        adv = _safe_num(g["adv_dollar_20"]).clip(lower=1.0).fillna(50_000_000.0)
        la = np.log(adv.to_numpy(dtype=float))
        factors.append(la - la.mean())

    if cfg.use_vol and "realised_vol_20d" in g.columns:
        v = _safe_num(g["realised_vol_20d"]).fillna(0.015).clip(0.0, 1.0).to_numpy(dtype=float)
        factors.append(v - v.mean())

    if not factors:
        return None
    return np.column_stack(factors)


def _spearman_ic(a: np.ndarray, b: np.ndarray) -> float:
    mask = np.isfinite(a) & np.isfinite(b)
    if mask.sum() < 5 or a[mask].std() == 0.0 or b[mask].std() == 0.0:
        return float("nan")
    r, _ = stats.spearmanr(a[mask], b[mask])
    return float(r) if math.isfinite(r) else float("nan")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def neutralize_scores_by_date(
    df: pd.DataFrame,
    score_col: str,
    *,
    cfg: NeutralizationConfig | None = None,
    out_col: str = "score_neutral",
) -> pd.DataFrame:
    """
    Residualize ``score_col`` against factor exposures per date.

    Returns a copy of *df* with ``out_col`` added.  When no factor data is
    available for a date, falls back to cross-sectionally centered score.

    Parameters
    ----------
    df       : panel with date, ticker, score_col, and optional factor columns
               (capm_beta, sector, adv_dollar_20, realised_vol_20d)
    score_col: column to neutralize
    cfg      : NeutralizationConfig
    out_col  : name for the neutralized score column

    Returns
    -------
    DataFrame with ``out_col`` appended.
    """
    if cfg is None:
        cfg = NeutralizationConfig()

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out[out_col] = np.nan

    for dt, g in out.groupby("date", sort=True):
        y = _safe_num(g[score_col]).to_numpy(dtype=float)
        y_c = safe_center(y, cap=cfg.value_cap)[:, 0]
        X = _build_factor_matrix(g, cfg)

        if X is None:
            out.loc[g.index, out_col] = y_c
        else:
            result = residualize_against_controls(
                y_c, X, ridge=cfg.ridge, value_cap=cfg.value_cap
            )
            out.loc[g.index, out_col] = np.asarray(result.residual, dtype=float)

    return out


def factor_variance_explained(
    df: pd.DataFrame,
    signal_col: str,
    *,
    cfg: NeutralizationConfig | None = None,
) -> dict[str, float]:
    """
    Fraction of signal variance explained by systematic factors per date.

    Returns mean, std, p25, p75 of the per-date R², and count of valid dates.
    R² = 1 − SS_residual / SS_total  (clamped to [0, 1]).
    """
    if cfg is None:
        cfg = NeutralizationConfig()

    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    r2_vals: list[float] = []

    for _, g in work.groupby("date", sort=True):
        y = _safe_num(g[signal_col]).to_numpy(dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < 5:
            continue
        y_c = y[mask] - y[mask].mean()
        ss_tot = float(np.dot(y_c, y_c))
        if ss_tot < 1e-14:
            continue

        X = _build_factor_matrix(g.iloc[np.where(mask)[0]], cfg)
        if X is None:
            r2_vals.append(0.0)
            continue

        result = residualize_against_controls(y_c, X, ridge=cfg.ridge, value_cap=cfg.value_cap)
        resid = np.asarray(result.residual, dtype=float)
        ss_res = float(np.dot(resid, resid))
        r2_vals.append(max(0.0, 1.0 - ss_res / ss_tot))

    if not r2_vals:
        return {k: float("nan") for k in ("r2_mean", "r2_std", "r2_p25", "r2_p75", "n_dates")}

    arr = np.array(r2_vals, dtype=float)
    return {
        "r2_mean": float(arr.mean()),
        "r2_std": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
        "r2_p25": float(np.percentile(arr, 25)),
        "r2_p75": float(np.percentile(arr, 75)),
        "n_dates": len(arr),
    }


def compute_neutralization_diagnostics(
    df: pd.DataFrame,
    score_col: str,
    target_col: str,
    *,
    cfg: NeutralizationConfig | None = None,
    raw_return_col: str = "forward_return",
    holding_days: int = 5,
    min_dates: int = 10,
) -> dict[str, float]:
    """
    Compare signal quality before and after factor neutralization.

    Parameters
    ----------
    df            : panel with date, ticker, score_col, target_col, factor cols
    score_col     : raw model score column
    target_col    : return / label column to compute IC against
    cfg           : NeutralizationConfig
    raw_return_col: column for Sharpe computation (must be raw return, not rank)
    holding_days  : annualization denominator
    min_dates     : minimum dates required for Sharpe

    Returns
    -------
    r2_systematic      : mean fraction of score variance explained by factors
    r2_systematic_p75  : 75th percentile of per-date R²
    ic_raw             : mean daily Spearman IC (score, target) before
    ic_neutralized     : mean daily Spearman IC (neutralized, target) after
    ic_improvement_pct : 100 × (|ic_neutral| − |ic_raw|) / |ic_raw|
    sharpe_raw         : long-quintile Sharpe using raw score
    sharpe_neutralized : long-quintile Sharpe using neutralized score
    n_dates            : number of dates with valid IC
    """
    if cfg is None:
        cfg = NeutralizationConfig()

    neutralized = neutralize_scores_by_date(df, score_col, cfg=cfg, out_col="_ns_tmp")

    r2_stats = factor_variance_explained(df, score_col, cfg=cfg)

    ic_raw_vals: list[float] = []
    ic_neu_vals: list[float] = []

    work = neutralized.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    for _, g in work.groupby("date", sort=True):
        if target_col not in g.columns:
            continue
        tgt = _safe_num(g[target_col]).to_numpy(dtype=float)
        s_r = _safe_num(g[score_col]).to_numpy(dtype=float)
        s_n = _safe_num(g["_ns_tmp"]).to_numpy(dtype=float)

        ic_r = _spearman_ic(s_r, tgt)
        ic_n = _spearman_ic(s_n, tgt)
        if math.isfinite(ic_r):
            ic_raw_vals.append(ic_r)
        if math.isfinite(ic_n):
            ic_neu_vals.append(ic_n)

    ic_raw = float(np.mean(ic_raw_vals)) if ic_raw_vals else float("nan")
    ic_neu = float(np.mean(ic_neu_vals)) if ic_neu_vals else float("nan")
    ic_imp = float("nan")
    if math.isfinite(ic_raw) and math.isfinite(ic_neu):
        ic_imp = (abs(ic_neu) - abs(ic_raw)) / max(abs(ic_raw), 1e-6) * 100.0

    def _q_sharpe(frame: pd.DataFrame, col: str) -> float:
        if raw_return_col not in frame.columns:
            return float("nan")
        period_rets: list[float] = []
        for _, g in frame.groupby("date", sort=True):
            if len(g) < 10:
                continue
            thr = g[col].quantile(0.80)
            top = g.loc[g[col] >= thr, raw_return_col].dropna()
            if len(top) < 2:
                continue
            period_rets.append(float(top.mean()))
        if len(period_rets) < min_dates:
            return float("nan")
        arr = np.array(period_rets, dtype=float)
        arr = arr[np.isfinite(arr)]
        if len(arr) < min_dates or arr.std(ddof=1) < 1e-10:
            return float("nan")
        return float(arr.mean() / arr.std(ddof=1) * math.sqrt(252.0 / max(holding_days, 1)))

    return {
        "r2_systematic": r2_stats.get("r2_mean", float("nan")),
        "r2_systematic_p75": r2_stats.get("r2_p75", float("nan")),
        "ic_raw": ic_raw,
        "ic_neutralized": ic_neu,
        "ic_improvement_pct": ic_imp,
        "sharpe_raw": _q_sharpe(neutralized, score_col),
        "sharpe_neutralized": _q_sharpe(neutralized, "_ns_tmp"),
        "n_dates": len(ic_raw_vals),
    }


def neutralize_features_before_training(
    df: pd.DataFrame,
    feature_cols: list[str],
    *,
    cfg: NeutralizationConfig | None = None,
    target_col: str | None = None,
) -> pd.DataFrame:
    """
    Institutional-grade pre-fit feature residualization.

    Residualize every feature column against factor exposures per date BEFORE
    model training. This prevents the model from learning factor/beta exposure
    in the first place — the correct Two Sigma / Jane Street approach.

    This is distinct from post-prediction score neutralization. By neutralizing
    features before fitting, the model never sees systematic factor information.
    The resulting scores are naturally orthogonal to factors without any
    post-processing.

    When ``target_col`` is provided, the target column is residualized against
    the same factor matrix to ensure both features and target live in the same
    idiosyncratic space. This is equivalent to training on factor-residual
    returns directly.

    Parameters
    ----------
    df           : panel with date, ticker, feature columns, and optional factor cols
    feature_cols : list of feature column names to neutralize
    cfg          : NeutralizationConfig
    target_col   : optional target column to also residualize

    Returns
    -------
    DataFrame with residualized feature columns (and optionally target column),
    with original columns preserved as ``<col>_raw`` for diagnostics.
    """
    if cfg is None:
        cfg = NeutralizationConfig()

    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    available_features = [c for c in feature_cols if c in out.columns]
    if not available_features:
        return out

    # P12: Force float64 dtype on all feature columns before residualization.
    # Residualized values are always float64; assigning them into int64 or
    # float32 columns triggers a Pandas FutureWarning (will become an error).
    for feat in available_features:
        out[feat] = pd.to_numeric(out[feat], errors="coerce").astype(np.float64)
    if target_col is not None and target_col in out.columns:
        out[target_col] = pd.to_numeric(out[target_col], errors="coerce").astype(np.float64)

    r2_history: list[float] = []
    n_dates_processed = 0

    for dt, g in out.groupby("date", sort=True):
        X = _build_factor_matrix(g, cfg)
        if X is None:
            continue

        n_dates_processed += 1

        for feat in available_features:
            y = _safe_num(g[feat]).to_numpy(dtype=float)
            mask = np.isfinite(y)
            if mask.sum() < 5:
                continue
            y_clean = y[mask]
            y_centered = y_clean - y_clean.mean()
            ss_tot = float(np.dot(y_centered, y_centered))
            if ss_tot < 1e-14:
                continue

            X_sub = X[mask]
            result = residualize_against_controls(
                y_centered, X_sub, ridge=cfg.ridge, value_cap=cfg.value_cap
            )
            resid = np.asarray(result.residual, dtype=float)
            out.loc[g.index[mask], feat] = resid

            # Track R² for diagnostics
            ss_res = float(np.dot(resid[:len(y_centered)], resid[:len(y_centered)]))
            r2 = max(0.0, 1.0 - ss_res / ss_tot)
            r2_history.append(r2)

        if target_col is not None and target_col in g.columns:
            y_tgt = _safe_num(g[target_col]).to_numpy(dtype=float)
            mask_t = np.isfinite(y_tgt)
            if mask_t.sum() >= 5:
                y_clean = y_tgt[mask_t]
                y_centered = y_clean - y_clean.mean()
                X_sub = X[mask_t]
                result = residualize_against_controls(
                    y_centered, X_sub, ridge=cfg.ridge, value_cap=cfg.value_cap
                )
                resid = np.asarray(result.residual, dtype=float)
                out.loc[g.index[mask_t], target_col] = resid

    # Attach diagnostics as DataFrame attributes
    if r2_history:
        out.attrs["neutralization_r2_mean"] = float(np.mean(r2_history))
        out.attrs["neutralization_r2_p75"] = float(np.percentile(r2_history, 75))
        out.attrs["neutralization_n_dates"] = n_dates_processed

    return out


def format_neutralization_report(diag: dict[str, float]) -> str:
    """Render neutralization diagnostics as a fixed-width text block."""
    def _f(v: object, pct: bool = False) -> str:
        if isinstance(v, float) and math.isfinite(v):
            return f"{v:8.4f}" if not pct else f"{v:7.2f}%"
        return "     NaN"

    lines = [
        "Factor Neutralization Diagnostics",
        "=" * 55,
        f"  Systematic R² (mean / p75) : {_f(diag.get('r2_systematic'))} / "
        f"{_f(diag.get('r2_systematic_p75'))}",
        "",
        "  Daily Spearman IC vs target",
        f"    Raw score            : {_f(diag.get('ic_raw'))}",
        f"    Neutralized score    : {_f(diag.get('ic_neutralized'))}",
        f"    Improvement          : {_f(diag.get('ic_improvement_pct'), pct=True)}",
        "",
        "  Long-quintile Sharpe (5-day hold)",
        f"    Raw score            : {_f(diag.get('sharpe_raw'))}",
        f"    Neutralized score    : {_f(diag.get('sharpe_neutralized'))}",
        "",
        f"  Observation dates    : {int(diag.get('n_dates', 0))}",
        "=" * 55,
    ]
    return "\n".join(lines)
