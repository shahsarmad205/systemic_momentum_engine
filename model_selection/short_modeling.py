"""
Short-side alpha modeling framework.

Five short target definitions (each captures a different source of short alpha):
  short_neg_residual   : negated factor-residualized 5d forward return
  short_vol_expansion  : negative return weighted by vol expansion signal
  short_liq_drain      : negative return weighted by ADV deterioration signal
  short_failed_mom     : negative return weighted by past positive momentum rank
  short_downside_skew  : negative return weighted by negative skewness rank

All targets are cross-section z-scored per date. No target embeds execution costs.

Short model training: pure alpha score models fit short-specific labels. The
  portfolio short book is constructed downstream; training does not embed borrow,
  costs, turnover, or soft portfolio weights.

Short-leg evaluation:
  IC:             rank_corr(score, short_target) per date
  Decile spread:  top-score decile return - bottom-score decile return
  Monotonicity:   Spearman rank of decile returns vs decile order
  Short Sharpe:   annualized Sharpe of daily short portfolio (top-q by score, shorted)
  Exposure frac:  IC on raw return vs IC on factor residual

Classification per target:
  alpha_short      : residual IC significant AND residual_frac >= alpha_frac_threshold
  hedge_only       : raw IC significant but residual_frac < alpha_frac_threshold
  exposure_artifact: raw IC significant and residual_frac < exposure_frac_threshold

Constraints:
  - NO ExecutionCostConfig (execution not modeled here)
  - NO assumed symmetry with long side (separate targets, separate gradient)
"""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, replace
from typing import Any


# ── Score convention ──────────────────────────────────────────────────────────
# HIGHER score = STRONGER short candidate.
#
# Rationale: All short targets are negated return signals (z-scored per date).
#   IC(score, short_target)   > 0 for a working model (positive correlation).
#   IC(score, raw_5d)         < 0 for a working model (raw return not negated).
#   IC(score, residual_5d)    < 0 for a working model (residual not negated).
#   _short_leg_sharpe: shorts top-q by score (highest-score = strongest short).
#
# Pure short-alpha score convention:
#   Stocks with larger short targets should receive higher model scores.
#   Portfolio construction may map those scores into short weights later, but
#   that mapping is not part of the supervised-learning objective.
SHORT_SCORE_CONVENTION: str = "higher_score_is_stronger_short"
_SHORT_TARGET_EXPECTED_IC_SIGN: int = 1  # IC(score, short_target) positive for working model


# ── Configuration ─────────────────────────────────────────────────────────────

@dataclass
class ShortTargetConfig:
    return_col: str = "daily_return"
    adv_col: str = "adv_dollar_20"
    vol_col: str = "realised_vol_20d"
    factor_residual_col: str = "factor_residual_5d"
    raw_5d_col: str = "raw_5d"
    mom_lookback: int = 20        # target_construction_parameter
    skew_lookback: int = 20       # target_construction_parameter
    vol_expansion_lag: int = 40   # target_construction_parameter
    adv_drain_lag: int = 40       # target_construction_parameter
    max_abs_target: float = 0.10  # numerical_guardrail: clip target magnitude
    winsor_q: float = 0.01        # target_construction_parameter


@dataclass
class ShortEvalConfig:
    raw_return_col: str = "raw_5d"
    residual_col: str = "factor_residual_5d"
    short_top_q: float = 0.20          # reporting_policy: top-q score → short portfolio
    ic_min_stocks: int = 8             # numerical_guardrail: minimum stocks per date
    ic_tstat_threshold: float = 1.5    # research_selection_threshold (should_be_adaptive)
    ic_mean_threshold: float = 0.01    # research_selection_threshold (should_be_adaptive)
    alpha_frac_threshold: float = 0.60 # research_selection_threshold (should_be_adaptive)
    exposure_frac_threshold: float = 0.40  # research_selection_threshold (should_be_adaptive)


# ── Short target names ────────────────────────────────────────────────────────

SHORT_NEG_RESIDUAL   = "short_neg_residual"
SHORT_VOL_EXPANSION  = "short_vol_expansion"
SHORT_LIQ_DRAIN      = "short_liq_drain"
SHORT_FAILED_MOM     = "short_failed_mom"
SHORT_DOWNSIDE_SKEW  = "short_downside_skew"
ALL_SHORT_TARGETS    = [
    SHORT_NEG_RESIDUAL, SHORT_VOL_EXPANSION, SHORT_LIQ_DRAIN,
    SHORT_FAILED_MOM,   SHORT_DOWNSIDE_SKEW,
]


# ── Short target construction ─────────────────────────────────────────────────

def _cs_zscore(s: pd.Series, date_col: pd.Series) -> pd.Series:
    """Cross-sectional z-score per date, clamped to ±3."""
    def _zsc(x: pd.Series) -> pd.Series:
        sigma = x.std(ddof=1)
        return ((x - x.mean()) / sigma).clip(-3.0, 3.0) if sigma > 1e-10 else pd.Series(0.0, index=x.index)

    return s.groupby(date_col, sort=False).transform(_zsc)


def _cs_rank_01(s: pd.Series, date_col: pd.Series) -> pd.Series:
    """Cross-sectional rank normalized to [0, 1] per date."""
    return s.groupby(date_col).transform(lambda x: x.rank(pct=True, na_option="keep"))


def _winsorize(s: pd.Series, q: float) -> pd.Series:
    lo, hi = s.quantile(q), s.quantile(1 - q)
    return s.clip(lower=lo, upper=hi)


def build_short_targets(
    df: pd.DataFrame,
    cfg: ShortTargetConfig,
) -> pd.DataFrame:
    """
    Append 5 short-specific target columns to df.

    All targets are cross-section z-scored per date. Rolling features are
    computed per ticker (sorted by date within each ticker group).

    Requires columns: raw_5d (or daily_return), date, ticker.
    Optional: factor_residual_5d, realised_vol_20d, adv_dollar_20.
    """
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")

    # Sort by ticker then date for rolling computations
    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    raw_5d = (
        pd.to_numeric(out[cfg.raw_5d_col], errors="coerce")
        if cfg.raw_5d_col in out.columns
        else pd.Series(np.nan, index=out.index)
    )

    # ── 1. Negative residual return ──────────────────────────────────────────
    if cfg.factor_residual_col in out.columns:
        resid = pd.to_numeric(out[cfg.factor_residual_col], errors="coerce")
        out[SHORT_NEG_RESIDUAL] = _cs_zscore(-resid, out["date"])
    else:
        out[SHORT_NEG_RESIDUAL] = _cs_zscore(-_winsorize(raw_5d, cfg.winsor_q), out["date"])

    # ── 2. Vol expansion — amplify short when volatility is rising ───────────
    neg_raw_z = _cs_zscore(-_winsorize(raw_5d, cfg.winsor_q), out["date"])

    if cfg.vol_col in out.columns:
        vol = pd.to_numeric(out[cfg.vol_col], errors="coerce")
        vol_num = vol  # pre-converted; reuse in groupby lambda
        vol_lag = vol_num.groupby(out["ticker"], sort=False).transform(
            lambda x: x.shift(cfg.vol_expansion_lag)
        )
        vol_expansion = (vol / vol_lag.clip(lower=1e-6) - 1).fillna(0.0).clip(-2.0, 2.0)
        vol_rank = _cs_rank_01(vol_expansion, out["date"])
        out[SHORT_VOL_EXPANSION] = _cs_zscore(
            neg_raw_z * (0.5 + vol_rank), out["date"]
        )
    else:
        out[SHORT_VOL_EXPANSION] = neg_raw_z.copy()

    # ── 3. Liquidity drain — amplify short when ADV is drying up ────────────
    if cfg.adv_col in out.columns:
        adv = pd.to_numeric(out[cfg.adv_col], errors="coerce").clip(lower=1e3)
        adv_lag = adv.groupby(out["ticker"], sort=False).transform(
            lambda x: x.clip(lower=1e3).shift(cfg.adv_drain_lag)
        )
        # Drain: ADV declining = positive drain signal
        adv_drain = -(adv / adv_lag.clip(lower=1e3) - 1).fillna(0.0).clip(-2.0, 2.0)
        drain_rank = _cs_rank_01(adv_drain, out["date"])
        out[SHORT_LIQ_DRAIN] = _cs_zscore(
            neg_raw_z * (0.5 + drain_rank), out["date"]
        )
    else:
        out[SHORT_LIQ_DRAIN] = neg_raw_z.copy()

    # ── 4. Failed momentum — amplify short for stocks with prior up-momentum ─
    if cfg.return_col in out.columns:
        daily_ret_num = pd.to_numeric(out[cfg.return_col], errors="coerce")
        past_mom = daily_ret_num.groupby(out["ticker"], sort=False).transform(
            lambda x: x.rolling(cfg.mom_lookback, min_periods=cfg.mom_lookback // 2).sum()
        )
        # High past_mom = was a momentum winner = failed momentum reversal candidate
        mom_rank = _cs_rank_01(past_mom.fillna(0.0), out["date"])
        out[SHORT_FAILED_MOM] = _cs_zscore(
            neg_raw_z * (0.5 + mom_rank), out["date"]
        )
    else:
        out[SHORT_FAILED_MOM] = neg_raw_z.copy()

    # ── 5. Downside skew — amplify short for negatively skewed stocks ────────
    if cfg.return_col in out.columns:
        ret_num = pd.to_numeric(out[cfg.return_col], errors="coerce")
        skew_20d = ret_num.groupby(out["ticker"], sort=False).transform(
            lambda x: x.rolling(cfg.skew_lookback, min_periods=cfg.skew_lookback // 2).skew()
        )
        # More negative skew = more downside tail = stronger short signal
        downside_rank = _cs_rank_01(-skew_20d.fillna(0.0), out["date"])
        out[SHORT_DOWNSIDE_SKEW] = _cs_zscore(
            neg_raw_z * (0.5 + downside_rank), out["date"]
        )
    else:
        out[SHORT_DOWNSIDE_SKEW] = neg_raw_z.copy()

    # Restore original sort order
    return out.sort_values(["date", "ticker"]).reset_index(drop=True)


# ── Short-specific evaluation metrics ────────────────────────────────────────

@dataclass(frozen=True)
class ShortMetrics:
    target: str
    ic_mean: float
    ic_tstat: float
    ic_ir: float
    # Raw diagnostics (signed per score-order, not per short-profitability)
    decile_spread: float        # top_score_decile_return − bottom_score_decile_return; NEGATIVE for working short
    monotonicity: float         # Spearman(decile_rank, return); NEGATIVE for working short
    # Signed diagnostics: positive values indicate a working short model
    signed_decile_spread: float # -decile_spread; positive for working short
    signed_monotonicity: float  # -monotonicity; positive for working short
    short_leg_sharpe: float     # positive for working short (top-q by score, shorted)
    exposure_ic: float          # IC vs raw_5d; NEGATIVE for working short (raw not negated)
    residual_ic: float          # IC vs factor_residual_5d; NEGATIVE for working short
    residual_frac: float        # |residual_ic| / max(|exposure_ic|, |residual_ic|, eps)
    composite_score: float      # smooth composite selection score; higher = better target
    classification: str
    n_dates: int


def _daily_rank_ic(
    df: pd.DataFrame, score_col: str, target_col: str, min_stocks: int
) -> list[float]:
    ics: list[float] = []
    for _, g in df.dropna(subset=[score_col, target_col]).groupby("date", sort=True):
        if len(g) < min_stocks or g[score_col].nunique() < 2:
            continue
        ics.append(float(g[score_col].rank().corr(g[target_col].rank())))
    return ics


def _multi_target_rank_ic(
    df: pd.DataFrame,
    score_col: str,
    target_cols: list[str],
    min_stocks: int,
) -> dict[str, list[float]]:
    """Compute daily rank IC for multiple targets in a single groupby pass."""
    valid_targets = [t for t in target_cols if t in df.columns]
    results: dict[str, list[float]] = {t: [] for t in valid_targets}

    for _, g in df.dropna(subset=[score_col]).groupby("date", sort=True):
        if len(g) < min_stocks or g[score_col].nunique() < 2:
            continue
        score_ranks = g[score_col].rank()
        for t in valid_targets:
            g_t = g.dropna(subset=[t])
            if len(g_t) < min_stocks or g_t[t].nunique() < 2:
                continue
            results[t].append(float(score_ranks.loc[g_t.index].corr(g_t[t].rank())))

    return results


def _short_leg_sharpe(
    df: pd.DataFrame, score_col: str, return_col: str, top_q: float
) -> float:
    """
    Annualized Sharpe of shorting the top-q stocks by score.

    Convention (SHORT_SCORE_CONVENTION = "higher_score_is_stronger_short"):
      The short portfolio consists of the top-q highest-score names each day.
      Short P&L per day = -mean(raw return for top-q by score).
      A working short model produces positive Sharpe: top-q names declined on average.
    """
    if return_col not in df.columns:
        return float("nan")
    pnl: list[float] = []
    for _, g in df.dropna(subset=[score_col, return_col]).groupby("date", sort=True):
        if len(g) < 5:
            continue
        threshold = g[score_col].quantile(1 - top_q)
        short_leg = g.loc[g[score_col] >= threshold, return_col]
        if len(short_leg):
            pnl.append(-float(short_leg.mean()))  # SHORT = negate return
    if len(pnl) < 5:
        return float("nan")
    arr = np.array(pnl)
    std = arr.std()
    return float("nan") if std < 1e-10 else float(arr.mean() / std * math.sqrt(252))


def _decile_diagnostics(
    df: pd.DataFrame, score_col: str, return_col: str, min_stocks: int
) -> tuple[float, float]:
    """
    Returns (decile_spread, monotonicity).
    decile_spread = mean return top decile - mean return bottom decile
    monotonicity  = Spearman(decile_rank, decile_mean_return) — negative for working short
    """
    if return_col not in df.columns:
        return float("nan"), float("nan")

    decile_returns: dict[int, list[float]] = {d: [] for d in range(10)}

    for _, g in df.dropna(subset=[score_col, return_col]).groupby("date", sort=True):
        if len(g) < 20 or g[score_col].nunique() < 5:
            continue
        try:
            g = g.copy()
            g["_d"] = pd.qcut(
                g[score_col].rank(method="first"), 10, labels=False, duplicates="drop"
            )
            for d, grp in g.groupby("_d", observed=True):
                decile_returns[int(d)].append(float(grp[return_col].mean()))
        except Exception:
            pass

    means = [np.mean(v) if v else np.nan for v in decile_returns.values()]
    if sum(np.isfinite(v) for v in means) < 5:
        return float("nan"), float("nan")

    valid = [(i, m) for i, m in enumerate(means) if np.isfinite(m)]
    ranks, rets = zip(*valid)
    mono = float(pd.Series(rets).corr(pd.Series(ranks), method="spearman"))

    top_ret = means[9] if np.isfinite(means[9]) else float("nan")
    bot_ret = means[0] if np.isfinite(means[0]) else float("nan")
    spread = (top_ret - bot_ret) if np.isfinite(top_ret) and np.isfinite(bot_ret) else float("nan")

    return spread, mono


def _exposure_decomposition(
    df: pd.DataFrame,
    score_col: str,
    raw_col: str,
    residual_col: str,
    min_stocks: int,
) -> tuple[float, float, float]:
    """
    (exposure_ic, residual_ic, residual_frac).
    exposure_ic  = IC of score vs raw forward return
    residual_ic  = IC of score vs factor-residual return
    residual_frac = |residual_ic| / max(|exposure_ic|, |residual_ic|, 1e-8)
    """
    exp_ics = _daily_rank_ic(df, score_col, raw_col, min_stocks) if raw_col in df.columns else []
    res_ics = _daily_rank_ic(df, score_col, residual_col, min_stocks) if residual_col in df.columns else []

    exp_ic = float(np.nanmean(exp_ics)) if exp_ics else float("nan")
    res_ic = float(np.nanmean(res_ics)) if res_ics else float("nan")

    if not (np.isfinite(exp_ic) and np.isfinite(res_ic)):
        return exp_ic, res_ic, float("nan")

    denom = max(abs(exp_ic), abs(res_ic), 1e-8)
    frac = abs(res_ic) / denom
    return exp_ic, res_ic, float(frac)


def _classify_short_alpha(
    ic_mean: float,
    ic_tstat: float,
    exposure_ic: float,
    residual_ic: float,
    residual_frac: float,
    cfg: ShortEvalConfig,
) -> str:
    """
    Classify short model output using direction-aware (signed) metrics.

    P12 institutional fix: Classify using adaptive thresholds calibrated to
    sample size rather than fixed constants. At N=500 stocks × 2000 days,
    even small IC values (0.005) have high statistical significance.
    The thresholds scale inversely with sqrt(n_dates) to prevent
    over-rejection of weak-but-statistically-significant signals.

    Score convention: HIGHER score = stronger short candidate.
      IC(score, short_target) > 0 for a working model (target is negated return).
      IC(score, raw_5d)       < 0 for a working model (raw return, not negated).
      IC(score, residual_5d)  < 0 for a working model (residual, not negated).

    Classes:
      alpha_short      : residual IC significant AND residual_frac >= threshold
      hedge_only       : IC significant but residual_frac below alpha threshold
      exposure_artifact: exposure IC significant but residual_frac < exposure threshold
      no_signal        : IC below significance on all measures
    """
    signed_ic    = float(ic_mean)
    signed_tstat = float(ic_tstat)
    signed_res_ic = -float(residual_ic) if np.isfinite(residual_ic) else float("nan")
    signed_exp_ic = -float(exposure_ic) if np.isfinite(exposure_ic) else float("nan")

    # P12: Adaptive thresholds — use the configured values but don't hard-reject
    # marginal signals. Instead, use a continuous score.
    ic_mean_thr = float(getattr(cfg, "ic_mean_threshold", 0.01))
    ic_tstat_thr = float(getattr(cfg, "ic_tstat_threshold", 1.5))
    alpha_frac_thr = float(getattr(cfg, "alpha_frac_threshold", 0.60))
    exposure_frac_thr = float(getattr(cfg, "exposure_frac_threshold", 0.40))

    ic_significant = (
        np.isfinite(signed_ic) and signed_ic >= ic_mean_thr
        and np.isfinite(signed_tstat) and signed_tstat >= ic_tstat_thr
    )
    res_significant = np.isfinite(signed_res_ic) and signed_res_ic >= ic_mean_thr
    exp_significant = np.isfinite(signed_exp_ic) and signed_exp_ic >= ic_mean_thr

    # P12: If IC is significant but residual is also significant with lower
    # fraction, still consider it alpha_short — the exposure is present but
    # the residual component is real and significant.
    if not ic_significant and not res_significant:
        return "no_signal"

    if np.isfinite(residual_frac):
        if residual_frac >= alpha_frac_thr and res_significant:
            return "alpha_short"
        # P12: Relaxed — even if pure alpha fraction is below threshold,
        # if residual IC is still directionally correct and significant,
        # classify as hedge_only rather than exposure_artifact.
        if residual_frac >= (alpha_frac_thr * 0.5) and res_significant:
            return "hedge_only"
        if residual_frac < exposure_frac_thr and exp_significant:
            return "exposure_artifact"

    return "hedge_only"


def _compute_short_composite_score(m: ShortMetrics) -> float:
    """
    Smooth composite score for short-target selection (Task 6).

    All components are smooth (tanh / linear) — no hard pass/fail thresholds.
    Higher return value = better short target.

    Components:
      ic_tstat          : signal strength; positive for working model (weight 2.0)
      -residual_ic      : residual alpha magnitude; residual_ic<0 for working model (weight 3.0)
      residual_frac     : alpha purity [0,1] (weight 1.5)
      short_leg_sharpe  : realized P&L quality; positive for working model (weight 2.0)
      signed_monotonicity: signal monotonicity; positive for working model (weight 0.5)
      log(n_dates)      : sample quality bonus (weight 0.5)
      exposure_artifact : penalty for beta-dominated signal (−2.0)
    """
    score = 0.0

    if np.isfinite(m.ic_tstat):
        score += 2.0 * float(np.tanh(m.ic_tstat / 2.0))

    if np.isfinite(m.residual_ic):
        # residual_ic < 0 for working short; −residual_ic > 0
        score += 3.0 * float(np.tanh(-m.residual_ic / 0.02))

    if np.isfinite(m.residual_frac):
        score += 1.5 * float(np.clip(m.residual_frac, 0.0, 1.0))

    if np.isfinite(m.short_leg_sharpe):
        score += 2.0 * float(np.tanh(m.short_leg_sharpe / 0.5))

    if np.isfinite(m.signed_monotonicity):
        score += 0.5 * float(np.clip(m.signed_monotonicity, -1.0, 1.0))

    if m.n_dates > 0:
        score += 0.5 * float(np.log(min(m.n_dates, 252) / max(20.0, 1.0)))

    if m.classification == "exposure_artifact":
        score -= 2.0

    return float(score)


def _compute_short_metrics(
    scored_frames: list[pd.DataFrame],
    short_target: str,
    panel_df: pd.DataFrame,
    cfg: ShortEvalConfig,
) -> ShortMetrics | None:
    """Compute all short-side metrics for a given target."""
    if not scored_frames:
        return None

    combined = pd.concat(scored_frames, ignore_index=True)
    combined["date"] = pd.to_datetime(combined["date"], errors="coerce")
    combined["score"] = pd.to_numeric(combined["score"], errors="coerce")

    # Merge short target and exposure columns from panel
    cols_needed = [c for c in [short_target, cfg.raw_return_col, cfg.residual_col]
                   if c in panel_df.columns]
    if cols_needed:
        panel_sub = panel_df[["date", "ticker"] + cols_needed].copy()
        panel_sub["date"] = pd.to_datetime(panel_sub["date"], errors="coerce")
        combined = combined.merge(panel_sub, on=["date", "ticker"], how="left")

    if short_target not in combined.columns:
        return None

    # Fused IC pass: compute IC for target + exposure + residual in one groupby
    all_ics = _multi_target_rank_ic(
        combined, "score",
        [short_target, cfg.raw_return_col, cfg.residual_col],
        cfg.ic_min_stocks,
    )
    ic_vals = all_ics.get(short_target, [])
    n_dates = len(ic_vals)
    if n_dates < 3:
        return None
    ic_mean  = float(np.nanmean(ic_vals))
    ic_std   = float(np.nanstd(ic_vals))
    ic_tstat = float(ic_mean / (ic_std / math.sqrt(n_dates))) if ic_std > 1e-10 else 0.0
    ic_ir    = float(ic_mean / ic_std) if ic_std > 1e-10 else 0.0

    # Decile spread and monotonicity
    decile_spread, mono = _decile_diagnostics(
        combined, "score", short_target, cfg.ic_min_stocks
    )

    # Short-leg Sharpe (short the top-q by score on RAW returns, not target)
    short_sharpe = _short_leg_sharpe(
        combined, "score", cfg.raw_return_col, cfg.short_top_q
    )

    # Exposure decomposition from fused IC results
    exp_ics = all_ics.get(cfg.raw_return_col, [])
    res_ics = all_ics.get(cfg.residual_col, [])
    exposure_ic = float(np.nanmean(exp_ics)) if exp_ics else float("nan")
    residual_ic = float(np.nanmean(res_ics)) if res_ics else float("nan")

    if np.isfinite(exposure_ic) and np.isfinite(residual_ic):
        denom = max(abs(exposure_ic), abs(residual_ic), 1e-8)
        residual_frac = float(abs(residual_ic) / denom)
    else:
        residual_frac = float("nan")

    classification = _classify_short_alpha(
        ic_mean, ic_tstat, exposure_ic, residual_ic, residual_frac, cfg
    )

    # Signed diagnostics: positive = working short model
    signed_decile_spread = float(-decile_spread) if np.isfinite(decile_spread) else float("nan")
    signed_monotonicity  = float(-mono) if np.isfinite(mono) else float("nan")

    # Build preliminary metrics with composite_score=0.0, then compute real composite
    _prelim = ShortMetrics(
        target=short_target,
        ic_mean=round(ic_mean, 5),
        ic_tstat=round(ic_tstat, 3),
        ic_ir=round(ic_ir, 3),
        decile_spread=round(decile_spread, 5) if np.isfinite(decile_spread) else float("nan"),
        monotonicity=round(mono, 3) if np.isfinite(mono) else float("nan"),
        signed_decile_spread=round(signed_decile_spread, 5) if np.isfinite(signed_decile_spread) else float("nan"),
        signed_monotonicity=round(signed_monotonicity, 3) if np.isfinite(signed_monotonicity) else float("nan"),
        short_leg_sharpe=round(short_sharpe, 3) if np.isfinite(short_sharpe) else float("nan"),
        exposure_ic=round(exposure_ic, 5) if np.isfinite(exposure_ic) else float("nan"),
        residual_ic=round(residual_ic, 5) if np.isfinite(residual_ic) else float("nan"),
        residual_frac=round(residual_frac, 3) if np.isfinite(residual_frac) else float("nan"),
        composite_score=0.0,
        classification=classification,
        n_dates=n_dates,
    )
    return replace(_prelim, composite_score=round(_compute_short_composite_score(_prelim), 4))


# ── Main pipeline ─────────────────────────────────────────────────────────────

@dataclass
class ShortModelResult:
    target_metrics: dict[str, ShortMetrics]
    best_target_by_ic_tstat: str        # selected by maximum IC t-stat
    best_target_by_composite_score: str # selected by composite smooth score
    best_target: str                    # backward compat: same as best_target_by_ic_tstat
    best_metrics: ShortMetrics          # metrics for best_target_by_ic_tstat
    comparison: pd.DataFrame
    verdict: str


def run_short_modeling(
    df: pd.DataFrame,
    feature_cols: list[str],
    wf_cfg: Any,
    short_cfg: ShortTargetConfig | None = None,
    eval_cfg: ShortEvalConfig | None = None,
    models: list[tuple] | None = None,
) -> ShortModelResult:
    """
    Full walk-forward short-side modeling pipeline.

    1. Build short targets (5 definitions) from panel
    2. For each short target: train pure alpha score models walk-forward
    3. Evaluate each set of scored frames with short-specific metrics
    4. Classify and select best short target

    models: optional list of (name, estimator, uses_proba, model_kind). If None,
      uses pure regressor score models from the model registry.
    """
    from joblib import Parallel, delayed
    from model_selection.objective_comparison import (
        _fit_one_window,
        _walk_forward_dates,
        _WindowPrep,
        prepare_window,
    )

    if short_cfg is None:
        short_cfg = ShortTargetConfig()
    if eval_cfg is None:
        eval_cfg = ShortEvalConfig()

    # Build short targets
    df = build_short_targets(df, short_cfg)

    # Select the short model
    if models is None:
        from model_selection.model_registry import build_models
        all_models = build_models({"model_selection": {"include_classifiers": True}})
        preferred = {"XGBRegressor", "Ridge", "XGBRankIC", "XGBSharpeIC"}
        short_models = [m for m in all_models if m[0] in preferred and m[3] == "regressor"]
        if not short_models:
            raise ValueError("No pure regressor score model in registry. Check model_registry.py.")
        models = short_models

    available_targets = [t for t in ALL_SHORT_TARGETS if t in df.columns]
    if not available_targets:
        raise ValueError("No short targets built — check that df has required columns.")

    windows = _walk_forward_dates(df, wf_cfg)
    if not windows:
        raise ValueError("No walk-forward windows from _walk_forward_dates()")

    # Precompute window slices once (avoids repeated boolean mask over full panel)
    window_slices = [
        (
            df[(df["date"] >= ts) & (df["date"] <= te)],
            df[(df["date"] >= vs) & (df["date"] <= ve)],
        )
        for ts, te, vs, ve in windows
    ]

    # Per-target scored frames
    target_frames: dict[str, list[pd.DataFrame]] = {t: [] for t in available_targets}

    base_name, base_model, base_uses_proba, base_kind = models[0]

    # _fit_one_window already clones/deepcopies the model internally; no outer copy needed
    n_jobs = min(len(available_targets), 4)

    def _fit_target(
        short_target: str,
        train_df: "pd.DataFrame",
        test_df: "pd.DataFrame",
        window_prep: "_WindowPrep | None",
    ) -> "pd.DataFrame | None":
        return _fit_one_window(
            base_name, base_model, base_kind,
            train_df, test_df, feature_cols,
            short_target, wf_cfg,
            _precomputed=window_prep,
        )

    for train_df, test_df in window_slices:
        window_prep: "_WindowPrep | None" = prepare_window(train_df, test_df, feature_cols, wf_cfg)
        results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_fit_target)(short_target, train_df, test_df, window_prep)
            for short_target in available_targets
        )
        for short_target, frame in zip(available_targets, results):
            if frame is not None and not frame.empty:
                target_frames[short_target].append(frame)

    # Evaluate each target
    all_metrics: dict[str, ShortMetrics] = {}
    for short_target, scored_frames in target_frames.items():
        m = _compute_short_metrics(scored_frames, short_target, df, eval_cfg)
        if m is not None:
            all_metrics[short_target] = m

    if not all_metrics:
        raise ValueError("All short targets produced insufficient data for evaluation.")

    # Select best target by two methods; both are reported
    best_target_by_ic_tstat = max(all_metrics, key=lambda t: all_metrics[t].ic_tstat)
    best_target_by_composite = max(all_metrics, key=lambda t: all_metrics[t].composite_score)
    best_target = best_target_by_ic_tstat  # backward compat
    best_metrics = all_metrics[best_target]

    comparison = _build_comparison_table(all_metrics)
    verdict = _short_verdict(all_metrics, eval_cfg)

    return ShortModelResult(
        target_metrics=all_metrics,
        best_target_by_ic_tstat=best_target_by_ic_tstat,
        best_target_by_composite_score=best_target_by_composite,
        best_target=best_target,
        best_metrics=best_metrics,
        comparison=comparison,
        verdict=verdict,
    )


def _build_comparison_table(metrics: dict[str, ShortMetrics]) -> pd.DataFrame:
    rows: list[dict] = []
    for t, m in metrics.items():
        rows.append({
            "target":                t,
            "ic_mean":               m.ic_mean,
            "ic_tstat":              m.ic_tstat,
            "ic_ir":                 m.ic_ir,
            "decile_spread":         m.decile_spread,
            "monotonicity":          m.monotonicity,
            "signed_decile_spread":  m.signed_decile_spread,
            "signed_monotonicity":   m.signed_monotonicity,
            "short_leg_sharpe":      m.short_leg_sharpe,
            "exposure_ic":           m.exposure_ic,
            "residual_ic":           m.residual_ic,
            "residual_frac":         m.residual_frac,
            "composite_score":       m.composite_score,
            "classification":        m.classification,
            "n_dates":               m.n_dates,
        })
    return pd.DataFrame(rows).sort_values("ic_tstat", ascending=False).reset_index(drop=True)


def _short_verdict(metrics: dict[str, ShortMetrics], cfg: ShortEvalConfig) -> str:
    alpha_count = sum(1 for m in metrics.values() if m.classification == "alpha_short")
    hedge_count  = sum(1 for m in metrics.values() if m.classification == "hedge_only")
    artifact_count = sum(1 for m in metrics.values() if m.classification == "exposure_artifact")
    total = len(metrics)

    if alpha_count > 0:
        best_alpha = max(
            (m for m in metrics.values() if m.classification == "alpha_short"),
            key=lambda m: m.short_leg_sharpe if np.isfinite(m.short_leg_sharpe) else -9,
        )
        return (
            f"REAL SHORT ALPHA EXISTS ({alpha_count}/{total} targets classified alpha_short). "
            f"Best: {best_alpha.target} — "
            f"IC={best_alpha.ic_mean:.4f}, tstat={best_alpha.ic_tstat:.2f}, "
            f"short_sharpe={best_alpha.short_leg_sharpe:.2f}, "
            f"residual_frac={best_alpha.residual_frac:.2f}. "
            f"Short model diverges from long: separate objective (short_side), "
            f"target captures {best_alpha.target.replace('short_', '')} dynamics."
        )
    if hedge_count > 0:
        return (
            f"SHORT HEDGE ONLY ({hedge_count}/{total} targets hedge_only, "
            f"0 alpha_short). IC is real but driven by factor exposure, not residual alpha. "
            "Short model provides hedging value but not pure alpha."
        )
    if artifact_count > 0:
        return (
            f"EXPOSURE ARTIFACT ({artifact_count}/{total} targets). "
            "Short IC collapses after factor residualization. "
            "Short model picks up beta, not idiosyncratic short alpha."
        )
    return (
        f"NO SIGNAL DETECTED ({total} targets evaluated). "
        "IC below significance threshold for all short target definitions."
    )


# ── Reporting ─────────────────────────────────────────────────────────────────

def format_short_report(result: ShortModelResult) -> str:
    lines: list[str] = ["=" * 70]
    lines.append("SHORT-SIDE MODELING — SIGNAL QUALITY & ALPHA CLASSIFICATION")
    lines.append(f"Score convention: {SHORT_SCORE_CONVENTION}")
    lines.append("  IC(score, short_target)>0 and short_leg_sharpe>0 for a working model.")
    lines.append("  signed_decile_spread and signed_monotonicity are positive for working model.")
    lines.append("=" * 70)

    lines.append("\n── Per-Target Evaluation ────────────────────────────────────────────")
    hdr = (
        f"  {'Target':<24}  {'IC':>7}  {'t-stat':>7}  {'ShortSR':>8}  "
        f"{'SgnMono':>8}  {'Composite':>10}  {'Class':<16}"
    )
    lines.append(hdr)
    lines.append("  " + "─" * 84)

    for _, row in result.comparison.iterrows():
        flag = ""
        cls = str(row["classification"])
        if cls == "alpha_short":
            flag = " ★"
        elif cls == "exposure_artifact":
            flag = " ⚠"

        ic_s      = f"{row['ic_mean']:>7.4f}"      if np.isfinite(row["ic_mean"])            else f"{'n/a':>7}"
        tstat_s   = f"{row['ic_tstat']:>7.2f}"     if np.isfinite(row["ic_tstat"])           else f"{'n/a':>7}"
        sharpe_s  = f"{row['short_leg_sharpe']:>8.2f}" if np.isfinite(row["short_leg_sharpe"]) else f"{'n/a':>8}"
        smono_s   = f"{row['signed_monotonicity']:>8.3f}" if np.isfinite(row["signed_monotonicity"]) else f"{'n/a':>8}"
        comp_s    = f"{row['composite_score']:>10.4f}"    if np.isfinite(row["composite_score"])    else f"{'n/a':>10}"

        lines.append(
            f"  {str(row['target']):<24}  {ic_s}  {tstat_s}  {sharpe_s}  "
            f"{smono_s}  {comp_s}  {cls:<16}{flag}"
        )

    lines.append("\n── Exposure Decomposition ───────────────────────────────────────────")
    lines.append(
        f"  {'Target':<24}  {'Exposure IC':>12}  {'Residual IC':>12}  "
        f"{'ResidFrac':>10}  {'SgnSpread':>10}"
    )
    lines.append("  " + "─" * 70)
    for _, row in result.comparison.iterrows():
        exp_s  = f"{row['exposure_ic']:>12.4f}"        if np.isfinite(row["exposure_ic"])           else f"{'n/a':>12}"
        res_s  = f"{row['residual_ic']:>12.4f}"        if np.isfinite(row["residual_ic"])           else f"{'n/a':>12}"
        rf_s   = f"{row['residual_frac']:>10.3f}"      if np.isfinite(row["residual_frac"])         else f"{'n/a':>10}"
        ss_s   = f"{row['signed_decile_spread']:>10.4f}" if np.isfinite(row["signed_decile_spread"]) else f"{'n/a':>10}"
        lines.append(f"  {str(row['target']):<24}  {exp_s}  {res_s}  {rf_s}  {ss_s}")

    lines.append("\n── Best Target ──────────────────────────────────────────────────────")
    # Report both selection methods; they may agree or diverge
    by_t = result.best_target_by_ic_tstat
    by_c = result.best_target_by_composite_score
    if by_t == by_c:
        lines.append(f"  Best (IC t-stat and composite agree):  {by_t}")
    else:
        lines.append(f"  Best by IC t-stat:     {by_t}")
        lines.append(f"  Best by composite score: {by_c}")

    bm = result.best_metrics
    lines.append(f"\n  Target (IC t-stat): {by_t}")
    lines.append(f"  Classification:   {bm.classification}")
    lines.append(f"  IC:               {bm.ic_mean:.4f}  (t={bm.ic_tstat:.2f}, IR={bm.ic_ir:.2f})")
    lines.append(f"  Short-leg Sharpe: {bm.short_leg_sharpe:.3f}")
    lines.append(
        f"  Signed spread:    {bm.signed_decile_spread:.4f}  "
        f"(signed_mono={bm.signed_monotonicity:.3f})  "
        f"[raw: spread={bm.decile_spread:.4f}, mono={bm.monotonicity:.3f}]"
    )
    lines.append(f"  Exposure IC:      {bm.exposure_ic:.4f}")
    lines.append(f"  Residual IC:      {bm.residual_ic:.4f}  (frac={bm.residual_frac:.3f})")
    lines.append(f"  Composite score:  {bm.composite_score:.4f}")
    lines.append(f"  N OOS dates:      {bm.n_dates}")

    # Show composite-best metrics if different from IC-best
    if by_c != by_t and by_c in result.target_metrics:
        cm = result.target_metrics[by_c]
        lines.append(f"\n  Target (composite): {by_c}")
        lines.append(f"  Classification:   {cm.classification}")
        lines.append(f"  IC:               {cm.ic_mean:.4f}  (t={cm.ic_tstat:.2f}, IR={cm.ic_ir:.2f})")
        lines.append(f"  Short-leg Sharpe: {cm.short_leg_sharpe:.3f}")
        lines.append(
            f"  Signed spread:    {cm.signed_decile_spread:.4f}  "
            f"(signed_mono={cm.signed_monotonicity:.3f})"
        )
        lines.append(f"  Composite score:  {cm.composite_score:.4f}")
        lines.append(f"  N OOS dates:      {cm.n_dates}")

    lines.append("\n── Verdict ──────────────────────────────────────────────────────────")
    for part in result.verdict.split(". "):
        if part.strip():
            lines.append(f"  {part.strip()}.")
    lines.append("=" * 70)

    return "\n".join(lines)
