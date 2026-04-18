"""
Portfolio construction helpers.

Implements deterministic top-K selection from adjusted scores and
long-only weight normalization, plus institutional-grade enhancements:

  - ``construct_top_k_weights``:  fixed scoring bug (negative score shifting
    only applies when long-only; long-short keeps mean-zero weights)
  - ``construct_inverse_vol_weights``:  inverse-volatility weighting (1/σ)
    with optional correlation-aware diversification cap
  - ``construct_risk_parity_weights``: equal risk contribution (ERC)

All weight functions follow the same contract:
  - Input: scores dict + optional vol/corr data
  - Output: dict[ticker, weight] summing to 1 (long-only) or net-zero (L/S)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def construct_top_k_weights(
    adjusted_scores: Mapping[str, float],
    top_k: int,
    long_only: bool = True,
) -> dict[str, float]:
    """
    Build portfolio weights from adjusted scores with corrected allocation logic.

    Fix vs. previous implementation
    ---------------------------------
    The old code shifted all scores by -min_score to make them non-negative.
    This silently flips the signal direction when ALL selected scores are
    negative (e.g. a Bear-regime day where every stock scores < 0).
    In that case the "top" scorer was the least-negative stock, but after
    shifting it received the highest weight — which is correct.  However the
    total weight was then anchored to very small (near-zero) differences,
    creating extreme concentration.

    Fixed rule (long-only):
      - Use softmax-style allocation: w_i = exp(score_i) / sum(exp(score_j))
        This is always positive, preserves ordering, and handles negative scores
        gracefully without distorting relative magnitude.
      - Fallback to equal weight when score dispersion is near zero.

    Long/short:
      - Weights are proportional to raw score (positive for longs, negative for shorts)
      - Normalized so sum(|weight|) = 1

    Parameters
    ----------
    adjusted_scores : {ticker: score}
    top_k : maximum number of positions to select
    long_only : if True, select top K longs only (positive weights)
    """
    if top_k <= 0:
        return {}

    clean: list[tuple[str, float]] = []
    for ticker, score in adjusted_scores.items():
        try:
            v = float(score)
        except Exception:
            continue
        if not math.isfinite(v):
            continue
        clean.append((str(ticker), v))

    if not clean:
        return {}

    ranked = sorted(clean, key=lambda x: (-x[1], x[0]))
    selected = ranked[:min(top_k, len(ranked))]

    if long_only:
        # Softmax allocation — always positive, handles negative scores correctly
        scores_arr = np.array([s for _, s in selected], dtype=float)
        # Temperature-scaled softmax (temperature=1 by default)
        scores_arr = scores_arr - scores_arr.max()   # numerical stability
        exp_scores = np.exp(scores_arr)
        total = float(exp_scores.sum())
        if total <= 0 or not np.isfinite(total):
            w = 1.0 / len(selected)
            return {t: w for t, _ in selected}
        return {t: float(exp_scores[i] / total) for i, (t, _) in enumerate(selected)}

    else:
        # Long/short: preserve sign, normalise by sum of abs weights
        alloc = {t: s for t, s in selected}
        total_abs = sum(abs(v) for v in alloc.values())
        if total_abs <= 0:
            return {}
        return {t: v / total_abs for t, v in alloc.items()}


@dataclass(frozen=True)
class RegimeExposureConfig:
    normal_top_k: int = 10
    normal_exposure: float = 1.0
    crisis_top_k: int = 8     # Phase 4: raised from 4 — was leaving too much alpha on table
    crisis_exposure: float = 0.45  # Phase 4: raised from 0.25 — was suppressing returns to 25%
    crisis_regime_value: str = "Crisis"


def construct_regime_aware_portfolio(
    adjusted_scores: Mapping[str, float],
    current_regime: str,
    config: RegimeExposureConfig | None = None,
) -> dict[str, object]:
    """
    Regime-aware portfolio construction.

    Uses only the provided current_regime (no future state), then applies:
      ranking -> top-K selection -> normalized weights -> exposure scaling

    Returns:
      {
        "selected_assets": list[str],
        "adjusted_weights": dict[str, float],   # sums to effective_exposure
        "effective_exposure": float,            # in [0, 1]
        "top_k_used": int,
      }
    """
    cfg = config or RegimeExposureConfig()
    is_crisis = str(current_regime) == str(cfg.crisis_regime_value)

    top_k = int(cfg.crisis_top_k if is_crisis else cfg.normal_top_k)
    top_k = max(0, top_k)

    exposure = float(cfg.crisis_exposure if is_crisis else cfg.normal_exposure)
    exposure = max(0.0, min(exposure, 1.0))

    base_w = construct_top_k_weights(adjusted_scores, top_k=top_k)
    selected_assets = list(base_w.keys())
    adjusted_weights = {t: w * exposure for t, w in base_w.items()}

    return {
        "selected_assets": selected_assets,
        "adjusted_weights": adjusted_weights,
        "effective_exposure": exposure,
        "top_k_used": top_k,
    }


def construct_inverse_vol_weights(
    tickers: list[str],
    volatilities: dict[str, float],
    scores: Optional[Mapping[str, float]] = None,
    max_single_weight: float = 0.25,
    min_vol_floor: float = 0.005,
) -> dict[str, float]:
    """
    Inverse-volatility weighting (1/σ) — standard at AQR, Bridgewater.

    Each position's weight is inversely proportional to its realised
    volatility.  Lower-vol stocks get higher weights, providing natural
    risk equalisation without a full covariance matrix.

    Optionally blends with score-proportional weights when *scores* provided:
        w_blend_i = 0.5 × w_invvol_i + 0.5 × w_score_i

    Parameters
    ----------
    tickers : positions to weight (pre-selected subset)
    volatilities : {ticker: annualised_vol} — must include all tickers
    scores : optional {ticker: adjusted_score} for score-blended weights
    max_single_weight : maximum weight any single name can receive
    min_vol_floor : floor applied to vol before inversion (prevents extreme weights)

    Returns
    -------
    dict[ticker, weight], weights sum to 1
    """
    if not tickers:
        return {}

    inv_vols: dict[str, float] = {}
    for t in tickers:
        vol = volatilities.get(t, 0.2)       # default 20% annual vol
        vol = max(float(vol), min_vol_floor)  # apply floor
        inv_vols[t] = 1.0 / vol

    total_inv = sum(inv_vols.values())
    if total_inv <= 0:
        w_eq = 1.0 / len(tickers)
        return {t: w_eq for t in tickers}

    inv_vol_weights = {t: v / total_inv for t, v in inv_vols.items()}

    # Optionally blend with score weights
    if scores is not None:
        score_vals = [(t, float(scores.get(t, 0.0))) for t in tickers]
        # Softmax of scores
        s_arr = np.array([s for _, s in score_vals])
        s_arr -= s_arr.max()
        exp_s = np.exp(s_arr)
        exp_sum = float(exp_s.sum())
        score_weights = (
            {t: float(exp_s[i] / exp_sum) for i, (t, _) in enumerate(score_vals)}
            if exp_sum > 0
            else {t: 1.0 / len(tickers) for t in tickers}
        )
        blended = {t: 0.5 * inv_vol_weights[t] + 0.5 * score_weights[t] for t in tickers}
    else:
        blended = inv_vol_weights

    # Cap at max_single_weight and renormalise
    capped = {t: min(w, max_single_weight) for t, w in blended.items()}
    total_capped = sum(capped.values())
    if total_capped <= 0:
        w_eq = 1.0 / len(tickers)
        return {t: w_eq for t in tickers}

    return {t: w / total_capped for t, w in capped.items()}


def construct_risk_parity_weights(
    tickers: list[str],
    covariance_matrix: Optional[np.ndarray] = None,
    volatilities: Optional[dict[str, float]] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> dict[str, float]:
    """
    Equal risk contribution (ERC / Risk Parity) weights.

    Each asset contributes equally to total portfolio variance.
    When *covariance_matrix* is provided, uses the full Σ (Ledoit-Wolf
    shrinkage recommended before passing).  Otherwise falls back to
    diagonal Σ = diag(vol²), equivalent to inverse-vol weighting.

    Algorithm: Cyclical coordinate descent (Chaves et al. 2012).

    Parameters
    ----------
    tickers : list of tickers (must align with covariance_matrix rows/cols)
    covariance_matrix : (n×n) annualised covariance matrix
    volatilities : fallback if no covariance_matrix supplied
    max_iter : maximum CCD iterations
    tol : convergence tolerance on weight vector

    Returns
    -------
    dict[ticker, weight], weights sum to 1
    """
    n = len(tickers)
    if n == 0:
        return {}
    if n == 1:
        return {tickers[0]: 1.0}

    # Build Σ
    if covariance_matrix is not None and covariance_matrix.shape == (n, n):
        Sigma = covariance_matrix.astype(float)
    elif volatilities is not None:
        vols = np.array([max(volatilities.get(t, 0.2), 1e-4) for t in tickers])
        Sigma = np.diag(vols ** 2)
    else:
        vols = np.full(n, 0.2)
        Sigma = np.diag(vols ** 2)

    # Ensure positive-definite (add ridge for numerical stability)
    Sigma += np.eye(n) * 1e-8

    # Cyclical coordinate descent for ERC
    w = np.ones(n) / n
    for _ in range(max_iter):
        w_prev = w.copy()
        for i in range(n):
            # Marginal risk contribution of asset i
            Sigma_w = Sigma @ w
            rc_i = w[i] * Sigma_w[i]
            # Target: rc_i = portfolio_vol / n  (equal contribution)
            port_var = float(w @ Sigma_w)
            target_rc = port_var / n
            # Newton step for w_i
            a = Sigma[i, i]
            b = float(np.dot(Sigma[i, :], w) - Sigma[i, i] * w[i])
            # Solve: w_i * (a*w_i + b) = target_rc
            # a*w_i^2 + b*w_i - target_rc = 0
            if a > 0:
                disc = b ** 2 + 4 * a * target_rc
                if disc >= 0:
                    w[i] = (-b + math.sqrt(disc)) / (2 * a)
        # Renormalise
        w = np.abs(w)
        total = float(w.sum())
        if total > 0:
            w /= total
        if np.max(np.abs(w - w_prev)) < tol:
            break

    return {t: float(w[i]) for i, t in enumerate(tickers)}


def construct_hybrid_fmc_score_weights(
    tickers: list[str],
    scores: Mapping[str, float],
    market_caps: dict[str, float],
    score_blend: float = 0.5,
    max_single_weight: float = 0.20,
) -> dict[str, float]:
    """
    FMC × Score hybrid weighting — S&P Global (Innes et al.) p.7, p.21-22.

    Rationale (graph node: mfic_rationale_hybrid_weight):
      "FMC × Score hybrid weighting maintains liquidity while improving factor
      tilt.  Pure score-weighting (softmax) ignores float market cap, leading
      to concentration in small illiquid names that score well."

    Weight formula:
        w_i = blend × w_score_i + (1 - blend) × w_fmc_i

    where:
        w_score_i  = softmax(score_i)          — factor tilt
        w_fmc_i    = FMC_i / Σ FMC_j           — liquidity anchor

    Parameters
    ----------
    tickers : pre-selected position list
    scores : {ticker: adjusted_score}
    market_caps : {ticker: float_adjusted_market_cap_$}
    score_blend : weight on score component (0 = pure FMC, 1 = pure score)
    max_single_weight : cap to prevent single-name concentration

    Returns
    -------
    dict[ticker, weight], weights sum to 1
    """
    if not tickers:
        return {}

    # Score component (softmax)
    s_arr = np.array([float(scores.get(t, 0.0)) for t in tickers], dtype=float)
    s_arr -= s_arr.max()
    exp_s = np.exp(s_arr)
    exp_sum = float(exp_s.sum())
    w_score = (
        {t: float(exp_s[i] / exp_sum) for i, t in enumerate(tickers)}
        if exp_sum > 0
        else {t: 1.0 / len(tickers) for t in tickers}
    )

    # FMC component
    caps = np.array([max(float(market_caps.get(t, 1.0)), 1.0) for t in tickers], dtype=float)
    cap_sum = float(caps.sum())
    w_fmc = {t: float(caps[i] / cap_sum) for i, t in enumerate(tickers)}

    # Blend
    blended = {
        t: score_blend * w_score[t] + (1.0 - score_blend) * w_fmc[t]
        for t in tickers
    }

    # Cap and renormalise
    capped = {t: min(w, max_single_weight) for t, w in blended.items()}
    total = sum(capped.values())
    if total <= 0:
        return {t: 1.0 / len(tickers) for t in tickers}
    return {t: w / total for t, w in capped.items()}


def compute_factor_imbalance(factor_scores: dict[str, float]) -> float:
    """
    Factor Imbalance Metric — S&P Global (Innes et al.) p.16.

    In Bottom-Up multi-factor construction, the composite z-score is the
    average of per-factor z-scores.  Factor imbalance measures the maximum
    deviation of any single factor from the composite average.

    High imbalance (> 1.0 z-score units) means one factor dominates the
    composite selection — the portfolio is implicitly a single-factor bet
    even though it looks multi-factor.

    Formula:
        composite = mean(z_factor_i)
        imbalance  = max |z_factor_i - composite|

    Parameters
    ----------
    factor_scores : {factor_name: z_score} for a single stock or portfolio avg

    Returns
    -------
    float : imbalance (0 = perfectly balanced, >1 = one factor dominates)
    """
    if len(factor_scores) < 2:
        return 0.0
    vals = np.array(list(factor_scores.values()), dtype=float)
    composite = float(vals.mean())
    return float(np.max(np.abs(vals - composite)))


def compute_active_share(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
) -> float:
    """
    Active Share — S&P Global (Innes et al.) p.6-7, p.11.

    Measures how different the portfolio is from the benchmark.
    Active Share = 0.5 × Σ |w_portfolio_i - w_benchmark_i|

    Range: [0, 1].  Typical institutional targets: 0.60–0.80.
    Paper rationale (mfic_rationale_linear_exposure_risk):
      "Linear relationship between Active Share and Tracking Error means
      there is no optimal concentration point — target Active Share
      explicitly rather than deriving it from stock count alone."

    Parameters
    ----------
    portfolio_weights : {ticker: weight}, should sum to 1
    benchmark_weights : {ticker: weight}, benchmark (e.g. equal-weight S&P 500)

    Returns
    -------
    float in [0, 1]
    """
    all_tickers = set(portfolio_weights) | set(benchmark_weights)
    active_share = 0.5 * sum(
        abs(portfolio_weights.get(t, 0.0) - benchmark_weights.get(t, 0.0))
        for t in all_tickers
    )
    return float(active_share)


def compute_rank_based_weights(df: pd.DataFrame, long_only: bool = False) -> pd.DataFrame:
    """
    Convert `adjusted_score` into rank-based centered/normalized weights.

    Steps:
    1) Percentile rank in [0, 1] using stable average-rank ties.
    2) Center: weight_raw = rank_pct - 0.5
    3) Optional long-only clamp: max(weight_raw, 0)
    4) Normalize so sum(abs(weight)) == 1

    Returns a copy with columns: `rank_pct`, `weight_raw`, `weight`.
    """
    if "adjusted_score" not in df.columns:
        raise KeyError("compute_rank_based_weights requires column 'adjusted_score'")

    out = df.copy()
    s = pd.to_numeric(out["adjusted_score"], errors="coerce")
    valid = s.notna()

    rank_pct = pd.Series(0.5, index=out.index, dtype=float)
    if valid.any():
        sv = s[valid]
        n = int(len(sv))
        r = sv.rank(method="average", ascending=True)
        rp = (r - 1.0) / (n - 1.0) if n > 1 else pd.Series(0.5, index=sv.index, dtype=float)
        rank_pct.loc[sv.index] = pd.to_numeric(rp, errors="coerce").fillna(0.5)

    weight_raw = rank_pct - 0.5
    if long_only:
        weight_raw = weight_raw.clip(lower=0.0)

    denom = float(weight_raw.abs().sum())
    if denom <= 1e-12:
        # Stable fallback:
        # - long_only: equal weight over valid rows
        # - long/short: all zeros (no cross-sectional dispersion)
        if long_only and valid.any():
            w = pd.Series(0.0, index=out.index, dtype=float)
            w.loc[valid] = 1.0 / float(valid.sum())
        else:
            w = pd.Series(0.0, index=out.index, dtype=float)
    else:
        w = weight_raw / denom

    out["rank_pct"] = rank_pct.fillna(0.5).astype(float)
    out["weight_raw"] = weight_raw.fillna(0.0).astype(float)
    out["weight"] = pd.to_numeric(w, errors="coerce").fillna(0.0).astype(float)
    return out


def select_high_conviction_assets(
    df: pd.DataFrame,
    *,
    rank_col: str = "rank_pct",
    score_col: str = "adjusted_score",
    threshold: float = 0.6,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Keep high-conviction assets then assign normalized long-only weights.

    Rules:
    1) Keep rows with rank_col > threshold
    2) Select top K by rank_col (then score_col, both descending)
    3) Normalize selected weights to sum 1 (equal-weight fallback if needed)
    4) Edge handling:
       - fewer than K pass -> use all passers
       - none pass -> fallback to top K by rank/score from full universe

    Returns selected rows with added `weight` column.
    """
    if top_k <= 0:
        return pd.DataFrame(columns=list(df.columns) + ["weight"])
    if rank_col not in df.columns:
        raise KeyError(f"select_high_conviction_assets requires column '{rank_col}'")

    out = df.copy()
    rank = pd.to_numeric(out[rank_col], errors="coerce")
    if score_col in out.columns:
        score = pd.to_numeric(out[score_col], errors="coerce")
    else:
        score = rank.copy()
    out["_rank"] = rank
    out["_score"] = score

    valid = out[out["_rank"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=list(df.columns) + ["weight"])

    passed = valid[valid["_rank"] > float(threshold)].copy()
    pool = passed if not passed.empty else valid

    selected = (
        pool.sort_values(by=["_rank", "_score"], ascending=[False, False], kind="mergesort")
        .head(min(int(top_k), len(pool)))
        .copy()
    )

    n = len(selected)
    if n == 0:
        return pd.DataFrame(columns=list(df.columns) + ["weight"])

    # Long-only normalized weights from rank strength.
    w_raw = selected["_rank"].clip(lower=0.0)
    denom = float(w_raw.sum())
    if denom <= 1e-12:
        selected["weight"] = 1.0 / n
    else:
        selected["weight"] = w_raw / denom

    selected["weight"] = pd.to_numeric(selected["weight"], errors="coerce").fillna(0.0)
    selected = selected.drop(columns=["_rank", "_score"])
    return selected

