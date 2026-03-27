"""
Portfolio construction helpers.

Implements deterministic top-K selection from adjusted scores and long-only
weight normalization to sum to 1.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd


def construct_top_k_weights(
    adjusted_scores: Mapping[str, float],
    top_k: int,
) -> dict[str, float]:
    """
    Build long-only portfolio weights from adjusted scores.

    Rules:
    - Select top K by adjusted score (descending), deterministic tie-break by ticker.
    - Convert selected scores into non-negative allocation signals:
      * if min(score) < 0, shift by -min(score)
      * otherwise use raw selected scores
    - If all allocation signals sum to 0, fallback to equal weights.
    - Return only selected tickers; weights sum to 1.
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
    selected = ranked[: min(top_k, len(ranked))]
    selected_scores = [s for _, s in selected]

    min_s = min(selected_scores)
    if min_s < 0:
        alloc = {t: s - min_s for t, s in selected}
    else:
        alloc = {t: s for t, s in selected}

    total = sum(alloc.values())
    if total <= 0:
        w = 1.0 / len(selected)
        return {t: w for t, _ in selected}

    return {t: v / total for t, v in alloc.items()}


@dataclass(frozen=True)
class RegimeExposureConfig:
    normal_top_k: int = 10
    normal_exposure: float = 1.0
    crisis_top_k: int = 4
    crisis_exposure: float = 0.25
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

