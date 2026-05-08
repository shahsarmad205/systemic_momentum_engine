#!/usr/bin/env python3
"""
Alpha-Transfer Decomposition Audit
===================================
Traces where positive IC turns into negative execution Sharpe.

For each model/horizon/rebalance date, computes:
1. Raw signal quality (score stats, IC, decile spread)
2. Decay-adjusted optimizer alpha (z-std, decay factor, alpha shrink)
3. Optimizer expression (alpha reward, risk/turnover/neutrality penalties, correlations)
4. PnL bridge (raw LO return, optimized gross, short-leg, cost, net, exec Sharpe)
5. Classification (binding failure mode)

Hard constraints:
- No model promotion
- Diagnostic-only
- No threshold/gate/model/optimizer/cost changes

This script is self-contained to avoid the features/ → pandas_ta import chain.
It replicates the minimal portfolio construction logic needed for decomposition.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from scipy.optimize import minimize

# ── Constants ────────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("output/models")
DECOMP_OUTPUT = OUTPUT_DIR / "alpha_transfer_decomposition.parquet"
PM_TABLE_OUTPUT = OUTPUT_DIR / "alpha_transfer_pm_table.csv"
PM_TABLE_TXT = OUTPUT_DIR / "alpha_transfer_pm_table.txt"


@dataclass
class RebalanceDecomposition:
    """Per-rebalance-date decomposition."""
    date: str = ""
    model_id: str = ""
    horizon: int = 0
    path: str = ""

    # 1. Raw signal quality
    score_mean: float = np.nan
    score_std: float = np.nan
    cs_ic_spearman: float = np.nan
    decile_spread: float = np.nan
    top_bottom_return: float = np.nan
    n_tickers: int = 0

    # 2. Decay-adjusted optimizer alpha
    z_alpha_std: float = np.nan
    halflife: float = np.nan
    decay_factor: float = np.nan
    alpha_std_after_decay: float = np.nan
    alpha_shrink_pct: float = np.nan

    # 3. Optimizer expression
    alpha_reward: float = np.nan
    risk_penalty: float = np.nan
    turnover_penalty: float = np.nan
    neutrality_penalty: float = np.nan
    score_weight_corr: float = np.nan
    score_active_weight_corr: float = np.nan
    gross_exposure: float = np.nan
    net_exposure: float = np.nan
    n_long: int = 0
    n_short: int = 0
    lambda_risk: float = np.nan
    gamma_turnover: float = np.nan

    # 4. PnL bridge
    raw_lo_return: float = np.nan
    optimized_gross_return: float = np.nan
    short_leg_return: float = np.nan
    cost_return: float = np.nan
    net_return: float = np.nan
    horizon_days_held: int = 0

    # 5. Classification
    binding_failure: str = ""
    failure_detail: str = ""


# ── Minimal Portfolio Constraints (replica of PortfolioConstraints) ──────────
@dataclass(frozen=True)
class PC:
    path: str = "long_short_spread"
    max_positions: int = 10
    min_positions: int = 3
    max_gross: float = 1.0
    max_net: float = 0.01
    max_name_weight: float = 0.10
    min_position_weight: float = 0.0
    use_optimizer: bool = True
    optimization_type: str = "l1"
    lambda_risk: float = 2.0
    gamma_turnover: float = 4.0
    lambda_turn_override: float | None = None
    factor_neutral: bool = True
    beta_neutral: bool = True
    sector_neutral: bool = True
    max_beta_abs: float = 0.15
    max_sector_abs: float = 0.12
    adv_fraction: float = 0.05
    capital: float = 10_000_000.0
    max_participation_rate: float = 0.10
    short_squeeze_filter: bool = True
    short_squeeze_max_risk: float = 0.75
    market_neutral_shorts: bool = True
    optimizer_alpha_scale: float = 1.0
    signal_halflife_days: float = float("nan")
    horizon_days: int = 5


# ── Minimal Cost Config ─────────────────────────────────────────────────────
@dataclass(frozen=True)
class CC:
    commission_bps: float = 5.0
    spread_bps: float = 10.0
    borrow_bps: float = 50.0
    default_adv_usd: float = 1e6
    default_daily_vol: float = 0.02
    capital: float = 1e7
    max_participation_rate: float = 0.10


# ── Signal Quality ───────────────────────────────────────────────────────────
def _compute_signal_quality(day: pd.DataFrame) -> dict[str, float]:
    scores = pd.to_numeric(day["score"], errors="coerce")
    next_ret = pd.to_numeric(day.get("next_return", pd.Series(0.0, index=day.index)), errors="coerce")

    if scores.nunique(dropna=True) < 2:
        return {
            "score_mean": 0.0, "score_std": 0.0, "cs_ic_spearman": 0.0,
            "decile_spread": 0.0, "top_bottom_return": 0.0,
        }

    valid = scores.notna() & next_ret.notna()
    s = scores[valid]
    r = next_ret[valid]

    ic, _ = spearmanr(s, r)
    if not np.isfinite(ic):
        ic = 0.0

    deciles = pd.qcut(s, q=10, labels=False, duplicates="drop")
    grouped = r.groupby(deciles).mean()
    decile_spread = float(grouped.iloc[-1] - grouped.iloc[0]) if len(grouped) >= 2 else 0.0

    sorted_idx = s.sort_values().index
    n = len(sorted_idx)
    top_n = max(1, n // 10)
    bot_n = max(1, n // 10)
    top_ret = float(r.loc[sorted_idx[-top_n:]].mean())
    bot_ret = float(r.loc[sorted_idx[:bot_n]].mean())

    return {
        "score_mean": float(s.mean()),
        "score_std": float(s.std(ddof=0)),
        "cs_ic_spearman": ic,
        "decile_spread": decile_spread,
        "top_bottom_return": top_ret - bot_ret,
    }


# ── Decay Diagnostics ────────────────────────────────────────────────────────
def _compute_decay_diagnostics(
    scores: pd.Series,
    halflife: float,
    horizon: int,
    path: str,
) -> dict[str, float]:
    if scores.nunique(dropna=True) < 2:
        return {
            "z_alpha_std": 0.0, "halflife": halflife, "decay_factor": 1.0,
            "alpha_std_after_decay": 0.0, "alpha_shrink_pct": 0.0,
        }

    centered = scores - float(scores.mean())
    scale = float(scores.std(ddof=0))
    z = centered / (scale if np.isfinite(scale) and scale > 1e-12 else 1.0)

    if path == "short_side":
        alpha = -scores.rank(pct=True, method="average").fillna(0.5)
    elif path == "long_only_overlay":
        alpha = z.clip(lower=0.0)
    else:
        alpha = z

    z_std = float(alpha.std(ddof=0)) if alpha.nunique(dropna=True) >= 2 else 0.0

    if np.isfinite(halflife) and halflife > 0:
        decay_factor = float(2.0 ** (-horizon / halflife))
        decay_factor = max(decay_factor, 0.01)
    else:
        decay_factor = 1.0

    alpha_std_after = z_std * decay_factor
    shrink_pct = (1.0 - decay_factor) * 100.0

    return {
        "z_alpha_std": z_std,
        "halflife": halflife,
        "decay_factor": decay_factor,
        "alpha_std_after_decay": alpha_std_after,
        "alpha_shrink_pct": shrink_pct,
    }


# ── Score to Alpha (replica of PortfolioConstructor._score_to_alpha) ─────────
def _score_to_alpha(
    scores: pd.Series,
    constraints: PC,
    halflife: float,
    horizon: int,
    tickers: list[str],
) -> pd.Series:
    if scores.nunique(dropna=True) < 2:
        return pd.Series(0.0, index=tickers, dtype=float)

    centered = scores - float(scores.mean())
    scale = float(scores.std(ddof=0))
    z = centered / (scale if np.isfinite(scale) and scale > 1e-12 else 1.0)

    path = str(constraints.path or "").lower()
    if path == "short_side":
        alpha = -scores.rank(pct=True, method="average").fillna(0.5)
    elif path == "long_only_overlay":
        alpha = z.clip(lower=0.0)
    else:
        alpha = z

    if np.isfinite(halflife) and halflife > 0:
        decay = float(2.0 ** (-horizon / halflife))
        decay = max(decay, 0.01)
        alpha = alpha * decay

    return (float(constraints.optimizer_alpha_scale) * alpha).replace(
        [np.inf, -np.inf], np.nan
    ).fillna(0.0)


# ── Optimizer Expression ─────────────────────────────────────────────────────
def _compute_optimizer_expression(
    alpha: pd.Series,
    weights: pd.Series,
    cov: np.ndarray,
    w_prev: pd.Series,
    scores: pd.Series,
    constraints: PC,
) -> dict[str, float]:
    tickers = list(weights.index)
    n = len(tickers)
    if n == 0:
        return {
            "alpha_reward": 0.0, "risk_penalty": 0.0, "turnover_penalty": 0.0,
            "neutrality_penalty": 0.0, "score_weight_corr": 0.0,
            "score_active_weight_corr": 0.0, "gross_exposure": 0.0,
            "net_exposure": 0.0, "n_long": 0, "n_short": 0,
            "lambda_risk": constraints.lambda_risk, "gamma_turnover": constraints.gamma_turnover,
        }

    w = weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    a = alpha.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    w0 = w_prev.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
    s = scores.reindex(tickers).fillna(0.0).to_numpy(dtype=float)

    alpha_reward = float(np.dot(w, a))
    risk_penalty = float(constraints.lambda_risk * np.dot(w, np.dot(cov, w)))
    dw = w - w0
    turnover_penalty = float(constraints.gamma_turnover * np.dot(dw, dw))
    net = float(w.sum())
    neutrality_penalty = 0.0
    if constraints.path == "long_short_spread":
        neutrality_penalty = float(100.0 * net ** 2)

    sw_corr, _ = spearmanr(s, w)
    if not np.isfinite(sw_corr):
        sw_corr = 0.0

    active = w - w0
    saw_corr, _ = spearmanr(s, active)
    if not np.isfinite(saw_corr):
        saw_corr = 0.0

    return {
        "alpha_reward": alpha_reward,
        "risk_penalty": risk_penalty,
        "turnover_penalty": turnover_penalty,
        "neutrality_penalty": neutrality_penalty,
        "score_weight_corr": sw_corr,
        "score_active_weight_corr": saw_corr,
        "gross_exposure": float(np.abs(w).sum()),
        "net_exposure": net,
        "n_long": int(np.sum(w > 1e-12)),
        "n_short": int(np.sum(w < -1e-12)),
        "lambda_risk": constraints.lambda_risk,
        "gamma_turnover": constraints.gamma_turnover,
    }


# ── PnL Bridge ───────────────────────────────────────────────────────────────
def _compute_pnl_bridge(
    day: pd.DataFrame,
    weights: pd.Series,
    horizon: int,
    costs: CC,
    w_prev: pd.Series,
    path: str,
) -> dict[str, float]:
    tickers = weights.index.tolist()
    next_ret = pd.to_numeric(day.get("next_return", pd.Series(0.0, index=day.index)), errors="coerce")
    ret_map = next_ret.set_index(day["ticker"]).reindex(tickers).fillna(0.0)

    w = weights.reindex(tickers).fillna(0.0)
    w_prev_aligned = w_prev.reindex(tickers).fillna(0.0)

    gross_return = float((w * ret_map).sum())
    long_return = float((w.clip(lower=0.0) * ret_map).sum())
    short_return = float((w.clip(upper=0.0) * ret_map).sum())

    lo_weights = w.clip(lower=0.0)
    lo_gross = float(lo_weights.sum())
    if lo_gross > 1e-12:
        lo_weights_norm = lo_weights / lo_gross
        raw_lo_return = float((lo_weights_norm * ret_map).sum())
    else:
        raw_lo_return = 0.0

    delta = (w - w_prev_aligned).abs().sum()
    trade_notional = float(delta) * 1e7
    commission = trade_notional * float(costs.commission_bps) / 10000.0
    spread_cost = trade_notional * float(costs.spread_bps) / 10000.0
    cost_return = float(commission + spread_cost)

    borrow = float(w.clip(upper=0.0).abs().sum()) * float(costs.borrow_bps) / 10000.0 / 252.0

    net_return = gross_return - cost_return - borrow

    return {
        "raw_lo_return": raw_lo_return,
        "optimized_gross_return": gross_return,
        "short_leg_return": short_return,
        "cost_return": cost_return,
        "net_return": net_return,
        "horizon_days_held": horizon,
    }


# ── Classification ───────────────────────────────────────────────────────────
def _classify_failure(decomp: RebalanceDecomposition) -> tuple[str, str]:
    d = decomp

    if d.alpha_shrink_pct > 70 and d.alpha_std_after_decay < 0.15:
        return "alpha_decay_dominated", (
            f"Decay shrinks alpha by {d.alpha_shrink_pct:.0f}%. "
            f"Alpha std after decay={d.alpha_std_after_decay:.4f} is too small "
            f"to overcome optimizer friction (risk={d.risk_penalty:.6f}, "
            f"turnover={d.turnover_penalty:.6f})."
        )

    if d.risk_penalty > abs(d.alpha_reward) * 2 and d.risk_penalty > 0.01:
        return "risk_penalty_dominated", (
            f"Risk penalty ({d.risk_penalty:.6f}) dominates alpha reward "
            f"({d.alpha_reward:.6f}). Lambda_risk={d.lambda_risk} is too high "
            f"for the available alpha signal (std={d.alpha_std_after_decay:.4f})."
        )

    if d.turnover_penalty > abs(d.alpha_reward) * 1.5 and d.turnover_penalty > 0.005:
        return "turnover_penalty_dominated", (
            f"Turnover penalty ({d.turnover_penalty:.6f}) exceeds alpha reward "
            f"({d.alpha_reward:.6f}). Gamma_turnover={d.gamma_turnover} is too high."
        )

    if d.neutrality_penalty > 0.01 and abs(d.net_exposure) < 0.005:
        return "neutrality_penalty_dominated", (
            f"Neutrality penalty ({d.neutrality_penalty:.6f}) forces near-zero "
            f"net exposure ({d.net_exposure:.6f}), constraining the optimizer "
            f"from taking meaningful directional bets."
        )

    if d.short_leg_return < -0.001 and d.path in {"long_short_spread", "short_side"}:
        return "short_leg_drag", (
            f"Short leg contributes {d.short_leg_return:.6f} return drag. "
            f"Long gross={d.optimized_gross_return:.6f}, short={d.short_leg_return:.6f}. "
            f"The short side is actively destroying value."
        )

    if d.cost_return > abs(d.net_return) and d.net_return < 0:
        return "cost_dominated", (
            f"Costs ({d.cost_return:.6f}) exceed net return ({d.net_return:.6f}). "
            f"Gross PnL is being fully consumed by execution costs."
        )

    if d.score_weight_corr < 0.1 and d.score_active_weight_corr < 0.1:
        return "alpha_not_transferred", (
            f"Score-weight correlation ({d.score_weight_corr:.3f}) and "
            f"score-active-weight correlation ({d.score_active_weight_corr:.3f}) "
            f"are near zero. The optimizer is not translating signal rank into "
            f"portfolio positioning."
        )

    if d.net_return < 0 and d.alpha_std_after_decay < 0.25:
        return "alpha_decay_dominated", (
            f"Net return negative with weak decay-adjusted alpha "
            f"(std={d.alpha_std_after_decay:.4f}). Signal is too weak after "
            f"decay to cover costs and optimizer friction."
        )

    if d.net_return < 0:
        return "alpha_not_transferred", (
            f"Net return negative ({d.net_return:.6f}) despite positive IC "
            f"({d.cs_ic_spearman:.4f}). The signal has predictive power but "
            f"the portfolio construction pipeline cannot extract it profitably."
        )

    return "none", "No binding failure — positive execution"


# ── Simple L1 Optimizer (replica for decomposition) ─────────────────────────
def _simple_l1_optimize(
    mu: np.ndarray,
    cov: np.ndarray,
    w_prev: np.ndarray,
    lambda_risk: float,
    gamma_turnover: float,
    max_weight: float,
    max_gross: float,
    net_target: float,
    n_long_target: int,
    n_short_target: int,
) -> np.ndarray:
    """Simple L1-style optimizer for decomposition (no factor constraints)."""
    n = len(mu)
    if n == 0:
        return np.zeros(0)

    def objective(w):
        return float(-np.dot(w, mu) + lambda_risk * np.dot(w, np.dot(cov, w)) + gamma_turnover * np.dot(w - w_prev, w - w_prev))

    bounds = [(-max_weight, max_weight) for _ in range(n)]
    constraints = []

    if abs(net_target) < 0.1:
        constraints.append({"type": "eq", "fun": lambda w: np.sum(w) - net_target})

    constraints.append({"type": "eq", "fun": lambda w: np.sum(np.abs(w)) - max_gross})

    w0 = np.zeros(n)
    if n_long_target > 0:
        idx = np.argsort(mu)[-n_long_target:]
        w0[idx] = max_gross / (2 * n_long_target)
    if n_short_target > 0:
        idx = np.argsort(mu)[:n_short_target]
        w0[idx] = -max_gross / (2 * n_short_target)

    try:
        result = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=constraints, options={"maxiter": 200})
        if result.success:
            return result.x
    except Exception:
        pass

    return w0


# ── Forward Returns ──────────────────────────────────────────────────────────
def _compute_forward_returns(df: pd.DataFrame, horizon_days: int) -> pd.Series:
    """Compute forward returns matching the simulation engine."""
    if "next_return" in df.columns:
        return pd.to_numeric(df["next_return"], errors="coerce")

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    all_dates = sorted(df["date"].dropna().unique())

    ret_map = {}
    for dt in all_dates:
        day_tickers = df.loc[df["date"] == dt, "ticker"].astype(str).values
        future_dates = [d for d in all_dates if d > dt]
        if not future_dates:
            for t in day_tickers:
                ret_map[(dt, str(t))] = 0.0
            continue

        target_idx = min(horizon_days - 1, len(future_dates) - 1)
        target_date = future_dates[target_idx]

        prices_today = df.loc[df["date"] == dt, ["ticker", "price"]].set_index("ticker")["price"]
        prices_future = df.loc[df["date"] == target_date, ["ticker", "price"]].set_index("ticker")["price"]

        for t in day_tickers:
            p0 = prices_today.get(t, np.nan)
            p1 = prices_future.get(t, np.nan)
            if np.isfinite(p0) and np.isfinite(p1) and p0 > 0:
                ret_map[(dt, str(t))] = (p1 - p0) / p0
            else:
                ret_map[(dt, str(t))] = 0.0

    idx = pd.MultiIndex.from_tuples(ret_map.keys(), names=["date", "ticker"])
    return pd.Series(list(ret_map.values()), index=idx)


# ── Covariance (Ledoit-Wolf replica) ─────────────────────────────────────────
def _compute_covariance(
    full_df: pd.DataFrame,
    day: pd.DataFrame,
    dt: pd.Timestamp,
    lookback_days: int = 60,
) -> np.ndarray:
    tickers = day["ticker"].astype(str).tolist()
    n = len(tickers)
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    if "daily_return" not in full_df.columns:
        vols = pd.to_numeric(day.get("realised_vol_20d", pd.Series(0.02, index=day.index)), errors="coerce")
        diag = (vols.fillna(0.02).clip(lower=1e-4).to_numpy(dtype=float) ** 2) * 252.0
        return np.diag(np.maximum(diag, 0.015 ** 2))

    hist_dates = sorted(
        pd.to_datetime(full_df.loc[full_df["date"] < dt, "date"], errors="coerce").dropna().unique()
    )
    lookback_dates = set(hist_dates[-lookback_days:])
    hist = full_df.loc[full_df["date"].isin(lookback_dates), ["date", "ticker", "daily_return"]].copy()
    if hist.empty or hist["date"].nunique() < 10:
        return np.eye(n, dtype=float) * 0.015 ** 2

    pivot = hist.pivot_table(index="date", columns="ticker", values="daily_return", aggfunc="last")
    pivot = pivot.reindex(columns=tickers)
    if pivot.shape[0] < 10:
        return np.eye(n, dtype=float) * 0.015 ** 2

    arr = pivot.astype(float).replace([np.inf, -np.inf], np.nan)
    arr = arr.sub(arr.mean(axis=0, skipna=True), axis=1).fillna(0.0)

    try:
        from sklearn.covariance import ledoit_wolf
        cov, _ = ledoit_wolf(arr.to_numpy(dtype=float), assume_centered=False)
        cov = cov * 252.0
    except Exception:
        cov = np.cov(arr.to_numpy(dtype=float), rowvar=False) * 252.0

    cov = np.atleast_2d(cov).astype(float)
    if cov.shape != (n, n) or not np.isfinite(cov).all():
        return np.eye(n, dtype=float) * 0.015 ** 2

    cov = (cov + cov.T) / 2.0
    min_eig = float(np.linalg.eigvalsh(cov)[0]) if n > 1 else float(cov[0, 0])
    if min_eig < 1e-8:
        cov = cov + np.eye(n) * (1e-8 - min_eig + 1e-8)

    return cov


# ── Canonicalize Scored Panel (replica) ──────────────────────────────────────
def _canonicalize(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "date" not in df.columns:
        raise ValueError("scored panel requires 'date' column")
    if "ticker" not in df.columns:
        raise ValueError("scored panel requires 'ticker' column")
    if "score" not in df.columns:
        raise ValueError("scored panel requires 'score' column")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str)
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    return df.dropna(subset=["date", "ticker", "score"]).sort_values(["date", "ticker"])


# ── Main Decomposition Engine ────────────────────────────────────────────────
def _run_decomposition(
    scored: pd.DataFrame,
    constraints: PC,
    costs: CC,
    model_id: str = "unknown",
) -> list[RebalanceDecomposition]:
    df = _canonicalize(scored)
    dates = sorted(df["date"].dropna().unique())
    rebalance_every = max(1, int(5))
    path = str(constraints.path or "").lower()
    halflife = float(getattr(constraints, "signal_halflife_days", float("nan")))
    horizon = int(constraints.horizon_days)

    forward_rets = _compute_forward_returns(df, horizon)

    results: list[RebalanceDecomposition] = []
    prev_target = pd.Series(dtype=float)

    for idx, dt in enumerate(dates):
        if idx % rebalance_every != 0:
            continue

        day = df.loc[df["date"] == dt].copy()
        tickers = day["ticker"].astype(str).tolist()
        if not tickers:
            continue

        rets = forward_rets.xs(dt, level="date", drop_level=False) if dt in forward_rets.index.get_level_values("date") else pd.Series(0.0, index=tickers)
        if isinstance(rets, pd.DataFrame):
            rets = rets.iloc[:, 0]
        day["next_return"] = rets.reindex(tickers).fillna(0.0).values

        scores = pd.to_numeric(day["score"], errors="coerce")
        scores.index = day["ticker"].astype(str).values

        cov = _compute_covariance(df, day, pd.Timestamp(dt))

        alpha = _score_to_alpha(scores, constraints, halflife, horizon, tickers)

        n_long = max(1, constraints.max_positions // 2)
        n_short = max(1, constraints.max_positions // 2)
        net_target = 0.0 if path == "long_short_spread" else (1.0 if path == "long_only_overlay" else -0.5)

        weights_arr = _simple_l1_optimize(
            mu=alpha.reindex(tickers).fillna(0.0).to_numpy(dtype=float),
            cov=cov,
            w_prev=prev_target.reindex(tickers).fillna(0.0).to_numpy(dtype=float),
            lambda_risk=constraints.lambda_risk,
            gamma_turnover=constraints.gamma_turnover,
            max_weight=constraints.max_name_weight,
            max_gross=constraints.max_gross,
            net_target=net_target,
            n_long_target=n_long,
            n_short_target=n_short,
        )
        weights = pd.Series(weights_arr, index=tickers, dtype=float)

        sq = _compute_signal_quality(day)
        dd = _compute_decay_diagnostics(scores, halflife, horizon, path)
        oe = _compute_optimizer_expression(alpha, weights, cov, prev_target, scores, constraints)
        pnl = _compute_pnl_bridge(day, weights, horizon, costs, prev_target, path)

        decomp = RebalanceDecomposition(
            date=str(pd.Timestamp(dt).date()),
            model_id=model_id,
            horizon=horizon,
            path=path,
            score_mean=sq["score_mean"],
            score_std=sq["score_std"],
            cs_ic_spearman=sq["cs_ic_spearman"],
            decile_spread=sq["decile_spread"],
            top_bottom_return=sq["top_bottom_return"],
            n_tickers=len(tickers),
            z_alpha_std=dd["z_alpha_std"],
            halflife=dd["halflife"],
            decay_factor=dd["decay_factor"],
            alpha_std_after_decay=dd["alpha_std_after_decay"],
            alpha_shrink_pct=dd["alpha_shrink_pct"],
            alpha_reward=oe["alpha_reward"],
            risk_penalty=oe["risk_penalty"],
            turnover_penalty=oe["turnover_penalty"],
            neutrality_penalty=oe["neutrality_penalty"],
            score_weight_corr=oe["score_weight_corr"],
            score_active_weight_corr=oe["score_active_weight_corr"],
            gross_exposure=oe["gross_exposure"],
            net_exposure=oe["net_exposure"],
            n_long=oe["n_long"],
            n_short=oe["n_short"],
            lambda_risk=oe["lambda_risk"],
            gamma_turnover=oe["gamma_turnover"],
            raw_lo_return=pnl["raw_lo_return"],
            optimized_gross_return=pnl["optimized_gross_return"],
            short_leg_return=pnl["short_leg_return"],
            cost_return=pnl["cost_return"],
            net_return=pnl["net_return"],
            horizon_days_held=pnl["horizon_days_held"],
        )
        decomp.binding_failure, decomp.failure_detail = _classify_failure(decomp)
        results.append(decomp)

        prev_target = weights[weights.abs() > 1e-12]

    return results


# ── PM Table ─────────────────────────────────────────────────────────────────
def _build_pm_table(decomps: list[RebalanceDecomposition]) -> pd.DataFrame:
    if not decomps:
        return pd.DataFrame()

    rows = []
    for d in decomps:
        rows.append({
            "model_id": d.model_id,
            "horizon": d.horizon,
            "path": d.path,
            "n_rebalances": 1,
            "mean_ic": d.cs_ic_spearman,
            "mean_score_std": d.score_std,
            "decay_factor": d.decay_factor,
            "alpha_shrink_pct": d.alpha_shrink_pct,
            "alpha_std_after_decay": d.alpha_std_after_decay,
            "score_weight_corr": d.score_weight_corr,
            "score_active_weight_corr": d.score_active_weight_corr,
            "mean_alpha_reward": d.alpha_reward,
            "mean_risk_penalty": d.risk_penalty,
            "mean_turnover_penalty": d.turnover_penalty,
            "mean_gross_return": d.optimized_gross_return,
            "mean_short_leg": d.short_leg_return,
            "mean_cost": d.cost_return,
            "mean_net_return": d.net_return,
            "binding_failure": d.binding_failure,
            "failure_detail": d.failure_detail,
        })

    df = pd.DataFrame(rows)

    agg = df.groupby(["model_id", "horizon", "path"]).agg(
        n_rebalances=("n_rebalances", "sum"),
        mean_ic=("mean_ic", "mean"),
        mean_score_std=("mean_score_std", "mean"),
        decay_factor=("decay_factor", "mean"),
        alpha_shrink_pct=("alpha_shrink_pct", "mean"),
        alpha_std_after_decay=("alpha_std_after_decay", "mean"),
        score_weight_corr=("score_weight_corr", "mean"),
        score_active_weight_corr=("score_active_weight_corr", "mean"),
        mean_alpha_reward=("mean_alpha_reward", "mean"),
        mean_risk_penalty=("mean_risk_penalty", "mean"),
        mean_turnover_penalty=("mean_turnover_penalty", "mean"),
        mean_gross_return=("mean_gross_return", "mean"),
        mean_short_leg=("mean_short_leg", "mean"),
        mean_cost=("mean_cost", "mean"),
        mean_net_return=("mean_net_return", "mean"),
        dominant_failure=("binding_failure", lambda x: x.value_counts().index[0] if len(x) > 0 else "none"),
    ).reset_index()

    return agg


def _recommend_action(row: pd.Series) -> str:
    failure = row.get("dominant_failure", row.get("binding_failure", ""))
    ic = row.get("mean_ic", 0)
    alpha_std = row.get("alpha_std_after_decay", 0)
    shrink = row.get("alpha_shrink_pct", 0)
    short_leg = row.get("mean_short_leg", 0)
    cost = row.get("mean_cost", 0)
    sw_corr = row.get("score_weight_corr", 0)

    if failure == "alpha_decay_dominated":
        if ic > 0.03:
            return "SLOWER_HORIZON — IC is positive but decay kills alpha. Test 10d/20d horizon or slower signal."
        return "ABANDON — IC too weak to survive decay at any horizon."

    if failure == "risk_penalty_dominated":
        return "RESEARCH_PC — Signal has merit but lambda_risk is too aggressive. Test lower risk penalty."

    if failure == "turnover_penalty_dominated":
        return "RESEARCH_PC — Turnover penalty blocks alpha transfer. Test lower gamma_turnover or longer rebalance."

    if failure == "neutrality_penalty_dominated":
        return "RESEARCH_PC — Neutrality constraints too tight. Test relaxed net exposure for this signal."

    if failure == "short_leg_drag":
        if short_leg < -0.005:
            return "LONG_ONLY — Short leg is value-destructive. Research as long-only overlay only."
        return "RESEARCH_SIGNAL — Short side underperforms. Investigate short-specific features."

    if failure == "cost_dominated":
        return "ABANDON — Costs consume all gross alpha. Signal not tradable at current frequency/size."

    if failure == "alpha_not_transferred":
        if sw_corr < 0.1:
            return "RESEARCH_PC — Optimizer not translating signal to weights. Check constraint conflicts."
        return "RESEARCH_SIGNAL — IC exists but not monetizable. Check target construction."

    if ic > 0.05 and alpha_std > 0.3:
        return "PROMOTE_CANDIDATE — Strong IC and alpha survival. Worthy of deeper research."

    if ic > 0.02:
        return "LONG_ONLY — Weak but positive IC. Test as long-only overlay with relaxed constraints."

    return "ABANDON — No monetizable alpha signal."


def _print_pm_table(agg: pd.DataFrame, out_path: Path):
    lines = []
    lines.append("=" * 120)
    lines.append("ALPHA-TRANSFER DECOMPOSITION — PM SUMMARY TABLE")
    lines.append("=" * 120)
    lines.append("")

    for _, row in agg.iterrows():
        model = row["model_id"]
        horizon = row["horizon"]
        path = row["path"]
        n_rb = int(row["n_rebalances"])
        ic = row["mean_ic"]
        shrink = row["alpha_shrink_pct"]
        alpha_std = row["alpha_std_after_decay"]
        sw_corr = row["score_weight_corr"]
        saw_corr = row["score_active_weight_corr"]
        gross = row["mean_gross_return"]
        short = row["mean_short_leg"]
        cost = row["mean_cost"]
        net = row["mean_net_return"]
        failure = row["dominant_failure"]
        action = _recommend_action(row)

        root_cause = "Signal Research" if any(k in action for k in ["SIGNAL", "ABANDON", "SLOWER"]) else ("Portfolio Construction" if "PC" in action else "Execution Cost")

        lines.append(f"Model: {model} | Horizon: {horizon}d | Path: {path} | Rebalances: {n_rb}")
        lines.append("-" * 100)
        lines.append(f"  Signal Quality:")
        lines.append(f"    Mean IC (Spearman):     {ic:+.4f}")
        lines.append(f"    Alpha std after decay:  {alpha_std:.4f}  (shrink: {shrink:.0f}%)")
        lines.append(f"    Score→Weight corr:      {sw_corr:+.3f}")
        lines.append(f"    Score→Active-Weight:    {saw_corr:+.3f}")
        lines.append("")
        lines.append(f"  PnL Bridge (per rebalance):")
        lines.append(f"    Gross return:           {gross:+.6f}")
        lines.append(f"    Short-leg contribution: {short:+.6f}")
        lines.append(f"    Cost drag:              {cost:+.6f}")
        lines.append(f"    Net return:             {net:+.6f}")
        lines.append("")
        lines.append(f"  Binding Failure:  {failure}")
        lines.append(f"  Root Cause:       {root_cause}")
        lines.append(f"  Recommendation:   {action}")
        lines.append("")
        lines.append("")

    text = "\n".join(lines)
    out_path.write_text(text)
    print(text)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    import glob as _glob

    scored_files = sorted(_glob.glob("output/research_state/universe_panel*.parquet"))
    if not scored_files:
        scored_files = sorted(_glob.glob("output/models/*_scored*.parquet"))

    if not scored_files:
        cmp = OUTPUT_DIR / "model_comparison.csv"
        if cmp.exists():
            df = pd.read_csv(cmp)
            print(f"Found model_comparison.csv with {len(df)} models.")
            print("To run full decomposition, provide scored panel parquet files.")
            print("Looking for any available parquet in output/...")
            all_parquet = sorted(_glob.glob("output/**/*.parquet", recursive=True))
            if all_parquet:
                print(f"Found {len(all_parquet)} parquet files:")
                for p in all_parquet[:20]:
                    print(f"  {p}")
            return
        print("No model results found. Run model selection first.")
        return

    all_decomps: list[RebalanceDecomposition] = []

    for sf in scored_files:
        print(f"\nProcessing: {sf}")
        try:
            scored = pd.read_parquet(sf)
        except Exception as e:
            print(f"  SKIP: {e}")
            continue

        if "score" not in scored.columns:
            print(f"  SKIP: no 'score' column")
            continue

        model_id = Path(sf).stem.replace("_scored", "").replace("universe_panel_", "")

        constraints = PC(
            horizon_days=5,
            path="long_short_spread",
            lambda_risk=2.0,
            gamma_turnover=4.0,
            max_gross=1.0,
            max_name_weight=0.10,
            max_net=0.01,
            use_optimizer=True,
            optimization_type="l1",
            factor_neutral=True,
            beta_neutral=True,
            sector_neutral=True,
            max_beta_abs=0.15,
            max_sector_abs=0.12,
            short_squeeze_filter=True,
            short_squeeze_max_risk=0.75,
            signal_halflife_days=2.3,
        )

        costs = CC(
            commission_bps=5.0,
            spread_bps=10.0,
            borrow_bps=50.0,
            default_adv_usd=1e6,
            default_daily_vol=0.02,
            capital=1e7,
            max_participation_rate=0.10,
        )

        decomps = _run_decomposition(scored, constraints, costs, model_id=model_id)
        print(f"  Decomposed {len(decomps)} rebalance dates")
        all_decomps.extend(decomps)

    if not all_decomps:
        print("No decompositions produced.")
        return

    df = pd.DataFrame([d.__dict__ for d in all_decomps])
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(DECOMP_OUTPUT, index=False)
    print(f"\nPer-rebalance decomposition saved to: {DECOMP_OUTPUT}")

    agg = _build_pm_table(all_decomps)
    agg.to_csv(PM_TABLE_OUTPUT, index=False)
    print(f"PM table saved to: {PM_TABLE_OUTPUT}")

    _print_pm_table(agg, PM_TABLE_TXT)


if __name__ == "__main__":
    main()
