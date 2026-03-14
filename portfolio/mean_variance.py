"""
Markowitz Mean-Variance optimization for portfolio weights.

Efficient frontier, max-Sharpe and min-variance weights via scipy.optimize.
Long-only, optional max weight per asset; no lookahead in rolling weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def _annualize_return(series: pd.Series) -> float:
    return float(series.mean() * 252)


def _annualize_vol(series: pd.Series) -> float:
    return float(series.std() * np.sqrt(252))


def compute_efficient_frontier(
    returns_df: pd.DataFrame,
    n_portfolios: int = 1000,
    risk_free_rate: float = 0.04,
) -> pd.DataFrame:
    """
    Monte Carlo simulation of efficient frontier.

    returns_df: DataFrame of daily returns, one column per ticker.
    Returns DataFrame with columns:
      weights (dict), expected_return, volatility, sharpe_ratio
    """
    if returns_df.empty or returns_df.shape[1] == 0:
        return pd.DataFrame(columns=["weights", "expected_return", "volatility", "sharpe_ratio"])

    tickers = list(returns_df.columns)
    n_assets = len(tickers)
    mu = returns_df.mean().values * 252
    cov = returns_df.cov().values * 252
    if cov.size == 0 or not np.isfinite(cov).all():
        return pd.DataFrame(columns=["weights", "expected_return", "volatility", "sharpe_ratio"])

    np.random.seed(42)
    results = []
    for _ in range(n_portfolios):
        w = np.random.dirichlet(np.ones(n_assets))
        ret = float(w @ mu)
        vol = float(np.sqrt(w @ cov @ w))
        if vol > 1e-12:
            sr = (ret - risk_free_rate) / vol
        else:
            sr = 0.0
        weights_dict = {tickers[j]: float(w[j]) for j in range(n_assets)}
        results.append({
            "weights": weights_dict,
            "expected_return": ret,
            "volatility": vol,
            "sharpe_ratio": sr,
        })
    return pd.DataFrame(results)


def max_sharpe_weights(
    returns_df: pd.DataFrame,
    risk_free_rate: float = 0.04,
    constraints: dict | None = None,
) -> dict:
    """
    Find weights that maximise Sharpe ratio.

    Uses scipy.optimize.minimize with SLSQP.
    Constraints:
      - weights sum to 1
      - each weight between 0 and max_weight (default 0.25)
      - long-only (no negative weights)
    Returns: {ticker: weight}
    """
    constraints = constraints or {}
    max_weight = float(constraints.get("max_weight", 0.25))

    if returns_df.empty or returns_df.shape[1] == 0:
        return {}
    tickers = list(returns_df.columns)
    n = len(tickers)
    if n == 1:
        return {tickers[0]: 1.0}

    mu = returns_df.mean().values * 252
    cov = returns_df.cov().values * 252
    if not np.isfinite(cov).all() or not np.isfinite(mu).all():
        return {t: 1.0 / n for t in tickers}

    def neg_sharpe(w: np.ndarray) -> float:
        ret = w @ mu
        vol = np.sqrt(w @ cov @ w)
        if vol < 1e-12:
            return 0.0
        return -(ret - risk_free_rate) / vol

    w0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    eq = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    res = minimize(neg_sharpe, w0, method="SLSQP", bounds=bounds, constraints=eq)
    if not res.success:
        return {t: 1.0 / n for t in tickers}
    w = np.clip(res.x, 0.0, max_weight)
    w = w / w.sum()
    return {tickers[j]: float(w[j]) for j in range(n)}


def min_variance_weights(
    returns_df: pd.DataFrame,
    constraints: dict | None = None,
) -> dict:
    """
    Find weights that minimise portfolio variance.

    Same constraints as max_sharpe (sum=1, 0 <= w <= max_weight, long-only).
    Returns: {ticker: weight}
    """
    constraints = constraints or {}
    max_weight = float(constraints.get("max_weight", 0.25))

    if returns_df.empty or returns_df.shape[1] == 0:
        return {}
    tickers = list(returns_df.columns)
    n = len(tickers)
    if n == 1:
        return {tickers[0]: 1.0}

    cov = returns_df.cov().values * 252
    if not np.isfinite(cov).all():
        return {t: 1.0 / n for t in tickers}

    def port_var(w: np.ndarray) -> float:
        return float(w @ cov @ w)

    w0 = np.ones(n) / n
    bounds = [(0.0, max_weight)] * n
    eq = {"type": "eq", "fun": lambda w: w.sum() - 1.0}
    res = minimize(port_var, w0, method="SLSQP", bounds=bounds, constraints=eq)
    if not res.success:
        return {t: 1.0 / n for t in tickers}
    w = np.clip(res.x, 0.0, max_weight)
    w = w / w.sum()
    return {tickers[j]: float(w[j]) for j in range(n)}


def rolling_mv_weights(
    returns_df: pd.DataFrame,
    window: int = 60,
    method: str = "max_sharpe",
    risk_free_rate: float = 0.04,
    rebalance_freq: int = 20,
    max_weight: float = 0.25,
) -> pd.DataFrame:
    """
    Compute MV weights on a rolling basis.

    Recomputes every rebalance_freq days using last window days.
    Uses only past data (no lookahead).
    Returns DataFrame of weights over time (index = date, columns = tickers).
    """
    if returns_df.empty or returns_df.shape[1] == 0:
        return pd.DataFrame()
    tickers = list(returns_df.columns)
    constraints = {"max_weight": max_weight}
    out = pd.DataFrame(index=returns_df.index, columns=tickers, dtype=float)
    last_rebalance_idx = -rebalance_freq - 1
    eqw = 1.0 / len(tickers)
    for i in range(len(returns_df)):
        idx = returns_df.index[i]
        if i < window - 1:
            out.loc[idx] = eqw
            continue
        if i - last_rebalance_idx >= rebalance_freq or last_rebalance_idx < 0:
            past = returns_df.iloc[max(0, i - window + 1) : i + 1]
            if past.shape[0] < 2:
                out.loc[idx] = out.iloc[i - 1].values if i > 0 else eqw
                continue
            if method == "max_sharpe":
                w = max_sharpe_weights(past, risk_free_rate=risk_free_rate, constraints=constraints)
            else:
                w = min_variance_weights(past, constraints=constraints)
            last_rebalance_idx = i
            out.loc[idx] = [w.get(t, eqw) for t in tickers]
        else:
            out.loc[idx] = out.iloc[i - 1].values
    return out
