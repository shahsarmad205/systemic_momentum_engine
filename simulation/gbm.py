"""
Geometric Brownian Motion price process simulation.

Parameter estimation from history, path simulation, price targets,
and backtest of forecast accuracy. No lookahead in rolling computations.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def estimate_gbm_params(price_series: pd.Series, window: int = 252) -> tuple[float, float]:
    """
    Estimate GBM parameters (mu, sigma) from historical prices.

    mu = annualised drift = mean(log_returns) * 252
    sigma = annualised vol = std(log_returns) * sqrt(252)
    Returns: (mu, sigma)
    """
    if price_series is None or len(price_series) < 2:
        return 0.0, 0.20
    prices = price_series.dropna()
    if len(prices) < 2:
        return 0.0, 0.20
    log_ret = np.log(prices / prices.shift(1)).dropna()
    use = log_ret.tail(window)
    if len(use) < 2:
        return 0.0, 0.20
    mu = float(use.mean() * 252)
    sigma = float(use.std() * np.sqrt(252))
    if sigma <= 0 or not np.isfinite(sigma):
        sigma = 0.20
    if not np.isfinite(mu):
        mu = 0.0
    return mu, sigma


def simulate_gbm(
    S0: float,
    mu: float,
    sigma: float,
    T: float,
    n_steps: int,
    n_paths: int = 1000,
    seed: int | None = None,
) -> np.ndarray:
    """
    Simulate GBM price paths.

    S0: starting price
    mu: annualised drift
    sigma: annualised volatility
    T: time horizon in years
    n_steps: number of time steps
    n_paths: number of simulated paths
    Returns: array of shape (n_paths, n_steps+1)

    Formula: S(t+dt) = S(t) * exp((mu - 0.5*sigma^2)*dt + sigma*sqrt(dt)*Z)
    where Z ~ N(0,1)
    """
    if n_steps <= 0 or n_paths <= 0:
        return np.full((n_paths, n_steps + 1), S0)
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    drift = (mu - 0.5 * sigma ** 2) * dt
    vol_sqrt_dt = sigma * np.sqrt(dt)
    paths = np.zeros((n_paths, n_steps + 1))
    paths[:, 0] = S0
    Z = rng.standard_normal((n_paths, n_steps))
    for t in range(n_steps):
        paths[:, t + 1] = paths[:, t] * np.exp(drift + vol_sqrt_dt * Z[:, t])
    return paths


def gbm_price_targets(
    price_series: pd.Series,
    horizon_days: int = 5,
    n_paths: int = 1000,
    confidence_levels: list[float] | None = None,
    window: int = 252,
    seed: int | None = None,
) -> pd.DataFrame:
    """
    For each date, simulate GBM forward horizon_days days using only past data.

    Returns DataFrame with columns:
      p05, p25, p50, p75, p95 (price percentiles at horizon)
      expected_return: (p50 - current) / current
      prob_positive: fraction of paths with positive return
      gbm_var_95: VaR from GBM simulation at 95% confidence (positive = loss level)
    """
    if confidence_levels is None:
        confidence_levels = [0.05, 0.25, 0.50, 0.75, 0.95]
    if price_series is None or len(price_series) < window:
        return pd.DataFrame()
    prices = price_series.dropna()
    if len(prices) < window:
        return pd.DataFrame()
    T = horizon_days / 252.0
    n_steps = max(1, horizon_days)
    out = pd.DataFrame(index=prices.index, columns=[
        "p05", "p25", "p50", "p75", "p95",
        "expected_return", "prob_positive", "gbm_var_95",
    ])
    for i in range(len(prices)):
        if i < window:
            continue
        past = prices.iloc[: i + 1].tail(window)
        S0 = float(prices.iloc[i])
        mu, sigma = estimate_gbm_params(past, window=len(past))
        paths = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths=n_paths, seed=seed)
        final = paths[:, -1]
        out.iloc[i, 0] = np.percentile(final, 5)
        out.iloc[i, 1] = np.percentile(final, 25)
        out.iloc[i, 2] = np.percentile(final, 50)
        out.iloc[i, 3] = np.percentile(final, 75)
        out.iloc[i, 4] = np.percentile(final, 95)
        rets = (final - S0) / S0
        out.iloc[i, 5] = (np.median(final) - S0) / S0 if S0 > 0 else 0.0
        out.iloc[i, 6] = float(np.mean(rets > 0))
        var95 = -np.percentile(rets, 5)
        out.iloc[i, 7] = max(0.0, var95)
    return out.dropna(how="all")


def backtest_gbm_accuracy(
    price_series: pd.Series,
    horizon_days: int = 5,
    n_paths: int = 500,
    window: int = 252,
    seed: int | None = None,
) -> dict:
    """
    Backtest how well GBM predicts actual price distribution.

    For each historical date (with enough past and future):
      1. Estimate mu, sigma from past 252 days
      2. Simulate forward horizon_days
      3. Compare simulated distribution to actual outcome
    Returns:
      coverage_95: fraction of actual outcomes within 95% CI
      coverage_50: fraction within 50% CI
      mean_error: average absolute prediction error (actual return vs median simulated)
    A well-calibrated GBM should show coverage_95 ≈ 0.95
    """
    if price_series is None or len(price_series) < window + horizon_days + 1:
        return {"coverage_95": np.nan, "coverage_50": np.nan, "mean_error": np.nan}
    prices = price_series.dropna()
    n = len(prices)
    if n < window + horizon_days + 1:
        return {"coverage_95": np.nan, "coverage_50": np.nan, "mean_error": np.nan}
    in_95 = 0
    in_50 = 0
    errors = []
    count = 0
    T = horizon_days / 252.0
    n_steps = max(1, horizon_days)
    for i in range(window, n - horizon_days):
        past = prices.iloc[i - window : i + 1]
        S0 = float(prices.iloc[i])
        actual_future = float(prices.iloc[i + horizon_days])
        actual_return = (actual_future - S0) / S0 if S0 > 0 else 0.0
        mu, sigma = estimate_gbm_params(past, window=len(past))
        paths = simulate_gbm(S0, mu, sigma, T, n_steps, n_paths=n_paths, seed=seed)
        final = paths[:, -1]
        p05 = np.percentile(final, 2.5)
        p95 = np.percentile(final, 97.5)
        p25 = np.percentile(final, 25)
        p75 = np.percentile(final, 75)
        med = np.median(final)
        pred_return = (med - S0) / S0 if S0 > 0 else 0.0
        if p05 <= actual_future <= p95:
            in_95 += 1
        if p25 <= actual_future <= p75:
            in_50 += 1
        errors.append(abs(actual_return - pred_return))
        count += 1
    coverage_95 = in_95 / count if count else np.nan
    coverage_50 = in_50 / count if count else np.nan
    mean_error = float(np.mean(errors)) if errors else np.nan
    return {"coverage_95": coverage_95, "coverage_50": coverage_50, "mean_error": mean_error}
