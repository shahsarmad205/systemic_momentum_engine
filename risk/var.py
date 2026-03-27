"""
Portfolio VaR helpers for pre-flight risk checks (historical / parametric / Monte Carlo).
"""

from __future__ import annotations

from statistics import NormalDist
from typing import Literal

import numpy as np
import pandas as pd

VarMethod = Literal["historical", "parametric", "monte_carlo"]


def compute_historical_var(returns: pd.Series | np.ndarray, confidence: float) -> float:
    """
    One-day historical VaR as a **positive** fraction of portfolio value (expected shortfall
    boundary: loss magnitude at the left tail).

    ``returns`` are simple daily returns (e.g. -0.02 for -2%).
    """
    r = pd.to_numeric(pd.Series(returns), errors="coerce").dropna().to_numpy()
    if len(r) < 5 or not (0 < confidence < 1):
        return float("nan")
    alpha = 1.0 - float(confidence)
    q = float(np.quantile(r, alpha))
    return float(-q)


def compute_parametric_var(returns: pd.Series | np.ndarray, confidence: float) -> float:
    """Gaussian VaR using sample mean and std of portfolio returns."""
    r = pd.to_numeric(pd.Series(returns), errors="coerce").dropna().to_numpy()
    if len(r) < 5 or not (0 < confidence < 1):
        return float("nan")
    mu = float(np.mean(r))
    sigma = float(np.std(r, ddof=1))
    if sigma <= 0 or not (sigma == sigma):
        return float(max(0.0, -mu))
    z = NormalDist().inv_cdf(1.0 - float(confidence))
    return float(-(mu + z * sigma))


def compute_monte_carlo_var(
    returns_df: pd.DataFrame,
    weights: pd.Series,
    confidence: float,
    *,
    n_samples: int = 20_000,
    seed: int | None = 42,
) -> float:
    """
    VaR by joint bootstrap: each draw samples one historical day (row) and applies weights.
    Preserves cross-sectional correlation in the window.
    """
    if returns_df.empty or len(returns_df) < 5:
        return float("nan")
    w = weights.reindex(returns_df.columns).fillna(0.0).to_numpy(dtype=float)
    if float(np.sum(np.abs(w))) < 1e-12:
        return 0.0
    mat = returns_df.to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(mat), size=int(n_samples))
    sim = mat[idx] @ w
    alpha = 1.0 - float(confidence)
    return float(-float(np.quantile(sim, alpha)))


def portfolio_var(
    tickers: list[str],
    weights: dict[str, float],
    returns_df: pd.DataFrame,
    *,
    confidence: float = 0.95,
    method: VarMethod = "historical",
) -> tuple[float, pd.Series]:
    """
    Portfolio daily returns = sum_j w_j r_{j,t}; then VaR.

    Weights should be **fractions of equity** (e.g. market_value / equity); cash left implicit.

    Returns
    -------
    var_pct : float
        Positive fraction (e.g. 0.03 for 3% one-day loss at confidence).
    port_ret : pd.Series
        Realized portfolio simple returns in window (for diagnostics).
    """
    if returns_df.empty:
        return float("nan"), pd.Series(dtype=float)

    w_ser = pd.Series({str(k).strip().upper(): float(v) for k, v in weights.items()})
    cols = [c for c in returns_df.columns if c in w_ser.index and float(w_ser[c]) != 0.0]
    if not cols:
        return 0.0, pd.Series(dtype=float)

    sub = returns_df[cols].astype(float)
    w = w_ser.reindex(sub.columns).fillna(0.0)
    port = (sub * w).sum(axis=1)
    port = port.dropna()
    if len(port) < 5:
        return float("nan"), port

    if method == "historical":
        var = compute_historical_var(port, confidence)
    elif method == "parametric":
        var = compute_parametric_var(port, confidence)
    elif method == "monte_carlo":
        var = compute_monte_carlo_var(sub, w, confidence)
    else:
        var = compute_historical_var(port, confidence)

    if var != var:  # NaN
        return float("nan"), port
    return float(max(0.0, var)), port
