"""
Value-at-Risk (VaR) and Expected Shortfall (CVaR).

Rolling Historical, Parametric (Gaussian), and Conditional VaR (CVaR)
for portfolio daily returns. Standalone module.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import norm


def historical_var(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
) -> pd.Series:
    """
    Rolling Historical VaR.

    For each day, VaR = negative quantile of last `window` returns
    at the given confidence level.
    VaR(95%) = -quantile(returns, 0.05)
    Returns positive number (loss amount).
    """
    if returns.empty:
        return pd.Series(dtype=float)
    q = 1.0 - confidence
    roll = returns.rolling(window=window, min_periods=max(2, window // 2))
    quant = roll.quantile(q)
    # VaR as positive loss: -quantile (so when return is -0.02, VaR = 0.02)
    return (-quant).reindex(returns.index)


def parametric_var(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
) -> pd.Series:
    """
    Parametric (Gaussian) VaR.

    Assumes normally distributed returns.
    VaR = mu - z_score * sigma
    where z_score = norm.ppf(1 - confidence)
    Returns positive number (loss amount).
    """
    if returns.empty:
        return pd.Series(dtype=float)
    z = norm.ppf(1.0 - confidence)
    roll = returns.rolling(window=window, min_periods=max(2, window // 2))
    mu = roll.mean()
    sigma = roll.std()
    # VaR = -(mu + z*sigma) when z is negative (e.g. 0.95 -> z=-1.65), so loss = -mu + 1.65*sigma
    # Convention: VaR is the loss level, so we want positive number: loss = -quantile = -(mu + z*sigma)
    var = -(mu + z * sigma)
    return var.reindex(returns.index)


def conditional_var(
    returns: pd.Series,
    confidence: float = 0.95,
    window: int = 252,
) -> pd.Series:
    """
    CVaR / Expected Shortfall.

    Average loss beyond the VaR threshold.
    CVaR = -mean(returns[returns < -VaR])
    Returns positive number (loss amount).
    """
    if returns.empty:
        return pd.Series(dtype=float)
    hvar = historical_var(returns, confidence=confidence, window=window)
    out = pd.Series(index=returns.index, dtype=float)
    for i in range(len(returns)):
        if i < window - 1:
            start = 0
        else:
            start = i - window + 1
        window_ret = returns.iloc[start : i + 1]
        var_i = hvar.iloc[i]
        if pd.isna(var_i) or var_i <= 0:
            out.iloc[i] = np.nan
            continue
        tail = window_ret[window_ret < -var_i]
        if tail.empty:
            out.iloc[i] = var_i
        else:
            out.iloc[i] = float(-tail.mean())
    return out


def portfolio_var_report(
    daily_equity: pd.DataFrame,
    confidence_levels: list[float] | None = None,
    holding_period_days: int = 1,
    window: int = 252,
) -> dict:
    """
    Full VaR report for the portfolio.

    Builds daily returns from daily_equity, computes Historical VaR (95%, 99%),
    CVaR (95%), and scales 1-day VaR to holding_period using sqrt(T) rule.
    Also computes VaR breach count and breach rate (95%).
    Returns dict with all VaR metrics (as decimals, e.g. 0.02 for 2%).
    """
    if confidence_levels is None:
        confidence_levels = [0.90, 0.95, 0.99]
    out = {}
    if daily_equity.empty or "equity" not in daily_equity.columns:
        out["var_95_1d"] = np.nan
        out["var_99_1d"] = np.nan
        out["cvar_95"] = np.nan
        out["var_95_5d"] = np.nan
        out["var_breach_rate_95"] = np.nan
        out["var_breach_count_95"] = 0
        return out

    eq = daily_equity.sort_values("date") if "date" in daily_equity.columns else daily_equity
    if "date" in eq.columns:
        equity_series = eq.set_index("date")["equity"]
    else:
        equity_series = eq["equity"]
    returns = equity_series.pct_change().dropna()
    if returns.empty or len(returns) < 2:
        out["var_95_1d"] = np.nan
        out["var_99_1d"] = np.nan
        out["cvar_95"] = np.nan
        out["var_95_5d"] = np.nan
        out["var_breach_rate_95"] = np.nan
        out["var_breach_count_95"] = 0
        return out

    hvar_95 = historical_var(returns, confidence=0.95, window=window)
    hvar_99 = historical_var(returns, confidence=0.99, window=window)
    cvar_95_series = conditional_var(returns, confidence=0.95, window=window)

    var_95_1d = float(hvar_95.dropna().iloc[-1]) if not hvar_95.dropna().empty else np.nan
    var_99_1d = float(hvar_99.dropna().iloc[-1]) if not hvar_99.dropna().empty else np.nan
    cvar_95 = float(cvar_95_series.dropna().iloc[-1]) if not cvar_95_series.dropna().empty else np.nan

    # Scale to holding period: VaR(T) ≈ VaR(1d) * sqrt(T)
    scale = np.sqrt(holding_period_days)
    var_95_5d = var_95_1d * scale if not np.isnan(var_95_1d) else np.nan

    out["var_95_1d"] = var_95_1d
    out["var_99_1d"] = var_99_1d
    out["cvar_95"] = cvar_95
    out["var_95_5d"] = var_95_5d

    # Breach: day when actual loss exceeded predicted VaR (loss = -return, so return < -VaR)
    aligned = hvar_95.reindex(returns.index).dropna()
    common = returns.index.intersection(aligned.index)
    if len(common) > 0:
        r = returns.loc[common]
        v = aligned.loc[common]
        breaches = (r < -v).sum()
        out["var_breach_count_95"] = int(breaches)
        out["var_breach_rate_95"] = float(breaches / len(common))
    else:
        out["var_breach_count_95"] = 0
        out["var_breach_rate_95"] = np.nan

    # Optional: store by confidence for report
    for c in confidence_levels:
        hv = historical_var(returns, confidence=c, window=window)
        last = hv.dropna().iloc[-1] if not hv.dropna().empty else np.nan
        out[f"var_{int(c*100)}_1d"] = float(last) if not np.isnan(last) else np.nan

    return out
