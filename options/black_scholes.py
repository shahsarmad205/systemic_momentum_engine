"""
Black-Scholes option pricing and analysis.

Standalone module: theoretical option prices and Greeks using
historical volatility as an IV proxy. No live options data required.
"""

from __future__ import annotations

from math import exp, log, sqrt

import numpy as np
import pandas as pd
from scipy.stats import norm


def bs_price(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> float:
    """
    Black-Scholes option price.

    S: current stock price
    K: strike price
    T: time to expiry in years
    r: risk-free rate (annualised)
    sigma: volatility (annualised)
    option_type: 'call' or 'put'

    Returns: theoretical option price
    """
    if T <= 0 or sigma <= 0:
        return 0.0
    sqrt_T = sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    N = norm.cdf
    if option_type == "call":
        return S * N(d1) - K * exp(-r * T) * N(d2)
    else:
        return K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


def bs_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str = "call",
) -> dict[str, float]:
    """
    Black-Scholes Greeks.

    Returns: delta, gamma, theta (per day), vega (per 1% vol), rho (per 1% rate).
    """
    out = {"delta": 0.0, "gamma": 0.0, "theta": 0.0, "vega": 0.0, "rho": 0.0}
    if T <= 0 or sigma <= 0:
        return out
    sqrt_T = sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    N = norm.cdf
    n = norm.pdf
    nd1 = n(d1)
    out["delta"] = N(d1) if option_type == "call" else N(d1) - 1.0
    out["gamma"] = nd1 / (S * sigma * sqrt_T) if S * sigma * sqrt_T > 1e-12 else 0.0
    # Theta: per year first, then per day
    if option_type == "call":
        theta_year = -S * nd1 * sigma / (2 * sqrt_T) - r * K * exp(-r * T) * N(d2)
    else:
        theta_year = -S * nd1 * sigma / (2 * sqrt_T) + r * K * exp(-r * T) * N(-d2)
    out["theta"] = theta_year / 365.0
    out["vega"] = S * sqrt_T * nd1 * 0.01  # per 1% move in vol
    out["rho"] = K * T * exp(-r * T) * (N(d2) if option_type == "call" else -N(-d2)) * 0.01  # per 1% in r
    return out


def implied_vol_from_historical(price_series: pd.Series, window: int = 30) -> pd.Series:
    """
    Estimate implied volatility proxy from historical realised volatility (annualised).

    Uses rolling std of log returns * sqrt(252).
    """
    if price_series is None or len(price_series) < 2:
        return pd.Series(dtype=float)
    log_ret = np.log(price_series / price_series.shift(1)).dropna()
    rolling_std = log_ret.rolling(window=window, min_periods=max(2, window // 2)).std()
    return (rolling_std * np.sqrt(252)).fillna(np.nan)


def options_strategy_signals(
    stock_data: pd.DataFrame,
    signal_data: pd.DataFrame,
    risk_free_rate: float = 0.04,
    expiry_days: int = 30,
) -> pd.DataFrame:
    """
    For each Bullish signal, compute:
    - ATM call price (strike = current price)
    - 5% OTM call price (strike = price * 1.05)
    - Delta of ATM call
    - Vega exposure
    - Break-even move required
    - Cost of hedging via put (ATM put price)

    Returns DataFrame with these columns added to signal_data.
    """
    out = signal_data.copy()
    if out.empty:
        return out
    price_col = "Close" if "Close" in stock_data.columns else ("AdjClose" if "AdjClose" in stock_data.columns else stock_data.columns[0])
    prices = stock_data[price_col].reindex(out.index).ffill().bfill()
    sigma_series = implied_vol_from_historical(prices, window=30)
    T = expiry_days / 365.0
    atm_call = []
    otm5_call = []
    atm_delta = []
    vega_exp = []
    breakeven_pct = []
    put_hedge = []
    for idx in out.index:
        S = float(prices.get(idx, np.nan))
        sigma = float(sigma_series.get(idx, 0.20))
        if np.isnan(S) or S <= 0 or sigma <= 0:
            sigma = 0.20
        K_atm = S
        K_otm5 = S * 1.05
        c_atm = bs_price(S, K_atm, T, risk_free_rate, sigma, "call")
        c_otm5 = bs_price(S, K_otm5, T, risk_free_rate, sigma, "call")
        g = bs_greeks(S, K_atm, T, risk_free_rate, sigma, "call")
        p_atm = bs_price(S, K_atm, T, risk_free_rate, sigma, "put")
        atm_call.append(c_atm)
        otm5_call.append(c_otm5)
        atm_delta.append(g["delta"])
        vega_exp.append(g["vega"])
        breakeven_pct.append(100.0 * (c_atm / S) if S > 0 else np.nan)
        put_hedge.append(p_atm)
    out["atm_call_price"] = atm_call
    out["otm5_call_price"] = otm5_call
    out["atm_call_delta"] = atm_delta
    out["vega_exposure"] = vega_exp
    out["breakeven_move_pct"] = breakeven_pct
    out["put_hedge_cost"] = put_hedge
    return out
