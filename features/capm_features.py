"""
CAPM features: Jensen's alpha, market beta, and residual volatility.

Rolling 60-day regression of stock returns on SPY returns:
  beta = Cov(stock, SPY) / Var(SPY)
  alpha = mean(stock) - beta * mean(SPY)
  residual_vol = std(stock - beta * SPY)
Alpha is then z-scored over trailing 252 days.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

CAPM_WINDOW = 60
CAPM_ZSCORE_WINDOW = 252


def compute_capm_features(
    stock_returns: pd.Series,
    spy_returns: pd.Series,
    window: int = CAPM_WINDOW,
    zscore_window: int = CAPM_ZSCORE_WINDOW,
) -> pd.DataFrame:
    """
    Compute capm_alpha (z-scored), capm_beta, capm_residual_vol.

    Parameters
    ----------
    stock_returns : pd.Series
        Daily returns of the stock (index = date).
    spy_returns : pd.Series
        Daily returns of SPY (market). Will be aligned to stock_returns index.
    window : int
        Rolling window for beta/alpha/residual vol (default 60).
    zscore_window : int
        Window for z-scoring alpha (default 252).

    Returns
    -------
    pd.DataFrame
        Index aligned to stock_returns. Columns: capm_alpha, capm_beta, capm_residual_vol.
    """
    # Align SPY to stock index (forward-fill then dropna so we only have common dates)
    spy_aligned = spy_returns.reindex(stock_returns.index).ffill().bfill().fillna(0.0)
    stock = stock_returns.astype(float).fillna(0.0)

    n = len(stock)
    capm_beta = pd.Series(np.nan, index=stock.index, dtype=float)
    capm_alpha_raw = pd.Series(np.nan, index=stock.index, dtype=float)
    capm_residual_vol = pd.Series(np.nan, index=stock.index, dtype=float)

    for i in range(window, n + 1):
        r_stock = stock.iloc[i - window:i]
        r_spy = spy_aligned.iloc[i - window:i]
        if r_spy.var() <= 1e-12:
            beta = 0.0
        else:
            beta = float(r_stock.cov(r_spy) / r_spy.var())
        beta = max(0.0, min(beta, 5.0))  # keep beta non-negative and bounded
        alpha = float(r_stock.mean() - beta * r_spy.mean())
        residual = r_stock.values - beta * r_spy.values
        res_vol = float(np.nanstd(residual)) if len(residual) > 1 else 0.0

        capm_beta.iloc[i - 1] = beta
        capm_alpha_raw.iloc[i - 1] = alpha
        capm_residual_vol.iloc[i - 1] = res_vol

    # Z-score alpha over trailing zscore_window
    alpha_mean = capm_alpha_raw.rolling(zscore_window, min_periods=min(60, zscore_window // 2)).mean()
    alpha_std = capm_alpha_raw.rolling(zscore_window, min_periods=min(60, zscore_window // 2)).std()
    capm_alpha = (capm_alpha_raw - alpha_mean) / alpha_std.replace(0, np.nan).fillna(0.0)

    return pd.DataFrame(
        {"capm_alpha": capm_alpha, "capm_beta": capm_beta, "capm_residual_vol": capm_residual_vol},
        index=stock.index,
    )
