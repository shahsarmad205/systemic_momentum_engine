"""
Load factor risk inputs (sector map, beta cache) from CSV.

Paths are typically set in ``backtest_config.yaml`` under ``risk_factors``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def load_sector_mapping(path: str | Path) -> dict[str, str]:
    """Return ``{ticker: sector}`` from CSV (columns: ticker/symbol, sector)."""
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {}
    if df.empty:
        return {}
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("ticker") or cols.get("symbol")
    scol = cols.get("sector")
    if not tcol or not scol:
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        t = str(row[tcol]).strip().upper()
        s = str(row[scol]).strip()
        if t and s and s.lower() != "nan":
            out[t] = s
    return out


def load_beta_cache(path: str | Path) -> dict[str, float]:
    """Return ``{ticker: beta}`` from CSV (columns: ticker/symbol, beta)."""
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        df = pd.read_csv(p)
    except Exception:
        return {}
    if df.empty:
        return {}
    cols = {c.lower(): c for c in df.columns}
    tcol = cols.get("ticker") or cols.get("symbol")
    bcol = cols.get("beta")
    if not tcol or not bcol:
        return {}
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        t = str(row[tcol]).strip().upper()
        try:
            b = float(row[bcol])
        except Exception:
            continue
        if not t or not (b == b) or not np.isfinite(b):
            continue
        out[t] = b
    return out


def compute_beta_ols(
    ticker_returns: pd.Series,
    market_returns: pd.Series,
    *,
    min_obs: int = 30,
) -> float | None:
    """Simple covariance/variance beta (aligned index)."""
    if ticker_returns is None or market_returns is None:
        return None
    both = pd.concat(
        {"s": pd.to_numeric(ticker_returns, errors="coerce"),
         "m": pd.to_numeric(market_returns, errors="coerce")},
        axis=1,
    ).dropna()
    if len(both) < min_obs:
        return None
    m_var = float(both["m"].var(ddof=0))
    if m_var <= 0 or not np.isfinite(m_var):
        return None
    cov = float(
        ((both["s"] - both["s"].mean()) * (both["m"] - both["m"].mean())).mean()
    )
    b = cov / m_var
    return float(b) if np.isfinite(b) else None
