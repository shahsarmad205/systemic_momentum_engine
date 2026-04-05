# region imports
from __future__ import annotations
try:
    from AlgorithmImports import *
except ImportError:
    pass
# endregion
"""
Consolidated Quantitative and Risk Utilities.
Combines returns loading, volatility scaling, risk loading, and sector aggregates.
"""

from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from utils.market_data import _cache_path

# --- Sector Mapping ---
SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "META": "Technology", "GOOG": "Technology", "AVGO": "Technology",
    "ADBE": "Technology", "CRM": "Technology", "AMD": "Technology",
    "INTC": "Technology", "ORCL": "Technology", "CSCO": "Technology",
    "IBM": "Technology", "QCOM": "Technology", "TXN": "Technology",
    "ACN": "Technology",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "JPM": "Financials", "V": "Financials", "MA": "Financials",
    "BAC": "Financials", "GS": "Financials", "AXP": "Financials",
    "BLK": "Financials", "MS": "Financials", "C": "Financials",
    "WFC": "Financials", "SCHW": "Financials",
    "JNJ": "Healthcare", "UNH": "Healthcare", "ABBV": "Healthcare",
    "PFE": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
    "TMO": "Healthcare", "DHR": "Healthcare", "ABT": "Healthcare",
    "MDT": "Healthcare", "AMGN": "Healthcare",
    "XOM": "Energy", "CVX": "Energy",
    "WMT": "Consumer Staples", "PG": "Consumer Staples",
    "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "PM": "Consumer Staples",
    "UNP": "Industrials", "CAT": "Industrials", "BA": "Industrials",
    "MMM": "Industrials", "GE": "Industrials",
    "DIS": "Communication Services",
    "NEE": "Utilities",
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "XLK": "ETF", "XLF": "ETF", "XLE": "ETF", "XLV": "ETF",
    "VTI": "ETF", "ARKK": "ETF",
}

def get_sector(ticker: str) -> str:
    return SECTOR_MAP.get(ticker.upper(), "Other")

def get_sectors() -> list[str]:
    sectors = set(SECTOR_MAP.values()) | {"Other"}
    return sorted(sectors)

def tickers_by_sector(sector_map: dict[str, str] | None = None) -> dict[str, list[str]]:
    m = sector_map or SECTOR_MAP
    out: dict[str, list[str]] = {}
    for ticker, sector in m.items():
        out.setdefault(sector, []).append(ticker)
    return out

# --- Returns Loading ---
def _close_returns_series(cache_dir: Path, ticker: str) -> pd.Series | None:
    path = _cache_path(str(cache_dir), ticker)
    if not path.is_file():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return None
    if df.empty or "Close" not in df.columns:
        return None
    idx = df.index
    if not isinstance(idx, pd.DatetimeIndex):
        idx = pd.to_datetime(idx, errors="coerce")
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_localize(None)
    close = pd.to_numeric(df["Close"], errors="coerce")
    close.index = idx
    ret = close.sort_index().pct_change()
    return ret.dropna()

def load_aligned_returns(
    tickers: list[str],
    cache_dir: Path,
    lookback: int,
    *,
    end_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    lookback = int(max(5, lookback))
    series_map: dict[str, pd.Series] = {}
    for raw in tickers:
        t = str(raw).strip()
        if not t: continue
        s = _close_returns_series(cache_dir, t)
        if s is None or len(s) < lookback: continue
        series_map[t.upper()] = s
    if not series_map: return pd.DataFrame(), []
    df = pd.DataFrame(series_map).dropna(how="any")
    if end_date is not None:
        end_ts = pd.Timestamp(end_date)
        df = df.loc[df.index <= end_ts]
    if len(df) < lookback: return pd.DataFrame(), list(series_map.keys())
    df = df.iloc[-lookback:]
    return df, list(df.columns)

# --- Volatility Sizing ---
def compute_vol_scaled_weight(
    raw_weight: float,
    volatility_20: float,
    target_vol: float = 0.15,
    min_vol_floor: float = 0.05,
    max_scale_cap: float = 3.0,
) -> float:
    vol = max(float(volatility_20), float(min_vol_floor))
    scale = float(target_vol) / vol
    if scale > max_scale_cap: scale = float(max_scale_cap)
    result = float(raw_weight) * scale
    result = float(round(result, 12))
    return 0.0 if result == -0.0 else result

def compute_portfolio_vol_weights(
    weights: dict[str, float],
    volatilities: dict[str, float],
    target_vol: float = 0.15,
) -> dict[str, float]:
    scaled: dict[str, float] = {}
    for ticker, w in weights.items():
        vol = float(volatilities.get(ticker, target_vol))
        scaled[ticker] = compute_vol_scaled_weight(w, vol, target_vol=target_vol)
    total_abs = sum(abs(v) for v in scaled.values())
    if total_abs <= 0.0: return {k: 0.0 for k in weights.keys()}
    return {k: v / total_abs for k, v in scaled.items()}

def compute_realized_vol_annualized(returns: pd.Series, window: int = 20) -> pd.Series:
    s = returns.astype(float)
    std = s.rolling(window, min_periods=1).std(ddof=0)
    return (std * np.sqrt(252.0)).astype(float)

def compute_vol_target_scaling_factor(
    vol: float | np.floating[Any],
    *,
    target_vol: float = 0.15,
    min_vol_floor: float = 0.05,
    max_scale_cap: float = 3.0,
) -> float:
    v = max(float(vol), float(min_vol_floor))
    scale = float(target_vol) / v
    return float(min(scale, float(max_scale_cap)))

def apply_vol_kill_switch(
    positions: float | np.ndarray | pd.Series,
    vol: float | np.ndarray | pd.Series,
    *,
    threshold_annual: float,
    cut_factor: float,
) -> float | np.ndarray | pd.Series:
    thr, cf = float(threshold_annual), float(cut_factor)
    if isinstance(positions, pd.Series):
        v_arr = pd.to_numeric(vol, errors="coerce").astype(float).to_numpy()
        trig = np.isfinite(v_arr) & (v_arr > thr)
        return positions.astype(float).where(~pd.Series(trig, index=positions.index), positions.astype(float) * cf)
    if isinstance(positions, np.ndarray):
        v_arr, p_arr = np.asarray(vol, dtype=float), np.asarray(positions, dtype=float)
        trig = np.isfinite(v_arr) & (v_arr > thr)
        return np.where(trig, p_arr * cf, p_arr).astype(float)
    pv, p = float(vol), float(positions)
    return p * cf if np.isfinite(pv) and pv > thr else p

# --- Risk Loading ---
def load_sector_mapping(path: str | Path) -> dict[str, str]:
    p = Path(path)
    if not p.is_file(): return {}
    try: df = pd.read_csv(p)
    except Exception: return {}
    if df.empty: return {}
    cols = {c.lower(): c for c in df.columns}
    tcol, scol = cols.get("ticker") or cols.get("symbol"), cols.get("sector")
    if not tcol or not scol: return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        t, s = str(row[tcol]).strip().upper(), str(row[scol]).strip()
        if t and s and s.lower() != "nan": out[t] = s
    return out

def load_beta_cache(path: str | Path) -> dict[str, float]:
    p = Path(path)
    if not p.is_file(): return {}
    try: df = pd.read_csv(p)
    except Exception: return {}
    if df.empty: return {}
    cols = {c.lower(): c for c in df.columns}
    tcol, bcol = cols.get("ticker") or cols.get("symbol"), cols.get("beta")
    if not tcol or not bcol: return {}
    out: dict[str, float] = {}
    for _, row in df.iterrows():
        t = str(row[tcol]).strip().upper()
        try: b = float(row[bcol])
        except Exception: continue
        if t and np.isfinite(b): out[t] = b
    return out

def compute_beta_ols(ticker_returns: pd.Series, market_returns: pd.Series, *, min_obs: int = 30) -> float | None:
    if ticker_returns is None or market_returns is None: return None
    both = pd.concat({"s": pd.to_numeric(ticker_returns, errors="coerce"), "m": pd.to_numeric(market_returns, errors="coerce")}, axis=1).dropna()
    if len(both) < min_obs: return None
    m_var = float(both["m"].var(ddof=0))
    if m_var <= 0 or not np.isfinite(m_var): return None
    cov = float(((both["s"] - both["s"].mean()) * (both["m"] - both["m"].mean())).mean())
    b = cov / m_var
    return float(b) if np.isfinite(b) else None

# --- Sector Aggregates ---
def compute_sector_aggregates(
    price_data: dict[str, pd.DataFrame],
    sector_map: dict[str, str] | None = None,
    sentiment_by_ticker: dict[str, float] | None = None,
) -> pd.DataFrame:
    sector_map = sector_map or SECTOR_MAP
    sector_to_tickers = tickers_by_sector(sector_map)
    all_dates = set()
    for df in price_data.values(): all_dates.update(df.index)
    dates = sorted(all_dates)
    ticker_momentum, ticker_vol = {}, {}
    for ticker, df in price_data.items():
        if df.empty or "Close" not in df.columns: continue
        close = df["Close"]
        ret = close.pct_change()
        ticker_momentum[ticker] = close.pct_change(20)
        ticker_vol[ticker] = ret.rolling(20).std()
    sectors_in_universe = {get_sector(t) for t in price_data}
    columns = []
    for sector in sorted(sectors_in_universe):
        columns.extend([f"sector_{sector}_{k}" for k in ["momentum", "volatility", "sentiment"]])
    result = pd.DataFrame(index=dates, columns=columns, dtype=float).fillna(0.0)
    for sector in sectors_in_universe:
        tickers_in_sector = [t for t in sector_to_tickers.get(sector, []) if t in price_data]
        if not tickers_in_sector: continue
        for col_pfx, ticker_map in [("momentum", ticker_momentum), ("volatility", ticker_vol)]:
            series_df = pd.DataFrame({t: ticker_map[t].reindex(dates) for t in tickers_in_sector if t in ticker_map})
            if not series_df.empty: result[f"sector_{sector}_{col_pfx}"] = series_df.mean(axis=1)
        if sentiment_by_ticker:
            s_vals = [sentiment_by_ticker.get(t, 0.0) for t in tickers_in_sector]
            result[f"sector_{sector}_sentiment"] = sum(s_vals)/len(s_vals) if s_vals else 0.0
    return result.fillna(0.0)

def apply_sector_adjustment(
    signal_data: dict[str, pd.DataFrame],
    sector_aggregates: pd.DataFrame,
    sector_map: dict[str, str],
    momentum_weight: float = 0.1,
    volatility_weight: float = -0.05,
    sentiment_weight: float = 0.1,
) -> None:
    # Use internal implementation of classify_final_signal to avoid cyclic imports
    def _classify(val):
        if val > 0.5: return "Bullish"
        if val < -0.5: return "Bearish"
        return "Neutral"
    for ticker, sig_df in signal_data.items():
        sector = sector_map.get(ticker, "Other")
        if f"sector_{sector}_momentum" not in sector_aggregates.columns: continue
        agg = sector_aggregates.reindex(sig_df.index).fillna(0)
        adj = sig_df["adjusted_score"].values + momentum_weight * agg[f"sector_{sector}_momentum"].values + \
              volatility_weight * agg[f"sector_{sector}_volatility"].values + sentiment_weight * agg[f"sector_{sector}_sentiment"].values
        sig_df["adjusted_score"] = adj
        sig_df["signal"] = pd.Series(adj, index=sig_df.index).apply(_classify)
