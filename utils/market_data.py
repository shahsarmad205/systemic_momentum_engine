"""
Market Data Layer
==================
Fetch OHLCV data from multiple providers with optional local caching.
Supports Yahoo Finance (default), Alpaca, and Finnhub.

Usage:
    from utils.market_data import get_ohlcv

    df = get_ohlcv("AAPL", "2020-01-01", "2024-01-01", provider="yahoo", use_cache=True)
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd


# Default cache directory (under project root)
# We cache one Parquet file per ticker under this directory, then slice by date.
DEFAULT_CACHE_DIR = "data/cache"
OHLCV_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


def _safe_filename(ticker: str) -> str:
    """Sanitize ticker for use in file paths."""
    return re.sub(r"[^\w\-.]", "_", ticker.upper())


def _cache_path(cache_dir: str, ticker: str) -> Path:
    """
    Path for per-ticker cache (Parquet). All history for a ticker is stored in a single file.
    """
    return Path(cache_dir) / f"{_safe_filename(ticker)}.parquet"


def _load_cached(path: Path, cache_ttl_days: int) -> pd.DataFrame | None:
    if not path.exists():
        return None
    if cache_ttl_days > 0:
        mtime = path.stat().st_mtime
        if (datetime.now().timestamp() - mtime) > cache_ttl_days * 86400:
            return None
    try:
        df = pd.read_parquet(path)
        # Require at least the core OHLCV columns; allow extra (AdjClose, Dividends, etc.).
        if not all(col in df.columns for col in OHLCV_COLUMNS):
            return None
        # Ensure DatetimeIndex
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        return df.sort_index()
    except Exception:
        return None


def _save_cache(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Persist all columns so adjusted/auxiliary fields are available on reload.
    df.to_parquet(path)


def _download_yahoo(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    import yfinance as yf

    def _fetch(period=None, start_=None, end_=None):
        if period:
            raw = yf.download(
                ticker,
                period=period,
                progress=False,
                auto_adjust=False,
                actions=True,
            )
        else:
            raw = yf.download(
                ticker,
                start=start_,
                end=end_,
                progress=False,
                auto_adjust=False,
                actions=True,
            )
        if raw.empty:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        for col in OHLCV_COLUMNS:
            if col not in raw.columns:
                return pd.DataFrame(columns=OHLCV_COLUMNS)
        df = raw[OHLCV_COLUMNS].dropna()
        # Preserve adjusted close and corporate actions when available
        if "Adj Close" in raw.columns:
            df["AdjClose"] = raw["Adj Close"].reindex(df.index)
        if "Dividends" in raw.columns:
            df["Dividends"] = raw["Dividends"].reindex(df.index)
        if "Stock Splits" in raw.columns:
            df["StockSplits"] = raw["Stock Splits"].reindex(df.index)
        return df

    # First try date range
    df = _fetch(start_=start, end_=end)
    # If too few rows, retry with period to help newer tickers / API gaps
    if len(df) < 30:
        df_6mo = _fetch(period="6mo")
        if len(df_6mo) > len(df):
            df = df_6mo
    if len(df) < 30:
        df_1y = _fetch(period="1y")
        if len(df_1y) > len(df):
            df = df_1y
    return df


def _attach_delisted_date(df: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """
    Attach a best-effort delisted_date column and drop rows after it.

    Many providers (including Yahoo via yfinance) simply stop returning
    data once a ticker is delisted. We treat the last available trading
    date before the requested end as the effective delisted_date and
    add it as a constant column. If the last date is >= end, the ticker
    is assumed to be live and delisted_date is NaT.
    """
    if df.empty:
        return df
    df = df.sort_index()
    last_date = df.index.max()
    # If the last available bar is strictly before the requested end date,
    # consider this our effective delisting date.
    delisted_date = last_date if last_date < end else pd.NaT
    df = df.loc[:last_date].copy()
    df["delisted_date"] = delisted_date
    return df


def _download_alpaca(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch OHLCV from Alpaca. Requires ALPACA_API_KEY and ALPACA_SECRET_KEY (or PAPER_KEY/SECRET)."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError:
        raise RuntimeError(
            "Alpaca provider requires alpaca-py. Install with: pip install alpaca-py"
        ) from None

    api_key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not secret:
        raise RuntimeError(
            "Alpaca provider requires ALPACA_API_KEY and ALPACA_SECRET_KEY (or APCA_* env vars)"
        )

    client = StockHistoricalDataClient(api_key, secret)
    request = StockBarsRequest(
        symbol_or_symbols=ticker,
        timeframe=TimeFrame.Day,
        start=start.to_pydatetime(),
        end=end.to_pydatetime(),
    )
    bars = client.get_stock_bars(request)
    if not bars or ticker not in bars.data:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    b = bars.data[ticker]
    df = pd.DataFrame(
        {
            "Open": [x.open for x in b],
            "High": [x.high for x in b],
            "Low": [x.low for x in b],
            "Close": [x.close for x in b],
            "Volume": [x.volume for x in b],
        },
        index=pd.DatetimeIndex([x.timestamp for x in b], name="Date"),
    )
    return df.sort_index()


def _download_finnhub(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch daily candles from Finnhub. Requires FINNHUB_API_KEY."""
    try:
        import finnhub
    except ImportError:
        raise RuntimeError(
            "Finnhub provider requires finnhub-python. Install with: pip install finnhub-python"
        ) from None

    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError("Finnhub provider requires FINNHUB_API_KEY environment variable")

    client = finnhub.Client(api_key=api_key)
    _from = int(start.timestamp())
    _to = int(end.timestamp())
    # candle: o, h, l, c, v, t
    data = client.stock_candles(ticker, "D", _from, _to)
    if not data or "c" not in data or not data["c"]:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = pd.DataFrame(
        {
            "Open": data["o"],
            "High": data["h"],
            "Low": data["l"],
            "Close": data["c"],
            "Volume": data.get("v", [0] * len(data["c"])),
        },
        index=pd.DatetimeIndex(
            pd.to_datetime(data["t"], unit="s"),
            name="Date",
        ),
    )
    return df.sort_index()


def _download_crypto_ccxt(
    symbol: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    exchange_id: str = "binance",
) -> pd.DataFrame:
    """
    Fetch daily OHLCV for crypto using CCXT.

    Notes
    -----
    - Assumes symbol in CCXT format, e.g. "BTC/USDT".
    - Uses 1d candles and respects 24/7 trading (no weekend filtering).
    """
    try:
        import ccxt  # type: ignore
    except ImportError:
        raise RuntimeError(
            "Crypto provider requires ccxt. Install with: pip install ccxt"
        ) from None

    if not hasattr(ccxt, exchange_id):
        raise RuntimeError(f"Unknown CCXT exchange '{exchange_id}'")

    exchange = getattr(ccxt, exchange_id)()
    timeframe = "1d"
    since = int(start.timestamp() * 1000)
    ohlcv: list[list] = []
    limit = 1000

    while True:
        batch = exchange.fetch_ohlcv(symbol, timeframe=timeframe, since=since, limit=limit)
        if not batch:
            break
        ohlcv.extend(batch)
        last_ts = batch[-1][0]
        if last_ts >= int(end.timestamp() * 1000):
            break
        since = last_ts + 1_000  # advance 1 second

    if not ohlcv:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = pd.DataFrame(
        {
            "Open": [x[1] for x in ohlcv],
            "High": [x[2] for x in ohlcv],
            "Low": [x[3] for x in ohlcv],
            "Close": [x[4] for x in ohlcv],
            "Volume": [x[5] for x in ohlcv],
        },
        index=pd.DatetimeIndex(pd.to_datetime([x[0] for x in ohlcv], unit="ms"), name="Date"),
    )
    return df.sort_index()


def _build_continuous_futures(
    contract_data: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Construct a simple continuous futures series from per-contract OHLCV.

    Roll logic
    ----------
    - Align all contracts on the union of dates.
    - For each date, pick the contract with the highest recent volume
      (using a 5-day rolling sum as a proxy for the front contract).
    - No back-adjustment is performed; prices are spliced.
    """
    if not contract_data:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    all_dates: set[pd.Timestamp] = set()
    for df in contract_data.values():
        all_dates.update(df.index)
    if not all_dates:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    all_idx = pd.DatetimeIndex(sorted(all_dates), name="Date")

    vol_roll = {}
    aligned = {}
    for symbol, df in contract_data.items():
        if df.empty:
            continue
        df = df.sort_index()
        aligned_df = df.reindex(all_idx)
        aligned[symbol] = aligned_df
        vol_roll[symbol] = aligned_df["Volume"].rolling(5, min_periods=1).sum()

    if not aligned:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    vol_roll_df = pd.DataFrame(vol_roll)
    best_contract = vol_roll_df.idxmax(axis=1)

    out = pd.DataFrame(index=all_idx, columns=OHLCV_COLUMNS, dtype=float)
    for date, contract in best_contract.items():
        if contract not in aligned:
            continue
        row = aligned[contract].loc[date]
        if row.isna().all():
            continue
        for col in OHLCV_COLUMNS:
            out.at[date, col] = row[col]

    out = out.dropna(subset=["Close"])
    return out


def get_ohlcv(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    provider: str = "yahoo",
    cache_dir: str | None = None,
    use_cache: bool = True,
    cache_ttl_days: int = 0,
    include_delisted: bool = True,
    asset_type: str = "equity",  # "equity" | "futures" | "crypto"
    futures_contracts: list[str] | None = None,
    crypto_exchange: str = "binance",
) -> pd.DataFrame:
    """
    Fetch OHLCV data for a ticker over a date range.

    Parameters
    ----------
    ticker : str
        Symbol (e.g. "AAPL", "SPY").
    start_date, end_date : str
        ISO date strings (inclusive range).
    provider : str
        "yahoo" (default), "alpaca", or "finnhub" for equities / ETFs.
    cache_dir : str, optional
        Directory for cached files. Default: data/cache/ohlcv.
    use_cache : bool
        If True, read from cache when available and write after download.
    cache_ttl_days : int
        Cache validity in days; 0 = use cache indefinitely.

    Returns
    -------
    pd.DataFrame
        Index: DatetimeIndex (Date). Columns: Open, High, Low, Close, Volume.
    """
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    # Verbose debug: show requested date range
    print(f"[get_ohlcv] {ticker}: requesting {start.strftime('%Y-%m-%d')} → {end.strftime('%Y-%m-%d')}")
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    path = _cache_path(cache_dir, ticker)

    if use_cache:
        cached = _load_cached(path, cache_ttl_days)
        if cached is not None and not cached.empty:
            cached_start = cached.index.min()
            cached_end = cached.index.max()
            # If the cached history fully covers the requested window, just slice.
            if start >= cached_start and end <= cached_end:
                df = cached.loc[(cached.index >= start) & (cached.index <= end)].copy()
                print(f"[get_ohlcv] {ticker}: cache hit, sliced shape={df.shape}")
                if include_delisted and asset_type == "equity" and not df.empty:
                    df = _attach_delisted_date(df, end)
                return df
            else:
                # Partial coverage: fall through to download additional history.
                # We will merge the newly downloaded data with the existing cache
                # in the save-cache step below.
                print(
                    f"[get_ohlcv] {ticker}: cache partial "
                    f"(cached {cached_start.date()} → {cached_end.date()}, "
                    f"requested {start.date()} → {end.date()}) — downloading extension"
                )

    provider = (provider or "yahoo").lower()

    # --- Dispatch by asset type ------------------------------------
    if asset_type == "crypto":
        df = _download_crypto_ccxt(ticker, start, end, exchange_id=crypto_exchange)
    elif asset_type == "futures":
        if not futures_contracts:
            raise ValueError("futures_contracts must be provided when asset_type='futures'")
        contract_data: dict[str, pd.DataFrame] = {}
        for contract in futures_contracts:
            if provider == "yahoo":
                c_df = _download_yahoo(contract, start, end)
            elif provider == "alpaca":
                c_df = _download_alpaca(contract, start, end)
            elif provider == "finnhub":
                c_df = _download_finnhub(contract, start, end)
            else:
                raise ValueError(f"Unknown provider for futures: {provider}. Use 'yahoo', 'alpaca', or 'finnhub'.")
            if not c_df.empty:
                contract_data[contract] = c_df
        df = _build_continuous_futures(contract_data)
    else:
        # Default: equities / ETFs
        if provider == "yahoo":
            df = _download_yahoo(ticker, start, end)
        elif provider == "alpaca":
            df = _download_alpaca(ticker, start, end)
        elif provider == "finnhub":
            df = _download_finnhub(ticker, start, end)
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'yahoo', 'alpaca', or 'finnhub'.")

    if df is None or df.empty:
        print(f"[get_ohlcv] {ticker}: download returned EMPTY frame")
    else:
        print(f"[get_ohlcv] {ticker}: download shape={df.shape}, date range {df.index.min()} → {df.index.max()}")

    if include_delisted and asset_type == "equity" and not df.empty:
        df = _attach_delisted_date(df, end)

    if use_cache and not df.empty:
        # Merge with any existing cache for this ticker and persist full history.
        try:
            existing = _load_cached(path, cache_ttl_days=0)  # ignore TTL when merging
        except Exception:
            existing = None
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = df.sort_index()
        _save_cache(path, combined)

    return df


def clear_cache(ticker: str | None = None, cache_dir: str | None = None) -> None:
    """
    Clear cached OHLCV data.

    Parameters
    ----------
    ticker : str or None
        If provided, clear only this ticker's cache file. If None, clear all.
    cache_dir : str or None
        Cache directory to operate on. Defaults to DEFAULT_CACHE_DIR.
    """
    cache_dir = cache_dir or DEFAULT_CACHE_DIR
    base = Path(cache_dir)
    if not base.exists():
        return
    if ticker is None:
        for p in base.glob("*.parquet"):
            try:
                p.unlink()
            except OSError:
                pass
    else:
        path = _cache_path(cache_dir, ticker)
        if path.exists():
            try:
                path.unlink()
            except OSError:
                pass
