# region imports
from __future__ import annotations
try:
    from AlgorithmImports import *  # noqa: F403
except ImportError:
    pass
# endregion
"""Explicit market-data provider facade.

This module intentionally keeps the historical ``get_ohlcv`` import path alive,
but provider selection is now explicit. Research and production code must pass
``provider=...`` or set ``TREND_DATA_PROVIDER``/``DATA_PROVIDER``. The facade
does not fall back from WRDS/Lean/etc. to Yahoo.
"""

import logging
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from providers.cache_manager import (
    DEFAULT_CACHE_DIR,
    OHLCV_COLUMNS,
    cache_path as _cache_path,
    clear_provider_cache,
    df_overlaps_window as _df_overlaps_window,
    load_cached as _load_cached,
    save_cache as _save_cache,
)
from providers.wrds_adapter import WRDSProvider
from providers.yahoo_adapter import YahooProvider

logger = logging.getLogger(__name__)

SUPPORTED_PROVIDERS = {"wrds", "yahoo", "alpaca", "finnhub", "lean"}


@dataclass(frozen=True)
class ProviderConfig:
    """Explicit provider configuration for OHLCV requests."""

    provider: str
    cache_dir: str = DEFAULT_CACHE_DIR
    use_cache: bool = True
    cache_ttl_days: int = 0
    wrds_username: str | None = None
    ticker_to_permno: dict[str, int] | None = None
    as_of_date: str | pd.Timestamp | None = None


def resolve_data_provider(provider: str | None = None) -> str:
    """Resolve a configured provider without implicit Yahoo degradation."""
    raw = provider or os.environ.get("TREND_DATA_PROVIDER") or os.environ.get("DATA_PROVIDER")
    if raw is None or not str(raw).strip():
        raise ValueError(
            "data provider must be explicit: set data.provider in config or pass provider='wrds'/'yahoo'"
        )
    resolved = str(raw).strip().lower()
    if resolved not in SUPPORTED_PROVIDERS:
        supported = ", ".join(sorted(SUPPORTED_PROVIDERS))
        raise ValueError(f"unknown data provider '{resolved}'. Supported providers: {supported}")
    return resolved


def get_provider(config: ProviderConfig):
    """Build the explicit provider adapter for a request."""
    provider = resolve_data_provider(config.provider)
    if provider == "wrds":
        return WRDSProvider(
            username=config.wrds_username,
            cache_dir=config.cache_dir if "wrds" in str(config.cache_dir).lower() else "data/cache/wrds",
            cache_ttl_days=config.cache_ttl_days,
            ticker_to_permno=config.ticker_to_permno,
            as_of_date=config.as_of_date,
        )
    if provider == "yahoo":
        return YahooProvider()
    return None


def _attach_delisted_date(df: pd.DataFrame, end: pd.Timestamp) -> pd.DataFrame:
    """Attach a best-effort delisted_date for non-WRDS providers."""
    if df.empty:
        return df
    df = df.sort_index()
    last_date = df.index.max()
    delisted_date = last_date if last_date < end else pd.NaT
    out = df.loc[:last_date].copy()
    out["delisted_date"] = delisted_date
    return out


def _download_alpaca(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch OHLCV from Alpaca. Requires ALPACA_API_KEY and ALPACA_SECRET_KEY."""
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame
    except ImportError as exc:
        raise RuntimeError("Alpaca provider requires alpaca-py to be installed") from exc

    api_key = os.environ.get("ALPACA_API_KEY") or os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("ALPACA_SECRET_KEY") or os.environ.get("APCA_API_SECRET_KEY")
    if not api_key or not secret:
        raise RuntimeError("Alpaca provider requires ALPACA_API_KEY and ALPACA_SECRET_KEY")

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
    return pd.DataFrame(
        {
            "Open": [x.open for x in b],
            "High": [x.high for x in b],
            "Low": [x.low for x in b],
            "Close": [x.close for x in b],
            "Volume": [x.volume for x in b],
        },
        index=pd.DatetimeIndex([x.timestamp for x in b], name="Date"),
    ).sort_index()


def _download_finnhub(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch daily candles from Finnhub. Requires FINNHUB_API_KEY."""
    try:
        import finnhub
    except ImportError as exc:
        raise RuntimeError("Finnhub provider requires finnhub-python to be installed") from exc

    api_key = os.environ.get("FINNHUB_API_KEY")
    if not api_key:
        raise RuntimeError("Finnhub provider requires FINNHUB_API_KEY")

    client = finnhub.Client(api_key=api_key)
    data = client.stock_candles(ticker, "D", int(start.timestamp()), int(end.timestamp()))
    if not data or "c" not in data or not data["c"]:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    return pd.DataFrame(
        {
            "Open": data["o"],
            "High": data["h"],
            "Low": data["l"],
            "Close": data["c"],
            "Volume": data.get("v", [0] * len(data["c"])),
        },
        index=pd.DatetimeIndex(pd.to_datetime(data["t"], unit="s"), name="Date"),
    ).sort_index()


def _download_lean(ticker: str, start: pd.Timestamp, end: pd.Timestamp) -> pd.DataFrame:
    """Fetch daily OHLCV from local Lean data storage. No external fallback."""
    symbol = str(ticker).lower()
    lean_data_path = Path("LeanCloud/data/equity/usa/daily") / f"{symbol}.zip"
    if not lean_data_path.exists():
        raise RuntimeError(f"Lean provider selected but no local Lean zip exists for {ticker}")

    with zipfile.ZipFile(lean_data_path, "r") as zf:
        csv_name = f"{symbol}.csv"
        if csv_name not in zf.namelist():
            csv_name = zf.namelist()[0]
        with zf.open(csv_name) as fh:
            df = pd.read_csv(
                fh,
                header=None,
                names=["DateTime", "Open", "High", "Low", "Close", "Volume"],
            )

    df["Date"] = pd.to_datetime(df["DateTime"].str.split(" ").str[0], format="%Y%m%d")
    df = df.set_index("Date")
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col] / 10000.0
    return df.loc[(df.index >= start) & (df.index <= end), OHLCV_COLUMNS].sort_index()


def _build_continuous_futures(contract_data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Construct a simple volume-selected continuous futures series."""
    if not contract_data:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    all_idx = pd.DatetimeIndex(sorted({dt for df in contract_data.values() for dt in df.index}), name="Date")
    if all_idx.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    aligned = {symbol: df.sort_index().reindex(all_idx) for symbol, df in contract_data.items() if not df.empty}
    if not aligned:
        return pd.DataFrame(columns=OHLCV_COLUMNS)
    vol_roll = pd.DataFrame({symbol: df["Volume"].rolling(5, min_periods=1).sum() for symbol, df in aligned.items()})
    best_contract = vol_roll.idxmax(axis=1)

    out = pd.DataFrame(index=all_idx, columns=OHLCV_COLUMNS, dtype=float)
    for date, contract in best_contract.items():
        row = aligned[contract].loc[date]
        if row.isna().all():
            continue
        for col in OHLCV_COLUMNS:
            out.at[date, col] = row[col]
    return out.dropna(subset=["Close"])


def get_ohlcv(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    provider: str | None = None,
    cache_dir: str | None = None,
    use_cache: bool = True,
    cache_ttl_days: int = 0,
    include_delisted: bool = True,
    asset_type: str = "equity",
    futures_contracts: list[str] | None = None,
    crypto_exchange: str = "binance",
    wrds_username: str | None = None,
    ticker_to_permno: dict[str, int] | None = None,
    as_of_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Fetch OHLCV through the explicitly configured provider."""
    start = pd.Timestamp(start_date)
    end = pd.Timestamp(end_date)
    provider_name = resolve_data_provider(provider)
    cache_root = cache_dir or DEFAULT_CACHE_DIR
    path = _cache_path(cache_root, ticker, provider_name)

    logger.debug(
        "[get_ohlcv] %s: %s -> %s provider=%s",
        ticker,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
        provider_name,
    )

    if use_cache:
        cached = _load_cached(path, cache_ttl_days)
        if cached is not None and not cached.empty:
            cached_start = cached.index.min()
            cached_end = cached.index.max()
            if start >= cached_start and end <= cached_end:
                df = cached.loc[(cached.index >= start) & (cached.index <= end)].copy()
                if include_delisted and asset_type == "equity" and provider_name != "wrds":
                    df = _attach_delisted_date(df, end)
                return df

    config = ProviderConfig(
        provider=provider_name,
        cache_dir=cache_root,
        use_cache=use_cache,
        cache_ttl_days=cache_ttl_days,
        wrds_username=wrds_username,
        ticker_to_permno=ticker_to_permno,
        as_of_date=as_of_date,
    )

    if asset_type == "futures":
        if not futures_contracts:
            raise ValueError("futures_contracts must be provided when asset_type='futures'")
        if provider_name != "yahoo":
            raise ValueError("continuous futures currently support only explicit provider='yahoo'")
        adapter = YahooProvider()
        contract_data = {
            contract: adapter.fetch_ohlcv(contract, start, end)
            for contract in futures_contracts
        }
        df = _build_continuous_futures(contract_data)
    elif asset_type == "crypto":
        raise ValueError("crypto OHLCV requires a dedicated provider adapter; no implicit fallback is allowed")
    else:
        adapter = get_provider(config)
        if adapter is not None:
            df = adapter.fetch_ohlcv(ticker, start, end)
        elif provider_name == "alpaca":
            df = _download_alpaca(ticker, start, end)
        elif provider_name == "finnhub":
            df = _download_finnhub(ticker, start, end)
        elif provider_name == "lean":
            df = _download_lean(ticker, start, end)
        else:
            raise ValueError(f"unsupported provider: {provider_name}")

    if df is None or df.empty:
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    if not _df_overlaps_window(df, start, end):
        return pd.DataFrame(columns=OHLCV_COLUMNS)

    df = df.loc[(df.index >= start) & (df.index <= end)].copy()
    if include_delisted and asset_type == "equity" and provider_name != "wrds":
        df = _attach_delisted_date(df, end)

    if use_cache and not df.empty:
        existing = _load_cached(path, cache_ttl_days=0)
        if existing is not None and not existing.empty:
            combined = pd.concat([existing, df])
            combined = combined[~combined.index.duplicated(keep="last")].sort_index()
        else:
            combined = df.sort_index()
        _save_cache(path, combined)

    return df


def clear_cache(ticker: str | None = None, cache_dir: str | None = None, provider: str = "yahoo") -> None:
    """Clear cached OHLCV data for an explicit provider namespace."""
    clear_provider_cache(ticker=ticker, cache_dir=cache_dir, provider=resolve_data_provider(provider))
