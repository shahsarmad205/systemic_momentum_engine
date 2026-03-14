"""
MarketDataLoader — abstraction over cached + API OHLCV.
Uses existing get_ohlcv implementation; no change to download/caching logic.
"""

from __future__ import annotations

import pandas as pd

from utils.market_data import get_ohlcv, DEFAULT_CACHE_DIR, OHLCV_COLUMNS


class MarketDataLoader:
    """
    Facade for price/volume history with local cache and flexible provider.
    """

    def __init__(
        self,
        provider: str = "yahoo",
        cache_dir: str | None = None,
        use_cache: bool = True,
        cache_ttl_days: int = 0,
    ):
        self.provider = provider
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR
        self.use_cache = use_cache
        self.cache_ttl_days = cache_ttl_days

    def load_price_history(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Load OHLCV; price history is the full OHLCV frame (Open/High/Low/Close)."""
        df = self._load_ohlcv(ticker, start_date, end_date)
        n = 0 if df is None or df.empty else len(df)
        print(f"  {ticker}: {n} rows loaded")
        return df

    def load_volume_history(
        self,
        ticker: str,
        start_date: str,
        end_date: str,
    ) -> pd.DataFrame:
        """Load volume series as a single-column DataFrame aligned with price index."""
        df = self._load_ohlcv(ticker, start_date, end_date)
        if df.empty or "Volume" not in df.columns:
            return pd.DataFrame(columns=["Volume"])
        return df[["Volume"]].copy()

    def _load_ohlcv(self, ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = get_ohlcv(
            ticker,
            start_date,
            end_date,
            provider=self.provider,
            cache_dir=self.cache_dir,
            use_cache=self.use_cache,
            cache_ttl_days=self.cache_ttl_days,
        )
        if df is not None and not df.empty:
            from data_processing.pipeline import process_market_data
            df = process_market_data(df)
        return df
