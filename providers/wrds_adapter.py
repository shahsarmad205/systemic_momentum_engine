"""Explicit WRDS/CRSP market-data adapter."""

from __future__ import annotations

import os

import pandas as pd

from providers.cache_manager import OHLCV_COLUMNS
from utils.wrds_universe import _is_missing_wrds_username


class WRDSProvider:
    """Fetch point-in-time CRSP data through WRDS with delisting returns applied."""

    name = "wrds"

    def __init__(
        self,
        *,
        username: str | None = None,
        cache_dir: str = "data/cache/wrds",
        cache_ttl_days: int = 30,
        ticker_to_permno: dict[str, int] | None = None,
        as_of_date: str | pd.Timestamp | None = None,
    ) -> None:
        self.username = username
        self.cache_dir = cache_dir
        self.cache_ttl_days = int(cache_ttl_days)
        self.ticker_to_permno = ticker_to_permno
        self.as_of_date = as_of_date

    def validate_available(self) -> None:
        user = self.username or os.environ.get("WRDS_USERNAME")
        if _is_missing_wrds_username(user):
            raise RuntimeError(
                "WRDS provider selected but WRDS_USERNAME is missing, placeholder, or no username was passed"
            )
        try:
            import wrds  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("WRDS provider requires the wrds package to be installed") from exc

    def fetch_ohlcv(
        self,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        **_: object,
    ) -> pd.DataFrame:
        self.validate_available()
        from utils.wrds_data import load_wrds_price_panel

        symbol = str(ticker).upper()
        panel = load_wrds_price_panel(
            [symbol],
            start_date=start,
            end_date=end,
            username=self.username,
            cache_dir=self.cache_dir,
            cache_ttl_days=self.cache_ttl_days,
            ticker_to_permno=self.ticker_to_permno,
            as_of_date=self.as_of_date or end,
        )
        df = panel.get(symbol, pd.DataFrame(columns=OHLCV_COLUMNS))
        if df is None:
            return pd.DataFrame(columns=OHLCV_COLUMNS)
        return df.sort_index()
