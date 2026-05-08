"""Explicit Yahoo Finance market-data adapter."""

from __future__ import annotations

import pandas as pd

from providers.cache_manager import OHLCV_COLUMNS, df_overlaps_window


class YahooProvider:
    """Fetch adjusted OHLCV from Yahoo via yfinance when explicitly configured."""

    name = "yahoo"

    def validate_available(self) -> None:
        try:
            import yfinance  # noqa: F401
        except ImportError as exc:
            raise RuntimeError("Yahoo provider requires yfinance to be installed") from exc

    def fetch_ohlcv(
        self,
        ticker: str,
        start: pd.Timestamp,
        end: pd.Timestamp,
        **_: object,
    ) -> pd.DataFrame:
        self.validate_available()
        import yfinance as yf

        def _fetch(period: str | None = None, start_: pd.Timestamp | None = None, end_: pd.Timestamp | None = None) -> pd.DataFrame:
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
            if not all(col in raw.columns for col in OHLCV_COLUMNS):
                return pd.DataFrame(columns=OHLCV_COLUMNS)

            df = raw[OHLCV_COLUMNS].dropna().copy()
            if "Adj Close" in raw.columns:
                adj = raw["Adj Close"].reindex(df.index)
                close_raw = df["Close"].copy()
                df["AdjClose"] = adj
                df["raw_price"] = close_raw
                df["Close"] = adj
                # Back-adjust Open, High, Low using the same adjustment ratio.
                # Without this, features like mr_overnight_raw = (open - prev_close) / prev_close
                # produce artifacts around split/dividend dates because Open/High/Low remain
                # unadjusted while Close is adjusted. This corrupts ~30% of technical features.
                adj_ratio = adj / close_raw.replace(0, float("nan"))
                for col in ("Open", "High", "Low"):
                    if col in df.columns:
                        df[col] = (df[col] * adj_ratio).where(adj_ratio.notna(), df[col])
                df["Volume"] = (df["Volume"] / adj_ratio).where(adj_ratio.notna() & (adj_ratio > 0), df["Volume"])
            if "Dividends" in raw.columns:
                df["Dividends"] = raw["Dividends"].reindex(df.index)
            if "Stock Splits" in raw.columns:
                df["StockSplits"] = raw["Stock Splits"].reindex(df.index)
            return df

        df = _fetch(start_=start, end_=end)
        if not df_overlaps_window(df, start, end):
            df = pd.DataFrame(columns=OHLCV_COLUMNS)

        # Yahoo sometimes returns sparse/empty rows for delisted names. Retry
        # recent-period endpoints only if they overlap the requested window.
        if len(df) < 30:
            df_6mo = _fetch(period="6mo")
            if df_overlaps_window(df_6mo, start, end) and len(df_6mo) > len(df):
                df = df_6mo
        if len(df) < 30:
            df_1y = _fetch(period="1y")
            if df_overlaps_window(df_1y, start, end) and len(df_1y) > len(df):
                df = df_1y

        if not df.empty:
            df = df.sort_index()
            df = df.loc[(df.index >= start) & (df.index <= end)]
        return df
