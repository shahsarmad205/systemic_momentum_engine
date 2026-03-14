"""
Market Regime Agent
=====================
Downloads SPY + VIX data and classifies each trading day into one of:
    Bull, Bear, Sideways, Crisis

Used by the backtester to adjust signal weights and position sizing
according to the prevailing market environment.
"""

import numpy as np
import pandas as pd
import yfinance as yf


class MarketRegimeAgent:
    LOOKBACK_BUFFER = 600          # calendar days before start for MA warm-up
    VIX_CRISIS_THRESHOLD = 30.0
    VIX_HIGH_THRESHOLD = 20.0

    def detect_regimes(
        self, start_date: str, end_date: str
    ) -> dict[pd.Timestamp, str]:
        """
        Return {date: regime_label} for every trading day in [start, end].

        Regime rules:
            Crisis   — VIX ≥ 30
            Bull     — SPY > SMA-200  AND  SMA-50 > SMA-200
            Bear     — SPY < SMA-200  AND  SMA-50 < SMA-200
            Sideways — everything else
        """
        dl_start = pd.Timestamp(start_date) - pd.Timedelta(days=self.LOOKBACK_BUFFER)
        dl_end = pd.Timestamp(end_date) + pd.Timedelta(days=30)

        spy = self._download("SPY", dl_start, dl_end)
        vix = self._download_vix(dl_start, dl_end, spy)

        if spy.empty:
            return {}

        sma200 = spy["Close"].rolling(200).mean()
        sma50 = spy["Close"].rolling(50).mean()

        regime_map: dict[pd.Timestamp, str] = {}
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=30)

        for date in spy.index:
            if date < start_ts or date > end_ts:
                continue
            if pd.isna(sma200.get(date)) or pd.isna(sma50.get(date)):
                regime_map[date] = "Sideways"
                continue

            close = float(spy.loc[date, "Close"])
            vix_val = float(vix.get(date, 15.0))

            if vix_val >= self.VIX_CRISIS_THRESHOLD:
                regime_map[date] = "Crisis"
            elif close > sma200[date] and sma50[date] > sma200[date]:
                regime_map[date] = "Bull"
            elif close < sma200[date] and sma50[date] < sma200[date]:
                regime_map[date] = "Bear"
            else:
                regime_map[date] = "Sideways"

        return regime_map

    # -- helpers ---------------------------------------------------

    @staticmethod
    def _download(ticker: str, start, end) -> pd.DataFrame:
        raw = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        keep = ["Open", "High", "Low", "Close", "Volume"]
        return raw[keep].dropna() if not raw.empty else pd.DataFrame()

    @staticmethod
    def _download_vix(start, end, spy_fallback: pd.DataFrame) -> dict:
        """Try ^VIX; fall back to annualised 20-day rolling vol of SPY."""
        try:
            raw = yf.download("^VIX", start=start, end=end, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            if not raw.empty and "Close" in raw.columns:
                return raw["Close"].to_dict()
        except Exception:
            pass

        if spy_fallback.empty:
            return {}

        returns = spy_fallback["Close"].pct_change()
        vol_proxy = returns.rolling(20).std() * np.sqrt(252) * 100
        return vol_proxy.to_dict()
