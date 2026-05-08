"""
Point-in-Time Universe Filter
================================
Addresses survivorship bias: our default universe is today's S&P 500 constituents.
Stocks that were delisted, merged, or removed between 2008-2022 are missing.
Including only today's survivors systematically overstates historical returns.

Two modes:
  1. PIT CSV mode (preferred): load a historical constituent file
     config/sp500_historical.csv with columns: ticker, date_added, date_removed
     This is the gold-standard fix when CRSP/Compustat constituent data is available.

  2. Cache proxy mode (fallback): use the first available date in the price cache
     as a proxy for IPO / S&P 500 entry. Any ticker whose cache starts after
     (backtest_date - min_history_days) is excluded from that date's universe.
     Limitation: does not capture stocks removed from the index (delistings) —
     only prevents using stocks before their data starts.

RESEARCH NOTE:
  True survivorship bias in S&P 500 backtests averages ~0.5-1.5% annualized
  (Elton, Gruber & Blake 1996; Brown, Goetzmann & Ross 1992).  The proxy fix
  here eliminates the pre-IPO bias but not the delisting bias.  For a full fix,
  load CRSP historical constituent data via the PIT CSV mode.
"""

from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# ── Default PIT CSV path (optional — place CRSP/Compustat data here) ─────────
_DEFAULT_PIT_CSV = "config/sp500_historical.csv"


class PointInTimeUniverse:
    """
    Filter tickers to those that were in the S&P 500 at a given historical date.

    Loading priority (first match wins):
      1. WRDS membership panel: data/cache/wrds/universe/sp500_membership_panel.parquet
         Columns: permno, ticker, company_name, start, ending
         This is the authoritative CRSP S&P 500 membership history (~2064 records).
      2. Custom PIT CSV: pit_csv path or config/sp500_historical.csv
         Columns: ticker, date_added, date_removed
      3. Cache proxy mode: first available date in per-ticker parquet cache ≥ min_history_days

    Parameters
    ----------
    cache_dir : str
        Root directory for the price cache (data/cache/).
    pit_csv : str | None
        Optional path to a custom CSV with columns: ticker, date_added, date_removed.
    min_history_days : int
        Proxy mode: min trading days before the date ticker must have been available.
    """

    # Paths searched for WRDS membership panel (relative to project root or absolute)
    _MEMBERSHIP_PANEL_PATHS = [
        "data/cache/wrds/universe/sp500_membership_panel.parquet",
        "data/cache/universe/sp500_membership_panel.parquet",
    ]

    def __init__(
        self,
        cache_dir: str = "data/cache",
        pit_csv: str | None = None,
        min_history_days: int = 252,
    ) -> None:
        self.cache_dir = cache_dir
        self.min_history_days = int(min_history_days)
        self._pit_table: pd.DataFrame | None = None
        self._first_date_cache: dict[str, pd.Timestamp] = {}

        # ── Priority 1: WRDS/CRSP sp500_membership_panel.parquet ─────────
        for panel_path in self._MEMBERSHIP_PANEL_PATHS:
            if os.path.exists(panel_path):
                try:
                    df = pd.read_parquet(panel_path)
                    df.columns = [c.lower().strip() for c in df.columns]
                    # Normalize: rename start/ending to date_added/date_removed
                    rename_map: dict[str, str] = {}
                    if "start" in df.columns and "date_added" not in df.columns:
                        rename_map["start"] = "date_added"
                    if "ending" in df.columns and "date_removed" not in df.columns:
                        rename_map["ending"] = "date_removed"
                    if rename_map:
                        df = df.rename(columns=rename_map)
                    if "ticker" in df.columns:
                        df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
                        # Drop rows with no ticker (pre-2000 historical entries)
                        df = df[df["ticker"].notna() & (df["ticker"] != "NONE") & (df["ticker"] != "NAN")]
                        for col in ("date_added", "date_removed"):
                            if col in df.columns:
                                df[col] = pd.to_datetime(df[col], errors="coerce")
                        self._pit_table = df.reset_index(drop=True)
                        logger.info(
                            "PointInTimeUniverse: loaded WRDS S&P 500 panel "
                            "(%d membership records, %d unique tickers) from %s",
                            len(df), df["ticker"].nunique(), panel_path,
                        )
                        break
                except Exception as exc:
                    logger.warning("PointInTimeUniverse: could not load panel %s: %s", panel_path, exc)

        # ── Priority 2: custom PIT CSV ────────────────────────────────────
        if self._pit_table is None:
            csv_path = pit_csv or _DEFAULT_PIT_CSV
            if os.path.exists(csv_path):
                try:
                    df = pd.read_csv(csv_path)
                    df.columns = [c.lower().strip() for c in df.columns]
                    if "ticker" in df.columns:
                        df["ticker"] = df["ticker"].str.strip().str.upper()
                        for col in ("date_added", "date_removed"):
                            if col in df.columns:
                                df[col] = pd.to_datetime(df[col], errors="coerce")
                        self._pit_table = df
                        logger.info(
                            "PointInTimeUniverse: loaded %d rows from %s", len(df), csv_path
                        )
                except Exception as exc:
                    logger.warning("PointInTimeUniverse: could not load %s: %s", csv_path, exc)

        if self._pit_table is None:
            logger.info(
                "PointInTimeUniverse: no PIT data found; using cache proxy mode "
                "(first cache date proxy for index entry, min_history_days=%d).",
                min_history_days,
            )

    def _first_available_date(self, ticker: str) -> pd.Timestamp | None:
        """Return the first date in the price cache for this ticker."""
        if ticker in self._first_date_cache:
            return self._first_date_cache[ticker]

        parquet_path = os.path.join(self.cache_dir, f"{ticker}.parquet")
        if not os.path.exists(parquet_path):
            self._first_date_cache[ticker] = None
            return None

        try:
            df = pd.read_parquet(parquet_path, columns=["Close"])
            if df.empty:
                self._first_date_cache[ticker] = None
                return None
            idx = pd.to_datetime(df.index)
            first = idx.min()
            self._first_date_cache[ticker] = first
            return first
        except Exception:
            self._first_date_cache[ticker] = None
            return None

    def is_available(self, ticker: str, date: pd.Timestamp) -> bool:
        """
        Return True if `ticker` was available in the universe on `date`.

        PIT CSV mode: ticker must appear with date_added ≤ date AND
        (date_removed is NaT OR date_removed > date).

        Cache proxy mode: first cache date must be ≤ (date - min_history_days).
        """
        ticker = ticker.strip().upper()
        date = pd.Timestamp(date)

        if self._pit_table is not None:
            rows = self._pit_table[self._pit_table["ticker"] == ticker]
            if rows.empty:
                # Ticker not in PIT table → not in S&P 500 at this date
                return False
            for _, row in rows.iterrows():
                added = row.get("date_added")
                removed = row.get("date_removed")
                if pd.isna(added):
                    continue
                if date < added:
                    continue
                if not pd.isna(removed) and date >= removed:
                    continue
                return True
            return False

        # Fallback: cache proxy
        first = self._first_available_date(ticker)
        if first is None:
            return False
        required_start = date - pd.Timedelta(days=int(self.min_history_days * 365 / 252))
        return first <= required_start

    def filter(
        self,
        tickers: list[str],
        date: pd.Timestamp,
    ) -> list[str]:
        """Return only those tickers available on `date`."""
        return [t for t in tickers if self.is_available(t, date)]

    def coverage_report(self, tickers: list[str], dates: list[pd.Timestamp]) -> pd.DataFrame:
        """
        Build a DataFrame showing what fraction of the candidate universe
        passes the PIT filter at each date.  Useful for spotting look-ahead
        regime changes in universe composition.
        """
        rows = []
        for d in dates:
            avail = self.filter(tickers, d)
            rows.append({
                "date": d,
                "n_candidates": len(tickers),
                "n_available": len(avail),
                "coverage_pct": len(avail) / len(tickers) if tickers else 0.0,
            })
        return pd.DataFrame(rows)


def load_pit_universe(
    cache_dir: str = "data/cache",
    pit_csv: str | None = None,
    min_history_days: int = 252,
) -> PointInTimeUniverse:
    """Convenience factory for PointInTimeUniverse."""
    return PointInTimeUniverse(
        cache_dir=cache_dir,
        pit_csv=pit_csv,
        min_history_days=min_history_days,
    )
