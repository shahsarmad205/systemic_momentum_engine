"""
WRDS CRSP Price/Return Loader
==============================
Loads daily security data from CRSP (crsp.dsf) and produces a canonical
DataFrame schema consumed by the feature pipeline and backtester.

Key correctness choices vs Yahoo Finance:
  1. Returns (ret): CRSP ret already incorporates dividends, splits, and all
     corporate actions. It is the correct total return for all feature and
     backtest calculations. Never recompute returns from price levels.
  2. Price levels (adj_price): Built as (1 + ret).cumprod() from a reference
     date — guaranteed split/dividend-consistent with zero artifacts.
  3. Delisting returns: Shumway (1997) shows ignoring delisting returns
     overstates alpha by ~1.5%/yr. We splice in dlret from crsp.dsedelist,
     or a conservative imputed loss (−30% NYSE/AMEX, −55% OTC) when dlret
     is missing and the delisting reason is performance-related.

Canonical output schema (Phase 3):
    ret            float64   Daily total return (CRSP ret + delisting splice)
    adj_price      float64   Cumulative-return price index (base=1.0 at start)
    raw_price      float64   abs(crsp.prc) — unadjusted closing price
    dollar_volume  float64   abs(prc) × vol (CRSP shares)
    shares_out     float64   shrout × 1000 (actual shares outstanding)
    market_cap     float64   raw_price × shares_out

    Backward-compatibility aliases written for the existing feature pipeline:
    Open           float64   raw_price (CRSP has no intraday; use Close as proxy)
    High           float64   raw_price × (1 + daily_range_estimate) — see note
    Low            float64   raw_price × (1 - daily_range_estimate)
    Close          float64   adj_price  (preferred for all downstream)
    AdjClose       float64   adj_price  (legacy alias; use Close going forward)
    Volume         float64   vol (CRSP raw share volume)

    NOTE on High/Low: CRSP daily file (dsf) does not contain intraday H/L.
    The proxied H/L use a 10-day rolling avg-range from ret volatility.
    These are approximate — features that need true H/L should use a TAQ
    or minute-bar source. Momentum, MA, reversal features are unaffected.

Usage:
    from utils.wrds_loader import WRDSLoader

    loader = WRDSLoader(db, cache_dir="data/cache/wrds")
    price_data = loader.load_universe(
        permnos=[10001, 10002, ...],
        ticker_map={10001: "AAPL", 10002: "MSFT"},
        start_date="2007-01-01",
        end_date="2022-12-31",
    )
    # price_data: dict[str, pd.DataFrame]  ticker → canonical DataFrame
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Imputed delisting returns (Shumway 1997 / Beaver-McNichols-Price 2007)
# Used when dlret is missing and the delist code indicates performance failure.
_DELISTING_IMPUTED_NYSE_AMEX = -0.30  # performance delistings 500–591 on NYSE/AMEX
_DELISTING_IMPUTED_OTC = -0.55         # performance delistings on NASDAQ/OTC
_PERFORMANCE_DELIST_CODES = set(range(500, 592))  # CRSP dlstcd 500–591

# CRSP exchange codes
_OTC_EXCHANGES = {3}  # exchcd=3 is NASDAQ; 1=NYSE, 2=AMEX


class WRDSLoader:
    """
    Load CRSP daily security data with delisting-return splicing.

    Parameters
    ----------
    db : wrds.Connection
    cache_dir : str
        Root directory for WRDS Parquet cache (separate from Yahoo cache).
    cache_ttl_days : int
        File-level cache TTL. 0 = always re-fetch.
    """

    def __init__(
        self,
        db,
        cache_dir: str = "data/cache/wrds",
        cache_ttl_days: int = 30,
    ) -> None:
        self._db = db
        self._cache_dir = Path(cache_dir) / "prices"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl_days = cache_ttl_days

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_universe(
        self,
        permnos: list[int],
        ticker_map: dict[int, str],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> dict[str, pd.DataFrame]:
        """
        Load price data for a list of PERMNOs and return a ticker-keyed dict.

        Parameters
        ----------
        permnos : list[int]
            CRSP security identifiers to load.
        ticker_map : dict[int, str]
            {permno: ticker} mapping (point-in-time from WRDSUniverse).
        start_date, end_date : str or Timestamp
            Date range for the backtest. We buffer by 400 days upstream so
            the feature pipeline has enough warm-up history.

        Returns
        -------
        dict[str, pd.DataFrame]
            {ticker: canonical_ohlcv_df} — same shape expected by the backtester.
        """
        s = pd.Timestamp(start_date)
        e = pd.Timestamp(end_date)

        # Batch-load raw CRSP data for all permnos in one SQL round-trip
        raw_panel = self._load_crsp_panel(permnos, s, e)

        # Splice delisting returns
        delist_map = self._load_delisting_returns(permnos, s, e)
        raw_panel = self._splice_delisting_returns(raw_panel, delist_map)

        # Split by permno, build canonical schema, key by ticker
        price_data: dict[str, pd.DataFrame] = {}
        for permno, group in raw_panel.groupby("permno"):
            ticker = ticker_map.get(int(permno), f"PERMNO_{permno}")
            df = self._build_canonical(group.sort_values("date"))
            if df.empty or len(df) < 20:
                logger.debug("Skipping %s (permno=%s): insufficient rows", ticker, permno)
                continue
            price_data[ticker] = df

        logger.info(
            "WRDSLoader: loaded %d securities for %s → %s",
            len(price_data), s.date(), e.date(),
        )
        return price_data

    def load_single(
        self,
        permno: int,
        ticker: str,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """Load a single security; returns canonical DataFrame or empty."""
        result = self.load_universe([permno], {permno: ticker}, start_date, end_date)
        return result.get(ticker, pd.DataFrame())

    # ------------------------------------------------------------------
    # Core: CRSP DSF fetch
    # ------------------------------------------------------------------

    def _load_crsp_panel(
        self,
        permnos: list[int],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Batch-fetch crsp.dsf for all permnos in one query.
        Caches the result per (permno_hash, start, end) to avoid re-querying.
        """
        cache_key = f"crsp_dsf_{abs(hash(tuple(sorted(permnos))))}_"
        cache_key += f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
        cache_path = self._cache_dir / cache_key

        if self._is_cache_fresh(cache_path):
            logger.debug("Loading CRSP DSF from cache: %s", cache_path.name)
            return pd.read_parquet(cache_path)

        permno_str = ",".join(str(p) for p in permnos)
        start_str = start.strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        sql = f"""
            SELECT permno, date, prc, ret, vol, shrout
            FROM crsp.dsf
            WHERE permno IN ({permno_str})
              AND date BETWEEN '{start_str}' AND '{end_str}'
            ORDER BY permno, date
        """
        logger.info("Fetching CRSP DSF for %d permnos (%s → %s)…", len(permnos), start_str, end_str)
        df = self._db.raw_sql(sql, date_cols=["date"])
        df["permno"] = df["permno"].astype(int)

        df.to_parquet(cache_path, index=False)
        logger.info("Cached CRSP DSF: %d rows for %d permnos", len(df), df["permno"].nunique())
        return df

    # ------------------------------------------------------------------
    # Delisting return splicing
    # ------------------------------------------------------------------

    def _load_delisting_returns(
        self,
        permnos: list[int],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> dict[int, dict]:
        """
        Pull crsp.dsedelist for the given permnos.
        Returns {permno: {"dlstdt": date, "dlret": float, "exchcd": int, "dlstcd": int}}.
        """
        cache_path = self._cache_dir / f"delist_{abs(hash(tuple(sorted(permnos))))}.parquet"
        if self._is_cache_fresh(cache_path):
            raw = pd.read_parquet(cache_path)
        else:
            permno_str = ",".join(str(p) for p in permnos)
            sql = f"""
                SELECT permno, dlstdt, dlret, hexcd, dlstcd
                FROM crsp.dsedelist
                WHERE permno IN ({permno_str})
                ORDER BY permno, dlstdt
            """
            raw = self._db.raw_sql(sql, date_cols=["dlstdt"])
            raw["permno"] = raw["permno"].astype(int)
            raw.to_parquet(cache_path, index=False)

        # Only keep delistings that fall within (or just after) our window
        raw = raw[raw["dlstdt"] <= (end + pd.Timedelta(days=5))]

        result: dict[int, dict] = {}
        for _, row in raw.iterrows():
            p = int(row["permno"])
            result[p] = {
                "dlstdt": row["dlstdt"],
                "dlret": row.get("dlret"),
                "exchcd": int(row.get("hexcd", 1) or 1),  # dsedelist uses hexcd
                "dlstcd": int(row.get("dlstcd", 0) or 0),
            }
        return result

    def _splice_delisting_returns(
        self,
        panel: pd.DataFrame,
        delist_map: dict[int, dict],
    ) -> pd.DataFrame:
        """
        For each delisted security, append one final row with the delisting return.

        Rules (Shumway 1997):
        - Use dlret when available and not NaN.
        - If dlret is missing AND dlstcd ∈ 500–591 (performance failure):
            NYSE/AMEX: impute −30%
            NASDAQ/OTC: impute −55%
        - Otherwise: no correction (voluntary delistings, mergers with
          proper merger returns already in ret).
        """
        extra_rows = []
        for permno, info in delist_map.items():
            dlstdt = pd.Timestamp(info["dlstdt"])
            dlret = info.get("dlret")
            exchcd = info.get("exchcd", 1)
            dlstcd = info.get("dlstcd", 0)

            # Determine the return to splice
            ret_to_splice = None
            if dlret is not None and not pd.isna(dlret):
                ret_to_splice = float(dlret)
            elif dlstcd in _PERFORMANCE_DELIST_CODES:
                if exchcd in _OTC_EXCHANGES:
                    ret_to_splice = _DELISTING_IMPUTED_OTC
                else:
                    ret_to_splice = _DELISTING_IMPUTED_NYSE_AMEX

            if ret_to_splice is None:
                continue

            # Check we don't already have a return for the delist date
            existing = panel[(panel["permno"] == permno) & (panel["date"] == dlstdt)]
            if not existing.empty:
                # Update in-place rather than append
                panel.loc[existing.index, "ret"] = ret_to_splice
                continue

            # Grab the last known row for this security to copy static fields
            last = panel[panel["permno"] == permno].sort_values("date").tail(1)
            if last.empty:
                continue

            new_row = last.copy()
            new_row["date"] = dlstdt
            new_row["ret"] = ret_to_splice
            new_row["prc"] = 0.0  # security effectively worthless
            new_row["vol"] = 0.0
            extra_rows.append(new_row)

        if extra_rows:
            panel = pd.concat([panel] + extra_rows, ignore_index=True)
            panel = panel.sort_values(["permno", "date"]).reset_index(drop=True)

        return panel

    # ------------------------------------------------------------------
    # Build canonical output DataFrame
    # ------------------------------------------------------------------

    def _build_canonical(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform a single-security CRSP slice into the canonical schema.

        CRSP prc quirk: negative prc = bid-ask midpoint (no trade that day).
        We always take abs(prc) as the raw price.
        """
        out = pd.DataFrame(index=df["date"].values)
        out.index = pd.DatetimeIndex(out.index)
        out.index.name = "Date"

        def _to_float(arr) -> np.ndarray:
            """Convert any pandas array type to plain float64 numpy array."""
            return np.array(pd.to_numeric(arr, errors="coerce"), dtype=np.float64)

        # ── Core return: use CRSP ret directly ───────────────────────────
        ret = _to_float(df["ret"].values)
        out["ret"] = ret

        # ── Raw price: abs(prc) ──────────────────────────────────────────
        raw_prc = np.abs(_to_float(df["prc"].values))
        out["raw_price"] = raw_prc

        # ── Adjusted price index: cumprod of (1 + ret) ──────────────────
        # Fill NaN returns with 0 (no price change) for cumprod continuity
        ret_filled = np.where(np.isfinite(ret), ret, 0.0)
        adj_price = np.cumprod(1.0 + ret_filled)
        out["adj_price"] = adj_price

        # ── Volume and market cap ────────────────────────────────────────
        vol = np.clip(_to_float(df["vol"].values), 0, None)
        shrout = np.clip(_to_float(df["shrout"].values), 0, None)

        out["dollar_volume"] = raw_prc * vol                   # shares × price
        out["shares_out"] = shrout * 1000.0                    # CRSP shrout is in thousands
        out["market_cap"] = raw_prc * out["shares_out"]
        out["Volume"] = vol                                     # raw share volume

        # ── Backward-compatibility columns for existing feature pipeline ──
        # Close = adj_price (preferred going forward; use ret for returns)
        out["Close"] = adj_price
        out["AdjClose"] = adj_price                            # legacy alias
        out["Open"] = adj_price                                # CRSP has no open; proxy
        out["Volume"] = vol

        # Approximate High/Low from rolling realized vol (for ATR and similar).
        # This is an estimate — not from true intraday data.
        daily_vol = (
            pd.Series(ret_filled, index=out.index)
            .rolling(10, min_periods=3)
            .std()
            .fillna(0.01)
        )
        out["High"] = adj_price * (1.0 + daily_vol)
        out["Low"] = adj_price * (1.0 - daily_vol)

        return out.dropna(subset=["ret"])

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    def _is_cache_fresh(self, path: Path) -> bool:
        if not path.exists():
            return False
        if self._cache_ttl_days <= 0:
            return False
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return age < timedelta(days=self._cache_ttl_days)
