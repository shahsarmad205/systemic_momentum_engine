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

Master-panel caching (institutional pattern):
  Call prefetch_universe_panel() ONCE before any walk-forward run.
  It downloads ALL PERMNOs × full date range in one SQL batch and writes:
      data/cache/wrds/prices/crsp_universe_panel_{start}_{end}.parquet
      data/cache/wrds/prices/crsp_universe_panel_{start}_{end}.meta.json
      data/cache/wrds/prices/crsp_delist_master.parquet
      data/cache/wrds/prices/crsp_delist_master.meta.json
  Subsequent calls to _load_crsp_panel() and _load_delisting_returns()
  slice from these local files — zero WRDS round-trips for all 8 walk-forward
  windows. Falls back to per-window WRDS queries when no master panel covers
  the requested range.

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

    # Run once before any walk-forward:
    loader.prefetch_universe_panel(
        permnos=[10001, 10002, ...],
        start_date="2006-01-01",
        end_date="2023-12-31",
    )

    # Every subsequent call (all 8 windows) reads from disk, zero WRDS hits:
    price_data = loader.load_universe(
        permnos=[10001, 10002, ...],
        ticker_map={10001: "AAPL", 10002: "MSFT"},
        start_date="2007-01-01",
        end_date="2022-12-31",
    )
    # price_data: dict[str, pd.DataFrame]  ticker → canonical DataFrame
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import sqlalchemy

logger = logging.getLogger(__name__)

# Imputed delisting returns (Shumway 1997 / Beaver-McNichols-Price 2007)
# Used when dlret is missing and the delist code indicates performance failure.
_DELISTING_IMPUTED_NYSE_AMEX = -0.30  # performance delistings 500–591 on NYSE/AMEX
_DELISTING_IMPUTED_OTC = -0.55         # performance delistings on NASDAQ/OTC
_PERFORMANCE_DELIST_CODES = set(range(500, 592))  # CRSP dlstcd 500–591

# CRSP exchange codes
_OTC_EXCHANGES = {3}  # exchcd=3 is NASDAQ; 1=NYSE, 2=AMEX

# Master panel filenames
_MASTER_DSF_PREFIX = "crsp_universe_panel_"
_MASTER_DELIST_STEM = "crsp_delist_master"


class WRDSLoader:
    """
    Load CRSP daily security data with delisting-return splicing.

    Parameters
    ----------
    db : wrds.Connection
    cache_dir : str
        Root directory for WRDS Parquet cache (separate from Yahoo cache).
    cache_ttl_days : int
        File-level cache TTL for per-window fallback caches. 0 = always re-fetch.
        Master panel files are never expired by TTL — CRSP history is immutable.
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
    # Internal helpers
    # ------------------------------------------------------------------

    def _raw_sql(self, sql: str, date_cols: list[str] | None = None) -> pd.DataFrame:
        """Execute SQL via SQLAlchemy directly, bypassing pd.read_sql_query.

        pd.read_sql_query fails to detect the wrds SA-2 Connection as a SQLAlchemy
        connectable (isinstance check mismatch across SA versions), falling into the
        legacy DBAPI2 path and crashing. Fetching via conn.execute() + fetchall()
        bypasses that entirely and works with any SA version.
        """
        with self._db.engine.connect() as conn:
            result = conn.execute(sqlalchemy.text(sql))
            df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
        if date_cols:
            for col in date_cols:
                if col in df.columns:
                    df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def prefetch_universe_panel(
        self,
        permnos: list[int],
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> Path:
        """
        Download ALL PERMNOs × full date range in ONE SQL batch and cache locally.

        Call this once before running any walk-forward backtest. All subsequent
        calls to load_universe() for any sub-window within [start_date, end_date]
        will slice from this local parquet with zero WRDS round-trips.

        Parameters
        ----------
        permnos : list[int]
            Full universe of CRSP PERMNOs to prefetch (e.g. all 500 tickers).
        start_date : str or Timestamp
            Earliest date needed (add feature warm-up buffer, e.g. 2006-01-01).
        end_date : str or Timestamp
            Latest date needed across all walk-forward windows.

        Returns
        -------
        Path
            Path to the written master DSF parquet file.
        """
        s = pd.Timestamp(start_date)
        e = pd.Timestamp(end_date)
        stem = f"{_MASTER_DSF_PREFIX}{s.strftime('%Y%m%d')}_{e.strftime('%Y%m%d')}"
        panel_path = self._cache_dir / f"{stem}.parquet"
        meta_path = self._cache_dir / f"{stem}.meta.json"

        if panel_path.exists() and meta_path.exists():
            existing = self._read_meta(meta_path)
            existing_permnos = set(existing.get("permnos", []))
            if set(int(p) for p in permnos) <= existing_permnos:
                logger.info("Master DSF panel already covers all permnos: %s", panel_path.name)
                # Still prefetch delist master if not present
                self._prefetch_delist_master(permnos)
                return panel_path

        permno_str = ",".join(str(p) for p in sorted(permnos))
        start_str = s.strftime("%Y-%m-%d")
        end_str = e.strftime("%Y-%m-%d")

        sql = f"""
            SELECT permno, date, prc, ret, vol, shrout
            FROM crsp.dsf
            WHERE permno IN ({permno_str})
              AND date BETWEEN '{start_str}' AND '{end_str}'
            ORDER BY permno, date
        """
        logger.info(
            "Prefetching master CRSP DSF panel: %d permnos, %s → %s…",
            len(permnos), start_str, end_str,
        )
        df = self._raw_sql(sql, date_cols=["date"])
        df["permno"] = df["permno"].astype(int)
        df.to_parquet(panel_path, index=False)

        meta: dict = {
            "start_date": start_str,
            "end_date": end_str,
            "permnos": sorted(int(p) for p in permnos),
            "row_count": len(df),
            "permno_count": int(df["permno"].nunique()),
            "created_at": datetime.now().isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info(
            "Master DSF panel written: %d rows, %d permnos → %s",
            len(df), df["permno"].nunique(), panel_path.name,
        )

        self._prefetch_delist_master(permnos)
        return panel_path

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

        # Batch-load raw CRSP data — master panel first, then WRDS fallback
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
    # Core: CRSP DSF fetch — master panel → per-window cache → WRDS
    # ------------------------------------------------------------------

    def _load_crsp_panel(
        self,
        permnos: list[int],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Return raw CRSP DSF rows for the requested permnos and date window.

        Resolution order:
          1. Master panel parquet (sliced locally — zero WRDS calls after prefetch)
          2. Per-window parquet cache (legacy fallback, keyed by permno_hash + dates)
          3. Direct WRDS SQL query (writes per-window cache for next run)
        """
        # ── 1. Master panel ──────────────────────────────────────────────
        master = self._find_master_dsf_panel(permnos, start, end)
        if master is not None:
            logger.debug(
                "Slicing master DSF panel for %d permnos (%s → %s)",
                len(permnos), start.date(), end.date(),
            )
            wanted = set(int(p) for p in permnos)
            try:
                # Predicate pushdown via pyarrow filters — leverages row-group
                # stats on permno since the parquet was written ORDER BY permno, date.
                df = pd.read_parquet(
                    master,
                    filters=[
                        ("permno", "in", list(wanted)),
                        ("date", ">=", start),
                        ("date", "<=", end),
                    ],
                )
            except Exception:
                # Fallback: full read + filter (pyarrow may not support all filter types)
                df = pd.read_parquet(master)
                df["date"] = pd.to_datetime(df["date"])
                df = df[df["permno"].isin(wanted) & (df["date"] >= start) & (df["date"] <= end)]
            df["permno"] = df["permno"].astype(int)
            return df.reset_index(drop=True)

        # ── 2. Per-window cache (legacy) ─────────────────────────────────
        cache_key = (
            f"crsp_dsf_{abs(hash(tuple(sorted(permnos))))}_"
            f"{start.strftime('%Y%m%d')}_{end.strftime('%Y%m%d')}.parquet"
        )
        cache_path = self._cache_dir / cache_key
        if self._is_cache_fresh(cache_path):
            logger.debug("Loading CRSP DSF from per-window cache: %s", cache_path.name)
            return pd.read_parquet(cache_path)

        # ── 3. WRDS SQL fallback ─────────────────────────────────────────
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
        logger.info(
            "Fetching CRSP DSF from WRDS for %d permnos (%s → %s)…",
            len(permnos), start_str, end_str,
        )
        df = self._raw_sql(sql, date_cols=["date"])
        df["permno"] = df["permno"].astype(int)
        df.to_parquet(cache_path, index=False)
        logger.info("Cached CRSP DSF: %d rows for %d permnos", len(df), df["permno"].nunique())
        return df

    # ------------------------------------------------------------------
    # Master-panel helpers
    # ------------------------------------------------------------------

    def _find_master_dsf_panel(
        self,
        permnos: list[int],
        start: pd.Timestamp,
        end: pd.Timestamp,
    ) -> Path | None:
        """
        Find an existing master DSF parquet whose permno set ⊇ requested and
        whose date range fully covers [start, end]. Returns the parquet path or None.
        """
        wanted = set(int(p) for p in permnos)
        for meta_path in sorted(self._cache_dir.glob(f"{_MASTER_DSF_PREFIX}*.meta.json")):
            meta = self._read_meta(meta_path)
            if not meta:
                continue
            panel_path = meta_path.with_suffix(".parquet")
            if not panel_path.exists():
                continue
            try:
                p_start = pd.Timestamp(meta["start_date"])
                p_end = pd.Timestamp(meta["end_date"])
                p_permnos = set(meta.get("permnos", []))
            except (KeyError, ValueError):
                continue
            if p_start <= start and p_end >= end and wanted <= p_permnos:
                return panel_path
        return None

    def _prefetch_delist_master(self, permnos: list[int]) -> Path:
        """
        Download crsp.dsedelist for ALL permnos and write a single master file.
        Skipped when the master already covers the full permno set.
        """
        meta_path = self._cache_dir / f"{_MASTER_DELIST_STEM}.meta.json"
        delist_path = self._cache_dir / f"{_MASTER_DELIST_STEM}.parquet"
        wanted = set(int(p) for p in permnos)

        if delist_path.exists() and meta_path.exists():
            meta = self._read_meta(meta_path)
            if wanted <= set(meta.get("permnos", [])):
                logger.debug("Master delist panel already covers all permnos.")
                return delist_path

        permno_str = ",".join(str(p) for p in sorted(permnos))
        sql = f"""
            SELECT permno, dlstdt, dlret, hexcd, dlstcd
            FROM crsp.dsedelist
            WHERE permno IN ({permno_str})
            ORDER BY permno, dlstdt
        """
        logger.info("Prefetching master delist panel for %d permnos…", len(permnos))
        raw = self._raw_sql(sql, date_cols=["dlstdt"])
        raw["permno"] = raw["permno"].astype(int)
        raw.to_parquet(delist_path, index=False)

        meta: dict = {
            "permnos": sorted(int(p) for p in permnos),
            "row_count": len(raw),
            "created_at": datetime.now().isoformat(),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        logger.info("Master delist panel written: %d rows → %s", len(raw), delist_path.name)
        return delist_path

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

        Resolution order:
          1. Master delist parquet (preferred — zero WRDS calls after prefetch)
          2. Per-permno-set parquet cache (legacy fallback)
          3. Direct WRDS SQL query
        """
        wanted = set(int(p) for p in permnos)

        # ── 1. Master delist panel ───────────────────────────────────────
        meta_path = self._cache_dir / f"{_MASTER_DELIST_STEM}.meta.json"
        master_path = self._cache_dir / f"{_MASTER_DELIST_STEM}.parquet"
        if master_path.exists() and meta_path.exists():
            meta = self._read_meta(meta_path)
            if wanted <= set(meta.get("permnos", [])):
                try:
                    raw = pd.read_parquet(
                        master_path,
                        filters=[("permno", "in", list(wanted))],
                    )
                except Exception:
                    raw = pd.read_parquet(master_path)
                    raw["permno"] = raw["permno"].astype(int)
                    raw = raw[raw["permno"].isin(wanted)]
                raw["permno"] = raw["permno"].astype(int)
                logger.debug(
                    "Sliced master delist panel: %d records for %d permnos",
                    len(raw), len(wanted),
                )
                return self._build_delist_map(raw, end)

        # ── 2. Per-permno-set cache (legacy) ─────────────────────────────
        cache_path = self._cache_dir / f"delist_{abs(hash(tuple(sorted(permnos))))}.parquet"
        if self._is_cache_fresh(cache_path):
            raw = pd.read_parquet(cache_path)
            return self._build_delist_map(raw, end)

        # ── 3. WRDS SQL fallback ─────────────────────────────────────────
        permno_str = ",".join(str(p) for p in permnos)
        sql = f"""
            SELECT permno, dlstdt, dlret, hexcd, dlstcd
            FROM crsp.dsedelist
            WHERE permno IN ({permno_str})
            ORDER BY permno, dlstdt
        """
        raw = self._raw_sql(sql, date_cols=["dlstdt"])
        raw["permno"] = raw["permno"].astype(int)
        raw.to_parquet(cache_path, index=False)
        return self._build_delist_map(raw, end)

    @staticmethod
    def _build_delist_map(raw: pd.DataFrame, end: pd.Timestamp) -> dict[int, dict]:
        """Convert a delist DataFrame to the {permno: info_dict} structure."""
        raw = raw[raw["dlstdt"] <= (end + pd.Timedelta(days=5))]
        result: dict[int, dict] = {}
        for _, row in raw.iterrows():
            p = int(row["permno"])
            result[p] = {
                "dlstdt": row["dlstdt"],
                "dlret": row.get("dlret"),
                "exchcd": int(row.get("hexcd", 1) or 1),
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
        if "delisting_return_applied" not in panel.columns:
            panel["delisting_return_applied"] = False
        if "delisting_return" not in panel.columns:
            panel["delisting_return"] = np.nan
        if "dlstcd" not in panel.columns:
            panel["dlstcd"] = np.nan
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
                panel.loc[existing.index, "ret"] = ret_to_splice
                panel.loc[existing.index, "delisting_return_applied"] = True
                panel.loc[existing.index, "delisting_return"] = ret_to_splice
                panel.loc[existing.index, "dlstcd"] = dlstcd
                continue

            # Grab the last known row for this security to copy static fields
            last = panel[panel["permno"] == permno].sort_values("date").tail(1)
            if last.empty:
                continue

            new_row = last.copy()
            new_row["date"] = dlstdt
            new_row["ret"] = ret_to_splice
            new_row["prc"] = 0.0
            new_row["vol"] = 0.0
            new_row["delisting_return_applied"] = True
            new_row["delisting_return"] = ret_to_splice
            new_row["dlstcd"] = dlstcd
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
        if "delisting_return_applied" in df.columns:
            out["delisting_return_applied"] = df["delisting_return_applied"].astype(bool).values
        else:
            out["delisting_return_applied"] = False
        if "delisting_return" in df.columns:
            out["delisting_return"] = _to_float(df["delisting_return"].values)
        else:
            out["delisting_return"] = np.nan
        if "dlstcd" in df.columns:
            out["dlstcd"] = _to_float(df["dlstcd"].values)
        else:
            out["dlstcd"] = np.nan

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

        out["dollar_volume"] = raw_prc * vol
        out["shares_out"] = shrout * 1000.0
        out["market_cap"] = raw_prc * out["shares_out"]
        out["Volume"] = vol

        # ── Backward-compatibility columns for existing feature pipeline ──
        out["Close"] = adj_price
        out["AdjClose"] = adj_price
        out["Open"] = adj_price
        out["Volume"] = vol

        # Approximate High/Low from rolling realized vol (for ATR and similar).
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

    @staticmethod
    def _read_meta(meta_path: Path) -> dict:
        try:
            return json.loads(meta_path.read_text())
        except Exception:
            return {}
