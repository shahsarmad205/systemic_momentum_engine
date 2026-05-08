"""
WRDS Universe Layer — Point-in-Time S&P 500 Membership
=======================================================
Replaces utils/universe.py's static Wikipedia approach with CRSP's
dsp500list table, which records the exact dates each security entered
and exited the S&P 500 index.

Why this matters:
  Wikipedia gives TODAY's constituents. Using them for a 2008 backtest
  includes Apple but excludes Lehman Brothers, Bear Stearns, Washington
  Mutual, Wachovia — stocks that were actually IN the index and went to
  near-zero. This survivorship bias inflates backtest Sharpe by ~0.1–0.2.

Usage:
    import os
    import wrds
    from utils.wrds_universe import WRDSUniverse

    db = wrds.Connection(wrds_username=os.environ["WRDS_USERNAME"])
    universe = WRDSUniverse(db, cache_dir="data/cache/wrds")

    # Point-in-time membership on a specific date
    members = universe.get_sp500_at_date("2008-09-15")   # includes Lehman
    print(members)   # DataFrame: permno, ticker, company_name

    # Full historical membership panel (start_date → end_date)
    panel = universe.get_sp500_panel("2008-01-01", "2022-12-31")
"""
from __future__ import annotations

import logging
import os
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
import numpy as np
from sqlalchemy import text

logger = logging.getLogger(__name__)

def wrds_query(db, sql: str, date_cols: list[str] | None = None) -> pd.DataFrame:
    """Robust query execution for WRDS/SQLAlchemy 2.0."""
    try:
        with db.engine.connect() as conn:
            return pd.read_sql_query(sql=text(sql), con=conn, parse_dates=date_cols)
    except Exception:
        with db.engine.connect() as conn:
            result = conn.execute(text(sql))
            df = pd.DataFrame(result.fetchall(), columns=result.keys())
            if date_cols:
                for col in date_cols:
                    if col in df.columns:
                        df[col] = pd.to_datetime(df[col], errors="coerce")
            return df


def _is_missing_wrds_username(raw: str | None) -> bool:
    user = str(raw or "").strip().lower()
    return user in {"", "your_wrds_username", "wrds_username", "username"}


class WRDSUniverse:
    """
    Point-in-time S&P 500 universe from CRSP dsp500list + stocknames.

    Parameters
    ----------
    db : wrds.Connection
        Active WRDS connection.
    cache_dir : str
        Root cache directory. Universe files go under cache_dir/universe/.
    cache_ttl_days : int
        How long to keep cached membership files before refreshing.
        Default 30 days — membership changes at most a few times per quarter.
    """

    def __init__(
        self,
        db,
        cache_dir: str = "data/cache/wrds",
        cache_ttl_days: int = 30,
    ) -> None:
        self._db = db
        self._cache_dir = Path(cache_dir) / "universe"
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._cache_ttl_days = cache_ttl_days
        self._membership_panel: pd.DataFrame | None = None  # lazy-loaded

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_sp500_at_date(self, date: str | pd.Timestamp) -> pd.DataFrame:
        """
        Return all S&P 500 members as of `date`.

        Columns returned: permno (int), ticker (str), company_name (str).

        Parameters
        ----------
        date : str or Timestamp
            Query date (e.g. "2008-09-15").
        """
        dt = pd.Timestamp(date)
        panel = self._load_full_panel()

        mask = (panel["start"] <= dt) & (
            panel["ending"].isna() | (panel["ending"] >= dt)
        )
        members = panel.loc[mask, ["permno", "ticker", "company_name"]].drop_duplicates(
            subset="permno"
        )
        logger.info("S&P 500 at %s: %d members", dt.date(), len(members))
        return members.reset_index(drop=True)

    def get_sp500_panel(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Return every (permno, ticker, start, ending) record where the security
        was in the S&P 500 at any point between start_date and end_date.

        This covers additions, removals, and re-additions within the window.
        """
        s = pd.Timestamp(start_date)
        e = pd.Timestamp(end_date)
        panel = self._load_full_panel()

        mask = (panel["start"] <= e) & (
            panel["ending"].isna() | (panel["ending"] >= s)
        )
        result = panel.loc[mask].copy()
        # Clip membership dates to the requested window
        result.loc[:, "effective_start"] = result["start"].clip(lower=s)
        result.loc[:, "effective_end"] = result["ending"].fillna(e).clip(upper=e)
        logger.info(
            "S&P 500 panel %s→%s: %d membership records, %d unique PERMNOs",
            s.date(), e.date(), len(result), result["permno"].nunique(),
        )
        return result.reset_index(drop=True)

    def get_unique_permnos(
        self,
        start_date: str | pd.Timestamp,
        end_date: str | pd.Timestamp,
    ) -> list[int]:
        """All unique PERMNOs that appeared in the S&P 500 over the window."""
        panel = self.get_sp500_panel(start_date, end_date)
        return sorted(panel["permno"].unique().tolist())

    def permno_to_ticker_map(
        self,
        permnos: list[int],
        date: str | pd.Timestamp,
    ) -> dict[int, str]:
        """
        Point-in-time PERMNO → ticker mapping from crsp.stocknames.

        Uses namedt ≤ date ≤ nameenddt to pick the ticker that was in use
        on the given date. Returns {permno: ticker}.
        """
        dt = pd.Timestamp(date)
        cache_path = self._cache_dir / f"stocknames_{dt.strftime('%Y%m%d')}.parquet"

        if self._is_cache_fresh(cache_path):
            sn = pd.read_parquet(cache_path)
        else:
            sn = self._fetch_stocknames(dt)
            sn.to_parquet(cache_path, index=False)

        sub = sn[sn["permno"].isin(permnos)]
        result: dict[int, str] = {}
        for _, row in sub.iterrows():
            p = int(row["permno"])
            t = str(row["ticker"]).strip()
            if t:
                result[p] = t
        return result

    # ------------------------------------------------------------------
    # Internal: load / cache the full historical membership panel
    # ------------------------------------------------------------------

    def _load_full_panel(self) -> pd.DataFrame:
        """
        Lazy-load the full dsp500list + stocknames join.
        Cached as a single Parquet file; refreshed if stale.
        """
        if self._membership_panel is not None:
            return self._membership_panel

        cache_path = self._cache_dir / "sp500_membership_panel.parquet"
        if self._is_cache_fresh(cache_path):
            self._membership_panel = pd.read_parquet(cache_path)
            logger.debug(
                "Loaded S&P 500 membership panel from cache (%d rows)", len(self._membership_panel)
            )
            return self._membership_panel

        logger.info("Fetching S&P 500 membership panel from WRDS (crsp.dsp500list + crsp.stocknames)…")
        self._membership_panel = self._fetch_membership_panel()
        self._membership_panel.to_parquet(cache_path, index=False)
        logger.info("Cached membership panel (%d rows)", len(self._membership_panel))
        return self._membership_panel

    def _fetch_membership_panel(self) -> pd.DataFrame:
        """
        Pull crsp.dsp500list (permno, start, ending) and join to
        crsp.stocknames (permno, ticker, comnam, namedt, nameenddt)
        to resolve the ticker that was in use at each entry date.
        """
        # ── Step 1: pull membership table ────────────────────────────────
        membership_sql = """
            SELECT permno, start, ending
            FROM crsp.dsp500list
            ORDER BY permno, start
        """
        membership = wrds_query(self._db, membership_sql, date_cols=["start", "ending"])
        membership["permno"] = membership["permno"].astype(int)
        membership["ending"] = pd.to_datetime(membership["ending"], errors="coerce")

        # ── Step 2: pull stocknames for ticker resolution ─────────────────
        stocknames_sql = """
            SELECT permno, ticker, comnam, namedt, nameenddt
            FROM crsp.stocknames
            ORDER BY permno, namedt
        """
        stocknames = wrds_query(
            self._db, stocknames_sql, date_cols=["namedt", "nameenddt"]
        )
        stocknames["permno"] = stocknames["permno"].astype(int)
        stocknames["nameenddt"] = pd.to_datetime(stocknames["nameenddt"], errors="coerce")

        # ── Step 3: for each membership record, pick the ticker that was
        #            in use at the `start` date of that membership window ──
        records = []
        stocknames_by_permno = stocknames.groupby("permno")

        for _, row in membership.iterrows():
            p = int(row["permno"])
            entry_date = row["start"]

            try:
                sn = stocknames_by_permno.get_group(p)
            except KeyError:
                ticker = ""
                company = ""
            else:
                # nameenddt NaT means still current
                valid = sn[
                    (sn["namedt"] <= entry_date)
                    & (sn["nameenddt"].isna() | (sn["nameenddt"] >= entry_date))
                ]
                if valid.empty:
                    # Fall back to the most recent name before entry_date
                    before = sn[sn["namedt"] <= entry_date]
                    valid = before.tail(1) if not before.empty else sn.head(1)
                ticker = str(valid["ticker"].iloc[0]).strip() if not valid.empty else ""
                company = str(valid["comnam"].iloc[0]).strip() if not valid.empty else ""

            records.append({
                "permno": p,
                "ticker": ticker,
                "company_name": company,
                "start": row["start"],
                "ending": row["ending"],
            })

        panel = pd.DataFrame(records)
        panel["start"] = pd.to_datetime(panel["start"])
        panel["ending"] = pd.to_datetime(panel["ending"], errors="coerce")
        return panel

    def _fetch_stocknames(self, date: pd.Timestamp) -> pd.DataFrame:
        """Pull stocknames valid on a specific date."""
        date_str = date.strftime("%Y-%m-%d")
        sql = f"""
            SELECT permno, ticker, comnam AS company_name, namedt, nameenddt
            FROM crsp.stocknames
            WHERE namedt <= '{date_str}'
              AND (nameenddt IS NULL OR nameenddt >= '{date_str}')
            ORDER BY permno, namedt
        """
        return wrds_query(self._db, sql, date_cols=["namedt", "nameenddt"])

    def _is_cache_fresh(self, path: Path) -> bool:
        """True if the file exists and is younger than cache_ttl_days."""
        if not path.exists():
            return False
        age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
        return age < timedelta(days=self._cache_ttl_days)


def build_backtest_universe(
    db,
    date: str | pd.Timestamp,
    min_price: float = 10.0,
    min_dollar_vol: float = 1e8,
    cache_dir: str = "data/cache/wrds/universe",
    cache_ttl_days: int = 1,
) -> list[int]:
    """
    Build a point-in-time investable universe for a given date.

    Steps:
      1. Get all S&P 500 members as of ``date`` from crsp.dsp500list.
      2. Filter by closing price ≥ min_price and dollar volume ≥ min_dollar_vol
         using the most recent available row in crsp.dsf on or before ``date``.
      3. Exclude PERMNOs whose dlstdt ≤ date in crsp.dsedelist
         (stock already delisted on the construction date — no live price).

    Returns a list of PERMNOs — the internal addressing key for CRSP data.
    Callers that need tickers should pass the result through
    ``WRDSUniverse.permno_to_ticker_map``.

    Parameters
    ----------
    db : wrds.Connection
    date : str or Timestamp
        Universe construction date (e.g. "2010-01-04" — first trading day of year).
    min_price : float
        Minimum stock price. $10 cuts out micro-caps, most institutional mandates.
    min_dollar_vol : float
        Minimum trailing-day dollar volume (abs(prc) × vol). $100 M filters
        illiquid names that are in the index but un-tradeable at scale.
    cache_dir : str
        Directory for caching universe snapshots (one file per date).
    cache_ttl_days : int
        Cache TTL; 0 = always re-query. Default 1 day.

    Returns
    -------
    list[int]  — PERMNOs in the investable universe on ``date``.
    """
    dt = pd.Timestamp(date)
    date_str = dt.strftime("%Y-%m-%d")

    cache_path = Path(cache_dir) / f"universe_{dt.strftime('%Y%m%d')}.parquet"
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_ttl_days > 0 and cache_path.exists():
        age = (pd.Timestamp.now() - pd.Timestamp(cache_path.stat().st_mtime, unit="s")).days
        if age < cache_ttl_days:
            return pd.read_parquet(cache_path)["permno"].tolist()

    # ── Step 1: S&P 500 members at date ──────────────────────────────────────
    universe_sql = f"""
        SELECT DISTINCT permno
        FROM crsp.dsp500list
        WHERE start <= '{date_str}'
          AND (ending IS NULL OR ending >= '{date_str}')
    """
    members = wrds_query(db, universe_sql)
    if members.empty:
        logger.warning("build_backtest_universe: no S&P 500 members found at %s", date_str)
        return []

    permno_list = members["permno"].astype(int).tolist()
    permno_str = ",".join(str(p) for p in permno_list)

    # ── Step 2: Price and liquidity filter from crsp.dsf ─────────────────────
    # Use the most recent trading day on or before `date` for each permno.
    liquidity_sql = f"""
        SELECT permno, date, ABS(prc) AS price, ABS(prc) * vol AS dollar_vol
        FROM crsp.dsf
        WHERE permno IN ({permno_str})
          AND date <= '{date_str}'
          AND date >= '{(dt - pd.Timedelta(days=10)).strftime("%Y-%m-%d")}'
        QUALIFY ROW_NUMBER() OVER (PARTITION BY permno ORDER BY date DESC) = 1
    """
    try:
        liq = wrds_query(db, liquidity_sql)
    except Exception:
        # Some WRDS setups don't support QUALIFY — fall back to subquery
        liquidity_sql_fallback = f"""
            SELECT a.permno, ABS(a.prc) AS price, ABS(a.prc) * a.vol AS dollar_vol
            FROM crsp.dsf a
            INNER JOIN (
                SELECT permno, MAX(date) AS max_date
                FROM crsp.dsf
                WHERE permno IN ({permno_str})
                  AND date <= '{date_str}'
                  AND date >= '{(dt - pd.Timedelta(days=10)).strftime("%Y-%m-%d")}'
                GROUP BY permno
            ) b ON a.permno = b.permno AND a.date = b.max_date
        """
        liq = wrds_query(db, liquidity_sql_fallback)

    liq["permno"] = liq["permno"].astype(int)
    liquid_permnos = set(
        liq.loc[
            (liq["price"] >= min_price) & (liq["dollar_vol"] >= min_dollar_vol),
            "permno",
        ].tolist()
    )

    # ── Step 3: Exclude already-delisted PERMNOs ─────────────────────────────
    delist_sql = f"""
        SELECT DISTINCT permno
        FROM crsp.dsedelist
        WHERE permno IN ({permno_str})
          AND dlstdt <= '{date_str}'
    """
    delisted = wrds_query(db, delist_sql)
    delisted_set = set(delisted["permno"].astype(int).tolist()) if not delisted.empty else set()

    # ── Combine filters ───────────────────────────────────────────────────────
    investable = [
        p for p in permno_list
        if p in liquid_permnos and p not in delisted_set
    ]

    # Cache result
    pd.DataFrame({"permno": investable}).to_parquet(cache_path, index=False)

    logger.info(
        "build_backtest_universe(%s): %d S&P500 members → %d liquid → %d ex-delist → %d investable",
        date_str, len(permno_list), len(liquid_permnos),
        len(liquid_permnos & delisted_set), len(investable),
    )
    return investable


_GLOBAL_WRDS_CONNECTION = None

def connect_wrds(username: str | None = None) -> "wrds.Connection":
    """
    Convenience wrapper: create a WRDS connection using the environment
    variable WRDS_USERNAME (or explicit username).

    Caches the connection to prevent connection-limit exhaustion (FATAL: too many connections).

    Usage:
        db = connect_wrds()   # reads WRDS_USERNAME from env
        db = connect_wrds("myusername")
    """
    global _GLOBAL_WRDS_CONNECTION
    if _GLOBAL_WRDS_CONNECTION is not None:
        return _GLOBAL_WRDS_CONNECTION

    try:
        import wrds  # type: ignore[import]
    except ImportError as exc:
        raise ImportError(
            "wrds package not installed. Run: pip install wrds"
        ) from exc

    user = username or os.environ.get("WRDS_USERNAME")
    if _is_missing_wrds_username(user):
        raise ValueError(
            "WRDS username required. Set WRDS_USERNAME environment variable "
            "or pass username= explicitly. Placeholder values like "
            "'your_wrds_username' are rejected to avoid interactive prompts."
        )
    _GLOBAL_WRDS_CONNECTION = wrds.Connection(wrds_username=user)
    return _GLOBAL_WRDS_CONNECTION
