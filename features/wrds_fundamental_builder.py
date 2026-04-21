"""
Compustat Fundamental Feature Builder (WRDS)
============================================
Replaces the SEC EDGAR XBRL approach with Compustat quarterly fundamentals
(`comp.fundq`) sourced through WRDS.

Key improvements over the EDGAR implementation:

1. Point-in-time via rdq:
   ``rdq`` is Compustat's earnings announcement date — the day the quarterly
   results were publicly released.  Using ``rdq`` as the availability date
   (instead of ``datadate`` or a fixed +45d lag) is the standard academic and
   practitioner approach for eliminating look-ahead bias in fundamental features.
   Livnat & Mendenhall (2006) document that market reaction to SUE is immediate
   at ``rdq``; any signal that uses a later date leaks information.

2. CRSP–Compustat link via ccmxpf_linktable:
   Translates PERMNO → GVKEY using the CRSP official linktable.
   linktype IN ('LU','LC','LS') and linkprim IN ('P','C') ensures we get the
   primary security link and exclude stale/duplicate links.

3. Same output schema as fundamental_builder.py:
   f_score, accruals_ratio, roa, delta_roa, delta_leverage
   Downstream code (feature_pipeline.py) requires no changes.

Compustat columns used (comp.fundq):
    atq   : Total Assets (quarterly, balance sheet)
    niq   : Net Income (quarterly, income statement)
    oancfy: Net Cash from Operating Activities (YTD; we convert to Q via diff)
    dlttq : Long-Term Debt (quarterly, balance sheet)
    actq  : Current Assets (quarterly, balance sheet)
    lctq  : Current Liabilities (quarterly, balance sheet)
    rdq   : Report Date of Quarterly Earnings (availability date)
    datadate: Quarter-end date (used to identify the period, not availability)
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

CACHE_DIR = Path("data/cache/wrds/fundamentals")
CACHE_TTL_DAYS = 30

_OUTPUT_COLS = ["f_score", "accruals_ratio", "roa", "delta_roa", "delta_leverage"]


def _zeros(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(0.0, index=dates, columns=_OUTPUT_COLS)


def _is_cache_fresh(path: Path, ttl_days: int = CACHE_TTL_DAYS) -> bool:
    if not path.exists():
        return False
    age = datetime.now() - datetime.fromtimestamp(path.stat().st_mtime)
    return age < timedelta(days=ttl_days)


# ---------------------------------------------------------------------------
# CRSP–Compustat link
# ---------------------------------------------------------------------------

def _get_gvkey(
    db,
    permno: int,
    cache_dir: Path,
) -> str | None:
    """
    Return the primary GVKEY for a PERMNO using the CRSP–Compustat linktable.

    Uses linktype IN ('LU','LC','LS') and linkprim IN ('P','C') per WRDS
    documentation on best-practice link selection.  Returns the most recently
    active link when multiple records exist.
    """
    cache_path = cache_dir / f"link_permno_{permno}.txt"
    if _is_cache_fresh(cache_path):
        return cache_path.read_text().strip() or None

    sql = f"""
        SELECT gvkey, linkdt, linkenddt
        FROM crsp.ccmxpf_linktable
        WHERE lpermno = {permno}
          AND linktype IN ('LU','LC','LS')
          AND linkprim IN ('P','C')
        ORDER BY linkenddt DESC NULLS FIRST, linkdt DESC
        LIMIT 1
    """
    try:
        df = db.raw_sql(sql)
        gvkey = str(df["gvkey"].iloc[0]).strip() if not df.empty else ""
    except Exception as exc:
        logger.warning("CRSP-Compustat link query failed for permno=%s: %s", permno, exc)
        gvkey = ""

    cache_path.write_text(gvkey)
    return gvkey or None


# ---------------------------------------------------------------------------
# Compustat quarterly fundamentals
# ---------------------------------------------------------------------------

def _fetch_compustat_quarterly(
    db,
    gvkey: str,
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path,
) -> pd.DataFrame:
    """
    Pull comp.fundq for a single GVKEY and return a cleaned DataFrame.

    Columns: datadate, rdq, atq, niq, oancfy, dlttq, actq, lctq.

    ``rdq`` is the key — it is the earnings announcement date (the day the
    data became public).  We use it as the point-in-time availability date.
    If ``rdq`` is null, we fall back to ``datadate + 45 days`` (conservative).

    ``oancfy`` is a YTD figure in Compustat.  We convert to standalone quarters
    via first-differences within each fiscal year, then patch the Q1 value
    (which is already standalone for the first quarter of the year).
    """
    cache_path = cache_dir / f"fundq_{gvkey}.parquet"
    if _is_cache_fresh(cache_path):
        df = pd.read_parquet(cache_path)
    else:
        # Buffer by 4 years before start to give delta_* features warm-up history
        buf_start = (start - pd.DateOffset(years=4)).strftime("%Y-%m-%d")
        end_str = end.strftime("%Y-%m-%d")

        sql = f"""
            SELECT gvkey, datadate, rdq, fqtr, fyearq,
                   atq, niq, oancfy, dlttq, actq, lctq
            FROM comp.fundq
            WHERE gvkey = '{gvkey}'
              AND datafmt = 'STD'
              AND indfmt = 'INDL'
              AND datadate BETWEEN '{buf_start}' AND '{end_str}'
            ORDER BY datadate
        """
        try:
            df = db.raw_sql(sql, date_cols=["datadate", "rdq"])
        except Exception as exc:
            logger.warning("comp.fundq query failed for gvkey=%s: %s", gvkey, exc)
            return pd.DataFrame()

        df.to_parquet(cache_path, index=False)

    if df.empty:
        return df

    # ── Convert oancfy (YTD) to standalone quarterly cash flow ──────────────
    # Sort by fiscal year and quarter so diff() works correctly.
    df = df.sort_values(["fyearq", "fqtr"]).copy()
    df["ocf_q"] = (
        df.groupby("fyearq")["oancfy"]
        .transform(lambda x: x.diff().where(x.index != x.index[0], x))
    )
    # Q1 of each fiscal year is already standalone (YTD = Q1 for first quarter)
    q1_mask = df["fqtr"] == 1
    df.loc[q1_mask, "ocf_q"] = df.loc[q1_mask, "oancfy"]

    # ── Point-in-time date: prefer rdq, fall back to datadate + 45d ─────────
    df["avail_date"] = df["rdq"].fillna(df["datadate"] + pd.Timedelta(days=45))
    df = df.sort_values("avail_date").reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def _compute_features_from_quarterly(
    quarterly: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Map point-in-time quarterly Compustat data onto a daily date index and
    compute Piotroski-style fundamental features.

    Availability date = avail_date (rdq or datadate+45d).
    Each value is forward-filled after its availability date.
    """
    if quarterly.empty:
        return _zeros(dates)

    def _pit_forward_fill(series: pd.Series, avail_dates: pd.Index) -> pd.Series:
        """Forward-fill a quarterly series indexed by avail_date onto daily dates."""
        s = series.copy()
        s.index = avail_dates
        s = s[~s.index.duplicated(keep="first")].sort_index()
        full_range = pd.date_range(
            start=min(s.index.min(), dates.min()),
            end=max(s.index.max(), dates.max()),
            freq="D",
        )
        return s.reindex(full_range).ffill().reindex(dates)

    avail = pd.DatetimeIndex(quarterly["avail_date"])
    atq = _pit_forward_fill(quarterly["atq"].values, avail)
    niq = _pit_forward_fill(quarterly["niq"].values, avail)
    ocf = _pit_forward_fill(quarterly["ocf_q"].values, avail)
    dlttq = _pit_forward_fill(quarterly["dlttq"].fillna(0.0).values, avail)
    actq = _pit_forward_fill(quarterly["actq"].values, avail)
    lctq = _pit_forward_fill(quarterly["lctq"].replace(0, np.nan).values, avail)

    # ── Core ratios ──────────────────────────────────────────────────────────
    ta_safe = atq.replace(0, np.nan)
    roa = (niq / ta_safe).clip(-5.0, 5.0)
    delta_roa = roa.diff()

    accruals = ((niq - ocf) / ta_safe).clip(-2.0, 2.0)

    leverage = (dlttq / ta_safe).clip(0.0, 10.0)
    delta_leverage = leverage.diff()

    current_ratio = (actq / lctq).clip(0.0, 20.0)
    delta_liquidity = current_ratio.diff()

    # ── Piotroski F-score (5 components) ────────────────────────────────────
    f1 = (roa > 0).astype(float)
    f2 = (delta_roa > 0).astype(float)
    f3 = (accruals < 0).astype(float)
    f4 = (delta_leverage < 0).astype(float)
    f5 = (delta_liquidity > 0).astype(float)
    f_score = (f1 + f2 + f3 + f4 + f5).clip(0, 5)

    result = pd.DataFrame({
        "f_score": f_score,
        "accruals_ratio": accruals,
        "roa": roa,
        "delta_roa": delta_roa,
        "delta_leverage": delta_leverage,
    }, index=dates)

    return result.fillna(0.0)


# ---------------------------------------------------------------------------
# Public API — mirrors fundamental_builder.fetch_fundamental_features
# ---------------------------------------------------------------------------

def fetch_fundamental_features(
    db,
    permno: int,
    dates: pd.DatetimeIndex,
    cache_dir: Path | None = None,
    cache_ttl_days: int = CACHE_TTL_DAYS,
) -> pd.DataFrame:
    """
    Return a DataFrame (indexed by `dates`) with Compustat-based fundamental
    features using CRSP PERMNO as the identifier.

    Parameters
    ----------
    db : wrds.Connection
        Active WRDS connection.
    permno : int
        CRSP PERMNO (from WRDSUniverse).
    dates : pd.DatetimeIndex
        Output daily index.
    cache_dir : Path, optional
        Override for the cache directory.
    cache_ttl_days : int
        Cache TTL for individual ticker feature files.

    Returns
    -------
    pd.DataFrame with columns: f_score, accruals_ratio, roa, delta_roa,
        delta_leverage. Filled with 0.0 when data is unavailable.
    """
    global CACHE_DIR
    cdir = cache_dir or CACHE_DIR
    cdir.mkdir(parents=True, exist_ok=True)

    feature_cache = cdir / f"features_permno_{permno}.parquet"
    if _is_cache_fresh(feature_cache, cache_ttl_days):
        try:
            cached = pd.read_parquet(feature_cache)
            cached.index = pd.DatetimeIndex(cached.index)
            if all(c in cached.columns for c in _OUTPUT_COLS):
                return cached.reindex(dates, method="ffill").fillna(0.0)[_OUTPUT_COLS]
        except Exception:
            pass

    try:
        start = dates.min()
        end = dates.max()

        gvkey = _get_gvkey(db, permno, cdir)
        if not gvkey:
            logger.debug("No CRSP-Compustat link for permno=%s; returning zeros.", permno)
            return _zeros(dates)

        quarterly = _fetch_compustat_quarterly(db, gvkey, start, end, cdir)
        if quarterly.empty:
            logger.debug("No Compustat data for gvkey=%s (permno=%s).", gvkey, permno)
            return _zeros(dates)

        result = _compute_features_from_quarterly(quarterly, dates)
        result.to_parquet(feature_cache)
        return result

    except Exception as exc:
        logger.warning(
            "fetch_fundamental_features failed for permno=%s: %s", permno, exc
        )
        return _zeros(dates)


def fetch_fundamental_features_batch(
    db,
    permno_dates_map: dict[int, pd.DatetimeIndex],
    cache_dir: Path | None = None,
    cache_ttl_days: int = CACHE_TTL_DAYS,
) -> dict[int, pd.DataFrame]:
    """
    Batch-load fundamental features for multiple PERMNOs.

    Fetches each PERMNO sequentially (Compustat queries are fast; GVKEY-level
    caching means subsequent calls are cache hits).

    Returns {permno: fundamental_feature_df}.
    """
    cdir = cache_dir or CACHE_DIR
    cdir.mkdir(parents=True, exist_ok=True)

    results: dict[int, pd.DataFrame] = {}
    for permno, dates in permno_dates_map.items():
        results[permno] = fetch_fundamental_features(
            db, permno, dates, cache_dir=cdir, cache_ttl_days=cache_ttl_days
        )
    logger.info(
        "WRDSFundamentals: loaded for %d PERMNOs", len(results)
    )
    return results
