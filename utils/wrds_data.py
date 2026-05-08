from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd

from utils.wrds_compat import wrds_raw_sql
from utils.wrds_loader import WRDSLoader
from utils.wrds_universe import connect_wrds

logger = logging.getLogger(__name__)


def resolve_data_provider(provider: str | None = None) -> str:
    """Resolve the historical price provider without implicit Yahoo fallback."""
    if provider:
        return str(provider).lower()
    raw = os.environ.get("TREND_DATA_PROVIDER") or os.environ.get("DATA_PROVIDER")
    if raw:
        return str(raw).lower()
    raise ValueError(
        "data provider must be explicit: pass data_provider='wrds'/'yahoo' "
        "or set TREND_DATA_PROVIDER"
    )


def _quote_sql_strings(values: list[str]) -> str:
    quoted = []
    for value in values:
        clean = str(value).replace("'", "''").strip()
        if clean:
            quoted.append(f"'{clean}'")
    return ",".join(quoted)


def resolve_ticker_to_permno(
    tickers: list[str],
    *,
    as_of_date: str | pd.Timestamp,
    username: str | None = None,
    cache_dir: str = "data/cache/wrds",
    existing_map: dict[str, int] | None = None,
) -> dict[str, int]:
    """
    Resolve arbitrary ticker symbols to CRSP PERMNOs on a point-in-time basis.

    `existing_map` wins when present so callers can preserve a PIT universe map
    built upstream and only fill the remaining benchmark/helper symbols here.
    """
    result: dict[str, int] = {
        str(t).upper(): int(p)
        for t, p in (existing_map or {}).items()
        if t is not None and p is not None
    }
    wanted = [str(t).upper() for t in tickers if str(t).strip()]
    missing = [t for t in wanted if t not in result]
    if not missing:
        return result

    dt = pd.Timestamp(as_of_date)
    cache_root = Path(cache_dir) / "universe"
    cache_root.mkdir(parents=True, exist_ok=True)
    cache_path = cache_root / f"ticker_permno_{dt.strftime('%Y%m%d')}.parquet"

    if cache_path.exists():
        try:
            cached = pd.read_parquet(cache_path)
            cached["ticker"] = cached["ticker"].astype(str).str.upper()
            cached_map = {
                str(row["ticker"]).upper(): int(row["permno"])
                for _, row in cached.iterrows()
                if pd.notna(row.get("permno"))
            }
            for ticker in missing:
                if ticker in cached_map:
                    result[ticker] = cached_map[ticker]
            missing = [t for t in wanted if t not in result]
            if not missing:
                return result
        except Exception:
            logger.exception("Failed to read WRDS ticker->permno cache from %s", cache_path)

    db = connect_wrds(username)
    dt_str = dt.strftime("%Y-%m-%d")
    ticker_sql = _quote_sql_strings(missing)
    if not ticker_sql:
        return result

    sql = f"""
        SELECT permno, ticker, namedt, nameenddt
        FROM crsp.stocknames
        WHERE UPPER(ticker) IN ({ticker_sql})
          AND namedt <= '{dt_str}'
          AND (nameenddt IS NULL OR nameenddt >= '{dt_str}')
        ORDER BY ticker, namedt DESC
    """
    names = wrds_raw_sql(db, sql, date_cols=["namedt", "nameenddt"])

    if names.empty:
        fallback_sql = f"""
            SELECT permno, ticker, namedt, nameenddt
            FROM crsp.stocknames
            WHERE UPPER(ticker) IN ({ticker_sql})
              AND namedt <= '{dt_str}'
            ORDER BY ticker, namedt DESC
        """
        names = wrds_raw_sql(db, fallback_sql, date_cols=["namedt", "nameenddt"])

    if not names.empty:
        names["ticker"] = names["ticker"].astype(str).str.upper()
        names["permno"] = names["permno"].astype(int)
        names = names.drop_duplicates(subset=["ticker"], keep="first")
        for _, row in names.iterrows():
            result[str(row["ticker"]).upper()] = int(row["permno"])

        try:
            all_rows = pd.DataFrame(
                [{"ticker": t, "permno": p} for t, p in sorted(result.items())]
            )
            all_rows.to_parquet(cache_path, index=False)
        except Exception:
            logger.exception("Failed to write WRDS ticker->permno cache to %s", cache_path)

    unresolved = [t for t in wanted if t not in result]
    if unresolved:
        logger.warning("WRDS ticker resolution failed for %d symbols: %s", len(unresolved), unresolved[:10])

    return result


def load_wrds_price_panel(
    tickers: list[str],
    *,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    username: str | None = None,
    cache_dir: str = "data/cache/wrds",
    cache_ttl_days: int = 30,
    ticker_to_permno: dict[str, int] | None = None,
    as_of_date: str | pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load CRSP price history for arbitrary ticker symbols through WRDS.
    """
    wanted = [str(t).upper() for t in tickers if str(t).strip()]
    if not wanted:
        return {}

    resolved = resolve_ticker_to_permno(
        wanted,
        as_of_date=as_of_date or end_date,
        username=username,
        cache_dir=cache_dir,
        existing_map=ticker_to_permno,
    )
    relevant = {t: int(resolved[t]) for t in wanted if t in resolved}
    if not relevant:
        return {}

    db = connect_wrds(username)
    loader = WRDSLoader(db, cache_dir=cache_dir, cache_ttl_days=cache_ttl_days)
    return loader.load_universe(
        permnos=sorted(set(relevant.values())),
        ticker_map={int(v): k for k, v in relevant.items()},
        start_date=start_date,
        end_date=end_date,
    )
