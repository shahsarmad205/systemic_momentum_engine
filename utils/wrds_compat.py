"""
wrds + pandas 2.x + SQLAlchemy 2.x compatibility shim.

pd.read_sql_query fails to detect the wrds SA-2 Connection as a SQLAlchemy
connectable (isinstance check mismatch across SA versions), falling into the
legacy DBAPI2 path and crashing with AttributeError / TypeError.
Fetching via conn.execute() + fetchall() bypasses pd.read_sql_query entirely.
"""
from __future__ import annotations

import sqlalchemy
import pandas as pd


def wrds_raw_sql(
    db,
    sql: str,
    date_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Execute SQL via SQLAlchemy directly, bypassing pd.read_sql_query."""
    with db.engine.connect() as conn:
        result = conn.execute(sqlalchemy.text(sql))
        df = pd.DataFrame(result.fetchall(), columns=list(result.keys()))
    if date_cols:
        for col in date_cols:
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
    return df
