"""
Fundamental Feature Router
===========================
Central dispatch layer for fundamental features.

When WRDS is configured (via ``configure_wrds``), routes requests to
``wrds_fundamental_builder`` (Compustat + rdq point-in-time).

Falls back to ``fundamental_builder`` (SEC EDGAR XBRL) when WRDS is not
configured or when the ticker has no PERMNO in the map.

Callers import from here instead of from either concrete module:

    from features.fundamental_router import fetch_fundamental_features, configure_wrds
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level WRDS state (set once at startup by run_backtest.py or similar)
# ---------------------------------------------------------------------------
_wrds_db = None
_ticker_to_permno: dict[str, int] = {}


def configure_wrds(db, ticker_to_permno: dict[str, int]) -> None:
    """
    Register a WRDS connection and ticker→PERMNO map so the router can use
    Compustat fundamentals instead of EDGAR.

    Call this once at startup (e.g. in run_backtest.py) before any features
    are computed:

        from features.fundamental_router import configure_wrds
        configure_wrds(db, universe.permno_to_ticker_map(...))
    """
    global _wrds_db, _ticker_to_permno
    _wrds_db = db
    _ticker_to_permno = ticker_to_permno
    logger.info(
        "fundamental_router: WRDS configured — %d tickers mapped to PERMNOs.",
        len(ticker_to_permno),
    )


def is_wrds_configured() -> bool:
    """True when WRDS has been configured and at least one PERMNO is mapped."""
    return _wrds_db is not None and bool(_ticker_to_permno)


def fetch_fundamental_features(
    ticker: str,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Return fundamental features for ``ticker`` on ``dates``.

    Routing priority:
      1. WRDS / Compustat path (if configured and ticker has a PERMNO)
      2. EDGAR XBRL fallback (always available, no credentials needed)

    Output columns: f_score, accruals_ratio, roa, delta_roa, delta_leverage.
    Returns 0.0 on any error — never raises, never blocks the pipeline.
    """
    _ZERO_COLS = ["f_score", "accruals_ratio", "roa", "delta_roa", "delta_leverage"]

    if _wrds_db is not None:
        permno = _ticker_to_permno.get(ticker)
        if permno is not None:
            try:
                from features.wrds_fundamental_builder import fetch_fundamental_features as _wrds
                return _wrds(_wrds_db, permno, dates)
            except Exception as exc:
                logger.warning(
                    "WRDS fundamental fetch failed for %s (permno=%s): %s — falling back to EDGAR.",
                    ticker, permno, exc,
                )

    # EDGAR fallback
    try:
        from features.fundamental_builder import fetch_fundamental_features as _edgar
        return _edgar(ticker, dates)
    except Exception as exc:
        logger.debug("EDGAR fundamental fetch failed for %s: %s", ticker, exc)

    return pd.DataFrame(0.0, index=dates, columns=_ZERO_COLS)
