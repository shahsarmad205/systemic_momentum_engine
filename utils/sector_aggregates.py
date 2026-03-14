"""
Sector-Level Aggregate Signals
===============================
Compute sector momentum, sector volatility, and average sector sentiment
from per-ticker data. Used to adjust individual stock signals by sector context.

All series use only past data (no look-ahead).
"""

from __future__ import annotations

import pandas as pd

from .sectors import SECTOR_MAP, get_sector, tickers_by_sector

MOMENTUM_WINDOW = 20
VOLATILITY_WINDOW = 20


def compute_sector_aggregates(
    price_data: dict[str, pd.DataFrame],
    sector_map: dict[str, str] | None = None,
    sentiment_by_ticker: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Compute per-date, per-sector aggregates from price_data (and optional sentiment).

    Returns a DataFrame with index = date and columns:
        sector_{name}_momentum   — average 20-day return of tickers in that sector
        sector_{name}_volatility — average 20-day rolling std of returns in that sector
        sector_{name}_sentiment  — average sentiment of tickers in that sector (0 if not provided)

    Only sectors that have at least one ticker in price_data are included.
    """
    sector_map = sector_map or SECTOR_MAP
    sector_to_tickers = tickers_by_sector(sector_map)

    # All dates from all tickers
    all_dates: set[pd.Timestamp] = set()
    for df in price_data.values():
        all_dates.update(df.index)
    dates = sorted(all_dates)

    # Per-ticker series: 20d return and 20d vol
    ticker_momentum: dict[str, pd.Series] = {}
    ticker_vol: dict[str, pd.Series] = {}
    for ticker, df in price_data.items():
        if df.empty or "Close" not in df.columns:
            continue
        close = df["Close"]
        ret = close.pct_change()
        ticker_momentum[ticker] = close.pct_change(MOMENTUM_WINDOW)
        ticker_vol[ticker] = ret.rolling(VOLATILITY_WINDOW).std()

    # Build sector aggregates per date
    sectors_in_universe = set()
    for t in price_data:
        sectors_in_universe.add(get_sector(t))

    columns = []
    for sector in sorted(sectors_in_universe):
        columns.append(f"sector_{sector}_momentum")
        columns.append(f"sector_{sector}_volatility")
        columns.append(f"sector_{sector}_sentiment")

    result = pd.DataFrame(index=dates, columns=columns, dtype=float)
    result.fillna(0.0, inplace=True)

    for sector in sectors_in_universe:
        tickers_in_sector = [t for t in sector_to_tickers.get(sector, []) if t in price_data]
        if not tickers_in_sector:
            continue

        mom_col = f"sector_{sector}_momentum"
        vol_col = f"sector_{sector}_volatility"
        sent_col = f"sector_{sector}_sentiment"

        # Average momentum per date (only tickers with valid data that date)
        mom_series = pd.DataFrame(
            {t: ticker_momentum[t].reindex(dates) for t in tickers_in_sector if t in ticker_momentum}
        )
        if not mom_series.empty:
            result[mom_col] = mom_series.mean(axis=1)

        vol_series = pd.DataFrame(
            {t: ticker_vol[t].reindex(dates) for t in tickers_in_sector if t in ticker_vol}
        )
        if not vol_series.empty:
            result[vol_col] = vol_series.mean(axis=1)

        if sentiment_by_ticker:
            sent_vals = [sentiment_by_ticker.get(t, 0.0) for t in tickers_in_sector]
            result[sent_col] = sum(sent_vals) / len(sent_vals) if sent_vals else 0.0

    return result.fillna(0.0)


def apply_sector_adjustment(
    signal_data: dict[str, pd.DataFrame],
    sector_aggregates: pd.DataFrame,
    sector_map: dict[str, str],
    momentum_weight: float = 0.1,
    volatility_weight: float = -0.05,
    sentiment_weight: float = 0.1,
) -> None:
    """
    Adjust each ticker's adjusted_score using its sector's aggregates (in-place).

    Formula: adjusted_new = adjusted_old
             + momentum_weight * sector_momentum
             + volatility_weight * sector_volatility
             + sentiment_weight * sector_sentiment

    Then reclassify signal from the new adjusted score.
    """
    from main import classify_final_signal

    for ticker, sig_df in signal_data.items():
        sector = sector_map.get(ticker, "Other")
        mom_col = f"sector_{sector}_momentum"
        vol_col = f"sector_{sector}_volatility"
        sent_col = f"sector_{sector}_sentiment"

        if mom_col not in sector_aggregates.columns:
            continue

        # Align sector aggregates to this ticker's index
        agg_aligned = sector_aggregates.reindex(sig_df.index).fillna(0)
        mom = agg_aligned[mom_col].values
        vol = agg_aligned[vol_col].values
        sent = agg_aligned[sent_col].values

        adj = sig_df["adjusted_score"].values.copy()
        adj = adj + momentum_weight * mom + volatility_weight * vol + sentiment_weight * sent
        sig_df["adjusted_score"] = adj
        sig_df["signal"] = pd.Series(adj, index=sig_df.index).apply(classify_final_signal)
