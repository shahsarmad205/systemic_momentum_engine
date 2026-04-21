"""
Export CRSP batch cache → per-ticker Parquet files in data/cache/
==================================================================
Bridges the WRDSLoader batch cache (data/cache/wrds/prices/*.parquet)
to the per-ticker cache format that get_ohlcv() reads
(data/cache/{TICKER}.parquet).

Run this AFTER pull_crsp_history.py completes:

    python scripts/export_crsp_to_cache.py \\
        --start 2005-01-01 --end 2024-12-31 \\
        --delete-yahoo

After this script, get_ohlcv("AAPL", ...) returns CRSP-sourced data
and run_model_selection.py / run_backtest.py read CRSP transparently.

Column schema written (matches what get_ohlcv / feature_builder expect):
    Open        adj_price  (CRSP has no intraday open)
    High        adj_price × (1 + rolling vol)
    Low         adj_price × (1 - rolling vol)
    Close       adj_price  (split/dividend-adjusted, canonical)
    Volume      vol (CRSP raw share count)
    AdjClose    adj_price  (legacy alias = Close for CRSP)
    raw_price   abs(prc)   (unadjusted close)
    ret         CRSP daily total return (source of truth)
    adj_price   same as Close
    dollar_volume, shares_out, market_cap
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("export_crsp_to_cache")


def parse_args():
    p = argparse.ArgumentParser(description="Export CRSP batch files to per-ticker cache")
    p.add_argument("--start", default="2005-01-01")
    p.add_argument("--end",   default="2024-12-31")
    p.add_argument(
        "--wrds-cache-dir", default="data/cache/wrds",
        help="Root WRDS cache directory (default: data/cache/wrds)",
    )
    p.add_argument(
        "--output-dir", default="data/cache",
        help="Per-ticker output directory (default: data/cache)",
    )
    p.add_argument(
        "--delete-yahoo",
        action="store_true",
        help="Delete existing Yahoo .parquet files in --output-dir before writing CRSP files.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    wrds_user = os.environ.get("WRDS_USERNAME")
    if not wrds_user:
        logger.error("WRDS_USERNAME not set. Run: export WRDS_USERNAME=your_username")
        sys.exit(1)

    from utils.wrds_universe import WRDSUniverse, connect_wrds
    from utils.wrds_loader import WRDSLoader

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Optional: delete Yahoo cache first ───────────────────────────────────
    if args.delete_yahoo:
        yahoo_files = list(out_dir.glob("*.parquet"))
        # Don't delete files that look like CRSP exports (we haven't written any yet)
        logger.warning("Deleting %d Yahoo Parquet files from %s …", len(yahoo_files), out_dir)
        for f in yahoo_files:
            f.unlink()
        logger.info("Yahoo cache cleared.")

    # ── Connect and load universe ─────────────────────────────────────────────
    logger.info("Connecting to WRDS as '%s' …", wrds_user)
    db = connect_wrds(wrds_user)
    logger.info("Connected.")

    start = pd.Timestamp(args.start)
    end   = pd.Timestamp(args.end)
    buf_start = start - pd.Timedelta(days=400)

    universe = WRDSUniverse(db, cache_dir=args.wrds_cache_dir, cache_ttl_days=30)
    panel = universe.get_sp500_panel(args.start, args.end)
    all_permnos = sorted(panel["permno"].unique().tolist())

    ticker_map = universe.permno_to_ticker_map(all_permnos, args.start)
    logger.info(
        "Universe: %d PERMNOs, %d with ticker mappings.",
        len(all_permnos), len(ticker_map),
    )

    # ── Load all price data from WRDS batch cache ─────────────────────────────
    logger.info("Loading CRSP price data from batch cache …")
    loader = WRDSLoader(db, cache_dir=args.wrds_cache_dir, cache_ttl_days=30)

    batch_size = 200
    batches = [all_permnos[i:i+batch_size] for i in range(0, len(all_permnos), batch_size)]

    written = 0
    skipped = 0
    for i, batch in enumerate(batches, 1):
        batch_map = {p: ticker_map.get(p, f"PERMNO_{p}") for p in batch}
        price_data = loader.load_universe(batch, batch_map, buf_start, end)

        for ticker, df in price_data.items():
            if df.empty:
                skipped += 1
                continue

            # Trim to requested window (buf_start was for warm-up only)
            df_out = df.loc[(df.index >= start) & (df.index <= end)].copy()
            if len(df_out) < 20:
                skipped += 1
                continue

            out_path = out_dir / f"{ticker}.parquet"
            df_out.to_parquet(out_path)
            written += 1

        logger.info(
            "Batch %d/%d — %d securities written (%d skipped so far).",
            i, len(batches), written, skipped,
        )

    logger.info(
        "\nExport complete: %d tickers written to %s  (%d skipped — insufficient data).",
        written, out_dir, skipped,
    )
    logger.info(
        "run_model_selection.py and run_backtest.py will now read CRSP data via get_ohlcv()."
    )


if __name__ == "__main__":
    main()
