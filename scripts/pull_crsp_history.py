"""
Pull full CRSP price history for the S&P 500 universe.

Usage:
    python scripts/pull_crsp_history.py --start 2005-01-01 --end 2024-12-31

This script:
  1. Loads the complete S&P 500 membership panel from crsp.dsp500list.
  2. Batch-fetches crsp.dsf for all PERMNOs that were ever in the index.
  3. Splices delisting returns from crsp.dsedelist.
  4. Writes one Parquet file per batch to data/cache/wrds/prices/.

Run this ONCE after deleting the old Yahoo cache. Subsequent runs are
cache hits (TTL=30 days) and return immediately.

Prerequisite:
    export WRDS_USERNAME=your_wrds_username
    # First connection prompts for password and caches ~/.pgpass automatically.
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
logger = logging.getLogger("pull_crsp_history")


def parse_args():
    p = argparse.ArgumentParser(description="Pull CRSP daily history for S&P 500")
    p.add_argument("--start", default="2005-01-01", help="Start date (inclusive)")
    p.add_argument("--end",   default="2024-12-31", help="End date (inclusive)")
    p.add_argument(
        "--batch-size", type=int, default=200,
        help="PERMNOs per SQL query (default 200; reduce if queries time out)",
    )
    p.add_argument(
        "--cache-dir", default="data/cache/wrds",
        help="Root cache directory (default: data/cache/wrds)",
    )
    return p.parse_args()


def main():
    args = parse_args()

    wrds_user = os.environ.get("WRDS_USERNAME")
    if not wrds_user:
        logger.error(
            "WRDS_USERNAME not set.\n"
            "  Fix: export WRDS_USERNAME=your_wrds_username\n"
            "  Then re-run this script."
        )
        sys.exit(1)

    from utils.wrds_universe import WRDSUniverse, connect_wrds
    from utils.wrds_loader import WRDSLoader

    # ── Connect ───────────────────────────────────────────────────────────────
    logger.info("Connecting to WRDS as '%s' …", wrds_user)
    db = connect_wrds(wrds_user)
    logger.info("Connected.")

    start = pd.Timestamp(args.start)
    end   = pd.Timestamp(args.end)

    # Buffer start for feature warm-up (momentum_12m_skip1 needs 252+21 days)
    buf_start = start - pd.Timedelta(days=400)
    logger.info("Buffered start: %s  (400 days before %s for feature warm-up)", buf_start.date(), start.date())

    # ── Universe: every PERMNO ever in the S&P 500 over the window ───────────
    logger.info("Loading S&P 500 membership panel …")
    universe = WRDSUniverse(db, cache_dir=args.cache_dir, cache_ttl_days=30)
    panel = universe.get_sp500_panel(args.start, args.end)
    all_permnos = sorted(panel["permno"].unique().tolist())
    logger.info("  %d unique PERMNOs ever in the index between %s and %s.", len(all_permnos), args.start, args.end)

    # ── Ticker map (point-in-time at start date for logging) ─────────────────
    ticker_map = universe.permno_to_ticker_map(all_permnos, args.start)
    logger.info("  Ticker map resolved for %d PERMNOs.", len(ticker_map))

    # ── Batch-load CRSP DSF ───────────────────────────────────────────────────
    loader = WRDSLoader(db, cache_dir=args.cache_dir, cache_ttl_days=30)
    batches = [
        all_permnos[i : i + args.batch_size]
        for i in range(0, len(all_permnos), args.batch_size)
    ]
    logger.info(
        "Fetching CRSP DSF in %d batch(es) of up to %d PERMNOs …",
        len(batches), args.batch_size,
    )

    total_securities = 0
    total_rows = 0
    for i, batch in enumerate(batches, 1):
        batch_map = {p: ticker_map.get(p, f"PERMNO_{p}") for p in batch}
        price_data = loader.load_universe(batch, batch_map, buf_start, end)
        n_sec = len(price_data)
        n_rows = sum(len(df) for df in price_data.values())
        total_securities += n_sec
        total_rows += n_rows
        logger.info(
            "  Batch %d/%d — %d securities, %d rows cached.",
            i, len(batches), n_sec, n_rows,
        )

    logger.info(
        "\nDone. %d securities | %d total rows | cache at %s/prices/",
        total_securities, total_rows, args.cache_dir,
    )


if __name__ == "__main__":
    main()
