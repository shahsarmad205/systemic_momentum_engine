"""
WRDS Cache Rebuild — Phase 7
=============================
Pre-warms the WRDS Parquet cache for the full backtest universe.

Run this ONCE after completing the WRDS migration to populate:
  data/cache/wrds/prices/     — CRSP daily price panels (crsp.dsf)
  data/cache/wrds/universe/   — S&P 500 membership panel
  data/cache/wrds/fundamentals/ — Compustat quarterly fundamentals

Usage:
    # Step 1: delete stale Yahoo cache (irreversible — ensure WRDS works first)
    # rm -rf data/cache/*.parquet

    # Step 2: run this script to pre-warm WRDS cache
    WRDS_USERNAME=your_username python scripts/rebuild_wrds_cache.py

    # Optional flags
    WRDS_USERNAME=your_username python scripts/rebuild_wrds_cache.py \\
        --start 2007-01-01 \\
        --end 2023-12-31 \\
        --delete-yahoo          # actually delete Yahoo parquet files
        --skip-fundamentals     # skip Compustat (faster re-run)
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rebuild_wrds_cache")


def parse_args():
    p = argparse.ArgumentParser(description="WRDS cache pre-warmer")
    p.add_argument("--start", default="2007-01-01", help="Backtest start date")
    p.add_argument("--end", default="2023-12-31", help="Backtest end date")
    p.add_argument(
        "--delete-yahoo",
        action="store_true",
        help="Delete data/cache/*.parquet (Yahoo cache) after WRDS cache is built. "
             "Irreversible — only do this once WRDS data is confirmed good.",
    )
    p.add_argument(
        "--skip-fundamentals",
        action="store_true",
        help="Skip Compustat fundamental pre-warming (price data only).",
    )
    p.add_argument(
        "--min-price", type=float, default=10.0,
        help="Minimum price filter for universe construction (default: $10).",
    )
    p.add_argument(
        "--min-dollar-vol", type=float, default=1e8,
        help="Minimum dollar volume filter (default: $100M).",
    )
    return p.parse_args()


def main():
    args = parse_args()
    wrds_user = os.environ.get("WRDS_USERNAME")
    if not wrds_user:
        logger.error("WRDS_USERNAME environment variable not set. Aborting.")
        sys.exit(1)

    # ── Connect ───────────────────────────────────────────────────────────────
    from utils.wrds_universe import WRDSUniverse, build_backtest_universe, connect_wrds
    from utils.wrds_loader import WRDSLoader

    logger.info("Connecting to WRDS as %s …", wrds_user)
    db = connect_wrds(wrds_user)
    logger.info("Connected.")

    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)

    # Buffer start by 400 days for feature warm-up (same as production)
    buf_start = start - pd.Timedelta(days=400)

    # ── Step 1: Universe membership panel ────────────────────────────────────
    logger.info("Step 1/4 — Loading S&P 500 membership panel (%s → %s) …", args.start, args.end)
    universe = WRDSUniverse(db, cache_ttl_days=30)
    panel = universe.get_sp500_panel(args.start, args.end)
    all_permnos = sorted(panel["permno"].unique().tolist())
    logger.info("  %d unique PERMNOs in membership panel.", len(all_permnos))

    # ── Step 2: Investable universe at start date ─────────────────────────────
    logger.info(
        "Step 2/4 — Building investable universe at %s "
        "(price≥$%.0f, dvol≥$%.0fM) …",
        args.start, args.min_price, args.min_dollar_vol / 1e6,
    )
    investable = build_backtest_universe(
        db,
        date=args.start,
        min_price=args.min_price,
        min_dollar_vol=args.min_dollar_vol,
    )
    logger.info("  %d / %d PERMNOs pass liquidity + delist filter.", len(investable), len(all_permnos))

    # ── Step 3: Pre-warm price data (CRSP DSF) ───────────────────────────────
    logger.info(
        "Step 3/4 — Pre-warming CRSP price cache for %d PERMNOs (%s → %s) …",
        len(all_permnos), buf_start.date(), end.date(),
    )
    loader = WRDSLoader(db, cache_ttl_days=30)
    ticker_map = universe.permno_to_ticker_map(all_permnos, args.start)

    # Load in batches of 200 to avoid very large SQL IN clauses
    batch_size = 200
    batches = [
        all_permnos[i : i + batch_size]
        for i in range(0, len(all_permnos), batch_size)
    ]
    logger.info("  Loading %d batch(es) of up to %d PERMNOs …", len(batches), batch_size)
    total_loaded = 0
    for i, batch in enumerate(batches, 1):
        batch_map = {p: ticker_map.get(p, f"PERMNO_{p}") for p in batch}
        price_data = loader.load_universe(batch, batch_map, buf_start, end)
        total_loaded += len(price_data)
        logger.info("  Batch %d/%d: %d securities cached.", i, len(batches), len(price_data))

    logger.info("  Price cache complete — %d securities total.", total_loaded)

    # ── Step 4: Pre-warm Compustat fundamentals ───────────────────────────────
    if not args.skip_fundamentals:
        logger.info(
            "Step 4/4 — Pre-warming Compustat fundamentals for %d PERMNOs …",
            len(investable),
        )
        from features.wrds_fundamental_builder import fetch_fundamental_features

        dates = pd.bdate_range(start=buf_start, end=end)
        failed = 0
        for idx, permno in enumerate(investable, 1):
            try:
                fetch_fundamental_features(db, permno, pd.DatetimeIndex(dates))
            except Exception as exc:
                logger.debug("  Fundamentals failed for permno=%s: %s", permno, exc)
                failed += 1
            if idx % 50 == 0:
                logger.info("  Fundamentals: %d/%d done (%d failed) …", idx, len(investable), failed)
        logger.info("  Fundamentals complete — %d failed out of %d.", failed, len(investable))
    else:
        logger.info("Step 4/4 — Skipping fundamentals (--skip-fundamentals).")

    # ── Step 5: Delete Yahoo cache (optional, last) ───────────────────────────
    if args.delete_yahoo:
        yahoo_cache = Path("data/cache")
        parquets = list(yahoo_cache.glob("*.parquet"))
        logger.warning(
            "Deleting %d Yahoo Parquet files from %s …", len(parquets), yahoo_cache
        )
        for f in parquets:
            f.unlink()
        logger.info("  Yahoo cache deleted.")
    else:
        logger.info(
            "Yahoo cache NOT deleted (omit --delete-yahoo to keep). "
            "Delete manually with: rm -rf data/cache/*.parquet"
        )

    logger.info(
        "\nCache rebuild complete.\n"
        "  Universe PERMNOs : %d (all members)\n"
        "  Investable at %s  : %d\n"
        "  Price data        : %d securities\n"
        "  Cache location    : data/cache/wrds/",
        len(all_permnos), args.start, len(investable), total_loaded,
    )


if __name__ == "__main__":
    main()
