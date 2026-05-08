"""
WRDS Universe Prefetch
======================
One-shot CLI that downloads the full CRSP price panel for all universe
tickers and writes the master parquet files that eliminate per-window WRDS
queries during walk-forward backtests.

Usage
-----
Run ONCE before any walk-forward backtest:

    python -m utils.wrds_prefetch

With explicit date range (overrides backtest_config.yaml):

    python -m utils.wrds_prefetch --start 2006-01-01 --end 2023-12-31

What it writes (under data/cache/wrds/prices/):
    crsp_universe_panel_{start}_{end}.parquet   — all PERMNOs × full history
    crsp_universe_panel_{start}_{end}.meta.json — permno list + coverage metadata
    crsp_delist_master.parquet                  — delisting returns, all PERMNOs
    crsp_delist_master.meta.json                — permno list metadata

After this runs, every WRDSLoader.load_universe() call for any walk-forward
window slices from these local files — zero WRDS round-trips.
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# ── project root on path ──────────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import yaml
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wrds_prefetch")

_CONFIG_PATH = _ROOT / "backtest_config.yaml"
# Feature warm-up buffer: momentum features need ~400 days of history before
# the first training window, so the prefetch start is pushed back by this much.
_WARMUP_BUFFER_DAYS = 400


def _load_config() -> dict:
    if not _CONFIG_PATH.exists():
        return {}
    with open(_CONFIG_PATH) as f:
        return yaml.safe_load(f) or {}


def _resolve_date_range(cfg: dict, args: argparse.Namespace) -> tuple[str, str]:
    """
    Resolve prefetch date range.
    Priority: CLI flags > backtest_config.yaml > hardcoded safe default.
    The start date is pushed back by _WARMUP_BUFFER_DAYS to cover feature warm-up.
    """
    if args.start and args.end:
        return args.start, args.end

    # Read training windows from config to derive full span
    windows: list[dict] = cfg.get("walk_forward", {}).get("windows", [])
    if windows:
        try:
            all_starts = [pd.Timestamp(w["train_start"]) for w in windows if "train_start" in w]
            all_ends = [pd.Timestamp(w["test_end"]) for w in windows if "test_end" in w]
            if all_starts and all_ends:
                earliest = min(all_starts) - pd.Timedelta(days=_WARMUP_BUFFER_DAYS)
                latest = max(all_ends)
                return earliest.strftime("%Y-%m-%d"), latest.strftime("%Y-%m-%d")
        except Exception:
            pass

    # Hardcoded safe range covering any standard 2008–2022 walk-forward
    logger.warning("Could not derive date range from config — using default 2006-01-01 → 2023-12-31")
    return "2006-01-01", "2023-12-31"


def _resolve_universe(cfg: dict, args: argparse.Namespace) -> list[str] | None:
    """Return explicit ticker list from CLI or config, or None to use WRDSUniverse."""
    if args.tickers:
        return [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    universe_cfg = cfg.get("universe", {})
    tickers = universe_cfg.get("tickers") or universe_cfg.get("symbols")
    if tickers and isinstance(tickers, list):
        return [str(t).upper() for t in tickers]
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prefetch full CRSP universe panel from WRDS (run once before backtest).",
    )
    parser.add_argument("--start", default=None, help="Start date YYYY-MM-DD (overrides config)")
    parser.add_argument("--end", default=None, help="End date YYYY-MM-DD (overrides config)")
    parser.add_argument("--tickers", default=None, help="Comma-separated tickers (overrides config)")
    parser.add_argument("--cache-dir", default="data/cache/wrds", help="Cache root directory")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if master panel already exists",
    )
    args = parser.parse_args()

    cfg = _load_config()
    start_str, end_str = _resolve_date_range(cfg, args)
    logger.info("Prefetch date range: %s → %s", start_str, end_str)

    # ── Connect to WRDS ───────────────────────────────────────────────────────
    from utils.wrds_universe import connect_wrds
    username = os.environ.get("WRDS_USERNAME") or cfg.get("data", {}).get("wrds_username")
    if not username:
        logger.error("WRDS_USERNAME not set. Export it or add data.wrds_username to config.")
        sys.exit(1)

    logger.info("Connecting to WRDS as %s…", username)
    db = connect_wrds(username)

    # ── Resolve tickers → PERMNOs ─────────────────────────────────────────────
    explicit_tickers = _resolve_universe(cfg, args)
    if explicit_tickers:
        logger.info("Universe from config/CLI: %d tickers", len(explicit_tickers))
        from utils.wrds_data import resolve_ticker_to_permno
        ticker_to_permno = resolve_ticker_to_permno(
            explicit_tickers,
            as_of_date=end_str,
            username=username,
            cache_dir=str(Path(args.cache_dir) / ".."),
        )
        permnos = sorted(set(ticker_to_permno.values()))
    else:
        # Fall back to WRDSUniverse dynamic resolution
        logger.info("No explicit ticker list — using WRDSUniverse for %s", end_str)
        from utils.wrds_universe import WRDSUniverse
        universe = WRDSUniverse(db, cache_dir=args.cache_dir)
        universe_df = universe.get_sp500_at_date(end_str)
        permnos = sorted(int(p) for p in universe_df["permno"].dropna().unique())

    if not permnos:
        logger.error("No PERMNOs resolved — cannot prefetch.")
        sys.exit(1)

    logger.info("Resolved %d PERMNOs for prefetch", len(permnos))

    # ── Force: delete existing master panel files ─────────────────────────────
    if args.force:
        cache_prices = Path(args.cache_dir) / "prices"
        for f in list(cache_prices.glob("crsp_universe_panel_*.parquet")) + \
                 list(cache_prices.glob("crsp_universe_panel_*.meta.json")) + \
                 list(cache_prices.glob("crsp_delist_master.*")):
            f.unlink(missing_ok=True)
            logger.info("Removed stale cache: %s", f.name)

    # ── Prefetch ──────────────────────────────────────────────────────────────
    from utils.wrds_loader import WRDSLoader
    loader = WRDSLoader(db, cache_dir=args.cache_dir, cache_ttl_days=0)
    panel_path = loader.prefetch_universe_panel(
        permnos=permnos,
        start_date=start_str,
        end_date=end_str,
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    size_mb = panel_path.stat().st_size / 1_048_576
    logger.info("=" * 60)
    logger.info("Prefetch complete.")
    logger.info("  DSF panel : %s  (%.1f MB)", panel_path.name, size_mb)
    delist_path = panel_path.parent / "crsp_delist_master.parquet"
    if delist_path.exists():
        delist_mb = delist_path.stat().st_size / 1_048_576
        logger.info("  Delist    : crsp_delist_master.parquet  (%.1f MB)", delist_mb)
    logger.info("All walk-forward windows will now read from disk — zero WRDS calls.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
