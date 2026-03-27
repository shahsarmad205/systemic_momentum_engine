#!/usr/bin/env python3
"""
Recompute average daily volume (ADV in shares) for all config tickers from OHLCV parquet,
write ``output/adv_cache.csv`` (path from ``risk_factors.liquidity``).

Run from ``trend_signal_engine`` root::

    python scripts/refresh_adv_cache.py

Schedule monthly via cron if ``refresh_cache_on_run`` is false in config.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent


def main() -> int:
    parser = argparse.ArgumentParser(description="Refresh ADV cache CSV from OHLCV parquets.")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "backtest_config.yaml",
        help="Path to backtest_config.yaml",
    )
    args = parser.parse_args()
    cfg_path = args.config
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 1

    with open(cfg_path, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    bt = cfg.get("backtest", cfg)
    rf = cfg.get("risk_factors") or {}
    liq = rf.get("liquidity") or {}
    lookback = int(liq.get("adv_lookback_days", 20))
    adv_rel = Path(str(liq.get("adv_cache_path", "output/adv_cache.csv")))
    adv_path = adv_rel if adv_rel.is_absolute() else ROOT / adv_rel

    cache_dir = Path(str(bt.get("cache_dir", "data/cache/ohlcv")))
    if not cache_dir.is_absolute():
        cache_dir = ROOT / cache_dir

    tickers = cfg.get("tickers") or []
    if not tickers:
        print("No tickers in config.", file=sys.stderr)
        return 1

    sys.path.insert(0, str(ROOT))
    from utils.adv_cache import mean_volume_from_ohlcv_parquet  # noqa: PLC0415

    rows: list[dict[str, float | str]] = []
    missing: list[str] = []
    for raw in tickers:
        sym = str(raw).strip().upper()
        if not sym:
            continue
        adv = mean_volume_from_ohlcv_parquet(sym, cache_dir, lookback)
        if adv is not None:
            rows.append({"ticker": sym, "adv": adv})
        else:
            missing.append(sym)

    if not rows:
        print("No ADV values computed (check cache_dir and OHLCV files).", file=sys.stderr)
        return 1

    adv_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).sort_values("ticker").to_csv(adv_path, index=False)
    print(f"Wrote {len(rows)} tickers to {adv_path}")
    if missing:
        print(f"Skipped (no OHLCV): {len(missing)} — {', '.join(missing[:20])}{'…' if len(missing) > 20 else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
