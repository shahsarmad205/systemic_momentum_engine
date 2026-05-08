#!/usr/bin/env python3
"""Daily WRDS integrity gate for PIT universe and CRSP delisting handling."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from utils.wrds_loader import WRDSLoader
from utils.wrds_universe import build_backtest_universe, connect_wrds

ROOT = Path(__file__).resolve().parents[1]


def _read_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _compare_snapshot(panel: dict[str, pd.DataFrame], snapshot_path: Path) -> list[str]:
    """Compare a small deterministic sample against a known-good snapshot JSON."""
    if not snapshot_path.is_file():
        return [f"known_good_snapshot_missing:{snapshot_path}"]
    snap = json.loads(snapshot_path.read_text(encoding="utf-8"))
    failures: list[str] = []
    for row in snap.get("sample", []):
        ticker = str(row["ticker"]).upper()
        df = panel.get(ticker)
        if df is None or df.empty:
            failures.append(f"snapshot_ticker_missing:{ticker}")
            continue
        last = df.iloc[-1]
        if "last_date" in row and pd.Timestamp(df.index[-1]) != pd.Timestamp(row["last_date"]):
            failures.append(f"snapshot_last_date_mismatch:{ticker}")
        if "last_ret" in row and abs(float(last["ret"]) - float(row["last_ret"])) > 1e-10:
            failures.append(f"snapshot_last_ret_mismatch:{ticker}")
        if "delisting_return_applied" in row and bool(last.get("delisting_return_applied", False)) != bool(row["delisting_return_applied"]):
            failures.append(f"snapshot_delist_flag_mismatch:{ticker}")
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description="Check WRDS PIT universe and delisting integrity")
    parser.add_argument("--config", default="backtest_config.yaml")
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    cfg_path = (ROOT / args.config).resolve()
    cfg = _read_config(cfg_path)
    data_cfg = cfg.get("data") or {}
    provider = str(data_cfg.get("provider") or "").lower()
    if provider != "wrds":
        raise SystemExit("WRDS integrity check requires data.provider: wrds")

    gov_cfg = ((cfg.get("governance") or {}).get("data_integrity") or {})
    out_dir = ROOT / str(gov_cfg.get("output_dir", "output/live/data_integrity"))
    out_dir.mkdir(parents=True, exist_ok=True)

    bt = cfg.get("backtest") or {}
    as_of = pd.Timestamp(gov_cfg.get("as_of_date") or bt.get("end_date") or pd.Timestamp.today()).normalize()
    start = pd.Timestamp(bt.get("start_date") or as_of - pd.Timedelta(days=370))
    sample_size = int(gov_cfg.get("sample_size", 5))
    cache_dir = str(data_cfg.get("cache_dir", "data/cache/wrds"))
    cache_ttl_days = int(data_cfg.get("cache_ttl_days", 1))

    failures: list[str] = []
    db = connect_wrds(os.environ.get("WRDS_USERNAME"))
    universe = build_backtest_universe(
        db,
        as_of,
        min_price=float(gov_cfg.get("min_price", 10.0)),
        min_dollar_vol=float(gov_cfg.get("min_dollar_vol", 100_000_000.0)),
        cache_dir=str(Path(cache_dir) / "universe"),
        cache_ttl_days=cache_ttl_days,
    )
    if not universe:
        failures.append("empty_pit_universe")

    sample_permnos = universe[:sample_size]
    ticker_map = {}
    if sample_permnos:
        from utils.wrds_universe import WRDSUniverse

        ticker_map = WRDSUniverse(db, cache_dir=cache_dir, cache_ttl_days=cache_ttl_days).permno_to_ticker_map(
            sample_permnos,
            as_of,
        )
    if sample_permnos and not ticker_map:
        failures.append("empty_permno_ticker_map")

    panel = WRDSLoader(db, cache_dir=cache_dir, cache_ttl_days=cache_ttl_days).load_universe(
        permnos=sample_permnos,
        ticker_map=ticker_map,
        start_date=start,
        end_date=as_of,
    )
    if sample_permnos and not panel:
        failures.append("empty_price_panel")

    for ticker, df in panel.items():
        missing = [c for c in ("ret", "Close", "Volume", "delisting_return_applied") if c not in df.columns]
        if missing:
            failures.append(f"{ticker}:missing_columns:{','.join(missing)}")
        if "ret" in df.columns and pd.to_numeric(df["ret"], errors="coerce").isna().all():
            failures.append(f"{ticker}:all_null_returns")

    snapshot_rel = str(gov_cfg.get("known_good_snapshot_path") or "").strip()
    if snapshot_rel:
        snapshot_path = Path(snapshot_rel)
        if not snapshot_path.is_absolute():
            snapshot_path = ROOT / snapshot_path
        failures.extend(_compare_snapshot(panel, snapshot_path))

    now = datetime.now(timezone.utc)
    payload = {
        "run_at_utc": now.isoformat(),
        "overall_status": "FAIL" if failures else "PASS",
        "as_of_date": as_of.strftime("%Y-%m-%d"),
        "universe_size": len(universe),
        "sample_permnos": sample_permnos,
        "tickers_checked": sorted(panel.keys()),
        "failures": failures,
    }
    latest = out_dir / "wrds_data_integrity_latest.json"
    latest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 2 if failures and args.strict else 0


if __name__ == "__main__":
    raise SystemExit(main())
