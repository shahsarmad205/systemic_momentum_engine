#!/usr/bin/env python3
"""
Poll Alpaca for orders listed in ``trades_pending.csv``; backfill fills and append to ``trades.csv``.

Run after the open (e.g. cron 9:45 ET) so overnight day orders show as filled::

    cd trend_signal_engine && python scripts/fetch_fills.py

Logs: stdout; redirect in cron to ``output/live/fetch_fills.log`` if desired.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description="Backfill Alpaca fills into trades CSV.")
    parser.add_argument("--config", type=Path, default=ROOT / "backtest_config.yaml")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    log = logging.getLogger("fetch_fills")

    if not args.config.is_file():
        log.error("Config not found: %s", args.config)
        return 1

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh) or {}

    st = cfg.get("slippage_tracking") or {}
    if not isinstance(st, dict) or not st.get("enabled", False):
        log.info("slippage_tracking.enabled is false; nothing to do.")
        return 0

    pend_rel = st.get("pending_trades_file") or st.get("output_path") or "output/live/trades_pending.csv"
    trades_rel = st.get("trades_file", "output/live/trades.csv")
    pending_path = ROOT / str(pend_rel)
    trades_path = ROOT / str(trades_rel)

    if not pending_path.is_file():
        log.info("No pending file at %s — exit 0.", pending_path)
        return 0

    from brokers.alpaca_broker import AlpacaBroker
    from utils.live_trades import TRADES_CSV_COLUMNS, adverse_slippage_bps

    try:
        broker = AlpacaBroker()
    except Exception as exc:  # noqa: BLE001
        log.exception("Alpaca broker: %s", exc)
        return 1

    pending = pd.read_csv(pending_path)
    if pending.empty:
        log.info("Pending trades file is empty.")
        return 0

    completed: list[dict[str, Any]] = []
    still: list[dict[str, Any]] = []

    for _, row in pending.iterrows():
        r = row.to_dict()
        oid_raw = r.get("order_id")
        if oid_raw is None or (isinstance(oid_raw, float) and pd.isna(oid_raw)):
            still.append(r)
            continue
        oid = str(oid_raw).strip()
        if not oid:
            still.append(r)
            continue

        try:
            o = broker.api.get_order(oid)
        except Exception as exc:  # noqa: BLE001
            log.warning("get_order %s failed: %s — leaving in pending", oid, exc)
            still.append(r)
            continue

        st_ord = str(o.status or "").lower()
        r["status"] = st_ord

        if st_ord != "filled":
            still.append(r)
            continue

        try:
            r["filled_qty"] = float(o.filled_qty) if o.filled_qty not in (None, "") else None
        except (TypeError, ValueError):
            r["filled_qty"] = None
        try:
            r["filled_avg_price"] = float(o.filled_avg_price) if o.filled_avg_price not in (None, "") else None
        except (TypeError, ValueError):
            r["filled_avg_price"] = None

        sig = r.get("signal_price")
        try:
            sig_f = float(sig) if sig is not None and not (isinstance(sig, float) and pd.isna(sig)) else None
        except (TypeError, ValueError):
            sig_f = None

        side = str(r.get("side", "buy")).lower()
        r["slippage_bps"] = adverse_slippage_bps(side, sig_f, r.get("filled_avg_price"))
        try:
            r["filled_at"] = str(getattr(o, "filled_at", None) or "")
        except Exception:
            r["filled_at"] = None

        completed.append(r)
        log.info("Filled %s %s order %s — slip=%s bps", r.get("ticker"), side, oid[:8], r.get("slippage_bps"))

    if completed:
        new_df = pd.DataFrame(completed).reindex(columns=TRADES_CSV_COLUMNS)
        if trades_path.is_file():
            old = pd.read_csv(trades_path)
            combined = pd.concat([old, new_df], ignore_index=True)
            if "order_id" in combined.columns:
                combined = combined.drop_duplicates(subset=["order_id"], keep="last")
        else:
            combined = new_df
        trades_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(trades_path, index=False)
        log.info("Appended %d row(s) → %s", len(completed), trades_path)

    if still:
        pd.DataFrame(still).to_csv(pending_path, index=False)
        log.info("%d row(s) remain pending → %s", len(still), pending_path)
    else:
        pd.DataFrame(columns=TRADES_CSV_COLUMNS).to_csv(pending_path, index=False)
        log.info("Cleared pending file %s", pending_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
