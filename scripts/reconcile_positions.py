#!/usr/bin/env python3
"""
Compare live Alpaca positions to the latest internal snapshot in ``output/portfolio/paper_positions.csv``.

Exit code 0 if ticker sets match (long only; ignores zero-share rows). Exit 1 on drift or error.
Run from ``trend_signal_engine/`` after configuring Alpaca (see config/alpaca_config.example.yaml).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _tickers_from_paper_csv(path: Path) -> set[str]:
    if not path.is_file():
        return set()
    df = pd.read_csv(path)
    if df.empty or "ticker" not in df.columns:
        return set()
    if "date" in df.columns:
        df = df.copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        dmax = df["date"].max()
        if pd.notna(dmax):
            df = df[df["date"] == dmax]
    tickers = (
        df["ticker"]
        .astype(str)
        .str.strip()
        .str.upper()
        .loc[lambda s: s.ne("") & s.ne("NAN")]
    )
    if "shares" in df.columns:
        sh = pd.to_numeric(df["shares"], errors="coerce").fillna(0.0)
        tickers = tickers.loc[sh > 0]
    return set(tickers.tolist())


def main() -> int:
    parser = argparse.ArgumentParser(description="Reconcile Alpaca positions vs paper_positions.csv")
    parser.add_argument(
        "--paper-csv",
        type=Path,
        default=_ROOT / "output" / "portfolio" / "paper_positions.csv",
        help="Internal positions snapshot CSV",
    )
    args = parser.parse_args()

    internal = _tickers_from_paper_csv(args.paper_csv.resolve())
    try:
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker()
        pos = broker.get_positions()
    except Exception as exc:  # noqa: BLE001
        print(f"[reconcile] broker error: {exc}", file=sys.stderr)
        return 1

    if pos is None or pos.empty:
        broker_set: set[str] = set()
    else:
        t = pos["ticker"].astype(str).str.strip().str.upper()
        if "qty" in pos.columns:
            q = pd.to_numeric(pos["qty"], errors="coerce").fillna(0.0)
            broker_set = set(t.loc[q.abs() > 1e-9].tolist())
        else:
            broker_set = set(t.tolist())

    only_broker = sorted(broker_set - internal)
    only_internal = sorted(internal - broker_set)
    if only_broker or only_internal:
        print("[reconcile] DRIFT detected")
        print(f"  Alpaca only: {only_broker}")
        print(f"  Internal only: {only_internal}")
        return 1

    print(f"[reconcile] OK — {len(broker_set)} positions match ({args.paper_csv.name}).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
