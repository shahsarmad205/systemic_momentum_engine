"""
Append-only live trade rows for slippage / cost calibration (CSV).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

TRADES_CSV_COLUMNS: list[str] = [
    "timestamp",
    "ticker",
    "side",
    "notional",
    "signal_price",
    "signal_price_source",
    "order_id",
    "status",
    "dry_run",
    "filled_qty",
    "filled_avg_price",
    "slippage_bps",
    "filled_at",
]


def append_trade_row(path: Path, record: dict[str, Any]) -> None:
    """Append one row; create file with header if missing."""
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {c: record.get(c) for c in TRADES_CSV_COLUMNS}
    df = pd.DataFrame([row])
    write_header = not path.is_file() or path.stat().st_size == 0
    df.to_csv(path, mode="a", index=False, header=write_header)


def adverse_slippage_bps(side: str, signal_price: float | None, fill_price: float | None) -> float | None:
    """
    Adverse execution cost in basis points (positive = worse than signal reference).
    Buy: paid more than signal. Sell: received less than signal.
    """
    if signal_price is None or fill_price is None:
        return None
    sp = float(signal_price)
    fp = float(fill_price)
    if sp <= 0 or fp != fp or sp != sp:
        return None
    s = str(side).strip().lower()
    if s == "buy":
        return (fp - sp) / sp * 10_000.0
    if s == "sell":
        return (sp - fp) / sp * 10_000.0
    return None
