from __future__ import annotations

import pandas as pd
import pytest

from utils.live_trades import TRADES_CSV_COLUMNS, adverse_slippage_bps, append_trade_row


def test_adverse_slippage_buy():
    assert adverse_slippage_bps("buy", 100.0, 100.5) == pytest.approx(50.0)


def test_adverse_slippage_sell_received_less_positive():
    assert adverse_slippage_bps("sell", 100.0, 99.5) == pytest.approx(50.0)


def test_adverse_slippage_none_inputs():
    assert adverse_slippage_bps("buy", None, 100.0) is None
    assert adverse_slippage_bps("buy", 100.0, None) is None


def test_append_trade_row_creates_with_header(tmp_path):
    p = tmp_path / "t.csv"
    row = {c: None for c in TRADES_CSV_COLUMNS}
    row.update({"timestamp": "t0", "ticker": "AAA", "side": "buy", "dry_run": False})
    append_trade_row(p, row)
    df = pd.read_csv(p)
    assert list(df.columns) == TRADES_CSV_COLUMNS
    assert len(df) == 1
