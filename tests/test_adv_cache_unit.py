from __future__ import annotations

from pathlib import Path

import pandas as pd

from utils.adv_cache import latest_close_from_ohlcv_parquet, mean_volume_from_ohlcv_parquet


def _write_ohlcv(tmp: Path, ticker: str = "TEST") -> Path:
    p = tmp / f"{ticker}.parquet"
    idx = pd.date_range("2024-01-01", periods=25, freq="B", tz=None)
    df = pd.DataFrame(
        {
            "Open": range(100, 125),
            "High": range(101, 126),
            "Low": range(99, 124),
            "Close": range(100, 125),
            "Volume": [1_000_000 + i * 1000 for i in range(25)],
        },
        index=idx,
    )
    df.to_parquet(p)
    return p


def test_mean_volume_tail(tmp_path: Path):
    _write_ohlcv(tmp_path)
    m = mean_volume_from_ohlcv_parquet("TEST", tmp_path, lookback=5)
    assert m is not None
    assert m > 1e6


def test_latest_close(tmp_path: Path):
    _write_ohlcv(tmp_path)
    c = latest_close_from_ohlcv_parquet("TEST", tmp_path)
    assert c == 124.0
