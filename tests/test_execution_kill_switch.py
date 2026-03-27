from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import yaml

from brokers.execution_engine import ExecutionEngine


class _StubBroker:
    def __init__(self) -> None:
        self.close_calls: list[str] = []
        self.place_calls: list[tuple[str, float]] = []

    def get_account(self) -> dict[str, float]:
        return {"equity": 100_000.0, "cash": 50_000.0, "buying_power": 50_000.0}

    def get_positions(self) -> pd.DataFrame:
        return pd.DataFrame(
            [{"ticker": "ZZZBAD", "qty": 1.0, "market_value": 100.0}],
        )

    def is_market_open(self) -> bool:
        return False

    def close_position(
        self,
        ticker: str,
        *,
        wait_for_fill: bool = False,
        signal_price: float | None = None,
    ) -> dict:
        self.close_calls.append(ticker)
        return {"success": True, "ticker": ticker, "order_id": "x"}

    def place_order(
        self,
        ticker: str,
        side: str,
        notional: float,
        wait_for_fill: bool = False,
        signal_price: float | None = None,
    ) -> dict:
        self.place_calls.append((ticker, float(notional)))
        return {"success": True, "ticker": ticker, "order_id": "y"}


def test_execute_respects_trading_halt(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "output" / "live").mkdir(parents=True)

    cfg = {
        "tickers": ["AAA", "BBB"],
        "backtest": {"max_positions": 8, "cache_dir": "data/cache/ohlcv"},
        "risk": {"max_position_pct_of_equity": 0.12},
        "live": {"trading_enabled": True, "trading_halt_env": "TRADING_HALT_TEST"},
        "slippage_tracking": {"enabled": False},
    }
    cfg_path = tmp_path / "backtest_config.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg), encoding="utf-8")

    signals = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "score": [2.0, 1.5],
        }
    )

    broker = _StubBroker()
    eng = ExecutionEngine(broker, config_path=str(cfg_path))

    import os

    os.environ["TRADING_HALT_TEST"] = "true"
    try:
        out = eng.execute(signals, dry_run=False)
    finally:
        del os.environ["TRADING_HALT_TEST"]

    assert out.get("live_trading_allowed") is False
    assert out.get("trading_halt_reason")
    assert broker.close_calls == []
    assert broker.place_calls == []

    logf = tmp_path / "output" / "live" / "execution_log.jsonl"
    assert logf.is_file()
    line = logf.read_text(encoding="utf-8").strip().splitlines()[-1]
    last = json.loads(line)
    assert last.get("live_trading_allowed") is False
