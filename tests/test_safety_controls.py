import json
from pathlib import Path

import pandas as pd
import pytest

from brokers.execution_engine import ExecutionEngine


class _FakeBroker:
    def __init__(self) -> None:
        self.place_calls: list[tuple[str, str, float]] = []
        self.close_calls: list[str] = []

    def get_account(self) -> dict[str, float]:
        return {"equity": 100_000.0, "cash": 100_000.0, "buying_power": 100_000.0, "portfolio_value": 100_000.0}

    def get_positions(self) -> pd.DataFrame:
        return pd.DataFrame([], columns=["ticker", "qty", "market_value"])

    def is_market_open(self) -> bool:
        return True

    def place_order(self, *, ticker: str, side: str, notional: float, **_: object) -> dict:
        self.place_calls.append((ticker, side, float(notional)))
        return {"success": True, "ticker": ticker, "side": side, "order_id": "oid", "status": "filled"}

    def close_position(self, ticker: str, **_: object) -> dict:
        self.close_calls.append(str(ticker))
        return {"success": True, "ticker": ticker, "side": "sell", "order_id": "oid", "status": "filled"}


def _write_cfg(tmp_path: Path) -> Path:
    p = tmp_path / "cfg.yaml"
    p.write_text(
        "\n".join(
            [
                "backtest:",
                "  max_positions: 2",
                "signals:",
                "  signal_confidence_multiplier: 0.0",
                "risk:",
                "  max_position_pct_of_equity: 0.5",
                "live:",
                "  trading_enabled: true",
                "  trading_halt_env: TRADING_HALTED",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    return p


def test_trading_halt_blocks_broker_calls(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TRADING_HALTED", "1")

    cfg = _write_cfg(tmp_path)
    broker = _FakeBroker()
    engine = ExecutionEngine(broker, config_path=str(cfg))

    signals = pd.DataFrame([{"ticker": "AAPL", "score": 1.0}, {"ticker": "MSFT", "score": 0.5}])
    out = engine.execute(signals, dry_run=False, extra_execution_log={"as_of": "2026-03-26"})

    assert out["live_trading_allowed"] is False
    assert out["trading_halt_reason"]
    assert broker.place_calls == []
    assert broker.close_calls == []


def test_idempotency_skips_duplicate_intents(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TRADING_HALTED", raising=False)

    # Pre-seed an existing intent for AAPL buy on the same as_of.
    intents = tmp_path / "output" / "live" / "order_intents.jsonl"
    intents.parent.mkdir(parents=True, exist_ok=True)
    intents.write_text(
        json.dumps(
            {
                "as_of": "2026-03-26",
                "intent_id": "2026-03-26|AAPL|buy",
                "ticker": "AAPL",
                "side": "buy",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    cfg = _write_cfg(tmp_path)
    broker = _FakeBroker()
    engine = ExecutionEngine(broker, config_path=str(cfg))

    signals = pd.DataFrame([{"ticker": "AAPL", "score": 1.0}, {"ticker": "MSFT", "score": 0.5}])
    engine.execute(signals, dry_run=False, extra_execution_log={"as_of": "2026-03-26"})

    # AAPL should be skipped; MSFT should be ordered.
    tickers = [t for (t, _, _) in broker.place_calls]
    assert "AAPL" not in tickers
    assert "MSFT" in tickers

