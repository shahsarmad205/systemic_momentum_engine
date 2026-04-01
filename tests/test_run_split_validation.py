from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import scripts.run_split_validation as rsv


def _write_cfg(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "model_selection:",
                "  split_rollout_gates:",
                "    min_short_pnl_total: 0.0",
                "    min_short_pnl_bear: 0.0",
                "    max_bull_short_damage_abs: 10.0",
                "    max_short_stoploss_rate_bull: 0.5",
                "    min_crisis_sharpe: -0.5",
                "    min_short_trades_bull: 1",
                "    min_crisis_days: 2",
                "    fail_on_insufficient_data: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _seed_inputs(tmp_path: Path, *, bull_short_pnl: float, bull_exit_reason: str = "time") -> None:
    trades = pd.DataFrame(
        [
            {"direction": "short", "pnl": 5.0, "regime": "Bear", "exit_reason": "time"},
            {"direction": "short", "pnl": bull_short_pnl, "regime": "Bull", "exit_reason": bull_exit_reason},
            {"direction": "long", "pnl": 3.0, "regime": "Bull", "exit_reason": "time"},
        ]
    )
    equity = pd.DataFrame(
        [
            {"date": "2026-01-01", "equity": 100000.0, "regime": "Crisis"},
            {"date": "2026-01-02", "equity": 100500.0, "regime": "Crisis"},
            {"date": "2026-01-03", "equity": 101000.0, "regime": "Crisis"},
            {"date": "2026-01-04", "equity": 101100.0, "regime": "Bull"},
        ]
    )
    out_bt = tmp_path / "output" / "backtests"
    out_bt.mkdir(parents=True, exist_ok=True)
    trades.to_csv(out_bt / "trades.csv", index=False)
    equity.to_csv(out_bt / "daily_equity.csv", index=False)


def _run(monkeypatch, tmp_path: Path, *extra: str) -> int:
    monkeypatch.setattr(rsv, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    argv = ["run_split_validation.py", "--config", "backtest_config.yaml", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return rsv.main()


def test_split_validation_strict_pass(monkeypatch, tmp_path: Path) -> None:
    _write_cfg(tmp_path / "backtest_config.yaml")
    _seed_inputs(tmp_path, bull_short_pnl=-5.0)

    rc = _run(monkeypatch, tmp_path, "--strict")
    assert rc == 0

    latest = tmp_path / "output" / "models" / "split_validation_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"


def test_split_validation_strict_fail(monkeypatch, tmp_path: Path) -> None:
    _write_cfg(tmp_path / "backtest_config.yaml")
    _seed_inputs(tmp_path, bull_short_pnl=-25.0, bull_exit_reason="stop_loss")

    rc = _run(monkeypatch, tmp_path, "--strict")
    assert rc == 2

    latest = tmp_path / "output" / "models" / "split_validation_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert "bull_short_damage_cap_breached" in payload["failures"]


def test_split_validation_normalizes_stop_loss_exit_reason(monkeypatch, tmp_path: Path) -> None:
    _write_cfg(tmp_path / "backtest_config.yaml")
    _seed_inputs(tmp_path, bull_short_pnl=-5.0, bull_exit_reason=" Stop_Loss ")

    rc = _run(monkeypatch, tmp_path, "--strict")
    assert rc == 2

    latest = tmp_path / "output" / "models" / "split_validation_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert "bull_short_stoploss_rate_too_high" in payload["failures"]
