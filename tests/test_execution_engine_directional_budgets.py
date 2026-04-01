from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from brokers.execution_engine import ExecutionEngine


class _NoopBroker:
    pass


def _write_cfg(tmp_path: Path, cfg: dict) -> Path:
    p = tmp_path / "cfg.yaml"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return p


def _base_cfg() -> dict:
    return {
        "backtest": {
            "max_positions": 4,
            "max_longs": 2,
            "max_shorts": 2,
        },
        "signals": {
            "signal_confidence_multiplier": 0.0,
        },
        "risk": {
            "max_position_pct_of_equity": 0.5,
            "max_gross_exposure": 1.5,
            "max_net_exposure": 1.5,
            "max_short_single_name": 0.2,
        },
        "execution": {
            "enable_shorts": True,
            "long_only": False,
        },
    }


def _weights_by_ticker(df: pd.DataFrame) -> dict[str, float]:
    ordered = df.sort_values("ticker")["target_weight"].astype(float).tolist()
    tickers = df.sort_values("ticker")["ticker"].astype(str).tolist()
    return dict(zip(tickers, ordered))


def test_directional_budgets_disabled_preserves_behavior(tmp_path: Path) -> None:
    cfg_a = _base_cfg()
    cfg_b = _base_cfg()
    cfg_b["risk"]["directional_budgets"] = {
        "enabled": False,
        "long_budget": 0.05,
        "short_budget": 0.05,
    }

    path_a = _write_cfg(tmp_path / "a", cfg_a)
    path_b = _write_cfg(tmp_path / "b", cfg_b)

    signals = pd.DataFrame(
        [
            {"ticker": "AAPL", "score": 2.0},
            {"ticker": "MSFT", "score": 1.0},
            {"ticker": "XOM", "score": -1.5},
            {"ticker": "CVX", "score": -1.0},
        ]
    )
    account = {"equity": 100000.0}

    eng_a = ExecutionEngine(_NoopBroker(), config_path=str(path_a))
    eng_b = ExecutionEngine(_NoopBroker(), config_path=str(path_b))

    out_a = eng_a.compute_target_portfolio(signals, account, verbose=False)
    out_b = eng_b.compute_target_portfolio(signals, account, verbose=False)

    assert _weights_by_ticker(out_a) == pytest.approx(_weights_by_ticker(out_b), abs=1e-12)


def test_directional_budgets_disabled_ignores_malformed_budget_values(tmp_path: Path) -> None:
    cfg_a = _base_cfg()
    cfg_b = _base_cfg()
    cfg_b["risk"]["directional_budgets"] = {
        "enabled": False,
        "long_budget": "not-a-number",
        "short_budget": "also-bad",
    }

    path_a = _write_cfg(tmp_path / "a", cfg_a)
    path_b = _write_cfg(tmp_path / "b", cfg_b)

    signals = pd.DataFrame(
        [
            {"ticker": "AAPL", "score": 2.0},
            {"ticker": "MSFT", "score": 1.0},
            {"ticker": "XOM", "score": -1.5},
            {"ticker": "CVX", "score": -1.0},
        ]
    )
    account = {"equity": 100000.0}

    eng_a = ExecutionEngine(_NoopBroker(), config_path=str(path_a))
    eng_b = ExecutionEngine(_NoopBroker(), config_path=str(path_b))

    out_a = eng_a.compute_target_portfolio(signals, account, verbose=False)
    out_b = eng_b.compute_target_portfolio(signals, account, verbose=False)

    assert _weights_by_ticker(out_a) == pytest.approx(_weights_by_ticker(out_b), abs=1e-12)


def test_directional_budgets_cap_long_side(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["execution"] = {"enable_shorts": False, "long_only": True}
    cfg["backtest"]["max_positions"] = 3
    cfg["backtest"]["max_longs"] = 3
    cfg["backtest"]["max_shorts"] = 0
    cfg["risk"]["directional_budgets"] = {
        "enabled": True,
        "long_budget": 0.60,
        "short_budget": 1.00,
    }

    path = _write_cfg(tmp_path, cfg)
    eng = ExecutionEngine(_NoopBroker(), config_path=str(path))

    signals = pd.DataFrame(
        [
            {"ticker": "AAPL", "score": 1.0},
            {"ticker": "MSFT", "score": 0.9},
            {"ticker": "NVDA", "score": 0.8},
        ]
    )
    out = eng.compute_target_portfolio(signals, {"equity": 100000.0}, verbose=False)

    long_sum = float(out[out["target_weight"] > 0]["target_weight"].sum())
    assert long_sum == pytest.approx(0.60, abs=1e-12)
    assert float(out["target_weight"].max()) <= 0.5


def test_directional_budgets_cap_short_side_with_single_name_limit(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["risk"]["directional_budgets"] = {
        "enabled": True,
        "long_budget": 1.0,
        "short_budget": 0.25,
    }

    path = _write_cfg(tmp_path, cfg)
    eng = ExecutionEngine(_NoopBroker(), config_path=str(path))

    signals = pd.DataFrame(
        [
            {"ticker": "AAPL", "score": 1.0},
            {"ticker": "TSLA", "score": -1.2},
            {"ticker": "NVDA", "score": -1.0},
        ]
    )
    out = eng.compute_target_portfolio(signals, {"equity": 100000.0}, verbose=False)

    shorts = out[out["target_weight"] < 0]["target_weight"].astype(float)
    short_sum_abs = float(-shorts.sum())
    assert short_sum_abs <= 0.25 + 1e-12
    assert float(shorts.abs().max()) <= 0.2 + 1e-12


def test_directional_budgets_still_respect_net_cap(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["backtest"]["max_positions"] = 3
    cfg["backtest"]["max_longs"] = 2
    cfg["backtest"]["max_shorts"] = 1
    cfg["risk"]["max_net_exposure"] = 0.10
    cfg["risk"]["directional_budgets"] = {
        "enabled": True,
        "long_budget": 0.60,
        "short_budget": 0.20,
    }

    path = _write_cfg(tmp_path, cfg)
    eng = ExecutionEngine(_NoopBroker(), config_path=str(path))

    signals = pd.DataFrame(
        [
            {"ticker": "AAPL", "score": 1.2},
            {"ticker": "MSFT", "score": 1.1},
            {"ticker": "XOM", "score": -1.0},
        ]
    )
    out = eng.compute_target_portfolio(signals, {"equity": 100000.0}, verbose=False)

    long_sum = float(out[out["target_weight"] > 0]["target_weight"].sum())
    short_sum_abs = float(-out[out["target_weight"] < 0]["target_weight"].sum())
    net = float(out["target_weight"].sum())

    assert abs(net) <= 0.10 + 1e-12
    assert long_sum <= 0.60 + 1e-12
    assert short_sum_abs <= 0.20 + 1e-12


def test_directional_budgets_still_respect_gross_cap(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["risk"]["max_gross_exposure"] = 0.50
    cfg["risk"]["directional_budgets"] = {
        "enabled": True,
        "long_budget": 0.60,
        "short_budget": 0.60,
    }

    path = _write_cfg(tmp_path, cfg)
    eng = ExecutionEngine(_NoopBroker(), config_path=str(path))

    signals = pd.DataFrame(
        [
            {"ticker": "AAPL", "score": 1.2},
            {"ticker": "MSFT", "score": 1.1},
            {"ticker": "XOM", "score": -1.0},
            {"ticker": "CVX", "score": -0.9},
        ]
    )
    out = eng.compute_target_portfolio(signals, {"equity": 100000.0}, verbose=False)

    gross = float(out["target_weight"].abs().sum())
    long_sum = float(out[out["target_weight"] > 0]["target_weight"].sum())
    short_sum_abs = float(-out[out["target_weight"] < 0]["target_weight"].sum())

    assert gross <= 0.50 + 1e-12
    assert long_sum <= 0.60 + 1e-12
    assert short_sum_abs <= 0.60 + 1e-12


def test_directional_budgets_apply_in_greedy_factor_limits_path(tmp_path: Path) -> None:
    cfg = _base_cfg()
    cfg["risk"]["directional_budgets"] = {
        "enabled": True,
        "long_budget": 0.30,
        "short_budget": 0.20,
    }
    # Any non-empty risk_factors block enables greedy path in execute().
    cfg["risk_factors"] = {"enabled": True}

    path = _write_cfg(tmp_path, cfg)
    eng = ExecutionEngine(_NoopBroker(), config_path=str(path))

    signals = pd.DataFrame(
        [
            {"ticker": "AAPL", "score": 1.2},
            {"ticker": "MSFT", "score": 1.1},
            {"ticker": "XOM", "score": -1.0},
            {"ticker": "CVX", "score": -0.9},
        ]
    )
    account = {"equity": 100000.0, "cash": 100000.0, "buying_power": 100000.0}
    target = eng.compute_target_portfolio(signals, account, verbose=False)

    filtered, _, _ = eng._apply_factor_limits(
        target,
        current_positions=pd.DataFrame([], columns=["ticker", "qty", "market_value"]),
        account=account,
        sizing_mult=1.0,
        verbose=False,
    )

    long_sum = float(filtered[filtered["target_weight"] > 0]["target_weight"].sum())
    short_sum_abs = float(-filtered[filtered["target_weight"] < 0]["target_weight"].sum())

    assert long_sum <= 0.30 + 1e-12
    assert short_sum_abs <= 0.20 + 1e-12
