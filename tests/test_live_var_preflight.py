import pandas as pd
import pytest

from run_live_trading import _preflight_var_check


def test_var_preflight_includes_signed_exposures(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, dict[str, float]] = {}

    def fake_load_aligned_returns(
        tickers: list[str],
        cache_dir: object,
        lookback: int,
    ) -> tuple[pd.DataFrame, None]:
        idx = pd.date_range("2026-01-01", periods=5, freq="D")
        cols = sorted(set(str(t).upper() for t in tickers))
        data = {c: [0.001, -0.002, 0.003, -0.001, 0.0] for c in cols}
        return pd.DataFrame(data, index=idx), None

    def fake_portfolio_var(
        tickers: list[str],
        weights: dict[str, float],
        returns_df: pd.DataFrame,
        confidence: float,
        method: str,
    ) -> tuple[float, None]:
        key = "target" if "TSLA" in weights else "current"
        captured[key] = dict(weights)
        return 0.01, None

    monkeypatch.setattr("utils.returns.load_aligned_returns", fake_load_aligned_returns)
    monkeypatch.setattr("risk.var.portfolio_var", fake_portfolio_var)

    cfg = {
        "backtest": {"cache_dir": "data/cache/ohlcv"},
        "risk": {
            "var_check": {
                "enabled": True,
                "confidence": 0.95,
                "lookback_days": 20,
                "max_var_pct": 0.05,
                "method": "historical",
                "check_target": True,
                "strict_coverage": True,
                "min_weight_coverage": 0.001,
            }
        },
    }
    account = {"equity": 100000.0}
    current_positions = pd.DataFrame(
        [
            {"ticker": "AAPL", "market_value": 20000.0},
            {"ticker": "MSFT", "market_value": -10000.0},
        ]
    )
    target = pd.DataFrame(
        [
            {"ticker": "AAPL", "target_value": 15000.0},
            {"ticker": "TSLA", "target_value": -5000.0},
        ]
    )

    ok, out = _preflight_var_check(
        config=cfg,
        account=account,
        current_positions=current_positions,
        target=target,
    )

    assert ok is True
    assert out["var_preflight_passed"] is True
    assert captured["current"]["MSFT"] < 0
    assert captured["target"]["TSLA"] < 0
