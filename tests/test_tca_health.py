from __future__ import annotations

import pandas as pd

from utils.tca_health import evaluate_tca_health


def test_tca_health_passes_under_thresholds() -> None:
    fills = pd.DataFrame({"slippage_bps": [4.0, 6.0, 8.0, 9.0, 7.0, 5.0]})
    out = evaluate_tca_health(
        fills,
        rolling_trades=6,
        max_avg_slippage_bps=10.0,
        max_p95_slippage_bps=15.0,
        min_fills=5,
        fail_on_no_data=False,
    )
    assert out["status"] == "PASS"
    assert out["failures"] == []


def test_tca_health_fails_when_avg_exceeds_threshold() -> None:
    fills = pd.DataFrame({"slippage_bps": [12.0, 13.0, 11.0, 10.5, 12.5]})
    out = evaluate_tca_health(
        fills,
        rolling_trades=5,
        max_avg_slippage_bps=10.0,
        max_p95_slippage_bps=20.0,
        min_fills=5,
        fail_on_no_data=False,
    )
    assert out["status"] == "FAIL"
    codes = {f.get("code") for f in out["failures"]}
    assert "AVG_SLIPPAGE_EXCEEDED" in codes


def test_tca_health_no_data_can_pass_when_configured() -> None:
    fills = pd.DataFrame(columns=["slippage_bps"])
    out = evaluate_tca_health(
        fills,
        rolling_trades=10,
        max_avg_slippage_bps=10.0,
        max_p95_slippage_bps=None,
        min_fills=5,
        fail_on_no_data=False,
    )
    assert out["status"] == "PASS"
    assert out["metrics"]["n_fills_total"] == 0


def test_tca_health_no_data_fails_when_fail_closed() -> None:
    fills = pd.DataFrame(columns=["slippage_bps"])
    out = evaluate_tca_health(
        fills,
        rolling_trades=10,
        max_avg_slippage_bps=10.0,
        max_p95_slippage_bps=None,
        min_fills=5,
        fail_on_no_data=True,
    )
    assert out["status"] == "FAIL"
    codes = {f.get("code") for f in out["failures"]}
    assert "NO_SLIPPAGE_DATA" in codes
