import pandas as pd
import pytest

from risk.hard_limits import HardLimitConfig, evaluate_target_hard_limits


def _limits(**overrides: float | bool | None) -> HardLimitConfig:
    base: dict[str, float | bool | None] = {
        "enabled": True,
        "fail_closed": True,
        "max_gross_exposure": 1.5,
        "max_abs_net_exposure": 0.5,
        "max_single_name_abs": 0.12,
        "max_short_single_name_abs": 0.12,
        "max_sector_exposure": 0.30,
    }
    base.update(overrides)
    return HardLimitConfig(**base)


def test_hard_limits_pass_when_within_limits() -> None:
    target = pd.DataFrame(
        [
            {"ticker": "AAPL", "target_weight": 0.10},
            {"ticker": "MSFT", "target_weight": 0.08},
            {"ticker": "XOM", "target_weight": -0.08},
        ]
    )
    sector_map = {"AAPL": "Technology", "MSFT": "Technology", "XOM": "Energy"}

    out = evaluate_target_hard_limits(
        target,
        equity=100000.0,
        limits=_limits(),
        sector_mapping=sector_map,
    )

    assert out["status"] == "PASS"
    assert out["failures"] == []


def test_hard_limits_fail_for_gross_and_single_name_breach() -> None:
    target = pd.DataFrame(
        [
            {"ticker": "AAPL", "target_weight": 0.13},
            {"ticker": "MSFT", "target_weight": 0.10},
            {"ticker": "XOM", "target_weight": -0.10},
        ]
    )

    out = evaluate_target_hard_limits(
        target,
        equity=100000.0,
        limits=_limits(max_gross_exposure=0.25, max_single_name_abs=0.12),
        sector_mapping={"AAPL": "Technology", "MSFT": "Technology", "XOM": "Energy"},
    )

    codes = {f.get("code") for f in out["failures"]}
    assert out["status"] == "FAIL"
    assert "MAX_GROSS_EXCEEDED" in codes
    assert "MAX_SINGLE_NAME_EXCEEDED" in codes


def test_hard_limits_fail_closed_on_missing_sector_mapping() -> None:
    target = pd.DataFrame(
        [
            {"ticker": "AAPL", "target_weight": 0.10},
            {"ticker": "MSFT", "target_weight": 0.10},
        ]
    )

    out = evaluate_target_hard_limits(
        target,
        equity=100000.0,
        limits=_limits(max_sector_exposure=0.30),
        sector_mapping={"AAPL": "Technology"},
    )

    assert out["status"] == "FAIL"
    assert any(f.get("code") == "MISSING_SECTOR_MAPPING" for f in out["failures"])


def test_hard_limits_can_derive_weights_from_target_value() -> None:
    target = pd.DataFrame(
        [
            {"ticker": "AAPL", "target_value": 10000.0},
            {"ticker": "MSFT", "target_value": -5000.0},
        ]
    )

    out = evaluate_target_hard_limits(
        target,
        equity=100000.0,
        limits=_limits(max_sector_exposure=None),
        sector_mapping={},
    )

    assert out["status"] == "PASS"
    assert out["metrics"]["gross_exposure"] == pytest.approx(0.15, abs=1e-12)