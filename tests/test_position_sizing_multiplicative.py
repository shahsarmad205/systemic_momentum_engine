from __future__ import annotations

import math

import pytest

from backtesting.position_sizing import compose_position_size


def test_components_applied_multiplicatively() -> None:
    s = compose_position_size(
        equity=100_000,
        weight=0.10,
        vol_scaling=0.5,
        regime_scaling=0.8,
        max_single_position_pct=0.12,
    )
    # 100k * 0.10 * 0.5 * 0.8 = 4000
    assert s == pytest.approx(4000.0)


def test_max_cap_enforced_at_12pct() -> None:
    s = compose_position_size(
        equity=100_000,
        weight=1.0,
        vol_scaling=2.0,
        regime_scaling=1.0,
        max_single_position_pct=0.12,
    )
    assert s == pytest.approx(12_000.0)


def test_no_nan_or_overflow_from_extreme_inputs() -> None:
    s = compose_position_size(
        equity=100_000,
        weight=0.2,
        vol_scaling=1e9,   # extreme scale
        regime_scaling=1.0,
        max_single_position_pct=0.12,
    )
    assert math.isfinite(s)
    assert s <= 12_000.0


def test_sum_of_positions_not_exceeding_equity_with_normalized_weights() -> None:
    equity = 100_000
    weights = [0.40, 0.35, 0.25]  # sum=1
    sizes = [
        compose_position_size(
            equity=equity,
            weight=w,
            vol_scaling=1.0,
            regime_scaling=1.0,
            max_single_position_pct=1.0,  # disable single-name cap for this conservation check
        )
        for w in weights
    ]
    assert sum(sizes) <= equity + 1e-8


def test_zero_equity_returns_zero() -> None:
    assert compose_position_size(0.0, 0.2, 1.0, 1.0) == 0.0


def test_negative_weight_long_only_clamped_to_zero() -> None:
    assert compose_position_size(100_000, -0.2, 1.0, 1.0, long_only=True) == 0.0


def test_negative_weight_allowed_when_not_long_only() -> None:
    s = compose_position_size(100_000, -0.2, 1.0, 1.0, long_only=False, max_single_position_pct=1.0)
    assert s == pytest.approx(-20_000.0)


def test_min_threshold_skips_tiny_positions() -> None:
    s = compose_position_size(
        equity=100_000,
        weight=0.001,     # 0.1%
        vol_scaling=1.0,
        regime_scaling=1.0,
        max_single_position_pct=0.12,
        min_single_position_pct=0.01,  # 1%
    )
    assert s == 0.0

