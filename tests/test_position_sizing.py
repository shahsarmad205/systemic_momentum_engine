import math

from backtesting.position_sizing import (
    PositionSizingParams,
    equal_size,
    vol_scaled_size,
    kelly_size,
    risk_parity_size,
)


def test_equal_size_basic():
    equity = 100_000.0
    max_positions = 10
    params = PositionSizingParams(max_position_pct_of_equity=0.25)

    size = equal_size(equity, max_positions, position_scale=1.0, params=params)
    # Equal size should be equity / max_positions = 10_000, below cap.
    assert math.isclose(size, 10_000.0, rel_tol=1e-6)


def test_vol_scaled_respects_risk_budget_and_cap():
    equity = 100_000.0
    max_positions = 10
    params = PositionSizingParams(
        max_position_pct_of_equity=0.25,
        target_risk_fraction=0.10,  # 10% of equity risk budget
    )

    # With vol = 20%, risk_budget = 10_000, naive size_vol = 10_000 / 0.2 = 50_000
    # Equal-weight = 10_000, capped at 3× equal (30_000) and then at 25% of equity (25_000).
    size = vol_scaled_size(
        equity=equity,
        max_positions=max_positions,
        position_scale=1.0,
        stock_vol_annual=0.20,
        params=params,
    )
    assert math.isclose(size, 25_000.0, rel_tol=1e-6)


def test_risk_parity_matches_vol_scaled_single_asset():
    equity = 100_000.0
    max_positions = 5
    params = PositionSizingParams(
        max_position_pct_of_equity=0.25,
        target_risk_fraction=0.05,
    )
    vol = 0.25

    vs = vol_scaled_size(equity, max_positions, 1.0, vol, params)
    rp = risk_parity_size(equity, max_positions, 1.0, vol, params)

    assert math.isclose(vs, rp, rel_tol=1e-6)


def test_kelly_size_scales_with_full_kelly():
    equity = 100_000.0
    max_positions = 10
    params = PositionSizingParams(
        max_position_pct_of_equity=0.50,
        kelly_fraction=1.0,
        kelly_win_rate=0.60,
        kelly_avg_win_return=0.02,
        kelly_avg_loss_return=0.01,
    )

    base = equity / max_positions
    size = kelly_size(equity, max_positions, 1.0, params)

    p = params.kelly_win_rate
    q = 1.0 - p
    b = params.kelly_avg_win_return / params.kelly_avg_loss_return
    full_kelly = max(0.0, min((p * b - q) / b, 1.0))
    expected_size = equity * full_kelly * params.kelly_fraction
    expected_size = max(expected_size, base * 0.5)
    cap = equity * params.max_position_pct_of_equity
    expected_size = min(expected_size, cap)
    assert math.isclose(size, expected_size, rel_tol=1e-6)

