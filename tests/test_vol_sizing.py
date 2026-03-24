from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.vol_sizing import (
    compute_realized_vol_annualized,
    compute_vol_target_scaling_factor,
)


def test_constant_returns_scaling_stable() -> None:
    # Constant daily return => rolling std ~0 => scale should clamp to max cap.
    r = pd.Series([0.001] * 120, dtype=float)
    vol = compute_realized_vol_annualized(r, window=20).iloc[-1]
    scale = compute_vol_target_scaling_factor(vol, target_vol=0.15, min_vol_floor=0.05, max_scale_cap=3.0)
    assert scale == pytest.approx(3.0, abs=1e-12)


def test_high_volatility_decreases_scaling() -> None:
    low_vol = 0.10
    high_vol = 0.40
    s_low = compute_vol_target_scaling_factor(low_vol, target_vol=0.15, min_vol_floor=0.05, max_scale_cap=3.0)
    s_high = compute_vol_target_scaling_factor(high_vol, target_vol=0.15, min_vol_floor=0.05, max_scale_cap=3.0)
    assert s_high < s_low
    assert s_high == pytest.approx(0.375, rel=1e-6)


def test_near_zero_volatility_no_explosion() -> None:
    # raw target/vol would explode; floor+cap should prevent it.
    scale = compute_vol_target_scaling_factor(1e-8, target_vol=0.15, min_vol_floor=0.05, max_scale_cap=2.5)
    assert np.isfinite(scale)
    assert scale <= 2.5
    assert scale == pytest.approx(2.5, abs=1e-12)

