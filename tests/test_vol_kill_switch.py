from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from utils.vol_sizing import apply_vol_kill_switch


def test_scalar_trigger_and_no_trigger() -> None:
    assert apply_vol_kill_switch(100.0, 0.30, threshold_annual=0.25, cut_factor=0.5) == pytest.approx(50.0)
    assert apply_vol_kill_switch(100.0, 0.20, threshold_annual=0.25, cut_factor=0.5) == pytest.approx(100.0)


def test_vectorized_series_behavior() -> None:
    idx = pd.date_range("2024-01-01", periods=4, freq="D")
    positions = pd.Series([10.0, 20.0, 30.0, 40.0], index=idx)
    vol = pd.Series([0.20, 0.30, 0.10, 0.40], index=idx)
    out = apply_vol_kill_switch(positions, vol, threshold_annual=0.25, cut_factor=0.5)
    exp = pd.Series([10.0, 10.0, 30.0, 20.0], index=idx)
    pd.testing.assert_series_equal(out, exp)


def test_no_nan_propagation_from_vol() -> None:
    positions = pd.Series([10.0, 20.0, 30.0])
    vol = pd.Series([0.30, np.nan, np.inf])
    out = apply_vol_kill_switch(positions, vol, threshold_annual=0.25, cut_factor=0.5)
    # first triggers, NaN/inf should not trigger
    assert out.iloc[0] == pytest.approx(5.0)
    assert out.iloc[1] == pytest.approx(20.0)
    assert out.iloc[2] == pytest.approx(30.0)
    assert np.isfinite(out.to_numpy()).all()


def test_numpy_array_vectorized() -> None:
    positions = np.array([1.0, 2.0, 3.0, 4.0])
    vol = np.array([0.26, 0.25, 0.10, 0.50])
    out = apply_vol_kill_switch(positions, vol, threshold_annual=0.25, cut_factor=0.5)
    assert np.allclose(out, np.array([0.5, 2.0, 3.0, 2.0]))

