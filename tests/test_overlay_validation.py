from __future__ import annotations

import pandas as pd

from utils.vol_sizing import apply_vol_kill_switch


def test_volatility_spike_kill_switch_reduces_exposure() -> None:
    positions = pd.Series([10000.0, 10000.0, 10000.0, 10000.0])
    realized_vol = pd.Series([0.15, 0.18, 0.30, 0.35])
    out = apply_vol_kill_switch(positions, realized_vol, threshold_annual=0.25, cut_factor=0.5)
    assert out.iloc[0] == 10000.0
    assert out.iloc[1] == 10000.0
    assert out.iloc[2] == 5000.0
    assert out.iloc[3] == 5000.0
