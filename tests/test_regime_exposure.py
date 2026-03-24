from __future__ import annotations

import pytest
import pandas as pd
from types import SimpleNamespace

from strategy.regime_exposure import (
    CANONICAL_REGIME_EXPOSURE,
    exposure_path_for_regimes,
    resolve_regime_position_scale,
    validate_regime_keys,
)
from strategy.candidates import build_ranked_candidates
from strategy.cross_sectional import build_cross_sectional_candidates


def test_each_regime_applies_correct_multiplier() -> None:
    assert resolve_regime_position_scale("Bull") == pytest.approx(1.0)
    assert resolve_regime_position_scale("Sideways") == pytest.approx(0.8)
    assert resolve_regime_position_scale("Bear") == pytest.approx(0.6)
    assert resolve_regime_position_scale("Crisis") == pytest.approx(0.3)


def test_invalid_regime_defaults_safely_or_raises() -> None:
    assert resolve_regime_position_scale("UnknownRegime", strict=False) == pytest.approx(0.8)
    with pytest.raises(ValueError):
        resolve_regime_position_scale("UnknownRegime", strict=True)


def test_validate_regime_keys_reports_missing_and_unexpected() -> None:
    missing, unexpected = validate_regime_keys(
        {
            "Bull": {"position_scale": 1.0},
            "Bear": {"position_scale": 0.6},
            "Alien": {"position_scale": 0.1},
        }
    )
    assert missing == {"Sideways", "Crisis"}
    assert unexpected == {"Alien"}


def test_exposure_decreases_monotonically_bull_to_crisis() -> None:
    path = exposure_path_for_regimes(["Bull", "Sideways", "Bear", "Crisis"])
    assert path == [1.0, 0.8, 0.6, 0.3]
    assert all(path[i] >= path[i + 1] for i in range(len(path) - 1))


def test_rapid_regime_switching_stress() -> None:
    seq = ["Bull", "Crisis", "Bull", "Bear", "Crisis", "Sideways"] * 20
    out = exposure_path_for_regimes(seq)
    assert len(out) == len(seq)
    assert min(out) >= min(CANONICAL_REGIME_EXPOSURE.values())
    assert max(out) <= max(CANONICAL_REGIME_EXPOSURE.values())


def test_long_crisis_period_stress() -> None:
    seq = ["Crisis"] * 250
    out = exposure_path_for_regimes(seq)
    assert len(out) == 250
    assert all(v == pytest.approx(0.3) for v in out)


def _dummy_config() -> SimpleNamespace:
    return SimpleNamespace(
        regime_adjustments={
            "Bull": {"score_mult": 1.0, "position_scale": 1.0},
            "Sideways": {"score_mult": 1.0, "position_scale": 0.8},
            "Bear": {"score_mult": 1.0, "position_scale": 0.6},
            "Crisis": {"score_mult": 1.0, "position_scale": 0.3},
        },
        vol_scaling_enabled=False,
        vol_scaling_target=0.15,
        signal_flip_threshold=0.15,
        min_signal_strength=0.0,
        signal_threshold_std_multiplier=None,
        signal_score_std=None,
        enable_shorts=False,
        execution_delay_days=0,
        dynamic_holding_enabled=False,
        holding_period_days=2,
        holding_period_by_strength=[],
        holding_period_by_signal={},
        bear_max_holding_days=3,
        top_longs=2,
        top_shorts=2,
        market_neutral=False,
    )


def test_candidates_builder_uses_regime_resolved_scale() -> None:
    cfg = _dummy_config()
    d = pd.Timestamp("2024-01-02")
    trading_days = [d, d + pd.Timedelta(days=1), d + pd.Timedelta(days=2), d + pd.Timedelta(days=3)]
    row = pd.Series({"adjusted_score": 0.9, "signal": "Bullish", "confidence": "High"})
    out = build_ranked_candidates(d, [("AAPL", row)], cfg, trading_days, 0, "Crisis")
    assert len(out) == 1
    assert out[0]["position_scale"] == pytest.approx(0.3)


def test_cross_sectional_builder_uses_regime_resolved_scale() -> None:
    cfg = _dummy_config()
    d = pd.Timestamp("2024-01-02")
    trading_days = [d, d + pd.Timedelta(days=1), d + pd.Timedelta(days=2), d + pd.Timedelta(days=3)]
    row1 = pd.Series({"adjusted_score": 0.9, "confidence": "High"})
    row2 = pd.Series({"adjusted_score": 0.4, "confidence": "High"})
    out, _ = build_cross_sectional_candidates(
        d,
        [("AAPL", row1), ("MSFT", row2)],
        cfg,
        trading_days,
        0,
        "Bear",
    )
    assert out
    assert all(e["position_scale"] == pytest.approx(0.6) for e in out)

