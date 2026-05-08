from risk.drawdown_overlay import compute_drawdown_overlay


def test_drawdown_overlay_disabled_is_noop() -> None:
    state = compute_drawdown_overlay(
        current_equity=90.0,
        peak_equity=100.0,
        overlay_cfg={"enabled": False},
    )

    assert state.drawdown == 0.10
    assert state.gross_multiplier == 1.0
    assert not state.halt_new_risk
    assert not state.flatten_all


def test_drawdown_overlay_scales_and_triggers_controls() -> None:
    state = compute_drawdown_overlay(
        current_equity=86.0,
        peak_equity=100.0,
        overlay_cfg={
            "enabled": True,
            "start_pct": 0.05,
            "max_pct": 0.15,
            "min_gross_multiplier": 0.4,
            "halt_new_risk_pct": 0.12,
            "flatten_all_pct": 0.18,
        },
    )

    assert round(state.drawdown, 4) == 0.14
    assert 0.4 <= state.gross_multiplier < 0.5
    assert state.halt_new_risk
    assert not state.flatten_all
