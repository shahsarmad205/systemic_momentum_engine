from __future__ import annotations

from run_retrain_baseline import build_oos_validation_windows


def test_build_oos_single_window():
    w = build_oos_validation_windows("2025-12-31", holdout_calendar_days=10, n_windows=1)
    assert len(w) == 1
    assert w[0][0] == "2025-12-21"
    assert w[0][1] == "2025-12-31"


def test_build_oos_three_windows_monotonic():
    w = build_oos_validation_windows("2025-12-31", holdout_calendar_days=90, n_windows=3)
    assert len(w) == 3
    for start, end in w:
        assert start <= end
