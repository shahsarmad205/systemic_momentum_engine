import warnings

import numpy as np
import pandas as pd
import pytest

import model_selection.factor_subsumption as factor_subsumption_mod
from model_selection.factor_subsumption import (
    build_factor_mimicking_returns,
    factor_subsumption_diagnostics,
)


def _scored_panel(days: int = 80, names: int = 20) -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=days)
    rows = []
    for d_i, dt in enumerate(dates):
        for i in range(names):
            size_signal = i - (names - 1) / 2.0
            rows.append(
                {
                    "date": dt,
                    "ticker": f"T{i}",
                    "daily_return": 0.0005 + 0.00015 * size_signal + 0.00002 * np.sin(d_i),
                    "forward_return": 0.002 + 0.0005 * size_signal,
                    "market_cap": 1_000_000_000 + i * 100_000_000,
                    "sector": "Tech" if i < names // 2 else "Industrials",
                    "ret_20d": size_signal,
                    "quality_score": -size_signal * 0.1,
                    "rolling_vol_20": 0.02 + i * 0.0005,
                }
            )
    return pd.DataFrame(rows)


def test_factor_subsumption_returns_alpha_and_loading_diagnostics() -> None:
    scored = _scored_panel()
    fac = build_factor_mimicking_returns(scored, horizon_days=1)
    assert "size" in fac.columns

    model_returns = fac["size"] * 0.8 + 0.0005
    stats = factor_subsumption_diagnostics(model_returns, scored, horizon_days=1, min_obs=40)

    assert stats["subsumption_n_obs"] >= 40
    assert stats["subsumption_alpha_ann"] > 0.0
    assert np.isfinite(stats["subsumption_max_abs_loading"])
    assert np.isfinite(stats["subsumption_r2"])


def test_factor_subsumption_handles_collinear_factor_block_without_runtime_warning(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    scored = _scored_panel(days=90, names=15)
    dates = pd.bdate_range("2021-01-04", periods=90)
    base = np.linspace(-0.01, 0.01, len(dates))
    fac = pd.DataFrame(
        {
            "market": base,
            "size": base,
            "momentum": base * 0.999999,
            "quality": -base,
        },
        index=dates,
    )

    monkeypatch.setattr(factor_subsumption_mod, "build_factor_mimicking_returns", lambda *_args, **_kwargs: fac)
    model_returns = pd.Series(base * 0.8 + 0.0002, index=dates)
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        stats = factor_subsumption_diagnostics(model_returns, scored, horizon_days=1, min_obs=40)

    assert stats["subsumption_n_obs"] >= 40
    assert "subsumption_alpha_ann" in stats
