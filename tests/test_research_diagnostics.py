from __future__ import annotations

import numpy as np
import pandas as pd

from model_selection.research_diagnostics import compute_full_diagnostics, long_only_evaluation


def test_full_diagnostics_populates_robust_halflife_from_available_signal_decay() -> None:
    dates = pd.bdate_range("2022-01-03", periods=12)
    tickers = [f"T{i}" for i in range(30)]
    rows: list[dict[str, object]] = []
    base_scores = np.linspace(-1.0, 1.0, len(tickers))
    for day_idx, dt in enumerate(dates):
        # Slowly rotate the ranking so both robust autocorrelation and decile
        # turnover have finite support.
        scores = np.roll(base_scores, day_idx % 3)
        for ticker, score in zip(tickers, scores):
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "score": float(score),
                    "target_return": float(score) * 0.001,
                    "target_return_net": float(score) * 0.0008,
                    "forward_return": float(score) * 0.001,
                    "daily_return": float(score) * 0.0002,
                }
            )
    oos = pd.DataFrame(rows)
    pnl = pd.DataFrame(
        {
            "date": dates,
            "daily_return": np.zeros(len(dates), dtype=float),
            "turnover": np.linspace(0.01, 0.03, len(dates)),
        }
    )

    out = compute_full_diagnostics(oos, pnl, {}, model_kind="regressor", horizon=10)

    assert np.isfinite(out["diag_robust_signal_halflife"])
    assert out["diag_robust_signal_halflife_source"] in {"rank_autocorrelation", "decile_turnover"}
    assert np.isfinite(out["diag_robust_cost_adjusted_ic"])


def test_long_only_diagnostic_prefers_deployable_forward_return() -> None:
    dates = pd.bdate_range("2022-03-01", periods=24)
    tickers = [f"T{i:02d}" for i in range(40)]
    rows: list[dict[str, object]] = []
    scores = np.linspace(-1.0, 1.0, len(tickers))
    for day_idx, dt in enumerate(dates):
        day_bump = (day_idx % 5) * 0.00001
        for ticker, score in zip(tickers, scores):
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "score": float(score),
                    "target_return": float(score) * 0.002 + day_bump,
                    "forward_return": -float(score) * 0.002 - day_bump,
                }
            )

    out = long_only_evaluation(pd.DataFrame(rows), target_horizon_days=10)

    assert out["diag_lo_sharpe"] < 0.0


def test_long_only_diagnostic_uses_executable_next_day_return_when_available() -> None:
    dates = pd.bdate_range("2022-04-01", periods=24)
    tickers = [f"T{i:02d}" for i in range(40)]
    rows: list[dict[str, object]] = []
    scores = np.linspace(-1.0, 1.0, len(tickers))
    for day_idx, dt in enumerate(dates):
        for ticker, score in zip(tickers, scores):
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "score": float(score),
                    "target_return": float(score) * 0.01,
                    "forward_return": float(score) * 0.01,
                    "daily_return": -float(score) * 0.002 + (day_idx % 3) * 0.00005,
                }
            )

    out = long_only_evaluation(pd.DataFrame(rows), target_horizon_days=10)

    assert out["diag_lo_sharpe"] < 0.0
