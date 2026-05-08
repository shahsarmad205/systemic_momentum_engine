from __future__ import annotations

import numpy as np
import pandas as pd

from agents.weight_learning_agent import feature_builder as fb
import features.wrds_panel_engine as wrds_panel_engine


def _price_frame(index: pd.DatetimeIndex) -> pd.DataFrame:
    base = np.linspace(100.0, 130.0, len(index))
    return pd.DataFrame(
        {
            "Open": base,
            "High": base * 1.01,
            "Low": base * 0.99,
            "Close": base,
            "Volume": np.linspace(1_000_000, 1_500_000, len(index)),
        },
        index=index,
    )


def test_build_feature_matrix_wrds_uses_preloaded_panel(monkeypatch):
    captured = {}

    def fake_batched_builder(tickers, **kwargs):
        captured["tickers"] = list(tickers)
        captured["kwargs"] = dict(kwargs)
        return pd.DataFrame(
            {
                "date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
                "ticker": ["AAPL", "AAPL"],
                "forward_return": [0.01, -0.01],
                "vix_zscore": [0.0, 0.0],
                "yield_curve_slope": [0.1, 0.1],
            }
        )

    import features.wrds_panel_engine as wrds_panel_engine

    monkeypatch.setattr(wrds_panel_engine, "build_wrds_feature_matrix_batched", fake_batched_builder)

    df = fb.build_feature_matrix(
        ["AAPL"],
        start_date="2020-01-02",
        end_date="2020-01-10",
        data_provider="wrds",
        wrds_username="tester",
    )

    assert not df.empty
    assert captured["tickers"] == ["AAPL"]
    assert captured["kwargs"]["wrds_username"] == "tester"
    assert "vix_zscore" in df.columns
    assert "yield_curve_slope" in df.columns


def test_attach_cross_sectional_zscore_suffix_block_is_blockwise_and_correct():
    df = pd.DataFrame(
        {
            "date": pd.to_datetime(
                ["2021-01-04", "2021-01-04", "2021-01-05", "2021-01-05"]
            ),
            "ticker": ["A", "B", "A", "B"],
            "feature_a": [1.0, 3.0, 2.0, 6.0],
            "feature_b": [10.0, 10.0, 5.0, 5.0],
            "feature_skip": [7.0, 8.0, 9.0, 10.0],
        }
    )

    out = fb._attach_cross_sectional_zscore_suffix_block(
        df,
        exclude_columns=frozenset({"feature_skip"}),
    )

    assert "feature_a_cs_z" in out.columns
    assert "feature_b_cs_z" in out.columns
    assert "feature_skip_cs_z" not in out.columns
    assert np.allclose(out.loc[[0, 1], "feature_a_cs_z"].to_numpy(dtype=float), [-1.0, 1.0])
    assert np.allclose(out.loc[[2, 3], "feature_a_cs_z"].to_numpy(dtype=float), [-1.0, 1.0])
    assert np.allclose(out["feature_b_cs_z"].to_numpy(dtype=float), 0.0)


def test_compute_capm_block_aligns_market_series_to_panel_rows():
    dates = pd.bdate_range("2021-01-01", periods=40)
    panel = pd.DataFrame(
        {
            "date": list(dates) + list(dates),
            "ticker": ["A"] * len(dates) + ["B"] * len(dates),
            "daily_return": np.linspace(-0.01, 0.02, len(dates)).tolist()
            + np.linspace(0.02, -0.01, len(dates)).tolist(),
            "ret_20d": np.linspace(-0.05, 0.05, len(dates)).tolist()
            + np.linspace(0.05, -0.05, len(dates)).tolist(),
        }
    )
    market_ret = pd.Series(np.linspace(-0.005, 0.01, len(dates)), index=dates, dtype=float)

    capm = wrds_panel_engine._compute_capm_block(panel, market_ret)

    assert len(capm) == len(panel)
    assert capm.index.equals(panel.index)
    assert np.isfinite(capm["rolling_corr_market_20"].to_numpy(dtype=float)).all()
    assert np.isfinite(capm["capm_beta"].to_numpy(dtype=float)).all()
    assert np.isfinite(capm["idio_momentum_20d"].to_numpy(dtype=float)).all()
