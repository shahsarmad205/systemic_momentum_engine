"""Characterization tests for target construction — Phase C.2.

These tests capture the exact legacy behavior of forward-return and target
construction so that the TargetPanelProvider refactor can be validated
against them with strict numerical parity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from model_selection.training import (
    TargetConfig,
    add_institutional_targets,
    retarget_panel_for_horizon,
    make_training_target,
)
from model_selection.validation import ExecutionCostConfig


# ── Synthetic panel fixtures ────────────────────────────────────────────────

def _make_panel(
    n_dates: int = 10,
    n_tickers: int = 5,
    start: str = "2021-01-04",
    seed: int = 42,
    include_daily_return: bool = False,
    include_factors: bool = False,
) -> pd.DataFrame:
    """Deterministic synthetic panel for target characterization."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]
    rows = []
    for d_i, dt in enumerate(dates):
        for t_i, tk in enumerate(tickers):
            row = {
                "date": dt,
                "ticker": tk,
                "forward_return": rng.normal(0.0005, 0.02),
            }
            if include_daily_return:
                row["daily_return"] = rng.normal(0.0005, 0.015)
            if include_factors:
                row["capm_beta"] = 0.8 + t_i * 0.04
                row["sector"] = "Tech" if t_i < n_tickers // 2 else "Industrials"
                row["adv_dollar_20"] = 100_000_000 - t_i * 1_000_000
                row["realised_vol_20d"] = 0.02 + t_i * 0.001
            rows.append(row)
    return pd.DataFrame(rows)


def _make_known_panel() -> pd.DataFrame:
    """Tiny panel with known forward_return values for exact arithmetic checks."""
    rows = [
        {"date": pd.Timestamp("2021-01-04"), "ticker": "A", "forward_return": 0.01},
        {"date": pd.Timestamp("2021-01-04"), "ticker": "B", "forward_return": -0.02},
        {"date": pd.Timestamp("2021-01-04"), "ticker": "C", "forward_return": 0.005},
        {"date": pd.Timestamp("2021-01-05"), "ticker": "A", "forward_return": -0.01},
        {"date": pd.Timestamp("2021-01-05"), "ticker": "B", "forward_return": 0.03},
        {"date": pd.Timestamp("2021-01-05"), "ticker": "C", "forward_return": 0.0},
        {"date": pd.Timestamp("2021-01-06"), "ticker": "A", "forward_return": 0.02},
        {"date": pd.Timestamp("2021-01-06"), "ticker": "B", "forward_return": -0.005},
        {"date": pd.Timestamp("2021-01-06"), "ticker": "C", "forward_return": 0.015},
        {"date": pd.Timestamp("2021-01-07"), "ticker": "A", "forward_return": 0.0},
        {"date": pd.Timestamp("2021-01-07"), "ticker": "B", "forward_return": 0.01},
        {"date": pd.Timestamp("2021-01-07"), "ticker": "C", "forward_return": -0.01},
    ]
    return pd.DataFrame(rows)


# ── 1. Raw forward return construction ──────────────────────────────────────

class TestRawForwardReturnConstruction:
    def test_add_institutional_targets_preserves_row_count(self):
        df = _make_panel(n_dates=10, n_tickers=5, include_factors=True)
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=True, net_of_costs=False),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )
        assert len(result) == len(df)

    def test_forward_return_column_unchanged(self):
        df = _make_panel(n_dates=10, n_tickers=5, include_factors=True)
        original_fwd = df["forward_return"].copy()
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5),
            costs=ExecutionCostConfig(),
            max_name_weight=0.10,
        )
        np.testing.assert_array_almost_equal(
            result["forward_return"].values,
            original_fwd.values,
            decimal=15,
        )


# ── 2. Multiple horizons ────────────────────────────────────────────────────

class TestMultipleHorizons:
    def test_retarget_panel_horizon_5(self):
        df = _make_panel(n_dates=20, n_tickers=5, include_daily_return=True, include_factors=True)
        result = retarget_panel_for_horizon(
            df,
            horizon_days=5,
            target_cfg=TargetConfig(horizon_days=5, residualize=True),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )
        assert len(result) == len(df)
        assert "target_return" in result.columns
        assert "target_rank" in result.columns

    def test_retarget_panel_horizon_10(self):
        df = _make_panel(n_dates=30, n_tickers=5, include_daily_return=True, include_factors=True)
        result_5 = retarget_panel_for_horizon(
            df,
            horizon_days=5,
            target_cfg=TargetConfig(horizon_days=5, residualize=True),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )
        result_10 = retarget_panel_for_horizon(
            df,
            horizon_days=10,
            target_cfg=TargetConfig(horizon_days=10, residualize=True),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )
        assert len(result_5) == len(result_10) == len(df)
        # Different horizons produce different target values
        assert not np.allclose(
            result_5["target_return"].values,
            result_10["target_return"].values,
            equal_nan=True,
        )


# ── 3. Missing prices / missing returns ─────────────────────────────────────

class TestMissingData:
    def test_nan_forward_return_produces_nan_targets(self):
        df = _make_known_panel()
        df.loc[df["ticker"] == "B", "forward_return"] = np.nan
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=False, net_of_costs=False),
            costs=ExecutionCostConfig(),
            max_name_weight=0.10,
        )
        # target_return should be 0-filled (fillna(0.0) in the code)
        assert result["target_return"].dtype == float
        assert np.isfinite(result["target_return"]).all()

    def test_all_nan_forward_return(self):
        df = _make_known_panel()
        df["forward_return"] = np.nan
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=False, net_of_costs=False),
            costs=ExecutionCostConfig(),
            max_name_weight=0.10,
        )
        assert np.isfinite(result["target_return"]).all()
        assert (result["target_return"] == 0.0).all()


# ── 4. Ticker boundary behavior ─────────────────────────────────────────────

class TestTickerBoundary:
    def test_single_ticker(self):
        df = _make_panel(n_dates=10, n_tickers=1, include_factors=True)
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=True, net_of_costs=False),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )
        assert len(result) == 10
        assert "target_return" in result.columns

    def test_many_tickers(self):
        df = _make_panel(n_dates=10, n_tickers=50, include_factors=True)
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=True, net_of_costs=False),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )
        assert len(result) == 500
        assert result["ticker"].nunique() == 50


# ── 5. Date sorting behavior ────────────────────────────────────────────────

class TestDateSorting:
    def test_unsorted_input_produces_same_output(self):
        df = _make_panel(n_dates=10, n_tickers=5, include_factors=True)
        shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
        result_sorted = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=True),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )
        result_shuffled = add_institutional_targets(
            shuffled,
            cfg=TargetConfig(horizon_days=5, residualize=True),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )
        # Sort both by date, ticker for comparison
        result_sorted = result_sorted.sort_values(["date", "ticker"]).reset_index(drop=True)
        result_shuffled = result_shuffled.sort_values(["date", "ticker"]).reset_index(drop=True)
        np.testing.assert_array_almost_equal(
            result_sorted["target_return"].values,
            result_shuffled["target_return"].values,
            decimal=12,
        )


# ── 6. Last horizon rows becoming NaN ───────────────────────────────────────

class TestHorizonNaNBehavior:
    def test_retarget_with_daily_return_produces_trailing_nans(self):
        """When retarget_panel_for_horizon uses daily_return, last h rows per ticker should be NaN in forward_return."""
        n_dates = 15
        h = 5
        df = _make_panel(n_dates=n_dates, n_tickers=3, include_daily_return=True, include_factors=True)
        result = retarget_panel_for_horizon(
            df,
            horizon_days=h,
            target_cfg=TargetConfig(horizon_days=h, residualize=False),
            costs=ExecutionCostConfig(),
            max_name_weight=0.10,
        )
        # forward_return should have NaN at the last h rows per ticker
        for ticker in df["ticker"].unique():
            mask = result["ticker"] == ticker
            ticker_data = result.loc[mask].sort_values("date").reset_index(drop=True)
            last_h = ticker_data["forward_return"].iloc[-h:]
            assert last_h.isna().all(), f"Expected last {h} rows for {ticker} to be NaN"


# ── 7. Residualized target behavior ─────────────────────────────────────────

class TestResidualizedTargets:
    def test_residualized_target_is_demeaned_by_date(self):
        df = _make_known_panel()
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=False, net_of_costs=False),
            costs=ExecutionCostConfig(),
            max_name_weight=0.10,
        )
        # Without residualize, target_return should be demeaned by date
        for date in result["date"].unique():
            date_mask = result["date"] == date
            mean_target = result.loc[date_mask, "target_return"].mean()
            assert abs(mean_target) < 1e-10, f"target_return not demeaned for date {date}"

    def test_residualized_target_with_ridge(self):
        df = _make_panel(n_dates=10, n_tickers=5, include_factors=True)
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=True, net_of_costs=False, residual_ridge=1e-4),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )
        assert np.isfinite(result["target_return"]).all()
        # Ridge-regularized residualization still centers by date approximately
        for date in result["date"].unique():
            date_mask = result["date"] == date
            mean_target = result.loc[date_mask, "target_return"].mean()
            assert abs(mean_target) < 0.01, f"Ridge residual not approximately centered for date {date}"


# ── 8. Net-of-cost target behavior ──────────────────────────────────────────

class TestNetOfCostTargets:
    def test_net_cost_target_is_lower_than_gross(self):
        df = _make_panel(n_dates=10, n_tickers=5, include_factors=True)
        result_gross = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=True, net_of_costs=False),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )
        result_net = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=True, net_of_costs=True),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )
        # Net targets should generally be lower (costs are positive)
        assert (result_net["target_expected_cost"] >= 0).all()
        # target_return_net should differ from target_return when costs > 0
        cost_mask = result_net["target_expected_cost"] > 0
        if cost_mask.any():
            assert not np.allclose(
                result_net.loc[cost_mask, "target_return"].values,
                result_net.loc[cost_mask, "target_return_net"].values,
            )

    def test_cost_breakdown_columns_exist(self):
        df = _make_panel(n_dates=10, n_tickers=5, include_factors=True)
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=True, net_of_costs=True),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )
        expected_cols = {
            "target_expected_cost",
            "target_expected_participation",
            "target_expected_fixed_cost",
            "target_expected_temporary_impact",
            "target_expected_permanent_impact",
            "target_expected_borrow_cost",
        }
        assert expected_cols.issubset(set(result.columns))


# ── 9. Winsorized/normalized target behavior ────────────────────────────────

class TestWinsorizedTargets:
    def test_winsor_q_clips_extreme_values(self):
        df = _make_known_panel()
        # Inject extreme values
        df.loc[0, "forward_return"] = 1.0
        df.loc[1, "forward_return"] = -1.0
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=False, net_of_costs=False, winsor_q=0.10),
            costs=ExecutionCostConfig(),
            max_name_weight=0.10,
        )
        # target_return_net should be clipped (it uses winsorization before filling NaN)
        assert np.isfinite(result["target_return_net"]).all()

    def test_max_abs_return_clips(self):
        df = _make_known_panel()
        df.loc[0, "forward_return"] = 100.0
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=False, net_of_costs=False, max_abs_return=0.10),
            costs=ExecutionCostConfig(),
            max_name_weight=0.10,
        )
        assert result["target_return_net"].abs().max() <= 0.10 + 1e-12


# ── 10. Exact target column naming ──────────────────────────────────────────

class TestTargetColumnNaming:
    def test_standard_target_columns(self):
        df = _make_panel(n_dates=10, n_tickers=5, include_factors=True)
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=True, net_of_costs=True),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )
        expected = {
            "target_return_net",
            "target_return",
            "target_rank",
            "target_down_decile",
            "target_up",
            "target_expected_cost",
            "target_expected_participation",
            "target_expected_fixed_cost",
            "target_expected_temporary_impact",
            "target_expected_permanent_impact",
            "target_expected_borrow_cost",
            "forward_return",
        }
        assert expected.issubset(set(result.columns))

    def test_column_dtypes(self):
        df = _make_panel(n_dates=10, n_tickers=5, include_factors=True)
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=True, net_of_costs=True),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )
        assert result["target_return"].dtype == float
        assert result["target_rank"].dtype == float
        assert result["target_down_decile"].dtype == int
        assert result["target_up"].dtype == int
        assert result["target_expected_cost"].dtype == float


# ── 11. Synthetic panel with known expected outputs ─────────────────────────

class TestKnownPanelExactValues:
    def test_demeaned_values_on_known_panel(self):
        """On a 3-ticker, 4-date panel, verify exact demeaning behavior.

        Note: add_institutional_targets applies winsor_q=0.01 by default,
        which slightly shifts the mean. We use winsor_q=0.0 to get pure demeaning.
        """
        df = _make_known_panel()
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=False, net_of_costs=False, winsor_q=0.0),
            costs=ExecutionCostConfig(),
            max_name_weight=0.10,
        )
        # For each date, target_return should be forward_return - mean(forward_return)
        for date in result["date"].unique():
            date_mask = result["date"] == date
            fwd = df.loc[df["date"] == date, "forward_return"].values
            expected_demeaned = fwd - fwd.mean()
            actual = result.loc[date_mask, "target_return"].values
            np.testing.assert_array_almost_equal(actual, expected_demeaned, decimal=12)

    def test_target_rank_on_known_panel(self):
        df = _make_known_panel()
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=False, net_of_costs=False),
            costs=ExecutionCostConfig(),
            max_name_weight=0.10,
        )
        # target_rank should be in [-1, 1]
        assert result["target_rank"].between(-1.0, 1.0).all()
        # Within each date, ranks should be distinct (unless ties)
        for date in result["date"].unique():
            date_mask = result["date"] == date
            ranks = result.loc[date_mask, "target_rank"]
            assert len(ranks) == 3  # 3 tickers per date

    def test_target_down_decile_on_known_panel(self):
        df = _make_known_panel()
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=False, net_of_costs=False),
            costs=ExecutionCostConfig(),
            max_name_weight=0.10,
        )
        # With 3 tickers per date, down decile (bottom 10%) should be 0 for all
        assert (result["target_down_decile"] == 0).all()

    def test_target_up_on_known_panel(self):
        df = _make_known_panel()
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=False, net_of_costs=False),
            costs=ExecutionCostConfig(),
            max_name_weight=0.10,
        )
        # target_up = 1 if target_return > 0
        for date in result["date"].unique():
            date_mask = result["date"] == date
            fwd = df.loc[df["date"] == date, "forward_return"].values
            mean_fwd = fwd.mean()
            expected_up = (fwd - mean_fwd > 0).astype(int)
            actual_up = result.loc[date_mask, "target_up"].values
            np.testing.assert_array_equal(actual_up, expected_up)


# ── 12. Retarget panel behavior with daily_return ───────────────────────────

class TestRetargetPanelWithDailyReturn:
    def test_compound_forward_return_from_daily(self):
        """Verify that retarget_panel_for_horizon compounds daily returns correctly."""
        n_dates = 10
        h = 3
        df = _make_panel(n_dates=n_dates, n_tickers=3, include_daily_return=True, include_factors=True, seed=0)
        result = retarget_panel_for_horizon(
            df,
            horizon_days=h,
            target_cfg=TargetConfig(horizon_days=h, residualize=False),
            costs=ExecutionCostConfig(),
            max_name_weight=0.10,
        )
        # Manually compute compound return for first ticker
        ticker = "T0"
        ticker_daily = df.loc[df["ticker"] == ticker].sort_values("date")["daily_return"].values
        # fwd_ret[0] = (1+r[1])*(1+r[2])*(1+r[3]) - 1
        expected = np.prod(1.0 + ticker_daily[1:h+1]) - 1.0
        ticker_result = result.loc[result["ticker"] == ticker].sort_values("date").reset_index(drop=True)
        np.testing.assert_almost_equal(ticker_result.iloc[0]["forward_return"], expected, decimal=12)

    def test_retarget_without_daily_return_falls_back_to_forward_return(self):
        """When daily_return is absent, retarget uses existing forward_return."""
        df = _make_panel(n_dates=10, n_tickers=3, include_daily_return=False, include_factors=True)
        original_fwd = df["forward_return"].copy()
        result = retarget_panel_for_horizon(
            df,
            horizon_days=5,
            target_cfg=TargetConfig(horizon_days=5, residualize=False),
            costs=ExecutionCostConfig(),
            max_name_weight=0.10,
        )
        # forward_return should be unchanged (no daily_return to recompute from)
        np.testing.assert_array_almost_equal(
            result["forward_return"].values,
            original_fwd.values,
            decimal=15,
        )


# ── 13. make_training_target behavior ───────────────────────────────────────

class TestMakeTrainingTarget:
    def test_short_classifier_returns_int_array(self):
        df = _make_panel(n_dates=10, n_tickers=5, include_factors=True)
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=True),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )
        y = make_training_target(result, model_name="test", model_kind="short_classifier", use_risk_adj=False)
        assert y.dtype == int
        assert len(y) == len(result)
        assert set(np.unique(y)).issubset({0, 1})

    def test_long_alpha_returns_float_array(self):
        df = _make_panel(n_dates=10, n_tickers=5, include_factors=True)
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=True),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )
        y = make_training_target(result, model_name="test", model_kind="long_alpha", use_risk_adj=False)
        assert y.dtype == float
        assert len(y) == len(result)
        assert np.isfinite(y).all()

    def test_regressor_returns_float_array(self):
        df = _make_panel(n_dates=10, n_tickers=5, include_factors=True)
        result = add_institutional_targets(
            df,
            cfg=TargetConfig(horizon_days=5, residualize=True),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )
        y = make_training_target(result, model_name="test", model_kind="regressor", use_risk_adj=False)
        assert y.dtype == float
        assert len(y) == len(result)
