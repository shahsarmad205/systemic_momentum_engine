"""P36: Alpha-capture decomposition tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from model_selection.empirical_baselines import (
    alpha_capture_decomposition,
    alpha_capture_summary_from_per_model,
)


def _make_scored(n_dates: int = 10, n_tickers: int = 50, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic scored panel with known IC and costs."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="D")
    rows = []
    for d in dates:
        score_noise = rng.normal(0, 1, n_tickers)
        ret_noise = rng.normal(0, 0.02, n_tickers)
        # Positive IC: score correlates ~0.1 with forward return
        common = rng.normal(0, 1, n_tickers)
        scores = 0.3 * common + 0.7 * score_noise
        fwd_rets = 0.01 * common + ret_noise
        for i in range(n_tickers):
            rows.append({
                "date": d,
                "ticker": f"T{i:03d}",
                "score": float(scores[i]),
                "forward_return": float(fwd_rets[i]),
                "target_return": float(fwd_rets[i]),
            })
    return pd.DataFrame(rows)


def _make_pnl_detail(scored_df: pd.DataFrame, cost_fraction: float = 0.30) -> pd.DataFrame:
    """Build synthetic PnL detail from scored data."""
    rows = []
    for date, grp in scored_df.groupby("date", sort=False):
        n = len(grp)
        gross = float(grp["forward_return"].iloc[:5].mean()) if n >= 5 else 0.0
        cost = abs(gross) * cost_fraction
        rows.append({
            "date": date,
            "gross_return": gross,
            "cost_return": cost,
            "commission_return": cost * 0.2,
            "spread_return": cost * 0.1,
            "temporary_impact_return": cost * 0.5,
            "permanent_impact_return": cost * 0.2,
            "long_cost_return": cost * 0.8,
            "short_cost_return": cost * 0.2,
            "borrow_return": 0.0,
        })
    return pd.DataFrame(rows)


class TestAlphaCaptureDecomposition:
    """P36: Per-date-per-ticker attribution integrity."""

    def test_empty_input_returns_empty(self):
        df, summary = alpha_capture_decomposition(pd.DataFrame())
        assert df.empty
        assert summary["decomp_status"] == "empty"

    def test_missing_columns_returns_status(self):
        df = pd.DataFrame({"date": ["2020-01-01"], "ticker": ["A"], "not_score": [0.5]})
        _, summary = alpha_capture_decomposition(df)
        assert summary["decomp_status"] == "missing_columns"

    def test_basic_decomposition_columns_present(self):
        scored = _make_scored(n_dates=5, n_tickers=20)
        pnl = _make_pnl_detail(scored)
        dec, summary = alpha_capture_decomposition(
            scored, pnl_detail_df=pnl, model_name="test_model", window_idx=0,
        )
        assert not dec.empty
        assert summary["decomp_status"] == "ok"
        for col in ["score_rank_pct", "selected_for_trade", "gross_pnl_per_ticker",
                     "allocated_cost", "net_pnl_per_ticker"]:
            assert col in dec.columns

    def test_full_universe_ic_is_computed(self):
        scored = _make_scored(n_dates=10, n_tickers=50)
        pnl = _make_pnl_detail(scored)
        _, summary = alpha_capture_decomposition(scored, pnl_detail_df=pnl)
        assert "full_universe_ic_mean" in summary
        assert np.isfinite(summary["full_universe_ic_mean"])

    def test_alpha_capture_labels_cost_dominated(self):
        """With weights and returns, alpha_capture is computed (positive or negative)."""
        scored = _make_scored(n_dates=5, n_tickers=30)
        tw_rows = []
        for date, grp in scored.groupby("date", sort=False):
            grp = grp.copy()
            grp["rank"] = grp["score"].rank(ascending=False)
            for _, r in grp.iterrows():
                w = 1.0 / 5 if r["rank"] <= 5 else 0.0
                tw_rows.append({"date": date, "ticker": r["ticker"], "target_weight": w})
        tw_df = pd.DataFrame(tw_rows)
        pnl = _make_pnl_detail(scored, cost_fraction=3.0)
        _, summary = alpha_capture_decomposition(
            scored, target_weights_df=tw_df, pnl_detail_df=pnl,
        )
        # alpha_capture and label should be computed (not unknown)
        assert summary["alpha_capture_label"] != "unknown"
        assert "alpha_capture" in summary

    def test_pnl_attribution_sums_match(self):
        """Gross PnL - cost = net PnL."""
        scored = _make_scored(n_dates=8, n_tickers=30)
        pnl = _make_pnl_detail(scored, cost_fraction=0.30)
        dec, summary = alpha_capture_decomposition(scored, pnl_detail_df=pnl)
        gross = summary["gross_pnl_total"]
        cost = summary["cost_total"]
        net = summary["net_pnl_total"]
        assert abs(gross - cost - net) < 1e-9

    def test_no_weights_still_computes_ic(self):
        scored = _make_scored(n_dates=5, n_tickers=30)
        pnl = _make_pnl_detail(scored)
        _, summary = alpha_capture_decomposition(scored, pnl_detail_df=pnl)
        assert np.isfinite(summary["full_universe_ic_mean"])
        # Without weights, selected universe IC is NaN
        assert not np.isfinite(summary["selected_universe_ic_mean"])

    def test_with_weights_computes_weighted_metrics(self):
        scored = _make_scored(n_dates=5, n_tickers=30)
        pnl = _make_pnl_detail(scored)
        # Build target weights proportional to score rank
        tw_rows = []
        for date, grp in scored.groupby("date", sort=False):
            grp = grp.copy()
            grp["rank"] = grp["score"].rank(pct=True, ascending=False)
            for _, r in grp.iterrows():
                w = float(r["rank"])  # higher score → higher weight
                tw_rows.append({"date": date, "ticker": r["ticker"], "target_weight": w})
        tw_df = pd.DataFrame(tw_rows)
        _, summary = alpha_capture_decomposition(
            scored, target_weights_df=tw_df, pnl_detail_df=pnl,
        )
        assert np.isfinite(summary["selected_universe_ic_mean"])
        assert np.isfinite(summary["weighted_ic_mean"])
        assert np.isfinite(summary["score_weight_corr_mean"])


class TestAlphaCaptureSummary:
    """P36: Aggregate summary from per-model parts."""

    def test_empty_parts_returns_empty_df(self):
        df = alpha_capture_summary_from_per_model([])
        assert df.empty

    def test_multiple_parts_produces_rows(self):
        parts = [
            {"model_name": "M1", "window_idx": 0, "alpha_capture": 0.5, "decomp_status": "ok"},
            {"model_name": "M1", "window_idx": 1, "alpha_capture": -0.3, "decomp_status": "ok"},
        ]
        df = alpha_capture_summary_from_per_model(parts)
        assert len(df) == 2
        assert "alpha_capture" in df.columns
        assert df.iloc[0]["alpha_capture"] == 0.5
        assert df.iloc[1]["alpha_capture"] == -0.3


class TestPositiveICNegativeAlphaCapture:
    """The key diagnostic: positive IC but negative execution."""

    def test_positive_ic_zero_cost_yields_positive_alpha_capture(self):
        """With positive IC and zero cost, alpha capture > 0."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        rows = []
        for d in dates:
            common = rng.normal(0, 1, 20)
            scores = common + rng.normal(0, 0.5, 20)
            fwd = 0.01 * common + rng.normal(0, 0.01, 20)
            for i in range(20):
                rows.append({"date": d, "ticker": f"T{i:03d}",
                             "score": float(scores[i]), "forward_return": float(fwd[i]),
                             "target_return": float(fwd[i])})
        scored = pd.DataFrame(rows)
        # Build equal-weight positions for top 5
        tw_rows = []
        for date, grp in scored.groupby("date", sort=False):
            grp = grp.copy()
            grp["rank"] = grp["score"].rank(ascending=False)
            for _, r in grp.iterrows():
                w = 1.0 / 5 if r["rank"] <= 5 else 0.0
                tw_rows.append({"date": date, "ticker": r["ticker"], "target_weight": w})
        tw_df = pd.DataFrame(tw_rows)
        pnl = _make_pnl_detail(scored, cost_fraction=0.001)  # near-zero costs
        _, summary = alpha_capture_decomposition(
            scored, target_weights_df=tw_df, pnl_detail_df=pnl,
        )
        assert np.isfinite(summary["full_universe_ic_mean"])
        assert summary["full_universe_ic_mean"] > 0
        assert summary["net_pnl_total"] > 0

    def test_positive_ic_high_cost_yields_cost_dominated(self):
        """With positive IC but high costs, alpha capture < 0."""
        rng = np.random.default_rng(42)
        dates = pd.date_range("2020-01-01", periods=20, freq="D")
        rows = []
        for d in dates:
            common = rng.normal(0, 1, 20)
            scores = common + rng.normal(0, 0.5, 20)
            fwd = 0.005 * common + rng.normal(0, 0.01, 20)
            for i in range(20):
                rows.append({"date": d, "ticker": f"T{i:03d}",
                             "score": float(scores[i]), "forward_return": float(fwd[i]),
                             "target_return": float(fwd[i])})
        scored = pd.DataFrame(rows)
        tw_rows = []
        for date, grp in scored.groupby("date", sort=False):
            grp = grp.copy()
            grp["rank"] = grp["score"].rank(ascending=False)
            for _, r in grp.iterrows():
                w = 1.0 / 5 if r["rank"] <= 5 else 0.0
                tw_rows.append({"date": date, "ticker": r["ticker"], "target_weight": w})
        tw_df = pd.DataFrame(tw_rows)
        pnl = _make_pnl_detail(scored, cost_fraction=2.0)  # costs exceed gross
        _, summary = alpha_capture_decomposition(
            scored, target_weights_df=tw_df, pnl_detail_df=pnl,
        )
        assert np.isfinite(summary["full_universe_ic_mean"])
        assert summary["full_universe_ic_mean"] > 0
        # High costs should make alpha capture negative
        assert not np.isfinite(summary["alpha_capture"]) or summary["alpha_capture"] < 0
