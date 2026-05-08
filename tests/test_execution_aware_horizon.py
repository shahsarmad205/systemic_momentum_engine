from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from model_selection.adaptive_params import (
    ExecutionAwareHorizonConfig,
    execution_aware_horizon_frontier,
    compute_rebalance_frontier,
    optimal_horizon_from_decay,
)


def _make_decay(features: list[str], horizons: list[int]) -> pd.DataFrame:
    """Build a minimal IC-decay table for testing."""
    rows = []
    for feat in features:
        for h in horizons:
            base_ic = 0.01 * (h ** 0.3)  # IC grows slowly with horizon
            noise = np.random.default_rng(42).normal(0, 0.001)
            rows.append({
                "feature": feat,
                "horizon_days": h,
                "target_type": "net_residual_return",
                "daily_spearman_ic": base_ic + noise,
                "daily_spearman_ic_std": 0.05,
                "daily_spearman_ic_tstat": (base_ic + noise) / 0.05 * np.sqrt(200),
                "signal_halflife_days": 3.0 if h <= 10 else 4.0,
                "positive_spread": 0.003,
                "positive_monotonicity": 0.55,
            })
    return pd.DataFrame(rows)


def _make_decay_with_fast_decay(features: list[str], horizons: list[int]) -> pd.DataFrame:
    """Build IC-decay with very short halflife (1.0d for all horizons)."""
    rows = []
    for feat in features:
        for h in horizons:
            rows.append({
                "feature": feat,
                "horizon_days": h,
                "target_type": "net_residual_return",
                "daily_spearman_ic": 0.025,
                "daily_spearman_ic_std": 0.05,
                "daily_spearman_ic_tstat": 0.025 / 0.05 * np.sqrt(200),
                "signal_halflife_days": 1.0,  # very fast decay
                "positive_spread": 0.003,
                "positive_monotonicity": 0.55,
            })
    return pd.DataFrame(rows)


class TestExecutionAwareHorizonFrontier:
    """P34: Execution-aware horizon frontier tests."""

    def test_empty_decay_returns_empty_frontier(self):
        frontier, diag = execution_aware_horizon_frontier(
            pd.DataFrame(), cfg=ExecutionAwareHorizonConfig()
        )
        assert frontier.empty
        assert "empty_decay_table" in diag["flags"]

    def test_all_candidates_in_frontier(self):
        decay = _make_decay(["f1", "f2", "f3"], [5, 10, 20, 63])
        cfg = ExecutionAwareHorizonConfig(candidate_horizons=(5, 10, 20, 63))
        frontier, diag = execution_aware_horizon_frontier(decay, cfg=cfg)
        # Should have at least the horizons that pass guardrails
        assert not frontier.empty
        assert "horizon_days" in frontier.columns
        for col in ["median_ic", "ic_ir", "signal_halflife_days", "rank_persistence_at_rebalance"]:
            assert col in frontier.columns

    def test_not_yet_available_metrics_marked(self):
        decay = _make_decay(["f1", "f2"], [5, 10])
        cfg = ExecutionAwareHorizonConfig(candidate_horizons=(5, 10))
        frontier, diag = execution_aware_horizon_frontier(decay, cfg=cfg)
        for na_col in ["avg_turnover", "alpha_capture", "gross_alpha",
                         "execution_cost_pnl", "net_execution_sharpe",
                         "psr", "dsr", "score_direction_stability"]:
            assert na_col in frontier.columns
            assert all(frontier[na_col] == "not_yet_available")

    def test_low_persistence_flag_raised(self):
        """High-IC horizon with fast signal decay should be flagged."""
        decay = _make_decay_with_fast_decay(["f1", "f2", "f3"], [5, 10, 20, 63])
        cfg = ExecutionAwareHorizonConfig(
            candidate_horizons=(5, 10, 20, 63),
            halflife_persistence_threshold=0.30,
        )
        frontier, diag = execution_aware_horizon_frontier(decay, cfg=cfg)
        # With halflife=1.0, rebalance=5 → persistence = 2^(-5/1) = 0.031 < 0.30
        assert any("high_ic_low_persistence" in flag for flag in diag.get("flags", []))
        assert not empty_df(frontier)

    def test_persistence_ok_when_halflife_long_enough(self):
        """Long-halflife signals pass persistence threshold."""
        rows = []
        for h in [5, 10, 20]:
            for feat in ["f1", "f2"]:
                rows.append({
                    "feature": feat,
                    "horizon_days": h,
                    "target_type": "net_residual_return",
                    "daily_spearman_ic": 0.015,
                    "daily_spearman_ic_std": 0.05,
                    "daily_spearman_ic_tstat": 0.015 / 0.05 * np.sqrt(200),
                    "signal_halflife_days": 15.0,  # long halflife
                    "positive_spread": 0.003,
                    "positive_monotonicity": 0.55,
                })
        decay = pd.DataFrame(rows)
        cfg = ExecutionAwareHorizonConfig(
            candidate_horizons=(5, 10, 20),
            halflife_persistence_threshold=0.30,
        )
        frontier, diag = execution_aware_horizon_frontier(decay, cfg=cfg)
        assert not frontier.empty
        # At halflife=15, rebalance=5 → persistence = 2^(-5/15) = 0.79 > 0.30
        assert all(
            p is not False
            for p in frontier["halflife_persistence_ok"]
            if p is not None
        )

    def test_composite_score_present(self):
        decay = _make_decay(["f1", "f2", "f3"], [5, 10, 20])
        cfg = ExecutionAwareHorizonConfig(candidate_horizons=(5, 10, 20))
        frontier, diag = execution_aware_horizon_frontier(decay, cfg=cfg)
        assert "composite_score" in frontier.columns
        assert len(frontier) >= 1

    def test_ic_only_selector_still_works(self):
        """optimal_horizon_from_decay is unaffected."""
        decay = _make_decay(["f1", "f2", "f3"], [5, 10, 20, 63])
        opt_h, diag = optimal_horizon_from_decay(
            decay, candidate_horizons=[5, 10, 20, 63]
        )
        assert opt_h in (5, 10, 20, 63)
        assert "scores" in diag

    def test_missing_halflife_metrics_not_crashed(self):
        """Missing halflife column should not crash the frontier."""
        rows = []
        for h in [5, 10]:
            for feat in ["f1", "f2"]:
                rows.append({
                    "feature": feat,
                    "horizon_days": h,
                    "target_type": "net_residual_return",
                    "daily_spearman_ic": 0.015,
                    "daily_spearman_ic_std": 0.05,
                    "daily_spearman_ic_tstat": 0.015 / 0.05 * np.sqrt(200),
                    # no signal_halflife_days column
                    "positive_spread": 0.003,
                    "positive_monotonicity": 0.55,
                })
        decay = pd.DataFrame(rows)
        cfg = ExecutionAwareHorizonConfig(candidate_horizons=(5, 10))
        frontier, diag = execution_aware_horizon_frontier(decay, cfg=cfg)
        assert not frontier.empty
        assert "signal_halflife_days" in frontier.columns


class TestRebalanceFrontier:
    """P34: Rebalance frequency frontier tests."""

    def test_empty_decay_returns_empty(self):
        frontier = compute_rebalance_frontier(
            pd.DataFrame(), cfg=ExecutionAwareHorizonConfig()
        )
        assert frontier.empty

    def test_all_rebalance_candidates_evaluated(self):
        decay = _make_decay(["f1", "f2"], [10, 20])
        cfg = ExecutionAwareHorizonConfig(
            candidate_horizons=(10, 20),
            candidate_rebalance_frequencies=(2, 5, 10, 20),
        )
        frontier = compute_rebalance_frontier(decay, cfg=cfg)
        assert not frontier.empty
        assert "target_horizon_days" in frontier.columns
        assert "rebalance_frequency_days" in frontier.columns
        assert "rank_persistence_at_rebalance" in frontier.columns
        # All rebalance frequencies <= target_horizon * max_ratio should be present
        for h in [10, 20]:
            for r in [2, 5, 10]:
                if r <= h * 1.0:
                    assert ((frontier["target_horizon_days"] == h) & (frontier["rebalance_frequency_days"] == r)).any()

    def test_persistence_decreases_with_rebalance(self):
        """Larger rebalance → lower persistence."""
        decay = _make_decay(["f1", "f2"], [20])
        cfg = ExecutionAwareHorizonConfig(
            candidate_horizons=(20,),
            candidate_rebalance_frequencies=(2, 5, 20),
        )
        frontier = compute_rebalance_frontier(decay, cfg=cfg)
        frontier = frontier.sort_values("rebalance_frequency_days")
        pers_values = frontier["rank_persistence_at_rebalance"].dropna()
        if len(pers_values) >= 2:
            # Persistence should decrease (or stay equal) as rebalance increases
            assert pers_values.iloc[0] >= pers_values.iloc[-1]


def empty_df(df: pd.DataFrame) -> bool:
    return df is None or df.empty
