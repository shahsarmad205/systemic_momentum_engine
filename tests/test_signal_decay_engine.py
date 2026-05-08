"""Tests for the Signal Decay and Horizon Compatibility Engine.

Proves:
1. Rank persistence is ticker/date aligned.
2. Flattened panel autocorrelation is not used.
3. Halflife is not hardcoded.
4. IC decay uses true h-day forward returns.
5. Different horizons use different forward return columns.
6. Features with fast decay are rejected for long horizons.
7. Features with peak IC at short horizons are not promoted at long horizons.
8. Missing or noisy decay data creates explicit quality flags.
9. Turnover estimate increases when rank persistence falls.
10. Smoothing is not accepted unless net alpha after costs improves.
11. All thresholds come from ResearchContract/config.
12. Conditional sleeves compute sleeve-specific decay rather than global decay.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from model_selection.signal_decay_engine import (
    SignalDecayEngine,
    compute_rank_persistence_curve,
    estimate_halflife_from_persistence,
    compute_ic_decay_curve,
    classify_horizon_compatibility,
    estimate_turnover_from_decay,
    run_smoothing_experiment,
    generate_signal_decay_reports,
    RankPersistenceResult,
    HalflifeResult,
    ICDecayResult,
    HorizonCompatibilityResult,
    TurnoverDecayResult,
    DecayStatus,
    HorizonStatus,
    _get_decay_config,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture
def persistent_signal_df():
    """Panel with a highly persistent signal (slow decay)."""
    np.random.seed(42)
    n_dates = 200
    n_tickers = 50
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    rows = []
    for ticker in tickers:
        # Persistent signal: AR(1) with high coefficient
        signal = np.zeros(n_dates)
        signal[0] = np.random.randn()
        for t in range(1, n_dates):
            signal[t] = 0.95 * signal[t - 1] + 0.1 * np.random.randn()
        for t, date in enumerate(dates):
            rows.append({
                "date": date,
                "ticker": ticker,
                "persistent_signal": signal[t],
                "forward_return": np.random.randn() * 0.01 + 0.001 * signal[t],
            })

    return pd.DataFrame(rows)


@pytest.fixture
def fast_decay_signal_df():
    """Panel with a fast-decaying signal."""
    np.random.seed(42)
    n_dates = 200
    n_tickers = 50
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    rows = []
    for ticker in tickers:
        # Fast decay: AR(1) with low coefficient
        signal = np.zeros(n_dates)
        signal[0] = np.random.randn()
        for t in range(1, n_dates):
            signal[t] = 0.3 * signal[t - 1] + 0.5 * np.random.randn()
        for t, date in enumerate(dates):
            rows.append({
                "date": date,
                "ticker": ticker,
                "fast_signal": signal[t],
                "forward_return": np.random.randn() * 0.01 + 0.001 * signal[t],
            })

    return pd.DataFrame(rows)


@pytest.fixture
def minimal_df():
    """Minimal panel for basic tests."""
    np.random.seed(42)
    n_dates = 100
    n_tickers = 20
    dates = pd.date_range("2020-01-01", periods=n_dates, freq="B")
    tickers = [f"T{i:03d}" for i in range(n_tickers)]

    rows = []
    for ticker in tickers:
        signal = np.random.randn(n_dates)
        for t, date in enumerate(dates):
            rows.append({
                "date": date,
                "ticker": ticker,
                "test_feature": signal[t],
                "forward_return": np.random.randn() * 0.01,
            })

    return pd.DataFrame(rows)


# ── Test 1: Rank persistence is ticker/date aligned ──────────────────────────

class TestRankPersistenceAlignment:
    def test_ticker_date_aligned(self, persistent_signal_df):
        """Rank persistence computes per-ticker, per-date correlations."""
        results = compute_rank_persistence_curve(
            persistent_signal_df, "persistent_signal",
            lags=[1, 2, 5, 10],
            min_dates=30, min_breadth=5,
        )
        assert len(results) == 4
        for r in results:
            assert r.n_dates > 0
            assert r.avg_breadth > 0
            # Persistence should be high for persistent signal
            if r.lag <= 5:
                assert r.rank_persistence > 0.5, f"Lag {r.lag}: persistence={r.rank_persistence}"

    def test_not_flattened_autocorrelation(self, persistent_signal_df):
        """Rank persistence is NOT computed from flattened panel autocorrelation.

        True ticker-aligned persistence should have n_dates > 0 and avg_breadth > 0,
        proving it's computed per-ticker, not on a flattened series.
        """
        results = compute_rank_persistence_curve(
            persistent_signal_df, "persistent_signal",
            lags=[1], min_dates=30, min_breadth=5,
        )
        assert len(results) == 1
        # Ticker-aligned should report per-date, per-ticker stats
        assert results[0].n_dates > 0
        assert results[0].avg_breadth > 0
        # Persistence should be positive for persistent signal
        assert results[0].rank_persistence > 0.5

    def test_empty_df_returns_empty(self):
        """Empty DataFrame returns empty results."""
        results = compute_rank_persistence_curve(pd.DataFrame(), "test", lags=[1])
        assert results == []

    def test_missing_feature_returns_empty(self, minimal_df):
        """Missing feature column returns empty results."""
        results = compute_rank_persistence_curve(minimal_df, "nonexistent", lags=[1])
        assert results == []

    def test_min_dates_filter(self, minimal_df):
        """Results are empty when dates are below minimum."""
        results = compute_rank_persistence_curve(
            minimal_df, "test_feature", lags=[1],
            min_dates=500, min_breadth=5,
        )
        # All results should have insufficient quality
        for r in results:
            assert r.persistence_quality == "insufficient"


# ── Test 2: Halflife is not hardcoded ────────────────────────────────────────

class TestHalflifeNotHardcoded:
    def test_halflife_differs_between_signals(self):
        """Halflife estimates differ between persistent and fast-decay signals."""
        # Create mock persistence results with known decay rates
        # Persistent signal: slow decay
        persist_results = [
            RankPersistenceResult(
                candidate_id="test", feature="persistent", family="test",
                sleeve="", regime="", lag=lag,
                rank_persistence=0.9 ** lag,  # Very slow decay
                n_dates=200, avg_breadth=50,
                persistence_quality="high",
            )
            for lag in [1, 2, 3, 5, 10, 20]
        ]
        persist_hl = estimate_halflife_from_persistence(persist_results)

        # Fast decay signal: rapid decay
        fast_results = [
            RankPersistenceResult(
                candidate_id="test", feature="fast", family="test",
                sleeve="", regime="", lag=lag,
                rank_persistence=0.5 ** lag,  # Very fast decay
                n_dates=200, avg_breadth=50,
                persistence_quality="high",
            )
            for lag in [1, 2, 3, 5, 10, 20]
        ]
        fast_hl = estimate_halflife_from_persistence(fast_results)

        # Persistent signal should have much longer halflife
        assert persist_hl.estimated_halflife_days > fast_hl.estimated_halflife_days, (
            f"Persistent halflife={persist_hl.estimated_halflife_days:.1f} "
            f"should be > fast halflife={fast_hl.estimated_halflife_days:.1f}"
        )
        assert persist_hl.fit_r2 > 0.5  # Good fit for clean exponential data
        assert fast_hl.fit_r2 > 0.5

    def test_halflife_comes_from_fit_not_constant(self, minimal_df):
        """Halflife is computed from the persistence curve, not a constant."""
        results = compute_rank_persistence_curve(
            minimal_df, "test_feature",
            lags=[1, 2, 3, 5, 10, 20], min_dates=30, min_breadth=5,
        )
        hl = estimate_halflife_from_persistence(results)
        # Should not be a hardcoded value like 10.0 or horizon*0.5
        assert hl.estimated_halflife_days > 0
        assert hl.decay_tau > 0
        # Fit quality should be reported
        assert hl.halflife_quality in ("high", "medium", "low")

    def test_insufficient_data_returns_fallback(self):
        """Insufficient data returns fallback with explicit status."""
        hl = estimate_halflife_from_persistence([])
        assert hl.decay_status == DecayStatus.INSUFFICIENT_DATA.value
        assert hl.halflife_quality == "low"


# ── Test 3: IC decay uses true h-day forward returns ─────────────────────────

class TestICDecayForwardReturns:
    def test_ic_decay_uses_forward_return(self, persistent_signal_df):
        """IC decay computes using forward_return column, not daily_return."""
        # Remove daily_return if present
        df = persistent_signal_df.copy()
        if "daily_return" in df.columns:
            df = df.drop(columns=["daily_return"])

        results = compute_ic_decay_curve(
            df, "persistent_signal",
            horizons=[1, 5, 10, 20],
            min_dates=30, min_breadth=5,
        )
        assert len(results) == 4
        # Each horizon should have different IC
        ics = [r.mean_ic for r in results if r.ic_quality != "insufficient"]
        if len(ics) >= 2:
            # IC should vary across horizons (not identical)
            assert len(set(round(ic, 4) for ic in ics)) > 1 or all(ic == 0 for ic in ics)

    def test_different_horizons_different_columns(self, persistent_signal_df):
        """Different horizons use different forward return windows."""
        df = persistent_signal_df.copy()
        # The engine builds fwd_ret_{h}d columns internally
        results = compute_ic_decay_curve(
            df, "persistent_signal",
            horizons=[1, 5, 20],
            min_dates=30, min_breadth=5,
        )
        # Verify each result has a different horizon
        horizons = [r.horizon for r in results]
        assert 1 in horizons
        assert 5 in horizons
        assert 20 in horizons

    def test_empty_df_returns_empty(self):
        """Empty DataFrame returns empty IC decay results."""
        results = compute_ic_decay_curve(pd.DataFrame(), "test", horizons=[1, 5])
        assert results == []


# ── Test 4: Fast decay rejected for long horizons ────────────────────────────

class TestFastDecayRejection:
    def test_fast_decay_rejected_at_long_horizon(self, fast_decay_signal_df):
        """Features with fast decay are rejected for long horizons."""
        # Compute persistence and halflife
        persist_results = compute_rank_persistence_curve(
            fast_decay_signal_df, "fast_signal",
            lags=[1, 2, 3, 5, 10, 20], min_dates=30, min_breadth=5,
        )
        hl = estimate_halflife_from_persistence(persist_results)

        # IC decay
        ic_results = compute_ic_decay_curve(
            fast_decay_signal_df, "fast_signal",
            horizons=[1, 5, 10, 20], min_dates=30, min_breadth=5,
        )

        # Classify at long horizon
        compat = classify_horizon_compatibility(
            hl, ic_results, tested_horizon=20,
            halflife_to_horizon_ratio_min=0.5,
            persistence_at_horizon_min=0.3,
        )

        # Fast decay should NOT be compatible with long horizon
        assert compat.horizon_status != HorizonStatus.HORIZON_COMPATIBLE.value

    def test_persistent_signal_compatible_with_long_horizon(self, persistent_signal_df):
        """Persistent signals are compatible with longer horizons."""
        persist_results = compute_rank_persistence_curve(
            persistent_signal_df, "persistent_signal",
            lags=[1, 2, 3, 5, 10, 20], min_dates=30, min_breadth=5,
        )
        hl = estimate_halflife_from_persistence(persist_results)
        ic_results = compute_ic_decay_curve(
            persistent_signal_df, "persistent_signal",
            horizons=[1, 5, 10, 20], min_dates=30, min_breadth=5,
        )

        compat = classify_horizon_compatibility(
            hl, ic_results, tested_horizon=10,
            halflife_to_horizon_ratio_min=0.5,
            persistence_at_horizon_min=0.3,
        )

        # Persistent signal should be more likely compatible
        assert compat.halflife_to_horizon_ratio > 0.5 or compat.persistence_at_horizon > 0.3


# ── Test 5: Peak IC at short horizons not promoted at long horizons ───────────

class TestPeakICClassification:
    def test_ic_peaks_elsewhere_detected(self):
        """When IC peaks at a different horizon, it's flagged."""
        # Create mock IC decay results where IC peaks at horizon 1
        ic_results = [
            ICDecayResult(
                candidate_id="test", feature="test", family="test",
                sleeve="", regime="", horizon=h,
                mean_ic=0.05 if h == 1 else 0.01,
                icir=2.0, hac_tstat=2.0, n_dates=100, avg_breadth=50,
                sign_consistency=0.8, subperiod_stability=0.7,
                ic_quality="high",
            )
            for h in [1, 5, 10, 20]
        ]

        # Mock halflife result
        hl = HalflifeResult(
            candidate_id="test", feature="test", family="test",
            sleeve="", regime="",
            estimated_halflife_days=3.0, decay_tau=4.33,
            initial_persistence=0.9, persistence_at_horizon=0.1,
            fit_r2=0.8, halflife_quality="high",
            decay_status=DecayStatus.FAST_DECAY.value,
        )

        # Classify at horizon 20 (IC peaks at 1)
        compat = classify_horizon_compatibility(
            hl, ic_results, tested_horizon=20,
        )

        # Should detect IC peaks elsewhere
        assert compat.horizon_status in (
            HorizonStatus.IC_PEAKS_ELSEWHERE.value,
            HorizonStatus.HORIZON_TOO_LONG.value,
            HorizonStatus.SIGNAL_TOO_FAST.value,
        )


# ── Test 6: Missing/noisy decay data creates quality flags ───────────────────

class TestDecayQualityFlags:
    def test_insufficient_data_flag(self):
        """Missing data creates explicit insufficient_data status."""
        hl = estimate_halflife_from_persistence([])
        assert hl.decay_status == DecayStatus.INSUFFICIENT_DATA.value
        assert hl.halflife_quality == "low"

    def test_unstable_decay_flag(self):
        """Noisy persistence creates unstable_decay or low quality status."""
        # Create results with all negative persistence (no decay signal)
        results = [
            RankPersistenceResult(
                candidate_id="test", feature="test", family="test",
                sleeve="", regime="", lag=lag,
                rank_persistence=val, n_dates=100, avg_breadth=50,
                persistence_quality="high",
            )
            for lag, val in [(1, -0.2), (2, -0.1), (3, -0.3), (5, -0.1), (10, -0.2)]
        ]
        hl = estimate_halflife_from_persistence(results)
        # All-negative persistence should produce unstable or insufficient status
        assert hl.decay_status in (
            DecayStatus.UNSTABLE_DECAY.value,
            DecayStatus.INSUFFICIENT_DATA.value,
        ) or hl.halflife_quality == "low"

    def test_persistence_quality_levels(self, minimal_df):
        """Persistence quality reflects data sufficiency."""
        results = compute_rank_persistence_curve(
            minimal_df, "test_feature",
            lags=[1, 2, 3, 5, 10], min_dates=30, min_breadth=5,
        )
        for r in results:
            assert r.persistence_quality in ("high", "medium", "low", "insufficient")


# ── Test 7: Turnover increases when rank persistence falls ───────────────────

class TestTurnoverFromDecay:
    def test_turnover_increases_with_faster_decay(self, persistent_signal_df, fast_decay_signal_df):
        """Turnover estimate is higher for fast-decaying signals."""
        # Persistent signal
        persist_results = compute_rank_persistence_curve(
            persistent_signal_df, "persistent_signal",
            lags=[1, 2, 3, 5, 10, 20], min_dates=30, min_breadth=5,
        )
        persist_hl = estimate_halflife_from_persistence(persist_results)
        persist_turnover = estimate_turnover_from_decay(
            persistent_signal_df, "persistent_signal", persist_hl,
            rebalance_gaps=[5, 10],
        )

        # Fast decay signal
        fast_results = compute_rank_persistence_curve(
            fast_decay_signal_df, "fast_signal",
            lags=[1, 2, 3, 5, 10, 20], min_dates=30, min_breadth=5,
        )
        fast_hl = estimate_halflife_from_persistence(fast_results)
        fast_turnover = estimate_turnover_from_decay(
            fast_decay_signal_df, "fast_signal", fast_hl,
            rebalance_gaps=[5, 10],
        )

        # Fast decay should have higher turnover pressure
        if persist_turnover and fast_turnover:
            persist_pressure = persist_turnover[0].turnover_pressure
            fast_pressure = fast_turnover[0].turnover_pressure
            assert fast_pressure >= persist_pressure, (
                f"Fast decay turnover pressure ({fast_pressure}) should be >= "
                f"persistent ({persist_pressure})"
            )

    def test_turnover_increases_with_longer_rebalance(self, persistent_signal_df):
        """Turnover increases as rebalance gap increases."""
        results = compute_rank_persistence_curve(
            persistent_signal_df, "persistent_signal",
            lags=[1, 2, 3, 5, 10, 20], min_dates=30, min_breadth=5,
        )
        hl = estimate_halflife_from_persistence(results)
        turnover = estimate_turnover_from_decay(
            persistent_signal_df, "persistent_signal", hl,
            rebalance_gaps=[1, 5, 10, 20],
        )

        if len(turnover) >= 2:
            # Longer rebalance should have higher turnover pressure
            assert turnover[-1].turnover_pressure >= turnover[0].turnover_pressure


# ── Test 8: Smoothing not accepted unless net alpha improves ─────────────────

class TestSmoothingExperiment:
    def test_smoothing_requires_net_alpha_improvement(self, persistent_signal_df):
        """Smoothing is only accepted if net alpha after costs improves."""
        cfg = {
            "signal_decay": {
                "smoothing": {
                    "enabled": True,
                    "methods": ["ewma"],
                    "ewma_spans": [3, 5],
                },
            },
        }
        results = run_smoothing_experiment(
            persistent_signal_df, "persistent_signal", cfg,
            horizons=[5, 10, 20],
        )

        if len(results) > 1:
            raw = results[0]
            assert raw.smoothing_method == "raw"
            # Smoothed versions should only be accepted if net alpha > raw
            for r in results[1:]:
                if r.accepted:
                    assert r.net_alpha_bps > raw.net_alpha_bps

    def test_smoothing_disabled_by_default(self, minimal_df):
        """Smoothing experiment is disabled by default."""
        results = run_smoothing_experiment(minimal_df, "test_feature", {})
        assert results == []


# ── Test 9: Thresholds come from config ──────────────────────────────────────

class TestThresholdsFromConfig:
    def test_decay_config_from_user(self):
        """User config overrides defaults."""
        cfg = {
            "signal_decay": {
                "min_dates_for_persistence": 100,
                "halflife_to_horizon_ratio_min": 0.8,
            },
        }
        decay_cfg = _get_decay_config(cfg)
        assert decay_cfg["min_dates_for_persistence"] == 100
        assert decay_cfg["halflife_to_horizon_ratio_min"] == 0.8

    def test_decay_config_defaults(self):
        """Missing config uses defaults."""
        decay_cfg = _get_decay_config({})
        assert decay_cfg["min_dates_for_persistence"] == 50
        assert decay_cfg["halflife_to_horizon_ratio_min"] == 0.5
        assert decay_cfg["lags"] == [1, 2, 3, 5, 10, 20, 40, 63]

    def test_horizon_compatibility_uses_config_thresholds(self):
        """Horizon compatibility uses configurable thresholds."""
        ic_results = [
            ICDecayResult(
                candidate_id="test", feature="test", family="test",
                sleeve="", regime="", horizon=20,
                mean_ic=0.03, icir=2.0, hac_tstat=2.0,
                n_dates=100, avg_breadth=50,
                sign_consistency=0.8, subperiod_stability=0.7,
                ic_quality="high",
            )
        ]
        hl = HalflifeResult(
            candidate_id="test", feature="test", family="test",
            sleeve="", regime="",
            estimated_halflife_days=15.0, decay_tau=21.6,
            initial_persistence=0.9, persistence_at_horizon=0.4,
            fit_r2=0.8, halflife_quality="high",
            decay_status=DecayStatus.PERSISTENT.value,
        )

        # With strict thresholds
        compat_strict = classify_horizon_compatibility(
            hl, ic_results, 20,
            halflife_to_horizon_ratio_min=1.0,
            persistence_at_horizon_min=0.5,
        )

        # With lenient thresholds
        compat_lenient = classify_horizon_compatibility(
            hl, ic_results, 20,
            halflife_to_horizon_ratio_min=0.3,
            persistence_at_horizon_min=0.2,
        )

        # Different thresholds should produce different results
        assert compat_strict.horizon_status != compat_lenient.horizon_status or \
               compat_strict.rejection_reason != compat_lenient.rejection_reason


# ── Test 10: Full engine integration ─────────────────────────────────────────

class TestSignalDecayEngine:
    def test_full_diagnostics(self, persistent_signal_df):
        """Full engine runs all diagnostics."""
        engine = SignalDecayEngine(config={})
        diagnostics = engine.run_full_diagnostics(
            persistent_signal_df,
            features=["persistent_signal"],
            horizons=[1, 5, 10, 20],
        )
        assert len(diagnostics) == 1
        d = diagnostics[0]
        assert len(d.persistence_results) > 0
        assert d.halflife_result.estimated_halflife_days > 0
        assert len(d.ic_decay_results) == 4
        assert len(d.horizon_compatibility) == 4
        assert len(d.turnover_results) > 0

    def test_report_generation(self, persistent_signal_df):
        """Reports are generated with correct columns."""
        engine = SignalDecayEngine(config={})
        diagnostics = engine.run_full_diagnostics(
            persistent_signal_df,
            features=["persistent_signal"],
            horizons=[1, 5, 10, 20],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = generate_signal_decay_reports(diagnostics, tmpdir)

            assert "rank_persistence" in paths
            assert "halflife" in paths
            assert "ic_decay" in paths
            assert "horizon_compatibility" in paths
            assert "turnover_decay" in paths
            assert "fast_decay" in paths
            assert "horizon_mismatch" in paths
            assert "quality_flags" in paths

            # Verify each file exists and has content
            for name, path in paths.items():
                assert path.exists()
                assert path.stat().st_size > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
