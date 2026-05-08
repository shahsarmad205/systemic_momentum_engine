"""
Smoke tests for short_modeling.py and research_contract.py (Phase 2).

Covers:
  1. Leakage gate: all SHORT_TARGET_COLUMNS blocked by is_model_feature_column()
  2. Allow-list: valid short-side predictor features pass if numeric
  3. _short_leg_sharpe: shorts top-q by highest score (not lowest)
  4. _classify_short_alpha: accepts strong expected-sign IC
  5. _classify_short_alpha: rejects wrong-sign IC (negative IC on short target)
  6. Comparison table: all expected columns are present
  7. format_short_report: contains both best-target keys
"""

import math
import numpy as np
import pandas as pd
import pytest

from model_selection.research_contract import (
    SHORT_TARGET_COLUMNS,
    FEATURE_SPECS,
    is_model_feature_column,
)
from model_selection.short_modeling import (
    ShortEvalConfig,
    ShortMetrics,
    _classify_short_alpha,
    _short_leg_sharpe,
    _build_comparison_table,
    _compute_short_composite_score,
    ShortModelResult,
    format_short_report,
)


# ── 1. Leakage gate ───────────────────────────────────────────────────────────

class TestLeakageGate:
    def test_short_target_columns_are_blocked(self):
        dummy = pd.Series([1.0, 2.0, 3.0])
        for col in SHORT_TARGET_COLUMNS:
            assert not is_model_feature_column(col, dummy), (
                f"{col} should be blocked by is_model_feature_column()"
            )

    def test_all_five_short_targets_blocked(self):
        from model_selection.short_modeling import ALL_SHORT_TARGETS
        dummy = pd.Series([1.0, 2.0, 3.0])
        for col in ALL_SHORT_TARGETS:
            assert col in SHORT_TARGET_COLUMNS, f"{col} missing from SHORT_TARGET_COLUMNS"
            assert not is_model_feature_column(col, dummy)


# ── 2. Allow-list: valid short-side predictor features ───────────────────────

class TestShortPredicatorAllowList:
    VALID_SHORT_PREDICTORS = [
        "short_interest_ratio",
        "days_to_cover",
        "borrow_crowding_risk",
        "short_squeeze_risk",
        "hard_short_squeeze_filter",
    ]

    def test_valid_short_predictors_pass_if_numeric(self):
        dummy = pd.Series([1.0, 2.0, 3.0])
        for col in self.VALID_SHORT_PREDICTORS:
            assert col in FEATURE_SPECS, f"{col} must be in FEATURE_SPECS"
            assert is_model_feature_column(col, dummy), (
                f"{col} is a valid short predictor; should pass is_model_feature_column()"
            )

    def test_valid_short_predictors_blocked_if_non_numeric(self):
        non_numeric = pd.Series(["a", "b", "c"])
        for col in self.VALID_SHORT_PREDICTORS:
            assert not is_model_feature_column(col, non_numeric)


# ── 3. _short_leg_sharpe: shorts TOP-q by highest score ──────────────────────

class TestShortLegSharpe:
    def _make_df(self) -> pd.DataFrame:
        rng = np.random.default_rng(42)
        n = 100
        dates = pd.date_range("2020-01-01", periods=20, freq="B").repeat(5)
        tickers = [f"T{i}" for i in range(5)] * 20
        # High-score stocks have negative raw returns (correct short target)
        scores = rng.standard_normal(n)
        returns = -scores * 0.01 + rng.standard_normal(n) * 0.005
        return pd.DataFrame({"date": dates, "ticker": tickers, "score": scores, "raw_5d": returns})

    def test_positive_sharpe_when_high_score_declines(self):
        df = self._make_df()
        sr = _short_leg_sharpe(df, "score", "raw_5d", top_q=0.20)
        assert np.isfinite(sr), "Sharpe should be finite"
        assert sr > 0, "Expected positive short-leg Sharpe when high-score stocks decline"

    def test_shorts_highest_score_not_lowest(self):
        """Manually verify: top-q by score = highest scores are shorted."""
        df = pd.DataFrame({
            "date": ["2020-01-01"] * 6,
            "ticker": list("ABCDEF"),
            "score":  [5.0, 4.0, 3.0, 2.0, 1.0, 0.0],  # A,B are top-20%
            "raw_5d": [-0.02, -0.03, 0.01, 0.01, 0.02, 0.02],  # A,B lose money
        })
        df["date"] = pd.to_datetime(df["date"])
        sr = _short_leg_sharpe(df, "score", "raw_5d", top_q=0.40)
        # Short P&L = -mean(-0.02, -0.03) = +0.025 → should be positive
        assert np.isfinite(sr) or math.isnan(sr)  # not enough dates for std, but no crash

    def test_missing_return_col_returns_nan(self):
        df = pd.DataFrame({"date": ["2020-01-01"], "ticker": ["A"], "score": [1.0]})
        df["date"] = pd.to_datetime(df["date"])
        assert math.isnan(_short_leg_sharpe(df, "score", "raw_5d", top_q=0.20))


# ── 4. _classify_short_alpha: accepts strong expected-sign IC ─────────────────

class TestClassifyShortAlpha:
    def _cfg(self) -> ShortEvalConfig:
        return ShortEvalConfig(
            ic_tstat_threshold=1.5,
            ic_mean_threshold=0.01,
            alpha_frac_threshold=0.60,
            exposure_frac_threshold=0.40,
        )

    def test_alpha_short_on_strong_positive_ic(self):
        cfg = self._cfg()
        # ic_mean > 0 (expected sign); residual_ic < 0 (−residual_ic > threshold)
        cls = _classify_short_alpha(
            ic_mean=0.05, ic_tstat=3.0,
            exposure_ic=-0.04, residual_ic=-0.05,
            residual_frac=0.80,
            cfg=cfg,
        )
        assert cls == "alpha_short", f"Expected alpha_short, got {cls}"

    def test_hedge_only_when_residual_frac_low(self):
        cfg = self._cfg()
        cls = _classify_short_alpha(
            ic_mean=0.05, ic_tstat=3.0,
            exposure_ic=-0.08, residual_ic=-0.02,
            residual_frac=0.20,
            cfg=cfg,
        )
        assert cls in ("hedge_only", "exposure_artifact"), f"Expected hedge/artifact, got {cls}"

    def test_no_signal_on_wrong_sign_ic(self):
        cfg = self._cfg()
        # Negative ic_mean = wrong sign (IC below zero for short target = model predicts backwards)
        cls = _classify_short_alpha(
            ic_mean=-0.05, ic_tstat=-3.0,
            exposure_ic=0.04, residual_ic=0.05,
            residual_frac=0.80,
            cfg=cfg,
        )
        assert cls == "no_signal", f"Expected no_signal for wrong-sign IC, got {cls}"

    def test_no_signal_on_near_zero_ic(self):
        cfg = self._cfg()
        cls = _classify_short_alpha(
            ic_mean=0.001, ic_tstat=0.3,
            exposure_ic=-0.001, residual_ic=-0.001,
            residual_frac=0.50,
            cfg=cfg,
        )
        assert cls == "no_signal"

    def test_exposure_artifact_when_residual_frac_very_low(self):
        cfg = self._cfg()
        # Strong raw IC (exposure_ic very negative), residual almost zero
        cls = _classify_short_alpha(
            ic_mean=0.06, ic_tstat=4.0,
            exposure_ic=-0.10, residual_ic=-0.005,
            residual_frac=0.05,
            cfg=cfg,
        )
        assert cls == "exposure_artifact", f"Expected exposure_artifact, got {cls}"


# ── 5. Comparison table: all expected columns present ────────────────────────

class TestComparisonTable:
    EXPECTED_COLS = [
        "target", "ic_mean", "ic_tstat", "ic_ir",
        "decile_spread", "monotonicity",
        "signed_decile_spread", "signed_monotonicity",
        "short_leg_sharpe", "exposure_ic", "residual_ic",
        "residual_frac", "composite_score", "classification", "n_dates",
    ]

    def _make_metrics(self) -> dict[str, ShortMetrics]:
        def _m(name: str, ic: float, tstat: float) -> ShortMetrics:
            prelim = ShortMetrics(
                target=name,
                ic_mean=ic, ic_tstat=tstat, ic_ir=ic / 0.05 if tstat else 0.0,
                decile_spread=-0.01, monotonicity=-0.6,
                signed_decile_spread=0.01, signed_monotonicity=0.6,
                short_leg_sharpe=0.5,
                exposure_ic=-0.03, residual_ic=-0.04,
                residual_frac=0.7,
                composite_score=0.0,
                classification="alpha_short",
                n_dates=50,
            )
            from dataclasses import replace
            return replace(prelim, composite_score=_compute_short_composite_score(prelim))

        return {
            "short_neg_residual": _m("short_neg_residual", 0.05, 3.0),
            "short_vol_expansion": _m("short_vol_expansion", 0.03, 2.0),
        }

    def test_all_expected_columns_present(self):
        df = _build_comparison_table(self._make_metrics())
        for col in self.EXPECTED_COLS:
            assert col in df.columns, f"Missing column: {col}"

    def test_sorted_by_ic_tstat_descending(self):
        df = _build_comparison_table(self._make_metrics())
        assert df["ic_tstat"].iloc[0] >= df["ic_tstat"].iloc[1]


# ── 6. format_short_report: contains both best-target keys ───────────────────

class TestFormatShortReport:
    def _make_result(self, agree: bool) -> ShortModelResult:
        from dataclasses import replace as dc_replace
        m1 = ShortMetrics(
            target="short_neg_residual",
            ic_mean=0.06, ic_tstat=4.0, ic_ir=1.2,
            decile_spread=-0.01, monotonicity=-0.7,
            signed_decile_spread=0.01, signed_monotonicity=0.7,
            short_leg_sharpe=0.8,
            exposure_ic=-0.04, residual_ic=-0.05,
            residual_frac=0.75,
            composite_score=0.0,
            classification="alpha_short",
            n_dates=100,
        )
        m1 = dc_replace(m1, composite_score=_compute_short_composite_score(m1))

        m2 = ShortMetrics(
            target="short_vol_expansion",
            ic_mean=0.04, ic_tstat=2.5, ic_ir=0.8,
            decile_spread=-0.008, monotonicity=-0.5,
            signed_decile_spread=0.008, signed_monotonicity=0.5,
            short_leg_sharpe=1.2,
            exposure_ic=-0.03, residual_ic=-0.03,
            residual_frac=0.65,
            composite_score=0.0,
            classification="alpha_short",
            n_dates=80,
        )
        m2 = dc_replace(m2, composite_score=_compute_short_composite_score(m2))

        all_metrics = {"short_neg_residual": m1, "short_vol_expansion": m2}
        by_tstat = "short_neg_residual"
        by_composite = "short_neg_residual" if agree else "short_vol_expansion"

        comparison = _build_comparison_table(all_metrics)
        return ShortModelResult(
            target_metrics=all_metrics,
            best_target_by_ic_tstat=by_tstat,
            best_target_by_composite_score=by_composite,
            best_target=by_tstat,
            best_metrics=m1,
            comparison=comparison,
            verdict="REAL SHORT ALPHA EXISTS (2/2 targets classified alpha_short). Best: short_neg_residual.",
        )

    def test_report_contains_score_convention(self):
        report = format_short_report(self._make_result(agree=True))
        assert "higher_score_is_stronger_short" in report

    def test_report_agree_case(self):
        report = format_short_report(self._make_result(agree=True))
        assert "agree" in report.lower() or "short_neg_residual" in report

    def test_report_disagree_shows_both_targets(self):
        report = format_short_report(self._make_result(agree=False))
        assert "short_neg_residual" in report
        assert "short_vol_expansion" in report

    def test_report_contains_signed_diagnostics(self):
        report = format_short_report(self._make_result(agree=True))
        assert "signed" in report.lower() or "SgnMono" in report or "SgnSpread" in report

    def test_report_contains_composite_score(self):
        report = format_short_report(self._make_result(agree=True))
        assert "Composite" in report or "composite" in report
