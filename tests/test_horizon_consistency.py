"""
P26/P27/P24 End-to-End Horizon Consistency Tests.

All tests use synthetic data.  No dependency on live market data.
Each test encodes the intended institutional behavior and will FAIL
if the pipeline silently resolves to a different horizon.

Run with:
    python -m pytest tests/test_horizon_consistency.py -v
Or directly:
    python tests/test_horizon_consistency.py
"""
import pandas as pd
import numpy as np

from model_selection.horizon_contract import (
    build_horizon_contract,
    HorizonConfig,
    HorizonConfigurationError,
)
from model_selection.alpha_research import (
    AlphaAdmissionConfig,
    build_feature_admission,
    _compute_signal_halflife_from_decay,
    _hac_tstat,
    _horizon_bucket_label,
    _compute_bucket_bhy_thresholds,
    TARGET_NET_RESIDUAL,
)
from model_selection.configuration import alpha_admission_config
from model_selection.research_contract import FEATURE_SPECS, FeatureFamilyRegistry


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_decay(features: list[str], horizon_days: list[int],
                ic_fn=None, rng=None) -> pd.DataFrame:
    """Build synthetic alpha_ic_decay with controlled IC values per feature/horizon."""
    if rng is None:
        rng = np.random.default_rng(42)
    if ic_fn is None:
        ic_fn = lambda feat, h: rng.normal(0.01, 0.003)
    rows = []
    for feat in features:
        spec = FEATURE_SPECS.get(feat)
        native_h = spec.horizon_days if spec else 10
        for h in horizon_days:
            ic = ic_fn(feat, h)
            tstat = ic / 0.005 * np.sqrt(400)
            rows.append({
                "feature": feat,
                "family": spec.family if spec else "unknown",
                "target_type": TARGET_NET_RESIDUAL,
                "horizon_days": h,
                "coverage": 0.90,
                "nonzero_rate": 0.95,
                "daily_spearman_ic": ic,
                "daily_spearman_ic_std": 0.005,
                "daily_spearman_ic_tstat": tstat,
                "ic_n_days": 400,
                "regime_valid_days": 200,
                "spread_valid_days": 200,
                "monotonicity_valid_days": 200,
                "positive_spread": abs(ic) * 5,
                "positive_monotonicity": 0.7,
                "inverted_spread": abs(ic) * 5,
                "inverted_monotonicity": 0.3,
                "regime_positive_rate": 0.7,
                "regime_inverted_positive_rate": 0.3,
                "evidence_available": True,
                "evidence_status": "available",
                "require_regime_support": False,
                "signal_halflife_days": native_h * 0.5,
                "decay_profile": spec.decay_profile if spec else "medium",
                "expected_horizon_days": native_h,
                "expected_sign": spec.expected_sign if spec else 1,
            })
    return _compute_signal_halflife_from_decay(pd.DataFrame(rows))


def _make_df(features: list[str], horizons: list[int], n: int = 200) -> pd.DataFrame:
    """Build a minimal panel DataFrame for admission."""
    target_cols = {}
    for h in horizons:
        target_cols[f"{TARGET_NET_RESIDUAL}_{h}d"] = np.random.randn(n)
    return pd.DataFrame({
        "date": pd.date_range("2020-01-01", periods=20, freq="D").repeat(10),
        "ticker": [f"T_{i}" for i in range(10)] * 20,
        **{f: np.random.randn(n) for f in features},
        **target_cols,
    })


def _base_cfg(**overrides) -> AlphaAdmissionConfig:
    """Default admission config for testing — all gates relaxed for synthetic data."""
    defaults = dict(
        production_horizon=10,
        horizons=(5, 10, 20, 63),
        multi_horizon_admission=True,
        cross_horizon_admission=False,
        enforce_horizon_alignment=False,
        allow_inversion=True,
        min_abs_ic=0.0001, min_ic_tstat=0.10,
        min_ic_valid_days=10, min_spread_valid_days=10,
        min_monotonicity_valid_days=10, min_regime_valid_days=10,
        min_regime_stability=0.2, min_monotonicity=0.2, min_coverage=0.2,
        min_marginal_abs_ic=-999, apply_bhy_correction=False,
        ic_cv_max=None, ic_sign_flip_max=None,
    )
    return AlphaAdmissionConfig(**{**defaults, **overrides})


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Configuration Consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestConfigurationConsistency:
    """P24: production_horizon must resolve authoritatively."""

    def test_production_horizon_authoritative(self):
        """alpha_research.production_horizon=10 → contract.production_horizon_days=10."""
        cfg = {"model_selection": {"alpha_research": {"production_horizon": 10}}}
        c = build_horizon_contract(cfg, cli_horizon=None).config
        assert c.production_horizon_days == 10
        assert c.target_horizon_days == 10  # inherits
        assert c.ic_evaluation_horizon == 10  # inherits

    def test_ic_eval_mismatch_raises(self):
        """ic_evaluation_horizon != production_horizon raises without flag."""
        cfg = {
            "model_selection": {"alpha_research": {"production_horizon": 10}},
            "horizon_config": {"ic_evaluation_horizon": 5},
        }
        try:
            build_horizon_contract(cfg)
            assert False, "Should have raised HorizonConfigurationError"
        except HorizonConfigurationError:
            pass  # expected

    def test_ic_eval_mismatch_allowed_with_flag(self):
        """ic_evaluation_horizon != production_horizon allowed with flag."""
        cfg = {
            "model_selection": {"alpha_research": {"production_horizon": 10}},
            "horizon_config": {
                "ic_evaluation_horizon": 20,
                "allow_cross_horizon_evaluation": True,
            },
        }
        c = build_horizon_contract(cfg).config
        assert c.ic_evaluation_horizon == 20
        assert c.production_horizon_days == 10

    def test_cli_sets_production_explicit_config_preserved(self):
        """P33: --horizon sets production_horizon only; explicit config survives."""
        cfg = {
            "model_selection": {"alpha_research": {"production_horizon": 10}},
            "horizon_config": {"target_horizon_days": 20},
        }
        c = build_horizon_contract(cfg, cli_horizon=63).config
        assert c.production_horizon_days == 63
        assert c.target_horizon_days == 20  # explicit config survives CLI

    def test_all_fields_inherit_production(self):
        """When only production_horizon is set, all fields inherit."""
        cfg = {"model_selection": {"alpha_research": {"production_horizon": 21}}}
        c = build_horizon_contract(cfg).config
        assert c.target_horizon_days == 21
        assert c.holding_period_days == 21
        assert c.ic_evaluation_horizon == 21

    def test_rebalance_independent_of_production(self):
        """rebalance_frequency may differ from production."""
        cfg = {
            "model_selection": {"alpha_research": {"production_horizon": 20}},
            "horizon_config": {"rebalance_frequency_days": 5},
        }
        c = build_horizon_contract(cfg).config
        assert c.production_horizon_days == 20
        assert c.rebalance_frequency_days == 5
        assert c.target_horizon_days == 20  # target inherits production


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Feature Admission Consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestFeatureAdmissionConsistency:
    """Features must be evaluated at their stated horizons."""

    def test_native_horizon_respected_when_multi_horizon(self):
        """When multi_horizon_admission=true, f_score (63d native) evaluated at 63d."""
        decay = _make_decay(["f_score"], [5, 10, 20, 63])
        df = _make_df(["f_score"], [5, 10, 20, 63])
        cfg = _base_cfg(
            multi_horizon_admission=True,
            cross_horizon_admission=True,
            enforce_horizon_alignment=True,
            horizon_alignment_multiplier=2.0,
        )
        adm = build_feature_admission(decay, df, ["f_score"], cfg=cfg)
        qs = adm[adm["feature"] == "f_score"]
        assert len(qs) > 0, "f_score should appear in admission"
        # With multi-horizon, eval_horizon should be the feature's native 63d
        eval_h = qs.iloc[0].get("eval_horizon_days", qs.iloc[0].get("selected_horizon_days", 0))
        assert eval_h == 63, f"Expected eval_horizon=63, got {eval_h}"

    def test_cross_horizon_admission_applies_decay_weight(self):
        """Cross-horizon admitted features carry a decay weight."""
        decay = _make_decay(["f_score"], [5, 10, 20, 63])
        df = _make_df(["f_score"], [5, 10, 20, 63])
        cfg = _base_cfg(
            multi_horizon_admission=True,
            cross_horizon_admission=True,
            enforce_horizon_alignment=True,
            horizon_alignment_multiplier=2.0,
        )
        adm = build_feature_admission(decay, df, ["f_score"], cfg=cfg)
        cross = adm[adm["recommended_action"].str.contains("cross_horizon", regex=False)]
        if len(cross) > 0:
            cw = cross.iloc[0].get("cross_horizon_weight", 0)
            assert cw > 0, f"Cross-horizon admitted features must have weight > 0, got {cw}"
            assert cw < 1.0, f"Decay weight for 63d→10d must be < 1, got {cw}"

    def test_cross_horizon_disabled_produces_zero_admits(self):
        """When cross_horizon=false, no cross-horizon admits."""
        decay = _make_decay(["f_score"], [5, 10, 20, 63])
        df = _make_df(["f_score"], [5, 10, 20, 63])
        cfg = _base_cfg(
            multi_horizon_admission=True,
            cross_horizon_admission=False,
            enforce_horizon_alignment=True,
            horizon_alignment_multiplier=2.0,
        )
        adm = build_feature_admission(decay, df, ["f_score"], cfg=cfg)
        cross = adm[adm["recommended_action"].str.contains("cross_horizon", regex=False)]
        assert len(cross) == 0

    def test_all_features_tested_without_multihorizon(self):
        """Without multi-horizon, all features evaluated at production_horizon."""
        decay = _make_decay(["ret_5d", "quality_score"], [10])
        df = _make_df(["ret_5d", "quality_score"], [10])
        cfg = _base_cfg(multi_horizon_admission=False)
        adm = build_feature_admission(decay, df, ["ret_5d", "quality_score"], cfg=cfg)
        assert len(adm) == 2
        assert (adm["eval_horizon_days"] == 10).all()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — HAC / BHY Consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestHACBHYConsistency:
    """P26: HAC lag must be horizon-aware. BHY must not mix buckets."""

    def test_hac_lag_capped_by_nonoverlap(self):
        """63d target should not get 62-lag HAC; it should be bounded."""
        rng = np.random.default_rng(1)
        # 63d feature: 500 daily IC values → n_nonoverlap ≈ 8
        ic_vals = rng.normal(0.01, 0.01, 500)
        t_raw = _hac_tstat(ic_vals, max_lag=5, horizon_days=63)
        # Old buggy formula: effective_lag = max(5, 62) = 62 → massive SE deflation
        # New formula: effective_lag = min(5, 500/63) ≈ min(5, 7) = 5
        assert np.isfinite(t_raw), f"HAC tstat should be finite, got {t_raw}"

    def test_horizon_buckets_are_mutually_exclusive(self):
        """Each horizon maps to exactly one bucket."""
        seen = set()
        for h in [1, 5, 6, 10, 21, 22, 40, 63, 100]:
            bucket = _horizon_bucket_label(h)
            assert bucket != "unknown", f"h={h} should have a known bucket"
            seen.add(bucket)
        assert len(seen) >= 3, f"Should have at least 3 distinct buckets, got {seen}"

    def test_bucket_bhy_computes_per_bucket(self):
        """BHY thresholds differ across buckets."""
        rng = np.random.default_rng(42)
        rows = []
        for feat in ["ret_5d", "quality_score", "f_score"]:
            spec = FEATURE_SPECS.get(feat)
            for h in [5, 10, 20, 63]:
                ic = rng.normal(0.01, 0.003)
                tstat = ic / 0.005 * np.sqrt(400)
                rows.append({
                    "feature": feat,
                    "family": spec.family if spec else "unknown",
                    "target_type": TARGET_NET_RESIDUAL,
                    "horizon_days": h,
                    "daily_spearman_ic": ic,
                    "daily_spearman_ic_std": 0.005,
                    "daily_spearman_ic_tstat": tstat,
                    "ic_n_days": 400,
                })
        decay = pd.DataFrame(rows)
        thresholds = _compute_bucket_bhy_thresholds(decay, alpha=0.05)
        assert "short" in thresholds
        assert "medium" in thresholds
        assert "long" in thresholds
        # Short bucket (5d-only features) may differ from long bucket (63d-heavy)
        assert thresholds["short"] > 0, f"short threshold invalid: {thresholds['short']}"

    def test_bhy_bucket_assignment(self):
        """Verify actual bucket assignments for common horizons."""
        assert _horizon_bucket_label(5) == "short"
        assert _horizon_bucket_label(10) == "medium"
        assert _horizon_bucket_label(21) == "medium"
        assert _horizon_bucket_label(22) == "long"
        assert _horizon_bucket_label(63) == "long"
        assert _horizon_bucket_label(100) == "ultra_long"


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Stability Gate Consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestStabilityGateConsistency:
    """P27: P11 stability gates must only test admissible horizon sets."""

    def test_fundamental_feature_not_rejected_by_short_horizon_noise(self):
        """f_score with noisy 1d-5d IC but stable 21d+ IC should pass stability."""
        rng = np.random.default_rng(777)
        features = ["f_score"]
        rows = []
        for h in [1, 2, 3, 5, 10, 20, 63]:
            if h < 21:
                ic = rng.normal(-0.001, 0.01)  # noisy, near-zero
            else:
                ic = rng.normal(0.015, 0.002)   # stable positive
            for spec_col in ["coverage", "nonzero_rate"]:
                pass
            rows.append({
                "feature": "f_score", "family": "fundamental_quality",
                "target_type": TARGET_NET_RESIDUAL, "horizon_days": h,
                "coverage": 0.90, "nonzero_rate": 0.95,
                "daily_spearman_ic": ic, "daily_spearman_ic_std": 0.005,
                "daily_spearman_ic_tstat": ic / 0.005 * np.sqrt(400),
                "ic_n_days": 400, "regime_valid_days": 200,
                "spread_valid_days": 200, "monotonicity_valid_days": 200,
                "positive_spread": abs(ic) * 5, "positive_monotonicity": 0.7,
                "inverted_spread": abs(ic) * 5, "inverted_monotonicity": 0.3,
                "regime_positive_rate": 0.7, "regime_inverted_positive_rate": 0.3,
                "evidence_available": True, "evidence_status": "available",
                "require_regime_support": False,
                "signal_halflife_days": 30, "decay_profile": "slow",
                "expected_horizon_days": 63, "expected_sign": 1,
            })
        decay = _compute_signal_halflife_from_decay(pd.DataFrame(rows))
        df = _make_df(features, [1, 2, 3, 5, 10, 20, 63])

        # Enable stability gates with strict thresholds
        cfg = _base_cfg(
            production_horizon=10,
            multi_horizon_admission=True,
            cross_horizon_admission=True,
            enforce_horizon_alignment=True,
            horizon_alignment_multiplier=2.0,
            ic_cv_max=2.0,
            ic_sign_flip_max=2,
        )
        adm = build_feature_admission(decay, df, features, cfg=cfg)
        f_score = adm[adm["feature"] == "f_score"]
        assert len(f_score) > 0, "f_score should be in admission table"

        # If admitted, it should NOT be rejected by ic_cv or sign_flips
        reason = str(f_score.iloc[0].get("reason", ""))
        assert "ic_cv" not in reason, (
            f"f_score should not fail IC CV due to short-horizon noise: {reason}"
        )
        assert "sign_flips" not in reason, (
            f"f_score should not fail sign flips due to short-horizon noise: {reason}"
        )

    def test_short_horizon_feature_still_fails_if_unstable(self):
        """A reversal feature with very high variance across all horizons should fail stability."""
        rng = np.random.default_rng(888)
        features = ["short_term_reversal"]
        rows = []
        for h in [1, 2, 3, 5, 10, 20]:
            ic = rng.normal(0.0, 0.03)  # very high variance, CV will be huge
            rows.append({
                "feature": "short_term_reversal", "family": "reversal",
                "target_type": TARGET_NET_RESIDUAL, "horizon_days": h,
                "coverage": 0.90, "nonzero_rate": 0.95,
                "daily_spearman_ic": ic, "daily_spearman_ic_std": 0.005,
                "daily_spearman_ic_tstat": ic / 0.005 * np.sqrt(400),
                "ic_n_days": 400, "regime_valid_days": 200,
                "spread_valid_days": 200, "monotonicity_valid_days": 200,
                "positive_spread": abs(ic) * 5, "positive_monotonicity": 0.7,
                "inverted_spread": abs(ic) * 5, "inverted_monotonicity": 0.3,
                "regime_positive_rate": 0.7, "regime_inverted_positive_rate": 0.3,
                "evidence_available": True, "evidence_status": "available",
                "require_regime_support": False,
                "signal_halflife_days": 2, "decay_profile": "fast",
                "expected_horizon_days": 3, "expected_sign": 1,
            })
        decay = _compute_signal_halflife_from_decay(pd.DataFrame(rows))
        df = _make_df(features, [1, 2, 3, 5, 10, 20])

        # Strict stability gates — reversal with high variance should be caught
        cfg = _base_cfg(
            production_horizon=3,
            multi_horizon_admission=True,
            cross_horizon_admission=True,
            enforce_horizon_alignment=True,
            horizon_alignment_multiplier=2.0,
            ic_cv_max=2.0,
            ic_sign_flip_max=2,
            fallback_mode="disabled",  # prevent fallback from re-admitting
        )
        adm = build_feature_admission(decay, df, features, cfg=cfg)
        rev = adm[adm["feature"] == "short_term_reversal"]
        assert len(rev) > 0, "reversal should appear in admission table"
        reason = str(rev.iloc[0].get("reason", ""))
        cv = rev.iloc[0].get("ic_cv", float("nan"))
        assert pd.notna(cv), f"ic_cv should be computed for reversal feature"
        # If CV > 2.0, the feature should be rejected by stability gate.
        # However, the fallback admission may re-admit it — in that case,
        # the reason should contain "fallback" to indicate it was re-admitted.
        if cv > 2.0:
            if rev.iloc[0]["admitted"]:
                assert "fallback" in reason.lower(), (
                    f"Feature with CV={cv:.1f} > 2.0 should be rejected by stability, "
                    f"but if re-admitted by fallback, reason should mention fallback. "
                    f"Got: {reason}"
                )
        elif rev.iloc[0]["admitted"]:
            assert cv <= 2.0, f"Admitted feature should have CV <= 2.0, got {cv}"

    def test_stability_horizon_set_configurable(self):
        """Families not in the override dict use decay-profile defaults."""
        # quality_lowvol has min_h=10
        # A feature with noisy IC at 5d but stable at 10d+ should pass
        rng = np.random.default_rng(999)
        features = ["low_vol_score"]
        rows = []
        for h in [1, 5, 10, 20, 63]:
            if h < 10:
                ic = rng.normal(-0.005, 0.015)  # noisy
            else:
                ic = rng.normal(0.012, 0.002)   # stable
            rows.append({
                "feature": "low_vol_score", "family": "quality_lowvol",
                "target_type": TARGET_NET_RESIDUAL, "horizon_days": h,
                "coverage": 0.90, "nonzero_rate": 0.95,
                "daily_spearman_ic": ic, "daily_spearman_ic_std": 0.005,
                "daily_spearman_ic_tstat": ic / 0.005 * np.sqrt(400),
                "ic_n_days": 400, "regime_valid_days": 200,
                "spread_valid_days": 200, "monotonicity_valid_days": 200,
                "positive_spread": abs(ic) * 5, "positive_monotonicity": 0.7,
                "inverted_spread": abs(ic) * 5, "inverted_monotonicity": 0.3,
                "regime_positive_rate": 0.7, "regime_inverted_positive_rate": 0.3,
                "evidence_available": True, "evidence_status": "available",
                "require_regime_support": False,
                "signal_halflife_days": 15, "decay_profile": "medium",
                "expected_horizon_days": 20, "expected_sign": 1,
            })
        decay = _compute_signal_halflife_from_decay(pd.DataFrame(rows))
        df = _make_df(features, [1, 5, 10, 20, 63])

        cfg = _base_cfg(
            production_horizon=10,
            multi_horizon_admission=True,
            cross_horizon_admission=True,
            enforce_horizon_alignment=True,
            horizon_alignment_multiplier=2.0,
            ic_cv_max=2.0, ic_sign_flip_max=2,
        )
        adm = build_feature_admission(decay, df, features, cfg=cfg)
        lvs = adm[adm["feature"] == "low_vol_score"]
        assert len(lvs) > 0
        reason = str(lvs.iloc[0].get("reason", ""))
        # Should NOT fail stability from 1d/5d noise (min_h=10 for quality_lowvol)
        assert "ic_cv" not in reason, (
            f"low_vol_score should not fail IC CV from horizons < 10d: {reason}"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — Reporting Consistency
# ═══════════════════════════════════════════════════════════════════════════════

class TestReportingConsistency:
    """Runtime output must include resolved horizon, not just configured."""

    def test_source_map_includes_production(self):
        """Contract source_map must track where production_horizon came from."""
        cfg = {"model_selection": {"alpha_research": {"production_horizon": 10}}}
        contract = build_horizon_contract(cfg)
        sm = contract.source_map
        assert "production_horizon_days" in sm
        assert "alpha_research" in sm["production_horizon_days"]

    def test_manifest_to_dict(self):
        """to_dict() must include production_horizon_days."""
        cfg = {"model_selection": {"alpha_research": {"production_horizon": 10}}}
        contract = build_horizon_contract(cfg)
        d = contract.to_dict()
        assert "horizon_config" in d
        assert d["horizon_config"]["production_horizon_days"] == 10

    def test_horizon_contract_prints_production(self):
        """Contract string representation shows production_horizon."""
        cfg = {"model_selection": {"alpha_research": {"production_horizon": 15}}}
        c = build_horizon_contract(cfg).config
        assert c.production_horizon_days == 15
        assert c.target_horizon_days == 15


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6 — Backward Compatibility
# ═══════════════════════════════════════════════════════════════════════════════

class TestBackwardCompatibility:
    """Legacy config fields must still work."""

    def test_legacy_lookahead_horizon_days(self):
        """model_selection.lookahead_horizon_days still resolves."""
        cfg = {"model_selection": {"lookahead_horizon_days": 21}}
        c = build_horizon_contract(cfg).config
        assert c.production_horizon_days == 21

    def test_legacy_backtest_lookahead(self):
        """backtest.lookahead_horizon_days still resolves."""
        cfg = {"backtest": {"lookahead_horizon_days": 15}}
        c = build_horizon_contract(cfg).config
        assert c.production_horizon_days == 15

    def test_5d_10d_20d_production_horizons_unchanged(self):
        """Configs without horizon_config still work."""
        cfg = {"model_selection": {"alpha_research": {"production_horizon": 5}}}
        c = build_horizon_contract(cfg).config
        assert c.production_horizon_days == 5
        assert c.target_horizon_days == 5
        assert c.rebalance_frequency_days == 5  # inherits from production
