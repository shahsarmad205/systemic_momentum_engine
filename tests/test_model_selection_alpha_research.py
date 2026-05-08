from __future__ import annotations

import numpy as np
import pandas as pd
import warnings

from model_selection.alpha_research import (
    AlphaAdmissionConfig,
    TARGET_NET_RESIDUAL,
    _compute_bhy_tstat_threshold,
    _target_horizon_from_column,
    apply_admitted_feature_transforms,
    build_feature_admission,
    run_alpha_research,
)
from model_selection.statistics import bhy_adjust_pvalues
from model_selection.training import TargetConfig
from model_selection.validation import ExecutionCostConfig


def _synthetic_alpha_panel(*, inverse: bool = False, clone: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    dates = pd.bdate_range("2020-01-01", periods=80)
    tickers = [f"T{i:02d}" for i in range(30)]
    rows: list[dict[str, object]] = []
    latent_by_ticker: dict[str, list[float]] = {}
    for ticker in tickers:
        latent_by_ticker[ticker] = rng.normal(0.0, 1.0, size=len(dates)).tolist()

    for ticker in tickers:
        latent = latent_by_ticker[ticker]
        for i, dt in enumerate(dates):
            prev_signal = latent[i - 1] if i > 0 else 0.0
            daily_return = 0.002 * prev_signal + rng.normal(0.0, 0.0002)
            signal = latent[i]
            feature = -signal if inverse else signal
            row = {
                "date": dt,
                "ticker": ticker,
                "sector": "Tech" if int(ticker[1:]) % 2 else "Industrials",
                "regime_label": "Bull" if i < len(dates) // 2 else "Normal",
                "daily_return": daily_return,
                "adv_dollar_20": 50_000_000.0,
                "realised_vol_20d": 0.02,
                "capm_beta": 1.0,
                "forward_return": np.nan,
                "alpha_signal": feature,
            }
            if clone:
                row["alpha_clone"] = feature + rng.normal(0.0, 0.00001)
            rows.append(row)
    return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)


def _cfg() -> AlphaAdmissionConfig:
    return AlphaAdmissionConfig(
        horizons=(1, 2),
        production_horizon=1,
        min_coverage=0.80,
        min_abs_ic=0.01,
        min_ic_tstat=1.0,
        min_monotonicity=0.50,
        min_regime_stability=0.50,
        min_marginal_abs_ic=0.001,
        minimum_admitted_features=1,
    )


def _sparse_cross_section_panel() -> pd.DataFrame:
    rng = np.random.default_rng(11)
    dates = pd.bdate_range("2021-01-01", periods=20)
    tickers = [f"S{i:02d}" for i in range(10)]
    rows: list[dict[str, object]] = []
    for i, dt in enumerate(dates):
        for j, ticker in enumerate(tickers):
            signal = float(j - len(tickers) / 2)
            rows.append(
                {
                    "date": dt,
                    "ticker": ticker,
                    "sector": "Tech",
                    "regime_label": "Bull" if i < len(dates) // 2 else "Normal",
                    "daily_return": 0.001 * signal + rng.normal(0.0, 0.0001),
                    "adv_dollar_20": 25_000_000.0,
                    "realised_vol_20d": 0.03,
                    "capm_beta": 1.0,
                    "forward_return": np.nan,
                    "alpha_signal": signal,
                }
            )
    return pd.DataFrame(rows).sort_values(["ticker", "date"]).reset_index(drop=True)


def test_alpha_research_computes_decay_and_admits_current_horizon_feature() -> None:
    df = _synthetic_alpha_panel()
    enriched, decay, admission = run_alpha_research(
        df,
        ["alpha_signal"],
        cfg=_cfg(),
        target_cfg=TargetConfig(horizon_days=1),
        costs=ExecutionCostConfig(),
        max_name_weight=0.10,
    )

    assert f"{TARGET_NET_RESIDUAL}_1d" in enriched.columns
    assert set(decay["target_type"]) >= {"raw_return", "residual_return", "net_residual_return"}
    row = admission.set_index("feature").loc["alpha_signal"]
    assert bool(row["admitted"]) is True
    assert int(row["transform_sign"]) == 1


def test_alpha_research_target_horizon_parser_handles_canonical_labels() -> None:
    assert _target_horizon_from_column("raw_return_1d") == 1
    assert _target_horizon_from_column("residual_return_2d") == 2
    assert _target_horizon_from_column("net_residual_return_10d") == 10
    assert _target_horizon_from_column("target_return") == 1


def test_bhy_feature_admission_uses_tstats_without_double_scaling() -> None:
    decay = pd.DataFrame(
        [
            {
                "feature": f"f{i}",
                "target_type": TARGET_NET_RESIDUAL,
                "horizon_days": 10,
                "daily_spearman_ic_tstat": t,
                "ic_n_days": 250,
            }
            for i, t in enumerate([0.5, 1.0, 2.5, 3.2, 4.0])
        ]
    )

    threshold = _compute_bhy_tstat_threshold(decay, alpha=0.10, production_horizon=10)

    assert np.isfinite(threshold)
    assert 0.0 < threshold < 6.0


def test_bhy_adjusted_pvalues_are_restored_to_original_order() -> None:
    p_values = np.asarray([0.04, 0.001, 0.02])
    q_values = bhy_adjust_pvalues(p_values)

    assert q_values[1] < q_values[2] < q_values[0]


def test_alpha_admission_does_not_fallback_rank_when_fail_closed() -> None:
    decay = pd.DataFrame(
        [
            {
                "feature": "weak_signal",
                "target_type": TARGET_NET_RESIDUAL,
                "horizon_days": 10,
                "coverage": 1.0,
                "daily_spearman_ic": 0.001,
                "daily_spearman_ic_tstat": 0.1,
                "ic_n_days": 120,
                "regime_valid_days": 120,
                "spread_valid_days": 120,
                "monotonicity_valid_days": 120,
                "positive_monotonicity": 0.50,
                "inverted_monotonicity": 0.50,
                "regime_positive_rate": 0.50,
                "regime_inverted_positive_rate": 0.50,
                "evidence_available": True,
                "evidence_status": "available",
            }
        ]
    )

    admission = build_feature_admission(
        decay,
        pd.DataFrame(),
        ["weak_signal"],
        cfg=AlphaAdmissionConfig(
            production_horizon=10,
            min_abs_ic=0.01,
            min_ic_tstat=2.0,
            minimum_admitted_features=1,
            fail_if_below_minimum=True,
            fallback_mode="percentile",
            min_marginal_abs_ic=0.0,
            enforce_horizon_alignment=False,
        ),
    )

    row = admission.set_index("feature").loc["weak_signal"]
    assert bool(row["admitted"]) is False
    assert "fallback" not in str(row["recommended_action"])


def test_alpha_research_inverts_sign_flipped_feature_before_training() -> None:
    df = _synthetic_alpha_panel(inverse=True)
    _, _, admission = run_alpha_research(
        df,
        ["alpha_signal"],
        cfg=_cfg(),
        target_cfg=TargetConfig(horizon_days=1),
        costs=ExecutionCostConfig(),
        max_name_weight=0.10,
    )

    row = admission.set_index("feature").loc["alpha_signal"]
    assert bool(row["admitted"]) is True
    assert int(row["transform_sign"]) == -1
    transformed = apply_admitted_feature_transforms(df, admission)
    assert np.sign(transformed["alpha_signal"].corr(df["alpha_signal"])) < 0


def test_alpha_research_removes_redundant_clone_by_marginal_contribution() -> None:
    df = _synthetic_alpha_panel(clone=True)
    _, _, admission = run_alpha_research(
        df,
        ["alpha_signal", "alpha_clone"],
        cfg=_cfg(),
        target_cfg=TargetConfig(horizon_days=1),
        costs=ExecutionCostConfig(),
        max_name_weight=0.10,
    )

    admitted = admission.loc[admission["admitted"].eq(True), "feature"].tolist()
    removed = admission.loc[admission["reason"].eq("fails_marginal_contribution"), "feature"].tolist()
    assert len(admitted) == 1
    assert len(removed) == 1
    removed_row = admission.loc[admission["reason"].eq("fails_marginal_contribution")].iloc[0]
    assert str(removed_row["redundant_with"]) in {"alpha_signal", "alpha_clone"}
    assert float(removed_row["redundancy_max_abs_corr"]) > 0.95


def test_alpha_research_marginal_projection_handles_pathological_clone_without_runtime_warning() -> None:
    df = _synthetic_alpha_panel(clone=True)
    df.loc[df.index[:20], "alpha_clone"] = 1e308
    df.loc[df.index[20:40], "alpha_clone"] = -1e308

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _, _, admission = run_alpha_research(
            df,
            ["alpha_signal", "alpha_clone"],
            cfg=_cfg(),
            target_cfg=TargetConfig(horizon_days=1),
            costs=ExecutionCostConfig(),
            max_name_weight=0.10,
        )

    assert np.isfinite(pd.to_numeric(admission["marginal_ic"], errors="coerce").fillna(0.0)).all()


def test_alpha_research_marks_sparse_bucket_metrics_as_structurally_unavailable() -> None:
    df = _sparse_cross_section_panel()

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        _, decay, admission = run_alpha_research(
            df,
            ["alpha_signal"],
            cfg=_cfg(),
            target_cfg=TargetConfig(horizon_days=1),
            costs=ExecutionCostConfig(),
            max_name_weight=0.10,
        )

    decay_row = decay[
        decay["feature"].eq("alpha_signal")
        & decay["target_type"].eq(TARGET_NET_RESIDUAL)
        & decay["horizon_days"].eq(1)
    ].iloc[0]
    assert bool(decay_row["evidence_available"]) is False
    assert int(decay_row["spread_valid_days"]) == 0
    assert int(decay_row["monotonicity_valid_days"]) == 0
    assert "insufficient_spread_support" in str(decay_row["evidence_status"])
    assert "insufficient_monotonicity_support" in str(decay_row["evidence_status"])

    admission_row = admission.set_index("feature").loc["alpha_signal"]
    assert bool(admission_row["admitted"]) is False
    assert str(admission_row["reason"]).startswith("insufficient_production_evidence:")


def test_alpha_admission_moves_horizon_misaligned_feature_before_model_fitting() -> None:
    decay = pd.DataFrame(
        [
            {
                "feature": "momentum_1m_skip_eom",
                "target_type": TARGET_NET_RESIDUAL,
                "horizon_days": 10,
                "coverage": 1.0,
                "daily_spearman_ic": 0.05,
                "daily_spearman_ic_tstat": 5.0,
                "ic_n_days": 120,
                "regime_valid_days": 120,
                "spread_valid_days": 120,
                "monotonicity_valid_days": 120,
                "positive_monotonicity": 0.70,
                "inverted_monotonicity": 0.30,
                "regime_positive_rate": 0.80,
                "regime_inverted_positive_rate": 0.20,
                "evidence_available": True,
                "evidence_status": "available",
            }
        ]
    )
    admission = build_feature_admission(
        decay,
        pd.DataFrame(),
        ["momentum_1m_skip_eom"],
        cfg=AlphaAdmissionConfig(
            production_horizon=10,
            min_abs_ic=0.001,
            min_ic_tstat=0.5,
            min_ic_valid_days=20,
            min_regime_valid_days=20,
            min_spread_valid_days=20,
            min_monotonicity_valid_days=20,
            min_marginal_abs_ic=0.0,
            enforce_horizon_alignment=True,
            horizon_alignment_multiplier=2.0,
        ),
    )

    row = admission.set_index("feature").loc["momentum_1m_skip_eom"]
    assert bool(row["admitted"]) is False
    assert row["recommended_action"] == "move_horizon"
    assert int(row["selected_horizon_days"]) == 21
    assert str(row["reason"]).startswith("horizon_misaligned:21d>20")
