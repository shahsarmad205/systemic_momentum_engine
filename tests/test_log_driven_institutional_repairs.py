import logging
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from model_selection.configuration import alpha_admission_config
from model_selection.cost_viability_engine import CostStatus
from model_selection.cost_viability_wiring import (
    FeatureCostResult,
    evaluate_feature_cost_viability,
    filter_feature_cost_results,
    feature_cost_results_to_horizon_contracts,
    summarize_feature_cost_gate,
)
from model_selection.horizon_contract import SweepMode, build_horizon_contract
from model_selection.horizon_eligibility import compute_ic_at_horizon, format_eligibility_report
from model_selection.horizon_gate import HorizonGate, HorizonGateConfig


def test_alpha_admission_uses_resolved_sweep_horizon_contract(caplog):
    cfg = {
        "horizon_config": {
            "target_horizon_days": 10,
            "holding_period_days": 10,
            "rebalance_frequency_days": 10,
            "allow_cross_horizon_evaluation": False,
        },
        "backtest": {
            "lookahead_horizon_days": 10,
            "holding_period_days": 10,
            "rebalance_every_trading_days": 10,
            "optimization_config": {"execution": {"horizon_days": 10}},
        },
        "model_selection": {
            "lookahead_horizon_days": 10,
            "production_horizons": [5, 10, 20, 63],
            "alpha_research": {
                "production_horizon": 10,
                "horizons": [5, 10, 20, 63],
            },
        },
    }
    contract = build_horizon_contract(
        cfg,
        active_horizon=63,
        sweep_mode=SweepMode.MULTI_HORIZON_SWEEP,
    )

    caplog.set_level(logging.WARNING, logger="model_selection.horizon_contract")
    caplog.clear()
    alpha_cfg = alpha_admission_config(cfg, horizon_contract=contract)

    assert alpha_cfg.production_horizon == 63
    assert not any("HORIZON CONTRACT ALIGNMENT FAILURE" in r.message for r in caplog.records)


def test_horizon_gate_ic_uses_daily_return_when_fwd_column_absent():
    rng = np.random.default_rng(7)
    dates = pd.date_range("2021-01-01", periods=40, freq="B")
    tickers = [f"T{i:02d}" for i in range(20)]
    rows = []
    for d_i, date in enumerate(dates):
        for t_i, ticker in enumerate(tickers):
            signal = (t_i - 10) / 10.0
            rows.append(
                {
                    "date": date,
                    "ticker": ticker,
                    "feature": signal,
                    "daily_return": 0.0005 + 0.01 * signal + rng.normal(0, 0.0001),
                }
            )
    panel = pd.DataFrame(rows).sample(frac=1.0, random_state=11).reset_index(drop=True)

    mean_ic, ic_series = compute_ic_at_horizon(panel, "feature", horizon=1)

    assert np.isfinite(mean_ic)
    assert mean_ic > 0.95
    assert len(ic_series) >= 5


def test_feature_cost_viability_consumes_alpha_admission_production_columns():
    alpha_admission = pd.DataFrame(
        [
            {
                "feature": "institutional_signal",
                "production_ic": 0.04,
                "production_ic_tstat": 3.2,
            }
        ]
    )
    alpha_decay = pd.DataFrame(
        [
            {
                "feature": "institutional_signal",
                "target_type": "net_residual_return",
                "horizon_days": 10,
                "daily_spearman_ic": 0.041,
                "daily_spearman_ic_tstat": 3.4,
                "signal_halflife_days": 30.0,
            },
            {
                "feature": "institutional_signal",
                "target_type": "raw_return",
                "horizon_days": 10,
                "daily_spearman_ic": 0.001,
                "daily_spearman_ic_tstat": 0.1,
                "signal_halflife_days": 2.0,
            },
        ]
    )
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=20, freq="B"),
            "ticker": ["A"] * 20,
            "daily_return": np.linspace(-0.01, 0.01, 20),
            "adv_dollar_20": [100_000_000.0] * 20,
        }
    )

    results = evaluate_feature_cost_viability(
        alpha_admission=alpha_admission,
        alpha_decay=alpha_decay,
        df=panel,
        cfg={},
        horizon=10,
    )

    assert len(results) == 1
    assert results[0].ic == 0.04
    assert results[0].ic_tstat == 3.2
    assert results[0].halflife == 30.0


def test_cost_viability_classification_thresholds_are_config_driven():
    alpha_admission = pd.DataFrame([{"feature": "institutional_signal", "production_ic": 0.10, "production_ic_tstat": 5.0}])
    alpha_decay = pd.DataFrame(
        [
            {
                "feature": "institutional_signal",
                "target_type": "net_residual_return",
                "horizon_days": 63,
                "daily_spearman_ic": 0.10,
                "daily_spearman_ic_tstat": 5.0,
                "signal_halflife_days": 126.0,
            }
        ]
    )
    panel = pd.DataFrame(
        {
            "date": pd.date_range("2021-01-01", periods=20, freq="B"),
            "ticker": ["A"] * 20,
            "daily_return": np.linspace(-0.01, 0.01, 20),
            "adv_dollar_20": [100_000_000.0] * 20,
        }
    )
    strict_cfg = {"model_selection": {"cost_viability": {"classification": {"min_adv_usd_viable": 200_000_000}}}}
    relaxed_cfg = {"model_selection": {"cost_viability": {"classification": {"min_adv_usd_viable": 1}}}}

    strict = evaluate_feature_cost_viability(alpha_admission, alpha_decay, panel, strict_cfg, 63)[0]
    relaxed = evaluate_feature_cost_viability(alpha_admission, alpha_decay, panel, relaxed_cfg, 63)[0]

    assert strict.cost_status == CostStatus.LIQUIDITY_INSUFFICIENT
    assert relaxed.cost_status != CostStatus.LIQUIDITY_INSUFFICIENT


def test_institutional_cost_contracts_support_legacy_horizon_gate_report():
    feature_results = [
        FeatureCostResult(
            feature="tradable_signal",
            family="momentum",
            horizon=63,
            ic=0.04,
            ic_tstat=3.0,
            halflife=90.0,
            expected_turnover=0.05,
            adv_usd=100_000_000.0,
            daily_vol=0.02,
            cost_status=CostStatus.COST_VIABLE,
            expected_alpha_bps=40.0,
            expected_cost_bps=5.0,
            net_expected_alpha_bps=35.0,
            alpha_cost_ratio=8.0,
            capacity_score=100.0,
            rejection_reason="",
        ),
        FeatureCostResult(
            feature="illiquid_signal",
            family="momentum",
            horizon=63,
            ic=0.03,
            ic_tstat=2.0,
            halflife=90.0,
            expected_turnover=0.05,
            adv_usd=2_000_000.0,
            daily_vol=0.02,
            cost_status=CostStatus.LIQUIDITY_INSUFFICIENT,
            expected_alpha_bps=30.0,
            expected_cost_bps=50.0,
            net_expected_alpha_bps=-20.0,
            alpha_cost_ratio=0.6,
            capacity_score=10.0,
            rejection_reason="adv_below_minimum",
        ),
    ]

    contracts = feature_cost_results_to_horizon_contracts(feature_results, horizon=63)
    report = format_eligibility_report(contracts, [63])
    gate = HorizonGate(
        contracts,
        config=HorizonGateConfig(
            min_production_features=1,
            min_families=1,
            max_family_concentration=1.0,
            min_effective_signals=1.0,
            use_production_level=True,
        ),
    ).evaluate(63)

    assert "tradable_signal" in report
    assert gate.eligible_features == ["tradable_signal"]
    assert gate.rejected_features["illiquid_signal"] == "liquidity_insufficient:adv_below_minimum"


def test_horizon_gate_reuses_production_cost_subset_without_threshold_changes():
    feature_results = [
        FeatureCostResult(
            feature="research_only_signal",
            family="momentum",
            horizon=63,
            ic=0.05,
            ic_tstat=3.0,
            halflife=90.0,
            expected_turnover=0.05,
            adv_usd=100_000_000.0,
            daily_vol=0.02,
            cost_status=CostStatus.COST_VIABLE,
            expected_alpha_bps=40.0,
            expected_cost_bps=5.0,
            net_expected_alpha_bps=35.0,
            alpha_cost_ratio=8.0,
            capacity_score=100.0,
            rejection_reason="",
        ),
        FeatureCostResult(
            feature="production_signal",
            family="quality",
            horizon=63,
            ic=0.04,
            ic_tstat=2.5,
            halflife=90.0,
            expected_turnover=0.05,
            adv_usd=2_000_000.0,
            daily_vol=0.02,
            cost_status=CostStatus.LIQUIDITY_INSUFFICIENT,
            expected_alpha_bps=30.0,
            expected_cost_bps=50.0,
            net_expected_alpha_bps=-20.0,
            alpha_cost_ratio=0.6,
            capacity_score=10.0,
            rejection_reason="adv_below_minimum",
        ),
    ]

    production_results = filter_feature_cost_results(feature_results, {"production_signal"})
    blocked, diagnostics = summarize_feature_cost_gate(
        production_results,
        {
            "model_selection": {
                "horizon_gate": {
                    "min_production_features": 1,
                    "min_families": 1,
                }
            }
        },
        63,
    )

    assert [r.feature for r in production_results] == ["production_signal"]
    assert blocked is True
    assert diagnostics["n_total"] == 1
    assert diagnostics["dominant_rejection_reasons"] == [("adv_below_minimum", 1)]


def test_feature_pipeline_import_does_not_load_pandas_ta_or_numba():
    code = (
        "import json, sys; "
        "import features.feature_pipeline; "
        "print(json.dumps({'pandas_ta': 'pandas_ta' in sys.modules, 'numba': 'numba' in sys.modules}))"
    )
    proc = subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
    assert json.loads(proc.stdout.strip()) == {"pandas_ta": False, "numba": False}
