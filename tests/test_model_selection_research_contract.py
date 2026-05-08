import numpy as np
import pandas as pd
import pytest

from model_selection.research_contract import (
    TimingContractViolation,
    audit_feature_contract,
    get_horizon_alignment_report,
    is_model_feature_column,
    summarize_feature_contract,
    validate_signal_execution_timing,
)


def _feature_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2022-01-03", periods=5)
    rows = []
    for dt in dates:
        for i in range(20):
            rows.append(
                {
                    "date": dt,
                    "ticker": f"T{i}",
                    "sector": "Tech",
                    "ret_5d": float(i),
                    "daily_return": i * 0.001,
                    "adv_dollar_20": 100_000_000.0,
                    "capm_beta": 1.0,
                    "forward_return": i * 0.002,
                    "target_return": i * 0.002,
                    "unknown_feature": -float(i),
                }
            )
    return pd.DataFrame(rows)


def test_model_feature_contract_excludes_risk_execution_and_targets() -> None:
    df = _feature_frame()

    assert is_model_feature_column("ret_5d", df["ret_5d"]) is True
    assert is_model_feature_column("daily_return", df["daily_return"]) is False
    assert is_model_feature_column("adv_dollar_20", df["adv_dollar_20"]) is False
    assert is_model_feature_column("capm_beta", df["capm_beta"]) is False
    assert is_model_feature_column("forward_return", df["forward_return"]) is False


def test_feature_research_ledger_reports_contract_and_monotonicity() -> None:
    df = _feature_frame()
    ledger = audit_feature_contract(df, ["ret_5d", "unknown_feature"], target_col="target_return")
    summary = summarize_feature_contract(ledger)

    ret_row = ledger.loc[ledger["feature"].eq("ret_5d")].iloc[0]
    unknown_row = ledger.loc[ledger["feature"].eq("unknown_feature")].iloc[0]

    assert bool(ret_row["known_contract"]) is True
    assert bool(unknown_row["known_contract"]) is False
    assert np.isfinite(float(ret_row["daily_spearman_ic"]))
    assert float(ret_row["quintile_spread"]) > 0.0
    assert summary["feature_known_contract_rate"] == 0.5


def test_signal_execution_timing_rejects_future_leaking_feature() -> None:
    df = _feature_frame()
    df["signal_time"] = df["date"]
    df["execution_time"] = df["date"] + pd.offsets.BDay(1)
    df["holding_period"] = 5
    df["ret_5d__available_at"] = df["date"] + pd.offsets.BDay(1)

    with pytest.raises(TimingContractViolation, match="ret_5d.*after signal_time"):
        validate_signal_execution_timing(
            df,
            ["ret_5d"],
            prediction_horizon_days=5,
        )


def test_signal_execution_timing_accepts_valid_close_t_features() -> None:
    df = _feature_frame()
    df["signal_time"] = df["date"]
    df["execution_time"] = df["date"] + pd.offsets.BDay(1)
    df["holding_period"] = 5
    df["ret_5d__available_at"] = df["date"]

    report = validate_signal_execution_timing(
        df,
        ["ret_5d"],
        prediction_horizon_days=5,
    )

    assert report.n_errors == 0
    assert report.contract.signal_time == "close_t"
    assert report.contract.execution_time == "next_open_or_next_vwap"


def test_signal_execution_timing_fails_on_enforced_horizon_misalignment() -> None:
    df = _feature_frame()
    df["signal_time"] = df["date"]
    df["execution_time"] = df["date"] + pd.offsets.BDay(1)
    df["holding_period"] = 5
    report = get_horizon_alignment_report(["quality_score"], 5, alignment_multiplier=2.0)

    with pytest.raises(TimingContractViolation, match="HorizonAlignment"):
        validate_signal_execution_timing(
            df,
            ["quality_score"],
            prediction_horizon_days=5,
            horizon_report=report,
            enforce_horizon_alignment=True,
        )
