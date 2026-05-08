import numpy as np
import pandas as pd
import warnings

from model_selection.training import (
    FeaturePreprocessor,
    TargetConfig,
    add_institutional_targets,
    make_training_target,
    retarget_panel_for_horizon,
)
from model_selection.validation import ExecutionCostConfig


def _training_frame() -> pd.DataFrame:
    dates = pd.bdate_range("2021-01-04", periods=6)
    rows = []
    for d_i, dt in enumerate(dates):
        for i in range(10):
            rows.append(
                {
                    "date": dt,
                    "ticker": f"T{i}",
                    "forward_return": (i - 4.5) * 0.003 + d_i * 0.0,
                    "capm_beta": 0.8 + i * 0.04,
                    "sector": "Tech" if i < 5 else "Industrials",
                    "adv_dollar_20": 100_000_000 - i * 1_000_000,
                    "realised_vol_20d": 0.02 + i * 0.001,
                    "feature_a": float(i),
                    "feature_b": 1.0,
                }
            )
    return pd.DataFrame(rows)


def test_targets_are_residual_net_of_cost_and_ranked_by_date() -> None:
    df = add_institutional_targets(
        _training_frame(),
        cfg=TargetConfig(horizon_days=5, residualize=True, net_of_costs=True),
        costs=ExecutionCostConfig(capital=10_000_000),
        max_name_weight=0.10,
    )

    assert {
        "target_return",
        "target_rank",
        "target_down_decile",
        "target_expected_cost",
        "target_expected_participation",
        "target_expected_temporary_impact",
        "target_expected_permanent_impact",
    }.issubset(df.columns)
    assert df["target_expected_cost"].ge(0.0).all()
    assert df["target_expected_participation"].between(0.0, 0.10).all()
    assert df["target_expected_temporary_impact"].ge(0.0).all()
    assert df["target_expected_permanent_impact"].ge(0.0).all()
    assert df.groupby("date")["target_return"].mean().abs().max() < 1e-10
    assert df["target_rank"].between(-1.0, 1.0).all()
    assert df.groupby("date")["target_down_decile"].sum().ge(1).all()


def test_target_residualization_handles_pathological_returns_without_runtime_warning() -> None:
    raw = _training_frame()
    raw.loc[raw.index[:3], "forward_return"] = 1e308
    raw.loc[raw.index[3:6], "capm_beta"] = np.inf

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        df = add_institutional_targets(
            raw,
            cfg=TargetConfig(horizon_days=5, residualize=True, net_of_costs=True),
            costs=ExecutionCostConfig(capital=10_000_000),
            max_name_weight=0.10,
        )

    assert np.isfinite(df["target_return"].to_numpy(dtype=float)).all()
    assert np.isfinite(df["target_return_net"].to_numpy(dtype=float)).all()
    assert df["target_return_net"].abs().max() <= 5.0
    assert np.isfinite(df["target_rank"].to_numpy(dtype=float)).all()


def test_feature_preprocessor_uses_train_schema_and_drops_constant_columns() -> None:
    train = _training_frame().iloc[:40].copy()
    test = _training_frame().iloc[40:].copy()
    test["feature_a"] = np.inf
    prep = FeaturePreprocessor.fit(train, ["feature_a", "feature_b", "missing_feature"], winsor_q=0.05)

    assert prep.active_features == ["feature_a"]
    x = prep.transform(test)

    assert x.shape == (len(test), 1)
    assert np.isfinite(x).all()
    assert x.max() <= float(prep.upper["feature_a"]) + 1e-12


def test_make_training_target_is_model_family_specific() -> None:
    df = add_institutional_targets(
        _training_frame(),
        cfg=TargetConfig(horizon_days=5),
        costs=ExecutionCostConfig(),
        max_name_weight=0.10,
    )

    reg = make_training_target(df, model_name="XGBRegressor", model_kind="regressor", use_risk_adj=False)
    rank = make_training_target(df, model_name="XGBRankIC", model_kind="regressor", use_risk_adj=False)
    short = make_training_target(df, model_name="ShortXGB", model_kind="short_classifier", use_risk_adj=False)
    cls = make_training_target(df, model_name="LogisticRegression", model_kind="classifier", use_risk_adj=False)

    assert reg.dtype.kind == "f"
    assert rank.min() >= -1.0 and rank.max() <= 1.0
    assert set(np.unique(short)).issubset({0, 1})
    assert set(np.unique(cls)).issubset({0, 1})


def test_retarget_panel_for_horizon_rebuilds_forward_returns_from_daily_returns() -> None:
    base = pd.DataFrame(
        {
            "date": pd.bdate_range("2021-01-04", periods=4),
            "ticker": ["T0"] * 4,
            "daily_return": [0.01, 0.02, 0.03, 0.04],
            "forward_return": [0.0] * 4,
            "capm_beta": [1.0] * 4,
            "sector": ["Tech"] * 4,
            "adv_dollar_20": [100_000_000.0] * 4,
            "realised_vol_20d": [0.02] * 4,
        }
    )

    out = retarget_panel_for_horizon(
        base,
        horizon_days=2,
        target_cfg=TargetConfig(horizon_days=2),
        costs=ExecutionCostConfig(capital=10_000_000),
        max_name_weight=0.10,
    )

    expected = (1.0 + 0.02) * (1.0 + 0.03) - 1.0
    assert np.isclose(float(out.loc[0, "forward_return"]), expected)
    assert np.isfinite(out["target_return"].to_numpy(dtype=float)).all()
