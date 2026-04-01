from __future__ import annotations

import pandas as pd

from agents.weight_learning_agent.feature_builder import attach_directional_labels
from run_model_selection import _feature_columns


def test_attach_directional_labels_default_thresholds() -> None:
    df = pd.DataFrame({"forward_return": [-0.02, -0.0, 0.01, 0.03]})
    out = attach_directional_labels(df)

    assert out["y_long"].tolist() == [0, 0, 1, 1]
    assert out["y_short"].tolist() == [1, 0, 0, 0]


def test_attach_directional_labels_with_no_trade_band() -> None:
    df = pd.DataFrame({"forward_return": [-0.015, -0.004, 0.003, 0.02]})
    out = attach_directional_labels(
        df,
        long_positive_threshold=0.005,
        short_negative_threshold=0.01,
    )

    # Only strong positive/negative moves get directional class labels.
    assert out["y_long"].tolist() == [0, 0, 0, 1]
    assert out["y_short"].tolist() == [1, 0, 0, 0]


def test_feature_columns_excludes_split_targets() -> None:
    df = pd.DataFrame(
        {
            "date": ["2026-01-01"],
            "ticker": ["AAPL"],
            "forward_return": [0.01],
            "y_bin": [1],
            "y_long": [1],
            "y_short": [0],
            "ret_5d": [0.2],
            "rolling_vol_20": [0.1],
        }
    )

    cols = _feature_columns(df)
    assert "y_long" not in cols
    assert "y_short" not in cols
    assert "ret_5d" in cols
    assert "rolling_vol_20" in cols
