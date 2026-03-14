import numpy as np
import pandas as pd

from agents.weight_learning_agent.weight_model import (
    WeightLearner,
    LearnedWeights,
    TARGET,
    COMPOUND_AND_PRICE_FEATURES,
)


def _make_synthetic_training_data(n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(123)
    dates = pd.date_range("2018-01-01", periods=n, freq="B")

    df = pd.DataFrame(index=dates)
    # Simple synthetic features: independent standard normals
    for col in COMPOUND_AND_PRICE_FEATURES:
        df[col] = rng.standard_normal(n)

    # Make target mainly driven by f_trend with some noise
    df[TARGET] = 0.5 * df["f_trend"] + 0.1 * rng.standard_normal(n)
    df["date"] = dates
    return df


def test_weight_learner_fit_produces_weights():
    df = _make_synthetic_training_data()

    learner = WeightLearner(model_type="ridge", target_type="regression")
    weights = learner.fit(df)

    assert isinstance(weights, LearnedWeights)
    # n_samples should match number of rows when 'date' is present
    assert weights.n_samples == len(df)
    # The learned trend weight should be non-zero and have the correct sign in expectation
    assert abs(weights.w_trend) > 0.01

