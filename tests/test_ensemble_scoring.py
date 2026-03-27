import pickle

import pandas as pd

from utils.ensemble_scoring import compute_ensemble_score, load_ensemble_models


class DummyClassifier:
    def predict_proba(self, X):
        p = X["f_trend"].astype(float).clip(-1.0, 1.0)
        p = (p + 1.0) / 2.0
        return pd.DataFrame({0: 1.0 - p, 1: p}).values


class DummyRegressor:
    def predict(self, X):
        return X["ret_5d"].astype(float).values


class DummyMaRegressor:
    def predict(self, X):
        return X["ma_crossover"].astype(float).values


def test_ensemble_load_and_score(tmp_path):
    clf_path = tmp_path / "logistic.pkl"
    reg_path = tmp_path / "ridge.pkl"
    with open(clf_path, "wb") as fh:
        pickle.dump(
            {
                "estimator": DummyClassifier(),
                "feature_columns": ["f_trend"],
                "model_name": "Logistic Regression",
            },
            fh,
        )
    with open(reg_path, "wb") as fh:
        pickle.dump(
            {
                "estimator": DummyRegressor(),
                "feature_columns": ["ret_5d"],
                "model_name": "Ridge",
            },
            fh,
        )

    cfg = {
        "models": [
            {"path": str(clf_path), "weight": 0.7, "type": "classifier"},
            {"path": str(reg_path), "weight": 0.3, "type": "regressor"},
        ],
        "normalize": True,
        "clip": False,
    }
    models = load_ensemble_models(cfg)
    assert len(models) == 2

    feats = pd.DataFrame(
        {
            "f_trend": [-0.8, -0.2, 0.2, 0.9],
            "ret_5d": [-0.05, 0.01, 0.02, 0.08],
        },
        index=["A", "B", "C", "D"],
    )
    scores = compute_ensemble_score(feats, models, normalize=True, clip=False)
    assert isinstance(scores, pd.Series)
    assert set(scores.index) == {"A", "B", "C", "D"}
    assert scores.notna().all()
    assert scores["D"] > scores["A"]


def test_feature_alignment_uses_ma_crossover(tmp_path):
    path = tmp_path / "random_forest.pkl"
    with open(path, "wb") as fh:
        pickle.dump(
            {
                "estimator": DummyMaRegressor(),
                "feature_columns": ["ma_crossover"],
                "model_name": "Random Forest Classifier",
            },
            fh,
        )
    models = load_ensemble_models(
        {
            "models": [{"path": str(path), "weight": 1.0, "type": "regressor"}],
            "normalize": False,
            "clip": False,
        }
    )
    feats = pd.DataFrame({"ma_crossover": [0.0, 1.0, -1.0]}, index=["A", "B", "C"])
    scores = compute_ensemble_score(feats, models, normalize=False, clip=False)
    assert float(scores["B"]) == 1.0
    assert float(scores["C"]) == -1.0

