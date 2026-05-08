from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from model_selection.configuration import parallel_research_config
from model_selection.model_registry import constrain_model_parallelism


def test_parallel_research_config_bounds_workers_and_limits_nested_by_default() -> None:
    cfg = {"model_selection": {"parallel_research": {"enabled": True}}}

    out = parallel_research_config(cfg, n_models=8)

    assert out["enabled"] is True
    assert 1 <= int(out["model_workers"]) <= 4
    assert int(out["model_workers"]) <= 8
    if int(out["model_workers"]) > 1:
        assert int(out["nested_candidate_workers"]) == 1


def test_parallel_research_config_honors_explicit_overrides() -> None:
    cfg = {
        "model_selection": {
            "parallel_research": {
                "enabled": True,
                "model_workers": 3,
                "nested_candidate_workers": 2,
            }
        }
    }

    out = parallel_research_config(cfg, n_models=5)

    assert out == {
        "enabled": True,
        "model_workers": 3,
        "economic_model_workers": 1,
        "nested_candidate_workers": 2,
    }


def test_constrain_model_parallelism_caps_inner_estimators() -> None:
    models = [
        (
            "RF",
            RandomForestClassifier(n_estimators=10, random_state=42, n_jobs=-1),
            True,
            "classifier",
        ),
        (
            "PipeRF",
            Pipeline([("model", RandomForestClassifier(n_estimators=5, random_state=42, n_jobs=-1))]),
            True,
            "classifier",
        ),
    ]

    constrained = constrain_model_parallelism(models, max_jobs=1)

    assert constrained[0][1].get_params()["n_jobs"] == 1
    assert constrained[1][1].get_params()["model__n_jobs"] == 1


def test_constrain_model_parallelism_skips_logistic_n_jobs_mutation() -> None:
    models = [
        (
            "Logit",
            LogisticRegression(C=1.0, max_iter=100),
            True,
            "classifier",
        ),
        (
            "PipeLogit",
            Pipeline([("model", LogisticRegression(C=1.0, max_iter=100))]),
            True,
            "classifier",
        ),
    ]

    constrained = constrain_model_parallelism(models, max_jobs=1)

    assert constrained[0][1].get_params().get("n_jobs") is None
    assert constrained[1][1].get_params().get("model__n_jobs") is None
