from __future__ import annotations

from model_selection.model_registry import (
    build_models,
    is_classifier_stock_selector,
    is_diagnostic_only,
)


def test_default_registry_excludes_binary_classifiers_for_stock_selection() -> None:
    models = build_models({"model_selection": {"model_tier": "core_plus_experimental"}})
    names = {name for name, *_ in models}
    kinds = {kind for *_, kind in models}

    assert "XGBRegressor" in names
    assert "XGBClassifier" not in names
    assert "LogisticRegression" not in names
    assert "ShortXGB" not in names
    assert "classifier" not in kinds
    assert "short_classifier" not in kinds


def test_explicit_classifier_registry_entries_are_diagnostic_only() -> None:
    models = build_models(
        {
            "model_selection": {
                "include_classifiers": True,
                "include_short_classifiers": True,
                "model_tier": "core_plus_experimental",
            }
        }
    )
    by_name = {name: kind for name, _, _, kind in models}

    assert by_name["XGBClassifier"] == "classifier"
    assert by_name["ShortXGB"] == "short_classifier"
    assert is_classifier_stock_selector("XGBClassifier", by_name["XGBClassifier"])
    assert is_classifier_stock_selector("ShortXGB", by_name["ShortXGB"])
    assert is_diagnostic_only("XGBClassifier")
    assert is_diagnostic_only("ShortXGB")
