from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.run_shadow_comparison as rsc


class DummyModel:
    def __init__(self, vals: list[float]) -> None:
        self._vals = np.array(vals, dtype=float)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        n = x.shape[0]
        p = self._vals[:n]
        p = np.clip(p, 0.0, 1.0)
        return np.column_stack([1.0 - p, p])


def _write_artifact(path: Path, *, values: list[float], model_type: str = "classifier") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "estimator": DummyModel(values),
        "feature_columns": ["f1", "f2"],
        "model_type": model_type,
        "model_name": "Dummy",
        "horizon_days": 1,
    }
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)


def _seed_config(path: Path) -> None:
    path.write_text(
        """
tickers: [AAPL, MSFT]
backtest:
  start_date: "2024-01-01"
  end_date: "2024-02-28"
governance:
  shadow_comparison:
    min_score_correlation: 0.80
    min_score_sign_agreement: 0.60
    max_score_mae: 0.30
    min_top_k_overlap_mean: 0.40
""".strip()
        + "\n",
        encoding="utf-8",
    )


def _seed_features() -> pd.DataFrame:
    rows = []
    for d in pd.date_range("2024-01-01", periods=6, freq="D"):
        for t in ("AAPL", "MSFT"):
            rows.append({"date": d, "ticker": t, "f1": 1.0, "f2": 2.0})
    return pd.DataFrame(rows)


def _run(monkeypatch, tmp_path: Path, *extra: str) -> int:
    monkeypatch.setattr(rsc, "ROOT", tmp_path)
    monkeypatch.setattr(rsc, "MODELS_DIR", tmp_path / "output" / "models")
    monkeypatch.setattr(rsc, "POINTER_PATH", tmp_path / "output" / "models" / "production_pointer.json")
    monkeypatch.setattr(rsc, "REGISTRY_PATH", tmp_path / "output" / "models" / "model_registry.json")
    monkeypatch.chdir(tmp_path)
    argv = ["run_shadow_comparison.py", "--config", "backtest_config.yaml", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return rsc.main()


def test_shadow_comparison_strict_pass(monkeypatch, tmp_path: Path) -> None:
    _seed_config(tmp_path / "backtest_config.yaml")
    features = _seed_features()

    prod = tmp_path / "output" / "models" / "prod.pkl"
    shad = tmp_path / "output" / "models" / "shadow.pkl"
    _write_artifact(prod, values=[0.8, 0.7, 0.75, 0.7, 0.82, 0.72, 0.78, 0.69, 0.76, 0.74, 0.79, 0.73])
    _write_artifact(shad, values=[0.79, 0.69, 0.74, 0.69, 0.81, 0.71, 0.77, 0.68, 0.75, 0.73, 0.78, 0.72])
    monkeypatch.setattr(rsc, "build_feature_matrix", lambda *a, **k: features.copy())

    rc = _run(
        monkeypatch,
        tmp_path,
        "--production-model-path",
        str(prod),
        "--shadow-model-path",
        str(shad),
        "--top-k",
        "2",
        "--strict",
    )
    assert rc == 0

    latest = tmp_path / "output" / "models" / "shadow_reports" / "latest_shadow_compare.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload["failures"] == []


def test_shadow_comparison_strict_fail(monkeypatch, tmp_path: Path) -> None:
    _seed_config(tmp_path / "backtest_config.yaml")
    features = _seed_features()

    prod = tmp_path / "output" / "models" / "prod.pkl"
    shad = tmp_path / "output" / "models" / "shadow.pkl"
    _write_artifact(prod, values=[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    _write_artifact(shad, values=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    monkeypatch.setattr(rsc, "build_feature_matrix", lambda *a, **k: features.copy())

    rc = _run(
        monkeypatch,
        tmp_path,
        "--production-model-path",
        str(prod),
        "--shadow-model-path",
        str(shad),
        "--top-k",
        "2",
        "--strict",
    )
    assert rc == 2

    latest = tmp_path / "output" / "models" / "shadow_reports" / "latest_shadow_compare.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert payload["failures"]


def test_shadow_comparison_non_strict_returns_zero_on_fail(monkeypatch, tmp_path: Path) -> None:
    _seed_config(tmp_path / "backtest_config.yaml")
    features = _seed_features()

    prod = tmp_path / "output" / "models" / "prod.pkl"
    shad = tmp_path / "output" / "models" / "shadow.pkl"
    _write_artifact(prod, values=[0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
    _write_artifact(shad, values=[0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
    monkeypatch.setattr(rsc, "build_feature_matrix", lambda *a, **k: features.copy())

    rc = _run(
        monkeypatch,
        tmp_path,
        "--production-model-path",
        str(prod),
        "--shadow-model-path",
        str(shad),
        "--top-k",
        "2",
    )
    assert rc == 0

    latest = tmp_path / "output" / "models" / "shadow_reports" / "latest_shadow_compare.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
