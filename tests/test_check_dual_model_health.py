from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import scripts.check_dual_model_health as cdmh


class DummyProbModel:
    def __init__(self, base: float) -> None:
        self.base = float(base)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        p = np.clip(self.base + 0.02 * np.tanh(x[:, 0]), 0.01, 0.99)
        return np.column_stack([1.0 - p, p])


def _write_config(path: Path, *, split_enabled: bool = True) -> None:
    path.write_text(
        "\n".join(
            [
                "tickers: [AAPL, MSFT]",
                "backtest:",
                "  end_date: '2026-03-01'",
                "model_selection:",
                f"  split_models:\n    enabled: {'true' if split_enabled else 'false'}",
                "signals:",
                f"  split_models:\n    enabled: {'true' if split_enabled else 'false'}\n    long_model_path: output/models/best_model_long.pkl\n    short_model_path: output/models/best_model_short.pkl",
                "governance:",
                "  dual_model_health:",
                "    enabled: true",
                "    max_artifact_age_hours: 9999",
                "    min_score_std: 0.00001",
                "    min_abs_score_ratio: 0.1",
                "    max_abs_score_ratio: 10.0",
                "    feature_lookback_days: 30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_artifact(path: Path, *, target: str, model_type: str, base: float) -> None:
    payload = {
        "estimator": DummyProbModel(base),
        "feature_columns": ["f1", "f2"],
        "model_type": model_type,
        "target": target,
        "horizon_days": 1,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(payload, fh)


def _seed_features() -> pd.DataFrame:
    rows = []
    for d in pd.date_range("2026-02-01", periods=10, freq="D"):
        for t in ("AAPL", "MSFT"):
            rows.append({"date": d, "ticker": t, "f1": 0.1 if t == "AAPL" else -0.2, "f2": 1.0})
    return pd.DataFrame(rows)


def _run(monkeypatch, tmp_path: Path, *extra: str) -> int:
    monkeypatch.setattr(cdmh, "ROOT", tmp_path)
    monkeypatch.chdir(tmp_path)
    argv = ["check_dual_model_health.py", "--config", "backtest_config.yaml", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return cdmh.main()


def test_dual_model_health_pass(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml", split_enabled=True)
    _write_artifact(tmp_path / "output" / "models" / "best_model_long.pkl", target="y_long", model_type="classifier", base=0.65)
    _write_artifact(tmp_path / "output" / "models" / "best_model_short.pkl", target="y_short", model_type="short_classifier", base=0.35)

    monkeypatch.setattr(cdmh, "build_feature_matrix", lambda *a, **k: _seed_features().copy())

    rc = _run(monkeypatch, tmp_path, "--strict")
    assert rc == 0

    latest = tmp_path / "output" / "live" / "dual_model_health" / "dual_model_health_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"


def test_dual_model_health_fail_missing_short_artifact(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml", split_enabled=True)
    _write_artifact(tmp_path / "output" / "models" / "best_model_long.pkl", target="y_long", model_type="classifier", base=0.65)

    monkeypatch.setattr(cdmh, "build_feature_matrix", lambda *a, **k: _seed_features().copy())

    rc = _run(monkeypatch, tmp_path, "--strict")
    assert rc == 2

    latest = tmp_path / "output" / "live" / "dual_model_health" / "dual_model_health_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "FAIL"
    assert "short_artifact_missing" in payload["failures"]


def test_dual_model_health_pass_when_split_disabled(monkeypatch, tmp_path: Path) -> None:
    _write_config(tmp_path / "backtest_config.yaml", split_enabled=False)

    rc = _run(monkeypatch, tmp_path, "--strict")
    assert rc == 0

    latest = tmp_path / "output" / "live" / "dual_model_health" / "dual_model_health_latest.json"
    payload = json.loads(latest.read_text(encoding="utf-8"))
    assert payload["status"] == "PASS"
    assert payload.get("reason") == "split_models_disabled"
