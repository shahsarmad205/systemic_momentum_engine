from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def _run_python(code: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )


def _module_import_status(module_name: str) -> dict[str, bool]:
    proc = _run_python(
        "import json, sys\n"
        f"import {module_name}\n"
        "print(json.dumps({"
        "'pandas_ta': 'pandas_ta' in sys.modules, "
        "'numba': 'numba' in sys.modules"
        "}))\n"
    )
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip())


def test_run_model_selection_import_smoke_does_not_load_pandas_ta_or_numba() -> None:
    status = _module_import_status("run_model_selection")
    assert status == {"pandas_ta": False, "numba": False}


def test_model_selection_validation_import_smoke_does_not_load_pandas_ta_or_numba() -> None:
    status = _module_import_status("model_selection.validation")
    assert status == {"pandas_ta": False, "numba": False}


def test_run_model_selection_help_succeeds_without_optional_indicator_stack() -> None:
    proc = subprocess.run(
        [sys.executable, "run_model_selection.py", "--help"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "usage: run_model_selection.py" in proc.stdout


def test_save_all_models_cli_option_is_removed() -> None:
    proc = subprocess.run(
        [sys.executable, "run_model_selection.py", "--save-all-models", "--run_sim_test"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert proc.returncode == 2
    assert "unrecognized arguments: --save-all-models" in proc.stderr


def test_phase_a2_deleted_private_helpers_are_not_exported() -> None:
    import run_model_selection as rms
    import model_selection.alpha_research as alpha_research
    import model_selection.ensemble_weighting as ensemble_weighting

    for name in [
        "_model_filename",
        "_safe_pearson",
        "_count_invested_days",
        "_chained_oos_metrics",
        "_is_economic_candidate_pool",
    ]:
        assert not hasattr(rms, name)
    assert not hasattr(alpha_research, "_regime_stability")
    assert not hasattr(ensemble_weighting, "_pairwise_signal_correlation")


def _research_args(**overrides: bool) -> SimpleNamespace:
    names = [
        "horizon_optimization",
        "confidence_weighting",
        "regime_gating",
        "asymmetry_correction",
        "capacity_analysis",
        "marginal_value",
        "cost_sensitivity",
        "joint_optimization",
        "deployability_ranking",
        "viability_check",
    ]
    values = {name: False for name in names}
    values.update(overrides)
    return SimpleNamespace(**values)


def _assert_research_route(monkeypatch, tmp_path: Path, args: SimpleNamespace, expected: list[str]) -> None:
    import run_model_selection as rms

    calls: list[str] = []
    out_dir = tmp_path / "models"
    out_dir.mkdir()
    (out_dir / "enriched_panel.parquet").write_text("already-present", encoding="utf-8")

    monkeypatch.setattr(rms, "_run_horizon_optimization", lambda *_args, **_kwargs: calls.append("horizon"))
    monkeypatch.setattr(rms, "_run_confidence_weighting", lambda *_args, **_kwargs: calls.append("confidence"))
    monkeypatch.setattr(rms, "_run_regime_gating", lambda *_args, **_kwargs: calls.append("regime"))
    monkeypatch.setattr(rms, "_run_asymmetry_correction", lambda *_args, **_kwargs: calls.append("asymmetry"))
    monkeypatch.setattr(rms, "_run_capacity_analysis", lambda *_args, **_kwargs: calls.append("capacity"))
    monkeypatch.setattr(rms, "_run_marginal_value_analysis", lambda *_args, **_kwargs: calls.append("marginal"))
    monkeypatch.setattr(rms, "_run_cost_sensitivity", lambda *_args, **_kwargs: calls.append("cost"))
    monkeypatch.setattr(rms, "_run_joint_optimization", lambda *_args, **_kwargs: calls.append("joint"))
    monkeypatch.setattr(rms, "_run_deployability_ranking", lambda *_args, **_kwargs: calls.append("deployability"))
    monkeypatch.setattr(rms, "_run_viability_check", lambda *_args, **_kwargs: calls.append("viability"))

    rms._run_optional_research_pillars(pd.DataFrame({"x": [1]}), out_dir, args, "cfg.yaml")

    assert calls == expected


def test_confidence_weighting_routes_independently(monkeypatch, tmp_path: Path) -> None:
    _assert_research_route(
        monkeypatch,
        tmp_path,
        _research_args(confidence_weighting=True),
        ["confidence"],
    )


def test_horizon_optimization_routes_without_confidence_weighting(monkeypatch, tmp_path: Path) -> None:
    _assert_research_route(
        monkeypatch,
        tmp_path,
        _research_args(horizon_optimization=True),
        ["horizon"],
    )


def test_horizon_and_confidence_weighting_route_in_deterministic_order(monkeypatch, tmp_path: Path) -> None:
    _assert_research_route(
        monkeypatch,
        tmp_path,
        _research_args(horizon_optimization=True, confidence_weighting=True),
        ["horizon", "confidence"],
    )
