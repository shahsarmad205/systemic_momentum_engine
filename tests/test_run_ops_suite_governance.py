from __future__ import annotations

import argparse
import sys
from pathlib import Path

import run_ops_suite as ros


def _ns(**kwargs):
    base = {
        "config": "backtest_config.yaml",
        "execute": False,
        "no_retrain": False,
        "force_retrain": False,
        "skip_stale_refresh": False,
        "no_feature_refresh": False,
        "live_extra": "",
        "strict_governance": True,
    }
    base.update(kwargs)
    return argparse.Namespace(**base)


def test_build_pipeline_argv_enforces_strict_preflight_by_default() -> None:
    argv = ros._build_pipeline_argv(_ns())
    assert "--strict-preflight" in argv
    assert "--dry-run" in argv


def test_build_pipeline_argv_no_strict_governance_omits_flag() -> None:
    argv = ros._build_pipeline_argv(_ns(strict_governance=False))
    assert "--strict-preflight" not in argv


def test_main_runs_governance_summary_after_pipeline(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def _fake_run(argv: list[str], *, logf, label: str, allow_nonzero: bool = False) -> int:
        del argv, logf, allow_nonzero
        calls.append(label)
        return 0

    monkeypatch.setattr(ros, "_preflight", lambda **kwargs: True)
    monkeypatch.setattr(ros, "_run_subprocess", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ops_suite.py",
            "--skip-preflight",
            "--skip-tracker",
            "--ops-log",
            str(tmp_path / "ops.log"),
        ],
    )

    ros.main()
    assert calls == ["daily_pipeline", "governance_summary"]


def test_main_skip_pipeline_skips_governance_summary(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def _fake_run(argv: list[str], *, logf, label: str, allow_nonzero: bool = False) -> int:
        del argv, logf, allow_nonzero
        calls.append(label)
        return 0

    monkeypatch.setattr(ros, "_preflight", lambda **kwargs: True)
    monkeypatch.setattr(ros, "_run_subprocess", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ops_suite.py",
            "--skip-preflight",
            "--skip-pipeline",
            "--skip-tracker",
            "--ops-log",
            str(tmp_path / "ops.log"),
        ],
    )

    ros.main()
    assert calls == []


def test_main_governance_failure_aborts(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def _fake_run(argv: list[str], *, logf, label: str, allow_nonzero: bool = False) -> int:
        del argv, logf, allow_nonzero
        calls.append(label)
        if label == "governance_summary":
            return 2
        return 0

    monkeypatch.setattr(ros, "_preflight", lambda **kwargs: True)
    monkeypatch.setattr(ros, "_run_subprocess", _fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_ops_suite.py",
            "--skip-preflight",
            "--skip-tracker",
            "--ops-log",
            str(tmp_path / "ops.log"),
        ],
    )

    try:
        ros.main()
        raise AssertionError("expected SystemExit")
    except SystemExit as exc:
        assert int(exc.code) == 2
    assert calls == ["daily_pipeline", "governance_summary"]
