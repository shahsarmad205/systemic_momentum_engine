from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _write_cfg(path: Path, *, bootstrap_allow_no_data: bool) -> None:
    path.write_text(
        "\n".join(
            [
                "slippage_tracking:",
                "  enabled: true",
                "  trades_file: output/live/trades.csv",
                "  rolling_trades: 20",
                "  alert_threshold_bps: 10",
                "governance:",
                "  tca_health:",
                "    rolling_trades: 20",
                "    max_avg_slippage_bps: 10.0",
                "    max_p95_slippage_bps: 20.0",
                "    min_fills: 10",
                f"    bootstrap_allow_no_data: {'true' if bootstrap_allow_no_data else 'false'}",
                "    fail_on_no_data: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def test_strict_tca_allows_bootstrap_no_data(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    _write_cfg(cfg, bootstrap_allow_no_data=True)
    (tmp_path / "output" / "live").mkdir(parents=True, exist_ok=True)
    (tmp_path / "output" / "live" / "trades.csv").write_text("timestamp\n", encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/check_tca_health.py",
        "--config",
        str(cfg),
        "--strict",
    ]
    proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True)
    assert proc.returncode == 0


def test_strict_tca_fails_without_bootstrap(tmp_path: Path) -> None:
    cfg = tmp_path / "cfg.yaml"
    _write_cfg(cfg, bootstrap_allow_no_data=False)
    (tmp_path / "output" / "live").mkdir(parents=True, exist_ok=True)
    (tmp_path / "output" / "live" / "trades.csv").write_text("timestamp\n", encoding="utf-8")

    cmd = [
        sys.executable,
        "scripts/check_tca_health.py",
        "--config",
        str(cfg),
        "--strict",
    ]
    proc = subprocess.run(cmd, cwd=str(Path(__file__).resolve().parents[1]), capture_output=True, text=True)
    assert proc.returncode == 2
