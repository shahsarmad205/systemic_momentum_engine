#!/usr/bin/env python3
"""
Industrial-style ops orchestrator for trend_signal_engine.

Runs a ordered sequence with unified logging:

1. **Preflight** — config paths, learned weights presence, optional Alpaca auth probe.
2. **Daily pipeline** — delegates to ``run_daily_pipeline.py`` (OHLCV / features, retrain, live).
3. **Performance tracker** — delegates to ``run_performance_tracker.py`` (IC, dashboards, alerts).

**Safety:** By default this passes ``--dry-run`` to the daily pipeline (no broker orders).
Use ``--execute`` only when you intend to place orders.

Usage (from ``trend_signal_engine/``)::

    python run_ops_suite.py                          # preflight + dry pipeline + tracker
    python run_ops_suite.py --execute                # live orders (use with care)
    python run_ops_suite.py --skip-preflight         # only pipeline + tracker
    python run_ops_suite.py --skip-tracker
    python run_ops_suite.py --no-retrain --dry-run   # forwards flags to daily pipeline

Logs append to ``output/live/ops_suite.log`` (and ``output/live/pipeline.log`` from the pipeline).
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any, TextIO

import yaml

_ROOT = Path(__file__).resolve().parent


def _chdir_root() -> None:
    os.chdir(_ROOT)
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))


def _ts() -> str:
    return datetime.now().isoformat()


def _open_ops_log(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", encoding="utf-8")


def _emit(logf: TextIO | None, msg: str, *, to_stdout: bool = True) -> None:
    line = f"[{_ts()}] {msg}"
    if logf is not None:
        logf.write(line + "\n")
        logf.flush()
    if to_stdout:
        print(line, flush=True)


def _load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _preflight(
    *,
    config_rel: str,
    logf: TextIO,
    strict_alpaca: bool,
) -> bool:
    """
    Return True if preflight passed (or non-strict warnings only).
    """
    ok = True
    cfg_path = (_ROOT / config_rel).resolve()
    _emit(logf, "preflight: start")

    if not cfg_path.is_file():
        _emit(logf, f"preflight: FAIL missing config {cfg_path}")
        return False

    try:
        cfg = _load_yaml(cfg_path)
    except Exception as exc:  # noqa: BLE001
        _emit(logf, f"preflight: FAIL cannot parse YAML: {exc}")
        return False

    bt = cfg.get("backtest", cfg) or {}
    cache_rel = str(bt.get("cache_dir", "data/cache/ohlcv"))
    cache_dir = (_ROOT / cache_rel).resolve()
    if not cache_dir.is_dir():
        _emit(
            logf,
            f"preflight: WARN OHLCV cache dir missing {cache_dir} (downloads may populate)",
        )

    sig = cfg.get("signals") or {}
    lw_rel = str(sig.get("learned_weights_path", "output/learned_weights.json"))
    lw_path = (_ROOT / lw_rel).resolve()
    if not lw_path.is_file():
        _emit(
            logf,
            f"preflight: WARN missing learned weights {lw_path} (signals may fail or use fallback)",
        )

    acfg = _ROOT / "config" / "alpaca_config.yaml"
    if not acfg.is_file():
        _emit(
            logf,
            f"preflight: WARN missing {acfg} (dry run may still write rankings)",
        )

    # Optional broker probe (real network)
    try:
        from alpaca_trade_api.rest import APIError as AlpacaAPIError

        from brokers.alpaca_broker import AlpacaBroker
    except ImportError as exc:
        _emit(logf, f"preflight: WARN Alpaca client not importable: {exc}")
        return ok

    try:
        broker = AlpacaBroker()
        _ = broker.get_account()
        _emit(logf, "preflight: Alpaca get_account OK")
    except AlpacaAPIError as exc:
        _emit(
            logf,
            f"preflight: Alpaca API error ({exc}) — fix keys/base_url for full execution",
        )
        if strict_alpaca:
            ok = False
    except (FileNotFoundError, ValueError) as exc:
        _emit(logf, f"preflight: WARN Alpaca not configured: {exc}")

    _emit(logf, f"preflight: {'PASS' if ok else 'FAIL'}")
    return ok


def _run_subprocess(
    argv: list[str],
    *,
    logf: TextIO,
    label: str,
    allow_nonzero: bool = False,
) -> int:
    _emit(logf, f"RUN {label}: {' '.join(argv)}")
    try:
        proc = subprocess.run(
            argv,
            cwd=str(_ROOT),
            check=False,
        )
    except OSError as exc:
        _emit(logf, f"FAIL {label}: {exc}")
        return 1
    if proc.returncode != 0:
        _emit(logf, f"FAIL {label}: exit code {proc.returncode}")
        if not allow_nonzero:
            return proc.returncode
    else:
        _emit(logf, f"OK {label}")
    return proc.returncode


def _build_pipeline_argv(ns: argparse.Namespace) -> list[str]:
    cmd: list[str] = [sys.executable, str(_ROOT / "run_daily_pipeline.py")]
    cmd.extend(["--config", ns.config])
    if not ns.execute:
        cmd.append("--dry-run")
    if ns.no_retrain:
        cmd.append("--no-retrain")
    if ns.force_retrain:
        cmd.append("--force-retrain")
    if ns.skip_stale_refresh:
        cmd.append("--skip-stale-refresh")
    if ns.no_feature_refresh:
        cmd.append("--no-feature-refresh")
    if ns.live_extra.strip():
        cmd.extend(["--live-extra", ns.live_extra.strip()])
    return cmd


def main() -> None:
    _chdir_root()

    parser = argparse.ArgumentParser(
        description="Ops suite: preflight + run_daily_pipeline + run_performance_tracker",
    )
    parser.add_argument(
        "--config",
        default="backtest_config.yaml",
        help="YAML config relative to trend_signal_engine/",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Allow real orders (omit --dry-run on daily pipeline). Default is dry-run.",
    )
    parser.add_argument("--skip-preflight", action="store_true")
    parser.add_argument("--skip-pipeline", action="store_true")
    parser.add_argument("--skip-tracker", action="store_true")
    parser.add_argument(
        "--strict-alpaca",
        action="store_true",
        help="Abort if Alpaca get_account fails (e.g. 401)",
    )
    parser.add_argument(
        "--strict-tracker",
        action="store_true",
        help="Exit non-zero if performance tracker fails",
    )
    # Forwarded to run_daily_pipeline.py
    parser.add_argument("--no-retrain", action="store_true")
    parser.add_argument("--force-retrain", action="store_true")
    parser.add_argument("--skip-stale-refresh", action="store_true")
    parser.add_argument("--no-feature-refresh", action="store_true")
    parser.add_argument(
        "--live-extra",
        default="",
        help="Quoted extra args for run_live_trading (via daily pipeline)",
    )
    parser.add_argument(
        "--tracker-extra",
        default="",
        help="Quoted extra CLI args for run_performance_tracker.py",
    )
    parser.add_argument(
        "--ops-log",
        default="output/live/ops_suite.log",
        help="Orchestrator log path (relative to trend_signal_engine/)",
    )

    ns = parser.parse_args()
    ops_log_path = (_ROOT / ns.ops_log).resolve()

    with _open_ops_log(ops_log_path) as logf:
        _emit(logf, "=" * 70)
        _emit(logf, "ops_suite start")
        _emit(logf, f"execute_orders={ns.execute} config={ns.config}")

        try:
            if not ns.skip_preflight:
                if not _preflight(
                    config_rel=ns.config,
                    logf=logf,
                    strict_alpaca=ns.strict_alpaca,
                ):
                    _emit(logf, "ops_suite ABORT (preflight)")
                    sys.exit(1)
            else:
                _emit(logf, "preflight skipped")

            if not ns.skip_pipeline:
                pv = _build_pipeline_argv(ns)
                rc = _run_subprocess(pv, logf=logf, label="daily_pipeline")
                if rc != 0:
                    _emit(logf, "ops_suite ABORT (pipeline)")
                    sys.exit(rc)
            else:
                _emit(logf, "pipeline skipped")

            if not ns.skip_tracker:
                tracker = [sys.executable, str(_ROOT / "run_performance_tracker.py")]
                extra = shlex.split(ns.tracker_extra) if ns.tracker_extra.strip() else []
                tracker.extend(extra)
                trc = _run_subprocess(
                    tracker,
                    logf=logf,
                    label="performance_tracker",
                    allow_nonzero=not ns.strict_tracker,
                )
                if trc != 0 and ns.strict_tracker:
                    _emit(logf, "ops_suite ABORT (tracker)")
                    sys.exit(trc)
                if trc != 0:
                    _emit(
                        logf,
                        "performance_tracker returned non-zero (ignored; use --strict-tracker to fail)",
                    )
            else:
                _emit(logf, "tracker skipped")

            _emit(logf, "ops_suite finished OK")
            _emit(logf, "=" * 70)

        except SystemExit:
            raise
        except Exception:
            _emit(logf, "ops_suite CRASH:\n" + traceback.format_exc())
            _emit(logf, "=" * 70)
            sys.exit(1)


if __name__ == "__main__":
    main()
