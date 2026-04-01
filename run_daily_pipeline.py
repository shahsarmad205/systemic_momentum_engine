#!/usr/bin/env python3
"""
Daily operational pipeline (run from ``trend_signal_engine/``):

1. Load ``backtest_config.yaml``.
2. Inspect OHLCV cache (``backtest.cache_dir``, default ``data/cache/ohlcv``): if the
   latest bar date is before the last completed US business day (not calendar
   ``yesterday``, so Mondays compare to Friday), log a warning and
   refresh data (extend cache via ``get_ohlcv``, then run ``build_feature_matrix`` to
   force a full aligned pull through the same path live scoring uses).
3. Optionally retrain if ``output/learned_weights.json`` is missing or older than
   ``pipeline.retrain_interval_days`` (default 30), by running a subprocess script
   (default ``run_retrain_baseline.py`` — set ``pipeline.retrain_script`` to
   ``run_model_selection.py`` if you prefer that heavy walk-forward job).
4. Run ``run_live_trading.py`` with ``--execute`` unless ``--dry-run`` is set.

Logs append to ``output/live/pipeline.log`` (same lines echoed to stdout).

Examples::

  python run_daily_pipeline.py
  python run_daily_pipeline.py --dry-run
  python run_daily_pipeline.py --no-retrain
  python run_daily_pipeline.py --force-retrain
  python run_daily_pipeline.py --live-extra '--date 2025-03-24'
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, TextIO

import pandas as pd
import yaml

from utils.artifacts import run_dir as _run_dir
from utils.artifacts import snapshot_config, snapshot_file, write_run_metadata
from utils.data_governance import (
    cache_covers_session,
    max_ohlcv_cache_bar_date,
    ohlcv_cache_dir,
    required_latest_cache_date,
)
from utils.run_context import ensure_run_id_in_env, format_ctx

_ROOT = Path(__file__).resolve().parent


def _require_python311() -> None:
    if sys.version_info[:2] != (3, 11):
        raise SystemExit(
            f"Python 3.11 required (CI/prod standard). Detected: {sys.version.split()[0]}"
        )


def _chdir_root() -> None:
    os.chdir(_ROOT)
    if str(_ROOT) not in sys.path:
        sys.path.insert(0, str(_ROOT))


def load_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def pipeline_log_path(cfg: dict[str, Any], root: Path) -> Path:
    pl_cfg = cfg.get("pipeline") or {}
    rel = pl_cfg.get("pipeline_log", "output/live/pipeline.log")
    return (root / rel).resolve()


def symbol_universe(cfg: dict[str, Any]) -> list[str]:
    raw = [str(t).strip() for t in (cfg.get("tickers") or []) if str(t).strip()]
    extra = ["SPY", "^VIX", "^VIX3M"]
    seen: set[str] = set()
    out: list[str] = []
    for t in raw + extra:
        if t.startswith("^"):
            n = "^" + t[1:].upper().replace(" ", "")
        else:
            n = t.upper().replace(" ", "")
        if n not in seen:
            seen.add(n)
            out.append(n)
    return out


def warm_ohlcv_cache(
    *,
    symbols: list[str],
    cache_dir: Path,
    provider: str,
    end_d: str,
    logf: TextIO,
    lookback_days: int = 450,
) -> None:
    """Extend parquet cache through ``end_d`` (merge download into cache)."""
    from utils.market_data import get_ohlcv

    end_ts = pd.Timestamp(end_d)
    start_ts = end_ts - pd.Timedelta(days=int(lookback_days))
    start_s = start_ts.strftime("%Y-%m-%d")
    end_s = end_ts.strftime("%Y-%m-%d")
    cache_s = str(cache_dir)
    for sym in symbols:
        log_line(logf, f"cache warm: {sym} {start_s} → {end_s} ({provider})")
        get_ohlcv(
            sym,
            start_s,
            end_s,
            provider=provider,
            cache_dir=cache_s,
            use_cache=True,
            cache_ttl_days=0,
        )


def run_feature_matrix_refresh(
    cfg: dict[str, Any],
    *,
    end_date: str,
    logf: TextIO,
) -> None:
    """Full feature build (same entry as live signals) to validate downloads."""
    from agents.weight_learning_agent.feature_builder import build_feature_matrix

    tickers = [str(t).strip() for t in (cfg.get("tickers") or []) if str(t).strip()]
    if not tickers:
        log_line(logf, "feature refresh skipped: no tickers in config")
        return
    start_ts = pd.Timestamp(end_date) - timedelta(days=400)
    start_date = start_ts.strftime("%Y-%m-%d")
    ms = cfg.get("model_selection") or {}
    horizon = int(ms.get("lookahead_horizon_days", 5) or 5)
    log_line(
        logf,
        f"build_feature_matrix({len(tickers)} tickers, {start_date} → {end_date}, h={horizon})",
    )
    build_feature_matrix(
        tickers,
        start_date=start_date,
        end_date=end_date,
        holding_period=horizon,
        feature_subset=None,
    )
    log_line(logf, "build_feature_matrix done")


def open_log(path: Path) -> TextIO:
    path.parent.mkdir(parents=True, exist_ok=True)
    return open(path, "a", encoding="utf-8")


def log_line(logf: TextIO, msg: str, *, structured: bool = True) -> None:
    ts = datetime.now().isoformat()
    suffix = (" " + format_ctx()).rstrip() if structured else ""
    line = f"[{ts}] {msg}{suffix}"
    logf.write(line + "\n")
    logf.flush()
    print(line, flush=True)


def learned_weights_path(cfg: dict[str, Any], root: Path) -> Path:
    pl = cfg.get("pipeline") or {}
    rel = pl.get("learned_weights_path", "output/learned_weights.json")
    return (root / rel).resolve()


def retrain_interval_days(cfg: dict[str, Any]) -> int:
    pl = cfg.get("pipeline") or {}
    return int(pl.get("retrain_interval_days", 30) or 30)


def retrain_script_name(cfg: dict[str, Any]) -> str:
    pl = cfg.get("pipeline") or {}
    return str(pl.get("retrain_script", "run_retrain_baseline.py") or "run_retrain_baseline.py")


def should_retrain(
    weights_path: Path,
    interval_days: int,
    *,
    no_retrain: bool,
    force_retrain: bool,
) -> bool:
    if no_retrain:
        return False
    if force_retrain:
        return True
    if not weights_path.is_file():
        return True
    age_days = (datetime.now().timestamp() - weights_path.stat().st_mtime) / 86400.0
    return age_days > float(interval_days)


def run_subprocess(
    argv: list[str],
    *,
    cwd: Path,
    logf: TextIO,
    label: str,
) -> None:
    log_line(logf, f"RUN {label}: {' '.join(argv)}")
    try:
        subprocess.run(argv, cwd=str(cwd), check=True)
    except subprocess.CalledProcessError as exc:
        log_line(logf, f"FAIL {label}: exit code {exc.returncode}")
        raise


def _write_preflight_incident_artifact(
    *,
    run_id: str,
    error: Exception,
    check_labels: list[str],
    failed_label: str,
    root: Path,
) -> Path:
    rd = _run_dir(run_id, root=root)
    incident_path = rd / "strict_preflight_incident.json"
    tb_tail = "\n".join(traceback.format_exception(type(error), error, error.__traceback__)[-50:])
    payload = {
        "run_id": run_id,
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "stage": "run_daily_pipeline.strict_preflight",
        "status": "FAIL",
        "failed_check_label": failed_label,
        "configured_checks": list(check_labels),
        "exception_type": type(error).__name__,
        "exception_message": str(error),
        "traceback_tail": tb_tail,
    }
    incident_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return incident_path


def run_governance_preflight(
    *,
    cfg: dict[str, Any],
    cfg_path: Path,
    cwd: Path,
    logf: TextIO,
) -> bool:
    from utils.trading_control import set_trading_halt_latch

    try:
        gov = cfg.get("governance") or {}
        pre = gov.get("preflight") or {}
        if not isinstance(pre, dict):
            pre = {}
        models_dir = cwd / "output" / "models"
        pointer_path = models_dir / "production_pointer.json"

        try:
            cfg_arg = cfg_path.relative_to(cwd)
        except ValueError:
            cfg_arg = cfg_path

        checks: list[tuple[str, list[str]]] = []
        shadow_enabled = bool(pre.get("run_shadow_monitor", True))
        bootstrap_allow_missing_production = bool(pre.get("bootstrap_allow_missing_production", True))
        if shadow_enabled and (not pointer_path.exists()) and bootstrap_allow_missing_production:
            log_line(logf, f"strict preflight: skipping shadow monitor (bootstrap; missing {pointer_path})")
            shadow_enabled = False

        if shadow_enabled:
            checks.append(
                (
                    "strict_preflight:shadow_monitor",
                    [
                        sys.executable,
                        str(cwd / "scripts" / "run_shadow_monitor.py"),
                        "--config",
                        str(cfg_arg),
                        "--strict",
                    ],
                )
            )
        if bool(pre.get("run_hard_limit_check", True)):
            target_rel = str(pre.get("hard_limit_target_csv", "output/live/target_executable_latest.csv") or "output/live/target_executable_latest.csv")
            target_path = Path(target_rel)
            if not target_path.is_absolute():
                target_path = (cwd / target_path).resolve()

            fresh_hours = float(pre.get("hard_limit_target_max_age_hours", 6.0) or 6.0)
            allow_skip_hard_limit = bool(pre.get("allow_skip_hard_limit_check", False))
            target_is_fresh = False
            if target_path.exists():
                age_hours = (datetime.now().timestamp() - float(target_path.stat().st_mtime)) / 3600.0
                target_is_fresh = age_hours <= fresh_hours
                if not target_is_fresh:
                    msg = f"strict preflight: hard-limit target stale age={age_hours:.2f}h > {fresh_hours:.2f}h"
                    if allow_skip_hard_limit:
                        log_line(logf, msg + "; skipping by config")
                    else:
                        raise RuntimeError(msg)
            else:
                msg = f"strict preflight: hard-limit target missing: {target_path}"
                if allow_skip_hard_limit:
                    log_line(logf, msg + "; skipping by config")
                else:
                    raise RuntimeError(msg)

            if target_is_fresh:
                target_arg = target_path
                try:
                    target_arg = target_path.relative_to(cwd)
                except ValueError:
                    pass
                checks.append(
                    (
                        "strict_preflight:hard_limit_check",
                        [
                            sys.executable,
                            str(cwd / "scripts" / "check_risk_limits.py"),
                            "--config",
                            str(cfg_arg),
                            "--target-csv",
                            str(target_arg),
                            "--strict",
                        ],
                    )
                )
        if bool(pre.get("run_tca_health_check", True)):
            checks.append(
                (
                    "strict_preflight:tca_health_check",
                    [
                        sys.executable,
                        str(cwd / "scripts" / "check_tca_health.py"),
                        "--config",
                        str(cfg_arg),
                        "--strict",
                    ],
                )
            )

        if not checks:
            if bool(pre.get("allow_empty_checks", False)):
                log_line(logf, "strict preflight enabled with no checks; continuing by config")
                return True
            raise RuntimeError("strict preflight enabled but no checks executed")

        failed_label = ""
        for label, argv in checks:
            failed_label = label
            run_subprocess(argv, cwd=cwd, logf=logf, label=label)

        latch_path = set_trading_halt_latch(
            cfg,
            active=False,
            reason="strict preflight passed",
            source="run_daily_pipeline.strict_preflight",
            details={"checks": [name for name, _ in checks]},
        )
        log_line(logf, f"strict preflight PASS; cleared halt latch at {latch_path}")
        return True
    except Exception as exc:
        # Best-effort incident artifact for postmortem; never block fail-closed latch.
        try:
            rid = (os.environ.get("RUN_ID") or "").strip() or ensure_run_id_in_env()
            check_labels = [name for name, _ in checks] if "checks" in locals() else []
            failed = failed_label if "failed_label" in locals() else ""
            incident_path = _write_preflight_incident_artifact(
                run_id=rid,
                error=exc,
                check_labels=check_labels,
                failed_label=failed,
                root=cwd,
            )
            log_line(logf, f"strict preflight incident artifact: {incident_path}")
        except Exception as incident_exc:
            log_line(logf, f"strict preflight incident artifact failed: {incident_exc}")

        latch_path = set_trading_halt_latch(
            cfg,
            active=True,
            reason=f"strict preflight failed: {exc}",
            source="run_daily_pipeline.strict_preflight",
        )
        log_line(logf, f"strict preflight FAIL; activated halt latch at {latch_path}")
        return False


def main() -> None:
    _require_python311()
    parser = argparse.ArgumentParser(description="Daily cache / retrain / live pipeline")
    parser.add_argument(
        "--config",
        type=str,
        default="backtest_config.yaml",
        help="Path to YAML config (relative to trend_signal_engine/)",
    )
    parser.add_argument(
        "--no-retrain",
        action="store_true",
        help="Skip model retraining even if due",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Force retraining subprocess",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Do not pass --execute to run_live_trading.py (no orders)",
    )
    parser.add_argument(
        "--skip-stale-refresh",
        action="store_true",
        help="Do not warm cache / rebuild features when OHLCV cache looks stale",
    )
    parser.add_argument(
        "--no-feature-refresh",
        action="store_true",
        help="Stale cache: only extend OHLCV parquet cache; skip build_feature_matrix",
    )
    parser.add_argument(
        "--live-extra",
        type=str,
        default="",
        help="Extra arguments for run_live_trading.py, as a single quoted string "
        '(e.g. \'--date 2025-03-24\')',
    )
    parser.add_argument(
        "--strict-preflight",
        action="store_true",
        help="Run strict governance checks (shadow monitor + hard-limit + TCA health) before live step; set/clear halt latch",
    )
    args = parser.parse_args()

    _chdir_root()
    rid = ensure_run_id_in_env()
    rd = _run_dir(rid, root=_ROOT)

    cfg_path = (_ROOT / args.config).resolve()
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(cfg_path)
    pl_cfg = cfg.get("pipeline") or {}
    strict_preflight_enabled = bool(args.strict_preflight or bool(pl_cfg.get("strict_preflight_default", False)))
    # Snapshot key inputs for auditability (best-effort; never fails the run).
    try:
        snaps: dict[str, Any] = {}
        snaps["config"] = snapshot_config(cfg_path, rd)
        snaps["requirements_lock"] = snapshot_file(_ROOT / "requirements-lock.txt", rd)
        snaps["learned_weights"] = snapshot_file(_ROOT / "output" / "learned_weights.json", rd)
        snaps["learned_weights_scaler"] = snapshot_file(_ROOT / "output" / "learned_weights_scaler.json", rd)
        write_run_metadata(
            rd,
            {
                "run_id": rid,
                "git_sha": format_ctx().split("git_sha=")[-1].split("]")[0] if "git_sha=" in format_ctx() else None,
                "argv": sys.argv,
                "dry_run": bool(args.dry_run),
                "no_retrain": bool(args.no_retrain),
                "force_retrain": bool(args.force_retrain),
                "snapshots": snaps,
            },
        )
    except Exception:
        pass
    log_path = pipeline_log_path(cfg, _ROOT)
    cache_dir = ohlcv_cache_dir(cfg, _ROOT)
    bt = cfg.get("backtest", cfg) or {}
    provider = str(bt.get("provider", "yahoo") or "yahoo")

    with open_log(log_path) as logf:
        log_line(logf, "=" * 60)
        log_line(logf, "pipeline start")
        log_line(
            logf,
            f"dry_run={args.dry_run} no_retrain={args.no_retrain} force_retrain={args.force_retrain} strict_preflight={strict_preflight_enabled}",
        )

        try:
            need = required_latest_cache_date()
            mx = max_ohlcv_cache_bar_date(cache_dir)
            mx_date = mx.normalize() if mx is not None else None

            if mx_date is None:
                log_line(
                    logf,
                    f"WARNING: no OHLCV cache in {cache_dir} (or no readable parquet); "
                    "refresh recommended.",
                )
                stale = True
            elif mx_date < need:
                log_line(
                    logf,
                    f"WARNING: OHLCV cache last bar {mx_date.date()} is before required "
                    f"{need.date()} — refreshing.",
                )
                stale = True
            else:
                log_line(
                    logf,
                    f"OHLCV cache OK: latest bar {mx_date.date()} (need >= {need.date()})",
                )
                stale = False

            if stale and not args.skip_stale_refresh:
                end_refresh = datetime.now().date().strftime("%Y-%m-%d")
                sym = symbol_universe(cfg)
                warm_ohlcv_cache(
                    symbols=sym,
                    cache_dir=cache_dir,
                    provider=provider,
                    end_d=end_refresh,
                    logf=logf,
                )
                end_fm = (datetime.now().date() - timedelta(days=1)).strftime("%Y-%m-%d")
                if not args.no_feature_refresh:
                    run_feature_matrix_refresh(cfg, end_date=end_fm, logf=logf)
                else:
                    log_line(logf, "skip build_feature_matrix (--no-feature-refresh)")

            elif stale and args.skip_stale_refresh:
                log_line(logf, "stale cache: --skip-stale-refresh set; not refreshing")

            pl_cfg = cfg.get("pipeline") or {}
            if bool(pl_cfg.get("fail_if_cache_stale_after_refresh", False)):
                mx_check = max_ohlcv_cache_bar_date(cache_dir)
                if not cache_covers_session(mx_check, need):
                    log_line(
                        logf,
                        "FATAL: OHLCV cache still stale vs required session "
                        f"(latest={mx_check}, need>={need.date()}). "
                        "Fix data pull or set pipeline.fail_if_cache_stale_after_refresh: false.",
                    )
                    sys.exit(1)

            w_path = learned_weights_path(cfg, _ROOT)
            interval = retrain_interval_days(cfg)
            script_name = retrain_script_name(cfg)
            retrain_py = (_ROOT / script_name).resolve()

            if not retrain_py.is_file():
                log_line(
                    logf,
                    f"WARNING: retrain script missing: {retrain_py} (skip retrain step)",
                )
            elif should_retrain(
                w_path,
                interval,
                no_retrain=args.no_retrain,
                force_retrain=args.force_retrain,
            ):
                try:
                    cfg_arg = cfg_path.relative_to(_ROOT)
                except ValueError:
                    cfg_arg = cfg_path
                argv = [sys.executable, str(retrain_py), "--config", str(cfg_arg)]
                run_subprocess(argv, cwd=_ROOT, logf=logf, label=f"retrain:{script_name}")
            else:
                log_line(
                    logf,
                    f"retrain skipped (weights mtime within {interval}d): {w_path}",
                )

            live_script = _ROOT / "run_live_trading.py"
            if not live_script.is_file():
                raise FileNotFoundError(f"Missing {live_script}")

            if strict_preflight_enabled:
                ok = run_governance_preflight(
                    cfg=cfg,
                    cfg_path=cfg_path,
                    cwd=_ROOT,
                    logf=logf,
                )
                if not ok:
                    log_line(logf, "FATAL: strict preflight failed; live step skipped")
                    sys.exit(1)

            live_argv = [sys.executable, str(live_script)]
            if not args.dry_run:
                live_argv.append("--execute")
            extra = shlex.split(args.live_extra) if args.live_extra.strip() else []
            live_argv.extend(extra)
            run_subprocess(live_argv, cwd=_ROOT, logf=logf, label="run_live_trading")

            log_line(logf, "pipeline finished OK")
            log_line(logf, "=" * 60)

        except SystemExit:
            raise
        except Exception:
            log_line(logf, "pipeline ABORTED with exception:\n" + traceback.format_exc())
            log_line(logf, "=" * 60)
            sys.exit(1)


if __name__ == "__main__":
    main()
