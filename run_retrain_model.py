#!/usr/bin/env python3
"""
Dynamic model retraining (scheduled or on-demand).

Reads ``retraining:`` from ``backtest_config.yaml`` and either:

- **learned_weights** — runs ``run_retrain_baseline.py`` (ridge ``WeightLearner`` + scaler), or
- **ml** — runs ``run_model_selection.py`` walk-forward selection and writes ``output/models/best_model.pkl``.

**Validation (learned_weights):** By default the last ``validation_holdout_calendar_days`` (e.g. 252) of
data are **excluded from the fit**; acceptance runs backtests only on that **out-of-sample** tail,
optionally split into ``validation_walk_forward_windows`` segments so the decision is less sensitive
to one noisy span. This avoids comparing a “0.92” full-backtest Sharpe to an in-sample quick check.

Logs append to ``output/live/retrain.log`` by default; failures can alert via ``monitoring:``.

Examples (from ``trend_signal_engine/``)::

  python run_retrain_model.py --verbose
  python run_retrain_model.py --force
  python run_retrain_model.py --model-type ml
  python run_retrain_model.py --as-of 2026-03-20 --no-validate
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import statistics
import subprocess
import sys
import traceback
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from utils.run_context import ensure_run_id_in_env, format_ctx


def _read_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _setup_logging(log_file: Path, *, verbose: bool) -> None:
    log_file.parent.mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG if verbose else logging.INFO)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.DEBUG if verbose else logging.INFO)
    sh.setFormatter(fmt)
    root.addHandler(fh)
    root.addHandler(sh)


def _load_retraining(cfg: dict[str, Any]) -> dict[str, Any]:
    raw = cfg.get("retraining")
    return raw if isinstance(raw, dict) else {}


def _matrix_start_date(train_end: str, *, train_start_floor: str, window_years: float) -> str:
    end_ts = pd.Timestamp(train_end)
    floor_ts = pd.Timestamp(str(train_start_floor).strip() or "2018-01-01")
    window_start = end_ts - pd.DateOffset(months=int(round(float(window_years) * 12)))
    return max(window_start, floor_ts).strftime("%Y-%m-%d")


def _should_skip_recent_success(
    *,
    root: Path,
    skip_days: int,
    force: bool,
    log: logging.Logger,
) -> bool:
    if force or skip_days <= 0:
        return False
    meta_path = root / "output" / "retrain_model_last.json"
    if not meta_path.is_file():
        return False
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("status") != "ok":
            return False
        ts = str(meta.get("timestamp", "")).strip()
        if not ts:
            return False
        last = pd.Timestamp(ts)
        if getattr(last, "tzinfo", None) is not None:
            last = last.tz_localize(None)
        now = pd.Timestamp.now()
        d = float((now - last).total_seconds()) / 86400.0
        if d < skip_days:
            log.info("Skip retrain: last success %.1f days ago (< %d).", float(d), skip_days)
            return True
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not read retrain_model_last.json: %s", exc)
    return False


def _alert_failure(message: str, *, config_path: Path) -> None:
    try:
        from run_performance_tracker import MonitoringConfig, send_alert

        mon = MonitoringConfig()
        try:
            with open(config_path, encoding="utf-8") as fh:
                cfg = yaml.safe_load(fh) or {}
            raw = cfg.get("monitoring") or {}
            if isinstance(raw, dict):
                mon.alert_method = str(raw.get("alert_method", mon.alert_method) or "print").strip().lower()
                mon.email_recipient = str(raw.get("email_recipient", mon.email_recipient) or "").strip()
                mon.slack_webhook = str(raw.get("slack_webhook", mon.slack_webhook) or "").strip()
        except Exception:
            pass
        subj = f"[TrendSignalEngine] Model retrain failed — {datetime.now().strftime('%Y-%m-%d %H:%M')}"
        send_alert(message, subject=subj, mon=mon)
    except Exception as exc:  # noqa: BLE001
        print(f"[retrain] alert dispatch failed: {exc}", file=sys.stderr)
        print(message, file=sys.stderr)


def _snapshot_model_release(
    *,
    root: Path,
    train_end: str,
    detail: dict[str, Any],
    model_type: str,
    rt: dict[str, Any],
    weights: Path | None,
    scaler: Path | None,
    ml_pkl: Path | None,
) -> None:
    if not bool(rt.get("release_snapshot_on_success", True)):
        return
    rel_base = Path(str(rt.get("releases_dir", "output/models/releases") or "output/models/releases"))
    if not rel_base.is_absolute():
        rel_base = root / rel_base
    stamp = train_end.replace("-", "")
    dest = rel_base / stamp
    dest.mkdir(parents=True, exist_ok=True)
    payload = {**detail, "release_dir": str(dest.relative_to(root))}
    (dest / "retrain_detail.json").write_text(
        json.dumps(payload, indent=2, default=str),
        encoding="utf-8",
    )
    if model_type in ("learned_weights", "learned", "weights"):
        if weights is not None and weights.is_file():
            shutil.copy2(weights, dest / weights.name)
        if scaler is not None and scaler.is_file():
            shutil.copy2(scaler, dest / scaler.name)
    elif model_type == "ml" and ml_pkl is not None and ml_pkl.is_file():
        shutil.copy2(ml_pkl, dest / ml_pkl.name)
        meta = ml_pkl.with_suffix(".meta.json")
        if meta.is_file():
            shutil.copy2(meta, dest / meta.name)


def _restore_learned_weights_backups(
    weights: Path,
    scaler: Path,
    suffix: str,
    log: logging.Logger,
) -> None:
    bw = weights.with_name(weights.name + suffix)
    bs = scaler.with_name(scaler.name + suffix)
    if bw.is_file():
        shutil.copy2(bw, weights)
        log.warning("Restored weights from %s", bw)
    if bs.is_file():
        shutil.copy2(bs, scaler)
        log.warning("Restored scaler from %s", bs)


def main() -> int:
    parser = argparse.ArgumentParser(description="Retrain learned weights or ML model per backtest_config.yaml.")
    parser.add_argument("--config", type=Path, default=_ROOT / "backtest_config.yaml")
    parser.add_argument("--force", action="store_true", help="Run even if retraining.enabled is false.")
    parser.add_argument(
        "--model-type",
        type=str,
        default="",
        help="Override retraining.model_type: learned_weights | ml",
    )
    parser.add_argument(
        "--as-of",
        type=str,
        default="",
        help="Training end date YYYY-MM-DD (default: yesterday local).",
    )
    parser.add_argument("--no-validate", action="store_true", help="Skip post-train Sharpe validation (learned only).")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    cfg_path = args.config
    if not cfg_path.is_file():
        print(f"Config not found: {cfg_path}", file=sys.stderr)
        return 1

    os.chdir(_ROOT)
    ensure_run_id_in_env()

    cfg = _read_config(cfg_path)
    rt = _load_retraining(cfg)
    log_path = Path(str(rt.get("log_path", "output/live/retrain.log")))
    if not log_path.is_absolute():
        log_path = _ROOT / log_path
    _setup_logging(log_path, verbose=bool(args.verbose))
    log = logging.getLogger("run_retrain_model")
    log.info("retrain start %s", format_ctx())

    if not rt.get("enabled", False) and not args.force:
        log.info("retraining.enabled is false (use --force to run anyway). Exiting 0.")
        return 0

    model_type = (args.model_type or rt.get("model_type", "learned_weights") or "learned_weights").strip().lower()
    train_end = str(args.as_of).strip()
    if not train_end:
        train_end = (datetime.now().date() - timedelta(days=1)).isoformat()

    train_start_floor = str(rt.get("train_start_date", "2018-01-01") or "2018-01-01").strip()
    window_years = float(rt.get("train_window_years", 5) or 5)
    skip_recent = int(rt.get("skip_if_trained_within_days", 0) or 0)
    if _should_skip_recent_success(root=_ROOT, skip_days=skip_recent, force=bool(args.force), log=log):
        return 0

    backup_suffix = str(rt.get("backup_suffix", ".before_retrain") or ".before_retrain")
    validate = bool(rt.get("validate_with_backtest", True)) and not args.no_validate
    val_days = int(rt.get("validation_backtest_days", 380) or 380)
    min_sharpe_ratio = float(rt.get("min_sharpe_ratio_vs_previous", 0.85) or 0.85)
    holdout_days = int(rt.get("validation_holdout_calendar_days", 252) or 252)
    n_val_windows = int(rt.get("validation_walk_forward_windows", 1) or 1)
    accept_if_better_on_windows = int(rt.get("accept_if_better_on_windows", 0) or 0)
    compare_last_accepted = bool(rt.get("compare_to_last_accepted_mean", False))
    min_ratio_last_accepted = float(rt.get("min_sharpe_ratio_vs_last_accepted", 0.85) or 0.85)
    last_accepted_path = Path(
        str(rt.get("last_accepted_oos_path", "output/retraining_last_accepted_oos.json"))
    )
    if not last_accepted_path.is_absolute():
        last_accepted_path = _ROOT / last_accepted_path
    alert_on_failure = bool(rt.get("alert_on_failure", True))

    out_weights = Path(str(rt.get("output_path", "output/learned_weights.json") or "output/learned_weights.json"))
    if not out_weights.is_absolute():
        out_weights = _ROOT / out_weights
    out_scaler = Path(
        str(rt.get("scaler_output_path", "output/learned_weights_scaler.json") or "output/learned_weights_scaler.json")
    )
    if not out_scaler.is_absolute():
        out_scaler = _ROOT / out_scaler

    matrix_start = _matrix_start_date(
        train_end,
        train_start_floor=train_start_floor,
        window_years=window_years,
    )

    detail: dict[str, Any] = {
        "timestamp": datetime.now().isoformat(),
        "model_type": model_type,
        "train_end": train_end,
        "matrix_start": matrix_start,
        "train_window_years": window_years,
        "validation_holdout_calendar_days": holdout_days,
        "validation_walk_forward_windows": n_val_windows,
        "status": "error",
    }

    try:
        if model_type in ("learned_weights", "learned", "weights"):
            cmd = [
                sys.executable,
                str(_ROOT / "run_retrain_baseline.py"),
                "--config",
                str(cfg_path),
                "--as-of",
                train_end,
                "--train-window-years",
                str(window_years),
                "--train-start-floor",
                train_start_floor,
                "--out-weights",
                str(out_weights),
                "--out-scaler",
                str(out_scaler),
                "--backup-suffix",
                backup_suffix,
                "--log-file",
                str(log_path),
            ]
            if holdout_days > 0:
                cmd.extend(["--holdout-calendar-days", str(holdout_days)])
            if args.verbose:
                cmd.append("--verbose")
            log.info("Running: %s", " ".join(cmd))
            log.info(
                "Retrain policy: holdout=%s calendar days (tail reserved for OOS), "
                "validation windows=%s, min_sharpe_ratio_vs_previous=%.2f, accept_if_better_on_windows=%s",
                holdout_days if holdout_days > 0 else "OFF (overlapping validation — not true OOS)",
                n_val_windows,
                min_sharpe_ratio,
                accept_if_better_on_windows,
            )
            subprocess.run(cmd, check=True, cwd=str(_ROOT))

            if validate:
                from run_retrain_baseline import (
                    build_oos_validation_windows,
                    learned_backtest_metrics_window,
                    metrics_sharpe,
                )

                backup_w = out_weights.with_name(out_weights.name + backup_suffix)
                if holdout_days > 0:
                    windows = build_oos_validation_windows(train_end, holdout_days, n_val_windows)
                    detail["validation_mode"] = "oos_holdout"
                else:
                    log.warning(
                        "validation_holdout_calendar_days=0: OOS validation overlaps training data "
                        "(same issue as full-period Sharpe on fit sample). Set to ~252+ for fair accepts."
                    )
                    end_ts = pd.Timestamp(train_end)
                    w0 = (end_ts - pd.Timedelta(days=val_days)).strftime("%Y-%m-%d")
                    windows = [(w0, train_end)]
                    detail["validation_mode"] = "overlapping_legacy"

                detail["validation_windows"] = [list(w) for w in windows]
                log.info("Validation backtest windows (%d): %s", len(windows), windows)

                sn_list: list[float | None] = []
                so_list: list[float | None] = []
                for w0, w1 in windows:
                    mn = learned_backtest_metrics_window(
                        config_path=cfg_path,
                        weights_path=out_weights,
                        start_date=w0,
                        end_date=w1,
                    )
                    mo = (
                        learned_backtest_metrics_window(
                            config_path=cfg_path,
                            weights_path=backup_w,
                            start_date=w0,
                            end_date=w1,
                        )
                        if backup_w.is_file()
                        else None
                    )
                    sn = metrics_sharpe(mn)
                    so = metrics_sharpe(mo)
                    sn_list.append(sn)
                    so_list.append(so)
                    log.info(
                        "OOS [%s → %s] NEW sharpe=%s RETURN=%s | PREV sharpe=%s RETURN=%s",
                        w0,
                        w1,
                        sn,
                        None if not mn else mn.get("total_return"),
                        so,
                        None if not mo else mo.get("total_return"),
                    )
                    if args.verbose and mn:
                        log.debug("NEW metrics window %s→%s: %s", w0, w1, mn)

                vals_new = [x for x in sn_list if x is not None and x == x]
                vals_old = [x for x in so_list if x is not None and x == x]
                mean_new = statistics.mean(vals_new) if vals_new else float("nan")
                mean_old = statistics.mean(vals_old) if vals_old else float("nan")
                better_count = sum(
                    1
                    for sn, so in zip(sn_list, so_list, strict=True)
                    if sn is not None
                    and so is not None
                    and sn == sn
                    and so == so
                    and sn > so
                )

                detail["validation_new_sharpe_per_window"] = sn_list
                detail["validation_previous_sharpe_per_window"] = so_list
                detail["validation_mean_new_sharpe"] = mean_new
                detail["validation_mean_previous_sharpe"] = mean_old
                detail["validation_better_windows_count"] = better_count

                reject = False
                reasons: list[str] = []

                if not vals_new:
                    reject = True
                    reasons.append(
                        "new model produced no valid OOS Sharpe on any validation window "
                        "(check OHLCV cache, date range, backtest errors)"
                    )

                if not reject and accept_if_better_on_windows > 0:
                    need_bw = int(accept_if_better_on_windows)
                    if better_count < need_bw:
                        reject = True
                        reasons.append(
                            f"need new>old on >= {need_bw} windows, got {better_count}"
                        )

                if (
                    not reject
                    and mean_old == mean_old
                    and mean_old > 1e-6
                    and mean_new == mean_new
                    and mean_new < float(min_sharpe_ratio) * float(mean_old)
                ):
                    reject = True
                    reasons.append(
                        f"mean OOS Sharpe {mean_new:.4f} < {min_sharpe_ratio:.2f} × "
                        f"previous mean {mean_old:.4f}"
                    )

                if (
                    not reject
                    and compare_last_accepted
                    and last_accepted_path.is_file()
                ):
                    try:
                        prev_accept = json.loads(last_accepted_path.read_text(encoding="utf-8"))
                        sm = float(prev_accept.get("mean_oos_sharpe", float("nan")))
                        if sm == sm and sm > 1e-6 and mean_new == mean_new:
                            if mean_new < float(min_ratio_last_accepted) * sm:
                                reject = True
                                reasons.append(
                                    f"mean OOS {mean_new:.4f} < {min_ratio_last_accepted:.2f} × "
                                    f"last accepted mean {sm:.4f}"
                                )
                    except Exception as exc:  # noqa: BLE001
                        log.warning("Could not read last accepted OOS baseline: %s", exc)

                if not backup_w.is_file():
                    log.warning("No backup weights at %s — skipped comparison to previous model.", backup_w)

                if reject:
                    msg = "Model rejected after OOS validation: " + "; ".join(reasons)
                    log.error(msg)
                    _restore_learned_weights_backups(out_weights, out_scaler, backup_suffix, log)
                    detail["status"] = "rejected_validation"
                    detail["reject_reason"] = msg
                    detail["reject_reasons"] = reasons
                    if alert_on_failure:
                        _alert_failure(msg, config_path=cfg_path)
                    (_ROOT / "output" / "retrain_model_last.json").parent.mkdir(parents=True, exist_ok=True)
                    (_ROOT / "output" / "retrain_model_last.json").write_text(
                        json.dumps(detail, indent=2, default=str),
                        encoding="utf-8",
                    )
                    return 1

                try:
                    last_accepted_path.parent.mkdir(parents=True, exist_ok=True)
                    last_accepted_path.write_text(
                        json.dumps(
                            {
                                "train_end": train_end,
                                "validated_at": datetime.now().isoformat(),
                                "holdout_calendar_days": holdout_days,
                                "validation_windows": [list(w) for w in windows],
                                "mean_oos_sharpe": mean_new,
                                "per_window_sharpe": sn_list,
                                "mean_previous_oos_sharpe": mean_old,
                            },
                            indent=2,
                            default=str,
                        ),
                        encoding="utf-8",
                    )
                    log.info("Wrote last-accepted OOS baseline → %s", last_accepted_path)
                except Exception as exc:  # noqa: BLE001
                    log.warning("Could not write last accepted OOS file: %s", exc)

        elif model_type == "ml":
            cmd = [
                sys.executable,
                str(_ROOT / "run_model_selection.py"),
                "--config",
                str(cfg_path),
                "--matrix-start-date",
                matrix_start,
                "--matrix-end-date",
                train_end,
            ]
            if args.verbose:
                pass  # run_model_selection has no verbose flag
            log.info("Running: %s", " ".join(cmd))
            subprocess.run(cmd, check=True, cwd=str(_ROOT))

            default_pkl = _ROOT / "output" / "models" / "best_model.pkl"
            dest_pkl = Path(str(rt.get("output_path", "output/models/best_model.pkl") or "output/models/best_model.pkl"))
            if not dest_pkl.is_absolute():
                dest_pkl = _ROOT / dest_pkl
            if dest_pkl.resolve() != default_pkl.resolve() and default_pkl.is_file():
                dest_pkl.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(default_pkl, dest_pkl)
                log.info("Copied %s → %s", default_pkl, dest_pkl)
            meta_src = _ROOT / "output" / "models" / "best_model.meta.json"
            dest_meta = dest_pkl.with_suffix(".meta.json")
            if meta_src.is_file() and dest_meta.resolve() != meta_src.resolve():
                shutil.copy2(meta_src, dest_meta)
                log.info("Copied meta → %s", dest_meta)
        else:
            raise SystemExit(f"Unknown model_type: {model_type!r} (use learned_weights or ml)")

        detail["status"] = "ok"
        ml_pkl: Path | None = None
        if model_type == "ml":
            ml_pkl = Path(str(rt.get("output_path", "output/models/best_model.pkl") or "output/models/best_model.pkl"))
            if not ml_pkl.is_absolute():
                ml_pkl = _ROOT / ml_pkl
        _snapshot_model_release(
            root=_ROOT,
            train_end=train_end,
            detail=dict(detail),
            model_type=model_type,
            rt=rt,
            weights=out_weights if model_type in ("learned_weights", "learned", "weights") else None,
            scaler=out_scaler if model_type in ("learned_weights", "learned", "weights") else None,
            ml_pkl=ml_pkl,
        )
        (_ROOT / "output" / "retrain_model_last.json").parent.mkdir(parents=True, exist_ok=True)
        (_ROOT / "output" / "retrain_model_last.json").write_text(
            json.dumps(detail, indent=2, default=str),
            encoding="utf-8",
        )
        log.info("Retrain finished successfully.")
        return 0

    except subprocess.CalledProcessError as exc:
        log.error("Subprocess failed: %s", exc)
        detail["error"] = f"subprocess exit {exc.returncode}"
        if alert_on_failure:
            _alert_failure(
                f"Retrain subprocess failed (exit {exc.returncode}).\n{traceback.format_exc()}",
                config_path=cfg_path,
            )
    except Exception as exc:  # noqa: BLE001
        log.exception("Retrain failed: %s", exc)
        detail["error"] = str(exc)
        if alert_on_failure:
            _alert_failure(
                f"Retrain error: {exc}\n{traceback.format_exc()}",
                config_path=cfg_path,
            )

    detail["status"] = "error"
    try:
        (_ROOT / "output" / "retrain_model_last.json").parent.mkdir(parents=True, exist_ok=True)
        (_ROOT / "output" / "retrain_model_last.json").write_text(
            json.dumps(detail, indent=2, default=str),
            encoding="utf-8",
        )
    except Exception:
        pass
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
