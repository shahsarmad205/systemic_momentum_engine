#!/usr/bin/env python3
"""Run fail-closed hard-limit risk checks on a target portfolio CSV."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"config not found: {path}")
    try:
        import yaml

        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        raise RuntimeError(f"failed to parse config {path}: {exc}") from exc


def _resolve_target_csv(arg_path: str) -> Path:
    p = Path(arg_path)
    if p.is_absolute():
        return p
    return (ROOT / p).resolve()


def main() -> int:
    parser = argparse.ArgumentParser(description="Check hard risk limits for target portfolio")
    parser.add_argument("--config", type=str, default="backtest_config.yaml")
    parser.add_argument(
        "--target-csv",
        type=str,
        default="output/live/target_executable_latest.csv",
        help="Target portfolio CSV with ticker and target_weight (or target_value)",
    )
    parser.add_argument("--equity", type=float, default=0.0, help="Account equity; if <=0 uses backtest.initial_capital")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero on hard-limit failure")
    parser.add_argument("--report", type=str, default="", help="Optional explicit report path")
    args = parser.parse_args()

    from risk.hard_limits import (
        evaluate_target_hard_limits,
        hard_limit_config_from_yaml,
        load_sector_mapping,
        write_hard_limit_report,
    )

    cfg_path = ROOT / args.config
    try:
        cfg = _read_config(cfg_path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    target_path = _resolve_target_csv(args.target_csv)
    if not target_path.exists():
        print(f"ERROR: target csv not found: {target_path}")
        return 1

    try:
        target = pd.read_csv(target_path)
    except Exception as exc:
        print(f"ERROR: failed to read target csv {target_path}: {exc}")
        return 1

    equity = float(args.equity)
    if equity <= 0:
        bt = cfg.get("backtest") or {}
        equity = float(bt.get("initial_capital", 100000.0) or 100000.0)

    limits = hard_limit_config_from_yaml(cfg)

    rf = cfg.get("risk_factors") or {}
    risk = cfg.get("risk") or {}
    sect_rel = (
        rf.get("sector_mapping_path")
        or rf.get("sector_map_path")
        or risk.get("sector_mapping_path")
        or risk.get("sector_map_path")
    )
    sector_mapping: dict[str, str] = {}
    if sect_rel:
        sector_mapping = load_sector_mapping((ROOT / str(sect_rel)).resolve())

    result = evaluate_target_hard_limits(
        target,
        equity=equity,
        limits=limits,
        sector_mapping=sector_mapping,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    report_dir = ROOT / "output" / "live" / "risk_gate"
    report_path = Path(args.report).resolve() if args.report else (report_dir / f"risk_gate_{ts}.json")
    latest_path = report_dir / "risk_gate_latest.json"

    payload = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_csv": str(target_path),
        "equity": float(equity),
        "limits": {
            "enabled": limits.enabled,
            "fail_closed": limits.fail_closed,
            "max_gross_exposure": limits.max_gross_exposure,
            "max_abs_net_exposure": limits.max_abs_net_exposure,
            "max_single_name_abs": limits.max_single_name_abs,
            "max_short_single_name_abs": limits.max_short_single_name_abs,
            "max_sector_exposure": limits.max_sector_exposure,
        },
        "status": result.get("status", "FAIL"),
        "metrics": result.get("metrics", {}),
        "failures": result.get("failures", []),
    }

    write_hard_limit_report(report_path, payload)
    if not args.report:
        write_hard_limit_report(latest_path, payload)

    print(f"Risk gate report: {report_path}")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "gross": payload["metrics"].get("gross_exposure"),
                "net": payload["metrics"].get("net_exposure"),
                "max_single": payload["metrics"].get("max_single_name_abs"),
                "n_failures": len(payload["failures"]),
            },
            indent=2,
        )
    )

    if payload["status"] != "PASS" and args.strict:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
