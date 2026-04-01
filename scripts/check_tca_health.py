#!/usr/bin/env python3
"""Check live fill slippage quality against governance thresholds."""

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


def main() -> int:
    parser = argparse.ArgumentParser(description="Check TCA/slippage health")
    parser.add_argument("--config", type=str, default="backtest_config.yaml")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when TCA status is FAIL")
    parser.add_argument("--report", type=str, default="", help="Optional explicit report path")
    args = parser.parse_args()

    from utils.tca_health import evaluate_tca_health

    cfg_path = ROOT / args.config
    try:
        cfg = _read_config(cfg_path)
    except RuntimeError as exc:
        print(f"ERROR: {exc}")
        return 1

    st = cfg.get("slippage_tracking") or {}
    gov = (cfg.get("governance") or {}).get("tca_health") or {}

    if not bool(st.get("enabled", False)):
        print("TCA check disabled: slippage_tracking.enabled is false")
        return 2 if args.strict else 0

    trades_rel = st.get("trades_file", "output/live/trades.csv")
    tp = Path(str(trades_rel))
    if tp.is_absolute():
        trades_path = tp
    else:
        trades_path = (cfg_path.parent / tp).resolve()
    fills = pd.DataFrame()
    if trades_path.is_file():
        try:
            fills = pd.read_csv(trades_path)
        except Exception as exc:
            print(f"ERROR: failed to read trades file {trades_path}: {exc}")
            return 1

    rolling_trades = int(gov.get("rolling_trades", st.get("rolling_trades", 20)) or 20)
    max_avg = float(gov.get("max_avg_slippage_bps", st.get("alert_threshold_bps", 10)) or 10)
    p95_raw = gov.get("max_p95_slippage_bps", None)
    max_p95 = float(p95_raw) if p95_raw is not None else None
    min_fills = int(gov.get("min_fills", max(5, rolling_trades // 2)) or max(5, rolling_trades // 2))
    fail_on_no_data = bool(gov.get("fail_on_no_data", False))
    bootstrap_allow_no_data = bool(gov.get("bootstrap_allow_no_data", True))
    # Strict mode remains fail-closed for slippage threshold breaches, but can be
    # configured to allow bootstrap when there is not enough fill history yet.
    effective_fail_on_no_data = bool(fail_on_no_data or (args.strict and not bootstrap_allow_no_data))

    result = evaluate_tca_health(
        fills,
        rolling_trades=rolling_trades,
        max_avg_slippage_bps=max_avg,
        max_p95_slippage_bps=max_p95,
        min_fills=min_fills,
        fail_on_no_data=effective_fail_on_no_data,
    )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = ROOT / "output" / "live" / "tca"
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report).resolve() if args.report else (out_dir / f"tca_health_{ts}.json")
    latest_path = out_dir / "tca_health_latest.json"

    payload = {
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "trades_file": str(trades_path),
        "status": result.get("status", "FAIL"),
        "metrics": result.get("metrics", {}),
        "failures": result.get("failures", []),
        "gates": {
            "rolling_trades": rolling_trades,
            "max_avg_slippage_bps": max_avg,
            "max_p95_slippage_bps": max_p95,
            "min_fills": min_fills,
            "fail_on_no_data": fail_on_no_data,
            "bootstrap_allow_no_data": bootstrap_allow_no_data,
            "effective_fail_on_no_data": effective_fail_on_no_data,
        },
    }
    report_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not args.report:
        latest_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

    print(f"TCA health report: {report_path}")
    print(
        json.dumps(
            {
                "status": payload["status"],
                "avg_slippage_bps": payload["metrics"].get("avg_slippage_bps"),
                "p95_slippage_bps": payload["metrics"].get("p95_slippage_bps"),
                "n_fills_window": payload["metrics"].get("n_fills_window"),
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
