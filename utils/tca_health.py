"""Transaction-cost analysis health checks for live fills."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def evaluate_tca_health(
    fills: pd.DataFrame,
    *,
    rolling_trades: int,
    max_avg_slippage_bps: float,
    max_p95_slippage_bps: float | None,
    min_fills: int,
    fail_on_no_data: bool,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []

    if fills is None or fills.empty or "slippage_bps" not in fills.columns:
        status = "FAIL" if fail_on_no_data else "PASS"
        if status == "FAIL":
            failures.append({"code": "NO_SLIPPAGE_DATA", "detail": "no fills with slippage_bps available"})
        return {
            "status": status,
            "metrics": {
                "n_fills_total": 0,
                "n_fills_window": 0,
                "avg_slippage_bps": None,
                "p95_slippage_bps": None,
            },
            "failures": failures,
        }

    sub = fills.copy()
    sub["slippage_bps"] = pd.to_numeric(sub["slippage_bps"], errors="coerce")
    sub = sub[sub["slippage_bps"].notna()].copy()
    if sub.empty:
        status = "FAIL" if fail_on_no_data else "PASS"
        if status == "FAIL":
            failures.append({"code": "NO_SLIPPAGE_DATA", "detail": "all slippage_bps are null/non-numeric"})
        return {
            "status": status,
            "metrics": {
                "n_fills_total": 0,
                "n_fills_window": 0,
                "avg_slippage_bps": None,
                "p95_slippage_bps": None,
            },
            "failures": failures,
        }

    window_n = int(max(1, rolling_trades))
    tail = sub.tail(window_n)["slippage_bps"].astype(float)
    avg = float(tail.mean()) if len(tail) else float("nan")
    p95 = float(np.percentile(tail, 95)) if len(tail) else float("nan")

    if len(tail) < int(max(1, min_fills)):
        if fail_on_no_data:
            failures.append(
                {
                    "code": "INSUFFICIENT_FILLS",
                    "measured": int(len(tail)),
                    "limit": int(max(1, min_fills)),
                }
            )

    if np.isfinite(avg) and avg > float(max_avg_slippage_bps):
        failures.append(
            {
                "code": "AVG_SLIPPAGE_EXCEEDED",
                "measured": float(avg),
                "limit": float(max_avg_slippage_bps),
            }
        )

    if max_p95_slippage_bps is not None and np.isfinite(p95) and p95 > float(max_p95_slippage_bps):
        failures.append(
            {
                "code": "P95_SLIPPAGE_EXCEEDED",
                "measured": float(p95),
                "limit": float(max_p95_slippage_bps),
            }
        )

    return {
        "status": "PASS" if not failures else "FAIL",
        "metrics": {
            "n_fills_total": int(len(sub)),
            "n_fills_window": int(len(tail)),
            "avg_slippage_bps": float(avg) if np.isfinite(avg) else None,
            "p95_slippage_bps": float(p95) if np.isfinite(p95) else None,
        },
        "failures": failures,
    }
