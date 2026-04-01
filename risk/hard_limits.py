"""Fail-closed hard-limit risk checks for target portfolios."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class HardLimitConfig:
    enabled: bool
    fail_closed: bool
    max_gross_exposure: float
    max_abs_net_exposure: float
    max_single_name_abs: float
    max_short_single_name_abs: float
    max_sector_exposure: float | None


def load_sector_mapping(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    if df.empty or "ticker" not in df.columns or "sector" not in df.columns:
        return {}
    out: dict[str, str] = {}
    for _, row in df.iterrows():
        t = str(row.get("ticker", "")).strip().upper()
        s = str(row.get("sector", "")).strip()
        if t and s and t != "NAN" and s != "NAN":
            out[t] = s
    return out


def hard_limit_config_from_yaml(cfg: dict[str, Any]) -> HardLimitConfig:
    risk = cfg.get("risk") or {}
    rf = cfg.get("risk_factors") or {}
    gate = risk.get("hard_limit_gate") or {}

    # Keep defaults aligned with execution config unless explicitly overridden.
    gross = float(gate.get("max_gross_exposure", risk.get("max_gross_exposure", 1.5)) or 1.5)
    abs_net = float(gate.get("max_abs_net_exposure", risk.get("max_net_exposure", 0.5)) or 0.5)
    single_name = float(
        gate.get(
            "max_single_name_abs",
            rf.get("max_single_name_pct", risk.get("max_position_pct_of_equity", 0.12)),
        )
        or 0.12
    )
    short_single = float(gate.get("max_short_single_name_abs", risk.get("max_short_single_name", single_name)) or single_name)
    sec_default = rf.get("max_sector_exposure", risk.get("max_sector_exposure"))
    max_sector_exposure = float(sec_default) if sec_default is not None else None
    if gate.get("max_sector_exposure") is not None:
        max_sector_exposure = float(gate.get("max_sector_exposure"))

    return HardLimitConfig(
        enabled=bool(gate.get("enabled", True)),
        fail_closed=bool(gate.get("fail_closed", True)),
        max_gross_exposure=gross,
        max_abs_net_exposure=abs_net,
        max_single_name_abs=single_name,
        max_short_single_name_abs=short_single,
        max_sector_exposure=max_sector_exposure,
    )


def evaluate_target_hard_limits(
    target: pd.DataFrame,
    *,
    equity: float,
    limits: HardLimitConfig,
    sector_mapping: dict[str, str] | None = None,
) -> dict[str, Any]:
    failures: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {
        "n_positions": 0,
        "gross_exposure": 0.0,
        "net_exposure": 0.0,
        "max_single_name_abs": 0.0,
        "max_short_single_name_abs": 0.0,
        "max_sector_exposure": None,
    }

    if not limits.enabled:
        return {
            "status": "PASS",
            "failures": failures,
            "metrics": metrics,
            "note": "hard_limit_gate_disabled",
        }

    if target is None or target.empty:
        return {"status": "PASS", "failures": failures, "metrics": metrics}

    df = target.copy()
    if "ticker" not in df.columns:
        failures.append({"code": "MISSING_REQUIRED_COLUMNS", "detail": "missing ticker"})
        return {"status": "FAIL", "failures": failures, "metrics": metrics}

    df["ticker"] = df["ticker"].astype(str).str.strip().str.upper()
    df = df[df["ticker"].ne("") & df["ticker"].ne("NAN")].copy()
    if df.empty:
        return {"status": "PASS", "failures": failures, "metrics": metrics}

    if "target_weight" in df.columns:
        weights = pd.to_numeric(df["target_weight"], errors="coerce")
    elif "target_value" in df.columns:
        if equity <= 0 or not np.isfinite(equity):
            failures.append(
                {
                    "code": "INVALID_EQUITY",
                    "detail": f"equity must be positive finite when using target_value, got {equity}",
                }
            )
            return {"status": "FAIL", "failures": failures, "metrics": metrics}
        weights = pd.to_numeric(df["target_value"], errors="coerce") / float(equity)
        df["target_weight"] = weights
    else:
        failures.append(
            {
                "code": "MISSING_REQUIRED_COLUMNS",
                "detail": "need target_weight or target_value",
            }
        )
        return {"status": "FAIL", "failures": failures, "metrics": metrics}

    if not np.isfinite(weights.to_numpy(dtype=float)).all():
        failures.append(
            {
                "code": "INVALID_NUMERIC_VALUE",
                "detail": "target weights contain non-finite values",
            }
        )
        return {"status": "FAIL", "failures": failures, "metrics": metrics}

    df["target_weight"] = weights.astype(float)
    by_ticker = (
        df.groupby("ticker", as_index=False)["target_weight"]
        .sum()
        .sort_values("ticker")
        .reset_index(drop=True)
    )

    w = by_ticker["target_weight"].astype(float)
    gross = float(w.abs().sum())
    net = float(w.sum())
    max_single = float(w.abs().max()) if len(w) else 0.0
    shorts = w[w < 0]
    max_short_single = float(shorts.abs().max()) if len(shorts) else 0.0

    metrics.update(
        {
            "n_positions": int(len(by_ticker)),
            "gross_exposure": gross,
            "net_exposure": net,
            "max_single_name_abs": max_single,
            "max_short_single_name_abs": max_short_single,
        }
    )

    if gross > float(limits.max_gross_exposure):
        failures.append(
            {
                "code": "MAX_GROSS_EXCEEDED",
                "measured": gross,
                "limit": float(limits.max_gross_exposure),
            }
        )
    if abs(net) > float(limits.max_abs_net_exposure):
        failures.append(
            {
                "code": "MAX_ABS_NET_EXCEEDED",
                "measured": abs(net),
                "limit": float(limits.max_abs_net_exposure),
            }
        )
    if max_single > float(limits.max_single_name_abs):
        top_idx = int(w.abs().idxmax())
        failures.append(
            {
                "code": "MAX_SINGLE_NAME_EXCEEDED",
                "ticker": str(by_ticker.loc[top_idx, "ticker"]),
                "measured": max_single,
                "limit": float(limits.max_single_name_abs),
            }
        )
    if max_short_single > float(limits.max_short_single_name_abs):
        short_ticker = str(by_ticker.loc[w.idxmin(), "ticker"]) if len(shorts) else ""
        failures.append(
            {
                "code": "MAX_SHORT_SINGLE_NAME_EXCEEDED",
                "ticker": short_ticker,
                "measured": max_short_single,
                "limit": float(limits.max_short_single_name_abs),
            }
        )

    if limits.max_sector_exposure is not None:
        sm = sector_mapping or {}
        if limits.fail_closed:
            missing = [
                str(t)
                for t in by_ticker["ticker"].tolist()
                if str(t).strip().upper() not in sm
            ]
            if missing:
                failures.append(
                    {
                        "code": "MISSING_SECTOR_MAPPING",
                        "detail": f"missing sector mapping for {len(missing)} ticker(s)",
                        "sample": missing[:10],
                    }
                )
        sec_df = by_ticker.copy()
        sec_df["sector"] = sec_df["ticker"].map(lambda x: sm.get(str(x).strip().upper(), "Unknown"))
        sec_df["abs_weight"] = sec_df["target_weight"].astype(float).abs()
        sec_exp = (
            sec_df.groupby("sector", as_index=False)["abs_weight"]
            .sum()
            .rename(columns={"abs_weight": "abs_exposure"})
            .sort_values("abs_exposure", ascending=False)
        )
        max_sector = float(sec_exp["abs_exposure"].max()) if not sec_exp.empty else 0.0
        metrics["max_sector_exposure"] = max_sector
        if max_sector > float(limits.max_sector_exposure):
            sector_name = str(sec_exp.iloc[0]["sector"]) if not sec_exp.empty else "Unknown"
            failures.append(
                {
                    "code": "MAX_SECTOR_EXCEEDED",
                    "sector": sector_name,
                    "measured": max_sector,
                    "limit": float(limits.max_sector_exposure),
                }
            )

    return {
        "status": "PASS" if not failures else "FAIL",
        "failures": failures,
        "metrics": metrics,
    }


def write_hard_limit_report(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2) + "\n")
