#!/usr/bin/env python3
"""
Flask API serving JSON from live monitoring CSVs / execution log.

Run from ``trend_signal_engine/``:
    pip install flask flask-cors
    python api/server.py

Default port: 5001 (override with PORT env).
"""

from __future__ import annotations

import json
import logging
import math
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, jsonify
from flask_cors import CORS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent.parent

PATHS = {
    "daily_pnl": _ROOT / "output" / "live" / "daily_pnl.csv",
    "ic_tracker": _ROOT / "output" / "live" / "ic_tracker.csv",
    "monitoring_metrics": _ROOT / "output" / "live" / "monitoring_metrics.csv",
    "execution_log": _ROOT / "output" / "live" / "execution_log.jsonl",
    "paper_positions": _ROOT / "output" / "portfolio" / "paper_positions.csv",
    "signals_dir": _ROOT / "output" / "signals",
}

app = Flask(__name__)
CORS(app)


def _json_sanitize(obj: Any) -> Any:
    """
    Make data safe for JSON (and JavaScript JSON.parse). Python/json allows NaN;
    the JSON spec does not — browsers reject it.
    """
    if obj is None:
        return None
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, int | np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        x = float(obj)
        if math.isnan(x) or math.isinf(x):
            return None
        return x
    if isinstance(obj, np.generic):
        try:
            return _json_sanitize(obj.item())
        except (ValueError, AttributeError):
            return None
    if isinstance(obj, pd.Timestamp):
        if pd.isna(obj):
            return None
        return obj.strftime("%Y-%m-%d")
    if isinstance(obj, datetime):
        return obj.replace(microsecond=0).isoformat()
    if isinstance(obj, date):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, str):
        return obj
    if isinstance(obj, np.ndarray):
        return _json_sanitize(obj.tolist())
    return str(obj)


def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    if not path.is_file():
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed reading %s: %s", path, exc)
        return None


def _last_execution_log_entry() -> dict[str, Any] | None:
    path = PATHS["execution_log"]
    if not path.is_file():
        return None
    try:
        lines = path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            return None
        return json.loads(lines[-1])
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed parsing execution log: %s", exc)
        return None


def _positions_from_execution_entry(entry: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    orders = entry.get("orders_placed") or []
    if not isinstance(orders, list):
        return out
    for o in orders:
        if not isinstance(o, dict) or not o.get("success"):
            continue
        t = o.get("ticker")
        if not t:
            continue
        out.append(
            {
                "ticker": str(t).upper(),
                "side": o.get("side", ""),
                "notional": o.get("notional"),
                "status": o.get("status", ""),
                "order_id": o.get("order_id", ""),
            }
        )
    return out


@app.route("/api/account")
def api_account() -> Any:
    """Latest equity/cash from daily_pnl; total return vs first row."""
    df = _safe_read_csv(PATHS["daily_pnl"])
    if df is None or df.empty:
        return jsonify(
            _json_sanitize(
                {
                    "equity": None,
                    "cash": None,
                    "buying_power": None,
                    "total_return": None,
                    "date": None,
                    "n_positions": None,
                }
            )
        )
    try:
        df = df.copy()
        df["date"] = df["date"].astype(str)
        last = df.iloc[-1]
        first = df.iloc[0]
        eq = float(pd.to_numeric(last.get("equity"), errors="coerce"))
        eq0 = float(pd.to_numeric(first.get("equity"), errors="coerce"))
        cash = last.get("cash")
        cash_f = float(pd.to_numeric(cash, errors="coerce")) if cash is not None else None
        npos = last.get("n_positions")
        npos_i = int(pd.to_numeric(npos, errors="coerce")) if npos is not None else None
        ret = (eq - eq0) / eq0 if eq0 and eq0 > 0 and pd.notna(eq) else None
        return jsonify(
            _json_sanitize(
                {
                    "equity": eq if pd.notna(eq) else None,
                    "cash": cash_f if cash_f is None or pd.notna(cash_f) else None,
                    "buying_power": None,
                    "total_return": ret if ret is None or pd.notna(ret) else None,
                    "date": str(last.get("date", "")),
                    "n_positions": npos_i,
                }
            )
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("api/account: %s", exc)
        return jsonify(
            _json_sanitize(
                {
                    "equity": None,
                    "cash": None,
                    "buying_power": None,
                    "total_return": None,
                    "date": None,
                    "n_positions": None,
                    "error": str(exc),
                }
            )
        )


@app.route("/api/positions")
def api_positions() -> Any:
    """Prefer paper_positions (latest date); else last execution_log buy orders."""
    df = _safe_read_csv(PATHS["paper_positions"])
    if df is not None and not df.empty and "ticker" in df.columns:
        try:
            if "date" in df.columns:
                df = df.copy()
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                dmax = df["date"].max()
                if pd.notna(dmax):
                    latest = df[df["date"] == dmax]
                    records = latest.where(pd.notnull(latest), None).to_dict(orient="records")
                    for r in records:
                        for k, v in list(r.items()):
                            if hasattr(v, "item"):
                                try:
                                    r[k] = v.item()
                                except (ValueError, AttributeError):
                                    pass
                    return jsonify(_json_sanitize({"source": "paper_positions.csv", "positions": records}))
        except Exception as exc:  # noqa: BLE001
            logger.exception("paper_positions read: %s", exc)

    entry = _last_execution_log_entry()
    if entry:
        pos = _positions_from_execution_entry(entry)
        return jsonify(
            _json_sanitize(
                {
                    "source": "execution_log.jsonl",
                    "timestamp": entry.get("timestamp"),
                    "positions": pos,
                }
            )
        )
    return jsonify(_json_sanitize({"source": None, "positions": []}))


@app.route("/api/signals")
def api_signals() -> Any:
    """Top 10 rows from newest *_rankings.csv."""
    d = PATHS["signals_dir"]
    if not d.is_dir():
        return jsonify(_json_sanitize({"file": None, "signals": []}))
    files = sorted(d.glob("*_rankings.csv"))
    if not files:
        return jsonify(_json_sanitize({"file": None, "signals": []}))
    path = files[-1]
    df = _safe_read_csv(path)
    if df is None or df.empty:
        return jsonify(_json_sanitize({"file": path.name, "signals": []}))
    try:
        top = df.head(10)
        recs = top.where(pd.notnull(top), None).to_dict(orient="records")
        return jsonify(_json_sanitize({"file": path.name, "signals": recs}))
    except Exception as exc:  # noqa: BLE001
        logger.exception("api/signals: %s", exc)
        return jsonify(_json_sanitize({"file": path.name, "signals": [], "error": str(exc)}))


@app.route("/api/equity")
def api_equity() -> Any:
    """Full {date, equity} from daily_pnl."""
    df = _safe_read_csv(PATHS["daily_pnl"])
    if df is None or df.empty or "equity" not in df.columns:
        return jsonify(_json_sanitize([]))
    try:
        out = [
            {"date": str(r["date"]), "equity": float(pd.to_numeric(r["equity"], errors="coerce"))}
            for _, r in df.iterrows()
            if pd.notna(pd.to_numeric(r.get("equity"), errors="coerce"))
        ]
        return jsonify(_json_sanitize(out))
    except Exception as exc:  # noqa: BLE001
        logger.exception("api/equity: %s", exc)
        return jsonify(_json_sanitize([]))


@app.route("/api/ic")
def api_ic() -> Any:
    """IC tracker dates and ic_daily / rolling_ic."""
    df = _safe_read_csv(PATHS["ic_tracker"])
    if df is None or df.empty:
        return jsonify(_json_sanitize([]))
    try:
        rows = []
        for _, r in df.iterrows():
            row: dict[str, Any] = {"date": str(r.get("date", ""))}
            if "ic_daily" in df.columns:
                v = pd.to_numeric(r.get("ic_daily"), errors="coerce")
                row["ic_daily"] = float(v) if pd.notna(v) else None
            if "rolling_ic" in df.columns:
                v = pd.to_numeric(r.get("rolling_ic"), errors="coerce")
                row["rolling_ic"] = float(v) if pd.notna(v) else None
            rows.append(row)
        return jsonify(_json_sanitize(rows))
    except Exception as exc:  # noqa: BLE001
        logger.exception("api/ic: %s", exc)
        return jsonify(_json_sanitize([]))


@app.route("/api/rolling")
def api_rolling() -> Any:
    """Latest rolling IC and rolling Sharpe from monitoring_metrics or fallbacks."""
    out: dict[str, Any] = {
        "rolling_ic": None,
        "rolling_sharpe": None,
        "date": None,
        "source": None,
    }
    df = _safe_read_csv(PATHS["monitoring_metrics"])
    if df is not None and not df.empty:
        try:
            last = df.iloc[-1]
            out["date"] = str(last.get("date", "")) if last.get("date") is not None else None
            ric = pd.to_numeric(last.get("rolling_ic"), errors="coerce")
            rsh = pd.to_numeric(last.get("rolling_sharpe"), errors="coerce")
            out["rolling_ic"] = float(ric) if pd.notna(ric) else None
            out["rolling_sharpe"] = float(rsh) if pd.notna(rsh) else None
            out["source"] = "monitoring_metrics.csv"
            return jsonify(_json_sanitize(out))
        except Exception as exc:  # noqa: BLE001
            logger.exception("api/rolling monitoring_metrics: %s", exc)

    df_ic = _safe_read_csv(PATHS["ic_tracker"])
    if df_ic is not None and not df_ic.empty and "rolling_ic" in df_ic.columns:
        last = df_ic.iloc[-1]
        ric = pd.to_numeric(last.get("rolling_ic"), errors="coerce")
        if pd.notna(ric):
            out["rolling_ic"] = float(ric)
            out["date"] = str(last.get("date", ""))
            out["source"] = "ic_tracker.csv"

    df_pnl = _safe_read_csv(PATHS["daily_pnl"])
    if df_pnl is not None and len(df_pnl) >= 3 and "equity" in df_pnl.columns:
        try:
            eq = pd.to_numeric(df_pnl["equity"], errors="coerce")
            ret = eq.pct_change().dropna()
            if len(ret) >= 2:
                sd = float(ret.std(ddof=1))
                if sd > 1e-12:
                    out["rolling_sharpe"] = float(ret.mean() / sd * (252**0.5))
                    if not out.get("date"):
                        out["date"] = str(df_pnl.iloc[-1].get("date", ""))
                    out["source"] = (
                        f"{out['source']}+pnl_sharpe" if out.get("source") else "daily_pnl_sharpe"
                    )
        except Exception as exc:  # noqa: BLE001
            logger.exception("api/rolling pnl sharpe: %s", exc)

    return jsonify(_json_sanitize(out))


@app.route("/health")
def health_live() -> Any:
    """Liveness: process is up."""
    return jsonify(_json_sanitize({"status": "ok", "service": "trend-signal-engine"}))


@app.route("/ready")
def health_ready() -> Any:
    """Readiness: primary config on disk (does not verify Alpaca credentials)."""
    cfg = _ROOT / "backtest_config.yaml"
    ok = cfg.is_file()
    body = {"ready": ok, "config_present": ok, "root": str(_ROOT)}
    return jsonify(_json_sanitize(body)), (200 if ok else 503)


@app.route("/api/health")
def api_health() -> Any:
    return jsonify(_json_sanitize({"ok": True, "root": str(_ROOT)}))


def main() -> None:
    port = int(os.environ.get("PORT", "5001"))
    logger.info("Serving from %s on port %s", _ROOT, port)
    app.run(host="0.0.0.0", port=port, debug=os.environ.get("FLASK_DEBUG") == "1")


if __name__ == "__main__":
    main()
