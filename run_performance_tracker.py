#!/usr/bin/env python3
"""
Performance Tracker
Run daily to monitor paper trading vs backtest expectations (Alpaca equity curve).

Computes rolling Information Coefficient (IC) from ``output/live/signal_history.csv``
vs realized forward returns (OHLCV via ``utils.market_data.get_ohlcv``, same provider
as ``backtest_config.yaml`` data.provider — Yahoo cache or Alpaca when configured).

Usage (from trend_signal_engine/):
    python run_performance_tracker.py
    python run_performance_tracker.py --ic-recent-days 180 --ic-alert-url https://...
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import smtplib
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime
from email.mime.text import MIMEText
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

logger = logging.getLogger(__name__)

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# Benchmarks aligned with output/backtests/latest_summary.txt (Net Sharpe, etc.)
BACKTEST_SHARPE = 1.547
BACKTEST_CAGR = 0.0899
BACKTEST_WIN_RATE = 0.484
BACKTEST_MAX_DD = -0.0706

DEFAULT_STARTING_EQUITY = 100_000.0


def _chdir_root() -> None:
    os.chdir(_ROOT)


@dataclass
class IcTrackerConfig:
    """Rolling IC between cross-sectional signal scores and forward returns."""

    enabled: bool = True
    # Wide default so sparse live history (e.g. weekly jobs) still includes older signal dates.
    recent_calendar_days: int = 730
    rolling_window: int = 20
    rolling_min_periods: int = 10
    threshold: float = 0.02
    forward_trading_days: int = 1
    min_names: int = 8
    alert_url: str = ""
    signal_history_path: Path = Path("output/live/signal_history.csv")
    out_csv: Path = Path("output/live/ic_tracker.csv")
    # When False, IC threshold / webhook alerts are deferred to ``monitor()`` (unified alerting).
    alert_on_threshold: bool = True


@dataclass
class MonitoringConfig:
    """Dashboard + rolling Sharpe merge + alert dispatch (see ``monitoring:`` in backtest_config.yaml)."""

    ic_rolling_window: int = 20
    sharpe_rolling_window: int = 20
    ic_alert_threshold: float = 0.02
    sharpe_alert_threshold: float = 0.0
    alert_method: str = "print"
    email_recipient: str = ""
    slack_webhook: str = ""
    dashboard_path: Path = Path("output/live/dashboard.html")
    generate_dashboard: bool = True
    monitoring_metrics_path: Path = Path("output/live/monitoring_metrics.csv")


def _load_data_settings() -> tuple[str, str]:
    """Provider and cache dir aligned with backtest_config.yaml ``data`` section."""
    path = _ROOT / "backtest_config.yaml"
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        data = cfg.get("data", {}) or {}
        provider = str(data.get("provider", "yahoo"))
        cache_dir = str(data.get("cache_dir", "data/cache/ohlcv"))
        ttl = int(data.get("cache_ttl_days", 0) or 0)
    except Exception:
        provider, cache_dir, ttl = "yahoo", "data/cache/ohlcv", 0
    return provider, cache_dir, ttl


def _load_monitoring_config() -> MonitoringConfig:
    path = _ROOT / "backtest_config.yaml"
    d = MonitoringConfig()
    try:
        with open(path, encoding="utf-8") as fh:
            cfg = yaml.safe_load(fh) or {}
        raw = cfg.get("monitoring") or {}
        if not isinstance(raw, dict):
            return d
        d.ic_rolling_window = int(raw.get("ic_rolling_window", d.ic_rolling_window))
        d.sharpe_rolling_window = int(raw.get("sharpe_rolling_window", d.sharpe_rolling_window))
        d.ic_alert_threshold = float(raw.get("ic_alert_threshold", d.ic_alert_threshold))
        d.sharpe_alert_threshold = float(raw.get("sharpe_alert_threshold", d.sharpe_alert_threshold))
        d.alert_method = str(raw.get("alert_method", d.alert_method) or "print").strip().lower()
        d.email_recipient = str(raw.get("email_recipient", d.email_recipient) or "").strip()
        wh = raw.get("slack_webhook", d.slack_webhook)
        d.slack_webhook = str(wh or "").strip()
        dp = raw.get("dashboard_path", str(d.dashboard_path))
        d.dashboard_path = Path(str(dp))
        d.generate_dashboard = bool(raw.get("generate_dashboard", d.generate_dashboard))
        mp = raw.get("monitoring_metrics_path", str(d.monitoring_metrics_path))
        d.monitoring_metrics_path = Path(str(mp))
    except Exception as exc:  # noqa: BLE001
        logger.warning("monitoring config load failed: %s", exc)
    return d


def _rolling_sharpe_series(
    returns: pd.Series,
    window: int,
    *,
    min_periods: int | None = None,
) -> pd.Series:
    """Annualized rolling Sharpe: mean/ std * sqrt(252) over trailing window."""
    r = pd.to_numeric(returns, errors="coerce")
    w = max(2, int(window))
    mp = max(2, int(min_periods)) if min_periods is not None else max(2, w // 2)

    def _sharpe(x: np.ndarray) -> float:
        x = x[np.isfinite(x)]
        if len(x) < 2:
            return float("nan")
        sd = float(np.std(x, ddof=1))
        if sd < 1e-12:
            return float("nan")
        return float(np.mean(x) / sd * np.sqrt(252))

    return r.rolling(window=w, min_periods=mp).apply(_sharpe, raw=True)


def _write_monitoring_metrics(
    *,
    ic_path: Path,
    pnl_path: Path,
    sharpe_window: int,
    out_csv: Path,
) -> pd.DataFrame | None:
    """Merge IC tracker with PnL-based rolling Sharpe; write ``monitoring_metrics.csv``."""
    if not pnl_path.is_file():
        return None
    pnl = pd.read_csv(pnl_path)
    if pnl.empty or "date" not in pnl.columns or "equity" not in pnl.columns:
        return None
    pnl = pnl.copy()
    pnl["date"] = pd.to_datetime(pnl["date"], errors="coerce").dt.strftime("%Y-%m-%d")
    pnl["equity"] = pd.to_numeric(pnl["equity"], errors="coerce")
    pnl = pnl.dropna(subset=["date", "equity"]).sort_values("date")
    pnl = pnl.drop_duplicates(subset=["date"], keep="last")
    pnl["ret"] = pnl["equity"].pct_change()
    pnl["rolling_sharpe"] = _rolling_sharpe_series(pnl["ret"], max(2, int(sharpe_window)))

    if not ic_path.is_file():
        out = pnl[["date", "equity", "ret", "rolling_sharpe"]].copy()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        return out

    ic = pd.read_csv(ic_path)
    if ic.empty or "date" not in ic.columns:
        out = pnl[["date", "equity", "ret", "rolling_sharpe"]].copy()
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(out_csv, index=False)
        return out
    ic = ic.copy()
    ic["date"] = ic["date"].astype(str).str.strip()
    merge_on = [c for c in ("date", "ic_daily", "rolling_ic", "n_names", "forward_trading_days", "below_threshold") if c in ic.columns]
    merged = pd.merge(
        ic[merge_on],
        pnl[["date", "equity", "ret", "rolling_sharpe"]],
        on="date",
        how="outer",
        sort=True,
    )
    merged = merged.sort_values("date").reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_csv, index=False)
    return merged


def send_alert(
    message: str,
    *,
    subject: str,
    mon: MonitoringConfig,
    dashboard_url_note: str = "",
) -> None:
    """Dispatch alert per ``monitoring.alert_method``."""
    method = (mon.alert_method or "print").strip().lower()
    body = message
    if dashboard_url_note:
        body = f"{message}\n\n{dashboard_url_note}"

    if method == "print":
        print(f"  [MONITOR ALERT] {body.replace(chr(10), ' ')}")
        return

    if method == "email":
        recipient = (mon.email_recipient or "").strip()
        if not recipient or "your@" in recipient.lower():
            print("  [MONITOR ALERT email] Set monitoring.email_recipient (or valid address).")
            print(f"  {body}")
            return
        smtp_host = (os.environ.get("SMTP_HOST") or "").strip()
        smtp_from = (os.environ.get("SMTP_FROM") or os.environ.get("SMTP_USER") or "").strip()
        if not smtp_host or not smtp_from:
            print("  [MONITOR ALERT email] Set SMTP_HOST and SMTP_FROM (or SMTP_USER) env vars.")
            print(f"  {body}")
            return
        port = int(os.environ.get("SMTP_PORT", "587"))
        user = (os.environ.get("SMTP_USER") or "").strip()
        password = (os.environ.get("SMTP_PASSWORD") or "").strip()
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"] = smtp_from
        msg["To"] = recipient
        try:
            with smtplib.SMTP(smtp_host, port, timeout=30) as server:
                server.starttls()
                if user:
                    server.login(user, password)
                server.sendmail(smtp_from, [recipient], msg.as_string())
            print("  [MONITOR ALERT] Email sent.")
        except Exception as exc:  # noqa: BLE001
            print(f"  [MONITOR ALERT] Email failed: {exc}")
        return

    if method == "slack":
        url = (mon.slack_webhook or "").strip()
        if not url or "hooks.slack.com" not in url:
            print("  [MONITOR ALERT slack] Set monitoring.slack_webhook to a valid URL.")
            print(f"  {body}")
            return
        try:
            import requests
        except ImportError:
            payload = json.dumps({"text": body[:3000]}).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=20) as resp:
                    _ = resp.read()
                print("  [MONITOR ALERT] Slack webhook sent (urllib).")
            except Exception as exc:  # noqa: BLE001
                print(f"  [MONITOR ALERT] Slack failed: {exc}")
            return
        try:
            r = requests.post(url, json={"text": body[:30000]}, timeout=20)
            r.raise_for_status()
            print("  [MONITOR ALERT] Slack webhook sent.")
        except Exception as exc:  # noqa: BLE001
            print(f"  [MONITOR ALERT] Slack failed: {exc}")
        return

    print(f"  [MONITOR ALERT] Unknown alert_method={method!r}; printing.")
    print(f"  {body}")


def generate_dashboard(
    mon: MonitoringConfig,
    *,
    pnl_path: Path,
    ic_path: Path,
    metrics_path: Path,
    positions: pd.DataFrame | None,
    account: dict[str, float] | None,
) -> None:
    """Write PNG charts + static HTML dashboard."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    live_dir = mon.dashboard_path.parent
    live_dir.mkdir(parents=True, exist_ok=True)

    png_equity = live_dir / "dashboard_equity.png"
    png_ic = live_dir / "dashboard_rolling_ic.png"
    png_sharpe = live_dir / "dashboard_rolling_sharpe.png"
    png_signals = live_dir / "dashboard_top_signals.png"

    # --- Data ---
    metrics = pd.read_csv(metrics_path) if metrics_path.is_file() else pd.DataFrame()
    pnl = pd.read_csv(pnl_path) if pnl_path.is_file() else pd.DataFrame()

    latest_ranking: pd.DataFrame | None = None
    sig_dir = _ROOT / "output" / "signals"
    if sig_dir.is_dir():
        rank_files = sorted(sig_dir.glob("*_rankings.csv"))
        if rank_files:
            try:
                latest_ranking = pd.read_csv(rank_files[-1])
            except Exception:  # noqa: BLE001
                latest_ranking = None

    # --- Plots ---
    fig, ax = plt.subplots(figsize=(9, 3.5))
    if not pnl.empty and "date" in pnl.columns and "equity" in pnl.columns:
        p2 = pnl.copy()
        p2["date"] = pd.to_datetime(p2["date"], errors="coerce")
        ax.plot(p2["date"], pd.to_numeric(p2["equity"], errors="coerce"), lw=1.5)
        ax.set_title("Equity (daily PnL)")
        ax.set_xlabel("Date")
        ax.set_ylabel("Equity ($)")
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No equity data", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(png_equity, dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    if not metrics.empty and "date" in metrics.columns and "rolling_ic" in metrics.columns:
        m = metrics.copy()
        m["date"] = pd.to_datetime(m["date"], errors="coerce")
        ax.plot(
            m["date"],
            pd.to_numeric(m["rolling_ic"], errors="coerce"),
            lw=1.5,
            color="#1f77b4",
        )
        ax.axhline(mon.ic_alert_threshold, color="r", ls="--", lw=1, label="threshold")
        ax.set_title("Rolling IC")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No rolling IC", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(png_ic, dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    if not metrics.empty and "date" in metrics.columns and "rolling_sharpe" in metrics.columns:
        m = metrics.copy()
        m["date"] = pd.to_datetime(m["date"], errors="coerce")
        ax.plot(
            m["date"],
            pd.to_numeric(m["rolling_sharpe"], errors="coerce"),
            lw=1.5,
            color="#2ca02c",
        )
        ax.axhline(mon.sharpe_alert_threshold, color="r", ls="--", lw=1, label="threshold")
        ax.set_title("Rolling Sharpe (annualized)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, "No rolling Sharpe", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(png_sharpe, dpi=120, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 4))
    if latest_ranking is not None and not latest_ranking.empty and "ticker" in latest_ranking.columns:
        dfp = latest_ranking.copy()
        if "score" in dfp.columns:
            dfp["score"] = pd.to_numeric(dfp["score"], errors="coerce")
            top = dfp.nlargest(10, "score")
            ax.barh(top["ticker"].astype(str), top["score"], color="#9467bd")
            ax.invert_yaxis()
            ax.set_title("Top 10 signals (latest rankings)")
            ax.grid(True, axis="x", alpha=0.3)
        else:
            ax.text(0.5, 0.5, "No score column", ha="center", va="center")
    else:
        ax.text(0.5, 0.5, "No rankings file", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(png_signals, dpi=120, bbox_inches="tight")
    plt.close(fig)

    # --- Latest KPIs ---
    last_eq = float("nan")
    peak_eq = float("nan")
    dd = float("nan")
    total_ret = float("nan")
    if not pnl.empty and "equity" in pnl.columns:
        eq = pd.to_numeric(pnl["equity"], errors="coerce").dropna()
        if len(eq):
            last_eq = float(eq.iloc[-1])
            peak_eq = float(eq.cummax().iloc[-1])
            if peak_eq > 0:
                dd = (last_eq - peak_eq) / peak_eq
            start = float(eq.iloc[0])
            if start > 0:
                total_ret = last_eq / start - 1.0

    ric = rs = None
    if not metrics.empty:
        tail = metrics.iloc[-1]
        if "rolling_ic" in metrics.columns:
            v = pd.to_numeric(tail.get("rolling_ic"), errors="coerce")
            ric = float(v) if pd.notna(v) else None
        if "rolling_sharpe" in metrics.columns:
            v = pd.to_numeric(tail.get("rolling_sharpe"), errors="coerce")
            rs = float(v) if pd.notna(v) else None

    acct_equity = float(account["equity"]) if account and "equity" in account else last_eq
    acct_cash = float(account.get("cash", float("nan"))) if account else float("nan")

    # Positions HTML
    pos_html = "<p>No position snapshot.</p>"
    if positions is not None and not positions.empty:
        cols = [c for c in ("ticker", "market_value", "unrealized_pnl", "unrealized_pnl_pct") if c in positions.columns]
        if cols:
            pos_html = positions[cols].to_html(index=False, float_format=lambda x: f"{x:,.2f}")

    sig_html = "<p>No signals file.</p>"
    if latest_ranking is not None and not latest_ranking.empty:
        show = latest_ranking.head(15)
        sig_html = show.to_html(index=False)

    def _pct(x: float) -> str:
        return f"{100.0 * float(x):.2f}%" if np.isfinite(x) else "—"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>Strategy health — {datetime.now().strftime("%Y-%m-%d %H:%M")}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 1.5rem; max-width: 1100px; background: #fafafa; }}
    h1 {{ font-size: 1.35rem; }}
    .kpi {{ display: flex; flex-wrap: wrap; gap: 1rem; margin: 1rem 0; }}
    .kpi div {{ background: #fff; padding: 0.75rem 1rem; border-radius: 8px; border: 1px solid #e0e0e0; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; border-radius: 6px; margin: 0.5rem 0; background: #fff; }}
    table {{ font-size: 0.9rem; }}
  </style>
</head>
<body>
  <h1>Live monitoring dashboard</h1>
  <p>Generated {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
  <div class="kpi">
    <div><strong>Account equity</strong><br/>{acct_equity:,.2f}</div>
    <div><strong>Cash</strong><br/>{acct_cash if np.isfinite(acct_cash) else "—"}</div>
    <div><strong>Total return (PnL series)</strong><br/>{_pct(total_ret)}</div>
    <div><strong>Drawdown (from peak in series)</strong><br/>{_pct(dd)}</div>
    <div><strong>Rolling IC</strong><br/>{f"{ric:.4f}" if ric is not None and np.isfinite(ric) else "—"}</div>
    <div><strong>Rolling Sharpe</strong><br/>{f"{rs:.3f}" if rs is not None and np.isfinite(rs) else "—"}</div>
  </div>
  <h2>Equity</h2>
  <img src="{png_equity.name}" alt="Equity"/>
  <h2>Rolling IC</h2>
  <img src="{png_ic.name}" alt="Rolling IC"/>
  <h2>Rolling Sharpe</h2>
  <img src="{png_sharpe.name}" alt="Rolling Sharpe"/>
  <h2>Top signals (bar)</h2>
  <img src="{png_signals.name}" alt="Signals"/>
  <h2>Open positions</h2>
  {pos_html}
  <h2>Latest rankings (top 15 rows)</h2>
  {sig_html}
</body>
</html>
"""
    mon.dashboard_path.parent.mkdir(parents=True, exist_ok=True)
    mon.dashboard_path.write_text(html, encoding="utf-8")
    print(f"  Dashboard → {mon.dashboard_path.resolve()}")


def monitor(
    mon: MonitoringConfig,
    *,
    ic_cfg: IcTrackerConfig,
    positions: pd.DataFrame | None,
    account: dict[str, float] | None,
) -> None:
    """Rolling-metrics file, threshold alerts, optional HTML dashboard."""
    try:
        metrics = _write_monitoring_metrics(
            ic_path=ic_cfg.out_csv,
            pnl_path=Path("output/live/daily_pnl.csv"),
            sharpe_window=mon.sharpe_rolling_window,
            out_csv=mon.monitoring_metrics_path,
        )
        if metrics is None or metrics.empty:
            print("  [monitor] No monitoring_metrics (need daily_pnl.csv with equity).")
            if mon.generate_dashboard:
                try:
                    generate_dashboard(
                        mon,
                        pnl_path=Path("output/live/daily_pnl.csv"),
                        ic_path=ic_cfg.out_csv,
                        metrics_path=mon.monitoring_metrics_path,
                        positions=positions,
                        account=account,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"  [monitor] dashboard: {exc}")
            return

        tail = metrics.iloc[-1]
        ric = pd.to_numeric(tail.get("rolling_ic"), errors="coerce")
        rsh = pd.to_numeric(tail.get("rolling_sharpe"), errors="coerce")
        breaches: list[str] = []
        if pd.notna(ric) and float(ric) < float(mon.ic_alert_threshold):
            breaches.append(
                f"rolling_ic={float(ric):.4f} < threshold {float(mon.ic_alert_threshold):.4f}"
            )
        if pd.notna(rsh) and float(rsh) < float(mon.sharpe_alert_threshold):
            breaches.append(
                f"rolling_sharpe={float(rsh):.4f} < threshold {float(mon.sharpe_alert_threshold):.4f}"
            )

        dash_note = f"Dashboard: file://{mon.dashboard_path.resolve()}"
        if breaches:
            subj = f"[TrendSignalEngine] Strategy health alert — {datetime.now().strftime('%Y-%m-%d')}"
            send_alert(
                "\n".join(breaches),
                subject=subj,
                mon=mon,
                dashboard_url_note=dash_note,
            )

        if mon.generate_dashboard:
            generate_dashboard(
                mon,
                pnl_path=Path("output/live/daily_pnl.csv"),
                ic_path=ic_cfg.out_csv,
                metrics_path=mon.monitoring_metrics_path,
                positions=positions,
                account=account,
            )
    except Exception as exc:  # noqa: BLE001
        print(f"  [monitor] Error (tracker continues): {exc}")


def _load_backtest_config_yaml() -> dict[str, Any]:
    path = Path("backtest_config.yaml")
    if not path.is_file():
        return {}
    try:
        with open(path, encoding="utf-8") as fh:
            return yaml.safe_load(fh) or {}
    except Exception:
        return {}


def _backtest_slippage_assumption_bps(cfg: dict[str, Any]) -> float:
    ex = cfg.get("execution") or {}
    if ex.get("slippage_bps") is not None:
        return float(ex["slippage_bps"])
    ec = cfg.get("execution_costs") or {}
    v = ec.get("slippage_bps")
    if v is None:
        return 5.0
    v = float(v)
    if abs(v) < 1.0:
        return v * 10_000.0
    return v


def run_slippage_metrics_section(cfg: dict[str, Any], mon: MonitoringConfig) -> None:
    st = cfg.get("slippage_tracking") or {}
    if not isinstance(st, dict) or not st.get("enabled", False):
        return
    trades_path = Path(str(st.get("trades_file", "output/live/trades.csv")))
    if not trades_path.is_file():
        print("  [slippage] No trades file yet — run live execute then scripts/fetch_fills.py.")
        return
    try:
        df = pd.read_csv(trades_path)
    except Exception as exc:  # noqa: BLE001
        print(f"  [slippage] Could not read {trades_path}: {exc}")
        return
    if df.empty or "slippage_bps" not in df.columns:
        print("  [slippage] Trades CSV empty or missing slippage_bps.")
        return
    sub = df[pd.to_numeric(df["slippage_bps"], errors="coerce").notna()].copy()
    if sub.empty:
        print("  [slippage] No rows with computed slippage yet (backfill fills first).")
        return
    last_n = int(st.get("rolling_trades", 20) or 20)
    threshold = float(st.get("alert_threshold_bps", 10) or 10)
    tail = sub.tail(last_n)["slippage_bps"].astype(float)
    avg = float(tail.mean())
    assum = _backtest_slippage_assumption_bps(cfg)
    print()
    print("  === REALISED SLIPPAGE (live) ===")
    print(f"  Trades file:        {trades_path}")
    print(f"  Last {len(tail)} fills avg (adverse bps): {avg:.2f} bps")
    print(f"  Backtest assumption (execution.slippage_bps): {assum:.2f} bps")
    if avg > assum:
        print(
            "  [!] Realised slippage exceeds backtest assumption — "
            "consider raising the cost model or reviewing execution."
        )
    metrics_path = Path(str(st.get("slippage_metrics_csv", "output/live/slippage_metrics.csv")))
    snap = pd.DataFrame(
        [
            {
                "timestamp": datetime.now().isoformat(),
                "n_trades_window": len(tail),
                "avg_slippage_bps": avg,
                "backtest_assumption_bps": assum,
            }
        ]
    )
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    if metrics_path.is_file():
        old = pd.read_csv(metrics_path)
        hist = pd.concat([old, snap], ignore_index=True)
    else:
        hist = snap
    hist.to_csv(metrics_path, index=False)
    print(f"  Slippage metrics log: {metrics_path}")

    if len(tail) >= 5 and avg > threshold:
        subj = f"[TrendSignalEngine] High realised slippage — {datetime.now().strftime('%Y-%m-%d')}"
        send_alert(
            f"Average adverse slippage over last {len(tail)} fills is {avg:.2f} bps "
            f"(alert threshold {threshold:.2f} bps; backtest assumption {assum:.2f} bps).",
            subject=subj,
            mon=mon,
        )


def _forward_trading_return(
    close: pd.Series,
    as_of: pd.Timestamp,
    n_forward_trading_days: int,
) -> float | None:
    """Close-only return from as-of session to n_forward_trading_days later."""
    s = pd.to_numeric(close, errors="coerce").dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if s.empty or n_forward_trading_days < 1:
        return None
    as_of = pd.Timestamp(as_of).normalize()
    idx = s.index
    pos = int(idx.searchsorted(as_of, side="right") - 1)
    if pos < 0:
        return None
    j = pos + int(n_forward_trading_days)
    if j >= len(s):
        return None
    a, b = float(s.iloc[pos]), float(s.iloc[j])
    if not (np.isfinite(a) and np.isfinite(b)) or a <= 0:
        return None
    return b / a - 1.0


def _cross_sectional_ic(scores: np.ndarray, fwd: np.ndarray) -> float | None:
    m = np.isfinite(scores) & np.isfinite(fwd)
    if int(m.sum()) < 3:
        return None
    x = scores[m].astype(float)
    y = fwd[m].astype(float)
    if np.nanstd(x) < 1e-12 or np.nanstd(y) < 1e-12:
        return None
    try:
        s = pd.Series(x).corr(pd.Series(y), method="spearman")
    except Exception:
        return None
    return float(s) if s == s and np.isfinite(s) else None


def _load_close_series(
    tickers: list[str],
    start_d: str,
    end_d: str,
    *,
    provider: str,
    cache_dir: str,
    cache_ttl_days: int,
) -> dict[str, pd.Series]:
    from utils.market_data import get_ohlcv

    out: dict[str, pd.Series] = {}
    for t in tickers:
        try:
            ohlcv = get_ohlcv(
                t,
                start_d,
                end_d,
                provider=provider,
                cache_dir=cache_dir,
                use_cache=True,
                cache_ttl_days=cache_ttl_days,
            )
        except Exception:
            continue
        if ohlcv is None or ohlcv.empty or "Close" not in ohlcv.columns:
            continue
        s = pd.to_numeric(ohlcv["Close"], errors="coerce").dropna()
        s.index = pd.to_datetime(s.index, errors="coerce")
        s = s[~s.index.isna()].sort_index()
        if not s.empty:
            out[str(t).upper()] = s
    return out


def run_ic_tracker_section(cfg: IcTrackerConfig) -> None:
    """
    Daily cross-sectional Spearman IC between ``score`` and forward return,
    then rolling mean over ``rolling_window`` dates. Warns if latest rolling IC
    is below ``threshold``; optional POST to ``alert_url``.
    """
    if not cfg.enabled:
        return

    hist = cfg.signal_history_path
    if not hist.is_file():
        print()
        print("  === IC TRACKER ===")
        print(f"  Skip: no {hist}")
        return

    signals = pd.read_csv(hist)
    if signals.empty or "date" not in signals.columns or "ticker" not in signals.columns:
        print()
        print("  === IC TRACKER ===")
        print("  Skip: signal_history missing date/ticker columns.")
        return

    if "score" not in signals.columns:
        print()
        print("  === IC TRACKER ===")
        print("  Skip: signal_history has no score column.")
        return

    signals = signals.copy()
    signals["date"] = pd.to_datetime(signals["date"], errors="coerce").dt.normalize()
    signals["ticker"] = signals["ticker"].astype(str).str.upper().str.strip()
    signals["score"] = pd.to_numeric(signals["score"], errors="coerce")
    signals = signals.dropna(subset=["date", "ticker", "score"])

    today = pd.Timestamp.now().normalize()
    cutoff_start = today - pd.Timedelta(days=int(max(30, cfg.recent_calendar_days)))
    signals = signals[signals["date"] >= cutoff_start]

    if signals["date"].nunique() < 2:
        print()
        print("  === IC TRACKER ===")
        print(
            f"  Skip: need at least 2 signal dates with scores in window "
            f"(have {signals['date'].nunique()})."
        )
        return

    tickers = sorted(signals["ticker"].unique().tolist())
    d_min = signals["date"].min() - pd.Timedelta(days=14)
    d_max = today + pd.Timedelta(days=15)
    start_str = d_min.strftime("%Y-%m-%d")
    end_str = d_max.strftime("%Y-%m-%d")

    provider, cache_dir, cache_ttl = _load_data_settings()
    print()
    print("  === IC TRACKER ===")
    print(
        f"  Signals: last {cfg.recent_calendar_days} cd | fwd={cfg.forward_trading_days} td | "
        f"roll={cfg.rolling_window} | min_names={cfg.min_names} | provider={provider}"
    )

    closes = _load_close_series(
        tickers,
        start_str,
        end_str,
        provider=provider,
        cache_dir=cache_dir,
        cache_ttl_days=cache_ttl,
    )
    if len(closes) < 5:
        print(f"  Skip: insufficient price series loaded ({len(closes)} tickers).")
        return

    daily_rows: list[dict[str, Any]] = []
    for dt in sorted(signals["date"].unique()):
        dt = pd.Timestamp(dt).normalize()
        g = signals[signals["date"] == dt]
        xs: list[float] = []
        ys: list[float] = []
        for _, r in g.iterrows():
            sym = str(r["ticker"])
            if sym not in closes:
                continue
            fr = _forward_trading_return(
                closes[sym], dt, cfg.forward_trading_days
            )
            if fr is None:
                continue
            xs.append(float(r["score"]))
            ys.append(float(fr))
        if len(xs) < cfg.min_names:
            continue
        ic = _cross_sectional_ic(np.array(xs), np.array(ys))
        if ic is None:
            continue
        daily_rows.append(
            {
                "date": dt.strftime("%Y-%m-%d"),
                "ic_daily": round(ic, 6),
                "n_names": len(xs),
                "forward_trading_days": cfg.forward_trading_days,
            }
        )

    if not daily_rows:
        print("  No daily IC rows (check prices / forward horizon / min_names).")
        return

    ic_df = pd.DataFrame(daily_rows).sort_values("date").reset_index(drop=True)

    # Merge with prior file to keep longer history; rolling_ic recomputed after merge.
    if cfg.out_csv.is_file():
        try:
            old = pd.read_csv(cfg.out_csv)
            old["date"] = old["date"].astype(str)
            ic_df["date"] = ic_df["date"].astype(str)
            tail_dates = set(ic_df["date"].tolist())
            old_kept = old[~old["date"].isin(tail_dates)]
            # Keep ic_daily from disk history; recompute rolling from full series below.
            keep_cols = [c for c in ("date", "ic_daily", "n_names", "forward_trading_days") if c in old_kept.columns]
            old_kept = old_kept[keep_cols] if keep_cols else old_kept[["date", "ic_daily"]]
            ic_df = pd.concat([old_kept, ic_df], ignore_index=True)
            ic_df = ic_df.drop_duplicates(subset=["date"], keep="last").sort_values("date")
        except Exception:
            ic_df = ic_df.sort_values("date")

    ic_df["ic_daily"] = pd.to_numeric(ic_df["ic_daily"], errors="coerce")
    ic_df["rolling_ic"] = ic_df["ic_daily"].rolling(
        int(cfg.rolling_window), min_periods=int(cfg.rolling_min_periods)
    ).mean()
    ic_df["below_threshold"] = ic_df["rolling_ic"].notna() & (
        ic_df["rolling_ic"] < float(cfg.threshold)
    )

    cfg.out_csv.parent.mkdir(parents=True, exist_ok=True)
    ic_df.to_csv(cfg.out_csv, index=False)
    print(f"  Saved {len(ic_df)} rows → {cfg.out_csv}")

    last_roll = float(ic_df["rolling_ic"].iloc[-1]) if len(ic_df) else float("nan")
    last_day = str(ic_df["date"].iloc[-1]) if len(ic_df) else ""
    if cfg.alert_on_threshold and np.isfinite(last_roll) and last_roll < float(cfg.threshold):
        msg = (
            f"[IC ALERT] Rolling IC {last_roll:.4f} < threshold {float(cfg.threshold):.4f} "
            f"(as of signal date {last_day}, window={cfg.rolling_window})."
        )
        print(f"  [!] {msg}")
        url = (cfg.alert_url or os.environ.get("IC_ALERT_WEBHOOK", "") or "").strip()
        if url:
            try:
                body = json.dumps({"text": msg}).encode("utf-8")
                req = urllib.request.Request(
                    url,
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib.request.urlopen(req, timeout=15) as resp:
                    _ = resp.read()
                print("  IC alert POST sent.")
            except urllib.error.URLError as e:
                print(f"  IC alert POST failed: {e}")
            except Exception as e:  # noqa: BLE001
                print(f"  IC alert POST failed: {e}")
    else:
        if np.isfinite(last_roll):
            print(
                f"  Latest rolling_ic={last_roll:.4f} (date={last_day}, threshold={float(cfg.threshold):.4f})."
            )
        else:
            print(
                f"  Latest rolling_ic not yet available (date={last_day}; need ≥{cfg.rolling_min_periods} "
                f"daily IC points for {cfg.rolling_window}-date roll)."
            )


def run_tracker(
    *,
    ic: IcTrackerConfig | None = None,
    skip_monitor: bool = False,
) -> None:
    _chdir_root()

    mon = _load_monitoring_config()
    ic_cfg = ic if ic is not None else IcTrackerConfig()
    if os.environ.get("IC_TRACKER_SKIP", "").lower() in ("1", "true", "yes"):
        ic_cfg = replace(ic_cfg, enabled=False)
    if skip_monitor:
        ic_cfg = replace(ic_cfg, alert_on_threshold=True)
    else:
        ic_cfg = replace(ic_cfg, alert_on_threshold=False)

    today = datetime.now().strftime("%Y-%m-%d")

    print()
    print(f"{'=' * 55}")
    print(f"  PERFORMANCE TRACKER — {today}")
    print(f"{'=' * 55}")

    cfg_bt = _load_backtest_config_yaml()

    history_file = Path("output/live/signal_history.csv")
    if not history_file.exists():
        print("  No signal history at output/live/signal_history.csv yet.")
        print("  (Optional) Run run_live_trading.py to populate; tracker still runs on Alpaca.")

    pnl_file = Path("output/live/daily_pnl.csv")

    account = None
    positions = pd.DataFrame()
    orders = pd.DataFrame()

    try:
        from brokers.alpaca_broker import AlpacaBroker

        broker = AlpacaBroker()
        account = broker.get_account()
        positions = broker.get_positions()
        orders = broker.get_orders(status="filled", limit=100)
    except Exception as e:  # noqa: BLE001
        print(f"  Could not connect to Alpaca: {e}")
        print("  Continuing with available local files only.")

    if account:
        equity = float(account["equity"])
        starting = DEFAULT_STARTING_EQUITY
        total_return = (equity - starting) / starting

        print()
        print("  === ACCOUNT STATE ===")
        print(f"  Baseline equity:  ${starting:>12,.2f}  (paper default; adjust if needed)")
        print(f"  Current equity:   ${equity:>12,.2f}")
        print(f"  Total return:     {total_return:>11.2%}")
        print(f"  Open positions:   {len(positions)}")
        if not orders.empty:
            print(f"  Filled orders (sample): {len(orders)} rows")

        today_entry = {
            "date": today,
            "equity": float(account["equity"]),
            "cash": float(account["cash"]),
            "n_positions": len(positions),
        }

        if pnl_file.exists():
            pnl_history = pd.read_csv(pnl_file)
            pnl_history = pd.concat(
                [pnl_history, pd.DataFrame([today_entry])],
                ignore_index=True,
            ).drop_duplicates(subset="date", keep="last")
        else:
            pnl_history = pd.DataFrame([today_entry])

        pnl_history = pnl_history.sort_values("date").reset_index(drop=True)
        pnl_file.parent.mkdir(parents=True, exist_ok=True)
        pnl_history.to_csv(pnl_file, index=False)

        if len(pnl_history) >= 2:
            pnl_history = pnl_history.copy()
            pnl_history["ret"] = pd.to_numeric(pnl_history["equity"], errors="coerce").pct_change()
            returns = pnl_history["ret"].dropna()
            n_days = len(returns)

            print()
            print(f"  === LIVE METRICS (from daily_pnl, {n_days} return obs) ===")
            print(f"  Backtest targets — Net Sharpe: {BACKTEST_SHARPE:.3f}, "
                  f"CAGR: {BACKTEST_CAGR:.2%}, Max DD: {BACKTEST_MAX_DD:.2%}, "
                  f"Win rate: {BACKTEST_WIN_RATE:.2%}")

            if n_days >= 5:
                std = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
                live_sharpe = (
                    float(returns.mean() / std * np.sqrt(252)) if std > 1e-12 else 0.0
                )
                live_cagr = float((1.0 + returns).prod() ** (252.0 / n_days) - 1.0)

                cumulative = (1.0 + returns).cumprod()
                rolling_max = cumulative.cummax()
                drawdowns = (cumulative - rolling_max) / rolling_max.replace(0, np.nan)
                live_max_dd = float(drawdowns.min())

                print(f"  Live Sharpe (ann., naive): {live_sharpe:.3f}")
                print(f"  Live CAGR (naive):        {live_cagr:.2%}")
                print(f"  Live Max DD:              {live_max_dd:.2%}")

                print()
                print("  === ALERTS ===")
                alerts: list[str] = []
                if n_days >= 20 and live_sharpe < 0.5:
                    alerts.append("[!] Sharpe below 0.5 over 20+ observations — review.")
                if live_max_dd < -0.15:
                    alerts.append("[!] Drawdown past 15% — consider risk-off / halt.")
                if n_days >= 60 and live_sharpe < BACKTEST_SHARPE * 0.5:
                    alerts.append("[!] Live Sharpe far below backtest Net Sharpe — review signals / fills.")

                if alerts:
                    for a in alerts:
                        print(f"  {a}")
                else:
                    print("  No threshold breaches — within scripted alert bands.")
            else:
                print(f"  Need 5+ return observations for live Sharpe/CAGR/DD (have {n_days}).")
        else:
            print()
            print("  Append more daily snapshots to output/live/daily_pnl.csv for live metrics.")
    else:
        print()
        print("  No live account snapshot — skipping equity curve append.")

    if history_file.exists():
        signals = pd.read_csv(history_file)
        print()
        print("  === SIGNAL HISTORY (output/live) ===")
        print(f"  Total rows:      {len(signals)}")
        if "date" in signals.columns:
            print(f"  Unique dates:    {signals['date'].nunique()}")
        if "ticker" in signals.columns:
            print(f"  Unique tickers:  {signals['ticker'].nunique()}")
        if "executed" in signals.columns:
            ex = signals["executed"].astype(str).str.lower().isin(("true", "1", "yes"))
            print(f"  Marked executed: {int(ex.sum())} rows")

    if not positions.empty:
        print()
        print("  === OPEN POSITIONS ===")
        print(f"  {'Ticker':<8} {'Value':>10} {'P&L':>10} {'P&L %':>8}")
        print(f"  {'-' * 40}")
        for _, pos in positions.iterrows():
            print(
                f"  {pos['ticker']:<8} "
                f"${float(pos['market_value']):>9,.0f} "
                f"${float(pos['unrealized_pnl']):>9,.0f} "
                f"{float(pos['unrealized_pnl_pct']) * 100:>7.1f}%"
            )

    run_ic_tracker_section(ic_cfg)

    try:
        run_slippage_metrics_section(cfg_bt, mon)
    except Exception as exc:  # noqa: BLE001
        print(f"  [slippage] Skipped: {exc}")

    if not skip_monitor:
        try:
            monitor(
                mon,
                ic_cfg=ic_cfg,
                positions=positions if not positions.empty else None,
                account=account,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"  [monitor] Skipped: {exc}")

    print()
    print(f"{'=' * 55}")
    print("  Tracker complete.")
    if pnl_file.exists():
        print(f"  PnL history: {pnl_file}")
    if not skip_monitor and mon.generate_dashboard:
        print(f"  Dashboard:   {mon.dashboard_path}")
    print(f"{'=' * 55}")


def main() -> None:
    _chdir_root()
    _mon = _load_monitoring_config()
    p = argparse.ArgumentParser(description="Paper performance tracker + rolling IC monitor.")
    p.add_argument(
        "--skip-ic",
        action="store_true",
        help="Disable IC tracker section.",
    )
    p.add_argument(
        "--skip-monitor",
        action="store_true",
        help="Skip monitoring metrics file, alerts, and dashboard.",
    )
    p.add_argument(
        "--ic-recent-days",
        type=int,
        default=int(os.environ.get("IC_TRACKER_RECENT_DAYS", "730")),
        help="Use signal_history rows from the last N calendar days (default 730).",
    )
    p.add_argument(
        "--ic-rolling-window",
        type=int,
        default=_mon.ic_rolling_window,
        help="Rolling mean window (trading days with IC); default from backtest_config monitoring.",
    )
    p.add_argument(
        "--ic-rolling-min",
        type=int,
        default=10,
        help="Min periods for rolling IC (default 10).",
    )
    p.add_argument(
        "--ic-threshold",
        type=float,
        default=_mon.ic_alert_threshold,
        help="IC threshold for alerts (monitoring); default from backtest_config.",
    )
    p.add_argument(
        "--ic-forward-days",
        type=int,
        default=int(os.environ.get("IC_TRACKER_FORWARD_DAYS", "1")),
        help="Forward return horizon in trading days (default 1).",
    )
    p.add_argument(
        "--ic-min-names",
        type=int,
        default=8,
        help="Minimum names with score+fwd return to compute daily IC.",
    )
    p.add_argument(
        "--ic-alert-url",
        type=str,
        default=os.environ.get("IC_ALERT_WEBHOOK", ""),
        help="Optional webhook URL (JSON POST {\"text\": ...}). Also IC_ALERT_WEBHOOK env.",
    )
    args = p.parse_args()
    ic = IcTrackerConfig(
        enabled=not args.skip_ic,
        recent_calendar_days=int(args.ic_recent_days),
        rolling_window=int(args.ic_rolling_window),
        rolling_min_periods=int(args.ic_rolling_min),
        threshold=float(args.ic_threshold),
        forward_trading_days=int(args.ic_forward_days),
        min_names=int(args.ic_min_names),
        alert_url=str(args.ic_alert_url or "").strip(),
    )
    run_tracker(ic=ic, skip_monitor=args.skip_monitor)


if __name__ == "__main__":
    main()
