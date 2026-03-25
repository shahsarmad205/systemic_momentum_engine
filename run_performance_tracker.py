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
import os
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

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
    if np.isfinite(last_roll) and last_roll < float(cfg.threshold):
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


def run_tracker(*, ic: IcTrackerConfig | None = None) -> None:
    _chdir_root()

    ic_cfg = ic if ic is not None else IcTrackerConfig()
    if os.environ.get("IC_TRACKER_SKIP", "").lower() in ("1", "true", "yes"):
        ic_cfg = replace(ic_cfg, enabled=False)

    today = datetime.now().strftime("%Y-%m-%d")

    print()
    print(f"{'=' * 55}")
    print(f"  PERFORMANCE TRACKER — {today}")
    print(f"{'=' * 55}")

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

    print()
    print(f"{'=' * 55}")
    print("  Tracker complete.")
    if pnl_file.exists():
        print(f"  PnL history: {pnl_file}")
    print(f"{'=' * 55}")


def main() -> None:
    p = argparse.ArgumentParser(description="Paper performance tracker + rolling IC monitor.")
    p.add_argument(
        "--skip-ic",
        action="store_true",
        help="Disable IC tracker section.",
    )
    p.add_argument(
        "--ic-recent-days",
        type=int,
        default=int(os.environ.get("IC_TRACKER_RECENT_DAYS", "730")),
        help="Use signal_history rows from the last N calendar days (default 730).",
    )
    p.add_argument("--ic-rolling-window", type=int, default=20, help="Rolling mean window (trading days with IC).")
    p.add_argument(
        "--ic-rolling-min",
        type=int,
        default=10,
        help="Min periods for rolling IC (default 10).",
    )
    p.add_argument("--ic-threshold", type=float, default=0.02, help="Warn if rolling IC falls below this.")
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
    run_tracker(ic=ic)


if __name__ == "__main__":
    main()
