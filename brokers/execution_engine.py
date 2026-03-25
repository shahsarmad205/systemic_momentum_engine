"""
Translate signal rankings into Alpaca orders (dry-run or live).

Uses :class:`AlpacaBroker` for account state and order placement.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from brokers.alpaca_broker import AlpacaBroker
from utils.market_data import get_ohlcv


class ExecutionEngine:
    """Build target weights from rankings and reconcile vs Alpaca positions."""

    def __init__(
        self,
        broker: AlpacaBroker,
        config_path: str = "backtest_config.yaml",
    ) -> None:
        self.broker = broker
        # Drawdown sizing state (populated during execute()).
        self._dd_current_equity: float | None = None
        self._dd_peak_equity: float | None = None
        self._dd_drawdown: float | None = None
        path = Path(config_path)
        if not path.is_file():
            raise FileNotFoundError(
                f"Config not found: {path.resolve()} — run from trend_signal_engine root."
            )
        with open(path, encoding="utf-8") as fh:
            self.config = yaml.safe_load(fh) or {}
        self.bt = self.config.get("backtest", self.config)
        risk = self.config.get("risk", {})

        self.max_position_pct = float(
            risk.get("max_position_pct_of_equity", 0.12)
        )
        # Optional factor risk limits (disabled when missing/None).
        self.max_beta: float | None = (
            float(risk["max_beta"]) if risk.get("max_beta") is not None else None
        )
        self.max_sector_exposure_default: float | None = (
            float(risk["max_sector_exposure"])
            if risk.get("max_sector_exposure") is not None
            else None
        )
        self.max_sector_exposure_by_sector: dict[str, float] = {}
        raw_by_sector = risk.get("max_sector_exposure_by_sector")
        if isinstance(raw_by_sector, dict):
            for k, v in raw_by_sector.items():
                if v is None:
                    continue
                self.max_sector_exposure_by_sector[str(k)] = float(v)
        self.max_single_name_pct: float | None = (
            float(risk["max_single_name_pct"])
            if risk.get("max_single_name_pct") is not None
            else None
        )
        # Paths are relative to trend_signal_engine root when run there.
        self.beta_cache_path = Path(risk.get("beta_cache_path", "output/beta_cache.csv"))
        self.sector_map_path = Path(risk.get("sector_map_path", "output/sector_map.csv"))
        self.max_positions = int(self.bt.get("max_positions", 8))

        sig = self.config.get("signals", {})
        self.min_signal_threshold = float(
            self.bt.get(
                "signal_confidence_multiplier",
                sig.get("signal_confidence_multiplier", 0.8),
            )
        )

    def compute_target_portfolio(
        self,
        signals_df: pd.DataFrame,
        account: dict[str, float],
        *,
        verbose: bool = True,
    ) -> pd.DataFrame:
        equity = float(account["equity"])

        if signals_df.empty or "score" not in signals_df.columns:
            if verbose:
                print("  [Execution] No scores in signals_df")
            return pd.DataFrame()
        if "ticker" not in signals_df.columns:
            raise ValueError("signals_df must include a 'ticker' column")

        score_std = float(signals_df["score"].std(ddof=0) or 0.0)
        if score_std <= 0 or not (score_std == score_std):  # nan check
            threshold = 0.0
        else:
            threshold = self.min_signal_threshold * score_std
        strong = signals_df[signals_df["score"].abs() > threshold].copy()

        longs = strong[strong["score"] > 0].head(self.max_positions)

        if len(longs) == 0:
            if verbose:
                print("  [Execution] No signals above threshold")
            return pd.DataFrame()

        weight_per_position = 1.0 / len(longs)
        max_weight = self.max_position_pct
        weight_per_position = min(weight_per_position, max_weight)

        longs = longs.copy()
        longs["target_weight"] = weight_per_position
        longs["target_value"] = longs["target_weight"] * equity

        if verbose:
            print(f"  [Execution] Target: {len(longs)} positions")
            print(f"  [Execution] Weight each: {weight_per_position:.1%}")
            print(f"  [Execution] Equity: ${equity:,.2f}")

        return longs

    def _load_beta_cache(self) -> dict[str, float]:
        if not self.beta_cache_path.exists():
            return {}
        try:
            df = pd.read_csv(self.beta_cache_path)
        except Exception:
            return {}
        if df.empty:
            return {}
        cols = {c.lower(): c for c in df.columns}
        tcol = cols.get("ticker") or cols.get("symbol")
        bcol = cols.get("beta")
        if not tcol or not bcol:
            return {}
        out: dict[str, float] = {}
        for _, row in df.iterrows():
            t = str(row[tcol]).strip()
            try:
                b = float(row[bcol])
            except Exception:
                continue
            if not t or not (b == b):
                continue
            out[t.upper()] = b
        return out

    def _append_beta_cache(self, betas: dict[str, float]) -> None:
        if not betas:
            return
        self.beta_cache_path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._load_beta_cache()
        merged = {**existing, **{k.upper(): float(v) for k, v in betas.items()}}
        df = pd.DataFrame([{"ticker": k, "beta": v} for k, v in sorted(merged.items())])
        try:
            df.to_csv(self.beta_cache_path, index=False)
        except Exception:
            return

    def _compute_beta_vs_spy(
        self,
        ticker: str,
        *,
        lookback_trading_days: int = 252,
    ) -> float | None:
        end = pd.Timestamp(datetime.now().date())
        start = end - pd.Timedelta(days=int(lookback_trading_days * 2.2))
        try:
            s_df = get_ohlcv(ticker, start.date().isoformat(), end.date().isoformat())
            m_df = get_ohlcv("SPY", start.date().isoformat(), end.date().isoformat())
        except Exception:
            return None

        if s_df is None or m_df is None or s_df.empty or m_df.empty:
            return None
        if "Close" not in s_df.columns or "Close" not in m_df.columns:
            return None

        s_ret = pd.to_numeric(s_df["Close"], errors="coerce").pct_change()
        m_ret = pd.to_numeric(m_df["Close"], errors="coerce").pct_change()
        both = pd.concat({"s": s_ret, "m": m_ret}, axis=1).dropna()
        if len(both) < max(30, int(lookback_trading_days * 0.5)):
            return None
        both = both.iloc[-lookback_trading_days:]
        if len(both) < 30:
            return None
        m_var = float(both["m"].var(ddof=0))
        if m_var <= 0:
            return None
        cov = float(
            ((both["s"] - both["s"].mean()) * (both["m"] - both["m"].mean())).mean()
        )
        return cov / m_var

    def _load_sector_map(self) -> dict[str, str]:
        if not self.sector_map_path.exists():
            return {}
        try:
            df = pd.read_csv(self.sector_map_path)
        except Exception:
            return {}
        if df.empty:
            return {}
        cols = {c.lower(): c for c in df.columns}
        tcol = cols.get("ticker") or cols.get("symbol")
        scol = cols.get("sector")
        if not tcol or not scol:
            return {}
        out: dict[str, str] = {}
        for _, row in df.iterrows():
            t = str(row[tcol]).strip().upper()
            s = str(row[scol]).strip()
            if t and s and s.lower() != "nan":
                out[t] = s
        return out

    def _append_sector_map(self, mapping: dict[str, str]) -> None:
        if not mapping:
            return
        self.sector_map_path.parent.mkdir(parents=True, exist_ok=True)
        existing = self._load_sector_map()
        merged = {**existing, **{k.upper(): str(v) for k, v in mapping.items()}}
        df = pd.DataFrame(
            [{"ticker": k, "sector": v} for k, v in sorted(merged.items())]
        )
        try:
            df.to_csv(self.sector_map_path, index=False)
        except Exception:
            return

    def _get_sector_yahoo_best_effort(self, ticker: str) -> str | None:
        try:
            import yfinance as yf

            info = yf.Ticker(ticker).info or {}
            sector = info.get("sector")
            if sector and isinstance(sector, str):
                sector = sector.strip()
                return sector or None
            return None
        except Exception:
            return None

    def _sector_cap_for(self, sector: str) -> float | None:
        if sector in self.max_sector_exposure_by_sector:
            return float(self.max_sector_exposure_by_sector[sector])
        return self.max_sector_exposure_default

    def _apply_factor_risk_limits_to_open_set(
        self,
        target: pd.DataFrame,
        to_open: set[str],
        to_hold: set[str],
    ) -> tuple[set[str], list[dict[str, Any]]]:
        skipped: list[dict[str, Any]] = []
        if target is None or target.empty:
            return to_open, skipped

        # Single-name cap (prefer the dedicated knob if provided).
        if self.max_single_name_pct is not None:
            cap = float(self.max_single_name_pct)
            if cap > 0 and "target_weight" in target.columns:
                tw = pd.to_numeric(target["target_weight"], errors="coerce").fillna(0.0)
                too_big = target[tw > cap]
                for t in too_big["ticker"].astype(str).tolist():
                    t_u = t.strip().upper()
                    if t_u in to_open:
                        to_open.remove(t_u)
                        skipped.append(
                            {
                                "ticker": t_u,
                                "reason": "max_single_name_pct",
                                "detail": f"target_weight>{cap:.2%}",
                            }
                        )

        if (
            self.max_beta is None
            and self.max_sector_exposure_default is None
            and not self.max_sector_exposure_by_sector
        ):
            return to_open, skipped

        beta_cache = self._load_beta_cache()
        sector_map = self._load_sector_map()
        new_betas: dict[str, float] = {}
        new_sectors: dict[str, str] = {}

        tdf = target.copy()
        if "score" in tdf.columns:
            tdf["score"] = pd.to_numeric(tdf["score"], errors="coerce")
            tdf = tdf.sort_values(["score"], ascending=False)
        if "target_weight" in tdf.columns:
            tdf["target_weight"] = (
                pd.to_numeric(tdf["target_weight"], errors="coerce").fillna(0.0)
            )

        sector_exposure: dict[str, float] = {}
        current_beta = 0.0

        # Seed exposures with positions we intend to keep (hold set).
        for _, row in tdf.iterrows():
            ticker = str(row.get("ticker", "")).strip().upper()
            if not ticker or ticker not in to_hold:
                continue
            w = float(row.get("target_weight", 0.0) or 0.0)
            if w <= 0:
                continue

            if self.max_beta is not None:
                b = beta_cache.get(ticker)
                if b is None:
                    b_calc = self._compute_beta_vs_spy(ticker)
                    if b_calc is not None:
                        b = float(b_calc)
                        new_betas[ticker] = b
                if b is None:
                    b = 1.0
                current_beta += w * float(b)

            if self.max_sector_exposure_default is not None or self.max_sector_exposure_by_sector:
                sec = sector_map.get(ticker)
                if sec is None:
                    sec_calc = self._get_sector_yahoo_best_effort(ticker)
                    if sec_calc is not None:
                        sec = sec_calc
                        new_sectors[ticker] = sec
                if sec is None:
                    sec = "Unknown"
                sector_exposure[sec] = float(sector_exposure.get(sec, 0.0) + w)

        filtered_open: set[str] = set()
        for _, row in tdf.iterrows():
            ticker = str(row.get("ticker", "")).strip().upper()
            if ticker not in to_open:
                continue
            w = float(row.get("target_weight", 0.0) or 0.0)
            if w <= 0:
                continue

            if self.max_beta is not None:
                b = beta_cache.get(ticker)
                if b is None:
                    b_calc = self._compute_beta_vs_spy(ticker)
                    if b_calc is not None:
                        b = float(b_calc)
                        new_betas[ticker] = b
                if b is None:
                    b = 1.0
                proposed_beta = current_beta + (w * float(b))
                if proposed_beta > float(self.max_beta):
                    skipped.append(
                        {
                            "ticker": ticker,
                            "reason": "max_beta",
                            "detail": f"proposed_portfolio_beta={proposed_beta:.3f} > cap={float(self.max_beta):.3f} (beta={float(b):.3f}, w={w:.2%})",
                        }
                    )
                    continue

            if self.max_sector_exposure_default is not None or self.max_sector_exposure_by_sector:
                sec = sector_map.get(ticker)
                if sec is None:
                    sec_calc = self._get_sector_yahoo_best_effort(ticker)
                    if sec_calc is not None:
                        sec = sec_calc
                        new_sectors[ticker] = sec
                if sec is None:
                    sec = "Unknown"
                cap = self._sector_cap_for(sec)
                if cap is not None and cap > 0:
                    proposed = float(sector_exposure.get(sec, 0.0) + w)
                    if proposed > float(cap):
                        skipped.append(
                            {
                                "ticker": ticker,
                                "reason": "max_sector_exposure",
                                "detail": f"sector={sec} proposed={proposed:.2%} > cap={float(cap):.2%}",
                            }
                        )
                        continue

            filtered_open.add(ticker)
            if self.max_beta is not None:
                b2 = beta_cache.get(ticker, new_betas.get(ticker, 1.0))
                current_beta += w * float(b2)
            if self.max_sector_exposure_default is not None or self.max_sector_exposure_by_sector:
                sec2 = sector_map.get(ticker, new_sectors.get(ticker, "Unknown"))
                sector_exposure[sec2] = float(sector_exposure.get(sec2, 0.0) + w)

        self._append_beta_cache(new_betas)
        self._append_sector_map(new_sectors)

        return filtered_open, skipped

    def reconcile(
        self,
        target: pd.DataFrame,
        current_positions: pd.DataFrame,
        *,
        verbose: bool = True,
    ) -> dict[str, set[str]]:
        target_tickers = (
            set(target["ticker"].astype(str).tolist()) if len(target) > 0 else set()
        )
        if current_positions is None or current_positions.empty:
            current_tickers = set()
        else:
            current_tickers = set(current_positions["ticker"].astype(str).tolist())

        to_open = target_tickers - current_tickers
        to_close = current_tickers - target_tickers
        to_hold = target_tickers & current_tickers

        filtered_to_open, skipped = self._apply_factor_risk_limits_to_open_set(
            target=target,
            to_open=set(str(t).strip().upper() for t in to_open),
            to_hold=set(str(t).strip().upper() for t in to_hold),
        )
        if skipped and verbose:
            print("  [Risk] Skipped opens due to risk limits:")
            for item in skipped:
                print(
                    f"    SKIP {item.get('ticker')} — {item.get('reason')}: {item.get('detail')}"
                )
        to_open = filtered_to_open

        if verbose:
            print(f"  [Reconcile] Open:  {sorted(to_open)}")
            print(f"  [Reconcile] Close: {sorted(to_close)}")
            print(f"  [Reconcile] Hold:  {sorted(to_hold)}")

        return {
            "to_open": to_open,
            "to_close": to_close,
            "to_hold": to_hold,
        }

    def _append_execution_log(self, log: dict[str, Any]) -> Path:
        log_dir = Path("output/live")
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / "execution_log.jsonl"
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log, default=str) + "\n")
        return log_file

    def _load_equity_history(self) -> list[float]:
        """
        Load historical equity values for drawdown-aware sizing.

        Source: output/live/daily_pnl.csv (written by run_performance_tracker.py).
        Returns empty list when the file doesn't exist or can't be parsed.
        """
        pnl_file = Path("output/live/daily_pnl.csv")
        if not pnl_file.exists():
            return []
        try:
            df = pd.read_csv(pnl_file)
            col = "equity" if "equity" in df.columns else "portfolio_value"
            if col not in df.columns:
                return []
            s = pd.to_numeric(df[col], errors="coerce").dropna()
            return [float(x) for x in s.tolist()]
        except Exception:
            return []

    def _get_drawdown_multiplier(self) -> float:
        """
        Drawdown-aware sizing multiplier.

        - Reads output/live/daily_pnl.csv (if present) to get equity history.
        - If the file doesn't exist, is malformed, or has < 2 entries: returns 1.0.
        - Uses peak equity from (history + current equity) and computes:
            drawdown = (peak - current_equity) / peak
        """
        current_equity = float(self._dd_current_equity or 0.0)

        # Default / fail-safe values.
        self._dd_peak_equity = current_equity if current_equity > 0 else None
        self._dd_drawdown = 0.0

        pnl_file = Path("output/live/daily_pnl.csv")
        if not pnl_file.exists():
            return 1.0

        try:
            df = pd.read_csv(pnl_file)
        except Exception:
            return 1.0

        col = "equity" if "equity" in df.columns else "portfolio_value"
        if col not in df.columns:
            return 1.0

        s = pd.to_numeric(df[col], errors="coerce").dropna()
        # Spec: if < 2 entries, do not use drawdown scaling.
        if len(s) < 2:
            return 1.0

        vals = [float(x) for x in s.tolist()]
        if current_equity > 0:
            vals.append(current_equity)
        vals = [x for x in vals if x == x and x > 0]  # drop NaN and non-positive
        if not vals:
            return 1.0

        peak = float(max(vals))
        self._dd_peak_equity = peak
        if peak <= 0 or current_equity <= 0:
            self._dd_drawdown = 0.0
            return 1.0

        drawdown = float((peak - current_equity) / peak)
        self._dd_drawdown = drawdown

        if drawdown < 0.05:
            return 1.0
        if drawdown < 0.10:
            return 0.8
        if drawdown < 0.15:
            return 0.6
        return 0.4

    def preview_planned_open_buy_notional(self, signals_df: pd.DataFrame) -> dict[str, Any]:
        """
        Same planning as execute() (target weights, drawdown multiplier, reconcile + risk filters),
        without placing orders. Sums target notional for new buys that would pass the min-$100 check.
        """
        account = self.broker.get_account()
        self._dd_current_equity = float(account["equity"])
        dd_mult = self._get_drawdown_multiplier()
        current_positions = self.broker.get_positions()
        target = self.compute_target_portfolio(signals_df, account, verbose=False)
        if len(target) > 0 and dd_mult != 1.0:
            target = target.copy()
            target["target_value"] = pd.to_numeric(target["target_value"], errors="coerce").fillna(0.0)
            target["target_value"] = target["target_value"] * float(dd_mult)
        if len(target) == 0:
            return {
                "total_open_notional": 0.0,
                "planned_buys": [],
                "cash": float(account["cash"]),
                "buying_power": float(account.get("buying_power", account["cash"])),
                "drawdown_multiplier": float(dd_mult),
            }
        changes = self.reconcile(target, current_positions, verbose=False)
        total = 0.0
        planned: list[dict[str, Any]] = []
        for ticker in sorted(changes["to_open"]):
            row = target[target["ticker"].astype(str) == ticker].iloc[0]
            notional = float(row["target_value"])
            if notional < 100:
                continue
            total += notional
            planned.append({"ticker": ticker, "notional": notional})
        return {
            "total_open_notional": float(total),
            "planned_buys": planned,
            "cash": float(account["cash"]),
            "buying_power": float(account.get("buying_power", account["cash"])),
            "drawdown_multiplier": float(dd_mult),
            "to_open": sorted(changes["to_open"]),
        }

    def execute(
        self,
        signals_df: pd.DataFrame,
        dry_run: bool = True,
    ) -> dict[str, Any]:
        print()
        print(f"{'=' * 50}")
        print(f"  EXECUTION ENGINE ({'DRY RUN' if dry_run else 'LIVE'})")
        print(f"{'=' * 50}")

        account = self.broker.get_account()
        print(f"  Equity: ${account['equity']:,.2f}")
        print(f"  Cash:   ${account['cash']:,.2f}")

        self._dd_current_equity = float(account["equity"])
        dd_mult = self._get_drawdown_multiplier()
        current_equity = float(self._dd_current_equity)
        peak_equity = float(self._dd_peak_equity or current_equity)
        drawdown = float(self._dd_drawdown or 0.0)
        print(f"  Drawdown: {drawdown:.2%}  (peak=${peak_equity:,.2f})")
        print(f"  Drawdown multiplier: {dd_mult:.2f}")

        current_positions = self.broker.get_positions()

        target = self.compute_target_portfolio(signals_df, account, verbose=True)
        if len(target) > 0 and dd_mult != 1.0:
            target = target.copy()
            target["target_value"] = pd.to_numeric(target["target_value"], errors="coerce").fillna(0.0)
            target["target_value"] = target["target_value"] * float(dd_mult)

        if len(target) == 0:
            print("  No trades to execute")
            skip_log: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "dry_run": dry_run,
                "equity": account["equity"],
                "peak_equity": float(peak_equity),
                "drawdown": float(drawdown),
                "drawdown_multiplier": float(dd_mult),
                "skipped": True,
                "reason": "no_target_positions",
                "n_open": 0,
                "n_close": 0,
                "n_hold": 0,
                "orders_placed": [],
                "orders_skipped": [],
            }
            log_file = self._append_execution_log(skip_log)
            print(f"  Log saved: {log_file}")
            print(f"{'=' * 50}")
            return skip_log

        changes = self.reconcile(target, current_positions, verbose=True)

        orders_placed: list[Any] = []
        orders_skipped: list[str] = []
        close_errors: list[dict[str, Any]] = []

        if dry_run:
            print()
            print("  [DRY RUN] Would execute:")

        for ticker in sorted(changes["to_close"]):
            sym = str(ticker).strip().upper()
            if dry_run:
                print(f"    CLOSE {ticker}")
            else:
                try:
                    result = self.broker.close_position(sym)
                except Exception as e:  # noqa: BLE001 — log unexpected broker errors
                    print(f"    [Close] FAILED {sym}: {e}")
                    result = {
                        "success": False,
                        "ticker": sym,
                        "error": str(e),
                    }
                orders_placed.append(result)
                if not result.get("success"):
                    close_errors.append(
                        {
                            "ticker": sym,
                            "error": result.get("error", "close_failed"),
                        }
                    )

        skip_opens_for_bp = False
        if not dry_run and len(changes["to_close"]) > 0:
            time.sleep(5)
            account = self.broker.get_account()
            current_positions = self.broker.get_positions()
            bp = float(account.get("buying_power", account["cash"]))
            print()
            print(
                "  Refreshed after closes — "
                f"Equity ${account['equity']:,.2f} | "
                f"Cash ${account['cash']:,.2f} | "
                f"BP ${bp:,.2f}"
            )
            if changes["to_open"] and bp < 1.0:
                skip_opens_for_bp = True
                print()
                print(
                    "  [!] Skipping buy leg: buying power still ~$0 — close orders are likely "
                    "unfilled (e.g. session closed). Re-run after the market opens and fills, "
                    "or run --execute during RTH so sells can fill before buys."
                )

        for ticker in sorted(changes["to_open"]):
            row = target[target["ticker"].astype(str) == ticker].iloc[0]
            notional = float(row["target_value"])

            if notional < 100:
                print(f"    SKIP {ticker}: too small ${notional:.0f}")
                orders_skipped.append(ticker)
                continue

            if skip_opens_for_bp:
                print(f"    SKIP {ticker}: no buying power (deferred)")
                orders_skipped.append(ticker)
                continue

            if dry_run:
                print(f"    BUY  {ticker} ${notional:,.2f}")
            else:
                result = self.broker.place_order(
                    ticker=ticker,
                    side="buy",
                    notional=notional,
                )
                orders_placed.append(result)

        log: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            "equity": account["equity"],
            "peak_equity": float(peak_equity),
            "drawdown": float(drawdown),
            "drawdown_multiplier": float(dd_mult),
            "skipped": False,
            "n_open": len(changes["to_open"]),
            "n_close": len(changes["to_close"]),
            "n_hold": len(changes["to_hold"]),
            "orders_placed": orders_placed,
            "orders_skipped": orders_skipped,
            "close_errors": close_errors,
            "skip_opens_for_bp": skip_opens_for_bp,
        }

        log_file = self._append_execution_log(log)

        print()
        print(f"  Log saved: {log_file}")
        print(f"{'=' * 50}")

        return log
