"""
Translate signal rankings into Alpaca orders (dry-run or live).

Uses :class:`AlpacaBroker` for account state and order placement.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from brokers.alpaca_broker import AlpacaBroker
from utils.adv_cache import (
    get_adv_shares,
    latest_close_from_ohlcv_parquet,
    latest_open_from_ohlcv_parquet,
)
from utils.live_trades import append_trade_row
from utils.market_data import get_ohlcv
from utils.risk_utils import load_beta_cache, load_sector_mapping
from utils.trading_control import is_live_trading_allowed, trading_halt_reason


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
        # Long/short exposure controls
        self.max_gross_exposure = float(risk.get("max_gross_exposure", 1.0) or 1.0)
        self.max_net_exposure = float(risk.get("max_net_exposure", 1.0) or 1.0)
        self.max_short_single_name = float(risk.get("max_short_single_name", 0.0) or 0.0)
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
        self.beta_lookback_days = int(risk.get("beta_lookback_days", 252))
        self.compute_beta_on_fly = bool(risk.get("compute_beta_on_fly", False))
        self.max_positions = int(self.bt.get("max_positions", 8))
        self.max_longs = int(self.bt.get("max_longs", (self.max_positions + 1) // 2) or 0)
        self.max_shorts = int(self.bt.get("max_shorts", self.max_positions // 2) or 0)
        ex = self.config.get("execution") or {}
        if not isinstance(ex, dict):
            ex = {}
        self.enable_shorts = bool(ex.get("enable_shorts", False))
        self.long_only = bool(ex.get("long_only", True))
        if not self.enable_shorts or self.long_only:
            self.max_shorts = 0
            self.max_longs = self.max_positions

        # Optional greedy factor filter block (see _apply_factor_limits); when non-empty,
        # overrides the paths/limits above where keys are present and disables open-only
        # filtering in _apply_factor_risk_limits_to_open_set (to avoid double application).
        rf = self.config.get("risk_factors") or {}
        self._use_greedy_factor_limits = bool(isinstance(rf, dict) and rf)
        if isinstance(rf, dict) and rf:
            if rf.get("max_beta") is not None:
                self.max_beta = float(rf["max_beta"])
            if rf.get("max_sector_exposure") is not None:
                self.max_sector_exposure_default = float(rf["max_sector_exposure"])
            if rf.get("max_single_name_pct") is not None:
                self.max_single_name_pct = float(rf["max_single_name_pct"])
            if rf.get("beta_cache_path"):
                self.beta_cache_path = Path(str(rf["beta_cache_path"]))
            sect_path = rf.get("sector_mapping_path") or rf.get("sector_map_path")
            if sect_path:
                self.sector_map_path = Path(str(sect_path))
            if rf.get("beta_lookback_days") is not None:
                self.beta_lookback_days = int(rf["beta_lookback_days"])
            if rf.get("compute_beta_on_fly") is not None:
                self.compute_beta_on_fly = bool(rf["compute_beta_on_fly"])

        liq_cfg: dict[str, Any] = {}
        if isinstance(rf, dict):
            liq_cfg = rf.get("liquidity") or {}
        if not isinstance(liq_cfg, dict):
            liq_cfg = {}
        self.liquidity_enabled = bool(liq_cfg.get("enabled", False))
        self.max_adv_pct = float(liq_cfg.get("max_adv_pct", 0.05))
        self.adv_lookback_days = int(liq_cfg.get("adv_lookback_days", 20))
        self.adv_cache_path = Path(str(liq_cfg.get("adv_cache_path", "output/adv_cache.csv")))
        self.adv_refresh_on_run = bool(liq_cfg.get("refresh_cache_on_run", False))
        self._ohlcv_cache_dir = Path(str(self.bt.get("cache_dir", "data/cache/ohlcv")))

        sig = self.config.get("signals", {})
        self.min_signal_threshold = float(
            self.bt.get(
                "signal_confidence_multiplier",
                sig.get("signal_confidence_multiplier", 0.8),
            )
        )

        st = self.config.get("slippage_tracking") or {}
        if not isinstance(st, dict):
            st = {}
        self.slippage_tracking_enabled = bool(st.get("enabled", False))
        self.slippage_signal_price_source = str(st.get("signal_price_source", "close")).strip().lower()
        engine_root = path.resolve().parent
        pend = st.get("pending_trades_file") or st.get("output_path") or "output/live/trades_pending.csv"
        tdone = st.get("trades_file", "output/live/trades.csv")
        self.slippage_pending_path = engine_root / str(pend)
        self.slippage_trades_path = engine_root / str(tdone)
        self._engine_root = engine_root

        live = self.config.get("live") or {}
        self._trading_halt_env = str(
            (live.get("trading_halt_env") if isinstance(live, dict) else None) or "TRADING_HALTED"
        ).strip()

    def _signal_reference_price(self, ticker: str) -> tuple[float | None, str]:
        """Reference price for slippage vs signal (OHLCV cache)."""
        t = str(ticker).strip().upper()
        cache_root = self._ohlcv_cache_dir
        if not cache_root.is_absolute():
            cache_root = Path.cwd() / cache_root
        if self.slippage_signal_price_source == "open":
            px = latest_open_from_ohlcv_parquet(t, cache_root)
            return px, "open"
        px = latest_close_from_ohlcv_parquet(t, cache_root)
        return px, "close"

    def _record_slippage_trade(
        self,
        *,
        dry_run: bool,
        ticker: str,
        side: str,
        notional: float | None,
        signal_price: float | None,
        signal_price_source: str,
        result: dict[str, Any],
    ) -> None:
        if not self.slippage_tracking_enabled or dry_run:
            return
        row: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "ticker": str(ticker).strip().upper(),
            "side": str(side).lower(),
            "notional": notional,
            "signal_price": signal_price,
            "signal_price_source": signal_price_source,
            "order_id": result.get("order_id"),
            "status": result.get("status"),
            "dry_run": False,
            "filled_qty": result.get("filled_qty"),
            "filled_avg_price": result.get("filled_avg_price"),
            "slippage_bps": result.get("slippage_bps"),
            "filled_at": result.get("filled_at"),
        }
        append_trade_row(self.slippage_pending_path, row)

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

        longs = strong[strong["score"] > 0].copy().sort_values("score", ascending=False).head(self.max_longs)
        shorts = (
            strong[strong["score"] < 0].copy().sort_values("score", ascending=True).head(self.max_shorts)
            if self.max_shorts > 0
            else strong.iloc[0:0].copy()
        )
        target = pd.concat([longs, shorts], ignore_index=True)
        if target.empty:
            if verbose:
                print("  [Execution] No signals above threshold")
            return pd.DataFrame()
        # Equal-weight magnitude (capped). Shorts are negative weights.
        w_each = min(1.0 / max(1, len(target)), float(self.max_position_pct))
        target = target.copy()
        target["target_weight"] = target["score"].apply(lambda s: w_each if float(s) > 0 else -w_each)

        # Enforce single-name short cap if provided
        short_cap = float(self.max_short_single_name) if self.max_short_single_name > 0 else float(self.max_position_pct)
        def _cap_w(w: float) -> float:
            if w > 0:
                return min(w, float(self.max_position_pct))
            return max(w, -short_cap)
        target["target_weight"] = target["target_weight"].astype(float).apply(_cap_w)

        # Enforce net exposure cap by scaling the dominant side
        long_sum = float(target[target["target_weight"] > 0]["target_weight"].sum())
        short_sum = float(-target[target["target_weight"] < 0]["target_weight"].sum())
        net = long_sum - short_sum
        cap_net = float(self.max_net_exposure)
        if cap_net > 0 and abs(net) > cap_net and (long_sum > 1e-12 or short_sum > 1e-12):
            if net > cap_net and long_sum > 1e-12:
                desired_long = cap_net + short_sum
                sL = max(0.0, min(1.0, desired_long / long_sum))
                target.loc[target["target_weight"] > 0, "target_weight"] *= sL
            elif net < -cap_net and short_sum > 1e-12:
                desired_short = long_sum + cap_net
                sS = max(0.0, min(1.0, desired_short / short_sum))
                target.loc[target["target_weight"] < 0, "target_weight"] *= sS

        # Enforce gross exposure cap by scaling both sides
        gross = float(target["target_weight"].abs().sum())
        cap_gross = float(self.max_gross_exposure) if self.max_gross_exposure > 0 else 1.0
        if cap_gross > 0 and gross > cap_gross and gross > 1e-12:
            scale = cap_gross / gross
            target["target_weight"] *= scale

        target["target_value"] = target["target_weight"] * equity

        if verbose:
            gross = float(target["target_weight"].abs().sum())
            net = float(target["target_weight"].sum())
            print(f"  [Execution] Target: {len(target)} positions (longs={int((target['target_weight']>0).sum())}, shorts={int((target['target_weight']<0).sum())})")
            print(f"  [Execution] Gross exposure: {gross:.2f}x  Net exposure: {net:.2f}x")
            print(f"  [Execution] Equity: ${equity:,.2f}")

        return target

    def _load_beta_cache(self) -> dict[str, float]:
        return load_beta_cache(self.beta_cache_path)

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
        lookback_trading_days: int | None = None,
    ) -> float | None:
        lb = int(
            lookback_trading_days
            if lookback_trading_days is not None
            else self.beta_lookback_days
        )
        end = pd.Timestamp(datetime.now().date())
        start = end - pd.Timedelta(days=int(lb * 2.2))
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
        if len(both) < max(30, int(lb * 0.5)):
            return None
        both = both.iloc[-lb:]
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
        return load_sector_mapping(self.sector_map_path)

    def _resolve_beta(self, ticker: str, beta_cache: dict[str, float]) -> float:
        t = str(ticker).strip().upper()
        b = beta_cache.get(t)
        if b is not None:
            return float(b)
        if self.compute_beta_on_fly:
            calc = self._compute_beta_vs_spy(t)
            if calc is not None:
                return float(calc)
        return 1.0

    def _liquidity_adv_check(self, ticker: str, notional: float) -> tuple[bool, str]:
        """
        True if proposed dollar notional is not too large vs ADV * price (max_adv_pct cap).

        Uses last OHLCV close in ``cache_dir`` for price; ADV from CSV or parquet.
        """
        t = str(ticker).strip().upper()
        if notional <= 0 or not self.liquidity_enabled:
            return True, ""

        cache_root = self._ohlcv_cache_dir
        if not cache_root.is_absolute():
            cache_root = Path.cwd() / cache_root

        price = latest_close_from_ohlcv_parquet(t, cache_root)
        if price is None or price <= 0:
            return False, "no_ohlcv_close_for_liquidity_check"

        shares_order = notional / price
        adv_path = self.adv_cache_path
        if not adv_path.is_absolute():
            adv_path = Path.cwd() / adv_path

        adv, src = get_adv_shares(
            t,
            cache_dir=cache_root,
            adv_cache_path=adv_path,
            lookback=self.adv_lookback_days,
            refresh=self.adv_refresh_on_run,
        )
        cap_shares = float(self.max_adv_pct) * adv
        if shares_order > cap_shares:
            return (
                False,
                (
                    f"shares_order≈{shares_order:.0f} > max_adv_pct*ADV≈{cap_shares:.0f} "
                    f"(ADV={adv:.0f} [{src}], px={price:.2f}, notional=${notional:,.0f})"
                ),
            )
        return True, ""

    def _apply_factor_limits(
        self,
        target: pd.DataFrame,
        current_positions: pd.DataFrame,
        account: dict[str, float],
        *,
        sizing_mult: float = 1.0,
        verbose: bool = True,
    ) -> tuple[pd.DataFrame, list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Greedy subset of target (score order) so that, after assigning equal weight among
        picks (capped by max_position_pct), portfolio beta / sector / single-name / ADV
        limits are not violated vs. a delta from current holdings.

        Returns filtered target, factor skip records, and liquidity skip records.
        """
        skips: list[dict[str, Any]] = []
        liquidity_skips: list[dict[str, Any]] = []
        if target is None or target.empty:
            return target, skips, liquidity_skips

        equity = float(account.get("equity") or 0.0)
        if equity <= 0:
            return target, skips, liquidity_skips

        no_factor_limits = (
            self.max_beta is None
            and self.max_sector_exposure_default is None
            and not self.max_sector_exposure_by_sector
            and self.max_single_name_pct is None
        )
        if no_factor_limits and not self.liquidity_enabled:
            return target, skips, liquidity_skips

        beta_cache: dict[str, float] = {}
        sector_map: dict[str, str] = {}
        cur_w: dict[str, float] = {}
        port_beta = 0.0
        sec_used: dict[str, float] = {}

        if not no_factor_limits:
            beta_cache = self._load_beta_cache()
            sector_map = self._load_sector_map()
            if current_positions is not None and not current_positions.empty:
                for _, pr in current_positions.iterrows():
                    sym = str(pr.get("ticker", "")).strip().upper()
                    if not sym:
                        continue
                    mv = float(pd.to_numeric(pr.get("market_value"), errors="coerce") or 0.0)
                    if abs(mv) <= 1e-9:
                        continue
                    cur_w[sym] = cur_w.get(sym, 0.0) + mv / equity

            for sym, w in cur_w.items():
                b = self._resolve_beta(sym, beta_cache)
                port_beta += w * b
                sec = sector_map.get(sym)
                if sec is None:
                    sec_calc = self._get_sector_yahoo_best_effort(sym)
                    sec = sec_calc or "Unknown"
                sec_used[sec] = sec_used.get(sec, 0.0) + abs(float(w))

        sm = float(sizing_mult) if (sizing_mult == sizing_mult) else 1.0
        if sm <= 0:
            sm = 1.0
        base_w = min(1.0 / max(1, self.max_positions), float(self.max_position_pct))
        cand_w_mag = base_w * sm
        if cand_w_mag <= 0:
            return target, skips, liquidity_skips

        tdf = target.copy()
        if "score" in tdf.columns:
            tdf["_score"] = pd.to_numeric(tdf["score"], errors="coerce")
            tdf = tdf.sort_values("_score", ascending=False)
        chosen_rows: list[pd.Series] = []

        for _, row in tdf.iterrows():
            if len(chosen_rows) >= self.max_positions:
                break
            t = str(row.get("ticker", "")).strip().upper()
            if not t:
                continue

            reason: str | None = None
            detail: str | None = None
            b = 1.0
            w_old = 0.0
            delta_w = 0.0
            sec = "Unknown"
            cap_sec: float | None = None
            score = float(pd.to_numeric(row.get("score"), errors="coerce") or 0.0)
            side_sign = 1.0 if score > 0 else -1.0 if score < 0 else 0.0
            if side_sign == 0.0:
                continue
            cand_w = float(side_sign) * float(cand_w_mag)

            if not no_factor_limits:
                b = self._resolve_beta(t, beta_cache)
                w_old = float(cur_w.get(t, 0.0))
                delta_w = cand_w - w_old

                sec = sector_map.get(t) or ""
                if not sec:
                    sec_calc = self._get_sector_yahoo_best_effort(t)
                    sec = sec_calc or "Unknown"
                cap_sec = self._sector_cap_for(sec)

                if self.max_single_name_pct is not None and abs(cand_w) > float(
                    self.max_single_name_pct
                ):
                    reason = "max_single_name_pct"
                    detail = f"|w_each|={abs(cand_w):.2%} > cap={float(self.max_single_name_pct):.2%}"
                elif self.max_beta is not None:
                    new_beta = port_beta + delta_w * b
                    if new_beta > float(self.max_beta):
                        reason = "max_beta"
                        detail = (
                            f"proposed_portfolio_beta={new_beta:.3f} > cap={float(self.max_beta):.3f} "
                            f"(beta={b:.3f}, delta_w={delta_w:+.2%})"
                        )
                if reason is None and cap_sec is not None and cap_sec > 0:
                    new_sec = sec_used.get(sec, 0.0) + abs(delta_w)
                    if new_sec > float(cap_sec):
                        reason = "max_sector_exposure"
                        detail = (
                            f"sector={sec} proposed={new_sec:.2%} > cap={float(cap_sec):.2%}"
                        )

            if reason is not None:
                skips.append({"ticker": t, "reason": reason, "detail": detail or ""})
                if verbose:
                    print(f"    SKIP {t} — {reason}: {detail}")
                continue

            if self.liquidity_enabled:
                notion_chk = abs(cand_w) * equity
                ok_liq, liq_detail = self._liquidity_adv_check(t, notion_chk)
                if not ok_liq:
                    liquidity_skips.append(
                        {
                            "ticker": t,
                            "reason": "liquidity_adv",
                            "detail": liq_detail,
                        }
                    )
                    if verbose:
                        print(f"    SKIP {t} — liquidity_adv: {liq_detail}")
                    continue

            chosen_rows.append(row)
            if not no_factor_limits:
                port_beta += delta_w * b
                sec_used[sec] = sec_used.get(sec, 0.0) + abs(delta_w)
                cur_w[t] = cand_w

        if not chosen_rows:
            return pd.DataFrame(), skips, liquidity_skips

        out = pd.DataFrame(chosen_rows)
        if "_score" in out.columns:
            out = out.drop(columns=["_score"], errors="ignore")

        # Assign equal weights; if liquidity is on, drop names that violate ADV and renormalize
        # until stable (renormalizing raises per-name notional when n falls).
        while len(out) > 0:
            n = len(out)
            w_each = min(1.0 / n, float(self.max_position_pct)) * sm
            out = out.copy()
            scores = pd.to_numeric(out.get("score"), errors="coerce").fillna(0.0)
            sign = scores.apply(lambda s: 1.0 if float(s) > 0 else (-1.0 if float(s) < 0 else 0.0))
            out["target_weight"] = sign * float(w_each)
            out["target_value"] = out["target_weight"] * float(equity)
            if not self.liquidity_enabled:
                break
            fail_info: list[tuple[Any, str, str]] = []
            for i, srow in out.iterrows():
                t2 = str(srow.get("ticker", "")).strip().upper()
                tv = abs(float(pd.to_numeric(srow.get("target_value"), errors="coerce") or 0.0))
                ok2, d2 = self._liquidity_adv_check(t2, tv)
                if not ok2:
                    fail_info.append((i, t2, d2))
            if not fail_info:
                break
            for _i, t2, d2 in fail_info:
                liquidity_skips.append(
                    {"ticker": t2, "reason": "liquidity_adv", "detail": d2}
                )
                if verbose:
                    print(f"    SKIP {t2} — liquidity_adv (sizing): {d2}")
            out = out.drop(index=[x[0] for x in fail_info]).reset_index(drop=True)

        if out.empty:
            return pd.DataFrame(), skips, liquidity_skips

        return out, skips, liquidity_skips

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
            if abs(w) <= 1e-12:
                continue
            w_abs = abs(float(w))

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
                proposed_beta = current_beta + (float(w) * float(b))
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
                    proposed = float(sector_exposure.get(sec, 0.0) + w_abs)
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
                current_beta += float(w) * float(b2)
            if self.max_sector_exposure_default is not None or self.max_sector_exposure_by_sector:
                sec2 = sector_map.get(ticker, new_sectors.get(ticker, "Unknown"))
                sector_exposure[sec2] = float(sector_exposure.get(sec2, 0.0) + w_abs)

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
        # Side-aware reconcile: if a name flips direction, close and reopen.
        tgt_dir: dict[str, int] = {}
        if target is not None and len(target) > 0:
            for _, r in target.iterrows():
                t = str(r.get("ticker") or "").strip().upper()
                if not t:
                    continue
                w = float(pd.to_numeric(r.get("target_weight"), errors="coerce") or 0.0)
                tgt_dir[t] = 1 if w > 0 else -1 if w < 0 else 0

        cur_dir: dict[str, int] = {}
        if current_positions is not None and not current_positions.empty:
            for _, r in current_positions.iterrows():
                t = str(r.get("ticker") or "").strip().upper()
                if not t:
                    continue
                d = r.get("direction")
                if d is not None and pd.notna(d):
                    cur_dir[t] = int(d)
                    continue
                nq = r.get("net_qty")
                if nq is not None and pd.notna(nq):
                    cur_dir[t] = 1 if float(nq) > 0 else -1 if float(nq) < 0 else 0
                    continue
                q = r.get("qty")
                if q is not None and pd.notna(q):
                    cur_dir[t] = 1 if float(q) > 0 else -1 if float(q) < 0 else 0

        target_tickers = {t for t, d in tgt_dir.items() if d != 0}
        current_tickers = {t for t, d in cur_dir.items() if d != 0}

        to_close: set[str] = set()
        to_open: set[str] = set()
        to_hold: set[str] = set()

        for t in current_tickers:
            if t not in target_tickers:
                to_close.add(t)
            else:
                if cur_dir.get(t, 0) != tgt_dir.get(t, 0):
                    to_close.add(t)
                    to_open.add(t)  # reopen after close
                else:
                    to_hold.add(t)

        for t in target_tickers:
            if t not in current_tickers:
                to_open.add(t)

        if self._use_greedy_factor_limits:
            filtered_to_open = set(str(t).strip().upper() for t in to_open)
            skipped: list[dict[str, Any]] = []
        else:
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

    def _order_intent_path(self) -> Path:
        p = Path("output/live/order_intents.jsonl")
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _intent_id(self, *, as_of: str, ticker: str, side: str) -> str:
        return f"{str(as_of).strip()}|{str(ticker).strip().upper()}|{str(side).strip().lower()}"

    def _load_intents_for_as_of(self, as_of: str) -> set[str]:
        p = self._order_intent_path()
        if not p.exists():
            return set()
        seen: set[str] = set()
        try:
            with open(p, encoding="utf-8") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if str(obj.get("as_of") or "").strip() != str(as_of).strip():
                        continue
                    iid = str(obj.get("intent_id") or "").strip()
                    if iid:
                        seen.add(iid)
        except Exception:
            return set()
        return seen

    def _append_intent(self, entry: dict[str, Any]) -> None:
        p = self._order_intent_path()
        with open(p, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry, default=str) + "\n")

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
        if (
            len(target) > 0
            and dd_mult != 1.0
            and not self._use_greedy_factor_limits
        ):
            target = target.copy()
            target["target_value"] = pd.to_numeric(target["target_value"], errors="coerce").fillna(0.0)
            target["target_value"] = target["target_value"] * float(dd_mult)
        risk_skips: list[dict[str, Any]] = []
        liquidity_skips: list[dict[str, Any]] = []
        if self._use_greedy_factor_limits and len(target) > 0:
            target, risk_skips, liquidity_skips = self._apply_factor_limits(
                target,
                current_positions,
                account,
                sizing_mult=float(dd_mult),
                verbose=False,
            )
        if len(target) == 0:
            return {
                "total_open_notional": 0.0,
                "planned_orders": [],
                "cash": float(account["cash"]),
                "buying_power": float(account.get("buying_power", account["cash"])),
                "drawdown_multiplier": float(dd_mult),
                "risk_skips": risk_skips,
                "liquidity_skips": liquidity_skips,
            }
        changes = self.reconcile(target, current_positions, verbose=False)
        total = 0.0
        planned: list[dict[str, Any]] = []
        for ticker in sorted(changes["to_open"]):
            row = target[target["ticker"].astype(str) == ticker].iloc[0]
            notional = float(row["target_value"])
            side = "buy" if notional > 0 else "sell"
            notional_abs = abs(notional)
            if notional_abs < 100:
                continue
            total += notional_abs
            planned.append({"ticker": ticker, "side": side, "notional": notional_abs})
        return {
            "total_open_notional": float(total),
            "planned_orders": planned,
            "cash": float(account["cash"]),
            "buying_power": float(account.get("buying_power", account["cash"])),
            "drawdown_multiplier": float(dd_mult),
            "to_open": sorted(changes["to_open"]),
            "risk_skips": risk_skips,
            "liquidity_skips": liquidity_skips,
        }

    def execute(
        self,
        signals_df: pd.DataFrame,
        dry_run: bool = True,
        *,
        extra_execution_log: dict[str, Any] | None = None,
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
        if (
            len(target) > 0
            and dd_mult != 1.0
            and not self._use_greedy_factor_limits
        ):
            target = target.copy()
            target["target_value"] = pd.to_numeric(target["target_value"], errors="coerce").fillna(0.0)
            target["target_value"] = target["target_value"] * float(dd_mult)

        risk_skips_exec: list[dict[str, Any]] = []
        liquidity_skips_exec: list[dict[str, Any]] = []
        if self._use_greedy_factor_limits and len(target) > 0:
            if dry_run:
                print()
                print("  [Risk] Greedy factor limits (beta / sector / single-name / liquidity)")
            target, risk_skips_exec, liquidity_skips_exec = self._apply_factor_limits(
                target,
                current_positions,
                account,
                sizing_mult=float(dd_mult),
                verbose=dry_run,
            )
            print(f"  Liquidity skips: {len(liquidity_skips_exec)}")
            if risk_skips_exec and not dry_run:
                print("  [Risk] Skips from greedy factor limits:")
                for item in risk_skips_exec:
                    print(
                        f"    SKIP {item.get('ticker')} — {item.get('reason')}: {item.get('detail')}"
                    )
            if liquidity_skips_exec and not dry_run:
                print("  [Risk] Skips from liquidity (ADV):")
                for item in liquidity_skips_exec:
                    print(
                        f"    SKIP {item.get('ticker')} — {item.get('reason')}: {item.get('detail')}"
                    )

        if len(target) == 0:
            print("  No trades to execute")
            place_live_empty = not dry_run and is_live_trading_allowed(
                self.config, halt_env_var=self._trading_halt_env
            )
            halt_reason_empty = (
                trading_halt_reason(self.config, halt_env_var=self._trading_halt_env)
                if not place_live_empty and not dry_run
                else None
            )
            skip_log: dict[str, Any] = {
                "timestamp": datetime.now().isoformat(),
                "dry_run": dry_run,
                "live_trading_allowed": place_live_empty,
                "trading_halt_reason": halt_reason_empty,
                "run_id": (os.environ.get("RUN_ID") or "").strip() or None,
                "equity": account["equity"],
                "peak_equity": float(peak_equity),
                "drawdown": float(drawdown),
                "drawdown_multiplier": float(dd_mult),
                "skipped": True,
                "reason": "no_target_positions_after_factor_limits"
                if (risk_skips_exec or liquidity_skips_exec)
                else "no_target_positions",
                "risk_skips": risk_skips_exec,
                "liquidity_skips": liquidity_skips_exec,
                "n_open": 0,
                "n_close": 0,
                "n_hold": 0,
                "orders_placed": [],
                "orders_skipped": [],
                "slippage_trades_logged": 0,
                "slippage_order_ids": [],
            }
            if extra_execution_log:
                skip_log = {**skip_log, **extra_execution_log}
            log_file = self._append_execution_log(skip_log)
            print(f"  Log saved: {log_file}")
            print(f"{'=' * 50}")
            return skip_log

        changes = self.reconcile(target, current_positions, verbose=True)

        as_of = str((extra_execution_log or {}).get("as_of") or "").strip() or datetime.now().strftime("%Y-%m-%d")
        existing_intents = self._load_intents_for_as_of(as_of)

        place_live = not dry_run and is_live_trading_allowed(self.config, halt_env_var=self._trading_halt_env)
        halt_reason = (
            trading_halt_reason(self.config, halt_env_var=self._trading_halt_env)
            if not place_live and not dry_run
            else None
        )

        orders_placed: list[Any] = []
        orders_skipped: list[str] = []
        close_errors: list[dict[str, Any]] = []
        slippage_trade_ids: list[str] = []

        if dry_run:
            print()
            print("  [DRY RUN] Would execute:")
        elif not place_live:
            print()
            print(f"  [Trading] HALTED — {halt_reason}; planned orders only (no broker calls).")

        for ticker in sorted(changes["to_close"]):
            sym = str(ticker).strip().upper()
            if dry_run or not place_live:
                print(f"    CLOSE {ticker}")
            else:
                intent_id = self._intent_id(as_of=as_of, ticker=sym, side="sell")
                if intent_id in existing_intents:
                    print(f"    SKIP CLOSE {sym}: already submitted (intent={intent_id})")
                    orders_skipped.append(sym)
                    continue
                try:
                    sig_px, sig_src = self._signal_reference_price(sym)
                    wait_fill = bool(self.broker.is_market_open())
                    result = self.broker.close_position(
                        sym,
                        wait_for_fill=wait_fill,
                        signal_price=sig_px,
                    )
                except Exception as e:  # noqa: BLE001 — log unexpected broker errors
                    print(f"    [Close] FAILED {sym}: {e}")
                    result = {
                        "success": False,
                        "ticker": sym,
                        "error": str(e),
                    }
                orders_placed.append(result)
                existing_intents.add(intent_id)
                self._append_intent(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "run_id": (os.environ.get("RUN_ID") or "").strip() or None,
                        "as_of": as_of,
                        "intent_id": intent_id,
                        "ticker": sym,
                        "side": "sell",
                        "order_id": result.get("order_id"),
                        "status": result.get("status"),
                        "success": bool(result.get("success")),
                    }
                )
                if not result.get("success"):
                    close_errors.append(
                        {
                            "ticker": sym,
                            "error": result.get("error", "close_failed"),
                        }
                    )
                else:
                    est_notional: float | None = None
                    fq, fap = result.get("filled_qty"), result.get("filled_avg_price")
                    if fq is not None and fap is not None:
                        try:
                            est_notional = float(fq) * float(fap)
                        except (TypeError, ValueError):
                            est_notional = None
                    self._record_slippage_trade(
                        dry_run=False,
                        ticker=sym,
                        side="sell",
                        notional=est_notional,
                        signal_price=sig_px,
                        signal_price_source=sig_src,
                        result=result,
                    )
                    oid = result.get("order_id")
                    if oid:
                        slippage_trade_ids.append(str(oid))

        skip_opens_for_bp = False
        if place_live and len(changes["to_close"]) > 0:
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
            side = "buy" if notional > 0 else "sell"
            notional_abs = abs(notional)

            if notional_abs < 100:
                print(f"    SKIP {ticker}: too small ${notional_abs:.0f}")
                orders_skipped.append(ticker)
                continue

            if skip_opens_for_bp:
                print(f"    SKIP {ticker}: no buying power (deferred)")
                orders_skipped.append(ticker)
                continue

            if dry_run or not place_live:
                verb = "BUY " if side == "buy" else "SELL"
                print(f"    {verb} {ticker} ${notional_abs:,.2f}")
            else:
                intent_id = self._intent_id(as_of=as_of, ticker=ticker, side=side)
                if intent_id in existing_intents:
                    print(f"    SKIP {side.upper()} {ticker}: already submitted (intent={intent_id})")
                    orders_skipped.append(ticker)
                    continue
                sig_px, sig_src = self._signal_reference_price(ticker)
                wait_fill = bool(self.broker.is_market_open())
                result = self.broker.place_order(
                    ticker=ticker,
                    side=side,
                    notional=notional_abs,
                    wait_for_fill=wait_fill,
                    signal_price=sig_px,
                )
                orders_placed.append(result)
                existing_intents.add(intent_id)
                self._append_intent(
                    {
                        "timestamp": datetime.now().isoformat(),
                        "run_id": (os.environ.get("RUN_ID") or "").strip() or None,
                        "as_of": as_of,
                        "intent_id": intent_id,
                        "ticker": str(ticker).strip().upper(),
                        "side": side,
                        "notional": float(notional_abs),
                        "order_id": result.get("order_id"),
                        "status": result.get("status"),
                        "success": bool(result.get("success")),
                    }
                )
                if result.get("success"):
                    self._record_slippage_trade(
                        dry_run=False,
                        ticker=ticker,
                        side=side,
                        notional=float(notional_abs),
                        signal_price=sig_px,
                        signal_price_source=sig_src,
                        result=result,
                    )
                    oid = result.get("order_id")
                    if oid:
                        slippage_trade_ids.append(str(oid))

        log: dict[str, Any] = {
            "timestamp": datetime.now().isoformat(),
            "dry_run": dry_run,
            "live_trading_allowed": place_live,
            "trading_halt_reason": halt_reason,
            "run_id": (os.environ.get("RUN_ID") or "").strip() or None,
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
            "risk_skips": risk_skips_exec,
            "liquidity_skips": liquidity_skips_exec,
            "slippage_trades_logged": len(slippage_trade_ids),
            "slippage_order_ids": slippage_trade_ids,
        }
        if extra_execution_log:
            log = {**log, **extra_execution_log}

        log_file = self._append_execution_log(log)

        print()
        print(f"  Log saved: {log_file}")
        print(f"{'=' * 50}")

        return log
