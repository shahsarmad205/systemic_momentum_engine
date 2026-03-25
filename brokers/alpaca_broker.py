"""
Alpaca Markets broker (paper or live).

Credentials (env overrides YAML so shell exports always match each other):
    ALPACA_API_KEY or ALPACA_KEY — Key ID (e.g. PK…)
    ALPACA_SECRET_KEY — Secret (long string; not the PK… id)

Usage:
    cp config/alpaca_config.example.yaml config/alpaca_config.yaml
    # Edit config/alpaca_config.yaml with your API keys

    python -m brokers.alpaca_broker
"""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

try:
    import alpaca_trade_api as tradeapi
    from alpaca_trade_api.rest import APIError
except ImportError as exc:  # pragma: no cover - runtime hint
    raise ImportError(
        "Install Alpaca client: pip install alpaca-trade-api"
    ) from exc


def _default_config_path() -> Path:
    root = Path(__file__).resolve().parent.parent
    env = os.environ.get("ALPACA_CONFIG")
    if env:
        return Path(env).expanduser()
    return root / "config" / "alpaca_config.yaml"


def _env_api_key() -> str | None:
    """ALPACA_API_KEY or ALPACA_KEY (alias)."""
    v = (os.environ.get("ALPACA_API_KEY") or os.environ.get("ALPACA_KEY") or "").strip()
    return v or None


def _resolve_cred(env_val: str | None, yaml_val: object) -> str:
    """Prefer env when set (avoids stale YAML pairing with ALPACA_SECRET_KEY); else YAML if not a placeholder."""
    e = (env_val or "").strip()
    if e:
        return e
    y = (str(yaml_val).strip() if yaml_val is not None else "")
    if y and not y.upper().startswith("YOUR"):
        return y
    return ""


def _validate_alpaca_key_pair(api_key: str, secret_key: str) -> None:
    """
    Alpaca expects two different strings: Key ID (often PK…) and a longer Secret.
    Using the Key ID as both values yields HTTP 401 / APIError unauthorized.
    """
    if api_key == secret_key:
        raise ValueError(
            "api_key and secret_key are identical. They must be different: "
            "Key ID → api_key; longer Secret → secret_key (Paper dashboard → API Keys)."
        )
    sk = secret_key.strip()
    # Key IDs are short and often start with PK (paper) or AK; secrets are longer random strings.
    if sk.startswith(("PK", "AK")) and len(sk) < 40:
        raise ValueError(
            "secret_key looks like an API Key ID (short value starting with PK/AK). "
            "Put that string in api_key / ALPACA_API_KEY only. "
            "secret_key / ALPACA_SECRET_KEY must be the separate Secret from Alpaca "
            "(Paper → API Keys → click eye / regenerate to copy)."
        )


class AlpacaBroker:
    """Thin wrapper around alpaca_trade_api.REST for paper checks and orders."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        path = Path(config_path) if config_path else _default_config_path()
        if not path.is_file():
            raise FileNotFoundError(
                f"Alpaca config not found: {path}\n"
                "Copy config/alpaca_config.example.yaml → config/alpaca_config.yaml "
                "and add your keys."
            )
        with open(path, encoding="utf-8") as fh:
            raw = yaml.safe_load(fh) or {}
        cfg = raw.get("alpaca")
        if not isinstance(cfg, dict):
            raise ValueError(f"Missing top-level 'alpaca:' block in {path}")

        api_key = _resolve_cred(_env_api_key(), cfg.get("api_key"))
        secret_key = _resolve_cred(
            os.environ.get("ALPACA_SECRET_KEY"), cfg.get("secret_key")
        )
        if not api_key:
            raise ValueError(
                "Set alpaca.api_key in config or ALPACA_API_KEY / ALPACA_KEY env var."
            )
        if not secret_key:
            raise ValueError(
                "Set alpaca.secret_key in config or ALPACA_SECRET_KEY env var. "
                "This must be the secret key from Alpaca (not the API key ID that starts with PK…)."
            )

        _validate_alpaca_key_pair(api_key, secret_key)

        base_url = str(cfg.get("base_url") or "https://paper-api.alpaca.markets").rstrip("/")
        self.api = tradeapi.REST(
            api_key,
            secret_key,
            base_url,
            api_version="v2",
        )
        self.paper = bool(cfg.get("paper_trading", True))
        self.max_order_value = float(cfg.get("max_order_value", 10_000))
        self.default_order_type = str(cfg.get("order_type", "market"))
        print(
            f"[Alpaca] REST client configured ({'PAPER' if self.paper else 'LIVE'}) — {base_url} "
            "(auth is checked on the first API call)"
        )

    def get_account(self) -> dict[str, float]:
        account = self.api.get_account()
        return {
            "equity": float(account.equity),
            "cash": float(account.cash),
            "buying_power": float(account.buying_power),
            "portfolio_value": float(account.portfolio_value),
        }

    def get_positions(self) -> pd.DataFrame:
        positions = self.api.list_positions()
        rows: list[dict[str, Any]] = []
        for p in positions:
            rows.append(
                {
                    "ticker": p.symbol,
                    "qty": float(p.qty),
                    "market_value": float(p.market_value),
                    "avg_entry_price": float(p.avg_entry_price),
                    "current_price": float(p.current_price),
                    "unrealized_pnl": float(p.unrealized_pl),
                    "unrealized_pnl_pct": float(p.unrealized_plpc),
                }
            )
        return pd.DataFrame(rows)

    def get_orders(self, status: str = "all", limit: int = 50) -> pd.DataFrame:
        orders = self.api.list_orders(status=status, limit=limit)
        rows: list[dict[str, Any]] = []
        for o in orders:
            rows.append(
                {
                    "id": o.id,
                    "ticker": o.symbol,
                    "qty": float(o.qty) if o.qty else 0.0,
                    "side": o.side,
                    "type": o.order_type,
                    "status": o.status,
                    "submitted_at": o.submitted_at,
                    "filled_at": o.filled_at,
                    "filled_avg_price": float(o.filled_avg_price)
                    if o.filled_avg_price
                    else None,
                }
            )
        return pd.DataFrame(rows)

    def place_order(
        self,
        ticker: str,
        side: str,
        notional: float,
        order_type: str | None = None,
        time_in_force: str = "day",
        *,
        apply_max_order_cap: bool = True,
    ) -> dict[str, Any]:
        """
        Submit a market order by notional. ``side`` is ``\"buy\"`` or ``\"sell\"`` (Alpaca API).

        ``apply_max_order_cap``: when True (default), notional is capped at ``max_order_value``
        from config. Set False for full-position closes so sells match full ``market_value``.
        """
        if not self.paper:
            raise ValueError("Safety check: place_order blocked when paper_trading is false")

        order_type = (order_type or self.default_order_type).lower()
        notional = float(notional)
        if apply_max_order_cap:
            notional = min(notional, self.max_order_value)

        try:
            order = self.api.submit_order(
                symbol=ticker,
                notional=round(notional, 2),
                side=side,
                type=order_type,
                time_in_force=time_in_force,
            )
            oid = order.id or ""
            short_id = f"{oid[:8]}…" if len(oid) > 8 else oid
            print(f"  [Order] {side.upper()} {ticker} ${notional:,.2f} — submitted {short_id}")
            return {
                "success": True,
                "order_id": order.id,
                "ticker": ticker,
                "side": side,
                "notional": notional,
                "status": order.status,
            }
        except Exception as e:  # noqa: BLE001 — surface broker errors
            print(f"  [Order] FAILED {ticker}: {e}")
            return {
                "success": False,
                "ticker": ticker,
                "error": str(e),
            }

    def _cancel_open_orders_for_symbol(self, symbol: str) -> int:
        """Cancel open orders for a symbol so shares are not parked as *available: 0*."""
        sym_u = str(symbol).strip().upper()
        n = 0
        try:
            for o in self.api.list_orders(status="open", limit=500):
                if str(getattr(o, "symbol", "")).strip().upper() == sym_u:
                    self.api.cancel_order(o.id)
                    n += 1
        except Exception:
            pass
        return n

    def close_position(self, ticker: str) -> dict[str, Any]:
        """
        Liquidate a long. Tries DELETE /positions (Alpaca default), then market sell by
        ``qty``, then by ``notional`` from ``market_value``. Cancels open orders for the
        symbol first — stale working orders often cause *insufficient qty available … available: 0*.
        """
        sym = str(ticker).strip().upper()
        errors: list[str] = []
        ot = self.default_order_type.lower()

        cancelled = self._cancel_open_orders_for_symbol(sym)
        if cancelled:
            print(f"  [Close] {sym} — cancelled {cancelled} open order(s) for symbol")

        try:
            self.api.close_position(sym)
            print(f"  [Close] {sym} — submitted")
            return {"success": True, "ticker": sym, "method": "delete"}
        except Exception as e:  # noqa: BLE001
            errors.append(f"delete: {e}")

        try:
            pos = self.api.get_position(sym)
        except Exception as e:  # noqa: BLE001
            err = "; ".join(errors + [f"get_position: {e}"])
            print(f"  [Close] FAILED {sym}: {err}")
            return {"success": False, "ticker": sym, "error": err}

        try:
            q = float(pos.qty)
            if q > 0:
                self.api.submit_order(
                    symbol=sym,
                    qty=pos.qty,
                    side="sell",
                    type=ot,
                    time_in_force="day",
                )
                print(f"  [Close] {sym} — submitted (market sell qty)")
                return {"success": True, "ticker": sym, "method": "sell_qty"}
        except Exception as e:  # noqa: BLE001
            errors.append(f"sell_qty: {e}")

        try:
            mv = float(pos.market_value)
            if mv > 0:
                self.api.submit_order(
                    symbol=sym,
                    notional=round(mv, 2),
                    side="sell",
                    type=ot,
                    time_in_force="day",
                )
                print(f"  [Close] {sym} — submitted (market sell notional)")
                return {"success": True, "ticker": sym, "method": "sell_notional"}
        except Exception as e:  # noqa: BLE001
            errors.append(f"sell_notional: {e}")

        err_msg = "; ".join(errors)
        print(f"  [Close] FAILED {sym}: {err_msg}")
        return {"success": False, "ticker": sym, "error": err_msg}

    def close_all_positions(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        positions = self.get_positions()
        for _, pos in positions.iterrows():
            results.append(self.close_position(str(pos["ticker"])))
        return results

    def is_market_open(self) -> bool:
        clock = self.api.get_clock()
        return bool(clock.is_open)

    def get_latest_price(self, ticker: str) -> float | None:
        try:
            trade = self.api.get_latest_trade(ticker)
            return float(trade.price)
        except Exception:
            return None


def check_setup() -> None:
    """Print account summary and market status (for paper verification)."""
    broker = AlpacaBroker()
    try:
        account = broker.get_account()
    except APIError as exc:
        msg = str(exc).lower()
        if "unauthorized" in msg or getattr(exc, "code", None) == 401:
            print(
                "\nAlpaca: unauthorized. Most often:\n"
                "  • api_key and secret must be from the **same** Paper key pair.\n"
                "  • Env wins over YAML: set ALPACA_API_KEY or ALPACA_KEY together with "
                "ALPACA_SECRET_KEY, or update config/alpaca_config.yaml so both match.\n"
                "  • secret must be the long Secret, not the PK… Key ID.\n"
                "  • base_url must be https://paper-api.alpaca.markets for Paper keys.\n"
            )
        raise
    print("=== Paper Account ===")
    for k, v in account.items():
        print(f"  {k}: {v}")
    print()
    print("Market open:", broker.is_market_open())
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("Local time:", now)


if __name__ == "__main__":
    check_setup()
