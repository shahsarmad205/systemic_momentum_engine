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
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yaml

from utils.live_trades import adverse_slippage_bps


class AlpacaAPIError(RuntimeError):
    def __init__(self, message: str, *, code: int | None = None, detail: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.detail = detail


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
    """Thin REST wrapper for Alpaca Trading API (paper or live)."""

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
        self._base_url = base_url
        self._session = requests.Session()
        self._session.headers.update(
            {
                "APCA-API-KEY-ID": api_key,
                "APCA-API-SECRET-KEY": secret_key,
                "Content-Type": "application/json",
            }
        )
        self.paper = bool(cfg.get("paper_trading", True))
        self.max_order_value = float(cfg.get("max_order_value", 10_000))
        self.default_order_type = str(cfg.get("order_type", "market"))
        print(
            f"[Alpaca] REST client configured ({'PAPER' if self.paper else 'LIVE'}) — {base_url} "
            "(auth is checked on the first API call)"
        )

    def _url(self, path: str) -> str:
        p = "/" + str(path).lstrip("/")
        # Trading endpoints live under /v2
        if not p.startswith("/v2/"):
            p = "/v2" + p
        return self._base_url + p

    def _request(self, method: str, path: str, *, params: dict[str, Any] | None = None, body: Any = None) -> Any:
        try:
            resp = self._session.request(
                method.upper(),
                self._url(path),
                params=params,
                json=body,
                timeout=20,
            )
        except Exception as exc:  # noqa: BLE001
            raise AlpacaAPIError(f"Alpaca request failed: {exc}") from exc

        if resp.status_code >= 400:
            detail = None
            try:
                detail = resp.text
            except Exception:
                detail = None
            raise AlpacaAPIError(
                f"Alpaca HTTP {resp.status_code} for {method.upper()} {path}",
                code=resp.status_code,
                detail=detail,
            )
        if not resp.content:
            return None
        try:
            return resp.json()
        except Exception:  # noqa: BLE001
            return resp.text

    def _poll_order_until_terminal(self, order_id: str, max_wait_s: float = 10.0) -> dict[str, Any] | None:
        deadline = time.monotonic() + float(max_wait_s)
        last: dict[str, Any] | None = None
        oid = str(order_id).strip()
        if not oid:
            return None
        while time.monotonic() < deadline:
            last = self.get_order(oid)
            st = str((last or {}).get("status") or "").lower()
            if st == "filled":
                return last
            if st in ("canceled", "expired", "rejected"):
                return last
            time.sleep(0.5)
        return last

    def _latest_order_id_for_symbol(self, symbol: str, side: str, *, lookback: int = 100) -> str | None:
        """Most recently updated order id for symbol+side (``status=all``)."""
        sym_u = str(symbol).strip().upper()
        try:
            orders = self.list_orders(status="all", limit=int(lookback))
        except Exception:
            return None
        candidates: list[dict[str, Any]] = []
        for o in (orders or []):
            if str(o.get("symbol", "")).upper() != sym_u:
                continue
            if str(o.get("side", "")).lower() != str(side).lower():
                continue
            candidates.append(o)
        if not candidates:
            return None

        def _ts(o: dict[str, Any]) -> str:
            return str(o.get("updated_at") or o.get("submitted_at") or "")

        candidates.sort(key=_ts, reverse=True)
        top = candidates[0]
        tid = top.get("id")
        return str(tid).strip() if tid else None

    def get_account(self) -> dict[str, float]:
        account = self._request("GET", "/account") or {}
        return {
            "equity": float(account.get("equity") or 0.0),
            "cash": float(account.get("cash") or 0.0),
            "buying_power": float(account.get("buying_power") or 0.0),
            "portfolio_value": float(account.get("portfolio_value") or 0.0),
        }

    def get_positions(self) -> pd.DataFrame:
        positions = self._request("GET", "/positions") or []
        rows: list[dict[str, Any]] = []
        for p in positions:
            side = str(p.get("side") or "").lower().strip()  # "long" | "short" (Alpaca)
            qty_raw = float(p.get("qty") or 0.0)
            # Alpaca may return qty as positive for both sides; preserve a signed view for downstream logic.
            signed_qty = -abs(qty_raw) if side == "short" else abs(qty_raw)
            rows.append(
                {
                    "ticker": p.get("symbol"),
                    "qty": qty_raw,
                    "side": side or None,
                    "net_qty": signed_qty,
                    "direction": -1 if side == "short" else 1,
                    "market_value": float(p.get("market_value") or 0.0),
                    "avg_entry_price": float(p.get("avg_entry_price") or 0.0),
                    "current_price": float(p.get("current_price") or 0.0),
                    "unrealized_pnl": float(p.get("unrealized_pl") or 0.0),
                    "unrealized_pnl_pct": float(p.get("unrealized_plpc") or 0.0),
                }
            )
        return pd.DataFrame(rows)

    def get_orders(self, status: str = "all", limit: int = 50) -> pd.DataFrame:
        orders = self.list_orders(status=status, limit=limit)
        rows: list[dict[str, Any]] = []
        for o in (orders or []):
            rows.append(
                {
                    "id": o.get("id"),
                    "ticker": o.get("symbol"),
                    "qty": float(o.get("qty") or 0.0),
                    "side": o.get("side"),
                    "type": o.get("order_type") or o.get("type"),
                    "status": o.get("status"),
                    "submitted_at": o.get("submitted_at"),
                    "filled_at": o.get("filled_at"),
                    "filled_avg_price": float(o.get("filled_avg_price") or 0.0) or None,
                }
            )
        return pd.DataFrame(rows)

    def list_orders(self, *, status: str = "all", limit: int = 100, nested: bool = False) -> list[dict[str, Any]]:
        params = {"status": status, "limit": int(limit), "nested": "true" if nested else "false"}
        out = self._request("GET", "/orders", params=params)
        return out or []

    def get_order(self, order_id: str) -> dict[str, Any]:
        oid = str(order_id).strip()
        return self._request("GET", f"/orders/{oid}") or {}

    def cancel_order(self, order_id: str) -> None:
        oid = str(order_id).strip()
        self._request("DELETE", f"/orders/{oid}")

    def submit_order(self, **payload: Any) -> dict[str, Any]:
        return self._request("POST", "/orders", body=payload) or {}

    def get_position(self, symbol: str) -> dict[str, Any]:
        sym = str(symbol).strip().upper()
        return self._request("GET", f"/positions/{sym}") or {}

    def close_position_api(self, symbol: str) -> Any:
        sym = str(symbol).strip().upper()
        return self._request("DELETE", f"/positions/{sym}")

    def place_order(
        self,
        ticker: str,
        side: str,
        notional: float,
        order_type: str | None = None,
        time_in_force: str = "day",
        *,
        apply_max_order_cap: bool = True,
        wait_for_fill: bool = True,
        signal_price: float | None = None,
        fill_poll_max_seconds: float = 10.0,
    ) -> dict[str, Any]:
        """
        Submit a market order by notional. ``side`` is ``\"buy\"`` or ``\"sell\"`` (Alpaca API).

        ``apply_max_order_cap``: when True (default), notional is capped at ``max_order_value``
        from config. Set False for full-position closes so sells match full ``market_value``.

        ``wait_for_fill``: poll Alpaca until ``filled`` / terminal state (set False when market
        is closed and fills will happen later — use ``scripts/fetch_fills.py`` to backfill).

        ``signal_price``: reference price at signal for adverse slippage in bps (optional).
        """
        # Belt-and-suspenders: execution engine also blocks on TRADING_HALTED, but the broker
        # must never place orders when halted.
        if (os.environ.get("TRADING_HALTED") or "").strip().lower() in ("1", "true", "yes", "on"):
            raise AlpacaAPIError("Trading halted via TRADING_HALTED", code=403)
        if not self.paper:
            raise ValueError("Safety check: place_order blocked when paper_trading is false")

        order_type = (order_type or self.default_order_type).lower()
        notional = float(notional)
        if apply_max_order_cap:
            notional = min(notional, self.max_order_value)

        result: dict[str, Any] = {
            "success": False,
            "ticker": ticker,
            "side": side,
            "notional": notional,
            "order_id": None,
            "status": None,
            "filled_qty": None,
            "filled_avg_price": None,
            "signal_price": signal_price,
            "slippage_bps": None,
            "filled_at": None,
            "error": None,
        }

        try:
            order = self.submit_order(
                symbol=ticker,
                notional=round(notional, 2),
                side=side,
                type=order_type,
                time_in_force=time_in_force,
            )
            oid = str(order.get("id") or "")
            short_id = f"{oid[:8]}…" if len(oid) > 8 else oid
            print(f"  [Order] {side.upper()} {ticker} ${notional:,.2f} — submitted {short_id}")
            result["success"] = True
            result["order_id"] = oid or None
            result["status"] = order.get("status")

            if wait_for_fill and oid:
                updated = self._poll_order_until_terminal(str(oid), max_wait_s=float(fill_poll_max_seconds))
                if updated is not None:
                    result["status"] = updated.get("status")
                    if updated.get("filled_qty") not in (None, ""):
                        try:
                            result["filled_qty"] = float(updated.get("filled_qty"))
                        except (TypeError, ValueError):
                            result["filled_qty"] = None
                    if updated.get("filled_avg_price") not in (None, ""):
                        try:
                            result["filled_avg_price"] = float(updated.get("filled_avg_price"))
                        except (TypeError, ValueError):
                            result["filled_avg_price"] = None
                    if result["status"] == "filled" and result["filled_avg_price"]:
                        result["slippage_bps"] = adverse_slippage_bps(
                            side, signal_price, result["filled_avg_price"]
                        )
                        try:
                            result["filled_at"] = str(updated.get("filled_at") or "")
                        except Exception:
                            result["filled_at"] = None
                elif result["status"] != "filled":
                    print(
                        f"  [Order] {ticker} — still not filled after "
                        f"{fill_poll_max_seconds:.0f}s (status={result['status']}); use fetch_fills later."
                    )

            return result
        except Exception as e:  # noqa: BLE001 — surface broker errors
            print(f"  [Order] FAILED {ticker}: {e}")
            result["error"] = str(e)
            return result

    def _cancel_open_orders_for_symbol(self, symbol: str) -> int:
        """Cancel open orders for a symbol so shares are not parked as *available: 0*."""
        sym_u = str(symbol).strip().upper()
        n = 0
        try:
            for o in self.list_orders(status="open", limit=500):
                if str(o.get("symbol", "")).strip().upper() == sym_u:
                    self.cancel_order(str(o.get("id") or ""))
                    n += 1
        except Exception:
            pass
        return n

    def close_position(
        self,
        ticker: str,
        *,
        wait_for_fill: bool = True,
        signal_price: float | None = None,
        fill_poll_max_seconds: float = 10.0,
    ) -> dict[str, Any]:
        """
        Liquidate a long. Tries DELETE /positions (Alpaca default), then market sell by
        ``qty``, then by ``notional`` from ``market_value``. Cancels open orders for the
        symbol first — stale working orders often cause *insufficient qty available … available: 0*.

        Returns ``order_id`` / fill fields when available; use ``wait_for_fill=False`` after hours.
        """
        if (os.environ.get("TRADING_HALTED") or "").strip().lower() in ("1", "true", "yes", "on"):
            raise AlpacaAPIError("Trading halted via TRADING_HALTED", code=403)
        sym = str(ticker).strip().upper()
        errors: list[str] = []
        ot = self.default_order_type.lower()

        def _enrich(d: dict[str, Any], *, side: str) -> dict[str, Any]:
            out = {
                **d,
                "signal_price": signal_price,
                "side": str(side).lower(),
                "filled_qty": d.get("filled_qty"),
                "filled_avg_price": d.get("filled_avg_price"),
                "slippage_bps": d.get("slippage_bps"),
                "status": d.get("status"),
            }
            oid = out.get("order_id")
            if (
                wait_for_fill
                and oid
                and out.get("success")
                and str(out.get("status") or "").lower() != "filled"
            ):
                updated = self._poll_order_until_terminal(str(oid), max_wait_s=float(fill_poll_max_seconds))
                if updated is not None:
                    out["status"] = updated.status
                    if getattr(updated, "filled_qty", None) not in (None, ""):
                        try:
                            out["filled_qty"] = float(updated.filled_qty)
                        except (TypeError, ValueError):
                            pass
                    if getattr(updated, "filled_avg_price", None) not in (None, ""):
                        try:
                            out["filled_avg_price"] = float(updated.filled_avg_price)
                        except (TypeError, ValueError):
                            pass
                    if out["status"] == "filled" and out.get("filled_avg_price") is not None:
                        out["slippage_bps"] = adverse_slippage_bps(
                            str(side).lower(), signal_price, out["filled_avg_price"]
                        )
                        try:
                            out["filled_at"] = str(getattr(updated, "filled_at", None) or "")
                        except Exception:
                            pass
            elif out.get("filled_avg_price") is not None and signal_price is not None:
                out["slippage_bps"] = adverse_slippage_bps(str(side).lower(), signal_price, out["filled_avg_price"])
            return out

        cancelled = self._cancel_open_orders_for_symbol(sym)
        if cancelled:
            print(f"  [Close] {sym} — cancelled {cancelled} open order(s) for symbol")

        try:
            self.close_position_api(sym)
            print(f"  [Close] {sym} — submitted")
            oid: str | None = None
            if wait_for_fill:
                time.sleep(1.5)
                oid = self._latest_order_id_for_symbol(sym, "sell") or self._latest_order_id_for_symbol(sym, "buy")
            base: dict[str, Any] = {"success": True, "ticker": sym, "method": "delete", "order_id": oid}
            if oid:
                o = self.get_order(oid)
                base["status"] = o.get("status")
                if o.get("filled_qty") not in (None, ""):
                    try:
                        base["filled_qty"] = float(o.get("filled_qty"))
                    except (TypeError, ValueError):
                        pass
                if o.get("filled_avg_price") not in (None, ""):
                    try:
                        base["filled_avg_price"] = float(o.get("filled_avg_price"))
                    except (TypeError, ValueError):
                        pass
                try:
                    base["filled_at"] = str(o.get("filled_at") or "")
                except Exception:
                    base["filled_at"] = None
            side = str((o or {}).get("side") or "sell").lower() if oid else "sell"
            return _enrich(base, side=side)
        except Exception as e:  # noqa: BLE001
            errors.append(f"delete: {e}")

        try:
            pos = self.get_position(sym)
        except Exception as e:  # noqa: BLE001
            err = "; ".join(errors + [f"get_position: {e}"])
            print(f"  [Close] FAILED {sym}: {err}")
            return {"success": False, "ticker": sym, "error": err, "order_id": None, "side": "sell"}

        try:
            q = float(pos.get("qty") or 0.0)
            side_pos = str(pos.get("side") or "").lower().strip()
            if not side_pos:
                side_pos = "short" if q < 0 else "long"
            if abs(q) > 0:
                order_side = "buy" if side_pos == "short" else "sell"
                order = self.submit_order(
                    symbol=sym,
                    qty=str(abs(q)),
                    side=order_side,
                    type=ot,
                    time_in_force="day",
                )
                print(f"  [Close] {sym} — submitted (market {order_side} qty)")
                base = {
                    "success": True,
                    "ticker": sym,
                    "method": f"{order_side}_qty",
                    "order_id": order.get("id"),
                    "status": order.get("status"),
                }
                return _enrich(base, side=order_side)
        except Exception as e:  # noqa: BLE001
            errors.append(f"order_qty: {e}")

        try:
            mv = float(pos.get("market_value") or 0.0)
            side_pos = str(pos.get("side") or "").lower().strip()
            if not side_pos:
                side_pos = "short" if q < 0 else "long"
            if abs(mv) > 0:
                order_side = "buy" if side_pos == "short" else "sell"
                order = self.submit_order(
                    symbol=sym,
                    notional=round(abs(mv), 2),
                    side=order_side,
                    type=ot,
                    time_in_force="day",
                )
                print(f"  [Close] {sym} — submitted (market {order_side} notional)")
                base = {
                    "success": True,
                    "ticker": sym,
                    "method": f"{order_side}_notional",
                    "order_id": order.get("id"),
                    "status": order.get("status"),
                }
                return _enrich(base, side=order_side)
        except Exception as e:  # noqa: BLE001
            errors.append(f"order_notional: {e}")

        err_msg = "; ".join(errors)
        print(f"  [Close] FAILED {sym}: {err_msg}")
        return {"success": False, "ticker": sym, "error": err_msg, "order_id": None}

    def close_all_positions(self) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        positions = self.get_positions()
        for _, pos in positions.iterrows():
            results.append(self.close_position(str(pos["ticker"])))
        return results

    def is_market_open(self) -> bool:
        clock = self._request("GET", "/clock") or {}
        return bool(clock.get("is_open"))

    def get_latest_price(self, ticker: str) -> float | None:
        # Optional data endpoint; not required for core execution paths.
        return None


def check_setup() -> None:
    """Print account summary and market status (for paper verification)."""
    broker = AlpacaBroker()
    try:
        account = broker.get_account()
    except AlpacaAPIError as exc:
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
