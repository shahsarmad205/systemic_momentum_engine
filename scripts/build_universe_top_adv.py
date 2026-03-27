#!/usr/bin/env python3
"""
Build a liquid universe list (top N by $ADV) from S&P 500 constituents.

This is designed for "staged universe expansion":
- fetch S&P 500 symbols (Wikipedia)
- download recent OHLCV to compute $ADV
- select top N tickers

Note: This does NOT require tickers to have history back to the backtest start date.
Newer IPOs are eligible; they will simply have shorter history in backtests.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent


def _sp500_symbols() -> list[str]:
    # Use Wikipedia API to avoid optional HTML parser deps (lxml/bs4).
    api = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "parse",
        "page": "List_of_S&P_500_companies",
        "prop": "wikitext",
        "formatversion": "2",
        "format": "json",
    }
    r = requests.get(
        api,
        params=params,
        timeout=20,
        headers={"User-Agent": "trend-signal-engine (student research)"},
    )
    r.raise_for_status()
    data = r.json()
    wikitext = (data.get("parse") or {}).get("wikitext")
    if not wikitext or not isinstance(wikitext, str):
        raise RuntimeError("Wikipedia API returned no wikitext")

    # Parse the constituents wikitable. Symbols are in rows like: |{{NyseSymbol|MMM}}
    sym_re = re.compile(r"^\|\{\{(?:NyseSymbol|NasdaqSymbol)\|([^}]+)\}\}\s*$")
    syms: list[str] = []
    for line in wikitext.splitlines():
        m = sym_re.match(line.strip())
        if not m:
            continue
        sym = m.group(1).strip().upper().replace(".", "-")
        sym = re.sub(r"\\s+", "", sym)
        if re.fullmatch(r"[A-Z]{1,5}(-[A-Z])?", sym):
            syms.append(sym)

    # yfinance uses '-' instead of '.' for class shares
    # e.g. BRK.B -> BRK-B, BF.B -> BF-B
    out = []
    seen = set()
    for s in syms:
        s = s.replace(".", "-")
        s = re.sub(r"\\s+", "", s)
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    return out


def _dollar_adv(df: pd.DataFrame, lookback: int = 20) -> float | None:
    if df is None or df.empty:
        return None
    if "Close" not in df.columns or "Volume" not in df.columns:
        return None
    c = pd.to_numeric(df["Close"], errors="coerce")
    v = pd.to_numeric(df["Volume"], errors="coerce")
    dv = (c * v).dropna()
    if dv.empty:
        return None
    return float(dv.tail(int(lookback)).mean())


def main() -> int:
    parser = argparse.ArgumentParser(description="Build top-N liquid universe from S&P 500.")
    parser.add_argument("--n", type=int, default=300, help="Number of tickers to select")
    parser.add_argument("--provider", type=str, default="yahoo", help="Data provider (default: yahoo)")
    parser.add_argument("--end", type=str, default=None, help="End date YYYY-MM-DD (default: today)")
    parser.add_argument("--lookback-days", type=int, default=450, help="Recent days to download for ADV ranking")
    parser.add_argument("--adv-lookback", type=int, default=20, help="$ADV window in trading days")
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "output" / "universe_top_adv.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    sys.path.insert(0, str(ROOT))
    from utils.market_data import get_ohlcv  # noqa: PLC0415

    tickers = _sp500_symbols()
    if not tickers:
        print("No symbols fetched.", file=sys.stderr)
        return 1

    end = pd.Timestamp(args.end) if args.end else pd.Timestamp.now().normalize()
    start = (end - pd.Timedelta(days=int(args.lookback_days))).normalize()
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    rows: list[dict[str, object]] = []
    for t in tickers:
        try:
            df = get_ohlcv(
                t,
                start_s,
                end_s,
                provider=str(args.provider),
                use_cache=True,
                cache_dir=str(ROOT / "data" / "cache" / "ohlcv"),
                cache_ttl_days=0,
            )
            adv = _dollar_adv(df, lookback=int(args.adv_lookback))
            if adv is None:
                continue
            fd = pd.Timestamp(df.index.min()).normalize() if not df.empty else None
            ld = pd.Timestamp(df.index.max()).normalize() if not df.empty else None
            rows.append({"ticker": t, "dollar_adv": adv, "first_date": fd, "last_date": ld})
        except Exception:
            continue

    if not rows:
        print("No ADV rows computed.", file=sys.stderr)
        return 1

    out_df = pd.DataFrame(rows).sort_values("dollar_adv", ascending=False).reset_index(drop=True)
    selected = out_df.head(int(args.n)).copy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    selected.to_csv(args.out, index=False)

    print(f"Selected {len(selected)} tickers. Wrote: {args.out}")
    print(", ".join(selected['ticker'].tolist()[:30]) + (" ..." if len(selected) > 30 else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

