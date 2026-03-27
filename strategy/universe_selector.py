"""
Liquidity-based universe filtering for multi-ticker panels.
"""

from __future__ import annotations

import pandas as pd


class UniverseSelector:
    """Filter tickers by average daily volume and dollar volume over a lookback."""

    def __init__(
        self,
        min_adv: float,
        min_dollar_vol: float,
        lookback_days: int,
    ) -> None:
        self.min_adv = float(min_adv)
        self.min_dollar_vol = float(min_dollar_vol)
        self.lookback_days = int(lookback_days)

    def select(self, panel: pd.DataFrame) -> list[str]:
        if panel.empty or "ticker" not in panel.columns:
            return []
        df = panel.copy()
        if "Volume" not in df.columns or "Close" not in df.columns:
            return []
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["dollar_vol"] = df["Volume"] * df["Close"]

        selected: list[str] = []
        for ticker, g in df.groupby("ticker", sort=False):
            g2 = g.sort_values("date").tail(self.lookback_days)
            if g2["Volume"].isna().all():
                continue
            adv = float(g2["Volume"].mean())
            dv = float(g2["dollar_vol"].mean())
            if adv >= self.min_adv and dv >= self.min_dollar_vol:
                selected.append(str(ticker))
        return selected
