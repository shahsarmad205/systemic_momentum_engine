"""
Portfolio & Position Management
=================================
Tracks cash, open positions, completed trade log, and daily equity snapshots.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


# ------------------------------------------------------------------
# Position data container
# ------------------------------------------------------------------

@dataclass
class Position:
    ticker: str
    signal: str            # "Bullish" / "Bearish"
    direction: int         # +1 long, −1 short
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    planned_exit_date: pd.Timestamp
    entry_price: float
    position_size: float   # dollar amount allocated
    shares: float
    adjusted_score: float
    confidence: str
    regime: str
    entry_cost: float
    impact_entry_cost: float = 0.0
    exit_reason: str = ""
    actual_exit_date: pd.Timestamp | None = None
    actual_holding_days: int = 0
    current_price: float = 0.0

    @property
    def unrealized_return(self) -> float:
        if self.entry_price <= 0:
            return 0.0
        return self.direction * (self.current_price - self.entry_price) / self.entry_price

    @property
    def market_value(self) -> float:
        return self.position_size * (1 + self.unrealized_return)


# ------------------------------------------------------------------
# Portfolio
# ------------------------------------------------------------------

class Portfolio:
    def __init__(self, initial_capital: float, max_positions: int):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.max_positions = max_positions
        self.positions: list[Position] = []
        self.trade_log: list[dict] = []
        self.equity_history: list[dict] = []

    # -- derived properties ----------------------------------------

    @property
    def equity(self) -> float:
        return self.cash + sum(p.market_value for p in self.positions)

    @property
    def available_slots(self) -> int:
        return self.max_positions - len(self.positions)

    # -- position sizing -------------------------------------------

    def compute_position_size(
        self,
        position_scale: float = 1.0,
        *,
        sizing_mode: str = "equal",
        volatility_annual: float | None = None,
        vol_target_annual: float = 0.15,
        kelly_fraction: float = 0.5,
        kelly_win_rate: float = 0.55,
        kelly_avg_win_return: float = 0.02,
        kelly_avg_loss_return: float = 0.015,
        max_position_pct_of_equity: float = 0.25,
    ) -> float:
        """
        Compute dollar position size. Equal weight by default; optional vol-scaled or Kelly.
        Always capped by max_position_pct_of_equity.
        """
        base = self.equity / self.max_positions
        size = max(0.0, base * position_scale)

        if sizing_mode == "vol_scaled" and volatility_annual is not None and volatility_annual > 1e-8:
            # Scale so risk per position is roughly constant: size ∝ vol_target / vol
            vol_ratio = vol_target_annual / max(volatility_annual, 0.01)
            size = size * min(vol_ratio, 3.0)  # cap leverage at 3x

        elif sizing_mode == "kelly":
            # Full Kelly fraction: f = (p*b - q) / b, b = avg_win/avg_loss
            p = kelly_win_rate
            q = 1.0 - p
            if kelly_avg_loss_return <= 0:
                b = 2.0
            else:
                b = kelly_avg_win_return / kelly_avg_loss_return
            full_kelly = (p * b - q) / b if b > 0 else 0.0
            full_kelly = max(0.0, min(full_kelly, 1.0))
            kelly_mult = 1.0 + kelly_fraction * full_kelly
            size = size * min(kelly_mult, 2.0)  # cap at 2x base

        cap = self.equity * max_position_pct_of_equity
        if cap > 0:
            size = min(size, cap)
        return max(0.0, size)

    # -- open a new position ----------------------------------------

    def open_position(
        self,
        ticker: str,
        signal: str,
        signal_date: pd.Timestamp,
        entry_date: pd.Timestamp,
        planned_exit_date: pd.Timestamp,
        entry_price: float,
        adjusted_score: float,
        confidence: str,
        regime: str,
        entry_cost: float,
        impact_entry_cost: float = 0.0,
        position_scale: float = 1.0,
        size_dollars: float | None = None,
    ) -> Position | None:
        if self.available_slots <= 0:
            return None

        if signal != "Bullish" and signal != "Bearish":
            return None  # do not trade Neutral signals

        if size_dollars is not None and size_dollars > 0:
            size = min(size_dollars, self.equity * 0.99)  # never allocate more than equity
        else:
            size = self.compute_position_size(position_scale)

        if size <= 0 or entry_price <= 0:
            return None

        if signal == "Bullish":
            direction = 1
        elif signal == "Bearish":
            direction = -1
        else:
            return None  # do not trade Neutral signals
        shares = size / entry_price

        pos = Position(
            ticker=ticker,
            signal=signal,
            direction=direction,
            signal_date=signal_date,
            entry_date=entry_date,
            planned_exit_date=planned_exit_date,
            entry_price=entry_price,
            position_size=size,
            shares=shares,
            adjusted_score=adjusted_score,
            confidence=confidence,
            regime=regime,
            entry_cost=entry_cost,
            impact_entry_cost=impact_entry_cost,
            current_price=entry_price,
        )

        self.cash -= size + entry_cost
        self.positions.append(pos)
        return pos

    # -- close an existing position ---------------------------------

    def close_position(
        self,
        pos: Position,
        exit_date: pd.Timestamp,
        exit_price: float,
        exit_cost: float,
        **extra_record: object,
    ) -> dict:
        trade_return = pos.direction * (exit_price - pos.entry_price) / pos.entry_price
        total_cost = pos.entry_cost + exit_cost
        pnl = pos.position_size * trade_return - total_cost
        net_return = (pnl / pos.position_size) if pos.position_size > 0 else 0.0

        self.cash += pos.position_size * (1 + trade_return) - exit_cost

        planned_holding_days = (pos.planned_exit_date - pos.entry_date).days
        actual_holding_days = (exit_date - pos.entry_date).days
        pos.actual_exit_date = exit_date
        pos.actual_holding_days = actual_holding_days

        record = {
            "ticker": pos.ticker,
            "signal": pos.signal,
            "direction": pos.direction,
            "signal_date": pos.signal_date,
            "entry_date": pos.entry_date,
            "exit_date": exit_date,
            "planned_exit_date": pos.planned_exit_date,
            "actual_exit_date": exit_date,
            "entry_price": round(pos.entry_price, 4),
            "exit_price": round(exit_price, 4),
            "position_size": round(pos.position_size, 2),
            "shares": round(pos.shares, 4),
            "return": round(trade_return, 6),
            "pnl": round(pnl, 2),
            "adjusted_score": pos.adjusted_score,
            "confidence": pos.confidence,
            "regime": pos.regime,
            "entry_cost": round(pos.entry_cost, 2),
            "exit_cost": round(exit_cost, 2),
            "total_cost": round(total_cost, 2),
            "impact_entry_cost": round(pos.impact_entry_cost, 2),
            "gross_return": round(trade_return, 6),
            "net_return": round(net_return, 6),
            "holding_days": actual_holding_days,
            "planned_holding_days": planned_holding_days,
            "actual_holding_days": actual_holding_days,
            "exit_reason": pos.exit_reason or "",
        }
        record.update(extra_record)
        self.trade_log.append(record)
        self.positions.remove(pos)
        return record

    # -- daily snapshot ---------------------------------------------

    def record_equity(self, date: pd.Timestamp, regime: str = "") -> None:
        self.equity_history.append({
            "date": date,
            "equity": round(self.equity, 2),
            "cash": round(self.cash, 2),
            "invested": round(sum(p.market_value for p in self.positions), 2),
            "n_positions": len(self.positions),
            "regime": regime,
        })

    # -- sector helpers ---------------------------------------------

    def get_sector_count(self, sector_map: dict[str, str]) -> dict[str, int]:
        counts: dict[str, int] = {}
        for pos in self.positions:
            sector = sector_map.get(pos.ticker, "Other")
            counts[sector] = counts.get(sector, 0) + 1
        return counts
