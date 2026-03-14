"""
Cross-sectional signal ranking and top/bottom portfolio selection.
Ranks all tickers by adjusted_score (high = long, low = short); selects
TOP_LONGS highest and TOP_SHORTS lowest; equal capital per selected name.
"""

from __future__ import annotations

import pandas as pd


def _holding_days(config, signal: str, adj_score: float) -> int:
    if not config.dynamic_holding_enabled:
        return config.holding_period_days
    abs_s = abs(adj_score)
    bands = sorted(config.holding_period_by_strength, key=lambda b: -b[0])
    for min_score, days in bands:
        if abs_s >= min_score:
            return days
    return config.holding_period_by_signal.get(signal, config.holding_period_days)


def build_cross_sectional_candidates(
    date: pd.Timestamp,
    daily_signals_at_date: list,
    config,
    trading_days: list,
    day_index: int,
    regime: str,
) -> tuple[list[dict], list[dict]]:
    """
    Build long and short candidate lists from cross-sectional rank.

    daily_signals_at_date: list of (ticker, sig_row) — should include all
    tickers with scores for the date (Neutral included for ranking).

    Returns:
        (pending_entries, daily_allocation_log_rows)
        pending_entries: list of entry dicts (Bullish for longs, Bearish for shorts)
        daily_allocation_log_rows: list of dicts for CSV (date, ticker, position_type, ...)
    """
    top_longs = getattr(config, "top_longs", 5)
    top_shorts = getattr(config, "top_shorts", 5)
    market_neutral = getattr(config, "market_neutral", True)
    min_strength = config.min_signal_strength

    regime_adj = config.regime_adjustments.get(regime, {})
    score_mult = regime_adj.get("score_mult", 1.0)
    position_scale = regime_adj.get("position_scale", 1.0)

    # Collect (ticker, adj_score, sig_row) for every ticker with valid score
    rows: list[tuple[str, float, any]] = []
    for ticker, sig_row in daily_signals_at_date:
        adj_score = float(sig_row["adjusted_score"]) * score_mult
        # Optional: still require |score| >= min for inclusion in universe
        if abs(adj_score) < min_strength:
            continue
        rows.append((ticker, adj_score, sig_row))

    if not rows:
        return [], []

    # Rank by score descending (highest first)
    rows.sort(key=lambda x: x[1], reverse=True)

    delay = int(getattr(config, "execution_delay_days", 0) or 0)
    next_idx = day_index + 1 + delay
    if next_idx >= len(trading_days):
        return [], []

    def make_entry(ticker: str, adj_score: float, sig_row, signal_type: str) -> dict | None:
        holding_days = _holding_days(config, signal_type, adj_score)
        exit_idx = next_idx + holding_days
        if exit_idx >= len(trading_days):
            return None
        return {
            "ticker": ticker,
            "signal": signal_type,
            "adjusted_score": round(adj_score, 4),
            "confidence": str(sig_row["confidence"]),
            "regime": regime,
            "signal_date": date,
            "exit_date": trading_days[exit_idx],
            "position_scale": position_scale,
            "_cross_sectional": True,
        }

    log_rows: list[dict] = []
    entries: list[dict] = []

    # Longs: top TOP_LONGS by score (highest)
    long_slice = rows[:top_longs]
    for ticker, adj_score, sig_row in long_slice:
        e = make_entry(ticker, adj_score, sig_row, "Bullish")
        if e:
            entries.append(e)
            log_rows.append({
                "date": date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "position_type": "long",
                "position_size": "",  # filled after equity known
                "signal_score": adj_score,
            })

    # Shorts: bottom TOP_SHORTS by score (lowest)
    if market_neutral or config.enable_shorts:
        short_slice = rows[-top_shorts:] if len(rows) >= top_shorts else []
        # Avoid duplicate ticker if same name in both legs
        long_tickers = {t for t, _, _ in long_slice}
        for ticker, adj_score, sig_row in short_slice:
            if ticker in long_tickers:
                continue
            e = make_entry(ticker, adj_score, sig_row, "Bearish")
            if e:
                entries.append(e)
                log_rows.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "ticker": ticker,
                    "position_type": "short",
                    "position_size": "",
                    "signal_score": adj_score,
                })

    return entries, log_rows
