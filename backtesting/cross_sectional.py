"""
Cross-sectional signal ranking and top/bottom portfolio selection.
Ranks all tickers by adjusted_score (high = long, low = short); selects
TOP_LONGS highest and TOP_SHORTS lowest; equal capital per selected name.
"""

from __future__ import annotations

import numpy as np
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
    currently_held_tickers: set[str] | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    Build long and short candidate lists from cross-sectional rank.
    Includes a 'Conviction Buffer' (hysteresis) to reduce turnover.
    """
    currently_held_tickers = currently_held_tickers or set()
    top_longs = getattr(config, "top_longs", 5)
    top_shorts = getattr(config, "top_shorts", 5)
    market_neutral = getattr(config, "market_neutral", True)
    
    min_strength = float(getattr(config, "min_signal_strength", 0.05))
    buffer = float(getattr(config, "conviction_buffer", 0.02) or 0.0)
    
    # If std-based threshold scanning is enabled, override `min_strength`
    # with the equivalent abs(adjusted_score) threshold.
    std_mult = getattr(config, "signal_threshold_std_multiplier", None)
    signal_score_std = getattr(config, "signal_score_std", None)
    if (
        std_mult is not None
        and signal_score_std is not None
        and float(signal_score_std) > 0
    ):
        min_strength = float(std_mult) * float(signal_score_std)

    regime_adj = config.regime_adjustments.get(regime, {})
    score_mult = regime_adj.get("score_mult", 1.0)
    position_scale = regime_adj.get("position_scale", 1.0)

    # First pass: collect all finite scores for cross-sectional normalization
    raw_candidates: list[tuple[str, float, any]] = []
    for ticker, sig_row in daily_signals_at_date:
        signal_raw = float(sig_row["adjusted_score"])
        if np.isfinite(signal_raw):
            raw_candidates.append((ticker, signal_raw, sig_row))

    # Cross-sectional z-score: when model outputs near-constant predictions
    # (std < 0.01), z-scoring restores meaningful rank dispersion across tickers.
    if len(raw_candidates) >= 5:
        all_scores = np.array([s for _, s, _ in raw_candidates], dtype=float)
        score_std = float(all_scores.std())
        score_mean = float(all_scores.mean())
        if score_std > 1e-8:
            raw_candidates = [
                (t, (s - score_mean) / score_std, sr)
                for t, s, sr in raw_candidates
            ]

    # Second pass: apply conviction-buffer threshold and regime multiplier
    rows: list[tuple[str, float, any]] = []
    for ticker, signal_norm, sig_row in raw_candidates:
        adj_score = signal_norm * score_mult

        # Hysteresis (Conviction Buffer):
        # Higher threshold for new entries, lower threshold for retention.
        is_held = ticker in currently_held_tickers
        effective_threshold = (min_strength - buffer) if is_held else (min_strength + buffer)

        if abs(signal_norm) < effective_threshold:
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

    # Shorts: if short_score_raw is available in signal rows, rank by it (dedicated short model).
    # Fallback: bottom of long z-score rank (weakest momentum = short candidates).
    if market_neutral or config.enable_shorts:
        long_tickers = {t for t, _, _ in long_slice}
        use_short_score = any(
            float(sig_row.get("short_score_raw", 0.0) or 0.0) != 0.0
            for _, _, sig_row in rows
        )
        if use_short_score:
            # Re-rank by short_score_raw ascending: lowest score = strongest short.
            # short_classifier output: -(2*p_down - 1) → p_down=0.9 → score=-0.8 (most negative = best short).
            # short regressor output: predicted forward_return → negative = expected decline = best short.
            # Both conventions are consistent: pick the LOWEST short_score_raw values.
            short_ranked = sorted(
                raw_candidates,
                key=lambda x: float(x[2].get("short_score_raw", 0.0) or 0.0),
                reverse=False,  # ascending: most negative score = strongest short candidate
            )
            short_slice = short_ranked[:top_shorts] if len(short_ranked) >= top_shorts else short_ranked
        else:
            short_slice = rows[-top_shorts:] if len(rows) >= top_shorts else []
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
