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


def _sector_neutral_zscore(
    candidates: list[tuple[str, float, any]],
    sector_map: dict[str, str],
) -> list[tuple[str, float, any]]:
    """
    Within-sector z-score demeaning before universe ranking.

    S&P Global (Innes et al.) p.5-6, p.12:
      "Sector-neutral construction z-scores within each sector first, then
      pools for selection.  Sector Neutrality reduces unintended active risk
      more than it costs in factor exposure — FER improves because tracking
      error falls faster than active factor exposure."

    For each GICS sector with ≥3 members: subtract sector mean and divide
    by sector std.  Sectors with <3 members keep raw scores (insufficient
    sample for reliable sector z-score).  Result is pooled back at
    universe level preserving relative rank within sector.
    """
    if not sector_map or not candidates:
        return candidates

    # Group by sector
    by_sector: dict[str, list[int]] = {}
    for idx, (ticker, _, _) in enumerate(candidates):
        sec = sector_map.get(ticker, "Other")
        by_sector.setdefault(sec, []).append(idx)

    scores = [s for _, s, _ in candidates]
    adjusted = list(scores)

    for sec_indices in by_sector.values():
        if len(sec_indices) < 3:
            continue  # too few members for reliable sector stat
        sec_scores = np.array([scores[i] for i in sec_indices], dtype=float)
        sec_mean = float(sec_scores.mean())
        sec_std = float(sec_scores.std())
        if sec_std < 1e-8:
            continue  # all identical scores in sector — leave unchanged
        for i in sec_indices:
            adjusted[i] = (scores[i] - sec_mean) / sec_std

    return [(t, adjusted[i], sr) for i, (t, _, sr) in enumerate(candidates)]


def build_cross_sectional_candidates(
    date: pd.Timestamp,
    daily_signals_at_date: list,
    config,
    trading_days: list,
    day_index: int,
    regime: str,
    currently_held_tickers: set[str] | None = None,
    sector_map: dict[str, str] | None = None,
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

    # Sector-neutral within-sector z-score (S&P Global, Innes et al. p.5-6, p.12).
    # Applied BEFORE universe z-score so that sector effects are removed from
    # raw scores first, then universe standardization gives rank dispersion.
    # Only active when sector_map is provided and config.neutralize_sector is set.
    neutralize_sector = getattr(config, "neutralize_sector", False) or getattr(
        config, "factor_neutralize_sector", False
    )
    if sector_map and neutralize_sector and len(raw_candidates) >= 5:
        raw_candidates = _sector_neutral_zscore(raw_candidates, sector_map)

    # Cross-sectional universe z-score: restores rank dispersion when model
    # outputs near-constant predictions (std < 0.01).
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

    # TS gate: only long stocks where time-series momentum (trend_score) is positive.
    # This implements the market-state adaptive property of TS momentum identified by
    # Bird, Gao & Yeung (2015): in bear markets, fewer stocks pass the gate, naturally
    # reducing the long book.  Pure CS always picks top_longs regardless of market
    # direction — the TS gate prevents "digging deeper" into weak longs in down markets.
    ts_gated_rows = [
        (t, s, sr) for t, s, sr in rows
        if float(sr.get("trend_score", 0.0) or 0.0) > 0.0
    ]

    # Information Discreteness sort (Da, Gurun & Warachka 2014 / FIP hypothesis).
    # Within the TS-gated long candidates, prefer CONTINUOUS momentum stocks (low ID)
    # over DISCRETE momentum stocks (high ID).
    #
    # Da et al. (2014) RFS: ID = sign(PRET) × [%neg − %pos]
    #   Low ID → gradual daily appreciation → investors underreact → alpha persists longer.
    #   High ID → a few large jump days → crowds chase → momentum mean-reverts fast.
    #
    # Implementation: sort TS-gated rows by a composite key:
    #   primary: adjusted_score descending (keep rank-order from CS z-score)
    #   secondary: information_discreteness ascending (prefer continuous within score tier)
    # The effect is a mild tiebreak that avoids "discrete" winners when score is similar.
    id_gate_enabled = getattr(config, "information_discreteness_sort", True)
    if id_gate_enabled and ts_gated_rows:
        ts_gated_rows = sorted(
            ts_gated_rows,
            key=lambda x: (
                # Primary: descending score (keep original CS rank as dominant signal)
                -round(x[1], 2),
                # Secondary: ascending ID (prefer continuous momentum within same score tier)
                # Da et al. (2014): low ID = gradual daily appreciation = longer alpha persistence
                float(x[2].get("information_discreteness", 0.0) or 0.0),
                # Tertiary: descending consistency (prefer all-horizon agreement)
                # Gupta & Kelly (2019): factor momentum everywhere = serial correlation
                # across look-back periods; multi-horizon agreement amplifies signal quality
                -float(x[2].get("momentum_consistency_score", 0.0) or 0.0),
            ),
        )

    long_slice = ts_gated_rows[:top_longs]
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

            # --- Turnover Gate (Medhat & Schmeling 2022, RFS) ---
            # CRITICAL: Low-turnover losers REVERSE (bounce UP). High-turnover losers
            # exhibit SHORT-TERM MOMENTUM (continue DOWN). Shorting low-turnover stocks
            # is betting against a ~17% per annum reversal — a systematic bleed.
            #
            # Gate rule: Only short stocks where turnover_pct_rank > 0.5 (above-median).
            # turnover_pct_rank is the rolling 63-day percentile of 20d avg share TO.
            # When the feature is unavailable, the gate is skipped (safe fallback).
            #
            # Also prefer stocks where short_term_momentum_score < 0 (the MS paper's
            # explicit interaction signal: negative return × high turnover = best short).
            turnover_gate_enabled = getattr(config, "short_turnover_gate", True)
            if turnover_gate_enabled:
                def _to_rank(sig_row) -> float:
                    return float(sig_row.get("turnover_pct_rank", 0.5) or 0.5)

                high_to = [c for c in short_ranked if _to_rank(c[2]) > 0.5]
                # Prefer short_term_momentum_score < 0 (high-TO loser with continuation signal)
                stmom_valid = [c for c in high_to if float(c[2].get("short_term_momentum_score", 0.0) or 0.0) < 0.0]
                # Fallback: all high-TO candidates if not enough with negative stmom score
                gated = stmom_valid if len(stmom_valid) >= top_shorts else (
                    stmom_valid + [c for c in high_to if c not in stmom_valid]
                )
                short_ranked = gated if len(gated) >= 1 else short_ranked  # keep original if gate yields 0

            short_slice = short_ranked[:top_shorts] if len(short_ranked) >= top_shorts else short_ranked
        else:
            # No dedicated short model output: skip shorts entirely.
            # Inverted long z-score (rows[-N:]) is NOT a valid short signal — it only
            # identifies "weakest longs", not stocks with negative expected returns.
            # Bird et al. (2015): short alpha in CS strategies is concentrated in the true
            # distributional tails, not just the bottom of a standardized long ranking.
            short_slice = []
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
