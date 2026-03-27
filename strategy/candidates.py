"""
Ranked daily candidates — extracted from backtester loop (identical logic).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MIN_SIGNAL_FLIP_THRESHOLD = 0.15
_LAST_SIGNAL_SCORE: dict[str, float] = {}


def build_ranked_candidates(
    date: pd.Timestamp,
    daily_signals_at_date: list,
    config,
    trading_days: list,
    day_index: int,
    regime: str,
    ticker_trade_counts: dict[str, int] | None = None,
) -> list[dict]:
    """
    Build sorted candidate entries for one calendar day.
    Same filtering and ranking as Backtester._simulate step 5.
    """
    regime_adj = config.regime_adjustments.get(regime, {})
    score_mult = regime_adj.get("score_mult", 1.0)
    base_position_scale = regime_adj.get("position_scale", 1.0)

    vol_scaling_enabled = getattr(config, "vol_scaling_enabled", True)
    vol_scaling_target = float(getattr(config, "vol_scaling_target", 0.15))
    flip_threshold = float(getattr(config, "signal_flip_threshold", MIN_SIGNAL_FLIP_THRESHOLD))

    candidates: list[dict] = []
    min_strength = getattr(config, "min_signal_strength", 0.3)
    # Optional std-based signal threshold scan.
    # Backtester computes `config.signal_score_std` from the signal distribution.
    std_mult = getattr(config, "signal_threshold_std_multiplier", None)
    signal_score_std = getattr(config, "signal_score_std", None)
    use_std_threshold = (
        std_mult is not None
        and signal_score_std is not None
        and np.isfinite(float(signal_score_std))
        and float(signal_score_std) > 0
    )
    threshold_abs = float(std_mult) * float(signal_score_std) if use_std_threshold else None
    for ticker, sig_row in daily_signals_at_date:
        adj_raw = float(sig_row["adjusted_score"])
        if not np.isfinite(adj_raw):
            continue

        if use_std_threshold:
            # Only enter if abs(signal_score) > threshold_abs
            if abs(adj_raw) <= float(threshold_abs):
                continue

            # Override direction from sign(adjusted_score) so the threshold
            # filtering is independent from upstream "Neutral" labeling.
            if adj_raw > 0:
                signal_str = "Bullish"
            elif adj_raw < 0:
                signal_str = "Bearish"
            else:
                continue

            if signal_str == "Bearish" and not config.enable_shorts:
                continue

            adj_score = adj_raw * score_mult
        else:
            signal = sig_row["signal"]
            # Normalize in case it's a numpy/pd scalar
            signal_str = str(signal).strip() if signal is not None else ""
            adj_score = adj_raw * score_mult

            # Apply min_signal_strength only to Neutral or effectively zero scores; allow directional (Bullish/Bearish) with any non-zero score
            if signal_str not in ("Bullish", "Bearish") or abs(adj_score) < 1e-9:
                if abs(adj_score) < min_strength:
                    continue
            if signal_str == "Bearish" and not config.enable_shorts:
                continue

        delay = int(getattr(config, "execution_delay_days", 0) or 0)
        next_idx = day_index + 1 + delay
        if next_idx >= len(trading_days):
            continue

        # Dynamic holding — mirror Backtester._get_holding_days
        if not config.dynamic_holding_enabled:
            holding_days = config.holding_period_days
        else:
            abs_s = abs(adj_score)
            bands = sorted(config.holding_period_by_strength, key=lambda b: -b[0])
            holding_days = config.holding_period_days
            for min_score, days in bands:
                if abs_s >= min_score:
                    holding_days = days
                    break
            else:
                holding_days = config.holding_period_by_signal.get(signal_str, config.holding_period_days)

        # Crisis regime: shorten max holding window.
        # (Positions still can close earlier via stop-loss/take-profit.)
        if regime == "Crisis":
            holding_days = min(int(holding_days), 3)
        # Bear regime: cap max holding (weak edge / faster turnover).
        if regime == "Bear":
            bear_cap = int(getattr(config, "bear_max_holding_days", 3) or 3)
            holding_days = min(int(holding_days), max(1, bear_cap))

        exit_idx = next_idx + holding_days
        if exit_idx >= len(trading_days):
            continue

        position_scale = base_position_scale

        # Turnover control: avoid flipping direction on tiny score changes
        prev_score = _LAST_SIGNAL_SCORE.get(ticker)
        prev_sign = 0.0 if prev_score is None else float(pd.Series(prev_score).apply(lambda x: 1.0 if x > 0 else (-1.0 if x < 0 else 0.0)).iloc[0])
        new_sign = 1.0 if signal_str == "Bullish" else -1.0 if signal_str == "Bearish" else 0.0
        if prev_sign != 0.0 and new_sign != 0.0 and prev_sign != new_sign and prev_score is not None:
            score_delta = abs(adj_score - prev_score)
            if score_delta < flip_threshold:
                continue

        if vol_scaling_enabled and "volatility_20" in (sig_row.index if hasattr(sig_row, "index") else sig_row):
            vol = sig_row.get("volatility_20")
            if pd.notna(vol) and vol > 0:
                from utils.vol_sizing import compute_vol_scaled_weight

                position_scale = compute_vol_scaled_weight(
                    position_scale,
                    float(vol),
                    target_vol=vol_scaling_target,
                )

        _LAST_SIGNAL_SCORE[ticker] = float(sig_row.get("smoothed_score", adj_score))

        entry = {
            "ticker": ticker,
            "signal": signal_str,
            "adjusted_score": round(adj_score, 4),
            "confidence": str(sig_row["confidence"]),
            "regime": regime,
            "signal_date": date,
            "exit_date": trading_days[exit_idx],
            "position_scale": position_scale,
        }
        # CAPM: pass through for beta-adjusted position sizing
        for key in ("capm_beta", "capm_alpha", "capm_residual_vol"):
            if key in sig_row and sig_row[key] is not None and pd.notna(sig_row[key]):
                entry[key] = float(sig_row[key])
        candidates.append(entry)

    # One entry per ticker per day (keep highest |score|) so we can open multiple positions across tickers
    seen: set[str] = set()
    unique: list[dict] = []
    for c in sorted(candidates, key=lambda x: abs(x["adjusted_score"]), reverse=True):
        tk = c["ticker"]
        if tk not in seen:
            seen.add(tk)
            unique.append(c)

    # If configured, split candidate budget between long/short slots.
    # This applies to the ranked (non-cross-sectional) path; cross-sectional
    # selection uses its own top_longs/top_shorts logic.
    max_positions = int(getattr(config, "max_positions", 10) or 10)
    max_longs = int(getattr(config, "max_longs", 0) or 0)
    max_shorts = int(getattr(config, "max_shorts", 0) or 0)
    if max_longs <= 0 and max_shorts <= 0:
        max_longs = int((max_positions + 1) // 2)
        max_shorts = int(max_positions // 2)
    if bool(getattr(config, "long_only", False)) or not getattr(config, "enable_shorts", False):
        max_shorts = 0
        max_longs = max_positions

    longs_only = [c for c in unique if c.get("signal") == "Bullish"]
    shorts_only = [c for c in unique if c.get("signal") == "Bearish"]
    selected = longs_only[:max_longs] + shorts_only[:max_shorts]
    # Keep final ordering by absolute score (so the open loop remains deterministic).
    selected.sort(key=lambda x: abs(float(x.get("adjusted_score", 0.0) or 0.0)), reverse=True)
    unique = selected[:max_positions]
    # Diversification: prefer tickers we've traded less so more names get filled (e.g. tickers traded >= 12)
    counts = ticker_trade_counts or {}
    diversification_penalty = 0.002  # 500 trades ≈ 1.0 score penalty
    unique.sort(
        key=lambda c: -(abs(c["adjusted_score"]) - diversification_penalty * counts.get(c["ticker"], 0)),
        reverse=False,
    )
    return unique
