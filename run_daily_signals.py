#!/usr/bin/env python3
"""
Daily Signal Generator
Run after market close (4:15pm ET) to generate
next-day trading signals.
Usage: python run_daily_signals.py [--date YYYY-MM-DD]
"""

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


def _to_float(row, key, default=0.0):
    v = row.get(key, default)
    try:
        if pd.isna(v):
            return float(default)
    except Exception:
        pass
    try:
        return float(v)
    except Exception:
        return float(default)


def compute_score(row, lw):
    return lw.compute_adjusted_score(
        f_trend=_to_float(row, "f_trend"),
        f_regional=_to_float(row, "f_regional"),
        f_global=_to_float(row, "f_global"),
        f_social=_to_float(row, "f_social"),
        ret_5d=_to_float(row, "ret_5d"),
        ret_10d=_to_float(row, "ret_10d"),
        ret_20d=_to_float(row, "ret_20d"),
        ret_60d=_to_float(row, "ret_60d"),
        cs_momentum_percentile=_to_float(row, "cs_momentum_percentile"),
        momentum_3m=_to_float(row, "momentum_3m"),
        momentum_6m=_to_float(row, "momentum_6m"),
        ma_crossover=_to_float(row, "ma_crossover"),
        rolling_vol_5=_to_float(row, "rolling_vol_5"),
        rolling_vol_10=_to_float(row, "rolling_vol_10"),
        rolling_vol=_to_float(row, "rolling_vol_20"),
        vol_of_vol_20=_to_float(row, "vol_of_vol_20"),
        jump_indicator=_to_float(row, "jump_indicator"),
        vol_rank=_to_float(row, "vol_rank"),
        relative_volume=_to_float(row, "relative_volume"),
        volume_zscore=_to_float(row, "volume_zscore"),
        rolling_corr_market=_to_float(row, "rolling_corr_market_20"),
        capm_beta=_to_float(row, "capm_beta"),
        vix_zscore=_to_float(row, "vix_zscore"),
        vol_spike=_to_float(row, "vol_spike"),
        vix_term_zscore=_to_float(row, "vix_term_zscore"),
        rsi_zscore=_to_float(row, "rsi_zscore"),
        bb_position=_to_float(row, "bb_position"),
        dist_high=_to_float(row, "dist_high"),
        dist_low=_to_float(row, "dist_low"),
        overnight_gap=_to_float(row, "overnight_gap"),
        intraday_rev=_to_float(row, "intraday_rev"),
        sector_relative_20d=_to_float(row, "sector_relative_20d"),
        sector_relative_60d=_to_float(row, "sector_relative_60d"),
    )


def _detect_regime(as_of_date: str) -> str:
    try:
        from backtesting.regime import MarketRegimeAgent

        start = (pd.Timestamp(as_of_date) - pd.Timedelta(days=800)).strftime("%Y-%m-%d")
        regime_map = MarketRegimeAgent().detect_regimes(start, as_of_date)
        if not regime_map:
            return "Sideways"
        keys = sorted(pd.to_datetime(list(regime_map.keys())))
        use_date = pd.Timestamp(as_of_date)
        if use_date not in regime_map:
            keys_le = [d for d in keys if d <= use_date]
            if not keys_le:
                return "Sideways"
            use_date = keys_le[-1]
        return str(regime_map.get(use_date, "Sideways"))
    except Exception:
        return "Sideways"


def run_daily_signals(as_of_date=None):
    as_of_date = as_of_date or datetime.today().strftime("%Y-%m-%d")
    print(f"{'='*55}")
    print(f"  Daily Signal Generator — {as_of_date}")
    print(f"{'='*55}")

    # Load config
    config = yaml.safe_load(open("backtest_config.yaml"))
    tickers = config.get("tickers", [])

    # Load learned weights
    weights_path = "output/learned_weights.json"
    weights = json.load(open(weights_path))
    model_name = weights.get("model_type")
    ic_val = weights.get("ic")
    print(f"  Model: {model_name}")
    if isinstance(ic_val, (int, float)):
        print(f"  Train IC: {float(ic_val):.4f}")
    else:
        print("  Train IC: N/A")
    print()

    # Build feature matrix for today
    print("  Building features...")
    from agents.weight_learning_agent.feature_builder import build_feature_matrix

    # Use extra lookback for rolling features
    lookback_start = (pd.Timestamp(as_of_date) - timedelta(days=400)).strftime("%Y-%m-%d")

    df = build_feature_matrix(tickers, start_date=lookback_start, end_date=as_of_date)
    if df.empty:
        raise RuntimeError("Feature matrix is empty; cannot generate daily signals.")
    df["date"] = pd.to_datetime(df["date"])

    # Get only latest row per ticker at or before as_of_date
    asof_ts = pd.Timestamp(as_of_date)
    eligible = df[df["date"] <= asof_ts].copy()
    if eligible.empty:
        raise RuntimeError("No feature rows at or before the requested date.")
    last_date = eligible["date"].max()
    today_features = eligible[eligible["date"] == last_date].copy()

    print(f"  Feature date: {last_date.date()}")
    print(f"  Tickers with data: {len(today_features)}")
    print()

    # Generate signals using learned weights
    from agents.weight_learning_agent.weight_model import LearnedWeights

    lw = LearnedWeights.load(weights_path)

    # Compute adjusted scores
    scores = {}
    for _, row in today_features.iterrows():
        ticker = row["ticker"]
        score = compute_score(row, lw)
        scores[ticker] = score

    # Rank tickers by score
    ranked = (
        pd.DataFrame([{"ticker": t, "score": s} for t, s in scores.items()])
        .sort_values("score", ascending=False)
        .reset_index(drop=True)
    )

    # Get current market regime
    regime = _detect_regime(as_of_date)
    print(f"  Market regime: {regime}")
    print()

    # Apply regime filter
    if regime in ("Crisis",):
        ranked = ranked[ranked["score"] > ranked["score"].quantile(0.95)]
        print("  Crisis filter: top 5% signals only")

    # Apply signal threshold
    multiplier = float(config.get("backtest", {}).get("signal_confidence_multiplier", 0.8))
    rolling_std = float(ranked["score"].std()) if len(ranked) > 1 else 0.0
    threshold = multiplier * rolling_std
    ranked = ranked[ranked["score"].abs() > threshold]

    # Top long candidates
    longs = ranked[ranked["score"] > 0].head(10).copy()

    # Output
    print(f"{'='*55}")
    print(f"  TOP LONG SIGNALS — {as_of_date}")
    print(f"{'='*55}")
    print(f"  {'Rank':<6} {'Ticker':<8} {'Score':>10} {'Signal'}")
    print(f"  {'-'*45}")
    for i, (_, row) in enumerate(longs.iterrows(), 1):
        signal = "STRONG" if abs(row["score"]) > threshold * 1.5 else "MODERATE"
        print(f"  {i:<6} {row['ticker']:<8} {row['score']:>10.4f} {signal}")

    print()
    print(f"  Regime:    {regime}")
    print(f"  Threshold: {threshold:.4f} ({multiplier}x std)")
    print(f"  Signals:   {len(longs)} tickers")
    print()

    # Save output
    output_dir = Path("output/signals")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{as_of_date}_rankings.csv"
    ranked.to_csv(output_file, index=False)

    # Append to signal history
    history_file = output_dir / "signal_history.csv"
    longs["date"] = as_of_date
    longs["regime"] = regime

    if history_file.exists():
        history = pd.read_csv(history_file)
        history = pd.concat([history, longs], ignore_index=True)
    else:
        history = longs
    history.to_csv(history_file, index=False)

    # Also write entries for paper-trading tracker.
    # Required schema:
    # date,ticker,entry_price,current_price,shares,direction,entry_date,signal_score
    portfolio_dir = Path("output/portfolio")
    portfolio_dir.mkdir(exist_ok=True)
    paper_positions_file = portfolio_dir / "paper_positions.csv"
    required_cols = [
        "date",
        "ticker",
        "entry_price",
        "current_price",
        "shares",
        "direction",
        "entry_date",
        "signal_score",
    ]
    paper_rows = longs[["ticker", "score"]].copy()
    paper_rows = paper_rows.rename(columns={"score": "signal_score"})
    paper_rows["date"] = as_of_date
    paper_rows["entry_date"] = as_of_date
    paper_rows["direction"] = "LONG"
    # Prices/shares are left for manual update by paper-trading workflow.
    paper_rows["entry_price"] = np.nan
    paper_rows["current_price"] = np.nan
    paper_rows["shares"] = np.nan
    paper_rows = paper_rows[required_cols]

    if paper_positions_file.exists():
        existing = pd.read_csv(paper_positions_file)
        for col in required_cols:
            if col not in existing.columns:
                existing[col] = np.nan
        existing = existing[required_cols]
        # Upsert by (date, ticker) so re-running same day doesn't duplicate entries.
        if not existing.empty:
            existing["_key"] = existing["date"].astype(str) + "|" + existing["ticker"].astype(str)
            paper_rows["_key"] = paper_rows["date"].astype(str) + "|" + paper_rows["ticker"].astype(str)
            existing = existing[~existing["_key"].isin(paper_rows["_key"])]
            existing = existing.drop(columns=["_key"], errors="ignore")
            paper_rows = paper_rows.drop(columns=["_key"], errors="ignore")
        merged_positions = pd.concat([existing, paper_rows], ignore_index=True)
    else:
        merged_positions = paper_rows.copy()
    merged_positions.to_csv(paper_positions_file, index=False)

    print(f"  Saved: {output_file}")
    print(f"  History: {history_file}")
    print(f"  Paper positions: {paper_positions_file}")
    print(f"{'='*55}")

    return longs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="As-of date YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    run_daily_signals(args.date)
