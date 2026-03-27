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
from utils.ensemble_scoring import compute_ensemble_score, load_ensemble_models


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
    config = yaml.safe_load(open("backtest_config.yaml", encoding="utf-8"))
    tickers = config.get("tickers", [])
    bt = config.get("backtest", {}) or {}
    ex = config.get("execution", {}) or {}
    sig_cfg = config.get("signals", {}) or {}
    signal_mode = str(sig_cfg.get("mode", "learned")).strip().lower()
    enable_shorts = bool(ex.get("enable_shorts", False))
    long_only = bool(ex.get("long_only", True))
    max_positions = int(bt.get("max_positions", 10) or 10)
    max_longs = int(bt.get("max_longs", (max_positions + 1) // 2) or 0)
    max_shorts = int(bt.get("max_shorts", max_positions // 2) or 0)
    if not enable_shorts or long_only:
        max_shorts = 0
        max_longs = max_positions

    weights_path = "output/learned_weights.json"

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
    scores: dict[str, float] = {}
    if signal_mode == "ensemble":
        ens_cfg = sig_cfg.get("ensemble", {}) or {}
        models = load_ensemble_models(ens_cfg)
        if not models:
            print("  [ensemble] WARN: no ensemble models loaded; falling back to learned weights.")
            signal_mode = "learned"
        else:
            feat_df = today_features.copy()
            if "ticker" in feat_df.columns:
                feat_df = feat_df.set_index("ticker")
            feat_df = feat_df.drop(columns=["date"], errors="ignore")
            ens_scores = compute_ensemble_score(
                feat_df,
                models,
                normalize=bool(ens_cfg.get("normalize", True)),
                clip=bool(ens_cfg.get("clip", False)),
            )
            if ens_scores.empty or ens_scores.isna().all():
                print("  [ensemble] WARN: empty predictions; falling back to learned weights.")
                signal_mode = "learned"
            else:
                scores = {str(t): float(s) for t, s in ens_scores.items()}
                print(f"  [ensemble] Loaded {len(models)} models")
    if signal_mode == "ml":
        ml_path = str(sig_cfg.get("ml_model_path", "output/models/best_model.pkl")).strip()
        ml_type = str(sig_cfg.get("ml_model_type", "classifier")).strip().lower()
        models = load_ensemble_models({"models": [{"path": ml_path, "weight": 1.0, "type": ml_type}]})
        if not models:
            print("  [ml] WARN: model not loaded; falling back to learned weights.")
            signal_mode = "learned"
        else:
            feat_df = today_features.copy()
            if "ticker" in feat_df.columns:
                feat_df = feat_df.set_index("ticker")
            feat_df = feat_df.drop(columns=["date"], errors="ignore")
            ml_scores = compute_ensemble_score(feat_df, models, normalize=False, clip=bool(sig_cfg.get("ml_clip", False)))
            if ml_scores.empty or ml_scores.isna().all():
                print("  [ml] WARN: empty predictions; falling back to learned weights.")
                signal_mode = "learned"
            else:
                scores = {str(t): float(s) for t, s in ml_scores.items()}
                print(f"  [ml] Loaded model: {ml_path}")
    if signal_mode not in {"ensemble", "ml", "learned"}:
        print(f"  [{signal_mode}] WARN: unsupported mode in run_daily_signals; falling back to learned.")
        signal_mode = "learned"
    if signal_mode == "learned":
        weights = json.load(open(weights_path, encoding="utf-8"))
        model_name = weights.get("model_type")
        ic_val = weights.get("ic")
        print(f"  Model: {model_name}")
        if isinstance(ic_val, int | float):
            print(f"  Train IC: {float(ic_val):.4f}")
        else:
            print("  Train IC: N/A")
        print()
        lw = LearnedWeights.load(weights_path)
        for _, row in today_features.iterrows():
            ticker = row["ticker"]
            score = compute_score(row, lw)
            scores[str(ticker)] = float(score)
    if not scores:
        raise RuntimeError("No scores generated. Check signals.mode and model artifact paths.")

    # Rank tickers by score
    ranked = pd.DataFrame([{"ticker": t, "score": s} for t, s in scores.items()])
    ranked = ranked.sort_values("score", ascending=False).reset_index(drop=True)

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
    ranked = ranked[ranked["score"].abs() > threshold].copy()
    ranked["signal"] = np.where(ranked["score"] > 0, 1, -1)

    longs = ranked[ranked["score"] > 0].copy().sort_values("score", ascending=False).head(max_longs)
    shorts = (
        ranked[ranked["score"] < 0].copy().sort_values("score", ascending=True).head(max_shorts)
        if max_shorts > 0
        else ranked.iloc[0:0].copy()
    )
    selected = pd.concat([longs, shorts], ignore_index=True)
    selected = (
        selected.sort_values("score", key=lambda s: s.abs(), ascending=False)
        .head(max_positions)
        .reset_index(drop=True)
    )

    # Output
    print(f"{'='*55}")
    print(f"  TOP SIGNALS (LONG/SHORT) — {as_of_date}")
    print(f"{'='*55}")
    print(f"  {'Rank':<6} {'Ticker':<8} {'Score':>10} {'Side'}")
    print(f"  {'-'*45}")
    for i, (_, row) in enumerate(selected.iterrows(), 1):
        side = "LONG" if float(row["score"]) > 0 else "SHORT"
        print(f"  {i:<6} {row['ticker']:<8} {row['score']:>10.4f} {side}")

    print()
    print(f"  Regime:    {regime}")
    print(f"  Threshold: {threshold:.4f} ({multiplier}x std)")
    print(f"  Signals:   {len(selected)} tickers (longs={len(longs)}, shorts={len(shorts)})")
    print()

    # Save output
    output_dir = Path("output/signals")
    output_dir.mkdir(exist_ok=True)

    output_file = output_dir / f"{as_of_date}_rankings.csv"
    ranked.to_csv(output_file, index=False)

    # Append to signal history
    history_file = output_dir / "signal_history.csv"
    selected["date"] = as_of_date
    selected["regime"] = regime

    if history_file.exists():
        history = pd.read_csv(history_file)
        history = pd.concat([history, selected], ignore_index=True)
    else:
        history = selected
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
    paper_rows = selected[["ticker", "score", "signal"]].copy()
    paper_rows = paper_rows.rename(columns={"score": "signal_score"})
    paper_rows["date"] = as_of_date
    paper_rows["entry_date"] = as_of_date
    paper_rows["direction"] = np.where(paper_rows["signal"] > 0, "LONG", "SHORT")
    paper_rows = paper_rows.drop(columns=["signal"], errors="ignore")
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

    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--date", type=str, default=None, help="As-of date YYYY-MM-DD (default: today)")
    args = parser.parse_args()
    run_daily_signals(args.date)
