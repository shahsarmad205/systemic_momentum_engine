"""
Trend Signal Engine — Multi-Agent Batch Runner (v4)
=====================================================
Runs five agents across 50 stock tickers:

    1. Trend Agent           → trend score, probability, raw signal
    2. Volatility Agent      → rolling volatility, confidence level
    3. Regional News Agent   → live stock/sector sentiment (hour-decayed)
    4. Global News Agent     → live macro/geopolitical sentiment (hour-decayed)
    5. Social Sentiment Agent → Reddit, Stocktwits sentiment (hour-decayed)

After agent outputs are collected, price data is checked for abnormal
returns and volume spikes (news impact detection).  When confirmed,
regional and global sentiment impact weights are boosted.

Adjusted trend score formula:

    adjusted = trend_score * trend_confidence
             + regional_sentiment * regional_impact   (×1.3 if news event)
             + global_sentiment   * global_impact     (×1.2 if news event)
             + social_sentiment   * social_impact

Signal thresholds (applied directly to adjusted score):
    > 0.5  → Bullish
    < -0.5 → Bearish
    else   → Neutral

All results are kept in memory (no CSV output).
Plots are saved to output/plots/ — existing plots are skipped.

Optional API keys for richer data:
    export FINNHUB_KEY="..."
    export NEWSAPI_KEY="..."

Usage:
    python main.py                     (single batch run)
    python scheduler.py                (continuous refresh loop)
"""

import os
import logging

import pandas as pd
import numpy as np
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from data_loader import download_stock_data

# --- Agent imports ---
from agents.trend_agent import build_features, run_trend_model
from agents.volatility_agent import run_volatility_model
#
# Some deployments of this repo include only the price-based / volatility
# components needed for backtesting. The live multi-agent news/sentiment
# agents are optional; make imports resilient so that signal generation and
# backtests can run even when those modules are absent.
#
try:
    from agents.regional_news_agent import run_regional_news_model
except ModuleNotFoundError:  # pragma: no cover
    def run_regional_news_model(*args, **kwargs):
        return {"regional_sentiment_score": 0.0, "impact_factor": 1.0}

try:
    from agents.global_news_agent import run_global_news_model
except ModuleNotFoundError:  # pragma: no cover
    def run_global_news_model(*args, **kwargs):
        return {"global_sentiment_score": 0.0, "impact_factor": 1.0}

try:
    from agents.social_sentiment_agent import run_social_sentiment_model
except ModuleNotFoundError:  # pragma: no cover
    def run_social_sentiment_model(*args, **kwargs):
        return {"social_sentiment_score": 0.0, "impact_factor": 1.0}

# Volatility helpers for rolling plots
from agents.volatility_agent.volatility_model import (
    compute_daily_returns,
    compute_rolling_volatility,
    compute_rolling_confidence,
)

# News-impact detection utility (optional for backtests)
try:
    from utils.news_impact import detect_news_impact
except ModuleNotFoundError:  # pragma: no cover
    def detect_news_impact(*args, **kwargs):
        return {}


# ---------------------------------------------------------------------------
# 1. Ticker universe — 50 stocks across sectors
# ---------------------------------------------------------------------------

TICKERS = [
    # Technology
    "AAPL", "NVDA", "TSLA", "META", "AMZN", "MSFT", "GOOG", "AVGO",
    "ADBE", "CRM", "AMD", "INTC", "ORCL", "CSCO", "IBM",

    # Broad-market & Thematic ETFs
    "SPY", "QQQ", "IWM", "ARKK", "DIA", "XLK", "VTI",

    # Finance
    "JPM", "BAC", "GS", "MS", "C", "WFC", "BLK", "SCHW",

    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT",

    # Consumer Staples & Discretionary
    "KO", "PG", "PEP", "COST", "WMT", "MCD", "NKE", "SBUX",

    # Energy & Industrials
    "XOM", "CVX", "CAT", "GE",
]

assert len(TICKERS) == 50, f"Expected 50 tickers, got {len(TICKERS)}"


# ---------------------------------------------------------------------------
# 2. Output paths
# ---------------------------------------------------------------------------

OUTPUT_DIR = "output"
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")


def ensure_output_dirs():
    os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 3. Adjusted trend score — new formula
# ---------------------------------------------------------------------------

CONFIDENCE_MULTIPLIER = {
    "High": 1.0,
    "Medium": 0.6,
    "Low": 0.3,
}


def compute_adjusted_trend_score(
    trend_score: float,
    confidence: str,
    regional_sentiment: float,
    regional_impact: float,
    global_sentiment: float,
    global_impact: float,
    social_sentiment: float,
    social_impact: float,
    learned_weights=None,
    ret_5d: float = 0.0,
    ret_10d: float = 0.0,
    rolling_vol: float = 0.0,
    relative_volume: float = 0.0,
) -> float:
    """
    Adjusted trend score formula:

        adjusted = w1 * trend_score * trend_confidence
                 + w2 * regional_sentiment * regional_impact
                 + ... + optional momentum/vol/volume terms when learned_weights has them.

    When *learned_weights* is provided, uses data-driven coefficients.
    """
    trend_confidence = CONFIDENCE_MULTIPLIER.get(confidence, 0.5)

    if learned_weights is not None:
        adjusted = learned_weights.compute_adjusted_score(
            f_trend=trend_score * trend_confidence,
            f_regional=regional_sentiment * regional_impact,
            f_global=global_sentiment * global_impact,
            f_social=social_sentiment * social_impact,
            ret_5d=ret_5d,
            ret_10d=ret_10d,
            rolling_vol=rolling_vol,
            relative_volume=relative_volume,
        )
    else:
        adjusted = (
            trend_score * trend_confidence
            + regional_sentiment * regional_impact
            + global_sentiment * global_impact
            + social_sentiment * social_impact
        )

    return round(adjusted, 4)


def classify_final_signal(adjusted_trend: float) -> str:
    """
    Map the adjusted trend score directly to a signal.

        > 0.5  → Bullish
        < -0.5 → Bearish
        else   → Neutral
    """
    if adjusted_trend > 0.5:
        return "Bullish"
    elif adjusted_trend < -0.5:
        return "Bearish"
    else:
        return "Neutral"


# ---------------------------------------------------------------------------
# 4. Process a single ticker through all five agents
# ---------------------------------------------------------------------------

def process_ticker(ticker: str, global_news_result: dict, learned_weights=None) -> dict | None:
    """
    Run the full five-agent pipeline for one ticker.

    The global news result is passed in (computed once for all tickers).
    When *learned_weights* is provided, the adjusted score uses data-driven
    coefficients instead of the default equal-weight formula.
    """
    # --- Download price data ---
    try:
        stock_data = download_stock_data(ticker, period="2y")
    except Exception as exc:
        print(f"    [ERROR] Download failed for {ticker}: {exc}")
        return None

    if stock_data.empty:
        print(f"    [SKIP]  No data returned for {ticker}")
        return None

    # --- Agent 1: Trend ---
    features = build_features(stock_data)

    if features.empty:
        print(f"    [SKIP]  Not enough history for trend features ({ticker})")
        return None

    trend_result = run_trend_model(features)

    # --- Agent 2: Volatility ---
    vol_result = run_volatility_model(stock_data)

    if vol_result is None:
        print(f"    [SKIP]  Not enough history for volatility ({ticker})")
        return None

    # --- Agent 3: Regional News ---
    regional_result = run_regional_news_model(ticker)

    # --- Agent 4: Global News (pre-computed) ---
    global_result = global_news_result

    # --- Agent 5: Social Sentiment ---
    social_result = run_social_sentiment_model(ticker)

    # --- News Impact Detection (price-based confirmation) ---
    impact_df = detect_news_impact(stock_data)
    latest_row = impact_df.dropna(subset=["abnormal_return", "volume_ratio"]).iloc[-1] \
        if not impact_df.dropna(subset=["abnormal_return", "volume_ratio"]).empty \
        else None

    news_event = False
    impact_score = 0.0

    if latest_row is not None:
        news_event = bool(latest_row["news_event_flag"])
        impact_score = round(float(latest_row["impact_score"]), 4)

    # Boost sentiment weights when price data confirms news impact
    reg_impact = regional_result["impact_factor"]
    glob_impact = global_result["impact_factor"]

    if news_event:
        reg_impact = min(reg_impact * 1.3, 1.0)
        glob_impact = min(glob_impact * 1.2, 1.0)

    # --- Optional: momentum/vol/volume for learned weights ---
    ret_5d = ret_10d = rolling_vol = relative_volume = 0.0
    if learned_weights is not None and not stock_data.empty and not features.empty:
        close = stock_data["Close"]
        volume = stock_data["Volume"]
        daily_ret = features["daily_return"]
        if len(close) >= 6:
            ret_5d = float(close.pct_change(5).iloc[-1]) if not pd.isna(close.pct_change(5).iloc[-1]) else 0.0
        if len(close) >= 11:
            ret_10d = float(close.pct_change(10).iloc[-1]) if not pd.isna(close.pct_change(10).iloc[-1]) else 0.0
        if len(daily_ret) >= 20:
            rv = daily_ret.rolling(20).std()
            rolling_vol = float(rv.iloc[-1]) if not pd.isna(rv.iloc[-1]) else 0.0
        if len(volume) >= 20:
            vma = volume.rolling(20).mean()
            rel = volume.iloc[-1] / vma.iloc[-1] if vma.iloc[-1] and not pd.isna(vma.iloc[-1]) else 1.0
            relative_volume = float(rel) if not (pd.isna(rel) or np.isinf(rel)) else 1.0

    # --- Compute adjusted trend score (with potentially boosted impacts) ---
    adjusted_score = compute_adjusted_trend_score(
        trend_score=trend_result["trend_score"],
        confidence=vol_result["confidence"],
        regional_sentiment=regional_result["regional_sentiment_score"],
        regional_impact=reg_impact,
        global_sentiment=global_result["global_sentiment_score"],
        global_impact=glob_impact,
        social_sentiment=social_result["social_sentiment_score"],
        social_impact=social_result["impact_factor"],
        learned_weights=learned_weights,
        ret_5d=ret_5d,
        ret_10d=ret_10d,
        rolling_vol=rolling_vol,
        relative_volume=relative_volume,
    )

    final_signal = classify_final_signal(adjusted_score)

    # --- Combine into one row (include sector for multi-asset context) ---
    from utils.sectors import get_sector
    combined = {
        "Ticker": ticker,
        "Sector": get_sector(ticker),
        "Trend Score": trend_result["trend_score"],
        "Probability Up": trend_result["probability_up"],
        "Volatility 20": vol_result["volatility_20"],
        "Volatility 50": vol_result["volatility_50"],
        "Confidence": vol_result["confidence"],
        "Regional Sentiment": regional_result["regional_sentiment_score"],
        "Global Sentiment": global_result["global_sentiment_score"],
        "Social Sentiment": social_result["social_sentiment_score"],
        "Adjusted Score": adjusted_score,
        "News Event": news_event,
        "Impact Score": impact_score,
        "Final Signal": final_signal,
    }

    return combined


# ---------------------------------------------------------------------------
# 5. Run the full pipeline (importable by dashboard.py)
# ---------------------------------------------------------------------------

LEARNED_WEIGHTS_PATH = "output/learned_weights.json"


def run_pipeline(use_learned_weights: bool = False) -> pd.DataFrame:
    """
    Execute the five-agent pipeline across all tickers.

    Returns a DataFrame with one row per successfully processed ticker,
    sorted by adjusted score (strongest bullish first).

    When *use_learned_weights* is True (or the file at LEARNED_WEIGHTS_PATH
    exists), the adjusted score is computed with data-driven weights.
    """
    # Optionally load learned weights
    lw = None
    if use_learned_weights:
        try:
            from agents.weight_learning_agent import LearnedWeights
            lw = LearnedWeights.load(LEARNED_WEIGHTS_PATH)
            print(f"  Using learned weights from {LEARNED_WEIGHTS_PATH}")
            print(f"    w_trend={lw.w_trend:.4f}  w_regional={lw.w_regional:.4f}  "
                  f"w_global={lw.w_global:.4f}  w_social={lw.w_social:.4f}  "
                  f"intercept={lw.intercept:.6f}")
        except FileNotFoundError:
            print(f"  [WARN] Learned weights not found at {LEARNED_WEIGHTS_PATH}, using rule-based")
        except Exception as exc:
            print(f"  [WARN] Failed to load learned weights: {exc}, using rule-based")

    print("  Running Global News Agent (once for all tickers)...")
    global_news_result = run_global_news_model()
    print(f"    Global Sentiment : {global_news_result['global_sentiment_score']:+.4f}")
    print(f"    Impact Factor    : {global_news_result['impact_factor']:.4f}")
    print()

    all_results = []
    total = len(TICKERS)

    for i, ticker in enumerate(TICKERS, start=1):
        print(f"  [{i}/{total}] {ticker}")
        result = process_ticker(ticker, global_news_result, learned_weights=lw)

        if result is not None:
            all_results.append(result)

    results_df = pd.DataFrame(all_results)

    if not results_df.empty:
        results_df.sort_values("Adjusted Score", ascending=False, inplace=True)
        results_df.reset_index(drop=True, inplace=True)

    return results_df


# ---------------------------------------------------------------------------
# 6. Rolling trend scores (for time-series overlay)
# ---------------------------------------------------------------------------

def compute_rolling_trend_scores(features: pd.DataFrame) -> pd.Series:
    """Vectorised trend-score formula applied to every row."""
    scale = 10.0

    scores = (
        0.30 * features["momentum_3m"] * scale
        + 0.25 * features["momentum_6m"] * scale
        + 0.25 * features["ma_crossover_signal"]
        + 0.20 * features["daily_return"] * scale
    )

    return scores


# ---------------------------------------------------------------------------
# 7. Detect abnormal price moves (proxy for news-impact events)
# ---------------------------------------------------------------------------

def detect_news_impact_events(features: pd.DataFrame) -> pd.DataFrame:
    """
    Identify days where the absolute daily return exceeds 2 standard
    deviations of the rolling 20-day return distribution.

    These outlier days are a price-based proxy for significant news events.
    Returns a filtered DataFrame of only the event days.
    """
    daily_ret = features["daily_return"]
    rolling_std = daily_ret.rolling(window=20).std()

    # Threshold: return magnitude is more than 2x the recent rolling std
    threshold = 2.0 * rolling_std
    is_event = daily_ret.abs() > threshold

    event_days = features[is_event].copy()
    return event_days


# ---------------------------------------------------------------------------
# 8. Plotting — enhanced with news-event markers + social sentiment
# ---------------------------------------------------------------------------

CONFIDENCE_COLOURS = {
    "High": "#2ecc71",
    "Medium": "#f1c40f",
    "Low": "#e74c3c",
}

SIGNAL_COLOURS = {
    "Bullish": "#2ecc71",
    "Bearish": "#e74c3c",
    "Neutral": "#f39c12",
}


def plot_combined_chart(
    ticker: str,
    features: pd.DataFrame,
    final_signal: str,
    confidence: str,
    regional_sentiment: float,
    global_sentiment: float,
    social_sentiment: float,
    adjusted_score: float,
):
    """
    Four-panel chart for one ticker:

        Panel 1 — Price + MAs + confidence shading + news-event markers
        Panel 2 — Rolling trend score bars
        Panel 3 — Rolling 20-day volatility
        Panel 4 — Sentiment summary (regional + global + social)

    Saved as {ticker}_trend_plot.png.
    """
    dates = features.index
    close_prices = features["Close"]
    ma_50 = features["ma_50"]
    ma_200 = features["ma_200"]

    rolling_scores = compute_rolling_trend_scores(features)

    daily_returns = compute_daily_returns(close_prices)
    rolling_vol_20 = compute_rolling_volatility(daily_returns, window=20)
    rolling_conf = compute_rolling_confidence(daily_returns, window=20)

    # Detect abnormal return days as news-event proxies
    event_days = detect_news_impact_events(features)

    signal_colour = SIGNAL_COLOURS.get(final_signal, "#888888")

    # --- Build four-panel figure ---
    fig, (ax_price, ax_score, ax_vol, ax_sent) = plt.subplots(
        nrows=4,
        ncols=1,
        figsize=(14, 13),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 1, 1, 0.8]},
    )

    fig.suptitle(
        f"{ticker}  —  Signal: {final_signal}  |  Confidence: {confidence}"
        f"  |  Adj. Score: {adjusted_score:+.2f}",
        fontsize=13,
        fontweight="bold",
        color=signal_colour,
    )

    # =================================================================
    # Panel 1: Price + MAs + Confidence shading + News-event markers
    # =================================================================
    ax_price.plot(dates, close_prices, linewidth=1.3, label="Close", color="#2c3e50")
    ax_price.plot(dates, ma_50, linewidth=0.9, label="50-day MA", color="#3498db", linestyle="--")
    ax_price.plot(dates, ma_200, linewidth=0.9, label="200-day MA", color="#e67e22", linestyle="--")

    price_min = close_prices.min() * 0.97
    price_max = close_prices.max() * 1.03

    # Confidence background shading
    for conf_label, conf_colour in CONFIDENCE_COLOURS.items():
        mask = rolling_conf == conf_label
        ax_price.fill_between(
            dates, price_min, price_max,
            where=mask, color=conf_colour, alpha=0.08,
            label=f"{conf_label} conf.",
        )

    # News-event markers: triangles on days with abnormal price moves
    if not event_days.empty:
        event_dates = event_days.index
        event_prices = event_days["Close"]
        event_returns = event_days["daily_return"]

        # Green up-triangle for positive events, red down-triangle for negative
        pos_mask = event_returns > 0
        neg_mask = event_returns <= 0

        if pos_mask.any():
            ax_price.scatter(
                event_dates[pos_mask], event_prices[pos_mask],
                marker="^", color="#27ae60", s=50, zorder=5,
                label="News impact (+)",
            )
        if neg_mask.any():
            ax_price.scatter(
                event_dates[neg_mask], event_prices[neg_mask],
                marker="v", color="#c0392b", s=50, zorder=5,
                label="News impact (−)",
            )

    ax_price.set_ylabel("Price ($)", fontsize=10)
    ax_price.set_ylim(price_min, price_max)
    ax_price.legend(loc="upper left", fontsize=7, ncol=4)
    ax_price.grid(True, alpha=0.3)

    ax_price.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_price.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # =================================================================
    # Panel 2: Rolling Trend Score
    # =================================================================
    score_colors = np.where(rolling_scores >= 0, "#2ecc71", "#e74c3c")
    ax_score.bar(dates, rolling_scores, color=score_colors, width=1.0, alpha=0.7)
    ax_score.axhline(y=0, color="#7f8c8d", linewidth=0.8)
    ax_score.set_ylabel("Trend Score", fontsize=10)
    ax_score.grid(True, alpha=0.3)

    ax_score.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_score.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # =================================================================
    # Panel 3: Rolling 20-day Volatility
    # =================================================================
    ax_vol.plot(dates, rolling_vol_20, linewidth=1.0, color="#8e44ad", label="20-day Vol")
    ax_vol.fill_between(dates, 0, rolling_vol_20, color="#8e44ad", alpha=0.15)
    ax_vol.set_ylabel("Volatility", fontsize=10)
    ax_vol.legend(loc="upper left", fontsize=7)
    ax_vol.grid(True, alpha=0.3)

    ax_vol.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax_vol.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))

    # =================================================================
    # Panel 4: Sentiment Summary — Regional + Global + Social
    # =================================================================
    bar_labels = ["Regional News", "Global News", "Social Media"]
    bar_values = [regional_sentiment, global_sentiment, social_sentiment]
    bar_colors = ["#2ecc71" if v >= 0 else "#e74c3c" for v in bar_values]

    bars = ax_sent.barh(bar_labels, bar_values, color=bar_colors, height=0.5, alpha=0.8)
    ax_sent.set_xlim(-1.0, 1.0)
    ax_sent.axvline(x=0, color="#7f8c8d", linewidth=0.8)
    ax_sent.set_xlabel("Sentiment Score", fontsize=10)
    ax_sent.grid(True, alpha=0.3, axis="x")

    # Value labels on bars
    for i, val in enumerate(bar_values):
        x_pos = val + 0.03 if val >= 0 else val - 0.03
        ha = "left" if val >= 0 else "right"
        ax_sent.text(
            x_pos, i, f"{val:+.2f}",
            va="center", ha=ha, fontsize=9, fontweight="bold",
        )

    fig.autofmt_xdate(rotation=30)
    plt.tight_layout(rect=[0, 0, 1, 0.95])

    plot_path = os.path.join(PLOTS_DIR, f"{ticker}_trend_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return plot_path


# ---------------------------------------------------------------------------
# 9. Generate plots — skip tickers whose plots already exist
# ---------------------------------------------------------------------------

def generate_all_plots(results_df: pd.DataFrame):
    """
    For each ticker, generate the combined chart.
    Skips tickers whose plot PNG already exists on disk.
    """
    total = len(results_df)

    for idx, row in results_df.iterrows():
        ticker = row["Ticker"]
        position = idx + 1

        existing_path = os.path.join(PLOTS_DIR, f"{ticker}_trend_plot.png")
        if os.path.exists(existing_path):
            print(f"  [{position}/{total}] {ticker} — plot exists, skipping")
            continue

        print(f"  [{position}/{total}] Plotting {ticker}...")

        try:
            stock_data = download_stock_data(ticker, period="2y")
            features = build_features(stock_data)

            if features.empty:
                print(f"    [SKIP] Not enough data for {ticker}")
                continue

            plot_path = plot_combined_chart(
                ticker=ticker,
                features=features,
                final_signal=row["Final Signal"],
                confidence=row["Confidence"],
                regional_sentiment=row["Regional Sentiment"],
                global_sentiment=row["Global Sentiment"],
                social_sentiment=row["Social Sentiment"],
                adjusted_score=row["Adjusted Score"],
            )
            print(f"    Saved → {plot_path}")

        except Exception as exc:
            print(f"    [ERROR] Plot failed for {ticker}: {exc}")


# ---------------------------------------------------------------------------
# 10. Console summary helpers
# ---------------------------------------------------------------------------

def print_summary(results_df: pd.DataFrame):
    """Print the results table and distribution summaries to the console."""
    display_cols = [
        "Ticker", "Trend Score", "Adjusted Score",
        "Confidence", "Regional Sentiment", "Global Sentiment",
        "Social Sentiment", "News Event", "Impact Score", "Final Signal",
    ]
    print(results_df[display_cols].to_string(index=False))
    print()

    # Signal distribution
    signal_counts = results_df["Final Signal"].value_counts()
    print("Final Signal Summary:")
    for name, count in signal_counts.items():
        print(f"  {name}: {count}")
    print()

    # Confidence distribution
    conf_counts = results_df["Confidence"].value_counts()
    print("Confidence Summary:")
    for name, count in conf_counts.items():
        print(f"  {name}: {count}")
    print()


# ---------------------------------------------------------------------------
# 11. Main entry point
# ---------------------------------------------------------------------------

def main():
    ensure_output_dirs()

    print("=" * 70)
    print("  Trend Signal Engine — Multi-Agent Batch Run (v3)")
    print("  Agents: Trend + Volatility + Regional News + Global News + Social")
    print("=" * 70)
    print()

    # ---- Phase 1: Run all five agents on every ticker ----
    results_df = run_pipeline()

    if results_df.empty:
        print("No results collected. All tickers failed.")
        return

    print()

    # ---- Phase 2: In-memory summary ----
    print_summary(results_df)

    # ---- Phase 3: Generate plots ----
    print("Generating enhanced plots (existing plots will be skipped)...")
    print()
    generate_all_plots(results_df)

    print()
    print("=" * 70)
    print("  Done. Plots saved to 'output/plots/'.")
    print("  All metrics are in memory (no CSV written).")
    print("  Run 'python dashboard.py' for an interactive Plotly report.")
    print("=" * 70)


if __name__ == "__main__":
    main()
