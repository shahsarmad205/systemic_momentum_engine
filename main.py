"""
Trend Signal Engine — Multi-Agent Batch Runner (v5)
=====================================================
Runs two agents across 50 stock tickers:

    1. Trend Agent      → trend score, probability, raw signal
    2. Volatility Agent → rolling volatility, confidence level

Adjusted trend score formula:

    adjusted = trend_score * trend_confidence   (×learned_weights when available)

Signal thresholds (applied directly to adjusted score):
    > 0.5  → Bullish
    < -0.5 → Bearish
    else   → Neutral

All results are kept in memory (no CSV output).
Plots are saved to output/plots/ — existing plots are skipped.

Usage:
    python main.py                     (single batch run)
    python scheduler.py                (continuous refresh loop)
"""

from __future__ import annotations

import logging
import os

import numpy as np
import pandas as pd

logging.getLogger("matplotlib").setLevel(logging.WARNING)
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt


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
# 3. Adjusted trend score
# ---------------------------------------------------------------------------

CONFIDENCE_MULTIPLIER = {
    "High": 1.0,
    "Medium": 0.6,
    "Low": 0.3,
}


def compute_adjusted_trend_score(
    trend_score: float,
    confidence: str,
    learned_weights=None,
    ret_5d: float = 0.0,
    ret_10d: float = 0.0,
    rolling_vol: float = 0.0,
    relative_volume: float = 0.0,
) -> float:
    """
    Adjusted trend score:

        adjusted = trend_score * trend_confidence
                 + optional momentum/vol/volume terms when learned_weights has them.

    When *learned_weights* is provided, uses data-driven coefficients.
    """
    trend_confidence = CONFIDENCE_MULTIPLIER.get(confidence, 0.5)

    if learned_weights is not None:
        adjusted = learned_weights.compute_adjusted_score(
            f_trend=trend_score * trend_confidence,
            ret_5d=ret_5d,
            ret_10d=ret_10d,
            rolling_vol=rolling_vol,
            relative_volume=relative_volume,
        )
    else:
        adjusted = trend_score * trend_confidence

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
# 4. Process a single ticker through the two agents
# ---------------------------------------------------------------------------

def process_ticker(ticker: str, learned_weights=None) -> dict | None:
    """
    Run the two-agent pipeline for one ticker.

    When *learned_weights* is provided, the adjusted score uses data-driven
    coefficients instead of the default formula.
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

    # --- Compute adjusted trend score ---
    adjusted_score = compute_adjusted_trend_score(
        trend_score=trend_result["trend_score"],
        confidence=vol_result["confidence"],
        learned_weights=learned_weights,
        ret_5d=ret_5d,
        ret_10d=ret_10d,
        rolling_vol=rolling_vol,
        relative_volume=relative_volume,
    )

    final_signal = classify_final_signal(adjusted_score)

    # --- Combine into one row ---
    from utils.quant_utils import get_sector
    combined = {
        "Ticker": ticker,
        "Sector": get_sector(ticker),
        "Trend Score": trend_result["trend_score"],
        "Probability Up": trend_result["probability_up"],
        "Volatility 20": vol_result["volatility_20"],
        "Volatility 50": vol_result["volatility_50"],
        "Confidence": vol_result["confidence"],
        "Adjusted Score": adjusted_score,
        "Final Signal": final_signal,
    }

    return combined


# ---------------------------------------------------------------------------
# 5. Run the full pipeline (importable by dashboard.py)
# ---------------------------------------------------------------------------

LEARNED_WEIGHTS_PATH = "output/learned_weights.json"


def run_pipeline(use_learned_weights: bool = False) -> pd.DataFrame:
    """
    Execute the two-agent pipeline across all tickers.

    Returns a DataFrame with one row per successfully processed ticker,
    sorted by adjusted score (strongest bullish first).

    When *use_learned_weights* is True (or the file at LEARNED_WEIGHTS_PATH
    exists), the adjusted score is computed with data-driven weights.
    """
    lw = None
    if use_learned_weights:
        try:
            from agents.weight_learning_agent import LearnedWeights
            lw = LearnedWeights.load(LEARNED_WEIGHTS_PATH)
            print(f"  Using learned weights from {LEARNED_WEIGHTS_PATH}")
            print(f"    w_trend={lw.w_trend:.4f}  intercept={lw.intercept:.6f}")
        except FileNotFoundError:
            print(f"  [WARN] Learned weights not found at {LEARNED_WEIGHTS_PATH}, using rule-based")
        except Exception as exc:
            print(f"  [WARN] Failed to load learned weights: {exc}, using rule-based")

    all_results = []
    total = len(TICKERS)

    for i, ticker in enumerate(TICKERS, start=1):
        print(f"  [{i}/{total}] {ticker}")
        result = process_ticker(ticker, learned_weights=lw)

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
    """
    daily_ret = features["daily_return"]
    rolling_std = daily_ret.rolling(window=20).std()
    threshold = 2.0 * rolling_std
    is_event = daily_ret.abs() > threshold
    return features[is_event].copy()


# ---------------------------------------------------------------------------
# 8. Plotting — three panels (price, trend score, volatility)
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
    adjusted_score: float,
):
    """
    Three-panel chart for one ticker:

        Panel 1 — Price + MAs + confidence shading + news-event markers
        Panel 2 — Rolling trend score bars
        Panel 3 — Rolling 20-day volatility

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

    event_days = detect_news_impact_events(features)
    signal_colour = SIGNAL_COLOURS.get(final_signal, "#888888")

    fig, (ax_price, ax_score, ax_vol) = plt.subplots(
        nrows=3,
        ncols=1,
        figsize=(14, 10),
        sharex=False,
        gridspec_kw={"height_ratios": [3, 1, 1]},
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

    for conf_label, conf_colour in CONFIDENCE_COLOURS.items():
        mask = rolling_conf == conf_label
        ax_price.fill_between(
            dates, price_min, price_max,
            where=mask, color=conf_colour, alpha=0.08,
            label=f"{conf_label} conf.",
        )

    if not event_days.empty:
        event_dates = event_days.index
        event_prices = event_days["Close"]
        event_returns = event_days["daily_return"]

        pos_mask = event_returns > 0
        neg_mask = event_returns <= 0

        if pos_mask.any():
            ax_price.scatter(
                event_dates[pos_mask], event_prices[pos_mask],
                marker="^", color="#27ae60", s=50, zorder=5,
                label="Price spike (+)",
            )
        if neg_mask.any():
            ax_price.scatter(
                event_dates[neg_mask], event_prices[neg_mask],
                marker="v", color="#c0392b", s=50, zorder=5,
                label="Price spike (−)",
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
        "Confidence", "Final Signal",
    ]
    print(results_df[display_cols].to_string(index=False))
    print()

    signal_counts = results_df["Final Signal"].value_counts()
    print("Final Signal Summary:")
    for name, count in signal_counts.items():
        print(f"  {name}: {count}")
    print()

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
    print("  Trend Signal Engine — Multi-Agent Batch Run (v5)")
    print("  Agents: Trend + Volatility")
    print("=" * 70)
    print()

    results_df = run_pipeline()

    if results_df.empty:
        print("No results collected. All tickers failed.")
        return

    print()
    print_summary(results_df)

    print("Generating plots (existing plots will be skipped)...")
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
