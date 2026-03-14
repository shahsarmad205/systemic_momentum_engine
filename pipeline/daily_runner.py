"""
Daily signal generation orchestrator.

Uses pipeline (news_scraper, sentiment_scorer), signals, features, data, and config. Caches results to output/cache/signals_YYYY-MM-DD.json.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timedelta, timezone
from typing import Any

from backtesting.signals import SignalEngine
from config import get_effective_tickers, load_config
from data import MarketDataLoader
from features import FeatureEngine
from pipeline.news_scraper import fetch_news_for_ticker
from pipeline.sentiment_scorer import SentimentScorer

logger = logging.getLogger(__name__)

# Lookback for price history (enough for 200d MA + momentum warm-up)
PRICE_LOOKBACK_DAYS = 400
MIN_PRICE_ROWS = 30  # minimum rows to produce signals (enough for 20-day rolling)
CACHE_DIR = "output/cache"
DEFAULT_TOP_LONGS = 5
DEFAULT_TOP_SHORTS = 5
# Default ticker universe (at least 20 for cross-sectional longs + shorts)
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "NVDA", "META", "AMZN", "TSLA", "JPM", "BAC", "JNJ",
    "XOM", "AMD", "NFLX", "DIS", "PYPL", "INTC", "BA", "GE", "PFE", "V",
]


def _date_str(dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%Y-%m-%d")


def _time_str(dt: datetime | None = None) -> str:
    if dt is None:
        dt = datetime.now(timezone.utc)
    return dt.strftime("%H:%M:%S")


def _composite_to_signal_score(composite: float) -> float:
    """Map composite in [-1, 1] to 1–10 scale."""
    return round(1.0 + (composite + 1.0) / 2.0 * 9.0, 2)


def _composite_to_direction(composite: float) -> str:
    """Map composite score to Bullish / Bearish / Neutral."""
    if composite >= 0.1:
        return "Bullish"
    if composite <= -0.1:
        return "Bearish"
    return "Neutral"


class DailyRunner:
    """
    Orchestrates daily signal generation: load price data, build features,
    fetch news, score sentiment, compute signal in full mode, blend quant + news,
    rank cross-sectionally, and cache result.
    """

    def __init__(
        self,
        config_path: str = "backtest_config.yaml",
        cache_dir: str = CACHE_DIR,
    ):
        self.config_path = config_path
        self.cache_dir = cache_dir
        self._config = load_config(config_path)
        self._market_loader = MarketDataLoader(
            provider=getattr(self._config, "data_provider", "yahoo"),
            cache_dir=getattr(self._config, "cache_dir", "data/cache/ohlcv"),
            use_cache=getattr(self._config, "cache_ohlcv", True),
            cache_ttl_days=int(getattr(self._config, "cache_ttl_days", 0)),
        )
        self._signal_engine = SignalEngine(
            weights=getattr(self._config, "signal_weights", None),
            learned_weights=getattr(self._config, "learned_weights", None),
            signal_smoothing_enabled=getattr(self._config, "signal_smoothing_enabled", True),
            signal_smoothing_span=int(getattr(self._config, "signal_smoothing_span", 5)),
        )
        self._sentiment_scorer = SentimentScorer()
        self._top_longs = int(getattr(self._config, "top_longs", DEFAULT_TOP_LONGS))
        self._top_shorts = int(getattr(self._config, "top_shorts", DEFAULT_TOP_SHORTS))

    def run(self, tickers: list[str]) -> dict[str, Any]:
        """
        For each ticker: load price data, build features, fetch news, score sentiment;
        compute signal in full mode; blend composite_score = 0.6*quant + 0.4*news;
        build top_longs / top_shorts; save to output/cache/signals_YYYY-MM-DD.json; return full dict.
        """
        now = datetime.now(timezone.utc)
        date_str = _date_str(now)
        time_str = _time_str(now)
        end_date = date_str
        start_dt = now - timedelta(days=PRICE_LOOKBACK_DAYS)
        start_date = start_dt.strftime("%Y-%m-%d")

        signals: dict[str, dict[str, Any]] = {}
        ticker_to_composite: dict[str, float] = {}
        all_news_sentiments: list[float] = []
        success_count = 0

        for ticker in tickers:
            try:
                # 1. Load price data
                df = self._market_loader.load_price_history(ticker, start_date, end_date)
                n_rows = 0 if df is None or df.empty else len(df)
                if df is None or df.empty:
                    print(f"  SKIP {ticker}: Insufficient price data (0 rows returned)")
                    continue
                if n_rows < MIN_PRICE_ROWS:
                    print(f"  SKIP {ticker}: Insufficient price data ({n_rows} rows, need >= {MIN_PRICE_ROWS})")
                    continue
                if not all(c in df.columns for c in ("Open", "High", "Low", "Close", "Volume")):
                    print(f"  SKIP {ticker}: Missing OHLCV columns (have: {list(df.columns)})")
                    continue

                # 2. Build features
                features_df = FeatureEngine.build_features(df)
                if features_df is None or features_df.empty:
                    print(f"  SKIP {ticker}: No features (input rows={n_rows})")
                    continue

                # 3. Full-mode signal: generate_signals + fetch_ticker_sentiments + apply_sentiment_overlay
                sig_df = self._signal_engine.generate_signals(df)
                if sig_df is None or sig_df.empty:
                    print(f"  SKIP {ticker}: No signals produced")
                    continue
                sentiments = self._signal_engine.fetch_ticker_sentiments(ticker)
                sig_df = self._signal_engine.apply_sentiment_overlay(sig_df, sentiments)
                last = sig_df.iloc[-1]
                quant_score_raw = float(last.get("adjusted_score", 0.0))
                quant_signal = max(-1.0, min(1.0, quant_score_raw))
                # Confidence from signal strength: min(abs(adjusted_score), 1.0), clamped [0.3, 0.95]
                confidence_num = last.get("confidence_numeric")
                if confidence_num is not None:
                    confidence_num = max(0.3, min(0.95, float(confidence_num)))
                else:
                    confidence_num = max(0.3, min(0.95, min(abs(quant_score_raw), 1.0)))

                # 4. Fetch news and score sentiment (pipeline)
                news_items = fetch_news_for_ticker(ticker, days_back=7)
                sentiment_result = self._sentiment_scorer.score_ticker_news(news_items)
                news_sentiment = float(sentiment_result.get("aggregate_score", 0.0))
                all_news_sentiments.append(news_sentiment)
                top_headlines = []
                for day in sentiment_result.get("daily_scores", [])[:3]:
                    top_headlines.extend((day.get("headlines") or [])[:2])
                top_headlines = list(dict.fromkeys(top_headlines))[:5]

                # 5. Composite and direction
                composite_score = 0.6 * quant_signal + 0.4 * news_sentiment
                composite_score = max(-1.0, min(1.0, composite_score))
                ticker_to_composite[ticker] = composite_score

                # Top features: use feature column names from last row if available
                feature_cols = [c for c in features_df.columns if c not in ("Open", "High", "Low", "Close", "Volume")]
                top_features = feature_cols[:5] if feature_cols else []

                signals[ticker] = {
                    "signal_score": _composite_to_signal_score(composite_score),
                    "composite_score": round(composite_score, 4),
                    "quant_score": round(quant_signal, 4),
                    "news_sentiment": round(news_sentiment, 4),
                    "direction": _composite_to_direction(composite_score),
                    "confidence": confidence_num,
                    "top_features": top_features,
                    "top_headlines": top_headlines,
                    "rank": 0,  # set after ranking
                }
                success_count += 1
            except Exception as e:
                print(f"  SKIP {ticker}: {type(e).__name__}: {e}")
                logger.debug("Error processing %s", ticker, exc_info=True)
                continue

        # 6. Assign ranks and top_longs / top_shorts from composite score (highest = longs, lowest = shorts)
        ordered = sorted(signals.keys(), key=lambda t: signals[t]["composite_score"], reverse=True)
        for rank, t in enumerate(ordered, start=1):
            signals[t]["rank"] = rank

        n = len(ordered)
        # Top N by score = longs; bottom N = shorts (least bullish). Allow overlap when n is small.
        top_longs = ordered[: min(self._top_longs, n)]
        top_shorts = ordered[-min(self._top_shorts, n) :] if n else []

        market_sentiment = (
            round(sum(all_news_sentiments) / len(all_news_sentiments), 4)
            if all_news_sentiments else 0.0
        )

        result = {
            "date": date_str,
            "generated_at": time_str + " UTC",
            "signals": signals,
            "top_longs": top_longs,
            "top_shorts": top_shorts,
            "market_sentiment": market_sentiment,
        }

        # 7. Save to cache
        os.makedirs(self.cache_dir, exist_ok=True)
        path = os.path.join(self.cache_dir, f"signals_{date_str}.json")
        with open(path, "w") as f:
            json.dump(result, f, indent=2)
        logger.info("Wrote %s", path)

        # Validation summary (if top_shorts is empty with signals present, ranking bug)
        signals_out = result["signals"]
        print(f"Generated {len(signals_out)} signals")
        print(f"Bullish: {sum(1 for s in signals_out.values() if s.get('direction') == 'Bullish')}")
        print(f"Bearish: {sum(1 for s in signals_out.values() if s.get('direction') == 'Bearish')}")
        print(f"Top longs: {result['top_longs']}")
        print(f"Top shorts: {result['top_shorts']}")
        print(f"\nPipeline complete: {success_count}/{len(tickers)} tickers succeeded")

        return result


if __name__ == "__main__":
    from config import setup_logging
    setup_logging(verbose=False)
    cfg = load_config("backtest_config.yaml")
    fallback = getattr(cfg, "tickers", None) or DEFAULT_TICKERS
    tickers = get_effective_tickers(cfg.tickers, fallback)
    runner = DailyRunner()
    out = runner.run(tickers)
    print("Date:", out["date"], "Generated:", out["generated_at"])
    print("Top longs:", out["top_longs"])
    print("Top shorts:", out["top_shorts"])
    print("Market sentiment:", out["market_sentiment"])
