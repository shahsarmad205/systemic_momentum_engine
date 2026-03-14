"""
SignalExplainer — plain-English explanations via Anthropic Claude API.
Uses file-based cache at output/cache/explanations_{date}.json keyed by ticker.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from typing import Any

import anthropic

logger = logging.getLogger(__name__)

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY")
CACHE_DIR = os.environ.get("SIGNALS_CACHE_DIR", "output/cache")
MODEL = "claude-sonnet-4-20250514"
SYSTEM_PROMPT = (
    "You are a financial analyst explaining stock signals to retail investors. "
    "Be concise, clear, and never use jargon. Always mention one risk."
)

FALLBACK = {"bullets": [], "risk": "", "summary": ""}


def _fallback() -> dict[str, Any]:
    return dict(FALLBACK)


def _user_prompt(signal_data: dict[str, Any]) -> str:
    ticker = signal_data.get("ticker", "Unknown")
    direction = signal_data.get("direction", "Neutral")
    signal_score = signal_data.get("signal_score", 0)
    top_features = signal_data.get("top_features") or []
    features_str = ", ".join(top_features) if isinstance(top_features, list) else str(top_features)
    news_sentiment = float(signal_data.get("news_sentiment", 0))
    headlines = signal_data.get("top_headlines") or []
    recent_headline = headlines[0] if headlines else ""
    return (
        f"Stock: {ticker}, Signal: {direction} ({signal_score}/10), "
        f"Key drivers: {features_str}, News sentiment: {news_sentiment:.2f}, "
        f"Recent headline: {recent_headline}"
    ).strip()


def _parse_response(text: str) -> dict[str, Any]:
    text = (text or "").strip()
    if not text:
        return _fallback()
    # Strip markdown code blocks if present
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines)
    try:
        out = json.loads(text)
        if not isinstance(out, dict):
            return _fallback()
        bullets = out.get("bullets")
        if not isinstance(bullets, list):
            out["bullets"] = []
        return {
            "bullets": out.get("bullets", [])[:3],
            "risk": str(out.get("risk", "")).strip()[:200],
            "summary": str(out.get("summary", "")).strip()[:300],
        }
    except json.JSONDecodeError:
        return _fallback()


def _explanation_cache_path(date_str: str, cache_dir: str = CACHE_DIR) -> str:
    return os.path.join(cache_dir, f"explanations_{date_str}.json")


def _load_explanations_for_date(date_str: str, cache_dir: str = CACHE_DIR) -> dict[str, dict]:
    path = _explanation_cache_path(date_str, cache_dir)
    if not os.path.isfile(path):
        return {}
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return {}


def _save_explanation(
    date_str: str, ticker: str, explanation: dict[str, Any], cache_dir: str = CACHE_DIR
) -> None:
    path = _explanation_cache_path(date_str, cache_dir)
    data = _load_explanations_for_date(date_str, cache_dir)
    data[ticker.upper()] = explanation
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


class SignalExplainer:
    """
    Generate plain-English explanations for signal data using Claude (Anthropic).
    Uses file-based cache: output/cache/explanations_{date}.json keyed by ticker.
    """

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir or CACHE_DIR

    def explain(self, signal_data: dict[str, Any]) -> dict[str, Any]:
        """
        Build user prompt from signal_data, call Claude, return parsed JSON
        with keys: bullets, risk, summary. Checks cache before calling API.
        On any error returns fallback dict with empty bullets.
        """
        ticker = (signal_data.get("ticker") or "unknown").upper()
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        cached = _load_explanations_for_date(date_str, self.cache_dir)
        if ticker in cached:
            return cached[ticker]

        if not ANTHROPIC_API_KEY:
            logger.warning("ANTHROPIC_API_KEY not set; returning fallback")
            result = _fallback()
            _save_explanation(date_str, ticker, result, self.cache_dir)
            return result

        user_text = _user_prompt(signal_data)
        instruction = (
            "Return ONLY a JSON object (no markdown) with exactly these fields:\n"
            "{\n"
            "  'bullets': ['reason 1 max 20 words', 'reason 2 max 20 words', 'reason 3 max 20 words'],\n"
            "  'risk': 'one risk factor max 20 words',\n"
            "  'summary': 'one sentence overall take max 25 words'\n"
            "}"
        )
        full_user = f"{user_text}\n\n{instruction}"

        try:
            client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
            message = client.messages.create(
                model=MODEL,
                max_tokens=512,
                system=SYSTEM_PROMPT,
                messages=[{"role": "user", "content": full_user}],
            )
            text = ""
            for block in (message.content or []):
                if getattr(block, "text", None):
                    text += block.text
            result = _parse_response(text)
            _save_explanation(date_str, ticker, result, self.cache_dir)
            return result
        except Exception as e:
            logger.exception("Explainer API error: %s", e)
            result = _fallback()
            _save_explanation(date_str, ticker, result, self.cache_dir)
            return result


def suggest_portfolio(
    weighted_avg_signal: float,
    tech_exposure: float,
    avg_news_sentiment: float,
    market_beta_estimate: float,
    risk_flags: list[str],
    holdings_summary: str,
) -> str:
    """
    One-sentence AI suggestion for portfolio context (SignalExplainer pattern).
    Uses Claude with portfolio metrics; returns fallback string on error or missing key.
    """
    if not ANTHROPIC_API_KEY:
        return "Set ANTHROPIC_API_KEY for AI suggestions. Review concentration and sector exposure."
    system = (
        "You are a financial analyst giving one-sentence portfolio suggestions to retail investors. "
        "Be concise, clear, and never use jargon. Mention at most one risk if relevant."
    )
    user = (
        f"Portfolio metrics: weighted_avg_signal={weighted_avg_signal:.2f}, "
        f"tech_exposure={tech_exposure:.1f}%, avg_news_sentiment={avg_news_sentiment:.2f}, "
        f"market_beta_estimate={market_beta_estimate:.2f}. "
        f"Risk flags: {risk_flags}. Holdings: {holdings_summary}. "
        "Return ONLY one sentence (max 25 words) as plain text, no JSON, no quotes."
    )
    try:
        client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
        message = client.messages.create(
            model=MODEL,
            max_tokens=128,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        text = ""
        for block in (message.content or []):
            if getattr(block, "text", None):
                text += block.text
        return (text or "Review concentration and sector exposure.").strip()[:300]
    except Exception as e:
        logger.exception("Portfolio suggestion API error: %s", e)
        return "Review concentration and sector exposure."
