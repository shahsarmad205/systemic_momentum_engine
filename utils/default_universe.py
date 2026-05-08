from __future__ import annotations

DEFAULT_TICKERS = [
    "AAPL", "NVDA", "TSLA", "META", "AMZN", "MSFT", "GOOG", "AVGO",
    "ADBE", "CRM", "AMD", "INTC", "ORCL", "CSCO", "IBM",
    "SPY", "QQQ", "IWM", "ARKK", "DIA", "XLK", "VTI",
    "JPM", "BAC", "GS", "MS", "C", "WFC", "BLK", "SCHW",
    "JNJ", "UNH", "PFE", "ABBV", "MRK", "LLY", "TMO", "ABT",
    "KO", "PG", "PEP", "COST", "WMT", "MCD", "NKE", "SBUX",
    "XOM", "CVX", "CAT", "GE",
]


def default_tickers() -> list[str]:
    return list(DEFAULT_TICKERS)
