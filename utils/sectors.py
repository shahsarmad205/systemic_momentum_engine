"""
Sector Mapping — Single source of truth for ticker → sector.
Used for sector aggregates, signal adjustment, and portfolio exposure limits.
"""

from __future__ import annotations

# Ticker → sector name (GICS-style)
SECTOR_MAP: dict[str, str] = {
    # Technology
    "AAPL": "Technology", "MSFT": "Technology", "NVDA": "Technology",
    "META": "Technology", "GOOG": "Technology", "AVGO": "Technology",
    "ADBE": "Technology", "CRM": "Technology", "AMD": "Technology",
    "INTC": "Technology", "ORCL": "Technology", "CSCO": "Technology",
    "IBM": "Technology", "QCOM": "Technology", "TXN": "Technology",
    "ACN": "Technology",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary",
    "HD": "Consumer Discretionary", "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "LOW": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    # Financials
    "JPM": "Financials", "V": "Financials", "MA": "Financials",
    "BAC": "Financials", "GS": "Financials", "AXP": "Financials",
    "BLK": "Financials", "MS": "Financials", "C": "Financials",
    "WFC": "Financials", "SCHW": "Financials",
    # Healthcare
    "JNJ": "Healthcare", "UNH": "Healthcare", "ABBV": "Healthcare",
    "PFE": "Healthcare", "MRK": "Healthcare", "LLY": "Healthcare",
    "TMO": "Healthcare", "DHR": "Healthcare", "ABT": "Healthcare",
    "MDT": "Healthcare", "AMGN": "Healthcare",
    # Energy
    "XOM": "Energy", "CVX": "Energy",
    # Consumer Staples
    "WMT": "Consumer Staples", "PG": "Consumer Staples",
    "KO": "Consumer Staples", "PEP": "Consumer Staples",
    "COST": "Consumer Staples", "PM": "Consumer Staples",
    # Industrials
    "UNP": "Industrials", "CAT": "Industrials", "BA": "Industrials",
    "MMM": "Industrials", "GE": "Industrials",
    # Communication Services
    "DIS": "Communication Services",
    # Utilities
    "NEE": "Utilities",
    # ETFs
    "SPY": "ETF", "QQQ": "ETF", "IWM": "ETF", "DIA": "ETF",
    "XLK": "ETF", "XLF": "ETF", "XLE": "ETF", "XLV": "ETF",
    "VTI": "ETF", "ARKK": "ETF",
}


def get_sector(ticker: str) -> str:
    """Return sector for ticker; 'Other' if unknown."""
    return SECTOR_MAP.get(ticker.upper(), "Other")


def get_sectors() -> list[str]:
    """Return sorted list of unique sector names (including Other)."""
    sectors = set(SECTOR_MAP.values()) | {"Other"}
    return sorted(sectors)


def tickers_by_sector(sector_map: dict[str, str] | None = None) -> dict[str, list[str]]:
    """Return sector -> list of tickers. Uses SECTOR_MAP if sector_map is None."""
    m = sector_map or SECTOR_MAP
    out: dict[str, list[str]] = {}
    for ticker, sector in m.items():
        out.setdefault(sector, []).append(ticker)
    return out
