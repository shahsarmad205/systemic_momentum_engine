"""
Data layer — market data loading, caching, API integration.
Delegates to utils.market_data; MarketDataLoader is the facade for research code.
"""

from .loader import MarketDataLoader

__all__ = ["MarketDataLoader"]
