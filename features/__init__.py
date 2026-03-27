"""
Public feature API for the trend_signal_engine project.

This package aggregates single-name and cross-sectional feature
builders so that other modules can import them from one place.
"""

from __future__ import annotations

from .breadth_features import calculate_market_breadth
from .engine import FeatureEngine
from .feature_pipeline import build_feature_matrix
from .latent_factor_features import extract_latent_factors
from .liquidity_features import calculate_liquidity_features
from .momentum_features import calculate_momentum_features
from .regime_features import detect_market_regime
from .volatility_features import calculate_volatility_features

__all__ = [
    "FeatureEngine",
    "build_feature_matrix",
    "calculate_momentum_features",
    "calculate_volatility_features",
    "calculate_liquidity_features",
    "detect_market_regime",
    "calculate_market_breadth",
    "extract_latent_factors",
]

