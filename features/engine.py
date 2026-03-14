"""
FeatureEngine facade for single-name OHLCV features.

This wraps the underlying feature pipeline so existing callers can
continue to use ``FeatureEngine.build_features`` while benefiting
from the richer feature set.
"""

from __future__ import annotations

import pandas as pd

from .feature_pipeline import build_feature_matrix


class FeatureEngine:
    """
    Exposes build_features(stock_data) using the enriched pipeline.
    """

    @staticmethod
    def build_features(stock_data: pd.DataFrame, config=None) -> pd.DataFrame:
        """
        Build an enriched feature matrix from raw OHLCV stock data.

        Parameters
        ----------
        stock_data : pd.DataFrame
            OHLCV time series for a single ticker.
        config : optional
            BacktestConfig (or any object with gbm_enabled, holding_period_days).
            When gbm_enabled=True, GBM-derived features are computed.

        Returns
        -------
        pd.DataFrame
            Feature matrix produced by ``build_feature_matrix``.
        """
        return build_feature_matrix(stock_data, config=config)

