"""
Daily Cross-Sectional Normalization
====================================
Standardizes features across the entire universe on each bar to remove 
stock-specific scale biases (e.g., TSLA volatility vs JNJ).
"""

import numpy as np
import pandas as pd
import logging

logger = logging.getLogger(__name__)

class CrossSectionalStandardizer:
    """
    Utility for performing 'Panel normalization' on a universe of tickers.
    Transformations:
      - 'zscore': (x - mu_cs) / sigma_cs
      - 'rank': percentile ranking (0 to 1)
    """

    @staticmethod
    def standardize_feature_dict(
        feature_dict: dict[str, pd.DataFrame], 
        method: str = "zscore",
        clip: float = 3.0
    ) -> dict[str, pd.DataFrame]:
        """
        Takes a dict of {ticker: df_features} and returns a dict with normalized values.
        Every numeric column in the DataFrames is treated as a feature to be normalized 
        cross-sectionally on each date.
        """
        if not feature_dict:
            return {}

        # 1. Identify all common features (columns) across the universe.
        all_cols = set()
        for df in feature_dict.values():
            all_cols.update(df.select_dtypes(include=[np.number]).columns)
        
        # Remove target/label columns if they exist (don't normalize them)
        protected = {"signal", "direction", "pnl", "return", "signal_date", "entry_date", "exit_date", "expiry_price", "expiry_return", "original_planned_exit_date"}
        features_to_norm = [c for c in all_cols if c not in protected]
        
        if not features_to_norm:
            return feature_dict

        # Use a copy to avoid mutating the original dict in place during calculation.
        norm_dict = {tk: df.copy() for tk, df in feature_dict.items()}

        for feat in features_to_norm:
            try:
                # 2. Pivot: Create a (Date x Ticker) matrix for this specific feature.
                # We use a loop for memory efficiency over pd.concat if the universe is large.
                # But for 300-500 tickers, we can just build a master frame.
                master_list = []
                for tk, df in feature_dict.items():
                    if feat in df.columns:
                        s = df[feat].copy()
                        s.name = tk
                        master_list.append(s)
                
                if not master_list:
                    continue
                
                cs_df = pd.concat(master_list, axis=1) # (Dates x Tickers)
                
                # 3. Apply Normalization across the Ticker axis (axis=1) for each Date (row).
                if method == "zscore":
                    # (x - mu) / sigma per row
                    mu = cs_df.mean(axis=1)
                    sigma = cs_df.std(axis=1).replace(0, np.nan)
                    norm_df = cs_df.sub(mu, axis=0).div(sigma, axis=0)
                    if clip is not None:
                        norm_df = norm_df.clip(-clip, clip)
                elif method == "rank":
                    # Percentile rank (0.0 to 1.0)
                    norm_df = cs_df.rank(axis=1, pct=True)
                else:
                    logger.warning("Unknown normalization method '%s'; skipping feature '%s'", method, feat)
                    continue

                # 4. Unpivot: Update the result dictionary.
                for tk in norm_df.columns:
                    if tk in norm_dict:
                        norm_dict[tk][feat] = norm_df[tk].fillna(0.0)

            except Exception as e:
                logger.error("Failed cross-sectional normalization for feature '%s': %s", feat, e)
                continue

        return norm_dict
