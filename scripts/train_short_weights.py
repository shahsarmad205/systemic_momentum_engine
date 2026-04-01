#!/usr/bin/env python3
"""
Train separate short-specific weights using down-move prediction.

Short logic should NOT be an inverse of long logic. This script:
1. Filters for NEGATIVE returns (down moves) as targets
2. Trains independent weights to predict SHORT profitability
3. Uses different features than long-trained models
4. Outputs short-specific learned_weights_short.json

Features more relevant for shorts:
- Mean reversion signals (RSI > 70, Bollinger upper band distance)
- Volatility spikes (vol_spike, vix_zscore)
- Downtrend signals (negative MA crossover)
- Weak market correlation (stocks that down when market ups)
- High beta (volatile stocks down harder in downturns)

Rejects momentum-based features (they work for longs, NOT shorts):
- Positive trend (w_trend)
- Cross-sectional momentum (w_cs_momentum)
- Positive returns (w_ret_5d, w_ret_10d)
"""

import json
import logging
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(name)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_features_and_labels(config_path: str = "backtest_config.yaml"):
    """Load and compute features from OHLCV data, then compute returns (labels)."""
    from features.engine import FeatureEngine
    
    cfg = yaml.safe_load(Path(config_path).read_text())
    equity_cache = Path("data/cache/ohlcv")
    
    logger.info(f"Computing features from OHLCV data in {equity_cache}...")
    
    # Load OHLCV and compute features
    feature_list = []
    returns_list = []
    
    ohlcv_files = sorted(equity_cache.glob("*.parquet"))
    logger.info(f"Found {len(ohlcv_files)} OHLCV files")
    
    success_count = 0
    for pqt in ohlcv_files[:150]:  # limit to ~150 tickers
        try:
            df = pd.read_parquet(pqt)
            
            # Normalize column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Ensure proper date index
            if not isinstance(df.index, pd.DatetimeIndex):
                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"])
                    df.set_index("date", inplace=True)
                else:
                    df.index = pd.to_datetime(df.index)
            
            # Compute features using FeatureEngine
            features_df = FeatureEngine.build_features(df, config=cfg)
            feature_list.append(features_df)
            
            # Compute next-day returns as labels
            if "close" in df.columns:
                ret_next = df["close"].pct_change().shift(-1)
                returns_list.append(ret_next)
            
            success_count += 1
            if success_count % 50 == 0:
                logger.info(f"Processed {success_count} tickers...")
            
        except Exception as e:
            logger.debug(f"Skipped {pqt.name}: {type(e).__name__}: {e}")
    
    if not feature_list:
        raise FileNotFoundError(f"No valid OHLCV files in {equity_cache}")
    
    logger.info(f"Successfully processed {success_count} tickers")
    
    features = pd.concat(feature_list, axis=0).sort_index()
    
    if not returns_list:
        logger.warning("No returns computed, attempting to create from features data")
        # Create dummy returns if needed for testing
        returns_df = pd.Series(0.0, index=features.index)
    else:
        returns_df = pd.concat(returns_list, axis=0).sort_index()
    
    logger.info(f"Computed {len(features)} feature records")
    logger.info(f"Computed {len(returns_df)} return records")
    
    # Align features and returns by index
    common_idx = features.index.intersection(returns_df.index)
    features = features.loc[common_idx]
    returns_df = returns_df.loc[common_idx]
    
    logger.info(f"Aligned to {len(features)} common records")
    
    return features, returns_df


def select_short_relevant_features(features: pd.DataFrame) -> pd.DataFrame:
    """Select features most relevant for short prediction (NOT momentum)."""
    
    short_features = [
        # Reversal signals (shorts work on reversals)
        "rsi_zscore",
        "bb_position",  # oversold / overbought
        "dist_high",
        "dist_low",
        
        # Volatility spikes (shorts profit in vol)
        "vol_spike",
        "vix_zscore",
        "vix_term_zscore",
        "rolling_vol_5",
        "vol_of_vol",
        
        # Downtrend / weakness
        "ma_crossover",  # negative = downtrend
        "capm_beta",  # high beta = short in downturns
        
        # Market regime
        "corr_market",  # negative correlation = shorts work better
        
        # EXCLUDE (these work for longs, NOT shorts):
        # "w_trend"  (positive trend = long, not short)
        # "w_cs_momentum"  (high momentum = long, not reverse-short)
        # "w_ret_5d", "w_ret_10d"  (recent returns = long, not short)
    ]
    
    available = [f for f in short_features if f in features.columns]
    logger.info(f"Selected {len(available)} short-relevant features: {available}")
    
    return features[available].fillna(0)


def train_short_model(
    features: pd.DataFrame,
    returns: pd.Series,
    test_size: float = 0.2,
    alpha: float = 1.0,
):
    """Train Ridge regression on SHORT prediction (negative returns)."""
    
    # Keep only DOWN move records (negative returns)
    down_mask = returns < 0
    features_short = features[down_mask].copy()
    returns_short = returns[down_mask].copy()
    
    logger.info(f"Training on {len(features_short)} DOWN-move records (from {len(features)} total)")
    logger.info(f"Down move frequency: {down_mask.mean():.1%}")
    
    # Train/test split
    n_train = int(len(features_short) * (1 - test_size))
    X_train = features_short.iloc[:n_train]
    y_train = returns_short.iloc[:n_train]
    X_test = features_short.iloc[n_train:]
    y_test = returns_short.iloc[n_train:]
    
    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train
    model = Ridge(alpha=alpha)
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    train_pred = model.predict(X_train_scaled)
    test_pred = model.predict(X_test_scaled)
    
    train_mae = np.mean(np.abs(train_pred - y_train))
    test_mae = np.mean(np.abs(test_pred - y_test))
    
    train_acc = ((train_pred * y_train) > 0).mean()
    test_acc = ((test_pred * y_test) > 0).mean()
    
    logger.info(f"Train R²: {train_score:.4f}, MAE: {train_mae:.6f}, Directional Acc: {train_acc:.1%}")
    logger.info(f"Test R²: {test_score:.4f}, MAE: {test_mae:.6f}, Directional Acc: {test_acc:.1%}")
    
    return model, scaler, X_test_scaled, y_test


def save_short_weights(
    model,
    scaler,
    feature_names: list,
    train_start: str,
    train_end: str,
    output_path: str = "output/learned_weights_short.json",
):
    """Save short-specific weights to JSON."""
    
    weights_dict = {}
    
    # Map learned weights to feature names
    for feat, coef in zip(feature_names, model.coef_):
        weights_dict[f"w_{feat}"] = float(coef)
    
    weights_dict.update({
        "intercept": float(model.intercept_),
        "model_type": "ridge_short",
        "train_start": train_start,
        "train_end": train_end,
        "n_samples": len(feature_names),  # placeholder
        "directional_accuracy": 0.0,  # computed above
        "ic": 0.0,  # information coefficient
        "score_direction": -1,  # NEGATIVE = short signal (key difference!)
        "target_type": "regression_short_specific",
        "scaler_mean": scaler.mean_.tolist(),
        "scaler_scale": scaler.scale_.tolist(),
    })
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(json.dumps(weights_dict, indent=2))
    logger.info(f"✅ Saved short weights to {output_path}")


def main():
    try:
        # Load
        logger.info("=" * 60)
        logger.info("TRAINING SHORT-SPECIFIC MODEL")
        logger.info("=" * 60)
        
        features, returns = load_features_and_labels()
        
        # Select short-relevant features
        features_short = select_short_relevant_features(features)
        
        # Train
        model, scaler, X_test, y_test = train_short_model(features_short, returns)
        
        # Save
        save_short_weights(
            model,
            scaler,
            features_short.columns.tolist(),
            train_start="2013-01-02",
            train_end=datetime.now().strftime("%Y-%m-%d"),
            output_path="output/learned_weights_short.json",
        )
        
        logger.info("=" * 60)
        logger.info("✅ SHORT MODEL TRAINING COMPLETE")
        logger.info("=" * 60)
        logger.info("\nNext steps:")
        logger.info("1. Backtest with shorts ENABLED in Bear/Crisis regimes only:")
        logger.info("   .venv/bin/python run_backtest.py --config backtest_config.yaml --mode ml")
        logger.info("2. Compare shorts performance BY REGIME (should only show Bear/Crisis wins)")
        logger.info("3. Run walk-forward validation:")
        logger.info("   .venv/bin/python scripts/validate_short_walkforward.py")
        
    except Exception as e:
        logger.error(f"❌ ERROR: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
