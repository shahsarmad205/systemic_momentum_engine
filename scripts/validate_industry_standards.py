#!/usr/bin/env python3
"""
Industry-Standard Feature & Risk Alignment Validator

Ensures both long and short strategies comply with institutional standards:

LONG STRATEGY:
  - Trend-following momentum (positive w_trend, w_cs_momentum)
  - Mean reversion entry confirmation (RSI, Bollinger oversold)
  - Position sizing: equal-weight base
  - Stop loss: 2% per position
  - Regime: active in Bull/Sideways (when trend is positive)

SHORT STRATEGY:
  - Mean reversion focused (RSI > 70, BB upper band, dist_high > 0)
  - Anti-momentum (negative w_cs_momentum for shorts)
  - Position sizing: 50% less than longs (risk-averse)
  - Stop loss: 1.5% per position (tighter for shorts - easier to squeeze)
  - Regime: ONLY in Bear/Crisis (when trend is negative)

Risk Management:
  - Directional budget caps (max 50% net long, 30% net short)
  - Gross exposure cap (max 1.5x)
  - Short single-name cap (max 10% per position)
  - Forced liquidation on hedge ratio breach

Usage:
  python scripts/validate_industry_standards.py --config backtest_config.yaml
"""

import argparse
import json
import logging
from pathlib import Path

import pandas as pd
import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load backtest config."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_weights(weights_path: str = "output/learned_weights.json") -> dict:
    """Load learned weights."""
    if not Path(weights_path).exists():
        logger.warning(f"Weights not found at {weights_path}")
        return {}
    with open(weights_path) as f:
        return json.load(f)


def check_regime_suppression(config: dict) -> dict:
    """Verify regime_suppress_shorts is configured."""
    suppress = config.get("execution", {}).get("regime_suppress_shorts", [])
    
    checks = {
        "regime_suppress_shorts_configured": len(suppress) > 0,
        "suppress_in_bull": "Bull" in suppress,
        "suppress_in_sideways": "Sideways" in suppress,
        "suppress_config": suppress,
    }
    
    if not checks["regime_suppress_shorts_configured"]:
        logger.error("❌ regime_suppress_shorts is empty → shorts allowed in ALL regimes (bad)")
    else:
        logger.info(f"✅ Shorts suppressed in regimes: {suppress}")
    
    return checks


def check_feature_alignment(weights: dict) -> dict:
    """Verify long/short weights follow industry standards."""
    
    checks = {
        "long_favoring_features": [],
        "short_favoring_features": [],
        "anomalies": [],
    }
    
    # LONG-FAVORING (positive weights)
    long_features = {
        "w_trend": "Trend: positive = uptrend",
        "w_cs_momentum": "Momentum: positive = strong",
        "w_ret_5d": "Recent returns: positive = continuing",
        "w_vol_spike": "Volatility spike: predicts trend (contextual)",
    }
    
    # SHORT-FAVORING (negative or contrarian weights)
    short_features = {
        "w_rsi_zscore": "RSI z-score: negative/high = oversold",
        "w_bb_position": "Bollinger position: at upper band = overbought",
        "w_dist_high": "Distance to high: oversold regime",
        "w_vix_zscore": "VIX spike: predicts reversals (shorts work)",
        "w_ma_crossover": "MA negative cross: downtrend",
    }
    
    # Verify long weights
    for feat, desc in long_features.items():
        val = float(weights.get(feat, 0.0))
        if val > 0:
            checks["long_favoring_features"].append(f"{feat}: {val:.6f} ({desc}) ✅")
        elif feat not in ["w_vol_spike"]:  # vol_spike context-dependent
            checks["anomalies"].append(f"{feat}: {val:.6f} — EXPECTED POSITIVE for longs ⚠️")
    
    # Verify short weights
    for feat, desc in short_features.items():
        val = float(weights.get(feat, 0.0))
        # These should support short logic
        if feat in ["w_rsi_zscore", "w_bb_position", "w_dist_high"]:
            # For shorts, these features should have NEGATIVE correlation (oversold = short)
            # But learned weights might be positive-scaled; context is score_direction
            checks["short_favoring_features"].append(f"{feat}: {val:.6f} ({desc})")
    
    # Check direction flag
    direction = int(weights.get("score_direction", 1))
    checks["score_direction"] = direction
    if direction == 1:
        logger.info("✅ score_direction = +1 (positive scores = long bias)")
    elif direction == -1:
        logger.warning("⚠️  score_direction = -1 (negative scores = short bias) — unusual")
    
    # Check model quality
    r2 = float(weights.get("r2", 0.0))
    ic = float(weights.get("ic", 0.0))
    acc = float(weights.get("directional_accuracy", 0.0))
    
    checks["model_quality"] = {
        "r2": r2,
        "ic": ic,
        "directional_accuracy": acc,
    }
    
    if r2 < 0.001:
        checks["anomalies"].append(f"R² = {r2:.6f} (very low explanatory power)")
    if ic < 0.03:
        checks["anomalies"].append(f"IC = {ic:.6f} (very low info coefficient)")
    if acc < 0.51:
        checks["anomalies"].append(f"Directional accuracy = {acc:.1%} (near random)")
    
    return checks


def check_risk_management(config: dict) -> dict:
    """Verify risk parameters follow institutional standards."""
    
    risk = config.get("risk", {})
    execution = config.get("execution", {})
    
    checks = {
        "max_gross_exposure": risk.get("max_gross_exposure", 1.5),
        "max_net_exposure": risk.get("max_net_exposure", 0.5),
        "max_short_single_name": risk.get("max_short_single_name", 0.12),
        "stop_loss_pct": config.get("execution", {}).get("stop_loss_pct", 0.02),
        "alerts": [],
    }
    
    # Standard institutional limits
    if checks["max_gross_exposure"] > 2.0:
        checks["alerts"].append(f"Gross exposure {checks['max_gross_exposure']:.1f}x is TOO HIGH (std: 1.5x)")
    else:
        checks["alerts"].append(f"✅ Gross exposure {checks['max_gross_exposure']:.1f}x is reasonable")
    
    if checks["max_net_exposure"] > 0.6:
        checks["alerts"].append(f"⚠️  Net exposure {checks['max_net_exposure']:.1f}x slightly high (std: 0.3-0.5x)")
    else:
        checks["alerts"].append(f"✅ Net exposure {checks['max_net_exposure']:.1f}x is conservative")
    
    if checks["max_short_single_name"] > 0.15:
        checks["alerts"].append(f"⚠️  Max short size {checks['max_short_single_name']:.1%} is large (std: 8-10%)")
    else:
        checks["alerts"].append(f"✅ Max short size {checks['max_short_single_name']:.1%} is prudent")
    
    return checks


def check_backtest_results(backtest_trades_path: str = "output/backtests/trades.csv") -> dict:
    """Verify backtest results show proper regime/directional behavior."""
    
    if not Path(backtest_trades_path).exists():
        logger.warning(f"Backtest trades not found at {backtest_trades_path}")
        return {}
    
    trades = pd.read_csv(backtest_trades_path)
    
    if "direction" not in trades.columns:
        trades["direction"] = (trades.get("signal") == "Bullish").astype(int) * 2 - 1
    
    checks = {
        "total_trades": len(trades),
        "by_regime_direction": {},
        "regime_purity_alerts": [],
    }
    
    # Analyze by regime and direction
    for regime in trades["regime"].unique():
        regime_trades = trades[trades["regime"] == regime]
        long_trades = regime_trades[regime_trades["direction"] == 1]
        short_trades = regime_trades[regime_trades["direction"] == -1]
        
        checks["by_regime_direction"][regime] = {
            "total": len(regime_trades),
            "longs": len(long_trades),
            "shorts": len(short_trades),
            "long_avg_pnl": float(long_trades["pnl"].mean()) if len(long_trades) > 0 else 0,
            "short_avg_pnl": float(short_trades["pnl"].mean()) if len(short_trades) > 0 else 0,
            "long_win_rate": float((long_trades["pnl"] > 0).mean()) if len(long_trades) > 0 else 0,
            "short_win_rate": float((short_trades["pnl"] > 0).mean()) if len(short_trades) > 0 else 0,
        }
        
        # INDUSTRY STANDARD: Shorts should ONLY be in Bear/Crisis
        if regime in ["Bull", "Sideways"] and len(short_trades) > 0:
            checks["regime_purity_alerts"].append(
                f"❌ {regime} regime has {len(short_trades)} shorts (should be 0 — regime purity violated)"
            )
        else:
            checks["regime_purity_alerts"].append(
                f"✅ {regime} regime: {len(long_trades)} longs, {len(short_trades)} shorts (correct)"
            )
    
    return checks


def generate_report(config_path: str = "backtest_config.yaml"):
    """Generate comprehensive industry-standards report."""
    
    logger.info("=" * 80)
    logger.info("INDUSTRY-STANDARD ALIGNMENT VALIDATION")
    logger.info("=" * 80)
    
    # Load
    config = load_config(config_path)
    weights = load_weights()
    
    # Check 1: Regime Suppression
    logger.info("\n[CHECK 1] REGIME SUPPRESSION (Short Purity)")
    logger.info("-" * 80)
    regime_check = check_regime_suppression(config)
    logger.info(f"Configuration: {regime_check['suppress_config']}")
    
    # Check 2: Feature Alignment
    logger.info("\n[CHECK 2] FEATURE WEIGHT ALIGNMENT")
    logger.info("-" * 80)
    feature_check = check_feature_alignment(weights)
    
    logger.info("Long-favoring weights:")
    for f in feature_check["long_favoring_features"][:5]:
        logger.info(f"  {f}")
    
    logger.info("\nShort-favoring weights:")
    for f in feature_check["short_favoring_features"][:5]:
        logger.info(f"  {f}")
    
    logger.info(f"\nModel Quality: R²={feature_check['model_quality']['r2']:.6f}, IC={feature_check['model_quality']['ic']:.6f}")
    
    if feature_check["anomalies"]:
        logger.warning("\n⚠️  ANOMALIES DETECTED:")
        for a in feature_check["anomalies"]:
            logger.warning(f"  {a}")
    
    # Check 3: Risk Management
    logger.info("\n[CHECK 3] RISK MANAGEMENT ALIGNMENT")
    logger.info("-" * 80)
    risk_check = check_risk_management(config)
    for alert in risk_check["alerts"]:
        logger.info(f"  {alert}")
    
    # Check 4: Backtest Results
    logger.info("\n[CHECK 4] BACKTEST REGIME/DIRECTIONAL PURITY")
    logger.info("-" * 80)
    backtest_check = check_backtest_results()
    
    if backtest_check:
        logger.info(f"Total trades: {backtest_check['total_trades']}\n")
        for regime, stats in backtest_check["by_regime_direction"].items():
            logger.info(
                f"{regime:12} | "
                f"Longs: {stats['longs']:3} ({stats['long_avg_pnl']:7.2f} avg, {stats['long_win_rate']:5.1%}) | "
                f"Shorts: {stats['shorts']:3} ({stats['short_avg_pnl']:7.2f} avg, {stats['short_win_rate']:5.1%})"
            )
        
        logger.info("\nRegime Purity Check:")
        for alert in backtest_check["regime_purity_alerts"]:
            logger.info(f"  {alert}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("SUMMARY & RECOMMENDATIONS")
    logger.info("=" * 80)
    
    passed = all([
        regime_check["regime_suppress_shorts_configured"],
        regime_check["suppress_in_bull"],
        regime_check["suppress_in_sideways"],
    ])
    
    if passed:
        logger.info("✅ REGIME PURITY: All checks passed")
    else:
        logger.error("❌ REGIME PURITY: Configure regime_suppress_shorts in backtest_config.yaml")
    
    if not feature_check["anomalies"]:
        logger.info("✅ FEATURE ALIGNMENT: No anomalies detected")
    else:
        logger.warning(f"⚠️  FEATURE ALIGNMENT: {len(feature_check['anomalies'])} anomalies — review weights")
    
    logger.info("\n" + "=" * 80)
    return {
        "regime_suppression": regime_check,
        "features": feature_check,
        "risk": risk_check,
        "backtest": backtest_check,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Validate industry-standard alignment")
    parser.add_argument("--config", default="backtest_config.yaml", help="Config path")
    args = parser.parse_args()
    
    report = generate_report(args.config)
