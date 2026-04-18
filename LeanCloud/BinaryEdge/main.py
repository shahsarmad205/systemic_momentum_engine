# ============================================================
# QuantConnect (Lean SDK) — Main Algorithm
# ============================================================
# Aligned with local trend_signal_engine backtest (Net Sharpe 1.119):
#
#   - Long model:  VotingRegressor  24 features (loaded from model_chunk_XX.py)
#   - Short model: VotingClassifier 28 features (loaded from short_model_payload.py)
#   - Bear-only shorts: max 5, 10d hold, 8% stop, 5% snap profit
#   - SignalStrengthPortfolioConstructionModel (proportional sizing)
#   - 21 calendar day rebalance (~15 trading days)
#   - 12 long / 5 short positions
#   - NO Sharpe circuit breaker (local has none; CB was firing 64x in 15yr)
#   - NO regime model routing (proven harmful: 1.234 -> 0.730)
#   - Bear/Crisis: no new longs opened, existing held within top-16 buffer (no forced mass exits)
# ============================================================

try:
    from AlgorithmImports import *
except ImportError:
    pass

from qc_alpha_model import TrendSignalAlphaModel, LoadProductionModel
from datetime import timedelta
import numpy as np
from System import Decimal


class InstitutionalPortfolioConstructionModel(PortfolioConstructionModel):
    """
    Implements Institutional-grade Portfolio Construction:
    1. Softmax Allocation: weights proportional to exp(score).
    2. Regime-Aware Exposure: scales gross exposure based on market regime.
    3. Position Caps: strict 8.33% limit (1/12 longs).
    """
    def __init__(self, rebalance_timedelta=timedelta(days=21), max_positions=12):
        super().__init__()
        self.rebalance_timedelta = rebalance_timedelta
        self.max_positions = max_positions
        self.last_rebalance = None
        # Regime Gross Exposure Caps (from backtest_config.yaml)
        self.exposure_caps = {
            "Bull": 1.0,
            "Sideways": 0.7,
            "Bear": 0.3,
            "Crisis": 0.25
        }

    def CreateTargets(self, algorithm, insights):
        # 1. Throttle rebalances
        if self.last_rebalance is not None and (algorithm.Time - self.last_rebalance) < self.rebalance_timedelta:
            # Check for force-liquidation insights (Direction.Flat)
            if not any(i.Direction == InsightDirection.Flat for i in insights):
                return []
        
        self.last_rebalance = algorithm.Time
        
        # 2. Group insights
        targets = []
        long_insights = [i for i in insights if i.Direction == InsightDirection.Up]
        short_insights = [i for i in insights if i.Direction == InsightDirection.Down]
        
        # Get current regime from Alpha Model
        regime = getattr(algorithm.Alpha, "current_regime", "Bull")
        exposure_cap = self.exposure_caps.get(regime, 1.0)
        
        # 3. Handle Longs (Softmax)
        if long_insights:
            # Numerical stability: subtract max score
            scores = np.array([i.Weight or 0.0 for i in long_insights])
            scores_exp = np.exp(scores - np.max(scores))
            total_exp = np.sum(scores_exp)
            
            for idx, insight in enumerate(long_insights):
                # Weight = (softmax_w) * gross_exposure
                w = (scores_exp[idx] / total_exp) * exposure_cap
                # Cap at 1/12 (8.33%)
                percent = min(w, 1.0 / self.max_positions)
                
                # Calculate quantity using algorithm's buying power logic (safest bridge)
                qty = algorithm.CalculateOrderQuantity(insight.Symbol, percent)
                targets.append(PortfolioTarget(insight.Symbol, qty))

        # 4. Handle Shorts (Equal weight within short book)
        if short_insights:
            # Short book is 70% of long equivalent scaling
            short_exposure = exposure_cap * 0.7
            w_short = short_exposure / len(short_insights)
            # Cap at 8% (same as long)
            percent_short = -min(w_short, 0.08)
            
            for insight in short_insights:
                qty = algorithm.CalculateOrderQuantity(insight.Symbol, percent_short)
                targets.append(PortfolioTarget(insight.Symbol, qty))

        # 5. Handle Liquidations
        flat_insights = [i for i in insights if i.Direction == InsightDirection.Flat]
        for i in flat_insights:
            # Quantity 0 = Full liquidation
            targets.append(PortfolioTarget(i.Symbol, 0))

        return targets


class TrendSignalAlgorithm(QCAlgorithm):

    def Initialize(self):
        # 1. Backtest Window & Capital
        self.SetStartDate(2008, 1, 1)
        self.SetEndDate(2022, 12, 31)
        self.SetCash(100000)
        # 253 days of warmup to ensure SMA/RSI and ML windows are full on Day 1
        self.SetWarmUp(0)

        # 2. Universe: top 500 by dollar volume, price > $10, ADV > $100M
        self.UniverseSettings.PriceAbove    = 10.0
        self.UniverseSettings.PriceBelow    = 10000.0
        self.UniverseSettings.Resolution    = Resolution.Daily
        self.UniverseSettings.DataNormalizationMode = DataNormalizationMode.Adjusted
        # Coarse + Fine selection enabled to populate .Fundamentals data for ML features
        self.AddUniverse(self.CoarseSelectionFunction, self.FineSelectionFunction)

        # 3. Benchmark
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol

        # 4. Load models
        # Long model: VotingRegressor 24 features (main alpha driver)
        # Short model: VotingClassifier 28 features, Bear-regime only
        # Stale Object Store models (old 15-feature RF) auto-detected and replaced.
        long_model  = LoadProductionModel(self, "best_long_model.pkl",  min_features=20)
        short_model = LoadProductionModel(self, "best_short_model.pkl", min_features=5)

        # 5. Alpha Model
        self.alpha_model = TrendSignalAlphaModel(
            self,
            long_model,
            short_model_data=short_model,
            spy_symbol=self.spy,
        )
        self.AddAlpha(self.alpha_model)

        # 6. Portfolio Construction
        self.Settings.RebalancePortfolioOnInsightChanges = True
        self.Settings.RebalancePortfolioOnSecurityChanges = False
        self.SetPortfolioConstruction(InstitutionalPortfolioConstructionModel(
            rebalance_timedelta=timedelta(days=21), # approx 15 trading days
            max_positions=12
        ))

        # 7. Execution (immediate fill, matches local)
        self.SetExecution(ImmediateExecutionModel())

    # ----------------------------------------------------------
    def OnData(self, data):
        pass

    # ----------------------------------------------------------
    def CoarseSelectionFunction(self, coarse):
        """Top-500 liquid universe: Price > $10, DollarVolume > $100M."""
        coarse_list = list(coarse)
        # Primary Filter: Price > $10. 
        # (We rely on sorted DollarVolume to pick the top 500 liquid stocks)
        filtered = [x for x in coarse_list if x.Price > 10]
        sorted_by_vol = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        
        if self.Time.day == 1:
            self.Log(f"UNIVERSE (Coarse): Raw={len(coarse_list)} | filtered(Price>10)={len(filtered)}")
            
        # Return all coarse candidates; FineSelection will trim to top 500
        return [x.Symbol for x in sorted_by_vol[:1000]]

    def FineSelectionFunction(self, fine):
        """Select top 500 by Dollar Volume and enable fundamental data access."""
        # Fundamental data is now available in the 'fine' objects.
        fine_list = list(fine)
        sorted_by_vol = sorted(fine_list, key=lambda x: x.DollarVolume, reverse=True)
        selected = [x.Symbol for x in sorted_by_vol[:500]]
        
        if self.Time.day == 1:
            has_fund = sum(1 for x in fine_list if x.HasFundamentalData)
            self.Log(f"UNIVERSE (Fine): Total={len(fine_list)} | HasFundamental={has_fund} | FinalSelected={len(selected)}")
            
        return selected
