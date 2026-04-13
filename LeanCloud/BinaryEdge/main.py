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
# ============================================================

try:
    from AlgorithmImports import *
except ImportError:
    pass

from qc_alpha_model import TrendSignalAlphaModel, LoadProductionModel
from datetime import timedelta
import numpy as np


class SignalStrengthPortfolioConstructionModel(EqualWeightingPortfolioConstructionModel):
    """
    Sizes positions proportional to ML signal score.
    Longs: equal base weight scaled by signal strength (max 12 positions).
    Shorts: 70% of long equivalent weight (matches ml_short_position_scale: 0.7).
    Total exposure capped at 95% buying power.
    """
    def __init__(self, rebalance_timedelta=timedelta(days=21),
                 max_long_positions=12, max_short_positions=5):
        super().__init__(rebalance_timedelta)
        self.max_long_positions  = max_long_positions
        self.max_short_positions = max_short_positions

    def DetermineTargetPercent(self, activeInsights):
        targets = {}

        long_insights  = [i for i in activeInsights if i.Direction == InsightDirection.Up]
        short_insights = [i for i in activeInsights if i.Direction == InsightDirection.Down]
        flat_insights  = [i for i in activeInsights if i.Direction == InsightDirection.Flat]

        # Base allocation per position
        long_base  = min(0.95 / float(self.max_long_positions), 1.0 / float(self.max_long_positions))
        short_base = long_base * 0.70   # ml_short_position_scale: 0.7

        raw = {}
        total_abs = 0.0

        for i in flat_insights:
            raw[i] = 0.0

        long_total_w  = sum(max(i.Weight or 0.01, 0.01) for i in long_insights)  or 1.0
        short_total_w = sum(max(i.Weight or 0.01, 0.01) for i in short_insights) or 1.0
        avg_long_w    = long_total_w  / max(len(long_insights),  1)
        avg_short_w   = short_total_w / max(len(short_insights), 1)

        for i in long_insights:
            w       = max(i.Weight or 0.01, 0.01)
            alloc   = min(long_base * (w / avg_long_w), long_base * 1.5)
            raw[i]  = alloc
            total_abs += alloc

        for i in short_insights:
            w       = max(i.Weight or 0.01, 0.01)
            alloc   = min(short_base * (w / avg_short_w), short_base * 1.5)
            raw[i]  = -alloc   # negative = short
            total_abs += alloc

        # Scale to 95% buying power
        if total_abs > 0.95:
            scale = 0.95 / total_abs
            for i in raw:
                raw[i] *= scale

        return raw


class TrendSignalAlgorithm(QCAlgorithm):

    def Initialize(self):
        # 1. Backtest Window & Capital
        self.SetStartDate(2008, 1, 1)
        self.SetEndDate(2022, 12, 31)
        self.SetCash(100000)
        self.SetWarmUp(0)

        # 2. Universe: top 500 by dollar volume, price > $10, ADV > $100M
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)

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
        # RebalancePortfolioOnInsightChanges=False: prevents micro-trades between rebalances.
        # Flat insights (exits) still trigger rebalance immediately.
        self.Settings.RebalancePortfolioOnInsightChanges = False
        self.Settings.RebalancePortfolioOnSecurityChanges = False
        self.SetPortfolioConstruction(SignalStrengthPortfolioConstructionModel(
            rebalance_timedelta=timedelta(days=21),
            max_long_positions=12,
            max_short_positions=5,
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
        filtered    = [x for x in coarse_list if x.Price > 10 and x.DollarVolume > 1e8]
        sorted_by_vol = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        selected    = [x.Symbol for x in sorted_by_vol[:500]]
        if self.Time.day == 1:
            self.Log(
                f"UNIVERSE: Total={len(coarse_list)} | "
                f"Filtered={len(filtered)} | Selected={len(selected)}"
            )
        return selected
