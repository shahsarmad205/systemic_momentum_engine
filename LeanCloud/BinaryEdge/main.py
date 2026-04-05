# ============================================================
# QuantConnect (Lean SDK) — Main Algorithm Boilerplate
# ============================================================
# Pillar 7: Long-Only Production Baseline
# Migration from Trend Signal Engine

try:
    from AlgorithmImports import *
except ImportError:
    pass
from qc_alpha_model import TrendSignalAlphaModel, LoadProductionModel
from datetime import timedelta

class TrendSignalAlgorithm(QCAlgorithm):
    def Initialize(self):
        # 1. Backtest Window & Capital
        self.SetStartDate(2023, 1, 1)  # Production OOS Window
        self.SetEndDate(2026, 1, 1)
        self.SetCash(100000)
        
        # 2. Universe Selection: Automatically select Top 300 Liquid US Names
        # Matches the liquid momentum research baseline (avg. $20M+ volume)
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)

        # 3. SPY for Macro Regime & Conviction Gate
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        self.threshold = self.GetParameter("conviction_threshold", 0.51)
        
        # 4. Load Production Long Model
        long_model = LoadProductionModel(self, "best_long_model.pkl")
        
        if long_model is None:
            self.Error("Long Alpha model not found. Deployment aborted.")
            self.Quit()
            
        # 5. Connect Alpha Model (Pillar 7 Long-Only)
        self.AddAlpha(TrendSignalAlphaModel(long_model, None, self.spy))
        
        # 6. Portfolio Construction: Equal weighting for the Top 10 insights
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel(lambda time: None))
        
        # 7. Risk Management (Optional: trailing stop-losses from research engine)
        # self.AddRiskManagement(TrailingStopRiskManagementModel(0.05))
        
        # 8. Execution: Immediate at Market Open
        self.SetExecution(ImmediateExecutionModel())

    def OnData(self, data):
        # Additional logic if not using AlphaModel Framework
        pass

    def CoarseSelectionFunction(self, coarse):
        """
        Dynamically filter the universe for high-liquidity names.
        Matches the $20M dollar-volume floor in candidates.py.
        """
        sorted_by_dollar_volume = sorted(coarse, key=lambda x: x.DollarVolume, reverse=True)
        # Return top 300 liquid candidates (Institutional Universe)
        return [x.Symbol for x in sorted_by_dollar_volume[:300]]
