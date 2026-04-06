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
        self.SetStartDate(2008, 1, 1)  # User requested: 2008-2022
        self.SetEndDate(2022, 12, 31)

        self.SetCash(100000)
        self.SetWarmUp(0)  # Instant Warm-up Pillar: Using manual History backfill
        
        # 2. Universe Selection: Institutional Liquid Universe
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)

        # 3. Benchmark for Macro Regime
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # 4. Load Models
        long_model = LoadProductionModel(self, "best_long_model.pkl")
        
        # 5. Connect Alpha Model
        self.AddAlpha(TrendSignalAlphaModel(self, long_model, None, self.spy))
        
        # 6. Portfolio Construction: Weekly rebalance only (matches 10-day holding period).
        # CRITICAL: RebalancePortfolioOnInsightChanges=False prevents the refresh insights
        # (emitted every 5 days to keep LEAN signals alive) from triggering a full
        # portfolio rebalance each time — that was causing 16,000+ orders and $16k fees.
        self.Settings.RebalancePortfolioOnInsightChanges = False
        self.Settings.RebalancePortfolioOnSecurityChanges = False
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel(
            timedelta(days=7)  # Rebalance once per week, matching ~5-day holding period
        ))
        
        # 8. Execution: Immediate at Market Open
        self.SetExecution(ImmediateExecutionModel())


    def OnData(self, data):
        # Additional logic if not using AlphaModel Framework
        pass

    def CoarseSelectionFunction(self, coarse):
        """
        Institutional Filter: Price > $10 and Dollar Volume > $100M.
        Mirrors the 'sp500' mode in backtest_config.yaml.
        """
        # Convert iterator to list once to prevent exhaustion
        coarse_list = list(coarse)
        
        # Filter for Price and Institutional Volume
        filtered = [x for x in coarse_list if x.Price > 10 and x.DollarVolume > 1e8]
        
        # Diagnostic Log: Monitor the universe funnel
        self.Log(f"UNIVERSE - Total Coarse: {len(coarse_list)} | Passed Filters: {len(filtered)}")
        
        # Sort by Volume and take Top 500 to match S&P 500 breadth
        sorted_by_vol = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sorted_by_vol[:500]]
