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
        self.SetEndDate(2024, 1, 1)
        self.SetCash(100000)
        
        # 2. Universe Selection: Top 300 Liquid US Names
        # For simplicity in Phase 1, we use manual tickers or a coarse universe
        # self.AddUniverse(self.CoarseSelectionFunction)
        self.tickers = [
    "NVDA", "TSLA", "MU", "MSFT", "SNDK", "AAPL", "AVGO", "AMZN", "GOOGL", "META",
    "PLTR", "AMD", "GOOG", "LITE", "NFLX", "ORCL", "INTC", "XOM", "JPM", "VRT",
    "CVX", "CRM", "LLY", "COHR", "WMT", "WDC", "COIN", "UNH", "LRCX", "AMAT",
    "APP", "BAC", "V", "GEV", "GS", "HOOD", "JNJ", "CAT", "NOW", "GLW", "INTU",
    "CSCO", "COST", "BKNG", "CRWD", "C", "PG", "MA", "ADBE", "ACN", "QCOM", "GE",
    "DELL", "ABBV", "WFC", "BA", "KLAC", "IBM", "HD", "COP", "STX", "UBER", "LIN",
    "VZ", "APH", "ABT", "NEM", "AXP", "KO", "ADI", "PANW", "TXN", "RTX", "BX", "T",
    "MRK", "MS", "CIEN", "LMT", "OXY", "DIS", "FCX", "TMUS", "SMCI", "PFE", "CMCSA",
    "VLO", "TER", "COF", "TMO", "MCD", "PEP", "SLB", "BLK", "SNPS", "NEE", "SPGI",
    "AMGN", "ETN", "DAL", "GILD", "HON", "CEG", "ADP", "BSX", "PYPL", "PM", "CVNA",
    "UAL", "NKE", "ANET", "DVN", "BMY", "CF", "SCHW", "SATS", "WBD", "PH", "EOG",
    "KKR", "TGT", "RCL", "SBUX", "DE", "FDX", "F", "CCL", "ISRG", "VST", "DHR",
    "UPS", "SYK", "WDAY", "APO", "TJX", "UNP", "ELV", "MPC", "MDT", "CME", "CDNS",
    "FANG", "MCK", "LYB", "WELL", "BKR", "MMM", "VRTX", "MO", "MCO", "LOW", "PGR",
    "CRH", "PWR", "XYZ", "PSX", "ROST", "DASH", "EQIX", "TTD", "HAL", "DUK", "NOC",
    "GM", "CVS", "FIX", "MDLZ", "CB", "EXPE", "AMT", "HCA", "CL", "HWM", "PNC",
    "MPWR", "HLT", "MCHP", "EQT", "USB", "ADSK", "CMG", "JCI", "TT", "ORLY", "ICE",
    "SHW", "LHX", "DDOG", "KR", "AZO", "DOW", "WM", "ABNB", "NRG", "SO", "NXPI",
    "MRSH", "ZTS", "ULTA", "AEP", "CSX", "MSI", "KEYS", "KMB", "MAR", "FITB", "LULU",
    "ROP", "MNST", "URI", "LYV", "DG", "CMI", "CI", "AJG", "FTNT", "NCLH", "EA",
    "ARES", "AKAM", "WMB", "TDG", "COR", "PLD", "TEL", "CTAS", "AXON", "KMI", "MRNA",
    "ECL", "EL", "EXC", "FICO", "REGN", "OKE", "HBAN", "AON", "HPE", "GD", "TRV",
    "EBAY", "ODFL", "KVUE", "EXE", "TFC", "ON", "PCG", "LUV", "BDX", "O", "PAYX",
    "CTSH", "CAH", "RF", "APD", "TTWO", "VMC", "BK", "GIS", "EMR", "DPZ", "CTRA",
    "KHC", "VRSK", "PSA", "DHI", "OMC", "FSLR", "TRGP", "CARR", "NSC", "EW", "ROK",
    "FISV", "FAST", "MLM", "HSY", "ITW", "XEL", "APA", "WAT", "JBL", "PCAR", "ALB",
    "AIG", "BBY", "HPQ", "DLTR", "DLR", "AME", "IBKR", "DRI", "SPG", "OTIS", "SRE",
    "MOS", "CTVA", "KEY", "CPRT", "IQV"
]
        for ticker in self.tickers:
            self.AddEquity(ticker, Resolution.Daily)
            
        # 3. SPY for Macro Regime & CAPM Benchmarking
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol
        
        # 4. Load ML Model (assuming path is local to the project folder)
        model = LoadProductionModel(self, "best_long_model.pkl")
        
        if model is None:
            self.Error("ML Model not found. Alpha generation will fail.")
            self.Quit()
            
        # 5. Connect Alpha Model
        self.AddAlpha(TrendSignalAlphaModel(model, self.spy))
        
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
        # Return top 200 liquid candidates
        return [x.Symbol for x in sorted_by_dollar_volume[:200]]
