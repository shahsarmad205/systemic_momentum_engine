# ============================================================
# QuantConnect (Lean SDK) — Main Algorithm
# ============================================================
# Synced with trend_signal_engine Phase A-D improvements.
# Phases implemented:
#   A1: Cross-sectional z-scores in qc_alpha_model.py
#   A3: 24-feature alignment with best_long_model.pkl
#   C2: Regime-conditional routing (Bear → xgb_regime_bear.pkl,
#       Crisis/HighVol → xgb_regime_highvol.pkl)
#   D1: Continuous regime score → linear gross cap scaling
#   D3: Sharpe circuit breaker (60d rolling Sharpe)
# ============================================================

try:
    from AlgorithmImports import *
except ImportError:
    pass

from qc_alpha_model import TrendSignalAlphaModel, LoadProductionModel
from datetime import timedelta
import numpy as np


class TrendSignalAlgorithm(QCAlgorithm):

    def Initialize(self):
        # 1. Backtest Window & Capital
        self.SetStartDate(2008, 1, 1)
        self.SetEndDate(2022, 12, 31)
        self.SetCash(100000)
        self.SetWarmUp(0)

        # 2. Universe Selection
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)

        # 3. Benchmark
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol

        # 4. Load Models
        long_model    = LoadProductionModel(self, "best_long_model.pkl")
        bear_model    = LoadProductionModel(self, "xgb_regime_bear.pkl")
        highvol_model = LoadProductionModel(self, "xgb_regime_highvol.pkl")

        # 5. Alpha Model
        self.alpha_model = TrendSignalAlphaModel(
            self,
            long_model,
            short_model=None,       # long-only for QC deployment
            spy_symbol=self.spy,
            bear_model_data=bear_model,
            highvol_model_data=highvol_model,
        )
        self.AddAlpha(self.alpha_model)

        # 6. Portfolio Construction: weekly rebalance
        self.Settings.RebalancePortfolioOnInsightChanges = False
        self.Settings.RebalancePortfolioOnSecurityChanges = False
        self.SetPortfolioConstruction(EqualWeightingPortfolioConstructionModel(
            timedelta(days=7)
        ))

        # 7. Execution
        self.SetExecution(ImmediateExecutionModel())

        # D3: Sharpe circuit breaker state
        self._daily_equity: list = []
        self._sharpe_cb_active = False
        self._sharpe_cb_window = 60
        self._sharpe_cb_threshold = 0.0
        self._sharpe_cb_recovery = 0.3
        self._sharpe_cb_scale = 0.5

    # ----------------------------------------------------------
    def OnData(self, data):
        pass

    def OnEndOfDay(self, symbol):
        """D3: Track daily equity and manage Sharpe circuit breaker."""
        if symbol != self.spy:
            return

        equity = float(self.Portfolio.TotalPortfolioValue)
        self._daily_equity.append(equity)

        if len(self._daily_equity) > self._sharpe_cb_window + 1:
            self._daily_equity = self._daily_equity[-(self._sharpe_cb_window + 1):]

        if len(self._daily_equity) >= self._sharpe_cb_window:
            eq_arr = np.array(self._daily_equity, dtype=float)
            rets = np.diff(eq_arr) / np.where(eq_arr[:-1] > 0, eq_arr[:-1], 1.0)
            if len(rets) >= 20:
                roll_sharpe = float(np.mean(rets) / max(np.std(rets, ddof=1), 1e-10) * np.sqrt(252))

                if not self._sharpe_cb_active and roll_sharpe < self._sharpe_cb_threshold:
                    self._sharpe_cb_active = True
                    self.Log(f"SHARPE_CB ACTIVATE: 60d Sharpe={roll_sharpe:.2f} < {self._sharpe_cb_threshold:.2f} — "
                             f"reducing gross cap by {self._sharpe_cb_scale:.0%}")
                    # Apply CB: reduce top_n in alpha model proportionally
                    self.alpha_model.top_n = max(1, int(10 * self._sharpe_cb_scale))

                elif self._sharpe_cb_active and roll_sharpe > self._sharpe_cb_recovery:
                    self._sharpe_cb_active = False
                    self.Log(f"SHARPE_CB DEACTIVATE: 60d Sharpe={roll_sharpe:.2f} > {self._sharpe_cb_recovery:.2f} — "
                             "restoring normal position count")
                    self.alpha_model.top_n = 10

    # ----------------------------------------------------------
    def CoarseSelectionFunction(self, coarse):
        """Top-500 liquid universe: Price > $10, DollarVolume > $100M."""
        coarse_list = list(coarse)
        filtered = [x for x in coarse_list if x.Price > 10 and x.DollarVolume > 1e8]
        self.Log(f"UNIVERSE: Total={len(coarse_list)} | Filtered={len(filtered)}")
        sorted_by_vol = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        return [x.Symbol for x in sorted_by_vol[:500]]
