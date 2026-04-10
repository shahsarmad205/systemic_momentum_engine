# ============================================================
# QuantConnect (Lean SDK) — Main Algorithm
# ============================================================
# Synced with trend_signal_engine local backtest state.
#
# Key alignment with local system:
#   - Global model only (no regime routing — was proven to hurt Sharpe)
#   - InsightWeighting for signal-strength based sizing (not equal weight)
#   - 21 calendar days rebalance (~15 trading days, matches local config)
#   - 12 long positions (matches top_longs: 12)
#   - D3: Sharpe circuit breaker preserved
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

        # 2. Universe Selection — top 500 by dollar volume, price > $10, ADV > $100M
        self.UniverseSettings.Resolution = Resolution.Daily
        self.AddUniverse(self.CoarseSelectionFunction)

        # 3. Benchmark
        self.spy = self.AddEquity("SPY", Resolution.Daily).Symbol

        # 4. Load Global Model Only
        # NOTE: Regime-conditional routing (Bear/HighVol models) was disabled after
        # local experiments showed it degrades Sharpe from 1.234 → 0.730.
        # Bull model IC=0.049 << global IC=0.106 — do NOT re-enable without validated
        # regime-specific FEATURES (credit_spread × beta interactions), not just weights.
        long_model = LoadProductionModel(self, "best_long_model.pkl")

        # 5. Alpha Model
        self.alpha_model = TrendSignalAlphaModel(
            self,
            long_model,
            spy_symbol=self.spy,
        )
        self.AddAlpha(self.alpha_model)

        # 6. Portfolio Construction
        # InsightWeighting sizes positions proportional to insight magnitude (ML score),
        # matching local signal_strength position sizing. Rebalances every ~15 trading
        # days (21 calendar days).
        self.Settings.RebalancePortfolioOnInsightChanges = False
        self.Settings.RebalancePortfolioOnSecurityChanges = False
        self.SetPortfolioConstruction(InsightWeightingPortfolioConstructionModel(
            timedelta(days=21)   # ~15 trading days, matches local rebalance_every_trading_days: 15
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
                             f"reducing position count by {1 - self._sharpe_cb_scale:.0%}")
                    self.alpha_model.top_n = max(1, int(12 * self._sharpe_cb_scale))

                elif self._sharpe_cb_active and roll_sharpe > self._sharpe_cb_recovery:
                    self._sharpe_cb_active = False
                    self.Log(f"SHARPE_CB DEACTIVATE: 60d Sharpe={roll_sharpe:.2f} > {self._sharpe_cb_recovery:.2f} — "
                             "restoring 12 positions")
                    self.alpha_model.top_n = 12

    # ----------------------------------------------------------
    def CoarseSelectionFunction(self, coarse):
        """Top-500 liquid universe: Price > $10, DollarVolume > $100M."""
        coarse_list = list(coarse)
        filtered = [x for x in coarse_list if x.Price > 10 and x.DollarVolume > 1e8]
        sorted_by_vol = sorted(filtered, key=lambda x: x.DollarVolume, reverse=True)
        selected = [x.Symbol for x in sorted_by_vol[:500]]
        self.Log(f"UNIVERSE: Total={len(coarse_list)} | Filtered={len(filtered)} | Selected={len(selected)}")
        return selected
