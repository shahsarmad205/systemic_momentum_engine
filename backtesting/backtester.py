"""
Backtester — Portfolio Simulation Orchestrator
=================================================
Coordinates data loading, signal generation, regime detection,
and a day-by-day portfolio simulation with realistic constraints.

Architecture:
    Phase 1  — Download OHLCV data, generate vectorised signals
    Phase 2  — Detect market regimes (SPY + VIX)
    Phase 3  — Day-by-day portfolio simulation
    Phase 4  — Metric computation
"""

from __future__ import annotations

import datetime
import logging
import math
from collections import defaultdict
from datetime import timedelta
from typing import Optional

import numpy as np
import pandas as pd

logging.getLogger("matplotlib").setLevel(logging.WARNING)
from pathlib import Path

import matplotlib.pyplot as plt

from .config import BacktestConfig
from .execution import ExecutionEngine
from .portfolio import Portfolio, Position
from .regime import MarketRegimeAgent
from .result import BacktestResult

try:
    from execution.cost_model import TransactionCostModel
except ImportError:
    TransactionCostModel = None

try:
    from execution.market_impact import MarketImpactModel, MarketImpactParams
except ImportError:
    MarketImpactModel = None  # type: ignore[misc,assignment]
    MarketImpactParams = None  # type: ignore[misc,assignment]
from utils.market_data import get_ohlcv
from utils.quant_utils import apply_sector_adjustment, compute_sector_aggregates
from utils.quant_utils import SECTOR_MAP

from .analytics import annualized_vol as _annualized_vol_fn, build_returns_matrix as _build_returns_matrix_fn
from .metrics import compute_all_metrics, compute_capm_metrics
from .signals import EXIT_BUFFER_DAYS, HISTORY_BUFFER_DAYS, SignalEngine

try:
    from agents.weight_learning_agent.feature_builder import (
        sector_relative_features_by_ticker,
        vol_rank_features_by_ticker,
    )
except ImportError:  # pragma: no cover
    sector_relative_features_by_ticker = None  # type: ignore[misc,assignment]
    vol_rank_features_by_ticker = None  # type: ignore[misc,assignment]
from backtesting.candidates import build_ranked_candidates
from backtesting.cross_sectional import build_cross_sectional_candidates
from backtesting.portfolio_construction import (
    construct_hybrid_fmc_score_weights,
    compute_active_share,
    compute_factor_imbalance,
)

try:
    from backtesting.regime_multipliers import (
        get_multiplier,
        load_regime_multipliers,
        signal_confidence_multiplier_for_regime as _scm_for_regime,
        threshold_aggressiveness_for_regime as _thresh_agg_for_regime,
    )
except ImportError:  # pragma: no cover
    load_regime_multipliers = None  # type: ignore[misc,assignment]
    get_multiplier = None  # type: ignore[misc,assignment]
    _scm_for_regime = None  # type: ignore[misc,assignment]
    _thresh_agg_for_regime = None  # type: ignore[misc,assignment]
from . import position_sizing

try:
    from options.black_scholes import (
        bs_greeks,
        bs_price,
        implied_vol_from_historical,
    )
except ImportError:
    bs_price = bs_greeks = implied_vol_from_historical = None

logger = logging.getLogger(__name__)

try:
    from research.factor_neutralization import (
        FactorNeutralizer,
        write_exposure_diagnostics,
        write_neutralization_report,
    )
except ImportError:
    FactorNeutralizer = None
    write_exposure_diagnostics = None
    write_neutralization_report = None


# BacktestResult is defined in result.py and imported above.
# Kept as a module-level name for backward compatibility with any code
# that does: from backtesting.backtester import BacktestResult


# ------------------------------------------------------------------
# Main Backtester
# ------------------------------------------------------------------

class Backtester:
    """
    Institutional-grade portfolio backtester.

    Usage:
        cfg = load_config("backtest_config.yaml")
        bt  = Backtester(cfg)
        res = bt.run()                         # default tickers
        res = bt.run(["AAPL", "MSFT", "NVDA"]) # custom tickers
    """

    def __init__(self, config: BacktestConfig):
        self.config = config
        self.portfolio = Portfolio(config.initial_capital, config.max_positions)
        self.ticker_trade_counts: dict[str, int] = {}
        # VIX-scaled cost scalar for the current simulation day (updated daily in the loop).
        self._cost_scalar_today: float = 1.0
        # Regime-specific confidence multipliers (safe defaults to 1.0 if loader unavailable).
        self.regime_multipliers: dict[str, float] = {
            "Bull": 1.0,
            "Bear": 1.0,
            "Cautious": 1.0,    # State Street (Ung 2025) "Cautious Decline" — 5th regime
            "Sideways": 1.0,
            "Crisis": 1.0,
        }
        if load_regime_multipliers is not None:
            try:
                self.regime_multipliers = load_regime_multipliers()
                logger.info("Regime multipliers loaded: %s", self.regime_multipliers)
            except Exception as exc:
                logger.warning("Failed to load regime multipliers (%s). Using defaults.", exc)
        # When True: use TransactionCostModel (execution_costs_* bps) and ExecutionEngine(commission=0).
        # When False: use config.slippage_bps + config.commission_per_trade (e.g. for cost sensitivity).
        self._execution_costs_enabled = getattr(config, "execution_costs_enabled", False)
        if self._execution_costs_enabled and TransactionCostModel is not None:
            self.cost_model = TransactionCostModel(
                commission_bps=getattr(config, "execution_costs_commission_bps", 2.0),
                spread_bps=getattr(config, "execution_costs_spread_bps", 2.0),
                slippage_bps=getattr(config, "execution_costs_slippage_bps", 1.0),
            )
            slippage_bps = getattr(config, "execution_costs_slippage_bps", 1.0)
            self.execution = ExecutionEngine(slippage_bps=slippage_bps, commission=0.0)
        else:
            self.cost_model = None
            self.execution = ExecutionEngine(config.slippage_bps, config.commission_per_trade)
        self.regime_agent = MarketRegimeAgent()

        # Almgren-Chriss market impact model (replaces ad-hoc k_bps * sqrt formula)
        self._market_impact_model: object = None
        if getattr(config, "market_impact_enabled", True) and MarketImpactModel is not None:
            _mi_params = MarketImpactParams(
                eta=float(getattr(config, "mi_eta", 0.142)),
                alpha=float(getattr(config, "mi_alpha", 0.314)),
                gamma=float(getattr(config, "mi_gamma", 0.6)),
                # Spread is per-leg; spread_bps from config is the full spread → halve it
                spread_half_bps=float(getattr(config, "execution_costs_spread_bps", 5.0)) / 2.0,
                commission_bps=float(getattr(config, "execution_costs_commission_bps", 2.0)),
            )
            _mi_adv = float(getattr(config, "mi_adv_usd_default", 50_000_000.0))
            self._market_impact_model = MarketImpactModel(adv_usd=_mi_adv, params=_mi_params)
            logger.info(
                "Almgren-Chriss market impact enabled  eta=%.3f  alpha=%.3f  gamma=%.2f",
                _mi_params.eta, _mi_params.alpha, _mi_params.gamma,
            )

        # Circuit breaker state
        self._equity_peak: float = config.initial_capital
        self._trading_halted: bool = False
        self._circuit_breaker_log: list[dict] = []
        # D3: Rolling Sharpe circuit breaker state
        self._sharpe_cb_active: bool = False
        self._daily_equity_returns: list[float] = []   # running equity values for rolling Sharpe CB

        learned_weights = None
        regime_weights = None
        regime_series = None
        if config.signal_mode == "learned" and config.learned_weights_path:
            from agents.weight_learning_agent import LearnedWeights
            from agents.weight_learning_agent.weight_model import load_regime_weights

            regime_weights, active_features = load_regime_weights(config.learned_weights_path)
            if regime_weights is not None:
                print(f"  Loaded regime-specific weights from {config.learned_weights_path}")
                print(f"    Regime models: {list(regime_weights.keys())}")
                from agents.weight_learning_agent.regime_detection import detect_regimes
                raw = detect_regimes(config.start_date, config.end_date)
                def _to_key(r):
                    if r == "HighVol":
                        return "high_vol"
                    return r.lower() if isinstance(r, str) else "normal"
                regime_series = raw.map(_to_key).reindex(raw.index).fillna("normal")
                counts = regime_series.value_counts()
                print(f"    Detected regimes (backtest window): {counts.to_dict()}")
                for reg in regime_weights:
                    w = regime_weights[reg]
                    print(f"    [{reg}] n_samples={getattr(w, 'n_samples', 0):,}  IC={getattr(w, 'ic', 0):.4f}")
            else:
                learned_weights = LearnedWeights.load(config.learned_weights_path)
                print(f"  Loaded learned weights from {config.learned_weights_path}")
                print(f"    w_trend={learned_weights.w_trend:.4f}  "
                      f"w_regional={learned_weights.w_regional:.4f}  "
                      f"w_global={learned_weights.w_global:.4f}  "
                      f"w_social={learned_weights.w_social:.4f}  "
                      f"intercept={learned_weights.intercept:.6f}")

        self.signal_engine = SignalEngine(
            weights=config.signal_weights,
            learned_weights=learned_weights,
            regime_weights=regime_weights,
            regime_series=regime_series,
            signal_smoothing_enabled=getattr(config, "signal_smoothing_enabled", True),
            signal_smoothing_span=int(getattr(config, "signal_smoothing_span", 5)),
        )
        # Provide full BacktestConfig to SignalEngine so it can respect
        # user-tunable thresholds (min_signal_strength, dynamic thresholds, etc.).
        self.signal_engine.config = config

    # ==============================================================
    # Public API
    # ==============================================================

    # Default fallback tickers when none are configured
    _FALLBACK_TICKERS = [
        "AAPL", "MSFT", "NVDA", "AMZN", "GOOG", "META", "TSLA",
        "JPM", "V", "JNJ", "WMT", "PG", "XOM", "UNH", "HD",
        "BAC", "MA", "ABBV", "PFE", "KO", "PEP", "MRK", "AVGO",
        "COST", "TMO", "CSCO", "ACN", "MCD", "NKE", "ADBE",
        "CRM", "AMD", "INTC", "ORCL", "IBM", "GS", "CAT", "BA",
        "SPY", "QQQ", "IWM", "DIA", "XLK", "XLF", "VTI",
    ]

    def run(
        self,
        tickers: list[str] | None = None,
        price_data: dict[str, pd.DataFrame] | None = None,
        signal_data: dict[str, pd.DataFrame] | None = None,
    ) -> BacktestResult:
        tickers = tickers or self.config.tickers
        if not tickers:
            try:
                from main import TICKERS
                tickers = list(TICKERS)
            except Exception:
                tickers = list(self._FALLBACK_TICKERS)

        # Phase 1 — data & signals
        if price_data is not None and signal_data is not None:
            print("Phase 1: Reusing pre-loaded data & signals…")
        else:
            mode = self.config.signal_mode
            mode_labels = {
                "price": "price-only (trend + volatility)",
                "full": "full (all agents)",
                "learned": "learned weights (data-driven)",
            }
            mode_label = mode_labels.get(mode, mode)
            print(f"Phase 1: Downloading data & generating signals [{mode_label}]…")
            price_data, signal_data = self.prepare_data(tickers)

        # Phase 1b — sector-level adjustment (momentum, volatility, sentiment)
        if self.config.sector_adjustment_enabled and price_data and signal_data:
            print("\nPhase 1b: Sector aggregates & signal adjustment…")
            sector_aggregates = compute_sector_aggregates(
                price_data,
                sector_map=SECTOR_MAP,
                sentiment_by_ticker=None,
            )
            apply_sector_adjustment(
                signal_data,
                sector_aggregates,
                SECTOR_MAP,
                momentum_weight=self.config.sector_momentum_weight,
                volatility_weight=self.config.sector_volatility_weight,
                sentiment_weight=self.config.sector_sentiment_weight,
            )
            print(f"  Applied sector context (momentum={self.config.sector_momentum_weight}, "
                  f"vol={self.config.sector_volatility_weight}, "
                  f"sentiment={self.config.sector_sentiment_weight})")

        # Phase 2 — regime detection
        regime_data: dict[pd.Timestamp, str] = {}
        regime_score_data: dict[pd.Timestamp, float] = {}
        if self.config.regime_enabled:
            print("\nPhase 2: Detecting market regimes (SPY + VIX)…")
            confirmation_days = int(getattr(self.config, "regime_confirmation_days", 3))
            regime_data = self.regime_agent.detect_regimes(
                self.config.start_date, self.config.end_date,
                confirmation_days=confirmation_days,
            )
            regime_counts = defaultdict(int)
            for r in regime_data.values():
                regime_counts[r] += 1
            print(f"  Regimes: {dict(regime_counts)}")
            # D1: compute continuous regime scores for linear gross exposure scaling
            if bool(getattr(self.config, "regime_continuous_score_enabled", True)):
                regime_score_data = self.regime_agent.detect_regime_scores(
                    self.config.start_date, self.config.end_date,
                )
                print(f"  Regime scores: {len(regime_score_data)} days computed")
        else:
            print("\nPhase 2: Regime detection disabled.")

        # Phase 3 — simulation
        print("\nPhase 3: Running day-by-day portfolio simulation…")
        _use_opt = bool(getattr(self.config, "use_continuous_optimization", False))
        if _use_opt:
            print("  [continuous-optimization mode]")
            trades, daily_equity = self._simulate_continuous(
                price_data, signal_data, regime_data
            )
        else:
            self._simulate(price_data, signal_data, regime_data, regime_score_data)
            trades = pd.DataFrame(self.portfolio.trade_log)
            daily_equity = pd.DataFrame(self.portfolio.equity_history)

        if not trades.empty:
            trades.sort_values("entry_date", inplace=True)
            trades.reset_index(drop=True, inplace=True)

        # Phase 4 — metrics
        print("Phase 4: Computing metrics…\n")
        metrics = compute_all_metrics(trades, daily_equity, self.config)
        metrics["crisis_entries_blocked_days_1_3"] = int(
            getattr(self, "_crisis_entries_blocked_days_1_3", 0)
        )

        # Phase 4b — trade-level diagnostics and P&L decomposition
        self._run_trade_diagnostics(trades, daily_equity)

        result = BacktestResult(trades, daily_equity, metrics, self.config)
        result.price_data = price_data
        result.signal_data = signal_data
        result.regime_data = regime_data

        # CAPM metrics (portfolio beta, alpha vs SPY, Treynor, information ratio)
        risk_free_rate = 0.04
        spy_df = price_data.get("SPY")
        if spy_df is None or spy_df.empty:
            try:
                spy_df = get_ohlcv(
                    "SPY",
                    self.config.start_date,
                    self.config.end_date,
                    cache_dir=getattr(self.config, "cache_dir", "data/cache/ohlcv"),
                    use_cache=getattr(self.config, "cache_ohlcv", True),
                )
            except Exception:
                spy_df = None
        # Active Share vs equal-weight benchmark (S&P Global, Innes et al. p.6-7).
        # Computed from the final portfolio snapshot (last day of backtest).
        # Institutional target: 0.60–0.80.
        try:
            if not trades.empty and "ticker" in trades.columns:
                final_positions = (
                    trades[trades["exit_date"].isna() | (pd.to_datetime(trades["exit_date"]) > pd.Timestamp(self.config.end_date))]
                    if "exit_date" in trades.columns else trades
                )
                # Fallback: use all trades, weight by position_size
                if final_positions.empty:
                    final_positions = trades
                if "position_size" in final_positions.columns and not final_positions.empty:
                    pos_sizes = final_positions.groupby("ticker")["position_size"].sum()
                    total_pos = pos_sizes.sum()
                    if total_pos > 0:
                        port_weights = (pos_sizes / total_pos).to_dict()
                        # Equal-weight benchmark over all tickers in the universe
                        all_tickers_universe = list(price_data.keys())
                        n_bench = max(len(all_tickers_universe), 1)
                        bench_weights = {t: 1.0 / n_bench for t in all_tickers_universe}
                        active_share = compute_active_share(port_weights, bench_weights)
                        result.metrics["active_share"] = round(active_share, 4)
                        _as_flag = " ✓ institutional" if 0.60 <= active_share <= 0.80 else (" ↑ high concentration" if active_share > 0.80 else " ↓ index-hugging")
                        print(f"  Active Share (vs equal-weight): {active_share:.2%}{_as_flag}")
                        print(f"    (Institutional target: 0.60–0.80)")
                        print()
        except Exception as _as_exc:
            logger.debug("Active share computation failed: %s", _as_exc)

        if spy_df is not None and not daily_equity.empty and "Close" in spy_df.columns:
            spy_ret = spy_df["Close"].pct_change()
            capm = compute_capm_metrics(daily_equity, spy_ret, risk_free_rate=risk_free_rate)
            result.metrics.update(capm)
            print("  CAPM (vs SPY):")
            print(f"    Portfolio Beta     : {capm['portfolio_beta']:.3f}" if pd.notna(capm.get("portfolio_beta")) else "    Portfolio Beta     : —")
            a = capm.get("portfolio_alpha_annual")
            print(f"    Portfolio Alpha    : {a:.2%} (ann.)" if a is not None and pd.notna(a) else "    Portfolio Alpha    : —")
            t = capm.get("treynor_ratio")
            print(f"    Treynor Ratio      : {t:.3f}" if t is not None and pd.notna(t) else "    Treynor Ratio      : —")
            ir = capm.get("information_ratio")
            print(f"    Information Ratio : {ir:.3f}" if ir is not None and pd.notna(ir) else "    Information Ratio : —")
            print()

        # Risk Metrics (VaR)
        v95 = result.metrics.get("var_95_1d")
        v99 = result.metrics.get("var_99_1d")
        cvar = result.metrics.get("cvar_95")
        v95_5d = result.metrics.get("var_95_5d")
        breach_count = result.metrics.get("var_breach_count_95", 0)
        breach_rate = result.metrics.get("var_breach_rate_95")
        if v95 is not None and not (isinstance(v95, float) and np.isnan(v95)):
            print("  Risk Metrics (VaR):")
            print(f"    Historical VaR (95%, 1d) : {v95:.2%}")
            if v99 is not None and not (isinstance(v99, float) and np.isnan(v99)):
                print(f"    Historical VaR (99%, 1d) : {v99:.2%}")
            if cvar is not None and not (isinstance(cvar, float) and np.isnan(cvar)):
                print(f"    CVaR / Expected Shortfall: {cvar:.2%}")
            if v95_5d is not None and not (isinstance(v95_5d, float) and np.isnan(v95_5d)):
                print(f"    VaR (95%, 5d scaled)     : {v95_5d:.2%}")
            print(f"    VaR breaches (95%)       : {breach_count} days (expected ~5% of trading days)")
            if breach_rate is not None and not (isinstance(breach_rate, float) and np.isnan(breach_rate)):
                print(f"    VaR breach rate          : {breach_rate:.1%} (should be ~5% if model correct)")
            print()

        # Position Sizing Comparison: run with equal and kelly to report both Sharpes
        orig_sizing = getattr(self.config, "position_sizing", "equal")
        orig_method = getattr(self.config, "position_sizing_method", None)
        equal_sharpe = kelly_sharpe = None
        kelly_win_rate_used = kelly_fraction_used = kelly_avg_position_pct = None
        try:
            # Run with equal
            self.config.position_sizing = "equal"
            self.config.position_sizing_method = "equal"
            self.portfolio = Portfolio(self.config.initial_capital, self.config.max_positions)
            self._simulate(price_data, signal_data, regime_data, regime_score_data=regime_score_data)
            eq_trades = pd.DataFrame(self.portfolio.trade_log)
            eq_equity = pd.DataFrame(self.portfolio.equity_history)
            if not eq_trades.empty:
                eq_trades.sort_values("entry_date", inplace=True)
            equal_metrics = compute_all_metrics(eq_trades, eq_equity, self.config)
            equal_sharpe = equal_metrics.get("sharpe_ratio")

            # Run with kelly
            self.config.position_sizing = "kelly"
            self.config.position_sizing_method = "kelly"
            self.portfolio = Portfolio(self.config.initial_capital, self.config.max_positions)
            self._simulate(price_data, signal_data, regime_data, regime_score_data=regime_score_data)
            kelly_trades = pd.DataFrame(self.portfolio.trade_log)
            kelly_equity = pd.DataFrame(self.portfolio.equity_history)
            if not kelly_trades.empty:
                kelly_trades.sort_values("entry_date", inplace=True)
            kelly_metrics = compute_all_metrics(kelly_trades, kelly_equity, self.config)
            kelly_sharpe = kelly_metrics.get("sharpe_ratio")
            kelly_fraction_used = getattr(self.config, "kelly_fraction", 0.5)
            kelly_win_rate_used = getattr(self, "_last_kelly_win_rate", None)
            kelly_avg_position_pct = getattr(self, "_last_kelly_position_pct", None)
            if kelly_avg_position_pct is None and not kelly_trades.empty and "position_size" in kelly_trades.columns:
                initial = self.config.initial_capital
                kelly_avg_position_pct = float(100.0 * kelly_trades["position_size"].mean() / initial) if initial else None
        finally:
            self.config.position_sizing = orig_sizing
            self.config.position_sizing_method = orig_method or orig_sizing

        # Attach comparison to result and print
        result.position_sizing_comparison = {
            "equal_weight_sharpe": equal_sharpe,
            "kelly_sharpe": kelly_sharpe,
            "kelly_win_rate_used": kelly_win_rate_used,
            "kelly_fraction": kelly_fraction_used,
            "kelly_avg_position_pct": kelly_avg_position_pct,
        }
        print("  Position Sizing Comparison:")
        print(f"    Equal-weight Sharpe  : {equal_sharpe:.3f}" if equal_sharpe is not None else "    Equal-weight Sharpe  : —")
        print(f"    Kelly (half) Sharpe  : {kelly_sharpe:.3f}" if kelly_sharpe is not None else "    Kelly (half) Sharpe  : —")
        wr = kelly_win_rate_used
        print(f"    Kelly win_rate used  : {wr:.1%}" if wr is not None else "    Kelly win_rate used  : —")
        print(f"    Kelly fraction       : {kelly_fraction_used}" if kelly_fraction_used is not None else "    Kelly fraction       : 0.5")
        ap = kelly_avg_position_pct
        print(f"    Kelly avg position   : {ap:.1f}% of equity" if ap is not None else "    Kelly avg position   : —")
        print()

        if getattr(self, "_factor_diagnostics", []) and write_neutralization_report is not None:
            report_path = getattr(
                self.config, "factor_neutralization_report_path",
                "output/research/factor_neutralization_report.json",
            )
            write_neutralization_report(
                self._factor_diagnostics,
                report_path,
                ic_after=metrics.get("information_coefficient"),
            )

        # IC (Information Coefficient) diagnostics
        # Grinold & Kahn: IR ≈ IC × sqrt(Breadth).  We print IC per signal
        # computed from completed trades (signal at entry vs realized return).
        self._print_ic_diagnostics(result.trades)

        return result

    # ==============================================================
    # IC (Information Coefficient) diagnostics
    # ==============================================================

    def _print_ic_diagnostics(self, trades: pd.DataFrame) -> None:
        """
        Compute and print IC (Information Coefficient) from completed trades.

        Pooled Spearman rank IC between adjusted_score at entry and realized
        holding-period return.  This is the simplest IC estimator and requires
        no minimum cross-section size per date.

        Interpretation (Grinold & Kahn 2000 benchmarks):
            IC  > 0.05  — usable signal (exploitable edge)
            IC  > 0.10  — good signal (institutional grade)
            IC  > 0.15  — exceptional (AQR-grade)
            |t| > 1.96  — statistically significant at 5% level
        """
        try:
            from scipy.stats import spearmanr
        except ImportError:
            return

        if trades is None or trades.empty:
            return
        if not {"adjusted_score", "pnl", "position_size"}.issubset(trades.columns):
            return

        df = trades.dropna(subset=["adjusted_score", "pnl", "position_size"]).copy()
        if len(df) < 10:
            return

        # Realized return per trade (direction-signed pnl / position size)
        pos_size = df["position_size"].astype(float).clip(lower=1.0)
        realized_ret = df["pnl"].astype(float) / pos_size
        signal_score = df["adjusted_score"].astype(float)

        valid = realized_ret.notna() & signal_score.notna() & np.isfinite(realized_ret) & np.isfinite(signal_score)
        n = int(valid.sum())
        if n < 10:
            return

        try:
            import warnings as _warnings
            with _warnings.catch_warnings():
                _warnings.simplefilter("ignore")
                ic, p_value = spearmanr(signal_score[valid].values, realized_ret[valid].values)
            if not np.isfinite(ic):
                return

            ic_std = float(np.std(
                [spearmanr(signal_score[valid].sample(max(n // 2, 5), replace=True, random_state=i).values,
                           realized_ret[valid].sample(max(n // 2, 5), replace=True, random_state=i).values)[0]
                 for i in range(50) if np.isfinite(
                     spearmanr(signal_score[valid].sample(max(n // 2, 5), replace=True, random_state=i).values,
                               realized_ret[valid].sample(max(n // 2, 5), replace=True, random_state=i).values)[0]
                 )],
                ddof=1,
            ))
            icir = ic / ic_std if ic_std > 0 else 0.0
            t_stat = ic / (1.0 / np.sqrt(n - 2 + 1e-9)) if n > 2 else 0.0
            sig = "*" if abs(t_stat) > 1.96 else " "

            _bench = ""
            if ic > 0.15:
                _bench = "(exceptional — AQR grade)"
            elif ic > 0.10:
                _bench = "(good — institutional grade)"
            elif ic > 0.05:
                _bench = "(usable signal)"
            elif ic > -0.05:
                _bench = "(weak — near noise)"
            elif ic > -0.10:
                _bench = "(anti-predictive — review signal direction)"
            else:
                _bench = "(strongly anti-predictive — signal likely inverted)"

            print("  IC Diagnostics (Grinold & Kahn):")
            print(f"    Signal: adjusted_score  IC={ic:+.4f}  ICIR={icir:+.3f}  "
                  f"t={t_stat:+.2f}{sig}  n={n}  {_bench}")
            print()
        except Exception as exc:
            logger.debug("IC diagnostics failed: %s", exc)

    # ==============================================================
    # Diagnostics — trade-level P&L decomposition
    # ==============================================================

    def _run_trade_diagnostics(self, trades: pd.DataFrame, daily_equity: pd.DataFrame) -> None:
        """
        Build a trade-level diagnostic report to understand where P&L comes from:
          - Long vs short P&L
          - Cost drag
          - Approximate short exposure
        """
        if trades is None or trades.empty:
            return

        df = trades.copy()
        # Ensure datetime
        df["exit_date"] = pd.to_datetime(df["exit_date"])
        df["entry_date"] = pd.to_datetime(df["entry_date"])

        # Position direction label
        df["position_direction"] = np.where(df["direction"] > 0, "long", "short")
        df["realized_pnl"] = df["pnl"].astype(float)
        df["transaction_cost"] = df["total_cost"].astype(float)
        df["signal_score"] = df.get("adjusted_score", np.nan).astype(float)

        # 1) Per-trade CSV (exit-date based)
        out_dir = Path("output/diagnostics")
        out_dir.mkdir(parents=True, exist_ok=True)
        diag_path = out_dir / "trade_diagnostics.csv"
        df_out = df[
            [
                "exit_date",
                "ticker",
                "signal_score",
                "position_direction",
                "realized_pnl",
                "transaction_cost",
            ]
        ].sort_values("exit_date")
        df_out.to_csv(diag_path, index=False)

        # 2) Aggregate diagnostics
        longs = df[df["direction"] > 0]
        shorts = df[df["direction"] < 0]

        avg_pnl_long = float(longs["realized_pnl"].mean()) if not longs.empty else float("nan")
        avg_pnl_short = float(shorts["realized_pnl"].mean()) if not shorts.empty else float("nan")

        # Average transaction cost as % of trade P&L (by absolute P&L).
        # Use a $1 floor so near-zero-P&L trades don't produce absurd ratios (e.g. $6 cost
        # on a $0.001 P&L = 600,000%). Cap each ratio at 5x (500%) before averaging.
        pnl_floor = df["realized_pnl"].abs().clip(lower=1.0)
        cost_pct = (df["transaction_cost"] / pnl_floor).clip(upper=5.0)
        avg_cost_pct = float(cost_pct.mean())

        # Approximate % of portfolio that is short on average: fraction of trades that are shorts
        frac_short_trades = float((df["direction"] < 0).mean())

        total_long_pnl = float(longs["realized_pnl"].sum()) if not longs.empty else 0.0
        total_short_pnl = float(shorts["realized_pnl"].sum()) if not shorts.empty else 0.0
        total_cost = float(df["transaction_cost"].sum())

        print("\nTrade-level diagnostics:")
        print(f"  Avg P&L per long trade    : {avg_pnl_long: .2f}")
        print(f"  Avg P&L per short trade   : {avg_pnl_short: .2f}")
        print(f"  Avg transaction cost / |P&L|: {avg_cost_pct: .2%}")
        print(f"  Fraction of trades that are shorts: {frac_short_trades: .2%}")
        print("  P&L contribution:")
        print(f"    Longs  : {total_long_pnl: .2f}")
        print(f"    Shorts : {total_short_pnl: .2f}")
        print(f"    Costs  : {-total_cost: .2f}  (negative = drag)")

        # 3) Sharpe by regime based on daily equity curve.
        if daily_equity is not None and not daily_equity.empty:
            de = daily_equity.sort_values("date").copy()
            de_idxed = de.set_index("date")
            rets = de_idxed["equity"].pct_change().dropna()
            regimes = de_idxed["regime"].reindex(rets.index)

            def _sharpe(x: pd.Series) -> float:
                if len(x) < 2:
                    return float("nan")
                std = float(x.std(ddof=1))
                if std == 0.0:
                    return float("nan")
                return float((x.mean() / std) * (252 ** 0.5))

            print("\nSharpe by regime (daily equity):")
            for reg in ["Bull", "Bear", "Cautious", "Sideways", "Crisis"]:
                mask = regimes == reg
                reg_rets = rets[mask]
                if not reg_rets.empty:
                    s = _sharpe(reg_rets)
                    print(f"  {reg:<8}: {s: .2f}")
            overall_s = _sharpe(rets)
            print(f"  Overall: {overall_s: .2f}")

        # 4) Cumulative P&L decomposition over time
        df_plot = df.sort_values("exit_date").copy()
        df_plot["long_pnl"] = np.where(df_plot["direction"] > 0, df_plot["realized_pnl"], 0.0)
        df_plot["short_pnl"] = np.where(df_plot["direction"] < 0, df_plot["realized_pnl"], 0.0)
        df_plot["cost_drag"] = -df_plot["transaction_cost"]

        df_cum = (
            df_plot.groupby("exit_date")[["long_pnl", "short_pnl", "cost_drag"]]
            .sum()
            .sort_index()
            .cumsum()
        )

        plt.figure(figsize=(9, 5))
        plt.plot(df_cum.index, df_cum["long_pnl"], label="Long P&L", color="green")
        plt.plot(df_cum.index, df_cum["short_pnl"], label="Short P&L", color="red")
        plt.plot(df_cum.index, df_cum["cost_drag"], label="Cost drag", color="orange")
        plt.axhline(0.0, color="black", linewidth=0.8)
        plt.xlabel("Date")
        plt.ylabel("Cumulative P&L (USD)")
        plt.title("P&L decomposition: longs vs shorts vs costs")
        plt.legend(loc="best")
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        plot_path = out_dir / "pnl_decomposition.png"
        plt.savefig(plot_path)
        plt.close()

    def run_with_custom_signals(
        self,
        price_data: dict,
        signal_data: dict,
        regime_data: dict,
        regime_score_data: dict | None = None,
    ):
        """
        Run simulation with pre-built price_data, signal_data, regime_data (no data load).
        Resets portfolio and runs _simulate. Used for Monte Carlo (noise, delay).
        """
        self.portfolio = Portfolio(self.config.initial_capital, self.config.max_positions)
        self._simulate(price_data, signal_data, regime_data, regime_score_data=regime_score_data or {})
        trades = pd.DataFrame(self.portfolio.trade_log)
        daily_equity = pd.DataFrame(self.portfolio.equity_history)
        if not trades.empty:
            trades.sort_values("entry_date", inplace=True)
            trades.reset_index(drop=True, inplace=True)
        metrics = compute_all_metrics(trades, daily_equity, self.config)
        metrics["crisis_entries_blocked_days_1_3"] = int(
            getattr(self, "_crisis_entries_blocked_days_1_3", 0)
        )
        result = BacktestResult(trades, daily_equity, metrics, self.config)
        result.price_data = price_data
        result.signal_data = signal_data
        result.regime_data = regime_data
        return result

    # ==============================================================
    # Phase 1 — data loading & signal generation
    # ==============================================================

    def prepare_data(self, tickers: list[str]) -> tuple[dict, dict]:
        start_ts = pd.Timestamp(self.config.start_date)
        dl_start = start_ts - timedelta(days=HISTORY_BUFFER_DAYS)
        # Date Clipping: ensure we never request data into the future which can cause Yahoo errors
        today_ts = pd.Timestamp(datetime.datetime.now())
        dl_end = pd.Timestamp(self.config.end_date) + timedelta(days=EXIT_BUFFER_DAYS)
        if dl_end > today_ts:
            dl_end = today_ts

        price_data: dict[str, pd.DataFrame] = {}
        signal_data: dict[str, pd.DataFrame] = {}
        pending_prices: dict[str, pd.DataFrame] = {}
        min_history_days = int(getattr(self.config, "min_history_days", 252))
        excluded_for_history: list[str] = []

        provider = getattr(self.config, "data_provider", "yahoo") or "yahoo"
        cache = getattr(self.config, "cache_ohlcv", True)
        cache_dir = getattr(self.config, "cache_dir", "data/cache/ohlcv")
        # Institutional Caching Fix: Default to 1-day TTL if not specified.
        # This prevents redundant 50-year downloads for 500 tickers on every run.
        cache_ttl = getattr(self.config, "cache_ttl_days", 1)

        # SPY/VIX Pre-load (Institutional Fix: load once with Tiingo to avoid Yahoo rate-limiting)
        spy_df_global = None
        vix_df_global = None
        vix3m_df_global = None
        try:
            print("\nPre-loading benchmarks (SPY, ^VIX, ^VIX3M) using Tiingo…")
            spy_df_global = get_ohlcv(
                "SPY",
                dl_start.strftime("%Y-%m-%d"),
                dl_end.strftime("%Y-%m-%d"),
                provider="tiingo",
                cache_dir=cache_dir,
                use_cache=cache,
                cache_ttl_days=cache_ttl,
            )
            vix_df_global = get_ohlcv(
                "^VIX",
                dl_start.strftime("%Y-%m-%d"),
                dl_end.strftime("%Y-%m-%d"),
                provider="tiingo",
                cache_dir=cache_dir,
                use_cache=cache,
                cache_ttl_days=cache_ttl,
            )
            vix3m_df_global = get_ohlcv(
                "^VIX3M",
                dl_start.strftime("%Y-%m-%d"),
                dl_end.strftime("%Y-%m-%d"),
                provider="tiingo",
                cache_dir=cache_dir,
                use_cache=cache,
                cache_ttl_days=cache_ttl,
            )
        except Exception as exc:
            logger.warning("Optional pre-load for benchmarks (SPY/VIX/VIX3M) failed: %s.", exc)

        for i, ticker in enumerate(tickers, 1):
            msg_prefix = f"[{i}/{len(tickers)}] {ticker}"
            print(f"  {msg_prefix}…", end=" ")
            logger.debug(
                "Requesting OHLCV for %s: %s → %s (provider=%s, cache_dir=%s, ttl=%d)",
                ticker,
                dl_start.strftime("%Y-%m-%d"),
                dl_end.strftime("%Y-%m-%d"),
                provider,
                cache_dir,
                cache_ttl,
            )
            try:
                raw = get_ohlcv(
                    ticker,
                    dl_start.strftime("%Y-%m-%d"),
                    dl_end.strftime("%Y-%m-%d"),
                    provider=provider,
                    cache_dir=cache_dir,
                    use_cache=cache,
                    cache_ttl_days=cache_ttl,
                )
                # Only require core OHLCV columns to be non-null; ignore NaNs in auxiliary columns
                data = raw.dropna(subset=["Close"])
                logger.info(
                    "Loaded OHLCV for %s: shape=%s",
                    ticker,
                    getattr(raw, "shape", None),
                )
                if not data.empty:
                    logger.debug(
                        "OHLCV window for %s: %s → %s (n=%d, cols=%s)",
                        ticker,
                        data.index.min(),
                        data.index.max(),
                        len(data),
                        list(data.columns),
                    )

                if data.empty or len(data) < 210:
                    print("insufficient data")
                    logger.warning(
                        "Skipping %s due to insufficient data (len=%d < 210 bars)",
                        ticker,
                        len(data),
                    )
                    continue
                # Debug: show actual data window returned before pre-history check.
                first_date = data.index.min()
                last_date = data.index.max()
                print(
                    f"{msg_prefix} OHLCV window: first_date={first_date.date()}, "
                    f"last_date={last_date.date()}, n={len(data)}"
                )
                # Allow tickers that list after the backtest start date (Unbalanced Panel Support)
                pass

                pending_prices[ticker] = data

            except Exception as exc:
                print(f"ERROR: {exc}")
                logger.exception("Error preparing data for %s: %s", ticker, exc)

        # Cross-sectional sector-relative momentum needs the full universe; compute once
        # then pass per-ticker series into generate_signals so learned adjusted_score matches training.
        sr_map: dict[str, pd.DataFrame] = {}
        if pending_prices and sector_relative_features_by_ticker is not None:
            try:
                sr_map = sector_relative_features_by_ticker(
                    pending_prices,
                    exclude_tickers=frozenset({"SPY"}),
                )
            except Exception:
                logger.exception(
                    "sector_relative_features_by_ticker failed; using zeros for sector-relative features"
                )
                sr_map = {}

        vr_map: dict[str, pd.DataFrame] = {}
        if pending_prices and vol_rank_features_by_ticker is not None:
            try:
                vr_map = vol_rank_features_by_ticker(
                    pending_prices,
                    exclude_tickers=frozenset({"SPY"}),
                )
            except Exception:
                logger.exception(
                    "vol_rank_features_by_ticker failed; using neutral vol_rank for learned signals"
                )
                vr_map = {}

        # Pillar 29: Institutional Joint-Panel Feature Parity
        # Pre-generate cross-sectional z-scores for all core ML features to ensure alignment with training.
        from agents.weight_learning_agent.feature_builder import (
            _CS_Z_PANEL_INPLACE_COLS,
            _apply_cross_sectional_zscore_columns,
        )
        panel_map: dict[str, pd.DataFrame] = {}
        if pending_prices and (getattr(self.config, "signal_mode", "price") in ("ml", "ensemble")):
            try:
                print(f"\nPhase 1b: Vectorizing {len(_CS_Z_PANEL_INPLACE_COLS)} joint-panel features across {len(pending_prices)} tickers…")
                ticker_dfs = []
                for tk, df in pending_prices.items():
                    if "Close" not in df.columns: continue
                    # Optimized vectorized feature builder
                    t_df = df[["Close", "Volume"]].copy()
                    t_df["ticker"] = tk
                    t_df["ret_5d"] = t_df["Close"].pct_change(5)
                    t_df["ret_10d"] = t_df["Close"].pct_change(10)
                    t_df["rolling_vol_20"] = t_df["Close"].pct_change(fill_method=None).rolling(20).std()
                    t_df["rolling_vol_60"] = t_df["Close"].pct_change(fill_method=None).rolling(60).std()
                    t_df["volume_zscore"] = (t_df["Volume"] - t_df["Volume"].rolling(20).mean()) / t_df["Volume"].rolling(20).std().replace(0, np.nan)
                    t_df["vix_zscore"] = 0.0
                    t_df["vol_spike"] = 0.0
                    # cs_momentum_raw: 6m return (126 days) excluding last month (shift 21)
                    # CS percentile rank applied below after panel concat
                    t_df["cs_momentum_raw"] = t_df["Close"].pct_change(126).shift(21)
                    # Pillar 29: Force lowercase 'date' identifier
                    t_df.index.name = "date"
                    ticker_dfs.append(t_df.reset_index())

                if ticker_dfs:
                    panel_df = pd.concat(ticker_dfs, ignore_index=True)
                    # Apply cross-sectional z-score per date slice
                    panel_df = _apply_cross_sectional_zscore_columns(panel_df, list(_CS_Z_PANEL_INPLACE_COLS))
                    # Cross-sectional momentum percentile: rank each ticker vs peers on same date
                    panel_df["cs_momentum_percentile"] = (
                        panel_df.groupby("date")["cs_momentum_raw"]
                        .rank(pct=True, method="average")
                    )
                    for tk in pending_prices.keys():
                        panel_map[tk] = panel_df[panel_df["ticker"] == tk].set_index("date").drop(columns=["ticker"])
                    print(f"  Vectorized synchronization complete.")
            except Exception as e:
                logger.error("Pillar 29: Joint-Panel synchronization failed: %s", e)
                panel_map = {}

        n_pending = len(pending_prices)
        for j, (ticker, data) in enumerate(pending_prices.items(), 1):
            print(f"  [signals {j}/{n_pending}] {ticker}…", end=" ")
            try:
                sr20_s = None
                sr60_s = None
                blk = sr_map.get(ticker)
                if blk is not None and not blk.empty:
                    sr20_s = blk["sector_relative_20d"]
                    sr60_s = blk["sector_relative_60d"]
                vr_s = None
                vblk = vr_map.get(ticker)
                if vblk is not None and not vblk.empty and "vol_rank" in vblk.columns:
                    vr_s = vblk["vol_rank"]
                signals = self.signal_engine.generate_signals(
                    data,
                    sector_relative_20d=sr20_s,
                    sector_relative_60d=sr60_s,
                    vol_rank=vr_s,
                    panel_features=panel_map.get(ticker),  # Pillar 29 Injection
                    spy_df=spy_df_global,
                    vix_df=vix_df_global,
                    vix3m_df=vix3m_df_global,
                    ticker=ticker,  # C1: earnings_surprise cache lookup
                )
                if signals.empty:
                    print("no signals")
                    logger.warning("No signals generated for %s; skipping.", ticker)
                    continue

                if self.config.signal_mode == "full":
                    sentiments = self.signal_engine.fetch_ticker_sentiments(ticker)
                    signals = self.signal_engine.apply_sentiment_overlay(signals, sentiments)
                    sent_parts = []
                    if sentiments["regional_sentiment"]:
                        sent_parts.append(f"reg={sentiments['regional_sentiment']:+.2f}")
                    if sentiments["global_sentiment"]:
                        sent_parts.append(f"glo={sentiments['global_sentiment']:+.2f}")
                    if sentiments["social_sentiment"]:
                        sent_parts.append(f"soc={sentiments['social_sentiment']:+.2f}")
                    sent_str = f" [{', '.join(sent_parts)}]" if sent_parts else ""
                else:
                    sent_str = ""

                price_data[ticker] = data
                signal_data[ticker] = signals

                non_neutral = int((signals["signal"] != "Neutral").sum())
                print(f"{len(data)} bars, {non_neutral} signals{sent_str}")
                logger.debug(
                    "Prepared data for %s: %d bars, %d non-neutral signals",
                    ticker,
                    len(data),
                    non_neutral,
                )
            except Exception as exc:
                print(f"ERROR: {exc}")
                logger.exception("Error generating signals for %s: %s", ticker, exc)

        # Always load SPY for volatility-based risk scaling.
        # Regime detection downloads SPY separately, but the simulation's
        # `price_data` dict may not include it.
        if "SPY" not in price_data:
            try:
                spy_raw = get_ohlcv(
                    "SPY",
                    dl_start.strftime("%Y-%m-%d"),
                    dl_end.strftime("%Y-%m-%d"),
                    provider=provider,
                    cache_dir=cache_dir,
                    use_cache=cache,
                    cache_ttl_days=cache_ttl,
                )
                spy_data = spy_raw.dropna(subset=["Close"])
                if not spy_data.empty:
                    price_data["SPY"] = spy_data
                    logger.info(
                        "Loaded SPY for vol scaling: shape=%s",
                        getattr(spy_data, "shape", None),
                    )
            except Exception:
                logger.exception("Failed to load SPY for vol scaling")

        if excluded_for_history:
            msg = (
                f"Excluding tickers with insufficient history "
                f"(<{min_history_days} days before start_date): "
                f"{', '.join(sorted(set(excluded_for_history)))}"
            )
            print(f"\n[WARN] {msg}")
            logger.warning(msg)

        return price_data, signal_data

    # ==============================================================
    # Phase 3 — day-by-day simulation
    # ==============================================================

    def _simulate(self, price_data, signal_data, regime_data, regime_score_data: dict | None = None):
        # D3: reset Sharpe CB state at start of each simulation run
        self._sharpe_cb_active = False
        self._daily_equity_returns = []
        self._factor_diagnostics = []
        # Precompute regime-level score dispersion once for fragmented-regime fallback
        # in the rolling-std confidence gate.
        self._precompute_regime_score_stats(signal_data, regime_data)
        # Build unified trading calendar
        all_dates: set[pd.Timestamp] = set()
        for df in price_data.values():
            all_dates.update(df.index)

        start_ts = pd.Timestamp(self.config.start_date)
        end_ts = pd.Timestamp(self.config.end_date)
        sim_end = end_ts + timedelta(days=EXIT_BUFFER_DAYS)

        trading_days = sorted(d for d in all_dates if start_ts <= d <= sim_end)
        if not trading_days:
            logger.warning("No trading days in simulation window %s → %s", start_ts, sim_end)
            return
        date_to_idx = {d: idx for idx, d in enumerate(trading_days)}
        min_hold = int(getattr(self.config, "min_holding_period_days", 0) or 0)
        crisis_beta_cutoff = float(getattr(self.config, "crisis_beta_cutoff", 0.8) or 0.8)
        crisis_signal_window_days = int(getattr(self.config, "crisis_signal_window_days", 60) or 60)
        crisis_signal_quantile = float(getattr(self.config, "crisis_signal_quantile", 0.95) or 0.95)
        bear_signal_window_days = int(getattr(self.config, "bear_signal_window_days", 60) or 60)
        bear_signal_quantile = float(getattr(self.config, "bear_signal_quantile", 0.70) or 0.70)
        early_exit_drawdown_pct = float(
            getattr(self.config, "min_holding_early_exit_drawdown_pct", 0.03) or 0.03
        )
        rebalance_every = int(getattr(self.config, "rebalance_every_trading_days", 1) or 1)
        if rebalance_every < 1:
            rebalance_every = 1

        def _held_trading_days(pos: Position, cur_i: int) -> int:
            entry_i = date_to_idx.get(pos.entry_date)
            if entry_i is None:
                # Fallback: should not happen because entry_date comes from trading_days.
                return cur_i
            return cur_i - entry_i

        def _threshold_exit_price_for_stop(pos: Position) -> float:
            if pos.entry_price <= 0:
                return float(pos.current_price or pos.entry_price or 0.0)
            if pos.direction > 0:
                return pos.entry_price * (1.0 - early_exit_drawdown_pct)
            return pos.entry_price * (1.0 + early_exit_drawdown_pct)

        def _min_hold_effective_for_pos(pos: Position) -> int:
            # Crisis transition: forced shortened exit must not be blocked by min-hold.
            if getattr(pos, "crisis_accelerated_exit", False):
                return 0
            # Crisis: allow positions to close on their planned exit date
            # (candidates.py shortens holding_days to <= 3). This disables the
            # global min-holding constraint for Crisis positions.
            if getattr(pos, "regime", None) == "Crisis":
                return 0
            # Bear: shortened planned holds (see candidates); allow expiry/stops without min-hold block.
            if getattr(pos, "regime", None) == "Bear":
                return 0
            return min_hold

        def _drawdown_exceeded_early_stop(pos: Position, tk: str, day: pd.Timestamp) -> bool:
            if tk not in price_data or day not in price_data[tk].index:
                return False
            if pos.entry_price <= 0:
                return False
            bar = price_data[tk].loc[day]
            entry = float(pos.entry_price)
            thr = early_exit_drawdown_pct
            if pos.direction > 0:
                low = float(bar.get("Low", np.nan))
                return np.isfinite(low) and low <= entry * (1.0 - thr)
            # For shorts, adverse move is price up.
            high = float(bar.get("High", np.nan))
            return np.isfinite(high) and high >= entry * (1.0 + thr)

        # ==============================================================
        # Volatility-based scaling (SPY 20d realized vs 252d average)
        # ==============================================================
        vol_scalar_series = pd.Series(1.0, index=pd.DatetimeIndex(trading_days))
        try:
            spy_df = price_data.get("SPY")
            if spy_df is not None and not spy_df.empty and "Close" in spy_df.columns:
                spy_close = pd.to_numeric(spy_df["Close"], errors="coerce").dropna().sort_index()
                if len(spy_close) >= 300:
                    spy_ret = spy_close.pct_change().dropna()
                    vol20_annual = spy_ret.rolling(20).std() * np.sqrt(252.0)
                    long_run_vol = vol20_annual.rolling(252).mean()
                    vol_ratio = vol20_annual / long_run_vol.replace(0.0, np.nan)
                    # scaling = min(1, 1/vol_ratio) so if vol is 2x normal => scale 0.5.
                    vol_scalar = np.minimum(1.0, 1.0 / vol_ratio)
                    vol_scalar = (
                        pd.to_numeric(vol_scalar, errors="coerce")
                        .replace([np.inf, -np.inf], np.nan)
                        .fillna(1.0)
                    )
                    vol_scalar_series = vol_scalar.reindex(trading_days).ffill().bfill().fillna(1.0)
        except Exception:
            logger.exception("SPY vol scaling failed; using scalar=1.0")

        # ==============================================================
        # VIX-threshold deleveraging (stacked on top of SPY vol scalar)
        # When VIX >= high_vix_threshold, multiply position sizes by
        # vix_deleverage_scaling_factor — reduces drawdown in risk-off spikes
        # before the full Crisis regime takes effect at VIX=30.
        # ==============================================================
        if bool(getattr(self.config, "vix_deleverage_enabled", False)):
            try:
                _vix_thresh = float(getattr(self.config, "vix_deleverage_high_threshold", 28.0))
                _vix_scale = float(getattr(self.config, "vix_deleverage_scaling_factor", 0.5))
                # Download VIX using the same helper from RegimeAgent to avoid code duplication.
                _vix_start = pd.Timestamp(self.config.start_date) - pd.Timedelta(days=600)
                _vix_end = pd.Timestamp(self.config.end_date) + pd.Timedelta(days=30)
                _vix_dict = MarketRegimeAgent._download_vix(_vix_start, _vix_end, price_data.get("SPY", pd.DataFrame()))
                if _vix_dict:
                    _vix_s = pd.Series(_vix_dict)
                    _vix_s.index = pd.to_datetime(_vix_s.index)
                    _vix_aligned = _vix_s.reindex(trading_days).ffill().bfill().fillna(15.0)
                    # Where VIX is elevated, replace vol_scalar with min of current and vix_scale
                    _elevated = _vix_aligned >= _vix_thresh
                    vol_scalar_series = vol_scalar_series.copy()
                    vol_scalar_series.loc[_elevated] = vol_scalar_series.loc[_elevated].clip(upper=_vix_scale)
                    n_elevated = int(_elevated.sum())
                    logger.info(
                        "VIX deleveraging active: %d/%d days VIX >= %.1f → scaling capped at %.2f",
                        n_elevated, len(trading_days), _vix_thresh, _vix_scale,
                    )
            except Exception:
                logger.exception("VIX deleveraging failed; vol_scalar unchanged")

        # ── VIX-Scaled Transaction Costs (Almgren-Chriss + Hasbrouck 2009) ──────────────
        # Bid-ask spreads expand 3-7× during stress. Replace fixed slippage_bps with a
        # dynamic scalar: cost_scalar = 1 + α × max(0, VIX_t − VIX_rolling_median_t).
        # α=0.08: at VIX=35 vs median=20 → scalar=2.2×; at VIX=80 (GFC peak) → scalar=5.8×.
        # Automatically self-limits turnover in expensive markets — no regime override needed.
        _vix_cost_scalar: dict[pd.Timestamp, float] = {}
        if bool(getattr(self.config, "vix_cost_scaling_enabled", True)):
            try:
                _vc_start = pd.Timestamp(self.config.start_date) - pd.Timedelta(days=600)
                _vc_end = pd.Timestamp(self.config.end_date) + pd.Timedelta(days=30)
                _vc_dict = MarketRegimeAgent._download_vix(
                    _vc_start, _vc_end, price_data.get("SPY", pd.DataFrame())
                )
                if _vc_dict:
                    _vc_raw = pd.Series(_vc_dict)
                    _vc_raw.index = pd.to_datetime(_vc_raw.index)
                    _vc_aligned = (
                        _vc_raw.reindex(pd.DatetimeIndex(trading_days))
                        .ffill().bfill().fillna(15.0)
                    )
                    _vc_median = (
                        _vc_aligned.rolling(252, min_periods=60)
                        .median()
                        .fillna(float(_vc_aligned.median()))
                    )
                    _vc_alpha = float(getattr(self.config, "vix_cost_alpha", 0.08))
                    _vc_scalars = 1.0 + _vc_alpha * (_vc_aligned - _vc_median).clip(lower=0.0)
                    for _d, _s in zip(trading_days, _vc_scalars):
                        _vix_cost_scalar[pd.Timestamp(_d)] = float(_s)
                    _peak_day = _vc_scalars.idxmax()
                    logger.info(
                        "VIX cost scaling active: α=%.3f | peak %.2f× on %s",
                        _vc_alpha, float(_vc_scalars.max()), _peak_day.date(),
                    )
            except Exception:
                logger.warning("VIX cost scaling init failed; using constant costs.")

        # Pre-index: calendar date → [(ticker, signal_row)] (string key so lookup matches regardless of type/tz)
        def _to_calendar_key(ts) -> str:
            if isinstance(ts, datetime.date) and not isinstance(ts, datetime.datetime):
                return ts.isoformat()
            return pd.Timestamp(ts).strftime("%Y-%m-%d")
        cross_sectional = getattr(self.config, "cross_sectional_ranking", False)
        # Optional std-based entry threshold scan:
        # When enabled, candidates are filtered later (in build_ranked_candidates)
        # using abs(adjusted_score) > (multiplier * signal_score_std).
        signal_threshold_std_multiplier = getattr(self.config, "signal_threshold_std_multiplier", None)
        signal_threshold_abs: float | None = None
        if signal_threshold_std_multiplier is not None:
            try:
                scores: list[float] = []
                for _tk, sig_df in signal_data.items():
                    if sig_df is None or sig_df.empty or "adjusted_score" not in sig_df.columns:
                        continue
                    mask = (sig_df.index >= start_ts) & (sig_df.index <= end_ts)
                    vals = pd.to_numeric(sig_df.loc[mask, "adjusted_score"], errors="coerce").dropna()
                    if not vals.empty:
                        scores.extend(vals.astype(float).tolist())
                if len(scores) > 1:
                    signal_score_std = float(pd.Series(scores).std(ddof=1))
                else:
                    signal_score_std = float("nan")
                # Store so candidate builders can use it.
                self.config.signal_score_std = signal_score_std
                if np.isfinite(signal_score_std) and signal_score_std > 0:
                    signal_threshold_abs = float(signal_threshold_std_multiplier) * signal_score_std
            except Exception:
                logger.exception("Failed to compute signal_score_std for std-based threshold scanning")

        base_min_strength = float(getattr(self.config, "min_signal_strength", 0.3))
        # If our std-based threshold is below the configured min_signal_strength,
        # broaden inclusion of Neutral scores so candidates can still be filtered later.
        if signal_threshold_abs is not None and np.isfinite(signal_threshold_abs):
            min_strength = min(base_min_strength, float(signal_threshold_abs))
        else:
            min_strength = base_min_strength
        daily_signals: dict[str, list] = defaultdict(list)
        for ticker, sig_df in signal_data.items():
            mask = (sig_df.index >= start_ts) & (sig_df.index <= end_ts)
            for date, row in sig_df[mask].iterrows():
                d = _to_calendar_key(date)
                if cross_sectional:
                    daily_signals[d].append((ticker, row))
                elif row["signal"] != "Neutral":
                    daily_signals[d].append((ticker, row))
                else:
                    # Include strong Neutral so more tickers qualify (long-only treats as weak long)
                    adj = row.get("adjusted_score")
                    if adj is not None and abs(float(adj)) >= min_strength:
                        daily_signals[d].append((ticker, row))
        logger.info(
            "Signal inclusion summary: days=%d, avg_candidates_per_day=%.2f, max_candidates_on_day=%d",
            len(daily_signals),
            float(np.mean([len(v) for v in daily_signals.values()])) if daily_signals else 0.0,
            max((len(v) for v in daily_signals.values()), default=0),
        )

        # Factor neutralizer (after signals, before ranking)
        factor_neutralizer = None
        if getattr(self.config, "factor_neutralization_enabled", False) and FactorNeutralizer is not None:
            try:
                from config import DEV_MODE
                dev_limit = 10 if DEV_MODE else None
            except ImportError:
                dev_limit = None
            factor_neutralizer = FactorNeutralizer(
                neutralize_market_beta=getattr(self.config, "factor_neutralize_market_beta", True),
                neutralize_sector=getattr(self.config, "factor_neutralize_sector", True),
                neutralize_size=getattr(self.config, "factor_neutralize_size", True),
                neutralize_smb=getattr(self.config, "factor_neutralize_smb", True),
                neutralize_rmw=getattr(self.config, "factor_neutralize_rmw", True),
                market_index=getattr(self.config, "factor_market_index", "SPY"),
                rolling_window=int(getattr(self.config, "factor_rolling_window", 60)),
                min_observations=10,
                dev_mode_limit=dev_limit,
                score_direction=float(getattr(self.config, "score_direction", 1.0)),
                min_signal_strength=float(getattr(self.config, "min_signal_strength", 0.001)),
            )
        factor_diagnostics: list[dict] = []
        execution_delay_days = int(getattr(self.config, "execution_delay_days", 0) or 0)

        scheduled_entries: list[tuple[int, dict]] = []  # (execute_at_idx, entry)
        pending_entries: list[dict] = []
        daily_allocation_rows: list[dict] = []  # for output/portfolio/daily_positions.csv

        crisis_consecutive_days = 0
        prev_regime: str | None = None
        crisis_entries_blocked_days_1_3 = 0
        universe_sizes = []

        for i, date in enumerate(trading_days):
            # Trade rebalance day (used to gate expiry exits and new signal updates).
            is_trade_rebalance_day_current = (rebalance_every == 1) or (i % rebalance_every == 0)
            # The entries created from signals computed on day i will execute at i+1+delay.
            execute_at = i + 1 + execution_delay_days
            is_trade_rebalance_day_next = (
                execute_at < len(trading_days)
                and ((rebalance_every == 1) or (execute_at % rebalance_every == 0))
            )
            # --- 0. Pull entries scheduled for today (repopulate every day from due scheduled entries) ---
            pending_entries = [e for (at, e) in scheduled_entries if at <= i]
            scheduled_entries = [(at, e) for (at, e) in scheduled_entries if at > i]

            # Today's market regime (used for risk exits and entry filters).
            regime_today = regime_data.get(date, "Sideways")
            # Update VIX-scaled cost scalar — used by entry slippage and _close_position this day.
            self._cost_scalar_today = _vix_cost_scalar.get(date, 1.0)
            prev_reg_before = prev_regime
            # C2: inform signal engine of current regime so it can route to regime-conditional model
            if hasattr(self.signal_engine, "set_regime"):
                self.signal_engine.set_regime(regime_today)

            # Bear entry: shorten remaining hold for longs that entered before the Bear regime.
            # Positions opened before the regime switch survive the one-time liquidation check
            # (e.g. already in the portfolio on a non-entry Bear day, or entered during hysteresis window).
            # Cap their planned_exit to at most bear_max_carry_days ahead so they don't drag for 10 days.
            _bear_carry_cap = int(getattr(self.config, "bear_max_carry_days", 2) or 2)
            if (
                regime_today == "Bear"
                and prev_reg_before is not None
                and prev_reg_before != "Bear"
            ):
                for pos in list(self.portfolio.positions):
                    if pos.direction > 0 and hasattr(pos, "planned_exit_date"):
                        cap_idx = min(i + _bear_carry_cap, len(trading_days) - 1)
                        forced_cap = pd.Timestamp(trading_days[cap_idx])
                        pe = pd.Timestamp(pos.planned_exit_date)
                        if pe > forced_cap:
                            pos.planned_exit_date = forced_cap

            if regime_today == "Crisis":
                crisis_consecutive_days = (crisis_consecutive_days + 1) if prev_regime == "Crisis" else 1
            else:
                crisis_consecutive_days = 0
            prev_regime = regime_today

            # Crisis transition: cap exit dates for existing winners so they don't drag into prolonged Crisis.
            # No forced closes — positions exit via their signal-based planned_exit_date.
            # Entry gates (crisis_block_all_new_entries) prevent new risk from being opened.
            if (
                regime_today == "Crisis"
                and prev_reg_before is not None
                and prev_reg_before != "Crisis"
                and bool(getattr(self.config, "crisis_transition_accel_enabled", True))
            ):
                winner_extra = int(getattr(self.config, "crisis_transition_winner_extra_days", 3) or 3)
                for pos in list(self.portfolio.positions):
                    if hasattr(pos, "planned_exit_date"):
                        cap_idx = min(i + winner_extra, len(trading_days) - 1)
                        forced_cap = pd.Timestamp(trading_days[cap_idx])
                        pe = pd.Timestamp(pos.planned_exit_date)
                        if pe > forced_cap:
                            pos.planned_exit_date = forced_cap
                            pos.crisis_accelerated_exit = True

            # --- 0a. Mean-Variance weights (rebalance every rebalance_days; no lookahead) ---
            method = getattr(self.config, "position_sizing_method", None) or getattr(
                self.config, "position_sizing", "equal"
            )
            if method in ("mv_max_sharpe", "mv_min_variance"):
                rebalance_days = getattr(self.config, "mean_variance_rebalance_days", 20)
                lookback = getattr(self.config, "mean_variance_lookback_days", 60)
                last_rebalance = getattr(self, "_last_mv_rebalance_idx", -99999)
                if i - last_rebalance >= rebalance_days or last_rebalance < 0:
                    returns_df = self._build_returns_matrix(price_data, date, lookback)
                    if returns_df is not None and not returns_df.empty and returns_df.shape[1] >= 1:
                        try:
                            from portfolio.mean_variance import (
                                max_sharpe_weights,
                                min_variance_weights,
                            )
                            max_w = getattr(self.config, "mean_variance_max_single_weight", 0.25)
                            constraints = {"max_weight": max_w}
                            if method == "mv_max_sharpe":
                                self._mv_weights = max_sharpe_weights(
                                    returns_df, risk_free_rate=0.04, constraints=constraints
                                )
                            else:
                                self._mv_weights = min_variance_weights(returns_df, constraints=constraints)
                            self._last_mv_rebalance_idx = i
                        except Exception:
                            self._mv_weights = {}
                    else:
                        self._mv_weights = {}
            else:
                self._mv_weights = {}

            # --- 0b. Circuit breaker: update drawdown and trading halt state ---
            equity = self.portfolio.equity
            if equity > self._equity_peak:
                self._equity_peak = equity
            drawdown = 0.0
            if self._equity_peak > 0:
                drawdown = (equity - self._equity_peak) / self._equity_peak  # negative when underwater
            dd_abs = -drawdown if drawdown < 0 else 0.0

            max_dd = float(getattr(self.config, "max_drawdown_pct", 0.0) or 0.0)
            resume_dd = float(getattr(self.config, "drawdown_resume_pct", max_dd))
            severe_dd = float(getattr(self.config, "severe_drawdown_close_all_pct", 0.0) or 0.0)

            # Enter halt state if drawdown breaches max_dd
            if max_dd > 0 and dd_abs > max_dd and not self._trading_halted:
                self._trading_halted = True
                self._circuit_breaker_log.append(
                    {"event": "halt", "date": date, "drawdown_pct": dd_abs}
                )
                logger.warning(
                    "Circuit breaker HALT at %s: drawdown=%.2f%% exceeds max_drawdown_pct=%.2f%%",
                    date,
                    dd_abs * 100.0,
                    max_dd * 100.0,
                )

            # Optional severe close-all if drawdown very large
            if severe_dd > 0 and dd_abs > severe_dd and self.portfolio.positions:
                logger.warning(
                    "Severe drawdown at %s: drawdown=%.2f%% exceeds severe_drawdown_close_all_pct=%.2f%%. "
                    "Closing all positions.",
                    date,
                    dd_abs * 100.0,
                    severe_dd * 100.0,
                )
                for pos in list(self.portfolio.positions):
                    self._close_position(pos, date, price_data, reason="circuit_close_all")

            # Resume trading once drawdown improves above resume_dd
            if (
                self._trading_halted
                and resume_dd >= 0
                and dd_abs < resume_dd
            ):
                self._trading_halted = False
                self._circuit_breaker_log.append(
                    {"event": "resume", "date": date, "drawdown_pct": dd_abs}
                )
                logger.info(
                    "Circuit breaker RESUME at %s: drawdown=%.2f%% below resume threshold=%.2f%%",
                    date,
                    dd_abs * 100.0,
                    resume_dd * 100.0,
                )

            # D3: Rolling Sharpe circuit breaker — track equity and update CB state
            self._daily_equity_returns.append(float(equity))
            _sharpe_cb_w = int(getattr(self.config, "sharpe_cb_window_days", 60) or 60)
            _sharpe_cb_thr = float(getattr(self.config, "sharpe_cb_threshold", 0.0) or 0.0)
            _sharpe_cb_rec = float(getattr(self.config, "sharpe_cb_recovery_threshold", 0.3) or 0.3)
            _sharpe_cb_en = bool(getattr(self.config, "sharpe_cb_enabled", True))
            if _sharpe_cb_en and len(self._daily_equity_returns) >= _sharpe_cb_w:
                eq_arr = np.array(self._daily_equity_returns[-_sharpe_cb_w - 1:], dtype=float)
                rets = np.diff(eq_arr) / np.where(eq_arr[:-1] > 0, eq_arr[:-1], 1.0)
                if len(rets) >= 20:
                    roll_sharpe = float(np.mean(rets) / max(np.std(rets, ddof=1), 1e-10) * np.sqrt(252))
                    if not self._sharpe_cb_active and roll_sharpe < _sharpe_cb_thr:
                        self._sharpe_cb_active = True
                        self._circuit_breaker_log.append({"event": "sharpe_cb_activate", "date": date, "sharpe": roll_sharpe})
                        logger.warning(
                            "Sharpe CB activated at %s: 60d Sharpe=%.2f < threshold=%.2f",
                            date, roll_sharpe, _sharpe_cb_thr,
                        )
                    elif self._sharpe_cb_active and roll_sharpe > _sharpe_cb_rec:
                        self._sharpe_cb_active = False
                        self._circuit_breaker_log.append({"event": "sharpe_cb_deactivate", "date": date, "sharpe": roll_sharpe})
                        logger.info(
                            "Sharpe CB deactivated at %s: 60d Sharpe=%.2f > recovery=%.2f",
                            date, roll_sharpe, _sharpe_cb_rec,
                        )

            # --- 0b. Regime-change based exits (optional early exit) ---
            if getattr(self.config, "regime_exit_on_change", False):
                regime_today = regime_data.get(date)
                if regime_today is not None:
                    for pos in list(self.portfolio.positions):
                        if pos.regime != regime_today:
                            held_days = _held_trading_days(pos, i)
                            min_hold_eff = _min_hold_effective_for_pos(pos)
                            if min_hold_eff > 0 and held_days < min_hold_eff:
                                if _drawdown_exceeded_early_stop(pos, pos.ticker, date):
                                    self._close_position(
                                        pos,
                                        date,
                                        price_data,
                                        exit_price_override=_threshold_exit_price_for_stop(pos),
                                        reason="stop_loss",
                                    )
                                continue
                            if is_trade_rebalance_day_current:
                                self._close_position(
                                    pos,
                                    date,
                                    price_data,
                                    reason="regime_change",
                                )
                            continue

            # --- 1. Close expired positions ---
            for pos in list(self.portfolio.positions):
                if pos.planned_exit_date <= date:
                    held_days = _held_trading_days(pos, i)
                    min_hold_eff = _min_hold_effective_for_pos(pos)
                    if min_hold_eff > 0 and held_days < min_hold_eff:
                        if _drawdown_exceeded_early_stop(pos, pos.ticker, date):
                            self._close_position(
                                pos,
                                date,
                                price_data,
                                exit_price_override=_threshold_exit_price_for_stop(pos),
                                reason="stop_loss",
                            )
                        continue
                    allow_expiry = is_trade_rebalance_day_current or getattr(
                        pos, "crisis_accelerated_exit", False
                    )
                    if allow_expiry:
                        px_override = None
                        # Any planned exit on a Crisis session: use open (with slippage) to limit same-day crash drag.
                        if regime_today == "Crisis" and bool(
                            getattr(self.config, "crisis_transition_use_open_exit", True)
                        ):
                            px_override = self._crisis_transition_exit_price(pos, date, price_data)
                        self._close_position(
                            pos,
                            date,
                            price_data,
                            exit_price_override=px_override,
                            reason="expiry",
                        )
                    continue

            # --- 1b. Stop-loss / take-profit (before entry loop so freed slots are available same day)
            # 0.0 means disabled for both
            stop_pct = getattr(self.config, "stop_loss_pct", 0.0) or 0.0
            take_pct = getattr(self.config, "take_profit_pct", 0.0) or 0.0
            for pos in list(self.portfolio.positions):
                tk = pos.ticker
                if tk not in price_data or date not in price_data[tk].index:
                    continue
                bar = price_data[tk].loc[date]
                high = float(bar.get("High", np.nan))
                low = float(bar.get("Low", np.nan))
                if not np.isfinite(high) or not np.isfinite(low):
                    continue
                long_side = pos.direction > 0
                stop_price = None
                take_price = None

                # Bear regime: exit longs on a tighter intraday drawdown (holdthrough + new risk).
                bear_exit_pct = float(
                    getattr(self.config, "bear_regime_intraday_exit_drawdown_pct", 0.02) or 0.0
                )
                if (
                    regime_today == "Bear"
                    and bear_exit_pct > 0
                    and long_side
                    and float(pos.entry_price or 0.0) > 0
                ):
                    bear_thr = float(pos.entry_price) * (1.0 - bear_exit_pct)
                    if np.isfinite(low) and low <= bear_thr <= high:
                        self._close_position(
                            pos,
                            date,
                            price_data,
                            exit_price_override=bear_thr,
                            reason="bear_regime_drawdown",
                        )
                        continue

                # Exception: allow early exit on adverse move > 3% even
                # if min holding isn't satisfied.
                held_days = _held_trading_days(pos, i)
                min_hold_eff = _min_hold_effective_for_pos(pos)
                if min_hold_eff > 0 and held_days < min_hold_eff and _drawdown_exceeded_early_stop(pos, tk, date):
                    self._close_position(
                        pos,
                        date,
                        price_data,
                        exit_price_override=_threshold_exit_price_for_stop(pos),
                        reason="stop_loss",
                    )
                    continue

                if stop_pct > 0:
                    stop_price = pos.entry_price * (1.0 - stop_pct) if long_side else pos.entry_price * (1.0 + stop_pct)
                if take_pct > 0:
                    take_price = pos.entry_price * (1.0 + take_pct) if long_side else pos.entry_price * (1.0 - take_pct)
                if stop_pct > 0 and stop_price is not None and low <= stop_price <= high:
                    self._close_position(pos, date, price_data, exit_price_override=stop_price, reason="stop_loss")
                    continue
                if take_pct > 0 and take_price is not None and low <= take_price <= high:
                    min_hold_eff = _min_hold_effective_for_pos(pos)
                    if min_hold_eff > 0 and held_days < min_hold_eff:
                        continue
                    self._close_position(pos, date, price_data, exit_price_override=take_price, reason="take_profit")
                    continue

                # Pillar 6: Short Snap Profit (Tactical Mean Reversion Fast Exit)
                snap_pct = float(getattr(self.config, "ml_short_snap_profit_pct", 0.0) or 0.0)
                if snap_pct > 0 and pos.direction < 0 and float(pos.entry_price or 0.0) > 0:
                    snap_thr = float(pos.entry_price) * (1.0 - snap_pct)
                    if np.isfinite(low) and low <= snap_thr:
                        self._close_position(
                            pos,
                            date,
                            price_data,
                            exit_price_override=snap_thr,
                            reason="short_snap_profit",
                        )
                        continue

                # Short stop loss: cap adverse moves on shorts (price rising against us).
                # Uses ml_short_stop_loss_pct — separate from the long stop_loss_pct so
                # we can tune independently without touching long exits.
                short_sl_pct = float(getattr(self.config, "ml_short_stop_loss_pct", 0.0) or 0.0)
                if short_sl_pct > 0 and pos.direction < 0 and float(pos.entry_price or 0.0) > 0:
                    short_sl_price = float(pos.entry_price) * (1.0 + short_sl_pct)
                    if np.isfinite(high) and high >= short_sl_price:
                        self._close_position(
                            pos,
                            date,
                            price_data,
                            exit_price_override=short_sl_price,
                            reason="short_stop_loss",
                        )
                        continue

                # ── ATR-Scaled Stop (Citadel/AQR standard) ───────────────────────────
                # Fires when intraday low drops > λ × ATR(14d) below entry price.
                # Regime-agnostic: the threshold self-calibrates to each stock's vol,
                # so a genuine fraud blow-up on a low-vol name fires while normal
                # momentum noise on a high-vol name does not.
                # λ=0.0 disables; set atr_stop_lambda: 2.5 in risk config to enable.
                _atr_lambda = float(getattr(self.config, "atr_stop_lambda", 0.0) or 0.0)
                if _atr_lambda > 0 and long_side and pos.entry_atr > 0:
                    _atr_stop_px = float(pos.entry_price) - _atr_lambda * pos.entry_atr
                    if np.isfinite(low) and low <= _atr_stop_px:
                        self._close_position(
                            pos, date, price_data,
                            exit_price_override=_atr_stop_px,
                            reason="atr_stop",
                        )
                        continue

                # ── Signal-Flip Exit ──────────────────────────────────────────────────
                # Close a long (short) when today's model score has inverted past the
                # threshold. The model itself is telling us the thesis no longer holds —
                # holding to planned expiry on a reversed signal is a look-ahead trap.
                # threshold=0.0 disables; set signal_flip_threshold: 0.3 to enable.
                _flip_thresh = float(getattr(self.config, "signal_flip_threshold", 0.0) or 0.0)
                if _flip_thresh > 0 and tk in signal_data:
                    _sig_df = signal_data.get(tk)
                    if _sig_df is not None and not _sig_df.empty and date in _sig_df.index:
                        _cur_score = float(_sig_df.loc[date, "adjusted_score"])
                        _flipped = (long_side and _cur_score < -_flip_thresh) or \
                                   (not long_side and _cur_score > _flip_thresh)
                        if _flipped:
                            self._close_position(pos, date, price_data, reason="signal_flip")
                            continue

            # --- 2. Execute pending entries at today's open ---
            existing_tickers = {p.ticker for p in self.portfolio.positions}

            # Cross-sectional daily rebalance: close positions not in today's target set
            if (
                cross_sectional
                and getattr(self.config, "cross_sectional_rebalance_daily", False)
                and pending_entries
                and is_trade_rebalance_day_current
            ):
                target_tickers = {e["ticker"] for e in pending_entries}
                if getattr(self.config, "no_trade_band_rebalance_enabled", True):
                    equity_now = float(self.portfolio.equity) if self.portfolio.equity else 0.0
                    if equity_now > 1e-12:
                        per_diff = float(getattr(self.config, "no_trade_band_weight_diff", 0.015) or 0.015)
                        port_drift_thr = float(getattr(self.config, "no_trade_band_total_drift", 0.05) or 0.05)
                        to_close: list = []
                        total_drift = 0.0
                        for pos in list(self.portfolio.positions):
                            if pos.ticker in target_tickers:
                                continue
                            w_to_zero = abs(float(pos.market_value)) / equity_now
                            if w_to_zero > per_diff:
                                to_close.append(pos)
                                total_drift += w_to_zero

                        # Only rebalance if the portfolio-level drift is meaningful.
                        if total_drift > port_drift_thr:
                            for pos in to_close:
                                held_days = _held_trading_days(pos, i)
                                min_hold_eff = _min_hold_effective_for_pos(pos)
                                if min_hold_eff > 0 and held_days < min_hold_eff:
                                    if _drawdown_exceeded_early_stop(pos, pos.ticker, date):
                                        self._close_position(
                                            pos,
                                            date,
                                            price_data,
                                            exit_price_override=_threshold_exit_price_for_stop(pos),
                                            reason="stop_loss",
                                        )
                                    continue
                                self._close_position(
                                    pos, date, price_data, reason="rebalance_no_trade_band"
                                )
                    else:
                        # If equity is effectively zero, fall back to legacy behaviour.
                        for pos in list(self.portfolio.positions):
                            if pos.ticker not in target_tickers:
                                held_days = _held_trading_days(pos, i)
                                min_hold_eff = _min_hold_effective_for_pos(pos)
                                if min_hold_eff > 0 and held_days < min_hold_eff:
                                    if _drawdown_exceeded_early_stop(pos, pos.ticker, date):
                                        self._close_position(
                                            pos,
                                            date,
                                            price_data,
                                            exit_price_override=_threshold_exit_price_for_stop(pos),
                                            reason="stop_loss",
                                        )
                                    continue
                                self._close_position(pos, date, price_data, reason="rebalance")
                else:
                    for pos in list(self.portfolio.positions):
                        if pos.ticker not in target_tickers:
                            held_days = _held_trading_days(pos, i)
                            min_hold_eff = _min_hold_effective_for_pos(pos)
                            if min_hold_eff > 0 and held_days < min_hold_eff:
                                if _drawdown_exceeded_early_stop(pos, pos.ticker, date):
                                    self._close_position(
                                        pos,
                                        date,
                                        price_data,
                                        exit_price_override=_threshold_exit_price_for_stop(pos),
                                        reason="stop_loss",
                                    )
                                continue
                            self._close_position(pos, date, price_data, reason="rebalance")
                existing_tickers = {p.ticker for p in self.portfolio.positions}

            # Equal capital per cross-sectional slot: equity divided by batch size (institutional equal-weight),
            # then adjusted by regime (Bull/Bear/Sideways/Crisis).
            cs_entries = [e for e in pending_entries if e.get("_cross_sectional")]
            equal_size = (
                self.portfolio.equity / max(len(cs_entries), 1) if cs_entries else None
            )
            if equal_size is not None:
                # Regime-aware adjustments for cross-sectional sizing.
                if regime_today == "Crisis":
                    # Crisis risk control is handled by the new vol-scaling + gross cap.
                    equal_size *= 1.0
                elif regime_today == "Cautious":
                    # State Street (Ung 2025): Cautious Decline has high uncertainty and equity
                    # returns of -0.7%/month. Cap position size more tightly than Sideways.
                    equal_size = min(equal_size, self.portfolio.equity * 0.045)
                elif regime_today == "Sideways":
                    # Cap position size at 5% of portfolio per ticker in Sideways regime.
                    equal_size = min(equal_size, self.portfolio.equity * 0.05)

            # Signal-strength sizing: precompute per-ticker sizes proportional to |adjusted_score|.
            _cs_sizing_method = getattr(self.config, "position_sizing", "equal")
            cs_signal_sizes: dict[str, float] | None = None
            if _cs_sizing_method == "signal_strength" and cs_entries:
                scores_abs = {
                    e["ticker"]: abs(float(e.get("adjusted_score", 0.0) or 0.0))
                    for e in cs_entries
                }
                total_abs = sum(scores_abs.values())
                _eq_cs = self.portfolio.equity
                _mxp_cs = float(getattr(self.config, "max_position_pct_of_equity", 0.12))
                if total_abs > 1e-12:
                    cs_signal_sizes = {
                        t: min(_eq_cs * (s / total_abs), _eq_cs * _mxp_cs)
                        for t, s in scores_abs.items()
                    }
            elif _cs_sizing_method == "fmc_hybrid" and cs_entries:
                # FMC × Score hybrid weighting — S&P Global (Innes et al.) p.7, p.21-22.
                # Uses float-adjusted market cap (proxied by 20d avg dollar volume when
                # true market cap data is unavailable) as the liquidity anchor,
                # blended with softmax-score tilt at score_blend ratio.
                _eq_cs = self.portfolio.equity
                _mxp_cs = float(getattr(self.config, "max_position_pct_of_equity", 0.12))
                _blend = float(getattr(self.config, "fmc_score_blend", 0.5))
                tickers_cs = [e["ticker"] for e in cs_entries]
                scores_cs = {e["ticker"]: float(e.get("adjusted_score", 0.0) or 0.0) for e in cs_entries}
                # Proxy FMC via 20d avg dollar volume (close × volume) when market cap unavailable
                market_caps_proxy: dict[str, float] = {}
                for _t in tickers_cs:
                    try:
                        _px = price_data.get(_t)
                        if _px is not None and not _px.empty:
                            _hist = _px.loc[_px.index <= date].tail(20)
                            _close_col = "Close" if "Close" in _hist.columns else "close"
                            _vol_col = "Volume" if "Volume" in _hist.columns else "volume"
                            if _close_col in _hist.columns and _vol_col in _hist.columns:
                                market_caps_proxy[_t] = float(
                                    (_hist[_close_col] * _hist[_vol_col]).mean()
                                )
                    except Exception:
                        pass
                    if _t not in market_caps_proxy:
                        market_caps_proxy[_t] = 1.0
                weights_fmc = construct_hybrid_fmc_score_weights(
                    tickers_cs, scores_cs, market_caps_proxy,
                    score_blend=_blend, max_single_weight=_mxp_cs,
                )
                cs_signal_sizes = {t: _eq_cs * w for t, w in weights_fmc.items()}

            # If circuit breaker is active, do not open new positions today
            if self._trading_halted and pending_entries:
                logger.info(
                    "Trading halted on %s: skipping %d pending entries due to circuit breaker.",
                    date,
                    len(pending_entries),
                )
                pending_entries = []

            # Correlation-aware position selection before entry processing.
            corr_window_days = int(getattr(self.config, "correlation_window_days", 60) or 60)
            corr_threshold = float(getattr(self.config, "max_pairwise_correlation", 0.70) or 0.70)
            if pending_entries and corr_threshold > 0:
                candidate_tickers: list[str] = []
                for e in pending_entries:
                    tk = e.get("ticker")
                    if tk and tk not in candidate_tickers:
                        candidate_tickers.append(tk)

                returns_60d: dict[str, pd.Series] = {}
                for tk in candidate_tickers:
                    if tk in price_data:
                        pxf = price_data[tk]
                        close_col = "close" if "close" in pxf.columns else "Close"
                        if close_col in pxf.columns:
                            p = pd.to_numeric(
                                pxf.loc[pxf.index <= date, close_col],
                                errors="coerce",
                            ).dropna()
                            returns_60d[tk] = p.pct_change(fill_method=None).tail(corr_window_days)

                if len(returns_60d) >= 2:
                    ret_df = pd.DataFrame(returns_60d).dropna()
                    corr_matrix = ret_df.corr() if not ret_df.empty else pd.DataFrame()
                else:
                    corr_matrix = pd.DataFrame()

                sorted_by_signal = sorted(
                    pending_entries,
                    key=lambda e: -abs(float(e.get("adjusted_score", 0.0) or 0.0)),
                )
                selected: list[str] = []
                selected_entries: list[dict] = []
                for candidate in sorted_by_signal:
                    tk = candidate.get("ticker")
                    if not tk:
                        continue
                    if len(selected) == 0:
                        selected.append(tk)
                        selected_entries.append(candidate)
                        continue

                    max_corr = 0.0
                    if not corr_matrix.empty:
                        for already in selected:
                            if tk in corr_matrix.index and already in corr_matrix.columns:
                                c = abs(float(corr_matrix.loc[tk, already]))
                                if np.isfinite(c):
                                    max_corr = max(max_corr, c)

                    if max_corr <= corr_threshold:
                        selected.append(tk)
                        selected_entries.append(candidate)

                pending_entries = selected_entries

            vol_scalar_today = float(vol_scalar_series.loc[date]) if date in vol_scalar_series.index else 1.0
            # Gross exposure caps (D1: continuous regime score → linear interpolation)
            # Precompute Sharpe CB params once per day so both gross_cap block and CB block share values.
            sharpe_cb_window = int(getattr(self.config, "sharpe_cb_window_days", 60) or 60)
            sharpe_cb_thr = float(getattr(self.config, "sharpe_cb_threshold", 0.0) or 0.0)
            sharpe_cb_recovery = float(getattr(self.config, "sharpe_cb_recovery_threshold", 0.3) or 0.3)
            sharpe_cb_scale = float(getattr(self.config, "sharpe_cb_exposure_scale", 0.5) or 0.5)
            sharpe_cb_enabled = bool(getattr(self.config, "sharpe_cb_enabled", True))

            _rs = float((regime_score_data or {}).get(date, float("nan")))
            if np.isfinite(_rs) and bool(getattr(self.config, "regime_continuous_score_enabled", True)):
                # D1: linear interpolation between bull cap and crisis cap using continuous score
                _bull_cap = float(getattr(self.config, "regime_score_bull_gross_cap", 1.0))
                _crisis_cap = float(getattr(self.config, "regime_score_crisis_gross_cap", 0.25))
                gross_cap_fraction_today = _bull_cap + _rs * (_crisis_cap - _bull_cap)
            else:
                # Fallback: hard regime labels (original behavior)
                gross_cap_fraction_today = 1.0
                if regime_today == "Crisis":
                    gross_cap_fraction_today = float(
                        getattr(self.config, "crisis_gross_cap_fraction", 0.6) or 0.6
                    )
                elif regime_today == "Cautious":
                    # State Street "Cautious Decline": between Sideways and Bear
                    gross_cap_fraction_today = float(
                        getattr(self.config, "cautious_gross_cap_fraction", 0.65) or 0.65
                    )
                elif regime_today == "Bear":
                    gross_cap_fraction_today = float(
                        getattr(self.config, "bear_gross_cap_fraction", 0.7) or 0.7
                    )

            # D3: Apply Sharpe CB scaling (stacks on top of regime gross cap)
            if self._sharpe_cb_active:
                gross_cap_fraction_today = gross_cap_fraction_today * sharpe_cb_scale

            # Precompute signal-score pool for volatility-scaled sizing (non-cross-sectional entries only).
            non_cs_entries = [
                e for e in pending_entries
                if not e.get("_cross_sectional")
            ]
            score_pool = []
            for e in non_cs_entries:
                # Respect long-only constraint when building the score pool.
                score = float(e.get("adjusted_score", 0.0) or 0.0)
                if getattr(self.config, "long_only", False):
                    if score <= 0 or e.get("signal") == "Bearish":
                        continue
                score_pool.append(abs(score))
            sum_abs_scores = float(sum(score_pool)) if score_pool else 0.0

            # Rebuild existing_tickers from current positions (after all exits) so closed
            # positions do not block re-entry on the same day.
            existing_tickers = {p.ticker for p in self.portfolio.positions}

            # Track effective universe size for this day
            active_tickers_today = [tk for tk, px in price_data.items() if date in px.index]
            universe_sizes.append(len(active_tickers_today))

            # Regime gross exposure cap enforcement (on trade rebalance days).
            if is_trade_rebalance_day_current and gross_cap_fraction_today < 0.999:
                equity_now = float(self.portfolio.equity) if self.portfolio.equity else 0.0
                max_gross_value_now = gross_cap_fraction_today * equity_now
                gross_now = sum(abs(p.market_value) for p in self.portfolio.positions)
                if equity_now > 1e-12 and gross_now > max_gross_value_now * 1.0001:
                    # Close largest positions first to get back under the cap.
                    positions_sorted = sorted(
                        list(self.portfolio.positions),
                        key=lambda p: abs(float(p.market_value)),
                        reverse=True,
                    )
                    for pos in positions_sorted:
                        gross_now = sum(abs(p.market_value) for p in self.portfolio.positions)
                        if gross_now <= max_gross_value_now * 1.0001:
                            break
                        held_days = _held_trading_days(pos, i)
                        min_hold_eff = _min_hold_effective_for_pos(pos)
                        if min_hold_eff > 0 and held_days < min_hold_eff:
                            if _drawdown_exceeded_early_stop(pos, pos.ticker, date):
                                self._close_position(
                                    pos,
                                    date,
                                    price_data,
                                    exit_price_override=_threshold_exit_price_for_stop(pos),
                                    reason="stop_loss",
                                )
                            continue
                        self._close_position(pos, date, price_data, reason="gross_cap_reduction")

            # --- No-trade band for position updates (reduce churn) ---
            # If a ticker is already held, we suppress re-entry scheduling when the
            # candidate's implied target weight is close to the current weight.
            # This prevents "replace everything" behaviour when signals jitter.
            no_trade_enabled = bool(getattr(self.config, "no_trade_band_rebalance_enabled", True))
            per_weight_diff = float(getattr(self.config, "no_trade_band_weight_diff", 0.015) or 0.015)
            portfolio_drift_thr = float(getattr(self.config, "no_trade_band_total_drift", 0.05) or 0.05)
            existing_pos_by_ticker = {p.ticker: p for p in self.portfolio.positions}
            equity_now = float(self.portfolio.equity) if self.portfolio.equity else 0.0
            implied_weight_diffs: dict[str, float] = {}
            portfolio_total_drift = 0.0
            if no_trade_enabled and equity_now > 1e-12:
                for entry in pending_entries:
                    tk = entry.get("ticker")
                    if not tk:
                        continue
                    if tk not in existing_pos_by_ticker:
                        continue
                    pos = existing_pos_by_ticker[tk]
                    current_w = abs(float(pos.market_value)) / equity_now
                    try:
                        # Match the *actual* target sizing path used in the open
                        # loop so the no-trade band gates re-entry correctly.
                        if entry.get("_cross_sectional"):
                            # Cross-sectional entries use precomputed `equal_size`.
                            target_size = min(float(equal_size or 0.0), equity_now * 0.99)
                        else:
                            target_size = float(
                                self._compute_position_size(entry, price_data, date)
                            )
                    except Exception:
                        # If we cannot compute target weight reliably, do not suppress.
                        target_size = float(pos.position_size)
                    target_w = abs(target_size) / equity_now if equity_now > 0 else 0.0
                    diff = abs(target_w - current_w)
                    # If multiple candidate entries exist for the same ticker, keep the max diff.
                    prev = implied_weight_diffs.get(tk)
                    if prev is None or diff > prev:
                        implied_weight_diffs[tk] = diff
                portfolio_total_drift = sum(
                    d for d in implied_weight_diffs.values() if d > per_weight_diff
                )
            allow_rebalance = (not no_trade_enabled) or (equity_now <= 1e-12) or (portfolio_total_drift > portfolio_drift_thr)

            # Rolling-std gate: threshold = scm × σ × regime_threshold_aggressiveness (per day).
            scm = self._signal_confidence_multiplier_for_regime(regime_today)
            th_agg = self._threshold_aggressiveness_for_regime(regime_today)
            if (
                regime_today == "Sideways"
                and pending_entries
                and scm is not None
                and float(scm) > 0
            ):
                logger.debug(
                    "SIDEWAYS: rolling-std gate scm=%.2f, aggressiveness=%.2f (threshold = scm×σ×aggr)",
                    float(scm),
                    float(th_agg),
                )

            for entry in pending_entries:
                tk = entry["ticker"]
                max_trades = int(getattr(self.config, "max_trades_per_ticker", 0) or 0)
                if max_trades > 0 and self.ticker_trade_counts.get(tk, 0) >= max_trades:
                    continue
                # Shorts gate: do not open short trades when allow_shorts is False.
                if entry.get("signal") == "Bearish" and not getattr(self.config, "allow_shorts", False):
                    continue
                # Long-only mode: skip Bearish; allow Bullish even with score <= 0 (we'll use fallback sizing).
                score = float(entry.get("adjusted_score", 0.0) or 0.0)
                if getattr(self.config, "long_only", False):
                    if entry.get("signal") == "Bearish":
                        continue

                # Regime-aware signal filtering: suppress shorts in configured regimes.
                _suppress = getattr(self.config, "regime_suppress_shorts", ["Bull", "Crisis"])
                if regime_today in _suppress and entry.get("signal") == "Bearish":
                    continue

                # Bear regime: block NEW LONG entries — no momentum edge in confirmed Bear.
                # Shorts are still allowed (regime_suppress_shorts controls that separately).
                # Existing positions run to their signal-based planned_exit_date (no forced close).
                if (
                    regime_today == "Bear"
                    and entry.get("signal") != "Bearish"
                    and bool(getattr(self.config, "bear_skip_new_entries", True))
                ):
                    continue

                # Crisis: block all new entries — HMM-assigned Crisis means systemic stress.
                # The entry gate replaces panic-liquidation: don't open new risk, let existing
                # positions exit naturally via planned_exit_date (capped by crisis_transition_accel).
                if regime_today == "Crisis" and bool(
                    getattr(self.config, "crisis_block_all_new_entries", True)
                ):
                    continue

                # Crisis day 1-3 hard block:
                # avoid opening fresh risk in the initial shock window.
                if regime_today == "Crisis" and crisis_consecutive_days <= 3:
                    crisis_entries_blocked_days_1_3 += 1
                    continue

                # Crisis (day >= 4): selective entry (top conviction + low rolling-beta).
                if regime_today == "Crisis":
                    entry_score = float(entry.get("adjusted_score", 0.0) or 0.0)

                    # 1) Top 5% of adjusted_score over a rolling 60 trading-day window.
                    sig_df = signal_data.get(tk) if isinstance(signal_data, dict) else None
                    if sig_df is not None and not sig_df.empty and "adjusted_score" in sig_df.columns:
                        try:
                            s = pd.to_numeric(
                                sig_df.loc[sig_df.index <= date, "adjusted_score"],
                                errors="coerce",
                            ).dropna()
                            if len(s) >= max(20, int(crisis_signal_window_days * 0.5)):
                                window = s.tail(crisis_signal_window_days)
                                thr = float(window.quantile(crisis_signal_quantile))
                                if entry_score < thr:
                                    continue
                        except Exception:
                            pass

                    # 2) Only low rolling-beta stocks (CAPM 60d beta).
                    rolling_beta_60d = float(entry.get("capm_beta", 0.0) or 0.0)
                    if np.isfinite(rolling_beta_60d) and rolling_beta_60d > crisis_beta_cutoff:
                        continue

                # Bear regime: only enter longs above rolling-window score quantile — longs have weak edge in Bear.
                # Apply regardless of long_only setting — shorts are handled separately by cross-sectional ranking.
                if regime_today == "Bear":
                    if entry.get("signal") == "Bullish":
                        reg_entry = str(entry.get("regime") or regime_today)
                        smult = float(
                            self.config.regime_adjustments.get(reg_entry, {}).get("score_mult", 1.0)
                            or 1.0
                        )
                        entry_adj = float(entry.get("adjusted_score", 0.0) or 0.0)
                        raw_score = entry_adj / smult if abs(smult) > 1e-12 else entry_adj
                        sig_df = signal_data.get(tk) if isinstance(signal_data, dict) else None
                        if sig_df is not None and not sig_df.empty and "adjusted_score" in sig_df.columns:
                            try:
                                s = pd.to_numeric(
                                    sig_df.loc[sig_df.index <= date, "adjusted_score"],
                                    errors="coerce",
                                ).dropna()
                                if len(s) >= max(20, int(bear_signal_window_days * 0.5)):
                                    window = s.tail(bear_signal_window_days)
                                    thr = float(window.quantile(bear_signal_quantile))
                                    if raw_score < thr:
                                        continue
                            except Exception:
                                pass

                # Optional: rolling score volatility confidence — skip weak signals vs recent dispersion.
                if scm is not None and float(scm) > 0:
                    sc_window = int(getattr(self.config, "signal_confidence_std_window", 60) or 60)
                    sc_min = int(getattr(self.config, "signal_confidence_min_periods", 20) or 20)
                    sig_df_conf = signal_data.get(tk) if isinstance(signal_data, dict) else None
                    if sig_df_conf is not None and not sig_df_conf.empty and "adjusted_score" in sig_df_conf.columns:
                        try:
                            s_hist = pd.to_numeric(
                                sig_df_conf.loc[sig_df_conf.index <= date, "adjusted_score"],
                                errors="coerce",
                            ).dropna()
                            rs = float("nan")
                            if len(s_hist) >= sc_min:
                                tail = s_hist.tail(sc_window)
                                rs = float(tail.std(ddof=1))
                            else:
                                # Fragmented regimes can have sparse per-ticker history.
                                # Fall back to global regime-level score std.
                                reg_key = str(regime_today or "Sideways")
                                rs = float((getattr(self, "_regime_score_std", {}) or {}).get(reg_key, np.nan))

                            reg_e = str(entry.get("regime") or regime_today)
                            smult_c = float(
                                self.config.regime_adjustments.get(reg_e, {}).get("score_mult", 1.0)
                                or 1.0
                            )
                            adj_e = float(entry.get("adjusted_score", 0.0) or 0.0)
                            raw_abs = abs(adj_e / smult_c) if abs(smult_c) > 1e-12 else abs(adj_e)
                            if np.isfinite(rs) and rs > 1e-12:
                                base_threshold = float(scm) * rs
                                final_threshold = base_threshold * float(th_agg)
                                if regime_today == "Sideways":
                                    logger.debug(
                                        "SIDEWAYS GATE: ticker=%s, raw_abs=%.4f, base_thresh=%.4f, "
                                        "aggressiveness=%.4f, final_thresh=%.4f, enter=%s",
                                        tk,
                                        raw_abs,
                                        base_threshold,
                                        float(th_agg),
                                        final_threshold,
                                        str(bool(raw_abs >= final_threshold)),
                                    )
                                if raw_abs < final_threshold:
                                    continue
                        except Exception:
                            pass

                if tk in existing_tickers:
                    # Reschedule for next trading day if still within exit window,
                    # but suppress when weight changes are small (no-trade band).
                    diff = implied_weight_diffs.get(tk, 0.0)
                    if no_trade_enabled and equity_now > 1e-12:
                        if (diff <= per_weight_diff) or (not allow_rebalance):
                            continue

                    next_at = i + 1
                    if next_at < len(trading_days):
                        original_exit = entry.get("exit_date")
                        next_date = trading_days[next_at]
                        if original_exit is None or next_date <= pd.Timestamp(original_exit):
                            scheduled_entries.append((next_at, entry))
                    continue
                if tk not in price_data or date not in price_data[tk].index:
                    next_at = i + 1
                    if next_at < len(trading_days):
                        original_exit = entry.get("exit_date")
                        next_date = trading_days[next_at]
                        if original_exit is None or next_date <= pd.Timestamp(original_exit):
                            scheduled_entries.append((next_at, entry))
                    continue
                if self.portfolio.available_slots <= 0:
                    # Reschedule all remaining entries for tomorrow instead of dropping them
                    idx = pending_entries.index(entry)
                    for rem in pending_entries[idx:]:
                        next_at = i + 1
                        if next_at < len(trading_days):
                            scheduled_entries.append((next_at, rem))
                    break

                open_price = float(price_data[tk].loc[date, "Open"])
                if not np.isfinite(open_price) or open_price <= 0:
                    next_at = i + 1
                    if next_at < len(trading_days):
                        original_exit = entry.get("exit_date")
                        next_date = trading_days[next_at]
                        if original_exit is None or next_date <= pd.Timestamp(original_exit):
                            scheduled_entries.append((next_at, entry))
                    continue
                entry_price = self.execution.apply_entry_slippage_scaled(
                    open_price, entry["signal"], self._cost_scalar_today
                )

                if entry.get("_cross_sectional"):
                    _mxp_cs = float(getattr(self.config, "max_position_pct_of_equity", 0.12))
                    if cs_signal_sizes is not None and tk in cs_signal_sizes:
                        # Signal-strength sizing: larger allocation for higher-conviction entries
                        size_dollars = cs_signal_sizes[tk]
                    else:
                        size_dollars = min(equal_size or 0, self.portfolio.equity * 0.99)
                    size_dollars = min(size_dollars, self.portfolio.equity * _mxp_cs)
                    # Enforce regime gross cap (including already-open positions).
                    if gross_cap_fraction_today < 0.999:
                        equity = self.portfolio.equity
                        gross_exposure = sum(abs(p.market_value) for p in self.portfolio.positions)
                        max_gross = gross_cap_fraction_today * equity
                        remaining = max(0.0, max_gross - gross_exposure)
                        size_dollars = min(size_dollars, remaining) if remaining > 0 else 0.0
                else:
                    method = getattr(self.config, "position_sizing_method", None) or getattr(
                        self.config, "position_sizing", "equal"
                    )
                    if method in ("mv_max_sharpe", "mv_min_variance"):
                        mv_weights = getattr(self, "_mv_weights", {}) or {}
                        equity = self.portfolio.equity
                        max_pos = self.portfolio.max_positions
                        w = mv_weights.get(tk, 1.0 / max_pos if max_pos else 0.1)
                        size_dollars = equity * float(w)
                        cap = equity * getattr(self.config, "max_position_pct_of_equity", 0.25)
                        if cap > 0:
                            size_dollars = min(size_dollars, cap)
                        size_dollars = max(0.0, size_dollars)
                        gross_exposure = sum(abs(p.market_value) for p in self.portfolio.positions)
                        remaining = max(0.0, gross_cap_fraction_today * equity - gross_exposure)
                        size_dollars = min(size_dollars, remaining) if remaining > 0 else 0.0
                        if regime_today == "Sideways":
                            size_dollars = min(size_dollars, equity * 0.05)
                        if regime_today == "Crisis":
                            size_dollars *= 1.0
                    elif method in ("equal", "kelly"):
                        size_dollars = self._compute_position_size(entry, price_data, date)
                        equity = self.portfolio.equity
                        gross_exposure = sum(abs(p.market_value) for p in self.portfolio.positions)
                        remaining = max(0.0, gross_cap_fraction_today * equity - gross_exposure)
                        size_dollars = min(size_dollars, remaining) if remaining > 0 else 0.0
                        if regime_today == "Sideways":
                            size_dollars = min(size_dollars, equity * 0.05)
                        if regime_today == "Crisis":
                            size_dollars *= 1.0
                    else:
                        # Volatility-scaled position sizing based on signal strength, stock volatility, and regime.
                        score = float(entry.get("adjusted_score", 0.0) or 0.0)
                        if sum_abs_scores is None or sum_abs_scores <= 0:
                            # Fallback to equal-weight so entries still open when score pool is empty (e.g. long_only filtered all)
                            equity = self.portfolio.equity
                            size_dollars = equity / self.config.max_positions
                            _mx_pct = float(getattr(self.config, "max_position_pct_of_equity", 0.12))
                            single_cap = _mx_pct * equity
                            if regime_today == "Sideways":
                                single_cap = min(single_cap, 0.05 * equity)
                            size_dollars = min(size_dollars, single_cap)
                            gross_exposure = sum(abs(p.market_value) for p in self.portfolio.positions)
                            remaining = max(0.0, gross_cap_fraction_today * equity - gross_exposure)
                            size_dollars = min(size_dollars, remaining)
                            if regime_today == "Crisis":
                                size_dollars *= 1.0
                        else:
                            # Realised volatility (20d) annualized
                            vol_annual = self._annualized_vol(
                                price_data,
                                tk,
                                date,
                                getattr(self.config, "vol_lookback_days", 20),
                            )
                            if vol_annual is None or vol_annual <= 1e-8:
                                # Fallback to equal-weight so missing vol does not block entry
                                equity = self.portfolio.equity
                                size_dollars = equity / self.config.max_positions
                                _mx_pct = float(getattr(self.config, "max_position_pct_of_equity", 0.12))
                                single_cap = _mx_pct * equity
                                if regime_today == "Sideways":
                                    single_cap = min(single_cap, 0.05 * equity)
                                size_dollars = min(size_dollars, single_cap)
                                gross_exposure = sum(abs(p.market_value) for p in self.portfolio.positions)
                                remaining = max(0.0, gross_cap_fraction_today * equity - gross_exposure)
                                size_dollars = min(size_dollars, remaining)
                                if regime_today == "Crisis":
                                    size_dollars *= 1.0
                            else:
                                target_vol = float(getattr(self.config, "vol_target_annual", 0.10) or 0.10)
                                risk_weight = abs(score) / sum_abs_scores
                                equity = self.portfolio.equity
                                # position_size_i ≈ equity * ( |score_i| / sum|scores| ) * (target_vol / vol_i )
                                raw_size = equity * risk_weight * (target_vol / max(vol_annual, 1e-3))
                                # Single-position cap from config; Sideways capped tighter at 5%.
                                _mx_pct = float(getattr(self.config, "max_position_pct_of_equity", 0.12))
                                single_cap = _mx_pct * equity
                                if regime_today == "Sideways":
                                    single_cap = min(single_cap, 0.05 * equity)
                                size_dollars = min(raw_size, single_cap)

                                # Cap gross exposure (sum |positions|) at 150% of equity.
                                gross_exposure = sum(abs(p.market_value) for p in self.portfolio.positions)
                                max_gross = gross_cap_fraction_today * equity
                                remaining = max_gross - gross_exposure
                                if remaining <= 0:
                                    size_dollars = 0.0  # no room, will reschedule below
                                else:
                                    size_dollars = min(size_dollars, remaining)
                                    # Ensure we don't zero out due to rounding: use at least a small equal-weight slice
                                    min_size = (equity / self.config.max_positions) * 0.5
                                    if (
                                        size_dollars > 0
                                        and size_dollars < min_size
                                        and remaining >= min_size
                                    ):
                                        size_dollars = min(min_size, remaining)

                                # In Crisis regime, further cut all position sizes by 50%.
                                if regime_today == "Crisis":
                                    size_dollars *= 1.0

                if size_dollars is None or size_dollars <= 0:
                    # Last-resort: we have slots and entry passed all gates; use minimum size so we actually open
                    if self.portfolio.available_slots > 0:
                        equity = self.portfolio.equity
                        gross = sum(abs(p.market_value) for p in self.portfolio.positions)
                        remaining = max(0.0, gross_cap_fraction_today * equity - gross)
                        _mx_pct = float(getattr(self.config, "max_position_pct_of_equity", 0.12))
                        size_dollars = min(
                            equity / self.config.max_positions,
                            equity * _mx_pct,
                            remaining,
                        )
                        if regime_today == "Crisis":
                            size_dollars *= 1.0
                    if size_dollars is None or size_dollars <= 0:
                        next_at = i + 1
                        if next_at < len(trading_days):
                            original_exit = entry.get("exit_date")
                            next_date = trading_days[next_at]
                            if original_exit is None or next_date <= pd.Timestamp(original_exit):
                                scheduled_entries.append((next_at, entry))
                        continue
                    # else: we set size_dollars in last-resort, fall through to sector cap and open_position

                # D2: short book sizing scale — size shorts at a fraction of long equivalent
                if entry.get("signal") == "Bearish":
                    _short_scale = float(getattr(self.config, "ml_short_position_scale", 0.7))
                    if size_dollars is not None and size_dollars > 0:
                        size_dollars = size_dollars * _short_scale

                # Beta adjustment: scale down high-beta positions to keep portfolio beta ~1
                beta = float(entry.get("capm_beta", 0.0) or 0.0)
                beta = max(0.5, min(beta, 2.5))
                target_beta = 1.0
                beta_scalar = target_beta / beta
                size_dollars = size_dollars * beta_scalar

                # Apply volatility scaling after all other size adjustments
                # so it truly affects the final order sizing.
                if size_dollars is not None and size_dollars > 0:
                    size_dollars *= vol_scalar_today

                # Re-apply gross cap after beta/vol adjustments (beta_scalar can
                # otherwise push us above the regime cap).
                if gross_cap_fraction_today < 0.999:
                    equity = self.portfolio.equity
                    gross_exposure = sum(abs(p.market_value) for p in self.portfolio.positions)
                    max_gross = gross_cap_fraction_today * equity
                    remaining = max_gross - gross_exposure
                    if remaining <= 0:
                        size_dollars = 0.0
                    else:
                        size_dollars = min(size_dollars, remaining)

                # Final hard cap vs equity (beta/vol scaling must not exceed max_position_pct_of_equity).
                _eq_mx = float(self.portfolio.equity)
                _pct_mx = float(getattr(self.config, "max_position_pct_of_equity", 0.12))
                if _eq_mx > 1e-12 and size_dollars is not None and float(size_dollars) > 0:
                    size_dollars = min(float(size_dollars), _eq_mx * _pct_mx)

                # Sector capital exposure cap (fraction of equity/capital)
                if self.config.sector_enabled:
                    sector = SECTOR_MAP.get(tk, "Other")
                    sector_exposure = sum(
                        p.market_value
                        for p in self.portfolio.positions
                        if SECTOR_MAP.get(p.ticker, "Other") == sector
                    )
                    projected_exposure = sector_exposure + size_dollars
                    total_capital = max(self.portfolio.equity, self.portfolio.initial_capital)
                    max_pct = float(getattr(self.config, "max_sector_exposure_pct", 0.3))
                    if total_capital > 0 and (projected_exposure / total_capital) > max_pct:
                        # Reschedule for next day instead of dropping
                        next_at = i + 1
                        if next_at < len(trading_days):
                            original_exit = entry.get("exit_date")
                            next_date = trading_days[next_at]
                            if original_exit is None or next_date <= pd.Timestamp(original_exit):
                                scheduled_entries.append((next_at, entry))
                        continue

                size_for_cost = size_dollars if size_dollars and size_dollars > 0 else (self.portfolio.equity / self.config.max_positions)

                # Base transaction cost
                base_cost = self.cost_model.cost_dollars(size_for_cost) if self.cost_model else self.execution.commission

                # Almgren-Chriss market impact (entry leg)
                # Replaces ad-hoc k_bps * sqrt(shares/adv) with calibrated A-C model.
                impact_cost = 0.0
                if self._market_impact_model is not None and size_for_cost > 0:
                    try:
                        _adv = self._compute_adv_usd(tk, price_data, date)
                        _vol = self._compute_vol_daily(tk, price_data, date)
                        _mi = MarketImpactModel(adv_usd=_adv, params=self._market_impact_model.params)
                        impact_cost = _mi.cost_dollars(size_for_cost, _vol, is_exit=False)
                    except Exception:
                        impact_cost = 0.0
                elif self._market_impact_model is None:
                    # Legacy fallback: ad-hoc shares-based square-root model
                    try:
                        adv_window = price_data[tk]["Volume"].loc[price_data[tk].index <= date].tail(20)
                        adv = float(adv_window.mean())
                        if adv > 0 and size_dollars and size_dollars > 0 and entry_price > 0:
                            shares = size_dollars / entry_price
                            rel_size = max(shares / adv, 0.0)
                            if rel_size > 0:
                                k_bps = float(getattr(self.config, "market_impact_k_bps", 10.0))
                                impact_bps = k_bps * (rel_size ** 0.5)
                                impact_cost = size_for_cost * (impact_bps / 10_000.0)
                    except Exception:
                        impact_cost = 0.0

                entry_cost = base_cost + impact_cost

                # Ensure we pass positive size when we have slots (avoid open_position returning None for size<=0)
                size_to_use = size_dollars if size_dollars and size_dollars > 0 else None
                if size_to_use is None and self.portfolio.available_slots > 0:
                    equity = self.portfolio.equity
                    gross = sum(abs(p.market_value) for p in self.portfolio.positions)
                    remaining = max(0.0, gross_cap_fraction_today * equity - gross)
                    _pct_mx = float(getattr(self.config, "max_position_pct_of_equity", 0.12))
                    size_to_use = min(
                        equity / self.config.max_positions,
                        equity * _pct_mx,
                        remaining,
                    )
                    if regime_today == "Crisis":
                        size_to_use *= 1.0
                size_to_use = size_to_use if size_to_use and size_to_use > 0 else None

                # Portfolio only opens Bullish/Bearish; treat strong Neutral as Bullish when long_only
                signal_for_open = entry["signal"]
                if signal_for_open == "Neutral" and getattr(self.config, "long_only", False):
                    signal_for_open = "Bullish"

                pos = self.portfolio.open_position(
                    ticker=tk,
                    signal=signal_for_open,
                    signal_date=entry["signal_date"],
                    entry_date=date,
                    planned_exit_date=entry["exit_date"],
                    original_planned_exit_date=entry.get("original_exit_date", entry["exit_date"]),
                    entry_price=entry_price,
                    adjusted_score=entry["adjusted_score"],
                    confidence=entry["confidence"],
                    regime=entry["regime"],
                    entry_cost=entry_cost,
                    impact_entry_cost=impact_cost,
                    position_scale=entry.get("position_scale", 1.0),
                    size_dollars=size_to_use,
                    max_position_pct_of_equity=float(
                        getattr(self.config, "max_position_pct_of_equity", 0.12)
                    ),
                )
                if pos:
                    self.ticker_trade_counts[tk] = self.ticker_trade_counts.get(tk, 0) + 1
                    # Stamp ATR at entry so the ATR stop-loss has its reference level.
                    if float(getattr(self.config, "atr_stop_lambda", 0.0) or 0.0) > 0:
                        pos.entry_atr = self._compute_atr(tk, price_data, date)

                if pos is None:
                    # Reschedule so we retry tomorrow (e.g. entry_price was 0 or size 0 this bar)
                    next_at = i + 1
                    if next_at < len(trading_days):
                        original_exit = entry.get("exit_date")
                        next_date = trading_days[next_at]
                        if original_exit is None or next_date <= pd.Timestamp(original_exit):
                            scheduled_entries.append((next_at, entry))
                    continue
                existing_tickers.add(tk)

            pending_entries = []  # will be refilled below from new_entries (or scheduled for delay)

            # --- 3. Mark-to-market ---
            is_first_crisis_day = (
                regime_today == "Crisis"
                and prev_reg_before is not None
                and prev_reg_before != "Crisis"
            )
            to_remove = []
            for pos in self.portfolio.positions:
                tk = pos.ticker
                if tk not in price_data or date not in price_data[tk].index:
                    # Delisting detected: data exists for some tickers but not this one today.
                    logger.warning("Delisting/Gap detected for %s on %s; force-closing.", tk, date)
                    pos.exit_reason = "delisted"
                    to_remove.append(pos)
                    continue

                bar = price_data[pos.ticker].loc[date]
                close_px = float(bar["Close"])
                use_open_mtm = False
                if regime_today == "Crisis" and getattr(pos, "crisis_accelerated_exit", False):
                    if bool(getattr(self.config, "crisis_accelerated_mtm_at_open", False)):
                        use_open_mtm = True
                    elif is_first_crisis_day and bool(
                        getattr(self.config, "crisis_first_day_winner_mtm_at_open", False)
                    ):
                        use_open_mtm = True
                if use_open_mtm:
                    o = float(bar.get("Open", np.nan))
                    pos.current_price = o if np.isfinite(o) and o > 0 else close_px
                else:
                    pos.current_price = close_px

            for pos in to_remove:
                self._close_position(pos, date, price_data, reason=pos.exit_reason)

            # --- 3.5. Volatility-based position scaling (SPY vol target) ---
            # Apply the rule: scale all position sizes by min(1, 1/vol_ratio).
            # This happens after mark-to-market at the daily close so we resize
            # *existing* exposures (not just new entries).
            if vol_scalar_today < 0.999:
                for pos in self.portfolio.positions:
                    old_market_value = float(pos.market_value)
                    new_market_value = old_market_value * vol_scalar_today
                    if old_market_value > 0 and new_market_value >= 0:
                        # Free the difference back to cash; equity stays unchanged
                        # at this timestamp, while future P&L scales down.
                        self.portfolio.cash += (old_market_value - new_market_value)
                        pos.position_size *= vol_scalar_today
                        pos.shares = pos.position_size / pos.entry_price if pos.entry_price > 0 else 0.0
                        # If we're effectively reducing exposure without logging
                        # an actual trade, we must proportionally reduce the
                        # pre-charged entry costs so later pnl subtraction
                        # doesn't over-penalize.
                        pos.entry_cost *= vol_scalar_today
                        pos.impact_entry_cost *= vol_scalar_today

            # Enforce regime gross exposure cap continuously (after vol scaling).
            # For long-only, `invested` ~ gross exposure. We still use abs(...) to
            # keep behaviour consistent if shorts are enabled later.
            if gross_cap_fraction_today < 0.999:
                equity_now = float(self.portfolio.equity)
                gross_now = sum(abs(p.market_value) for p in self.portfolio.positions)
                max_gross_value = gross_cap_fraction_today * equity_now
                if equity_now > 1e-12 and gross_now > max_gross_value * 1.0001 and gross_now > 0:
                    cap_scalar = max_gross_value / gross_now
                    for pos in self.portfolio.positions:
                        old_market_value = float(pos.market_value)
                        new_market_value = old_market_value * cap_scalar
                        if old_market_value > 0 and new_market_value >= 0:
                            self.portfolio.cash += (old_market_value - new_market_value)
                            pos.position_size *= cap_scalar
                            pos.shares = pos.position_size / pos.entry_price if pos.entry_price > 0 else 0.0
                            # Adjust stored costs for remaining notional.
                            pos.entry_cost *= cap_scalar
                            pos.impact_entry_cost *= cap_scalar

            # --- 4. Record daily equity ---
            regime = regime_data.get(date, "Sideways")
            # Short borrow cost drag (applied as daily cash charge on short notional).
            short_borrow_cost = 0.0
            borrow_bps = float(getattr(self.config, "short_borrow_cost_bps", 0.0) or 0.0)
            equity_now = float(self.portfolio.equity) if self.portfolio.equity else 0.0
            # Use shares*price (not Position.market_value) for exposure/borrow calculations.
            def _notional(pos) -> float:
                try:
                    return abs(float(pos.shares) * float(pos.current_price))
                except Exception:
                    return abs(float(getattr(pos, "position_size", 0.0) or 0.0))
            gross_exposure = (
                sum(_notional(p) for p in self.portfolio.positions) / equity_now
                if equity_now > 1e-12
                else 0.0
            )
            net_exposure = (
                sum(float(p.direction) * _notional(p) for p in self.portfolio.positions) / equity_now
                if equity_now > 1e-12
                else 0.0
            )
            if borrow_bps > 0 and self.portfolio.positions:
                daily_rate = (borrow_bps / 10_000.0) / 252.0
                short_notional = sum(_notional(p) for p in self.portfolio.positions if int(getattr(p, "direction", 1)) < 0)
                short_borrow_cost = float(short_notional) * float(daily_rate)
                if short_borrow_cost > 0:
                    self.portfolio.cash -= short_borrow_cost
            self.portfolio.record_equity(
                date,
                regime,
                crisis_consecutive_days=int(crisis_consecutive_days),
                gross_exposure=gross_exposure,
                net_exposure=net_exposure,
                short_borrow_cost=short_borrow_cost if short_borrow_cost > 0 else 0.0,
            )

            # --- 5. Queue new signals (only inside backtest window) ---
            date_key = _to_calendar_key(date)
            if date > end_ts or date_key not in daily_signals:
                continue
            # Rebalance cadence:
            # Signals are only allowed to generate new entries when the
            # resulting execution day (i+1+delay) lands on a trade
            # rebalance day. This prevents turnover on non-rebalance days.
            if not is_trade_rebalance_day_next:
                continue

            signals_at_date = daily_signals[date_key]
            if factor_neutralizer is not None and signals_at_date:
                neutralized, diag = factor_neutralizer.neutralize(
                    date, signals_at_date, price_data, collect_diagnostics=True
                )
                daily_signals[date_key] = neutralized
                if diag:
                    factor_diagnostics.append(diag)

            if cross_sectional:
                new_entries, log_rows = build_cross_sectional_candidates(
                    date, daily_signals[date_key], self.config, trading_days, i, regime,
                    sector_map=SECTOR_MAP if getattr(self.config, "sector_enabled", False) else None,
                )
                n = len(new_entries)
                eq = self.portfolio.equity
                per = round(eq / n, 2) if n else 0.0
                for r in log_rows:
                    r["position_size"] = per
                daily_allocation_rows.extend(log_rows)

                # Factor Imbalance diagnostic — S&P Global (Innes et al.) p.16.
                # Measures whether one factor dominates the composite score today.
                # High imbalance (>1.0) means the portfolio is implicitly single-factor.
                if new_entries and getattr(self.config, "log_factor_imbalance", True):
                    _fi_scores: dict[str, float] = {}
                    for _sig_ticker, _sig_row in daily_signals[date_key]:
                        _fi_scores["adjusted_score"] = _fi_scores.get("adjusted_score", 0.0) + float(_sig_row.get("adjusted_score", 0.0) or 0.0)
                        _fi_scores["trend_score"] = _fi_scores.get("trend_score", 0.0) + float(_sig_row.get("trend_score", 0.0) or 0.0)
                        _fi_scores["short_score"] = _fi_scores.get("short_score", 0.0) + float(_sig_row.get("short_score_raw", 0.0) or 0.0)
                    n_sigs = max(len(daily_signals[date_key]), 1)
                    _fi_avg = {k: v / n_sigs for k, v in _fi_scores.items()}
                    _imbalance = compute_factor_imbalance(_fi_avg)
                    if _imbalance > 1.0:
                        logger.warning(
                            "Factor imbalance %.2f on %s — one factor dominates composite (threshold 1.0).",
                            _imbalance, date.date(),
                        )
                # Always execute cross-sectional entries at the next trading day
                # (plus any configured execution_delay_days) at that day's open.
                # Do not schedule new entries for tickers we already hold (re-entry only after expiry).
                execute_at = i + 1 + execution_delay_days
                if execute_at < len(trading_days) and trading_days[execute_at] <= end_ts:
                    currently_open = {p.ticker for p in self.portfolio.positions}
                    for e in new_entries:
                        if e["ticker"] not in currently_open:
                            scheduled_entries.append((execute_at, e))
            else:
                # Diversification: prefer tickers with fewer trades so more names get filled
                ticker_trade_counts: dict[str, int] = {}
                for r in self.portfolio.trade_log:
                    t = r.get("ticker")
                    if t:
                        ticker_trade_counts[t] = ticker_trade_counts.get(t, 0) + 1
                new_entries = build_ranked_candidates(
                    date, daily_signals[date_key], self.config, trading_days, i, regime,
                    ticker_trade_counts=ticker_trade_counts,
                )
                # For ranked candidates, also execute at next trading day open
                # (plus any configured execution_delay_days).
                # Do not schedule new entries for tickers we already hold (re-entry only after expiry).
                execute_at = i + 1 + execution_delay_days
                if execute_at < len(trading_days) and trading_days[execute_at] <= end_ts:
                    currently_open = {p.ticker for p in self.portfolio.positions}
                    for e in new_entries:
                        if e["ticker"] not in currently_open:
                            scheduled_entries.append((execute_at, e))

        # --- Save daily portfolio allocations (cross-sectional) ---
        if daily_allocation_rows:
            import os
            path = getattr(self.config, "daily_positions_csv_path", "output/portfolio/daily_positions.csv")
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            pd.DataFrame(daily_allocation_rows).to_csv(path, index=False)

        # --- Factor neutralization diagnostics (report with IC written from run() after metrics) ---
        self._factor_diagnostics = factor_diagnostics
        self._crisis_entries_blocked_days_1_3 = int(crisis_entries_blocked_days_1_3)
        if factor_diagnostics and write_exposure_diagnostics is not None:
            exposures_path = getattr(
                self.config, "factor_exposures_path", "output/research/factor_exposures.csv"
            )
            write_exposure_diagnostics(factor_diagnostics, exposures_path)

        # --- Force-close any leftover positions ---
        for pos in list(self.portfolio.positions):
            last_day = trading_days[-1] if trading_days else pd.Timestamp.now()
            self._close_position(pos, last_day, price_data, reason="force_close")
        
        self._universe_sizes = universe_sizes

    def _build_returns_matrix(
        self,
        price_data: dict,
        as_of_date: pd.Timestamp,
        lookback_days: int,
    ) -> pd.DataFrame | None:
        """Thin wrapper — logic lives in analytics.build_returns_matrix."""
        return _build_returns_matrix_fn(price_data, as_of_date, lookback_days)

    def _annualized_vol(
        self,
        price_data: dict,
        ticker: str,
        as_of_date: pd.Timestamp,
        lookback_days: int = 20,
    ) -> float | None:
        """Thin wrapper — logic lives in analytics.annualized_vol."""
        return _annualized_vol_fn(price_data, ticker, as_of_date, lookback_days)

    def _get_rolling_kelly_params(self) -> tuple[float, float, float] | None:
        """
        Estimate win_rate, avg_win_return, avg_loss_return from last 50 closed trades.
        Returns None if fewer than 20 trades (caller uses config defaults).
        """
        log = getattr(self.portfolio, "trade_log", []) or []
        n = min(50, len(log))
        if n < 20:
            return None
        recent = log[-n:]
        returns = [r.get("return") for r in recent if r.get("return") is not None]
        if len(returns) < 20:
            return None
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        win_rate = len(wins) / len(returns) if returns else 0.5
        avg_win = (sum(wins) / len(wins)) if wins else 0.02
        avg_loss = (sum(abs(x) for x in losses) / len(losses)) if losses else 0.015
        return (win_rate, avg_win, avg_loss)

    def _compute_position_size(self, entry: dict, price_data: dict, date: pd.Timestamp) -> float:
        """Compute dollar position size from config (equal / vol_scaled / kelly / risk_parity)."""
        position_scale = entry.get("position_scale", 1.0)
        method = getattr(self.config, "position_sizing_method", None) or getattr(
            self.config, "position_sizing", "equal"
        )

        vol_annual = None
        if method in ("vol_scaled", "risk_parity") and entry.get("ticker"):
            vol_annual = self._annualized_vol(
                price_data,
                entry["ticker"],
                date,
                getattr(self.config, "vol_lookback_days", 20),
            )

        kelly_win = getattr(self.config, "kelly_win_rate", 0.55)
        kelly_avg_win = getattr(self.config, "kelly_avg_win_return", 0.02)
        kelly_avg_loss = getattr(self.config, "kelly_avg_loss_return", 0.015)
        if method == "kelly":
            rolling = self._get_rolling_kelly_params()
            if rolling is not None:
                kelly_win, kelly_avg_win, kelly_avg_loss = rolling
            self._last_kelly_win_rate = kelly_win

        params = position_sizing.PositionSizingParams(
            method=method,
            max_position_pct_of_equity=getattr(
                self.config, "max_position_pct_of_equity", 0.25
            ),
            target_risk_fraction=getattr(self.config, "vol_target_annual", 0.15),
            kelly_fraction=getattr(self.config, "kelly_fraction", 0.5),
            kelly_win_rate=kelly_win,
            kelly_avg_win_return=kelly_avg_win,
            kelly_avg_loss_return=kelly_avg_loss,
        )

        equity = self.portfolio.equity
        max_positions = self.portfolio.max_positions

        if method == "vol_scaled":
            return position_sizing.vol_scaled_size(
                equity=equity,
                max_positions=max_positions,
                position_scale=position_scale,
                stock_vol_annual=vol_annual,
                params=params,
            )
        if method == "kelly":
            size = position_sizing.kelly_size(
                equity=equity,
                max_positions=max_positions,
                position_scale=position_scale,
                params=params,
            )
            if equity > 0 and size > 0:
                self._last_kelly_position_pct = 100.0 * size / equity
            return size
        if method == "risk_parity":
            return position_sizing.risk_parity_size(
                equity=equity,
                max_positions=max_positions,
                position_scale=position_scale,
                stock_vol_annual=vol_annual,
                params=params,
            )

        # default: equal-weight sizing
        return position_sizing.equal_size(
            equity=equity,
            max_positions=max_positions,
            position_scale=position_scale,
            params=params,
        )

    def _get_holding_days(self, signal: str, adjusted_score: float) -> int:
        """
        Return holding period in days for this signal.
        When dynamic_holding_enabled: use by_strength bands first (strongest |score| wins),
        then by_signal, else config.holding_period_days.
        """
        if not self.config.dynamic_holding_enabled:
            return self.config.holding_period_days
        abs_s = abs(adjusted_score)
        # Bands: list of (min_abs_score, days), use descending order so strongest first
        bands = sorted(self.config.holding_period_by_strength, key=lambda b: -b[0])
        for min_score, days in bands:
            if abs_s >= min_score:
                return days
        return self.config.holding_period_by_signal.get(
            signal, self.config.holding_period_days
        )

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _precompute_regime_score_stats(self, signal_data, regime_data) -> None:
        """Precompute global score std per regime for fallback in confidence gate."""
        self._regime_score_std = {}
        all_scores_by_regime: dict[str, list[float]] = {
            "Bull": [],
            "Bear": [],
            "Sideways": [],
            "Crisis": [],
        }

        if isinstance(signal_data, dict):
            for _tk, df in signal_data.items():
                if df is None or df.empty or "adjusted_score" not in df.columns:
                    continue
                for date, row in df.iterrows():
                    reg = "Sideways"
                    if regime_data:
                        try:
                            reg = str(regime_data.get(date, "Sideways"))
                        except Exception:
                            reg = "Sideways"
                    if reg not in all_scores_by_regime:
                        all_scores_by_regime[reg] = []
                    score = pd.to_numeric(row.get("adjusted_score"), errors="coerce")
                    if pd.notna(score):
                        all_scores_by_regime[reg].append(float(score))

        for reg, scores in all_scores_by_regime.items():
            if len(scores) >= 10:
                self._regime_score_std[reg] = float(np.std(scores, ddof=1))
            else:
                self._regime_score_std[reg] = 0.1  # default fallback

        logger.info("Precomputed regime score std: %s", self._regime_score_std)

    def _signal_confidence_multiplier_for_regime(self, regime_today: str) -> float | None:
        """Thin wrapper — logic lives in regime_multipliers.signal_confidence_multiplier_for_regime."""
        if _scm_for_regime is not None:
            multiplier = _scm_for_regime(
                regime_today,
                self.config,
                loaded_multipliers=getattr(self, "regime_multipliers", None),
            )
        else:
            base = getattr(self.config, "signal_confidence_multiplier", None)
            multiplier = float(base) if base is not None else 1.0

        regime_key = str(regime_today or "Sideways")
        if regime_key != "Sideways" and abs(multiplier - 1.0) > 1e-12:
            logger.debug(
                "Regime %s: applying multiplier %.2f to confidence threshold",
                regime_key,
                multiplier,
            )
        return multiplier

    def _threshold_aggressiveness_for_regime(self, regime_today: str) -> float:
        """Thin wrapper — logic lives in regime_multipliers.threshold_aggressiveness_for_regime."""
        if _thresh_agg_for_regime is not None:
            return _thresh_agg_for_regime(regime_today, self.config)
        return 1.0

    def _compute_adv_usd(self, ticker: str, price_data: dict, date) -> float:
        """20-day average daily dollar volume. Falls back to config default."""
        try:
            df = price_data.get(ticker)
            if df is None or df.empty or "Volume" not in df.columns or "Close" not in df.columns:
                return float(getattr(self.config, "mi_adv_usd_default", 50_000_000.0))
            hist = df.loc[df.index <= pd.Timestamp(date)]
            if len(hist) < 2:
                return float(getattr(self.config, "mi_adv_usd_default", 50_000_000.0))
            dvol = (hist["Volume"] * hist["Close"]).tail(20)
            adv = float(dvol.mean())
            return adv if adv > 0 else float(getattr(self.config, "mi_adv_usd_default", 50_000_000.0))
        except Exception:
            return float(getattr(self.config, "mi_adv_usd_default", 50_000_000.0))

    def _compute_vol_daily(self, ticker: str, price_data: dict, date) -> float:
        """20-day realised daily return volatility. Falls back to 1.5% (S&P average)."""
        try:
            df = price_data.get(ticker)
            if df is None or df.empty or "Close" not in df.columns:
                return 0.015
            hist = df["Close"].loc[df.index <= pd.Timestamp(date)].tail(21)
            if len(hist) < 3:
                return 0.015
            vol = float(hist.pct_change().dropna().std())
            return vol if vol > 0 else 0.015
        except Exception:
            return 0.015

    def _compute_atr(self, ticker: str, price_data: dict, date: pd.Timestamp, window: int = 14) -> float:
        """ATR(window) in dollars at the given date.

        True Range = max(H-L, |H-C_prev|, |L-C_prev|).
        Returns 0.0 when price history is insufficient rather than raising.
        Used by the ATR-scaled stop-loss so each stock's stop self-calibrates to
        its own realised volatility — a 2.5×ATR stop means the same 'noise units'
        regardless of whether the stock moves $0.50 or $5 per day.
        """
        try:
            df = price_data.get(ticker)
            if df is None or df.empty:
                return 0.0
            cols = {"High", "Low", "Close"}
            if not cols.issubset(df.columns):
                return 0.0
            hist = df.loc[df.index <= pd.Timestamp(date)].tail(window + 1)
            if len(hist) < 2:
                return 0.0
            high = hist["High"]
            low = hist["Low"]
            close_prev = hist["Close"].shift(1)
            tr = pd.concat(
                [(high - low), (high - close_prev).abs(), (low - close_prev).abs()],
                axis=1,
            ).max(axis=1)
            atr = float(tr.iloc[1:].mean())   # drop first row (NaN prev close)
            return atr if np.isfinite(atr) and atr > 0 else 0.0
        except Exception:
            return 0.0

    def _crisis_transition_exit_price(self, pos, date, price_data) -> float | None:
        """Exit at Open when enabled; else None (caller uses Close in _close_position)."""
        if not bool(getattr(self.config, "crisis_transition_use_open_exit", True)):
            return None
        tk = pos.ticker
        if tk not in price_data or date not in price_data[tk].index:
            return None
        bar = price_data[tk].loc[date]
        o = float(bar.get("Open", np.nan))
        if np.isfinite(o) and o > 0:
            return float(self.execution.apply_exit_slippage(o, pos.signal))
        return None

    def _close_position(self, pos, date, price_data, exit_price_override: float | None = None, reason: str = ""):
        tk = pos.ticker
        if tk in price_data and date in price_data[tk].index:
            base_price = float(price_data[tk].loc[date, "Close"])
        elif tk in price_data and not price_data[tk].empty:
            base_price = float(price_data[tk]["Close"].iloc[-1])
        else:
            base_price = pos.current_price or pos.entry_price

        if exit_price_override is not None:
            base_price = exit_price_override

        exit_price = self.execution.apply_exit_slippage_scaled(
            base_price, pos.signal, getattr(self, "_cost_scalar_today", 1.0)
        )
        exit_cost = self.cost_model.cost_dollars(pos.position_size) if self.cost_model else self.execution.commission
        # Almgren-Chriss exit impact (no permanent impact on exit leg)
        if self._market_impact_model is not None and pos.position_size > 0:
            try:
                _adv = self._compute_adv_usd(tk, price_data, date)
                _vol = self._compute_vol_daily(tk, price_data, date)
                _mi = MarketImpactModel(adv_usd=_adv, params=self._market_impact_model.params)
                exit_cost += _mi.cost_dollars(pos.position_size, _vol, is_exit=True)
            except Exception:
                pass
        # Tag exit reason on the position before logging the trade
        pos.exit_reason = reason or pos.exit_reason

        extra = {}
        if getattr(self.config, "options_analysis", False) and bs_price is not None and bs_greeks is not None and implied_vol_from_historical is not None:
            S = pos.entry_price
            T = getattr(self.config, "options_expiry_days", 30) / 365.0
            r = getattr(self.config, "options_risk_free_rate", 0.04)
            sigma = 0.20
            if tk in price_data and not price_data[tk].empty:
                series = price_data[tk]["Close"].loc[price_data[tk].index <= pos.entry_date]
                if len(series) >= 2:
                    iv_series = implied_vol_from_historical(series, window=30)
                    last_iv = iv_series.dropna().iloc[-1] if not iv_series.dropna().empty else 0.20
                    if last_iv > 0:
                        sigma = float(last_iv)
            atm_call = bs_price(S, S, T, r, sigma, "call")
            greeks = bs_greeks(S, S, T, r, sigma, "call")
            put_hedge = bs_price(S, S, T, r, sigma, "put")
            extra = {
                "atm_call_price": round(atm_call, 4),
                "atm_call_delta": round(greeks["delta"], 6),
                "breakeven_pct": round(100.0 * atm_call / S, 4) if S > 0 else None,
                "put_hedge_cost": round(put_hedge, 4),
                "implied_vol": round(sigma, 4),
            }
        self.portfolio.close_position(pos, date, exit_price, exit_cost, **extra)

    # ==============================================================
    # Continuous Optimization Simulation
    # ==============================================================

    def _simulate_continuous(
        self,
        price_data: dict,
        signal_data: dict,
        regime_data: dict,
    ) -> tuple:
        """
        Forecast-driven continuous portfolio optimization simulation.

        Architecture:
            signal → forecast → risk_model → optimizer → trade_scheduler
                                                        → mark-to-market

        Replaces the discrete top-N rebalance loop with a daily QP that
        naturally controls turnover via the turnover penalty γ||Δw||².

        Returns
        -------
        (trades_df, daily_equity_df) compatible with existing metrics pipeline.
        """
        from .forecast import ForecastEngine
        from .risk_model import RiskModel
        from .optimizer import PortfolioOptimizer
        from .trade_scheduler import TradeScheduler

        cfg = self.config
        opt_cfg = getattr(cfg, "optimization_config", {}) or {}

        # ── Build components from config ────────────────────────────────────
        fc_cfg = opt_cfg.get("forecast", {})
        _use_demean = bool(fc_cfg.get("use_forecast_demean", True))
        _max_alpha = float(fc_cfg.get("max_alpha", 0.05))
        forecast_engine = ForecastEngine(
            tau_days=float(fc_cfg.get("tau_days", 6.0)),
            smoothing_span=int(fc_cfg.get("smoothing_span", 5)),
            scale_factor=float(fc_cfg.get("scale_factor", 0.012)),
        )

        rm_cfg = opt_cfg.get("risk_model", {})
        risk_model = RiskModel(
            window=int(rm_cfg.get("window", 60)),
            min_periods=int(rm_cfg.get("min_periods", 20)),
            method=str(rm_cfg.get("method", "ledoit_wolf")),
            annualize=True,
        )

        op_cfg = opt_cfg.get("optimizer", {})
        fm_cfg = opt_cfg.get("factor_model", {})
        _market_ticker = str(fm_cfg.get("market_ticker", "SPY"))
        # eta_beta and xi_sector removed: when using Σ_idio (factor-neutralized
        # covariance), market and sector variance are already absent. Additive
        # factor penalties on top of Σ_idio double-count and make λ uncalibrated.
        optimizer = PortfolioOptimizer(
            lambda_risk=float(op_cfg.get("lambda_risk", 2.0)),
            gamma_turnover=float(op_cfg.get("gamma_turnover", 4.0)),
            max_weight=float(op_cfg.get("max_weight", getattr(cfg, "max_position_pct_of_equity", 0.10))),
            net_exposure_max=float(op_cfg.get("net_exposure_max", 1.0)),
            long_only=bool(op_cfg.get("long_only", not getattr(cfg, "enable_shorts", False))),
            gross_cap=float(op_cfg.get("gross_cap", getattr(cfg, "max_gross_exposure", 1.0))),
            min_position_weight=float(op_cfg.get("min_position_weight", 0.0)),
        )

        ex_cfg = opt_cfg.get("execution", {})
        scheduler = TradeScheduler(
            horizon=int(ex_cfg.get("horizon_days", 3)),
            min_trade=float(ex_cfg.get("min_trade_weight", 0.001)),
            short_close_horizon=int(ex_cfg.get("short_close_horizon_days", ex_cfg.get("horizon_days", 3))),
        )

        # ── Exposure scaling config ──────────────────────────────────────────
        es_cfg = opt_cfg.get("exposure_scaling", {})
        _use_exposure_scaling = bool(es_cfg.get("enabled", False))
        _strength_window = int(es_cfg.get("strength_window", 60))
        _strength_pct = float(es_cfg.get("strength_percentile", 75))
        _es_ema_span = int(es_cfg.get("ema_span", 5))
        _min_exposure = float(es_cfg.get("min_exposure", 0.1))
        _es_alpha = 2.0 / (_es_ema_span + 1)  # EMA decay factor

        # ── Multi-alpha engine (optional) ────────────────────────────────────
        from .multi_alpha import MultiAlphaEngine
        ma_cfg = opt_cfg.get("multi_alpha", {})
        _use_multi_alpha = bool(ma_cfg.get("enabled", False))
        _mae: Optional[MultiAlphaEngine] = None
        if _use_multi_alpha:
            _mae = MultiAlphaEngine(
                base_weights=tuple(ma_cfg.get("weights", [0.5, 0.25, 0.25])),
                reversal_days=int(ma_cfg.get("reversal_days", 1)),
                momentum_days=int(ma_cfg.get("momentum_days", 20)),
                clip_sigma=float(ma_cfg.get("clip_sigma", 3.0)),
                ic_window=int(ma_cfg.get("ic_window", 60)),
                corr_window=int(ma_cfg.get("corr_window", 60)),
                target_vol=float(ma_cfg.get("target_vol", 0.15)),
                vol_scale_window=int(ma_cfg.get("vol_scale_window", 20)),
                vol_scale_lo=float(ma_cfg.get("vol_scale_lo", 0.5)),
                vol_scale_hi=float(ma_cfg.get("vol_scale_hi", 2.0)),
                weight_ridge=float(ma_cfg.get("weight_ridge", 1e-3)),
                use_volume_divergence=bool(ma_cfg.get("use_volume_divergence", True)),
                # CRITICAL: IC calibration horizon must match portfolio holding period.
                # Default 1 (old behavior) trains on 1-day IC while positions are held
                # 5 days — this upweights α₁ reversal which is anti-predictive at 5d.
                ic_horizon_days=int(ma_cfg.get("ic_horizon_days", getattr(cfg, "holding_period_days", 5))),
            )

        # ── Sector mapping (always loaded — needed for Σ_idio + MAE neutralization) ──
        # Previously gated on _use_factor_model + xi_sector > 0. Now unconditional:
        # the sector map is required for consistent return-space neutralization in
        # both the risk model (Σ_idio) and the MultiAlphaEngine (_factor_neutralize).
        _sector_id_map: dict[str, int] = {}
        _n_sectors: int = 0
        _sector_map_path = str(fm_cfg.get("sector_mapping_path", "config/sector_mapping.csv"))
        try:
            import os
            _sm_path = _sector_map_path if os.path.isabs(_sector_map_path) else os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))), _sector_map_path
            )
            _sm_df = pd.read_csv(_sm_path)
            _sector_labels = sorted(_sm_df["sector"].dropna().unique().tolist())
            _sector_label_to_id = {s: i for i, s in enumerate(_sector_labels)}
            _n_sectors = len(_sector_labels)
            for _, row in _sm_df.iterrows():
                t_sym = str(row["ticker"])
                t_sec = str(row["sector"])
                if t_sec in _sector_label_to_id:
                    _sector_id_map[t_sym] = _sector_label_to_id[t_sec]
            logger.info(
                "_simulate_continuous: loaded %d sector mappings, %d sectors",
                len(_sector_id_map), _n_sectors,
            )
        except Exception as exc:
            logger.warning(
                "_simulate_continuous: could not load sector mapping (%s); "
                "neutralization will be market-only (no sector demeaning).", exc
            )

        # ── Pre-compute smoothed forecast series (vectorised, no per-bar overhead) ──
        signal_data_with_fc = forecast_engine.build_forecast_series(signal_data)

        # ── Collect all dates in simulation window ───────────────────────────
        start_ts = pd.Timestamp(cfg.start_date)
        end_ts = pd.Timestamp(cfg.end_date)
        all_dates: set = set()
        for df in price_data.values():
            if df is not None and not df.empty:
                all_dates.update(df.index)
        trading_days = sorted(d for d in all_dates if start_ts <= d <= end_ts)
        if not trading_days:
            logger.warning("_simulate_continuous: no trading days found.")
            return pd.DataFrame(), pd.DataFrame()

        tickers_universe = [t for t in signal_data_with_fc if t in price_data]

        # ── MultiAlphaEngine precompute (once, before loop) ──────────────────
        if _mae is not None:
            _mae.precompute(
                price_data,
                tickers_universe,
                market_ticker=str(ma_cfg.get("market_ticker", "SPY")),
                sector_id_map=_sector_id_map if _sector_id_map else None,
            )

        # ── State ────────────────────────────────────────────────────────────
        equity = float(cfg.initial_capital)
        w_current: dict[str, float] = {t: 0.0 for t in tickers_universe}

        # Exposure scaling state
        from collections import deque
        _strength_history: deque = deque(maxlen=_strength_window)
        _alpha_ema: float = 1.0  # Start fully invested; ramp down on weak days

        # Virtual position tracking (for trade log compatibility)
        _POS_THRESHOLD = 0.005  # treat |w| > 0.5% as an open position
        virt_positions: dict[str, dict] = {}  # ticker → entry info

        equity_history: list[dict] = []
        trade_log: list[dict] = []
        total_cost_paid = 0.0

        # Slippage / commission cost per unit of weight change (fraction)
        cost_per_unit = (
            float(getattr(cfg, "slippage_bps", 1.0)) +
            float(getattr(cfg, "execution_costs_commission_bps", 1.0)) +
            float(getattr(cfg, "execution_costs_spread_bps", 1.0))
        ) / 10_000.0

        for date in trading_days:
            regime = regime_data.get(date, "Sideways")

            # ── 1. Regime-based gross cap override ──────────────────────────
            regime_gross_cap = optimizer.gross_cap
            if regime == "Crisis":
                regime_gross_cap = min(regime_gross_cap, float(getattr(cfg, "crisis_gross_cap_fraction", 0.30)))
            elif regime == "Bear":
                regime_gross_cap = min(regime_gross_cap, float(getattr(cfg, "bear_gross_cap_fraction", 0.50)))
            optimizer.gross_cap = regime_gross_cap

            # ── 2. Build today's forecasts for all tickers ───────────────────
            raw_forecasts = forecast_engine.current_forecasts(
                signal_data_with_fc, date, score_col="adjusted_score"
            )

            # ── 3. Regime multiplier on forecasts ────────────────────────────
            regime_adj = getattr(cfg, "regime_adjustments", {}).get(regime, {})
            score_mult = float(regime_adj.get("score_mult", 1.0))
            forecasts = {t: v * score_mult for t, v in raw_forecasts.items()}

            # ── 3b. Forecast normalisation / multi-alpha combination ─────────
            _signal_strength = 0.0
            if forecasts:
                if _mae is not None:
                    # Multi-alpha path: MAE handles CS norm, orthogonalization,
                    # correlation-adjusted weighting, and vol regime scaling internally.
                    forecasts = _mae.current_forecasts(date, forecasts)
                else:
                    # Single-alpha path: demean (remove negative-mean bias) + clip.
                    # Do NOT divide by std — that destroys signal magnitude and makes
                    # weak-signal days indistinguishable from strong ones.
                    fc_arr = np.array(list(forecasts.values()))
                    if _use_demean:
                        fc_mean = float(fc_arr.mean())
                        forecasts = {t: v - fc_mean for t, v in forecasts.items()}
                        fc_arr = np.array(list(forecasts.values()))
                    if _max_alpha > 0.0:
                        forecasts = {t: float(np.clip(v, -_max_alpha, _max_alpha))
                                     for t, v in forecasts.items()}
                        fc_arr = np.array(list(forecasts.values()))

                # Track CS std for exposure scaling (both paths)
                fc_arr = np.array(list(forecasts.values()))
                _signal_strength = float(fc_arr.std()) if len(fc_arr) > 1 else 0.0
                _strength_history.append(_signal_strength)

            # ── 4. Active tickers: universe ∩ those with signals today ───────
            active_tickers = [t for t in tickers_universe if t in forecasts]
            if not active_tickers:
                # No signals → hold current weights, no trades
                equity_history.append({"date": date, "equity": equity, "n_positions": 0,
                                        "gross_exposure": sum(abs(v) for v in w_current.values()),
                                        "regime": regime})
                continue

            # ── 5. Covariance matrix for active tickers (Σ_idio) ───────────
            # Pass sector_id_map so risk model estimates Σ from idiosyncratic
            # returns — the same return space as the alpha signals. Without
            # this, λw'Σ_raw·w and the idio-space forecast μ have inconsistent
            # units, making λ uncalibrated and the optimizer solution invalid.
            cov, cov_tickers = risk_model.fit_at_date(
                price_data, active_tickers, date,
                sector_id_map=_sector_id_map if _sector_id_map else None,
            )
            if len(cov_tickers) < 2:
                # Fallback: keep current weights
                equity_history.append({"date": date, "equity": equity, "n_positions": 0,
                                        "gross_exposure": sum(abs(v) for v in w_current.values()),
                                        "regime": regime})
                continue

            # Align forecast and w_prev to cov_tickers order
            fc_vec = {t: forecasts.get(t, 0.0) for t in cov_tickers}
            wp_vec = {t: w_current.get(t, 0.0) for t in cov_tickers}

            # ── 6. Optimise ──────────────────────────────────────────────────
            # Factor penalties (eta_beta, xi_sector) removed. Σ_idio is estimated
            # from idiosyncratic returns, so market and sector variance are already
            # absent from the covariance. No additive penalties needed.
            w_target_map = optimizer.optimize(fc_vec, cov, wp_vec, cov_tickers)

            # Zero out tickers with no signal (they were in cov but not in forecasts)
            for t in cov_tickers:
                if t not in forecasts:
                    w_target_map[t] = 0.0

            # ── 7. Regime short suppression ──────────────────────────────────
            # In Crisis/Bear regimes the short book gets destroyed by market beta
            # surges (e.g. 2009 recovery, 2020 rebound). Suppress shorts entirely
            # in Crisis; allow only weak shorts (≤50% of normal cap) in Bear.
            if getattr(cfg, "regime_short_suppression", True):
                if regime == "Crisis":
                    w_target_map = {t: max(0.0, v) for t, v in w_target_map.items()}
                elif regime == "Bear":
                    _bear_short_cap = float(getattr(cfg, "bear_short_cap", 0.5))
                    w_target_map = {
                        t: max(v, -abs(optimizer.max_weight) * _bear_short_cap)
                        for t, v in w_target_map.items()
                    }

            # ── 8. Exposure scaling: α_t × w_opt ─────────────────────────────
            # Scales the whole portfolio by cross-sectional signal strength.
            # Weak-signal days (flat cross-section) → small positions.
            # α_t is EMA-smoothed to prevent day-to-day instability.
            if _use_exposure_scaling and len(_strength_history) >= max(5, _strength_window // 4):
                _hist_arr = np.array(_strength_history)
                _threshold = float(np.percentile(_hist_arr, _strength_pct))
                if _threshold > 1e-9:
                    _alpha_raw = min(1.0, _signal_strength / _threshold)
                else:
                    _alpha_raw = 1.0
                _alpha_ema = _alpha_raw * _es_alpha + _alpha_ema * (1.0 - _es_alpha)
                _scale = max(_min_exposure, _alpha_ema)
                w_target_map = {t: v * _scale for t, v in w_target_map.items()}

            # ── 10. Gradual execution ────────────────────────────────────────
            actual_delta = scheduler.step(w_target_map, w_current)

            # ── 11. Apply cost and update weights ────────────────────────────
            for t, dw in actual_delta.items():
                if abs(dw) < scheduler.min_trade:
                    continue
                trade_dollar = abs(dw) * equity
                # Fix 3: one-way cost per execution step — no ×2 multiplier.
                # Round-trip cost is naturally paid as two separate one-way steps
                # (open leg + close leg). Applying ×2 per step means a position
                # opened over 3 scheduler steps pays 6× the correct one-way cost.
                cost = trade_dollar * cost_per_unit
                equity -= cost
                total_cost_paid += cost
                w_current[t] = w_current.get(t, 0.0) + dw
                # Accumulate per-position cost for trade log attribution (Fix 3)
                if t in virt_positions:
                    virt_positions[t]["cumulative_cost"] = (
                        virt_positions[t].get("cumulative_cost", 0.0) + cost
                    )

            # Clamp tiny residuals
            for t in list(w_current.keys()):
                if abs(w_current[t]) < 1e-5:
                    w_current[t] = 0.0

            # ── 12. Mark-to-market: compute today's P&L ─────────────────────
            daily_pnl = 0.0
            for t, w in w_current.items():
                if abs(w) < 1e-6 or t not in price_data:
                    continue
                df_px = price_data[t]
                if df_px is None or df_px.empty:
                    continue
                close_col = "Close" if "Close" in df_px.columns else "close"
                # Find today and previous trading day close
                idx_arr = df_px.index
                loc = idx_arr.searchsorted(date)
                if loc == 0 or loc >= len(idx_arr) or idx_arr[loc] != date:
                    continue
                try:
                    c_today = float(pd.to_numeric(df_px[close_col].iloc[loc], errors="coerce"))
                    c_prev = float(pd.to_numeric(df_px[close_col].iloc[loc - 1], errors="coerce"))
                except (IndexError, ValueError):
                    continue
                if not (np.isfinite(c_today) and np.isfinite(c_prev) and c_prev > 0):
                    continue
                ret = (c_today - c_prev) / c_prev
                daily_pnl += w * ret

            equity = equity * (1.0 + daily_pnl)

            # ── 10. Virtual position tracking → trade log ────────────────────
            for t in tickers_universe:
                w = w_current.get(t, 0.0)
                direction = 1 if w > 0 else (-1 if w < 0 else 0)
                prev_pos = virt_positions.get(t)

                if prev_pos is None and abs(w) >= _POS_THRESHOLD:
                    # Open a virtual position
                    px = self._get_close_price(t, date, price_data)
                    virt_positions[t] = {
                        "entry_date": date,
                        "entry_price": px,
                        "direction": direction,
                        "entry_weight": abs(w),
                        "entry_equity": equity,             # Fix 4: equity at open for accurate position_size
                        "entry_score": float(forecasts.get(t, 0.0)),  # Fix 1: entry-day signal for IC
                        "cumulative_cost": 0.0,             # Fix 3: accumulate tx costs during hold
                        "signal_date": date,
                    }
                elif prev_pos is not None:
                    direction_flip = (prev_pos["direction"] > 0) != (direction > 0)
                    closed = abs(w) < _POS_THRESHOLD or direction_flip
                    if closed:
                        # Close virtual position and record trade
                        px_exit = self._get_close_price(t, date, price_data)
                        px_entry = prev_pos["entry_price"]
                        raw_ret = prev_pos["direction"] * (px_exit - px_entry) / (px_entry + 1e-12)
                        # Fix 4: use entry equity so position_size reflects actual capital at risk
                        _entry_equity = prev_pos.get("entry_equity", equity)
                        _pos_size = prev_pos["entry_weight"] * _entry_equity
                        _total_cost = prev_pos.get("cumulative_cost", 0.0)
                        holding_days = max(1, (date - prev_pos["entry_date"]).days)
                        trade_log.append({
                            "ticker": t,
                            "signal": "Bullish" if prev_pos["direction"] > 0 else "Bearish",
                            "direction": prev_pos["direction"],
                            "signal_date": prev_pos["signal_date"],
                            "entry_date": prev_pos["entry_date"],
                            "exit_date": date,
                            "entry_price": px_entry,
                            "exit_price": px_exit,
                            "position_size": _pos_size,
                            "shares": _pos_size / max(px_entry, 1e-6),
                            "return": raw_ret,
                            "pnl": raw_ret * _pos_size,
                            "realized_pnl": raw_ret * _pos_size,
                            # Fix 1: entry-day signal (not exit-day) for valid IC measurement
                            "adjusted_score": prev_pos.get("entry_score", 0.0),
                            "confidence": "High",
                            "regime": regime,
                            "holding_days": holding_days,
                            # Fix 3: expose actual accumulated transaction costs
                            "transaction_cost": _total_cost,
                            "entry_cost": 0.0,
                            "exit_cost": 0.0,
                            "total_cost": _total_cost,
                        })
                        del virt_positions[t]
                        # Re-open if still has weight (direction flip)
                        if abs(w) >= _POS_THRESHOLD and direction_flip:
                            px_new = self._get_close_price(t, date, price_data)
                            virt_positions[t] = {
                                "entry_date": date,
                                "entry_price": px_new,
                                "direction": direction,
                                "entry_weight": abs(w),
                                "entry_equity": equity,
                                "entry_score": float(forecasts.get(t, 0.0)),
                                "cumulative_cost": 0.0,
                                "signal_date": date,
                            }

            n_open = int(sum(1 for v in w_current.values() if abs(v) >= _POS_THRESHOLD))
            gross_exp = float(sum(abs(v) for v in w_current.values()))
            equity_history.append({
                "date": date,
                "equity": equity,
                "n_positions": n_open,
                "gross_exposure": gross_exp,
                "regime": regime,
            })

        # Close any remaining virtual positions at last trading day
        last_date = trading_days[-1] if trading_days else None
        if last_date is not None:
            for t, prev_pos in list(virt_positions.items()):
                px_exit = self._get_close_price(t, last_date, price_data)
                px_entry = prev_pos["entry_price"]
                raw_ret = prev_pos["direction"] * (px_exit - px_entry) / (px_entry + 1e-12)
                avg_w = prev_pos["entry_weight"]
                # Fix 4: use entry equity for consistent position_size
                _entry_equity_eod = prev_pos.get("entry_equity", equity)
                _pos_size_eod = avg_w * _entry_equity_eod
                _total_cost_eod = prev_pos.get("cumulative_cost", 0.0)
                holding_days = max(1, (last_date - prev_pos["entry_date"]).days)
                trade_log.append({
                    "ticker": t,
                    "signal": "Bullish" if prev_pos["direction"] > 0 else "Bearish",
                    "direction": prev_pos["direction"],
                    "signal_date": prev_pos["signal_date"],
                    "entry_date": prev_pos["entry_date"],
                    "exit_date": last_date,
                    "entry_price": px_entry,
                    "exit_price": px_exit,
                    "position_size": _pos_size_eod,
                    "shares": _pos_size_eod / max(px_entry, 1e-6),
                    "return": raw_ret,
                    "pnl": raw_ret * _pos_size_eod,
                    "realized_pnl": raw_ret * _pos_size_eod,
                    # Fix 1: use stored entry-day signal; 0.0 fallback means this
                    # position was opened before the fix — tolerable for final-day close
                    "adjusted_score": prev_pos.get("entry_score", 0.0),
                    "confidence": "High",
                    "regime": regime_data.get(last_date, "Sideways"),
                    "holding_days": holding_days,
                    # Fix 3: expose actual accumulated costs
                    "transaction_cost": _total_cost_eod,
                    "entry_cost": 0.0,
                    "exit_cost": 0.0,
                    "total_cost": _total_cost_eod,
                })

        trades_df = pd.DataFrame(trade_log)
        equity_df = pd.DataFrame(equity_history)

        gross = sum(abs(v) for v in w_current.values())
        n_pos = sum(1 for v in w_current.values() if abs(v) >= _POS_THRESHOLD)
        print(f"  Continuous sim complete: {len(trading_days)} days, "
              f"{len(trade_log)} virtual trades, "
              f"final equity ${equity:,.0f}, "
              f"gross exposure {gross:.1%}, "
              f"total cost ${total_cost_paid:,.0f}")

        # ── Tier 1 Diagnostics ───────────────────────────────────────────────
        # D1-D5 per the institutional validation checklist.
        # These run once at end-of-simulation from the accumulated history.

        # ── D0: MultiAlpha signal diagnostics ───────────────────────────────
        if _mae is not None:
            print(f"  MultiAlpha correlations: {_mae.correlation_summary()}")
            try:
                ic_report = _mae.compute_ic_report(price_data)
                print(ic_report)
            except Exception as _ic_exc:
                logger.debug("MultiAlpha IC report failed: %s", _ic_exc)

        # ── D1: Cross-sectional mean(r_idio[t]) ≈ 0 ∀ t ────────────────────
        # Verifies that factor neutralization is active. If mean is non-zero,
        # either market demeaning failed or sector_id_map was not loaded.
        if _mae is not None and _mae._ret_1d_df is not None:
            try:
                _idio_cs_mean = _mae._ret_1d_df.mean(axis=1)
                _d1_mean = float(_idio_cs_mean.mean())
                _d1_std = float(_idio_cs_mean.std())
                print(f"\n  [D1] CS mean(r_idio): mean={_d1_mean:+.6f}  std={_d1_std:.6f}"
                      f"  {'PASS' if abs(_d1_mean) < 1e-4 else 'FAIL — neutralization not active'}")
            except Exception as _d1_exc:
                logger.debug("D1 diagnostics failed: %s", _d1_exc)

        # ── D2: Eigenvalue spectrum of alpha signal covariance ───────────────
        # N_eff_signal ≈ K=3 means alphas are orthogonal. N_eff_signal ≈ 1
        # means all three alphas collapse onto the same factor (neutralization
        # didn't work or alpha operators are not independent).
        if _mae is not None:
            _neff_signal = _mae.effective_breadth()
            print(f"  [D2] N_eff_signal (alpha covariance) = {_neff_signal:.2f} / 3.00"
                  f"  {'PASS' if _neff_signal >= 2.0 else 'WARN — alphas not independent'}")

        # ── D3: N_eff_portfolio from realized daily PnL covariance ───────────
        # This is the TRUE effective breadth: how many independent bets does
        # the portfolio actually hold? Distinct from N_eff_signal (which only
        # measures alpha vector orthogonality, max=K=3).
        # Computed from the equity_df daily returns per position is not trivially
        # available here without per-position PnL tracking. Approximate from
        # the covariance of daily equity changes — this gives a lower bound.
        # Full per-position PnL covariance requires trade_log enrichment (future).
        if equity_df is not None and len(equity_df) > 20:
            try:
                _eq_vals = equity_df["equity"].values.astype(float)
                _daily_ret = np.diff(_eq_vals) / (_eq_vals[:-1] + 1e-10)
                # Proxy: rolling 60d window covariance of daily gross_exposure-weighted return
                # True D3 requires per-position PnL matrix; log the proxy clearly.
                _eq_ret_series = pd.Series(_daily_ret)
                _roll_var = _eq_ret_series.rolling(60, min_periods=20).var().dropna().values
                if len(_roll_var) > 5:
                    _s1 = float(np.sum(_roll_var))
                    _s2 = float(np.sum(_roll_var ** 2))
                    _neff_proxy = (_s1 ** 2) / (_s2 + 1e-12)
                    print(f"  [D3] N_eff_portfolio (rolling-variance proxy) = {_neff_proxy:.1f}"
                          f"  [NOTE: proxy only — full per-position PnL covariance not yet tracked]")
            except Exception as _d3_exc:
                logger.debug("D3 diagnostics failed: %s", _d3_exc)

        # ── D4: IC_idio vs IC_raw comparison ─────────────────────────────────
        # Reports IC from MAE's rolling history. These are already computed
        # against r_idio (after Step 1 fix). Compare to raw-return IC to
        # quantify how much the IC was inflated by market direction.
        # Full raw vs idio split requires running MAE twice — log what we have.
        if _mae is not None and any(_mae._ic_orth[k] for k in range(3)):
            _labels = ["α₁ reversal", "α₂ core-ML ", "α₃ momentum"]
            print(f"  [D4] IC summary (vs r_idio — corrected target):")
            for k, lbl in enumerate(_labels):
                _raw_h = list(_mae._ic_raw[k])
                _orth_h = list(_mae._ic_orth[k])
                if _raw_h:
                    print(f"       {lbl}: IC_raw={np.mean(_raw_h):+.4f}"
                          f"  IC_orth={np.mean(_orth_h):+.4f}"
                          f"  ICIR={np.mean(_orth_h) / (np.std(_orth_h) + 1e-9):.2f}")

        # ── D5: Turnover decomposition ────────────────────────────────────────
        # Classify each trade into:
        #   signal-driven: alpha signal changed direction / decayed past threshold
        #   risk-driven:   optimizer reduced position to enforce gross/sector cap
        #   mechanical:    residual (rebalance timer, min_trade threshold rounding)
        # Full decomposition requires annotating trades at source. Proxy: log
        # total annualised turnover; flag if it exceeds 500%/yr (timer-driven sign).
        if trades_df is not None and len(trades_df) > 0 and len(equity_df) > 1:
            try:
                _n_days = (equity_df["date"].iloc[-1] - equity_df["date"].iloc[0]).days
                _years = max(_n_days / 365.25, 0.01)
                _total_traded = float(trades_df["position_size"].abs().sum())
                _avg_equity = float(equity_df["equity"].mean())
                _to_ann = (_total_traded / (_avg_equity + 1e-10)) / _years
                _to_flag = "WARN — may be timer-driven" if _to_ann > 5.0 else "OK"
                print(f"  [D5] Annualised turnover ≈ {_to_ann:.1%}/yr  {_to_flag}"
                      f"\n       [Full signal/risk/mechanical decomposition requires per-trade annotation]")
            except Exception as _d5_exc:
                logger.debug("D5 diagnostics failed: %s", _d5_exc)

        return trades_df, equity_df

    def _get_close_price(
        self,
        ticker: str,
        date: pd.Timestamp,
        price_data: dict,
    ) -> float:
        """Lookup close price for a ticker on a date; fallback to last available."""
        df = price_data.get(ticker)
        if df is None or df.empty:
            return 1.0
        close_col = "Close" if "Close" in df.columns else "close"
        if close_col not in df.columns:
            return 1.0
        hist = df.loc[df.index <= date, close_col]
        if hist.empty:
            return 1.0
        val = float(pd.to_numeric(hist.iloc[-1], errors="coerce"))
        return val if np.isfinite(val) and val > 0 else 1.0
