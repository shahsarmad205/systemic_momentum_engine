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

from collections import defaultdict
import datetime
from datetime import date as date_type, timedelta
import logging

import numpy as np
import pandas as pd
logging.getLogger("matplotlib").setLevel(logging.WARNING)
import matplotlib.pyplot as plt
from pathlib import Path
from .config import BacktestConfig
from .portfolio import Portfolio
from .execution import ExecutionEngine
from .regime import MarketRegimeAgent

try:
    from execution.cost_model import TransactionCostModel
except ImportError:
    TransactionCostModel = None
from .signals import SignalEngine, HISTORY_BUFFER_DAYS, EXIT_BUFFER_DAYS
from .metrics import compute_all_metrics, compute_capm_metrics

from utils.sectors import SECTOR_MAP
from utils.sector_aggregates import compute_sector_aggregates, apply_sector_adjustment
from utils.market_data import get_ohlcv

try:
    from agents.weight_learning_agent.feature_builder import (
        sector_relative_features_by_ticker,
        vol_rank_features_by_ticker,
    )
except ImportError:  # pragma: no cover
    sector_relative_features_by_ticker = None  # type: ignore[misc,assignment]
    vol_rank_features_by_ticker = None  # type: ignore[misc,assignment]
from strategy.candidates import build_ranked_candidates
from strategy.cross_sectional import build_cross_sectional_candidates
from . import position_sizing

try:
    from options.black_scholes import (
        bs_price,
        bs_greeks,
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


# ------------------------------------------------------------------
# Result container
# ------------------------------------------------------------------

class BacktestResult:
    """Holds everything produced by a single backtest run."""

    __slots__ = ("trades", "daily_equity", "metrics", "config",
                 "price_data", "signal_data", "regime_data", "position_sizing_comparison")

    def __init__(
        self,
        trades: pd.DataFrame,
        daily_equity: pd.DataFrame,
        metrics: dict,
        config: BacktestConfig,
    ):
        self.trades = trades
        self.daily_equity = daily_equity
        self.metrics = metrics
        self.config = config
        self.price_data: dict[str, pd.DataFrame] = {}
        self.signal_data: dict[str, pd.DataFrame] = {}
        self.regime_data: dict[pd.Timestamp, str] = {}
        self.position_sizing_comparison: dict | None = None


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

        # Circuit breaker state
        self._equity_peak: float = config.initial_capital
        self._trading_halted: bool = False
        self._circuit_breaker_log: list[dict] = []

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

    def run(self, tickers: list[str] | None = None) -> BacktestResult:
        tickers = tickers or self.config.tickers
        if not tickers:
            try:
                from main import TICKERS
                tickers = list(TICKERS)
            except Exception:
                tickers = list(self._FALLBACK_TICKERS)

        # Phase 1 — data & signals
        mode = self.config.signal_mode
        mode_labels = {
            "price": "price-only (trend + volatility)",
            "full": "full (all agents)",
            "learned": "learned weights (data-driven)",
        }
        mode_label = mode_labels.get(mode, mode)
        print(f"Phase 1: Downloading data & generating signals [{mode_label}]…")
        price_data, signal_data = self._prepare_data(tickers)

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
        if self.config.regime_enabled:
            print("\nPhase 2: Detecting market regimes (SPY + VIX)…")
            regime_data = self.regime_agent.detect_regimes(
                self.config.start_date, self.config.end_date,
            )
            regime_counts = defaultdict(int)
            for r in regime_data.values():
                regime_counts[r] += 1
            print(f"  Regimes: {dict(regime_counts)}")
        else:
            print("\nPhase 2: Regime detection disabled.")

        # Phase 3 — simulation
        print("\nPhase 3: Running day-by-day portfolio simulation…")
        self._simulate(price_data, signal_data, regime_data)

        # Collect outputs
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
            self._simulate(price_data, signal_data, regime_data)
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
            self._simulate(price_data, signal_data, regime_data)
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
        return result

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

        # Average transaction cost as % of trade P&L (by absolute P&L to avoid sign issues)
        cost_pct = df["transaction_cost"] / (df["realized_pnl"].abs() + 1e-8)
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
            for reg in ["Bull", "Bear", "Sideways", "Crisis"]:
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
    ):
        """
        Run simulation with pre-built price_data, signal_data, regime_data (no data load).
        Resets portfolio and runs _simulate. Used for Monte Carlo (noise, delay).
        """
        self.portfolio = Portfolio(self.config.initial_capital, self.config.max_positions)
        self._simulate(price_data, signal_data, regime_data)
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

    def _prepare_data(self, tickers: list[str]):
        start_ts = pd.Timestamp(self.config.start_date)
        dl_start = start_ts - timedelta(days=HISTORY_BUFFER_DAYS)
        dl_end = pd.Timestamp(self.config.end_date) + timedelta(days=EXIT_BUFFER_DAYS)

        price_data: dict[str, pd.DataFrame] = {}
        signal_data: dict[str, pd.DataFrame] = {}
        pending_prices: dict[str, pd.DataFrame] = {}
        min_history_days = int(getattr(self.config, "min_history_days", 252))
        excluded_for_history: list[str] = []

        provider = getattr(self.config, "data_provider", "yahoo") or "yahoo"
        cache = getattr(self.config, "cache_ohlcv", True)
        cache_dir = getattr(self.config, "cache_dir", "data/cache/ohlcv")
        cache_ttl = getattr(self.config, "cache_ttl_days", 0)

        for i, ticker in enumerate(tickers, 1):
            msg_prefix = f"[{i}/{len(tickers)}] {ticker}"
            print(f"  {msg_prefix}…", end=" ")
            logger.info(
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
                if (start_ts - first_date).days < min_history_days:
                    print(f"insufficient pre-history (<{min_history_days}d before start)")
                    excluded_for_history.append(ticker)
                    logger.warning(
                        "Excluding %s due to insufficient pre-history: first_date=%s, required=%d days before %s",
                        ticker,
                        first_date,
                        min_history_days,
                        start_ts,
                    )
                    continue

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
                logger.info(
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

    def _simulate(self, price_data, signal_data, regime_data):
        self._factor_diagnostics = []
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
        crisis_max_holding_days = int(getattr(self.config, "crisis_max_holding_days", 3) or 3)
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
                market_index=getattr(self.config, "factor_market_index", "SPY"),
                rolling_window=int(getattr(self.config, "factor_rolling_window", 60)),
                min_observations=10,
                dev_mode_limit=dev_limit,
            )
        factor_diagnostics: list[dict] = []
        execution_delay_days = int(getattr(self.config, "execution_delay_days", 0) or 0)

        scheduled_entries: list[tuple[int, dict]] = []  # (execute_at_idx, entry)
        pending_entries: list[dict] = []
        daily_allocation_rows: list[dict] = []  # for output/portfolio/daily_positions.csv

        crisis_consecutive_days = 0
        prev_regime: str | None = None
        crisis_entries_blocked_days_1_3 = 0

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
            prev_reg_before = prev_regime

            # Long-only: first day of a Bear spell — exit inherited longs (holdthrough drag on Bear days).
            if (
                bool(getattr(self.config, "bear_liquidate_longs_on_regime_entry", False))
                and getattr(self.config, "long_only", False)
                and regime_today == "Bear"
                and prev_reg_before is not None
                and prev_reg_before != "Bear"
            ):
                for pos in list(self.portfolio.positions):
                    if pos.direction > 0:
                        self._close_position(
                            pos,
                            date,
                            price_data,
                            reason="bear_regime_entry_liquidate",
                        )

            if regime_today == "Crisis":
                crisis_consecutive_days = (crisis_consecutive_days + 1) if prev_regime == "Crisis" else 1
            else:
                crisis_consecutive_days = 0
            prev_regime = regime_today

            # Crisis transition: accelerate exit for positions entered before Crisis (holdthrough P&L drag).
            # Losers: optional same-day forced close; winners: cap planned exit up to N days ahead.
            if (
                regime_today == "Crisis"
                and prev_reg_before is not None
                and prev_reg_before != "Crisis"
                and bool(getattr(self.config, "crisis_transition_accel_enabled", True))
            ):
                flatten_all = bool(getattr(self.config, "crisis_transition_flatten_all", False))
                if flatten_all:
                    for pos in list(self.portfolio.positions):
                        px = self._crisis_transition_exit_price(pos, date, price_data)
                        self._close_position(
                            pos,
                            date,
                            price_data,
                            exit_price_override=px,
                            reason="crisis_transition_flatten_all",
                        )
                else:
                    loser_days = int(getattr(self.config, "crisis_transition_loser_max_hold_days", 0) or 0)
                    winner_extra = int(getattr(self.config, "crisis_transition_winner_extra_days", 3) or 3)
                    force_loser_same_day = bool(
                        getattr(self.config, "crisis_transition_force_close_losers_same_day", True)
                    )
                    for pos in list(self.portfolio.positions):
                        tk = pos.ticker
                        if tk in price_data and date in price_data[tk].index:
                            bar = price_data[tk].loc[date]
                            o = float(bar.get("Open", np.nan))
                            c = float(bar.get("Close", np.nan))
                            px = o if np.isfinite(o) else c
                            if np.isfinite(px):
                                pos.current_price = px
                        prof = float(pos.unrealized_return) > 0
                        if (not prof) and force_loser_same_day:
                            px = self._crisis_transition_exit_price(pos, date, price_data)
                            self._close_position(
                                pos,
                                date,
                                price_data,
                                exit_price_override=px,
                                reason="crisis_transition_loser",
                            )
                            continue
                        extra_idx = winner_extra if prof else loser_days
                        cap_idx = min(i + extra_idx, len(trading_days) - 1)
                        forced_cap = pd.Timestamp(trading_days[cap_idx])
                        pe = pd.Timestamp(pos.planned_exit_date)
                        pos.planned_exit_date = min(pe, forced_cap)
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
                            from portfolio.mean_variance import max_sharpe_weights, min_variance_weights
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
                elif regime_today == "Sideways":
                    # Cap position size at 5% of portfolio per ticker in Sideways regime.
                    equal_size = min(equal_size, self.portfolio.equity * 0.05)

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
                            returns_60d[tk] = p.pct_change().tail(corr_window_days)

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
            # Gross exposure caps:
            # - Normal  : 100% gross (no extra cap)
            # - Crisis  : 40% gross
            # - Bear    : 70% gross
            gross_cap_fraction_today = 1.0
            if regime_today == "Crisis":
                gross_cap_fraction_today = 0.4
            elif regime_today == "Bear":
                gross_cap_fraction_today = float(
                    getattr(self.config, "bear_gross_cap_fraction", 0.7) or 0.7
                )

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

            for entry in pending_entries:
                tk = entry["ticker"]
                # Shorts gate: do not open short trades when allow_shorts is False.
                if entry.get("signal") == "Bearish" and not getattr(self.config, "allow_shorts", False):
                    continue
                # Long-only mode: skip Bearish; allow Bullish even with score <= 0 (we'll use fallback sizing).
                score = float(entry.get("adjusted_score", 0.0) or 0.0)
                if getattr(self.config, "long_only", False):
                    if entry.get("signal") == "Bearish":
                        continue

                # Regime-aware signal filtering:
                # - Bull regime : suppress all shorts
                # - Bear regime : allow both longs and shorts
                # - Crisis      : long-only (suppress shorts)
                # - Sideways    : allow both
                if regime_today in ("Bull", "Crisis") and entry.get("signal") == "Bearish":
                    continue

                # Bear + long-only: optional hard block — no new entries (weak / no edge).
                if (
                    regime_today == "Bear"
                    and getattr(self.config, "long_only", False)
                    and bool(getattr(self.config, "bear_skip_new_entries", False))
                ):
                    continue

                # Crisis: optional hard block — no new entries for entire Crisis spell (fixes Crisis Sharpe from day 4+ entries).
                if regime_today == "Crisis" and bool(
                    getattr(self.config, "crisis_block_all_new_entries", False)
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

                # Bear (long-only): only enter longs above rolling-window score quantile — weak edge in Bear.
                if regime_today == "Bear" and getattr(self.config, "long_only", False):
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
                scm = self._signal_confidence_multiplier_for_regime(regime_today)
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
                            if len(s_hist) >= sc_min:
                                tail = s_hist.tail(sc_window)
                                rs = float(tail.std(ddof=1))
                                reg_e = str(entry.get("regime") or regime_today)
                                smult_c = float(
                                    self.config.regime_adjustments.get(reg_e, {}).get("score_mult", 1.0)
                                    or 1.0
                                )
                                adj_e = float(entry.get("adjusted_score", 0.0) or 0.0)
                                raw_abs = abs(adj_e / smult_c) if abs(smult_c) > 1e-12 else abs(adj_e)
                                if np.isfinite(rs) and rs > 1e-12 and raw_abs < float(scm) * rs:
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
                entry_price = self.execution.apply_entry_slippage(open_price, entry["signal"])

                if entry.get("_cross_sectional"):
                    # Cross-sectional portfolios keep their existing equal-weight sizing.
                    size_dollars = min(equal_size or 0, self.portfolio.equity * 0.99)
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
                            single_cap = 0.10 * equity
                            if regime_today == "Sideways":
                                single_cap = 0.05 * equity
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
                                single_cap = 0.10 * equity
                                if regime_today == "Sideways":
                                    single_cap = 0.05 * equity
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
                                # Base single-position cap: 10% of portfolio equity,
                                # tightened to 5% in Sideways regime.
                                single_cap = 0.10 * equity
                                if regime_today == "Sideways":
                                    single_cap = 0.05 * equity
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
                                    if size_dollars > 0 and size_dollars < min_size and remaining >= min_size:
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
                        size_dollars = min(equity / self.config.max_positions, equity * 0.10, remaining)
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

                # Market impact component: k * sqrt(relative_size) bps
                impact_cost = 0.0
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
                    size_to_use = min(equity / self.config.max_positions, equity * 0.10, remaining)
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
                    entry_price=entry_price,
                    adjusted_score=entry["adjusted_score"],
                    confidence=entry["confidence"],
                    regime=entry["regime"],
                    entry_cost=entry_cost,
                    impact_entry_cost=impact_cost,
                    position_scale=entry.get("position_scale", 1.0),
                    size_dollars=size_to_use,
                )
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
            for pos in self.portfolio.positions:
                if pos.ticker in price_data and date in price_data[pos.ticker].index:
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
            self.portfolio.record_equity(
                date,
                regime,
                crisis_consecutive_days=int(crisis_consecutive_days),
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
                    date, daily_signals[date_key], self.config, trading_days, i, regime
                )
                n = len(new_entries)
                eq = self.portfolio.equity
                per = round(eq / n, 2) if n else 0.0
                for r in log_rows:
                    r["position_size"] = per
                daily_allocation_rows.extend(log_rows)
                # Always execute cross-sectional entries at the next trading day
                # (plus any configured execution_delay_days) at that day's open.
                # Do not schedule new entries for tickers we already hold (re-entry only after expiry).
                execute_at = i + 1 + execution_delay_days
                if execute_at < len(trading_days):
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
                if execute_at < len(trading_days):
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

    def _build_returns_matrix(
        self,
        price_data: dict,
        as_of_date: pd.Timestamp,
        lookback_days: int,
    ) -> pd.DataFrame | None:
        """
        Build DataFrame of daily returns for all tickers, last lookback_days up to as_of_date.
        Uses only past data (no lookahead). Returns None if insufficient data.
        """
        series_list = []
        for ticker, df in price_data.items():
            if df is None or df.empty or "Close" not in df.columns:
                continue
            price_col = "AdjClose" if "AdjClose" in df.columns else "Close"
            past = df[price_col].loc[df.index <= as_of_date].tail(lookback_days + 1)
            if len(past) < 2:
                continue
            rets = past.pct_change().dropna()
            if len(rets) < 2:
                continue
            rets.name = ticker
            series_list.append(rets)
        if not series_list:
            return None
        out = pd.concat(series_list, axis=1, join="inner")
        if out.empty or out.shape[0] < 2 or out.shape[1] < 1:
            return None
        out = out.tail(lookback_days)
        if len(out) < 2:
            return None
        return out

    def _annualized_vol(
        self,
        price_data: dict,
        ticker: str,
        as_of_date: pd.Timestamp,
        lookback_days: int = 20,
    ) -> float | None:
        """Return annualized volatility (std of daily returns) for ticker as of date, or None."""
        if ticker not in price_data:
            return None
        df = price_data[ticker]
        if "Close" not in df.columns or df.empty:
            return None
        price_col = "AdjClose" if "AdjClose" in df.columns else "Close"
        series = df[price_col].loc[df.index <= as_of_date].tail(lookback_days + 1)
        if len(series) < 2:
            return None
        rets = series.pct_change().dropna()
        if len(rets) < 2:
            return None
        return float(rets.std() * (252 ** 0.5))

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
            setattr(self, "_last_kelly_win_rate", kelly_win)

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
                setattr(self, "_last_kelly_position_pct", 100.0 * size / equity)
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

    def _signal_confidence_multiplier_for_regime(self, regime_today: str) -> float | None:
        """Rolling-std confidence gate: regime-specific σ multiplier, else global signal_confidence_multiplier."""
        m = {
            "Bull": getattr(self.config, "signal_confidence_multiplier_bull", None),
            "Sideways": getattr(self.config, "signal_confidence_multiplier_sideways", None),
            "Bear": getattr(self.config, "signal_confidence_multiplier_bear", None),
            "Crisis": getattr(self.config, "signal_confidence_multiplier_crisis", None),
        }.get(regime_today)
        if m is not None:
            return float(m)
        base = getattr(self.config, "signal_confidence_multiplier", None)
        return float(base) if base is not None else None

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

        exit_price = self.execution.apply_exit_slippage(base_price, pos.signal)
        exit_cost = self.cost_model.cost_dollars(pos.position_size) if self.cost_model else self.execution.commission
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
