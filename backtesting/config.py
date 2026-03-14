"""
Backtest Configuration
========================
Loads settings from a YAML file and provides typed defaults.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import yaml


_DEFAULT_REGIME_ADJ = {
    "Bull":     {"score_mult": 1.2, "position_scale": 1.0},
    "Bear":     {"score_mult": 0.8, "position_scale": 0.8},
    "Sideways": {"score_mult": 1.0, "position_scale": 0.7},
    "Crisis":   {"score_mult": 0.6, "position_scale": 0.5},
}


@dataclass
class BacktestConfig:
    # Tickers: any list of symbols (config, CLI, or fallback). No fixed embedded list required.
    tickers: list[str] = field(default_factory=list)

    # Market data: provider and cache
    data_provider: str = "yahoo"       # "yahoo" | "alpaca" | "finnhub"
    cache_ohlcv: bool = True          # cache downloaded OHLCV to avoid repeated API calls
    cache_dir: str = "data/cache/ohlcv"
    cache_ttl_days: int = 0            # 0 = use cache indefinitely

    # Backtest window
    start_date: str = "2018-01-01"
    end_date: str = "2024-01-01"

    # Capital management
    initial_capital: float = 100_000
    max_positions: int = 10
    holding_period_days: int = 5

    # Cross-sectional ranking portfolio (institutional-style top/bottom selection)
    cross_sectional_ranking: bool = False
    top_longs: int = 5
    top_shorts: int = 5
    market_neutral: bool = False  # if True, add bottom TOP_SHORTS as shorts; else long-only from top
    cross_sectional_rebalance_daily: bool = False  # if True, close positions not in today's target before opens
    daily_positions_csv_path: str = "output/portfolio/daily_positions.csv"

    # Risk & position management
    position_sizing: str = "equal"           # "equal" | "vol_scaled" | "kelly" | "risk_parity" | "mv_max_sharpe" | "mv_min_variance"
    position_sizing_method: str = "equal"    # same options as position_sizing
    # Mean-Variance (Markowitz) sizing when position_sizing is mv_max_sharpe or mv_min_variance
    mean_variance_enabled: bool = False
    mean_variance_method: str = "max_sharpe"  # "max_sharpe" | "min_variance"
    mean_variance_lookback_days: int = 60
    mean_variance_rebalance_days: int = 20
    mean_variance_max_single_weight: float = 0.25
    vol_target_annual: float = 0.15          # target annual vol for vol_scaled sizing
    vol_lookback_days: int = 20             # rolling window for volatility estimate
    kelly_fraction: float = 0.5             # fraction of full Kelly when position_sizing="kelly"
    kelly_win_rate: float = 0.55            # used for Kelly (or from backtest stats)
    kelly_avg_win_return: float = 0.02     # average winning return
    kelly_avg_loss_return: float = 0.015   # average losing return (positive number)
    stop_loss_pct: float = 0.0             # 0 = disabled; e.g. 0.02 = -2% exit
    take_profit_pct: float = 0.0           # 0 = disabled; e.g. 0.05 = +5% exit
    max_position_pct_of_equity: float = 0.25  # max single position as fraction of equity (e.g. 0.25 = 25%)
    vol_scaling_enabled: bool = True
    vol_scaling_target: float = 0.15
    max_drawdown_pct: float = 0.20         # circuit breaker: halt new trades beyond this DD (e.g. 0.20 = -20%)
    drawdown_resume_pct: float = 0.10      # resume trading once DD improves above this level
    severe_drawdown_close_all_pct: float = 0.0  # 0=disabled; e.g. 0.30 = close all positions if DD worse than -30%

    # Dynamic holding (optional): hold longer for stronger/different signal types
    dynamic_holding_enabled: bool = False
    holding_period_by_signal: dict[str, int] = field(default_factory=lambda: {"Bullish": 5, "Bearish": 3})
    holding_period_by_strength: list[tuple[float, int]] = field(default_factory=lambda: [(0.5, 10), (0.3, 5)])

    # Execution
    slippage_bps: float = 5.0
    commission_per_trade: float = 1.0
    allow_shorts: bool = False  # True = allow short trades; False = bullish-only for validation
    enable_shorts: bool = False  # Alias for allow_shorts (kept for backward compatibility)
    execution_delay_days: int = 0  # Monte Carlo: delay entry by N days (0 = next day)
    # Long-only override: when True, ignore bearish signals and do not open shorts.
    long_only: bool = True

    # Execution costs (realistic trading: commission + spread + slippage in bps)
    execution_costs_enabled: bool = True
    execution_costs_commission_bps: float = 2.0
    execution_costs_spread_bps: float = 2.0
    execution_costs_slippage_bps: float = 1.0
    execution_costs_sensitivity_test: bool = False
    execution_costs_scenarios: list[float] = field(default_factory=lambda: [5.0, 10.0, 20.0])  # total bps
    execution_costs_sensitivity_report_path: str = "output/research/cost_sensitivity.csv"

    # Signal filtering
    min_signal_strength: float = 0.3
    signal_mode: str = "price"      # "price" = trend+vol, "full" = all agents, "learned" = dynamic weights
    signal_weights: dict = field(default_factory=lambda: {
        "trend": 1.0,
        "regional_news": 0.5,
        "global_news": 0.4,
        "social": 0.3,
    })
    learned_weights_path: str = ""  # path to learned_weights.json (used when signal_mode="learned")
    signal_smoothing_span: int = 5
    signal_smoothing_enabled: bool = True
    signal_flip_threshold: float = 0.15
    dynamic_thresholds_enabled: bool = True
    base_signal_threshold: float = 0.5

    # Market regime
    regime_enabled: bool = True
    regime_adjustments: dict = field(default_factory=lambda: dict(_DEFAULT_REGIME_ADJ))
    regime_exit_on_change: bool = False

    # Sector allocation and signal adjustment
    sector_enabled: bool = False
    max_sector_exposure: float = 0.3              # legacy: fraction of positions
    max_sector_exposure_pct: float = 0.3          # fraction of equity/capital per sector (0.3 = 30%)
    sector_adjustment_enabled: bool = False
    sector_momentum_weight: float = 0.1
    sector_volatility_weight: float = -0.05
    sector_sentiment_weight: float = 0.1

    # Factor exposure neutralization (remove market/sector/size bias from signals)
    factor_neutralization_enabled: bool = False
    factor_neutralize_market_beta: bool = True
    factor_neutralize_sector: bool = True
    factor_neutralize_size: bool = True
    factor_market_index: str = "SPY"
    factor_rolling_window: int = 60
    factor_exposures_path: str = "output/research/factor_exposures.csv"
    factor_neutralization_report_path: str = "output/research/factor_neutralization_report.json"

    # Monte Carlo robustness testing
    monte_carlo_enabled: bool = False
    monte_carlo_runs: int = 500
    monte_carlo_signal_noise: float = 0.02
    monte_carlo_max_execution_delay_days: int = 2
    monte_carlo_results_path: str = "output/research/monte_carlo_results.csv"
    monte_carlo_summary_path: str = "output/research/monte_carlo_summary.json"
    monte_carlo_equity_curves_path: str = "output/research/monte_carlo_equity_curves.csv"
    monte_carlo_equity_plot_path: str = "output/research/monte_carlo_equity_distribution.png"
    monte_carlo_robustness_report_path: str = "output/research/monte_carlo_robustness_report.json"

    # Simulation (GBM features in pipeline; optional)
    gbm_enabled: bool = False

    # Research / analytics
    walk_forward_enabled: bool = True
    train_years: int = 5
    test_years: int = 1
    step_years: int = 1
    walk_forward_windows: int = 4
    walk_forward_train_ratio: float = 0.7
    walk_forward_train_weights: bool = True   # train weight model on train window before OOS backtest (learned mode)
    walk_forward_report_path: str = "output/backtests/walk_forward_validation_report.csv"
    ic_decay_lags: list[int] = field(default_factory=lambda: [1, 5, 10, 20])
    cost_sensitivity_scenarios: list[dict] | None = None  # optional; else use DEFAULT_COST_SCENARIOS

    # Options analysis (Black-Scholes at entry; adds columns to trades.csv)
    options_analysis: bool = False
    options_expiry_days: int = 30
    options_risk_free_rate: float = 0.04

    # Output paths
    save_trades_csv: bool = True
    save_equity_csv: bool = True
    equity_curve_path: str = "output/backtests/equity_curve.png"
    trades_csv_path: str = "output/backtests/trades.csv"
    equity_csv_path: str = "output/backtests/daily_equity.csv"
    cost_sensitivity_report_path: str = "output/backtests/cost_sensitivity_report.csv"


def load_config(path: str = "backtest_config.yaml") -> BacktestConfig:
    """Read YAML and merge into a BacktestConfig, using defaults for missing keys."""
    try:
        with open(path) as fh:
            raw = yaml.safe_load(fh) or {}
    except FileNotFoundError:
        raw = {}

    cfg = BacktestConfig()

    cfg.tickers = raw.get("tickers", cfg.tickers) or []

    data = raw.get("data", {})
    cfg.data_provider = data.get("provider", cfg.data_provider)
    cfg.cache_ohlcv = data.get("cache_ohlcv", cfg.cache_ohlcv)
    cfg.cache_dir = data.get("cache_dir", cfg.cache_dir)
    cfg.cache_ttl_days = int(data.get("cache_ttl_days", cfg.cache_ttl_days))

    bt = raw.get("backtest", {})
    cfg.start_date = bt.get("start_date", cfg.start_date)
    cfg.end_date = bt.get("end_date", cfg.end_date)
    cfg.initial_capital = float(bt.get("initial_capital", cfg.initial_capital))
    cfg.max_positions = int(bt.get("max_positions", cfg.max_positions))
    cfg.holding_period_days = int(bt.get("holding_period_days", cfg.holding_period_days))
    cs = bt.get("cross_sectional", {})
    cfg.cross_sectional_ranking = cs.get("enabled", cfg.cross_sectional_ranking)
    cfg.top_longs = int(cs.get("top_longs", cfg.top_longs))
    cfg.top_shorts = int(cs.get("top_shorts", cfg.top_shorts))
    cfg.market_neutral = cs.get("market_neutral", cfg.market_neutral)
    cfg.cross_sectional_rebalance_daily = cs.get("rebalance_daily", cfg.cross_sectional_rebalance_daily)
    cfg.daily_positions_csv_path = cs.get("daily_positions_csv_path", cfg.daily_positions_csv_path)
    dh = bt.get("dynamic_holding", {})
    cfg.dynamic_holding_enabled = dh.get("enabled", cfg.dynamic_holding_enabled)
    if "by_signal" in dh:
        cfg.holding_period_by_signal = dict(dh["by_signal"])
    if "by_strength" in dh:
        cfg.holding_period_by_strength = [tuple(b) for b in dh["by_strength"]]

    risk = raw.get("risk", {})
    cfg.position_sizing = risk.get("position_sizing", cfg.position_sizing)
    cfg.position_sizing_method = risk.get(
        "position_sizing_method",
        risk.get("position_sizing", cfg.position_sizing_method),
    )
    cfg.vol_target_annual = float(risk.get("vol_target_annual", cfg.vol_target_annual))
    cfg.vol_lookback_days = int(risk.get("vol_lookback_days", cfg.vol_lookback_days))
    cfg.vol_scaling_enabled = risk.get("vol_scaling_enabled", cfg.vol_scaling_enabled)
    cfg.vol_scaling_target = float(risk.get("vol_scaling_target", cfg.vol_scaling_target))
    cfg.kelly_fraction = float(risk.get("kelly_fraction", cfg.kelly_fraction))
    cfg.kelly_win_rate = float(risk.get("kelly_win_rate", cfg.kelly_win_rate))
    cfg.kelly_avg_win_return = float(risk.get("kelly_avg_win_return", cfg.kelly_avg_win_return))
    cfg.kelly_avg_loss_return = float(risk.get("kelly_avg_loss_return", cfg.kelly_avg_loss_return))
    cfg.stop_loss_pct = float(risk.get("stop_loss_pct", cfg.stop_loss_pct))
    cfg.take_profit_pct = float(risk.get("take_profit_pct", cfg.take_profit_pct))
    cfg.max_position_pct_of_equity = float(risk.get("max_position_pct_of_equity", cfg.max_position_pct_of_equity))
    cfg.max_drawdown_pct = float(risk.get("max_drawdown_pct", cfg.max_drawdown_pct))
    cfg.drawdown_resume_pct = float(risk.get("drawdown_resume_pct", cfg.drawdown_resume_pct))
    cfg.severe_drawdown_close_all_pct = float(
        risk.get("severe_drawdown_close_all_pct", cfg.severe_drawdown_close_all_pct)
    )
    mv = raw.get("mean_variance", {})
    cfg.mean_variance_enabled = mv.get("enabled", cfg.mean_variance_enabled)
    cfg.mean_variance_method = str(mv.get("method", cfg.mean_variance_method))
    cfg.mean_variance_lookback_days = int(mv.get("lookback_days", cfg.mean_variance_lookback_days))
    cfg.mean_variance_rebalance_days = int(mv.get("rebalance_days", cfg.mean_variance_rebalance_days))
    cfg.mean_variance_max_single_weight = float(mv.get("max_single_weight", cfg.mean_variance_max_single_weight))
    if cfg.mean_variance_enabled:
        cfg.position_sizing = "mv_max_sharpe" if cfg.mean_variance_method == "max_sharpe" else "mv_min_variance"
        cfg.position_sizing_method = cfg.position_sizing

    ex = raw.get("execution", {})
    cfg.slippage_bps = float(ex.get("slippage_bps", cfg.slippage_bps))
    cfg.commission_per_trade = float(ex.get("commission_per_trade", cfg.commission_per_trade))
    cfg.allow_shorts = ex.get("allow_shorts", ex.get("enable_shorts", cfg.allow_shorts))
    cfg.enable_shorts = cfg.allow_shorts  # keep in sync for backward compatibility
    cfg.long_only = ex.get("long_only", cfg.long_only)

    ec = raw.get("execution_costs", {})
    cfg.execution_costs_enabled = ec.get("enabled", cfg.execution_costs_enabled)
    cfg.execution_costs_commission_bps = float(ec.get("commission_bps", cfg.execution_costs_commission_bps))
    cfg.execution_costs_spread_bps = float(ec.get("spread_bps", cfg.execution_costs_spread_bps))
    cfg.execution_costs_slippage_bps = float(ec.get("slippage_bps", cfg.execution_costs_slippage_bps))
    cfg.execution_costs_sensitivity_test = ec.get("sensitivity_test", cfg.execution_costs_sensitivity_test)
    if "scenarios" in ec:
        cfg.execution_costs_scenarios = [float(s) for s in ec["scenarios"]]
    cfg.execution_costs_sensitivity_report_path = str(
        ec.get("sensitivity_report_path", cfg.execution_costs_sensitivity_report_path)
    )
    # Cross-sectional short leg requires shorts
    if getattr(cfg, "cross_sectional_ranking", False) and getattr(cfg, "market_neutral", False):
        cfg.allow_shorts = True
        cfg.enable_shorts = True

    sig = raw.get("signals", {})
    cfg.min_signal_strength = float(sig.get("min_signal_strength", cfg.min_signal_strength))
    cfg.signal_mode = sig.get("mode", cfg.signal_mode)
    if "weights" in sig:
        cfg.signal_weights.update(sig["weights"])
    cfg.learned_weights_path = sig.get("learned_weights_path", cfg.learned_weights_path)
    cfg.signal_smoothing_span = int(sig.get("smoothing_span", cfg.signal_smoothing_span))
    cfg.signal_smoothing_enabled = sig.get("smoothing_enabled", cfg.signal_smoothing_enabled)
    cfg.signal_flip_threshold = float(sig.get("flip_threshold", cfg.signal_flip_threshold))
    cfg.dynamic_thresholds_enabled = sig.get(
        "dynamic_thresholds_enabled", cfg.dynamic_thresholds_enabled
    )
    cfg.base_signal_threshold = float(
        sig.get(
            "base_signal_threshold",
            sig.get("base_threshold", cfg.base_signal_threshold),
        )
    )

    reg = raw.get("regime", {})
    cfg.regime_enabled = reg.get("enabled", cfg.regime_enabled)
    cfg.regime_adjustments = reg.get("adjustments", cfg.regime_adjustments)
    cfg.regime_exit_on_change = reg.get("exit_on_change", cfg.regime_exit_on_change)

    sec = raw.get("sectors", {})
    cfg.sector_enabled = sec.get("enabled", cfg.sector_enabled)
    cfg.max_sector_exposure = float(sec.get("max_exposure", cfg.max_sector_exposure))
    cfg.max_sector_exposure_pct = float(
        sec.get(
            "max_exposure_pct",
            sec.get("max_exposure", cfg.max_sector_exposure_pct),
        )
    )
    cfg.sector_adjustment_enabled = sec.get("adjustment_enabled", cfg.sector_adjustment_enabled)
    cfg.sector_momentum_weight = float(sec.get("momentum_weight", cfg.sector_momentum_weight))
    cfg.sector_volatility_weight = float(sec.get("volatility_weight", cfg.sector_volatility_weight))
    cfg.sector_sentiment_weight = float(sec.get("sentiment_weight", cfg.sector_sentiment_weight))

    fn = raw.get("factor_neutralization", {})
    cfg.factor_neutralization_enabled = fn.get("enabled", cfg.factor_neutralization_enabled)
    cfg.factor_neutralize_market_beta = fn.get("neutralize_market_beta", cfg.factor_neutralize_market_beta)
    cfg.factor_neutralize_sector = fn.get("neutralize_sector", cfg.factor_neutralize_sector)
    cfg.factor_neutralize_size = fn.get("neutralize_size", cfg.factor_neutralize_size)
    cfg.factor_market_index = str(fn.get("market_index", cfg.factor_market_index))
    cfg.factor_rolling_window = int(fn.get("rolling_window", cfg.factor_rolling_window))
    cfg.factor_exposures_path = str(fn.get("factor_exposures_path", cfg.factor_exposures_path))
    cfg.factor_neutralization_report_path = str(
        fn.get("factor_neutralization_report_path", cfg.factor_neutralization_report_path)
    )

    mc = raw.get("monte_carlo", {})
    cfg.monte_carlo_enabled = mc.get("enabled", cfg.monte_carlo_enabled)
    cfg.monte_carlo_runs = int(mc.get("runs", cfg.monte_carlo_runs))
    cfg.monte_carlo_signal_noise = float(mc.get("signal_noise", cfg.monte_carlo_signal_noise))
    cfg.monte_carlo_max_execution_delay_days = int(
        mc.get("max_execution_delay_days", cfg.monte_carlo_max_execution_delay_days)
    )
    cfg.monte_carlo_results_path = str(mc.get("results_path", cfg.monte_carlo_results_path))
    cfg.monte_carlo_summary_path = str(mc.get("summary_path", cfg.monte_carlo_summary_path))
    cfg.monte_carlo_equity_curves_path = str(mc.get("equity_curves_path", cfg.monte_carlo_equity_curves_path))
    cfg.monte_carlo_equity_plot_path = str(mc.get("equity_plot_path", cfg.monte_carlo_equity_plot_path))
    cfg.monte_carlo_robustness_report_path = str(
        mc.get("robustness_report_path", cfg.monte_carlo_robustness_report_path)
    )

    sim = raw.get("simulation", {})
    cfg.gbm_enabled = sim.get("gbm_enabled", cfg.gbm_enabled)

    res = raw.get("research", {})
    cfg.walk_forward_enabled = res.get("walk_forward_enabled", cfg.walk_forward_enabled)
    cfg.train_years = int(res.get("train_years", cfg.train_years))
    cfg.test_years = int(res.get("test_years", cfg.test_years))
    cfg.step_years = int(res.get("step_years", cfg.step_years))
    cfg.walk_forward_windows = int(res.get("walk_forward_windows", cfg.walk_forward_windows))
    cfg.walk_forward_train_ratio = float(res.get("walk_forward_train_ratio", cfg.walk_forward_train_ratio))
    cfg.walk_forward_train_weights = res.get("walk_forward_train_weights", cfg.walk_forward_train_weights)
    cfg.walk_forward_report_path = res.get("walk_forward_report_path", cfg.walk_forward_report_path)
    cfg.ic_decay_lags = res.get("ic_decay_lags", cfg.ic_decay_lags)
    if "cost_sensitivity_scenarios" in res:
        cfg.cost_sensitivity_scenarios = [dict(s) for s in res["cost_sensitivity_scenarios"]]

    opt = raw.get("options", {})
    cfg.options_analysis = opt.get("analysis", cfg.options_analysis)
    cfg.options_expiry_days = int(opt.get("expiry_days", cfg.options_expiry_days))
    cfg.options_risk_free_rate = float(opt.get("risk_free_rate", cfg.options_risk_free_rate))

    out = raw.get("output", {})
    cfg.save_trades_csv = out.get("save_trades_csv", cfg.save_trades_csv)
    cfg.save_equity_csv = out.get("save_equity_csv", cfg.save_equity_csv)
    cfg.equity_curve_path = out.get("equity_curve_path", cfg.equity_curve_path)
    cfg.trades_csv_path = out.get("trades_csv_path", cfg.trades_csv_path)
    cfg.equity_csv_path = out.get("equity_csv_path", cfg.equity_csv_path)
    cfg.cost_sensitivity_report_path = out.get("cost_sensitivity_report_path", cfg.cost_sensitivity_report_path)

    return cfg
