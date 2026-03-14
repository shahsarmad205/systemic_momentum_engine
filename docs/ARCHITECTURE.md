# Modular Research Platform — Architecture

## Layout

| Module | Path | Role |
|--------|------|------|
| **data** | `data/` | `MarketDataLoader` — cached + API OHLCV (`load_price_history`, `load_volume_history`) |
| **features** | `features/` | `FeatureEngine.build_features(df)` — delegates to `agents.trend_agent.feature_engineering` |
| **signals** | `signals/` | `SignalEngineFacade` — wraps `backtesting.signals.SignalEngine`; `compute_signal_score(...)` |
| **strategy** | `strategy/` | `StrategyEngine` + `build_ranked_candidates` (per-signal) + `build_cross_sectional_candidates` (top/bottom daily rank, optional market-neutral shorts) |
| **portfolio** | `portfolio/` | `PortfolioEngine` — `open_position` / `close_position` / `update_portfolio` over `backtesting.portfolio.Portfolio` |
| **backtest** | `backtest/` | `BacktestEngine.run_backtest()` — coordinates simulation via `Backtester`; experiment snapshots under `output/experiments/` |
| **analytics** | `analytics/` | `PerformanceAnalyzer` — `compute_performance_metrics`, IC decay helpers, `generate_reports` |
| **dashboard** | `dashboard/` | `prepare_equity_series`, `prepare_trades_summary` for visualization consumers |
| **legacy** | `backtesting/` | Unchanged math: `Backtester`, `Portfolio`, `SignalEngine`, `metrics`, `analytics`, `execution` |
| **execution** | `execution/` | `TransactionCostModel` — commission + spread + slippage in bps; cost_dollars(position_size) per leg |
| **research** | `research/` | `WalkForwardValidator`; `FactorNeutralizer`; `MonteCarloRobustnessTester` — walk-forward, factor neutralization, Monte Carlo robustness |

## Configuration

- **`config.py`** (project root): `DEV_MODE`, `DEV_TICKER_LIMIT`, `get_effective_tickers`, `setup_logging`; re-exports `load_config` / `BacktestConfig`.
- **`backtest_config.yaml`** remains the primary YAML; loaded via `backtesting.config.load_config`.

## Backward compatibility

- **`run_backtest.py`** — uses `BacktestEngine`; still supports `--walk-forward`, `--ic-decay`, `--cost-sensitivity`.
- **`run_walk_forward.py`** — dedicated walk-forward entry point: loads config, runs `WalkForwardValidator`, prints summary, saves outputs to `output/experiments/<timestamp>/` (or `output/research/` with `--no-experiment`).
- **`run_weight_learning.py`** — uses `config.get_effective_tickers` for dev ticker cap.

## Experiments

Each standard backtest run writes a timestamped folder under `output/experiments/<UTC>/`:

- `trades.csv`, `daily_equity.csv`, `metrics.json`, optional `backtest_config.yaml` copy

## Walk-Forward Research Pipeline

The platform provides a **walk-forward validation** pipeline designed to prevent overfitting and deliver **true out-of-sample (OOS) performance evaluation** suitable for institutional quant research.

### What It Does

- **Calendar-year rolling windows**: Training and test periods are defined in calendar years (e.g. train 5 years, test 1 year, step 1 year). No test-date information is used during training.
- **Train → freeze → test**: For each window, a weight model (e.g. ridge regression) is trained on the train period only. Weights are frozen; the backtester runs **only** on the test window with those weights. No retraining on test data.
- **Strict temporal split**: Test windows are always strictly after the training window. Data from the test period is never used for fitting or feature construction for that segment.

This design **avoids look-ahead bias** and **prevents overfitting** to the test set: each test window is genuinely unseen at training time.

### How Overfitting Is Prevented

| Concern | Mitigation |
|--------|------------|
| Fitting on test data | Train/test split is strictly temporal; test dates are excluded from training and from any weight learning. |
| Peeking at future returns | Backtester runs only over the test window; signals and weights depend only on data up to the end of the train window. |
| Reusing same data for train and test | Each segment uses disjoint train and test periods. No overlap. |
| Optimizing on OOS window | Weights are fixed before the test run; no gradient or tuning step sees test-period outcomes. |

The result is **true OOS evaluation**: reported metrics (Sharpe, return, max drawdown, win rate, profit factor, IC) for each test window reflect performance of a strategy that was fully specified before that window began.

### Pipeline Components

- **`research/walk_forward.py`** — `WalkForwardValidator`: generates calendar-year windows, trains `WeightLearner` on each train window, runs `Backtester` on the corresponding test window with frozen weights, and aggregates per-period metrics and chained equity.
- **`run_walk_forward.py`** — CLI entry point: loads config, applies dev-mode limits, runs the validator, prints summary and per-period table, and writes all outputs to an experiment directory (or `output/research/` if `--no-experiment`).
- **Config** (`backtest_config.yaml` under `research:`): `walk_forward_enabled`, `train_years`, `test_years`, `step_years` control the rolling scheme.

### Outputs and Experiment Logging

Each walk-forward run can write a **timestamped experiment directory** `output/experiments/<timestamp>/` containing:

| Artifact | Description |
|----------|-------------|
| `config_snapshot.yaml` | Full backtest (and research) config used for the run; ensures reproducibility. |
| `walk_forward_results.csv` | One row per test period: train/test dates, Sharpe, total return, max drawdown, win rate, average return, profit factor, information coefficient, rank IC, trade counts. |
| `summary.json` | Aggregate statistics across windows: mean/median Sharpe, mean return, mean drawdown, mean IC, segment count. |
| `equity_curve.png` | Chained OOS equity curve (each segment’s equity scaled so the series is continuous). |
| `walk_forward_equity.csv` | Daily chained equity series. |

These outputs support **reproducible, auditable research**: every run is tied to a config snapshot and a full set of OOS metrics and curves.

### Institutional-Level Rigor

The pipeline is built to support **institutional-style quant research**:

- **No in-sample leakage**: Training uses only data before the test window; test performance is out-of-sample by construction.
- **Explicit train/test boundaries**: Calendar-year windows are documented in the results CSV and in the validator output.
- **Full config capture**: Each experiment stores a config snapshot so runs can be reproduced and compared.
- **Rich OOS metrics**: Per-period and aggregate statistics (Sharpe, return, drawdown, win rate, profit factor, IC) are computed and saved for reporting and compliance.
- **Single entry point**: `python run_walk_forward.py` (with optional `--config`, `--no-experiment`) runs the full pipeline and produces reports suitable for internal or institutional review.

Researchers can run rigorous walk-forward experiments and produce **out-of-sample performance reports** that clearly separate in-sample fitting from OOS evaluation and that are suitable for institutional-level quant research and reporting.

## Transaction Cost and Slippage Modeling

Realistic execution modeling is necessary to avoid **optimistic backtest bias**. The platform models transaction costs and slippage so that backtest results reflect realistic trading conditions.

### Transaction Cost Model

- **Module**: `execution/cost_model.py` — `TransactionCostModel` computes total execution cost per trade leg (entry or exit) from:
  - **Commission** (basis points of notional)
  - **Bid–ask spread** (basis points)
  - **Slippage** (basis points; market impact / worse fill)
- **Formula**: `total_cost_bps = commission_bps + spread_bps + slippage_bps`. Cost in dollars: `cost_dollars = position_size * (total_bps / 10_000)`.
- **Default assumptions** (configurable in `backtest_config.yaml` under `execution_costs:`): e.g. `commission_bps: 2`, `spread_bps: 2`, `slippage_bps: 1` → ~5 bps per leg (~10 bps round-trip).
- When **`execution_costs.enabled: false`**, the backtest runs without these costs (zero cost model) for research comparisons.

### How Slippage Is Simulated

- **Price adjustment** (in `backtesting/execution.py`): executed price is worsened so that:
  - **Long entry**: `execution_price = market_price * (1 + slippage_bps/10_000)` (buy higher).
  - **Short entry**: `execution_price = market_price * (1 - slippage_bps/10_000)` (short lower).
  - Exits are symmetric (sell lower for longs, cover higher for shorts). This reflects worse fills due to market impact and spread.
- When execution costs are **enabled**, the same `slippage_bps` from the cost model is used for both this price adjustment and the slippage component of the dollar cost.

### How Costs Affect Portfolio Returns

- **Per trade**: On open, `entry_cost = cost_model.cost_dollars(position_size)` is deducted from cash. On close, `exit_cost = cost_model.cost_dollars(position_size)` is deducted; **net return** = gross return minus (entry_cost + exit_cost) in dollar terms, so `net_return = gross_return - total_cost/position_size` (as a fraction).
- **Portfolio equity**: Cash and PnL already reflect costs (costs are subtracted when opening and closing). Daily equity and final capital are **after** all transaction costs.
- **Metrics**: The report includes **Gross Return** (before costs), **Net Return** (after costs), **Total Transaction Costs**, and **Average Cost per Trade**. Trades CSV includes `entry_cost`, `exit_cost`, `total_cost`, `gross_return`, `net_return` for analysis.

### Cost Sensitivity Testing

- Optional **stress testing** (`execution_costs.sensitivity_test: true` and `scenarios: [5, 10, 20]`) runs additional backtests with total cost set to 5, 10, and 20 bps per leg (split across commission/spread/slippage).
- Results are written to **`output/research/cost_sensitivity.csv`** (path configurable) with columns such as `total_bps`, `total_return`, `gross_return`, `net_return`, `sharpe_ratio`, `max_drawdown`, `total_transaction_costs`, etc. This verifies whether the strategy remains profitable under higher trading costs.

### Why This Matters

Without realistic execution modeling, backtests tend to **overstate** performance: live trading incurs commissions, spread, and slippage. By applying a configurable cost model and slippage to every trade and reporting gross vs net returns, the platform produces **cost-aware backtests** that are closer to real-world performance and help researchers avoid strategies that are only profitable in a frictionless backtest.

## Factor Exposure Neutralization

Trading signals can be unintentionally driven by **common risk factors** (market beta, sector, size). The platform includes a **neutralization layer** so that signals represent **idiosyncratic alpha** rather than hidden factor bets. **Institutional quantitative funds** routinely neutralize signals against these factors to improve robustness and generalization.

### Factor Exposure Risk in Quant Models

- **Market beta**: Strategies that simply follow the market (high beta) can show strong backtest returns that are mostly systematic, not alpha.
- **Sector bias**: Overweighting technology or healthcare can look like skill but is sector exposure.
- **Size factor**: Tilting toward small or large caps can dominate signal strength.  
If left in the signal, these exposures **confound research**: backtest performance may not generalize, and live portfolios can have unintended factor tilts.

### How Neutralization Removes Systematic Biases

- **Placement**: Neutralization runs **after** signal generation and **before** cross-sectional ranking and portfolio construction. Flow: features → signal model → **factor neutralization** → ranking → portfolio.
- **Method**: For each trading date, a **cross-sectional regression** is run:  
  `signal = β_market × market_beta + β_sector × sector_dummies + β_size × size_factor + residual`.  
  The **residual** is used as the neutralized signal. It has minimal correlation with the included factors.
- **Factors**:
  - **Market beta**: Rolling 60-day beta vs a market index (e.g. SPY). No look-ahead: uses only data up to that date.
  - **Sector**: One-hot encoding from the platform’s sector map (Technology, Healthcare, Financials, etc.); one category dropped to avoid collinearity.
  - **Size**: Proxy `log(price × average_volume)` (or `log(price)` if volume unavailable).
- **Configuration** (`backtest_config.yaml` → `factor_neutralization`): `enabled`, `neutralize_market_beta`, `neutralize_sector`, `neutralize_size`. If `enabled: false`, signals pass through unchanged.

### Why This Improves Robustness and Generalization

- **Cleaner alpha**: The residual is the part of the signal **orthogonal** to the chosen factors, so reported performance is closer to true stock-specific prediction.
- **Fewer unintended bets**: Portfolio construction then ranks on factor-neutral scores, reducing accidental market/sector/size tilts.
- **Walk-forward safe**: Factor values are computed **within** each backtest window using only data up to the current date, so there is no look-ahead; the same logic works in walk-forward validation.
- **Diagnostics**: The platform writes **`output/research/factor_exposures.csv`** (per-date factor correlations with signal before/after) and **`output/research/factor_neutralization_report.json`** (mean factor correlations before/after, and signal IC after neutralization). Researchers can verify that post-neutralization factor correlations are near zero and that the signal remains predictive (IC).

## Monte Carlo Robustness Testing

**Institutional quantitative research** uses Monte Carlo simulations to ensure strategies are **stable under uncertainty**, not the result of random luck or a lucky trade sequence. The platform includes a dedicated robustness tester that runs hundreds of randomized simulations and reports whether performance holds up.

### Why Monte Carlo Testing Matters

- A single backtest can be **overfit** or **lucky**: one favourable ordering of trades or one set of signal realizations may flatter the strategy.
- In live trading, **execution order**, **slippage**, and **signal noise** vary. If the strategy’s reported edge **disappears** under small perturbations, it is **fragile**.
- Monte Carlo testing **stress-tests** the strategy by applying controlled randomness and checking how often metrics (Sharpe, return, drawdown) remain acceptable.

### How Random Perturbations Stress-Test the Strategy

Four perturbation types are implemented:

1. **Trade order shuffle**: The sequence of trade returns is randomly permuted; the equity curve is rebuilt from this new order. This checks whether **drawdown and path** depend heavily on the specific order of wins and losses.
2. **Return bootstrapping**: Trade returns are **resampled with replacement** to form a new return sequence; equity and metrics are recomputed. This probes **sampling variation** and whether the edge is robust to different return draws.
3. **Signal noise injection**: Small Gaussian noise is added to signal scores before ranking (`noisy_signal = signal + N(0, noise_level)`). The full backtest is re-run with these noisy signals. This tests **sensitivity to small signal changes** (e.g. noise_level = 0.02).
4. **Execution delay simulation**: Trades are executed **0–2 days** (configurable) after the signal. The backtester supports `execution_delay_days`; each run can use a random delay to mimic **real-world execution lag**.

Each simulation records **Sharpe ratio**, **total return**, **max drawdown**, **win rate**, **profit factor**, and **information coefficient**. Results are written to **`output/research/monte_carlo_results.csv`**; summary stats and robustness diagnostics to JSON and a plot.

### How to Interpret Robustness Metrics

- **probability_sharpe_positive**: Fraction of runs with Sharpe > 0. High (e.g. > 90%) suggests the strategy is **often** risk-adjusted profitable under perturbation.
- **probability_profitable**: Fraction of runs with total return > 0. **High** (e.g. > 95%) suggests the strategy stays profitable in most perturbed scenarios.
- **percentile_5_return** and **percentile_95_return**: The 5th and 95th percentiles of total return across runs. **Narrow** range and **positive** 5th percentile indicate **stable** performance; a **negative** 5th percentile means a meaningful chance of loss under randomness.
- **Distribution of final equity** (saved as **`output/research/monte_carlo_equity_distribution.png`**): Shows how much final capital varies. A **tight** distribution around a positive level is more reassuring than a **wide** or **bimodal** one.

If most simulations remain profitable and Sharpe stays positive, the strategy is **robust**. If metrics **collapse** (e.g. median return turns negative, or most runs lose money), the strategy is **fragile** and likely not suitable for institutional use without further work.

### Configuration and Runner

- **Config** (`backtest_config.yaml` → `monte_carlo`): `enabled`, `runs` (default 500), `signal_noise`, `max_execution_delay_days`, and paths for results, summary, equity curves, and robustness report.
- **Runner**: **`python run_monte_carlo.py`** loads config, applies **DEV_MODE** (50 runs, 10 tickers when enabled), runs **MonteCarloRobustnessTester**, saves all outputs, and prints the robustness summary.

## Principle

New packages are **facades**: they expose clean interfaces without duplicating numerical logic. The backtester still owns the day-loop and execution slippage; strategy module owns cross-sectional ranking only.
