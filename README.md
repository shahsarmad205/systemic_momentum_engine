# Trend Signal Engine

A production-grade quantitative research system for systematic momentum signal generation, weight learning, portfolio optimization, and walk-forward validation.

---

## What this is

The system turns historical price data into momentum-based trading signals by combining multiple technical and fundamental-style inputs into a single score. It then trains machine-learning models across five model types (logistic, ridge, random forest, XGBoost, gradient boosting) to learn how to weight those inputs from past data. To avoid look-ahead bias, all out-of-sample evaluation is done with walk-forward validation: the model is trained only on data before each test window and tested on unseen future data. Six financial models are built in: half-Kelly position sizing with rolling win-rate estimation, CAPM-based beta and alpha, Black-Scholes theoretical option pricing on each signal, Value-at-Risk and Expected Shortfall for portfolio risk, Markowitz mean-variance optimization for weights, and Geometric Brownian Motion for price-path simulation and calibration checks.

---

## Quick start

```bash
# Install
pip install -r requirements.txt

# Run weight learning (logistic model — best OOS IC)
python run_weight_learning.py --model logistic

# Run full backtest
python run_backtest.py

# Run walk-forward validation (true OOS)
python run_backtest.py --walk-forward

# Run options analysis
python run_options_analysis.py

# Run VaR analysis
python run_var_analysis.py

# Run GBM simulation
python run_gbm_analysis.py
```

---

## Architecture

Market data is loaded via `utils/market_data.py` and cached locally. Features are built in `features/engine.py` and in `agents/trend_agent/feature_engineering.py`, which produce the inputs used for scoring. Signals are generated and thresholded in `backtesting/signals.py`, which consumes the feature matrices and outputs Bullish, Bearish, or Neutral per date and ticker. When using learned weights, the weight model is trained in `agents/weight_learning_agent/weight_model.py` on historical forward returns and the resulting coefficients are applied to the same features at inference time. The main simulation loop runs in `backtesting/backtester.py`, which steps through calendar days, applies regime filters, opens and closes positions, and records trades. Regime labels (Bull, Bear, Sideways, Crisis) come from `backtesting/regime.py` using SPY and VIX. The financial models live in dedicated modules: `options/black_scholes.py` for theoretical option pricing, `risk/var.py` for VaR and CVaR, `portfolio/mean_variance.py` for Markowitz optimization, and `simulation/gbm.py` for GBM simulation and calibration.

---

## Backtest results

Full backtest (2018–2024, 9 US large-cap tech tickers, long-only, Kelly position sizing):

| Metric | Value |
|--------|-------|
| Sharpe Ratio | 0.741 |
| Sortino Ratio | 0.911 |
| CAGR | 21.87% |
| Max Drawdown | -15.24% |
| Win Rate | 47.7% |
| Trades/year | 642 |
| Avg Hold | 3.6 days |
| Total Return | +227.5% |

Walk-forward OOS validation (weights retrained per window, no look-ahead):

| Window | Period | OOS Sharpe | Return | Dir Acc | OOS IC |
|--------|--------|------------|--------|---------|--------|
| 1 | Jan–Jul 2019 | 1.43 | +10.2% | 52.6% | 0.015 |
| 2 | Jul–Dec 2020 | 0.89 | +10.6% | 45.8% | 0.009 |
| 3 | Jan–Jun 2022 | -1.05 | -5.6% | 29.2% | 0.038 |
| 4 | Jul–Dec 2023 | 0.31 | +3.6% | 44.2% | 0.050 |
| **Mean** | | **0.40** | **+4.7%** | **43.0%** | **0.028** |

Regime breakdown:

| Regime | Sharpe | Notes |
|--------|--------|-------|
| Bull | 2.05 | Strong momentum environment |
| Bear | 0.82 | Moderate — signal holds |
| Sideways | 0.32 | Weak signal in choppy markets |
| Crisis | -4.10 | Long-only fails in sharp drawdowns |

Walk-forward OOS Sharpe of 0.40 vs full-period Sharpe of 0.741 represents a 46% degradation — within the acceptable range for systematic momentum strategies. Window 3 (2022 H1) fails consistently across all tested models, coinciding with the Federal Reserve rate hiking cycle and growth-to-value rotation. This is a known limitation of pure momentum factors and is documented rather than papered over.

---

## ML model comparison

| Model | WF IC | WF Dir Acc | WF AUC | Selected |
|-------|-------|------------|--------|----------|
| Logistic | 0.039 | 56.4% | 0.511 | ✓ |
| Ridge | 0.041 | 56.0% | 0.511 | |
| Random Forest | 0.033 | 56.5% | 0.504 | |
| XGBoost | 0.043 | 53.6% | 0.513 | |
| GBR | 0.028 | 54.8% | 0.503 | |

Logistic regression is selected as the production model: it is the simplest, has competitive IC, provides calibrated probability outputs, and has the lowest overfit risk. XGBoost shows the highest IC but the worst directional accuracy — a sign of overfit to return magnitude rather than direction.

---

## Financial models

**Kelly Criterion** (configured in `backtest_config.yaml` and used in the backtester): Position sizing uses half-Kelly with rolling estimates of win rate and average win/loss from the last 50 closed trades. When fewer than 20 trades are available, config defaults (e.g. 0.55 win rate, 0.02 avg win, 0.015 avg loss) are used. The result in the reference backtest is Sharpe 0.714 and average position size about 14.4% of equity.

**CAPM** (`backtesting/signals.py`, `features/engine.py`): Rolling 60-day regression of stock returns on SPY yields market beta and Jensen’s alpha; alpha is z-scored over a trailing 252-day window. The universe shows a mean beta of about 1.30 versus SPY, with alpha used as a cross-sectional signal after z-scoring.

**Black-Scholes** (`options/black_scholes.py`): On each signal entry the system prices theoretical at-the-money calls and puts using the Black-Scholes formula. Greeks (delta, gamma, theta, vega, rho) are computed at entry. Historical realised volatility (rolling standard deviation of log returns annualised) is used as an implied-volatility proxy; no live options data is required. Put-call parity is checked in validation.

**Value at Risk** (`risk/var.py`): Three methods are implemented: historical VaR (rolling quantile of past returns), parametric VaR (Gaussian, using mean and standard deviation), and conditional VaR (Expected Shortfall) as the average loss beyond the VaR threshold. For the validation series, historical VaR at 95% is about 2.94% and CVaR about 3.53%.

**Mean-Variance Optimization** (`portfolio/mean_variance.py`): Markowitz efficient frontier is approximated by simulating 1000 random long-only portfolios from the covariance matrix of returns. Max-Sharpe and min-variance weights are computed via `scipy.optimize.minimize` (SLSQP) with constraints that weights sum to one and each weight is between zero and a maximum (e.g. 0.25). The efficient frontier plot shows the scatter of simulated portfolios and the upper envelope.

**Geometric Brownian Motion** (`simulation/gbm.py`): Drift and volatility (mu, sigma) are estimated from historical log returns (mu = mean × 252, sigma = std × √252). Paths are simulated with the standard GBM formula; price targets and calibration are backtested by comparing simulated confidence intervals to realised outcomes. Validation reports coverage of the 95% CI; in the test run, mean log return over one year is 0.0818 vs theoretical 0.0800.

---

## Known limitations

The backtest universe is limited to nine large-cap US technology names, so results are not representative of a diversified multi-sector portfolio. The strategy is long-only, so it carries full market beta and performs poorly in Crisis regimes (e.g. Sharpe -4.10 when the regime classifier labels Crisis). Walk-forward out-of-sample Sharpe is about 46% lower than the full-period backtest Sharpe, which is typical for momentum strategies but should be expected when moving to live or paper trading. The full-period information coefficient is negative (-0.0086) in the backtest; reported returns are driven by position sizing (Kelly), regime filtering, and execution rules rather than by raw directional prediction. Turnover is high (642 trades per year), so more realistic transaction cost assumptions would reduce net returns relative to the current conservative cost setup.

---

## Future work

- Regime-adaptive signal gating using HMM (replace hardcoded SPY+VIX thresholds)
- Multi-sector universe expansion for cross-sectional breadth
- Factor neutralization (market beta, sector, size) to isolate stock-specific momentum
- Live paper trading integration via `pipeline/daily_runner.py`

---

This system is a research tool. Nothing here constitutes investment advice. Past backtest performance does not guarantee future results.
