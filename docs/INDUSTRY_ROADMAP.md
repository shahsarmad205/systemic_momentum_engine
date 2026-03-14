# Industry-Level Roadmap: Making the Trend Signal Engine Institutional-Grade

This document outlines what “industry level” means for a systematic equity/trend engine and gives a **prioritized, actionable plan** to get there. You already have: learned weights, walk-forward, factor neutralization, regime logic, Monte Carlo, and execution costs. The gaps are in **alpha quality**, **risk controls**, **validation rigor**, and **operational discipline**.

---

## 1. Alpha & Signal (Make the Edge Real)

**Goal:** Improve predictive power and robustness of the signal so it works across regimes and time.

| Priority | What to do | Why it matters |
|----------|------------|----------------|
| **P0** | **Walk-forward as the gate** — Never trust in-sample backtest. Always report OOS walk-forward Sharpe/CAGR and use it to choose models. Run `python run_weight_learning.py --compare` and `python run_backtest.py --walk-forward` and **optimize for OOS**, not single-window backtest. | In-sample overfitting is the #1 reason retail/systematic strategies fail live. |
| **P0** | **Target forward returns, not labels** — In weight learning, use `--target-type regression` and `--return-target excess` (excess over SPY) or `sharpe_scaled` so the model learns *magnitude* of opportunity, not just direction. Tune `--holding-period` (e.g. 5, 10, 20 days) via OOS. | Industry models predict forward returns or risk-adjusted returns; classification alone is weak. |
| **P1** | **Regime-specific models** — You have `--regime` in weight learning. Train separate weights for Bull/Bear/Sideways/Crisis and use them in the backtester (you already have regime in the loop). Validate each regime’s OOS performance. | Reduces drawdowns in bad regimes and improves risk-adjusted returns. |
| **P1** | **Cross-sectional rank, not level** — Use `cross_sectional.enabled: true` with `top_longs: 5` (or 10). Rank by `adjusted_score` and take top N; rebalance daily or weekly. This is how most quant equity funds work (long top quintile, avoid bottom). | Level scores are noisy; rank is more stable and tradable. |
| **P2** | **More/better features** — Add: short-term momentum (e.g. 5d, 10d), earnings/revision proxies if you have data, volume profile (you have some), and simple mean-reversion for oversold bounces. Keep feature set small enough to avoid overfitting; validate with IC decay and stability. | Richer feature set improves signal; stability analysis keeps you honest. |
| **P2** | **Signal decay / horizon** — Run `python run_backtest.py --ic-decay` and pick the forward horizon (e.g. 5d or 10d) where IC is highest and most stable. Align `holding_period_days` with that. | Matching hold period to signal horizon improves hit rate and turnover. |

---

## 2. Risk & Execution (Protect Capital and Control Drawdowns)

**Goal:** Size positions by risk, cap drawdowns, and trade realistically.

| Priority | What to do | Why it matters |
|----------|------------|----------------|
| **P0** | **Vol-targeted sizing** — In `backtest_config.yaml` set `risk.position_sizing: "vol_scaled"` and `vol_target_annual: 0.10`–0.15. You already have the code; enable it. Size each position so that contribution to portfolio vol is roughly equal. | Industry standard: equal risk per position, not equal dollar. |
| **P0** | **Stop-loss** — Set `risk.stop_loss_pct: 0.02` or 0.03 (2–3% per position). Your backtester already supports it. This cuts tail losses and improves Sharpe even if win rate drops slightly. | Large single-trade losses (e.g. -14% on META) kill compounded returns. |
| **P1** | **Circuit breaker for live** — Keep `max_drawdown_pct: 0` in research; for live or paper trading, set e.g. 10–15% and `drawdown_resume_pct` slightly lower so you pause new entries until drawdown recovers. | Prevents runaway drawdowns in production. |
| **P1** | **Sector/name concentration** — Turn on `sectors.enabled: true` and `max_exposure: 0.25`–0.30 so no single sector dominates. Optionally cap single-name weight (e.g. max 15% of equity per ticker). | Avoids blow-ups from one sector or name. |
| **P2** | **Realistic costs** — Use `execution_costs.enabled: true` with realistic bps (e.g. 5–10 bps total for large-cap US). Run cost sensitivity and ensure strategy is still positive after costs. | Many strategies are marginal before costs and negative after. |

---

## 3. Validation & Robustness (Prove It’s Not Luck)

**Goal:** Demonstrate that the edge is stable and not overfit.

| Priority | What to do | Why it matters |
|----------|------------|----------------|
| **P0** | **Report OOS only for decisions** — When you tune anything (holding period, min_signal_strength, model type), choose by **walk-forward OOS** (e.g. last 1–2 years out-of-sample). Never choose by in-sample or single backtest. | Only OOS is a fair estimate of live performance. |
| **P1** | **Monte Carlo** — You have 500 runs with signal noise and execution delay. Run it regularly; require e.g. 90% of runs to have positive OOS Sharpe before considering the strategy “robust.” | Tests sensitivity to small changes in inputs. |
| **P1** | **Multiple universes** — Backtest on: (1) large-cap only, (2) mid-cap, (3) sector-specific (e.g. tech). If the strategy only works on one slice, document it; if it works across slices, confidence is higher. | Reduces regime/universe bias. |
| **P2** | **Bootstrap confidence intervals** — You already have Sharpe 95% CI in metrics. Use them: if the lower bound is negative, the strategy may not be statistically significant. | Separates signal from noise. |

---

## 4. Infrastructure & Ops (Run It Like a Fund)

**Goal:** Reproducibility, data quality, and clear live vs research separation.

| Priority | What to do | Why it matters |
|----------|------------|----------------|
| **P1** | **Pin data and code** — For every backtest or walk-forward run, log: config hash, data range, and git commit (or artifact version). Save a snapshot of `backtest_config.yaml` and learned weights in `output/experiments/<run_id>/`. You already have experiment snapshots; make them the source of truth. | Reproducibility is non-negotiable for institutional use. |
| **P1** | **Data quality checks** — Before backtest, check: no missing bars on key dates, no obvious bad ticks (e.g. 50% single-day move), and alignment of signal dates with price dates. One script that validates inputs and fails fast. | Bad data causes phantom edges or false confidence. |
| **P2** | **Separate research vs production config** — Research: no circuit breaker, maybe higher costs for stress. Production/paper: circuit breaker on, realistic costs, and optional kill switch (e.g. max daily loss). | Prevents research assumptions from leaking into live. |
| **P2** | **Simple monitoring** — If you run live or paper: log daily P&L, position count, and drawdown. Alert when drawdown or position count exceeds thresholds. | Catches regime shifts and execution issues early. |

---

## 5. Prioritized Action Plan (Next 2–4 Weeks)

Do these in order. Each step should be validated with **walk-forward OOS** before moving on.

1. **Week 1 – Risk and sizing**
   - Enable **vol_scaled** position sizing and **stop_loss_pct: 0.02** in `backtest_config.yaml`.
   - Re-run backtest and walk-forward. Target: OOS Sharpe positive and max drawdown &lt; 15%.

2. **Week 2 – Alpha**
   - Run weight learning with **regression**, **excess return** target, and **--compare** (walk-forward). Try holding period 5, 10, 20 and pick best OOS.
   - Enable **cross_sectional** (top_longs: 5–10) and compare OOS to current long-only single-name approach.

3. **Week 3 – Validation**
   - Run **Monte Carlo** (500 runs). If &lt; 80% of runs have positive OOS Sharpe, tighten signal (e.g. higher min_signal_strength or fewer names) and re-test.
   - Document: “We use OOS walk-forward and Monte Carlo; we do not optimize on in-sample backtest.”

4. **Week 4 – Reproducibility and docs**
   - Ensure every `run_backtest.py` and `run_weight_learning.py` write to `output/experiments/<timestamp>/` with full config and weights.
   - Add a one-page “Strategy Summary” (universe, signal, holding period, risk limits, OOS Sharpe band) for internal or investor use.

---

## 6. What “Industry Level” Means in One Sentence

**Use out-of-sample walk-forward and Monte Carlo to choose and validate the strategy; size by risk (vol targeting), cap drawdowns (stops and circuit breaker), and run with realistic costs and clear separation between research and production.** Your codebase already has the building blocks; the gap is **discipline in how you use them** (OOS-first, risk-first) and a few config changes (vol sizing, stops, cross-sectional).

Start with the Week 1 actions; they will immediately improve risk-adjusted results and drawdown behavior without changing your alpha logic.
