# Phase 2 Ridge ablation

| Step | Group | n active | WF IC | Train IC | Net Sharpe | Win% | CAGR | vs prev |
|------|-------|----------|-------|----------|------------|------|------|---------|
| 0 | Phase1 baseline | 9 | 0.0302 | 0.0224 | 1.041 | 46.2% | 9.10% | baseline |
| 1 | Phase1 baseline → + Group A (RSI+BB) | 11 | 0.0323 | 0.0230 | 1.491 | 46.8% | 14.60% | KEEP |
| 2 | Phase1 baseline → + Group A (RSI+BB) → + | 13 | 0.0364 | 0.0314 | 1.260 | 46.9% | 10.52% | REGRESS |
| 3 | Phase1 baseline → + Group A (RSI+BB) → + | 15 | 0.0364 | 0.0313 | 1.095 | 46.8% | 8.99% | REGRESS |
| 4 | Phase1 baseline → + Group A (RSI+BB) → + | 17 | 0.0336 | 0.0330 | 1.466 | 47.0% | 12.51% | KEEP |
| 5 | Phase1 baseline → + Group A (RSI+BB) → + | 20 | 0.0440 | 0.0284 | 1.208 | 46.7% | 9.37% | REGRESS |
| 6 | Phase1 baseline → + Group A (RSI+BB) → + | 25 | 0.0589 | 0.0411 | 1.577 | 47.1% | 13.99% | KEEP |

**Best Net Sharpe step:** `6` (sharpe=1.577)

Re-train with that step: `export TSE_ABLATION_STEP=6` then `python run_weight_learning.py --model ridge ...`