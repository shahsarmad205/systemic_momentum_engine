# Trend Signal Engine — Modular Research Platform

## Overview

The system is a modular research platform that supports:

- **Dynamic signal weighting** — rule-based, learned (Ridge/GBR), or regime-specific weights
- **Regime-aware modeling** — Bull/Bear/Sideways/Crisis adjustments and regime-specific weight models
- **Sector-aware trading** — sector exposure limits and sector momentum/vol/sentiment adjustment
- **Realistic cost modeling** — slippage, commission, and transaction cost sensitivity analysis
- **Robust walk-forward validation** — sequential train/test windows, optional per-window weight training, OOS metrics (Sharpe, drawdown, directional accuracy, IC)
- **Flexible ticker simulation** — any ticker list via config or CLI; Yahoo/Alpaca/Finnhub with local cache
- **Real-time preparation** — async data ingestion hooks, signal streaming interface, stateless fast prediction

## Fast Experimentation by Default

When no tickers are specified in config or CLI:

- **Backtest** (`run_backtest.py`): uses a **10-ticker development universe** (from `main.TICKERS` or fallback list, limited to 10).
- **Weight learning** (`run_weight_learning.py`): same 10-ticker limit for quick runs.

Override with:

- `backtest_config.yaml`: set `tickers: [AAPL, MSFT, ...]`
- CLI: `python run_backtest.py --tickers AAPL MSFT GOOG ...`

## Real-Time Integration

The `realtime` package prepares for live trading:

1. **Data ingestion hooks** (`realtime.DataIngestionHooks`) — register callbacks for `on_bar` or `on_data_ready`; async feeders notify when new data is available.
2. **Signal streaming** (`realtime.SignalStream`) — subscribe with callbacks; `publish(ticker, signal_dict)` delivers to dashboard or a queue adapter (e.g. MQ/WebSocket).
3. **Stateless prediction** (`realtime.predict_latest_signal`) — given `(ticker, ohlcv_df, signal_engine)`, returns the latest signal row; no global state, fast for real-time updates.

See `realtime/` and `realtime/runner.py` for wiring data → predict → stream.

## Web UI (Signal Ranking Engine)

- **`webapp/`** — Flask + static SPA for ticker universe, cross-sectional controls, ranking table, daily positions, backtest trigger, equity chart.
- Run: `cd webapp && python app.py` → http://127.0.0.1:5000
- See `webapp/README.md`.
