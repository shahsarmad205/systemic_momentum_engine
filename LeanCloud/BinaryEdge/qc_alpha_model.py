# ============================================================
# QuantConnect (Lean SDK) — ML Alpha Model
# ============================================================
# Aligned with local trend_signal_engine backtest.
#
# Feature computation matches the 19-feature baseline that
# produces local Net Sharpe 1.119, CAGR 11.55%.
#
# Key alignment decisions:
#   - No regime model routing (proven harmful locally)
#   - quality_score = rolling 60d return/vol ratio (price-only, matches local)
#   - down_up_vol_ratio = std(negative_rets) / std(positive_rets) (proper computation)
#   - sector_relative = stock_ret - universe_mean (best proxy without sector data)
#   - top_n=12, hold_n=16, refresh_days=15 (match local config)
# ============================================================

try:
    from AlgorithmImports import *
except ImportError:
    class AlphaModel: pass
    class RollingWindow(list):
        def __init__(self, type_or_size, size=None):
            self.size = size if size is not None else (type_or_size if isinstance(type_or_size, int) else 0)
        def __getitem__(self, key): return list.__getitem__(self, key)
        def __class_getitem__(cls, item): return cls
        @property
        def IsReady(self): return len(self) >= self.size
        @property
        def Count(self): return len(self)
        def Add(self, item):
            self.insert(0, item)
            if len(self) > self.size: self.pop()
    class Resolution: Daily = 'Daily'

import pandas as pd
import numpy as np
import pickle
import io
from datetime import timedelta


# ============================================================
# LGBMRankerWrapper stub — must be at module level so that
# the pickled VotingRegressor can deserialize correctly.
# Returns zeros if LGBMRanker is not available (graceful fallback).
# ============================================================
try:
    from sklearn.base import BaseEstimator, RegressorMixin as _RegressorMixin
    import lightgbm as _lgb

    class LGBMRankerWrapper(_RegressorMixin, BaseEstimator):
        def __init__(self, n_estimators=100, max_depth=4, learning_rate=0.05,
                     num_leaves=15, min_child_samples=20, subsample=0.8,
                     colsample_bytree=0.8):
            self.n_estimators = n_estimators
            self.max_depth = max_depth
            self.learning_rate = learning_rate
            self.num_leaves = num_leaves
            self.min_child_samples = min_child_samples
            self.subsample = subsample
            self.colsample_bytree = colsample_bytree
            self._model = None
            self._preloaded_date_groups = None

        def set_date_context(self, date_groups):
            self._preloaded_date_groups = date_groups
            return self

        def fit(self, X, y, **kw):
            return self  # inference-only in QC

        def predict(self, X):
            if self._model is None:
                return np.zeros(len(X), dtype=float)
            return self._model.predict(X).astype(float)

except ImportError:
    try:
        from sklearn.base import BaseEstimator, RegressorMixin as _RegressorMixin

        class LGBMRankerWrapper(_RegressorMixin, BaseEstimator):
            def __init__(self, **kw): pass
            def fit(self, X, y, **kw): return self
            def predict(self, X): return np.zeros(len(X), dtype=float)

    except ImportError:
        class LGBMRankerWrapper:
            def __init__(self, **kw): pass
            def fit(self, X, y, **kw): return self
            def predict(self, X): return np.zeros(len(X), dtype=float)


# ============================================================
# Model Loader
# ============================================================

class SimpleModel:
    """
    Fallback model when pickle fails to load.
    Returns zeros — caller must detect isinstance(model, SimpleModel)
    and use a feature-based fallback score instead.
    """
    def predict(self, X):
        return np.zeros(len(X), dtype=float)
    def predict_proba(self, X):
        return np.array([[0.5, 0.5]] * len(X))


def LoadProductionModel(algorithm, model_path="best_long_model.pkl"):
    """
    Load model artifact. Load order:
      1. model_payload.py (base64+zlib encoded .py file — QC syncs these)
      2. Local file path (works when running locally or in QC with file in project)
      3. Object Store (uploaded via Research notebook)
    Returns (estimator, feature_columns_list).
    """
    import joblib
    import sys
    import os
    import base64
    import zlib

    # Patch __main__ so pickle can resolve LGBMRankerWrapper
    _main = sys.modules.get("__main__")
    if _main is not None and not hasattr(_main, "LGBMRankerWrapper"):
        setattr(_main, "LGBMRankerWrapper", LGBMRankerWrapper)

    obj = None
    source = "UNKNOWN"

    # --- 1. model_payload.py (primary QC path: .py files ARE synced, .pkl are not) ---
    # model_payload.py joins 12 model_chunk_XX.py files (each <64KB, QC limit)
    try:
        from model_payload import load_model_bytes
        raw_bytes = load_model_bytes()
        try:
            obj = joblib.load(io.BytesIO(raw_bytes))
        except Exception:
            obj = pickle.loads(raw_bytes)
        source = "MODEL_PAYLOAD_PY"
        algorithm.Log(f"LoadProductionModel: loaded from model_payload.py ({len(raw_bytes):,} bytes decompressed)")
    except ImportError:
        algorithm.Log("model_payload.py not found — trying local file and Object Store.")
    except Exception as exc:
        algorithm.Log(f"model_payload.py decode failed: {exc} — falling back.")
        obj = None

    # --- 2. Local file path ---
    if obj is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, model_path)
        if os.path.isfile(full_path):
            try:
                obj = joblib.load(full_path)
                source = "LOCAL_FILE"
                algorithm.Log(f"LoadProductionModel: loaded from local file {full_path}")
            except Exception as exc:
                algorithm.Log(f"Local file load failed: {exc}")

    # --- 3. Object Store ---
    if obj is None:
        try:
            if algorithm.ObjectStore.ContainsKey(model_path):
                data = bytes(algorithm.ObjectStore.ReadBytes(model_path))
                try:
                    obj = joblib.load(io.BytesIO(data))
                except Exception:
                    obj = pickle.loads(data)
                source = "OBJECT_STORE"
                algorithm.Log(f"LoadProductionModel: loaded from Object Store ({len(data):,} bytes)")
        except Exception as exc:
            algorithm.Log(f"Object Store load failed: {exc}")

    if obj is None:
        algorithm.Error(
            "CRITICAL: best_long_model.pkl not found via model_payload.py, local file, or Object Store. "
            "Run encode_model.py locally then commit model_payload.py to sync to QC. "
            "Using f_trend momentum fallback."
        )
        return (SimpleModel(), None)

    if isinstance(obj, dict) and "estimator" in obj:
        est = obj["estimator"]
        feat_cols = obj.get("feature_columns")
        algorithm.Log(f"Model loaded ({source}): {type(est).__name__} | features={len(feat_cols) if feat_cols else 'unknown'}")
        return (est, feat_cols)
    else:
        algorithm.Log(f"Model loaded ({source}): {type(obj).__name__} (no metadata)")
        return (obj, None)


# ============================================================
# Alpha Model
# ============================================================

class TrendSignalAlphaModel(AlphaModel):
    def __init__(self, algorithm, model_data, spy_symbol=None):
        # Unpack (estimator, feature_columns) tuple from LoadProductionModel
        if isinstance(model_data, tuple):
            self.long_model, _feat_cols = model_data
        else:
            self.long_model, _feat_cols = model_data, None

        self.spy_symbol = spy_symbol or algorithm.AddEquity("SPY", Resolution.Daily).Symbol
        self.symbol_data = {}
        self.active_signals = {}
        self.last_refresh = {}

        # Feature columns — prefer pkl metadata, fall back to the 19-feature baseline
        # that produced local Net Sharpe 1.119 (matches backtest_config.yaml feature_subset)
        self.features = _feat_cols if _feat_cols else [
            "f_trend", "ret_5d", "ret_10d", "ret_20d", "rolling_vol_20",
            "cs_momentum_percentile", "rsi_14", "ret_1d", "vol_ratio_5_20",
            "capm_residual_vol", "short_term_reversal", "nearness_52w_low",
            "low_vol_score", "quality_score", "nearness_52w_high", "capm_alpha",
            "momentum_acceleration", "sector_relative_20d", "sector_relative_60d",
        ]

        # Strategy params — aligned with local backtest_config.yaml
        self.top_n = 12          # top_longs: 12
        self.hold_n = 16         # hold positions ranked up to 16 (buffer)
        self.min_score = 0.10    # min cross-sectional percentile rank (after transform)
        self.min_dollar_volume = 1e8
        self.refresh_days = 15   # rebalance_every_trading_days: 15

        # Regime state (for logging only — NOT used for model routing)
        self.current_regime = "Bull"
        self.spy_close_window = RollingWindow[float](210)

        # Warm up SPY history
        spy_hist = algorithm.History(self.spy_symbol, 210, Resolution.Daily)
        if spy_hist is not None and len(spy_hist) > 0:
            for _, row in spy_hist.iterrows():
                p = float(row.get('close', row.get('Close', 0)))
                if p > 0:
                    self.spy_close_window.Add(p)

        # Detect fallback mode — pkl was not found/loaded
        self._model_is_fallback = isinstance(self.long_model, SimpleModel)
        if self._model_is_fallback:
            algorithm.Error(
                "CRITICAL: best_long_model.pkl not found in project files or Object Store. "
                "Upload via: ObjectStore.SaveBytes('best_long_model.pkl', open('best_long_model.pkl','rb').read()) "
                "in a Research notebook. Using f_trend momentum fallback until pkl is loaded."
            )
        algorithm.Log(
            f"TrendSignalAlphaModel initialized | "
            f"features={len(self.features)} | model={type(self.long_model).__name__} | "
            f"fallback_mode={self._model_is_fallback} | "
            f"top_n={self.top_n} | refresh_days={self.refresh_days}"
        )

    # ----------------------------------------------------------
    def _update_regime_label(self):
        """Update regime label for logging. NOT used for model routing."""
        if self.spy_close_window.Count < 205:
            return
        closes = np.array([self.spy_close_window[i] for i in range(self.spy_close_window.Count)][::-1])
        sma200 = float(np.mean(closes[-200:]))
        sma50 = float(np.mean(closes[-50:]))
        close = closes[-1]
        if len(closes) >= 21:
            rets = np.diff(closes[-21:]) / closes[-21:-1]
            vix_proxy = float(np.std(rets) * np.sqrt(252) * 100)
        else:
            vix_proxy = 15.0
        if vix_proxy >= 30.0:
            self.current_regime = "Crisis"
        elif close > sma200 and sma50 > sma200:
            self.current_regime = "Bull"
        elif close < sma200 and sma50 < sma200:
            self.current_regime = "Bear"
        else:
            self.current_regime = "Sideways"

    # ----------------------------------------------------------
    def OnSecuritiesChanged(self, algorithm, changes):
        for added in changes.AddedSecurities:
            if added.Symbol not in self.symbol_data:
                self.symbol_data[added.Symbol] = SymbolData(
                    algorithm, added.Symbol, self.spy_close_window
                )
        for removed in changes.RemovedSecurities:
            self.symbol_data.pop(removed.Symbol, None)

    def Update(self, algorithm, data):
        # Update SPY window and regime label (logging only)
        spy_bar = data.Bars.get(self.spy_symbol)
        if spy_bar:
            self.spy_close_window.Add(spy_bar.Close)
        self._update_regime_label()

        # Gather per-stock features
        raw_features = {}
        for symbol, sd in self.symbol_data.items():
            if symbol == self.spy_symbol or not data.Bars.ContainsKey(symbol):
                continue
            sd.Update(data.Bars[symbol])
            if not sd.IsReady:
                continue
            if hasattr(algorithm.Securities[symbol], "DollarVolume"):
                if algorithm.Securities[symbol].DollarVolume < self.min_dollar_volume:
                    continue
            feats = sd.GetFeatures()
            if feats:
                raw_features[symbol] = feats

        if not raw_features:
            total_tracked = len(self.symbol_data)
            algorithm.Log(
                f"NO_FEATURES: 0/{total_tracked} symbols ready. "
                f"Check: close_window>=253, sma50.IsReady, rsi.IsReady, spy_window>=50."
            )
            return []

        # Build cross-sectional panel
        df = pd.DataFrame.from_dict(raw_features, orient='index')

        # Cross-sectional z-scores — matches local pipeline's cs z-scoring
        for col in ['ret_5d', 'ret_10d', 'ret_20d', 'rolling_vol_20', 'capm_residual_vol']:
            if col in df.columns:
                std = df[col].std(ddof=0)
                mean = df[col].mean()
                df[col] = ((df[col] - mean) / std).clip(-4, 4) if std > 1e-9 else 0.0

        # cs_momentum_percentile: rank of 6-month momentum
        if "momentum_6m" in df.columns:
            df["cs_momentum_percentile"] = df["momentum_6m"].rank(pct=True).fillna(0.5)
        else:
            df["cs_momentum_percentile"] = 0.5

        # sector_relative: stock return minus universe mean (best proxy without sector CSV)
        # Local uses sector median; this approximation loses some signal but preserves direction
        for col_raw, col_out in [("ret_20d", "sector_relative_20d"), ("ret_60d", "sector_relative_60d")]:
            if col_raw in df.columns:
                df[col_out] = df[col_raw] - df[col_raw].mean()
            else:
                df[col_out] = 0.0

        # Model inference — always use long_model (no regime routing)
        long_raw_scores = {}

        if self._model_is_fallback:
            # pkl not loaded: score = f_trend * 0.5 + cs_momentum_percentile - 0.5
            # This gives cross-sectionally meaningful signal so trades still happen.
            for symbol, row in df.iterrows():
                f_t = float(row.get("f_trend", 0.0))
                cs_mom = float(row.get("cs_momentum_percentile", 0.5))
                score = float(np.clip(0.5 * f_t + (cs_mom - 0.5), -2.0, 2.0))
                long_raw_scores[symbol] = score
        else:
            for symbol, row in df.iterrows():
                feat_vec = np.array([float(row.get(f, 0.0)) for f in self.features], dtype=float)
                feat_vec = np.where(np.isfinite(feat_vec), feat_vec, 0.0)
                feat_vec = np.clip(feat_vec, -10.0, 10.0)
                try:
                    if hasattr(self.long_model, "predict_proba"):
                        proba = self.long_model.predict_proba([feat_vec])
                        score = float(2.0 * proba[0][1] - 1.0)
                    else:
                        score = float(self.long_model.predict([feat_vec])[0])
                    long_raw_scores[symbol] = score
                except Exception as exc:
                    algorithm.Log(f"Inference error {symbol.Value}: {exc}")

        if not long_raw_scores:
            return []

        # Cross-sectional rank normalization → [-1, 1]
        l_series = pd.Series(long_raw_scores).rank(pct=True, method='average').fillna(0.5)
        l_norm = (l_series * 2.0) - 1.0

        top_candidates = sorted(l_norm.items(), key=lambda x: x[1], reverse=True)
        symbols_by_rank = [s for s, _ in top_candidates]

        insights = []

        # Close positions that fell out of hold_n or below min_score
        to_remove = []
        for symbol, direction in list(self.active_signals.items()):
            try:
                rank = symbols_by_rank.index(symbol) + 1
                score = float(l_norm.get(symbol, 0.0))
                if rank > self.hold_n or score < self.min_score:
                    insights.append(Insight.Price(symbol, timedelta(days=5), InsightDirection.Flat, score))
                    to_remove.append(symbol)
                    self.last_refresh.pop(symbol, None)
                    continue
                last_time = self.last_refresh.get(symbol)
                if last_time is None or (algorithm.Time - last_time).days >= self.refresh_days:
                    insights.append(Insight.Price(
                        symbol, timedelta(days=self.refresh_days * 2), direction, score
                    ))
                    self.last_refresh[symbol] = algorithm.Time
            except ValueError:
                insights.append(Insight.Price(symbol, timedelta(days=5), InsightDirection.Flat, 0.0))
                to_remove.append(symbol)
                self.last_refresh.pop(symbol, None)

        for s in to_remove:
            self.active_signals.pop(s, None)

        # Open new positions for top_n candidates
        for symbol, score in top_candidates:
            rank = symbols_by_rank.index(symbol) + 1
            if rank <= self.top_n and symbol not in self.active_signals and score >= self.min_score:
                insight = Insight.Price(
                    symbol, timedelta(days=self.refresh_days * 2),
                    InsightDirection.Up, score
                )
                insights.append(insight)
                self.active_signals[symbol] = InsightDirection.Up
                self.last_refresh[symbol] = algorithm.Time

        algorithm.Log(
            f"Alpha | regime={self.current_regime} | "
            f"active={len(self.active_signals)} | insights={len(insights)} | "
            f"candidates={len(long_raw_scores)}"
        )
        return insights


# ============================================================
# Per-Symbol Feature Builder
# ============================================================

class SymbolData:
    def __init__(self, algorithm, symbol, spy_close_window):
        self.symbol = symbol
        self.spy_close_window = spy_close_window

        # QC indicators
        self.rsi = RelativeStrengthIndex(14)
        self.std20 = StandardDeviation(20)
        self.std5 = StandardDeviation(5)
        self.sma50 = SimpleMovingAverage(50)
        self.sma200 = SimpleMovingAverage(200)

        # Rolling windows
        self.close_window = RollingWindow[float](253)
        self.returns_window = RollingWindow[float](2)
        self.vol_history = RollingWindow[float](252)

        # Warm up from history
        history = algorithm.History(symbol, 253, Resolution.Daily)
        if history is not None and len(history) > 0:
            for idx, row in history.iterrows():
                try:
                    close = float(row['close'])
                except Exception:
                    close = float(row[4]) if len(row) > 4 else 0.0
                close = max(0.01, min(20000.0, round(close, 4)))
                if not np.isfinite(close):
                    continue
                time = idx[1] if isinstance(idx, tuple) else idx
                self.close_window.Add(close)
                try:
                    self.rsi.Update(time, close)
                    self.sma50.Update(time, close)
                    self.sma200.Update(time, close)
                    self.returns_window.Add(close)
                    if self.returns_window.Count >= 2:
                        ret = max(-0.5, min(0.5, (self.returns_window[0] / self.returns_window[1]) - 1.0))
                        self.std20.Update(time, ret)
                        self.std5.Update(time, ret)
                        self.vol_history.Add(self.std20.Current.Value * np.sqrt(252))
                except Exception:
                    pass

    @property
    def IsReady(self):
        return (
            self.close_window.Count >= 253 and
            self.sma50.IsReady and
            self.rsi.IsReady and
            self.spy_close_window.Count >= 50
        )

    def Update(self, bar):
        if bar is None:
            return
        close = float(bar.Close) if hasattr(bar, 'Close') else float(bar['close'])
        close = max(0.01, min(20000.0, round(close, 4)))
        if not np.isfinite(close):
            return
        time = bar.EndTime if hasattr(bar, 'EndTime') else None
        if time is None:
            return
        try:
            self.close_window.Add(close)
            self.rsi.Update(time, close)
            self.sma50.Update(time, close)
            self.sma200.Update(time, close)
            self.returns_window.Add(close)
            if self.returns_window.Count >= 2:
                ret = max(-0.5, min(0.5, (self.returns_window[0] / self.returns_window[1]) - 1.0))
                self.std20.Update(time, ret)
                self.std5.Update(time, ret)
                self.vol_history.Add(self.std20.Current.Value * np.sqrt(252))
        except Exception:
            pass

    def GetFeatures(self):
        """
        Build the 19-feature vector matching local best_long_model.pkl training.
        All features are price-derived — no EDGAR data needed.
        Feature names match backtest_config.yaml feature_subset exactly.
        """
        if not self.IsReady:
            return None

        close = self.close_window[0]

        # --- Returns ---
        def safe_ret(n):
            return (close / self.close_window[n]) - 1.0 if self.close_window.Count > n else 0.0

        ret_1d  = safe_ret(1)
        ret_5d  = safe_ret(5)
        ret_10d = safe_ret(10)
        ret_20d = safe_ret(20)
        ret_60d = safe_ret(60)
        mom_6m  = safe_ret(126)  # used for cs_momentum_percentile, not directly a model feature

        # --- Trend composite (matches local f_trend computation) ---
        ma50  = self.sma50.Current.Value
        ma200 = self.sma200.Current.Value
        ma_cross = 1.0 if ma50 > ma200 else -1.0
        mom_3m = safe_ret(63)
        f_trend = (0.30 * mom_3m * 10.0 +
                   0.25 * mom_6m * 10.0 +
                   0.25 * ma_cross +
                   0.20 * ret_1d * 10.0)

        # --- Volatility ---
        vol_current = self.std20.Current.Value * np.sqrt(252)
        vol_5d = self.std5.Current.Value * np.sqrt(252)
        vol_ratio_5_20 = vol_5d / vol_current if vol_current > 1e-9 else 1.0

        # Vol percentile (for low_vol_score)
        vol_perc = 0.5
        if self.vol_history.Count > 20:
            hist = [self.vol_history[i] for i in range(self.vol_history.Count)]
            vol_perc = sum(1 for v in hist if v < vol_current) / float(len(hist))

        # --- RSI ---
        rsi_val = self.rsi.Current.Value
        rsi_overbought = 1.0 if rsi_val > 70 else 0.0

        # --- 52-week high/low ---
        win52 = min(252, self.close_window.Count - 1)
        prices_52 = [self.close_window[i] for i in range(win52)]
        high_52w = max(prices_52) if prices_52 else close
        low_52w  = min(prices_52) if prices_52 else close
        nearness_52w_high = float(np.clip(close / max(high_52w, 1e-6), 0.0, 1.0))
        dist_from_52w_high = float(np.clip((high_52w - close) / max(high_52w, 1e-6), 0.0, 1.0))
        dist_low = (close - low_52w) / max(low_52w, 1e-6)
        nearness_52w_low  = 1.0 / (1.0 + max(0.0, dist_low))

        # --- CAPM alpha and residual vol (60d rolling regression vs SPY) ---
        capm_alpha = 0.0
        capm_res_vol = 0.05
        try:
            win_c = min(60, self.close_window.Count - 1, self.spy_close_window.Count - 1)
            if win_c >= 20:
                stock_rets = np.array(
                    [(self.close_window[i] / self.close_window[i + 1]) - 1.0 for i in range(win_c)],
                    dtype=float
                )
                spy_rets = np.array(
                    [(self.spy_close_window[i] / self.spy_close_window[i + 1]) - 1.0 for i in range(win_c)],
                    dtype=float
                )
                var_spy = float(np.var(spy_rets))
                beta = float(np.cov(stock_rets, spy_rets)[0][1] / var_spy) if var_spy > 1e-9 else 1.0
                residuals = stock_rets - beta * spy_rets
                capm_res_vol = float(np.std(residuals))
                capm_alpha   = float(np.mean(residuals) * 252)  # annualized
        except Exception:
            pass

        # --- All-weather features ---
        short_term_reversal = float(np.clip(-ret_5d, -0.5, 0.5))
        low_vol_score = 1.0 - vol_perc

        # quality_score: rolling 60d return / vol — same computation as local feature_builder
        # (local uses rolling Sharpe-like metric, not EDGAR; verified consistent with 19-feature set)
        win_q = min(60, self.close_window.Count - 1)
        if win_q >= 20:
            rets_q = np.array(
                [(self.close_window[i] / self.close_window[i + 1]) - 1.0 for i in range(win_q)],
                dtype=float
            )
            std_q = float(np.std(rets_q))
            quality_score = float(np.clip(
                np.mean(rets_q) / std_q * np.sqrt(252) if std_q > 1e-9 else 0.0,
                -5.0, 5.0
            ))
        else:
            quality_score = 0.0

        # --- down_up_vol_ratio: std(negative daily returns) / std(positive daily returns) ---
        # Fixed: was hardcoded to 1.0 (zero cross-sectional signal). Now properly computed.
        win_d = min(60, self.close_window.Count - 1)
        down_up_vol_ratio = 1.0
        if win_d >= 20:
            rets_d = np.array(
                [(self.close_window[i] / self.close_window[i + 1]) - 1.0 for i in range(win_d)],
                dtype=float
            )
            up_rets   = rets_d[rets_d > 0]
            down_rets = rets_d[rets_d <= 0]
            vol_up   = float(np.std(up_rets))   if len(up_rets)   > 2 else 1e-9
            vol_down = float(np.std(down_rets)) if len(down_rets) > 2 else 1e-9
            down_up_vol_ratio = float(np.clip(vol_down / max(vol_up, 1e-9), 0.1, 10.0))

        # --- Momentum acceleration ---
        momentum_acceleration = ret_5d - ret_10d

        return {
            # Core momentum & trend
            "f_trend":                f_trend,
            "ret_1d":                 ret_1d,
            "ret_5d":                 ret_5d,
            "ret_10d":                ret_10d,
            "ret_20d":                ret_20d,
            "ret_60d":                ret_60d,       # used to compute sector_relative_60d in Update
            "momentum_6m":            mom_6m,        # used to compute cs_momentum_percentile in Update
            # Volatility
            "rolling_vol_20":         vol_current,
            "vol_ratio_5_20":         vol_ratio_5_20,
            # RSI
            "rsi_14":                 rsi_val,
            "rsi_overbought":         rsi_overbought,
            # 52-week proximity
            "nearness_52w_high":      nearness_52w_high,
            "nearness_52w_low":       nearness_52w_low,
            "dist_from_52w_high":     dist_from_52w_high,
            # CAPM
            "capm_residual_vol":      capm_res_vol,
            "capm_alpha":             capm_alpha,
            # All-weather
            "short_term_reversal":    short_term_reversal,
            "low_vol_score":          low_vol_score,
            "quality_score":          quality_score,
            # Volume ratio (now properly computed, not hardcoded)
            "down_up_vol_ratio":      down_up_vol_ratio,
            # Momentum
            "momentum_acceleration":  momentum_acceleration,
            # Sector relative and cs_momentum_percentile overwritten in Update()
            "sector_relative_20d":    0.0,
            "sector_relative_60d":    0.0,
            "cs_momentum_percentile": 0.5,
        }
