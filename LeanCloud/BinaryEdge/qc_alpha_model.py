# ============================================================
# QuantConnect (Lean SDK) — ML Alpha Model
# ============================================================
# Synced with trend_signal_engine backtesting pipeline.
# Implements all Phase A-D improvements:
#   A1: Cross-sectional z-scores at inference
#   A2: Short model drives short selection (long-only mode here)
#   A3: Full 23-feature alignment with best_long_model.pkl (earnings_surprise removed)
#   C2: Regime-conditional routing for Bear/Crisis/HighVol
#   D1: Continuous regime score → linear gross exposure scaling
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
# LGBMRankerWrapper stub — must be defined at module level
# so that the pickled VotingRegressor (best_long_model.pkl)
# can deserialize its LGBMRankerWrapper sub-estimator.
# The stub delegates to the real LGBMRanker if lightgbm is
# available; otherwise returns zeros gracefully.
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
    """Fallback neutral model when pickle fails."""
    def predict(self, X):
        return np.zeros(len(X), dtype=float)
    def predict_proba(self, X):
        return np.array([[0.5, 0.5]] * len(X))


def LoadProductionModel(algorithm, model_path="best_long_model.pkl"):
    """
    Load model artifact from local project file or Object Store.
    Injects LGBMRankerWrapper into __main__ before unpickling so the
    VotingRegressor sub-estimator deserializes correctly.
    Returns (estimator, feature_columns_list).
    """
    import joblib
    import sys
    import os

    # Patch __main__ with LGBMRankerWrapper so pickle can resolve it
    _main = sys.modules.get("__main__")
    if _main is not None and not hasattr(_main, "LGBMRankerWrapper"):
        setattr(_main, "LGBMRankerWrapper", LGBMRankerWrapper)

    obj = None
    source = "UNKNOWN"
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_path = os.path.join(base_dir, model_path)

    try:
        algorithm.Log(f"LoadProductionModel: looking for {full_path}")
        if os.path.isfile(full_path):
            try:
                obj = joblib.load(full_path)
                source = "LOCAL_FILE"
            except Exception:
                with open(full_path, "rb") as fh:
                    obj = pickle.load(fh)
                source = "LOCAL_PICKLE"
        else:
            algorithm.Log(f"Local file not found: {full_path}. Trying Object Store.")
            if algorithm.ObjectStore.ContainsKey(model_path):
                data = bytes(algorithm.ObjectStore.ReadBytes(model_path))
                try:
                    obj = joblib.load(io.BytesIO(data))
                except Exception:
                    obj = pickle.loads(data)
                source = "OBJECT_STORE"
            else:
                algorithm.Error(f"Model '{model_path}' not found anywhere.")
                return (SimpleModel(), None)
    except Exception as exc:
        algorithm.Error(f"LoadProductionModel failed for {model_path}: {exc}")
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
    def __init__(self, algorithm, model_data, short_model=None, spy_symbol=None,
                 bear_model_data=None, highvol_model_data=None):
        # Unpack (estimator, feature_columns) tuples
        if isinstance(model_data, tuple):
            self.long_model, _feat_cols = model_data
        else:
            self.long_model, _feat_cols = model_data, None

        if isinstance(short_model, tuple):
            self.short_model, _ = short_model
        else:
            self.short_model = short_model

        # C2: regime-specific models (Bear / HighVol/Crisis only)
        if isinstance(bear_model_data, tuple):
            self.bear_model, _ = bear_model_data
        else:
            self.bear_model = bear_model_data

        if isinstance(highvol_model_data, tuple):
            self.highvol_model, _ = highvol_model_data
        else:
            self.highvol_model = highvol_model_data

        self.spy_symbol = spy_symbol or algorithm.AddEquity("SPY", Resolution.Daily).Symbol
        self.symbol_data = {}
        self.active_signals = {}
        self.last_refresh = {}
        self.long_only = True

        # Model feature columns (from artifact metadata)
        self.features = _feat_cols if _feat_cols else [
            "vol_expansion", "f_trend", "short_term_reversal", "ret_20d",
            "ret_1d", "vol_ratio_5_20", "nearness_52w_low", "sector_relative_60d",
            "nearness_52w_high", "cs_momentum_percentile",
            "dist_from_52w_high", "low_vol_score", "momentum_acceleration",
            "ret_5d", "capm_residual_vol", "down_up_vol_ratio", "rsi_14",
            "capm_alpha", "ret_10d", "sector_relative_20d", "rsi_overbought",
            "rolling_vol_20", "quality_score",
        ]

        # Strategy params
        self.top_n = 10
        self.hold_n = 15
        self.min_score = 0.20
        self.min_dollar_volume = 1e8
        self.flip_buffer = 0.05
        self.refresh_days = 10

        # D1: Continuous regime score state
        self.current_regime = "Bull"        # label: Bull/Bear/Crisis/Sideways
        self.regime_score = 0.0             # 0=Bull, 1=Crisis
        self.bull_gross_cap = 1.0
        self.crisis_gross_cap = 0.25

        # SPY warm-up for regime detection
        self.spy_close_window = RollingWindow[float](210)  # 200d MA + buffer
        self.vix_window = RollingWindow[float](20)         # for VIX proxy

        spy_hist = algorithm.History(self.spy_symbol, 210, Resolution.Daily)
        if spy_hist is not None and len(spy_hist) > 0:
            for _, row in spy_hist.iterrows():
                p = float(row.get('close', row.get('Close', 0)))
                if p > 0:
                    self.spy_close_window.Add(p)

        algorithm.Log(f"TrendSignalAlphaModel initialized | features={len(self.features)} | "
                      f"bear_model={'yes' if self.bear_model else 'no'} | "
                      f"highvol_model={'yes' if self.highvol_model else 'no'}")

    # ----------------------------------------------------------
    def _sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-float(x)))

    def _update_regime(self, algorithm):
        """D1: Update regime label + continuous regime score from SPY+VIX proxy."""
        if self.spy_close_window.Count < 205:
            return

        closes = [self.spy_close_window[i] for i in range(self.spy_close_window.Count)]
        closes_arr = np.array(closes[::-1], dtype=float)  # oldest first

        sma200 = float(np.mean(closes_arr[-200:]))
        sma50  = float(np.mean(closes_arr[-50:]))
        close  = closes_arr[-1]

        # SPY realized vol (20d) as VIX proxy (annualized %)
        if len(closes_arr) >= 21:
            rets = np.diff(closes_arr[-21:]) / closes_arr[-21:-1]
            vix_proxy = float(np.std(rets) * np.sqrt(252) * 100)
        else:
            vix_proxy = 15.0

        # Hard label
        if vix_proxy >= 30.0:
            self.current_regime = "Crisis"
        elif close > sma200 and sma50 > sma200:
            self.current_regime = "Bull"
        elif close < sma200 and sma50 < sma200:
            self.current_regime = "Bear"
        else:
            self.current_regime = "Sideways"

        # D1: continuous score [0=Bull, 1=Crisis]
        vix_score = self._sigmoid((vix_proxy - 25.0) / 4.0)
        sma_gap = (sma200 - close) / max(sma200, 1e-6)
        sma_score = self._sigmoid(sma_gap / 0.025)
        self.regime_score = float(np.clip(max(vix_score, sma_score), 0.0, 1.0))

    def _gross_cap(self):
        """D1: Linear interpolation of gross cap using continuous regime score."""
        return self.bull_gross_cap + self.regime_score * (self.crisis_gross_cap - self.bull_gross_cap)

    def _select_model(self):
        """C2: Return the best model for the current regime."""
        if self.current_regime == "Bear" and self.bear_model is not None:
            return self.bear_model
        if self.current_regime in ("Crisis", "HighVol") and self.highvol_model is not None:
            return self.highvol_model
        return self.long_model

    # ----------------------------------------------------------
    def OnSecuritiesChanged(self, algorithm, changes):
        for added in changes.AddedSecurities:
            if added.Symbol not in self.symbol_data:
                self.symbol_data[added.Symbol] = SymbolData(algorithm, added.Symbol, self.spy_close_window)
        for removed in changes.RemovedSecurities:
            self.symbol_data.pop(removed.Symbol, None)

    def Update(self, algorithm, data):
        # Update SPY window
        spy_bar = data.Bars.get(self.spy_symbol)
        if spy_bar:
            self.spy_close_window.Add(spy_bar.Close)

        # Update regime
        self._update_regime(algorithm)
        gross_cap = self._gross_cap()
        effective_top_n = max(1, int(self.top_n * gross_cap))

        # Gather features
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
            feats = sd.GetFullFeatures()
            if feats:
                raw_features[symbol] = feats

        algorithm.Log(f"UPDATE | regime={self.current_regime} score={self.regime_score:.2f} "
                      f"cap={gross_cap:.2f} top_n={effective_top_n} symbols={len(raw_features)}")

        if not raw_features:
            return []

        # Build panel DataFrame
        df = pd.DataFrame.from_dict(raw_features, orient='index')

        # A1: Cross-sectional z-scores (matches training)
        cs_z_cols = ['ret_5d', 'ret_10d', 'rolling_vol_20']
        for col in cs_z_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std(ddof=0)
                df[col] = (df[col] - mean) / std if std > 1e-9 else 0.0

        # Cross-sectional momentum percentile
        if "momentum_6m" in df.columns:
            df["cs_momentum_percentile"] = df["momentum_6m"].rank(pct=True).fillna(0.5)
        else:
            df["cs_momentum_percentile"] = 0.5

        # sector_relative_20d/60d: approximate as (stock ret - universe mean ret)
        for col in ["ret_20d", "ret_60d"]:
            if col in df.columns:
                universe_mean = df[col].mean()
                suffix = "20d" if "20" in col else "60d"
                df[f"sector_relative_{suffix}"] = df[col] - universe_mean

        # Select model for current regime (C2)
        active_model = self._select_model()

        # Inference
        long_raw_scores = {}
        for symbol, row in df.iterrows():
            ordered = np.array([float(row.get(f, 0.0)) for f in self.features], dtype=float)
            ordered = np.where(np.isfinite(ordered), ordered, 0.0)
            ordered = np.clip(ordered, -10.0, 10.0)
            try:
                if hasattr(active_model, "predict_proba"):
                    proba = active_model.predict_proba([ordered])
                    score = (2.0 * float(proba[0][1])) - 1.0
                else:
                    score = float(active_model.predict([ordered])[0])
                long_raw_scores[symbol] = score
            except Exception as exc:
                algorithm.Log(f"Inference error {symbol.Value}: {exc}")

        if not long_raw_scores:
            return []

        # Cross-sectional rank normalization
        l_series = pd.Series(long_raw_scores).rank(pct=True, method='average').fillna(0.5)
        l_norm = (l_series * 2.0) - 1.0

        top_candidates = sorted(l_norm.items(), key=lambda x: x[1], reverse=True)
        symbols_by_rank = [s for s, _ in top_candidates]

        insights = []

        # Update existing holdings
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
                    insights.append(Insight.Price(symbol, timedelta(days=self.refresh_days * 2), direction, score))
                    self.last_refresh[symbol] = algorithm.Time
            except ValueError:
                insights.append(Insight.Price(symbol, timedelta(days=5), InsightDirection.Flat, 0.0))
                to_remove.append(symbol)
                self.last_refresh.pop(symbol, None)

        for s in to_remove:
            self.active_signals.pop(s, None)

        # New entries (capped by D1 gross cap)
        for symbol, score in top_candidates:
            rank = symbols_by_rank.index(symbol) + 1
            if rank <= effective_top_n and symbol not in self.active_signals and score >= self.min_score:
                insight = Insight.Price(symbol, timedelta(days=self.refresh_days * 2),
                                        InsightDirection.Up, score)
                insights.append(insight)
                self.active_signals[symbol] = InsightDirection.Up
                self.last_refresh[symbol] = algorithm.Time

        if insights:
            algorithm.Log(f"Alpha: Active={len(self.active_signals)} Insights={len(insights)} "
                          f"Model={type(active_model).__name__}")

        return insights


# ============================================================
# Per-Symbol Feature Builder
# ============================================================

class SymbolData:
    def __init__(self, algorithm, symbol, spy_close_window):
        self.symbol = symbol
        self.spy_close_window = spy_close_window  # shared reference

        # Indicators
        self.rsi = RelativeStrengthIndex(14)
        self.std20 = StandardDeviation(20)
        self.std5 = StandardDeviation(5)
        self.sma50 = SimpleMovingAverage(50)
        self.sma200 = SimpleMovingAverage(200)

        # Windows
        self.close_window = RollingWindow[float](253)
        self.returns_window = RollingWindow[float](2)
        self.vol_history = RollingWindow[float](252)

        # Warm-up from history
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
        return (self.close_window.Count >= 253 and
                self.sma50.IsReady and self.rsi.IsReady and
                self.spy_close_window.Count >= 50)

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

    def GetFullFeatures(self):
        """Build the 24-feature vector matching best_long_model.pkl training."""
        if not self.IsReady:
            return None

        close = self.close_window[0]
        ret_1d  = (close / self.close_window[1])  - 1.0 if self.close_window.Count > 1  else 0.0
        ret_5d  = (close / self.close_window[5])  - 1.0 if self.close_window.Count > 5  else 0.0
        ret_10d = (close / self.close_window[10]) - 1.0 if self.close_window.Count > 10 else 0.0
        ret_20d = (close / self.close_window[20]) - 1.0 if self.close_window.Count > 20 else 0.0
        ret_60d = (close / self.close_window[60]) - 1.0 if self.close_window.Count > 60 else 0.0
        mom3m   = (close / self.close_window[63]) - 1.0 if self.close_window.Count > 63 else 0.0
        mom6m   = (close / self.close_window[126])- 1.0 if self.close_window.Count > 126 else 0.0

        # Trend composite
        ma50  = self.sma50.Current.Value
        ma200 = self.sma200.Current.Value
        ma_cross = 1.0 if ma50 > ma200 else -1.0
        f_trend = 0.30 * mom3m * 10.0 + 0.25 * mom6m * 10.0 + 0.25 * ma_cross + 0.20 * ret_1d * 10.0

        # Volatility
        vol_current = self.std20.Current.Value * np.sqrt(252)
        vol_5d = self.std5.Current.Value * np.sqrt(252)
        vol_ratio_5_20 = vol_5d / vol_current if vol_current > 1e-9 else 1.0
        vol_expansion = vol_ratio_5_20

        # Vol percentile for low_vol_score
        vol_perc = 0.5
        if self.vol_history.Count > 20:
            hist = [self.vol_history[i] for i in range(self.vol_history.Count)]
            vol_perc = sum(1 for v in hist if v < vol_current) / float(len(hist))

        # RSI
        rsi_val = self.rsi.Current.Value
        rsi_overbought = 1.0 if rsi_val > 70 else 0.0

        # 52-week high/low
        win52 = min(252, self.close_window.Count - 1)
        prices_52 = [self.close_window[i] for i in range(win52)]
        high_52w = max(prices_52) if prices_52 else close
        low_52w  = min(prices_52) if prices_52 else close
        dist_from_52w_high = (close / high_52w) - 1.0
        nearness_52w_high  = float(np.clip(close / max(high_52w, 1e-6), 0.0, 1.0))
        dist_low = (close - low_52w) / max(low_52w, 1e-6)
        nearness_52w_low = 1.0 / (1.0 + max(0.0, dist_low))

        # CAPM alpha & residual vol (60d rolling regression vs SPY)
        capm_alpha = 0.0
        capm_res_vol = 0.05
        try:
            win_c = min(60, self.close_window.Count - 1, self.spy_close_window.Count - 1)
            if win_c >= 20:
                stock_rets = np.array([(self.close_window[i] / self.close_window[i+1]) - 1.0
                                       for i in range(win_c)], dtype=float)
                spy_rets   = np.array([(self.spy_close_window[i] / self.spy_close_window[i+1]) - 1.0
                                       for i in range(win_c)], dtype=float)
                var_spy = float(np.var(spy_rets))
                beta = float(np.cov(stock_rets, spy_rets)[0][1] / var_spy) if var_spy > 1e-9 else 1.0
                residuals = stock_rets - beta * spy_rets
                capm_res_vol = float(np.std(residuals))
                capm_alpha = float(np.mean(residuals) * 252)  # annualized
        except Exception:
            pass

        # All-weather features
        short_term_reversal = float(np.clip(-ret_5d, -0.5, 0.5))
        low_vol_score = 1.0 - vol_perc
        win_q = min(60, self.close_window.Count - 1)
        if win_q >= 20:
            rets_q = np.array([(self.close_window[i] / self.close_window[i+1]) - 1.0
                               for i in range(win_q)], dtype=float)
            std_q = float(np.std(rets_q))
            quality_score = float(np.clip(np.mean(rets_q) / std_q * np.sqrt(252)
                                          if std_q > 1e-9 else 0.0, -5.0, 5.0))
        else:
            quality_score = 0.0

        # down_up_vol_ratio: proxy using return sign (rolling 20-day)
        # (proper version needs per-day volume — approximate as 1.0)
        down_up_vol_ratio = 1.0

        # momentum_acceleration
        momentum_acceleration = ret_5d - ret_10d

        return {
            # Core momentum & trend
            "f_trend":                f_trend,
            "ret_1d":                 ret_1d,
            "ret_5d":                 ret_5d,
            "ret_10d":                ret_10d,
            "ret_20d":                ret_20d,
            "momentum_6m":            mom6m,           # used for cs_momentum_percentile
            "ret_60d":                ret_60d,          # used for sector_relative_60d
            # Volatility
            "rolling_vol_20":         vol_current,
            "vol_ratio_5_20":         vol_ratio_5_20,
            "vol_expansion":          vol_expansion,
            # RSI & mean-reversion
            "rsi_14":                 rsi_val,
            "rsi_overbought":         rsi_overbought,
            "momentum_acceleration":  momentum_acceleration,
            "down_up_vol_ratio":      down_up_vol_ratio,
            # 52-week proximity
            "dist_from_52w_high":     dist_from_52w_high,
            "nearness_52w_high":      nearness_52w_high,
            "nearness_52w_low":       nearness_52w_low,
            # CAPM
            "capm_residual_vol":      capm_res_vol,
            "capm_alpha":             capm_alpha,
            # All-weather
            "short_term_reversal":    short_term_reversal,
            "low_vol_score":          low_vol_score,
            "quality_score":          quality_score,
            # Sector relative (computed in Update from panel)
            "sector_relative_20d":    0.0,  # overwritten in Update()
            "sector_relative_60d":    0.0,  # overwritten in Update()
            # cs_momentum_percentile overwritten in Update()
            "cs_momentum_percentile": 0.5,
        }
