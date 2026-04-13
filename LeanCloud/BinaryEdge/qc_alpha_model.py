# ============================================================
# QuantConnect (Lean SDK) — ML Alpha Model
# ============================================================
# Aligned with local trend_signal_engine backtest (Net Sharpe 1.119):
#
#   Long model:  VotingRegressor  24 features — loaded from model_chunk_XX.py
#   Short model: VotingClassifier 28 features — loaded from short_model_payload.py
#
#   Short rules match local backtest_config.yaml exactly:
#     - Bear regime ONLY  (ml_short_allowed_regimes: ["Bear"])
#     - Max 5 shorts      (top_shorts: 5)
#     - 10-day hold       (ml_short_holding_period_days: 10)
#     - 8% stop loss      (ml_short_stop_loss_pct: 0.08)
#     - 5% snap profit    (ml_short_snap_profit_pct: 0.05)
#     - Min signal 0.7    (ml_short_min_signal_strength: 0.7)
#
#   Regime exposure (longs):
#     Bull/Sideways: 12  |  Bear: 0 new longs (hold existing)  |  Crisis: 4
#
#   No regime model routing. No Sharpe circuit breaker.
#   Stale Object Store model auto-detected (<20 features) and replaced from chunks.
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
# LGBMRankerWrapper stub — needed for VotingRegressor unpickling
# ============================================================
try:
    from sklearn.base import BaseEstimator, RegressorMixin as _RegressorMixin
    import lightgbm as _lgb

    class LGBMRankerWrapper(_RegressorMixin, BaseEstimator):
        def __init__(self, n_estimators=100, max_depth=4, learning_rate=0.05,
                     num_leaves=15, min_child_samples=20, subsample=0.8,
                     colsample_bytree=0.8):
            self.n_estimators   = n_estimators
            self.max_depth      = max_depth
            self.learning_rate  = learning_rate
            self.num_leaves     = num_leaves
            self.min_child_samples = min_child_samples
            self.subsample      = subsample
            self.colsample_bytree = colsample_bytree
            self._model = None

        def fit(self, X, y, **kw): return self
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
    """Fallback when pkl fails to load — triggers momentum fallback scoring."""
    def predict(self, X): return np.zeros(len(X), dtype=float)
    def predict_proba(self, X): return np.array([[0.5, 0.5]] * len(X))


def _load_obj_from_bytes(raw_bytes):
    import joblib
    try:
        return joblib.load(io.BytesIO(raw_bytes))
    except Exception:
        return pickle.loads(raw_bytes)


def _validate_obj(obj, min_features=20):
    """Reject stale models (old 15-feature RF uploaded manually to Object Store)."""
    if not isinstance(obj, dict) or "estimator" not in obj:
        return False
    return len(obj.get("feature_columns") or []) >= min_features


def LoadProductionModel(algorithm, model_path="best_long_model.pkl", min_features=20):
    """
    Load order:
      1. Object Store — validated (>= min_features). Stale models auto-deleted.
      2. model_chunk_XX.py / short_model_payload.py — git-synced, each <64KB.
         After loading, saves correct model to Object Store for future runs.
      3. Local file — for local testing.
    Returns (estimator, feature_columns_list).
    """
    import joblib, sys, os

    _main = sys.modules.get("__main__")
    if _main is not None and not hasattr(_main, "LGBMRankerWrapper"):
        setattr(_main, "LGBMRankerWrapper", LGBMRankerWrapper)

    obj    = None
    source = "UNKNOWN"

    # --- 1. Object Store (fast; validated) ---
    try:
        if algorithm.ObjectStore.ContainsKey(model_path):
            data      = bytes(algorithm.ObjectStore.ReadBytes(model_path))
            candidate = _load_obj_from_bytes(data)
            if _validate_obj(candidate, min_features):
                obj    = candidate
                source = "OBJECT_STORE"
                algorithm.Log(
                    f"LoadProductionModel [{model_path}]: Object Store OK "
                    f"({len(data):,} bytes, "
                    f"{len(obj.get('feature_columns', []))} features)"
                )
            else:
                n_feat = len(candidate.get("feature_columns", [])) if isinstance(candidate, dict) else 0
                algorithm.Log(
                    f"LoadProductionModel [{model_path}]: Object Store STALE "
                    f"({n_feat} features < {min_features}). Deleting and reloading from .py files."
                )
                try:
                    algorithm.ObjectStore.Delete(model_path)
                except Exception:
                    pass
    except Exception as exc:
        algorithm.Log(f"Object Store load failed [{model_path}]: {exc}")

    # --- 2. Synced .py payload files ---
    if obj is None:
        raw_bytes = None
        try:
            if model_path == "best_long_model.pkl":
                from model_payload import load_model_bytes
                raw_bytes = load_model_bytes()
            elif model_path == "best_short_model.pkl":
                from short_model_payload import load_short_model_bytes
                raw_bytes = load_short_model_bytes()
        except ImportError as e:
            algorithm.Log(f"Payload import failed [{model_path}]: {e}")
        except Exception as e:
            algorithm.Log(f"Payload load failed [{model_path}]: {e}")

        if raw_bytes is not None:
            try:
                candidate = _load_obj_from_bytes(raw_bytes)
                if _validate_obj(candidate, min_features):
                    obj    = candidate
                    source = "PAYLOAD_PY"
                    algorithm.Log(
                        f"LoadProductionModel [{model_path}]: loaded from .py payload "
                        f"({len(raw_bytes):,} bytes). Caching to Object Store..."
                    )
                    try:
                        algorithm.ObjectStore.SaveBytes(model_path, bytearray(raw_bytes))
                        algorithm.Log(f"Object Store cache saved [{model_path}].")
                    except Exception as e:
                        algorithm.Log(f"Object Store save failed (non-fatal): {e}")
                else:
                    algorithm.Log(f"Payload model [{model_path}] invalid — too few features.")
            except Exception as e:
                algorithm.Log(f"Payload decode failed [{model_path}]: {e}")

    # --- 3. Local file (local testing fallback) ---
    if obj is None:
        base_dir  = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, model_path)
        if os.path.isfile(full_path):
            try:
                candidate = joblib.load(full_path)
                if _validate_obj(candidate, min_features):
                    obj    = candidate
                    source = "LOCAL_FILE"
                    algorithm.Log(f"LoadProductionModel [{model_path}]: loaded from local file.")
            except Exception as e:
                algorithm.Log(f"Local file load failed [{model_path}]: {e}")

    if obj is None:
        algorithm.Error(
            f"CRITICAL: '{model_path}' not found or stale. "
            "Commit model_chunk_*.py + short_model_payload.py and push to QC. "
            "Using SimpleModel fallback."
        )
        return (SimpleModel(), None)

    est      = obj["estimator"]
    feat_cols = obj.get("feature_columns")
    algorithm.Log(
        f"Model loaded ({source}) [{model_path}]: {type(est).__name__} | "
        f"features={len(feat_cols) if feat_cols else 'unknown'}"
    )
    return (est, feat_cols)


# ============================================================
# Alpha Model
# ============================================================

class TrendSignalAlphaModel(AlphaModel):
    def __init__(self, algorithm, model_data, short_model_data=None, spy_symbol=None):
        # Long model
        if isinstance(model_data, tuple):
            self.long_model, _feat_cols = model_data
        else:
            self.long_model, _feat_cols = model_data, None

        # Short model (optional — Bear-only)
        if short_model_data is not None and isinstance(short_model_data, tuple):
            self.short_model, _short_feat_cols = short_model_data
        elif short_model_data is not None:
            self.short_model, _short_feat_cols = short_model_data, None
        else:
            self.short_model, _short_feat_cols = None, None

        self.spy_symbol  = spy_symbol or algorithm.AddEquity("SPY", Resolution.Daily).Symbol
        self.symbol_data = {}
        self.long_signals  = {}   # symbol -> score (float)
        self.short_signals = {}   # symbol -> entry_price (float) — for snap profit / stop tracking
        self.last_refresh  = {}

        # Long model feature columns (prefer pkl metadata → 24-feature set)
        self.features = _feat_cols if _feat_cols else [
            "f_trend", "ret_5d", "ret_10d", "ret_20d", "rolling_vol_20",
            "cs_momentum_percentile", "rsi_14", "ret_1d", "vol_ratio_5_20",
            "capm_residual_vol", "short_term_reversal", "nearness_52w_low",
            "low_vol_score", "quality_score", "nearness_52w_high", "capm_alpha",
            "momentum_acceleration", "sector_relative_20d", "sector_relative_60d",
            "dist_from_52w_high", "rsi_overbought", "down_up_vol_ratio",
            "vol_expansion", "earnings_surprise",
        ]
        # Short model feature columns (prefer pkl metadata → 28-feature set)
        self.short_features = _short_feat_cols if _short_feat_cols else [
            "sector_relative_20d", "vol_expansion", "nearness_52w_low",
            "f_score", "delta_roa", "rolling_vol_20", "delta_leverage", "roa",
            "sector_relative_60d", "vol_ratio_5_20", "accruals_ratio", "quality_score",
            "dist_from_52w_high", "ret_5d", "capm_residual_vol", "ret_1d", "capm_alpha",
            "cs_momentum_percentile", "ret_10d", "momentum_acceleration", "rsi_14",
            "f_trend", "ret_20d", "down_up_vol_ratio", "low_vol_score", "rsi_overbought",
            "short_term_reversal", "nearness_52w_high",
        ]

        # Fill values for features unavailable in QC
        self.DEFAULT_FILLS = {
            "low_vol_score": 0.5,
            "nearness_52w_high": 0.5,
            "nearness_52w_low": 0.5,
            "cs_momentum_percentile": 0.5,
            # Fundamental features — zeroed (no EDGAR in QC)
            "vol_expansion": 0.0,
            "earnings_surprise": 0.0,
            "f_score": 0.0,
            "delta_roa": 0.0,
            "delta_leverage": 0.0,
            "roa": 0.0,
            "accruals_ratio": 0.0,
        }

        # Long strategy params (match local backtest_config.yaml)
        self.top_n            = 12    # top_longs: 12
        self.hold_n           = 16    # hold buffer: 16
        self.min_score        = 0.10  # min CS percentile rank
        self.min_dollar_volume = 1e8
        self.refresh_days     = 21    # matches PCM rebalance

        # Short strategy params (match local backtest_config.yaml)
        self.top_shorts              = 5      # top_shorts: 5
        self.short_hold_n            = 7      # hold buffer for shorts
        self.short_min_signal        = 0.70   # ml_short_min_signal_strength: 0.7
        self.short_holding_days      = 10     # ml_short_holding_period_days: 10
        self.short_stop_pct          = 0.08   # ml_short_stop_loss_pct: 0.08
        self.short_snap_profit_pct   = 0.05   # ml_short_snap_profit_pct: 0.05
        self.short_min_liquidity_usd = 2e7    # ml_short_min_liquidity_usd: 20M

        # Regime state
        self.current_regime   = "Bull"
        self.spy_close_window = RollingWindow[float](210)
        self.last_rebalance_time = None
        # Track short entry prices for snap-profit / stop management
        self.short_entry_prices = {}  # symbol -> entry_price

        # Warm up SPY
        spy_hist = algorithm.History(self.spy_symbol, 210, Resolution.Daily)
        if spy_hist is not None and len(spy_hist) > 0:
            for _, row in spy_hist.iterrows():
                p = float(row.get('close', row.get('Close', 0)))
                if p > 0:
                    self.spy_close_window.Add(p)

        self._long_fallback  = isinstance(self.long_model, SimpleModel)
        self._short_enabled  = self.short_model is not None and not isinstance(self.short_model, SimpleModel)

        if self._long_fallback:
            algorithm.Error("CRITICAL: long model not loaded — using f_trend fallback.")
        algorithm.Log(
            f"TrendSignalAlphaModel initialized | "
            f"long_model={type(self.long_model).__name__} ({len(self.features)} feats) | "
            f"short_model={'enabled' if self._short_enabled else 'disabled'} | "
            f"top_n={self.top_n} | top_shorts={self.top_shorts} | "
            f"refresh_days={self.refresh_days}"
        )

    # ----------------------------------------------------------
    def _update_regime_label(self):
        """Regime label — used to gate shorts (Bear-only) and crisis scaling."""
        if self.spy_close_window.Count < 205:
            return
        closes = np.array(
            [self.spy_close_window[i] for i in range(self.spy_close_window.Count)][::-1]
        )
        sma200    = float(np.mean(closes[-200:]))
        sma50     = float(np.mean(closes[-50:]))
        close     = closes[-1]
        vix_proxy = 15.0
        if len(closes) >= 21:
            rets      = np.diff(closes[-21:]) / closes[-21:-1]
            vix_proxy = float(np.std(rets) * np.sqrt(252) * 100)
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
        new_symbols = [
            s.Symbol for s in changes.AddedSecurities
            if s.Symbol not in self.symbol_data
        ]
        if new_symbols:
            # Pre-build dict keyed by Symbol to avoid QC's PandasMapper KeyError
            # (.loc[symbol] raises a non-catchable C#-wrapped exception for missing symbols)
            hist_by_symbol = {}
            try:
                all_history = algorithm.History(new_symbols, 253, Resolution.Daily)
                if all_history is not None and len(all_history) > 0:
                    if algorithm.Time.day == 1:
                        algorithm.Log(
                            f"BATCH_HISTORY: {len(new_symbols)} symbols → "
                            f"{len(all_history)} rows"
                        )
                    try:
                        # groupby level=0 splits MultiIndex DataFrame into per-symbol slices
                        # .get() on a plain dict never raises — safe for missing symbols
                        hist_by_symbol = dict(tuple(all_history.groupby(level=0)))
                    except Exception as exc:
                        algorithm.Log(f"History groupby failed ({exc}) — individual fallback")
            except Exception as exc:
                algorithm.Log(f"Batch History failed ({exc}) — warming up from live bars")

            for symbol in new_symbols:
                sym_hist = hist_by_symbol.get(symbol, None)
                self.symbol_data[symbol] = SymbolData(
                    algorithm, symbol, self.spy_close_window, sym_hist
                )

        for s in changes.RemovedSecurities:
            self.symbol_data.pop(s.Symbol, None)

    # ----------------------------------------------------------
    def Update(self, algorithm, data):
        # Track SPY for regime
        if data.Bars.ContainsKey(self.spy_symbol):
            self.spy_close_window.Add(data.Bars[self.spy_symbol].Close)
        prev_regime = self.current_regime
        self._update_regime_label()
        regime_changed_to_bear = (
            self.current_regime == "Bear" and prev_regime != "Bear"
        )

        # Update indicators every day
        for symbol, sd in self.symbol_data.items():
            if data.Bars.ContainsKey(symbol):
                sd.Update(data.Bars[symbol])

        # --- Intraday short management (runs every day, not throttled) ---
        short_intraday_insights = self._manage_short_intraday(algorithm, data)

        # Throttle ML pipeline to rebalance frequency
        if (self.last_rebalance_time is not None and not regime_changed_to_bear
                and (algorithm.Time - self.last_rebalance_time).days < self.refresh_days):
            return short_intraday_insights
        self.last_rebalance_time = algorithm.Time

        # Gather features
        raw_features = {}
        for symbol, sd in self.symbol_data.items():
            if symbol == self.spy_symbol or not sd.IsReady:
                continue
            if not algorithm.Securities.ContainsKey(symbol):
                continue
            sec = algorithm.Securities[symbol]
            if sec.Price <= 0:
                continue
            if hasattr(sec, "DollarVolume") and sec.DollarVolume < self.min_dollar_volume:
                continue
            feats = sd.GetFeatures()
            if feats:
                raw_features[symbol] = feats

        if not raw_features:
            if getattr(self, "_no_feat_month", None) != algorithm.Time.month:
                algorithm.Log(
                    f"NO_FEATURES: 0/{len(self.symbol_data)} symbols ready."
                )
                self._no_feat_month = algorithm.Time.month
            return short_intraday_insights

        df = pd.DataFrame.from_dict(raw_features, orient='index')

        # CS z-score (matches training pipeline)
        for col in ['ret_5d', 'ret_10d', 'ret_20d', 'rolling_vol_20', 'capm_residual_vol']:
            if col in df.columns:
                std  = df[col].std(ddof=0)
                mean = df[col].mean()
                df[col] = ((df[col] - mean) / std).clip(-4, 4) if std > 1e-9 else 0.0

        if "momentum_6m" in df.columns:
            df["cs_momentum_percentile"] = df["momentum_6m"].rank(pct=True).fillna(0.5)
        else:
            df["cs_momentum_percentile"] = 0.5

        for raw_col, out_col in [("ret_20d", "sector_relative_20d"),
                                  ("ret_60d", "sector_relative_60d")]:
            if raw_col in df.columns:
                df[out_col] = df[raw_col] - df[raw_col].mean()
            else:
                df[out_col] = 0.0

        # Long model inference
        insights = list(short_intraday_insights)
        insights += self._update_longs(algorithm, df)

        # Short model inference (Bear regime only — matches local)
        if self._short_enabled and self.current_regime == "Bear":
            insights += self._update_shorts(algorithm, df)
        elif self.current_regime != "Bear":
            # Close all shorts when leaving Bear
            insights += self._close_all_shorts(algorithm)

        if getattr(self, "_log_month", None) != algorithm.Time.month:
            algorithm.Log(
                f"Alpha | regime={self.current_regime} | "
                f"longs={len(self.long_signals)} | shorts={len(self.short_signals)} | "
                f"insights={len(insights)} | candidates={len(raw_features)}"
            )
            self._log_month = algorithm.Time.month

        return insights

    # ----------------------------------------------------------
    def _update_longs(self, algorithm, df):
        """Long position management — always runs."""
        # Long model scoring
        raw_scores = {}
        if self._long_fallback:
            for symbol, row in df.iterrows():
                f_t  = float(row.get("f_trend", 0.0))
                cs_m = float(row.get("cs_momentum_percentile", 0.5))
                raw_scores[symbol] = float(np.clip(0.5 * f_t + (cs_m - 0.5), -2.0, 2.0))
        else:
            for symbol, row in df.iterrows():
                fv = np.array(
                    [float(row.get(f, self.DEFAULT_FILLS.get(f, 0.0))) for f in self.features],
                    dtype=float)
                fv = np.clip(np.where(np.isfinite(fv), fv, 0.0), -10.0, 10.0)
                try:
                    if hasattr(self.long_model, "predict_proba"):
                        score = float(2.0 * self.long_model.predict_proba([fv])[0][1] - 1.0)
                    else:
                        score = float(self.long_model.predict([fv])[0])
                    raw_scores[symbol] = score
                except Exception as exc:
                    pass

        if not raw_scores:
            return []

        l_norm         = (pd.Series(raw_scores).rank(pct=True, method='average').fillna(0.5) * 2.0) - 1.0
        top_candidates = sorted(l_norm.items(), key=lambda x: x[1], reverse=True)
        sym_by_rank    = [s for s, _ in top_candidates]

        # Regime-based long count (matches local bear_regime_entry_liquidate)
        if self.current_regime == "Crisis":
            target_n = max(1, int(self.top_n * 0.3))   # ~4 positions
            hold_n   = max(2, int(self.hold_n * 0.3))
        elif self.current_regime == "Bear":
            target_n = 0    # No new longs in Bear (hold existing until they exit)
            hold_n   = self.hold_n
        else:
            target_n = self.top_n    # 12
            hold_n   = self.hold_n   # 16

        insights   = []
        to_remove  = []

        # Manage existing longs
        for symbol, orig_weight in list(self.long_signals.items()):
            try:
                rank  = sym_by_rank.index(symbol) + 1
                score = float(l_norm.get(symbol, 0.0))
            except ValueError:
                self._emit_flat(algorithm, symbol, 0.0, insights)
                to_remove.append(symbol)
                self.last_refresh.pop(symbol, None)
                continue

            if rank > hold_n or score < self.min_score:
                self._emit_flat(algorithm, symbol, score, insights)
                to_remove.append(symbol)
                self.last_refresh.pop(symbol, None)
            else:
                last = self.last_refresh.get(symbol)
                if last is None or (algorithm.Time - last).days >= self.refresh_days:
                    self._emit_long(algorithm, symbol, score, insights)
                    self.last_refresh[symbol]   = algorithm.Time
                    self.long_signals[symbol]   = score

        for s in to_remove:
            self.long_signals.pop(s, None)

        # Open new longs
        for symbol, score in top_candidates:
            if len(self.long_signals) >= target_n:
                break
            rank = sym_by_rank.index(symbol) + 1
            if (rank <= target_n
                    and symbol not in self.long_signals
                    and symbol not in self.short_signals
                    and score >= self.min_score
                    and self._security_ok(algorithm, symbol)):
                self._emit_long(algorithm, symbol, score, insights)
                self.long_signals[symbol]  = score
                self.last_refresh[symbol]  = algorithm.Time

        return insights

    # ----------------------------------------------------------
    def _update_shorts(self, algorithm, df):
        """
        Short position management — Bear regime only.
        Matches local: top_shorts=5, hold=10d, stop=8%, snap_profit=5%.
        Fundamental features (f_score, roa, etc.) zeroed — price-based features still active.
        """
        # Short model scoring
        raw_scores = {}
        for symbol, row in df.iterrows():
            fv = np.array(
                [float(row.get(f, self.DEFAULT_FILLS.get(f, 0.0))) for f in self.short_features],
                dtype=float)
            fv = np.clip(np.where(np.isfinite(fv), fv, 0.0), -10.0, 10.0)
            try:
                if hasattr(self.short_model, "predict_proba"):
                    # Probability of class 1 (bearish) → higher = stronger short
                    prob  = float(self.short_model.predict_proba([fv])[0][1])
                    score = (prob * 2.0) - 1.0   # normalise to [-1, 1]
                else:
                    score = float(self.short_model.predict([fv])[0])
                raw_scores[symbol] = score
            except Exception:
                pass

        if not raw_scores:
            return []

        # Rank: lowest (most negative) = strongest short candidate
        s_norm          = (pd.Series(raw_scores).rank(pct=True, method='average').fillna(0.5) * 2.0) - 1.0
        top_short_cands = sorted(s_norm.items(), key=lambda x: x[1])   # ascending → strongest short first
        sym_by_rank     = [s for s, _ in top_short_cands]

        insights  = []
        to_remove = []

        # Manage existing shorts (close if ranked out or signal weakened)
        for symbol in list(self.short_signals.keys()):
            try:
                rank  = sym_by_rank.index(symbol) + 1
                score = float(s_norm.get(symbol, 0.0))
            except ValueError:
                self._emit_flat(algorithm, symbol, 0.0, insights)
                to_remove.append(symbol)
                continue
            if rank > self.short_hold_n or score > -self.min_score:
                self._emit_flat(algorithm, symbol, score, insights)
                to_remove.append(symbol)

        for s in to_remove:
            self.short_signals.pop(s, None)
            self.short_entry_prices.pop(s, None)

        # Open new shorts
        for symbol, score in top_short_cands:
            if len(self.short_signals) >= self.top_shorts:
                break
            rank = sym_by_rank.index(symbol) + 1
            if (rank <= self.top_shorts
                    and symbol not in self.short_signals
                    and symbol not in self.long_signals
                    and score <= -self.short_min_signal
                    and self._security_ok(algorithm, symbol)
                    and self._short_liquidity_ok(algorithm, symbol)):
                insight = Insight.Price(
                    symbol, timedelta(days=self.short_holding_days),
                    InsightDirection.Down, abs(score), None, None, abs(score)
                )
                insights.append(insight)
                self.short_signals[symbol] = abs(score)
                # Record entry price for snap-profit / stop tracking
                self.short_entry_prices[symbol] = float(
                    algorithm.Securities[symbol].Price
                )

        return insights

    # ----------------------------------------------------------
    def _manage_short_intraday(self, algorithm, data):
        """
        Runs every day (not throttled).
        Implements snap-profit (5% decline) and stop-loss (8% adverse) for shorts.
        Matches local: ml_short_snap_profit_pct=0.05, ml_short_stop_loss_pct=0.08
        """
        insights  = []
        to_remove = []

        for symbol, weight in list(self.short_signals.items()):
            if not algorithm.Securities.ContainsKey(symbol):
                continue
            entry_price = self.short_entry_prices.get(symbol, 0.0)
            if entry_price <= 0:
                continue
            current_price = float(algorithm.Securities[symbol].Price)
            if current_price <= 0:
                continue

            pct_change = (current_price - entry_price) / entry_price  # negative = profitable short

            # Snap profit: price fell >= 5% → take profit
            if pct_change <= -self.short_snap_profit_pct:
                self._emit_flat(algorithm, symbol, 0.0, insights)
                to_remove.append(symbol)
                algorithm.Log(
                    f"SHORT snap_profit {symbol.Value}: "
                    f"entry={entry_price:.2f} current={current_price:.2f} "
                    f"({pct_change:.1%})"
                )
                continue

            # Stop loss: price rose >= 8% against short → cut loss
            if pct_change >= self.short_stop_pct:
                self._emit_flat(algorithm, symbol, 0.0, insights)
                to_remove.append(symbol)
                algorithm.Log(
                    f"SHORT stop_loss {symbol.Value}: "
                    f"entry={entry_price:.2f} current={current_price:.2f} "
                    f"({pct_change:.1%})"
                )

        for s in to_remove:
            self.short_signals.pop(s, None)
            self.short_entry_prices.pop(s, None)

        return insights

    # ----------------------------------------------------------
    def _close_all_shorts(self, algorithm):
        """Close all short positions when leaving Bear regime."""
        insights  = []
        to_remove = list(self.short_signals.keys())
        for symbol in to_remove:
            self._emit_flat(algorithm, symbol, 0.0, insights)
        for s in to_remove:
            self.short_signals.pop(s, None)
            self.short_entry_prices.pop(s, None)
        if to_remove:
            algorithm.Log(
                f"Closed {len(to_remove)} shorts: regime={self.current_regime} (left Bear)"
            )
        return insights

    # ----------------------------------------------------------
    def _security_ok(self, algorithm, symbol):
        return (algorithm.Securities.ContainsKey(symbol)
                and algorithm.Securities[symbol].Price > 0)

    def _short_liquidity_ok(self, algorithm, symbol):
        try:
            sec = algorithm.Securities[symbol]
            dv  = getattr(sec, "DollarVolume", None)
            return dv is None or dv >= self.short_min_liquidity_usd
        except Exception:
            return True

    def _emit_long(self, algorithm, symbol, score, insights):
        if self._security_ok(algorithm, symbol):
            insights.append(Insight.Price(
                symbol, timedelta(days=self.refresh_days * 2),
                InsightDirection.Up, score, None, None, score
            ))

    def _emit_flat(self, algorithm, symbol, score, insights):
        if self._security_ok(algorithm, symbol):
            insights.append(Insight.Price(
                symbol, timedelta(days=5), InsightDirection.Flat, score
            ))


# ============================================================
# Per-Symbol Feature Builder
# ============================================================

class SymbolData:
    def __init__(self, algorithm, symbol, spy_close_window, prefetched_history=None):
        self.symbol          = symbol
        self.spy_close_window = spy_close_window

        self.rsi    = RelativeStrengthIndex(14)
        self.std20  = StandardDeviation(20)
        self.std5   = StandardDeviation(5)
        self.sma50  = SimpleMovingAverage(50)
        self.sma200 = SimpleMovingAverage(200)

        self.close_window   = RollingWindow[float](253)
        self.volume_window  = RollingWindow[float](60)
        self.returns_window = RollingWindow[float](2)
        self.vol_history    = RollingWindow[float](252)

        history = (prefetched_history if prefetched_history is not None
                   else algorithm.History(symbol, 253, Resolution.Daily))
        if history is not None and len(history) > 0:
            for idx, row in history.iterrows():
                try:
                    close = float(row['close'])
                except Exception:
                    close = float(row[4]) if len(row) > 4 else 0.0
                close = max(0.01, min(20000.0, round(close, 4)))
                if not np.isfinite(close):
                    continue
                try:
                    volume = float(row['volume'])
                except Exception:
                    volume = float(row[5]) if len(row) > 5 else 0.0
                time = idx[1] if isinstance(idx, tuple) else idx
                self.close_window.Add(close)
                self.volume_window.Add(volume)
                try:
                    self.rsi.Update(time, close)
                    self.sma50.Update(time, close)
                    self.sma200.Update(time, close)
                    self.returns_window.Add(close)
                    if self.returns_window.Count >= 2:
                        ret = max(-0.5, min(0.5,
                            (self.returns_window[0] / self.returns_window[1]) - 1.0))
                        self.std20.Update(time, ret)
                        self.std5.Update(time, ret)
                        self.vol_history.Add(self.std20.Current.Value * np.sqrt(252))
                except Exception:
                    pass

    @property
    def IsReady(self):
        return (self.close_window.Count >= 253 and self.sma50.IsReady
                and self.rsi.IsReady and self.spy_close_window.Count >= 50)

    def Update(self, bar):
        if bar is None:
            return
        close = float(bar.Close) if hasattr(bar, 'Close') else float(bar['close'])
        close = max(0.01, min(20000.0, round(close, 4)))
        if not np.isfinite(close):
            return
        time   = bar.EndTime if hasattr(bar, 'EndTime') else None
        volume = float(bar.Volume) if hasattr(bar, 'Volume') else 0.0
        if time is None:
            return
        try:
            self.close_window.Add(close)
            self.volume_window.Add(volume)
            self.rsi.Update(time, close)
            self.sma50.Update(time, close)
            self.sma200.Update(time, close)
            self.returns_window.Add(close)
            if self.returns_window.Count >= 2:
                ret = max(-0.5, min(0.5,
                    (self.returns_window[0] / self.returns_window[1]) - 1.0))
                self.std20.Update(time, ret)
                self.std5.Update(time, ret)
                self.vol_history.Add(self.std20.Current.Value * np.sqrt(252))
        except Exception:
            pass

    def GetFeatures(self):
        if not self.IsReady:
            return None

        close = self.close_window[0]

        def safe_ret(n):
            return (close / self.close_window[n]) - 1.0 if self.close_window.Count > n else 0.0

        ret_1d  = safe_ret(1)
        ret_5d  = safe_ret(5)
        ret_10d = safe_ret(10)
        ret_20d = safe_ret(20)
        ret_60d = safe_ret(60)
        mom_6m  = safe_ret(126)

        ma50     = self.sma50.Current.Value
        ma200    = self.sma200.Current.Value
        ma_cross = 1.0 if ma50 > ma200 else -1.0
        mom_3m   = safe_ret(63)
        f_trend  = (0.30 * mom_3m * 10.0 + 0.25 * mom_6m * 10.0
                    + 0.25 * ma_cross + 0.20 * ret_1d * 10.0)

        vol_current    = self.std20.Current.Value * np.sqrt(252)
        vol_5d         = self.std5.Current.Value  * np.sqrt(252)
        vol_ratio_5_20 = vol_5d / vol_current if vol_current > 1e-9 else 1.0

        vol_perc = 0.5
        if self.vol_history.Count > 20:
            hist     = [self.vol_history[i] for i in range(self.vol_history.Count)]
            vol_perc = sum(1 for v in hist if v < vol_current) / float(len(hist))

        rsi_val        = self.rsi.Current.Value
        rsi_overbought = 1.0 if rsi_val > 70 else 0.0

        win52      = min(252, self.close_window.Count - 1)
        prices_52  = [self.close_window[i] for i in range(win52)]
        high_52w   = max(prices_52) if prices_52 else close
        low_52w    = min(prices_52) if prices_52 else close
        nearness_52w_high  = float(np.clip(close / max(high_52w, 1e-6), 0.0, 1.0))
        dist_from_52w_high = float(np.clip((close - high_52w) / max(high_52w, 1e-6), -1.0, 0.0))
        dist_low           = (close - low_52w) / max(low_52w, 1e-6)
        nearness_52w_low   = 1.0 / (1.0 + max(0.0, dist_low))

        capm_alpha   = 0.0
        capm_res_vol = 0.05
        try:
            win_c = min(60, self.close_window.Count - 1, self.spy_close_window.Count - 1)
            if win_c >= 20:
                stock_rets = np.array(
                    [(self.close_window[i] / self.close_window[i+1]) - 1.0
                     for i in range(win_c)], dtype=float)
                spy_rets = np.array(
                    [(self.spy_close_window[i] / self.spy_close_window[i+1]) - 1.0
                     for i in range(win_c)], dtype=float)
                var_spy      = float(np.var(spy_rets))
                beta         = (float(np.cov(stock_rets, spy_rets)[0][1] / var_spy)
                                if var_spy > 1e-9 else 1.0)
                residuals    = stock_rets - beta * spy_rets
                capm_res_vol = float(np.std(residuals))
                capm_alpha   = float(np.mean(residuals) * 252)
        except Exception:
            pass

        short_term_reversal = float(np.clip(-ret_5d, -0.5, 0.5))
        low_vol_score       = 1.0 - vol_perc

        win_q         = min(60, self.close_window.Count - 1)
        quality_score = 0.0
        if win_q >= 20:
            rets_q = np.array(
                [(self.close_window[i] / self.close_window[i+1]) - 1.0
                 for i in range(win_q)], dtype=float)
            std_q = float(np.std(rets_q))
            quality_score = float(np.clip(
                np.mean(rets_q) / std_q * np.sqrt(252) if std_q > 1e-9 else 0.0,
                -5.0, 5.0))

        win_d             = min(60, self.close_window.Count - 1)
        down_up_vol_ratio = 1.0
        if win_d >= 20:
            rets_d = np.array(
                [(self.close_window[i] / self.close_window[i+1]) - 1.0
                 for i in range(win_d)], dtype=float)
            up_r   = rets_d[rets_d > 0]
            down_r = rets_d[rets_d <= 0]
            vol_up   = float(np.std(up_r))   if len(up_r)   > 2 else 1e-9
            vol_down = float(np.std(down_r)) if len(down_r) > 2 else 1e-9
            down_up_vol_ratio = float(np.clip(vol_down / max(vol_up, 1e-9), 0.1, 10.0))

        momentum_acceleration = ret_5d - ret_10d

        return {
            "f_trend":               f_trend,
            "ret_1d":                ret_1d,
            "ret_5d":                ret_5d,
            "ret_10d":               ret_10d,
            "ret_20d":               ret_20d,
            "ret_60d":               ret_60d,
            "momentum_6m":           mom_6m,
            "rolling_vol_20":        vol_current,
            "vol_ratio_5_20":        vol_ratio_5_20,
            "rsi_14":                rsi_val,
            "rsi_overbought":        rsi_overbought,
            "nearness_52w_high":     nearness_52w_high,
            "nearness_52w_low":      nearness_52w_low,
            "dist_from_52w_high":    dist_from_52w_high,
            "capm_residual_vol":     capm_res_vol,
            "capm_alpha":            capm_alpha,
            "short_term_reversal":   short_term_reversal,
            "low_vol_score":         low_vol_score,
            "quality_score":         quality_score,
            "down_up_vol_ratio":     down_up_vol_ratio,
            "momentum_acceleration": momentum_acceleration,
            # Unavailable in QC — zeroed
            "vol_expansion":         0.0,
            "earnings_surprise":     0.0,
            "f_score":               0.0,
            "delta_roa":             0.0,
            "delta_leverage":        0.0,
            "roa":                   0.0,
            "accruals_ratio":        0.0,
            # Overwritten by Update() cross-sectional pass
            "sector_relative_20d":   0.0,
            "sector_relative_60d":   0.0,
            "cs_momentum_percentile": 0.5,
        }
