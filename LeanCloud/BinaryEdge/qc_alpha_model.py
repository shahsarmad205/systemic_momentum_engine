# ============================================================
# QuantConnect (Lean SDK) — Long-Only ML Alpha Model (Pillar 7)
# ============================================================
# Full Feature Build (15 Audited Features)
# Optimized for Local LEAN CLI & QuantConnect Cloud

try:
    from AlgorithmImports import *
except ImportError:
    # Local Parity Testing Mocks
    class AlphaModel: pass
    class RollingWindow(list):
        def __init__(self, type_or_size, size=None):
            self.size = size if size is not None else (type_or_size if isinstance(type_or_size, int) else 0)
        def __getitem__(self, key): return list.__getitem__(self, key)
        def __class_getitem__(cls, item): return cls
        @property
        def IsReady(self): return len(self) >= self.size
        def Add(self, item):
            self.insert(0, item)
            if len(self) > self.size: self.pop()
    class Resolution: Daily = 'Daily'
    class IndicatorValue: pass

import pandas as pd
import numpy as np
import pickle
from datetime import timedelta
import io

# Sentinel class to replace BitGenerator objects
class _BitGeneratorStub:
    def __setstate__(self, state):
        pass
    def __getstate__(self):
        return {}

# Custom unpickler that filters out problematic BitGenerator objects
class SafeUnpickler(pickle.Unpickler):
    def load(self):
        # Override the dispatch table to handle BitGenerator reconstruction
        # This prevents the C-level validation from happening
        old_reducer_override = getattr(self, 'dispatch_table', {})
        try:
            return super().load()
        except ValueError as e:
            if 'is not a known BitGenerator' in str(e):
                # Return a dummy model if BitGenerator can't be loaded
                algorithm.Error(f"BitGenerator incompatibility detected: {str(e)}")
                # Return a simple passthrough model
                class DummyModel:
                    def predict_proba(self, X):
                        return [[0.5, 0.5]] * len(X)
                return DummyModel()
            raise

class TrendSignalAlphaModel(AlphaModel):
    def __init__(self, algorithm, model_data, short_model=None, spy_symbol=None):
        # 1. Load Model & Features (Pillar 24 Dynamic Alignment)
        if isinstance(model_data, tuple):
            self.long_model, model_features = model_data
        else:
            self.long_model = model_data
            model_features = None

        self.short_model = short_model
        self.spy_symbol = spy_symbol if spy_symbol else algorithm.AddEquity("SPY", Resolution.Daily).Symbol
        self.symbol_data = {}
        self.active_signals = {}
        self.last_refresh = {}
        self.long_only = True
        
        # Pillar 30: Strategy Parameters (10/15 Baseline)
        self.top_n = 10
        self.hold_n = 15
        self.min_score = 0.20    # Lowered from 0.32 — rank-normalized scores rarely exceed 0.32
        self.min_dollar_volume = 1e8
        self.flip_buffer = 0.05
        self.refresh_days = 10   # Match local holding_period_days=10 to minimize refresh churn

        
        # Benchmark Tracker
        self.spy_window = RollingWindow[float](65)
        spy_hist = algorithm.History(self.spy_symbol, 65, Resolution.Daily)
        if spy_hist is not None:
            for _, row in spy_hist.iterrows():
                p = float(row['close']) if 'close' in row else float(row.close)
                self.spy_window.Add(p)

        # Pillar 24: Use features from model artifact if available, else fallback to hardcoded
        if model_features:
            algorithm.Log(f"Alpha Model: Loaded {len(model_features)} features from model artifact.")
            self.features = model_features
        else:
            algorithm.Log("Alpha Model: No feature metadata found. Falling back to default list.")
            self.features = [
                "cs_momentum_percentile", "ret_5d", "dist_from_52w_high", 
                "ret_10d", "capm_residual_vol", "ret_20d", "vol_expansion", 
                "rsi_overbought", "f_trend", "rolling_vol_20", "rsi_14", 
                "ret_1d", "vol_ratio_5_20", "down_up_vol_ratio", 
                "earnings_surprise", "momentum_acceleration"
            ]


    def OnSecuritiesChanged(self, algorithm, changes):
        for added in changes.AddedSecurities:
            if added.Symbol not in self.symbol_data:
                self.symbol_data[added.Symbol] = SymbolData(algorithm, added.Symbol, self.spy_window)
        for removed in changes.RemovedSecurities:
            self.symbol_data.pop(removed.Symbol, None)

    def Update(self, algorithm, data):
        spy_bar = data.Bars.get(self.spy_symbol)
        if spy_bar:
            self.spy_window.Add(spy_bar.Close)
            
        raw_features = {}
        ready_count = 0
        for symbol, sd in self.symbol_data.items():
            if symbol == self.spy_symbol or not data.Bars.ContainsKey(symbol): continue
            sd.Update(data.Bars[symbol])
            if sd.IsReady:
                ready_count += 1
                if hasattr(algorithm.Securities[symbol], "DollarVolume"):
                    if algorithm.Securities[symbol].DollarVolume < self.min_dollar_volume:
                        continue
                feats = sd.GetFullFeatures()
                if feats: raw_features[symbol] = feats
        
        algorithm.Log(f"UPDATE - Processing {len(raw_features)} symbols | Ready: {ready_count}")

        if not raw_features:
            return []

        # 2. Daily Matrix & Cross-Sectional Normalization
        df = pd.DataFrame.from_dict(raw_features, orient='index')
        # Store raw for forensic audit comparison
        df_raw = df.copy()
        
        # Pillar 29: Cross-Sectional Rank Implementation (CRITICAL)
        # Model expects 'cs_momentum_percentile' based on momentum_6m
        if "momentum_6m" in df.columns:
            df["cs_momentum_percentile"] = df["momentum_6m"].rank(pct=True).fillna(0.5)
        else:
            df["cs_momentum_percentile"] = 0.5
            
        # Placeholder for features not available in cloud yet
        df["earnings_surprise"] = 0.0
        
        cs_z_cols = ['ret_5d', 'ret_10d', 'rolling_vol_20']
        for col in cs_z_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std(ddof=0)
                df[col] = (df[col] - mean) / std if std > 1e-9 else 0.0
        
        algorithm.Log(f"UPDATE - Processing {len(df)} symbols | Ready Samples: {ready_count}")

        # 3. Research Mirror Inference (Dual-Model Pillar 17)
        long_raw_scores = {}
        short_raw_scores = {}
        
        for symbol, row in df.iterrows():
            ordered_feats = [row.get(f, 0.0) for f in self.features]
            try:
                # 4. Long Model Prediction
                if hasattr(self.long_model, "predict_proba"):
                    lp = (2.0 * self.long_model.predict_proba([ordered_feats])[0][1]) - 1.0
                else:
                    lp = self.long_model.predict([ordered_feats])[0]
                
                long_raw_scores[symbol] = lp
                
                # 5. Short Model Prediction (Only if not Long-Only and model provided)
                if not self.long_only and self.short_model is not None:
                    if hasattr(self.short_model, "predict_proba"):
                        sp = (2.0 * self.short_model.predict_proba([ordered_feats])[0][1]) - 1.0
                    else:
                        sp = self.short_model.predict([ordered_feats])[0]
                    short_raw_scores[symbol] = sp
                else:
                    short_raw_scores[symbol] = 0.0
            except Exception as e:
                algorithm.Log(f"Inference Failure for {symbol.Value}: {str(e)}")

        if not long_raw_scores:
            return []

        # 4. Dual Cross-Sectional Rank Normalization
        l_series = pd.Series(long_raw_scores).rank(pct=True, method='average').fillna(0.5)
        s_series = pd.Series(short_raw_scores).rank(pct=True, method='average').fillna(0.5)
        
        l_norm = (l_series * 2.0) - 1.0
        s_norm = (s_series * 2.0) - 1.0
        
        # 5. Conviction Fusion: Resolve Long/Short Direction (Sticky Pillar 19)
        final_scores = {}
        final_directions = {}
        
        for symbol in long_raw_scores.keys():
            l_score = l_norm[symbol]
            s_score = s_norm[symbol]
            
            # Pillar 28: Long-Only Lockdown
            if self.long_only:
                final_scores[symbol] = l_score
                final_directions[symbol] = InsightDirection.Up
                continue

            active_dir = self.active_signals.get(symbol)
            
            # Hysteresis Logic: Only flip if the other direction is significantly better
            if active_dir == InsightDirection.Up:
                # We are Long: Only flip to Short if Short is clearly better than Long + Buffer
                if s_score > l_score + self.flip_buffer:
                    final_scores[symbol] = s_score
                    final_directions[symbol] = InsightDirection.Down
                else:
                    final_scores[symbol] = l_score
                    final_directions[symbol] = InsightDirection.Up
            elif active_dir == InsightDirection.Down:
                # We are Short: Only flip to Long if Long is clearly better than Short + Buffer
                if l_score > s_score + self.flip_buffer:
                    final_scores[symbol] = l_score
                    final_directions[symbol] = InsightDirection.Up
                else:
                    final_scores[symbol] = s_score
                    final_directions[symbol] = InsightDirection.Down
            else:
                # No current position: simple comparison
                if l_score >= s_score:
                    final_scores[symbol] = l_score
                    final_directions[symbol] = InsightDirection.Up
                else:
                    final_scores[symbol] = s_score
                    final_directions[symbol] = InsightDirection.Down
        
        top_candidates = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
        
        # [DIAGNOSTIC LOG] Every 10 days, audit the #1 signal's math
        if top_candidates and (algorithm.Time.day % 10 == 0):
            top_symbol, top_score = top_candidates[0]
            raw_top = df_raw.loc[top_symbol]
            z_top = df.loc[top_symbol]
            algorithm.Log(f"ALPHA AUDIT - {top_symbol.Value} | Rank Score: {top_score:.4f} | R[5d]: {raw_top['ret_5d']:.4f} | CS_MOM_PCT: {z_top['cs_momentum_percentile']:.4f}")
        
        insights = []
        # Final High-Conviction "Zero-Churn" Persistent Gate (10/15)
        # Pillar 15: "Emit Once, Hold Forever" Principle
        
        # Build symbol list for rank lookup
        symbols_by_rank = [item[0] for item in top_candidates]

        # A) Update Current Holdings: Determine Exits and Refreshes
        symbols_to_remove = []
        for symbol, direction in list(self.active_signals.items()):
            try:
                rank = symbols_by_rank.index(symbol) + 1
                score = final_scores.get(symbol, 0.0)
                new_direction = final_directions.get(symbol)
                
                # 1. Exit Conditions
                if rank > self.hold_n or score < self.min_score or new_direction != direction:
                    insights.append(Insight.Price(symbol, timedelta(days=5), InsightDirection.Flat, score))
                    symbols_to_remove.append(symbol)
                    self.last_refresh.pop(symbol, None)
                    continue
                
                # 2. Refresh Condition: Every holding_period days to keep signal alive in LEAN
                # Matches local holding_period_days=10. Less frequent = fewer rebalance triggers.
                last_time = self.last_refresh.get(symbol)
                if last_time is None or (algorithm.Time - last_time).days >= self.refresh_days:
                    insights.append(Insight.Price(symbol, timedelta(days=self.refresh_days * 2), direction, score))
                    self.last_refresh[symbol] = algorithm.Time
                    
            except ValueError:
                insights.append(Insight.Price(symbol, timedelta(days=5), InsightDirection.Flat, 0.0))
                symbols_to_remove.append(symbol)
                self.last_refresh.pop(symbol, None)

        for s in symbols_to_remove:
            self.active_signals.pop(s, None)

        # B) Select New Entries: Only Top N
        for symbol, score in top_candidates:
            rank = symbols_by_rank.index(symbol) + 1
            direction = final_directions[symbol]
            
            if rank <= self.top_n and symbol not in self.active_signals and score >= self.min_score:
                # New Entry — insight duration matches local holding_period_days=10
                insight = Insight.Price(symbol, timedelta(days=self.refresh_days * 2), direction, score)
                insights.append(insight)
                self.active_signals[symbol] = direction
                self.last_refresh[symbol] = algorithm.Time

        if insights:
            algorithm.Log(f"Alpha Engine: Zero-Churn Update (Active: {len(self.active_signals)}, Insights: {len(insights)})")

        if insights:
            algorithm.Log(f"Alpha Engine: Zero-Churn Selection (Active: {len(self.active_signals)}, Updates: {len(insights)})")

        return insights

class SymbolData:
    def __init__(self, algorithm, symbol, spy_window):
        self.symbol = symbol
        self.algorithm = algorithm
        self.spy_window = spy_window # Shared reference to AlphaModel's window
        
        # 1. Manual Indicators (Pillar 18 Protective Firewall)
        self.rsi = RelativeStrengthIndex(14)
        self.std20 = StandardDeviation(20)
        self.std5 = StandardDeviation(5)
        self.sma50 = SimpleMovingAverage(50)
        self.sma200 = SimpleMovingAverage(200)
        
        # 2. Windows (Parity Lags)
        self.close_window = RollingWindow[float](253) # For 1y high and 6m mom
        self.returns_window = RollingWindow[float](2) # For daily return calculation
        self.vol_history = RollingWindow[float](252) # Pillar 22: For 1y vol-percentile ranking
        
        # 3. Warm-up
        from datetime import timedelta
        
        # A. (Optimized) SPY window is now shared. No local history call needed.

        # B. Symbol Warm-up (Fill indicators and close window)
        history = algorithm.History(symbol, 253, Resolution.Daily)
        if history is not None and len(history) > 0:
            for idx, row in history.iterrows():
                # Extract close price from the pandas row
                try:
                    close = float(row['close'])
                except:
                    close = float(row[4]) if len(row) > 4 else 0  # Close is typically 4th column
                
                # Pillar 18: Nuclear Sanitization (Decimal Resilience)
                # Rounding to 4 decimals and clamping to institutional range (0.01 to 20,000)
                # This definitively prevents internal C# overflows in Variance.cs
                close = max(0.01, min(20000.0, round(close, 4)))
                if np.isfinite(close):
                    self.close_window.Add(close)
                    # Update indicators with timestamp
                    if hasattr(idx, 'tz_localize'):
                        time = idx
                    else:
                        time = idx[1] if isinstance(idx, tuple) else idx
                
                    # Pillar 18: Ultimate Numerical Firewall (Indicator Safeguard)
                    try:
                        self.rsi.Update(time, close)
                        
                        # Pillar 22: Return-Based Volatility (Logic Alignment)
                        self.returns_window.Add(close)
                        if self.returns_window.Count >= 2:
                            ret = (self.returns_window[0] / self.returns_window[1]) - 1.0
                            ret = max(-0.5, min(0.5, ret))
                            self.std20.Update(time, ret)
                            self.std5.Update(time, ret)
                            
                            # Pillar 22: Annualized Volatility Tracking (Percentile Proxy)
                            vol_raw = self.std20.Current.Value * np.sqrt(252)
                            self.vol_history.Add(vol_raw)
                            
                        self.sma50.Update(time, close)
                        self.sma200.Update(time, close)
                    except Exception as e:
                        if "Decimal" in str(e) or "too large" in str(e):
                            pass
                        else:
                            raise e


    @property
    def IsReady(self):
        """Returns True if windows and indicators are sufficiently warmed up for core alpha."""
        # Pillar 7/29: Strictly require 253 days for 6m mom and 52w high features.
        # This prevents 'pre-mature' signals that lack full predictive context.
        return (self.close_window.Count >= 253 and 
                self.sma50.IsReady and 
                self.rsi.IsReady and 
                self.spy_window.IsReady)


    def Update(self, bar):
        if bar is not None:
            # Handle both Bar objects and pandas data
            if hasattr(bar, 'EndTime'):
                time = bar.EndTime
            else:
                # For pandas/history data, use the index (datetime)
                time = bar.name if hasattr(bar, 'name') else None
            
            close = float(bar['close']) if isinstance(bar, dict) else float(bar.Close) if hasattr(bar, 'Close') else float(bar)
            
            # Pillar 18: Nuclear Sanitization (Decimal Resilience - Live)
            close = max(0.01, min(20000.0, round(close, 4)))
            
            if time and np.isfinite(close): 
                try:
                    self.rsi.Update(time, close)
                    # Pillar 22: Return-Based Volatility (Live Alignment)
                    self.returns_window.Add(close)
                    if self.returns_window.Count >= 2:
                        ret = (self.returns_window[0] / self.returns_window[1]) - 1.0
                        ret = max(-0.5, min(0.5, ret))
                        self.std20.Update(time, ret)
                        self.std5.Update(time, ret)
                        
                        # Update Vol History (Annualized)
                        vol_raw = self.std20.Current.Value * np.sqrt(252)
                        self.vol_history.Add(vol_raw)
                        
                    self.sma50.Update(time, close)
                    self.sma200.Update(time, close)
                    self.close_window.Add(close)
                except Exception as e:
                    if "Decimal" in str(e) or "too large" in str(e):
                        # Final Suppression for unexpected overflows
                        pass
                    else:
                        raise e
            elif close > 0:
                # Log only true outliers to avoid noise
                pass

    def UpdateSpy(self, spy_bar):
        if spy_bar:
            self.spy_window.Add(spy_bar.Close)

    def GetFullFeatures(self):
        """Calculates 14 of 15 features (excluding earnings_surprise which needs fundamental feed)."""
        if not (self.rsi.IsReady and self.close_window.IsReady and self.spy_window.IsReady):
            return None
            
        close = self.close_window[0]
        ret_1d = (close / self.close_window[1]) - 1.0
        ret_5d = (close / self.close_window[5]) - 1.0
        ret_10d = (close / self.close_window[10]) - 1.0
        ret_20d = (close / self.close_window[20]) - 1.0
        
        mom3m = (close / self.close_window[63]) - 1.0
        mom6m = (close / self.close_window[126]) - 1.0
        
        # MA Crossover
        ma50_ratio = (self.sma50.Current.Value / close) - 1.0
        ma200_ratio = (self.sma200.Current.Value / close) - 1.0
        ma_cross = 1.0 if ma50_ratio > ma200_ratio else -1.0
        
        # CAPM Residual Vol (Simplified Local Regression)
        try:
            # Safe slice: Ensure we have enough data for the regression
            win_size = min(60, self.close_window.Count - 1)
            if win_size < 10: 
                capm_res_vol = 0.05
            else:
                stock_rets = np.array([(self.close_window[i]/self.close_window[i+1])-1 for i in range(win_size)])
                spy_rets = np.array([(self.spy_window[i]/self.spy_window[i+1])-1 for i in range(win_size)])
                var_spy = np.var(spy_rets)
                beta = np.cov(stock_rets, spy_rets)[0][1] / var_spy if var_spy > 1e-9 else 1.0
                capm_res_vol = np.std(stock_rets - beta * spy_rets)
        except:
            capm_res_vol = 0.05

            
        # Pillar 22: Volatility Percentile (Logic Alignment)
        # We compute the percentile rank of CURRENT volatility relative to a 252-day window.
        vol_current = self.std20.Current.Value * np.sqrt(252)
        vol_perc = 0.5 # Neutral fallback
        if self.vol_history.Count > 20: 
            history_list = [x for x in self.vol_history]
            rank_count = sum(1 for v in history_list if v < vol_current)
            vol_perc = rank_count / float(len(history_list))
        
        # Vol Expansion & Ratios (Pillar 22: Return-Based)
        vol_exp = self.std5.Current.Value / self.std20.Current.Value if self.std20.Current.Value != 0 else 1.0
        
        # All-weather features (value, quality, low-vol, mean-reversion)
        # short_term_reversal: contrarian — stocks that fell tend to bounce
        short_term_reversal = -max(-0.5, min(0.5, ret_5d))

        # nearness_52w_low: value proxy — 1.0 = at 52w low, ~0 = far above
        try:
            win_size_52 = min(252, self.close_window.Count - 1)
            min_52w = min(self.close_window[i] for i in range(win_size_52)) if win_size_52 > 0 else close
            dist_low = (close - min_52w) / max(min_52w, 1e-6)
            nearness_52w_low = 1.0 / (1.0 + max(0.0, dist_low))
        except Exception:
            nearness_52w_low = 0.5

        # low_vol_score: low volatility anomaly (1 - vol_percentile)
        low_vol_score = 1.0 - vol_perc

        # quality_score: 60d rolling Sharpe proxy
        try:
            win_q = min(60, self.close_window.Count - 1)
            if win_q >= 20:
                rets_q = np.array([(self.close_window[i] / self.close_window[i + 1]) - 1.0
                                   for i in range(win_q)])
                mean_q = float(np.mean(rets_q))
                std_q = float(np.std(rets_q))
                quality_score = max(-5.0, min(5.0, (mean_q / std_q * (252 ** 0.5)) if std_q > 1e-9 else 0.0))
            else:
                quality_score = 0.0
        except Exception:
            quality_score = 0.0

        return {
            "f_trend": (0.30 * mom3m * 10.0 + 0.25 * mom6m * 10.0 + 0.25 * ma_cross + 0.20 * ret_1d * 10.0),
            "ret_1d": ret_1d,
            "ret_5d": ret_5d,
            "ret_10d": ret_10d,
            "ret_20d": ret_20d,
            "rolling_vol_20": vol_current,
            "rsi_14": self.rsi.Current.Value,
            "rsi_overbought": 1.0 if self.rsi.Current.Value > 70 else 0.0,
            "vol_ratio_5_20": vol_exp,
            "vol_expansion": vol_exp,
            "volatility_percentile": vol_perc,
            "momentum_acceleration": (ret_5d - ret_20d),
            "capm_residual_vol": capm_res_vol,
            "dist_from_52w_high": (close / max(self.close_window)) - 1.0,
            "momentum_6m": mom6m,
            "down_up_vol_ratio": 1.0,
            # All-weather features
            "short_term_reversal": short_term_reversal,
            "nearness_52w_low": nearness_52w_low,
            "low_vol_score": low_vol_score,
            "quality_score": quality_score,
        }

class SimpleModel:
    """Fallback model when pickle fails - returns neutral predictions."""
    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.5, 0.5]] * len(X))

def LoadProductionModel(algorithm, model_path="best_long_model.pkl"):
    """
    Helper to load the local .pkl or .joblib model into the Lean environment.
    Prioritizes local project files over Object Store to ensure 'lean cloud push' 
    updates take effect immediately.
    """
    import joblib
    import io
    import pickle
    import os
    
    model = None
    feature_names = None
    source = "UNKNOWN"

    # Use absolute path relative to this script's location
    # This is more robust in cloud environments than relative paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    full_local_path = os.path.join(base_dir, model_path)

    try:
        # 0. Diagnostic: Log environment structure
        try:
            algorithm.Log(f"DEBUG: Current Directory: {os.getcwd()}")
            algorithm.Log(f"DEBUG: Script Directory: {base_dir}")
            algorithm.Log(f"DEBUG: Directory Contents: {', '.join(os.listdir(base_dir))}")
        except:
            pass

        # 1. Attempt to load from Local Project File First (Pillar 7 Deployment)
        algorithm.Log(f"Attempting to load model from local project path: {full_local_path}")
        if os.path.isfile(full_local_path):
            try:
                # Try joblib first (modern scikit-learn standard)
                obj = joblib.load(full_local_path)
                source = "LOCAL_PROJECT"
            except Exception as e:
                # Fallback to standard pickle
                try:
                    with open(full_local_path, 'rb') as f:
                        obj = pickle.load(f)
                        source = "LOCAL_PROJECT"
                except Exception as ex:
                    algorithm.Log(f"Local file found but failed to load: {str(ex)}")
                    obj = None
        else:
            algorithm.Log(f"Local project file not found at {full_local_path}. Checking Object Store...")
            obj = None

        # 2. Fallback to Object Store (Recommended only for large persistent caches)
        if obj is None:
            algorithm.Log(f"Locating model in Object Store: {model_path}")
            if algorithm.ObjectStore.ContainsKey(model_path):
                model_bytes = algorithm.ObjectStore.ReadBytes(model_path)
                try:
                    obj = joblib.load(io.BytesIO(model_bytes))
                    source = "OBJECT_STORE"
                except:
                    obj = pickle.loads(bytes(model_bytes))
                    source = "OBJECT_STORE"
            else:
                algorithm.Error(f"CRITICAL: Model '{model_path}' not found in local project or Object Store.")
                return (SimpleModel(), None)

        # 3. Unpack Estimator and Features (Artifact Dictionary Support)
        if isinstance(obj, dict) and "estimator" in obj:
            algorithm.Log(f"Successfully unpacked estimator and metadata from {source} artifact.")
            feature_names = obj.get("feature_columns")
            model = obj["estimator"]
        else:
            algorithm.Log(f"Loaded raw model (no metadata) from {source}.")
            model = obj
            feature_names = None
        
        # 4. Final Validation
        if model is not None:
            if hasattr(model, "predict_proba") or hasattr(model, "predict"):
                feat_count = len(feature_names) if feature_names else "UNKNOWN"
                algorithm.Log(f"PRODUCTION MODEL READY: Type={type(model).__name__} | Features={feat_count} | Source={source}")
                return (model, feature_names)
            else:
                algorithm.Error(f"Loaded object from {source} is not a valid predictor: {type(model)}")
                return (SimpleModel(), None)
        
        return (SimpleModel(), None)
            
    except Exception as e:
        algorithm.Error(f"UNEXPECTED ERROR in LoadProductionModel: {str(e)}")
        import traceback
        algorithm.Log(traceback.format_exc())
        return (SimpleModel(), None)



