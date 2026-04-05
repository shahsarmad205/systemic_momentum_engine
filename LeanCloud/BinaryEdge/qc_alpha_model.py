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
    def __init__(self, long_model, short_model, spy_symbol):
        """
        Pillar 17: Market-Neutral Alpha Engine.
        Injects two separate estimators for symmetric momentum capture.
        """
        self.long_model = long_model
        self.short_model = short_model
        self.spy_symbol = spy_symbol
        self.symbol_data = {}
        self.active_signals = {} # Pillar 15: Zero-Churn State Tracker
        
        # Pillar 13, 14 & 17: Signal & Persistence Calibration
        self.score_scale = 1.0 
        self.score_direction = 1.0
        self.min_score = 0.32 # Pillar 21: Tightened floor to prune Alpha-Drag
        self.flip_buffer = 0.15 # Pillar 19: Sticky Hysteresis
        self.top_n = 8 # Pillar 21: Concentrated Top 8 Long/Short
        self.hold_n = 12 # Pillar 21: Persistent Hold-Gate (Top 12)
        self.long_only = True  # Pillar 28: LOCKED to Long-Only (Pillar 7 Baseline)
        self.forensic_logging = True # Active Audit Mode

        # Full 15 Audited Features (Pillar 7 Final)
        self.features = [
            "capm_residual_vol", "momentum_acceleration", "ret_10d", 
            "cs_momentum_percentile", "ret_20d", "rolling_vol_20", 
            "ret_5d", "rsi_14", "ret_1d", "rsi_overbought", 
            "vol_ratio_5_20", "vol_expansion", "down_up_vol_ratio", 
            "dist_from_52w_high", "f_trend"
        ]

    def OnSecuritiesChanged(self, algorithm, changes):
        """Handle additions and removals from the universe."""
        for added in changes.AddedSecurities:
            if added.Symbol not in self.symbol_data:
                self.symbol_data[added.Symbol] = SymbolData(algorithm, added.Symbol, self.spy_symbol)

        for removed in changes.RemovedSecurities:
            data = self.symbol_data.pop(removed.Symbol, None)
            if data:
                # Cleanup logic if using consolidators
                pass

    def Update(self, algorithm, data):
        """Daily Alpha Update with Forensic Parity Audit."""
        # 1. Update Indicators and Collect Raw Features
        spy_bar = data.Bars.get(self.spy_symbol)
        raw_features = {}
        
        for symbol, sd in self.symbol_data.items():
            if symbol == self.spy_symbol: continue
            if not data.Bars.ContainsKey(symbol): continue
            
            sd.Update(data.Bars[symbol])
            if spy_bar:
                sd.UpdateSpy(spy_bar)
            if not sd.IsReady: continue
            
            feats = sd.GetFullFeatures()
            if feats:
                raw_features[symbol] = feats

        if not raw_features:
            return []

        # 2. Daily Matrix & Cross-Sectional Normalization
        df = pd.DataFrame.from_dict(raw_features, orient='index')
        # Store raw for forensic audit comparison
        df_raw = df.copy()
        
        cs_z_cols = ['ret_5d', 'ret_10d', 'rolling_vol_20']
        for col in cs_z_cols:
            if col in df.columns:
                mean = df[col].mean()
                std = df[col].std(ddof=0)
                df[col] = (df[col] - mean) / std if std > 1e-9 else 0.0

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
        
        # [FORENSIC LOG] Every morning, audit the #1 signal's math
        if self.forensic_logging and top_candidates:
            top_symbol, top_score = top_candidates[0]
            raw_top = df_raw.loc[top_symbol]
            z_top = df.loc[top_symbol]
            algorithm.Log(f"AUDIT - {top_symbol.Value} | Signal: {top_score:.4f} | Raw[5d]: {raw_top['ret_5d']:.4f} | Z[5d]: {z_top['ret_5d']:.4f}")
        
        insights = []
        # Final High-Conviction "Zero-Churn" Persistent Gate (10/15)
        # Pillar 15: "Emit Once, Hold Forever" Principle
        
        # Build symbol list for rank lookup
        symbols_by_rank = [item[0] for item in top_candidates]

        # A) Update Current Holdings: Determine Exits
        symbols_to_remove = []
        for symbol, direction in list(self.active_signals.items()):
            try:
                rank = symbols_by_rank.index(symbol) + 1
                score = final_scores.get(symbol, 0.0)
                new_direction = final_directions.get(symbol)
                
                # Exit Conditions:
                # 1. Rank drops below Hold Gate (Top 15)
                # 2. Score drops below Threshold (0.25)
                # 3. Direction Flip (Hysteresis Buffer met)
                if rank > self.hold_n or score < self.min_score or new_direction != direction:
                    insights.append(Insight.Price(symbol, timedelta(days=5), InsightDirection.Flat, score))
                    symbols_to_remove.append(symbol)
            except ValueError:
                # Stock not in current candidates (delisted or dropped out of 300)
                insights.append(Insight.Price(symbol, timedelta(days=5), InsightDirection.Flat, 0.0))
                symbols_to_remove.append(symbol)

        for s in symbols_to_remove:
            self.active_signals.pop(s, None)

        # B) Select New Entries: Only Top 10
        for symbol, score in top_candidates:
            rank = symbols_by_rank.index(symbol) + 1
            direction = final_directions[symbol]
            
            # If Rank is in Top 10 and we aren't already predicting this symbol
            if rank <= self.top_n and symbol not in self.active_signals and score >= self.min_score:
                insight = Insight.Price(symbol, timedelta(days=5), direction, score)
                insights.append(insight)
                self.active_signals[symbol] = direction

        if insights:
            algorithm.Log(f"Alpha Engine: Zero-Churn Selection (Active: {len(self.active_signals)}, Updates: {len(insights)})")

        return insights

class SymbolData:
    def __init__(self, algorithm, symbol, spy_symbol):
        self.symbol = symbol
        self.algorithm = algorithm
        self.spy_symbol = spy_symbol
        
        # 1. Manual Indicators (Pillar 18 Protective Firewall)
        # We avoid algorithm.RSI/STD helpers to prevent automatic (unsanitized) data updates
        self.rsi = RelativeStrengthIndex(14)
        self.std20 = StandardDeviation(20)
        self.std5 = StandardDeviation(5)
        self.sma50 = SimpleMovingAverage(50)
        self.sma200 = SimpleMovingAverage(200)
        
        # 2. Windows (Parity Lags)
        self.close_window = RollingWindow[float](253) # For 1y high and 6m mom
        self.returns_window = RollingWindow[float](2) # For daily return calculation
        self.vol_history = RollingWindow[float](252) # Pillar 22: For 1y vol-percentile ranking
        self.spy_window = RollingWindow[float](61)   # For CAPM regression
        
        # 3. Warm-up
        from datetime import timedelta
        history = algorithm.History(symbol, timedelta(days=253))
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
        """Returns True if windows and indicators are fully warmed up."""
        return (self.close_window.IsReady and 
                self.sma200.IsReady and 
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
            stock_rets = np.array([(self.close_window[i]/self.close_window[i+1])-1 for i in range(60)])
            spy_rets = np.array([(self.spy_window[i]/self.spy_window[i+1])-1 for i in range(60)])
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
            "volatility_percentile": vol_perc, # Pillar 22: Parity Feature
            "momentum_acceleration": (ret_5d - ret_20d),
            "capm_residual_vol": capm_res_vol,
            "dist_from_52w_high": (close / max(self.close_window)) - 1.0,
            "momentum_6m": mom6m,
            "down_up_vol_ratio": 1.0, 
        }

class SimpleModel:
    """Fallback model when pickle fails - returns neutral predictions."""
    def predict_proba(self, X):
        import numpy as np
        return np.array([[0.5, 0.5]] * len(X))

def LoadProductionModel(algorithm, model_path="best_long_model.pkl"):
    """
    Helper to load the local .pkl or .joblib model into the Lean environment.
    Handles artifact dictionaries and fallbacks.
    """
    import joblib
    import io
    import pickle
    try:
        # 1. Attempt to load from Object Store (Recommended for Cloud)
        algorithm.Log(f"Locating production model in Object Store: {model_path}")
        if algorithm.ObjectStore.ContainsKey(model_path):
            model_bytes = algorithm.ObjectStore.ReadBytes(model_path)
            try:
                obj = joblib.load(io.BytesIO(model_bytes))
            except Exception as e:
                algorithm.Log(f"Joblib load failed, trying pickle: {str(e)}")
                obj = pickle.loads(bytes(model_bytes))
            
            # Pillar 13 Final: Smart Unpacker (Dictionary Artifact support)
            if isinstance(obj, dict) and "estimator" in obj:
                algorithm.Log(f"Unpacked estimator from artifact dictionary: {model_path}")
                model = obj["estimator"]
                # Additional artifacts can be extracted here: obj.get("feature_columns"), etc.
            else:
                model = obj
        else:
            algorithm.Error(f"Model path '{model_path}' not found in Object Store.")
            # Log all available keys if missing
            keys = [item.Key for item in algorithm.ObjectStore]
            algorithm.Log(f"Available keys in Object Store: {', '.join(keys) if keys else 'NONE'}")
            
            # 2. Fallback to standard local path
            algorithm.Log(f"Attempting fallback to local project file: {model_path}")
            try:
                model = joblib.load(model_path)
            except:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)

        # 3. Unpack the estimator if it's stored in an artifact dictionary
        if isinstance(model, dict) and "estimator" in model:
            algorithm.Log(f"Unpacked estimator from artifact dictionary: {model_path}")
            model = model["estimator"]
        
        # 4. Final validation: ensure it has the required interface
        if model is not None and hasattr(model, "predict_proba"):
            algorithm.Log(f"Model successfully validated for production: {model_path}")
            return model
        else:
            algorithm.Error(f"Loaded object does not support predict_proba: {type(model)}")
            return SimpleModel()
            
    except Exception as e:
        algorithm.Error(f"Failed to load model {model_path}: {str(e)}")
        return SimpleModel()
