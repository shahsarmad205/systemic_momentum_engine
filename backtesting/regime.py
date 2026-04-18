"""
Market Regime Agent
=====================
Downloads SPY + VIX data and classifies each trading day into one of:
    Bull, Bear, Cautious, Sideways, Crisis

Architecture: Gaussian HMM (industry-standard, AQR / Two Sigma / State Street)
----------------------------------------------------------------------
A Gaussian HMM learns five latent states from macro features and outputs
P(state | observations) — a probability vector — rather than hard threshold labels.
This eliminates the fragile "VIX >= 30 = Crisis" hard boundary that over-fires
during moderate stress periods and produces whipsaw regime flips.

Five macro features (daily, computed from local cache):
    1. SPY_1m_return     — 21-day rolling return (trend signal)
    2. VIX_level         — spot VIX (stress signal)
    3. VIX_5d_change     — 5-day VIX change (momentum of fear)
    4. SPY_30d_vol       — 30-day annualized realized volatility (risk signal)
    5. SPY_sma_gap       — (Close - SMA200) / SMA200 (position relative to trend)

Training:
    - Pre-fit window: data from (start_date - 10 years) → (start_date - 1 year),
      i.e., we never use data from the production window to fit the HMM.
    - Re-estimation: every 63 trading days (~1 quarter), the HMM is re-fit on
      an expanding window ending 252 days before the current date (1-year buffer
      prevents look-ahead on regime labels).

State labelling:
    After HMM fitting, states are semantically labeled by comparing their mean
    feature vectors:
        - Highest VIX + highest vol + negative SPY return  → "Crisis"
        - Negative SPY return + VIX ∈ [20, 30)            → "Bear"
        - Negative SPY return + VIX in mild range          → "Cautious"
        - VIX low + SPY near flat                          → "Sideways"
        - Positive SPY return + low VIX                    → "Bull"

    The MAP state (argmax of probability vector) is the regime label.
    The probability vector is stored in `regime_probs` for downstream
    use in continuous position sizing (no forced liquidations needed).

Fallback:
    If hmmlearn is unavailable or HMM fitting fails (< 250 training rows),
    the system falls back to the original threshold-based classifier with
    hysteresis, which is still significantly better than a single-threshold approach.

References:
    - Hamilton (1989): original Markov-Switching Regime model
    - Kritzman et al. (2012): turbulence index and regime detection
    - State Street Ung (2025): 5-state macro factor regime model
    - AQR Asness et al. (2020): regime-conditional momentum
    - Man Group (2024): momentum performance across 5 economic regimes
"""

import logging
import warnings
from collections import Counter
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Ordered regime labels for HMM state assignment
REGIME_LABELS = ["Bull", "Cautious", "Bear", "Sideways", "Crisis"]
N_STATES = 5


class MarketRegimeAgent:
    LOOKBACK_BUFFER = 600          # calendar days before start for MA warm-up
    VIX_CRISIS_THRESHOLD = 30.0
    VIX_HIGH_THRESHOLD = 20.0

    # HMM re-estimation cadence (trading days between quarterly refits)
    HMM_REFIT_INTERVAL = 63        # ~1 quarter
    # Minimum rows needed to attempt HMM fitting; below this, use threshold fallback
    HMM_MIN_ROWS = 250
    # Gap between fitting window end and production window start (look-ahead buffer)
    HMM_LABEL_BUFFER_DAYS = 252    # 1 year

    def __init__(self):
        self._hmm_model = None          # fitted GaussianHMM instance
        self._hmm_state_map = None      # {hmm_state_int: regime_label}
        self._hmm_fit_end_date = None   # last date used in fitting window
        # Probability output: date → {regime: probability}
        self._regime_probs: dict[pd.Timestamp, dict[str, float]] = {}

    # ------------------------------------------------------------------ #
    #  Public interface                                                    #
    # ------------------------------------------------------------------ #

    def detect_regimes(
        self,
        start_date: str,
        end_date: str,
        confirmation_days: int = 3,
    ) -> dict[pd.Timestamp, str]:
        """
        Return {date: regime_label} for every trading day in [start, end].

        Uses Gaussian HMM when hmmlearn is available and enough training data
        exists. Falls back to threshold rules + hysteresis otherwise.

        Side effect: populates `self._regime_probs` with per-day probability
        vectors that downstream sizing can use for continuous exposure scaling.
        """
        dl_start = pd.Timestamp(start_date) - pd.Timedelta(days=self.LOOKBACK_BUFFER + 3650)
        dl_end = pd.Timestamp(end_date) + pd.Timedelta(days=30)
        prod_start = pd.Timestamp(start_date)
        prod_end = pd.Timestamp(end_date) + pd.Timedelta(days=30)

        spy = self._download("SPY", dl_start, dl_end)
        vix = self._download_vix(dl_start, dl_end, spy)

        if spy.empty:
            logger.warning("MarketRegimeAgent: SPY download empty — defaulting all dates to Sideways")
            return {}

        features_df = self._build_features(spy, vix)

        # Attempt HMM-based classification
        try:
            result = self._hmm_classify(features_df, prod_start, prod_end)
            if result:
                logger.info(
                    "MarketRegimeAgent: HMM regime detection produced %d labels", len(result)
                )
                self._log_distribution(result)
                return result
        except Exception as exc:
            logger.warning(
                "HMM classification failed (%s) — falling back to threshold rules.", exc
            )

        # Fallback: threshold rules + hysteresis
        logger.info("MarketRegimeAgent: using threshold-based fallback regime detection.")
        return self._threshold_classify(spy, vix, features_df, prod_start, prod_end, confirmation_days)

    def get_regime_probs(self) -> dict[pd.Timestamp, dict[str, float]]:
        """
        Return {date: {regime: probability}} populated after detect_regimes().

        Use this for continuous position sizing:
            exposure_scalar = sum(P(k) * E(k) for k in regimes)
        where E(k) is the target exposure for regime k.
        """
        return self._regime_probs

    def detect_regime_scores(
        self,
        start_date: str,
        end_date: str,
    ) -> dict[pd.Timestamp, float]:
        """
        Return {date: risk_score} in [0.0, 1.0] for every trading day.

        0.0 = pure Bull (full exposure), 1.0 = pure Crisis (minimum exposure).

        When HMM probs are available, computes as a weighted combination of
        regime-specific risk scores. Otherwise falls back to VIX/SMA sigmoid.
        """
        # If we have HMM probs, derive score from probability-weighted regimes
        if self._regime_probs:
            risk_weights = {
                "Bull": 0.0,
                "Sideways": 0.25,
                "Cautious": 0.55,
                "Bear": 0.75,
                "Crisis": 1.0,
            }
            scores = {}
            for date, probs in self._regime_probs.items():
                scores[date] = float(
                    sum(probs.get(r, 0.0) * w for r, w in risk_weights.items())
                )
            return scores

        # Fallback: sigmoid-based scores from raw indicators
        dl_start = pd.Timestamp(start_date) - pd.Timedelta(days=self.LOOKBACK_BUFFER)
        dl_end = pd.Timestamp(end_date) + pd.Timedelta(days=30)

        spy = self._download("SPY", dl_start, dl_end)
        vix = self._download_vix(dl_start, dl_end, spy)

        if spy.empty:
            logger.warning("detect_regime_scores: SPY download empty — returning empty scores")
            return {}

        sma200 = spy["Close"].rolling(200).mean()
        vix_series = pd.Series(vix).reindex(spy.index).ffill().fillna(15.0)

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=30)

        scores: dict[pd.Timestamp, float] = {}
        for date in spy.index:
            if date < start_ts or date > end_ts:
                continue
            if pd.isna(sma200.get(date)):
                scores[date] = 0.5
                continue

            close_val = float(spy.loc[date, "Close"])
            sma200_val = float(sma200[date])
            vix_val = float(vix_series.get(date, 15.0))

            vix_score = float(1.0 / (1.0 + np.exp(-(vix_val - 25.0) / 4.0)))
            sma_gap = (sma200_val - close_val) / sma200_val
            sma_score = float(1.0 / (1.0 + np.exp(-sma_gap / 0.025)))
            scores[date] = float(np.clip(max(vix_score, sma_score), 0.0, 1.0))

        logger.info(
            "detect_regime_scores: computed %d daily scores (range %.3f–%.3f)",
            len(scores),
            min(scores.values()) if scores else float("nan"),
            max(scores.values()) if scores else float("nan"),
        )
        return scores

    # ------------------------------------------------------------------ #
    #  HMM machinery                                                       #
    # ------------------------------------------------------------------ #

    def _build_features(
        self, spy: pd.DataFrame, vix: dict
    ) -> pd.DataFrame:
        """
        Construct daily feature matrix from SPY + VIX.

        Features:
            spy_1m_return  — 21-day rolling SPY return
            vix_level      — VIX closing level
            vix_5d_change  — 5-day change in VIX
            spy_30d_vol    — 30-day annualised SPY realised volatility (×100)
            spy_sma_gap    — (Close - SMA200) / SMA200
        """
        df = pd.DataFrame(index=spy.index)
        close = spy["Close"]

        df["spy_1m_return"] = close.pct_change(21)
        df["spy_30d_vol"] = close.pct_change().rolling(30).std() * np.sqrt(252) * 100

        sma200 = close.rolling(200).mean()
        df["spy_sma_gap"] = (close - sma200) / sma200

        vix_series = pd.Series(vix).reindex(spy.index).ffill().fillna(15.0)
        df["vix_level"] = vix_series
        df["vix_5d_change"] = vix_series.diff(5)

        return df.dropna()

    def _hmm_classify(
        self,
        features_df: pd.DataFrame,
        prod_start: pd.Timestamp,
        prod_end: pd.Timestamp,
    ) -> dict[pd.Timestamp, str]:
        """
        Fit a Gaussian HMM on pre-production data, then label production dates.

        The model is re-fit every HMM_REFIT_INTERVAL trading days on an expanding
        window, but the fitting window always ends at least HMM_LABEL_BUFFER_DAYS
        before the current production date to prevent look-ahead.
        """
        try:
            from hmmlearn.hmm import GaussianHMM
        except ImportError:
            raise ImportError("hmmlearn not installed — run: pip install hmmlearn")

        prod_dates = features_df.index[
            (features_df.index >= prod_start) & (features_df.index <= prod_end)
        ]
        if len(prod_dates) == 0:
            return {}

        # Use all pre-production data for initial fit (no look-ahead)
        train_end = prod_start - pd.Timedelta(days=self.HMM_LABEL_BUFFER_DAYS)
        train_mask = features_df.index < train_end

        if train_mask.sum() < self.HMM_MIN_ROWS:
            raise ValueError(
                f"Only {train_mask.sum()} pre-production rows — below minimum {self.HMM_MIN_ROWS} for HMM"
            )

        feature_cols = ["spy_1m_return", "vix_level", "vix_5d_change", "spy_30d_vol", "spy_sma_gap"]

        results: dict[pd.Timestamp, str] = {}
        self._regime_probs = {}

        # Fit the initial model
        X_train = features_df.loc[train_mask, feature_cols].values
        model, state_map = self._fit_hmm(GaussianHMM, X_train)
        self._hmm_model = model
        self._hmm_state_map = state_map
        scaler = getattr(model, "_feature_scaler", None)
        last_refit_idx = 0

        for i, date in enumerate(prod_dates):
            # Periodic re-estimation on expanding window
            if i > 0 and i % self.HMM_REFIT_INTERVAL == 0:
                refit_end = date - pd.Timedelta(days=self.HMM_LABEL_BUFFER_DAYS)
                refit_mask = features_df.index < refit_end
                if refit_mask.sum() >= self.HMM_MIN_ROWS:
                    try:
                        X_refit = features_df.loc[refit_mask, feature_cols].values
                        new_model, new_state_map = self._fit_hmm(GaussianHMM, X_refit)
                        model = new_model
                        state_map = new_state_map
                        scaler = getattr(model, "_feature_scaler", None)
                        self._hmm_model = model
                        self._hmm_state_map = state_map
                        logger.info(
                            "HMM re-fitted at %s on %d rows", date.date(), refit_mask.sum()
                        )
                    except Exception as exc:
                        logger.warning("HMM re-fit failed at %s (%s) — keeping previous model.", date.date(), exc)

            if date not in features_df.index:
                # Carry forward last known probs
                if self._regime_probs:
                    last_probs = list(self._regime_probs.values())[-1]
                    self._regime_probs[date] = last_probs
                    results[date] = max(last_probs, key=last_probs.get)
                else:
                    self._regime_probs[date] = {r: 1.0 / N_STATES for r in REGIME_LABELS}
                    results[date] = "Sideways"
                continue

            try:
                # Use a 20-day context window ending at today for posterior inference.
                # hmmlearn's predict_proba uses forward-backward algorithm on the full
                # sequence, so a short window gives a good approximation of P(state | obs).
                ctx_start = max(0, features_df.index.get_loc(date) - 20)
                ctx_end = features_df.index.get_loc(date) + 1
                X_ctx = features_df.iloc[ctx_start:ctx_end][feature_cols].values
                if scaler is not None:
                    X_ctx = np.clip(scaler.transform(X_ctx), -5.0, 5.0)
                if len(X_ctx) > 0:
                    posteriors = model.predict_proba(X_ctx)  # (T, n_states)
                    proba_vec = posteriors[-1]  # last row = today's posterior
                else:
                    proba_vec = np.ones(N_STATES) / N_STATES
            except Exception:
                proba_vec = np.ones(N_STATES) / N_STATES

            # Map HMM state indices → regime labels
            probs_by_regime = {
                state_map[s]: float(proba_vec[s]) for s in range(N_STATES)
            }
            self._regime_probs[date] = probs_by_regime

            # MAP regime: highest-probability state
            map_regime = max(probs_by_regime, key=probs_by_regime.get)
            results[date] = map_regime

        return results

    @staticmethod
    def _fit_hmm(GaussianHMM_cls, X_train: np.ndarray):
        """
        Fit a Gaussian HMM and return (model, state_map, scaler).

        Features span very different scales (VIX 10-80 vs returns ±0.3), so we
        standardize with StandardScaler before fitting. The scaler is stored on
        the model object so that predict_proba calls use the same transform.

        state_map: {hmm_state_int: regime_label}

        States are matched to regimes by ranking their mean feature vectors:
            - Crisis:   highest VIX mean AND most negative SPY return
            - Bear:     2nd highest VIX, negative SPY return
            - Cautious: moderate VIX, mild negative SPY return
            - Sideways: low VIX, near-zero SPY return
            - Bull:     lowest VIX, highest positive SPY return
        """
        from sklearn.preprocessing import StandardScaler

        # Drop any rows with Inf/NaN that survived dropna() (e.g. pct_change on zero price)
        finite_mask = np.isfinite(X_train).all(axis=1)
        X_train = X_train[finite_mask]

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)
        # Clip to ±5 sigma so K-means++ distance matmul cannot overflow
        X_scaled = np.clip(X_scaled, -5.0, 5.0)

        model = GaussianHMM_cls(
            n_components=N_STATES,
            covariance_type="diag",   # diag more numerically stable than full on 5 features
            n_iter=200,
            tol=1e-4,
            random_state=42,
            init_params="kmeans",
        )
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*overflow.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*divide by zero.*")
            warnings.filterwarnings("ignore", category=RuntimeWarning, message=".*invalid value.*")
            model.fit(X_scaled)
        # Attach scaler so _hmm_classify can retrieve it without re-fitting
        model._feature_scaler = scaler

        means = model.means_  # (n_states, n_features)
        # Feature column order: spy_1m_return, vix_level, vix_5d_change, spy_30d_vol, spy_sma_gap
        # Indices:                   0              1           2             3             4
        vix_means = means[:, 1]     # VIX level per state
        ret_means = means[:, 0]     # SPY 1m return per state

        # Sort states from lowest to highest VIX
        vix_rank = np.argsort(vix_means)  # vix_rank[0] = lowest-VIX state

        # Assign regimes based on VIX rank and return sign
        # Start with VIX-ordered assignment, then refine by return
        n = N_STATES
        regime_assignment = {}

        # Crisis = highest VIX state
        regime_assignment[vix_rank[-1]] = "Crisis"
        # Bull = lowest VIX + positive return
        regime_assignment[vix_rank[0]] = "Bull"

        # Middle three states: rank by return to distinguish Bear/Cautious/Sideways
        middle_states = [vix_rank[i] for i in range(1, n - 1)]
        middle_ret = [(s, ret_means[s]) for s in middle_states]
        middle_sorted = sorted(middle_ret, key=lambda x: x[1])  # ascending return

        # Most negative return among middle → Bear
        regime_assignment[middle_sorted[0][0]] = "Bear"
        # Middle return → Cautious
        regime_assignment[middle_sorted[1][0]] = "Cautious"
        # Most positive return among middle → Sideways (recovering)
        regime_assignment[middle_sorted[2][0]] = "Sideways"

        # Log the assignment for transparency
        for state_idx, label in regime_assignment.items():
            logger.debug(
                "HMM state %d → %s (VIX mean=%.1f, ret mean=%.3f)",
                state_idx, label, vix_means[state_idx], ret_means[state_idx],
            )

        return model, regime_assignment

    # ------------------------------------------------------------------ #
    #  Threshold-based fallback (original logic)                          #
    # ------------------------------------------------------------------ #

    def _threshold_classify(
        self,
        spy: pd.DataFrame,
        vix: dict,
        features_df: pd.DataFrame,
        prod_start: pd.Timestamp,
        prod_end: pd.Timestamp,
        confirmation_days: int,
    ) -> dict[pd.Timestamp, str]:
        """
        Original threshold-based regime classification with hysteresis.
        Used as fallback when HMM is unavailable or fails.
        """
        sma200 = spy["Close"].rolling(200).mean()
        sma50  = spy["Close"].rolling(50).mean()
        vix_series = pd.Series(vix).reindex(spy.index).ffill().fillna(15.0)
        vix_slope  = vix_series.diff(5).fillna(0.0)

        # Pass 1: raw tentative regimes
        raw_regimes: dict[pd.Timestamp, str] = {}
        for date in spy.index:
            if date < prod_start or date > prod_end:
                continue
            if pd.isna(sma200.get(date)) or pd.isna(sma50.get(date)):
                raw_regimes[date] = "Sideways"
                continue

            close   = float(spy.loc[date, "Close"])
            vix_val = float(vix.get(date, 15.0))

            if vix_val >= self.VIX_CRISIS_THRESHOLD:
                raw_regimes[date] = "Crisis"
            elif (
                vix_val >= self.VIX_HIGH_THRESHOLD
                and (
                    close < float(sma50.get(date, close + 1))
                    or float(vix_slope.get(date, 0.0)) > 2.0
                )
            ):
                raw_regimes[date] = "Cautious"
            elif close > sma200[date] and sma50[date] > sma200[date]:
                raw_regimes[date] = "Bull"
            elif close < sma200[date] and sma50[date] < sma200[date]:
                raw_regimes[date] = "Bear"
            else:
                raw_regimes[date] = "Sideways"

            # Populate simple uniform probs for fallback (no HMM)
            regime = raw_regimes[date]
            probs = {r: 0.05 for r in REGIME_LABELS}
            probs[regime] = 0.80
            total = sum(probs.values())
            self._regime_probs[date] = {r: v / total for r, v in probs.items()}

        if confirmation_days <= 1:
            return raw_regimes

        # Pass 2: hysteresis
        confirmed_regimes: dict[pd.Timestamp, str] = {}
        sorted_dates = sorted(raw_regimes.keys())
        current_official = "Sideways"
        pending_regime = "Sideways"
        pending_count = 0

        for date in sorted_dates:
            raw = raw_regimes[date]

            if raw == "Crisis":
                current_official = "Crisis"
                pending_regime = "Crisis"
                pending_count = confirmation_days
                confirmed_regimes[date] = "Crisis"
                continue

            if current_official == "Crisis" and raw != "Crisis":
                if raw != pending_regime:
                    pending_regime = raw
                    pending_count = 1
                else:
                    pending_count += 1
                if pending_count >= confirmation_days:
                    current_official = pending_regime
                confirmed_regimes[date] = current_official
                continue

            if raw == pending_regime:
                pending_count += 1
            else:
                pending_regime = raw
                pending_count = 1

            if pending_count >= confirmation_days:
                current_official = pending_regime
            confirmed_regimes[date] = current_official

        n_switches = sum(
            1 for i in range(1, len(sorted_dates))
            if confirmed_regimes.get(sorted_dates[i]) != confirmed_regimes.get(sorted_dates[i - 1])
        )
        logger.info(
            "Threshold regime detection: %d switches (confirmation=%d days)",
            n_switches, confirmation_days,
        )
        return confirmed_regimes

    # ------------------------------------------------------------------ #
    #  Helpers                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _log_distribution(regimes: dict) -> None:
        dist = Counter(regimes.values())
        total = len(regimes)
        if total == 0:
            return
        dist_str = " | ".join(
            f"{r}: {dist.get(r, 0)} ({dist.get(r, 0)/total:.1%})"
            for r in ["Bull", "Cautious", "Bear", "Sideways", "Crisis"]
        )
        logger.info("Regime distribution: %s", dist_str)
        crisis_pct = dist.get("Crisis", 0) / total
        if crisis_pct > 0.40:
            logger.warning(
                "⚠ %.1f%% of days classified as CRISIS — this is unusually high. "
                "HMM state labelling may need review if VIX data is incomplete.",
                crisis_pct * 100,
            )

    @staticmethod
    def _load_from_cache(ticker: str, cache_ticker: str, start, end) -> pd.DataFrame:
        """
        Try to load OHLCV from local parquet cache before hitting yfinance.
        """
        from pathlib import Path

        base = Path(__file__).parent.parent
        cache_path = base / "data" / "cache" / f"{cache_ticker}.parquet"

        if not cache_path.exists():
            cache_path = Path("data") / "cache" / f"{cache_ticker}.parquet"

        if cache_path.exists():
            try:
                df = pd.read_parquet(cache_path)
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                df = df.loc[pd.Timestamp(start):pd.Timestamp(end)]
                keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
                df = df[keep].dropna(subset=["Close"])
                if not df.empty:
                    logger.info(
                        "regime._load_from_cache: loaded %s from cache (%d rows, %s – %s)",
                        cache_ticker, len(df), df.index[0].date(), df.index[-1].date(),
                    )
                    return df
                logger.warning("Cache file for %s exists but is empty after filtering.", cache_ticker)
            except Exception as exc:
                logger.warning("Failed to load %s from cache (%s). Falling back to yfinance.", cache_ticker, exc)
        return pd.DataFrame()

    @staticmethod
    def _download(ticker: str, start, end) -> pd.DataFrame:
        # 1. Try local parquet cache first
        cached = MarketRegimeAgent._load_from_cache(ticker, ticker, start, end)
        if not cached.empty:
            return cached

        # 2. Fall back to yfinance
        import yfinance as yf

        logger.info("regime._download: %s not in cache — downloading from yfinance.", ticker)
        try:
            raw = yf.download(ticker, start=start, end=end, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            keep = ["Open", "High", "Low", "Close", "Volume"]
            result = raw[keep].dropna() if not raw.empty else pd.DataFrame()
            if result.empty:
                logger.warning(
                    "regime._download: yfinance returned empty DataFrame for %s. "
                    "Regime will default to Sideways for all dates.",
                    ticker,
                )
            return result
        except Exception as exc:
            logger.warning("regime._download: yfinance failed for %s (%s).", ticker, exc)
            return pd.DataFrame()

    @staticmethod
    def _download_vix(start, end, spy_fallback: pd.DataFrame) -> dict:
        """Try local _VIX cache → yfinance ^VIX → SPY vol proxy (in that order)."""
        cached = MarketRegimeAgent._load_from_cache("^VIX", "_VIX", start, end)
        if not cached.empty and "Close" in cached.columns:
            series = cached["Close"]
            if hasattr(series, "squeeze"):
                series = series.squeeze()
            vix_dict = series.to_dict()
            if vix_dict:
                logger.info("VIX loaded from local cache: %d days", len(vix_dict))
                return vix_dict

        try:
            import yfinance as yf
            raw = yf.download("^VIX", start=start, end=end, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            raw.columns = [c.capitalize() if isinstance(c, str) else c for c in raw.columns]
            if not raw.empty and "Close" in raw.columns:
                series = raw["Close"]
                if hasattr(series, "squeeze"):
                    series = series.squeeze()
                vix_dict = series.to_dict()
                if vix_dict:
                    logger.info("VIX downloaded from yfinance: %d days", len(vix_dict))
                    return vix_dict
        except Exception as exc:
            logger.warning("VIX yfinance download failed (%s).", exc)

        logger.warning(
            "⚠ Using SPY realized-vol proxy for VIX — regime classification may be less accurate."
        )
        if spy_fallback.empty:
            return {}

        returns = spy_fallback["Close"].pct_change()
        vol_proxy = returns.rolling(20).std() * np.sqrt(252) * 100 * 1.2
        return vol_proxy.to_dict()
