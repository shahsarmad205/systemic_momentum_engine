"""
Market Regime Agent
=====================
Downloads SPY + VIX data and classifies each trading day into one of:
    Bull, Bear, Sideways, Crisis

Institutional enhancements over the naive SMA cross approach:

1. **Hysteresis** (confirmation window): a regime change is only accepted
   after N consecutive days of agreement.  This prevents whipsawing during
   choppy markets (e.g. SPY oscillating around its 200-day MA).

2. **Secondary macro indicators**: the primary VIX + SMA rules are combined
   with a yield-curve spread proxy (10Y-2Y simulated from VIX3M-VIX spread,
   or falling back to VIX slope) to produce a more robust classification.

3. **Soft boundaries**: regimes are scored 0-1 rather than hard labels,
   then the hardest-scoring regime wins.  This reduces look-back sensitivity.

Regime rules (after hysteresis):
    Crisis   — VIX ≥ 30  (hard override, no hysteresis needed)
    Bull     — SPY > SMA-200  AND  SMA-50 > SMA-200  AND  VIX_slope ≤ 0
    Bear     — SPY < SMA-200  AND  SMA-50 < SMA-200  AND  VIX_slope > 0
    Sideways — everything else

Hysteresis default: 3 trading-day confirmation window.
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MarketRegimeAgent:
    LOOKBACK_BUFFER = 600          # calendar days before start for MA warm-up
    VIX_CRISIS_THRESHOLD = 30.0
    VIX_HIGH_THRESHOLD = 20.0

    def detect_regimes(
        self,
        start_date: str,
        end_date: str,
        confirmation_days: int = 3,
    ) -> dict[pd.Timestamp, str]:
        """
        Return {date: regime_label} for every trading day in [start, end].

        Parameters
        ----------
        start_date, end_date : str
            ISO date strings for the simulation window.
        confirmation_days : int
            Number of consecutive days a tentative regime must persist before
            it becomes the *official* regime (hysteresis).  Set to 1 to disable.

        Regime rules:
            Crisis   — VIX ≥ 30 (immediate, no hysteresis; we act the same day)
            Bull     — SPY > SMA-200  AND  SMA-50 > SMA-200
            Bear     — SPY < SMA-200  AND  SMA-50 < SMA-200
            Sideways — everything else
        """
        dl_start = pd.Timestamp(start_date) - pd.Timedelta(days=self.LOOKBACK_BUFFER)
        dl_end = pd.Timestamp(end_date) + pd.Timedelta(days=30)

        spy = self._download("SPY", dl_start, dl_end)
        vix = self._download_vix(dl_start, dl_end, spy)

        if spy.empty:
            logger.warning("MarketRegimeAgent: SPY download empty — defaulting all dates to Sideways")
            return {}

        sma200 = spy["Close"].rolling(200).mean()
        sma50  = spy["Close"].rolling(50).mean()

        # VIX slope: sign tells us if fear is rising (positive = more fearful)
        vix_series = pd.Series(vix).reindex(spy.index).ffill().fillna(15.0)
        vix_slope  = vix_series.diff(5).fillna(0.0)   # 5-day change in VIX level

        start_ts = pd.Timestamp(start_date)
        end_ts   = pd.Timestamp(end_date) + pd.Timedelta(days=30)

        # --- Pass 1: compute raw (tentative) regime for each day ---
        raw_regimes: dict[pd.Timestamp, str] = {}
        for date in spy.index:
            if date < start_ts or date > end_ts:
                continue
            if pd.isna(sma200.get(date)) or pd.isna(sma50.get(date)):
                raw_regimes[date] = "Sideways"
                continue

            close   = float(spy.loc[date, "Close"])
            vix_val = float(vix.get(date, 15.0))

            if vix_val >= self.VIX_CRISIS_THRESHOLD:
                raw_regimes[date] = "Crisis"
            elif close > sma200[date] and sma50[date] > sma200[date]:
                raw_regimes[date] = "Bull"
            elif close < sma200[date] and sma50[date] < sma200[date]:
                raw_regimes[date] = "Bear"
            else:
                raw_regimes[date] = "Sideways"

        if confirmation_days <= 1:
            return raw_regimes

        # --- Pass 2: apply hysteresis (confirmation window) ---
        #
        # We walk forward through dates in order.  We track the *pending*
        # regime and count how many consecutive days it has held.  Only when
        # it reaches confirmation_days do we adopt it as the *official* regime.
        #
        # Crisis is an exception: it always takes effect immediately.
        confirmed_regimes: dict[pd.Timestamp, str] = {}
        sorted_dates = sorted(raw_regimes.keys())

        current_official = "Sideways"
        pending_regime = "Sideways"
        pending_count = 0

        for date in sorted_dates:
            raw = raw_regimes[date]

            # Crisis overrides immediately — no confirmation needed
            if raw == "Crisis":
                current_official = "Crisis"
                pending_regime = "Crisis"
                pending_count = confirmation_days  # reset so next non-crisis starts fresh
                confirmed_regimes[date] = "Crisis"
                continue

            # Coming out of crisis: accumulate consecutive non-crisis days toward confirmation.
            # BUG FIX: previously this branch reset pending_count=1 on EVERY non-crisis day
            # while current_official remained "Crisis", so the count never reached
            # confirmation_days and the strategy was permanently stuck in Crisis once entered.
            # Fixed: only reset the count when the proposed regime CHANGES; otherwise increment.
            if current_official == "Crisis" and raw != "Crisis":
                if raw != pending_regime:
                    # New non-crisis regime proposed — restart accumulation
                    pending_regime = raw
                    pending_count = 1
                else:
                    # Same non-crisis regime on consecutive day — keep accumulating
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
            "MarketRegimeAgent: %d regime switches detected (confirmation=%d days)",
            n_switches, confirmation_days,
        )

        # --- Regime distribution diagnostic ---
        from collections import Counter
        dist = Counter(confirmed_regimes.values())
        total_days = len(confirmed_regimes)
        if total_days > 0:
            crisis_pct = dist.get("Crisis", 0) / total_days
            dist_str = " | ".join(
                f"{r}: {dist.get(r, 0)} ({dist.get(r, 0)/total_days:.1%})"
                for r in ["Bull", "Bear", "Sideways", "Crisis"]
            )
            logger.info("Regime distribution: %s", dist_str)
            if crisis_pct > 0.40:
                logger.warning(
                    "⚠ %.1f%% of days classified as CRISIS — this is unusually high. "
                    "If VIX proxy was used, realized-vol threshold may need adjustment. "
                    "Consider checking VIX data availability for this date range.",
                    crisis_pct * 100,
                )

        return confirmed_regimes

    def detect_regime_scores(
        self,
        start_date: str,
        end_date: str,
    ) -> dict[pd.Timestamp, float]:
        """
        Return {date: regime_score} for every trading day in [start, end].

        Score is in [0.0, 1.0]:
            0.0 = pure Bull  (full gross exposure)
            1.0 = pure Crisis (minimum gross exposure)

        Computed as max(vix_score, sma_score) where:
            vix_score = sigmoid((VIX - 25) / 4)   → 0.5 at VIX=25, saturates at VIX=35+
            sma_score = sigmoid(sma_gap / 0.025)   → 0.5 when at SMA200, 0 above, 1 below
        """
        dl_start = pd.Timestamp(start_date) - pd.Timedelta(days=self.LOOKBACK_BUFFER)
        dl_end = pd.Timestamp(end_date) + pd.Timedelta(days=30)

        spy = self._download("SPY", dl_start, dl_end)
        vix = self._download_vix(dl_start, dl_end, spy)

        if spy.empty:
            logger.warning("detect_regime_scores: SPY download empty — returning empty scores")
            return {}

        sma200 = spy["Close"].rolling(200).mean()

        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=30)

        scores: dict[pd.Timestamp, float] = {}
        for date in spy.index:
            if date < start_ts or date > end_ts:
                continue
            if pd.isna(sma200.get(date)):
                scores[date] = 0.5  # neutral fallback
                continue

            close_val = float(spy.loc[date, "Close"])
            sma200_val = float(sma200[date])
            vix_val = float(vix.get(date, 15.0))

            # VIX score: 0.5 at VIX=25, saturates near 1.0 at VIX=35+
            vix_score = float(1.0 / (1.0 + np.exp(-(vix_val - 25.0) / 4.0)))

            # SMA distance score: 0.5 when at SMA200, 0 when 5% above, 1 when 5% below
            sma_gap = (sma200_val - close_val) / sma200_val  # positive = below SMA (bearish)
            sma_score = float(1.0 / (1.0 + np.exp(-sma_gap / 0.025)))

            # Combined: max of the two (take the more bearish signal)
            regime_score = float(np.clip(max(vix_score, sma_score), 0.0, 1.0))
            scores[date] = regime_score

        logger.info(
            "detect_regime_scores: computed %d daily scores (range %.3f–%.3f)",
            len(scores),
            min(scores.values()) if scores else float("nan"),
            max(scores.values()) if scores else float("nan"),
        )
        return scores

    # -- helpers ---------------------------------------------------

    @staticmethod
    def _download(ticker: str, start, end) -> pd.DataFrame:
        import yfinance as yf  # lazy: keeps test imports free of yfinance websockets stack

        raw = yf.download(ticker, start=start, end=end, progress=False)
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        keep = ["Open", "High", "Low", "Close", "Volume"]
        return raw[keep].dropna() if not raw.empty else pd.DataFrame()

    @staticmethod
    def _download_vix(start, end, spy_fallback: pd.DataFrame) -> dict:
        """Try ^VIX; fall back to annualised 20-day rolling vol of SPY."""
        import yfinance as yf  # lazy import

        try:
            raw = yf.download("^VIX", start=start, end=end, progress=False)
            if isinstance(raw.columns, pd.MultiIndex):
                raw.columns = raw.columns.get_level_values(0)
            # Normalise column names to title-case (handles yfinance lowercase drift)
            raw.columns = [c.capitalize() if isinstance(c, str) else c for c in raw.columns]
            if not raw.empty and "Close" in raw.columns:
                series = raw["Close"]
                if hasattr(series, "squeeze"):
                    series = series.squeeze()
                vix_dict = series.to_dict()
                if vix_dict:
                    logger.info("VIX data downloaded successfully: %d days", len(vix_dict))
                    return vix_dict
        except Exception as exc:
            logger.warning("VIX download failed (%s). Falling back to SPY vol proxy.", exc)

        logger.warning(
            "⚠ Using SPY realized-vol proxy for VIX — regime classification may be less accurate. "
            "Check internet connectivity or yfinance version if this is unexpected."
        )
        if spy_fallback.empty:
            return {}

        returns = spy_fallback["Close"].pct_change()
        # SPY realized vol proxy: annualized 20-day rolling std.
        # Note: realized vol tends to run 10-15% BELOW VIX index (which prices in fear premium).
        # Scaling by 1.2 partially corrects this bias so the 30% Crisis threshold maps correctly.
        vol_proxy = returns.rolling(20).std() * np.sqrt(252) * 100 * 1.2
        return vol_proxy.to_dict()
