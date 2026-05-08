"""Adaptive portfolio-control coefficients for executable validation.

This module calibrates the scalar coefficients already present in the
mean-variance + turnover objective:

    maximize  w'alpha - lambda_risk * w'Sigma w - gamma_turnover * ||w-w_prev||^2

It does not change the optimizer objective.  It only estimates date-level
``lambda_risk`` and ``gamma_turnover`` from trailing, point-in-time evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AdaptivePortfolioControlConfig:
    """Configuration for causal portfolio-control calibration."""

    enabled: bool = False
    lookback_days: int = 252
    min_history_days: int = 60
    ema_span: int = 20
    target_volatility: float = 0.15
    horizon_days: int = 5
    lambda_floor_factor: float = 0.50
    lambda_ceil_factor: float = 4.00
    gamma_floor_factor: float = 0.125
    gamma_ceil_factor: float = 4.00
    min_expected_alpha: float = 1e-4


@dataclass(frozen=True)
class PortfolioControlSnapshot:
    """Date-level adaptive coefficients plus explanatory state."""

    date: pd.Timestamp
    lambda_risk: float
    gamma_turnover: float
    expected_alpha: float
    expected_cost: float
    realized_volatility: float
    signal_ic: float
    support_days: int
    status: str


class AdaptivePortfolioController:
    """Stateful EMA smoother for adaptive risk and turnover coefficients."""

    def __init__(
        self,
        cfg: AdaptivePortfolioControlConfig,
        *,
        base_lambda: float,
        base_gamma: float,
        cost_config: Any,
    ) -> None:
        self.cfg = cfg
        self.base_lambda = float(base_lambda)
        self.base_gamma = float(base_gamma)
        self.cost_config = cost_config
        self._ema_alpha = 2.0 / (max(1.0, float(cfg.ema_span)) + 1.0)
        self._lambda_ema: float | None = None
        self._gamma_ema: float | None = None
        self.snapshots: list[PortfolioControlSnapshot] = []

    def snapshot_for_date(
        self,
        *,
        date: pd.Timestamp,
        full_df: pd.DataFrame,
        day_df: pd.DataFrame,
        market_state: Any | None,
    ) -> PortfolioControlSnapshot:
        """Return causal adaptive coefficients for ``date``.

        Only observations with ``history.date < date`` are used for signal and
        volatility estimates.  Current market-state inputs are limited to
        trailing/known liquidity and volatility primitives.
        """
        dt = pd.Timestamp(date)
        if not self.cfg.enabled:
            snap = self._base_snapshot(dt, status="disabled")
            self.snapshots.append(snap)
            return snap

        history = self._history(full_df, dt)
        support_days = int(history["date"].nunique()) if "date" in history.columns and not history.empty else 0
        if support_days < int(self.cfg.min_history_days):
            snap = self._base_snapshot(dt, status="insufficient_history", support_days=support_days)
            self.snapshots.append(snap)
            return snap

        signal_ic = self._trailing_ic(history)
        realized_vol = self._realized_volatility(history, market_state)
        expected_alpha = self._expected_alpha(history, signal_ic)
        expected_cost = self._expected_cost(day_df, market_state)

        lambda_raw = self._lambda_from_vol(realized_vol)
        gamma_raw = self._gamma_from_cost_alpha(expected_cost, expected_alpha)
        lambda_smoothed = self._smooth("lambda", lambda_raw)
        gamma_smoothed = self._smooth("gamma", gamma_raw)

        snap = PortfolioControlSnapshot(
            date=dt,
            lambda_risk=lambda_smoothed,
            gamma_turnover=gamma_smoothed,
            expected_alpha=expected_alpha,
            expected_cost=expected_cost,
            realized_volatility=realized_vol,
            signal_ic=signal_ic,
            support_days=support_days,
            status="adaptive",
        )
        self.snapshots.append(snap)
        return snap

    def snapshots_frame(self) -> pd.DataFrame:
        if not self.snapshots:
            return pd.DataFrame(
                columns=[
                    "date",
                    "lambda_risk",
                    "gamma_turnover",
                    "expected_alpha",
                    "expected_cost",
                    "realized_volatility",
                    "signal_ic",
                    "support_days",
                    "status",
                ]
            )
        return pd.DataFrame([s.__dict__ for s in self.snapshots])

    def _base_snapshot(
        self,
        dt: pd.Timestamp,
        *,
        status: str,
        support_days: int = 0,
    ) -> PortfolioControlSnapshot:
        return PortfolioControlSnapshot(
            date=dt,
            lambda_risk=self.base_lambda,
            gamma_turnover=self.base_gamma,
            expected_alpha=float("nan"),
            expected_cost=float("nan"),
            realized_volatility=float("nan"),
            signal_ic=float("nan"),
            support_days=int(support_days),
            status=status,
        )

    def _history(self, full_df: pd.DataFrame, dt: pd.Timestamp) -> pd.DataFrame:
        if full_df is None or full_df.empty or "date" not in full_df.columns:
            return pd.DataFrame()
        frame = full_df.copy()
        frame["date"] = pd.to_datetime(frame["date"], errors="coerce")
        revealed_cutoff = dt - pd.tseries.offsets.BDay(max(1, int(self.cfg.horizon_days)))
        start = pd.Timestamp(revealed_cutoff) - pd.Timedelta(days=int(self.cfg.lookback_days) * 2)
        hist = frame[(frame["date"] <= pd.Timestamp(revealed_cutoff)) & (frame["date"] >= start)].copy()
        if hist.empty:
            return hist
        dates = sorted(hist["date"].dropna().unique())[-int(self.cfg.lookback_days):]
        return hist[hist["date"].isin(dates)].copy()

    def _target_column(self, frame: pd.DataFrame) -> str | None:
        for col in ("next_return", "target_return", "daily_return"):
            if col in frame.columns:
                return col
        return None

    def _trailing_ic(self, history: pd.DataFrame) -> float:
        target_col = self._target_column(history)
        if target_col is None or "score" not in history.columns:
            return 0.0
        daily_ic: list[float] = []
        for _, group in history.groupby("date", sort=True):
            x = pd.to_numeric(group["score"], errors="coerce")
            y = pd.to_numeric(group[target_col], errors="coerce")
            valid = x.notna() & y.notna()
            if int(valid.sum()) < 5 or x.loc[valid].nunique() < 2 or y.loc[valid].nunique() < 2:
                continue
            corr = x.loc[valid].corr(y.loc[valid], method="spearman")
            if np.isfinite(corr):
                daily_ic.append(float(corr))
        return float(np.nanmean(daily_ic)) if daily_ic else 0.0

    def _realized_volatility(self, history: pd.DataFrame, market_state: Any | None) -> float:
        target_col = self._target_column(history)
        if target_col is not None and "date" in history.columns:
            daily = (
                history.assign(_ret=pd.to_numeric(history[target_col], errors="coerce"))
                .groupby("date")["_ret"]
                .mean()
                .dropna()
            )
            if len(daily) >= 20:
                vol = float(daily.std(ddof=1) * np.sqrt(252.0))
                if np.isfinite(vol) and vol > 1e-8:
                    return vol
        if market_state is not None and getattr(market_state, "daily_vol", None) is not None:
            arr = np.asarray(market_state.daily_vol, dtype=float)
            arr = arr[np.isfinite(arr) & (arr > 0.0)]
            if arr.size:
                return float(np.median(arr) * np.sqrt(252.0))
        return float(self.cfg.target_volatility)

    def _expected_alpha(self, history: pd.DataFrame, signal_ic: float) -> float:
        target_col = self._target_column(history)
        if target_col is None or "date" not in history.columns:
            return float(self.cfg.min_expected_alpha)
        cs_std = (
            history.assign(_ret=pd.to_numeric(history[target_col], errors="coerce"))
            .groupby("date")["_ret"]
            .std(ddof=1)
            .dropna()
        )
        sigma_cs = float(cs_std.median()) if not cs_std.empty else float("nan")
        if not np.isfinite(sigma_cs) or sigma_cs <= 0.0:
            sigma_cs = float(self.cfg.min_expected_alpha)
        alpha = abs(float(signal_ic)) * sigma_cs
        return float(max(alpha, float(self.cfg.min_expected_alpha)))

    def _expected_cost(self, day_df: pd.DataFrame, market_state: Any | None) -> float:
        one_way = (
            float(getattr(self.cost_config, "commission_bps", 1.0))
            + 0.5 * float(getattr(self.cost_config, "spread_bps", 1.0))
        ) / 10_000.0
        max_pov = max(1e-6, float(getattr(self.cost_config, "max_participation_rate", 0.10)))
        if market_state is not None and getattr(market_state, "daily_vol", None) is not None:
            vol_arr = np.asarray(market_state.daily_vol, dtype=float)
            vol_arr = vol_arr[np.isfinite(vol_arr) & (vol_arr > 0.0)]
            daily_vol = float(np.median(vol_arr)) if vol_arr.size else float(getattr(self.cost_config, "default_daily_vol", 0.02))
        elif "realised_vol_20d" in day_df.columns:
            vals = pd.to_numeric(day_df["realised_vol_20d"], errors="coerce").dropna()
            daily_vol = float(vals.median()) if not vals.empty else float(getattr(self.cost_config, "default_daily_vol", 0.02))
        else:
            daily_vol = float(getattr(self.cost_config, "default_daily_vol", 0.02))
        impact = float(getattr(self.cost_config, "impact_eta", 0.142)) * daily_vol * (max_pov ** float(getattr(self.cost_config, "impact_alpha", 0.314)))
        return float(max(one_way + impact, 0.0))

    def _lambda_from_vol(self, realized_vol: float) -> float:
        target = max(float(self.cfg.target_volatility), 1e-8)
        ratio = max(float(realized_vol), 1e-8) / target
        raw = self.base_lambda * ratio * ratio
        return float(np.clip(raw, self.base_lambda * self.cfg.lambda_floor_factor, self.base_lambda * self.cfg.lambda_ceil_factor))

    def _gamma_from_cost_alpha(self, expected_cost: float, expected_alpha: float) -> float:
        denom = max(float(expected_alpha), float(self.cfg.min_expected_alpha), 1e-12)
        ratio = float(expected_cost) / denom
        raw = self.base_gamma * ratio
        return float(np.clip(raw, self.base_gamma * self.cfg.gamma_floor_factor, self.base_gamma * self.cfg.gamma_ceil_factor))

    def _smooth(self, which: str, raw: float) -> float:
        if which == "lambda":
            prev = self._lambda_ema
            new = raw if prev is None else self._ema_alpha * raw + (1.0 - self._ema_alpha) * prev
            self._lambda_ema = float(new)
            return float(new)
        prev = self._gamma_ema
        new = raw if prev is None else self._ema_alpha * raw + (1.0 - self._ema_alpha) * prev
        self._gamma_ema = float(new)
        return float(new)
