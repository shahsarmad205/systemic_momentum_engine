"""
MultiAlphaEngine  (v3 — rank-correct architecture)
====================================================
Addresses the fundamental structural constraint identified by quant PM review:

  PROBLEM: All three alphas (reversal, core-ML, momentum) load on the same
           market + sector factor. Gram-Schmidt on market-contaminated inputs
           does NOT increase the rank of the alpha space — it merely rotates
           the same dominant eigenvector. N_eff stays at ~5.

  ROOT CAUSE: α₁ = −ret_1d, α₃ = ret_20d are computed on RAW returns.
              Raw returns contain ~40% market variance + ~20% sector variance.
              The first eigenvector of the alpha covariance ≈ market beta.
              Every subsequent alpha is a projection of this same factor.

  FIX: Factor-neutralize return inputs BEFORE alpha construction.
       r_idio[i,t] = r[i,t] − market_mean[t] − sector_mean[i,t]
       α₁ and α₃ are computed from r_idio, which is zero-mean cross-sectionally
       and has the systematic common factors removed.

  EXPECTED OUTCOME:
    • The dominant market eigenvalue is removed from the alpha covariance.
    • Eigenvalue spectrum flattens → N_eff increases from ~5 toward 15-25.
    • α₂ (ML signal) is already ~factor-neutral via CS normalization.
    • Gram-Schmidt now operates on genuinely independent inputs.
    • Short signals become real: bottom-ranked stocks by idio momentum
      are genuine short candidates, not just "underperformed when market fell."

Algorithmic sections
---------------------
  §1  Feature construction  — idiosyncratic returns (market + sector neutral)
                               + volume-price divergence (Alpha101 paper §III)
  §2  K=3 alphas            — α₁=idio_reversal(τ=2d), α₂=core-ML(τ=7d),
                               α₃=idio_momentum+vol_div(τ=20d)
  §3  CS normalization       — independent z-score + clip(±3σ) per alpha
  §4  Full Gram-Schmidt       — α̃₁=α₁; α̃₂=α₂−proj(α₂|α̃₁);
                               α̃₃=α₃−proj(α₃|[α̃₁,α̃₂])
  §5  Data-driven weights    — w=Σ⁻¹μ from rolling IC history
  §6  Horizon-aware EMA      — span=2τ−1 per alpha: (3, 13, 39)
  §7  Vol-targeting          — s_t=clip(σ*/σ_mkt, lo, hi); continuous, not binary
  §9  Diagnostics            — IC (raw+ortho), corr matrix, N_eff (3×3 eigenvalues)

Research references
-------------------
  Shin et al. (2024) arXiv:2401.02710 — synergistic alpha sets via RL;
    top Alpha101 formula: −Corr(close_ret, volume, 10) [IC=0.035 on CSI300].
    Mapped to: vol-price divergence blend in α₃.

  Frisch-Waugh-Lovell theorem — factor residualization is equivalent to
    adding factor controls to any regression. Applied here cross-sectionally.

  Grinold & Kahn (2000) — IR=IC×√N; N is effective INDEPENDENT bets.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Horizon constants
# ---------------------------------------------------------------------------
_TAU = (2, 7, 20)                              # decay constants (days) per alpha
_EMA_SPAN = tuple(max(2, 2 * t - 1) for t in _TAU)   # EMA span ≈ 2τ−1 → (3, 13, 39)
_MIN_IC_HISTORY = 30                            # days before switching to Σ⁻¹μ weights


# ---------------------------------------------------------------------------
# MultiAlphaEngine
# ---------------------------------------------------------------------------

class MultiAlphaEngine:
    """
    Parameters
    ----------
    base_weights : tuple[float, float, float]
        Fallback weights for (α₁, α₂, α₃) before _MIN_IC_HISTORY IC days.
    reversal_days : int
        Idiosyncratic reversal lookback (1–3 recommended).
    momentum_days : int
        Idiosyncratic momentum lookback (20–40 recommended).
    clip_sigma : float
        Hard clip z-score threshold per alpha.  Default 3.
    ic_window : int
        Rolling days for IC history and Σ⁻¹μ weight update.
    corr_window : int
        Rolling window for correlation / N_eff diagnostics.
    target_vol : float
        Annualised vol target for §7 scaling.  E.g. 0.15 = 15%.
    vol_scale_window : int
        SPY rolling std window.  Default 20.
    vol_scale_lo / hi : float
        Bounds on continuous scaling factor s_t.
    weight_ridge : float
        Ridge on IC covariance Σ for numerical stability.
    use_volume_divergence : bool
        Blend −corr(idio_ret, volume, 10) into α₃.  Default True.
    """

    def __init__(
        self,
        base_weights: tuple[float, float, float] = (0.5, 0.25, 0.25),
        reversal_days: int = 1,
        momentum_days: int = 20,
        clip_sigma: float = 3.0,
        ic_window: int = 60,
        corr_window: int = 60,
        target_vol: float = 0.15,
        vol_scale_window: int = 20,
        vol_scale_lo: float = 0.5,
        vol_scale_hi: float = 2.0,
        weight_ridge: float = 1e-3,
        use_volume_divergence: bool = True,
        ic_horizon_days: int = 1,
    ):
        w = np.asarray(base_weights, dtype=float)
        self._base_weights: np.ndarray = w / w.sum()
        self._optimal_weights: np.ndarray = self._base_weights.copy()

        self.reversal_days = int(reversal_days)
        self.momentum_days = int(momentum_days)
        self.clip_sigma = float(clip_sigma)
        self.ic_window = int(ic_window)
        self.corr_window = int(corr_window)
        self.target_vol = float(target_vol)
        self.vol_scale_window = int(vol_scale_window)
        self.vol_scale_lo = float(vol_scale_lo)
        self.vol_scale_hi = float(vol_scale_hi)
        self.weight_ridge = float(weight_ridge)
        self.use_volume_divergence = bool(use_volume_divergence)
        # IC calibration horizon: must match the portfolio holding period.
        # Using 1-day IC to calibrate weights applied to 5-day positions causes
        # the Σ⁻¹μ to upweight α₁ (reversal, good at 1d) which is anti-predictive
        # at 5-day horizon — inverting the combined signal.
        self.ic_horizon_days = max(1, int(ic_horizon_days))

        # Precomputed DataFrames  (T × N)
        self._alpha1_df: Optional[pd.DataFrame] = None   # idio reversal + EMA
        self._alpha3_df: Optional[pd.DataFrame] = None   # idio momentum + vol-div + EMA
        self._ret_1d_df: Optional[pd.DataFrame] = None   # idio 1d returns (diagnostics only)
        self._ret_Nd_df: Optional[pd.DataFrame] = None   # ic_horizon_days-ahead idio returns (IC calibration)
        self._mkt_vol_ann: Optional[pd.Series] = None    # annualised SPY σ

        # Rolling IC tracking  (K=3)
        self._ic_raw: list[deque] = [deque(maxlen=ic_window) for _ in range(3)]
        self._ic_orth: list[deque] = [deque(maxlen=ic_window) for _ in range(3)]

        # Cross-sectional correlation (ortho alphas)
        self._corr12: deque = deque(maxlen=corr_window)
        self._corr13: deque = deque(maxlen=corr_window)
        self._corr23: deque = deque(maxlen=corr_window)

        # Alpha signal lag queue for IC computation.
        # _prev: 1-day lag (yesterday's signals vs today's 1d returns) — diagnostics only.
        # _prev_Nd: ring buffer of last ic_horizon_days alpha snapshots so that
        #   _update_ic() can pair signals from N days ago with today's N-day return.
        self._prev: Optional[dict] = None
        self._prev_Nd: deque = deque(maxlen=max(1, self.ic_horizon_days))

        # N_eff (alpha eigenvalue breadth)
        self._neff_history: deque = deque(maxlen=corr_window)

    # ------------------------------------------------------------------
    # §1 / §6  Precompute  (once before the simulation loop)
    # ------------------------------------------------------------------

    def precompute(
        self,
        price_data: dict,
        tickers: list[str],
        market_ticker: str = "SPY",
        sector_id_map: dict | None = None,
    ) -> None:
        """
        Build idiosyncratic-return-based alpha DataFrames.

        §1  Factor neutralize: r_idio = r − market_mean − sector_mean
        §6  Apply per-alpha EMA smoothing (span = 2τ−1)

        Parameters
        ----------
        price_data : dict[str, pd.DataFrame]
        tickers : list[str]  — universe
        market_ticker : str  — SPY or equivalent for vol-targeting
        sector_id_map : dict[str, int]  — ticker → 0-based sector index.
                        If None/empty, only market-neutralization is applied.
        """
        sector_id_map = sector_id_map or {}

        # ── Build close and volume matrices ─────────────────────────────
        close_dict: dict[str, pd.Series] = {}
        volume_dict: dict[str, pd.Series] = {}
        for t in tickers:
            df = price_data.get(t)
            if df is None or df.empty:
                continue
            for col in ("Close", "AdjClose", "close"):
                if col in df.columns:
                    px = pd.to_numeric(df[col], errors="coerce")
                    if px.notna().sum() >= max(self.momentum_days + 10, 30):
                        close_dict[t] = px
                    break
            for vcol in ("Volume", "volume"):
                if vcol in df.columns:
                    volume_dict[t] = pd.to_numeric(df[vcol], errors="coerce").clip(lower=0)
                    break

        if len(close_dict) < 2:
            logger.warning("MultiAlphaEngine: <2 tickers with price data; all alphas disabled.")
            return

        prices = pd.DataFrame(close_dict).sort_index()
        prices.index = pd.to_datetime(prices.index)
        tickers_avail = list(prices.columns)

        # ── §1 Raw 1d returns ─────────────────────────────────────────────
        ret_1d_raw = prices.pct_change(1, fill_method=None)

        # ── §1 FACTOR NEUTRALIZATION ─────────────────────────────────────
        # Structural fix: remove market + sector variance from returns BEFORE
        # computing any alpha.  Without this, all alphas load on the same
        # dominant market eigenvector → Gram-Schmidt is cosmetic.
        ret_idio = _factor_neutralize(ret_1d_raw, sector_id_map, tickers_avail)

        # IC target MUST be r_idio, not r_raw.
        # Using raw returns inflates IC on up-market days: all stocks move together,
        # making any alpha with positive bias appear predictive.
        # _ret_1d_df: 1-day idio returns (diagnostics only — do NOT use for Σ⁻¹μ calibration)
        # _ret_Nd_df: ic_horizon_days rolling sum of idio returns (used for IC calibration).
        #   This MUST match the portfolio holding period so Σ⁻¹μ weights reflect
        #   which alpha actually predicts over the horizon being traded.
        self._ret_1d_df = ret_idio
        N = self.ic_horizon_days
        if N > 1:
            # Rolling sum of last N days of idio returns = N-day cumulative idio return.
            # At time T, this is the return from T-N+1 to T, matching a signal entered
            # at T-N+1 and held until T.  _update_ic() looks up this value at today's
            # date against the alpha signal stored N-1 days ago via _prev_Nd queue.
            self._ret_Nd_df = ret_idio.rolling(N, min_periods=max(1, N // 2)).sum()
        else:
            self._ret_Nd_df = ret_idio

        # ── §2 / §6  Alpha 1: idiosyncratic reversal, τ=2d, EMA span=3 ──
        # −r_idio[t] → expect positive alpha from previous idio losers.
        # LOOK-AHEAD FIX: shift(1) before EMA so that α₁ at date T is based
        # entirely on data up to T-1.  Without shift, the EMA at T includes
        # ret_idio[T] (today's close), which the strategy then uses to trade
        # at the SAME close — a direct 1-day look-ahead.  With span=3, today's
        # return gets weight 0.5 in the unshifted EMA, making the look-ahead
        # substantial (not just a numerical artefact).
        self._alpha1_df = (
            (-ret_idio).shift(1)
            .ewm(span=_EMA_SPAN[0], min_periods=1)
            .mean()
        )

        # ── §2 / §6  Alpha 3: idio momentum + vol-divergence, τ=20d ─────
        # Cumulative idiosyncratic return over [t−momentum_days, t−reversal_days]
        # (skip last `reversal_days` to avoid contamination by α₁)
        ret_idio_shifted = ret_idio.shift(self.reversal_days)
        mom_window = max(5, self.momentum_days - self.reversal_days)
        # Vectorised cumulative sum of idio daily returns (no pct_change on prices;
        # sum of daily idio returns ≈ log-return approximation, avoids price lookback)
        mom_raw = ret_idio_shifted.rolling(mom_window, min_periods=max(5, mom_window // 4)).sum()

        if self.use_volume_divergence and len(volume_dict) >= 2:
            # Alpha101 paper: −corr(close_ret, volume, 10) [IC=0.035 on CSI300]
            # Use idiosyncratic returns (not raw) for the correlation computation.
            vol_df = pd.DataFrame(volume_dict).reindex(columns=tickers_avail)
            vol_ranked = vol_df.rank(axis=0, pct=True)
            vol_div = -(ret_idio.rolling(10, min_periods=5).corr(vol_ranked))
            # 70% momentum + 30% vol-divergence blend
            alpha3_raw = 0.7 * mom_raw + 0.3 * vol_div.reindex(columns=tickers_avail)
        else:
            alpha3_raw = mom_raw

        # §6: per-alpha EMA decay (τ=20 → span=39)
        self._alpha3_df = alpha3_raw.ewm(span=_EMA_SPAN[2], min_periods=1).mean()

        # ── §7 Market vol series (annualised SPY σ) ──────────────────────
        df_spy = price_data.get(market_ticker)
        if df_spy is not None and not df_spy.empty:
            for col in ("Close", "AdjClose", "close"):
                if col in df_spy.columns:
                    spy_ret = pd.to_numeric(df_spy[col], errors="coerce").pct_change(fill_method=None)
                    self._mkt_vol_ann = (
                        spy_ret
                        .rolling(self.vol_scale_window, min_periods=5)
                        .std()
                        .mul(np.sqrt(252))
                    )
                    self._mkt_vol_ann.index = pd.to_datetime(self._mkt_vol_ann.index)
                    break

        logger.info(
            "MultiAlphaEngine precomputed: %d tickers | α₁=%s | α₃=%s | "
            "sector_map=%d entries | τ=(%d,%d,%d)d | EMA_span=(%d,%d,%d)",
            len(tickers_avail),
            self._alpha1_df.shape, self._alpha3_df.shape,
            len(sector_id_map), *_TAU, *_EMA_SPAN,
        )

    # ------------------------------------------------------------------
    # §2–§7  Per-date combined forecast
    # ------------------------------------------------------------------

    def current_forecasts(
        self,
        date: pd.Timestamp,
        main_forecasts: dict[str, float],
    ) -> dict[str, float]:
        """
        §2  Lookup α₁(idio reversal), α₂(core ML), α₃(idio momentum) at `date`.
        §3  CS z-score + clip each alpha independently.
        §4  Full Gram-Schmidt: α̃₁=α₁; α̃₂=α₂−proj(α₂|α̃₁); α̃₃=α₃−proj(α₃|[α̃₁,α̃₂]).
        §5  Update Σ⁻¹μ weights from rolling IC history.
        §7  Continuous vol-targeting: s_t = clip(σ*/σ_mkt, lo, hi).
        """
        if not main_forecasts:
            return {}

        active = list(main_forecasts.keys())
        n = len(active)

        # ── §2 Raw alphas ────────────────────────────────────────────────
        a1_raw = self._lookup(self._alpha1_df, date, active, n)      # idio reversal
        a2_raw = np.array([main_forecasts[t] for t in active], dtype=float)  # core ML
        a3_raw = self._lookup(self._alpha3_df, date, active, n)      # idio momentum

        # ── §3 CS normalize independently ────────────────────────────────
        a1_n = _cs_norm(a1_raw, self.clip_sigma)
        a2_n = _cs_norm(a2_raw, self.clip_sigma)
        a3_n = _cs_norm(a3_raw, self.clip_sigma)

        # ── §4 Full Gram-Schmidt orthogonalization ───────────────────────
        a1_orth = a1_n.copy()
        a2_orth = _cs_norm(_proj_residual(a2_n, a1_orth), self.clip_sigma)
        a3_orth = _cs_norm(
            _proj_residual(_proj_residual(a3_n, a1_orth), a2_orth),
            self.clip_sigma,
        )

        # ── §9A Track pairwise correlations (ortho alphas) ───────────────
        valid3 = np.isfinite(a1_orth) & np.isfinite(a2_orth) & np.isfinite(a3_orth)
        if valid3.sum() >= 5:
            c12, c13, c23 = _pairwise_corr(a1_orth, a2_orth, a3_orth, valid3)
            for val, buf in zip((c12, c13, c23), (self._corr12, self._corr13, self._corr23)):
                if np.isfinite(val):
                    buf.append(val)
            # §9C Alpha-space effective breadth
            alpha_mat = np.vstack([a1_orth[valid3], a2_orth[valid3], a3_orth[valid3]])
            self._neff_history.append(_effective_n(np.cov(alpha_mat)))

        # ── §5 IC-based weight update (ic_horizon_days lag) ─────────────
        # Push today's snapshot into the N-day ring buffer BEFORE calling
        # _update_ic, so that _update_ic can pop the oldest entry (N days ago).
        snap = {
            "date": date,
            "tickers": active,
            "raw": np.vstack([a1_raw, a2_raw, a3_raw]),
            "orth": np.vstack([a1_orth, a2_orth, a3_orth]),
        }
        self._prev_Nd.append(snap)
        self._update_ic(date, active, a1_raw, a2_raw, a3_raw, a1_orth, a2_orth, a3_orth)
        weights = self._compute_weights()

        # Store this date's alphas for 1-day IC diagnostic (_prev is 1-day lag only)
        self._prev = snap

        # ── Combine ──────────────────────────────────────────────────────
        a1s = np.where(np.isfinite(a1_orth), a1_orth, 0.0)
        a2s = np.where(np.isfinite(a2_orth), a2_orth, 0.0)
        a3s = np.where(np.isfinite(a3_orth), a3_orth, 0.0)

        # ── Short-side reversal mask ─────────────────────────────────────
        # α₁ (reversal) is anti-predictive for shorts: recent winners tend
        # to persist (momentum dominates reversal past day 2-3), so using
        # α₁ to select short candidates produces consistently wrong-direction
        # positions.  Clip α₁ to positive values only — reversal signal is
        # used solely to identify long candidates (buy recent idio losers).
        a1s = np.where(a1s > 0, a1s, 0.0)

        combined = weights[0] * a1s + weights[1] * a2s + weights[2] * a3s

        # ── §7 Continuous vol-targeting ──────────────────────────────────
        combined = combined * self._vol_scale(date)

        return {t: float(combined[i]) for i, t in enumerate(active)}

    # ------------------------------------------------------------------
    # §5  Rolling IC → Σ⁻¹μ weights
    # ------------------------------------------------------------------

    def _update_ic(
        self,
        date: pd.Timestamp,
        active: list[str],
        a1_raw: np.ndarray, a2_raw: np.ndarray, a3_raw: np.ndarray,
        a1_orth: np.ndarray, a2_orth: np.ndarray, a3_orth: np.ndarray,
    ) -> None:
        # Use the snapshot from ic_horizon_days ago (oldest entry in _prev_Nd).
        # _prev_Nd has maxlen=ic_horizon_days, so when it is full, [0] is the
        # entry from exactly ic_horizon_days ago.  _ret_Nd_df holds the rolling
        # ic_horizon_days cumulative idio return ending at `date`, so pairing
        # the N-day-old signal with today's N-day cumulative return gives a
        # proper N-day-horizon IC.
        if self._ret_Nd_df is None:
            return
        nd_prev = self._prev_Nd[0] if len(self._prev_Nd) == self._prev_Nd.maxlen else None
        if nd_prev is None:
            return
        prev_tickers = nd_prev["tickers"]
        ret_Nd = self._lookup(self._ret_Nd_df, date, prev_tickers, len(prev_tickers))
        valid = np.isfinite(ret_Nd)
        if valid.sum() < 10:
            return
        r = ret_Nd[valid]
        for k, arr in enumerate(nd_prev["raw"]):
            ic = _rank_ic(arr[valid], r)
            if np.isfinite(ic):
                self._ic_raw[k].append(ic)
        for k, arr in enumerate(nd_prev["orth"]):
            ic = _rank_ic(arr[valid], r)
            if np.isfinite(ic):
                self._ic_orth[k].append(ic)

    def _compute_weights(self) -> np.ndarray:
        """w = Σ⁻¹μ from rolling IC of orthogonalized alphas."""
        if any(len(h) < _MIN_IC_HISTORY for h in self._ic_orth):
            return self._base_weights.copy()
        mu = np.array([float(np.mean(h)) for h in self._ic_orth])
        ic_mat = np.column_stack([np.array(list(h)) for h in self._ic_orth])
        sigma = np.cov(ic_mat.T) + self.weight_ridge * np.eye(3)
        try:
            w_raw = np.linalg.solve(sigma, mu)
        except np.linalg.LinAlgError:
            return self._optimal_weights.copy()
        w_sum = float(np.sum(np.abs(w_raw)))
        if w_sum < 1e-9:
            return self._optimal_weights.copy()
        self._optimal_weights = w_raw / w_sum
        return self._optimal_weights.copy()

    # ------------------------------------------------------------------
    # §7  Continuous vol-targeting
    # ------------------------------------------------------------------

    def _vol_scale(self, date: pd.Timestamp) -> float:
        if self._mkt_vol_ann is None:
            return 1.0
        try:
            sigma_mkt = float(self._mkt_vol_ann.asof(date))
        except Exception:
            return 1.0
        if not np.isfinite(sigma_mkt) or sigma_mkt < 0.01:
            return 1.0
        return float(np.clip(self.target_vol / sigma_mkt, self.vol_scale_lo, self.vol_scale_hi))

    # ------------------------------------------------------------------
    # §9  Diagnostics
    # ------------------------------------------------------------------

    def effective_breadth(self) -> float:
        return float(np.mean(self._neff_history)) if self._neff_history else float("nan")

    def correlation_summary(self) -> str:
        if not self._corr12:
            return "corr: (warming up)"
        w = self._optimal_weights
        return (
            f"corr(α̃₁↔α̃₂)={np.mean(self._corr12):+.3f}  "
            f"corr(α̃₁↔α̃₃)={np.mean(self._corr13):+.3f}  "
            f"corr(α̃₂↔α̃₃)={np.mean(self._corr23):+.3f}  "
            f"N_eff={self.effective_breadth():.2f}/3.00  "
            f"w=[{w[0]:+.3f},{w[1]:+.3f},{w[2]:+.3f}]"
        )

    def compute_ic_report(self, price_data: dict | None = None) -> str:
        lines = ["=" * 55, "MultiAlpha Diagnostics Report", "=" * 55]
        labels = ["α₁  idio-reversal", "α₂  core-ML     ", "α₃  idio-momentum"]
        for k, lbl in enumerate(labels):
            raw_h = list(self._ic_raw[k])
            orth_h = list(self._ic_orth[k])
            if raw_h:
                lines.append(
                    f"  {lbl}: "
                    f"IC_raw={np.mean(raw_h):+.4f}(σ={np.std(raw_h):.4f})  "
                    f"IC_orth={np.mean(orth_h):+.4f}(σ={np.std(orth_h):.4f})"
                )
        lines.append("")
        lines.append("  Orthogonal alpha correlation matrix (rolling):")
        if self._corr12:
            m12, m13, m23 = np.mean(self._corr12), np.mean(self._corr13), np.mean(self._corr23)
            lines += [
                f"    [[1.000  {m12:+.3f}  {m13:+.3f}]",
                f"     [{m12:+.3f}   1.000  {m23:+.3f}]",
                f"     [{m13:+.3f}  {m23:+.3f}   1.000]]",
            ]
        neff = self.effective_breadth()
        lines.append(f"\n  Alpha-space N_eff = {neff:.2f} / 3.00  (1.00 = single-factor)")
        lines.append(f"  Current Σ⁻¹μ weights: {self._optimal_weights}")
        lines.append("=" * 55)
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _lookup(
        self,
        df: Optional[pd.DataFrame],
        date: pd.Timestamp,
        active: list[str],
        n: int,
    ) -> np.ndarray:
        if df is None:
            return np.full(n, np.nan)
        loc = df.index.searchsorted(date, side="right") - 1
        if loc < 0:
            return np.full(n, np.nan)
        return df.iloc[loc].reindex(active).values.astype(float)


# ---------------------------------------------------------------------------
# Module-level pure functions
# ---------------------------------------------------------------------------

def _factor_neutralize(
    ret_df: pd.DataFrame,
    sector_id_map: dict,
    tickers: list[str],
) -> pd.DataFrame:
    """
    §1 Factor neutralization: remove market mean + sector mean from each date.

    Step 1 — Market: r_dm[i,t] = r[i,t] − mean_cross_section(r[:,t])
    Step 2 — Sector: r_idio[i,t] = r_dm[i,t] − mean_within_sector(r_dm[:,t])

    This is the Frisch-Waugh residualization with market intercept +
    sector dummies. Vectorized: O(T × N), no per-date Python loops.

    Why this matters: Without neutralization, the dominant eigenvalue of
    the alpha covariance ≈ market beta. All alphas project onto it.
    After neutralization the eigenspectrum flattens → N_eff increases.
    """
    # Step 1: Cross-sectional market demeaning
    mkt_mean = ret_df.mean(axis=1)
    ret_dm = ret_df.sub(mkt_mean, axis=0)

    if not sector_id_map:
        return ret_dm

    # Step 2: Within-sector demeaning
    sector_arr = np.array([sector_id_map.get(t, -1) for t in tickers])
    unique_sectors = sorted(s for s in np.unique(sector_arr) if s >= 0)

    ret_idio = ret_dm.copy()
    for s in unique_sectors:
        mask = sector_arr == s
        if mask.sum() < 2:
            continue
        cols = [tickers[i] for i, m in enumerate(mask) if m]
        sector_mean = ret_dm[cols].mean(axis=1)
        ret_idio[cols] = ret_dm[cols].sub(sector_mean, axis=0)

    return ret_idio


def _cs_norm(arr: np.ndarray, clip_sigma: float) -> np.ndarray:
    """§3: Cross-sectional z-score + clip.  NaN-safe."""
    valid = np.isfinite(arr)
    if valid.sum() < 5:
        return arr.copy()
    vals = arr[valid]
    out = arr.copy()
    out[valid] = np.clip((vals - vals.mean()) / (vals.std() + 1e-9), -clip_sigma, clip_sigma)
    return out


def _proj_residual(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    """§4: OLS residual y ← y − (cov(y,x)/var(x))·x.  NaN-safe."""
    valid = np.isfinite(x) & np.isfinite(y)
    if valid.sum() < 5:
        return y.copy()
    xv, yv = x[valid], y[valid]
    beta = float(np.dot(xv, yv) / (np.dot(xv, xv) + 1e-12))
    out = y.copy()
    out[valid] -= beta * xv
    return out


def _pairwise_corr(
    a1: np.ndarray, a2: np.ndarray, a3: np.ndarray, valid: np.ndarray,
) -> tuple[float, float, float]:
    return (
        float(np.corrcoef(a1[valid], a2[valid])[0, 1]),
        float(np.corrcoef(a1[valid], a3[valid])[0, 1]),
        float(np.corrcoef(a2[valid], a3[valid])[0, 1]),
    )


def _effective_n(cov: np.ndarray) -> float:
    """§9C: N_eff = (Σλ)² / Σλ²  ∈ [1, K].  Max when alphas are independent."""
    ev = np.linalg.eigvalsh(cov)
    ev = ev[ev > 1e-10]
    if len(ev) == 0:
        return 1.0
    s1, s2 = float(ev.sum()), float((ev ** 2).sum())
    return (s1 ** 2) / s2 if s2 > 1e-12 else 1.0


def _rank_ic(alpha: np.ndarray, fwd_ret: np.ndarray) -> float:
    """Spearman rank IC (finite values only)."""
    from scipy.stats import spearmanr
    valid = np.isfinite(alpha) & np.isfinite(fwd_ret)
    if valid.sum() < 5:
        return float("nan")
    rho, _ = spearmanr(alpha[valid], fwd_ret[valid])
    return float(rho)
