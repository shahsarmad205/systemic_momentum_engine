"""
Direction Meta-Model
=====================
Walk-forward meta-learning of signal direction (+1 / -1) from market state.

Motivation
----------
The static IC-sign rule (direction = sign(IC_raw)) is correct in expectation
but fragile: it treats every window's IC observation as equally reliable and
ignores the market context that drives IC sign stability.

This module trains a logistic regression meta-model that maps a market-state
feature vector to P(direction=+1) using only past-window observations —
preserving strict OOS integrity.

State features (per window)
---------------------------
  From the panel cross-section:
    cs_vol_mean     : mean cross-sectional realized vol (market vol proxy)
    mkt_trend       : cumulative equal-weighted return (trend direction)
    mkt_drawdown    : mean peak-to-current drawdown (stress indicator)
    cs_dispersion   : mean cross-sectional daily-return std (alpha opportunity)
    liq_change      : mean change in log-median ADV (liquidity direction)
  From signal history:
    ic_hist_mean    : mean IC over last K windows (signal momentum)
    ic_hist_std     : IC variability over last K windows (signal stability)
    ic_hist_pos_frac: fraction of last K windows with IC > 0
  From model scores:
    signal_disp     : mean cross-sectional std of model scores

Model
-----
LogisticRegression (L2) with class-balanced weighting.
Scaler: StandardScaler (fitted on train windows, applied to test window).
Minimum windows before activation: ``DirectionModelConfig.min_windows_to_train``.
Fallback to IC-sign rule when insufficient history.

Walk-forward protocol
---------------------
For window t:
1. Extract state_t from window_t panel + ic_history[0..t-1]
2. Call dm.predict(state_t, fallback_ic=IC_raw_t_minus_1)
3. Apply predicted direction to score
4. After OOS IC_raw_t is observed, call dm.record_window(state_t, sign(IC_raw_t))
5. Call dm.fit() before the next window

Diagnostics
-----------
evaluate_direction_stability : flip rate, correct-direction fraction, lift vs static
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class DirectionModelConfig:
    min_windows_to_train: int = 5     # activate meta-model only after this many windows
    ic_history_window: int = 5        # number of past windows for IC history features
    regularization_C: float = 1.0     # LogisticRegression C
    random_state: int = 42
    # panel column names for market state extraction
    return_col: str = "daily_return"
    vol_col: str = "realised_vol_20d"
    adv_col: str = "adv_dollar_20"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DirectionPrediction:
    direction: int           # +1 or -1
    direction_prob: float    # P(direction=+1), NaN when fallback
    fallback: bool           # True = IC-sign rule was used
    reason: str              # human-readable explanation


# ---------------------------------------------------------------------------
# Meta-model
# ---------------------------------------------------------------------------


class DirectionMetaModel:
    """
    Walk-forward meta-model that learns signal direction from market state.

    Usage (per model, per outer walk-forward window)
    ------------------------------------------------
    dm = DirectionMetaModel(cfg)

    for window t in outer_windows:
        state = dm.extract_state(window_panel_df, ic_history[:t])
        pred  = dm.predict(state, fallback_ic=nested_ic_raw)

        # Use pred.direction to orient scores for this OOS window.
        # After OOS IC is observed (ic_raw_t):
        dm.record_window(state, direction_label=1 if ic_raw_t >= 0 else -1)
        dm.fit()   # retrain on all recorded windows
    """

    def __init__(self, cfg: DirectionModelConfig | None = None) -> None:
        self.cfg = cfg or DirectionModelConfig()
        self._states: list[dict[str, float]] = []
        self._labels: list[int] = []
        self._scaler: StandardScaler | None = None
        self._model: LogisticRegression | None = None
        self._trained_on: int = 0
        self._all_feature_keys: list[str] = []

    # ── state extraction ──────────────────────────────────────────────────────

    def extract_state(
        self,
        window_df: pd.DataFrame,
        ic_history: list[float],
        *,
        score_col: str | None = "score",
    ) -> dict[str, float]:
        """
        Compute the state feature vector for one walk-forward window.

        Parameters
        ----------
        window_df  : panel DataFrame for this window (date, ticker, ...)
        ic_history : raw IC values from all past windows, most recent last
        score_col  : column name for model scores (used for signal dispersion)

        Returns
        -------
        dict mapping feature name → float value (NaN → 0.0 before training)
        """
        cfg = self.cfg
        state: dict[str, float] = {}

        work = window_df.copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")

        # ── Cross-sectional / market features ────────────────────────────────
        if cfg.return_col in work.columns:
            r = work[cfg.return_col].replace([np.inf, -np.inf], np.nan)
            by_date = r.groupby(work["date"])
            cs_disp = by_date.std()
            mkt_ret = by_date.mean()
            state["cs_dispersion"] = float(cs_disp.mean()) if not cs_disp.empty else 0.0
            state["mkt_trend"] = float(mkt_ret.sum()) if not mkt_ret.empty else 0.0

            # Drawdown: mean (cumret - rolling_peak)
            cum = mkt_ret.cumsum().ffill()
            peak = cum.expanding().max()
            state["mkt_drawdown"] = float((cum - peak).mean())

        if cfg.vol_col in work.columns:
            v = work[cfg.vol_col].replace([np.inf, -np.inf], np.nan)
            state["cs_vol_mean"] = float(v.groupby(work["date"]).mean().mean())

        if cfg.adv_col in work.columns:
            a = work[cfg.adv_col].replace([np.inf, -np.inf], np.nan).clip(lower=1.0)
            log_adv = np.log(a.to_numpy(dtype=float))
            by_date_adv = pd.Series(log_adv, index=work.index).groupby(work["date"])
            daily_median = by_date_adv.median()
            state["liq_change"] = float(daily_median.diff().mean())

        # ── IC history features ───────────────────────────────────────────────
        recent_all = [x for x in ic_history if math.isfinite(x)]
        recent_k = recent_all[-cfg.ic_history_window:]
        if recent_k:
            state["ic_hist_mean"] = float(np.mean(recent_k))
            state["ic_hist_std"] = (
                float(np.std(recent_k, ddof=1)) if len(recent_k) > 1 else 0.0
            )
            state["ic_hist_pos_frac"] = float(
                sum(1 for x in recent_k if x > 0) / len(recent_k)
            )
        else:
            state["ic_hist_mean"] = 0.0
            state["ic_hist_std"] = 0.0
            state["ic_hist_pos_frac"] = 0.5

        # ── Signal dispersion ─────────────────────────────────────────────────
        if score_col is not None and score_col in work.columns:
            sc = work[score_col].replace([np.inf, -np.inf], np.nan)
            disp_by_date = sc.groupby(work["date"]).std()
            state["signal_disp"] = float(disp_by_date.mean())
        else:
            state["signal_disp"] = 0.0

        # NaN-guard: replace any non-finite value with 0.0
        return {k: (v if math.isfinite(v) else 0.0) for k, v in state.items()}

    # ── recording and training ────────────────────────────────────────────────

    def record_window(self, state: dict[str, float], direction_label: int) -> None:
        """
        Store an observed (state, direction) pair.

        direction_label must be +1 (IC_raw >= 0) or -1 (IC_raw < 0).
        Call this *after* the OOS IC is observed — not before.
        """
        if direction_label not in (1, -1):
            raise ValueError(
                f"direction_label must be +1 or -1, got {direction_label!r}"
            )
        self._states.append(dict(state))
        self._labels.append(direction_label)

    def _feature_matrix(self, states: list[dict[str, float]]) -> np.ndarray:
        """Convert a list of state dicts to a (N, F) float matrix."""
        if not states:
            return np.empty((0, 0))
        if not self._all_feature_keys:
            self._all_feature_keys = sorted(states[0].keys())
        keys = self._all_feature_keys
        return np.array(
            [[s.get(k, 0.0) for k in keys] for s in states],
            dtype=float,
        )

    def fit(self) -> bool:
        """
        (Re-)train on all recorded (state, label) pairs.

        Returns True if training succeeded, False if insufficient data or
        single-class labels (meta-model stays unactivated).
        """
        n = len(self._states)
        if n < self.cfg.min_windows_to_train:
            return False

        labels = np.array(self._labels, dtype=int)
        if len(np.unique(labels)) < 2:
            return False

        X = self._feature_matrix(self._states)
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf = LogisticRegression(
                C=self.cfg.regularization_C,
                max_iter=1000,
                random_state=self.cfg.random_state,
                class_weight="balanced",
                solver="lbfgs",
            )
            clf.fit(Xs, labels)

        self._scaler = scaler
        self._model = clf
        self._trained_on = n
        return True

    # ── prediction ────────────────────────────────────────────────────────────

    def predict(
        self,
        state: dict[str, float],
        *,
        fallback_ic: float = float("nan"),
    ) -> DirectionPrediction:
        """
        Predict direction for the next window given the current market state.

        Falls back to IC-sign rule when the meta-model is not yet trained.
        """
        if self._model is not None and self._scaler is not None:
            if not self._all_feature_keys:
                self._all_feature_keys = sorted(state.keys())
            x = np.array(
                [[state.get(k, 0.0) for k in self._all_feature_keys]],
                dtype=float,
            )
            xs = self._scaler.transform(x)
            classes = list(self._model.classes_)
            probs = self._model.predict_proba(xs)[0]
            prob_pos = float(probs[classes.index(1)]) if 1 in classes else 0.5
            direction = 1 if prob_pos >= 0.5 else -1
            return DirectionPrediction(
                direction=direction,
                direction_prob=prob_pos,
                fallback=False,
                reason=(
                    f"meta-model: P(+1)={prob_pos:.3f}, "
                    f"n_train={self._trained_on}"
                ),
            )

        # Fallback: IC-sign rule
        if math.isfinite(fallback_ic):
            direction = 1 if fallback_ic >= 0.0 else -1
            return DirectionPrediction(
                direction=direction,
                direction_prob=float("nan"),
                fallback=True,
                reason=f"IC-sign fallback: IC={fallback_ic:.4f}",
            )

        return DirectionPrediction(
            direction=1,
            direction_prob=float("nan"),
            fallback=True,
            reason="IC-sign fallback: IC not finite — default +1",
        )

    # ── convenience properties ────────────────────────────────────────────────

    @property
    def n_windows_recorded(self) -> int:
        return len(self._states)

    @property
    def is_active(self) -> bool:
        return self._model is not None


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def evaluate_direction_stability(
    directions: list[int],
    ic_series: list[float],
) -> dict[str, float]:
    """
    Compute stability metrics for a sequence of direction predictions.

    Parameters
    ----------
    directions : predicted direction (+1/-1) per window
    ic_series  : observed IC_raw per window (used as ground truth for label)

    Returns
    -------
    n_windows            : total windows
    flip_rate            : fraction of consecutive-window direction changes
    direction_pos_frac   : fraction of windows with direction=+1
    static_correct_frac  : fraction where static +1 matches sign(IC_raw)
    learned_correct_frac : fraction where predicted direction matches sign(IC_raw)
    lift_vs_static       : learned_correct_frac − static_correct_frac
    """
    n = len(directions)
    if n == 0 or len(ic_series) != n:
        return {k: float("nan") for k in (
            "n_windows", "flip_rate", "direction_pos_frac",
            "static_correct_frac", "learned_correct_frac", "lift_vs_static",
        )}

    ic_signs = [1 if ic >= 0.0 else -1 for ic in ic_series]
    flips = sum(1 for i in range(1, n) if directions[i] != directions[i - 1])

    static_correct = sum(1 for s in ic_signs if s == 1) / n
    learned_correct = sum(1 for d, s in zip(directions, ic_signs) if d == s) / n

    return {
        "n_windows": float(n),
        "flip_rate": flips / (n - 1) if n > 1 else 0.0,
        "direction_pos_frac": sum(1 for d in directions if d == 1) / n,
        "static_correct_frac": static_correct,
        "learned_correct_frac": learned_correct,
        "lift_vs_static": learned_correct - static_correct,
    }


def format_direction_report(
    stability: dict[str, float],
    predictions: list[DirectionPrediction] | None = None,
) -> str:
    """Render direction model diagnostics as a fixed-width text block."""
    def _f(v: object) -> str:
        if isinstance(v, float) and math.isfinite(v):
            return f"{v:7.4f}"
        return "    NaN"

    def _pct(v: object) -> str:
        if isinstance(v, float) and math.isfinite(v):
            return f"{v * 100:6.1f}%"
        return "   NaN"

    lines = [
        "Direction Meta-Model Diagnostics",
        "=" * 52,
        f"  Windows evaluated      : {int(stability.get('n_windows', 0))}",
        f"  Flip rate              : {_pct(stability.get('flip_rate'))}",
        f"  Direction=+1 fraction  : {_pct(stability.get('direction_pos_frac'))}",
        "",
        "  Correct-direction accuracy",
        f"    Static (+1 always)   : {_pct(stability.get('static_correct_frac'))}",
        f"    Learned direction    : {_pct(stability.get('learned_correct_frac'))}",
        f"    Lift vs static       : {_f(stability.get('lift_vs_static'))}",
    ]

    if predictions:
        fallback_cnt = sum(1 for p in predictions if p.fallback)
        lines += [
            "",
            f"  Meta-model activations  : {len(predictions) - fallback_cnt} / {len(predictions)}",
            f"  Fallback (IC-sign rule) : {fallback_cnt} / {len(predictions)}",
        ]

    lines.append("=" * 52)
    return "\n".join(lines)
