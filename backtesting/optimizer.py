"""
Portfolio Optimizer
=====================
Solves the mean-variance + turnover-penalty QP:

    maximize  w' μ  −  λ · w' Σ_idio w  −  γ · ||w − w_prev||²

subject to:
    |w_i| ≤ max_weight          (position bounds)
    |Σ w_i| ≤ net_exposure_max  (net exposure, optional)
    long_only: w_i ≥ 0          (optional)
    Σ |w_i| ≤ gross_cap         (gross leverage)

Solver: FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) — pure numpy.

WHY NOT scipy.optimize.minimize:
  Every scipy solver (SLSQP, trust-constr, L-BFGS-B) internally calls back
  into our objective/gradient functions via Python dispatch.  Those functions
  compute `cov @ w` and `w @ cov @ w`.  On macOS ARM64, Apple's Accelerate
  BLAS triggers RuntimeWarning "divide by zero / overflow encountered in matmul"
  for near-subnormal float64 values encountered at intermediate solver states,
  even when the final result is correct.  This is a BLAS-level floating-point
  exception (FPE) in Apple's BLAS, not a mathematical error in our code.
  Suppressing the warnings (np.errstate, warnings.filterwarnings) would hide
  real errors too.

WHY FISTA:
  Our QP has a CONSTANT, KNOWN gradient:
      ∇f(w) = −μ + 2λΣw + 2γ(w − w₀)
  and a CONSTANT Lipschitz constant:
      L = λ_max(2λΣ + 2γI) = 2λ·λ_max(Σ) + 2γ

  FISTA (Beck & Teboulle 2009) solves  min f(w) + g(w)  where g is the
  indicator of the feasible set (projection = proximal operator of g).

  All operations are pure numpy:
    - matrix-vector products via np.dot (not @)  ← avoids Accelerate FPE path
    - projection via closed-form bisection  ← O(N log N), no scipy
    - momentum update  ← scalar arithmetic

  This completely bypasses Apple Accelerate's FPE-triggering matmul dispatch.
  Convergence is O(1/k²) for convex QP — for N≤100 assets, 500 iterations
  reaches machine precision.

WHY np.dot NOT @:
  Python's @ operator calls __matmul__ which dispatches through numpy's
  dispatch protocol into the active BLAS backend (Accelerate on macOS ARM64).
  np.dot on contiguous float64 arrays calls BLAS dgemv directly via a
  different internal path that does NOT trigger the same FPE signal handling
  in Accelerate's implementation.

Reference:
  Beck & Teboulle (2009) "A Fast Iterative Shrinkage-Thresholding Algorithm
    for Linear Inverse Problems." SIAM J. Imaging Sci.
  Grinold & Kahn (2000) Active Portfolio Management, Ch. 14.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


class PortfolioOptimizer:
    """
    Parameters
    ----------
    lambda_risk : float
        Risk aversion. λ=2 standard; calibrated at idio vol ≈ 15-20%.
    gamma_turnover : float
        Turnover penalty. Turnover emerges from signal decay × γ, not timers.
    max_weight : float
        Maximum absolute weight per position.
    net_exposure_max : float
        Maximum |sum(w)|. 1.0 = unconstrained; 0.1 = near-dollar-neutral.
    long_only : bool
        Enforce w_i ≥ 0.
    gross_cap : float
        Maximum sum(|w|).
    """

    def __init__(
        self,
        lambda_risk: float = 2.0,
        gamma_turnover: float = 4.0,
        max_weight: float = 0.10,
        net_exposure_max: float = 1.0,
        long_only: bool = True,
        gross_cap: float = 1.0,
        min_position_weight: float = 0.0,
    ):
        self.lambda_risk = float(lambda_risk)
        self.gamma_turnover = float(gamma_turnover)
        self.max_weight = float(max_weight)
        self.net_exposure_max = float(net_exposure_max)
        self.long_only = bool(long_only)
        self.gross_cap = float(gross_cap)
        self.min_position_weight = float(min_position_weight)

    # ------------------------------------------------------------------
    # Public: solve QP via FISTA
    # ------------------------------------------------------------------

    def optimize(
        self,
        forecasts: dict,
        cov: np.ndarray,
        w_prev: dict,
        tickers: list,
    ) -> dict:
        """
        Solve the mean-variance + turnover QP using FISTA (pure numpy).

        FISTA gradient step uses np.dot (not @) to avoid triggering
        Apple Accelerate BLAS FPE signals that produce RuntimeWarnings
        in scipy.optimize.minimize callback dispatch.
        """
        N = len(tickers)
        if N == 0:
            return {}

        mu = np.array([forecasts.get(t, 0.0) for t in tickers], dtype=float)
        w0 = np.array([w_prev.get(t, 0.0) for t in tickers], dtype=float)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        w0 = np.where(np.isfinite(w0), w0, 0.0)
        # Clip to ±3.0: handles both return-space (~0.012) and z-score (~±2) inputs.
        # The old ±0.10 clip was calibrated only for return-space; the MultiAlphaEngine
        # outputs cross-sectionally normalized z-scores (typically ±1–3). Clipping at
        # ±0.10 reduces z=2 → 0.10, giving w* ≈ 1.22% per ticker. With 500 tickers this
        # means gross-cap projection scales all weights to <0.5% → no positions open.
        mu = np.clip(mu, -3.0, 3.0)

        # ── Covariance validation ────────────────────────────────────────
        if cov.shape != (N, N) or not np.all(np.isfinite(cov)):
            logger.warning("Optimizer: invalid cov; using identity fallback.")
            cov = np.eye(N) * (0.15 ** 2)
        else:
            cov = (cov + cov.T) / 2.0
            ev = np.linalg.eigvalsh(cov)
            if float(ev[0]) < 1e-8:
                cov += np.eye(N) * (1e-8 - float(ev[0]) + 1e-8)

        # Make C-contiguous float64 — required for safe np.dot dispatch
        cov = np.ascontiguousarray(cov, dtype=np.float64)
        mu = np.ascontiguousarray(mu, dtype=np.float64)
        w0 = np.ascontiguousarray(w0, dtype=np.float64)

        w_opt = self._fista(mu, cov, w0, N)
        lb = 0.0 if self.long_only else -self.max_weight
        w_opt = np.clip(w_opt, lb, self.max_weight)
        w_opt[np.abs(w_opt) < 1e-5] = 0.0

        # ── Min-position threshold: zero out sub-threshold weights to force
        #    concentration and eliminate index-hugging from hundreds of tiny positions.
        #    Re-project onto gross_cap AND net_exposure_max after zeroing.
        #    Without the net re-projection, small shorts that provided net-neutrality
        #    get zeroed, pushing the portfolio net-long above net_exposure_max.
        if self.min_position_weight > 0.0:
            w_opt[np.abs(w_opt) < self.min_position_weight] = 0.0
            gross = float(np.sum(np.abs(w_opt)))
            if gross > self.gross_cap + 1e-10:
                w_opt = self._project_gross_cap(w_opt, lb, self.max_weight)
            if self.net_exposure_max < 0.999:
                net = float(np.sum(w_opt))
                if abs(net) > self.net_exposure_max + 1e-10:
                    w_opt = self._project_net_exposure(w_opt, lb, self.max_weight)

        return {t: float(w_opt[i]) for i, t in enumerate(tickers)}

    # ------------------------------------------------------------------
    # Core: FISTA projected gradient descent
    # ------------------------------------------------------------------

    def _fista(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        w0: np.ndarray,
        N: int,
    ) -> np.ndarray:
        """
        FISTA for  min  f(w) + g(w)
          f(w) = -w'μ + λ·w'Σw + γ·||w-w₀||²   (smooth, quadratic)
          g(w) = indicator of feasible set         (proximal via projection)

        Gradient of f at w:
          ∇f(w) = -μ + 2λΣw + 2γ(w-w₀)
                = (2λΣ + 2γI)w - (μ + 2γw₀)
                = H·w - c   where c = μ + 2γw₀

        Lipschitz constant of ∇f:
          L = λ_max(H) = 2λ·λ_max(Σ) + 2γ

        Step size α = 1/L ensures f(w - α·∇f(w)) ≤ f(w) for all w.

        FISTA update (Beck & Teboulle 2009, Algorithm 1):
          y_k = w_k + ((t_{k-1} - 1) / t_k) (w_k - w_{k-1})  ← momentum
          w_{k+1} = proj(y_k - α·∇f(y_k))                     ← gradient + project

        Uses np.dot (not @) for all matrix-vector products.
        """
        lam = self.lambda_risk
        gam = self.gamma_turnover
        lb = 0.0 if self.long_only else -self.max_weight
        ub = self.max_weight

        # ── Lipschitz constant and step size ────────────────────────────
        # λ_max(Σ) via eigvalsh (ascending); λ_max(H) = 2λ·λ_max(Σ) + 2γ
        lam_max_cov = float(np.linalg.eigvalsh(cov)[-1])
        L = 2.0 * lam * lam_max_cov + 2.0 * gam
        if L < 1e-10:
            L = 1.0
        alpha = 1.0 / L   # gradient step size

        # Pre-compute constant rhs term: c = μ + 2γw₀
        c = mu + 2.0 * gam * w0   # shape (N,)

        # ── Feasible initial point ───────────────────────────────────────
        w = np.clip(w0.copy(), lb, ub)
        w = self._project(w, lb, ub)

        # ── FISTA with restart ───────────────────────────────────────────
        # Restart on monotone increase (adaptive restart, Su et al. 2014):
        # if <∇f(y_k), w_k - w_{k-1}> > 0 then restart (t=1, y=w).
        y = w.copy()
        t = 1.0
        w_prev_iter = w.copy()

        for _ in range(600):
            # ── Gradient: ∇f(y) = H·y - c = (2λΣ + 2γI)y - c
            # np.dot avoids @ dispatch into Accelerate FPE path
            Sy = np.dot(cov, y)           # Σ·y  (BLAS dgemv via np.dot)
            grad_y = 2.0 * lam * Sy + 2.0 * gam * y - c

            # ── Gradient step + projection
            w_new = self._project(y - alpha * grad_y, lb, ub)

            # ── FISTA momentum (Nesterov acceleration)
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))

            # ── Adaptive restart: if gradient direction is wrong, reset momentum
            restart = float(np.dot(grad_y, w_new - w)) > 0.0
            if restart:
                y = w_new.copy()
                t_new = 1.0
            else:
                beta = (t - 1.0) / t_new
                y = w_new + beta * (w_new - w)

            w_prev_iter = w.copy()
            w = w_new
            t = t_new

        return w

    # ------------------------------------------------------------------
    # Projection onto feasible set
    # ------------------------------------------------------------------

    def _project(self, v: np.ndarray, lb: float, ub: float) -> np.ndarray:
        """
        Project v onto:  {w : lb ≤ w_i ≤ ub}  ∩  {Σ|w_i| ≤ gross_cap}
                       ∩  {|Σw_i| ≤ net_exposure_max}

        Step 1: box projection via np.clip — O(N)
        Step 2: gross leverage via bisection water-filling — O(N log N) per iteration
                (actually O(N·64) = O(N) since 64 bisection steps)
        Step 3: net exposure via bisection shift (only when net_exposure_max < 1) — O(N·64)

        All operations are pure numpy — no scipy calls.
        """
        # ── Step 1: box ──────────────────────────────────────────────────
        w = np.clip(v, lb, ub)

        # ── Step 2: gross leverage cap ───────────────────────────────────
        gross = float(np.sum(np.abs(w)))
        if gross > self.gross_cap + 1e-10:
            w = self._project_gross_cap(w, lb, ub)

        # ── Step 3: net exposure cap (only when active) ──────────────────
        if self.net_exposure_max < 0.999:
            net = float(np.sum(w))
            if abs(net) > self.net_exposure_max + 1e-10:
                w = self._project_net_exposure(w, lb, ub)

        return w

    def _project_gross_cap(
        self, w: np.ndarray, lb: float, ub: float
    ) -> np.ndarray:
        """
        Project w onto {Σ|w_i| ≤ gross_cap, lb ≤ w_i ≤ ub} via bisection.

        For long-only (lb=0): w* = clip(v - θ, 0, ub)
          where θ ≥ 0 is found s.t. Σ clip(v - θ, 0, ub) = gross_cap.

        For L/S (lb < 0): w* = sign(v) · clip(|v| - θ, 0, ub)
          where θ ≥ 0 is found s.t. Σ clip(|v| - θ, 0, ub) = gross_cap.

        Both cases: monotone function of θ → bisection in 64 iterations.
        """
        G = self.gross_cap

        if self.long_only:
            # Monotone: f(θ) = Σ clip(w - θ, 0, ub) decreasing in θ
            lo, hi = 0.0, float(np.max(w)) + 1e-10
            for _ in range(64):
                mid = (lo + hi) * 0.5
                if float(np.sum(np.clip(w - mid, 0.0, ub))) > G:
                    lo = mid
                else:
                    hi = mid
            theta = (lo + hi) * 0.5
            return np.clip(w - theta, 0.0, ub)
        else:
            # Soft-threshold on absolute values, preserve signs
            absw = np.abs(w)
            signs = np.sign(w)
            lo, hi = 0.0, float(np.max(absw)) + 1e-10
            for _ in range(64):
                mid = (lo + hi) * 0.5
                if float(np.sum(np.clip(absw - mid, 0.0, ub))) > G:
                    lo = mid
                else:
                    hi = mid
            theta = (lo + hi) * 0.5
            return signs * np.clip(absw - theta, 0.0, ub)

    def _project_net_exposure(
        self, w: np.ndarray, lb: float, ub: float
    ) -> np.ndarray:
        """
        Project w onto {|Σw_i| ≤ net_exposure_max, lb ≤ w_i ≤ ub}.

        Strategy: shift all weights by a scalar δ (uniform tilt removal).
          w* = clip(w - δ, lb, ub)
          Find δ s.t. |Σ clip(w - δ, lb, ub)| = net_exposure_max.

        This is a monotone function → bisection.
        """
        net = float(np.sum(w))
        E = self.net_exposure_max

        if net > E:
            # net too positive: shift down (δ > 0)
            lo, hi = 0.0, net - lb * len(w) + 1e-10
            for _ in range(64):
                mid = (lo + hi) * 0.5
                if float(np.sum(np.clip(w - mid, lb, ub))) > E:
                    lo = mid
                else:
                    hi = mid
            delta = (lo + hi) * 0.5
            return np.clip(w - delta, lb, ub)
        elif net < -E:
            # net too negative: shift up (δ < 0, i.e. add |δ|)
            lo, hi = 0.0, ub * len(w) - net + 1e-10
            for _ in range(64):
                mid = (lo + hi) * 0.5
                if float(np.sum(np.clip(w + mid, lb, ub))) < -E:
                    lo = mid
                else:
                    hi = mid
            delta = (lo + hi) * 0.5
            return np.clip(w + delta, lb, ub)
        return w

    # ------------------------------------------------------------------
    # Closed-form unconstrained optimum (for diagnostics)
    # ------------------------------------------------------------------

    def unconstrained_optimal(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        w0: np.ndarray,
    ) -> np.ndarray:
        """
        Closed-form ignoring bounds/constraints:
          w* = (2λΣ + 2γI)^{-1} (μ + 2γ w0)

        Uses np.linalg.solve — safe, no pinvh, no lstsq.
        """
        N = len(mu)
        H = 2.0 * self.lambda_risk * cov + 2.0 * self.gamma_turnover * np.eye(N)
        rhs = mu + 2.0 * self.gamma_turnover * w0
        try:
            return np.linalg.solve(H, rhs)
        except np.linalg.LinAlgError:
            return np.zeros(N)
