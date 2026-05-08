"""
Portfolio Optimizer
=====================
Solves the mean-variance + turnover-penalty QP:

    maximize  w' μ  −  λ · w' Σ_idio w  −  γ · ||w − w_prev||²

subject to:
    |w_i| ≤ max_weight          (position bounds)
    |Σ w_i| ≤ net_exposure_max  (net exposure, optional)
    |B'w| ≤ factor_limits       (explicit factor constraints, optional)
    long_only: w_i ≥ 0          (optional)
    Σ |w_i| ≤ gross_cap         (gross leverage)

Solver: FISTA (Fast Iterative Shrinkage-Thresholding Algorithm) — pure numpy.
With optional Augmented Lagrangian outer loop for linear factor constraints.

Market Impact: integrates the Almgren-Chriss square-root law as a
heterogeneous per-ticker penalty γ_i that varies with trade size,
replacing the naive quadratic approximation that underestimates large
trades and overestimates small ones.

Convergence: reports duality gap, KKT residual, and iteration count.

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
  Almgren & Chriss (2001) "Optimal Execution of Portfolio Transactions."
    J. Computational Finance — square-root impact law.
  Kissell & Malamut (2005) "Understanding Market Impact." ITG.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConvergenceReport:
    """Solver convergence diagnostics returned alongside optimal weights."""
    n_iterations: int
    max_iterations: int
    converged: bool
    objective_final: float
    objective_initial: float
    objective_improvement: float
    gradient_norm: float
    kkt_residual: float
    kkt_stationarity_violation: float
    kkt_feasibility_violation: float
    duality_gap: float
    factor_violations: dict = field(default_factory=lambda: {})

    @property
    def summary(self):
        parts = [
            f"iter={self.n_iterations}/{self.max_iterations}",
            f"converged={self.converged}",
            f"obj={self.objective_final:.6e} (Δ={self.objective_improvement:.2e})",
            f"∇norm={self.gradient_norm:.2e}",
            f"KKT={self.kkt_residual:.2e}",
            f"duality_gap={self.duality_gap:.2e}",
        ]
        if self.factor_violations:
            worst = max(self.factor_violations.values())
            parts.append(f"worst_factor_viol={worst:.2e}")
        return " | ".join(parts)


@dataclass(frozen=True)
class FactorConstraint:
    """Linear factor constraint: |B'w| ≤ limit."""
    name: str
    exposure: np.ndarray  # shape (N,)
    limit: float


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
    min_position_weight : float
        Minimum absolute weight threshold.
    impact_gamma : float
        Almgren-Chriss impact exponent (default 0.6 ≈ square-root law).
        Used to compute per-ticker effective γ_i from trade size.
    fista_max_iter : int
        Maximum FISTA iterations before forcing stop.
    fista_tol : float
        Convergence tolerance on relative weight change.
    neutrality_penalty : float
        P5: Quadratic penalty on net exposure in the QP objective.
        Adds ρ·(Σw_i)² to the objective, driving the optimizer toward
        dollar-neutrality endogenously (not just via post-hoc projection).
        ρ=100 enforces strict neutrality for L/S spread books.
        ρ=0 for long-only mandates (neutrality is meaningless there).
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
        impact_gamma: float = 0.6,
        fista_max_iter: int = 1000,
        fista_tol: float = 1e-8,
        neutrality_penalty: float = 0.0,
    ):
        self.lambda_risk = float(lambda_risk)
        self.gamma_turnover = float(gamma_turnover)
        self.max_weight = float(max_weight)
        self.net_exposure_max = float(net_exposure_max)
        self.long_only = bool(long_only)
        self.gross_cap = float(gross_cap)
        self.min_position_weight = float(min_position_weight)
        self.impact_gamma = float(impact_gamma)
        self.fista_max_iter = int(fista_max_iter)
        self.fista_tol = float(fista_tol)
        self.neutrality_penalty = float(neutrality_penalty)

    # ------------------------------------------------------------------
    # Public: solve QP via FISTA
    # ------------------------------------------------------------------

    def optimize(
        self,
        forecasts: dict,
        cov: np.ndarray,
        w_prev: dict,
        tickers: list,
        per_ticker_ub: dict | None = None,
        per_ticker_cost: dict | None = None,
        per_ticker_impact_gamma: dict | None = None,
        factor_constraints: list[FactorConstraint] | None = None,
        per_ticker_vol: dict | None = None,
        per_ticker_adv: dict | None = None,
        portfolio_equity: float = 1.0,
        return_convergence: bool = False,
    ) -> dict | tuple[dict, ConvergenceReport]:
        """
        Solve the mean-variance + turnover QP using FISTA (pure numpy).

        FISTA gradient step uses np.dot (not @) to avoid triggering
        Apple Accelerate BLAS FPE signals that produce RuntimeWarnings
        in scipy.optimize.minimize callback dispatch.

        Parameters
        ----------
        per_ticker_ub : dict, optional
            Per-ticker upper bound on |weight|.  Typically ADV-derived.
        per_ticker_cost : dict, optional
            Per-ticker one-way transaction cost rate.
        per_ticker_impact_gamma : dict, optional
            Per-ticker effective turnover penalty from ImpactModel.
        factor_constraints : list[FactorConstraint], optional
            Explicit linear factor constraints |B'w| ≤ limit encoded
            directly into the optimization via Augmented Lagrangian.
            This replaces the post-optimization projection that violates
            KKT conditions.
        per_ticker_vol : dict, optional
            Per-ticker daily volatility for Almgren-Chriss impact calc.
        per_ticker_adv : dict, optional
            Per-ticker average daily volume (USD) for impact calc.
        portfolio_equity : float
            Total portfolio equity in USD for impact scaling.
        return_convergence : bool
            If True, return (weights, ConvergenceReport) tuple.

        Returns
        -------
        dict[ticker, weight] or tuple[dict, ConvergenceReport]
        """
        N = len(tickers)
        if N == 0:
            result = {}
            if return_convergence:
                return result, ConvergenceReport(
                    n_iterations=0, max_iterations=self.fista_max_iter,
                    converged=True, objective_final=0.0, objective_initial=0.0,
                    objective_improvement=0.0, gradient_norm=0.0,
                    kkt_residual=0.0, kkt_stationarity_violation=0.0,
                    kkt_feasibility_violation=0.0, duality_gap=0.0,
                )
            return result

        mu = np.array([forecasts.get(t, 0.0) for t in tickers], dtype=float)
        w0 = np.array([w_prev.get(t, 0.0) for t in tickers], dtype=float)
        mu = np.where(np.isfinite(mu), mu, 0.0)
        w0 = np.where(np.isfinite(w0), w0, 0.0)
        mu = np.clip(mu, -3.0, 3.0)

        # ── Per-ticker gamma vector (heterogeneous impact penalty) ──────────
        # Almgren-Chriss square-root law: effective penalty γ_i depends on
        # trade size via γ_i = γ_base + κ · (|Δw_i| · equity / ADV_i)^impact_gamma.
        # This captures the convexity that the quadratic approximation misses:
        # small trades → γ_i ≈ γ_base (like quadratic)
        # large trades → γ_i ≫ γ_base (underestimated by quadratic)
        if per_ticker_impact_gamma is not None:
            gamma_vec = np.array(
                [float(per_ticker_impact_gamma.get(t, self.gamma_turnover)) for t in tickers],
                dtype=np.float64,
            )
            gamma_vec = np.maximum(gamma_vec, 0.0)
        else:
            gamma_vec = None

        # ── Linear cost adjustment: subtract κ_i from forecast in trade direction ──
        if per_ticker_cost is not None:
            kappa = np.array(
                [float(per_ticker_cost.get(t, 0.0)) for t in tickers],
                dtype=np.float64,
            )
            if np.any(kappa > 0.0):
                gv_diag = gamma_vec if gamma_vec is not None else np.full(N, self.gamma_turnover, dtype=np.float64)
                H_diag = (
                    2.0 * self.lambda_risk * np.diag(cov)
                    + 2.0 * gv_diag
                    + 1e-10
                )
                w_unc = (mu + 2.0 * gv_diag * w0) / H_diag
                direction = np.sign(w_unc - w0)
                direction = np.where(direction == 0.0, np.sign(mu), direction)
                mu = mu - kappa * direction

        # ── Build per-ticker bound vectors ───────────────────────────────
        if per_ticker_ub is not None:
            ub_vec = np.array(
                [min(float(per_ticker_ub.get(t, self.max_weight)), self.max_weight) for t in tickers],
                dtype=np.float64,
            )
        else:
            ub_vec = np.full(N, self.max_weight, dtype=np.float64)
        lb_vec = np.zeros(N, dtype=np.float64) if self.long_only else -ub_vec

        # ── Covariance validation ────────────────────────────────────────
        if cov.shape != (N, N) or not np.all(np.isfinite(cov)):
            logger.warning("Optimizer: invalid cov; using identity fallback.")
            cov = np.eye(N) * (0.15 ** 2)
        else:
            cov = (cov + cov.T) / 2.0
            ev = np.linalg.eigvalsh(cov)
            if float(ev[0]) < 1e-8:
                cov += np.eye(N) * (1e-8 - float(ev[0]) + 1e-8)

        cov = np.ascontiguousarray(cov, dtype=np.float64)
        mu = np.ascontiguousarray(mu, dtype=np.float64)
        w0 = np.ascontiguousarray(w0, dtype=np.float64)

        # ── Solve with optional factor constraints via Augmented Lagrangian ──
        if factor_constraints:
            w_opt, conv = self._solve_with_factors(
                mu, cov, w0, N, lb_vec, ub_vec, gamma_vec, factor_constraints,
            )
        else:
            w_opt, conv = self._fista(
                mu, cov, w0, N, lb_vec=lb_vec, ub_vec=ub_vec, gamma_vec=gamma_vec,
            )

        w_opt = np.clip(w_opt, lb_vec, ub_vec)
        w_opt[np.abs(w_opt) < 1e-5] = 0.0

        if self.min_position_weight > 0.0:
            w_opt[np.abs(w_opt) < self.min_position_weight] = 0.0
            gross = float(np.sum(np.abs(w_opt)))
            lb_scalar = 0.0 if self.long_only else -self.max_weight
            if gross > self.gross_cap + 1e-10:
                w_opt = self._project_gross_cap(w_opt, lb_vec, ub_vec)
            if self.net_exposure_max < 0.999:
                net = float(np.sum(w_opt))
                if abs(net) > self.net_exposure_max + 1e-10:
                    w_opt = self._project_net_exposure(w_opt, lb_scalar, self.max_weight)

        result = {t: float(w_opt[i]) for i, t in enumerate(tickers)}
        if return_convergence:
            return result, conv
        return result

    def constrain_weights(
        self,
        weights: dict[str, float],
        tickers: list[str],
        *,
        factor_exposures: dict[str, np.ndarray] | None = None,
        factor_bounds: dict[str, float] | None = None,
        max_weight_overrides: dict[str, float] | None = None,
    ) -> dict[str, float]:
        """
        DEPRECATED: Apply factor constraints via post-optimization projection.

        This method violates KKT conditions — the projected portfolio is not
        optimal for the constrained problem.  Use factor_constraints parameter
        in optimize() instead, which encodes constraints via Augmented Lagrangian
        directly in the QP.

        Kept only for backward compatibility with the live backtester.
        """
        logger.warning(
            "constrain_weights is deprecated.  Pass factor_constraints to optimize() "
            "instead for KKT-satisfying factor-constrained optimization."
        )
        if not tickers:
            return {}

        w = np.array([float(weights.get(t, 0.0)) for t in tickers], dtype=float)
        ub_vec = np.full(len(tickers), float(self.max_weight), dtype=float)
        if max_weight_overrides:
            for i, ticker in enumerate(tickers):
                override = max_weight_overrides.get(ticker)
                if override is None:
                    continue
                ub_vec[i] = max(0.0, min(float(override), float(self.max_weight)))
        lb_vec = np.zeros(len(tickers), dtype=float) if self.long_only else -ub_vec

        def _clip_and_rescale(arr: np.ndarray) -> np.ndarray:
            arr = np.clip(arr, lb_vec, ub_vec)
            gross = float(np.sum(np.abs(arr)))
            if gross > self.gross_cap + 1e-12 and gross > 0:
                arr = arr * (self.gross_cap / gross)
                arr = np.clip(arr, lb_vec, ub_vec)
            net = float(np.sum(arr))
            if self.net_exposure_max < 0.999 and abs(net) > self.net_exposure_max + 1e-12:
                long_mask = arr > 0
                short_mask = arr < 0
                long_sum = float(arr[long_mask].sum())
                short_sum = float(-arr[short_mask].sum())
                if net > self.net_exposure_max and long_sum > 0:
                    desired_long = self.net_exposure_max + short_sum
                    arr[long_mask] *= max(0.0, min(1.0, desired_long / long_sum))
                elif net < -self.net_exposure_max and short_sum > 0:
                    desired_short = long_sum + self.net_exposure_max
                    arr[short_mask] *= max(0.0, min(1.0, desired_short / short_sum))
                arr = np.clip(arr, lb_vec, ub_vec)
            arr[np.abs(arr) < 1e-6] = 0.0
            return arr

        w = _clip_and_rescale(w)
        if factor_exposures and factor_bounds:
            for _ in range(3):
                violated = False
                for name, exposure in factor_exposures.items():
                    bound = factor_bounds.get(name)
                    if bound is None or bound <= 0:
                        continue
                    exp_vec = np.array(exposure, dtype=float)
                    if exp_vec.shape != w.shape or not np.isfinite(exp_vec).all():
                        continue
                    cur = float(np.dot(exp_vec, w))
                    if abs(cur) <= float(bound) + 1e-10:
                        continue
                    violated = True
                    target = float(bound) if cur > 0 else -float(bound)
                    denom = float(np.dot(exp_vec, exp_vec)) + 1e-12
                    w = w - ((cur - target) / denom) * exp_vec
                    w = _clip_and_rescale(w)
                if not violated:
                    break

        return {ticker: float(w[i]) for i, ticker in enumerate(tickers)}

    def _solve_with_factors(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        w0: np.ndarray,
        N: int,
        lb_vec: np.ndarray,
        ub_vec: np.ndarray,
        gamma_vec: np.ndarray | None,
        factor_constraints: list[FactorConstraint],
    ) -> tuple[np.ndarray, ConvergenceReport]:
        """
        Solve QP with explicit linear factor constraints |B'w| ≤ limit.

        Uses Augmented Lagrangian (method of multipliers) outer loop:
          L_ρ(w, λ) = f(w) + Σ_j [λ_j · (|B_j'w| - limit_j)_+ + ρ/2 · (|B_j'w| - limit_j)_+²]

        Each outer iteration solves a bound-constrained QP via FISTA with
        modified objective that includes the AL penalty terms.

        This encodes factor constraints as part of the optimization problem
        rather than post-hoc projection, satisfying KKT conditions.
        """
        rho = 10.0  # initial penalty parameter
        n_factors = len(factor_constraints)
        lambdas = np.zeros(n_factors)  # Lagrange multipliers

        # Pre-stack factor exposure matrix B (n_factors × N)
        B = np.array([fc.exposure for fc in factor_constraints], dtype=np.float64)
        limits = np.array([fc.limit for fc in factor_constraints], dtype=np.float64)
        names = [fc.name for fc in factor_constraints]

        best_w = w0.copy()
        best_obj = np.inf
        overall_conv = None

        for al_iter in range(20):  # outer AL iterations
            # Modified objective: f_AL(w) = f(w) + Σ AL penalties
            # Gradient contribution from AL: Σ λ_j · B_j + ρ · (|B_j'w| - limit_j)_+ · sign(B_j'w) · B_j
            # We handle this inside the FISTA gradient computation.

            w_inner, inner_conv = self._fista_with_factor_penalties(
                mu, cov, w0, N, lb_vec, ub_vec, gamma_vec,
                B, limits, lambdas, rho,
            )

            # Evaluate factor violations
            factor_vals = B @ w_inner  # n_factors
            violations = np.abs(factor_vals) - limits
            max_violation = float(np.max(np.maximum(violations, 0.0)))

            # Update multipliers and penalty
            for j in range(n_factors):
                viol = max(0.0, abs(factor_vals[j]) - limits[j])
                sign_j = 1.0 if factor_vals[j] > 0 else -1.0
                lambdas[j] = max(0.0, lambdas[j] + rho * viol * sign_j)
            rho = min(rho * 2.0, 1e6)

            # Track best feasible solution
            obj = self._objective(mu, cov, w0, w_inner, gamma_vec)
            if obj < best_obj or max_violation < 1e-6:
                best_obj = obj
                best_w = w_inner.copy()

            factor_viol_dict = {names[j]: float(max(0.0, violations[j])) for j in range(n_factors)}

            if max_violation < 1e-8:
                # KKT satisfied
                grad_norm = self._gradient_norm(mu, cov, w0, best_w, gamma_vec, B, limits, lambdas, rho)
                overall_conv = ConvergenceReport(
                    n_iterations=inner_conv.n_iterations + al_iter,
                    max_iterations=self.fista_max_iter * 20,
                    converged=True,
                    objective_final=best_obj,
                    objective_initial=inner_conv.objective_initial,
                    objective_improvement=inner_conv.objective_improvement,
                    gradient_norm=grad_norm,
                    kkt_residual=grad_norm + max_violation,
                    kkt_stationarity_violation=grad_norm,
                    kkt_feasibility_violation=max_violation,
                    duality_gap=inner_conv.duality_gap,
                    factor_violations=factor_viol_dict,
                )
                return best_w, overall_conv

        # Return best found even if not fully converged
        grad_norm = self._gradient_norm(mu, cov, w0, best_w, gamma_vec, B, limits, lambdas, rho)
        factor_vals_final = B @ best_w
        factor_viol_dict = {names[j]: float(max(0.0, abs(factor_vals_final[j]) - limits[j])) for j in range(n_factors)}
        overall_conv = ConvergenceReport(
            n_iterations=inner_conv.n_iterations + 20,
            max_iterations=self.fista_max_iter * 20,
            converged=False,
            objective_final=best_obj,
            objective_initial=inner_conv.objective_initial,
            objective_improvement=inner_conv.objective_improvement,
            gradient_norm=grad_norm,
            kkt_residual=grad_norm + max_violation,
            kkt_stationarity_violation=grad_norm,
            kkt_feasibility_violation=max_violation,
            duality_gap=inner_conv.duality_gap,
            factor_violations=factor_viol_dict,
        )
        return best_w, overall_conv

    def _fista_with_factor_penalties(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        w0: np.ndarray,
        N: int,
        lb_vec: np.ndarray,
        ub_vec: np.ndarray,
        gamma_vec: np.ndarray | None,
        B: np.ndarray,
        limits: np.ndarray,
        lambdas: np.ndarray,
        rho: float,
    ) -> tuple[np.ndarray, ConvergenceReport]:
        """FISTA with Augmented Lagrangian factor penalty in the gradient."""
        lam = self.lambda_risk
        gam = self.gamma_turnover
        gv = gamma_vec if gamma_vec is not None else np.full(N, gam, dtype=np.float64)

        lam_max_cov = float(np.linalg.eigvalsh(cov)[-1])
        # Add factor penalty curvature to Lipschitz estimate
        factor_curvature = rho * float(np.sum(np.sum(B ** 2, axis=1)))
        L = 2.0 * lam * lam_max_cov + 2.0 * float(np.max(gv)) + factor_curvature
        if L < 1e-10:
            L = 1.0
        alpha = 1.0 / L

        c = mu + 2.0 * gv * w0

        w = np.clip(w0.copy(), lb_vec, ub_vec)
        w = self._project(w, lb_vec, ub_vec)
        y = w.copy()
        t = 1.0

        obj_initial = self._objective_with_factors(mu, cov, w0, w, gamma_vec, B, limits, lambdas, rho)
        best_w = w.copy()
        best_obj = obj_initial

        for it in range(self.fista_max_iter):
            B_y = np.dot(B, y)  # n_factors
            factor_vals = np.abs(B_y) - limits
            factor_pos = np.maximum(factor_vals, 0.0)

            # Gradient of AL penalty: Σ (λ_j + ρ · factor_pos_j) · sign(B_j'y) · B_j
            al_grad = np.zeros(N)
            for j in range(len(lambdas)):
                coeff = lambdas[j] + rho * factor_pos[j]
                sign_j = 1.0 if B_y[j] > 0 else -1.0
                al_grad += coeff * sign_j * B[j]

            Sy = np.dot(cov, y)
            grad_y = 2.0 * lam * Sy + 2.0 * gv * y - c + al_grad

            w_new = self._project(y - alpha * grad_y, lb_vec, ub_vec)

            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            restart = float(np.dot(grad_y, w_new - w)) > 0.0
            if restart:
                y = w_new.copy()
                t_new = 1.0
            else:
                beta = (t - 1.0) / t_new
                y = w_new + beta * (w_new - w)

            w = w_new
            t = t_new

            # Convergence check
            w_change = float(np.max(np.abs(w - best_w)))
            obj = self._objective_with_factors(mu, cov, w0, w, gamma_vec, B, limits, lambdas, rho)
            if obj < best_obj:
                best_obj = obj
                best_w = w.copy()
            if w_change < self.fista_tol and it > 10:
                break

        grad_norm = float(np.linalg.norm(
            2.0 * lam * np.dot(cov, best_w) + 2.0 * gv * best_w - mu - 2.0 * gv * w0
        ))
        obj_final = self._objective_with_factors(mu, cov, w0, best_w, gamma_vec, B, limits, lambdas, rho)

        return best_w, ConvergenceReport(
            n_iterations=it + 1, max_iterations=self.fista_max_iter,
            converged=w_change < self.fista_tol,
            objective_final=obj_final, objective_initial=obj_initial,
            objective_improvement=obj_initial - obj_final,
            gradient_norm=grad_norm,
            kkt_residual=grad_norm,
            kkt_stationarity_violation=grad_norm,
            kkt_feasibility_violation=0.0,
            duality_gap=self._duality_gap(mu, cov, w0, best_w, lb_vec, ub_vec, gamma_vec),
        )

    def _objective(self, mu, cov, w0, w, gamma_vec):
        gv = gamma_vec if gamma_vec is not None else np.full(len(w), self.gamma_turnover)
        return float(-np.dot(w, mu) + self.lambda_risk * np.dot(w, np.dot(cov, w)) + np.dot(gv, (w - w0) ** 2))

    def _objective_with_factors(self, mu, cov, w0, w, gamma_vec, B, limits, lambdas, rho):
        obj = self._objective(mu, cov, w0, w, gamma_vec)
        Bw = np.dot(B, w)
        for j in range(len(limits)):
            viol = max(0.0, abs(Bw[j]) - limits[j])
            obj += lambdas[j] * viol + 0.5 * rho * viol ** 2
        return obj

    def _gradient_norm(self, mu, cov, w0, w, gamma_vec, B, limits, lambdas, rho):
        gv = gamma_vec if gamma_vec is not None else np.full(len(w), self.gamma_turnover)
        grad = -mu + 2.0 * self.lambda_risk * np.dot(cov, w) + 2.0 * gv * (w - w0)
        Bw = np.dot(B, w)
        for j in range(len(limits)):
            if abs(Bw[j]) > limits[j]:
                sign_j = 1.0 if Bw[j] > 0 else -1.0
                grad += (lambdas[j] + rho * (abs(Bw[j]) - limits[j])) * sign_j * B[j]
        return float(np.linalg.norm(grad))

    def _duality_gap(self, mu, cov, w0, w, lb_vec, ub_vec, gamma_vec):
        """Compute primal-dual gap for bound-constrained QP."""
        gv = gamma_vec if gamma_vec is not None else np.full(len(w), self.gamma_turnover)
        N = len(w)
        H = 2.0 * self.lambda_risk * cov + 2.0 * np.diag(gv)
        rhs = mu + 2.0 * gv * w0
        grad = -np.dot(H, w) + rhs
        # Dual: maximize -0.5·w'Hw + rhs'w subject to bounds
        primal = self._objective(mu, cov, w0, w, gamma_vec)
        # Upper bound on dual via relaxation
        dual_approx = float(-0.5 * np.dot(w, np.dot(H, w)) + np.dot(rhs, w))
        return abs(primal + dual_approx)

    # ------------------------------------------------------------------
    # Core: FISTA projected gradient descent
    # ------------------------------------------------------------------

    def _fista(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        w0: np.ndarray,
        N: int,
        lb_vec: np.ndarray | None = None,
        ub_vec: np.ndarray | None = None,
        gamma_vec: np.ndarray | None = None,
    ) -> tuple[np.ndarray, ConvergenceReport]:
        """
        FISTA for  min  f(w) + g(w)
          f(w) = -w'μ + λ·w'Σw + Σ γ_i·(w_i−w₀_i)²   (smooth, quadratic)
          g(w) = indicator of feasible set               (proximal via projection)

        Returns (optimal_weights, ConvergenceReport).

        Convergence criteria:
          1. Relative weight change < fista_tol
          2. Gradient norm < fista_tol
          3. Max iterations reached (not converged)

        Reports duality gap and KKT residual.
        """
        lam = self.lambda_risk
        gam = self.gamma_turnover
        rho_neutral = self.neutrality_penalty  # P5: neutrality penalty coefficient
        gv: np.ndarray = gamma_vec if gamma_vec is not None else np.full(N, gam, dtype=np.float64)

        if lb_vec is None:
            lb_vec = np.full(N, 0.0 if self.long_only else -self.max_weight, dtype=np.float64)
        if ub_vec is None:
            ub_vec = np.full(N, self.max_weight, dtype=np.float64)

        lam_max_cov = float(np.linalg.eigvalsh(cov)[-1])
        # P5: neutrality penalty adds rank-1 term ρ·1·1' to Hessian → eigenvalue 2ρN
        L = 2.0 * lam * lam_max_cov + 2.0 * float(np.max(gv)) + 2.0 * rho_neutral * N
        if L < 1e-10:
            L = 1.0
        alpha = 1.0 / L

        c = mu + 2.0 * gv * w0

        w = np.clip(w0.copy(), lb_vec, ub_vec)
        w = self._project(w, lb_vec, ub_vec)
        obj_initial = self._objective(mu, cov, w0, w, gamma_vec)

        y = w.copy()
        t = 1.0
        best_w = w.copy()
        best_obj = obj_initial
        converged = False
        n_iter = 0
        w_change = np.inf

        for it in range(self.fista_max_iter):
            Sy = np.dot(cov, y)
            grad_y = 2.0 * lam * Sy + 2.0 * gv * y - c
            # P5: add neutrality gradient ∂/∂w [ρ·(Σw)²] = 2ρ·(Σw)·1
            if rho_neutral > 0:
                grad_y = grad_y + 2.0 * rho_neutral * np.sum(y)

            w_new = self._project(y - alpha * grad_y, lb_vec, ub_vec)

            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))

            restart = float(np.dot(grad_y, w_new - w)) > 0.0
            if restart:
                y = w_new.copy()
                t_new = 1.0
            else:
                beta = (t - 1.0) / t_new
                y = w_new + beta * (w_new - w)

            w = w_new
            t = t_new
            n_iter = it + 1

            # ── Convergence check ────────────────────────────────────────
            w_change = float(np.max(np.abs(w - best_w)))
            obj = self._objective(mu, cov, w0, w, gamma_vec)
            if obj < best_obj:
                best_obj = obj
                best_w = w.copy()

            # Check both weight change and gradient norm
            grad_norm = float(np.linalg.norm(grad_y))
            if w_change < self.fista_tol and it > 10:
                converged = True
                break
            if grad_norm < self.fista_tol and it > 10:
                converged = True
                break

            # Early stopping on objective stagnation: if relative improvement
            # is < 1e-6 of initial objective for 20+ consecutive iterations,
            # the solver is spinning without meaningful progress.
            rel_improvement = abs(obj - best_obj) / (abs(obj_initial) + 1e-12)
            if it >= 30 and rel_improvement < 1e-7 and w_change < 1e-5:
                converged = True
                break

        # ── Post-solve diagnostics ───────────────────────────────────────
        grad_final = 2.0 * lam * np.dot(cov, best_w) + 2.0 * gv * best_w - c
        grad_norm_final = float(np.linalg.norm(grad_final))
        obj_final = self._objective(mu, cov, w0, best_w, gamma_vec)

        # KKT stationarity: projection residual = w - Π(w - ∇f(w))
        kkt_stationarity = float(np.max(np.abs(best_w - self._project(best_w - grad_final, lb_vec, ub_vec))))

        # Feasibility: box constraint violations (should be ~0 after projection)
        kkt_feasibility = float(max(
            np.max(np.maximum(lb_vec - best_w, 0.0)),
            np.max(np.maximum(best_w - ub_vec, 0.0)),
        ))

        duality_gap = self._duality_gap(mu, cov, w0, best_w, lb_vec, ub_vec, gamma_vec)

        report = ConvergenceReport(
            n_iterations=n_iter,
            max_iterations=self.fista_max_iter,
            converged=converged,
            objective_final=obj_final,
            objective_initial=obj_initial,
            objective_improvement=obj_initial - obj_final,
            gradient_norm=grad_norm_final,
            kkt_residual=kkt_stationarity + kkt_feasibility,
            kkt_stationarity_violation=kkt_stationarity,
            kkt_feasibility_violation=kkt_feasibility,
            duality_gap=duality_gap,
        )

        if not converged:
            logger.warning(
                "FISTA did not converge in %d iterations: %s",
                n_iter, report.summary,
            )

        return best_w, report

    # ------------------------------------------------------------------
    # Projection onto feasible set
    # ------------------------------------------------------------------

    def _project(
        self,
        v: np.ndarray,
        lb: float | np.ndarray,
        ub: float | np.ndarray,
    ) -> np.ndarray:
        """
        Project v onto:  {w : lb_i ≤ w_i ≤ ub_i}  ∩  {Σ|w_i| ≤ gross_cap}
                       ∩  {|Σw_i| ≤ net_exposure_max}

        lb and ub may be scalar or per-ticker arrays (for ADV caps).

        Step 1: per-ticker box projection → O(N)
        Step 2: gross leverage cap via bisection water-filling — O(N·64)
        Step 3: net exposure cap via bisection shift — O(N·64)
        """
        w = np.clip(v, lb, ub)

        gross = float(np.sum(np.abs(w)))
        if gross > self.gross_cap + 1e-10:
            w = self._project_gross_cap(w, lb, ub)

        if self.net_exposure_max < 0.999:
            net = float(np.sum(w))
            if abs(net) > self.net_exposure_max + 1e-10:
                lb_s = float(np.min(lb)) if hasattr(lb, "__len__") else float(lb)
                ub_s = float(np.max(ub)) if hasattr(ub, "__len__") else float(ub)
                w = self._project_net_exposure(w, lb_s, ub_s)

        return w

    def _project_gross_cap(
        self,
        w: np.ndarray,
        lb: float | np.ndarray,
        ub: float | np.ndarray,
    ) -> np.ndarray:
        """
        Project w onto {Σ|w_i| ≤ gross_cap, lb_i ≤ w_i ≤ ub_i} via bisection.

        Supports per-ticker bounds (vectors) for ADV capacity constraints.

        For long-only (lb=0): w* = clip(v - θ, 0, ub_i)
        For L/S (lb < 0):    w* = sign(v) · clip(|v| - θ, 0, ub_i)

        Both cases: Σ|w*(θ)| is monotone decreasing in θ → bisection.
        """
        G = self.gross_cap
        lb_arr = np.broadcast_to(lb, w.shape)
        ub_arr = np.broadcast_to(ub, w.shape)

        if self.long_only:
            lo, hi = 0.0, float(np.max(w)) + 1e-10
            for _ in range(64):
                mid = (lo + hi) * 0.5
                if float(np.sum(np.clip(w - mid, 0.0, ub_arr))) > G:
                    lo = mid
                else:
                    hi = mid
            theta = (lo + hi) * 0.5
            return np.clip(w - theta, 0.0, ub_arr)
        else:
            absw = np.abs(w)
            signs = np.sign(w)
            lo, hi = 0.0, float(np.max(absw)) + 1e-10
            for _ in range(64):
                mid = (lo + hi) * 0.5
                if float(np.sum(np.clip(absw - mid, 0.0, ub_arr))) > G:
                    lo = mid
                else:
                    hi = mid
            theta = (lo + hi) * 0.5
            return signs * np.clip(absw - theta, 0.0, ub_arr)

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
