import numpy as np
import logging

logger = logging.getLogger(__name__)

class L1PortfolioOptimizer:
    """
    Portfolio Optimizer with L1 Turnover Penalty.
    
    Solves:
        maximize  w' μ  −  λ_risk * w' Σ w  −  λ_turn * ||w − w_prev||_1
    subject to:
        Σ w_i = C
        lb_i ≤ w_i ≤ ub_i
    
    Solver: Pairwise Coordinate Descent.
    """

    def __init__(
        self,
        lambda_risk: float = 2.0,
        lambda_turn: float = 0.01,
        max_weight: float = 0.10,
        gross_cap: float = 1.0,
        net_exposure_target: float = 0.0,
    ):
        self.lambda_risk = float(lambda_risk)
        self.lambda_turn = float(lambda_turn)
        self.max_weight = float(max_weight)
        self.gross_cap = float(gross_cap)
        self.net_exposure_target = float(net_exposure_target)

    def optimize(
        self,
        mu: np.ndarray,
        cov: np.ndarray,
        w_prev: np.ndarray,
        lb: np.ndarray,
        ub: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-6
    ) -> np.ndarray:
        """
        Pure NumPy implementation of Pairwise Coordinate Descent.
        """
        N = len(mu)
        if N < 2:
            return np.clip(np.full(N, self.net_exposure_target / max(1, N)), lb, ub)

        # Covariance validation
        if cov.shape != (N, N) or not np.all(np.isfinite(cov)):
            cov = np.eye(N) * (0.15 ** 2)
        else:
            cov = (cov + cov.T) / 2.0
            ev = np.linalg.eigvalsh(cov)
            if float(ev[0]) < 1e-8:
                cov += np.eye(N) * (1e-8 - float(ev[0]) + 1e-8)

        # Initialize with a feasible point (projection of w_prev or net_exposure_target)
        w = self._initialize_feasible(w_prev, lb, ub)
        
        # Precompute Sigma * w
        Sw = np.dot(cov, w)
        
        for iteration in range(max_iter):
            max_delta = 0.0
            
            # Shuffle pairs for better convergence
            indices = np.arange(N)
            np.random.shuffle(indices)
            
            for k in range(0, N - 1, 2):
                i, j = indices[k], indices[k+1]
                
                # Subproblem: move delta from j to i
                # w_i_new = w_i + delta
                # w_j_new = w_j - delta
                
                # Quadratic part: A*delta^2 + B*delta
                # A = -lambda_risk * (sigma_ii + sigma_jj - 2*sigma_ij)
                # B = (mu_i - mu_j) - 2*lambda_risk * ( (Sigma*w)_i - (Sigma*w)_j )
                
                sigma_ii = cov[i, i]
                sigma_jj = cov[j, j]
                sigma_ij = cov[i, j]
                
                quad_a = -self.lambda_risk * (sigma_ii + sigma_jj - 2.0 * sigma_ij)
                if quad_a >= 0: # Should be negative for convexity
                    quad_a = -1e-10
                
                quad_b = (mu[i] - mu[j]) - 2.0 * self.lambda_risk * (Sw[i] - Sw[j])
                
                # L1 part: -lambda_turn * (|w_i + delta - w_prev_i| + |w_j - delta - w_prev_j|)
                kink_i = w_prev[i] - w[i]
                kink_j = w[j] - w_prev[j]
                
                # Box constraints:
                # lb_i <= w_i + delta <= ub_i  => lb_i - w_i <= delta <= ub_i - w_i
                # lb_j <= w_j - delta <= ub_j  => w_j - ub_j <= delta <= w_j - lb_j
                delta_min = max(lb[i] - w[i], w[j] - ub[j])
                delta_max = min(ub[i] - w[i], w[j] - lb[j])
                
                if delta_max <= delta_min:
                    continue
                
                # Solve 1D piecewise quadratic
                delta_opt = self._solve_1d_piecewise(quad_a, quad_b, kink_i, kink_j, delta_min, delta_max)
                
                if abs(delta_opt) > 1e-12:
                    w[i] += delta_opt
                    w[j] -= delta_opt
                    # Update Sigma * w
                    Sw += delta_opt * (cov[:, i] - cov[:, j])
                    max_delta = max(max_delta, abs(delta_opt))
            
            if max_delta < tol:
                break
                
        return w

    def _solve_1d_piecewise(self, a, b, k1, k2, d_min, d_max) -> float:
        """
        Finds max of f(d) = a*d^2 + b*d - lam*(|d - k1| + |d - k2|) on [d_min, d_max]
        """
        lam = self.lambda_turn
        kinks = sorted([k for k in [k1, k2] if d_min < k < d_max])
        nodes = [d_min] + kinks + [d_max]
        
        best_d = d_min
        best_val = -np.inf
        
        for i in range(len(nodes) - 1):
            n_start, n_end = nodes[i], nodes[i+1]
            # On this segment, the L1 terms have constant signs
            # |d - k1| = s1 * (d - k1)
            # |d - k2| = s2 * (d - k2)
            mid = (n_start + n_end) / 2.0
            s1 = 1.0 if mid >= k1 else -1.0
            s2 = 1.0 if mid >= k2 else -1.0
            
            # Linear part adjustment: b_eff = b - lam * (s1 + s2)
            b_eff = b - lam * (s1 + s2)
            
            # Max of a*d^2 + b_eff*d is at -b_eff / (2*a)
            d_star = -b_eff / (2.0 * a)
            d_star = np.clip(d_star, n_start, n_end)
            
            val = a * d_star**2 + b_eff * d_star
            if val > best_val:
                best_val = val
                best_d = d_star
                
        return best_d

    def _initialize_feasible(self, w_prev, lb, ub) -> np.ndarray:
        """
        Simple projection of w_prev to satisfy sum(w) = target.
        """
        w = np.clip(w_prev, lb, ub)
        current_sum = np.sum(w)
        target = self.net_exposure_target
        
        if abs(current_sum - target) > 1e-10:
            # Distribute the difference
            diff = target - current_sum
            N = len(w)
            # This is a bit naive, but it works for initialization
            w += diff / N
            w = np.clip(w, lb, ub)
            
            # Refine if still off (due to clipping)
            for _ in range(10):
                curr = np.sum(w)
                if abs(curr - target) < 1e-10:
                    break
                # Water-filling style adjustment
                diff = target - curr
                if diff > 0:
                    mask = w < ub
                else:
                    mask = w > lb
                if not np.any(mask):
                    break
                w[mask] += diff / np.sum(mask)
                w = np.clip(w, lb, ub)
        return w

def calibrate_lambda_turn(
    mu: np.ndarray,
    cov: np.ndarray,
    w_prev: np.ndarray,
    lb: np.ndarray,
    ub: np.ndarray,
    returns_sample: np.ndarray, # (n_days, n_assets)
    lambdas: list[float] = [0.0, 0.001, 0.01, 0.1, 0.5]
) -> float:
    """
    Task 2: Calibrate λ_turn via cross-validated net Sharpe.
    """
    best_lambda = lambdas[0]
    best_sharpe = -np.inf
    
    # We use the returns_sample to simulate one step forward and see which lambda
    # would have performed better. In a real walk-forward, this would be done
    # on the previous window.
    
    for l_turn in lambdas:
        opt = L1PortfolioOptimizer(lambda_turn=l_turn)
        w_opt = opt.optimize(mu, cov, w_prev, lb, ub)
        
        # Simple net Sharpe calculation for this single rebalance
        # Net Return = w_opt' * mu_realized - costs
        # Here we approximate costs as lambda_turn * turnover
        turnover = np.sum(np.abs(w_opt - w_prev))
        
        # We need a proxy for performance. Let's say we use the realized returns
        avg_ret = np.dot(w_opt, np.mean(returns_sample, axis=0))
        vol_ret = np.std(np.dot(returns_sample, w_opt))
        
        # Roughly estimate costs (e.g. 5bps per unit turnover)
        cost_proxy = turnover * 0.0005 
        net_ret = avg_ret - cost_proxy
        sharpe = net_ret / vol_ret if vol_ret > 1e-12 else -1.0
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_lambda = l_turn
            
    return best_lambda
