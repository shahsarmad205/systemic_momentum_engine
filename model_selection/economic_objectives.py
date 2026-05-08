from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class EconomicObjectiveConfig:
    """Date-grouped portfolio utility objective for model selection."""

    objective: str = "long_short_spread"
    temperature: float = 0.20
    cost_penalty: float = 1.0
    turnover_penalty: float = 0.05
    l2: float = 1e-3
    learning_rate: float = 0.05
    max_iter: int = 150
    gradient_clip: float = 5.0
    random_state: int = 42


def _softmax(x: np.ndarray, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    z = np.asarray(x, dtype=float) / temp
    z = np.clip(z - np.nanmax(z), -50.0, 50.0)
    exp_z = np.exp(z)
    denom = float(exp_z.sum())
    if denom <= 1e-12 or not np.isfinite(denom):
        return np.ones(len(z), dtype=float) / max(len(z), 1)
    return exp_z / denom


def _signed_weights(scores: np.ndarray, objective: str, temperature: float) -> np.ndarray:
    path = str(objective or "long_short_spread").lower()
    if path == "long_only_overlay":
        return _softmax(scores, temperature)
    if path == "short_side":
        return -_softmax(-scores, temperature)
    long_w = _softmax(scores, temperature)
    short_w = _softmax(-scores, temperature)
    return 0.5 * long_w - 0.5 * short_w


def _portfolio_gradient(
    scores: np.ndarray,
    target: np.ndarray,
    cost: np.ndarray,
    *,
    objective: str,
    temperature: float,
    cost_penalty: float,
) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    path = str(objective or "long_short_spread").lower()
    y = np.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
    c = np.nan_to_num(cost, nan=0.0, posinf=0.0, neginf=0.0)

    if path == "long_only_overlay":
        long_w = _softmax(scores, temp)
        utility_vec = y - float(cost_penalty) * c
        avg = float(long_w @ utility_vec)
        return long_w * (utility_vec - avg) / temp

    if path == "short_side":
        short_w = _softmax(-scores, temp)
        utility_vec = y + float(cost_penalty) * c
        avg = float(short_w @ utility_vec)
        return short_w * (utility_vec - avg) / temp

    long_w = _softmax(scores, temp)
    short_w = _softmax(-scores, temp)
    long_vec = y - float(cost_penalty) * c
    short_vec = y + float(cost_penalty) * c
    grad_long = 0.5 * long_w * (long_vec - float(long_w @ long_vec)) / temp
    grad_short = 0.5 * short_w * (short_vec - float(short_w @ short_vec)) / temp
    return grad_long + grad_short


def _turnover_gradient(
    scores: np.ndarray,
    tickers: np.ndarray,
    prev_tickers: np.ndarray | None,
    prev_weights: np.ndarray | None,
    *,
    objective: str,
    temperature: float,
    turnover_penalty: float,
) -> np.ndarray:
    if prev_tickers is None or prev_weights is None or float(turnover_penalty) <= 0.0:
        return np.zeros(len(scores), dtype=float)
    signed = _signed_weights(scores, objective, temperature)
    prev_map = {str(t): float(w) for t, w in zip(prev_tickers, prev_weights, strict=False)}
    aligned_prev = np.asarray([prev_map.get(str(t), 0.0) for t in tickers], dtype=float)
    delta = signed - aligned_prev
    v = float(turnover_penalty) * delta / np.sqrt(delta * delta + 1e-8)

    temp = max(float(temperature), 1e-6)
    path = str(objective or "long_short_spread").lower()
    if path == "long_only_overlay":
        w = _softmax(scores, temp)
        return w * (v - float(w @ v)) / temp
    if path == "short_side":
        w = _softmax(-scores, temp)
        return w * (v - float(w @ v)) / temp

    long_w = _softmax(scores, temp)
    short_w = _softmax(-scores, temp)
    grad_long = 0.5 * long_w * (v - float(long_w @ v)) / temp
    grad_short = 0.5 * short_w * (v - float(short_w @ v)) / temp
    return grad_long + grad_short


class DateGroupedEconomicModel:
    """
    Deprecated portfolio-utility learner retained for research archaeology.

    This class intentionally fails closed by default. Alpha production models
    must emit cross-sectional scores only; expected implementation cost,
    turnover, and portfolio objectives belong in validation/construction.
    """

    def __init__(
        self,
        objective: str = "long_short_spread",
        temperature: float = 0.20,
        cost_penalty: float = 1.0,
        turnover_penalty: float = 0.05,
        l2: float = 1e-3,
        learning_rate: float = 0.05,
        max_iter: int = 150,
        gradient_clip: float = 5.0,
        random_state: int = 42,
        early_stopping: bool = False,
        patience: int = 10,
        min_delta: float = 1e-4,
        allow_portfolio_objective_training: bool = False,
    ) -> None:
        self.objective = objective
        self.temperature = temperature
        self.cost_penalty = cost_penalty
        self.turnover_penalty = turnover_penalty
        self.l2 = l2
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.gradient_clip = gradient_clip
        self.random_state = random_state
        self.early_stopping = early_stopping
        self.patience = patience
        self.min_delta = min_delta
        self.allow_portfolio_objective_training = allow_portfolio_objective_training
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0
        self.fit_info_: dict[str, Any] = {}
        self._date_groups: np.ndarray | None = None
        self._tickers: np.ndarray | None = None
        self._cost: np.ndarray | None = None

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "temperature": self.temperature,
            "cost_penalty": self.cost_penalty,
            "turnover_penalty": self.turnover_penalty,
            "l2": self.l2,
            "learning_rate": self.learning_rate,
            "max_iter": self.max_iter,
            "gradient_clip": self.gradient_clip,
            "random_state": self.random_state,
            "early_stopping": self.early_stopping,
            "patience": self.patience,
            "min_delta": self.min_delta,
            "allow_portfolio_objective_training": self.allow_portfolio_objective_training,
        }

    def set_params(self, **params: Any) -> "DateGroupedEconomicModel":
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def set_training_context(
        self,
        *,
        date_groups: np.ndarray | None = None,
        tickers: np.ndarray | None = None,
        cost: np.ndarray | None = None,
    ) -> "DateGroupedEconomicModel":
        self._date_groups = None if date_groups is None else np.asarray(date_groups)
        self._tickers = None if tickers is None else np.asarray(tickers).astype(str)
        self._cost = None if cost is None else np.asarray(cost, dtype=float)
        return self

    def fit(self, X: np.ndarray, y: np.ndarray, **kw: Any) -> "DateGroupedEconomicModel":
        if not bool(self.allow_portfolio_objective_training):
            raise RuntimeError(
                "DateGroupedEconomicModel embeds portfolio utility, cost, and "
                "turnover terms in training. It is not a valid AlphaModel. "
                "Use pure score models for alpha generation and apply costs/"
                "constraints in executable validation or portfolio construction."
            )
        x = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n, p = x.shape if x.ndim == 2 else (len(y_arr), 0)
        if p == 0 or n == 0:
            self.coef_ = np.zeros(p, dtype=float)
            self.intercept_ = 0.0
            return self

        date_groups = kw.get("_date_groups")
        tickers = kw.get("_tickers")
        cost = kw.get("_cost")
        if date_groups is None:
            date_groups = self._date_groups
        if tickers is None:
            tickers = self._tickers
        if cost is None:
            cost = self._cost

        if date_groups is None or len(date_groups) != n:
            date_groups = np.arange(n, dtype=int) // 500
        date_groups = np.asarray(date_groups)
        tickers = np.asarray(tickers).astype(str) if tickers is not None and len(tickers) == n else np.arange(n).astype(str)
        cost_arr = np.asarray(cost, dtype=float) if cost is not None and len(cost) == n else np.zeros(n, dtype=float)
        y_arr = np.nan_to_num(y_arr, nan=0.0, posinf=0.0, neginf=0.0)
        cost_arr = np.nan_to_num(cost_arr, nan=0.0, posinf=0.0, neginf=0.0).clip(min=0.0)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        rng = np.random.default_rng(int(self.random_state))
        coef = rng.normal(0.0, 1e-3, size=p)
        intercept = 0.0
        groups = [np.flatnonzero(date_groups == g) for g in np.unique(date_groups)]
        groups = [g for g in groups if len(g) >= 5]
        if not groups:
            self.coef_ = coef
            self.intercept_ = intercept
            return self

        lr0 = float(self.learning_rate)
        
        import time
        t0 = time.perf_counter()
        patience_counter = 0
        stopped_early = False
        grad_norm = 0.0
        weight_delta = 0.0
        iterations_used = 0
        
        for it in range(max(1, int(self.max_iter))):
            iterations_used = it + 1
            prev_coef = coef.copy()
            prev_intercept = intercept
            grad_coef = np.zeros(p, dtype=float)
            grad_intercept = 0.0
            prev_tickers: np.ndarray | None = None
            prev_weights: np.ndarray | None = None
            for idx in groups:
                xg = x[idx]
                yg = y_arr[idx]
                cg = cost_arr[idx]
                tg = tickers[idx]
                scores = xg @ coef + intercept
                scores = np.clip(scores, -25.0, 25.0)
                grad_s = _portfolio_gradient(
                    scores,
                    yg,
                    cg,
                    objective=str(self.objective),
                    temperature=float(self.temperature),
                    cost_penalty=float(self.cost_penalty),
                )
                grad_s -= _turnover_gradient(
                    scores,
                    tg,
                    prev_tickers,
                    prev_weights,
                    objective=str(self.objective),
                    temperature=float(self.temperature),
                    turnover_penalty=float(self.turnover_penalty),
                )
                grad_s = np.clip(grad_s, -float(self.gradient_clip), float(self.gradient_clip))
                grad_coef += xg.T @ grad_s
                grad_intercept += float(grad_s.sum())
                prev_tickers = tg
                prev_weights = _signed_weights(scores, str(self.objective), float(self.temperature))

            scale = 1.0 / max(len(groups), 1)
            grad_coef = grad_coef * scale - float(self.l2) * coef
            grad_intercept *= scale
            lr = lr0 / np.sqrt(1.0 + it / 25.0)
            coef += lr * np.clip(grad_coef, -float(self.gradient_clip), float(self.gradient_clip))
            intercept += lr * float(np.clip(grad_intercept, -float(self.gradient_clip), float(self.gradient_clip)))
            coef = np.nan_to_num(coef, nan=0.0, posinf=0.0, neginf=0.0)
            
            grad_norm = float(np.linalg.norm(grad_coef)) + abs(float(grad_intercept))
            weight_delta = float(np.linalg.norm(coef - prev_coef)) + abs(float(intercept - prev_intercept))
            
            if not np.isfinite(grad_norm) or not np.isfinite(weight_delta):
                coef = np.zeros_like(coef)
                intercept = 0.0
                stopped_early = True
                break
                
            if self.early_stopping:
                if grad_norm < self.min_delta and weight_delta < self.min_delta:
                    patience_counter += 1
                else:
                    patience_counter = 0
                    
                if patience_counter >= self.patience:
                    stopped_early = True
                    break

        t1 = time.perf_counter()
        time_elapsed_s = float(t1 - t0)
        time_per_iter = time_elapsed_s / max(1, iterations_used)
        estimated_runtime_saved_s = time_per_iter * max(0, int(self.max_iter) - iterations_used)

        self.coef_ = coef
        self.intercept_ = float(intercept)
        self.fit_info_ = {
            "iterations_used": int(iterations_used),
            "stopped_early": bool(stopped_early),
            "estimated_runtime_saved_s": round(estimated_runtime_saved_s, 3),
            "final_grad_norm": float(grad_norm),
            "final_weight_delta": float(weight_delta),
        }
        
        if self.early_stopping and stopped_early:
            import logging
            logging.info(
                f"[DateGroupedEconomicModel] Early stopping triggered at iter {iterations_used}/{self.max_iter}. "
                f"Grad norm: {grad_norm:.6f}, Weight delta: {weight_delta:.6f}. "
                f"Estimated time saved: {estimated_runtime_saved_s:.2f}s."
            )
            
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        x = np.asarray(X, dtype=float)
        if self.coef_ is None:
            return np.zeros(len(x), dtype=float)
        return np.nan_to_num(x @ self.coef_ + float(self.intercept_), nan=0.0, posinf=0.0, neginf=0.0)
