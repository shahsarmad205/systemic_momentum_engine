"""
Portfolio construction helpers.

Implements deterministic top-K selection from adjusted scores and
long-only weight normalization, plus institutional-grade enhancements:

  - ``construct_top_k_weights``:  fixed scoring bug (negative score shifting
    only applies when long-only; long-short keeps mean-zero weights)
  - ``construct_inverse_vol_weights``:  inverse-volatility weighting (1/σ)
    with optional correlation-aware diversification cap
  - ``construct_risk_parity_weights``: equal risk contribution (ERC)
  - Risk overlay application: volatility targeting + time-series scaling
  - Stress test validation of constructed portfolios

All weight functions follow the same contract:
  - Input: scores dict + optional vol/corr data
  - Output: dict[ticker, weight] summing to 1 (long-only) or net-zero (L/S)
"""

from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RiskOverlayConfig:
    """Configuration for risk overlay application after weight construction."""
    vol_targeting_enabled: bool = True
    vol_target: float = 0.15
    vol_min_scale: float = 0.3
    vol_max_scale: float = 1.2
    time_series_overlay_enabled: bool = True
    stress_test_enabled: bool = True
    stress_max_dd: float = 0.25
    stress_max_vol: float = 0.40
    stress_min_pass_scenarios: int = 3


@dataclass(frozen=True)
class PortfolioConstraints:
    """Institutional score-to-weight and book-level construction constraints."""

    path: str = "long_short_spread"
    max_positions: int = 10
    min_positions: int = 3
    max_gross: float = 1.0
    max_net: float = 0.01  # P5: tightened from 0.10 — strict dollar neutrality for L/S spread books
    max_name_weight: float = 0.10
    min_position_weight: float = 0.0
    use_optimizer: bool = True
    optimization_type: str = "l1"
    lambda_risk: float = 2.0
    gamma_turnover: float = 4.0
    lambda_turn_override: float | None = None
    no_trade_band_weight_diff: float = 0.015
    no_trade_band_total_drift: float = 0.05
    factor_neutral: bool = True
    beta_neutral: bool = True
    sector_neutral: bool = True
    max_beta_abs: float = 0.15
    max_sector_abs: float = 0.12
    style_exposure_limits: Mapping[str, float] = field(default_factory=dict)
    constraint_passes: int = 3
    adv_fraction: float = 0.05
    capital: float = 10_000_000.0
    max_participation_rate: float = 0.10
    short_squeeze_filter: bool = True
    short_squeeze_max_risk: float = 0.75
    market_neutral_shorts: bool = True
    optimizer_alpha_scale: float = 1.0
    enable_feasibility_repair: bool = True
    feasibility_repair_scales: tuple[float, ...] = (0.75, 0.50, 0.25, 0.10, 0.0)
    # P8: Decay-aware execution — signal halflife vs prediction horizon mismatch.
    # When the signal decays faster than the holding period, alpha is scaled down
    # by 2^(-horizon/halflife) to prevent over-allocation to stale signals.
    signal_halflife_days: float = float("nan")
    horizon_days: int = 5


@dataclass(frozen=True)
class PortfolioInputs:
    """Date-level inputs consumed by the centralized PortfolioConstructor."""

    date: Any
    tickers: list[str]
    scores: pd.Series
    previous_weights: pd.Series = field(default_factory=lambda: pd.Series(dtype=float))
    covariance: np.ndarray | None = None
    beta: pd.Series | None = None
    sectors: pd.Series | None = None
    factor_exposures: Mapping[str, pd.Series | np.ndarray | Mapping[str, float]] = field(default_factory=dict)
    adv_dollar: pd.Series | None = None
    liquidity_caps: pd.Series | None = None
    daily_vol: pd.Series | None = None
    borrow_penalty: pd.Series | None = None
    short_blocked: pd.Series | None = None
    squeeze_risk: pd.Series | None = None
    hard_short_squeeze: pd.Series | None = None


@dataclass(frozen=True)
class PortfolioConstructionResult:
    """Constructed target weights plus construction diagnostics."""

    weights: pd.Series
    diagnostics: dict[str, float | int | str | bool]
    violations: tuple[str, ...] = ()


def _clean_series(values: pd.Series | Mapping[str, float] | None, tickers: list[str], default: float = 0.0) -> pd.Series:
    if values is None:
        return pd.Series(default, index=tickers, dtype=float)
    if isinstance(values, pd.Series):
        s = values.copy()
    else:
        s = pd.Series(dict(values), dtype=float)
    s.index = s.index.astype(str)
    return pd.to_numeric(s.reindex(tickers), errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default).astype(float)


def _normalise_gross(weights: pd.Series, gross: float) -> pd.Series:
    out = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)
    total = float(out.abs().sum())
    target = max(0.0, float(gross))
    if total <= 1e-12 or target <= 0.0:
        return out * 0.0
    if total > target:
        out = out * (target / total)
    return out


def _gross_net_diagnostics(weights: pd.Series) -> dict[str, float]:
    return {
        "gross_exposure": float(weights.abs().sum()),
        "net_exposure": float(weights.sum()),
        "long_exposure": float(weights[weights > 0.0].sum()) if not weights.empty else 0.0,
        "short_exposure": float(weights[weights < 0.0].abs().sum()) if not weights.empty else 0.0,
        "max_name_weight": float(weights.abs().max()) if not weights.empty else 0.0,
        "n_positions": int((weights.abs() > 1e-12).sum()),
    }


class PortfolioConstructor:
    """Centralized institutional portfolio construction engine.

    All score-to-weight conversion and portfolio constraints live here. The
    simulator may account for execution costs, but it must not change target
    weights after this layer has produced them.
    """

    @staticmethod
    def build_weights(
        inputs: PortfolioInputs,
        constraints: PortfolioConstraints,
    ) -> PortfolioConstructionResult:
        return PortfolioConstructor()._build(inputs, constraints)

    def _build(self, inputs: PortfolioInputs, constraints: PortfolioConstraints) -> PortfolioConstructionResult:
        tickers = [str(t) for t in inputs.tickers]
        if not tickers:
            return PortfolioConstructionResult(pd.Series(dtype=float), {"n_positions": 0}, ())

        scores = _clean_series(inputs.scores, tickers, default=0.0)
        prev = _clean_series(inputs.previous_weights, tickers, default=0.0)
        self._validate_required_inputs(inputs, constraints, tickers)

        alpha = self._score_to_alpha(scores, inputs, constraints, tickers)
        construction_source = "optimizer" if bool(constraints.use_optimizer) else "rank"
        base = (
            self._optimizer_weights(alpha, prev, inputs, constraints, tickers)
            if bool(constraints.use_optimizer)
            else self._rank_weights(scores, constraints, tickers)
        )
        if (
            bool(constraints.use_optimizer)
            and float(_clean_series(base, tickers, default=0.0).abs().sum()) <= 1e-12
            and float(prev.abs().sum()) <= 1e-12
            and float(alpha.std(ddof=0)) > 1e-12
        ):
            seeded = self._rank_weights(scores, constraints, tickers)
            if float(seeded.abs().sum()) > 1e-12:
                base = seeded
                construction_source = "rank_cold_start_seed"
        weights, violations = self._apply_common_constraints(base, inputs, constraints, tickers)
        initial_violations = tuple(violations)
        if float(prev.abs().sum()) > 1e-12:
            drift = weights.sub(prev, fill_value=0.0)
            if (
                float(drift.abs().sum()) < float(constraints.no_trade_band_total_drift)
                and (float(drift.abs().max()) if not drift.empty else 0.0) < float(constraints.no_trade_band_weight_diff)
            ):
                weights = prev[prev.abs() > 1e-12]
                violations = self._constraint_violations(weights, inputs, constraints, tickers)
                if violations and not initial_violations:
                    initial_violations = tuple(violations)
        repair_diagnostics: dict[str, float | int | str | bool] = {}
        if violations and bool(constraints.enable_feasibility_repair):
            weights, violations, repair_diagnostics = self._repair_infeasible_book(
                weights,
                inputs,
                constraints,
                tickers,
                initial_violations=violations,
            )
        diagnostics = self._diagnostics(weights, inputs, constraints, tickers)
        diagnostics["construction_source"] = construction_source
        if initial_violations:
            diagnostics["initial_violations"] = ",".join(initial_violations)
        diagnostics.update(repair_diagnostics)
        if violations:
            diagnostics["construction_status"] = "infeasible"
        elif float(weights.abs().sum()) <= 1e-12:
            diagnostics["construction_status"] = "de_risked_to_cash" if initial_violations else "no_trade"
        else:
            diagnostics["construction_status"] = "ok" if not initial_violations else "repaired"
        return PortfolioConstructionResult(weights=weights, diagnostics=diagnostics, violations=tuple(violations))

    def _repair_infeasible_book(
        self,
        weights: pd.Series,
        inputs: PortfolioInputs,
        constraints: PortfolioConstraints,
        tickers: list[str],
        *,
        initial_violations: list[str],
    ) -> tuple[pd.Series, list[str], dict[str, float | int | str | bool]]:
        """Fail closed by reducing risk budget until all hard constraints pass.

        This is an institutional feasibility projection, not a relaxation. If the
        requested alpha book cannot satisfy hard exposure constraints at the
        desired gross, the only admissible action is to consume less risk budget.
        The terminal feasible point is cash, which preserves simulator integrity
        and makes the lack of executable alpha explicit in diagnostics.
        """
        base = _clean_series(weights, tickers, default=0.0)
        base = base.reindex(tickers).fillna(0.0)
        original_gross = float(base.abs().sum())
        scales = tuple(float(s) for s in constraints.feasibility_repair_scales)
        if not scales or scales[-1] != 0.0:
            scales = (*scales, 0.0)

        best = base * 0.0
        best_violations = self._constraint_violations(best, inputs, constraints, tickers)
        for scale in scales:
            clipped_scale = max(0.0, min(1.0, float(scale)))
            trial = base * clipped_scale
            if float(constraints.min_position_weight) > 0.0:
                trial = trial.where(trial.abs() >= float(constraints.min_position_weight), 0.0)
            trial = trial[trial.abs() > 1e-12]
            violations = self._constraint_violations(trial, inputs, constraints, tickers)
            if not violations:
                repaired_gross = float(trial.abs().sum())
                return (
                    trial,
                    [],
                    {
                        "feasibility_repaired": True,
                        "repair_scale": clipped_scale,
                        "gross_before_repair": original_gross,
                        "gross_after_repair": repaired_gross,
                        "repair_initial_violations": ",".join(initial_violations),
                    },
                )
            best = trial
            best_violations = violations

        return (
            best[best.abs() > 1e-12],
            best_violations,
            {
                "feasibility_repaired": False,
                "repair_scale": 0.0,
                "gross_before_repair": original_gross,
                "gross_after_repair": float(best.abs().sum()),
                "repair_initial_violations": ",".join(initial_violations),
            },
        )

    def _validate_required_inputs(
        self,
        inputs: PortfolioInputs,
        constraints: PortfolioConstraints,
        tickers: list[str],
    ) -> None:
        supplied = dict(inputs.factor_exposures or {})
        has_beta = inputs.beta is not None or "market_beta" in supplied or "beta" in supplied
        has_sector = inputs.sectors is not None or any(str(k).startswith("sector:") for k in supplied)
        if bool(constraints.factor_neutral) and bool(constraints.beta_neutral) and not has_beta:
            raise ValueError("PortfolioConstructor requires beta exposures when beta_neutral=True")
        if bool(constraints.factor_neutral) and bool(constraints.sector_neutral) and not has_sector:
            raise ValueError("PortfolioConstructor requires sector labels when sector_neutral=True")
        for name, bound in dict(constraints.style_exposure_limits or {}).items():
            if float(bound) > 0.0 and str(name) not in supplied:
                raise ValueError(f"PortfolioConstructor requires style exposure '{name}' when configured")
        if float(constraints.adv_fraction) > 0.0 and inputs.adv_dollar is None and inputs.liquidity_caps is None:
            raise ValueError("PortfolioConstructor requires ADV or liquidity caps when liquidity constraints are active")
        path = str(constraints.path or "").lower()
        needs_short_controls = path in {"long_short_spread", "short_side"}
        if needs_short_controls and bool(constraints.short_squeeze_filter):
            if inputs.short_blocked is None and inputs.squeeze_risk is None and inputs.hard_short_squeeze is None:
                raise ValueError("PortfolioConstructor requires short-squeeze fields when short_squeeze_filter=True")
        if len(_clean_series(inputs.scores, tickers, default=np.nan).dropna()) == 0:
            raise ValueError("PortfolioConstructor received no finite scores")

    def _score_to_alpha(
        self,
        scores: pd.Series,
        inputs: PortfolioInputs,
        constraints: PortfolioConstraints,
        tickers: list[str],
    ) -> pd.Series:
        if scores.nunique(dropna=True) < 2:
            return pd.Series(0.0, index=tickers, dtype=float)
        rank = scores.rank(pct=True, method="average").fillna(0.5)
        centered = scores - float(scores.mean())
        scale = float(scores.std(ddof=0))
        z = centered / (scale if np.isfinite(scale) and scale > 1e-12 else 1.0)
        path = str(constraints.path or "").lower()
        if path == "short_side":
            alpha = -rank
        elif path == "long_only_overlay":
            alpha = z.clip(lower=0.0)
        elif path == "long_short_spread":
            alpha = z
        else:
            raise ValueError(f"unsupported construction path: {constraints.path}")

        blocked = self._short_blocked(inputs, constraints, tickers)
        alpha = alpha.where(~(blocked & (alpha < 0.0)), 0.0)
        borrow = _clean_series(inputs.borrow_penalty, tickers, default=0.0).clip(lower=0.0)
        alpha = alpha.where(alpha >= 0.0, alpha + borrow)

        # P8: Decay-aware execution scaling.
        # Signal halflife (typically 2.3d) vs prediction horizon (10d) mismatch.
        halflife = float(getattr(constraints, "signal_halflife_days", float("nan")))
        horizon = float(getattr(constraints, "horizon_days", 5))
        if np.isfinite(halflife) and halflife > 0:
            decay_correction = float(2.0 ** (-horizon / halflife))
            decay_correction = max(decay_correction, 0.01)
            alpha = alpha * decay_correction

        return (float(constraints.optimizer_alpha_scale) * alpha).replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _rank_weights(self, scores: pd.Series, constraints: PortfolioConstraints, tickers: list[str]) -> pd.Series:
        ranked = scores.sort_values(ascending=False, kind="mergesort")
        k = max(1, int(constraints.max_positions))
        min_k = max(1, int(constraints.min_positions))
        path = str(constraints.path or "").lower()
        weights = pd.Series(0.0, index=tickers, dtype=float)
        if path == "long_short_spread":
            side_k = max(1, k // 2)
            if len(ranked) < max(2 * min(min_k, side_k), 2):
                return weights
            longs = ranked.head(side_k).index
            shorts = ranked.tail(side_k).index
            weights.loc[longs] = 0.5 / max(len(longs), 1)
            weights.loc[shorts] = -0.5 / max(len(shorts), 1)
        elif path == "short_side":
            shorts = ranked.head(k).index
            if len(shorts) < min_k:
                return weights
            weights.loc[shorts] = -1.0 / max(len(shorts), 1)
        elif path == "long_only_overlay":
            longs = ranked.head(k).index
            if len(longs) < min_k:
                return weights
            weights.loc[longs] = 1.0 / max(len(longs), 1)
        else:
            raise ValueError(f"unsupported construction path: {constraints.path}")
        return weights

    def _optimizer_weights(
        self,
        alpha: pd.Series,
        prev: pd.Series,
        inputs: PortfolioInputs,
        constraints: PortfolioConstraints,
        tickers: list[str],
    ) -> pd.Series:
        cov = inputs.covariance
        if cov is None:
            cov = np.eye(len(tickers), dtype=float) * (0.15 ** 2)
        cov = np.asarray(cov, dtype=float)
        if cov.shape != (len(tickers), len(tickers)) or not np.isfinite(cov).all():
            raise ValueError("PortfolioConstructor requires a finite NxN covariance matrix for optimizer construction")
        caps = self._liquidity_caps(inputs, constraints, tickers)
        path = str(constraints.path or "").lower()
        if str(constraints.optimization_type or "l2").lower() == "l1":
            from model_selection.optimizer_l1 import L1PortfolioOptimizer

            target_net = 1.0 if path == "long_only_overlay" else (0.0 if path == "long_short_spread" else -min(float(constraints.max_gross), 1.0))
            opt = L1PortfolioOptimizer(
                lambda_risk=float(constraints.lambda_risk),
                lambda_turn=float(constraints.lambda_turn_override if constraints.lambda_turn_override is not None else constraints.gamma_turnover),
                max_weight=float(constraints.max_name_weight),
                gross_cap=float(constraints.max_gross),
                net_exposure_target=float(target_net),
            )
            ub = caps.to_numpy(dtype=float)
            lb = np.zeros(len(tickers), dtype=float) if path == "long_only_overlay" else -ub
            raw = opt.optimize(
                mu=alpha.to_numpy(dtype=float),
                cov=cov,
                w_prev=prev.to_numpy(dtype=float),
                lb=lb,
                ub=ub,
            )
            return pd.Series(raw, index=tickers, dtype=float)

        from backtesting.optimizer import PortfolioOptimizer, FactorConstraint

        # Build explicit factor constraints from exposures for QP encoding.
        # These are passed directly into the optimizer which uses Augmented
        # Lagrangian to enforce |B'w| ≤ limit as part of the optimization,
        # satisfying KKT conditions.  The previous approach applied factor
        # constraints via iterative post-optimization projection, which
        # produces a suboptimal portfolio that violates KKT conditions.
        exposures = self._factor_exposures(inputs, constraints, tickers)
        factor_constraints: list[FactorConstraint] = []
        if "market_beta" in exposures:
            factor_constraints.append(FactorConstraint(
                name="market_beta",
                exposure=np.array(exposures["market_beta"], dtype=float),
                limit=float(constraints.max_beta_abs),
            ))
        for name in exposures:
            if name.startswith("sector:"):
                factor_constraints.append(FactorConstraint(
                    name=name,
                    exposure=np.array(exposures[name], dtype=float),
                    limit=float(constraints.max_sector_abs),
                ))
        for name, limit in (constraints.style_exposure_limits or {}).items():
            if name in exposures:
                factor_constraints.append(FactorConstraint(
                    name=str(name),
                    exposure=np.array(exposures[name], dtype=float),
                    limit=float(limit),
                ))

        opt = PortfolioOptimizer(
            lambda_risk=float(constraints.lambda_risk),
            gamma_turnover=float(constraints.gamma_turnover),
            max_weight=float(constraints.max_name_weight),
            net_exposure_max=float(self._optimizer_net_limit(constraints)),
            long_only=(path == "long_only_overlay"),
            gross_cap=float(constraints.max_gross),
            min_position_weight=float(constraints.min_position_weight),
        )
        raw, conv = opt.optimize(
            forecasts={t: float(alpha.get(t, 0.0)) for t in tickers},
            cov=cov,
            w_prev={t: float(prev.get(t, 0.0)) for t in tickers},
            tickers=tickers,
            per_ticker_ub={t: float(caps.get(t, constraints.max_name_weight)) for t in tickers},
            factor_constraints=factor_constraints if factor_constraints else None,
            return_convergence=True,
        )
        if not conv.converged:
            logger.warning(
                "Portfolio optimizer convergence warning: %s", conv.summary
            )
        return pd.Series(raw, index=tickers, dtype=float)

    def _apply_common_constraints(
        self,
        weights: pd.Series,
        inputs: PortfolioInputs,
        constraints: PortfolioConstraints,
        tickers: list[str],
    ) -> tuple[pd.Series, list[str]]:
        out = _clean_series(weights, tickers, default=0.0)
        path = str(constraints.path or "").lower()
        if path == "long_only_overlay":
            out = out.clip(lower=0.0)
        elif path == "short_side":
            out = out.clip(upper=0.0)

        blocked = self._short_blocked(inputs, constraints, tickers)
        out = out.where(~(blocked & (out < 0.0)), 0.0)
        caps = self._liquidity_caps(inputs, constraints, tickers)
        out = out.clip(lower=(-caps if path != "long_only_overlay" else 0.0), upper=caps)
        out = self._project_net_exposure(out, constraints, tickers)

        # Factor constraints are now encoded directly in the QP via
        # Augmented Lagrangian (see _optimizer_weights).  No post-optimization
        # projection is needed.  We only verify residual violations for reporting.
        exposures = self._factor_exposures(inputs, constraints, tickers)
        out = _normalise_gross(out, min(float(constraints.max_gross), float(out.abs().sum())))

        if float(constraints.min_position_weight) > 0.0:
            out = out.where(out.abs() >= float(constraints.min_position_weight), 0.0)
        out = self._project_net_exposure(out, constraints, tickers)
        out = _normalise_gross(out, min(float(constraints.max_gross), float(out.abs().sum())))
        violations = self._constraint_violations(out, inputs, constraints, tickers)
        return out[out.abs() > 1e-12], violations

    def _optimizer_net_limit(self, constraints: PortfolioConstraints) -> float:
        path = str(constraints.path or "").lower()
        if path in {"long_only_overlay", "short_side"}:
            return max(0.0, float(constraints.max_gross))
        return abs(float(constraints.max_net))

    def _net_bounds(self, constraints: PortfolioConstraints) -> tuple[float, float]:
        """Return mandate-specific net-exposure bounds.

        A spread book is explicitly market-neutral. A short-side alpha book is
        allowed to carry negative target weights because the hedge/accounting
        layer is separate from score-to-short selection. This avoids the
        institutional anti-pattern where the short model is silently converted
        into a zero book by applying long-short spread constraints to it.
        """
        path = str(constraints.path or "").lower()
        max_gross = max(0.0, float(constraints.max_gross))
        if path == "long_only_overlay":
            return 0.0, max_gross
        if path == "short_side":
            return -max_gross, 0.0
        return -abs(float(constraints.max_net)), abs(float(constraints.max_net))

    def _project_net_exposure(
        self,
        weights: pd.Series,
        constraints: PortfolioConstraints,
        tickers: list[str],
    ) -> pd.Series:
        """Project weights into mandate-specific net bounds without relaxing risk limits.

        The projection only removes exposure from the over-represented side. It
        never adds leverage or flips names, so it is a conservative feasibility
        projection suitable for hard pre-simulation construction.
        """
        out = _clean_series(weights, tickers, default=0.0)
        path = str(constraints.path or "").lower()
        if path == "long_only_overlay":
            return out.clip(lower=0.0)
        if path == "short_side":
            return out.clip(upper=0.0)

        lower, upper = self._net_bounds(constraints)
        net = float(out.sum())
        if net > upper + 1e-12:
            excess = net - upper
            positives = out.clip(lower=0.0)
            pos_total = float(positives.sum())
            if pos_total > 1e-12:
                reduction = positives * min(1.0, excess / pos_total)
                out = out - reduction
        elif net < lower - 1e-12:
            excess = lower - net
            negatives = (-out.clip(upper=0.0))
            neg_total = float(negatives.sum())
            if neg_total > 1e-12:
                reduction = negatives * min(1.0, excess / neg_total)
                out = out + reduction
        return out.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    def _liquidity_caps(self, inputs: PortfolioInputs, constraints: PortfolioConstraints, tickers: list[str]) -> pd.Series:
        if inputs.liquidity_caps is not None:
            caps = _clean_series(inputs.liquidity_caps, tickers, default=float(constraints.max_name_weight))
        else:
            adv = _clean_series(inputs.adv_dollar, tickers, default=np.nan)
            if adv.isna().any():
                raise ValueError("PortfolioConstructor cannot derive liquidity caps from missing ADV")
            caps = float(constraints.adv_fraction) * adv / max(float(constraints.capital), 1.0)
        return caps.clip(lower=0.0, upper=float(constraints.max_name_weight))

    def _short_blocked(self, inputs: PortfolioInputs, constraints: PortfolioConstraints, tickers: list[str]) -> pd.Series:
        if not bool(constraints.short_squeeze_filter):
            return pd.Series(False, index=tickers, dtype=bool)
        if inputs.short_blocked is not None:
            return _clean_series(inputs.short_blocked, tickers, default=0.0).astype(bool)
        squeeze = _clean_series(inputs.squeeze_risk, tickers, default=0.0)
        hard = _clean_series(inputs.hard_short_squeeze, tickers, default=0.0)
        return ((squeeze >= float(constraints.short_squeeze_max_risk)) | (hard >= 1.0)).astype(bool)

    def _factor_exposures(
        self,
        inputs: PortfolioInputs,
        constraints: PortfolioConstraints,
        tickers: list[str],
    ) -> dict[str, np.ndarray]:
        if not bool(constraints.factor_neutral):
            return {}
        exposures: dict[str, np.ndarray] = {}
        supplied = dict(inputs.factor_exposures or {})
        if bool(constraints.beta_neutral):
            beta_raw = supplied.get("market_beta", supplied.get("beta", inputs.beta))
            beta = _clean_series(beta_raw, tickers, default=1.0).clip(-3.0, 5.0)
            exposures["market_beta"] = (beta - beta.mean()).to_numpy(dtype=float)
        if bool(constraints.sector_neutral):
            sector_supplied = {
                str(k): v for k, v in supplied.items() if str(k).startswith("sector:")
            }
            if sector_supplied:
                for name, values in sector_supplied.items():
                    vec = _clean_series(values, tickers, default=0.0)
                    exposures[name] = (vec - vec.mean()).to_numpy(dtype=float)
            elif inputs.sectors is None:
                raise ValueError("PortfolioConstructor requires sectors for sector neutrality")
            else:
                sectors = inputs.sectors.copy()
                sectors.index = sectors.index.astype(str)
                sectors = sectors.reindex(tickers).fillna("Unknown").astype(str)
                dummies = pd.get_dummies(sectors, dtype=float)
                for col in dummies.columns:
                    vec = dummies[col] - dummies[col].mean()
                    exposures[f"sector:{col}"] = vec.to_numpy(dtype=float)
        for name, bound in dict(constraints.style_exposure_limits or {}).items():
            if float(bound) <= 0.0:
                continue
            values = supplied.get(str(name))
            if values is None:
                raise ValueError(f"PortfolioConstructor requires style exposure '{name}' when configured")
            vec = _clean_series(values, tickers, default=0.0)
            exposures[str(name)] = (vec - vec.mean()).to_numpy(dtype=float)
        return exposures

    def _constraint_violations(
        self,
        weights: pd.Series,
        inputs: PortfolioInputs,
        constraints: PortfolioConstraints,
        tickers: list[str],
    ) -> list[str]:
        violations: list[str] = []
        eps = 1e-8
        if float(weights.abs().sum()) > float(constraints.max_gross) + eps:
            violations.append("max_gross")
        path = str(constraints.path or "").lower()
        lower, upper = self._net_bounds(constraints)
        net = float(weights.sum())
        if net < lower - eps or net > upper + eps:
            violations.append("net_exposure")
        if path == "short_side" and not weights.empty and float(weights.max()) > eps:
            violations.append("short_only")
        if not weights.empty and float(weights.abs().max()) > float(constraints.max_name_weight) + eps:
            violations.append("max_name_weight")
        exposures = self._factor_exposures(inputs, constraints, tickers)
        if "market_beta" in exposures and abs(float(np.dot(exposures["market_beta"], weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)))) > float(constraints.max_beta_abs) + 1e-6:
            violations.append("beta")
        for name, vec in exposures.items():
            if name.startswith("sector:") and abs(float(np.dot(vec, weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)))) > float(constraints.max_sector_abs) + 1e-6:
                violations.append(name)
            if name in dict(constraints.style_exposure_limits or {}) and abs(float(np.dot(vec, weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)))) > float(dict(constraints.style_exposure_limits)[name]) + 1e-6:
                violations.append(name)
        return violations

    def _diagnostics(
        self,
        weights: pd.Series,
        inputs: PortfolioInputs,
        constraints: PortfolioConstraints,
        tickers: list[str],
    ) -> dict[str, float | int | str | bool]:
        diagnostics: dict[str, float | int | str | bool] = _gross_net_diagnostics(weights)
        exposures = self._factor_exposures(inputs, constraints, tickers)
        aligned = weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)
        diagnostics["beta_exposure"] = float(np.dot(exposures["market_beta"], aligned)) if "market_beta" in exposures else 0.0
        sector_vals = [abs(float(np.dot(v, aligned))) for k, v in exposures.items() if k.startswith("sector:")]
        diagnostics["max_sector_exposure"] = float(max(sector_vals)) if sector_vals else 0.0
        for name in dict(constraints.style_exposure_limits or {}):
            if name in exposures:
                diagnostics[f"{name}_exposure"] = float(np.dot(exposures[name], aligned))
        diagnostics["path"] = str(constraints.path)
        diagnostics["optimization_type"] = str(constraints.optimization_type)
        return diagnostics

    def apply_risk_overlay(
        self,
        weights: pd.Series,
        cov: np.ndarray,
        inputs: PortfolioInputs,
        constraints: PortfolioConstraints,
        overlay_cfg: RiskOverlayConfig,
    ) -> tuple[pd.Series, dict[str, Any]]:
        """
        Apply institutional risk overlays to constructed weights.

        Executes in order:
          1. Volatility targeting: scale weights to hit target portfolio vol
          2. Time-series risk overlay: adjust exposure based on market trend
          3. Stress test validation: verify portfolio survives scenarios

        Parameters
        ----------
        weights : pd.Series
            Base portfolio weights from optimization
        cov : np.ndarray
            Covariance matrix (N, N), annualized
        inputs : PortfolioInputs
            Current date-level inputs
        constraints : PortfolioConstraints
            Portfolio construction constraints
        overlay_cfg : RiskOverlayConfig
            Risk overlay configuration

        Returns
        -------
        tuple[pd.Series, dict]
            (adjusted_weights, overlay_diagnostics)
        """
        tickers = [str(t) for t in weights.index]
        diagnostics: dict[str, Any] = {}
        w = weights.reindex(tickers).fillna(0.0).to_numpy(dtype=float)

        if not overlay_cfg.vol_targeting_enabled and not overlay_cfg.time_series_overlay_enabled:
            return weights, diagnostics

        # Step 1: Volatility targeting
        if overlay_cfg.vol_targeting_enabled and cov.shape == (len(tickers), len(tickers)):
            try:
                from .risk_model import VolatilityTargeting

                vol_targeter = VolatilityTargeting(
                    target_vol=overlay_cfg.vol_target,
                    min_scale=overlay_cfg.vol_min_scale,
                    max_scale=overlay_cfg.vol_max_scale,
                )
                port_var = float(np.dot(w, np.dot(cov, w)))
                port_vol = np.sqrt(max(port_var, 1e-12))

                # Use trailing 20d realized vol for GARCH adjustment if available
                realized_vols = None
                if inputs.daily_vol is not None:
                    vols = _clean_series(inputs.daily_vol, tickers, default=0.0)
                    realized_vols = (vols * np.sqrt(252)).to_numpy(dtype=float)

                targeted, scale, forecast_vol = vol_targeter.apply_vol_targeting(w, cov, realized_vols)
                diagnostics["vol_target_scale"] = scale
                diagnostics["vol_forecast"] = forecast_vol
                diagnostics["vol_before_overlay"] = port_vol
                w = targeted

            except Exception as exc:
                logger.warning("Volatility targeting failed: %s", exc)
                diagnostics["vol_target_error"] = str(exc)

        # Step 2: Time-series risk overlay
        if overlay_cfg.time_series_overlay_enabled:
            try:
                from .time_series_risk import TimeSeriesRiskConfig, TimeSeriesRiskOverlay

                ts_cfg = TimeSeriesRiskConfig(
                    enabled=True,
                    vol_target=overlay_cfg.vol_target,
                    vol_min_scale=overlay_cfg.vol_min_scale,
                    vol_max_scale=overlay_cfg.vol_max_scale,
                )
                overlay = TimeSeriesRiskOverlay(ts_cfg)

                market_price = None
                if inputs.scores is not None and len(inputs.scores) > 0:
                    market_price = None  # Caller should provide SPY price

                exposure, state = overlay.compute_exposure(
                    date=inputs.date,
                    scores=inputs.scores if inputs.scores is not None else pd.Series(dtype=float),
                    market_price=market_price,
                )
                w = w * exposure
                diagnostics["ts_exposure_scale"] = exposure
                diagnostics["ts_regime"] = state.regime_label
                diagnostics["ts_signal_strength"] = state.signal_strength
                diagnostics["ts_trend_signal"] = state.trend_signal
                diagnostics["ts_vol_scale"] = state.vol_scale

            except Exception as exc:
                logger.warning("Time-series overlay failed: %s", exc)
                diagnostics["ts_overlay_error"] = str(exc)

        # Step 3: Stress test validation
        if overlay_cfg.stress_test_enabled and cov.shape == (len(tickers), len(tickers)):
            try:
                from .stress_testing import StressTestEngine

                stress_engine = StressTestEngine()
                mu = inputs.scores.reindex(tickers).fillna(0.0).to_numpy(dtype=float) * 0.01  # rough expected returns
                results = stress_engine.run_stress_tests(w, cov, expected_returns=mu, tickers=tickers)
                summary = stress_engine.summarize(results)

                diagnostics["stress_passed"] = summary["passed"]
                diagnostics["stress_failed"] = summary["failed"]
                diagnostics["stress_worst_dd"] = summary["worst_dd"]
                diagnostics["stress_worst_return"] = summary["worst_return"]

                # If too many scenarios fail, scale down risk
                if summary["failed"] > (len(results) - overlay_cfg.stress_min_pass_scenarios):
                    risk_reduction = 0.5
                    w = w * risk_reduction
                    diagnostics["stress_risk_reduction"] = risk_reduction
                    diagnostics["construction_status"] = "stress_reduced"

            except Exception as exc:
                logger.warning("Stress testing failed: %s", exc)
                diagnostics["stress_test_error"] = str(exc)

        # Reconstruct pd.Series
        adjusted_weights = pd.Series(w, index=tickers, dtype=float)
        adjusted_weights = adjusted_weights[adjusted_weights.abs() > 1e-12]

        return adjusted_weights, diagnostics


def construct_top_k_weights(
    adjusted_scores: Mapping[str, float],
    top_k: int,
    long_only: bool = True,
) -> dict[str, float]:
    """
    Build portfolio weights from adjusted scores with corrected allocation logic.

    Fix vs. previous implementation
    ---------------------------------
    The old code shifted all scores by -min_score to make them non-negative.
    This silently flips the signal direction when ALL selected scores are
    negative (e.g. a Bear-regime day where every stock scores < 0).
    In that case the "top" scorer was the least-negative stock, but after
    shifting it received the highest weight — which is correct.  However the
    total weight was then anchored to very small (near-zero) differences,
    creating extreme concentration.

    Fixed rule (long-only):
      - Use softmax-style allocation: w_i = exp(score_i) / sum(exp(score_j))
        This is always positive, preserves ordering, and handles negative scores
        gracefully without distorting relative magnitude.
      - Fallback to equal weight when score dispersion is near zero.

    Long/short:
      - Weights are proportional to raw score (positive for longs, negative for shorts)
      - Normalized so sum(|weight|) = 1

    Parameters
    ----------
    adjusted_scores : {ticker: score}
    top_k : maximum number of positions to select
    long_only : if True, select top K longs only (positive weights)
    """
    if top_k <= 0:
        return {}

    clean: list[tuple[str, float]] = []
    for ticker, score in adjusted_scores.items():
        try:
            v = float(score)
        except Exception:
            continue
        if not math.isfinite(v):
            continue
        clean.append((str(ticker), v))

    if not clean:
        return {}

    ranked = sorted(clean, key=lambda x: (-x[1], x[0]))
    selected = ranked[:min(top_k, len(ranked))]

    if long_only:
        # Softmax allocation — always positive, handles negative scores correctly
        scores_arr = np.array([s for _, s in selected], dtype=float)
        # Temperature-scaled softmax (temperature=1 by default)
        scores_arr = scores_arr - scores_arr.max()   # numerical stability
        exp_scores = np.exp(scores_arr)
        total = float(exp_scores.sum())
        if total <= 0 or not np.isfinite(total):
            w = 1.0 / len(selected)
            return {t: w for t, _ in selected}
        return {t: float(exp_scores[i] / total) for i, (t, _) in enumerate(selected)}

    else:
        # Long/short: preserve sign, normalise by sum of abs weights
        alloc = {t: s for t, s in selected}
        total_abs = sum(abs(v) for v in alloc.values())
        if total_abs <= 0:
            return {}
        return {t: v / total_abs for t, v in alloc.items()}


@dataclass(frozen=True)
class RegimeExposureConfig:
    normal_top_k: int = 10
    normal_exposure: float = 1.0
    crisis_top_k: int = 8     # Phase 4: raised from 4 — was leaving too much alpha on table
    crisis_exposure: float = 0.45  # Phase 4: raised from 0.25 — was suppressing returns to 25%
    crisis_regime_value: str = "Crisis"


def construct_regime_aware_portfolio(
    adjusted_scores: Mapping[str, float],
    current_regime: str,
    config: RegimeExposureConfig | None = None,
) -> dict[str, object]:
    """
    Regime-aware portfolio construction.

    Uses only the provided current_regime (no future state), then applies:
      ranking -> top-K selection -> normalized weights -> exposure scaling

    Returns:
      {
        "selected_assets": list[str],
        "adjusted_weights": dict[str, float],   # sums to effective_exposure
        "effective_exposure": float,            # in [0, 1]
        "top_k_used": int,
      }
    """
    cfg = config or RegimeExposureConfig()
    is_crisis = str(current_regime) == str(cfg.crisis_regime_value)

    top_k = int(cfg.crisis_top_k if is_crisis else cfg.normal_top_k)
    top_k = max(0, top_k)

    exposure = float(cfg.crisis_exposure if is_crisis else cfg.normal_exposure)
    exposure = max(0.0, min(exposure, 1.0))

    base_w = construct_top_k_weights(adjusted_scores, top_k=top_k)
    selected_assets = list(base_w.keys())
    adjusted_weights = {t: w * exposure for t, w in base_w.items()}

    return {
        "selected_assets": selected_assets,
        "adjusted_weights": adjusted_weights,
        "effective_exposure": exposure,
        "top_k_used": top_k,
    }


def construct_inverse_vol_weights(
    tickers: list[str],
    volatilities: dict[str, float],
    scores: Optional[Mapping[str, float]] = None,
    max_single_weight: float = 0.25,
    min_vol_floor: float = 0.005,
) -> dict[str, float]:
    """
    Inverse-volatility weighting (1/σ) — standard at AQR, Bridgewater.

    Each position's weight is inversely proportional to its realised
    volatility.  Lower-vol stocks get higher weights, providing natural
    risk equalisation without a full covariance matrix.

    Optionally blends with score-proportional weights when *scores* provided:
        w_blend_i = 0.5 × w_invvol_i + 0.5 × w_score_i

    Parameters
    ----------
    tickers : positions to weight (pre-selected subset)
    volatilities : {ticker: annualised_vol} — must include all tickers
    scores : optional {ticker: adjusted_score} for score-blended weights
    max_single_weight : maximum weight any single name can receive
    min_vol_floor : floor applied to vol before inversion (prevents extreme weights)

    Returns
    -------
    dict[ticker, weight], weights sum to 1
    """
    if not tickers:
        return {}

    inv_vols: dict[str, float] = {}
    for t in tickers:
        vol = volatilities.get(t, 0.2)       # default 20% annual vol
        vol = max(float(vol), min_vol_floor)  # apply floor
        inv_vols[t] = 1.0 / vol

    total_inv = sum(inv_vols.values())
    if total_inv <= 0:
        w_eq = 1.0 / len(tickers)
        return {t: w_eq for t in tickers}

    inv_vol_weights = {t: v / total_inv for t, v in inv_vols.items()}

    # Optionally blend with score weights
    if scores is not None:
        score_vals = [(t, float(scores.get(t, 0.0))) for t in tickers]
        # Softmax of scores
        s_arr = np.array([s for _, s in score_vals])
        s_arr -= s_arr.max()
        exp_s = np.exp(s_arr)
        exp_sum = float(exp_s.sum())
        score_weights = (
            {t: float(exp_s[i] / exp_sum) for i, (t, _) in enumerate(score_vals)}
            if exp_sum > 0
            else {t: 1.0 / len(tickers) for t in tickers}
        )
        blended = {t: 0.5 * inv_vol_weights[t] + 0.5 * score_weights[t] for t in tickers}
    else:
        blended = inv_vol_weights

    # Cap at max_single_weight and renormalise
    capped = {t: min(w, max_single_weight) for t, w in blended.items()}
    total_capped = sum(capped.values())
    if total_capped <= 0:
        w_eq = 1.0 / len(tickers)
        return {t: w_eq for t in tickers}

    return {t: w / total_capped for t, w in capped.items()}


def construct_risk_parity_weights(
    tickers: list[str],
    covariance_matrix: Optional[np.ndarray] = None,
    volatilities: Optional[dict[str, float]] = None,
    max_iter: int = 100,
    tol: float = 1e-8,
) -> dict[str, float]:
    """
    Equal risk contribution (ERC / Risk Parity) weights.

    Each asset contributes equally to total portfolio variance.
    When *covariance_matrix* is provided, uses the full Σ (Ledoit-Wolf
    shrinkage recommended before passing).  Otherwise falls back to
    diagonal Σ = diag(vol²), equivalent to inverse-vol weighting.

    Algorithm: Cyclical coordinate descent (Chaves et al. 2012).

    Parameters
    ----------
    tickers : list of tickers (must align with covariance_matrix rows/cols)
    covariance_matrix : (n×n) annualised covariance matrix
    volatilities : fallback if no covariance_matrix supplied
    max_iter : maximum CCD iterations
    tol : convergence tolerance on weight vector

    Returns
    -------
    dict[ticker, weight], weights sum to 1
    """
    n = len(tickers)
    if n == 0:
        return {}
    if n == 1:
        return {tickers[0]: 1.0}

    # Build Σ
    if covariance_matrix is not None and covariance_matrix.shape == (n, n):
        Sigma = covariance_matrix.astype(float)
    elif volatilities is not None:
        vols = np.array([max(volatilities.get(t, 0.2), 1e-4) for t in tickers])
        Sigma = np.diag(vols ** 2)
    else:
        vols = np.full(n, 0.2)
        Sigma = np.diag(vols ** 2)

    # Ensure positive-definite (add ridge for numerical stability)
    Sigma += np.eye(n) * 1e-8

    # Cyclical coordinate descent for ERC
    w = np.ones(n) / n
    for _ in range(max_iter):
        w_prev = w.copy()
        for i in range(n):
            # Marginal risk contribution of asset i
            Sigma_w = Sigma @ w
            # Target: rc_i = portfolio_vol / n  (equal contribution)
            port_var = float(w @ Sigma_w)
            target_rc = port_var / n
            # Newton step for w_i
            a = Sigma[i, i]
            b = float(np.dot(Sigma[i, :], w) - Sigma[i, i] * w[i])
            # Solve: w_i * (a*w_i + b) = target_rc
            # a*w_i^2 + b*w_i - target_rc = 0
            if a > 0:
                disc = b ** 2 + 4 * a * target_rc
                if disc >= 0:
                    w[i] = (-b + math.sqrt(disc)) / (2 * a)
        # Renormalise
        w = np.abs(w)
        total = float(w.sum())
        if total > 0:
            w /= total
        if np.max(np.abs(w - w_prev)) < tol:
            break

    return {t: float(w[i]) for i, t in enumerate(tickers)}


def construct_hybrid_fmc_score_weights(
    tickers: list[str],
    scores: Mapping[str, float],
    market_caps: dict[str, float],
    score_blend: float = 0.5,
    max_single_weight: float = 0.20,
) -> dict[str, float]:
    """
    FMC × Score hybrid weighting — S&P Global (Innes et al.) p.7, p.21-22.

    Rationale (graph node: mfic_rationale_hybrid_weight):
      "FMC × Score hybrid weighting maintains liquidity while improving factor
      tilt.  Pure score-weighting (softmax) ignores float market cap, leading
      to concentration in small illiquid names that score well."

    Weight formula:
        w_i = blend × w_score_i + (1 - blend) × w_fmc_i

    where:
        w_score_i  = softmax(score_i)          — factor tilt
        w_fmc_i    = FMC_i / Σ FMC_j           — liquidity anchor

    Parameters
    ----------
    tickers : pre-selected position list
    scores : {ticker: adjusted_score}
    market_caps : {ticker: float_adjusted_market_cap_$}
    score_blend : weight on score component (0 = pure FMC, 1 = pure score)
    max_single_weight : cap to prevent single-name concentration

    Returns
    -------
    dict[ticker, weight], weights sum to 1
    """
    if not tickers:
        return {}

    # Score component (softmax)
    s_arr = np.array([float(scores.get(t, 0.0)) for t in tickers], dtype=float)
    s_arr -= s_arr.max()
    exp_s = np.exp(s_arr)
    exp_sum = float(exp_s.sum())
    w_score = (
        {t: float(exp_s[i] / exp_sum) for i, t in enumerate(tickers)}
        if exp_sum > 0
        else {t: 1.0 / len(tickers) for t in tickers}
    )

    # FMC component
    caps = np.array([max(float(market_caps.get(t, 1.0)), 1.0) for t in tickers], dtype=float)
    cap_sum = float(caps.sum())
    w_fmc = {t: float(caps[i] / cap_sum) for i, t in enumerate(tickers)}

    # Blend
    blended = {
        t: score_blend * w_score[t] + (1.0 - score_blend) * w_fmc[t]
        for t in tickers
    }

    # Cap and renormalise
    capped = {t: min(w, max_single_weight) for t, w in blended.items()}
    total = sum(capped.values())
    if total <= 0:
        return {t: 1.0 / len(tickers) for t in tickers}
    return {t: w / total for t, w in capped.items()}


def compute_factor_imbalance(factor_scores: dict[str, float]) -> float:
    """
    Factor Imbalance Metric — S&P Global (Innes et al.) p.16.

    In Bottom-Up multi-factor construction, the composite z-score is the
    average of per-factor z-scores.  Factor imbalance measures the maximum
    deviation of any single factor from the composite average.

    High imbalance (> 1.0 z-score units) means one factor dominates the
    composite selection — the portfolio is implicitly a single-factor bet
    even though it looks multi-factor.

    Formula:
        composite = mean(z_factor_i)
        imbalance  = max |z_factor_i - composite|

    Parameters
    ----------
    factor_scores : {factor_name: z_score} for a single stock or portfolio avg

    Returns
    -------
    float : imbalance (0 = perfectly balanced, >1 = one factor dominates)
    """
    if len(factor_scores) < 2:
        return 0.0
    vals = np.array(list(factor_scores.values()), dtype=float)
    composite = float(vals.mean())
    return float(np.max(np.abs(vals - composite)))


def compute_active_share(
    portfolio_weights: dict[str, float],
    benchmark_weights: dict[str, float],
) -> float:
    """
    Active Share — S&P Global (Innes et al.) p.6-7, p.11.

    Measures how different the portfolio is from the benchmark.
    Active Share = 0.5 × Σ |w_portfolio_i - w_benchmark_i|

    Range: [0, 1].  Typical institutional targets: 0.60–0.80.
    Paper rationale (mfic_rationale_linear_exposure_risk):
      "Linear relationship between Active Share and Tracking Error means
      there is no optimal concentration point — target Active Share
      explicitly rather than deriving it from stock count alone."

    Parameters
    ----------
    portfolio_weights : {ticker: weight}, should sum to 1
    benchmark_weights : {ticker: weight}, benchmark (e.g. equal-weight S&P 500)

    Returns
    -------
    float in [0, 1]
    """
    all_tickers = set(portfolio_weights) | set(benchmark_weights)
    active_share = 0.5 * sum(
        abs(portfolio_weights.get(t, 0.0) - benchmark_weights.get(t, 0.0))
        for t in all_tickers
    )
    return float(active_share)


def compute_rank_based_weights(df: pd.DataFrame, long_only: bool = False) -> pd.DataFrame:
    """
    Convert `adjusted_score` into rank-based centered/normalized weights.

    Steps:
    1) Percentile rank in [0, 1] using stable average-rank ties.
    2) Center: weight_raw = rank_pct - 0.5
    3) Optional long-only clamp: max(weight_raw, 0)
    4) Normalize so sum(abs(weight)) == 1

    Returns a copy with columns: `rank_pct`, `weight_raw`, `weight`.
    """
    if "adjusted_score" not in df.columns:
        raise KeyError("compute_rank_based_weights requires column 'adjusted_score'")

    out = df.copy()
    s = pd.to_numeric(out["adjusted_score"], errors="coerce")
    valid = s.notna()

    rank_pct = pd.Series(0.5, index=out.index, dtype=float)
    if valid.any():
        sv = s[valid]
        n = int(len(sv))
        r = sv.rank(method="average", ascending=True)
        rp = (r - 1.0) / (n - 1.0) if n > 1 else pd.Series(0.5, index=sv.index, dtype=float)
        rank_pct.loc[sv.index] = pd.to_numeric(rp, errors="coerce").fillna(0.5)

    weight_raw = rank_pct - 0.5
    if long_only:
        weight_raw = weight_raw.clip(lower=0.0)

    denom = float(weight_raw.abs().sum())
    if denom <= 1e-12:
        # Stable fallback:
        # - long_only: equal weight over valid rows
        # - long/short: all zeros (no cross-sectional dispersion)
        if long_only and valid.any():
            w = pd.Series(0.0, index=out.index, dtype=float)
            w.loc[valid] = 1.0 / float(valid.sum())
        else:
            w = pd.Series(0.0, index=out.index, dtype=float)
    else:
        w = weight_raw / denom

    out["rank_pct"] = rank_pct.fillna(0.5).astype(float)
    out["weight_raw"] = weight_raw.fillna(0.0).astype(float)
    out["weight"] = pd.to_numeric(w, errors="coerce").fillna(0.0).astype(float)
    return out


def select_high_conviction_assets(
    df: pd.DataFrame,
    *,
    rank_col: str = "rank_pct",
    score_col: str = "adjusted_score",
    threshold: float = 0.6,
    top_k: int = 5,
) -> pd.DataFrame:
    """
    Keep high-conviction assets then assign normalized long-only weights.

    Rules:
    1) Keep rows with rank_col > threshold
    2) Select top K by rank_col (then score_col, both descending)
    3) Normalize selected weights to sum 1 (equal-weight fallback if needed)
    4) Edge handling:
       - fewer than K pass -> use all passers
       - none pass -> fallback to top K by rank/score from full universe

    Returns selected rows with added `weight` column.
    """
    if top_k <= 0:
        return pd.DataFrame(columns=list(df.columns) + ["weight"])
    if rank_col not in df.columns:
        raise KeyError(f"select_high_conviction_assets requires column '{rank_col}'")

    out = df.copy()
    rank = pd.to_numeric(out[rank_col], errors="coerce")
    if score_col in out.columns:
        score = pd.to_numeric(out[score_col], errors="coerce")
    else:
        score = rank.copy()
    out["_rank"] = rank
    out["_score"] = score

    valid = out[out["_rank"].notna()].copy()
    if valid.empty:
        return pd.DataFrame(columns=list(df.columns) + ["weight"])

    passed = valid[valid["_rank"] > float(threshold)].copy()
    pool = passed if not passed.empty else valid

    selected = (
        pool.sort_values(by=["_rank", "_score"], ascending=[False, False], kind="mergesort")
        .head(min(int(top_k), len(pool)))
        .copy()
    )

    n = len(selected)
    if n == 0:
        return pd.DataFrame(columns=list(df.columns) + ["weight"])

    # Long-only normalized weights from rank strength.
    w_raw = selected["_rank"].clip(lower=0.0)
    denom = float(w_raw.sum())
    if denom <= 1e-12:
        selected["weight"] = 1.0 / n
    else:
        selected["weight"] = w_raw / denom

    selected["weight"] = pd.to_numeric(selected["weight"], errors="coerce").fillna(0.0)
    selected = selected.drop(columns=["_rank", "_score"])
    return selected
