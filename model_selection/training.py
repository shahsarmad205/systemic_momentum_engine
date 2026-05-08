from __future__ import annotations

from dataclasses import dataclass
import warnings

import numpy as np
import pandas as pd

from model_selection.residualization import residualize_against_controls, safe_center
from model_selection.validation import ExecutionCostConfig


@dataclass(frozen=True)
class TargetConfig:
    """Pure alpha supervised-learning target construction."""

    horizon_days: int = 5
    residualize: bool = True
    net_of_costs: bool = False
    residual_ridge: float = 1e-4
    winsor_q: float = 0.01
    max_abs_return: float = 5.0


@dataclass
class FeaturePreprocessor:
    """
    Train-only preprocessing artifact for model selection.

    The prior flow filled missing values with zero and clipped globally. This
    object estimates medians/winsorization bounds/constant-column masks only on
    the training slice, then applies the same schema to validation/OOS/live.
    """

    feature_columns: list[str]
    medians: pd.Series
    lower: pd.Series
    upper: pd.Series
    active_features: list[str]

    @classmethod
    def fit(
        cls,
        df: pd.DataFrame,
        feature_columns: list[str],
        *,
        winsor_q: float = 0.01,
        min_std: float = 1e-6,
    ) -> "FeaturePreprocessor":
        cols = [c for c in feature_columns if c in df.columns]
        x = df[cols].apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        med = x.median(axis=0).fillna(0.0)
        q = float(min(max(winsor_q, 0.0), 0.20))
        lo = x.quantile(q).fillna(med)
        hi = x.quantile(1.0 - q).fillna(med)
        spread = (hi - lo).abs()
        hi = hi.where(spread > 1e-12, med + 10.0)
        lo = lo.where(spread > 1e-12, med - 10.0)
        clipped = x.fillna(med).clip(lower=lo, upper=hi, axis=1)
        std = clipped.std(axis=0).fillna(0.0)
        active = [c for c in cols if float(std.get(c, 0.0)) >= float(min_std)]
        return cls(feature_columns=cols, medians=med, lower=lo, upper=hi, active_features=active)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        cols = self.active_features
        if not cols:
            return np.zeros((len(df), 0), dtype=float)
        x = pd.DataFrame(index=df.index)
        for col in cols:
            if col in df.columns:
                x[col] = pd.to_numeric(df[col], errors="coerce")
            else:
                x[col] = np.nan
        x = x.replace([np.inf, -np.inf], np.nan)
        x = x.fillna(self.medians.reindex(cols).fillna(0.0))
        x = x.clip(
            lower=self.lower.reindex(cols).fillna(-10.0),
            upper=self.upper.reindex(cols).fillna(10.0),
            axis=1,
        )
        return np.nan_to_num(x.to_numpy(dtype=float), nan=0.0, posinf=0.0, neginf=0.0).copy()


def _expected_round_trip_cost_breakdown(
    df: pd.DataFrame,
    *,
    costs: ExecutionCostConfig,
    horizon_days: int,
    max_name_weight: float,
) -> pd.DataFrame:
    if "expected_round_trip_cost_frac" in df.columns:
        raw = pd.to_numeric(df["expected_round_trip_cost_frac"], errors="coerce")
        total = raw.replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(lower=0.0)
        return pd.DataFrame(
            {
                "expected_round_trip_cost": total,
                "expected_participation": np.nan,
                "expected_fixed_cost": np.nan,
                "expected_temporary_impact": np.nan,
                "expected_permanent_impact": np.nan,
                "expected_borrow_cost": np.nan,
            },
            index=df.index,
        )

    adv_raw = df["adv_dollar_20"] if "adv_dollar_20" in df.columns else pd.Series(costs.default_adv_usd, index=df.index)
    adv = pd.to_numeric(adv_raw, errors="coerce")
    adv = adv.replace([np.inf, -np.inf], np.nan).fillna(costs.default_adv_usd).clip(lower=1.0)
    vol_raw = df["realised_vol_20d"] if "realised_vol_20d" in df.columns else pd.Series(costs.default_daily_vol, index=df.index)
    vol = pd.to_numeric(vol_raw, errors="coerce")
    vol = vol.replace([np.inf, -np.inf], np.nan).fillna(costs.default_daily_vol).abs().clip(lower=1e-6)
    order_usd = float(max_name_weight) * float(costs.capital)
    participation = (order_usd / adv).clip(lower=0.0, upper=max(float(costs.max_participation_rate), 1e-6))
    commission = float(costs.commission_bps) / 10_000.0
    spread = float(costs.spread_bps) / 2.0 / 10_000.0
    fixed_one_way = commission + spread
    temporary = float(costs.impact_eta) * vol * (participation ** float(costs.impact_gamma))
    permanent = float(costs.impact_alpha) * vol * (participation ** float(costs.impact_gamma))
    borrow = float(costs.borrow_bps) / 10_000.0 * max(1, int(horizon_days)) / 252.0
    fixed_round_trip = 2.0 * fixed_one_way
    total = (fixed_round_trip + temporary + permanent + borrow).clip(lower=0.0)
    return pd.DataFrame(
        {
            "expected_round_trip_cost": total,
            "expected_participation": participation.astype(float),
            "expected_fixed_cost": pd.Series(fixed_round_trip, index=df.index, dtype=float),
            "expected_temporary_impact": temporary.astype(float),
            "expected_permanent_impact": permanent.astype(float),
            "expected_borrow_cost": pd.Series(borrow, index=df.index, dtype=float),
        },
        index=df.index,
    )


def _expected_round_trip_cost(
    df: pd.DataFrame,
    *,
    costs: ExecutionCostConfig,
    horizon_days: int,
    max_name_weight: float,
) -> pd.Series:
    return _expected_round_trip_cost_breakdown(
        df,
        costs=costs,
        horizon_days=horizon_days,
        max_name_weight=max_name_weight,
    )["expected_round_trip_cost"]


def _residualize_by_date(
    df: pd.DataFrame,
    target: pd.Series,
    *,
    ridge: float,
) -> pd.Series:
    out = pd.Series(np.nan, index=df.index, dtype=float)
    work = df.copy()
    work["_target"] = pd.to_numeric(target, errors="coerce")
    work["date"] = pd.to_datetime(work["date"], errors="coerce")

    for _, g in work.groupby("date", sort=True):
        y = pd.to_numeric(g["_target"], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(y)
        if mask.sum() < 8:
            out.loc[g.index] = safe_center(y, cap=5.0)[:, 0] if np.isfinite(y).any() else 0.0
            continue

        factors: list[pd.Series] = []
        if "capm_beta" in g.columns:
            beta = pd.to_numeric(g["capm_beta"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(1.0)
            factors.append(beta.clip(-3.0, 3.0) - beta.mean())
        if "sector" in g.columns:
            dummies = pd.get_dummies(g["sector"].fillna("Unknown").astype(str), dtype=float)
            for col in dummies.columns:
                factors.append(dummies[col] - dummies[col].mean())
        if "adv_dollar_20" in g.columns:
            adv = pd.to_numeric(g["adv_dollar_20"], errors="coerce").replace([np.inf, -np.inf], np.nan)
            log_adv = np.log(adv.clip(lower=1.0)).fillna(np.log(50_000_000.0))
            factors.append(log_adv - log_adv.mean())
        if "realised_vol_20d" in g.columns:
            vol = pd.to_numeric(g["realised_vol_20d"], errors="coerce").replace([np.inf, -np.inf], np.nan)
            vol = vol.fillna(vol.median()).clip(lower=0.0, upper=1.0)
            factors.append(vol - vol.mean())

        if not factors:
            out.loc[g.index] = safe_center(y, cap=5.0)[:, 0]
            continue

        x = pd.concat(factors, axis=1).to_numpy(dtype=float)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        y0 = safe_center(y, cap=5.0)[:, 0]
        y_abs_max = float(np.abs(y0).max()) if len(y0) else 0.0
        if not np.isfinite(y_abs_max):
            out.loc[g.index] = 0.0
            continue
        if y_abs_max > 5.0:
            # Equity forward returns beyond +/-500% are almost always bad corporate-action
            # adjustments or data spikes. Keep the cross-sectional order direction but avoid
            # letting one pathological target blow up factor residualization.
            y0 = np.clip(y0, -5.0, 5.0)
        col_std = x.std(axis=0)
        keep = np.isfinite(col_std) & (col_std > 1e-10)
        x = x[:, keep]
        if x.size == 0:
            out.loc[g.index] = y0
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                with np.errstate(over="raise", invalid="raise", divide="raise"):
                    resid_result = residualize_against_controls(
                        y0,
                        x,
                        ridge=float(ridge),
                        value_cap=5.0,
                        control_cap=10.0,
                    )
                    resid = np.asarray(resid_result.residual, dtype=float)
                    if not np.isfinite(resid).all():
                        resid = y0
        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
            resid = y0
        out.loc[g.index] = resid
    return out.fillna(0.0)


def add_institutional_targets(
    df: pd.DataFrame,
    *,
    cfg: TargetConfig,
    costs: ExecutionCostConfig,
    max_name_weight: float,
) -> pd.DataFrame:
    """Add net-of-cost, factor-residualized, rank, and short-decile targets."""
    out = df.copy()
    fwd = pd.to_numeric(out["forward_return"], errors="coerce").replace([np.inf, -np.inf], np.nan)
    max_abs_return = abs(float(cfg.max_abs_return))
    if max_abs_return > 0.0:
        fwd = fwd.clip(lower=-max_abs_return, upper=max_abs_return)
    cost_breakdown = _expected_round_trip_cost_breakdown(
        out,
        costs=costs,
        horizon_days=int(cfg.horizon_days),
        max_name_weight=float(max_name_weight),
    )
    cost = cost_breakdown["expected_round_trip_cost"]
    target = fwd - cost if cfg.net_of_costs else fwd.copy()
    target = target.replace([np.inf, -np.inf], np.nan)
    if max_abs_return > 0.0:
        target = target.clip(lower=-max_abs_return, upper=max_abs_return)
    q = float(min(max(cfg.winsor_q, 0.0), 0.20))
    if q > 0:
        lo = target.groupby(pd.to_datetime(out["date"], errors="coerce")).transform(lambda x: x.quantile(q))
        hi = target.groupby(pd.to_datetime(out["date"], errors="coerce")).transform(lambda x: x.quantile(1.0 - q))
        target = target.clip(lower=lo, upper=hi)
    out["target_return_net"] = target.fillna(0.0)
    out["target_return"] = (
        _residualize_by_date(out, target, ridge=float(cfg.residual_ridge))
        if cfg.residualize
        else target.groupby(pd.to_datetime(out["date"], errors="coerce")).transform(lambda x: x - x.mean()).fillna(0.0)
    )
    rank_pct = out["target_return"].groupby(pd.to_datetime(out["date"], errors="coerce")).transform(
        lambda x: x.rank(pct=True, na_option="keep")
    )
    out["target_rank"] = (rank_pct.fillna(0.5) * 2.0 - 1.0).astype(float)
    out["target_down_decile"] = (rank_pct.fillna(0.5) <= 0.10).astype(int)
    out["target_up"] = (out["target_return"] > 0.0).astype(int)
    out["target_expected_cost"] = cost.astype(float)
    out["target_expected_participation"] = cost_breakdown["expected_participation"].astype(float)
    out["target_expected_fixed_cost"] = cost_breakdown["expected_fixed_cost"].astype(float)
    out["target_expected_temporary_impact"] = cost_breakdown["expected_temporary_impact"].astype(float)
    out["target_expected_permanent_impact"] = cost_breakdown["expected_permanent_impact"].astype(float)
    out["target_expected_borrow_cost"] = cost_breakdown["expected_borrow_cost"].astype(float)
    return out


def retarget_panel_for_horizon(
    df: pd.DataFrame,
    *,
    horizon_days: int,
    target_cfg: TargetConfig,
    costs: ExecutionCostConfig,
    max_name_weight: float,
) -> pd.DataFrame:
    """
    Rebuild forward returns and downstream targets for a candidate holding horizon.

    If ``daily_return`` is available we recompute the true forward compounded return
    from next-day realized returns. Otherwise we preserve the existing
    ``forward_return`` column and only rebuild the net-of-cost target stack.
    """
    out = df.copy()
    horizon = max(1, int(horizon_days))
    if "daily_return" in out.columns:
        work = out[["ticker", "date", "daily_return"]].copy()
        work["date"] = pd.to_datetime(work["date"], errors="coerce")
        work["daily_return"] = pd.to_numeric(work["daily_return"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        work = work.sort_values(["ticker", "date"])

        def _future_compound(s: pd.Series) -> pd.Series:
            lead = (1.0 + pd.to_numeric(s, errors="coerce")).shift(-1)
            prod = lead.rolling(horizon, min_periods=horizon).apply(np.prod, raw=True)
            return prod.shift(-(horizon - 1)) - 1.0

        work["_forward_h"] = work.groupby("ticker", sort=False)["daily_return"].transform(_future_compound)
        out = out.join(work["_forward_h"])
        out["forward_return"] = pd.to_numeric(out["_forward_h"], errors="coerce").replace([np.inf, -np.inf], np.nan)
        out = out.drop(columns=["_forward_h"])

    cfg_h = TargetConfig(
        horizon_days=horizon,
        residualize=bool(target_cfg.residualize),
        net_of_costs=bool(target_cfg.net_of_costs),
        residual_ridge=float(target_cfg.residual_ridge),
        winsor_q=float(target_cfg.winsor_q),
        max_abs_return=float(target_cfg.max_abs_return),
    )
    return add_institutional_targets(
        out,
        cfg=cfg_h,
        costs=costs,
        max_name_weight=max_name_weight,
    )


def make_training_target(
    df: pd.DataFrame,
    *,
    model_name: str,
    model_kind: str,
    use_risk_adj: bool,
) -> np.ndarray:
    """Return the supervised target aligned to the model family."""
    if model_kind == "short_classifier":
        return df["target_down_decile"].fillna(0).astype(int).to_numpy()
    if model_kind in {"long_alpha", "overlay_alpha", "short_alpha"}:
        return df["target_rank"].fillna(0.0).to_numpy(dtype=float)
    if model_kind == "regressor":
        if use_risk_adj and "forward_return_risk_adj" in df.columns:
            return (
                pd.to_numeric(df["forward_return_risk_adj"], errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .fillna(df["target_return"])
                .clip(-10.0, 10.0)
                .to_numpy(dtype=float)
            )
        # Ranking models receive ordinal rank labels; the wrappers convert them to
        # per-cross-section decile integers internally before calling the ranker API.
        _RANK_TARGET_MODELS = {
            "XGBRankIC", "LGBMRanker",
            "XGBRankerPairwise", "XGBRankerNDCG",
        }
        if model_name in _RANK_TARGET_MODELS:
            return df["target_rank"].fillna(0.0).to_numpy(dtype=float)
        # Quantile regressors are more stable on the winsorized residualized return
        # than on raw rank percentiles, but either is valid — use target_return.
        return df["target_return"].fillna(0.0).clip(-0.5, 0.5).to_numpy(dtype=float)
    return df["target_up"].fillna(0).astype(int).to_numpy()
