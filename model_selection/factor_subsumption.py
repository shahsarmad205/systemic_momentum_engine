from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from model_selection.research_contract import FEATURE_SPECS
from model_selection.residualization import safe_center


def _next_day_returns(scored: pd.DataFrame, *, horizon_days: int) -> pd.Series:
    if "daily_return" in scored.columns:
        tmp = scored[["date", "ticker", "daily_return"]].copy()
        tmp["date"] = pd.to_datetime(tmp["date"], errors="coerce")
        tmp = tmp.sort_values(["ticker", "date"])
        nxt = pd.to_numeric(tmp["daily_return"], errors="coerce").groupby(tmp["ticker"]).shift(-1)
        return pd.Series(nxt.to_numpy(dtype=float), index=scored.index)
    return pd.to_numeric(scored.get("forward_return", 0.0), errors="coerce") / max(1.0, float(horizon_days))


def _top_bottom_spread(signal: pd.Series, future_ret: pd.Series, *, quantile: float = 0.20) -> float:
    s = pd.to_numeric(signal, errors="coerce")
    r = pd.to_numeric(future_ret, errors="coerce")
    mask = s.notna() & r.notna()
    if int(mask.sum()) < 10:
        return float("nan")
    s = s.loc[mask]
    r = r.loc[mask]
    if s.nunique(dropna=True) < 2 or r.nunique(dropna=True) < 2:
        return float("nan")
    rank = s.rank(pct=True, method="average")
    q = float(min(max(quantile, 0.05), 0.45))
    long_mask = rank >= (1.0 - q)
    short_mask = rank <= q
    if int(long_mask.sum()) < 2 or int(short_mask.sum()) < 2:
        return float("nan")
    return float(r.loc[long_mask].mean() - r.loc[short_mask].mean())


def build_factor_mimicking_returns(scored: pd.DataFrame, *, horizon_days: int = 5) -> pd.DataFrame:
    if scored is None or scored.empty:
        return pd.DataFrame()
    df = scored.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["next_return"] = _next_day_returns(df, horizon_days=int(horizon_days))
    df = df.dropna(subset=["date", "ticker", "next_return"]).sort_values(["date", "ticker"])
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, float]] = []
    internal_families = {"trend", "residual_alpha", "sector_relative", "reversal"}
    internal_cols = [
        c for c in df.columns
        if c in FEATURE_SPECS and FEATURE_SPECS[c].family in internal_families
    ]
    for dt, day in df.groupby("date", sort=True):
        future = pd.to_numeric(day["next_return"], errors="coerce")
        row: dict[str, float] = {
            "date": pd.Timestamp(dt),
            "market": float(future.mean()) if len(future) else float("nan"),
        }

        if "sector" in day.columns:
            sector_ret = day.groupby(day["sector"].fillna("Unknown"))["next_return"].mean()
            market = row["market"]
            for sector_name, sec_ret in sector_ret.items():
                row[f"sector:{sector_name}"] = float(sec_ret - market)

        style_map: dict[str, pd.Series] = {}
        if "market_cap" in day.columns:
            style_map["size"] = -np.log(pd.to_numeric(day["market_cap"], errors="coerce").clip(lower=1.0))
        elif "adv_dollar_20" in day.columns:
            style_map["size"] = -np.log(pd.to_numeric(day["adv_dollar_20"], errors="coerce").clip(lower=1.0))
        if "book_to_market" in day.columns:
            style_map["value"] = pd.to_numeric(day["book_to_market"], errors="coerce")
        elif "bm" in day.columns:
            style_map["value"] = pd.to_numeric(day["bm"], errors="coerce")
        elif "nearness_52w_low" in day.columns:
            style_map["value"] = pd.to_numeric(day["nearness_52w_low"], errors="coerce")
        for name, columns in {
            "momentum": ("momentum_12m_skip1", "cs_momentum_percentile", "ret_20d"),
            "quality": ("quality_score", "roa", "f_score", "gross_margin"),
            "low_vol": ("low_vol_score",),
        }.items():
            for col in columns:
                if col in day.columns:
                    style_map[name] = pd.to_numeric(day[col], errors="coerce")
                    break
        if "low_vol" not in style_map:
            if "rolling_vol_20" in day.columns:
                style_map["low_vol"] = -pd.to_numeric(day["rolling_vol_20"], errors="coerce")
            elif "volatility_20" in day.columns:
                style_map["low_vol"] = -pd.to_numeric(day["volatility_20"], errors="coerce")

        if internal_cols:
            internal = day[internal_cols].apply(pd.to_numeric, errors="coerce").mean(axis=1)
            style_map["internal_alpha"] = internal

        for factor_name, signal in style_map.items():
            row[factor_name] = _top_bottom_spread(signal, future)
        rows.append(row)

    fac = pd.DataFrame(rows).set_index("date").sort_index()
    keep = []
    for col in fac.columns:
        s = pd.to_numeric(fac[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
        if s.notna().sum() >= 20 and s.std(ddof=0) > 1e-12:
            keep.append(col)
    return fac[keep] if keep else pd.DataFrame(index=fac.index)


def _hac_covariance(x: np.ndarray, resid: np.ndarray, *, max_lag: int = 5) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    resid = np.nan_to_num(np.asarray(resid, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0:
        return np.zeros((0, 0), dtype=float)
    x = np.clip(x, -1e6, 1e6)
    resid = np.clip(resid, -1e6, 1e6)
    n, k = x.shape
    with np.errstate(all="ignore"):
        xtx = np.nan_to_num(x.T @ x, nan=0.0, posinf=0.0, neginf=0.0)
    xtx_inv = np.linalg.pinv(xtx)
    s = np.zeros((k, k), dtype=float)
    xe = x * resid[:, None]
    with np.errstate(all="ignore"):
        s += np.nan_to_num(xe.T @ xe, nan=0.0, posinf=0.0, neginf=0.0)
    max_lag = min(max(0, int(max_lag)), n - 1)
    for lag in range(1, max_lag + 1):
        weight = 1.0 - lag / (max_lag + 1.0)
        with np.errstate(all="ignore"):
            gamma = np.nan_to_num(xe[lag:].T @ xe[:-lag], nan=0.0, posinf=0.0, neginf=0.0)
        s += weight * (gamma + gamma.T)
    return xtx_inv @ s @ xtx_inv


def factor_subsumption_diagnostics(
    model_returns: pd.Series | np.ndarray,
    scored: pd.DataFrame,
    *,
    horizon_days: int = 5,
    min_obs: int = 40,
    precomputed_fac: pd.DataFrame | None = None,
) -> dict[str, float]:
    fac = precomputed_fac if (precomputed_fac is not None and not precomputed_fac.empty) else build_factor_mimicking_returns(scored, horizon_days=int(horizon_days))
    r = pd.Series(model_returns).copy()
    r.index = pd.to_datetime(r.index, errors="coerce")
    r = pd.to_numeric(r, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if fac.empty or r.empty:
        return {
            "subsumption_alpha_ann": float("nan"),
            "subsumption_alpha_tstat": float("nan"),
            "subsumption_r2": float("nan"),
            "subsumption_max_abs_loading": float("nan"),
            "subsumption_n_obs": 0,
        }

    data = fac.join(r.rename("model_return"), how="inner").dropna()
    if len(data) < int(min_obs):
        return {
            "subsumption_alpha_ann": float("nan"),
            "subsumption_alpha_tstat": float("nan"),
            "subsumption_r2": float("nan"),
            "subsumption_max_abs_loading": float("nan"),
            "subsumption_n_obs": int(len(data)),
        }

    y = data["model_return"].to_numpy(dtype=float)
    x_cols = [c for c in data.columns if c != "model_return"]
    x_raw = data[x_cols].to_numpy(dtype=float)
    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    x_raw = np.nan_to_num(x_raw, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.clip(y, -1.0, 1.0)
    x_raw = np.clip(x_raw, -1.0, 1.0)
    alpha_daily = float(np.mean(y))
    y_centered = safe_center(y, cap=1.0)[:, 0]
    z = safe_center(x_raw, cap=1.0)
    control_std = np.std(z, axis=0)
    keep = np.isfinite(control_std) & (control_std > 1e-10)
    z = z[:, keep]
    kept_cols = [name for name, use in zip(x_cols, keep, strict=False) if use]
    if z.size == 0:
        resid = y_centered
        r2 = float("nan")
        alpha_t = float("nan")
        loadings: dict[str, float] = {}
    else:
        scale = np.std(z, axis=0)
        keep_scale = np.isfinite(scale) & (scale > 1e-10)
        z = z[:, keep_scale]
        kept_cols = [name for name, use in zip(kept_cols, keep_scale, strict=False) if use]
        if z.size == 0:
            resid = y_centered
            r2 = float("nan")
            alpha_t = float("nan")
            loadings = {}
        else:
            z = z / scale[keep_scale]
            z = np.nan_to_num(z, nan=0.0, posinf=0.0, neginf=0.0)
            # Stage 1 — lstsq with strict numerical validation.
            # Only coefficient estimation needs tight FP guards; downstream stats
            # (r2, HAC t-stat) use plain arithmetic and get their own fallback.
            _coef_ok = False
            coef: np.ndarray = np.array([], dtype=float)
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("error", RuntimeWarning)
                    with np.errstate(over="raise", invalid="raise", divide="raise"):
                        k = int(z.shape[1])
                        sqrt_ridge = float(np.sqrt(1e-6))
                        lhs = np.vstack([z, sqrt_ridge * np.eye(k, dtype=float)])
                        rhs = np.concatenate([y_centered, np.zeros(k, dtype=float)])
                        coef, _, _, singular_values = np.linalg.lstsq(lhs, rhs, rcond=None)
                        coef = np.asarray(coef, dtype=float)
                        if not np.isfinite(coef).all():
                            raise FloatingPointError("non_finite_coef")
                        coef_abs = float(np.max(np.abs(coef), initial=0.0))
                        if coef_abs > 1e4:
                            raise FloatingPointError("coef_scale_invalid")
                        sv = np.asarray(singular_values, dtype=float)
                        sv = sv[np.isfinite(sv) & (sv > 1e-10)]
                        if sv.size and float(np.max(sv) / np.min(sv)) > 1e8:
                            raise FloatingPointError("condition_number_invalid")
                        _coef_ok = True
            except (RuntimeWarning, ValueError, FloatingPointError, np.linalg.LinAlgError):
                pass

            if not _coef_ok:
                resid = y_centered
                r2 = float("nan")
                alpha_t = float("nan")
                loadings = {}
            else:
                # Stage 2 — r2 and HAC t-stat from validated coefficients.
                # Uses np.errstate(all="ignore") to tolerate minor numerical noise
                # without discarding results that the lstsq validation already passed.
                try:
                    with np.errstate(all="ignore"):
                        fitted_centered = z @ coef
                        fitted = alpha_daily + fitted_centered
                        resid = np.nan_to_num(y - fitted, nan=0.0, posinf=0.0, neginf=0.0)
                        ss_tot = float(np.sum((y - y.mean()) ** 2))
                        ss_res = float(np.sum(resid ** 2))
                        r2 = (
                            1.0 - ss_res / ss_tot
                            if (np.isfinite(ss_tot) and ss_tot > 1e-12)
                            else float("nan")
                        )
                        x_hac = np.c_[np.ones(len(data), dtype=float), z]
                        cov = _hac_covariance(x_hac, resid, max_lag=5)
                        se = np.sqrt(np.clip(np.diag(cov), 0.0, None))
                        alpha_t = (
                            alpha_daily / se[0]
                            if (np.isfinite(se[0]) and se[0] > 1e-12)
                            else float("nan")
                        )
                    loadings = {name: float(val) for name, val in zip(kept_cols, coef, strict=False)}
                except Exception:
                    resid = y_centered
                    r2 = float("nan")
                    alpha_t = float("nan")
                    loadings = {}
    max_abs_loading = max((abs(v) for v in loadings.values()), default=float("nan"))
    out: dict[str, float] = {
        "subsumption_alpha_ann": alpha_daily * 252.0,
        "subsumption_alpha_tstat": float(alpha_t),
        "subsumption_r2": float(r2) if np.isfinite(r2) else float("nan"),
        "subsumption_max_abs_loading": float(max_abs_loading) if np.isfinite(max_abs_loading) else float("nan"),
        "subsumption_n_obs": int(len(data)),
    }
    for name, value in loadings.items():
        safe_name = name.replace(":", "_").replace("-", "_")
        out[f"subsumption_loading_{safe_name}"] = float(value)
    return out
