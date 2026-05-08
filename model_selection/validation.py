from __future__ import annotations

import enum
import hashlib
import logging
import math
import pickle
import tempfile
import time
import warnings
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scipy import stats as _sps
from model_selection.adaptive_portfolio_control import (
    AdaptivePortfolioControlConfig,
    AdaptivePortfolioController,
)

logger = logging.getLogger(__name__)

_PORTFOLIO_CONSTRUCTION_TYPES: tuple[Any, Any, Any] | None = None
_RISK_MODEL_TYPE: Any | None = None


def _stable_cache_token(value: Any) -> str:
    """Deterministic cache token; avoids Python's process-randomized built-in hashing."""
    digest = hashlib.sha256()
    if isinstance(value, (pd.Series, pd.Index)):
        arr = value.to_numpy()
    else:
        arr = np.asarray(value)
    digest.update(str(arr.dtype).encode("utf-8"))
    digest.update(str(arr.shape).encode("utf-8"))
    if arr.dtype == object:
        encoded = pd.Series(arr.reshape(-1)).astype(str).tolist()
        digest.update(repr(encoded).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(arr).tobytes())
    return digest.hexdigest()[:24]


def _load_backtesting_leaf_module(module_name: str, file_name: str) -> Any:
    """Load a backtesting leaf module without executing backtesting/__init__.py."""
    import importlib.util
    import sys

    cached = sys.modules.get(module_name)
    if cached is not None:
        return cached
    module_path = Path(__file__).resolve().parents[1] / "backtesting" / file_name
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load {module_name} from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    return module


def _load_portfolio_construction() -> tuple[Any, Any, Any]:
    """Load optional portfolio construction dependencies at execution time."""
    global _PORTFOLIO_CONSTRUCTION_TYPES
    if _PORTFOLIO_CONSTRUCTION_TYPES is not None:
        return _PORTFOLIO_CONSTRUCTION_TYPES
    try:
        module = _load_backtesting_leaf_module(
            "_model_selection_backtesting_portfolio_construction",
            "portfolio_construction.py",
        )
        _PORTFOLIO_CONSTRUCTION_TYPES = (
            module.PortfolioConstructor,
            module.PortfolioConstraints,
            module.PortfolioInputs,
        )
    except Exception as exc:
        raise RuntimeError(
            "Executable portfolio construction requires backtesting.portfolio_construction. "
            "That optional path could not be imported; check optional backtesting dependencies."
        ) from exc
    return _PORTFOLIO_CONSTRUCTION_TYPES


def _load_risk_model() -> Any:
    """Load the optional production risk model only when validation state is built."""
    global _RISK_MODEL_TYPE
    if _RISK_MODEL_TYPE is not None:
        return _RISK_MODEL_TYPE
    try:
        module = _load_backtesting_leaf_module(
            "_model_selection_backtesting_risk_model",
            "risk_model.py",
        )
        _RISK_MODEL_TYPE = module.RiskModel
    except Exception as exc:
        raise RuntimeError(
            "ValidationStateCache requires backtesting.risk_model. "
            "That optional path could not be imported; check optional backtesting dependencies."
        ) from exc
    return _RISK_MODEL_TYPE

class MetricIntegrityError(Exception):
    """Raised when a required statistical metric is missing from evaluation results."""
    pass

DEBUG_DIAGNOSTICS = False
TRACE_DIAGNOSTICS = False

# Global telemetry for market state cache performance (Task 2)
_MARKET_STATE_STATS = {
    "hits": 0,
    "misses": 0,
    "build_time_s": 0.0,
}

def get_market_state_stats() -> dict[str, Any]:
    """Safe accessor for market state telemetry (Task 4)."""
    return dict(_MARKET_STATE_STATS)


@dataclass(frozen=True)
class ExecutionCostConfig:
    """Cost model used inside model selection, expressed in portfolio-return units."""

    capital: float = 10_000_000.0
    commission_bps: float = 1.0
    spread_bps: float = 1.0
    borrow_bps: float = 50.0
    impact_eta: float = 0.142
    impact_alpha: float = 0.314
    impact_gamma: float = 0.6
    default_adv_usd: float = 50_000_000.0
    default_daily_vol: float = 0.02
    max_participation_rate: float = 0.10
    permanent_impact_decay_days: int = 5


@dataclass(frozen=True)
class TradeCostBreakdown:
    """Per-trade execution economics in portfolio-return units."""

    cost_return: float
    commission_return: float
    spread_return: float
    fixed_cost_return: float
    temporary_impact_return: float
    permanent_impact_return: float
    trade_weight_abs: float
    trade_notional: float
    adv_dollar: float
    daily_vol: float
    participation_rate: float
    participation_capped: bool


@dataclass(frozen=True)
class DateLevelMarketState:
    """Reusable date-level market state for executable portfolio construction."""

    date: pd.Timestamp
    tickers: tuple[str, ...]
    ticker_to_idx: dict[str, int]
    covariance: np.ndarray
    specific_risk: np.ndarray
    factor_exposures: dict[str, np.ndarray]
    adv_dollar: np.ndarray
    daily_vol: np.ndarray
    liquidity_caps: np.ndarray
    participation_scale: np.ndarray
    max_participation_rate: float
    borrow_penalty_horizon: np.ndarray
    crowding_risk: np.ndarray
    short_interest_ratio: np.ndarray
    squeeze_risk: np.ndarray
    short_blocked: np.ndarray


@dataclass(frozen=True)
class EvaluationConfig:
    """Executable model-selection portfolio assumptions."""

    max_positions: int = 10
    min_positions: int = 3
    horizon_days: int = 5
    path: str = "long_short_spread"
    rebalance_every_days: int = 5
    factor_neutral: bool = True
    beta_neutral: bool = True
    sector_neutral: bool = True
    max_beta_abs: float = 0.15
    max_sector_abs: float = 0.12
    adv_fraction: float = 0.05
    max_gross: float = 1.0
    max_name_weight: float = 0.10
    constraint_passes: int = 3
    costs: ExecutionCostConfig = ExecutionCostConfig()
    use_optimizer: bool = True
    lambda_risk: float = 2.0
    gamma_turnover: float = 4.0
    net_exposure_max: float = 0.01  # P5: tightened from 0.10 — strict dollar neutrality for L/S spread books
    min_position_weight: float = 0.0
    no_trade_band_weight_diff: float = 0.015
    no_trade_band_total_drift: float = 0.05
    optimizer_lookback_days: int = 60
    optimizer_alpha_scale: float = 1.0
    short_squeeze_filter: bool = True
    short_squeeze_max_risk: float = 0.75
    style_exposure_limits: dict[str, float] = field(default_factory=dict)
    # Market-neutral adjustment for pure short mandates: subtract (net_exposure ×
    # cross-sectional mean return) from each day's P&L so the Sharpe measures
    # return RELATIVE TO the market rather than absolute return.  Without this,
    # a short book with 84% directional accuracy still shows Sharpe ≈ −6 in a
    # sustained bull market — a structural eval artefact, not a model failure.
    market_neutral_shorts: bool = True
    optimization_type: str = "l1"  # "l2" or "l1"
    lambda_turn_override: float | None = None
    signal_halflife_days: float = float("nan")  # P8: for decay-aware execution scaling
    adaptive_control_enabled: bool = False
    adaptive_control_lookback_days: int = 252
    adaptive_control_min_history_days: int = 60
    adaptive_control_ema_span: int = 20
    adaptive_control_target_volatility: float = 0.15
    adaptive_lambda_floor_factor: float = 0.50
    adaptive_lambda_ceil_factor: float = 4.00
    adaptive_gamma_floor_factor: float = 0.125
    adaptive_gamma_ceil_factor: float = 4.00
    adaptive_min_expected_alpha: float = 1e-4


@dataclass(frozen=True)
class PromotionGateConfig:
    """Hard gates for model artifacts that can be exported or promoted."""

    min_sharpe: float = 0.50
    min_ic_tstat: float = 2.0
    min_ic_ir: float = 0.75
    min_beat_rate: float = 0.625
    max_drawdown: float = -0.25
    min_cost_aware_sharpe: float = 0.25
    min_windows: int = 6
    min_psr: float = 0.60
    max_beta_abs_mean: float = 0.15
    max_sector_abs_mean: float = 0.12
    max_cost_to_gross_pnl: float = 0.50
    min_decile_spread: float = 0.0
    min_tail_monotonicity: float = 0.50
    min_long_leg_sharpe: float = 0.0
    min_short_leg_sharpe: float = 0.0
    min_subsumption_alpha_ann: float = 0.0
    min_subsumption_alpha_tstat: float = 1.0
    max_subsumption_r2: float = 0.80
    max_subsumption_loading_abs: float = 1.50
    
    # Execution Robustness Gates
    execution_robustness_enabled: bool = True
    execution_robustness_affect_selection: bool = False
    execution_robustness_fail_on_missing: bool = True
    min_signal_halflife_buffer: float = 1.0  # Buffer added to rebalance_frequency
    min_caic_to_ic_ratio: float = 0.30
    max_avg_turnover: float = 0.80          # Daily turnover threshold
    dynamic_thresholds_enabled: bool = True
    dynamic_threshold_confidence: float = 0.95
    dynamic_threshold_min_effective_obs: int = 12
    dynamic_threshold_reference_ic_std: float = 0.05
    dynamic_threshold_reference_turnover: float = 0.35
    
    # Per-path threshold overrides. Keys are model_kind (e.g. "short_classifier")
    # or the family alias "short_side" (matches short_classifier + short_alpha).
    # Exact model_kind takes precedence over the family alias.
    # Any scalar field on this dataclass can be overridden here.
    path_overrides: dict = field(default_factory=dict)


class PromotionTier(str, enum.Enum):
    """Hierarchical classification of model promotion readiness."""

    PRODUCTION = "production"
    LONG_ALPHA_CAND = "long_alpha_candidate"
    HEDGE_ONLY = "hedge_only"
    DIAGNOSTIC_ONLY = "diagnostic_only"
    REJECTED = "rejected"


@dataclass(frozen=True)
class LongAlphaCandidateConfig:
    """Gates for the long_alpha_candidate tier."""

    min_long_leg_sharpe: float = 0.0
    min_ic_tstat: float = 1.0
    min_beat_rate: float = 0.50
    max_drawdown: float = -0.35
    min_psr: float = 0.50
    max_cost_to_gross_pnl: float = 0.75
    min_decile_spread: float = 0.0
    min_windows: int = 4


def _risk_price_data_from_panel(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Convert the validation panel into RiskModel-compatible per-ticker frames.

    Institutional ownership rule:
    - RiskModel estimates exposures/covariance.
    - PortfolioConstructor enforces exposure constraints.
    - Simulator computes PnL/costs and must not neutralize weights.
    """
    if df is None or df.empty:
        return {}
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work = work.dropna(subset=["date", "ticker"]).sort_values(["ticker", "date"])
    price_data: dict[str, pd.DataFrame] = {}
    passthrough_cols = [
        "market_beta",
        "capm_beta",
        "beta",
        "market_cap",
        "log_market_cap",
        "size",
        "size_exposure",
        "book_to_market",
        "bm",
        "value_score",
        "quality_score",
        "roa",
        "profitability",
        "momentum",
        "momentum_exposure",
        "momentum_12m_skip1",
    ]
    for ticker, grp in work.groupby("ticker", sort=False):
        g = grp.sort_values("date").set_index("date")
        out = pd.DataFrame(index=pd.to_datetime(g.index))
        close_col = next((c for c in ("AdjClose", "Adj Close", "Close", "close") if c in g.columns), None)
        if close_col is not None:
            close = pd.to_numeric(g[close_col], errors="coerce")
        elif "daily_return" in g.columns:
            ret = pd.to_numeric(g["daily_return"], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
            close = 100.0 * (1.0 + ret.clip(-0.95, 10.0)).cumprod()
        else:
            close = pd.Series(100.0, index=g.index, dtype=float)
        out["Close"] = close.replace([np.inf, -np.inf], np.nan).ffill().fillna(100.0)
        if "Volume" in g.columns:
            out["Volume"] = pd.to_numeric(g["Volume"], errors="coerce")
        elif "volume" in g.columns:
            out["Volume"] = pd.to_numeric(g["volume"], errors="coerce")
        elif "adv_dollar_20" in g.columns:
            adv = pd.to_numeric(g["adv_dollar_20"], errors="coerce")
            out["Volume"] = (adv / out["Close"].replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        for col in passthrough_cols:
            if col in g.columns:
                out[col] = pd.to_numeric(g[col], errors="coerce")
        price_data[str(ticker)] = out
    return price_data


def _sector_maps_for_day(day: pd.DataFrame) -> tuple[dict[str, int], dict[int, str]]:
    if day is None or day.empty or "sector" not in day.columns:
        return {}, {}
    labels = day["sector"].fillna("Unknown").astype(str)
    unique = {label: idx for idx, label in enumerate(sorted(labels.unique()))}
    sector_id_map = {
        str(ticker): int(unique[label])
        for ticker, label in zip(day["ticker"].astype(str), labels, strict=False)
    }
    sector_labels = {idx: label for label, idx in unique.items()}
    return sector_id_map, sector_labels


# --- OPTIMIZED KERNELS ---


@dataclass(frozen=True)
class ChurnFilterConfig:
    """P18: Rank-persistence eligibility filter for portfolio construction."""

    enabled: bool = False
    min_consecutive_top_decile_days: int = 2
    top_decile_fraction: float = 0.10
    min_eligible_names: int = 10
    apply_to_paths: tuple[str, ...] = ("long_only_overlay",)
    score_penalty_for_ineligible: float = -1e6  # effectively removes from optimizer


def _churn_filter_top_decile_membership(
    scored: pd.DataFrame,
) -> pd.DataFrame:
    """
    Tag each row with whether its ticker is in the top decile on that date.

    Returns copy of scored with added column ``_top_decile`` (bool).
    """
    work = scored[["date", "ticker", "score"]].copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    work["score"] = pd.to_numeric(work["score"], errors="coerce")
    work["_pct"] = work.groupby("date")["score"].rank(pct=True)
    work["_top_decile"] = work["_pct"] >= (1.0 - 0.10)
    return work.drop(columns=["_pct"])


def apply_churn_filter(
    te_scored: pd.DataFrame,
    *,
    cfg: ChurnFilterConfig,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """
    P18: Persistence eligibility filter.

    Only stocks that have been in the top decile for at least
    ``min_consecutive_top_decile_days`` consecutive signal dates are eligible
    for portfolio allocation.  Ineligible stocks have their score set to a
    large negative value so the QP optimizer never selects them.

    Returns (filtered_scored, diagnostics_dict).
    """
    if te_scored is None or te_scored.empty:
        return te_scored, {"churn_filter_status": "empty_panel"}

    diag: dict[str, Any] = {"churn_filter_status": "applied"}
    n_days = int(cfg.min_consecutive_top_decile_days)

    # 1. Tag top-decile membership
    tagged = _churn_filter_top_decile_membership(te_scored)

    # 2. For each ticker, count consecutive top-decile days
    tagged = tagged.sort_values(["ticker", "date"])
    tagged["_streak"] = 0

    for ticker, grp in tagged.groupby("ticker", sort=False):
        streak = 0
        streak_vals = np.zeros(len(grp), dtype=int)
        for i in range(len(grp)):
            if grp.iloc[i]["_top_decile"]:
                streak += 1
            else:
                streak = 0
            streak_vals[i] = streak
        tagged.loc[grp.index, "_streak"] = streak_vals

    # 3. Eligible if streak >= min_consecutive_top_decile_days
    tagged["_eligible"] = tagged["_streak"] >= n_days
    eligible_mask = tagged["_eligible"]

    # 4. Build diagnostics
    n_eligible = int(eligible_mask.sum())
    n_total = len(tagged)
    diag["churn_filter_n_eligible"] = n_eligible
    diag["churn_filter_n_total"] = n_total
    diag["churn_filter_eligible_pct"] = round(n_eligible / max(n_total, 1) * 100, 1)
    diag["churn_filter_min_streak_days"] = n_days

    if n_eligible < cfg.min_eligible_names:
        diag["churn_filter_status"] = "fallback_unfiltered"
        diag["churn_filter_fallback_reason"] = (
            f"Only {n_eligible} eligible names (min={cfg.min_eligible_names})"
        )
        # Return unfiltered panel but with diagnostics
        return te_scored, diag

    # 5. Score eligible vs ineligible
    eligible_scores = tagged.loc[eligible_mask, "score"]
    ineligible_scores = tagged.loc[~eligible_mask, "score"]
    diag["churn_filter_eligible_score_mean"] = float(eligible_scores.mean())
    diag["churn_filter_ineligible_score_mean"] = (
        float(ineligible_scores.mean()) if len(ineligible_scores) > 0 else float("nan")
    )

    # 6. Build filtered panel: keep all rows but penalize ineligible scores
    #   The QP optimizer will naturally avoid deeply negative scores.
    out = te_scored.copy()
    ineligible_idx = tagged.index[~eligible_mask]
    if "score" in out.columns:
        out.loc[ineligible_idx, "score"] = float(cfg.score_penalty_for_ineligible)

    # 7. Per-date eligible count
    tagged["date"] = pd.to_datetime(tagged["date"], errors="coerce")
    diag["churn_filter_eligible_per_date_mean"] = float(
        tagged.groupby("date")["_eligible"].sum().mean()
    )
    diag["churn_filter_eligible_per_date_min"] = int(
        tagged.groupby("date")["_eligible"].sum().min()
    )

    # 8. Average streak of eligible stocks
    eligible_streaks = tagged.loc[eligible_mask, "_streak"]
    if len(eligible_streaks) > 0:
        diag["churn_filter_eligible_streak_mean"] = float(eligible_streaks.mean())

    return out, diag


def _get_implied_weights(
    df: pd.DataFrame,
    *,
    primary_path: str,
    precomputed_ranks: pd.Series | np.ndarray | None = None,
) -> pd.DataFrame:
    """Compute implied portfolio weights from cross-sectional scores using optimized NumPy ops."""
    if df.empty or "score" not in df.columns or "date" not in df.columns:
        return pd.DataFrame()

    # Reuse data as much as possible, avoiding extra columns
    work = df[["date", "ticker", "score"]].dropna()
    if work.empty:
        return pd.DataFrame()

    # Use global date indices to ensure alignment across different functions
    all_dates = sorted(df["date"].unique())
    date_map = {d: i for i, d in enumerate(all_dates)}
    date_indices = work["date"].map(date_map).values

    if precomputed_ranks is not None:
        # Align ranks with current work (which has its own dropna)
        if isinstance(precomputed_ranks, pd.Series):
            ranks = precomputed_ranks.reindex(work.index).values
        else:
            ranks = precomputed_ranks
    else:
        # If no ranks provided, compute them cross-sectionally
        sort_idx = np.argsort(date_indices)
        work_sorted = work.iloc[sort_idx]
        d_s = date_indices[sort_idx]
        scores = work_sorted["score"].values

        # Grouped rank calculation using shift/diff to find boundaries
        boundaries = np.where(d_s[1:] != d_s[:-1])[0] + 1
        bounds = np.concatenate(([0], boundaries, [len(d_s)]))
        ranks_sorted = np.zeros_like(scores, dtype=float)
        for i in range(len(bounds) - 1):
            s, e = bounds[i], bounds[i + 1]
            n = e - s
            if n > 0:
                ranks_sorted[s:e] = rankdata(scores[s:e], method="average") / n
        
        ranks = np.zeros_like(ranks_sorted)
        ranks[sort_idx] = ranks_sorted

    # Apply mandate-specific weighting logic
    # Ensure ranks is a numpy array for vectorization, handle NaNs from reindex
    r_vals = np.nan_to_num(ranks, nan=0.5)
    
    if primary_path == "long_only_overlay":
        w_raw = np.maximum(r_vals - 0.5, 0.0)
    elif primary_path == "short_side":
        w_raw = -np.maximum(r_vals - 0.5, 0.0)
    else:  # long_short_spread
        w_raw = r_vals - 0.5

    # Grouped normalization: sum(|w|) per date using bincount
    abs_w = np.abs(w_raw)
    date_sums = np.bincount(date_indices, weights=abs_w, minlength=len(all_dates))
    expanded_sums = date_sums[date_indices]

    valid_mask = expanded_sums > 1e-12
    w_norm = np.zeros_like(w_raw)
    w_norm[valid_mask] = w_raw[valid_mask] / expanded_sums[valid_mask]

    res = work.copy()
    res["_w"] = w_norm

    # --- Weight Summary Trace ---
    if not res.empty:
        stats = res.groupby("date")["_w"].agg([
            lambda x: np.abs(x).sum(),
            lambda x: x.sum(),
            lambda x: (x > 1e-12).sum(),
            lambda x: (x < -1e-12).sum(),
            lambda x: x[x > 0].sum() if (x > 0).any() else 0.0,
            lambda x: x[x < 0].sum() if (x < 0).any() else 0.0
        ])
        stats.columns = ["sum_abs", "net", "n_long", "n_short", "long_exp", "short_exp"]
        
        m_sum_abs = float(stats["sum_abs"].mean())
        m_net = float(stats["net"].mean())
        max_net_a = float(stats["net"].abs().max())
        m_long = float(stats["long_exp"].mean())
        m_short = float(stats["short_exp"].mean())
        z_long = int((stats["n_long"] == 0).sum())
        z_short = int((stats["n_short"] == 0).sum())
        
        if DEBUG_DIAGNOSTICS:
            print(f"[Weight Summary] path={primary_path} | mean|w|={m_sum_abs:.4f} | net={m_net:.4f} | max|net|={max_net_a:.4f} | long={m_long:.4f} | short={m_short:.4f} | zero_long_days={z_long} | zero_short_days={z_short}")
        
        if TRACE_DIAGNOSTICS:
            for dt, group in res.groupby("date"):
                _abs_sum = np.abs(group["_w"]).sum()
                _net = group["_w"].sum()
                print(f"[Weight Audit] {dt.date()} | Sum|w|={_abs_sum:.4f} | Net={_net:.4f}")
    # ----------------------------

    return res[res["_w"].abs() > 1e-12]



def compute_execution_robustness(
    scored: pd.DataFrame,
    *,
    primary_path: str,
    target_col: str = "forward_return",
    model_name: str | None = None,
    window_idx: int | None = None,
) -> dict[str, float]:
    """Highly optimized flat-array execution robustness diagnostics with exact parity."""
    out = {
        "ic_mean": float("nan"),
        "icir": float("nan"),
        "halflife": float("nan"),
        "turnover_mean": float("nan"),
        "turnover_vol": float("nan"),
        "cost_ic": float("nan"),
        "caic_ratio": float("nan"),
        "capacity_weighted_ic": float("nan"),
        "decile_tail_stability": float("nan"),
        "hhi_concentration": float("nan"),
        "robustness_score": float("nan"),
        "robustness_reason": "none",
        "ic_valid_ratio": float("nan"),
        "ic_nan_ratio": float("nan"),
        # Legacy/Internal names for compatibility
        "cs_ic_spearman_mean": float("nan"),
        "daily_ic_annualized_icir": float("nan"),
        "daily_icir": float("nan"),
        "signal_halflife_days": float("nan"),
        "turnover_volatility": float("nan"),
        "cost_adjusted_ic_mean": float("nan"),
    }
    if scored.empty or "score" not in scored.columns:
        return out

    # 1. Spearman IC and Cost-Aware IC (Computed on valid subsets)
    # Raw IC on the primary target
    raw_ic_res = cross_sectional_ic(scored, target_col=target_col, model_name=model_name, window_idx=window_idx)
    out["cs_ic_spearman_mean"] = float(raw_ic_res.get("cs_ic_spearman_mean", float("nan")))
    out["daily_ic_annualized_icir"] = float(raw_ic_res.get("daily_ic_annualized_icir", float("nan")))
    out["ic_mean"] = out["cs_ic_spearman_mean"]
    out["ic_tstat"] = float(raw_ic_res.get("cs_ic_spearman_tstat", float("nan")))
    out["icir"] = out["daily_ic_annualized_icir"]
    out["daily_icir"] = out["daily_ic_annualized_icir"]
    out["ic_valid_days"] = int(raw_ic_res.get("ic_n_days", 0))
    out["ic_nan_days"] = int(raw_ic_res.get("ic_nan_days", 0))
    out["ic_valid_ratio"] = float(raw_ic_res.get("ic_valid_ratio", float("nan")))
    out["ic_nan_ratio"] = float(raw_ic_res.get("ic_nan_ratio", float("nan")))
    # P19: Per-window IC integrity counters for cross-process aggregation
    out["ic_constant_days"] = int(raw_ic_res.get("ic_constant_days", 0))
    out["ic_small_sample_days"] = int(raw_ic_res.get("ic_small_sample_days", 0))
    out["ic_nan_inf_days"] = int(raw_ic_res.get("ic_nan_inf_days", 0))

    # Cost-adjusted IC on net target
    cost_col = "target_return_net" if "target_return_net" in scored.columns else target_col
    ic_res = cross_sectional_ic(scored, target_col=cost_col, quiet=True)
    out["cost_ic"] = float(ic_res.get("cs_ic_spearman_mean", float("nan")))
    out["cost_adjusted_ic_mean"] = out["cost_ic"]
    
    if np.isfinite(out["ic_mean"]) and abs(out["ic_mean"]) > 1e-12:
        out["caic_ratio"] = out["cost_ic"] / out["ic_mean"]

    # 2. Compute Full Population Ranks (for Turnover, Halflife, HHI)
    work = scored[["date", "ticker", "score"]].dropna(subset=["date", "score"]).copy()
    if work.empty:
        return out

    all_dates = sorted(scored["date"].unique())
    date_map = {d: i for i, d in enumerate(all_dates)}
    date_indices = work["date"].map(date_map).values
    
    sort_idx = np.argsort(date_indices)
    work_s = work.iloc[sort_idx]
    d_s = date_indices[sort_idx]
    sc_s = work_s["score"].values
    
    boundaries = np.where(d_s[1:] != d_s[:-1])[0] + 1
    bounds = np.concatenate(([0], boundaries, [len(d_s)]))
    ranks_s = np.zeros_like(sc_s, dtype=float)
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        n = e - s
        if n > 0:
            ranks_s[s:e] = rankdata(sc_s[s:e], method="average") / n
    
    full_ranks = np.zeros_like(ranks_s)
    full_ranks[sort_idx] = ranks_s
    work["rank"] = full_ranks

    # 3. Implied Weights and HHI
    w_df = _get_implied_weights(work, primary_path=primary_path, precomputed_ranks=work["rank"])
    if not w_df.empty:
        w_vals = w_df["_w"].values
        w_date_indices = w_df["date"].map(date_map).values
        hhi_per_date = np.bincount(w_date_indices, weights=w_vals**2, minlength=len(all_dates))
        active_dates = np.bincount(w_date_indices, minlength=len(all_dates)) > 0
        if np.any(active_dates):
            out["hhi_concentration"] = float(np.mean(hhi_per_date[active_dates]))

        # 4. Turnover Volatility (Shift-Join to match pivot.diff() exactly)
        # Create a 'previous day' dataset for comparison
        prev_w = w_df[["date", "ticker", "_w"]].copy()
        prev_w["date_idx"] = prev_w["date"].map(date_map) + 1
        
        curr_w = w_df[["date", "ticker", "_w"]].copy()
        curr_w["date_idx"] = curr_w["date"].map(date_map)
        
        # Merge to find (w_t, w_{t-1}) pairs. Using outer join to capture entries and exits.
        # But wait, pivot().diff() only has rows for dates present in the index.
        # If we want to match piv.diff().abs().sum(axis=1), we only need Turnover for dates in all_dates.
        merged = pd.merge(
            curr_w, 
            prev_w, 
            on=["date_idx", "ticker"], 
            how="outer", 
            suffixes=("", "_prev")
        ).fillna(0.0)
        
        # We only care about turnover for dates in the original sample
        merged = merged[merged["date_idx"] < len(all_dates)]
        merged["diff"] = merged["_w"] - merged["_w_prev"]
        
        # Sum absolute diffs per date
        turnover_by_date = merged.groupby("date_idx")["diff"].apply(lambda x: x.abs().sum()).reindex(range(len(all_dates)), fill_value=0.0).values
        
        # Ensure array is writable (avoid ValueError: assignment destination is read-only)
        turnover_by_date = np.array(turnover_by_date, dtype=np.float64, copy=True)
        
        # Match pivot.diff() by ensuring first day turnover is 0.0
        if turnover_by_date.size > 0:
            turnover_by_date[0] = 0.0
            
        # Match pandas.std() on pivot.diff() which has T rows
        if len(turnover_by_date) > 1:
            out["turnover_vol"] = float(np.std(turnover_by_date, ddof=1))
            out["turnover_volatility"] = out["turnover_vol"]
        
        # New: Average Turnover for gating (matches pivot.diff().abs().sum().iloc[1:].mean())
        if len(turnover_by_date) > 1:
            out["turnover_mean"] = float(np.mean(turnover_by_date[1:]))
        elif len(turnover_by_date) == 1:
             out["turnover_mean"] = 0.0


    # 5. Signal Halflife (Average of Daily Correlations)
    ticker_ids_r = work["ticker"].astype("category").cat.codes.values
    sort_idx_r = np.lexsort((date_indices, ticker_ids_r))
    t_r = ticker_ids_r[sort_idx_r]
    d_r = date_indices[sort_idx_r]
    v_r = work["rank"].values[sort_idx_r]

    is_pair = (t_r[1:] == t_r[:-1]) & (d_r[1:] == d_r[:-1] + 1)
    if np.any(is_pair):
        v0 = v_r[:-1][is_pair]
        v1 = v_r[1:][is_pair]
        dates_pair = d_r[1:][is_pair]
        
        # Vectorized grouped correlation components
        sum_v0 = np.bincount(dates_pair, weights=v0, minlength=len(all_dates))
        sum_v1 = np.bincount(dates_pair, weights=v1, minlength=len(all_dates))
        sum_v0v1 = np.bincount(dates_pair, weights=v0 * v1, minlength=len(all_dates))
        sum_v0sq = np.bincount(dates_pair, weights=v0**2, minlength=len(all_dates))
        sum_v1sq = np.bincount(dates_pair, weights=v1**2, minlength=len(all_dates))
        n_pairs = np.bincount(dates_pair, minlength=len(all_dates))
        
        # Only compute for days with sufficient pairs
        valid_days = n_pairs > 10
        if np.any(valid_days):
            n = n_pairs[valid_days]
            mu0, mu1 = sum_v0[valid_days] / n, sum_v1[valid_days] / n
            cov = (sum_v0v1[valid_days] / n) - (mu0 * mu1)
            var0 = (sum_v0sq[valid_days] / n) - (mu0**2)
            var1 = (sum_v1sq[valid_days] / n) - (mu1**2)
            
            denom = np.sqrt(np.maximum(var0, 0) * np.maximum(var1, 0))
            rho_t = np.full_like(denom, np.nan, dtype=float)
            np.divide(cov, denom, out=rho_t, where=denom > 1e-12)

            finite_rho = rho_t[np.isfinite(rho_t)]
            if finite_rho.size:
                avg_rho = float(np.mean(finite_rho))
                if 0 < avg_rho < 1.0:
                    out["halflife"] = -np.log(2) / np.log(avg_rho)
                elif avg_rho >= 1.0:
                    out["halflife"] = 100.0
                else:
                    out["halflife"] = 0.0
                out["signal_halflife_days"] = out["halflife"]

    # 6. Capacity Weighted IC
    cap_col = (
        "adv_dollar_20"
        if "adv_dollar_20" in scored.columns
        else ("dollar_volume" if "dollar_volume" in scored.columns else None)
    )
    if cap_col and cap_col in scored.columns:
        df_cap = scored[["date", "score", target_col, cap_col]].dropna()
        if not df_cap.empty:
            d_codes = df_cap["date"].map(date_map).values
            sc = df_cap["score"].values
            tc = df_cap[target_col].values
            caps = df_cap[cap_col].values

            sum_caps = np.bincount(d_codes, weights=caps, minlength=len(all_dates))
            m_sc = np.bincount(d_codes, weights=sc * caps, minlength=len(all_dates)) / (sum_caps + 1e-12)
            m_tc = np.bincount(d_codes, weights=tc * caps, minlength=len(all_dates)) / (sum_caps + 1e-12)

            sc_dm = sc - m_sc[d_codes]
            tc_dm = tc - m_tc[d_codes]

            w_cov = np.bincount(d_codes, weights=caps * sc_dm * tc_dm, minlength=len(all_dates))
            w_var_sc = np.bincount(d_codes, weights=caps * sc_dm**2, minlength=len(all_dates))
            w_var_tc = np.bincount(d_codes, weights=caps * tc_dm**2, minlength=len(all_dates))

            denom = np.sqrt(w_var_sc * w_var_tc)
            w_ic = np.full_like(denom, np.nan, dtype=float)
            np.divide(w_cov, denom, out=w_ic, where=denom > 1e-12)
            finite_w_ic = w_ic[np.isfinite(w_ic)]
            if finite_w_ic.size:
                out["capacity_weighted_ic"] = float(np.mean(finite_w_ic))

    # 7. Final Schema Validation (Task 4)
    required_keys = ["halflife", "turnover_mean", "turnover_vol", "cost_ic", "caic_ratio", "ic_mean", "icir"]
    for k in required_keys:
        if k not in out:
            raise MetricIntegrityError(f"Missing required metric key: {k}")

    return out

# ── Cross-sectional IC cache ─────────────────────────────────────────────────
# Cache IC results by (score_hash, target_hash, horizon_days) to avoid redundant
# computation when the same scored/target data is passed multiple times.
_ic_cache: dict[tuple, tuple[dict[str, Any], pd.Series | None]] = {}
_IC_CACHE_MAX_ENTRIES = 512


def robust_spearman(a: np.ndarray, b: np.ndarray, *, min_obs: int = 5, min_unique: int = 2) -> float:
    """
    P19: Safe Spearman rank correlation with explicit edge-case handling.

    Returns NaN when:
      - fewer than min_obs valid observations
      - fewer than min_unique distinct values in either input (constant array)
      - zero variance in either input
      - correlation computation fails

    This prevents the Pandas/Scipy ``ConstantInputWarning`` at the source
    by checking pre-conditions before delegating to ``scipy.stats.spearmanr``.

    All internal IC/diagnostic call sites that currently call spearmanr,
    .corr(method='spearman'), or .rank().corr() directly should migrate
    to this function to eliminate spurious warnings and ensure consistent
    NaN semantics for degenerate inputs.
    """
    import contextlib, io, warnings as _w

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    n_valid = mask.sum()

    if n_valid < min_obs:
        return float("nan")

    a_valid = a[mask]
    b_valid = b[mask]

    if np.max(a_valid) == np.min(a_valid) or np.max(b_valid) == np.min(b_valid):
        return float("nan")

    if len(np.unique(a_valid)) < min_unique or len(np.unique(b_valid)) < min_unique:
        return float("nan")

    with _w.catch_warnings():
        _w.simplefilter("ignore", category=RuntimeWarning)
        r, _ = _sps.spearmanr(a_valid, b_valid)
        return float(r) if math.isfinite(r) else float("nan")


# P19: Observability — IC integrity tracking via per-call counters returned in
# the result dict.  The old module-level global `_IC_INTEGRITY` was process-local
# and invisible across fork boundaries, so the main process always saw zeros.
# Counters are now embedded in the `cross_sectional_ic` return dict and aggregated
# by the orchestrator from worker result payloads.
#
# The legacy global and `ic_integrity_report()` / `ic_integrity_check()` are
# retained for backward compatibility but emit a deprecation warning.  New code
# should use `aggregate_ic_integrity()` in the orchestrator.

_IC_INTEGRITY = {"total": 0, "valid": 0, "invalid_constant": 0, "invalid_small": 0, "invalid_nan": 0}


def _track_ic_call(valid: bool, reason: str = "") -> None:
    _IC_INTEGRITY["total"] += 1
    if valid:
        _IC_INTEGRITY["valid"] += 1
    elif "constant" in reason:
        _IC_INTEGRITY["invalid_constant"] += 1
    elif "nan" in reason or "inf" in reason:
        _IC_INTEGRITY["invalid_nan"] += 1
    else:
        _IC_INTEGRITY["invalid_small"] += 1


def ic_integrity_report() -> dict[str, int | float]:
    """Legacy: reads the process-local global.  Returns zeros in fork-worker mode.
    Use `aggregate_ic_integrity(worker_results)` in the orchestrator instead."""
    t = _IC_INTEGRITY
    total = max(t["total"], 1)
    return {
        "ic_total_evaluations": t["total"],
        "ic_valid_count": t["valid"],
        "ic_invalid_constant_count": t["invalid_constant"],
        "ic_invalid_small_sample_count": t["invalid_small"],
        "ic_invalid_nan_count": t.get("invalid_nan", 0),
        "ic_invalid_ratio": round((total - t["valid"]) / total, 4),
    }


def aggregate_ic_integrity(
    worker_results: list[dict[str, Any]],
    *,
    max_invalid_ratio: float = 0.20,
) -> dict[str, int | float]:
    """Aggregate IC integrity counters from worker result payloads.

    Each worker result dict may contain per-window IC counters:
      ic_n_days, ic_nan_days, ic_valid_ratio, ic_constant_days, ic_small_sample_days

    Returns a consolidated report with:
      ic_total_evaluations, ic_valid_count, ic_invalid_constant_count,
      ic_invalid_small_sample_count, ic_invalid_nan_count, ic_invalid_ratio,
      ic_status (PASS/FAIL)
    """
    total = 0
    valid = 0
    constant = 0
    small = 0
    nan_inf = 0

    for result in worker_results:
        if not isinstance(result, dict):
            continue
        # Per-window IC stats from cross_sectional_ic return dict
        n_valid = int(result.get("ic_n_days", 0) or 0)
        n_nan = int(result.get("ic_nan_days", 0) or 0)
        n_constant = int(result.get("ic_constant_days", 0) or 0)
        n_small = int(result.get("ic_small_sample_days", 0) or 0)
        n_nan_inf = int(result.get("ic_nan_inf_days", 0) or 0)

        window_total = n_valid + n_nan
        if window_total > 0:
            total += window_total
            valid += n_valid
            constant += n_constant
            small += n_small
            nan_inf += n_nan_inf

    # Fallback: if no per-window counters found, try legacy global
    if total == 0:
        legacy = ic_integrity_report()
        total = legacy["ic_total_evaluations"]
        valid = legacy["ic_valid_count"]
        constant = legacy["ic_invalid_constant_count"]
        small = legacy["ic_invalid_small_sample_count"]
        nan_inf = legacy.get("ic_invalid_nan_count", 0)

    invalid = total - valid
    total_safe = max(total, 1)
    return {
        "ic_total_evaluations": total,
        "ic_valid_count": valid,
        "ic_invalid_constant_count": constant,
        "ic_invalid_small_sample_count": small,
        "ic_invalid_nan_count": nan_inf,
        "ic_invalid_ratio": round(invalid / total_safe, 4),
        "ic_status": "PASS" if (invalid / total_safe) <= max_invalid_ratio else "FAIL",
    }


def ic_integrity_check(max_invalid_ratio: float = 0.20) -> None:
    """P19 fail-fast: raise if too many IC evaluations are degenerate."""
    report = ic_integrity_report()
    if report["ic_invalid_ratio"] > max_invalid_ratio:
        raise RuntimeError(
            f"IC integrity failure: {report['ic_invalid_ratio']*100:.1f}% of IC evaluations are invalid "
            f"({report['ic_invalid_constant_count']} constant, {report['ic_invalid_small_sample_count']} small-sample "
            f"out of {report['ic_total_evaluations']} total). "
            f"Threshold={max_invalid_ratio*100:.0f}%. Check feature quality, target coverage, or cross-sectional size."
        )


def cross_sectional_ic(
    scored: pd.DataFrame,
    *,
    target_col: str = "forward_return",
    return_ranks: bool = False,
    model_name: str | None = None,
    window_idx: int | None = None,
    quiet: bool = False,
    horizon_days: int = 5,
) -> dict[str, Any] | tuple[dict[str, Any], pd.Series | None]:
    """Daily cross-sectional Pearson/Spearman IC with HAC t-stat over dates.

    horizon_days is used to set the NW HAC lag (max_lag = max(5, h-1)) to account for
    autocorrelation introduced by overlapping h-day forward returns, and to compute
    horizon_adj_ic_ir = IC_mean/IC_std × sqrt(252/h) — the non-overlapping Grinold IR.
    """
    empty_stats = {
        "cs_ic_pearson_mean": float("nan"),
        "cs_ic_spearman_mean": float("nan"),
        "cs_ic_spearman_std": float("nan"),
        "cs_ic_spearman_ir": float("nan"),
        "cs_ic_spearman_annualized_icir": float("nan"),
        "cs_ic_spearman_tstat": float("nan"),
        "cs_ic_positive_rate": float("nan"),
        "cs_ic_n_days": 0,
        "daily_ic_mean": float("nan"),
        "daily_ic_std": float("nan"),
        "daily_ic_annualized_icir": float("nan"),
        "daily_ic_hac_tstat": float("nan"),
        "daily_ic_positive_rate": float("nan"),
        "daily_ic_n_days": 0,
        "ic_valid_ratio": float("nan"),
        "ic_nan_ratio": float("nan"),
        "horizon_adj_ic_ir": float("nan"),
        "horizon_adj_ic_tstat": float("nan"),
        # P19: Per-window IC integrity counters for cross-process aggregation
        "ic_n_days": 0,
        "ic_nan_days": 0,
        "ic_constant_days": 0,
        "ic_small_sample_days": 0,
        "ic_nan_inf_days": 0,
    }
    # [IC PIPELINE TRACE 1: cross_sectional_ic]
    from model_selection.validation import TRACE_DIAGNOSTICS
    if TRACE_DIAGNOSTICS:
        print(f"\n[IC Pipeline Trace: cross_sectional_ic]")
        print(f"  target_col   : {target_col}")
        print(f"  n_obs        : {len(scored) if scored is not None else 0}")
        print(f"  n_dates      : {scored['date'].nunique() if scored is not None and not scored.empty else 0}")
        print(f"  grouping     : per-date (cross-sectional)")
        print(f"  method       : spearman (rank)")
        print(f"  weights      : none applied")
    # ------------------------------------------
    if scored is None or scored.empty or target_col not in scored.columns:
        return (empty_stats, None) if return_ranks else empty_stats

    df = scored[["date", "score", target_col]].dropna()
    if df.empty:
        return (empty_stats, None) if return_ranks else empty_stats

    if not return_ranks:
        score_hash = _stable_cache_token(df["score"])
        target_hash = _stable_cache_token(df[target_col])
        cache_key = (score_hash, target_hash, horizon_days)
        if cache_key in _ic_cache:
            cached_result, _ = _ic_cache[cache_key]
            # P31: Track cached hits for integrity reporting
            _track_ic_call(valid=True)
            return cached_result

    # Keep track of original index for rank alignment
    original_idx = df.index
    df = df.copy()
    df["date_id"] = df["date"].astype("category").cat.codes
    # Sort for boundary-based reductions
    sort_idx = np.argsort(df["date_id"].values)
    df_sorted = df.iloc[sort_idx]

    dates = df_sorted["date_id"].values
    scores = df_sorted["score"].values
    targets = df_sorted[target_col].values

    boundaries = np.where(dates[1:] != dates[:-1])[0] + 1
    bounds = np.concatenate(([0], boundaries, [len(dates)]))

    pearson, spearman = [], []
    all_sc_ranks = [] if return_ranks else None
    # P19: Per-window IC integrity counters
    n_constant_days = 0
    n_small_sample_days = 0
    n_nan_inf_days = 0

    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i + 1]
        n = e - s
        if n < 5:
            _track_ic_call(valid=False, reason="small_sample")
            n_small_sample_days += 1
            if return_ranks:
                all_sc_ranks.append(np.full(n, np.nan))
            pearson.append(np.nan)
            spearman.append(np.nan)
            continue

        sc, tc = scores[s:e], targets[s:e]
        if np.max(sc) == np.min(sc) or np.max(tc) == np.min(tc):
            _track_ic_call(valid=False, reason="constant")
            n_constant_days += 1
            if return_ranks:
                all_sc_ranks.append(rankdata(sc, method="average") / n)
            pearson.append(np.nan)
            spearman.append(np.nan)
            continue

        sc_m, tc_m = sc - np.mean(sc), tc - np.mean(tc)
        cov = np.sum(sc_m * tc_m)
        var = np.sqrt(np.sum(sc_m ** 2) * np.sum(tc_m ** 2))
        p_val = cov / var if var > 1e-12 else np.nan
        pearson.append(p_val)

        sc_rank, tc_rank = rankdata(sc, method="average"), rankdata(tc, method="average")
        if return_ranks:
            all_sc_ranks.append(sc_rank / n)

        sc_rm, tc_rm = sc_rank - np.mean(sc_rank), tc_rank - np.mean(tc_rank)
        scov = np.sum(sc_rm * tc_rm)
        svar = np.sqrt(np.sum(sc_rm ** 2) * np.sum(tc_rm ** 2))
        s_val = scov / svar if svar > 1e-12 else np.nan
        spearman.append(s_val)
        _track_ic_call(valid=True)

    s_arr = np.array(spearman)
    p_arr = np.array(pearson)
    
    total_days = len(bounds) - 1
    valid_days = int(np.isfinite(s_arr).sum())
    # Count NaN/inf days that are NOT constant or small-sample (i.e., degenerate correlations)
    n_nan_inf_days = int(np.sum(~np.isfinite(s_arr) & (s_arr != s_arr)))  # NaN from var<=0
    # Also count inf values
    n_nan_inf_days += int(np.sum(np.isinf(s_arr)))
    
    unique_dates = df_sorted["date"].unique()
    s = pd.Series(s_arr, index=unique_dates, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    p = pd.Series(p_arr, index=unique_dates, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()

    if s.empty:
        stats = empty_stats.copy()
        stats["ic_valid_ratio"] = 0.0
        stats["ic_nan_ratio"] = 1.0
        stats["ic_nan_days"] = total_days
        stats["ic_constant_days"] = n_constant_days
        stats["ic_small_sample_days"] = n_small_sample_days
        stats["ic_nan_inf_days"] = n_nan_inf_days
        return (stats, None) if return_ranks else stats

    # Task 1: Suppress expected warnings locally
    with np.errstate(divide="ignore", invalid="ignore"):
        mu = float(s.mean())
        std = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
        ir = mu / std if np.isfinite(std) and std > 1e-12 else float("nan")

        # HAC t-stat: NW lag set to max(5, h-1) so full h-day return overlap autocorrelation
        # (ρ_k ≈ (h-k)/h for k<h) is captured. At h=10, lag-9 IC observations share 10% overlap.
        r = s.to_numpy(dtype=float)
        demeaned = r - mu
        _h = max(1, int(horizon_days))
        max_lag = min(max(5, _h - 1), len(r) - 1)
        var_hac = float(np.dot(demeaned, demeaned) / len(r))
        for lag in range(1, max_lag + 1):
            gamma = float(np.dot(demeaned[lag:], demeaned[:-lag]) / len(r))
            var_hac += 2.0 * (1.0 - lag / (max_lag + 1.0)) * gamma

        se_hac = np.sqrt(var_hac / len(r)) if var_hac > 1e-12 else float("nan")
        t_hac = mu / se_hac if np.isfinite(se_hac) and se_hac > 1e-12 else float("nan")

        # --- IC Summary ---
        ic_n_days = len(r)
        ic_nan_days = total_days - valid_days
        
        if DEBUG_DIAGNOSTICS and not quiet:
            m_str = f" model={model_name}" if model_name else ""
            w_str = f" window={window_idx}" if window_idx is not None else ""
            print(f"[IC Summary]{m_str}{w_str} | target={target_col} | mean={mu:.6f} | std={std:.6f} | ir={ir:.6f} | tstat={t_hac:.4f} | valid_days={valid_days} | nan_days={ic_nan_days}")
        
        if DEBUG_DIAGNOSTICS and not quiet and abs(mu) < 0.002 and abs(t_hac) < 1.0:
            print(f"[Signal Warning]\nWeak or no predictive signal detected\nic_mean={mu:.6f}\nic_tstat={t_hac:.4f}")
            
        if TRACE_DIAGNOSTICS and not quiet:
            for dt, ic_val in s.items():
                print(f"[IC Trace] {dt.date()} | spearman={ic_val:.4f}")
        # ------------------

        # horizon_adj_ic_ir: non-overlapping Grinold IR = IC_mean/IC_std × sqrt(252/h).
        # Grinold (1994) annualised IR for a signal evaluated at h-day horizon uses
        # sqrt(252/h) non-overlapping periods per year, not sqrt(252).
        # cs_ic_spearman_annualized_icir uses sqrt(252), which overstates IR by sqrt(h).
        _adj_ir = ir * np.sqrt(252.0 / float(_h)) if np.isfinite(ir) else float("nan")
        # horizon_adj_ic_tstat: non-overlapping equivalent t-stat = IC_mean × sqrt(N/h) / IC_std.
        # Uses IC_std (raw daily IC std) as the per-observation noise scale; N/h gives the
        # number of effectively independent observations.  Do NOT use var_hac here — that is the
        # HAC denominator for the N-obs HAC t-stat; substituting it would double-count the
        # overlap correction (var_hac ≈ std² × 6.7 at h=10 already accounts for the MA(h-1)
        # autocorrelation, so dividing again by sqrt(h) would under-state the t-stat by sqrt(6.7/h)).
        _n_indep = max(1.0, float(len(r)) / float(_h))
        _adj_tstat = (
            mu * np.sqrt(_n_indep) / std
            if np.isfinite(std) and std > 1e-9
            else float("nan")
        )

        se = np.sqrt(max(var_hac, 0.0) / len(r)) if len(r) else float("nan")
        tstat = mu / se if np.isfinite(se) and se > 1e-12 else float("nan")

        res_dict = {
            "cs_ic_pearson_mean": float(p.mean()) if not p.empty else float("nan"),
            "cs_ic_spearman_mean": mu,
            "cs_ic_spearman_std": std,
            "cs_ic_spearman_ir": ir,
            "cs_ic_spearman_annualized_icir": ir * np.sqrt(252.0) if np.isfinite(ir) else float("nan"),
            "cs_ic_spearman_tstat": tstat,
            "cs_ic_positive_rate": float((s > 0).mean()),
            "cs_ic_n_days": len(s),
            "daily_ic_mean": mu,
            "daily_ic_std": std,
            "daily_ic_annualized_icir": ir * np.sqrt(252.0) if np.isfinite(ir) else float("nan"),
            "daily_ic_hac_tstat": tstat,
            "daily_ic_positive_rate": float((s > 0).mean()),
            "daily_ic_n_days": len(s),
            "ic_n_days": valid_days,
            "ic_nan_days": total_days - valid_days,
            "ic_valid_ratio": float(valid_days / total_days) if total_days > 0 else 0.0,
            "ic_nan_ratio": 1.0 - (float(valid_days / total_days) if total_days > 0 else 0.0),
            "horizon_adj_ic_ir": float(_adj_ir) if np.isfinite(_adj_ir) else float("nan"),
            "horizon_adj_ic_tstat": float(_adj_tstat) if np.isfinite(_adj_tstat) else float("nan"),
            # P19: Per-window IC integrity counters for cross-process aggregation
            "ic_constant_days": n_constant_days,
            "ic_small_sample_days": n_small_sample_days,
            "ic_nan_inf_days": n_nan_inf_days,
        }

    if return_ranks:
        final_ranks_sorted = np.concatenate(all_sc_ranks)
        # Map back to original order
        final_ranks = np.zeros_like(final_ranks_sorted)
        final_ranks[sort_idx] = final_ranks_sorted
        return res_dict, pd.Series(final_ranks, index=original_idx)

    if len(_ic_cache) < _IC_CACHE_MAX_ENTRIES:
        _ic_cache[cache_key] = (res_dict.copy(), None)

    return res_dict



# --- SIMULATION ENGINE ---


def simulate_executable_portfolio(
    scored: pd.DataFrame,
    cfg: EvaluationConfig,
    *,
    state_cache: ValidationStateCache | None = None,
    target_weights: pd.DataFrame | None = None,
) -> tuple[pd.Series, pd.DataFrame]:
    """Daily MTM simulator using pre-constructed target weights and execution costs."""
    if scored is None or scored.empty:
        return pd.Series(dtype=float, name="daily_return"), pd.DataFrame()

    df = _canonicalize_scored_panel(scored)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["next_return"] = _next_day_returns(df, horizon_days=int(cfg.horizon_days))
    df = df.dropna(subset=["date", "ticker", "next_return"]).sort_values(["date", "ticker"])

    cache_path = _simulation_cache_path(df, cfg, state_cache, target_weights=target_weights)
    cached = _read_simulation_cache(cache_path)
    if cached is not None:
        return cached

    targets = (
        target_weights.copy()
        if target_weights is not None
        else build_target_weights(df, cfg, state_cache=state_cache)
    )
    if not targets.empty:
        required_cols = {"date", "ticker", "target_weight"}
        missing = required_cols - set(targets.columns)
        if missing:
            raise MetricIntegrityError(f"target_weights missing required columns: {sorted(missing)}")
        targets["date"] = pd.to_datetime(targets["date"], errors="coerce")
        targets["ticker"] = targets["ticker"].astype(str)
        targets["target_weight"] = pd.to_numeric(targets["target_weight"], errors="coerce").fillna(0.0)
        _assert_unique_target_weights(targets)
    if targets.empty:
        pnl = _zero_execution_pnl(df)
        ser = pd.Series(
            pnl["daily_return"].to_numpy(dtype=float),
            index=pd.to_datetime(pnl["date"]),
            name="daily_return",
        )
        _write_simulation_cache(cache_path, ser, pnl)
        return ser, pnl

    rows: list[dict[str, Any]] = []
    current = pd.Series(dtype=float)
    permanent_impact_state = pd.Series(dtype=float)
    target_by_date = {pd.Timestamp(k): g for k, g in targets.groupby("date", sort=True)}
    
    path_audit = str(cfg.path or "").lower()

    date_groups = list(df.groupby("date", sort=True))
    all_dates = [pd.Timestamp(k) for k, _ in date_groups]

    for raw_dt, day in date_groups:
        dt = pd.Timestamp(raw_dt)
        market_state = state_cache.get(dt) if state_cache is not None else None

        if not permanent_impact_state.empty:
            decay_days = max(1, int(cfg.costs.permanent_impact_decay_days))
            permanent_before_decay = float(permanent_impact_state.abs().sum())
            permanent_impact_state *= max(0.0, 1.0 - 1.0 / float(decay_days))
            permanent_impact_state = permanent_impact_state[permanent_impact_state.abs() > 1e-12]
            permanent_decay = max(0.0, permanent_before_decay - float(permanent_impact_state.abs().sum()))
        else:
            permanent_decay = 0.0

        delta = pd.Series(dtype=float)
        cost = long_cost = short_cost = commission = spread = fixed_cost = temporary_impact = permanent_impact = 0.0
        trade_notional = 0.0
        trade_count = capped_trade_count = 0
        participation_rates: list[float] = []
        control_metrics: dict[str, float | int | str] = {}

        if dt in target_by_date:
            target_frame = target_by_date[dt]
            for col in (
                "_lambda_risk",
                "_gamma_turnover",
                "_expected_alpha",
                "_expected_cost",
                "_realized_volatility",
                "_signal_ic",
                "_control_support_days",
            ):
                if col in target_frame.columns:
                    vals = target_frame[col].dropna()
                    if not vals.empty:
                        control_metrics[col[1:]] = float(vals.iloc[0])
            if "_control_status" in target_frame.columns:
                vals = target_frame["_control_status"].dropna()
                if not vals.empty:
                    control_metrics["control_status"] = str(vals.iloc[0])
            target = target_frame.set_index("ticker")["target_weight"].astype(float)
            immutable_target = target.copy(deep=True)
            if path_audit == "long_short_spread" and abs(float(target.sum())) > float(cfg.net_exposure_max) + 1e-8:
                raise MetricIntegrityError(
                    f"Simulator received non-neutral target weights on {dt.date()}: "
                    f"net={float(target.sum()):.6f}, limit={float(cfg.net_exposure_max):.6f}. "
                    "PortfolioConstructor must enforce neutrality before simulation."
                )

            union = current.index.union(target.index)
            prev = current.reindex(union).fillna(0.0)
            nxt = target.reindex(union).fillna(0.0)
            if not target.equals(immutable_target):
                raise MetricIntegrityError("Simulator attempted to mutate target weights after construction")
            delta = nxt - prev
            delta_active = delta[delta.abs() > 1e-12]
            vector_costs = None
            if market_state is not None and not delta_active.empty:
                vector_costs = _vectorized_trade_costs_from_market_state(
                    delta_active.index.to_numpy(dtype=object),
                    delta_active.to_numpy(dtype=float),
                    prev.reindex(delta_active.index).to_numpy(dtype=float),
                    nxt.reindex(delta_active.index).to_numpy(dtype=float),
                    market_state,
                    cfg,
                )
            if vector_costs is not None:
                cost += float(vector_costs["cost"])
                long_cost += float(vector_costs["long_cost"])
                short_cost += float(vector_costs["short_cost"])
                commission += float(vector_costs["commission"])
                spread += float(vector_costs["spread"])
                fixed_cost += float(vector_costs["fixed_cost"])
                temporary_impact += float(vector_costs["temporary_impact"])
                permanent_impact += float(vector_costs["permanent_impact"])
                trade_notional += float(vector_costs["trade_notional"])
                trade_count += int(vector_costs["trade_count"])
                capped_trade_count += int(vector_costs["capped_trade_count"])
                participation_rates = list(np.asarray(vector_costs["participation_rates"], dtype=float))
                permanent_updates = vector_costs["permanent_updates"]
                if isinstance(permanent_updates, pd.Series) and not permanent_updates.empty:
                    permanent_impact_state = permanent_impact_state.add(permanent_updates, fill_value=0.0)
            else:
                by_ticker = day.set_index("ticker")
                for ticker, dw in delta_active.items():
                    is_exit = abs(float(nxt.get(ticker, 0.0))) < abs(float(prev.get(ticker, 0.0)))
                    if market_state is not None:
                        breakdown = _trade_cost_breakdown_from_market_state(
                            float(dw), str(ticker), market_state, cfg, is_exit=is_exit
                        )
                    else:
                        row = by_ticker.loc[ticker] if ticker in by_ticker.index else pd.Series(dtype=float)
                        breakdown = _trade_cost_breakdown(float(dw), row, cfg, is_exit=is_exit)

                    cost += breakdown.cost_return
                    if float(nxt.get(ticker, 0.0)) >= 0.0:
                        long_cost += breakdown.cost_return
                    else:
                        short_cost += breakdown.cost_return

                    commission += breakdown.commission_return
                    spread += breakdown.spread_return
                    fixed_cost += breakdown.fixed_cost_return
                    temporary_impact += breakdown.temporary_impact_return
                    permanent_impact += breakdown.permanent_impact_return
                    trade_notional += breakdown.trade_notional
                    trade_count += 1
                    capped_trade_count += int(breakdown.participation_capped)
                    participation_rates.append(breakdown.participation_rate)

                    if breakdown.permanent_impact_return > 0.0:
                        permanent_impact_state.loc[str(ticker)] = float(
                            permanent_impact_state.get(str(ticker), 0.0)
                        ) + np.sign(float(dw)) * breakdown.permanent_impact_return
            current = nxt[nxt.abs() > 1e-12]

        ret_map = pd.Series(
            pd.to_numeric(day["next_return"], errors="coerce").to_numpy(dtype=float),
            index=day["ticker"].astype(str).to_numpy(dtype=object),
            dtype=float,
        )
        aligned = current.reindex(ret_map.index).fillna(0.0)
        gross_pnl = float((aligned * ret_map).sum())
        long_pnl = float((aligned[aligned > 0.0] * ret_map.reindex(aligned.index).fillna(0.0)).sum())
        short_pnl = float((aligned[aligned < 0.0] * ret_map.reindex(aligned.index).fillna(0.0)).sum())
        
        # --- Diagnostic Trace ---
        path_lower = str(cfg.path or "").lower()
        if path_lower in ["long_short_spread", "short_side", "long_only_overlay"]:
            n_long = int((aligned > 1e-12).sum())
            n_short = int((aligned < -1e-12).sum())
            w_long = float(aligned[aligned > 0.0].sum())
            w_short = float(aligned[aligned < 0.0].sum())
            check_pnl = long_pnl + short_pnl
            if TRACE_DIAGNOSTICS:
                print(f"[PnL Trace] {dt.date()} | Path: {path_lower} | Gross: {gross_pnl:.6f} | Long: {long_pnl:.6f} | Short: {short_pnl:.6f} | Check: {check_pnl:.6f}")
                print(f"    Positions: L={n_long}, S={n_short} | Weights: L={w_long:.4f}, S={w_short:.4f}")
            
            if dt == all_dates[-1] and DEBUG_DIAGNOSTICS:
                day_sorted = day.sort_values("score", ascending=False)
                top_10 = day_sorted.head(10).copy()
                bot_10 = day_sorted.tail(10).copy()
                top_10["weight"] = top_10["ticker"].map(aligned).fillna(0.0)
                bot_10["weight"] = bot_10["ticker"].map(aligned).fillna(0.0)
                print(f"--- Score/Weight Audit ({dt.date()}, Path: {path_lower}) ---")
                print("Top 10 Scores (Highest):")
                print(top_10[["ticker", "score", "weight"]])
                print("Bottom 10 Scores (Lowest):")
                print(bot_10[["ticker", "score", "weight"]])
        # ------------------------

        borrow = float(aligned[aligned < 0.0].abs().sum()) * float(cfg.costs.borrow_bps) / 10_000.0 / 252.0
        
        _mkt_adj = 0.0
        if path_lower == "short_side" and bool(getattr(cfg, "market_neutral_shorts", True)):
            _mkt_ret = float(ret_map.mean()) if (not ret_map.empty and np.isfinite(ret_map.mean())) else 0.0
            _net_exp = float(current.sum())
            if np.isfinite(_mkt_ret) and np.isfinite(_net_exp) and not current.empty:
                _mkt_adj = _net_exp * _mkt_ret
                if TRACE_DIAGNOSTICS:
                    print(f"[MktAdj Audit] {dt.date()} | PnL_raw: {gross_pnl:.6f} | NetExp: {_net_exp:.4f} | MktRet: {_mkt_ret:.6f} | Adj: {_mkt_adj:.6f} | PnL_adj: {gross_pnl - _mkt_adj:.6f}")
        net = gross_pnl - _mkt_adj - cost - borrow



        rows.append(
            {
                "date": dt,
                "daily_return": net,
                "gross_return": gross_pnl,
                "market_adj_return": _mkt_adj,
                "long_gross_return": long_pnl,
                "short_gross_return": short_pnl,
                "long_cost_return": long_cost,
                "short_cost_return": short_cost,
                "cost_return": cost,
                "commission_return": commission,
                "spread_return": spread,
                "fixed_cost_return": fixed_cost,
                "temporary_impact_return": temporary_impact,
                "permanent_impact_return": permanent_impact,
                "permanent_impact_decay_return": permanent_decay,
                "permanent_impact_unamortized_return": float(permanent_impact_state.abs().sum())
                if not permanent_impact_state.empty
                else 0.0,
                "borrow_return": borrow,
                "gross_exposure": float(current.abs().sum()),
                "long_exposure": float(current[current > 0.0].sum()) if not current.empty else 0.0,
                "short_exposure": float(current[current < 0.0].abs().sum()) if not current.empty else 0.0,
                "net_exposure": float(current.sum()),
                "turnover": float(delta.abs().sum()) if dt in target_by_date else 0.0,
                "trade_count": int(trade_count),
                "trade_notional": float(trade_notional),
                "participation_rate_mean": float(np.mean(participation_rates)) if participation_rates else 0.0,
                "participation_rate_p95": float(np.percentile(participation_rates, 95)) if participation_rates else 0.0,
                "participation_rate_max": float(np.max(participation_rates)) if participation_rates else 0.0,
                "participation_over_5pct_count": int(np.sum(np.asarray(participation_rates) > 0.05))
                if participation_rates
                else 0,
                "participation_over_10pct_count": int(np.sum(np.asarray(participation_rates) > 0.10))
                if participation_rates
                else 0,
                "participation_capped_count": int(capped_trade_count),
                "beta_exposure": _portfolio_beta_from_aligned_day(day, aligned),
                "max_sector_exposure": _max_sector_exposure_from_aligned_day(day, aligned),
                "n_positions": int((current.abs() > 1e-12).sum()),
                "n_long_positions": int((current > 1e-12).sum()),
                "n_short_positions": int((current < -1e-12).sum()),
                "lambda_risk": float(control_metrics.get("lambda_risk", cfg.lambda_risk)),
                "gamma_turnover": float(control_metrics.get("gamma_turnover", cfg.gamma_turnover)),
                "expected_alpha": float(control_metrics.get("expected_alpha", np.nan)),
                "expected_cost": float(control_metrics.get("expected_cost", np.nan)),
                "realized_volatility": float(control_metrics.get("realized_volatility", np.nan)),
                "signal_ic": float(control_metrics.get("signal_ic", np.nan)),
                "control_support_days": int(control_metrics.get("control_support_days", 0)),
                "control_status": str(control_metrics.get("control_status", "not_rebalanced")),
            }
        )

    pnl = pd.DataFrame(rows)
    
    if not pnl.empty:
        # --- PnL Summary and Integrity Check ---
        total_pnl_s = pnl["gross_return"]
        long_pnl_s = pnl["long_gross_return"]
        short_pnl_s = pnl["short_gross_return"]
        cost_total = pnl["cost_return"].sum()
        total_pnl = total_pnl_s.sum()
        long_pnl = long_pnl_s.sum()
        short_pnl = short_pnl_s.sum()
        
        recon_err = float((total_pnl_s - (long_pnl_s + short_pnl_s)).abs().max())
        
        if DEBUG_DIAGNOSTICS:
            print(f"\n[PnL Summary] Path: {path_audit}")
            print(f"  total_pnl            : {total_pnl:>10.6f}")
            print(f"  long_pnl             : {long_pnl:>10.6f}")
            print(f"  short_pnl            : {short_pnl:>10.6f}")
            print(f"  cost_total           : {cost_total:>10.6f}")
            print(f"  reconstruction_error : {recon_err:>10.10f}")

        def _sharpe(s):
            mu = s.mean()
            std = s.std()
            if np.isfinite(std) and std > 1e-12:
                return float(mu / std * np.sqrt(252))
            return 0.0

        sharpe_total = _sharpe(total_pnl_s)
        sharpe_long = _sharpe(long_pnl_s)
        sharpe_short = _sharpe(short_pnl_s)

        if DEBUG_DIAGNOSTICS:
            print(f"[PnL Summary] total_sharpe={sharpe_total:.4f} | long_sharpe={sharpe_long:.4f} | short_sharpe={sharpe_short:.4f} | reconstruction_error={recon_err:.2e}")

        if recon_err > 1e-8:
            raise MetricIntegrityError(f"PnL reconstruction failed with error {recon_err:.2e}")
        
        # --- Exposure Summary ---
        m_gross = float(pnl["gross_exposure"].mean())
        m_net = float(pnl["net_exposure"].mean())
        max_abs_net = float(pnl["net_exposure"].abs().max())
        
        if DEBUG_DIAGNOSTICS:
            print(f"\n[Exposure Summary] Path: {path_audit}")
            print(f"  gross            : {m_gross:>10.6f}")
            print(f"  net              : {m_net:>10.6f}")
            print(f"  max|net|         : {max_abs_net:>10.6f}")
        
        if path_lower == "long_short_spread" and abs(m_net) > 0.01:
            logger.warning("Non-neutral portfolio detected: net=%.4f", m_net)
        # ------------------------

        avg_turnover = float(pnl["turnover"].mean())
        avg_pov = float(pnl["participation_rate_mean"].mean())
        if avg_turnover > 0.2 and avg_pov < 1e-6:
            raise AssertionError(
                f"Turnover/POV integrity failure: Avg Turnover={avg_turnover:.4f} but Avg POV={avg_pov:.6f}. "
                "This suggests turnover is occurring without execution cost or participation tracking."
            )

    if pnl.empty:
        return pd.Series(dtype=float, name="daily_return"), pnl
    ser = pd.Series(
        pnl["daily_return"].to_numpy(dtype=float), index=pd.to_datetime(pnl["date"]), name="daily_return"
    )
    _write_simulation_cache(cache_path, ser, pnl)
    return ser, pnl


def _zero_execution_pnl(df: pd.DataFrame) -> pd.DataFrame:
    """Return an explicit zero-book execution ledger for infeasible/no-trade books."""
    dates = sorted(pd.to_datetime(df["date"], errors="coerce").dropna().unique())
    rows: list[dict[str, Any]] = []
    for dt in dates:
        rows.append(
            {
                "date": pd.Timestamp(dt),
                "daily_return": 0.0,
                "gross_return": 0.0,
                "market_adj_return": 0.0,
                "long_gross_return": 0.0,
                "short_gross_return": 0.0,
                "long_cost_return": 0.0,
                "short_cost_return": 0.0,
                "cost_return": 0.0,
                "commission_return": 0.0,
                "spread_return": 0.0,
                "fixed_cost_return": 0.0,
                "temporary_impact_return": 0.0,
                "permanent_impact_return": 0.0,
                "permanent_impact_decay_return": 0.0,
                "permanent_impact_unamortized_return": 0.0,
                "borrow_return": 0.0,
                "gross_exposure": 0.0,
                "long_exposure": 0.0,
                "short_exposure": 0.0,
                "net_exposure": 0.0,
                "turnover": 0.0,
                "trade_count": 0,
                "trade_notional": 0.0,
                "participation_rate_mean": 0.0,
                "participation_rate_p95": 0.0,
                "participation_rate_max": 0.0,
                "participation_over_5pct_count": 0,
                "participation_over_10pct_count": 0,
                "participation_capped_count": 0,
                "beta_exposure": 0.0,
                "max_sector_exposure": 0.0,
                "n_positions": 0,
                "n_long_positions": 0,
                "n_short_positions": 0,
            }
        )
    return pd.DataFrame(rows)


def build_target_weights(
    scored: pd.DataFrame,
    cfg: EvaluationConfig,
    *,
    state_cache: ValidationStateCache | None = None,
) -> pd.DataFrame:
    """Build target weights through the centralized PortfolioConstructor."""
    if scored is None or scored.empty:
        return pd.DataFrame(columns=["date", "ticker", "target_weight"])

    df = _canonicalize_scored_panel(scored)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["score"] = pd.to_numeric(df["score"], errors="coerce")
    df = df.dropna(subset=["date", "ticker", "score"]).sort_values(["date", "ticker"])
    if df.empty:
        return pd.DataFrame(columns=["date", "ticker", "target_weight"])

    parts: list[pd.DataFrame] = []
    dates = sorted(df["date"].dropna().unique())
    rebalance_every = max(1, int(cfg.rebalance_every_days))
    prev_target = pd.Series(dtype=float)
    PortfolioConstructor, _, _ = _load_portfolio_construction()
    constructor = PortfolioConstructor()
    control_cfg = AdaptivePortfolioControlConfig(
        enabled=bool(cfg.adaptive_control_enabled),
        lookback_days=int(cfg.adaptive_control_lookback_days),
        min_history_days=int(cfg.adaptive_control_min_history_days),
        ema_span=int(cfg.adaptive_control_ema_span),
        target_volatility=float(cfg.adaptive_control_target_volatility),
        horizon_days=int(cfg.horizon_days),
        lambda_floor_factor=float(cfg.adaptive_lambda_floor_factor),
        lambda_ceil_factor=float(cfg.adaptive_lambda_ceil_factor),
        gamma_floor_factor=float(cfg.adaptive_gamma_floor_factor),
        gamma_ceil_factor=float(cfg.adaptive_gamma_ceil_factor),
        min_expected_alpha=float(cfg.adaptive_min_expected_alpha),
    )
    controller = AdaptivePortfolioController(
        control_cfg,
        base_lambda=float(cfg.lambda_risk),
        base_gamma=float(cfg.gamma_turnover),
        cost_config=cfg.costs,
    )

    for idx, dt in enumerate(dates):
        day = df.loc[df["date"] == dt].copy()
        if idx % rebalance_every != 0:
            continue
        market_state = state_cache.get(pd.Timestamp(dt)) if state_cache is not None else None
        construction_day = day
        if market_state is not None:
            tickers = [str(t) for t in market_state.tickers]
            aligned = day.set_index("ticker", drop=False).reindex(tickers)
            aligned["date"] = pd.Timestamp(dt)
            aligned["ticker"] = tickers
            construction_day = aligned.reset_index(drop=True)
        else:
            tickers = construction_day["ticker"].astype(str).tolist()
            construction_day["date"] = pd.Timestamp(dt)
        if not tickers:
            continue

        inputs = _portfolio_inputs_for_day(
            construction_day,
            cfg,
            pd.Timestamp(dt),
            prev_target,
            full_df=df,
            market_state=market_state,
        )
        control = controller.snapshot_for_date(
            date=pd.Timestamp(dt),
            full_df=df,
            day_df=construction_day,
            market_state=market_state,
        )
        day_cfg = replace(
            cfg,
            lambda_risk=float(control.lambda_risk),
            gamma_turnover=float(control.gamma_turnover),
        )
        constraints = _portfolio_constraints_from_eval_config(day_cfg)
        result = constructor.build_weights(inputs, constraints)
        if result.violations:
            raise MetricIntegrityError(
                f"Portfolio construction constraint violation on {pd.Timestamp(dt).date()}: "
                + ",".join(result.violations)
            )
        weights_by_ticker = result.weights.copy()
        diagnostics = dict(result.diagnostics)
        if str(cfg.path or "").lower() == "long_only_overlay":
            derisk_scale = 1.0
            raw_beta = float("nan")
            raw_sector = float("nan")
            if bool(cfg.beta_neutral) and "capm_beta" in construction_day.columns:
                beta = pd.to_numeric(
                    construction_day.set_index("ticker")["capm_beta"].reindex(weights_by_ticker.index),
                    errors="coerce",
                ).fillna(1.0)
                raw_beta = float((weights_by_ticker.reindex(beta.index).fillna(0.0) * beta).sum())
                if float(cfg.max_beta_abs) > 0.0 and abs(raw_beta) > float(cfg.max_beta_abs) + 1e-8:
                    derisk_scale = min(derisk_scale, float(cfg.max_beta_abs) / max(abs(raw_beta), 1e-12))
            if bool(cfg.sector_neutral) and "sector" in construction_day.columns:
                sectors = construction_day.set_index("ticker")["sector"].reindex(weights_by_ticker.index)
                sector_exposure = weights_by_ticker.groupby(sectors.fillna("Unknown").astype(str)).sum()
                raw_sector = float(sector_exposure.abs().max()) if len(sector_exposure) else float("nan")
                if float(cfg.max_sector_abs) > 0.0 and np.isfinite(raw_sector) and raw_sector > float(cfg.max_sector_abs) + 1e-8:
                    derisk_scale = min(derisk_scale, float(cfg.max_sector_abs) / max(raw_sector, 1e-12))
            if derisk_scale < 1.0:
                weights_by_ticker = (weights_by_ticker * max(0.0, derisk_scale))
                weights_by_ticker = weights_by_ticker[weights_by_ticker.abs() > 1e-12]
                diagnostics["construction_status"] = (
                    "beta_sector_derisked" if str(diagnostics.get("construction_status", "")) == "ok"
                    else f"{diagnostics.get('construction_status', 'unknown')}|beta_sector_derisked"
                )
                diagnostics["raw_beta_before_derisk"] = raw_beta
                diagnostics["raw_sector_before_derisk"] = raw_sector
                diagnostics["derisk_scale"] = float(derisk_scale)
                if np.isfinite(raw_beta):
                    diagnostics["raw_beta_after_derisk"] = float(raw_beta * derisk_scale)
                if np.isfinite(raw_sector):
                    diagnostics["raw_sector_after_derisk"] = float(raw_sector * derisk_scale)
        weights = construction_day["ticker"].astype(str).map(weights_by_ticker).fillna(0.0)
        weights.index = construction_day.index
        prev_target = weights_by_ticker[weights_by_ticker.abs() > 1e-12]

        out = construction_day[["date", "ticker"]].copy()
        out["target_weight"] = weights.reindex(construction_day.index).fillna(0.0).to_numpy(dtype=float)
        out["_constructed_by"] = "PortfolioConstructor"
        out["_lambda_risk"] = float(control.lambda_risk)
        out["_gamma_turnover"] = float(control.gamma_turnover)
        out["_expected_alpha"] = float(control.expected_alpha)
        out["_expected_cost"] = float(control.expected_cost)
        out["_realized_volatility"] = float(control.realized_volatility)
        out["_signal_ic"] = float(control.signal_ic)
        out["_control_support_days"] = int(control.support_days)
        out["_control_status"] = str(control.status)
        for key in (
            "gross_exposure",
            "net_exposure",
            "beta_exposure",
            "max_sector_exposure",
            "construction_status",
            "raw_beta_before_derisk",
            "raw_sector_before_derisk",
            "raw_beta_after_derisk",
            "raw_sector_after_derisk",
            "derisk_scale",
        ):
            if key in diagnostics:
                out[f"_constructed_{key}"] = diagnostics[key]
        parts.append(out)

    if not parts:
        return pd.DataFrame(columns=["date", "ticker", "target_weight"])
    built = pd.concat(parts, ignore_index=True)
    _assert_unique_target_weights(built)
    built.attrs["adaptive_control_timeseries"] = controller.snapshots_frame()
    return built


def _portfolio_constraints_from_eval_config(cfg: EvaluationConfig) -> PortfolioConstraints:
    _, PortfolioConstraints, _ = _load_portfolio_construction()
    return PortfolioConstraints(
        path=str(cfg.path),
        max_positions=int(cfg.max_positions),
        min_positions=int(cfg.min_positions),
        max_gross=float(cfg.max_gross),
        max_net=float(cfg.net_exposure_max),
        max_name_weight=float(cfg.max_name_weight),
        min_position_weight=float(cfg.min_position_weight),
        use_optimizer=bool(cfg.use_optimizer),
        optimization_type=str(cfg.optimization_type),
        lambda_risk=float(cfg.lambda_risk),
        gamma_turnover=float(cfg.gamma_turnover),
        lambda_turn_override=cfg.lambda_turn_override,
        no_trade_band_weight_diff=float(cfg.no_trade_band_weight_diff),
        no_trade_band_total_drift=float(cfg.no_trade_band_total_drift),
        factor_neutral=bool(cfg.factor_neutral),
        beta_neutral=bool(cfg.beta_neutral),
        sector_neutral=bool(cfg.sector_neutral),
        max_beta_abs=float(cfg.max_beta_abs),
        max_sector_abs=float(cfg.max_sector_abs),
        constraint_passes=int(cfg.constraint_passes),
        adv_fraction=float(cfg.adv_fraction),
        capital=float(cfg.costs.capital),
        max_participation_rate=float(cfg.costs.max_participation_rate),
        short_squeeze_filter=bool(cfg.short_squeeze_filter),
        short_squeeze_max_risk=float(cfg.short_squeeze_max_risk),
        market_neutral_shorts=bool(cfg.market_neutral_shorts),
        optimizer_alpha_scale=float(cfg.optimizer_alpha_scale),
        style_exposure_limits=dict(cfg.style_exposure_limits or {}),
        signal_halflife_days=float(getattr(cfg, "signal_halflife_days", float("nan"))),
        horizon_days=int(cfg.horizon_days),
    )


def _portfolio_inputs_for_day(
    day: pd.DataFrame,
    cfg: EvaluationConfig,
    dt: pd.Timestamp,
    prev_target: pd.Series,
    *,
    full_df: pd.DataFrame,
    market_state: DateLevelMarketState | None,
) -> PortfolioInputs:
    _, _, PortfolioInputs = _load_portfolio_construction()
    tickers = day["ticker"].astype(str).tolist()
    by_ticker = day.set_index("ticker", drop=False)
    scores = pd.Series(pd.to_numeric(day["score"], errors="coerce").to_numpy(dtype=float), index=tickers, dtype=float)
    beta = (
        pd.Series(pd.to_numeric(day["capm_beta"], errors="coerce").to_numpy(dtype=float), index=tickers, dtype=float)
        if "capm_beta" in day.columns
        else (
            pd.Series(market_state.factor_exposures["market_beta"], index=tickers, dtype=float)
            if market_state is not None and "market_beta" in market_state.factor_exposures
            else None
        )
    )
    sectors = (
        pd.Series(day["sector"].fillna("Unknown").astype(str).to_numpy(), index=tickers, dtype=object)
        if "sector" in day.columns
        else None
    )
    if market_state is not None:
        covariance = market_state.covariance
        adv = pd.Series(market_state.adv_dollar, index=tickers, dtype=float)
        liquidity_caps = pd.Series(market_state.liquidity_caps, index=tickers, dtype=float)
        daily_vol = pd.Series(market_state.daily_vol, index=tickers, dtype=float)
        borrow = pd.Series(market_state.borrow_penalty_horizon, index=tickers, dtype=float)
        short_blocked = pd.Series(market_state.short_blocked, index=tickers, dtype=bool)
        squeeze = pd.Series(market_state.squeeze_risk, index=tickers, dtype=float)
    else:
        covariance = _covariance_for_day(full_df, day, dt, cfg) if bool(cfg.use_optimizer) else None
        adv = pd.Series(
            _numeric_day_column(day, "adv_dollar_20", cfg.costs.default_adv_usd).to_numpy(dtype=float),
            index=tickers,
            dtype=float,
        )
        liquidity_caps = None
        daily_vol = pd.Series(
            _numeric_day_column(day, "realised_vol_20d", cfg.costs.default_daily_vol).to_numpy(dtype=float),
            index=tickers,
            dtype=float,
        )
        borrow_horizon = float(cfg.costs.borrow_bps) / 10_000.0 * max(1, int(cfg.horizon_days)) / 252.0
        crowding = _numeric_day_column(day, "borrow_crowding_risk", 0.0).clip(0.0, 1.0)
        short_interest = _numeric_day_column(day, "short_interest_ratio", 0.0).clip(0.0, 1.0)
        borrow = pd.Series((borrow_horizon * (1.0 + crowding + short_interest)).to_numpy(dtype=float), index=tickers, dtype=float)
        squeeze = pd.Series(_numeric_day_column(day, "short_squeeze_risk", 0.0).to_numpy(dtype=float), index=tickers, dtype=float)
        hard = pd.Series(_numeric_day_column(day, "hard_short_squeeze_filter", 0.0).to_numpy(dtype=float), index=tickers, dtype=float)
        short_blocked = ((squeeze >= float(cfg.short_squeeze_max_risk)) | (hard >= 1.0)).astype(bool)
    hard = (
        pd.Series(_numeric_day_column(day, "hard_short_squeeze_filter", 0.0).to_numpy(dtype=float), index=tickers, dtype=float)
        if "hard_short_squeeze_filter" in by_ticker.columns
        else pd.Series(0.0, index=tickers, dtype=float)
    )
    return PortfolioInputs(
        date=dt,
        tickers=tickers,
        scores=scores,
        previous_weights=prev_target,
        covariance=covariance,
        beta=beta,
        sectors=sectors,
        factor_exposures=(
            {
                name: pd.Series(values, index=tickers, dtype=float)
                for name, values in (market_state.factor_exposures if market_state is not None else {}).items()
            }
        ),
        adv_dollar=adv,
        liquidity_caps=liquidity_caps,
        daily_vol=daily_vol,
        borrow_penalty=borrow,
        short_blocked=short_blocked,
        squeeze_risk=squeeze,
        hard_short_squeeze=hard,
    )


def _rank_weights_for_day(day: pd.DataFrame, cfg: EvaluationConfig) -> pd.Series:
    day = day.sort_values("score", ascending=False)
    k = max(1, int(cfg.max_positions))
    min_k = max(1, int(cfg.min_positions))
    path = str(cfg.path or "").lower()
    weights = pd.Series(0.0, index=day.index, dtype=float)

    if path == "long_short_spread":
        side_k = max(1, k // 2)
        min_side = max(1, min(min_k, side_k))
        if len(day) < max(2 * min_side, 2):
            return weights
        longs = day.head(side_k)
        shorts = day.tail(side_k)
        if len(longs) < min_side or len(shorts) < min_side:
            return weights
        weights.loc[longs.index] = 0.5 / len(longs)
        weights.loc[shorts.index] = -0.5 / len(shorts)
    elif path == "short_side":
        shorts = day.head(k)
        if len(shorts) < min_k:
            return weights
        weights.loc[shorts.index] = -1.0 / len(shorts)
        assert (weights <= 0).all(), "short_side: no positive weights allowed"
        assert (weights.loc[shorts.index] < 0).all(), "short_side: top-k must have negative weights"
    elif path == "long_only_overlay":
        longs = day.head(k)
        if len(longs) < min_k:
            return weights
        weights.loc[longs.index] = 1.0 / len(longs)
    else:
        raise ValueError(f"unsupported evaluation path: {cfg.path}")

    if cfg.max_name_weight > 0:
        weights = weights.clip(lower=-cfg.max_name_weight, upper=cfg.max_name_weight)
    weights = _apply_short_squeeze_constraints(day, weights, cfg)
    return _normalise_gross(weights, float(cfg.max_gross))


# ── Ledoit-Wolf covariance cache ─────────────────────────────────────────────
# Cache covariance matrices by (universe_hash, lookback, lookback_date) to avoid
# O(N³) recomputation when the universe composition hasn't changed between
# consecutive rebalance dates. This is the standard institutional approach:
# covariance is slow to evolve and recomputing it for identical universes is
# pure waste.
_covariance_cache: dict[tuple, tuple[np.ndarray, pd.Timestamp]] = {}
_COVARIANCE_CACHE_MAX_ENTRIES = 256


def _covariance_for_day(
    full_df: pd.DataFrame, day: pd.DataFrame, dt: pd.Timestamp, cfg: EvaluationConfig
) -> np.ndarray:
    tickers = day["ticker"].astype(str).tolist()
    n = len(tickers)
    if n == 0:
        return np.zeros((0, 0), dtype=float)

    default_var = float(cfg.costs.default_daily_vol) ** 2 * 252.0
    if "daily_return" not in full_df.columns:
        vols = _numeric_day_column(day, "realised_vol_20d", cfg.costs.default_daily_vol)
        diag = (vols.fillna(cfg.costs.default_daily_vol).clip(lower=1e-4).to_numpy(dtype=float) ** 2) * 252.0
        return np.diag(np.maximum(diag, default_var * 0.05))

    hist_dates = sorted(
        pd.to_datetime(full_df.loc[full_df["date"] < dt, "date"], errors="coerce").dropna().unique()
    )
    lookback_days = max(10, int(cfg.optimizer_lookback_days))
    lookback_dates = set(hist_dates[-lookback_days:])
    hist = full_df.loc[full_df["date"].isin(lookback_dates), ["date", "ticker", "daily_return"]].copy()
    if hist.empty or hist["date"].nunique() < 10:
        return np.eye(n, dtype=float) * default_var

    pivot = hist.pivot_table(index="date", columns="ticker", values="daily_return", aggfunc="last")
    pivot = pivot.reindex(columns=tickers)
    if pivot.shape[0] < 10:
        return np.eye(n, dtype=float) * default_var

    universe_hash = _stable_cache_token(sorted(str(t) for t in tickers))
    cache_key = (universe_hash, lookback_days, hist_dates[-1] if hist_dates else pd.Timestamp.min)
    if cache_key in _covariance_cache:
        cached_cov, cached_dt = _covariance_cache[cache_key]
        if cached_dt == hist_dates[-1]:
            return cached_cov.copy()

    arr = pivot.astype(float).replace([np.inf, -np.inf], np.nan)
    arr = arr.sub(arr.mean(axis=0, skipna=True), axis=1).fillna(0.0)
    # Use the same Ledoit-Wolf estimator as the production RiskModel.
    # The previous 80/20 convex shrinkage (0.80*sample + 0.20*diag) is ad-hoc
    # and produces a different covariance structure than the production path,
    # causing validated models to behave materially differently in production.
    # Ledoit-Wolf (OAS) is the institutional standard and ensures estimator
    # consistency between validation and production.
    try:
        from sklearn.covariance import ledoit_wolf
        cov, _ = ledoit_wolf(arr.to_numpy(dtype=float), assume_centered=False)
        cov = cov * 252.0  # annualize
    except Exception:
        cov = np.cov(arr.to_numpy(dtype=float), rowvar=False) * 252.0

    cov = np.atleast_2d(cov).astype(float)
    if cov.shape != (n, n) or not np.isfinite(cov).all():
        return np.eye(n, dtype=float) * default_var

    # ── PD enforcement + condition number bound (matches RiskModel) ──────
    cov = (cov + cov.T) / 2.0
    min_eig = float(np.linalg.eigvalsh(cov)[0]) if n > 1 else float(cov[0, 0])
    if min_eig < 1e-8:
        cov = cov + np.eye(n) * (1e-8 - min_eig + 1e-8)

    MAX_COND = 1000.0
    eigvals = np.linalg.eigvalsh(cov)
    lam_max = float(eigvals[-1])
    lam_min = float(eigvals[0])
    if lam_min > 0 and (lam_max / lam_min) > MAX_COND:
        ridge = (lam_max - MAX_COND * lam_min) / (MAX_COND - 1.0)
        cov += np.eye(n) * ridge

    if len(_covariance_cache) < _COVARIANCE_CACHE_MAX_ENTRIES:
        _covariance_cache[cache_key] = (cov.copy(), hist_dates[-1])

    return cov


def _factor_exposures_for_day(day: pd.DataFrame) -> dict[str, np.ndarray]:
    exposures: dict[str, np.ndarray] = {}
    if "capm_beta" in day.columns:
        beta = pd.to_numeric(day["capm_beta"], errors="coerce").fillna(1.0).clip(-3.0, 3.0)
        exposures["beta"] = (beta - beta.mean()).to_numpy(dtype=float)
    if "sector" in day.columns:
        sectors = day["sector"].fillna("Unknown").astype(str)
        dummies = pd.get_dummies(sectors, dtype=float)
        for col in dummies.columns:
            vec = dummies[col] - dummies[col].mean()
            exposures[f"sector:{col}"] = vec.to_numpy(dtype=float)
    return exposures


def _trade_cost_breakdown(
    trade_weight: float, row: pd.Series, cfg: EvaluationConfig, *, is_exit: bool
) -> TradeCostBreakdown:
    trade_abs = abs(float(trade_weight))
    if trade_abs <= 0:
        return TradeCostBreakdown(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
        )
    costs = cfg.costs
    adv = max(_safe_float(row.get("adv_dollar_20", np.nan), costs.default_adv_usd), 1.0)
    vol = abs(_safe_float(row.get("realised_vol_20d", np.nan), costs.default_daily_vol))
    vol = vol if vol > 1e-6 else costs.default_daily_vol
    order_usd = trade_abs * float(costs.capital)
    raw_participation = order_usd / adv
    max_participation = max(float(costs.max_participation_rate), 1e-6)
    participation = min(raw_participation, max_participation)
    commission = float(costs.commission_bps) / 10_000.0
    spread = float(costs.spread_bps) / 2.0 / 10_000.0
    temporary = float(costs.impact_eta) * vol * (participation ** float(costs.impact_gamma))
    permanent = (
        0.0 if is_exit else float(costs.impact_alpha) * vol * (participation ** float(costs.impact_gamma))
    )
    commission_return = trade_abs * commission
    spread_return = trade_abs * spread
    temporary_return = trade_abs * temporary
    permanent_return = trade_abs * permanent
    fixed_return = commission_return + spread_return
    return TradeCostBreakdown(
        cost_return=fixed_return + temporary_return + permanent_return,
        commission_return=commission_return,
        spread_return=spread_return,
        fixed_cost_return=fixed_return,
        temporary_impact_return=temporary_return,
        permanent_impact_return=permanent_return,
        trade_weight_abs=trade_abs,
        trade_notional=order_usd,
        adv_dollar=adv,
        daily_vol=vol,
        participation_rate=participation,
        participation_capped=raw_participation > max_participation,
    )


def _trade_cost_breakdown_from_market_state(
    trade_weight: float,
    ticker: str,
    market_state: DateLevelMarketState,
    cfg: EvaluationConfig,
    *,
    is_exit: bool,
) -> TradeCostBreakdown:
    trade_abs = abs(float(trade_weight))
    if trade_abs <= 0:
        return TradeCostBreakdown(
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, False
        )
    idx = market_state.ticker_to_idx.get(str(ticker))
    if idx is None:
        return _trade_cost_breakdown(trade_weight, pd.Series(dtype=float), cfg, is_exit=is_exit)
    costs = cfg.costs
    adv = max(float(market_state.adv_dollar[idx]), 1.0)
    vol = max(abs(float(market_state.daily_vol[idx])), 1e-6)
    order_usd = trade_abs * float(costs.capital)
    raw_participation = trade_abs * float(market_state.participation_scale[idx])
    max_participation = max(float(market_state.max_participation_rate), 1e-6)
    participation = min(raw_participation, max_participation)
    commission = float(costs.commission_bps) / 10_000.0
    spread = float(costs.spread_bps) / 2.0 / 10_000.0
    temporary = float(costs.impact_eta) * vol * (participation ** float(costs.impact_gamma))
    permanent = (
        0.0 if is_exit else float(costs.impact_alpha) * vol * (participation ** float(costs.impact_gamma))
    )
    commission_return = trade_abs * commission
    spread_return = trade_abs * spread
    temporary_return = trade_abs * temporary
    permanent_return = trade_abs * permanent
    fixed_return = commission_return + spread_return
    return TradeCostBreakdown(
        cost_return=fixed_return + temporary_return + permanent_return,
        commission_return=commission_return,
        spread_return=spread_return,
        fixed_cost_return=fixed_return,
        temporary_impact_return=temporary_return,
        permanent_impact_return=permanent_return,
        trade_weight_abs=trade_abs,
        trade_notional=order_usd,
        adv_dollar=adv,
        daily_vol=vol,
        participation_rate=participation,
        participation_capped=raw_participation > max_participation,
    )


def _vectorized_trade_costs_from_market_state(
    trade_tickers: np.ndarray,
    trade_weights: np.ndarray,
    previous_weights: np.ndarray,
    next_weights: np.ndarray,
    market_state: DateLevelMarketState,
    cfg: EvaluationConfig,
) -> dict[str, Any] | None:
    """Vectorized execution-cost accounting for one rebalance date.

    This preserves the same cost equations as ``_trade_cost_breakdown_from_market_state``
    but avoids one Python function/dataclass allocation per traded name.
    """

    if len(trade_tickers) == 0:
        return None
    idx = np.array([market_state.ticker_to_idx.get(str(t), -1) for t in trade_tickers], dtype=int)
    if np.any(idx < 0):
        return None

    costs = cfg.costs
    trade_abs = np.abs(np.asarray(trade_weights, dtype=float))
    active = trade_abs > 1e-12
    if not np.any(active):
        return None
    idx = idx[active]
    tickers = np.asarray(trade_tickers, dtype=object)[active]
    dw = np.asarray(trade_weights, dtype=float)[active]
    prev = np.asarray(previous_weights, dtype=float)[active]
    nxt = np.asarray(next_weights, dtype=float)[active]
    trade_abs = trade_abs[active]

    vol = np.maximum(np.abs(market_state.daily_vol[idx].astype(float)), 1e-6)
    raw_participation = trade_abs * market_state.participation_scale[idx].astype(float)
    max_participation = max(float(market_state.max_participation_rate), 1e-6)
    participation = np.minimum(raw_participation, max_participation)
    impact_power = participation ** float(costs.impact_gamma)

    commission_return = trade_abs * (float(costs.commission_bps) / 10_000.0)
    spread_return = trade_abs * (float(costs.spread_bps) / 2.0 / 10_000.0)
    temporary_return = trade_abs * (float(costs.impact_eta) * vol * impact_power)
    is_exit = np.abs(nxt) < np.abs(prev)
    permanent_return = trade_abs * np.where(
        is_exit,
        0.0,
        float(costs.impact_alpha) * vol * impact_power,
    )
    fixed_return = commission_return + spread_return
    cost_return = fixed_return + temporary_return + permanent_return
    long_mask = nxt >= 0.0

    permanent_mask = permanent_return > 0.0
    permanent_updates = pd.Series(
        np.sign(dw[permanent_mask]) * permanent_return[permanent_mask],
        index=pd.Index(tickers[permanent_mask].astype(str), dtype=str),
        dtype=float,
    )

    return {
        "cost": float(np.sum(cost_return)),
        "long_cost": float(np.sum(cost_return[long_mask])),
        "short_cost": float(np.sum(cost_return[~long_mask])),
        "commission": float(np.sum(commission_return)),
        "spread": float(np.sum(spread_return)),
        "fixed_cost": float(np.sum(fixed_return)),
        "temporary_impact": float(np.sum(temporary_return)),
        "permanent_impact": float(np.sum(permanent_return)),
        "trade_notional": float(np.sum(trade_abs) * float(costs.capital)),
        "trade_count": int(len(trade_abs)),
        "capped_trade_count": int(np.sum(raw_participation > max_participation)),
        "participation_rates": participation.astype(float),
        "permanent_updates": permanent_updates,
    }


def _apply_execution_constraints(day: pd.DataFrame, weights: pd.Series, cfg: EvaluationConfig) -> pd.Series:
    out = weights.copy()
    passes = max(1, int(cfg.constraint_passes))
    for _ in range(passes):
        prev = out.copy()
        out = _neutralize_day_weights(day, out, cfg)
        out = _apply_liquidity_caps(day, out, cfg)
        if float((out.reindex(prev.index).fillna(0.0) - prev.fillna(0.0)).abs().sum()) < 1e-10:
            break
    return _normalise_gross(out, min(float(cfg.max_gross), float(out.abs().sum())))


def _neutralize_day_weights(day: pd.DataFrame, weights: pd.Series, cfg: EvaluationConfig) -> pd.Series:
    if not bool(cfg.factor_neutral):
        return weights
    weight_probe = pd.to_numeric(weights, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
    max_abs_weight = float(weight_probe.abs().max()) if len(weight_probe) else 0.0
    if max_abs_weight <= 1e-12:
        return weights
    if max_abs_weight > 10.0:
        return _normalise_gross(weight_probe.clip(-1.0, 1.0), float(cfg.max_gross))

    factors: list[pd.Series] = []
    if bool(cfg.beta_neutral) and "capm_beta" in day.columns:
        beta = pd.to_numeric(day["capm_beta"], errors="coerce").fillna(1.0).clip(-3.0, 3.0)
        factors.append(beta - beta.mean())
    if bool(cfg.sector_neutral) and "sector" in day.columns:
        sectors = day["sector"].fillna("Unknown").astype(str)
        dummies = pd.get_dummies(sectors, dtype=float)
        for col in dummies.columns:
            factors.append(dummies[col] - dummies[col].mean())

    if not factors:
        return weights

    x = pd.concat(factors, axis=1).reindex(day.index).fillna(0.0).to_numpy(dtype=float)
    w = weights.reindex(day.index).fillna(0.0).to_numpy(dtype=float)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    w = np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0)
    if x.size == 0 or w.size == 0 or not np.isfinite(x).all() or not np.isfinite(w).all():
        return weights

    col_std = x.std(axis=0)
    keep = np.isfinite(col_std) & (col_std > 1e-10)
    x = x[:, keep]
    if x.size == 0 or x.shape[0] <= 2:
        return weights

    x = x - x.mean(axis=0, keepdims=True)
    col_norm = np.sqrt(np.sum(x * x, axis=0))
    keep = np.isfinite(col_norm) & (col_norm > 1e-10)
    x = x[:, keep]
    col_norm = col_norm[keep]
    if x.size == 0:
        return weights
    x = x / col_norm

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                xtx = x.T @ x
                xtw = x.T @ w
                ridge = max(1e-6, float(np.trace(xtx)) / max(1, xtx.shape[0]) * 1e-6)
                beta_hat = np.linalg.solve(xtx + ridge * np.eye(xtx.shape[0]), xtw)
                neutral = w - x @ beta_hat
    except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
        return weights

    out = pd.Series(neutral, index=day.index, dtype=float)
    if cfg.max_name_weight > 0:
        out = out.clip(lower=-cfg.max_name_weight, upper=cfg.max_name_weight)
    return _normalise_gross(out, float(cfg.max_gross))


def _apply_liquidity_caps(day: pd.DataFrame, weights: pd.Series, cfg: EvaluationConfig) -> pd.Series:
    if "adv_dollar_20" not in day.columns or cfg.adv_fraction <= 0:
        return weights
    adv = pd.to_numeric(day["adv_dollar_20"], errors="coerce").fillna(cfg.costs.default_adv_usd)
    cap = (float(cfg.adv_fraction) * adv / max(float(cfg.costs.capital), 1.0)).clip(lower=0.0)
    capped = weights.clip(lower=-cap, upper=cap)
    return _normalise_gross(capped, min(float(cfg.max_gross), float(capped.abs().sum())))


def _apply_short_squeeze_constraints(
    day: pd.DataFrame, weights: pd.Series, cfg: EvaluationConfig
) -> pd.Series:
    if not bool(cfg.short_squeeze_filter) or str(cfg.path or "").lower() not in {
        "short_side",
        "long_short_spread",
    }:
        return weights
    if weights.empty:
        return weights
    squeeze = _numeric_day_column(day, "short_squeeze_risk", 0.0)
    hard = _numeric_day_column(day, "hard_short_squeeze_filter", 0.0)
    blocked_idx = day.index[(squeeze >= float(cfg.short_squeeze_max_risk)) | (hard >= 1.0)]
    out = weights.copy()
    short_blocked = out.index.intersection(blocked_idx)
    out.loc[short_blocked] = out.loc[short_blocked].clip(lower=0.0)
    return out


def _next_day_returns(scored: pd.DataFrame, *, horizon_days: int) -> pd.Series:
    if "daily_return" in scored.columns:
        dates = pd.to_datetime(scored["date"], errors="coerce").to_numpy(dtype="datetime64[ns]")
        tickers = scored["ticker"].astype(str).to_numpy(dtype=object)
        daily = pd.to_numeric(scored["daily_return"], errors="coerce").to_numpy(dtype=float)
        ticker_codes, _ = pd.factorize(tickers, sort=False)
        order = np.lexsort((dates, ticker_codes))
        sorted_codes = ticker_codes[order]
        sorted_daily = daily[order]
        shifted_sorted = np.full(len(scored), np.nan, dtype=float)
        if len(order) > 1:
            same_ticker_next = sorted_codes[:-1] == sorted_codes[1:]
            shifted_sorted[:-1] = np.where(same_ticker_next, sorted_daily[1:], np.nan)
        out = np.full(len(scored), np.nan, dtype=float)
        out[order] = shifted_sorted
        return pd.Series(out, index=scored.index)
    return pd.to_numeric(scored.get("forward_return", 0.0), errors="coerce") / max(1.0, float(horizon_days))


def _portfolio_beta(day: pd.DataFrame, weights: pd.Series) -> float:
    if "capm_beta" not in day.columns or weights.empty:
        return 0.0
    beta = pd.to_numeric(day.set_index("ticker")["capm_beta"], errors="coerce").fillna(1.0)
    return float((weights.reindex(beta.index).fillna(0.0) * beta).sum())


def _max_sector_exposure(day: pd.DataFrame, weights: pd.Series) -> float:
    if "sector" not in day.columns or weights.empty:
        return 0.0
    tmp = pd.DataFrame({"weight": weights})
    sectors = day.set_index("ticker")["sector"].fillna("Unknown").astype(str)
    tmp["sector"] = sectors.reindex(tmp.index).fillna("Unknown")
    return float(tmp.groupby("sector")["weight"].sum().abs().max()) if not tmp.empty else 0.0


def _portfolio_beta_from_aligned_day(day: pd.DataFrame, aligned_weights: pd.Series) -> float:
    """Fast beta exposure when weights are already aligned to ``day`` ticker order."""
    if "capm_beta" not in day.columns or aligned_weights.empty:
        return 0.0
    beta = pd.to_numeric(day["capm_beta"], errors="coerce").fillna(1.0).to_numpy(dtype=float)
    w = aligned_weights.to_numpy(dtype=float)
    if beta.shape[0] != w.shape[0]:
        return _portfolio_beta(day, aligned_weights)
    return float(np.dot(w, beta))


def _max_sector_exposure_from_aligned_day(day: pd.DataFrame, aligned_weights: pd.Series) -> float:
    """Fast sector exposure via factorized sectors and ``np.bincount``."""
    if "sector" not in day.columns or aligned_weights.empty:
        return 0.0
    sectors = day["sector"].fillna("Unknown").astype(str).to_numpy(dtype=object)
    w = aligned_weights.to_numpy(dtype=float)
    if sectors.shape[0] != w.shape[0]:
        return _max_sector_exposure(day, aligned_weights)
    codes, _ = pd.factorize(sectors, sort=False)
    exposure = np.bincount(codes, weights=w, minlength=int(codes.max()) + 1 if codes.size else 0)
    return float(np.max(np.abs(exposure))) if exposure.size else 0.0


def max_drawdown(returns: np.ndarray | pd.Series) -> float:
    r = pd.to_numeric(pd.Series(returns), errors="coerce").fillna(0.0).to_numpy(dtype=float)
    if len(r) == 0:
        return 0.0
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = equity / np.where(peak == 0.0, 1.0, peak) - 1.0
    return float(np.nanmin(dd))


def annualized_sharpe(returns: np.ndarray | pd.Series, *, hac_lags: int = 5) -> float:
    r = pd.to_numeric(pd.Series(returns), errors="coerce").dropna().to_numpy(dtype=float)
    if len(r) < 10:
        return float("nan")
    mu = float(np.mean(r))
    demeaned = r - mu
    max_lag = min(max(0, int(hac_lags)), len(r) - 1)
    var = float(np.dot(demeaned, demeaned) / len(demeaned))
    for lag in range(1, max_lag + 1):
        cov = float(np.dot(demeaned[lag:], demeaned[:-lag]) / len(demeaned))
        var += 2.0 * (1.0 - lag / (max_lag + 1.0)) * cov
    if var <= 1e-16:
        return 0.0 if abs(mu) <= 1e-12 else float("nan")
    return float(mu / np.sqrt(var) * np.sqrt(252.0))


def _calibrate_l1_turnover_penalty(
    scored: pd.DataFrame,
    cfg: EvaluationConfig,
    *,
    state_cache: ValidationStateCache | None = None,
) -> float:
    """Task 2: Calibrate lambda_turn via in-fold cross-validation of net Sharpe."""
    from dataclasses import replace as _dc_replace
    if scored.empty:
        return 0.0
        
    # Sweep values for lambda_turn (aliased as gamma_turnover in config)
    lambdas = [0.0, 0.01, 0.05, 0.1, 0.5, 1.0]
    best_lambda = 0.1
    best_sharpe = -np.inf
    
    # Use a faster proxy or a smaller subset of dates for calibration
    dates = sorted(scored["date"].unique())
    if len(dates) < 5:
        return 0.1
        
    # Split into 3 sub-folds if possible
    n_folds = 3
    fold_size = len(dates) // n_folds
    
    for l_val in lambdas:
        fold_sharpes = []
        test_cfg = _dc_replace(cfg, gamma_turnover=l_val, optimization_type="l1")
        
        # Fast evaluation across sub-folds
        for f in range(n_folds):
            start_idx = f * fold_size
            end_idx = (f + 1) * fold_size
            test_dates = dates[start_idx:end_idx]
            fold_scored = scored[scored["date"].isin(test_dates)]
            
            if fold_scored.empty:
                continue
                
            daily, _ = simulate_executable_portfolio(fold_scored, test_cfg, state_cache=state_cache)
            sharpe = _sharpe_from_series(daily.to_numpy(dtype=float), horizon=int(cfg.horizon_days))
            if np.isfinite(sharpe):
                fold_sharpes.append(sharpe)
                
        if fold_sharpes:
            mean_sharpe = float(np.mean(fold_sharpes))
            if mean_sharpe > best_sharpe:
                best_sharpe = mean_sharpe
                best_lambda = l_val
                
    return best_lambda


def _sharpe_from_series(pnl: np.ndarray, horizon: int = 1) -> float:
    """
    Annualised Sharpe with Newey-West correction for serial correlation.

    When `horizon > 1`, overlapping forward-return windows create autocorrelation
    that inflates the naive Sharpe (std is underestimated). The Newey-West HAC
    estimator corrects the standard error using lags up to `horizon - 1`.
    Correction factor: sqrt(1 + 2 * sum_k(1 - k/horizon) * rho_k) where rho_k
    is the k-lag autocorrelation of the return series.
    """
    pnl = pnl.astype(float)
    pnl = pnl[np.isfinite(pnl)]
    if len(pnl) < 10:
        return float("nan")
    mu = float(np.mean(pnl))
    sd = float(np.std(pnl, ddof=1))
    if sd < 1e-12:
        return 0.0 if abs(mu) <= 1e-12 else float("nan")

    if horizon > 1 and len(pnl) > horizon * 2:
        # Newey-West variance: V_NW = gamma_0 + 2 * sum_{k=1}^{h-1} (1 - k/h) * gamma_k
        gamma_0 = float(np.var(pnl, ddof=1))
        nw_var = gamma_0
        for k in range(1, int(horizon)):
            if k >= len(pnl):
                break
            gamma_k = float(np.cov(pnl[k:], pnl[:-k])[0, 1])
            nw_var += 2.0 * (1.0 - k / horizon) * gamma_k
        nw_var = max(nw_var, gamma_0 * 0.1)  # floor at 10% of naive variance
        sd = float(np.sqrt(nw_var))
        if sd < 1e-12:
            return 0.0 if abs(mu) <= 1e-12 else float("nan")

    return float((mu / sd) * np.sqrt(252.0))


def cagr(returns: np.ndarray | pd.Series) -> float:
    r = pd.to_numeric(pd.Series(returns), errors="coerce").dropna().to_numpy(dtype=float)
    if len(r) == 0:
        return float("nan")
    total = float(np.prod(1.0 + r))
    years = len(r) / 252.0
    if years <= 0 or total <= 0:
        return float("nan")
    return float(total ** (1.0 / years) - 1.0)


def decile_return_diagnostics(
    scored: pd.DataFrame, *, target_col: str = "target_return", n_bins: int = 10
) -> dict[str, float]:
    if scored is None or scored.empty or target_col not in scored.columns:
        return {
            "decile_top_mean": float("nan"),
            "decile_bottom_mean": float("nan"),
            "decile_spread": float("nan"),
            "decile_monotonicity": float("nan"),
            "decile_n_days": 0,
        }
    df = scored[["date", "score", target_col]].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna()
    tops, bottoms, spreads, monotonicity = [], [], [], []
    for _, g in df.groupby("date", sort=True):
        if len(g) < max(10, n_bins * 3) or g["score"].nunique() < n_bins:
            continue
        try:
            bucket = pd.qcut(g["score"].rank(method="first"), n_bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        by_bucket = g.groupby(bucket, observed=True)[target_col].mean().sort_index()
        if len(by_bucket) < 3:
            continue
        bottom, top = float(by_bucket.iloc[0]), float(by_bucket.iloc[-1])
        diffs = np.diff(by_bucket.to_numpy(dtype=float))
        tops.append(top); bottoms.append(bottom); spreads.append(top - bottom)
        monotonicity.append(float((diffs >= 0.0).mean()) if len(diffs) else float("nan"))
    return {
        "decile_top_mean": float(np.nanmean(tops)) if tops else float("nan"),
        "decile_bottom_mean": float(np.nanmean(bottoms)) if bottoms else float("nan"),
        "decile_spread": float(np.nanmean(spreads)) if spreads else float("nan"),
        "decile_monotonicity": float(np.nanmean(monotonicity)) if monotonicity else float("nan"),
        "decile_n_days": int(len(spreads)),
    }


def simulate_proxy_portfolio(
    scored: pd.DataFrame, cfg: EvaluationConfig
) -> tuple[pd.Series, pd.DataFrame]:
    from dataclasses import replace as _dc_replace

    proxy_cfg = _dc_replace(cfg, use_optimizer=False)
    returns, pnl = simulate_executable_portfolio(scored, proxy_cfg, state_cache=None)
    if not pnl.empty:
        pnl = pnl.copy()
        pnl["_simulation_mode"] = "proxy"
    return returns, pnl


def proxy_executable_metrics(scored: pd.DataFrame, cfg: EvaluationConfig) -> dict[str, float]:
    returns, pnl = simulate_proxy_portfolio(scored, cfg)
    arr = returns.to_numpy(dtype=float)
    costs = float(pnl["cost_return"].sum()) if not pnl.empty and "cost_return" in pnl else 0.0
    gross = float(pnl["gross_return"].sum()) if not pnl.empty and "gross_return" in pnl else 0.0
    return {
        "proxy_exec_sharpe": annualized_sharpe(arr, hac_lags=max(1, cfg.horizon_days)),
        "proxy_exec_cagr": cagr(arr),
        "proxy_exec_max_dd": max_drawdown(arr),
        "proxy_exec_cost_to_gross_pnl": abs(costs) / abs(gross)
        if abs(gross) > 1e-12
        else (0.0 if abs(costs) <= 1e-12 else float("inf")),
        "proxy_exec_n_days": int(len(arr)),
        "proxy_exec_beta_abs_mean": float(pnl["beta_exposure"].abs().mean())
        if not pnl.empty and "beta_exposure" in pnl
        else float("nan"),
        "_simulation_mode": "proxy",
    }


def executable_metrics(
    scored: pd.DataFrame, cfg: EvaluationConfig, *, state_cache: ValidationStateCache | None = None
) -> dict[str, float]:
    returns, pnl = simulate_executable_portfolio(scored, cfg, state_cache=state_cache)
    arr = returns.to_numpy(dtype=float)
    costs = float(pnl["cost_return"].sum()) if not pnl.empty and "cost_return" in pnl else 0.0
    gross = float(pnl["gross_return"].sum()) if not pnl.empty and "gross_return" in pnl else 0.0

    def _col_arr(name: str) -> np.ndarray:
        if pnl.empty or name not in pnl:
            return np.array([], dtype=float)
        return pd.to_numeric(pnl[name], errors="coerce").fillna(0.0).to_numpy(dtype=float)

    long_gross_arr = _col_arr("long_gross_return")
    short_gross_arr = _col_arr("short_gross_return")
    long_cost_arr = _col_arr("long_cost_return")
    short_cost_arr = _col_arr("short_cost_return")
    borrow_arr = _col_arr("borrow_return")
    mkt_adj_arr = _col_arr("market_adj_return")

    long_arr = long_gross_arr - long_cost_arr
    short_arr = short_gross_arr - short_cost_arr - borrow_arr - mkt_adj_arr

    def _sum_col(name: str) -> float:
        return float(pd.to_numeric(pnl.get(name, pd.Series(dtype=float)), errors="coerce").fillna(0.0).sum())

    def _mean_col(name: str) -> float:
        s = pd.to_numeric(pnl.get(name, pd.Series(dtype=float)), errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        return float(s.mean()) if not s.dropna().empty else float("nan")

    def _max_col(name: str) -> float:
        s = pd.to_numeric(pnl.get(name, pd.Series(dtype=float)), errors="coerce").replace(
            [np.inf, -np.inf], np.nan
        )
        return float(s.max()) if not s.dropna().empty else float("nan")

    return {
        "exec_sharpe": annualized_sharpe(arr, hac_lags=max(1, cfg.horizon_days)),
        "exec_long_leg_sharpe": annualized_sharpe(long_arr, hac_lags=max(1, cfg.horizon_days)),
        "exec_short_leg_sharpe": annualized_sharpe(short_arr, hac_lags=max(1, cfg.horizon_days)),
        "exec_cagr": cagr(arr),
        "exec_max_dd": max_drawdown(arr),
        "exec_win_rate": float((arr > 0.0).mean()) if len(arr) else float("nan"),
        "exec_cost_return_sum": costs,
        "exec_long_cost_return_sum": _sum_col("long_cost_return"),
        "exec_short_cost_return_sum": _sum_col("short_cost_return"),
        "exec_borrow_return_sum": _sum_col("borrow_return"),
        "exec_commission_return_sum": _sum_col("commission_return"),
        "exec_spread_return_sum": _sum_col("spread_return"),
        "exec_fixed_cost_return_sum": _sum_col("fixed_cost_return"),
        "exec_temporary_impact_return_sum": _sum_col("temporary_impact_return"),
        "exec_permanent_impact_return_sum": _sum_col("permanent_impact_return"),
        "exec_permanent_impact_decay_return_sum": _sum_col("permanent_impact_decay_return"),
        "exec_permanent_impact_unamortized_mean": _mean_col("permanent_impact_unamortized_return"),
        "exec_turnover_mean": _mean_col("turnover"),
        "exec_trade_count_sum": _sum_col("trade_count"),
        "exec_trade_notional_sum": _sum_col("trade_notional"),
        "exec_participation_mean": _mean_col("participation_rate_mean"),
        "exec_participation_p95": _mean_col("participation_rate_p95"),
        "exec_participation_max": _max_col("participation_rate_max"),
        "exec_participation_over_5pct_count": _sum_col("participation_over_5pct_count"),
        "exec_participation_over_10pct_count": _sum_col("participation_over_10pct_count"),
        "exec_participation_capped_count": _sum_col("participation_capped_count"),
        "exec_gross_exposure_mean": _mean_col("gross_exposure"),
        "exec_net_exposure_abs_mean": float(pnl["net_exposure"].abs().mean())
        if not pnl.empty and "net_exposure" in pnl
        else float("nan"),
        "exec_long_exposure_mean": _mean_col("long_exposure"),
        "exec_short_exposure_mean": _mean_col("short_exposure"),
        "exec_beta_abs_mean": _mean_col("beta_exposure"),
        "exec_max_sector_abs_mean": _mean_col("max_sector_exposure"),
        "exec_cost_to_gross_pnl": abs(costs) / abs(gross)
        if abs(gross) > 1e-12
        else (0.0 if abs(costs) <= 1e-12 else float("inf")),
        "exec_n_days": int(len(arr)),
        **decile_return_diagnostics(scored, target_col="target_return"),
    }


def evaluate_promotion_gates(
    row: pd.Series | dict[str, Any],
    cfg: PromotionGateConfig,
    *,
    model_kind: str = "",
    long_cand_cfg: LongAlphaCandidateConfig | None = None,
) -> dict[str, Any]:
    """Evaluate hard promotion gates."""
    data = dict(row)
    _kind = str(model_kind or data.get("model_kind", "") or "").lower().strip()
    _is_short = _kind in {"short_classifier", "short_alpha"}
    threshold_audit = _dynamic_thresholds(data, cfg, model_kind=_kind, is_short=_is_short)

    def _thr(field_name: str) -> float:
        if field_name in threshold_audit:
            return float(threshold_audit[field_name]["effective"])
        return _resolve_threshold(cfg, _kind, _is_short, field_name)

    # --- Gate Audit Trace ---
    path_audit = data.get("oos_evaluation_path", "unknown")
    _is_long_only_path = path_audit == "long_only_overlay"
    if path_audit == "long_short_spread":
        l_sharpe = data.get("exec_long_leg_sharpe")
        s_sharpe = data.get("exec_short_leg_sharpe")
        l_thr = _thr("min_long_leg_sharpe")
        s_thr = _thr("min_short_leg_sharpe")
        if DEBUG_DIAGNOSTICS:
            print(f"[Gate Audit] Path: {path_audit} | Model: {data.get('model_name', '??')}")
            print(f"    Long Leg Sharpe: {l_sharpe} (Thr: {l_thr})")
            print(f"    Short Leg Sharpe: {s_sharpe} (Thr: {s_thr})")
    # ------------------------

    checks = {
        "min_windows": _safe_float(data.get("n_windows"), 0.0) >= _thr("min_windows"),
        "min_sharpe": _safe_float(data.get("oos_sharpe_chained"), -999.0) >= _thr("min_sharpe"),
        "min_cost_aware_sharpe": _safe_float(data.get("exec_sharpe"), -999.0) >= _thr("min_cost_aware_sharpe"),
        # horizon_adj_ic_tstat: non-overlapping t-stat (N/h effective d.f.) checked first;
        # falls back to window-level oos_ic_tstat only if horizon-adjusted is unavailable.
        "min_ic_tstat": _first_valid_float(
            ["horizon_adj_ic_tstat", "oos_ic_tstat", "cs_ic_spearman_tstat"], data, -999.0
        ) >= _thr("min_ic_tstat"),
        # horizon_adj_ic_ir: non-overlapping Grinold IR = IC_mean/IC_std × sqrt(252/h);
        # checked before oos_ic_ir (window-level, uncorrected) and the overstate annualized IR.
        "min_ic_ir": _first_valid_float(
            ["horizon_adj_ic_ir", "oos_ic_ir", "cs_ic_spearman_annualized_icir", "cs_ic_spearman_ir"], data, -999.0
        ) >= _thr("min_ic_ir"),
        "min_beat_rate": _safe_float(data.get("oos_beat_rate"), -999.0) >= _thr("min_beat_rate"),
        "max_drawdown": _safe_float(data.get("exec_max_dd", data.get("oos_max_dd")), -999.0)
        >= _thr("max_drawdown"),
        "min_psr": _safe_float(data.get("oos_psr"), 0.0) >= _thr("min_psr"),
        "max_beta_abs_mean": True
        if _is_short
        else (_safe_float(data.get("exec_beta_abs_mean"), 999.0) <= _thr("max_beta_abs_mean")),
        "max_sector_abs_mean": _safe_float(data.get("exec_max_sector_abs_mean"), 999.0)
        <= _thr("max_sector_abs_mean"),
        "max_cost_to_gross_pnl": _safe_float(data.get("exec_cost_to_gross_pnl"), 999.0)
        <= _thr("max_cost_to_gross_pnl"),
        "min_decile_spread": _safe_float(data.get("decile_spread"), -999.0) >= _thr("min_decile_spread"),
        "min_tail_monotonicity": _safe_float(data.get("decile_monotonicity"), -999.0)
        >= _thr("min_tail_monotonicity"),
        "min_long_leg_sharpe": True
        if _is_short or _is_long_only_path
        else (_safe_float(data.get("exec_long_leg_sharpe"), -999.0) >= _thr("min_long_leg_sharpe")),
        "min_short_leg_sharpe": True
        if _is_short or _is_long_only_path
        else (_safe_float(data.get("exec_short_leg_sharpe"), -999.0) >= _thr("min_short_leg_sharpe")),
        "min_subsumption_alpha_ann": _gate_optional(
            data.get("subsumption_alpha_ann"), lambda v, t: v >= t, _thr("min_subsumption_alpha_ann")
        ),
        "min_subsumption_alpha_tstat": _gate_optional(
            data.get("subsumption_alpha_tstat"), lambda v, t: v >= t, _thr("min_subsumption_alpha_tstat")
        ),
        "max_subsumption_r2": _gate_optional(
            data.get("subsumption_r2"), lambda v, t: v <= t, _thr("max_subsumption_r2")
        ),
        "max_subsumption_loading_abs": _gate_optional(
            data.get("subsumption_max_abs_loading"), lambda v, t: v <= t, _thr("max_subsumption_loading_abs")
        ),
    }
    
    # ── Execution Robustness Gates ────────────────────────────────────────────
    if cfg.execution_robustness_enabled:
        # 1. Signal Halflife vs Rebalance Frequency
        rebalance_freq = _safe_float(data.get("rebalance_frequency", data.get("rebalance_every_days")), 5.0)
        halflife = _safe_float(data.get("diag_robust_signal_halflife", data.get("signal_halflife_days")), -1.0)
        halflife_req = rebalance_freq + _thr("min_signal_halflife_buffer")
        halflife_ok = halflife >= halflife_req if halflife >= 0 else False
        
        # 2. CAIC / Raw IC Ratio
        caic_ratio = _safe_float(data.get("diag_caic_to_raw_ic_ratio"), -1.0)
        caic_ok = caic_ratio >= _thr("min_caic_to_ic_ratio") if caic_ratio >= 0 else False
        
        # 3. Average Turnover
        turn_mean = _safe_float(data.get("diag_robust_turnover_mean", data.get("avg_turnover")), 999.0)
        turn_ok = turn_mean <= _thr("max_avg_turnover")
        
        # 4. Check for missing metrics if required
        if cfg.execution_robustness_fail_on_missing:
            if not np.isfinite(halflife) or not np.isfinite(caic_ratio) or not np.isfinite(turn_mean):
                # If any are missing, the whole robustness gate fails
                halflife_ok = caic_ok = turn_ok = False

        checks.update({
            "robust_halflife": halflife_ok,
            "robust_caic_ratio": caic_ok,
            "robust_turnover": turn_ok
        })

    # Determine failures
    # Robustness gates are tagged as diagnostic or blocking depending on policy.
    # They are NEVER silently dropped from the final report.
    _robustness_gate_names = {"robust_halflife", "robust_caic_ratio", "robust_turnover"}
    std_gate_names = {
        "min_windows", "min_sharpe", "min_cost_aware_sharpe", "min_ic_tstat", "min_ic_ir",
        "min_beat_rate", "max_drawdown", "min_psr", "max_beta_abs_mean", "max_sector_abs_mean",
        "max_cost_to_gross_pnl", "min_decile_spread", "min_tail_monotonicity",
        "min_long_leg_sharpe", "min_short_leg_sharpe", "min_subsumption_alpha_ann",
        "min_subsumption_alpha_tstat", "max_subsumption_r2", "max_subsumption_loading_abs",
    }

    if cfg.execution_robustness_affect_selection:
        # Robustness failures are selection-blocking
        failures = [name for name, ok in checks.items() if not ok]
        blocking_failures = list(failures)
    else:
        # Standard gates only — but robustness failures are still reported
        # with a diagnostic prefix so they appear in the failure report.
        standard_failures = [name for name, ok in checks.items() if not ok and name in std_gate_names]
        diagnostic_failures = [name for name, ok in checks.items() if not ok and name in _robustness_gate_names]
        failures = standard_failures + [f"diagnostic:{d}" for d in diagnostic_failures]
        blocking_failures = list(standard_failures)

    # --- Gate Failure Summary ---
    if DEBUG_DIAGNOSTICS:
        for name in failures:
            metric_map = {
                "min_windows": "n_windows", "min_sharpe": "oos_sharpe_chained",
                "min_cost_aware_sharpe": "exec_sharpe", "min_ic_tstat": "oos_ic_tstat",
                "min_ic_ir": "oos_ic_ir", "min_beat_rate": "oos_beat_rate",
                "max_drawdown": "exec_max_dd", "min_psr": "oos_psr",
                "max_beta_abs_mean": "exec_beta_abs_mean", "max_sector_abs_mean": "exec_max_sector_abs_mean",
                "max_cost_to_gross_pnl": "exec_cost_to_gross_pnl", "min_decile_spread": "decile_spread",
                "min_tail_monotonicity": "decile_monotonicity", "min_long_leg_sharpe": "exec_long_leg_sharpe",
                "min_short_leg_sharpe": "exec_short_leg_sharpe", "min_subsumption_alpha_ann": "subsumption_alpha_ann",
                "min_subsumption_alpha_tstat": "subsumption_alpha_tstat", "max_subsumption_r2": "subsumption_r2",
                "max_subsumption_loading_abs": "subsumption_max_abs_loading",
                "robust_halflife": "diag_robust_signal_halflife", "robust_caic_ratio": "diag_caic_to_raw_ic_ratio",
                "robust_turnover": "diag_robust_turnover_mean"
            }
            metric_name = metric_map.get(name, "unknown")
            val = data.get(metric_name)
            thr = _thr(name)
            is_nan = not np.isfinite(_safe_float(val, np.nan))
            log_line = f"[Gate Failure] model={data.get('model_name', '??')} | path={path_audit} | gate={name} | value={val} | threshold={thr} | is_nan={is_nan}"
            if is_nan:
                log_line += " | reason=missing_metric_not_model_failure"
            print(log_line)
    # ----------------------------

    return {
        "promotion_pass": not blocking_failures,
        "promotion_failures": ",".join(failures),
        **{f"gate_{name}": bool(ok) for name, ok in checks.items()},
        **{
            f"gate_threshold_static_{name}": float(info["static"])
            for name, info in threshold_audit.items()
        },
        **{
            f"gate_threshold_dynamic_{name}": float(info["dynamic"])
            for name, info in threshold_audit.items()
        },
        **{
            f"gate_threshold_effective_{name}": float(info["effective"])
            for name, info in threshold_audit.items()
        },
        **{
            f"gate_threshold_reason_{name}": str(info["reason"])
            for name, info in threshold_audit.items()
        },
    }


def classify_promotion_tier(
    row: dict[str, Any],
    gate_cfg: PromotionGateConfig,
    long_cand_cfg: LongAlphaCandidateConfig,
    *,
    model_kind: str = "",
) -> PromotionTier:
    """Classify a model into a promotion tier based on its gate results and metrics."""
    data = dict(row)
    _kind = str(model_kind or data.get("model_kind", "") or "").lower().strip()
    _is_short = _kind in {"short_classifier", "short_alpha"}
    _is_long = _kind in {"long_alpha", "regressor", "long_short_spread"}
    _is_overlay = _kind == "overlay_alpha"

    if bool(data.get("promotion_pass", False)):
        return PromotionTier.PRODUCTION

    if _is_long or _is_overlay:
        long_leg_ok = (
            _safe_float(data.get("exec_long_leg_sharpe"), -999.0) >= long_cand_cfg.min_long_leg_sharpe
        )
        ic_ok = (
            _first_valid_float(["oos_ic_tstat", "cs_ic_spearman_tstat"], data, -999.0)
            >= long_cand_cfg.min_ic_tstat
        )
        beat_ok = _safe_float(data.get("oos_beat_rate"), -999.0) >= long_cand_cfg.min_beat_rate
        dd_ok = (
            _safe_float(data.get("exec_max_dd", data.get("oos_max_dd")), -999.0)
            >= long_cand_cfg.max_drawdown
        )
        windows_ok = _safe_float(data.get("n_windows"), 0.0) >= long_cand_cfg.min_windows
        cost_ok = (
            _safe_float(data.get("exec_cost_to_gross_pnl"), 999.0) <= long_cand_cfg.max_cost_to_gross_pnl
        )
        decile_ok = _safe_float(data.get("decile_spread"), -999.0) >= long_cand_cfg.min_decile_spread
        if long_leg_ok and ic_ok and beat_ok and dd_ok and windows_ok and cost_ok and decile_ok:
            return PromotionTier.LONG_ALPHA_CAND

    if _is_short:
        sharpe_pos = _safe_float(data.get("exec_sharpe"), -999.0) > 0.0
        beat_pos = _safe_float(data.get("oos_beat_rate"), -999.0) >= 0.5
        if sharpe_pos and beat_pos:
            return PromotionTier.HEDGE_ONLY

    ic_tstat = _first_valid_float(["oos_ic_tstat", "cs_ic_spearman_tstat"], data, -999.0)
    beat_rate = _safe_float(data.get("oos_beat_rate"), -999.0)
    oos_sharpe = _safe_float(data.get("oos_sharpe_chained"), -999.0)
    if (ic_tstat > 0.0) or (beat_rate > 0.5) or (oos_sharpe > 0.0):
        return PromotionTier.DIAGNOSTIC_ONLY

    return PromotionTier.REJECTED


class ValidationStateCache:
    """Persistent date-level market state for executable validation."""

    def __init__(
        self, full_df: pd.DataFrame, *, cfg: EvaluationConfig, artifact_dir: str | Path | None = None
    ) -> None:
        cols = [
            "date",
            "ticker",
            "daily_return",
            "Close",
            "close",
            "AdjClose",
            "Adj Close",
            "Volume",
            "volume",
            "market_cap",
            "log_market_cap",
            "size",
            "size_exposure",
            "market_beta",
            "adv_dollar_20",
            "realised_vol_20d",
            "capm_beta",
            "sector",
            "book_to_market",
            "bm",
            "value_score",
            "quality_score",
            "roa",
            "profitability",
            "momentum",
            "momentum_exposure",
            "momentum_12m_skip1",
            "short_squeeze_risk",
            "hard_short_squeeze_filter",
            "borrow_crowding_risk",
            "short_interest_ratio",
        ]
        keep = [c for c in cols if c in full_df.columns]
        df = full_df.loc[:, keep].copy()
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date", "ticker"]).sort_values(["date", "ticker"]).reset_index(drop=True)
        self.full_df = df
        self.cfg = cfg
        RiskModel = _load_risk_model()
        self.risk_model = RiskModel(
            window=max(20, int(cfg.optimizer_lookback_days)),
            min_periods=min(20, max(5, int(cfg.optimizer_lookback_days) // 3)),
            method="ledoit_wolf",
            annualize=True,
        )
        self.price_data = _risk_price_data_from_panel(df)
        self._cache: dict[pd.Timestamp, DateLevelMarketState] = {}
        self.artifact_dir = Path(artifact_dir) if artifact_dir is not None else None
        if self.artifact_dir is not None:
            self.artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "dates": [
                str(pd.Timestamp(v).date())
                for v in sorted(pd.to_datetime(df["date"], errors="coerce").dropna().unique())
            ],
            "tickers": sorted(df["ticker"].astype(str).unique().tolist()),
            "lookback_days": int(cfg.optimizer_lookback_days),
        }
        self.signature = hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:20]

    def get(self, dt: pd.Timestamp) -> DateLevelMarketState:
        dt = pd.Timestamp(dt)
        if dt in self._cache:
            _MARKET_STATE_STATS["hits"] += 1
            return self._cache[dt]
        path = (
            self.artifact_dir / f"{dt.strftime('%Y%m%d')}_{self.signature}.pkl"
            if self.artifact_dir
            else None
        )
        if path and path.exists():
            with open(path, "rb") as fh:
                state = pickle.load(fh)
            self._cache[dt] = state
            _MARKET_STATE_STATS["hits"] += 1
            return state

        _MARKET_STATE_STATS["misses"] += 1
        t0 = time.perf_counter()
        built = self._build_state_for_day(dt)
        _MARKET_STATE_STATS["build_time_s"] += (time.perf_counter() - t0)
        
        if path:
            with open(path, "wb") as fh:
                pickle.dump(built, fh)
        self._cache[dt] = built
        return built

    def _build_state_for_day(self, dt: pd.Timestamp) -> DateLevelMarketState:
        day = self.full_df.loc[self.full_df["date"] == dt].copy()
        tickers = tuple(day["ticker"].astype(str).tolist())
        cov = _covariance_for_day(self.full_df, day, dt, self.cfg)
        sector_id_map, sector_labels = _sector_maps_for_day(day)
        exposures = self.risk_model.compute_factor_exposures_at_date(
            self.price_data,
            list(tickers),
            dt,
            sector_id_map=sector_id_map,
            sector_labels=sector_labels,
        )
        adv = _numeric_day_column(day, "adv_dollar_20", self.cfg.costs.default_adv_usd).to_numpy(dtype=float)
        vol = (
            _numeric_day_column(day, "realised_vol_20d", self.cfg.costs.default_daily_vol)
            .clip(lower=1e-6)
            .to_numpy(dtype=float)
        )
        liq_caps = np.minimum(
            float(self.cfg.max_name_weight),
            float(self.cfg.adv_fraction) * adv / max(float(self.cfg.costs.capital), 1.0),
        )
        participation_scale = float(self.cfg.costs.capital) / np.clip(adv, 1.0, None)
        crowding = _numeric_day_column(day, "borrow_crowding_risk", 0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        si = _numeric_day_column(day, "short_interest_ratio", 0.0).clip(0.0, 1.0).to_numpy(dtype=float)
        borrow_penalty = (
            float(self.cfg.costs.borrow_bps) / 10_000.0 * max(1, int(self.cfg.horizon_days)) / 252.0
        ) * (1.0 + crowding + si)
        squeeze = _numeric_day_column(day, "short_squeeze_risk", 0.0)
        hard = _numeric_day_column(day, "hard_short_squeeze_filter", 0.0)
        blocked = ((squeeze >= float(self.cfg.short_squeeze_max_risk)) | (hard >= 1.0)).to_numpy(dtype=bool)
        spec_risk = np.sqrt(np.clip(np.diag(cov), 0.0, None))
        return DateLevelMarketState(
            date=dt,
            tickers=tickers,
            ticker_to_idx={t: i for i, t in enumerate(tickers)},
            covariance=cov,
            specific_risk=spec_risk,
            factor_exposures={k: v for k, v in exposures.items()},
            adv_dollar=adv,
            daily_vol=vol,
            liquidity_caps=liq_caps,
            participation_scale=participation_scale,
            max_participation_rate=float(self.cfg.costs.max_participation_rate),
            borrow_penalty_horizon=borrow_penalty,
            crowding_risk=crowding,
            short_interest_ratio=si,
            squeeze_risk=squeeze.to_numpy(dtype=float),
            short_blocked=blocked,
        )


def _safe_float(raw: Any, default: float = 0.0) -> float:
    try:
        val = float(raw)
        return val if np.isfinite(val) else default
    except Exception:
        return default


def _first_valid_float(keys: list[str], data: dict[str, Any], default: float) -> float:
    for k in keys:
        v = data.get(k)
        if v is not None:
            try:
                fv = float(v)
                if np.isfinite(fv):
                    return fv
            except Exception:
                pass
    return default


def _gate_optional(raw: Any, op: Any, threshold: float) -> bool:
    try:
        val = float(raw)
        if not np.isfinite(val):
            return True
        return bool(op(val, threshold))
    except Exception:
        return True


def _resolve_threshold(
    cfg: PromotionGateConfig, model_kind: str, is_short: bool, field_name: str
) -> float:
    overrides = getattr(cfg, "path_overrides", {}) or {}
    kind_overrides = overrides.get(model_kind, {}) or {}
    if field_name in kind_overrides:
        return float(kind_overrides[field_name])
    if is_short:
        family_overrides = overrides.get("short_side", {}) or {}
        if field_name in family_overrides:
            return float(family_overrides[field_name])
    return float(getattr(cfg, field_name))


def _normal_critical_value(confidence: float) -> float:
    confidence = min(0.995, max(0.50, float(confidence)))
    if confidence >= 0.99:
        return 2.58
    if confidence >= 0.975:
        return 2.24
    if confidence >= 0.95:
        return 1.96
    if confidence >= 0.90:
        return 1.65
    return 1.28


def _effective_observations(data: dict[str, Any]) -> tuple[float, int, int]:
    horizon = max(1, int(_first_valid_float(["horizon_days", "nested_selected_horizon_mode"], data, 1.0)))
    n_days = int(
        max(
            0,
            _first_valid_float(["daily_ic_n_days", "ic_n_days", "exec_n_days", "decile_n_days"], data, 0.0),
        )
    )
    if n_days <= 0:
        n_windows = int(max(0, _safe_float(data.get("n_windows"), 0.0)))
        n_days = n_windows * horizon
    n_eff = float(n_days) / float(horizon)
    return n_eff, n_days, horizon


def _dynamic_thresholds(
    data: dict[str, Any],
    cfg: PromotionGateConfig,
    *,
    model_kind: str,
    is_short: bool,
) -> dict[str, dict[str, Any]]:
    """Calibrate promotion thresholds from sample geometry and execution regime.

    Static config values remain floors. Dynamic thresholds only tighten gates
    when sample support, horizon scaling, IC variance, turnover, or costs imply
    that the nominal threshold would be statistically underpowered.
    """

    fields = [
        "min_ic_tstat",
        "min_ic_ir",
        "min_sharpe",
        "min_cost_aware_sharpe",
        "min_beat_rate",
    ]
    out: dict[str, dict[str, Any]] = {}
    for field in fields:
        static = _resolve_threshold(cfg, model_kind, is_short, field)
        out[field] = {
            "static": float(static),
            "dynamic": float(static),
            "effective": float(static),
            "reason": "static_fallback",
        }
    if not bool(getattr(cfg, "dynamic_thresholds_enabled", True)):
        return out

    n_eff, n_days, horizon = _effective_observations(data)
    tcrit = _normal_critical_value(getattr(cfg, "dynamic_threshold_confidence", 0.95))
    min_eff = max(1, int(getattr(cfg, "dynamic_threshold_min_effective_obs", 12)))
    ic_std = _first_valid_float(["daily_ic_std", "oos_ic_std"], data, float("nan"))
    ref_ic_std = max(1e-6, float(getattr(cfg, "dynamic_threshold_reference_ic_std", 0.05)))
    variance_multiplier = 1.0
    if np.isfinite(ic_std) and ic_std > 0:
        variance_multiplier = min(2.0, max(1.0, float(ic_std) / ref_ic_std))
    sample_multiplier = 1.0 if n_eff >= 30 else min(2.0, math.sqrt(30.0 / max(1.0, n_eff)))

    if n_eff < min_eff:
        dyn_t = float("inf")
        reason_t = f"insufficient_effective_observations(n_eff={n_eff:.1f}, min={min_eff})"
    else:
        dyn_t = float(tcrit * sample_multiplier)
        reason_t = f"zcrit={tcrit:.2f}, n_eff={n_eff:.1f}, horizon={horizon}d"
    out["min_ic_tstat"]["dynamic"] = dyn_t
    out["min_ic_tstat"]["effective"] = max(out["min_ic_tstat"]["static"], dyn_t)
    out["min_ic_tstat"]["reason"] = reason_t

    years_eff = max(1e-6, n_days / 252.0)
    dyn_ir = float(tcrit * variance_multiplier / math.sqrt(years_eff)) if n_eff >= min_eff else float("inf")
    out["min_ic_ir"]["dynamic"] = dyn_ir
    out["min_ic_ir"]["effective"] = max(out["min_ic_ir"]["static"], dyn_ir)
    out["min_ic_ir"]["reason"] = (
        f"tcrit/sqrt(years_eff) with variance_multiplier={variance_multiplier:.2f}, "
        f"years_eff={years_eff:.2f}, horizon={horizon}d"
    )

    turnover = _first_valid_float(
        ["exec_turnover_mean", "diag_robust_turnover_mean", "avg_turnover"],
        data,
        float("nan"),
    )
    cost_ratio = _first_valid_float(["exec_cost_to_gross_pnl", "diag_cost_to_gross_ratio"], data, float("nan"))
    ref_turnover = max(1e-6, float(getattr(cfg, "dynamic_threshold_reference_turnover", 0.35)))
    turnover_uplift = 0.0
    if np.isfinite(turnover) and turnover > ref_turnover:
        turnover_uplift = min(0.50, 0.10 * (float(turnover) / ref_turnover - 1.0))
    cost_uplift = 0.0
    if np.isfinite(cost_ratio):
        cost_uplift = min(0.75, max(0.0, 0.25 * float(cost_ratio)))
    sample_uplift = 0.0 if n_eff >= 30 else min(0.25, 0.05 * (30.0 / max(1.0, n_eff) - 1.0))

    dyn_sharpe = float(out["min_sharpe"]["static"] + turnover_uplift + sample_uplift)
    out["min_sharpe"]["dynamic"] = dyn_sharpe
    out["min_sharpe"]["effective"] = max(out["min_sharpe"]["static"], dyn_sharpe)
    out["min_sharpe"]["reason"] = (
        f"turnover_uplift={turnover_uplift:.3f}, sample_uplift={sample_uplift:.3f}, n_eff={n_eff:.1f}"
    )

    dyn_exec_sharpe = float(out["min_cost_aware_sharpe"]["static"] + turnover_uplift + cost_uplift + sample_uplift)
    out["min_cost_aware_sharpe"]["dynamic"] = dyn_exec_sharpe
    out["min_cost_aware_sharpe"]["effective"] = max(out["min_cost_aware_sharpe"]["static"], dyn_exec_sharpe)
    out["min_cost_aware_sharpe"]["reason"] = (
        f"cost_uplift={cost_uplift:.3f}, turnover_uplift={turnover_uplift:.3f}, "
        f"sample_uplift={sample_uplift:.3f}"
    )

    dyn_beat = float(out["min_beat_rate"]["static"] + (0.05 if n_eff < 30 else 0.0))
    out["min_beat_rate"]["dynamic"] = dyn_beat
    out["min_beat_rate"]["effective"] = min(0.80, max(out["min_beat_rate"]["static"], dyn_beat))
    out["min_beat_rate"]["reason"] = f"small_sample_buffer={'yes' if n_eff < 30 else 'no'}"
    return out


def _numeric_day_column(day: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in day.columns:
        return (
            pd.to_numeric(day[column], errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(default)
        )
    return pd.Series(float(default), index=day.index, dtype=float)


def _normalise_gross(weights: pd.Series, max_gross: float) -> pd.Series:
    gross = float(weights.abs().sum())
    if gross <= 1e-12:
        return weights * 0.0
    return weights * min(float(max_gross), gross) / gross


def _canonicalize_scored_panel(scored: pd.DataFrame) -> pd.DataFrame:
    """Enforce one executable observation per date/ticker.

    Feature joins can legitimately create duplicate rows during research, but
    portfolio construction and simulation are position ledgers. Allowing
    duplicate ticker rows causes a single constructed weight to be counted
    multiple times by the simulator, which is how a neutral book can appear
    non-neutral after construction. Numeric fields are averaged; categorical
    fields use the last non-null observation for deterministic lineage.
    """
    if scored is None or scored.empty:
        return pd.DataFrame() if scored is None else scored.copy()
    df = scored.copy()
    if "date" not in df.columns or "ticker" not in df.columns:
        return df
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["ticker"] = df["ticker"].astype(str)
    df = df.dropna(subset=["date", "ticker"])
    if not bool(df.duplicated(["date", "ticker"]).any()):
        return df.sort_values(["date", "ticker"]).reset_index(drop=True)

    agg: dict[str, str] = {}
    for col in df.columns:
        if col in {"date", "ticker"}:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            agg[col] = "mean"
        else:
            agg[col] = "last"
    out = df.groupby(["date", "ticker"], as_index=False, sort=True).agg(agg)
    out.attrs["deduplicated_rows"] = int(len(df) - len(out))
    return out.reset_index(drop=True)


def _assert_unique_target_weights(targets: pd.DataFrame) -> None:
    """Fail loudly if a target-weight ledger is not uniquely keyed."""
    if targets is None or targets.empty:
        return
    dup = targets.duplicated(["date", "ticker"], keep=False)
    if bool(dup.any()):
        sample = (
            targets.loc[dup, ["date", "ticker"]]
            .drop_duplicates()
            .head(5)
            .to_dict("records")
        )
        raise MetricIntegrityError(
            "target_weights must be unique by date/ticker before simulation; "
            f"duplicates={int(dup.sum())} sample={sample}"
        )


def _write_simulation_cache(path: Path | None, daily: pd.Series, pnl: pd.DataFrame) -> None:
    if path is None:
        return
    tmp: Path | None = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("wb", dir=path.parent, delete=False) as fh:
            pickle.dump({"daily": daily, "pnl": pnl}, fh)
            tmp = Path(fh.name)
        tmp.replace(path)
    except Exception:
        if tmp is not None:
            tmp.unlink(missing_ok=True)


def _read_simulation_cache(path: Path | None) -> tuple[pd.Series, pd.DataFrame] | None:
    if path is None or not path.exists():
        return None
    try:
        with open(path, "rb") as fh:
            payload = pickle.load(fh)
        daily, pnl = payload.get("daily"), payload.get("pnl")
        if isinstance(daily, pd.Series) and isinstance(pnl, pd.DataFrame):
            daily = daily.copy()
            pnl = pnl.copy()
            daily.attrs["_is_cached"] = True
            pnl.attrs["_is_cached"] = True
            return daily, pnl
    except Exception:
        return None
    return None


def _simulation_cache_path(
    df: pd.DataFrame,
    cfg: EvaluationConfig,
    state_cache: ValidationStateCache | None,
    *,
    target_weights: pd.DataFrame | None = None,
) -> Path | None:
    if state_cache is None or state_cache.artifact_dir is None:
        return None
    try:
        weight_digest = None
        if target_weights is not None and not target_weights.empty:
            tw = target_weights[["date", "ticker", "target_weight"]].copy()
            tw["date"] = pd.to_datetime(tw["date"], errors="coerce").astype("int64", copy=False)
            tw["ticker"] = tw["ticker"].astype(str)
            tw["target_weight"] = pd.to_numeric(tw["target_weight"], errors="coerce").fillna(0.0).round(12)
            tw = tw.sort_values(["date", "ticker"]).reset_index(drop=True)
            weight_digest = hashlib.sha1(pd.util.hash_pandas_object(tw, index=False).values.tobytes()).hexdigest()
        else:
            cols = [c for c in ["date", "ticker", "score", "next_return", "daily_return", "forward_return"] if c in df.columns]
            content = df[cols].copy().sort_values([c for c in ["date", "ticker"] if c in cols]).reset_index(drop=True)
            weight_digest = hashlib.sha1(pd.util.hash_pandas_object(content, index=False).values.tobytes()).hexdigest()
        payload = {
            "cfg": asdict(cfg),
            "rows": int(len(df)),
            "cols": sorted(df.columns.tolist()),
            "signature": state_cache.signature,
            "target_weights": weight_digest,
        }
        digest = hashlib.sha1(repr(payload).encode("utf-8")).hexdigest()[:24]
        cache_dir = state_cache.artifact_dir / "simulation_results"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir / f"{digest}.pkl"
    except Exception:
        return None
