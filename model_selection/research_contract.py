from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from model_selection.research_numerics_core import compute_daily_ic_series


@dataclass(frozen=True)
class FeatureSpec:
    """Research contract for a production alpha feature.

    P10: decay_profile tags the expected signal persistence:
      - "fast": price-based signals (momentum, reversal) — halflife 1-5 days
      - "medium": technical/flow signals (volume, volatility) — halflife 5-15 days
      - "slow": fundamental signals (quality, accruals, margins) — halflife 20-90 days
    This enables the model to learn horizon-appropriate feature weighting.
    """

    name: str
    family: str
    expected_sign: int
    horizon_days: int
    timestamp: str
    hypothesis: str
    source: str
    decay_profile: str = "medium"  # fast | medium | slow


@dataclass(frozen=True)
class SignalExecutionTimingContract:
    """Microstructure timing contract for alpha research panels."""

    signal_time: str = "close_t"
    execution_time: str = "next_open_or_next_vwap"
    holding_period_name: str = "horizon"


@dataclass(frozen=True)
class TimingValidationReport:
    """Result of enforcing signal -> execution timing alignment."""

    contract: SignalExecutionTimingContract
    n_features: int
    n_errors: int
    n_warnings: int
    errors: tuple[str, ...]
    warnings: tuple[str, ...]


class TimingContractViolation(ValueError):
    """Raised when a feature panel violates the signal/execution timing contract."""


FEATURE_SPECS: dict[str, FeatureSpec] = {
    "f_trend": FeatureSpec("f_trend", "trend", 1, 5, "close_t", "Composite trend should rank continuation candidates.", "internal"),
    "ret_5d": FeatureSpec("ret_5d", "momentum", 1, 5, "close_t", "Short-horizon winners should continue over the rebalance horizon.", "Jegadeesh-Titman"),
    "ret_10d": FeatureSpec("ret_10d", "momentum", 1, 5, "close_t", "Two-week winners should continue over the next week.", "Jegadeesh-Titman"),
    "ret_20d": FeatureSpec("ret_20d", "momentum", 1, 5, "close_t", "One-month cross-sectional momentum should rank near-term continuation.", "Jegadeesh-Titman"),
    "cs_momentum_percentile": FeatureSpec("cs_momentum_percentile", "momentum", 1, 20, "close_t", "Six-month skip-month momentum percentile should rank medium-term continuation.", "Jegadeesh-Titman"),
    "momentum_12m_skip1": FeatureSpec("momentum_12m_skip1", "momentum", 1, 20, "close_t", "12M-1M momentum is a replicated cross-sectional momentum factor.", "Jegadeesh-Titman"),
    "momentum_1m_skip_eom": FeatureSpec("momentum_1m_skip_eom", "short_momentum", 1, 21, "close_t", "One-month momentum excluding end-of-month flow should persist.", "Medhat-Schmeling"),
    "short_term_momentum_score": FeatureSpec("short_term_momentum_score", "short_momentum", 1, 21, "close_t", "Turnover-conditioned losers should continue under short-term momentum.", "Medhat-Schmeling"),
    "short_term_reversal": FeatureSpec("short_term_reversal", "reversal", 1, 3, "close_t", "Recent losers should mean-revert over short horizons.", "Jegadeesh"),
    "industry_relative_reversal": FeatureSpec("industry_relative_reversal", "reversal", 1, 5, "close_t", "Industry-relative extremes should reverse after stripping sector momentum.", "RRLP"),
    "high_vol_reversal_flag": FeatureSpec("high_vol_reversal_flag", "reversal_conditioner", 1, 5, "close_t", "High volatility accelerates reversal realization.", "RRLP"),
    "nearness_52w_low": FeatureSpec("nearness_52w_low", "reversal", 1, 10, "close_t", "Near-low names may bounce when selling pressure is exhausted.", "internal"),
    "nearness_52w_high": FeatureSpec("nearness_52w_high", "momentum", 1, 20, "close_t", "52-week-high proximity captures continuation and anchoring.", "George-Hwang"),
    "low_vol_score": FeatureSpec("low_vol_score", "quality_lowvol", 1, 20, "close_t", "Low-vol anomaly should improve risk-adjusted ranking.", "Frazzini-Pedersen"),
    "quality_score": FeatureSpec("quality_score", "quality", 1, 20, "close_t", "Stable positive return over volatility proxies quality/profitability.", "quality anomaly"),
    "capm_alpha": FeatureSpec("capm_alpha", "residual_alpha", 1, 5, "close_t", "Positive residual alpha should persist after market adjustment.", "CAPM"),
    "capm_residual_vol": FeatureSpec("capm_residual_vol", "risk", -1, 5, "close_t", "High idiosyncratic volatility should be penalized unless separately conditioned.", "idiosyncratic vol anomaly"),
    "vol_ratio_5_20": FeatureSpec("vol_ratio_5_20", "regime", 0, 5, "close_t", "Volatility term structure is a conditional regime feature.", "internal"),
    "rolling_vol_20": FeatureSpec("rolling_vol_20", "risk", -1, 5, "close_t", "Realized volatility is a risk conditioner, not standalone alpha.", "risk control"),
    "turnover_pct_rank": FeatureSpec("turnover_pct_rank", "liquidity", 0, 21, "close_t", "Turnover conditions whether short-term return is momentum or reversal.", "Medhat-Schmeling"),
    "sector_relative_20d": FeatureSpec("sector_relative_20d", "sector_relative", 1, 5, "close_t", "Stock momentum relative to sector should rank idiosyncratic continuation.", "multi-factor construction"),
    "sector_relative_60d": FeatureSpec("sector_relative_60d", "sector_relative", 1, 20, "close_t", "Slower sector-relative trend should rank idiosyncratic continuation.", "multi-factor construction"),
    "momentum_acceleration": FeatureSpec("momentum_acceleration", "momentum", 1, 5, "close_t", "Acceleration in short-horizon returns should identify fresh momentum.", "internal"),
    # P10: Cross-horizon fundamental features — slow decay profile (halflife 20-90 days).
    # These provide persistent alpha signals that complement fast price-based features.
    "f_score": FeatureSpec("f_score", "fundamental_quality", 1, 63, "filing_or_rdq", "Higher Piotroski quality should improve medium-horizon alpha.", "Piotroski", "slow"),
    "accruals_ratio": FeatureSpec("accruals_ratio", "fundamental_quality", -1, 63, "filing_or_rdq", "High accruals imply lower earnings quality.", "Sloan", "slow"),
    "roa": FeatureSpec("roa", "fundamental_quality", 1, 63, "filing_or_rdq", "Profitability should be rewarded after accounting for availability lag.", "quality anomaly", "slow"),
    "delta_roa": FeatureSpec("delta_roa", "fundamental_quality", 1, 63, "filing_or_rdq", "Improving profitability should predict relative strength.", "quality anomaly", "slow"),
    "delta_leverage": FeatureSpec("delta_leverage", "fundamental_quality", -1, 63, "filing_or_rdq", "Rising leverage should penalize balance-sheet quality.", "quality anomaly", "slow"),
    "gross_margin": FeatureSpec("gross_margin", "fundamental_quality", 1, 63, "filing_or_rdq", "Higher gross margins indicate stronger unit economics.", "Compustat", "slow"),
    "delta_gross_margin": FeatureSpec("delta_gross_margin", "fundamental_deterioration", 1, 63, "filing_or_rdq", "Improving gross margins should improve forward alpha.", "Compustat", "slow"),
    "operating_margin": FeatureSpec("operating_margin", "fundamental_quality", 1, 63, "filing_or_rdq", "Higher operating margins indicate stronger profitability.", "Compustat", "slow"),
    "delta_operating_margin": FeatureSpec("delta_operating_margin", "fundamental_deterioration", 1, 63, "filing_or_rdq", "Improving operating margins should improve forward alpha.", "Compustat", "slow"),
    "margin_deterioration": FeatureSpec("margin_deterioration", "fundamental_deterioration", -1, 63, "filing_or_rdq", "Declining margins are a short-book deterioration signal.", "Compustat", "slow"),
    "debt_to_assets": FeatureSpec("debt_to_assets", "fundamental_leverage", -1, 63, "filing_or_rdq", "High debt-to-assets increases balance-sheet fragility.", "Compustat", "slow"),
    "total_debt_to_assets": FeatureSpec("total_debt_to_assets", "fundamental_leverage", -1, 63, "filing_or_rdq", "High total debt-to-assets should be penalized.", "Compustat", "slow"),
    "weak_profitability": FeatureSpec("weak_profitability", "fundamental_deterioration", -1, 63, "filing_or_rdq", "Weak profitability is a short-risk signal.", "Compustat", "slow"),
    "share_issuance_growth": FeatureSpec("share_issuance_growth", "dilution", -1, 63, "filing_or_rdq", "Rising shares outstanding indicates dilution pressure.", "Compustat", "slow"),
    "dilution_pressure": FeatureSpec("dilution_pressure", "dilution", -1, 63, "filing_or_rdq", "Positive dilution pressure should weaken future returns.", "Compustat", "slow"),
    "late_filing_flag": FeatureSpec("late_filing_flag", "reporting_quality", -1, 63, "filing_or_rdq", "Late reporting can proxy accounting uncertainty.", "Compustat", "slow"),
    "restatement_like_flag": FeatureSpec("restatement_like_flag", "reporting_quality", -1, 63, "filing_or_rdq", "Large accounting changes and late reports proxy restatement-like risk.", "Compustat", "slow"),
    "fundamental_deterioration_score": FeatureSpec("fundamental_deterioration_score", "fundamental_deterioration", -1, 63, "filing_or_rdq", "Composite deterioration should identify weak short candidates.", "Compustat", "slow"),
    # P10: fundamental_coverage — fraction of fundamental features available per stock-date.
    # Conditioning feature (expected_sign=0): tells the model when fundamental alphas are reliable.
    # Small caps with sparse filings get low coverage → model should downweight fundamental signals.
    # Family "liquidity" ensures it is available to all model kinds (not excluded by economic mandate).
    "fundamental_coverage": FeatureSpec("fundamental_coverage", "liquidity", 0, 63, "close_t", "Higher fundamental data coverage indicates more reliable fundamental signals.", "internal", "slow"),
    "short_interest_ratio": FeatureSpec("short_interest_ratio", "crowding", -1, 21, "exchange_short_interest_publication", "High short interest can indicate crowding and borrow risk.", "WRDS optional", "medium"),
    "days_to_cover": FeatureSpec("days_to_cover", "crowding", -1, 21, "exchange_short_interest_publication", "High days-to-cover increases squeeze risk.", "WRDS optional", "medium"),
    "borrow_crowding_risk": FeatureSpec("borrow_crowding_risk", "crowding", -1, 21, "securities_lending_publication", "Expensive/crowded borrow should reduce short attractiveness.", "WRDS optional", "medium"),
    "short_squeeze_risk": FeatureSpec("short_squeeze_risk", "squeeze_filter", -1, 5, "close_t", "High squeeze risk should block or downweight shorts.", "risk control"),
    "hard_short_squeeze_filter": FeatureSpec("hard_short_squeeze_filter", "squeeze_filter", -1, 5, "close_t", "Hard squeeze filter prevents initiating dangerous shorts.", "risk control"),
    # P3: Regime probability features — HMM posterior probabilities as model inputs.
    # These let the model learn regime-conditional feature weights without separate training.
    "regime_score": FeatureSpec("regime_score", "regime", 0, 5, "close_t", "Continuous regime risk score (0=Bull, 1=Crisis) enables regime-conditional weighting.", "HMM posterior"),
    "regime_proba_bull": FeatureSpec("regime_proba_bull", "regime", 0, 5, "close_t", "Probability of Bull regime — high values indicate favorable risk environment.", "HMM posterior"),
    "regime_proba_bear": FeatureSpec("regime_proba_bear", "regime", 0, 5, "close_t", "Probability of Bear regime — risk-off context feature.", "HMM posterior"),
    "regime_proba_crisis": FeatureSpec("regime_proba_crisis", "regime", 0, 5, "close_t", "Probability of Crisis regime — extreme stress context feature.", "HMM posterior"),
    # P4: Regime-aware cross-asset features — provide macro context for alpha conditioning.
    "vol_risk_premium": FeatureSpec("vol_risk_premium", "regime", 0, 5, "close_t", "VRP = implied vol - realized vol; high VRP signals risk aversion premium.", "VIX vs SPY RV"),
    "credit_spread_zscore": FeatureSpec("credit_spread_proxy", "regime", 0, 5, "close_t", "Widening credit spreads (HYG-IEF) signal risk-off regime.", "credit markets"),
    "yield_curve_zscore": FeatureSpec("yield_curve_slope", "regime", 0, 5, "close_t", "Inverted yield curve (IEF-SHY) signals recession risk.", "rates"),
    # P4: Additional computed features — already in pipeline but not registered as alpha features.
    "momentum_consistency_score": FeatureSpec("momentum_consistency_score", "momentum", 1, 10, "close_t", "Consistency of momentum across sub-periods; more reliable than raw momentum.", "info discreteness"),
    "information_discreteness": FeatureSpec("information_discreteness", "momentum", 1, 10, "close_t", "Da et al. 2014: continuous-momentum stocks outperform discrete jumpers.", "information discreteness"),
    "volatility_trend": FeatureSpec("volatility_trend", "regime", 0, 5, "close_t", "Direction of volatility trend (rising/falling) is a regime context signal.", "vol dynamics"),
}


# ── P21: FeatureFamilyRegistry — structured family grouping ───────────────────

class FeatureFamilyRegistry:
    """Read-only registry grouping FEATURE_SPECS into logical families.

    Supports:
      - listing all families
      - listing features by family
      - mapping feature -> family
      - returning metadata for a feature
      - handling unknown features as "unknown"

    This is a lightweight wrapper around the existing FEATURE_SPECS dict.
    It does NOT duplicate metadata — it reads from the source-of-truth.
    """

    def __init__(self, specs: dict[str, FeatureSpec] | None = None):
        self._specs: dict[str, FeatureSpec] = dict(specs or FEATURE_SPECS)
        self._family_map: dict[str, list[str]] = {}
        for name, spec in self._specs.items():
            self._family_map.setdefault(spec.family, []).append(name)

    @property
    def families(self) -> list[str]:
        return sorted(self._family_map)

    @property
    def n_features(self) -> int:
        return len(self._specs)

    @property
    def n_families(self) -> int:
        return len(self._family_map)

    def features_by_family(self, family: str) -> list[str]:
        return list(self._family_map.get(family, []))

    def family_of(self, feature: str) -> str:
        spec = self._specs.get(feature)
        return spec.family if spec else "unknown"

    def spec_of(self, feature: str) -> FeatureSpec | None:
        return self._specs.get(feature)

    def list_all_features(self) -> list[str]:
        return sorted(self._specs)

    def horizon_days_of(self, feature: str) -> int:
        spec = self._specs.get(feature)
        return spec.horizon_days if spec else 0

    def decay_profile_of(self, feature: str) -> str:
        spec = self._specs.get(feature)
        return spec.decay_profile if spec else "unknown"

    def expected_sign_of(self, feature: str) -> int:
        spec = self._specs.get(feature)
        return spec.expected_sign if spec else 0


ALPHA_METADATA_COLUMNS = {
    "date",
    "ticker",
    "sector",
    "regime_label",
    "direction",
    "signal_time",
    "execution_time",
    "holding_period",
    "horizon",
}

TARGET_COLUMNS = {
    "forward_return",
    "forward_return_risk_adj",
    "forward_return_excess",
    "target_return",
    "target_return_net",
    "target_rank",
    "target_down_decile",
    "target_up",
    "target_expected_cost",
    "spy_forward_5d",
    # target_construction.py target menu columns
    "raw_5d",
    "raw_10d",
    "market_residual_5d",
    "sector_residual_5d",
    "factor_residual_5d",
    "vol_scaled_5d",
    "cost_adjusted_5d",
}

# Short-side target columns built by short_modeling.build_short_targets().
# These are derived from forward returns and MUST NOT become model features.
# Valid short-side predictor features (short_interest_ratio, short_squeeze_risk, etc.)
# are registered in FEATURE_SPECS and are explicitly allowed as numeric features.
SHORT_TARGET_COLUMNS: frozenset[str] = frozenset({
    "short_neg_residual",
    "short_vol_expansion",
    "short_liq_drain",
    "short_failed_mom",
    "short_downside_skew",
})

RISK_EXECUTION_COLUMNS = {
    "daily_return",
    "adv_dollar_20",
    "realised_vol_20d",
    "vol_20_simple",
    "capm_beta",
    "expected_round_trip_cost_frac",
}

TIMING_AUDIT_COLUMNS = {
    "feature_timestamp",
    "feature_asof",
    "feature_available_at",
}


def is_model_feature_column(column: str, series: pd.Series) -> bool:
    """Return True only for columns eligible to be learned alpha predictors.

    Leakage protection layers (applied in order):
    1. Explicit block lists: metadata, target, risk/execution columns.
    2. Short target columns (short_modeling.build_short_targets outputs) — always blocked.
    3. Pattern blocks: target_*, *forward*, *direction*.
    4. Short-prefixed columns not in FEATURE_SPECS — blocked unless explicitly contracted.
       Valid short-side predictor features (short_interest_ratio, short_squeeze_risk,
       hard_short_squeeze_filter, borrow_crowding_risk, days_to_cover) are in FEATURE_SPECS
       and remain allowed when numeric.
    5. Must be numeric dtype.
    """
    c = str(column)
    if c in ALPHA_METADATA_COLUMNS or c in TARGET_COLUMNS or c in RISK_EXECUTION_COLUMNS or c in TIMING_AUDIT_COLUMNS:
        return False
    if c.endswith(("_timestamp", "__timestamp", "_asof", "__asof", "_available_at", "__available_at")):
        return False
    if c in SHORT_TARGET_COLUMNS:
        return False
    if c.startswith("target_") or "forward" in c.lower() or "direction" in c.lower():
        return False
    if c.startswith("short_") and c not in FEATURE_SPECS:
        return False
    return pd.api.types.is_numeric_dtype(series)


def _normalise_signal_dates(values: pd.Series) -> pd.Series:
    return pd.to_datetime(values, errors="coerce").dt.normalize()


def _availability_columns_for_feature(feature: str, columns: Iterable[str]) -> list[str]:
    names = {
        f"{feature}__timestamp",
        f"{feature}_timestamp",
        f"{feature}__asof",
        f"{feature}_asof",
        f"{feature}__available_at",
        f"{feature}_available_at",
    }
    return [c for c in columns if c in names]


def validate_signal_execution_timing(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    prediction_horizon_days: int,
    horizon_report: dict[str, object] | None = None,
    enforce_horizon_alignment: bool = False,
    contract: SignalExecutionTimingContract | None = None,
) -> TimingValidationReport:
    """Enforce the institutional signal -> execution timing contract.

    Contract:
    - ``signal_time`` is close_t, represented by the panel ``date``.
    - execution is next open or next VWAP, represented by optional
      ``execution_time`` values strictly after signal date.
    - holding period equals the configured model horizon.

    The function does not alter feature definitions. It only validates that
    feature availability metadata, when present, is no later than signal_time
    and that documented feature timestamps do not imply future data.
    """
    timing_contract = contract or SignalExecutionTimingContract()
    errors: list[str] = []
    warnings: list[str] = []
    features = [str(c) for c in feature_columns]
    if df is None or df.empty:
        errors.append("timing panel is empty")
        report = TimingValidationReport(timing_contract, len(features), len(errors), len(warnings), tuple(errors), tuple(warnings))
        raise TimingContractViolation("; ".join(errors))
    signal_date = pd.Series(pd.NaT, index=df.index)
    if "date" not in df.columns:
        errors.append("timing panel missing required signal date column 'date'")
    else:
        signal_date = _normalise_signal_dates(df["date"])
        if signal_date.isna().any():
            errors.append("signal_time contains invalid dates")

    if "signal_time" in df.columns and "date" in df.columns:
        signal_meta = _normalise_signal_dates(df["signal_time"])
        bad = signal_meta.notna() & signal_date.notna() & (signal_meta != signal_date)
        if bool(bad.any()):
            errors.append(f"signal_time metadata not aligned to close_t/date on {int(bad.sum())} rows")

    if "execution_time" in df.columns and "date" in df.columns:
        execution_date = _normalise_signal_dates(df["execution_time"])
        bad = execution_date.notna() & signal_date.notna() & (execution_date <= signal_date)
        if bool(bad.any()):
            errors.append(f"execution_time must be after signal_time on {int(bad.sum())} rows")

    holding_col = "holding_period" if "holding_period" in df.columns else ("horizon" if "horizon" in df.columns else None)
    if holding_col is not None:
        hp = pd.to_numeric(df[holding_col], errors="coerce")
        bad = hp.notna() & (hp.astype(float) != float(prediction_horizon_days))
        if bool(bad.any()):
            errors.append(f"{holding_col} differs from configured horizon on {int(bad.sum())} rows")

    allowed_spec_timestamps = {
        "close_t",
        "filing_or_rdq",
        "exchange_short_interest_publication",
        "securities_lending_publication",
    }
    for feature in features:
        spec = FEATURE_SPECS.get(feature)
        if spec is not None and spec.timestamp not in allowed_spec_timestamps:
            errors.append(f"{feature}: unsupported feature timestamp contract '{spec.timestamp}'")
        for col in _availability_columns_for_feature(feature, df.columns):
            available = _normalise_signal_dates(df[col])
            bad = available.notna() & signal_date.notna() & (available > signal_date)
            if bool(bad.any()):
                errors.append(f"{feature}: {col} is after signal_time on {int(bad.sum())} rows")

    if horizon_report:
        n_misaligned = int(horizon_report.get("n_misaligned", 0) or 0)
        if n_misaligned > 0:
            names = [m[0] for m in horizon_report.get("misaligned", [])]
            msg = f"HorizonAlignment found {n_misaligned} misaligned feature(s): {names}"
            if enforce_horizon_alignment:
                errors.append(msg)
            else:
                warnings.append(msg)

    report = TimingValidationReport(
        timing_contract,
        len(features),
        len(errors),
        len(warnings),
        tuple(errors),
        tuple(warnings),
    )
    if errors:
        raise TimingContractViolation("; ".join(errors))
    return report


def _daily_rank_ic(df: pd.DataFrame, feature: str, target_col: str) -> float:
    vals: list[float] = []
    for _, g in df[["date", feature, target_col]].dropna().groupby("date", sort=True):
        if len(g) < 8 or g[feature].nunique() < 2 or g[target_col].nunique() < 2:
            continue
        corr = g[feature].corr(g[target_col], method="spearman")
        if np.isfinite(corr):
            vals.append(float(corr))
    return float(np.mean(vals)) if vals else float("nan")


def _monotonicity(df: pd.DataFrame, feature: str, target_col: str, *, bins: int = 5) -> tuple[float, float]:
    spreads: list[float] = []
    mono_scores: list[float] = []
    for _, g in df[["date", feature, target_col]].dropna().groupby("date", sort=True):
        if len(g) < bins * 3 or g[feature].nunique() < bins:
            continue
        try:
            bucket = pd.qcut(g[feature].rank(method="first"), bins, labels=False, duplicates="drop")
        except ValueError:
            continue
        by_bucket = g.groupby(bucket, observed=True)[target_col].mean()
        if len(by_bucket) < 3:
            continue
        ordered = by_bucket.sort_index()
        spreads.append(float(ordered.iloc[-1] - ordered.iloc[0]))
        diffs = np.diff(ordered.to_numpy(dtype=float))
        mono_scores.append(float((diffs >= 0.0).mean()) if len(diffs) else float("nan"))
    spread = float(np.nanmean(spreads)) if spreads else float("nan")
    mono = float(np.nanmean(mono_scores)) if mono_scores else float("nan")
    return spread, mono


def audit_feature_contract(
    df: pd.DataFrame,
    feature_columns: Iterable[str],
    *,
    target_col: str = "target_return",
) -> pd.DataFrame:
    """
    Produce an auditable feature ledger for model-selection inputs.

    This is intentionally feature-level, not model-level: a model that passes OOS
    but relies on undocumented or non-monotonic features is not production-ready.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    work = df.copy()
    work["date"] = pd.to_datetime(work["date"], errors="coerce")
    if target_col not in work.columns:
        return pd.DataFrame()

    existing = [f for f in feature_columns if f in work.columns]
    if not existing:
        return pd.DataFrame()

    # Clean all features and target upfront (one pass instead of per-feature)
    for c in existing + [target_col]:
        work[c] = pd.to_numeric(work[c], errors="coerce").replace([np.inf, -np.inf], np.nan)

    n = max(len(work), 1)

    # ── Batch IC: compute for all features at once using tensor backend ──
    ic_lookup: dict[str, float] = {}
    try:
        ic_df, _, _ = compute_daily_ic_series(
            work, existing, target_col,
            min_breadth=8, mode="auto",
        )
        ic_lookup = ic_df.mean(axis=0).to_dict() if ic_df is not None else {}
    except Exception:
        for feat in existing:
            ic_lookup[feat] = _daily_rank_ic(work, feat, target_col)

    rows: list[dict[str, object]] = []
    for feature in existing:
        spec = FEATURE_SPECS.get(feature)
        s = work[feature]
        coverage = float(s.notna().mean())
        nonzero_rate = float((s.fillna(0.0).abs() > 1e-12).sum() / n)
        ic = ic_lookup.get(feature, float("nan"))
        spread, monotonicity = _monotonicity(work, feature, target_col)
        expected_sign = int(spec.expected_sign) if spec else 0
        sign_aligned = (
            bool(np.sign(ic) == expected_sign)
            if expected_sign != 0 and np.isfinite(ic) and abs(ic) > 1e-12
            else None
        )
        rows.append(
            {
                "feature": feature,
                "known_contract": spec is not None,
                "family": spec.family if spec else "unknown",
                "timestamp": spec.timestamp if spec else "unknown",
                "expected_sign": expected_sign,
                "horizon_days": spec.horizon_days if spec else np.nan,
                "coverage": coverage,
                "nonzero_rate": nonzero_rate,
                "daily_spearman_ic": ic,
                "quintile_spread": spread,
                "quintile_monotonicity": monotonicity,
                "sign_aligned": sign_aligned,
                "hypothesis": spec.hypothesis if spec else "",
                "source": spec.source if spec else "",
            }
        )
    return pd.DataFrame(rows)


def get_horizon_alignment_report(
    feature_columns: Iterable[str],
    prediction_horizon_days: int,
    *,
    alignment_multiplier: float = 2.0,
) -> dict[str, object]:
    """Report feature horizon alignment for a given prediction horizon.

    A feature is *aligned* when its documented ``horizon_days`` satisfies:
        feature.horizon_days <= prediction_horizon_days * alignment_multiplier

    Features with ``horizon_days > threshold`` are *misaligned*: they predict
    alpha at a timescale much longer than the target window.  Mixing them in a
    5-day model creates spurious in-sample fit that collapses OOS.

    Returns a dict with:
      aligned         — list[str] of feature names within horizon
      misaligned      — list[tuple[str, int, str]] of (name, horizon_days, timestamp)
      unknown         — list[str] of features not in FEATURE_SPECS
      prediction_horizon  — int, the target horizon supplied
      multiplier          — float, the alignment_multiplier used
      n_aligned / n_misaligned / n_unknown — counts
    """
    aligned: list[str] = []
    misaligned: list[tuple[str, int, str]] = []
    unknown: list[str] = []
    threshold = int(prediction_horizon_days) * float(alignment_multiplier)
    for col in feature_columns:
        spec = FEATURE_SPECS.get(str(col))
        if spec is None:
            unknown.append(str(col))
            continue
        if spec.horizon_days > threshold:
            misaligned.append((str(col), int(spec.horizon_days), str(spec.timestamp)))
        else:
            aligned.append(str(col))
    return {
        "aligned": aligned,
        "misaligned": misaligned,
        "unknown": unknown,
        "prediction_horizon": int(prediction_horizon_days),
        "multiplier": float(alignment_multiplier),
        "n_aligned": len(aligned),
        "n_misaligned": len(misaligned),
        "n_unknown": len(unknown),
    }


def filter_horizon_aligned_features(
    feature_columns: Iterable[str],
    prediction_horizon_days: int,
    *,
    alignment_multiplier: float = 2.0,
) -> list[str]:
    """Return only features whose documented horizon is compatible with the target horizon.

    Wraps ``get_horizon_alignment_report`` — use when enforce=True.
    Features not in FEATURE_SPECS are retained (unknown ≠ misaligned).
    """
    report = get_horizon_alignment_report(
        feature_columns,
        prediction_horizon_days,
        alignment_multiplier=alignment_multiplier,
    )
    return report["aligned"] + report["unknown"]


def summarize_feature_contract(ledger: pd.DataFrame) -> dict[str, float]:
    if ledger is None or ledger.empty:
        return {
            "feature_known_contract_rate": float("nan"),
            "feature_min_coverage": float("nan"),
            "feature_positive_ic_rate": float("nan"),
            "feature_sign_aligned_rate": float("nan"),
        }
    known = pd.to_numeric(ledger["known_contract"], errors="coerce")
    ic = pd.to_numeric(ledger["daily_spearman_ic"], errors="coerce")
    sign = ledger["sign_aligned"].dropna()
    return {
        "feature_known_contract_rate": float(known.mean()),
        "feature_min_coverage": float(pd.to_numeric(ledger["coverage"], errors="coerce").min()),
        "feature_positive_ic_rate": float((ic > 0.0).mean()) if len(ic) else float("nan"),
        "feature_sign_aligned_rate": float(sign.astype(bool).mean()) if len(sign) else float("nan"),
    }
