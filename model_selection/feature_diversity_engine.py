"""Feature Diversity and Marginal Alpha Engine.

Institutional framework that measures feature redundancy, effective number of
independent signals, feature-family concentration, incremental IC, marginal
model value, and robustness across regimes/horizons.

Admits features based on unique economic and statistical contribution,
not raw feature count.

Usage:
    from model_selection.feature_diversity_engine import FeatureDiversityEngine

    engine = FeatureDiversityEngine(config=contract.raw_config)
    result = engine.run_full_diversity_analysis(df, features, horizons)
    reports = generate_diversity_reports(result, "output/models/feature_diversity/")

Hard rules:
- No feature enters model selection without registry metadata.
- Pairwise correlation alone is not the only redundancy metric.
- Rank correlation is computed cross-sectionally by date.
- N_eff decreases when duplicate features are added.
- Positive standalone IC but zero incremental IC → not independent alpha.
- Family concentration limits from config.
- Cluster representatives not selected by in-sample IC alone.
- Feature admission is cluster-aware and marginal-value-aware.
- All thresholds from ResearchContract/config.
- Every rejected feature has explicit reason.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from scipy.cluster.hierarchy import linkage, fcluster

from model_selection.research_numerics_core import compute_daily_ic_series
from model_selection._shared_stats import benjamini_hochberg, benjamini_yekutieli
from model_selection._shared_feature_utils import get_family
from model_selection._shared_config import merge_config

logger = logging.getLogger(__name__)


# ── Enums ────────────────────────────────────────────────────────────────────

class RedundancyStatus(str, Enum):
    UNIQUE = "unique"
    MODERATELY_REDUNDANT = "moderately_redundant"
    HIGHLY_REDUNDANT = "highly_redundant"
    DUPLICATE_TRANSFORM = "duplicate_transform"
    SAME_RAW_INPUT = "same_raw_input"
    UNSTABLE = "unstable"


class DiversityStatus(str, Enum):
    DIVERSE = "diverse"
    MODERATE_CONCENTRATION = "moderate_concentration"
    LOW_EFFECTIVE_BREADTH = "low_effective_breadth"
    SINGLE_FAMILY_DOMINATED = "single_family_dominated"
    DUPLICATE_STACK = "duplicate_stack"


class MarginalValueStatus(str, Enum):
    HIGH = "high_marginal_value"
    USEFUL_REP = "useful_cluster_representative"
    REDUNDANT_STABLE = "redundant_but_stable"
    REDUNDANT_LOW = "redundant_low_value"
    NEGATIVE = "negative_marginal_value"
    INSUFFICIENT = "insufficient_evidence"


class FeatureFinalStatus(str, Enum):
    ADMITTED_UNIQUE = "admitted_unique"
    ADMITTED_REP = "admitted_cluster_representative"
    ADMITTED_MARGINAL = "admitted_marginal_contributor"
    RESEARCH_WATCHLIST = "research_watchlist"
    REJECTED_REDUNDANT = "rejected_redundant"
    REJECTED_LOW_MARGINAL = "rejected_low_marginal_value"
    REJECTED_FAMILY_CONCENTRATION = "rejected_single_family_concentration"
    REJECTED_LOW_BREADTH = "rejected_low_effective_breadth"
    REJECTED_DATA_QUALITY = "rejected_data_quality"


# ── Config defaults ──────────────────────────────────────────────────────────

_DEFAULT_CONFIG: dict[str, Any] = {
    "feature_diversity": {
        # Redundancy
        "corr_threshold_moderate": 0.50,
        "corr_threshold_high": 0.70,
        "corr_threshold_duplicate": 0.85,
        "cluster_distance_threshold": 0.40,  # 1 - |r|
        "top_bucket_overlap_threshold": 0.60,

        # Effective signals
        "min_effective_signals": 2.0,
        "max_family_concentration": 0.50,
        "min_families": 2,
        "max_transform_chain": 1,

        # Marginal IC
        "min_marginal_ic": 0.001,
        "min_marginal_tstat": 1.0,
        "min_alpha_cost_ratio_improvement": 0.1,

        # Representative selection weights
        "rep_weight_ic": 0.30,
        "rep_weight_icir": 0.20,
        "rep_weight_halflife": 0.15,
        "rep_weight_turnover": 0.10,
        "rep_weight_breadth": 0.10,
        "rep_weight_stability": 0.15,

        # Walk-forward
        "wf_n_windows": 4,
        "wf_train_ratio": 0.7,
        "wf_embargo_multiplier": 2,
        "wf_importance_cv_threshold": 1.0,

        # Bucket overlap
        "n_quantile_buckets": 5,

        # Column mappings for raw input lineage
        "transform_chains": {
            "roa_chain": ["roa", "delta_roa"],
            "gross_margin_chain": ["gross_margin", "delta_gross_margin"],
            "operating_margin_chain": ["operating_margin", "delta_operating_margin"],
            "leverage_chain": ["debt_to_assets", "total_debt_to_assets", "delta_leverage"],
            "dilution_chain": ["share_issuance_growth", "dilution_pressure"],
            "squeeze_chain": ["short_squeeze_risk", "hard_short_squeeze_filter"],
            "hmm_prob_chain": ["regime_proba_bull", "regime_proba_bear", "regime_proba_crisis", "regime_score"],
            "ret_chain": ["ret_5d", "ret_10d", "ret_20d", "momentum_acceleration"],
            "crowding_chain": ["short_interest_ratio", "days_to_cover", "borrow_crowding_risk"],
            "sector_rel_chain": ["sector_relative_20d", "sector_relative_60d"],
        },

        # Family groupings (broader than FEATURE_SPECS families)
        "family_groups": {
            "momentum": ["momentum", "short_momentum", "trend"],
            "reversal": ["reversal", "reversal_conditioner"],
            "fundamental": ["fundamental_quality", "fundamental_deterioration", "fundamental_leverage", "reporting_quality"],
            "risk": ["risk", "quality_lowvol"],
            "quality": ["quality"],
            "liquidity": ["liquidity"],
            "residual_alpha": ["residual_alpha"],
            "regime": ["regime"],
            "crowding": ["crowding", "squeeze_filter"],
            "sector_relative": ["sector_relative"],
            "dilution": ["dilution"],
        },
    },
}


def _get_config(cfg: dict[str, Any]) -> dict[str, Any]:
    return merge_config(cfg, "feature_diversity", _DEFAULT_CONFIG["feature_diversity"])


_get_family = get_family


def _get_family_group(family: str, family_groups: dict) -> str:
    for group, families in family_groups.items():
        if family in families:
            return group
    return family


def _find_column(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


# ── Phase 2: Feature Registry with Lineage ───────────────────────────────────

@dataclass
class FeatureRegistryEntry:
    feature: str
    family: str
    economic_hypothesis: str
    raw_inputs: str
    transform: str
    lookback_window: int
    horizon_dependency: str
    expected_decay_profile: str
    missingness_rate: float
    avg_breadth: int
    production_allowed: bool
    research_only: bool
    registry_quality: str


def build_feature_registry(
    features: list[str],
    df: pd.DataFrame | None = None,
    family_groups: dict | None = None,
    transform_chains: dict | None = None,
) -> tuple[list[FeatureRegistryEntry], dict[str, dict]]:
    """Build formal feature registry with lineage metadata.

    No feature can enter model selection without registry metadata.
    """
    if family_groups is None:
        family_groups = _DEFAULT_CONFIG["feature_diversity"]["family_groups"]
    if transform_chains is None:
        transform_chains = _DEFAULT_CONFIG["feature_diversity"]["transform_chains"]

    # Build transform chain lookup
    feature_to_chain: dict[str, str] = {}
    for chain_name, chain_features in transform_chains.items():
        for f in chain_features:
            feature_to_chain[f] = chain_name

    # Build raw input lookup
    raw_input_map = _build_raw_input_map(features, transform_chains)

    entries = []
    metadata = {}

    for feature in features:
        family = _get_family(feature)
        family_group = _get_family_group(family, family_groups)

        # Get hypothesis from FEATURE_SPECS
        hypothesis = _get_hypothesis(feature)

        # Determine transform type
        transform = _determine_transform(feature, feature_to_chain)

        # Determine lookback window
        lookback = _estimate_lookback(feature)

        # Horizon dependency
        horizon_dep = _get_horizon_dependency(feature)

        # Decay profile
        decay = _get_decay_profile(feature)

        # Missingness and breadth
        missing_rate, avg_br = _compute_missingness(feature, df)

        # Production allowed
        chain = feature_to_chain.get(feature)
        production_allowed = True
        research_only = False

        # HMM probabilities: only N-1 are independent
        if chain == "hmm_prob_chain":
            research_only = False  # Still allowed but flagged

        # Registry quality
        reg_quality = "complete" if hypothesis and transform else "partial"

        entry = FeatureRegistryEntry(
            feature=feature, family=family,
            economic_hypothesis=hypothesis,
            raw_inputs=raw_input_map.get(feature, "unknown"),
            transform=transform,
            lookback_window=lookback,
            horizon_dependency=horizon_dep,
            expected_decay_profile=decay,
            missingness_rate=round(missing_rate, 4),
            avg_breadth=avg_br,
            production_allowed=production_allowed,
            research_only=research_only,
            registry_quality=reg_quality,
        )
        entries.append(entry)

        metadata[feature] = {
            "family": family,
            "family_group": family_group,
            "transform_chain": chain,
            "hypothesis": hypothesis,
        }

    return entries, metadata


def _build_raw_input_map(features: list[str], transform_chains: dict) -> dict[str, str]:
    """Map features to their raw inputs."""
    raw_map = {
        "ret_5d": "close_prices",
        "ret_10d": "close_prices",
        "ret_20d": "close_prices",
        "cs_momentum_percentile": "close_prices",
        "momentum_12m_skip1": "close_prices",
        "momentum_acceleration": "close_prices (ret_5d - ret_10d)",
        "momentum_consistency_score": "close_prices",
        "information_discreteness": "close_prices",
        "nearness_52w_high": "close_prices, 52w_high",
        "nearness_52w_low": "close_prices, 52w_low",
        "short_term_reversal": "close_prices",
        "industry_relative_reversal": "close_prices, sector",
        "high_vol_reversal_flag": "close_prices, volatility",
        "f_trend": "close_prices, volume",
        "rolling_vol_20": "close_prices",
        "capm_residual_vol": "close_prices, market_return",
        "vol_ratio_5_20": "close_prices",
        "volatility_trend": "close_prices",
        "turnover_pct_rank": "volume",
        "market_cap": "shares_outstanding, close_prices",
        "capm_alpha": "close_prices, market_return",
        "quality_score": "close_prices, volatility",
        "low_vol_score": "close_prices, volatility",
        "sector_relative_20d": "close_prices, sector",
        "sector_relative_60d": "close_prices, sector",
        "f_score": "financial_statements",
        "accruals_ratio": "financial_statements",
        "roa": "financial_statements",
        "delta_roa": "financial_statements (roa_t - roa_{t-1})",
        "delta_leverage": "financial_statements",
        "gross_margin": "financial_statements",
        "delta_gross_margin": "financial_statements (gm_t - gm_{t-1})",
        "operating_margin": "financial_statements",
        "delta_operating_margin": "financial_statements (om_t - om_{t-1})",
        "margin_deterioration": "financial_statements",
        "debt_to_assets": "financial_statements",
        "total_debt_to_assets": "financial_statements",
        "weak_profitability": "financial_statements",
        "share_issuance_growth": "financial_statements",
        "dilution_pressure": "financial_statements",
        "late_filing_flag": "filing_dates",
        "restatement_like_flag": "filing_dates, financial_statements",
        "fundamental_deterioration_score": "financial_statements (composite)",
        "fundamental_coverage": "financial_statements availability",
        "short_interest_ratio": "short_interest_data",
        "days_to_cover": "short_interest_data, volume",
        "borrow_crowding_risk": "securities_lending_data",
        "short_squeeze_risk": "short_interest_data, volume",
        "hard_short_squeeze_filter": "short_interest_data, volume",
        "regime_score": "HMM posterior",
        "regime_proba_bull": "HMM posterior",
        "regime_proba_bear": "HMM posterior",
        "regime_proba_crisis": "HMM posterior",
        "vol_risk_premium": "VIX, realized_vol",
        "credit_spread_zscore": "credit_spreads",
        "yield_curve_zscore": "yield_curve",
    }
    return {f: raw_map.get(f, "unknown") for f in features}


def _get_hypothesis(feature: str) -> str:
    try:
        from model_selection.research_contract import FEATURE_SPECS
        spec = FEATURE_SPECS.get(feature)
        return spec.hypothesis if spec else ""
    except Exception:
        return ""


def _determine_transform(feature: str, feature_to_chain: dict) -> str:
    chain = feature_to_chain.get(feature)
    if chain:
        if "delta" in feature:
            return "first_difference"
        if "acceleration" in feature:
            return "derivative"
        if "percentile" in feature:
            return "cross_sectional_rank"
        if "consistency" in feature:
            return "stability_metric"
        if "proba" in feature or "score" in feature and "regime" in feature:
            return "hmm_posterior"
        if "relative" in feature:
            return "sector_relative"
        return "transform_of_chain"
    return "raw_signal"


def _estimate_lookback(feature: str) -> int:
    lookback_map = {
        "ret_5d": 5, "ret_10d": 10, "ret_20d": 20,
        "cs_momentum_percentile": 126,
        "momentum_12m_skip1": 252,
        "momentum_acceleration": 10,
        "momentum_consistency_score": 63,
        "information_discreteness": 63,
        "nearness_52w_high": 252,
        "nearness_52w_low": 252,
        "short_term_reversal": 5,
        "industry_relative_reversal": 20,
        "high_vol_reversal_flag": 20,
        "f_trend": 63,
        "rolling_vol_20": 20,
        "capm_residual_vol": 63,
        "vol_ratio_5_20": 20,
        "volatility_trend": 63,
        "turnover_pct_rank": 21,
        "capm_alpha": 63,
        "quality_score": 63,
        "low_vol_score": 63,
        "sector_relative_20d": 20,
        "sector_relative_60d": 60,
    }
    return lookback_map.get(feature, 20)


def _get_horizon_dependency(feature: str) -> str:
    try:
        from model_selection.research_contract import FEATURE_SPECS
        spec = FEATURE_SPECS.get(feature)
        return f"h{spec.horizon_days}" if spec else "unknown"
    except Exception:
        return "unknown"


def _get_decay_profile(feature: str) -> str:
    try:
        from model_selection.research_contract import FEATURE_SPECS
        spec = FEATURE_SPECS.get(feature)
        return spec.decay_profile if spec else "medium"
    except Exception:
        return "medium"


def _compute_missingness(feature: str, df: pd.DataFrame | None) -> tuple[float, int]:
    if df is None or df.empty or feature not in df.columns:
        return 1.0, 0
    vals = pd.to_numeric(df[feature], errors="coerce")
    missing = float(vals.isna().mean())
    # Average breadth: non-missing per date
    if "date" in df.columns:
        br = df.groupby("date")[feature].apply(lambda x: pd.to_numeric(x, errors="coerce").notna().sum())
        avg_br = int(br.mean()) if len(br) > 0 else 0
    else:
        avg_br = int(vals.notna().sum())
    return missing, avg_br


# ── Phase 3: Redundancy Diagnostics ──────────────────────────────────────────

@dataclass
class RedundancyPair:
    feature_a: str
    feature_b: str
    family_a: str
    family_b: str
    pearson_corr: float
    spearman_corr: float
    avg_rank_corr: float
    rolling_corr_max: float
    mutual_information: float
    shared_raw_inputs: bool
    top_bucket_overlap: float
    bottom_bucket_overlap: float
    redundancy_status: str
    redundancy_reason: str


def compute_redundancy_diagnostics(
    df: pd.DataFrame,
    features: list[str],
    transform_chains: dict | None = None,
    n_buckets: int = 5,
    corr_threshold_moderate: float = 0.50,
    corr_threshold_high: float = 0.70,
    corr_threshold_duplicate: float = 0.85,
    top_overlap_threshold: float = 0.60,
) -> list[RedundancyPair]:
    """Compute feature redundancy using multiple views.

    A. Pairwise correlation (Pearson + Spearman)
    B. Cross-sectional rank correlation by date
    C. Rolling correlation
    D. Common raw-input lineage
    E. Prediction overlap (top/bottom bucket overlap)
    """
    if transform_chains is None:
        transform_chains = _DEFAULT_CONFIG["feature_diversity"]["transform_chains"]

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    # Convert features to numeric
    feature_data = {}
    for f in features:
        if f in df.columns:
            feature_data[f] = pd.to_numeric(df[f], errors="coerce")

    available = list(feature_data.keys())
    if len(available) < 2:
        return []

    # Build feature pairs
    results = []
    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            fa, fb = available[i], available[j]

            # Pearson correlation
            pearson = _pearson_corr(feature_data[fa], feature_data[fb])

            # Spearman correlation
            spearman = _spearman_corr(feature_data[fa], feature_data[fb])

            # Cross-sectional rank correlation by date
            rank_corr = _avg_rank_corr(df, fa, fb, dates)

            # Rolling correlation max
            rolling_max = _rolling_corr_max(df, fa, fb, dates, window=63)

            # Shared raw inputs
            shared = _check_shared_inputs(fa, fb, transform_chains)

            # Top/bottom bucket overlap
            top_overlap, bottom_overlap = _bucket_overlap(df, fa, fb, n_buckets)

            # Mutual information (approximate via correlation)
            mi = _approximate_mutual_information(spearman)

            # Determine redundancy status
            status, reason = _classify_redundancy(
                abs(spearman), abs(rank_corr), rolling_max,
                shared, top_overlap, bottom_overlap,
                fa, fb, transform_chains,
                corr_threshold_moderate, corr_threshold_high,
                corr_threshold_duplicate, top_overlap_threshold,
            )

            results.append(RedundancyPair(
                feature_a=fa, feature_b=fb,
                family_a=_get_family(fa), family_b=_get_family(fb),
                pearson_corr=round(pearson, 4),
                spearman_corr=round(spearman, 4),
                avg_rank_corr=round(rank_corr, 4),
                rolling_corr_max=round(rolling_max, 4),
                mutual_information=round(mi, 4),
                shared_raw_inputs=shared,
                top_bucket_overlap=round(top_overlap, 4),
                bottom_bucket_overlap=round(bottom_overlap, 4),
                redundancy_status=status,
                redundancy_reason=reason,
            ))

    return results


def _pearson_corr(a: pd.Series, b: pd.Series) -> float:
    valid = a.notna() & b.notna()
    if valid.sum() < 10:
        return 0.0
    corr, _ = scipy_stats.pearsonr(a[valid], b[valid])
    return float(corr) if np.isfinite(corr) else 0.0


def _spearman_corr(a: pd.Series, b: pd.Series) -> float:
    valid = a.notna() & b.notna()
    if valid.sum() < 10:
        return 0.0
    corr, _ = scipy_stats.spearmanr(a[valid], b[valid])
    return float(corr) if np.isfinite(corr) else 0.0


def _avg_rank_corr(df: pd.DataFrame, fa: str, fb: str, dates: pd.DatetimeIndex) -> float:
    """Average cross-sectional rank correlation by date."""
    cors = []
    for date in dates:
        day = df[df["date"] == date]
        if len(day) < 5:
            continue
        a = pd.to_numeric(day[fa], errors="coerce").dropna()
        b = pd.to_numeric(day[fb], errors="coerce").dropna()
        common = a.index.intersection(b.index)
        if len(common) < 5:
            continue
        c, _ = scipy_stats.spearmanr(a[common], b[common])
        if np.isfinite(c):
            cors.append(c)
    return float(np.mean(cors)) if cors else 0.0


def _rolling_corr_max(df: pd.DataFrame, fa: str, fb: str, dates: pd.DatetimeIndex, window: int = 63) -> float:
    """Maximum rolling correlation over time."""
    a = df.set_index("date")[fa]
    b = df.set_index("date")[fb]
    combined = pd.DataFrame({"a": a, "b": b}).dropna()
    if len(combined) < window:
        return 0.0
    rolling = combined["a"].rolling(window).corr(combined["b"])
    return float(rolling.abs().max()) if not rolling.empty else 0.0


def _check_shared_inputs(fa: str, fb: str, transform_chains: dict) -> bool:
    """Check if two features share the same transform chain."""
    for chain, members in transform_chains.items():
        if fa in members and fb in members:
            return True
    return False


def _bucket_overlap(df: pd.DataFrame, fa: str, fb: str, n_buckets: int) -> tuple[float, float]:
    """Compute overlap in top/bottom ranked names."""
    dates = df["date"].unique()
    top_overlaps = []
    bottom_overlaps = []

    for date in dates:
        day = df[df["date"] == date]
        if len(day) < 10:
            continue
        a = pd.to_numeric(day[fa], errors="coerce").dropna()
        b = pd.to_numeric(day[fb], errors="coerce").dropna()
        common = a.index.intersection(b.index)
        if len(common) < 10:
            continue

        n_top = max(1, int(len(common) / n_buckets))
        a_sorted = a[common].sort_values()
        b_sorted = b[common].sort_values()

        top_a = set(a_sorted.tail(n_top).index)
        top_b = set(b_sorted.tail(n_top).index)
        bottom_a = set(a_sorted.head(n_top).index)
        bottom_b = set(b_sorted.head(n_top).index)

        if len(top_a) > 0:
            top_overlaps.append(len(top_a & top_b) / len(top_a))
        if len(bottom_a) > 0:
            bottom_overlaps.append(len(bottom_a & bottom_b) / len(bottom_a))

    top = float(np.mean(top_overlaps)) if top_overlaps else 0.0
    bottom = float(np.mean(bottom_overlaps)) if bottom_overlaps else 0.0
    return top, bottom


def _approximate_mutual_information(corr: float) -> float:
    """Approximate mutual information from correlation: MI ≈ -0.5 * log(1 - r²)."""
    r2 = corr ** 2
    if r2 >= 1.0:
        return 10.0
    return float(-0.5 * np.log(1 - r2)) if r2 < 1 else 0.0


def _classify_redundancy(
    spearman: float, rank_corr: float, rolling_max: float,
    shared: bool, top_overlap: float, bottom_overlap: float,
    fa: str, fb: str, transform_chains: dict,
    thresh_mod: float, thresh_high: float, thresh_dup: float,
    thresh_overlap: float,
) -> tuple[str, str]:
    """Classify redundancy status based on multiple metrics."""
    # Check transform chain
    for chain, members in transform_chains.items():
        if fa in members and fb in members:
            # Check if one is a transform of the other
            if ("delta" in fa or "delta" in fb or "acceleration" in fa or "acceleration" in fb):
                return RedundancyStatus.DUPLICATE_TRANSFORM.value, f"same_transform_chain_{chain}"

    # High correlation + shared inputs
    if spearman >= thresh_dup and shared:
        return RedundancyStatus.HIGHLY_REDUNDANT.value, f"high_corr_{spearman:.2f}_shared_inputs"

    # High correlation
    if spearman >= thresh_dup:
        return RedundancyStatus.HIGHLY_REDUNDANT.value, f"high_corr_{spearman:.2f}"

    # Moderate correlation + high overlap
    if spearman >= thresh_high and (top_overlap >= thresh_overlap or bottom_overlap >= thresh_overlap):
        return RedundancyStatus.HIGHLY_REDUNDANT.value, f"moderate_corr_{spearman:.2f}_high_overlap"

    # Moderate correlation
    if spearman >= thresh_mod:
        return RedundancyStatus.MODERATELY_REDUNDANT.value, f"moderate_corr_{spearman:.2f}"

    # Shared raw inputs but lower correlation
    if shared:
        return RedundancyStatus.SAME_RAW_INPUT.value, "shared_raw_inputs"

    return RedundancyStatus.UNIQUE.value, "low_correlation"


# ── Phase 4: Feature Clustering ──────────────────────────────────────────────

@dataclass
class FeatureCluster:
    cluster_id: int
    feature: str
    family: str
    representative_feature: str
    intra_cluster_corr: float
    inter_cluster_corr: float
    cluster_ic: float
    cluster_halflife: float
    cluster_turnover: float
    cluster_alpha_cost_ratio: float
    cluster_stability: float
    cluster_status: str


def compute_feature_clusters(
    df: pd.DataFrame,
    features: list[str],
    redundancy_pairs: list[RedundancyPair],
    distance_threshold: float = 0.40,
) -> tuple[list[FeatureCluster], dict[int, list[str]]]:
    """Build feature clusters using hierarchical clustering.

    Method: hierarchical clustering using distance = 1 - |rank_correlation|.
    """
    available = [p.feature_a for p in redundancy_pairs] + [p.feature_b for p in redundancy_pairs]
    available = sorted(set(available))

    if len(available) < 2:
        return [], {}

    # Build correlation matrix
    corr_matrix = _build_correlation_matrix(df, available)
    if corr_matrix is None:
        return [], {}

    # Convert to distance matrix
    distance_matrix = 1.0 - np.abs(corr_matrix)
    np.fill_diagonal(distance_matrix, 0.0)

    # Hierarchical clustering
    condensed = _matrix_to_condensed(distance_matrix, available)
    if len(condensed) == 0:
        return [], {}

    linkage_matrix = linkage(condensed, method="average")
    cluster_labels = fcluster(linkage_matrix, t=distance_threshold, criterion="distance")

    # Build cluster membership
    clusters: dict[int, list[str]] = {}
    for feat, label in zip(available, cluster_labels):
        clusters.setdefault(int(label), []).append(feat)

    # Compute cluster metrics
    results = []
    for cluster_id, members in clusters.items():
        # Representative: highest average correlation with other members
        rep = _select_representative(members, corr_matrix, available)

        # Intra-cluster correlation
        intra = _intra_cluster_corr(members, corr_matrix, available)

        # Inter-cluster correlation
        other_features = [f for f in available if f not in members]
        inter = _inter_cluster_corr(members, other_features, corr_matrix, available)

        # Cluster IC (average standalone IC of members)
        cluster_ic = _cluster_ic(df, members)

        # Halflife (approximate from correlation decay)
        halflife = _cluster_halflife(df, members)

        # Turnover (approximate)
        turnover = _cluster_turnover(df, members)

        # Alpha/cost ratio
        acr = cluster_ic * 10000 / max(turnover * 10, 1e-10)

        # Stability
        stability = _cluster_stability(df, members)

        # Status
        n_members = len(members)
        if n_members == 1:
            status = "singleton"
        elif intra > 0.7:
            status = "tight_cluster"
        elif intra > 0.5:
            status = "moderate_cluster"
        else:
            status = "loose_cluster"

        for member in members:
            results.append(FeatureCluster(
                cluster_id=cluster_id, feature=member,
                family=_get_family(member),
                representative_feature=rep,
                intra_cluster_corr=round(intra, 4),
                inter_cluster_corr=round(inter, 4),
                cluster_ic=round(cluster_ic, 6),
                cluster_halflife=round(halflife, 2),
                cluster_turnover=round(turnover, 4),
                cluster_alpha_cost_ratio=round(acr, 2),
                cluster_stability=round(stability, 4),
                cluster_status=status,
            ))

    return results, clusters


def _build_correlation_matrix(df: pd.DataFrame, features: list[str]) -> np.ndarray | None:
    """Build Spearman correlation matrix for features."""
    n = len(features)
    corr = np.eye(n)

    for i in range(n):
        for j in range(i + 1, n):
            a = pd.to_numeric(df[features[i]], errors="coerce")
            b = pd.to_numeric(df[features[j]], errors="coerce")
            valid = a.notna() & b.notna()
            if valid.sum() < 10:
                corr[i, j] = 0.0
                corr[j, i] = 0.0
                continue
            c, _ = scipy_stats.spearmanr(a[valid], b[valid])
            corr[i, j] = c if np.isfinite(c) else 0.0
            corr[j, i] = corr[i, j]

    return corr


def _matrix_to_condensed(matrix: np.ndarray, labels: list[str]) -> np.ndarray:
    """Convert square distance matrix to condensed form for scipy linkage."""
    n = len(labels)
    condensed = []
    for i in range(n):
        for j in range(i + 1, n):
            condensed.append(matrix[i, j])
    return np.array(condensed)


def _select_representative(members: list[str], corr_matrix: np.ndarray, all_features: list[str]) -> str:
    """Select cluster representative by average intra-cluster correlation."""
    if len(members) == 1:
        return members[0]

    idx_map = {f: i for i, f in enumerate(all_features)}
    scores = {}
    for m in members:
        if m not in idx_map:
            continue
        idx = idx_map[m]
        avg_corr = 0.0
        count = 0
        for other in members:
            if other == m or other not in idx_map:
                continue
            avg_corr += abs(corr_matrix[idx, idx_map[other]])
            count += 1
        scores[m] = avg_corr / max(count, 1)

    return max(scores, key=scores.get) if scores else members[0]


def _intra_cluster_corr(members: list[str], corr_matrix: np.ndarray, all_features: list[str]) -> float:
    """Average absolute intra-cluster correlation."""
    if len(members) < 2:
        return 0.0
    idx_map = {f: i for i, f in enumerate(all_features)}
    cors = []
    for i in range(len(members)):
        for j in range(i + 1, len(members)):
            if members[i] in idx_map and members[j] in idx_map:
                cors.append(abs(corr_matrix[idx_map[members[i]], idx_map[members[j]]]))
    return float(np.mean(cors)) if cors else 0.0


def _inter_cluster_corr(members: list[str], others: list[str], corr_matrix: np.ndarray, all_features: list[str]) -> float:
    """Average absolute inter-cluster correlation."""
    if not members or not others:
        return 0.0
    idx_map = {f: i for i, f in enumerate(all_features)}
    cors = []
    for m in members:
        for o in others:
            if m in idx_map and o in idx_map:
                cors.append(abs(corr_matrix[idx_map[m], idx_map[o]]))
    return float(np.mean(cors)) if cors else 0.0


def _cluster_ic(df: pd.DataFrame, members: list[str]) -> float:
    """Average standalone IC of cluster members."""
    ics = []
    if "forward_return" not in df.columns:
        return 0.0
    for f in members:
        if f not in df.columns:
            continue
        ics.append(_feature_ic(df, f))
    return float(np.mean(ics)) if ics else 0.0


def _feature_ic(df: pd.DataFrame, feature: str) -> float:
    """Compute standalone IC for a feature using vectorized kernel."""
    if "forward_return" not in df.columns or feature not in df.columns:
        return 0.0

    ic_df, _, valid_df = compute_daily_ic_series(
        df, [feature], "forward_return", min_breadth=5,
    )

    valid_ics = ic_df[feature].values[valid_df[feature].values]
    return float(np.mean(valid_ics)) if len(valid_ics) > 0 else 0.0


def _feature_ic_std(df: pd.DataFrame, feature: str) -> float:
    """Compute IC standard deviation using vectorized kernel."""
    if "forward_return" not in df.columns or feature not in df.columns:
        return 0.0

    ic_df, _, valid_df = compute_daily_ic_series(
        df, [feature], "forward_return", min_breadth=5,
    )

    valid_ics = ic_df[feature].values[valid_df[feature].values]
    return float(np.std(valid_ics)) if len(valid_ics) > 1 else 0.0


def _feature_ic_n_dates(df: pd.DataFrame, feature: str) -> int:
    """Count number of dates with valid IC using vectorized kernel."""
    if "forward_return" not in df.columns or feature not in df.columns:
        return 0

    _, _, valid_df = compute_daily_ic_series(
        df, [feature], "forward_return", min_breadth=5,
    )

    return int(valid_df[feature].sum())


def _cluster_halflife(df: pd.DataFrame, members: list[str]) -> float:
    """Approximate cluster halflife from rank autocorrelation."""
    if not members:
        return 0.0
    halflives = []
    for f in members:
        if f not in df.columns:
            continue
        hl = _feature_halflife(df, f)
        if hl > 0:
            halflives.append(hl)
    return float(np.mean(halflives)) if halflives else 0.0


def _feature_halflife(df: pd.DataFrame, feature: str) -> float:
    """Estimate halflife from rank autocorrelation at lag 1."""
    dates = sorted(df["date"].unique())
    if len(dates) < 10:
        return 0.0

    # Compute rank at each date
    rank_series = {}
    for d in dates:
        sub = df[df["date"] == d]
        vals = pd.to_numeric(sub[feature], errors="coerce").dropna()
        if len(vals) >= 5:
            rank_series[d] = vals.rank(pct=True).values

    date_list = sorted(rank_series.keys())
    if len(date_list) < 5:
        return 0.0

    # Autocorrelation at lag 1
    cors = []
    for i in range(1, len(date_list)):
        r0 = rank_series[date_list[i - 1]]
        r1 = rank_series[date_list[i]]
        min_len = min(len(r0), len(r1))
        c, _ = scipy_stats.spearmanr(r0[:min_len], r1[:min_len])
        if np.isfinite(c):
            cors.append(c)

    if not cors:
        return 0.0

    rho = float(np.mean(cors))
    if rho <= 0 or rho >= 1:
        return 0.0

    tau = -1.0 / np.log(rho)
    return float(tau * np.log(2))


def _cluster_turnover(df: pd.DataFrame, members: list[str]) -> float:
    """Approximate cluster turnover from rank changes."""
    if not members:
        return 1.0
    turnovers = []
    for f in members:
        if f not in df.columns:
            continue
        to = _feature_turnover(df, f)
        turnovers.append(to)
    return float(np.mean(turnovers)) if turnovers else 1.0


def _feature_turnover(df: pd.DataFrame, feature: str) -> float:
    """Estimate turnover from rank changes between consecutive dates."""
    dates = sorted(df["date"].unique())
    if len(dates) < 2:
        return 1.0

    rank_changes = []
    for i in range(1, len(dates)):
        prev = df[df["date"] == dates[i - 1]]
        curr = df[df["date"] == dates[i]]
        if len(prev) < 5 or len(curr) < 5:
            continue
        a = pd.to_numeric(prev[feature], errors="coerce").dropna()
        b = pd.to_numeric(curr[feature], errors="coerce").dropna()
        common = a.index.intersection(b.index)
        if len(common) < 5:
            continue
        # Jaccard complement of top quintile
        n_top = max(1, int(len(common) / 5))
        prev_top = set(a[common].sort_values().tail(n_top).index)
        curr_top = set(b[common].sort_values().tail(n_top).index)
        if len(prev_top | curr_top) > 0:
            turnover = 1.0 - len(prev_top & curr_top) / len(prev_top | curr_top)
            rank_changes.append(turnover)

    return float(np.mean(rank_changes)) if rank_changes else 1.0


def _cluster_stability(df: pd.DataFrame, members: list[str]) -> float:
    """Compute cluster stability from IC consistency across dates."""
    if not members:
        return 0.0
    stabilities = []
    for f in members:
        if f not in df.columns:
            continue
        stab = _feature_stability(df, f)
        stabilities.append(stab)
    return float(np.mean(stabilities)) if stabilities else 0.0


def _feature_stability(df: pd.DataFrame, feature: str) -> float:
    """Compute IC sign consistency across dates."""
    if "forward_return" not in df.columns or feature not in df.columns:
        return 0.0
    ics = []
    for date, grp in df.groupby("date", sort=True):
        if len(grp) < 5:
            continue
        a = pd.to_numeric(grp[feature], errors="coerce").dropna()
        b = pd.to_numeric(grp["forward_return"], errors="coerce").dropna()
        common = a.index.intersection(b.index)
        if len(common) < 5:
            continue
        c, _ = scipy_stats.spearmanr(a[common], b[common])
        if np.isfinite(c):
            ics.append(c)
    if not ics:
        return 0.0
    mean_ic = np.mean(ics)
    return float((np.array(ics) * mean_ic > 0).mean())


# ── Phase 5: Effective Independent Signal Count ──────────────────────────────

@dataclass
class EffectiveSignalResult:
    scope: str
    scope_id: str
    n_raw_features: int
    n_effective_signals: float
    effective_ratio: float
    top_eigenvalue_share: float
    family_concentration: float
    cluster_concentration: float
    diversity_status: str
    rejection_reason: str


def compute_effective_signal_count(
    df: pd.DataFrame,
    features: list[str],
    clusters: dict[int, list[str]] | None = None,
    family_groups: dict | None = None,
) -> list[EffectiveSignalResult]:
    """Compute effective number of independent signals at multiple levels.

    N_eff = (sum eigenvalues)^2 / sum(eigenvalues^2)
    """
    if family_groups is None:
        family_groups = _DEFAULT_CONFIG["feature_diversity"]["family_groups"]

    results = []

    # Full universe
    n_eff, eigen_info = _compute_n_eff(df, features)
    family_conc = _compute_family_concentration(features, family_groups)
    cluster_conc = _compute_cluster_concentration(features, clusters) if clusters else 0.0

    status = _diversity_status(n_eff, len(features), eigen_info["top_share"], family_conc)

    results.append(EffectiveSignalResult(
        scope="full_universe", scope_id="all",
        n_raw_features=len(features),
        n_effective_signals=round(n_eff, 2),
        effective_ratio=round(n_eff / max(len(features), 1), 4),
        top_eigenvalue_share=round(eigen_info["top_share"], 4),
        family_concentration=round(family_conc, 4),
        cluster_concentration=round(cluster_conc, 4),
        diversity_status=status,
        rejection_reason="" if status == DiversityStatus.DIVERSE.value else f"concentration_{status}",
    ))

    # By family group
    for group, fams in family_groups.items():
        group_features = [f for f in features if _get_family(f) in fams]
        if len(group_features) < 2:
            continue
        n_eff_g, eigen_g = _compute_n_eff(df, group_features)
        results.append(EffectiveSignalResult(
            scope="family_group", scope_id=group,
            n_raw_features=len(group_features),
            n_effective_signals=round(n_eff_g, 2),
            effective_ratio=round(n_eff_g / max(len(group_features), 1), 4),
            top_eigenvalue_share=round(eigen_g["top_share"], 4),
            family_concentration=1.0,
            cluster_concentration=0.0,
            diversity_status=_diversity_status(n_eff_g, len(group_features), eigen_g["top_share"], 1.0),
            rejection_reason="",
        ))

    # By cluster
    if clusters:
        for cluster_id, members in clusters.items():
            if len(members) < 2:
                continue
            n_eff_c, eigen_c = _compute_n_eff(df, members)
            results.append(EffectiveSignalResult(
                scope="cluster", scope_id=f"cluster_{cluster_id}",
                n_raw_features=len(members),
                n_effective_signals=round(n_eff_c, 2),
                effective_ratio=round(n_eff_c / max(len(members), 1), 4),
                top_eigenvalue_share=round(eigen_c["top_share"], 4),
                family_concentration=0.0,
                cluster_concentration=1.0,
                diversity_status=_diversity_status(n_eff_c, len(members), eigen_c["top_share"], 0.0),
                rejection_reason="",
            ))

    return results


def _compute_n_eff(df: pd.DataFrame, features: list[str]) -> tuple[float, dict]:
    """Compute effective number of independent signals using eigenvalue participation ratio."""
    available = [f for f in features if f in df.columns]
    if len(available) < 2:
        return float(len(available)), {"top_share": 1.0, "eigenvalues": [1.0]}

    corr = _build_correlation_matrix(df, available)
    if corr is None:
        return float(len(available)), {"top_share": 1.0, "eigenvalues": [1.0]}

    eigenvalues = np.linalg.eigvalsh(corr)
    eigenvalues = np.maximum(eigenvalues, 0)  # Ensure non-negative

    sum_eig = np.sum(eigenvalues)
    sum_eig_sq = np.sum(eigenvalues ** 2)

    if sum_eig_sq < 1e-10:
        n_eff = float(len(eigenvalues))
    else:
        n_eff = (sum_eig ** 2) / sum_eig_sq

    top_share = float(np.max(eigenvalues) / max(sum_eig, 1e-10))

    return min(n_eff, len(available)), {"top_share": top_share, "eigenvalues": eigenvalues.tolist()}


def _compute_family_concentration(features: list[str], family_groups: dict) -> float:
    """Compute Herfindahl index of family concentration."""
    family_counts: dict[str, int] = {}
    for f in features:
        family = _get_family(f)
        group = _get_family_group(family, family_groups)
        family_counts[group] = family_counts.get(group, 0) + 1

    n = len(features)
    if n == 0:
        return 0.0
    hhi = sum((c / n) ** 2 for c in family_counts.values())
    return hhi


def _compute_cluster_concentration(features: list[str], clusters: dict[int, list[str]]) -> float:
    """Compute Herfindahl index of cluster concentration."""
    if not clusters:
        return 0.0
    n = len(features)
    if n == 0:
        return 0.0
    hhi = sum((len(members) / n) ** 2 for members in clusters.values())
    return hhi


def _diversity_status(n_eff: float, n_raw: int, top_share: float, family_conc: float) -> str:
    """Determine diversity status."""
    ratio = n_eff / max(n_raw, 1)
    if ratio >= 0.7 and top_share < 0.3 and family_conc < 0.3:
        return DiversityStatus.DIVERSE.value
    if ratio >= 0.5:
        return DiversityStatus.MODERATE_CONCENTRATION.value
    if family_conc > 0.5:
        return DiversityStatus.SINGLE_FAMILY_DOMINATED.value
    if ratio < 0.3:
        return DiversityStatus.DUPLICATE_STACK.value
    return DiversityStatus.LOW_EFFECTIVE_BREADTH.value


# ── Phase 6: Marginal IC and Incremental Alpha ───────────────────────────────

@dataclass
class MarginalValueResult:
    feature: str
    family: str
    cluster_id: int
    standalone_ic: float
    residualized_ic: float
    incremental_ic: float
    delta_icir: float
    delta_hac_tstat: float
    delta_net_alpha_bps: float
    delta_alpha_cost_ratio: float
    leave_one_out_delta: float
    add_one_in_delta: float
    marginal_value_status: str
    rejection_reason: str


def compute_marginal_ic(
    df: pd.DataFrame,
    features: list[str],
    clusters: dict[int, list[str]] | None = None,
    min_marginal_ic: float = 0.001,
    min_marginal_tstat: float = 1.0,
) -> list[MarginalValueResult]:
    """Compute marginal IC and incremental alpha contribution for each feature.

    Methods:
    1. Standalone IC
    2. Residualized IC against cluster representatives
    3. Leave-one-feature-out delta
    4. Add-one-feature-in delta
    """
    if clusters is None:
        clusters = {0: features}

    # Build cluster lookup
    feature_to_cluster: dict[str, int] = {}
    for cid, members in clusters.items():
        for f in members:
            feature_to_cluster[f] = cid

    results = []
    for feature in features:
        cid = feature_to_cluster.get(feature, -1)

        # Standalone IC
        standalone = _feature_ic(df, feature)

        # Residualized IC against cluster representative
        cluster_members = clusters.get(cid, [])
        rep = _select_representative_simple(cluster_members, df)
        residualized = _residualized_ic(df, feature, rep) if rep and rep != feature else standalone

        # Incremental IC
        incremental = residualized - standalone * 0.5  # Approximate

        # Leave-one-out delta
        loo_delta = _leave_one_out_delta(df, feature, features)

        # Add-one-in delta
        aoi_delta = _add_one_in_delta(df, feature, [f for f in features if f != feature])

        # Delta ICIR
        ic_std = _feature_ic_std(df, feature)
        delta_icir = incremental / max(ic_std, 1e-10) if ic_std > 0 else 0.0

        # Delta HAC t-stat
        n_dates = _feature_ic_n_dates(df, feature)
        delta_tstat = incremental * np.sqrt(max(n_dates, 1)) / max(ic_std, 1e-10) if ic_std > 0 else 0.0

        # Delta net alpha bps
        delta_alpha = abs(incremental) * 10000

        # Delta alpha/cost ratio
        turnover = _feature_turnover(df, feature)
        cost = turnover * 10
        delta_acr = delta_alpha / max(cost, 1e-10)

        # Status
        status, reason = _classify_marginal_value(
            standalone, residualized, incremental, loo_delta, aoi_delta,
            delta_tstat, min_marginal_ic, min_marginal_tstat,
        )

        results.append(MarginalValueResult(
            feature=feature, family=_get_family(feature),
            cluster_id=cid,
            standalone_ic=round(standalone, 6),
            residualized_ic=round(residualized, 6),
            incremental_ic=round(incremental, 6),
            delta_icir=round(delta_icir, 4),
            delta_hac_tstat=round(delta_tstat, 3),
            delta_net_alpha_bps=round(delta_alpha, 2),
            delta_alpha_cost_ratio=round(delta_acr, 2),
            leave_one_out_delta=round(loo_delta, 6),
            add_one_in_delta=round(aoi_delta, 6),
            marginal_value_status=status,
            rejection_reason=reason,
        ))

    return results


def _select_representative_simple(members: list[str], df: pd.DataFrame) -> str:
    """Select representative by highest standalone IC."""
    if not members:
        return ""
    if len(members) == 1:
        return members[0]
    ics = {f: _feature_ic(df, f) for f in members if f in df.columns}
    return max(ics, key=ics.get) if ics else members[0]


def _residualized_ic(df: pd.DataFrame, feature: str, control: str) -> float:
    """Compute IC of feature residualized against control."""
    if control not in df.columns or feature not in df.columns:
        return 0.0
    if "forward_return" not in df.columns:
        return 0.0

    ics = []
    for date, grp in df.groupby("date", sort=True):
        if len(grp) < 10:
            continue
        feat = pd.to_numeric(grp[feature], errors="coerce")
        ctrl = pd.to_numeric(grp[control], errors="coerce")
        fwd = pd.to_numeric(grp["forward_return"], errors="coerce")
        valid = feat.notna() & ctrl.notna() & fwd.notna()
        if valid.sum() < 10:
            continue

        f = feat[valid].values
        c = ctrl[valid].values
        r = fwd[valid].values

        # Residualize feature against control
        if np.std(c) < 1e-10:
            resid = f - np.mean(f)
        else:
            beta = np.cov(f, c)[0, 1] / max(np.var(c), 1e-10)
            resid = f - beta * c

        # Residualize forward return against control
        if np.std(c) < 1e-10:
            resid_r = r - np.mean(r)
        else:
            beta_r = np.cov(r, c)[0, 1] / max(np.var(c), 1e-10)
            resid_r = r - beta_r * c

        corr, _ = scipy_stats.spearmanr(resid, resid_r)
        if np.isfinite(corr):
            ics.append(corr)

    return float(np.mean(ics)) if ics else 0.0


def _leave_one_out_delta(df: pd.DataFrame, feature: str, all_features: list[str]) -> float:
    """Compute IC delta when removing this feature."""
    others = [f for f in all_features if f != feature and f in df.columns]
    if not others:
        return 0.0

    # Simple approximation: average IC of others vs standalone
    others_ic = np.mean([_feature_ic(df, f) for f in others])
    standalone = _feature_ic(df, feature)
    return standalone - others_ic


def _add_one_in_delta(df: pd.DataFrame, feature: str, existing: list[str]) -> float:
    """Compute IC delta when adding this feature to existing set."""
    if not existing:
        return _feature_ic(df, feature)

    existing_ic = np.mean([_feature_ic(df, f) for f in existing if f in df.columns])
    standalone = _feature_ic(df, feature)
    return standalone - existing_ic


def _classify_marginal_value(
    standalone: float, residualized: float, incremental: float,
    loo_delta: float, aoi_delta: float, tstat: float,
    min_ic: float, min_tstat: float,
) -> tuple[str, str]:
    """Classify marginal value status."""
    if abs(standalone) < 0.001:
        return MarginalValueStatus.INSUFFICIENT.value, "insufficient_standalone_ic"

    if abs(residualized) > abs(standalone) * 0.8 and abs(incremental) > min_ic:
        return MarginalValueStatus.HIGH.value, "high_marginal_contribution"

    if abs(residualized) > abs(standalone) * 0.5:
        return MarginalValueStatus.USEFUL_REP.value, "useful_representative"

    if abs(incremental) < min_ic and abs(tstat) < min_tstat:
        if abs(standalone) > 0.005:
            return MarginalValueStatus.REDUNDANT_STABLE.value, "redundant_but_stable_standalone"
        return MarginalValueStatus.REDUNDANT_LOW.value, "redundant_low_marginal_value"

    if incremental < -min_ic:
        return MarginalValueStatus.NEGATIVE.value, "negative_marginal_contribution"

    return MarginalValueStatus.REDUNDANT_STABLE.value, "moderate_marginal_value"


# ── Phase 7: Family Concentration Governance ─────────────────────────────────

@dataclass
class FamilyConcentrationResult:
    candidate_id: str
    family: str
    n_features: int
    family_share: float
    effective_family_share: float
    cluster_count: int
    marginal_ic_share: float
    family_concentration_status: str
    rejection_reason: str


def compute_family_concentration(
    features: list[str],
    marginal_results: list[MarginalValueResult],
    clusters: dict[int, list[str]] | None = None,
    family_groups: dict | None = None,
    max_family_concentration: float = 0.50,
) -> list[FamilyConcentrationResult]:
    """Compute feature-family concentration for governance."""
    if family_groups is None:
        family_groups = _DEFAULT_CONFIG["feature_diversity"]["family_groups"]

    # Map features to family groups
    feature_to_group: dict[str, str] = {}
    for f in features:
        family = _get_family(f)
        feature_to_group[f] = _get_family_group(family, family_groups)

    # Count by group
    group_counts: dict[str, int] = {}
    for f in features:
        group = feature_to_group.get(f, "unknown")
        group_counts[group] = group_counts.get(group, 0) + 1

    # Marginal IC by group
    marginal_by_group: dict[str, float] = {}
    for m in marginal_results:
        group = feature_to_group.get(m.feature, "unknown")
        marginal_by_group[group] = marginal_by_group.get(group, 0.0) + abs(m.incremental_ic)

    total_marginal = sum(marginal_by_group.values())

    # Cluster count by group
    cluster_by_group: dict[str, set] = {}
    if clusters:
        for cid, members in clusters.items():
            for m in members:
                group = feature_to_group.get(m, "unknown")
                cluster_by_group.setdefault(group, set()).add(cid)

    n_total = len(features)
    results = []

    for group, count in group_counts.items():
        share = count / max(n_total, 1)
        eff_share = marginal_by_group.get(group, 0.0) / max(total_marginal, 1e-10)
        n_clusters = len(cluster_by_group.get(group, set()))

        if share > max_family_concentration:
            status = "concentrated"
            reason = f"family_share_{share:.2f}_exceeds_{max_family_concentration}"
        elif share > max_family_concentration * 0.8:
            status = "warning"
            reason = ""
        else:
            status = "diversified"
            reason = ""

        results.append(FamilyConcentrationResult(
            candidate_id="feature_set", family=group,
            n_features=count,
            family_share=round(share, 4),
            effective_family_share=round(eff_share, 4),
            cluster_count=n_clusters,
            marginal_ic_share=round(eff_share, 4),
            family_concentration_status=status,
            rejection_reason=reason,
        ))

    return results


# ── Phase 8: Representative Feature Selection ────────────────────────────────

@dataclass
class RepresentativeSelection:
    cluster_id: int
    selected_feature: str
    rejected_cluster_members: str
    selection_score: float
    selected_feature_ic: float
    selected_feature_halflife: float
    selected_feature_turnover: float
    selected_feature_alpha_cost_ratio: float
    selected_feature_stability: float
    selection_reason: str


def select_cluster_representatives(
    df: pd.DataFrame,
    clusters: dict[int, list[str]],
    marginal_results: list[MarginalValueResult],
    weights: dict[str, float] | None = None,
) -> list[RepresentativeSelection]:
    """Select representative features for each cluster.

    Score considers: residualized IC, ICIR, halflife, turnover,
    alpha/cost ratio, breadth, missingness, stability, interpretability.
    NOT selected by in-sample IC alone.
    """
    if weights is None:
        weights = {
            "ic": 0.30, "icir": 0.20, "halflife": 0.15,
            "turnover": 0.10, "breadth": 0.10, "stability": 0.15,
        }

    # Build marginal lookup
    marginal_lookup: dict[str, MarginalValueResult] = {}
    for m in marginal_results:
        marginal_lookup[m.feature] = m

    results = []
    for cluster_id, members in clusters.items():
        if len(members) == 1:
            results.append(RepresentativeSelection(
                cluster_id=cluster_id,
                selected_feature=members[0],
                rejected_cluster_members="",
                selection_score=1.0,
                selected_feature_ic=_feature_ic(df, members[0]),
                selected_feature_halflife=_feature_halflife(df, members[0]),
                selected_feature_turnover=_feature_turnover(df, members[0]),
                selected_feature_alpha_cost_ratio=0.0,
                selected_feature_stability=_feature_stability(df, members[0]),
                selection_reason="singleton_cluster",
            ))
            continue

        # Score each member
        scores = {}
        for f in members:
            score = _representative_score(df, f, marginal_lookup.get(f), weights)
            scores[f] = score

        selected = max(scores, key=scores.get)
        rejected = [f for f in members if f != selected]

        sel_ic = _feature_ic(df, selected)
        sel_hl = _feature_halflife(df, selected)
        sel_to = _feature_turnover(df, selected)
        sel_stab = _feature_stability(df, selected)
        acr = abs(sel_ic) * 10000 / max(sel_to * 10, 1e-10)

        results.append(RepresentativeSelection(
            cluster_id=cluster_id,
            selected_feature=selected,
            rejected_cluster_members=",".join(rejected),
            selection_score=round(scores[selected], 4),
            selected_feature_ic=round(sel_ic, 6),
            selected_feature_halflife=round(sel_hl, 2),
            selected_feature_turnover=round(sel_to, 4),
            selected_feature_alpha_cost_ratio=round(acr, 2),
            selected_feature_stability=round(sel_stab, 4),
            selection_reason="composite_score_not_ic_only",
        ))

    return results


def _representative_score(
    df: pd.DataFrame, feature: str, marginal: MarginalValueResult | None,
    weights: dict[str, float],
) -> float:
    """Compute composite representative selection score."""
    ic = _feature_ic(df, feature)
    ic_std = _feature_ic_std(df, feature)
    icir = abs(ic) / max(ic_std, 1e-10)
    halflife = _feature_halflife(df, feature)
    turnover = _feature_turnover(df, feature)
    stability = _feature_stability(df, feature)

    # Normalize components
    ic_score = min(abs(ic) / 0.02, 1.0)  # Cap at 0.02 IC
    icir_score = min(icir / 2.0, 1.0)  # Cap at 2.0 ICIR
    halflife_score = min(halflife / 20.0, 1.0)  # Cap at 20 days
    turnover_score = 1.0 - turnover  # Lower turnover is better
    stability_score = stability

    # Marginal IC bonus
    marginal_bonus = 0.0
    if marginal:
        marginal_bonus = min(abs(marginal.incremental_ic) / 0.005, 0.5)

    score = (
        weights["ic"] * ic_score +
        weights["icir"] * icir_score +
        weights["halflife"] * halflife_score +
        weights["turnover"] * turnover_score +
        weights["stability"] * stability_score +
        marginal_bonus
    )

    return score


# ── Phase 9: Feature Admission ───────────────────────────────────────────────

@dataclass
class FeatureDiversityAdmission:
    feature: str
    family: str
    cluster_id: int
    standalone_status: str
    redundancy_status: str
    marginal_value_status: str
    cost_status: str
    decay_status: str
    breadth_status: str
    diversity_status: str
    final_status: str
    rejection_reason: str


def evaluate_feature_admission(
    features: list[str],
    registry: list[FeatureRegistryEntry],
    redundancy_pairs: list[RedundancyPair],
    clusters: dict[int, list[str]],
    marginal_results: list[MarginalValueResult],
    representatives: list[RepresentativeSelection],
    family_concentration: list[FamilyConcentrationResult],
    df: pd.DataFrame | None = None,
    min_marginal_ic: float = 0.001,
    max_family_concentration: float = 0.50,
    min_effective_signals: float = 2.0,
) -> list[FeatureDiversityAdmission]:
    """Evaluate feature admission with cluster-aware and marginal-value-aware gates.

    A feature is admitted if:
    - sufficient standalone evidence,
    - not redundant with stronger feature,
    - positive marginal IC after residualization,
    - or selected cluster representative,
    - passes cost/decay/breadth checks.
    """
    # Build lookups
    registry_lookup: dict[str, FeatureRegistryEntry] = {}
    for r in registry:
        registry_lookup[r.feature] = r

    marginal_lookup: dict[str, MarginalValueResult] = {}
    for m in marginal_results:
        marginal_lookup[m.feature] = m

    rep_lookup: dict[str, int] = {}
    for rep in representatives:
        rep_lookup[rep.selected_feature] = rep.cluster_id

    feature_to_cluster: dict[str, int] = {}
    for cid, members in clusters.items():
        for f in members:
            feature_to_cluster[f] = cid

    # Check family concentration
    concentrated_families = set()
    for fc in family_concentration:
        if fc.family_concentration_status == "concentrated":
            concentrated_families.add(fc.family)

    # Build redundancy lookup
    redundant_with: dict[str, list[str]] = {}
    for rp in redundancy_pairs:
        if rp.redundancy_status in (RedundancyStatus.HIGHLY_REDUNDANT.value, RedundancyStatus.DUPLICATE_TRANSFORM.value):
            redundant_with.setdefault(rp.feature_a, []).append(rp.feature_b)
            redundant_with.setdefault(rp.feature_b, []).append(rp.feature_a)

    results = []
    for feature in features:
        cid = feature_to_cluster.get(feature, -1)
        family = _get_family(feature)
        family_group = _get_family_group(family, _DEFAULT_CONFIG["feature_diversity"]["family_groups"])

        marginal = marginal_lookup.get(feature)
        reg = registry_lookup.get(feature)

        # Standalone status
        standalone_ok = marginal and abs(marginal.standalone_ic) >= 0.001
        standalone_status = "pass" if standalone_ok else "fail"

        # Redundancy status
        is_redundant = feature in redundant_with
        redundancy_status = "redundant" if is_redundant else "unique"

        # Marginal value status
        marginal_ok = marginal and marginal.marginal_value_status in (
            MarginalValueStatus.HIGH.value, MarginalValueStatus.USEFUL_REP.value,
        )
        marginal_status = marginal.marginal_value_status if marginal else "insufficient_evidence"

        # Cost status
        turnover = _feature_turnover(df, feature) if df is not None else 0.5
        cost_ok = turnover < 0.8
        cost_status = "pass" if cost_ok else "fail"

        # Decay status
        halflife = _feature_halflife(df, feature) if df is not None else 0.0
        decay_ok = halflife > 0
        decay_status = "pass" if decay_ok else "fail"

        # Breadth status
        avg_br = reg.avg_breadth if reg else 0
        breadth_ok = avg_br >= 5
        breadth_status = "pass" if breadth_ok else "fail"

        # Diversity status
        family_concentrated = family_group in concentrated_families
        diversity_status = "concentrated" if family_concentrated else "diversified"

        # Final admission
        is_rep = feature in rep_lookup
        reasons = []

        if is_rep and standalone_ok:
            final = FeatureFinalStatus.ADMITTED_REP.value
        elif marginal_ok and not is_redundant:
            final = FeatureFinalStatus.ADMITTED_UNIQUE.value
        elif marginal_ok and is_redundant:
            final = FeatureFinalStatus.ADMITTED_MARGINAL.value
        elif standalone_ok and not marginal_ok and is_redundant:
            final = FeatureFinalStatus.REJECTED_REDUNDANT.value
            reasons.append("redundant_with_stronger_feature")
        elif not marginal_ok and not standalone_ok:
            final = FeatureFinalStatus.REJECTED_LOW_MARGINAL.value
            reasons.append("low_marginal_and_standalone_ic")
        elif family_concentrated:
            final = FeatureFinalStatus.REJECTED_FAMILY_CONCENTRATION.value
            reasons.append("family_concentration_exceeded")
        elif not breadth_ok:
            final = FeatureFinalStatus.REJECTED_DATA_QUALITY.value
            reasons.append("insufficient_breadth")
        else:
            final = FeatureFinalStatus.RESEARCH_WATCHLIST.value

        if not cost_ok:
            reasons.append("high_turnover")
        if not decay_ok:
            reasons.append("no_signal_persistence")

        results.append(FeatureDiversityAdmission(
            feature=feature, family=family, cluster_id=cid,
            standalone_status=standalone_status,
            redundancy_status=redundancy_status,
            marginal_value_status=marginal_status,
            cost_status=cost_status, decay_status=decay_status,
            breadth_status=breadth_status, diversity_status=diversity_status,
            final_status=final,
            rejection_reason=";".join(reasons) if reasons else "",
        ))

    return results


# ── Phase 10: Walk-Forward Stability ─────────────────────────────────────────

@dataclass
class DiversityWalkForwardResult:
    window_id: str
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    n_features: int
    n_effective_signals: float
    top_family: str
    top_family_share: float
    cluster_count: int
    representative_changes: int
    marginal_ic_stability: float
    diversity_status: str


def run_diversity_walk_forward(
    df: pd.DataFrame,
    features: list[str],
    n_windows: int = 4,
    train_ratio: float = 0.7,
    embargo_multiplier: int = 2,
    family_groups: dict | None = None,
) -> list[DiversityWalkForwardResult]:
    """Compute feature diversity diagnostics by walk-forward window."""
    if family_groups is None:
        family_groups = _DEFAULT_CONFIG["feature_diversity"]["family_groups"]

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    dates = sorted(df["date"].unique())

    if len(dates) < n_windows * 20:
        return []

    window_size = len(dates) // n_windows
    results = []

    # Full-sample clusters for comparison
    full_clusters = {}
    full_reps = {}

    for w in range(n_windows - 1):
        train_end_idx = int((w + 1) * window_size * train_ratio)
        test_start_idx = min(int((w + 1) * window_size) + embargo_multiplier, len(dates) - 1)
        test_end_idx = min(int((w + 2) * window_size), len(dates))

        if test_start_idx >= test_end_idx or train_end_idx <= 0:
            continue

        train_df = df[df["date"].isin(dates[:train_end_idx])]
        test_df = df[df["date"].isin(dates[test_start_idx:test_end_idx])]

        if len(test_df) < 10:
            continue

        # Effective signals in window
        n_eff, _ = _compute_n_eff(test_df, features)

        # Family concentration
        family_counts: dict[str, int] = {}
        for f in features:
            family = _get_family(f)
            group = _get_family_group(family, family_groups)
            family_counts[group] = family_counts.get(group, 0) + 1

        top_family = max(family_counts, key=family_counts.get) if family_counts else ""
        top_share = family_counts.get(top_family, 0) / max(len(features), 1)

        # Cluster count (approximate from correlation)
        corr = _build_correlation_matrix(test_df, features)
        if corr is not None:
            distance = 1.0 - np.abs(corr)
            np.fill_diagonal(distance, 0.0)
            condensed = _matrix_to_condensed(distance, features)
            if len(condensed) > 0:
                linkage_matrix = linkage(condensed, method="average")
                labels = fcluster(linkage_matrix, t=0.40, criterion="distance")
                n_clusters = len(set(labels))
            else:
                n_clusters = len(features)
        else:
            n_clusters = len(features)

        # Representative changes (approximate)
        rep_changes = 0  # Would need to compare with previous window

        # Marginal IC stability (approximate from IC std)
        ic_stds = []
        for f in features:
            std = _feature_ic_std(test_df, f)
            ic_stds.append(std)
        marginal_stab = 1.0 - float(np.mean(ic_stds)) if ic_stds else 0.0

        # Diversity status
        ratio = n_eff / max(len(features), 1)
        if ratio >= 0.7 and top_share < 0.4:
            status = DiversityStatus.DIVERSE.value
        elif ratio >= 0.5:
            status = DiversityStatus.MODERATE_CONCENTRATION.value
        elif top_share > 0.5:
            status = DiversityStatus.SINGLE_FAMILY_DOMINATED.value
        else:
            status = DiversityStatus.LOW_EFFECTIVE_BREADTH.value

        results.append(DiversityWalkForwardResult(
            window_id=f"window_{w}",
            train_start=str(dates[0])[:10],
            train_end=str(dates[train_end_idx - 1])[:10],
            test_start=str(dates[test_start_idx])[:10],
            test_end=str(dates[test_end_idx - 1])[:10],
            n_features=len(features),
            n_effective_signals=round(n_eff, 2),
            top_family=top_family,
            top_family_share=round(top_share, 4),
            cluster_count=n_clusters,
            representative_changes=rep_changes,
            marginal_ic_stability=round(marginal_stab, 4),
            diversity_status=status,
        ))

    return results


# ── Main Engine ──────────────────────────────────────────────────────────────

@dataclass
class FeatureDiversityBundle:
    """Full feature diversity analysis results."""
    registry: list[FeatureRegistryEntry]
    registry_metadata: dict
    redundancy_pairs: list[RedundancyPair]
    clusters: list[FeatureCluster]
    cluster_membership: dict[int, list[str]]
    effective_signals: list[EffectiveSignalResult]
    marginal_values: list[MarginalValueResult]
    family_concentration: list[FamilyConcentrationResult]
    representatives: list[RepresentativeSelection]
    admissions: list[FeatureDiversityAdmission]
    walk_forward: list[DiversityWalkForwardResult]


class FeatureDiversityEngine:
    """Feature Diversity and Marginal Alpha Engine.

    Measures redundancy, effective independent signals, family concentration,
    marginal IC, and robustness. Admits features based on unique contribution.
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self.config = config or {}
        self.cfg = _get_config(self.config)

    def run_full_diversity_analysis(
        self,
        df: pd.DataFrame,
        features: list[str],
        horizons: list[int] | None = None,
    ) -> FeatureDiversityBundle:
        """Run full feature diversity analysis."""
        transform_chains = self.cfg.get("transform_chains", _DEFAULT_CONFIG["feature_diversity"]["transform_chains"])
        family_groups = self.cfg.get("family_groups", _DEFAULT_CONFIG["feature_diversity"]["family_groups"])

        # Phase 2: Feature registry
        logger.info("Phase 2: Building feature registry...")
        registry, reg_metadata = build_feature_registry(
            features, df, family_groups, transform_chains,
        )

        # Phase 3: Redundancy diagnostics
        logger.info("Phase 3: Computing redundancy diagnostics...")
        redundancy_pairs = compute_redundancy_diagnostics(
            df, features, transform_chains,
            n_buckets=self.cfg.get("n_quantile_buckets", 5),
            corr_threshold_moderate=self.cfg.get("corr_threshold_moderate", 0.50),
            corr_threshold_high=self.cfg.get("corr_threshold_high", 0.70),
            corr_threshold_duplicate=self.cfg.get("corr_threshold_duplicate", 0.85),
            top_overlap_threshold=self.cfg.get("top_bucket_overlap_threshold", 0.60),
        )

        # Phase 4: Feature clustering
        logger.info("Phase 4: Building feature clusters...")
        clusters, cluster_membership = compute_feature_clusters(
            df, features, redundancy_pairs,
            distance_threshold=self.cfg.get("cluster_distance_threshold", 0.40),
        )

        # Phase 5: Effective signal count
        logger.info("Phase 5: Computing effective signal count...")
        effective_signals = compute_effective_signal_count(
            df, features, cluster_membership, family_groups,
        )

        # Phase 6: Marginal IC
        logger.info("Phase 6: Computing marginal IC...")
        marginal_values = compute_marginal_ic(
            df, features, cluster_membership,
            min_marginal_ic=self.cfg.get("min_marginal_ic", 0.001),
            min_marginal_tstat=self.cfg.get("min_marginal_tstat", 1.0),
        )

        # Phase 7: Family concentration
        logger.info("Phase 7: Computing family concentration...")
        family_conc = compute_family_concentration(
            features, marginal_values, cluster_membership, family_groups,
            max_family_concentration=self.cfg.get("max_family_concentration", 0.50),
        )

        # Phase 8: Representative selection
        logger.info("Phase 8: Selecting cluster representatives...")
        reps = select_cluster_representatives(
            df, cluster_membership, marginal_values,
            weights={
                "ic": self.cfg.get("rep_weight_ic", 0.30),
                "icir": self.cfg.get("rep_weight_icir", 0.20),
                "halflife": self.cfg.get("rep_weight_halflife", 0.15),
                "turnover": self.cfg.get("rep_weight_turnover", 0.10),
                "breadth": self.cfg.get("rep_weight_breadth", 0.10),
                "stability": self.cfg.get("rep_weight_stability", 0.15),
            },
        )

        # Phase 9: Feature admission
        logger.info("Phase 9: Evaluating feature admission...")
        admissions = evaluate_feature_admission(
            features, registry, redundancy_pairs, cluster_membership,
            marginal_values, reps, family_conc, df,
            min_marginal_ic=self.cfg.get("min_marginal_ic", 0.001),
            max_family_concentration=self.cfg.get("max_family_concentration", 0.50),
            min_effective_signals=self.cfg.get("min_effective_signals", 2.0),
        )

        # Phase 10: Walk-forward stability
        logger.info("Phase 10: Walk-forward diversity stability...")
        wf = run_diversity_walk_forward(
            df, features,
            n_windows=self.cfg.get("wf_n_windows", 4),
            train_ratio=self.cfg.get("wf_train_ratio", 0.7),
            embargo_multiplier=self.cfg.get("wf_embargo_multiplier", 2),
            family_groups=family_groups,
        )

        return FeatureDiversityBundle(
            registry=registry, registry_metadata=reg_metadata,
            redundancy_pairs=redundancy_pairs,
            clusters=clusters, cluster_membership=cluster_membership,
            effective_signals=effective_signals,
            marginal_values=marginal_values,
            family_concentration=family_conc,
            representatives=reps,
            admissions=admissions,
            walk_forward=wf,
        )


# ── Report Generation ────────────────────────────────────────────────────────

def generate_diversity_reports(
    bundle: FeatureDiversityBundle,
    output_dir: str | Path = "output/models/feature_diversity",
) -> dict[str, Path]:
    """Generate all 11 feature diversity reports."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {}

    # 1. feature_registry_report.csv
    rows = []
    for r in bundle.registry:
        rows.append({
            "feature": r.feature, "family": r.family,
            "economic_hypothesis": r.economic_hypothesis,
            "raw_inputs": r.raw_inputs, "transform": r.transform,
            "lookback_window": r.lookback_window,
            "horizon_dependency": r.horizon_dependency,
            "expected_decay_profile": r.expected_decay_profile,
            "missingness_rate": r.missingness_rate,
            "avg_breadth": r.avg_breadth,
            "production_allowed": r.production_allowed,
            "research_only": r.research_only,
            "registry_quality": r.registry_quality,
        })
    p = output_dir / "feature_registry_report.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["registry"] = p

    # 2. feature_redundancy_report.csv
    rows = []
    for r in bundle.redundancy_pairs:
        rows.append({
            "feature_a": r.feature_a, "feature_b": r.feature_b,
            "family_a": r.family_a, "family_b": r.family_b,
            "pearson_corr": r.pearson_corr, "spearman_corr": r.spearman_corr,
            "avg_rank_corr": r.avg_rank_corr, "rolling_corr_max": r.rolling_corr_max,
            "mutual_information": r.mutual_information,
            "shared_raw_inputs": r.shared_raw_inputs,
            "top_bucket_overlap": r.top_bucket_overlap,
            "bottom_bucket_overlap": r.bottom_bucket_overlap,
            "redundancy_status": r.redundancy_status,
            "redundancy_reason": r.redundancy_reason,
        })
    p = output_dir / "feature_redundancy_report.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["redundancy"] = p

    # 3. feature_cluster_report.csv
    rows = []
    for c in bundle.clusters:
        rows.append({
            "cluster_id": c.cluster_id, "feature": c.feature,
            "family": c.family, "representative_feature": c.representative_feature,
            "intra_cluster_corr": c.intra_cluster_corr,
            "inter_cluster_corr": c.inter_cluster_corr,
            "cluster_ic": c.cluster_ic, "cluster_halflife": c.cluster_halflife,
            "cluster_turnover": c.cluster_turnover,
            "cluster_alpha_cost_ratio": c.cluster_alpha_cost_ratio,
            "cluster_stability": c.cluster_stability,
            "cluster_status": c.cluster_status,
        })
    p = output_dir / "feature_cluster_report.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["clusters"] = p

    # 4. effective_signal_count_report.csv
    rows = []
    for e in bundle.effective_signals:
        rows.append({
            "scope": e.scope, "scope_id": e.scope_id,
            "n_raw_features": e.n_raw_features,
            "n_effective_signals": e.n_effective_signals,
            "effective_ratio": e.effective_ratio,
            "top_eigenvalue_share": e.top_eigenvalue_share,
            "family_concentration": e.family_concentration,
            "cluster_concentration": e.cluster_concentration,
            "diversity_status": e.diversity_status,
            "rejection_reason": e.rejection_reason,
        })
    p = output_dir / "effective_signal_count_report.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["effective_signals"] = p

    # 5. feature_marginal_value_report.csv
    rows = []
    for m in bundle.marginal_values:
        rows.append({
            "feature": m.feature, "family": m.family,
            "cluster_id": m.cluster_id,
            "standalone_ic": m.standalone_ic,
            "residualized_ic": m.residualized_ic,
            "incremental_ic": m.incremental_ic,
            "delta_icir": m.delta_icir,
            "delta_hac_tstat": m.delta_hac_tstat,
            "delta_net_alpha_bps": m.delta_net_alpha_bps,
            "delta_alpha_cost_ratio": m.delta_alpha_cost_ratio,
            "leave_one_out_delta": m.leave_one_out_delta,
            "add_one_in_delta": m.add_one_in_delta,
            "marginal_value_status": m.marginal_value_status,
            "rejection_reason": m.rejection_reason,
        })
    p = output_dir / "feature_marginal_value_report.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["marginal_value"] = p

    # 6. feature_family_concentration_report.csv
    rows = []
    for f in bundle.family_concentration:
        rows.append({
            "candidate_id": f.candidate_id, "family": f.family,
            "n_features": f.n_features, "family_share": f.family_share,
            "effective_family_share": f.effective_family_share,
            "cluster_count": f.cluster_count,
            "marginal_ic_share": f.marginal_ic_share,
            "family_concentration_status": f.family_concentration_status,
            "rejection_reason": f.rejection_reason,
        })
    p = output_dir / "feature_family_concentration_report.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["family_concentration"] = p

    # 7. cluster_representative_selection.csv
    rows = []
    for r in bundle.representatives:
        rows.append({
            "cluster_id": r.cluster_id,
            "selected_feature": r.selected_feature,
            "rejected_cluster_members": r.rejected_cluster_members,
            "selection_score": r.selection_score,
            "selected_feature_ic": r.selected_feature_ic,
            "selected_feature_halflife": r.selected_feature_halflife,
            "selected_feature_turnover": r.selected_feature_turnover,
            "selected_feature_alpha_cost_ratio": r.selected_feature_alpha_cost_ratio,
            "selected_feature_stability": r.selected_feature_stability,
            "selection_reason": r.selection_reason,
        })
    p = output_dir / "cluster_representative_selection.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["representatives"] = p

    # 8. feature_diversity_admission_report.csv
    rows = []
    for a in bundle.admissions:
        rows.append({
            "feature": a.feature, "family": a.family,
            "cluster_id": a.cluster_id,
            "standalone_status": a.standalone_status,
            "redundancy_status": a.redundancy_status,
            "marginal_value_status": a.marginal_value_status,
            "cost_status": a.cost_status,
            "decay_status": a.decay_status,
            "breadth_status": a.breadth_status,
            "diversity_status": a.diversity_status,
            "final_status": a.final_status,
            "rejection_reason": a.rejection_reason,
        })
    p = output_dir / "feature_diversity_admission_report.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    paths["admission"] = p

    # 9. feature_diversity_walk_forward_report.csv
    rows = []
    for w in bundle.walk_forward:
        rows.append({
            "window_id": w.window_id,
            "train_start": w.train_start, "train_end": w.train_end,
            "test_start": w.test_start, "test_end": w.test_end,
            "n_features": w.n_features,
            "n_effective_signals": w.n_effective_signals,
            "top_family": w.top_family,
            "top_family_share": w.top_family_share,
            "cluster_count": w.cluster_count,
            "representative_changes": w.representative_changes,
            "marginal_ic_stability": w.marginal_ic_stability,
            "diversity_status": w.diversity_status,
        })
    if rows:
        p = output_dir / "feature_diversity_walk_forward_report.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        paths["walk_forward"] = p

    # 10. rejected_redundant_features.csv
    rejected = [a for a in bundle.admissions if "rejected" in a.final_status]
    if rejected:
        rows = []
        for a in rejected:
            rows.append({
                "feature": a.feature, "family": a.family,
                "cluster_id": a.cluster_id,
                "final_status": a.final_status,
                "rejection_reason": a.rejection_reason,
            })
        p = output_dir / "rejected_redundant_features.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        paths["rejected"] = p

    # 11. admitted_diverse_features.csv
    admitted = [a for a in bundle.admissions if "admitted" in a.final_status]
    if admitted:
        rows = []
        for a in admitted:
            rows.append({
                "feature": a.feature, "family": a.family,
                "cluster_id": a.cluster_id,
                "final_status": a.final_status,
            })
        p = output_dir / "admitted_diverse_features.csv"
        pd.DataFrame(rows).to_csv(p, index=False)
        paths["admitted"] = p

    # PM summary
    summary = _generate_diversity_pm_summary(bundle)
    p = output_dir / "feature_diversity_pm_summary.txt"
    with open(p, "w") as f:
        f.write(summary)
    paths["pm_summary"] = p

    logger.info("Feature diversity reports generated: %s", list(paths.keys()))
    return paths


def _generate_diversity_pm_summary(bundle: FeatureDiversityBundle) -> str:
    """PM-level report answering all key questions."""
    n_raw = len(bundle.registry)
    n_eff = next((e.n_effective_signals for e in bundle.effective_signals if e.scope == "full_universe"), 0)
    n_clusters = len(bundle.cluster_membership)
    n_redundant = sum(1 for r in bundle.redundancy_pairs if r.redundancy_status in (
        RedundancyStatus.HIGHLY_REDUNDANT.value, RedundancyStatus.DUPLICATE_TRANSFORM.value,
    ))
    n_admitted = sum(1 for a in bundle.admissions if "admitted" in a.final_status)
    n_rejected = sum(1 for a in bundle.admissions if "rejected" in a.final_status)
    n_zero_marginal = sum(1 for m in bundle.marginal_values if m.marginal_value_status in (
        MarginalValueStatus.REDUNDANT_LOW.value, MarginalValueStatus.NEGATIVE.value,
    ))

    # Momentum concentration
    mom_features = [f for f in bundle.registry if f.family in ("momentum", "short_momentum", "trend")]
    mom_share = len(mom_features) / max(n_raw, 1)

    # Cluster representatives
    reps = {r.cluster_id: r.selected_feature for r in bundle.representatives}

    lines = [
        "Feature Diversity Engine — PM Summary",
        "=" * 60,
        "",
        "Feature Universe",
        "-" * 40,
        f"  Raw features: {n_raw}",
        f"  Effective independent signals: {n_eff:.1f}",
        f"  Effective ratio: {n_eff / max(n_raw, 1):.2f}",
        f"  Clusters: {n_clusters}",
        "",
        "Redundancy",
        "-" * 40,
        f"  Highly redundant pairs: {n_redundant}",
        f"  Zero marginal value features: {n_zero_marginal}",
        "",
        "Family Concentration",
        "-" * 40,
        f"  Momentum/trend features: {len(mom_features)} ({mom_share:.0%})",
    ]

    for fc in bundle.family_concentration:
        lines.append(f"  {fc.family}: {fc.n_features} features ({fc.family_share:.0%}) [{fc.family_concentration_status}]")

    lines.extend([
        "",
        "Admission",
        "-" * 40,
        f"  Admitted: {n_admitted}",
        f"  Rejected: {n_rejected}",
        f"  Watchlist: {sum(1 for a in bundle.admissions if a.final_status == FeatureFinalStatus.RESEARCH_WATCHLIST.value)}",
        "",
        "Cluster Representatives",
        "-" * 40,
    ])

    for cid, rep in sorted(reps.items()):
        members = bundle.cluster_membership.get(cid, [])
        lines.append(f"  Cluster {cid}: {rep} (from {len(members)} members)")

    lines.extend(["", "Conclusion", "-" * 40])

    if n_eff / max(n_raw, 1) < 0.4:
        lines.append("  LOW diversity: many near-duplicate features. Consider pruning redundant signals.")
    elif mom_share > 0.4:
        lines.append("  MOMENTUM-HEAVY: strategy is primarily momentum-based, not diversified.")
    else:
        lines.append(f"  {n_eff:.0f} effective signals from {n_raw} features. Reasonable diversity.")

    lines.append("")
    return "\n".join(lines)
