"""
P21: Feature/Alpha Redesign Diagnostics — Read-only feature-family audit.

Reads existing artifacts (alpha_ic_decay.csv, feature_admission.csv,
horizon_alignment_report.csv) and FEATURE_SPECS to produce:
  - per-family IC decay analysis
  - sign stability audit
  - admission/inversion rates
  - contribution decomposition
  - research recommendations

All thresholds are configurable via FeatureAuditConfig.
This module does NOT change models, scores, gates, or promotion.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from model_selection.research_contract import FEATURE_SPECS, FeatureFamilyRegistry


# ── Configuration ────────────────────────────────────────────────────────────

@dataclass
class FeatureAuditConfig:
    """All configurable thresholds for the feature audit."""

    min_abs_ic_for_sign: float = 0.001   # IC below this is treated as ~zero
    max_sign_flips_for_stable: int = 1   # features with <= this many flips are "stable"
    min_halflife_for_slow: float = 15.0  # halflife above this is "slow" decay
    max_halflife_for_fast: float = 5.0   # halflife below this is "fast" decay
    min_admission_rate_for_promising: float = 0.30  # at least 30% admitted
    max_inversion_rate_for_stable: float = 0.50     # at most 50% inverted
    min_contrib_pct_for_significant: float = 5.0    # at least 5% contribution


# ── Dataclasses ──────────────────────────────────────────────────────────────

@dataclass
class FeatureSignStability:
    """Sign stability audit for a single feature across horizons."""
    feature: str
    family: str
    ic_signs: list[int]          # +1, -1, or 0 per horizon
    n_horizons: int
    n_sign_flips: int
    majority_sign: int            # +1, -1, or 0
    majority_fraction: float      # fraction with majority sign
    is_stable: bool
    horizons_with_sign: list[int]


@dataclass
class FeatureFamilySummary:
    """Aggregated diagnostics for one feature family."""
    name: str
    n_features: int
    features: list[str]

    # IC
    ic_by_horizon: dict[int, float]
    mean_abs_ic: float
    median_ic: float

    # Sign stability
    ic_sign_stability: float         # fraction of features that are "stable"
    n_sign_flips_mean: float

    # Admission
    n_admitted: int
    admission_rate: float
    n_inverted: int
    inversion_rate: float

    # Decay
    median_halflife: float
    dominant_decay_profile: str      # fast | medium | slow | unknown

    # Quality
    coverage_mean: float
    regime_stability_mean: float

    # Alignment
    horizon_alignment_rate: float    # fraction of features aligned with target

    # Marginal contribution
    marginal_ic_sum: float
    marginal_ic_mean: float
    ic_contribution_pct: float       # % of total absolute marginal IC

    # Classification
    bottlenecks: list[str]
    recommendation: str              # promising_stable | unstable_sign | etc.


@dataclass
class FeatureAuditReport:
    """Complete feature audit report."""
    config: dict[str, Any]
    families: list[FeatureFamilySummary]
    sign_stability: list[FeatureSignStability]
    summary: dict[str, Any]          # executive summary
    recommendations: list[str]

    def save(self, path: Path) -> None:
        """Write JSON report to disk."""
        data = {
            "config": self.config,
            "families": [asdict(f) for f in self.families],
            "summary": self.summary,
            "recommendations": self.recommendations,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")

    def save_markdown(self, path: Path) -> None:
        """Write human-readable markdown report."""
        lines = ["# P21: Feature/Alpha Redesign Diagnostics", ""]
        lines.append(f"**Horizon evaluated**: {self.summary.get('horizon', 'N/A')}")
        lines.append(f"**Total features**: {self.summary.get('n_features', 0)}")
        lines.append(f"**Families**: {self.summary.get('n_families', 0)}")
        lines.append("")

        lines.append("## Executive Summary")
        for r in self.recommendations:
            lines.append(f"- {r}")
        lines.append("")

        lines.append("## Family Rankings by Marginal IC Contribution")
        lines.append("| Family | Features | Admitted | Inv% | |IC| | Halflife | Decay | SignStab | Contrib% | Recommendation |")
        lines.append("|--------|----------|----------|------|------|----------|-------|----------|----------|----------------|")
        for f in sorted(self.families, key=lambda x: abs(x.ic_contribution_pct), reverse=True):
            lines.append(
                f"| {f.name:<20} | {f.n_features:>3} | {f.n_admitted:>3} | "
                f"{f.inversion_rate*100:>3.0f}% | {f.mean_abs_ic:>5.4f} | "
                f"{f.median_halflife:>5.1f}d | {f.dominant_decay_profile:<5} | "
                f"{f.ic_sign_stability:>5.2f} | {f.ic_contribution_pct:>5.1f}% | "
                f"{f.recommendation:<20} |"
            )
        lines.append("")

        lines.append("## Recommendations")
        for f in self.families:
            if f.bottlenecks:
                lines.append(f"### {f.name}")
                lines.append(f"**Recommendation**: {f.recommendation}")
                lines.append(f"**Bottlenecks**: {', '.join(f.bottlenecks)}")
                lines.append("")
                if f.ic_by_horizon:
                    lines.append("IC by horizon:")
                    for h, ic in sorted(f.ic_by_horizon.items()):
                        lines.append(f"  - {h}d: {ic:.5f}")
                    lines.append("")

        lines.append("## Sign Stability Audit")
        lines.append("| Feature | Family | Flips | Stable | Majority |")
        lines.append("|---------|--------|-------|--------|----------|")
        for s in sorted(self.sign_stability, key=lambda x: x.n_sign_flips, reverse=True):
            lines.append(
                f"| {s.feature:<25} | {s.family:<15} | {s.n_sign_flips} | "
                f"{'Yes' if s.is_stable else 'No'} | {'+' if s.majority_sign > 0 else ('-' if s.majority_sign < 0 else '~')} |"
            )
        lines.append("")

        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines), encoding="utf-8")


# ── Artifact loaders ─────────────────────────────────────────────────────────

def _safe_read_csv(path: Path | str, name: str, required_cols: list[str]) -> pd.DataFrame:
    """Read CSV with graceful degradation on missing files or columns."""
    if path is None or not Path(path).exists():
        print(f"[FeatureAudit] {name} not found at {path} — skipping")
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        print(f"[FeatureAudit] Failed to read {name}: {exc} — skipping")
        return pd.DataFrame()
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[FeatureAudit] {name} missing required columns: {missing} — degrading")
    return df


def load_alpha_ic_decay(path: Path | str) -> pd.DataFrame:
    return _safe_read_csv(path, "alpha_ic_decay", [
        "feature", "family", "horizon_days", "target_type", "daily_spearman_ic",
        "daily_spearman_ic_tstat", "signal_halflife_days",
    ])


def load_feature_admission(path: Path | str) -> pd.DataFrame:
    return _safe_read_csv(path, "feature_admission", [
        "feature", "family", "admitted", "transform_sign", "recommended_action",
    ])


# ── Sign stability ───────────────────────────────────────────────────────────

def compute_sign_stability(
    decay: pd.DataFrame,
    registry: FeatureFamilyRegistry,
    cfg: FeatureAuditConfig,
) -> list[FeatureSignStability]:
    """Compute sign stability for every feature across all available horizons."""
    if decay.empty or "horizon_days" not in decay.columns:
        return []

    results: list[FeatureSignStability] = []
    required = {"feature", "horizon_days", "daily_spearman_ic"}
    if not required.issubset(decay.columns):
        return []

    for feature in sorted(decay["feature"].unique()):
        fd = decay[decay["feature"] == feature].sort_values("horizon_days")
        if fd.empty:
            continue

        ics = pd.to_numeric(fd["daily_spearman_ic"], errors="coerce")
        signs: list[int] = []
        horizons_seen: list[int] = []
        for _, row in fd.iterrows():
            ic = row.get("daily_spearman_ic", np.nan)
            h = int(row.get("horizon_days", 0))
            if not np.isfinite(ic) or abs(ic) < cfg.min_abs_ic_for_sign:
                signs.append(0)
            else:
                signs.append(1 if ic > 0 else -1)
            horizons_seen.append(h)

        # Count sign flips (between consecutive non-zero signs)
        nonzero_signs = [s for s in signs if s != 0]
        n_flips = sum(
            1 for i in range(1, len(nonzero_signs))
            if nonzero_signs[i] != nonzero_signs[i - 1]
        )

        # Majority sign
        pos = sum(1 for s in nonzero_signs if s > 0)
        neg = sum(1 for s in nonzero_signs if s < 0)
        total = max(len(nonzero_signs), 1)
        majority = 1 if pos > neg else (-1 if neg > pos else 0)
        majority_frac = max(pos, neg) / total

        family = registry.family_of(feature)
        results.append(FeatureSignStability(
            feature=feature,
            family=family,
            ic_signs=signs,
            n_horizons=len(horizons_seen),
            n_sign_flips=n_flips,
            majority_sign=majority,
            majority_fraction=round(majority_frac, 3),
            is_stable=n_flips <= cfg.max_sign_flips_for_stable,
            horizons_with_sign=horizons_seen,
        ))

    return results


# ── Family aggregation ───────────────────────────────────────────────────────

def compute_family_summaries(
    registry: FeatureFamilyRegistry,
    decay: pd.DataFrame,
    admission: pd.DataFrame,
    horizon_alignment: dict[str, object] | None,
    sign_stability: list[FeatureSignStability],
    target_horizon: int,
    cfg: FeatureAuditConfig,
) -> list[FeatureFamilySummary]:
    """Aggregate per-feature diagnostics into per-family summaries."""
    summaries: list[FeatureFamilySummary] = []

    # Build admission lookup
    adm_cols = {"feature", "admitted", "transform_sign"}
    adm_lookup: dict[str, dict[str, Any]] = {}
    if not admission.empty and adm_cols.issubset(admission.columns):
        for _, row in admission.iterrows():
            adm_lookup[str(row["feature"])] = {
                "admitted": bool(row.get("admitted", False)),
                "inverted": int(row.get("transform_sign", 1)) < 0,
                "marginal_ic": float(row.get("marginal_ic", np.nan) or np.nan),
            }

    # Build sign stability lookup
    sign_lookup: dict[str, FeatureSignStability] = {
        s.feature: s for s in sign_stability
    }

    # Build alignment lookup
    aligned_set: set[str] = set()
    if horizon_alignment:
        aligned_set = set(str(a) for a in (horizon_alignment.get("aligned", []) or []))

    total_marginal_abs = 0.0
    for f in adm_lookup:
        mi = adm_lookup[f]["marginal_ic"]
        if np.isfinite(mi):
            total_marginal_abs += abs(mi)

    for family in registry.families:
        features = registry.features_by_family(family)
        n = len(features)

        # IC by horizon
        ic_by_horizon: dict[int, float] = {}
        if not decay.empty and "family" in decay.columns and "horizon_days" in decay.columns:
            fd = decay[decay["feature"].isin(features)]
            if not fd.empty:
                for h in sorted(fd["horizon_days"].dropna().unique()):
                    ics = pd.to_numeric(
                        fd[fd["horizon_days"] == int(h)]["daily_spearman_ic"],
                        errors="coerce",
                    ).dropna()
                    if len(ics) > 0:
                        ic_by_horizon[int(h)] = float(ics.mean())

        all_ics = []
        if not decay.empty and "daily_spearman_ic" in decay.columns:
            all_ics = pd.to_numeric(
                decay[decay["feature"].isin(features)]["daily_spearman_ic"],
                errors="coerce",
            ).dropna().tolist()

        mean_abs_ic = float(np.mean(np.abs(all_ics))) if all_ics else float("nan")
        median_ic = float(np.median(all_ics)) if all_ics else float("nan")

        # Sign stability
        family_signs = [s for s in sign_stability if s.family == family]
        n_stable = sum(1 for s in family_signs if s.is_stable)
        sign_stab = n_stable / max(len(family_signs), 1)
        n_flips_mean = float(np.mean([s.n_sign_flips for s in family_signs])) if family_signs else float("nan")

        # Admission
        admitted = sum(1 for f in features if adm_lookup.get(f, {}).get("admitted", False))
        inverted = sum(1 for f in features if adm_lookup.get(f, {}).get("inverted", False))
        admission_rate = admitted / n if n > 0 else 0.0
        inversion_rate = inverted / max(admitted, 1) if admitted > 0 else 0.0

        # Decay
        halflifes = []
        profiles: list[str] = []
        for f in features:
            spec = registry.spec_of(f)
            if spec:
                profiles.append(spec.decay_profile)
            if not decay.empty and "signal_halflife_days" in decay.columns:
                fd = decay[(decay["feature"] == f) & decay["signal_halflife_days"].notna()]
                if not fd.empty:
                    halflifes.append(float(fd["signal_halflife_days"].iloc[0]))

        median_halflife = float(np.median(halflifes)) if halflifes else float("nan")
        dominant_profile = max(set(profiles), key=profiles.count) if profiles else "unknown"

        # Quality
        coverages = []
        regimes = []
        if not decay.empty:
            for f in features:
                fd = decay[decay["feature"] == f]
                if "coverage" in fd.columns:
                    cv = pd.to_numeric(fd["coverage"], errors="coerce").mean()
                    if np.isfinite(cv):
                        coverages.append(float(cv))
                if "regime_positive_rate" in fd.columns:
                    rp = pd.to_numeric(fd["regime_positive_rate"], errors="coerce").mean()
                    if np.isfinite(rp):
                        regimes.append(float(rp))

        coverage_mean = float(np.mean(coverages)) if coverages else float("nan")
        regime_stability_mean = float(np.mean(regimes)) if regimes else float("nan")

        # Alignment
        n_aligned = len([f for f in features if f in aligned_set])
        alignment_rate = n_aligned / n if n > 0 else 0.0

        # Marginal contribution
        marg_ics = [
            adm_lookup[f]["marginal_ic"]
            for f in features
            if f in adm_lookup and np.isfinite(adm_lookup[f].get("marginal_ic", np.nan))
        ]
        marginal_sum = float(sum(marg_ics)) if marg_ics else float("nan")
        marginal_mean = float(np.mean(marg_ics)) if marg_ics else float("nan")
        marginal_abs = float(sum(abs(x) for x in marg_ics)) if marg_ics else 0.0
        contrib_pct = (marginal_abs / total_marginal_abs * 100) if total_marginal_abs > 0 else float("nan")

        # Bottlenecks
        bottlenecks: list[str] = []
        if admission_rate < cfg.min_admission_rate_for_promising:
            bottlenecks.append(f"low_admission ({admission_rate:.0%})")
        if inversion_rate > cfg.max_inversion_rate_for_stable:
            bottlenecks.append(f"high_inversion ({inversion_rate:.0%})")
        if sign_stab < 0.5:
            bottlenecks.append(f"unstable_signs ({sign_stab:.0%})")
        if np.isfinite(median_halflife) and median_halflife < cfg.max_halflife_for_fast:
            bottlenecks.append(f"fast_decay ({median_halflife:.1f}d)")
        if np.isfinite(contrib_pct) and contrib_pct < cfg.min_contrib_pct_for_significant:
            bottlenecks.append(f"low_contribution ({contrib_pct:.1f}%)")

        # Recommendation
        recommendation = _classify_family(
            admission_rate=admission_rate,
            inversion_rate=inversion_rate,
            sign_stab=sign_stab,
            median_halflife=median_halflife,
            contrib_pct=contrib_pct,
            cfg=cfg,
        )

        summaries.append(FeatureFamilySummary(
            name=family,
            n_features=n,
            features=features,
            ic_by_horizon=ic_by_horizon,
            mean_abs_ic=mean_abs_ic,
            median_ic=median_ic,
            ic_sign_stability=round(sign_stab, 4),
            n_sign_flips_mean=round(n_flips_mean, 2),
            n_admitted=admitted,
            admission_rate=round(admission_rate, 4),
            n_inverted=inverted,
            inversion_rate=round(inversion_rate, 4),
            median_halflife=round(median_halflife, 2) if np.isfinite(median_halflife) else float("nan"),
            dominant_decay_profile=dominant_profile,
            coverage_mean=round(coverage_mean, 4) if np.isfinite(coverage_mean) else float("nan"),
            regime_stability_mean=round(regime_stability_mean, 4) if np.isfinite(regime_stability_mean) else float("nan"),
            horizon_alignment_rate=round(alignment_rate, 4),
            marginal_ic_sum=round(marginal_sum, 6) if np.isfinite(marginal_sum) else float("nan"),
            marginal_ic_mean=round(marginal_mean, 6) if np.isfinite(marginal_mean) else float("nan"),
            ic_contribution_pct=round(contrib_pct, 2) if np.isfinite(contrib_pct) else float("nan"),
            bottlenecks=bottlenecks,
            recommendation=recommendation,
        ))

    return summaries


def _classify_family(
    admission_rate: float,
    inversion_rate: float,
    sign_stab: float,
    median_halflife: float,
    contrib_pct: float,
    cfg: FeatureAuditConfig,
) -> str:
    """Classify a feature family into a recommendation tier."""
    if not np.isfinite(contrib_pct) or contrib_pct == 0:
        return "insufficient_evidence"

    if admission_rate >= cfg.min_admission_rate_for_promising:
        if inversion_rate <= cfg.max_inversion_rate_for_stable:
            if sign_stab >= 0.5:
                if np.isfinite(median_halflife) and median_halflife < cfg.max_halflife_for_fast:
                    return "promising_but_fast_decay"
                return "promising_stable"

    if inversion_rate > 0.8:
        return "inverted_alpha"

    if sign_stab < 0.3:
        return "unstable_sign"

    if admission_rate == 0:
        return "mostly_rejected"

    if contrib_pct < cfg.min_contrib_pct_for_significant:
        return "noisy_family"

    return "insufficient_evidence"


# ── Main entry point ─────────────────────────────────────────────────────────

def run_feature_audit(
    *,
    decay_path: Path | str | None = None,
    admission_path: Path | str | None = None,
    horizon_alignment_report: dict[str, object] | None = None,
    feature_specs: dict | None = None,
    target_horizon: int = 10,
    out_dir: Path | str | None = None,
    config: FeatureAuditConfig | None = None,
) -> FeatureAuditReport:
    """
    P21: Run the complete feature-family audit.

    Reads existing CSV artifacts, aggregates by family, audits sign stability,
    and produces a research recommendation report.

    Parameters
    ----------
    decay_path              : path to alpha_ic_decay.csv
    admission_path          : path to feature_admission.csv
    horizon_alignment_report: dict from get_horizon_alignment_report()
    feature_specs           : optional override for FEATURE_SPECS
    target_horizon          : production horizon in days
    out_dir                 : directory to write reports to
    config                  : FeatureAuditConfig override

    Returns
    -------
    FeatureAuditReport with families, sign stability, summary, recommendations.
    """
    if config is None:
        config = FeatureAuditConfig()

    registry = FeatureFamilyRegistry(feature_specs)

    # Load artifacts
    decay = load_alpha_ic_decay(decay_path) if decay_path else pd.DataFrame()
    admission = load_feature_admission(admission_path) if admission_path else pd.DataFrame()

    # Sign stability
    sign_stability = compute_sign_stability(decay, registry, config)

    # Family summaries
    families = compute_family_summaries(
        registry, decay, admission, horizon_alignment_report,
        sign_stability, target_horizon, config,
    )

    # Build summary
    n_admitted_total = sum(f.n_admitted for f in families)
    n_inverted_total = sum(f.n_inverted for f in families)
    n_unstable = sum(1 for f in families if f.recommendation == "unstable_sign")
    n_fast = sum(1 for f in families if "fast_decay" in f.recommendation)
    n_inverted = sum(1 for f in families if f.recommendation == "inverted_alpha")
    n_promising = sum(1 for f in families if "promising" in f.recommendation)

    summary = {
        "horizon": target_horizon,
        "n_features": registry.n_features,
        "n_families": registry.n_families,
        "n_admitted_total": n_admitted_total,
        "n_inverted_total": n_inverted_total,
        "n_promising_families": n_promising,
        "n_unstable_families": n_unstable,
        "n_fast_decay_families": n_fast,
        "n_inverted_families": n_inverted,
        "top_family_by_contrib": max(families, key=lambda f: abs(f.ic_contribution_pct) if np.isfinite(f.ic_contribution_pct) else 0).name if families else "N/A",
    }

    # Recommendations
    recommendations: list[str] = []
    for f in families:
        if f.recommendation == "promising_stable" and f.ic_contribution_pct > 10:
            recommendations.append(
                f"KEEP {f.name}: {f.n_admitted}/{f.n_features} admitted, "
                f"|IC|={f.mean_abs_ic:.4f}, contrib={f.ic_contribution_pct:.1f}%"
            )
        elif f.recommendation == "promising_but_fast_decay":
            recommendations.append(
                f"WATCH {f.name}: good IC but fast decay (halflife={f.median_halflife:.1f}d). "
                f"Consider shorter rebalance or persistence filtering."
            )
        elif f.recommendation == "inverted_alpha":
            recommendations.append(
                f"INVESTIGATE {f.name}: {f.n_inverted}/{f.n_admitted} admitted features are sign-inverted. "
                f"Feature direction may be anti-correlated with target."
            )
        elif f.recommendation == "unstable_sign":
            recommendations.append(
                f"REVIEW {f.name}: IC sign flips across horizons (stability={f.ic_sign_stability:.2f}). "
                f"May need horizon-specific admission or removal."
            )
        elif f.recommendation == "mostly_rejected":
            recommendations.append(
                f"REMOVE {f.name}: 0/{f.n_features} features admitted. "
                f"Could not pass IC/monotonicity/coverage gates."
            )

    if not recommendations:
        recommendations.append("No feature families show sufficient signal quality. Consider expanding feature universe or redesigning targets.")

    report = FeatureAuditReport(
        config=asdict(config),
        families=families,
        sign_stability=sign_stability,
        summary=summary,
        recommendations=recommendations,
    )

    # Write outputs if out_dir provided
    if out_dir is not None:
        out = Path(out_dir)
        report.save(out / "feature_audit_report.json")
        report.save_markdown(out / "feature_audit_recommendation.md")
        print(f"[FeatureAudit] Reports written to {out}")

    return report
