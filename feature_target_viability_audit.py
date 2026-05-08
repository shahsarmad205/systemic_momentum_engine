"""
Feature & Target Viability Research Framework (Minimal, Fast)
==============================================================
Only computes what's needed for the audit classification.
"""

from __future__ import annotations

import math
import json
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

OUT_DIR = Path("output/models/feature_target_viability")
OUT_DIR.mkdir(parents=True, exist_ok=True)

HORIZONS = [5, 10, 20, 63]

FEATURE_FAMILIES = {
    "trend": ["f_trend"],
    "momentum": ["ret_5d", "ret_10d", "ret_20d", "cs_momentum_percentile",
                 "momentum_12m_skip1", "nearness_52w_high", "momentum_acceleration"],
    "reversal": ["short_term_reversal", "industry_relative_reversal", "nearness_52w_low"],
    "quality_lowvol": ["low_vol_score", "quality_score"],
    "residual_alpha": ["capm_alpha"],
    "risk": ["capm_residual_vol", "rolling_vol_20"],
    "regime": ["vol_ratio_5_20"],
    "liquidity": ["turnover_pct_rank"],
    "sector_relative": ["sector_relative_20d", "sector_relative_60d"],
}

ALL_FEATURES = sorted(set(f for fam in FEATURE_FAMILIES.values() for f in fam))

TARGET_NAMES = [
    "raw_forward_return", "centered_forward_return", "sector_neutral_residual",
    "beta_neutral_residual", "market_neutral_residual", "vol_adjusted_forward_return",
    "cost_adjusted_forward_return", "risk_adjusted_forward_return",
    "regime_conditioned_forward_return",
]


def compute_forward_returns_fast(df, horizons):
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    log_ret = np.log1p(df["daily_return"].values)
    ticker_ids = df["ticker"].values
    n = len(df)
    changes = np.where(ticker_ids[1:] != ticker_ids[:-1])[0] + 1
    boundaries = np.concatenate([[0], changes, [n]])
    for h in horizons:
        fwd = np.full(n, np.nan)
        for i in range(len(boundaries) - 1):
            s, e = boundaries[i], boundaries[i + 1]
            if e - s <= h:
                continue
            lc = np.cumsum(log_ret[s:e])
            valid = np.arange(s, e - h)
            fwd[valid] = np.exp(lc[valid - s + h] - lc[valid - s]) - 1.0
        df[f"fwd_{h}d"] = fwd
    return df


def compute_ic_and_halflife(df, feature_cols, target_col):
    """Compute IC series and halflife for all features vs one target. Fast."""
    work = df[["date", target_col] + feature_cols].dropna(subset=[target_col])
    if len(work) < 100:
        return {}

    # Rank all features + target within each date
    for col in feature_cols:
        work[f"{col}_r"] = work.groupby("date")[col].rank(pct=True, method="average")
    work["t_r"] = work.groupby("date")[target_col].rank(pct=True, method="average")

    g = work.groupby("date", sort=True)
    n_per_date = g["t_r"].count().values.astype(float)
    sum_ty = g["t_r"].sum().values
    sum_ty2 = (work["t_r"] ** 2).groupby(work["date"]).sum().values

    results = {}
    for feat in feature_cols:
        rc = f"{feat}_r"
        sum_tx = g[rc].sum().values
        sum_txy = (work[rc] * work["t_r"]).groupby(work["date"]).sum().values
        sum_tx2 = (work[rc] ** 2).groupby(work["date"]).sum().values
        denom = np.sqrt(np.maximum((sum_tx2 - sum_tx**2 / n_per_date) * (sum_ty2 - sum_ty**2 / n_per_date), 0))
        ic = np.where(denom > 1e-10, (sum_txy - sum_tx * sum_ty / n_per_date) / denom, np.nan)
        ic = ic[(n_per_date >= 10) & np.isfinite(ic)]

        if len(ic) < 5:
            continue

        ic_mean = ic.mean()
        ic_std = ic.std(ddof=1)
        ic_tstat = ic_mean / (ic_std / math.sqrt(len(ic))) if ic_std > 0 else np.nan
        ic_ir = abs(ic_mean) / ic_std if ic_std > 0 else np.nan

        # Halflife
        v = ic[np.isfinite(ic)]
        max_lag = min(40, len(v) // 3) if len(v) >= 6 else 1
        halflife = np.nan
        if len(v) >= 6 and max_lag >= 2:
            mu, var = v.mean(), v.var()
            if var > 1e-12:
                acf = np.array([np.mean((v[:-lag] - mu) * (v[lag:] - mu)) / var
                                for lag in range(1, max_lag + 1)])
                for i in range(len(acf) - 1):
                    if acf[i] >= 0.5 and acf[i + 1] < 0.5:
                        halflife = float(i + 1 + (0.5 - acf[i + 1]) / (acf[i] - acf[i + 1]))
                        break
                if np.isnan(halflife):
                    halflife = 1.0 if acf[0] < 0.5 else float(max_lag)

        # Selected-book IC (top 20%) - vectorized
        sel = work[work[rc] >= 0.80]
        if len(sel) > 100:
            sel = sel.copy()
            sel["fx"] = sel.groupby("date")[feat].rank(pct=True)
            sel["ty"] = sel.groupby("date")[target_col].rank(pct=True)
            gs = sel.groupby("date", sort=True)
            ns = gs["fx"].count().values.astype(float)
            sx = gs["fx"].sum().values
            sy = gs["ty"].sum().values
            sxy = (sel["fx"] * sel["ty"]).groupby(sel["date"]).sum().values
            sx2 = (sel["fx"] ** 2).groupby(sel["date"]).sum().values
            sy2 = (sel["ty"] ** 2).groupby(sel["date"]).sum().values
            ds = np.sqrt(np.maximum((sx2 - sx**2 / ns) * (sy2 - sy**2 / ns), 0))
            sic = np.where(ds > 1e-10, (sxy - sx * sy / ns) / ds, np.nan)
            sic = sic[(ns >= 10) & np.isfinite(sic)]
            sel_ic = float(sic.mean()) if len(sic) > 0 else np.nan
        else:
            sel_ic = np.nan

        # Alpha/cost ratio
        acr = np.nan
        if "expected_round_trip_cost_frac" in work.columns:
            top = work[work[rc] >= 0.90]
            if len(top) > 10:
                mu_a = top[target_col].mean()
                mu_c = top["expected_round_trip_cost_frac"].mean()
                acr = float(mu_a / mu_c) if mu_c > 1e-8 else np.nan

        results[feat] = {
            "ic_mean": ic_mean, "ic_std": ic_std, "ic_tstat": ic_tstat,
            "ic_ir": ic_ir, "n_days": len(ic), "pct_positive": (ic > 0).mean(),
            "halflife": halflife, "selected_book_ic": sel_ic, "alpha_cost_ratio": acr,
        }

    return results


def classify_feature(row, horizon):
    ic = row.get("ic_mean", 0) or 0
    sel = row.get("selected_book_ic", 0) or 0
    acr = row.get("alpha_cost_ratio", 0) or 0
    hl = row.get("halflife", 0) or 0
    if ic > 0.005 and sel < -0.005:
        return "selection_distorted"
    if ic < 0.005 or sel < -0.01:
        return "non_viable"
    if acr < 0.5:
        return "cost_dominated"
    if hl < horizon / 2:
        return "decay_dominated"
    if ic > 0.02 and sel > 0 and acr > 1 and hl >= horizon:
        return "globally_viable"
    if ic > 0.01:
        return "conditionally_viable"
    return "non_viable"


def construct_targets_fast(df, horizon):
    """Vectorized target construction using pandas groupby."""
    fwd_col = f"fwd_{horizon}d"
    fwd = df[fwd_col].values.astype(np.float64)
    targets = {}

    targets["raw_forward_return"] = fwd.copy()

    # Centered
    targets["centered_forward_return"] = fwd - df.groupby("date")[fwd_col].transform("mean").values

    # Sector-neutral
    if "sector" in df.columns:
        sec = df["sector"].fillna("Unknown")
        sn = fwd - df.groupby(["date", sec])[fwd_col].transform("mean").values
        sn = sn - df.groupby("date")[sn].transform("mean").values
        targets["sector_neutral_residual"] = sn
    else:
        targets["sector_neutral_residual"] = targets["centered_forward_return"].copy()

    # Beta-neutral
    if "capm_beta" in df.columns:
        beta = df["capm_beta"].fillna(1.0).clip(-3, 3).values.astype(np.float64)
        beta_c = beta - df.groupby("date")[beta].transform("mean").values
        fwd_s = pd.Series(fwd, index=df.index)
        beta_s = pd.Series(beta_c, index=df.index)
        cov_yb = (fwd_s * beta_s).groupby(df["date"]).transform("mean") - fwd_s.groupby(df["date"]).transform("mean") * beta_s.groupby(df["date"]).transform("mean")
        var_b = (beta_s ** 2).groupby(df["date"]).transform("mean") - beta_s.groupby(df["date"]).transform("mean") ** 2
        b = cov_yb / var_b.replace(0, np.nan)
        bn = fwd - b.values * beta_c
        bn = bn - df.groupby("date")[bn].transform("mean").values
        targets["beta_neutral_residual"] = bn
    else:
        targets["beta_neutral_residual"] = targets["centered_forward_return"].copy()

    # Market-neutral (skip expensive groupby apply, use simple centering)
    targets["market_neutral_residual"] = targets["centered_forward_return"].copy()

    # Vol-adjusted
    if "realised_vol_20d" in df.columns:
        vol = df["realised_vol_20d"].fillna(0.015).clip(lower=1e-4).values.astype(np.float64)
        va = np.clip(fwd / (vol * math.sqrt(horizon)), -5, 5)
        va = va - df.groupby("date")[va].transform("mean").values
        targets["vol_adjusted_forward_return"] = va
    else:
        targets["vol_adjusted_forward_return"] = targets["centered_forward_return"].copy()

    # Cost-adjusted
    if "expected_round_trip_cost_frac" in df.columns:
        costs = df["expected_round_trip_cost_frac"].fillna(0.001).values.astype(np.float64)
        ca = fwd - costs
        ca = ca - df.groupby("date")[ca].transform("mean").values
        targets["cost_adjusted_forward_return"] = ca
    else:
        targets["cost_adjusted_forward_return"] = targets["centered_forward_return"].copy()

    # Risk-adjusted
    if "realised_vol_20d" in df.columns:
        vol = df["realised_vol_20d"].fillna(0.015).clip(lower=1e-4).values.astype(np.float64)
        ra = np.clip(fwd / vol, -10, 10)
        ra = ra - df.groupby("date")[ra].transform("mean").values
        targets["risk_adjusted_forward_return"] = ra
    else:
        targets["risk_adjusted_forward_return"] = targets["centered_forward_return"].copy()

    # Regime-conditioned
    if "regime_label" in df.columns:
        regs = df["regime_label"].fillna("Unknown")
        rc = fwd - df.groupby(["date", regs])[fwd_col].transform("mean").values
        rc = rc - df.groupby("date")[rc].transform("mean").values
        targets["regime_conditioned_forward_return"] = rc
    else:
        targets["regime_conditioned_forward_return"] = targets["centered_forward_return"].copy()

    return targets


def run_phase1(df, horizon=10):
    target_col = f"fwd_{horizon}d"
    if target_col not in df.columns:
        return pd.DataFrame(), pd.DataFrame()

    active = [f for f in ALL_FEATURES if f in df.columns]
    print(f"    IC + halflife + sel_ic + acr for {len(active)} features...")
    metrics = compute_ic_and_halflife(df, active, target_col)

    results = []
    for family, features in FEATURE_FAMILIES.items():
        for feat in features:
            if feat not in metrics:
                continue
            row = {"feature": feat, "family": family, "horizon": horizon}
            row.update(metrics[feat])
            row["classification"] = classify_feature(row, horizon)
            results.append(row)

    feat_df = pd.DataFrame(results)

    # Conditional IC for top 5 features
    cond_rows = []
    if not feat_df.empty:
        top_feats = feat_df.nlargest(5, "ic_mean")["feature"].tolist()
        print(f"    Conditional IC for: {top_feats}")

        df_b = df[["date", target_col] + top_feats].copy()
        df_b["bucket_sector"] = df["sector"].fillna("Unknown")
        if "adv_dollar_20" in df.columns:
            try:
                df_b["bucket_cap"] = pd.qcut(np.log(df["adv_dollar_20"].clip(lower=1)), 3,
                                              labels=["small", "mid", "large"], duplicates="drop")
            except:
                df_b["bucket_cap"] = "unknown"
        if "realised_vol_20d" in df.columns:
            try:
                df_b["bucket_vol"] = pd.qcut(df["realised_vol_20d"], 3,
                                              labels=["low", "med", "high"], duplicates="drop")
            except:
                df_b["bucket_vol"] = "unknown"
        if "regime_label" in df.columns:
            df_b["bucket_regime"] = df["regime_label"].fillna("Unknown")

        for feat in top_feats:
            family = feat_df[feat_df["feature"] == feat]["family"].iloc[0]
            for bcol in ["bucket_sector", "bucket_cap", "bucket_vol", "bucket_regime"]:
                if bcol not in df_b.columns:
                    continue
                for bv, sub in df_b.groupby(bcol, sort=True):
                    if sub[feat].isna().all() or sub[target_col].isna().all():
                        continue
                    sub_metrics = compute_ic_and_halflife(sub, [feat], target_col)
                    if feat not in sub_metrics:
                        continue
                    sm = sub_metrics[feat]
                    cond_rows.append({
                        "feature": feat, "family": family, "horizon": horizon,
                        "dimension": bcol, "bucket": str(bv),
                        "ic_mean": sm["ic_mean"], "ic_tstat": sm["ic_tstat"],
                        "ic_ir": sm["ic_ir"], "n_days": sm["n_days"],
                        "spread_mean": np.nan,
                    })

    return feat_df, pd.DataFrame(cond_rows)


def run_phase2(df, horizon=10):
    targets = construct_targets_fast(df, horizon)
    proxy = "f_trend" if "f_trend" in df.columns else ALL_FEATURES[0]
    results = []

    for tname in TARGET_NAMES:
        tcol = f"_t_{tname}"
        df[tcol] = targets[tname]
        metrics = compute_ic_and_halflife(df, [proxy], tcol)
        if proxy not in metrics:
            continue
        m = metrics[proxy]

        # Zero-exposure rate
        work = df[[proxy, tcol, "expected_round_trip_cost_frac", "date"]].dropna()
        work = work.copy()
        work["_r"] = work.groupby("date")[proxy].rank(pct=True)
        top = work[work["_r"] >= 0.90]
        n_dates = top["date"].nunique()
        n_zero = sum(1 for _, g in top.groupby("date", sort=True)
                     if g[tcol].mean() < g["expected_round_trip_cost_frac"].mean())
        zero_rate = n_zero / n_dates if n_dates > 0 else np.nan

        ds = m["halflife"] / horizon if m["halflife"] and horizon > 0 else np.nan

        results.append({
            "target": tname, "horizon": horizon,
            "full_universe_ic": m["ic_mean"], "full_universe_ic_tstat": m["ic_tstat"],
            "full_universe_ic_ir": m["ic_ir"],
            "selected_book_ic": m["selected_book_ic"], "weighted_book_ic": np.nan,
            "spread_mean": np.nan, "mono_mean": np.nan,
            "alpha_cost_ratio": m["alpha_cost_ratio"], "halflife": m["halflife"],
            "decay_survival": ds, "net_sharpe": np.nan, "gross_sharpe": np.nan,
            "zero_exposure_rate": zero_rate, "alpha_capture_ratio": np.nan,
        })

    return pd.DataFrame(results)


def run_phase3(df, horizon=10):
    targets = construct_targets_fast(df, horizon)
    results = []
    for tname in TARGET_NAMES:
        tcol = f"_t_{tname}"
        df[tcol] = targets[tname]
        active = [f for f in ALL_FEATURES if f in df.columns]
        metrics = compute_ic_and_halflife(df, active, tcol)

        for family, features in FEATURE_FAMILIES.items():
            for feat in features:
                if feat not in metrics:
                    continue
                m = metrics[feat]
                halflife = m["halflife"]
                rejected = halflife < horizon / 2 if not np.isnan(halflife) else True
                results.append({
                    "feature": feat, "family": family, "target": tname, "horizon": horizon,
                    "halflife": halflife,
                    "horizon_ratio": halflife / horizon if not np.isnan(halflife) else np.nan,
                    "ic_mean": m["ic_mean"], "ic_tstat": m["ic_tstat"],
                    "rejected": rejected,
                    "rejection_reason": f"halflife ({halflife:.1f}d) < horizon/2 ({horizon/2}d)" if rejected else "",
                })
    return pd.DataFrame(results)


def generate_synthesis_report(p1f, p1c, p2, p3, horizon=10):
    lines = []
    lines.append("=" * 100)
    lines.append("FEATURE & TARGET VIABILITY RESEARCH FRAMEWORK — SYNTHESIS REPORT")
    lines.append("=" * 100)
    lines.append(f"Date: 2026-05-04 | Horizon: {horizon}d")
    lines.append("")

    lines.append("-" * 100)
    lines.append("PHASE 1: CONDITIONAL FEATURE VIABILITY AUDIT")
    lines.append("-" * 100)
    if not p1f.empty:
        cd = p1f["classification"].value_counts()
        lines.append("Classification Distribution:")
        for c, n in cd.items():
            lines.append(f"  {c}: {n}")
        lines.append("")
        top = p1f.nlargest(10, "ic_mean")[["feature", "family", "ic_mean", "ic_tstat",
                                            "selected_book_ic", "alpha_cost_ratio", "halflife", "classification"]]
        lines.append("Top 10 Features by IC:")
        lines.append(top.to_string(index=False))
        lines.append("")
        sd = p1f[p1f["classification"] == "selection_distorted"]
        if not sd.empty:
            lines.append(f"SELECTION-DISTORTED ({len(sd)}):")
            lines.append(sd[["feature", "ic_mean", "selected_book_ic", "alpha_cost_ratio"]].to_string(index=False))
            lines.append("")
        if not p1c.empty:
            lines.append("Conditional Highlights:")
            for dim in p1c["dimension"].unique():
                dd = p1c[p1c["dimension"] == dim]
                if dd["ic_mean"].notna().sum() < 2:
                    continue
                b = dd.loc[dd["ic_mean"].idxmax()]
                w = dd.loc[dd["ic_mean"].idxmin()]
                if abs(b["ic_mean"]) > 0.005:
                    lines.append(f"  {dim}: best={b['bucket']} (IC={b['ic_mean']:.4f}), worst={w['bucket']} (IC={w['ic_mean']:.4f})")
            lines.append("")

    lines.append("-" * 100)
    lines.append("PHASE 2: TARGET CONSTRUCTION TOURNAMENT")
    lines.append("-" * 100)
    if not p2.empty:
        p2s = p2.sort_values("full_universe_ic", ascending=False)
        cols = ["target", "full_universe_ic", "full_universe_ic_tstat", "selected_book_ic",
                "alpha_cost_ratio", "halflife", "zero_exposure_rate"]
        lines.append(p2s[cols].to_string(index=False))
        lines.append("")
        b = p2s.iloc[0]
        lines.append(f"Best: {b['target']} (IC={b['full_universe_ic']:.4f}, sel_ic={b['selected_book_ic']:.4f})")
        lines.append("")

    lines.append("-" * 100)
    lines.append("PHASE 3: HALFLIFE COMPATIBILITY")
    lines.append("-" * 100)
    if not p3.empty:
        rej = p3["rejected"].sum()
        lines.append(f"Rejected: {rej}/{len(p3)} ({rej/len(p3)*100:.0f}%)")
        lines.append("")

    lines.append("=" * 100)
    lines.append("SYNTHESIS: NEXT VIABLE PATH")
    lines.append("=" * 100)
    lines.append("")

    decisions = []
    if not p1f.empty:
        sp = (p1f["classification"] == "selection_distorted").mean()
        if sp > 0.3:
            decisions.append(("selection_distorted", f"{sp:.0%} features selection-distorted."))
        dp = (p1f["classification"] == "decay_dominated").mean()
        if dp > 0.3:
            decisions.append(("horizon_redesign", f"{dp:.0%} decay-dominated."))
        cp = (p1f["classification"] == "cost_dominated").mean()
        if cp > 0.5:
            decisions.append(("cost_dominated", f"{cp:.0%} cost-dominated."))

    if not p2.empty:
        bi = p2["full_universe_ic"].max()
        bs = p2.loc[p2["full_universe_ic"].idxmax(), "selected_book_ic"]
        if bi < 0.015:
            decisions.append(("better_features", f"Best IC={bi:.4f} < 0.015."))
        elif bs < 0:
            decisions.append(("better_targets", f"Best IC={bi:.4f} but sel_ic={bs:.4f}."))

    if not p3.empty:
        rr = p3["rejected"].mean()
        if rr > 0.7:
            decisions.append(("horizon_redesign", f"{rr:.0%} combos rejected."))

    if not p1c.empty:
        strong = [d for d in p1c["dimension"].unique() if (p1c[p1c["dimension"] == d]["ic_mean"] > 0.02).any()]
        if strong:
            decisions.append(("conditional_deployment", f"Strong conditional IC in: {', '.join(strong)}."))

    if not decisions:
        decisions.append(("reject_signal_family", "No viable path identified."))

    for i, (p, r) in enumerate(decisions, 1):
        lines.append(f"PATH {i}: {p.upper().replace('_', ' ')}")
        lines.append(f"  {r}")
        lines.append("")

    lines.append("PRIORITY:")
    for i, (p, _) in enumerate(decisions, 1):
        lines.append(f"  P{i}: {p}")
    lines.append("")
    lines.append("=" * 100)
    return "\n".join(lines)


def main():
    t0 = time.time()
    print("Loading panel...")
    df = pd.read_parquet("output/models/enriched_panel.parquet")
    print(f"  {df.shape}, {df['date'].nunique()} dates, {df['ticker'].nunique()} tickers")

    print("Computing forward returns...")
    df = compute_forward_returns_fast(df, HORIZONS)
    for h in HORIZONS:
        print(f"  fwd_{h}d: {df[f'fwd_{h}d'].notna().sum()} non-NaN")

    phase1, phase2, phase3 = {}, {}, {}

    for h in HORIZONS:
        t1 = time.time()
        print(f"\n--- Horizon {h}d ---")
        print("Phase 1...")
        f, c = run_phase1(df, horizon=h)
        phase1[h] = {"features": f, "conditional": c}
        f.to_parquet(OUT_DIR / f"phase1_feature_viability_h{h}d.parquet", index=False)
        c.to_parquet(OUT_DIR / f"phase1_conditional_breakdown_h{h}d.parquet", index=False)
        print(f"  {len(f)} features, {len(c)} cond rows, {time.time()-t1:.1f}s")
        if not f.empty:
            print(f"  Classifications: {f['classification'].value_counts().to_dict()}")

        t2 = time.time()
        print("Phase 2...")
        p2 = run_phase2(df, horizon=h)
        phase2[h] = p2
        p2.to_parquet(OUT_DIR / f"phase2_target_tournament_h{h}d.parquet", index=False)
        print(f"  {len(p2)} targets, {time.time()-t2:.1f}s")
        if not p2.empty:
            b = p2.loc[p2["full_universe_ic"].idxmax()]
            print(f"  Best: {b['target']} (IC={b['full_universe_ic']:.4f})")

        t3 = time.time()
        print("Phase 3...")
        p3 = run_phase3(df, horizon=h)
        phase3[h] = p3
        p3.to_parquet(OUT_DIR / f"phase3_halflife_compatibility_h{h}d.parquet", index=False)
        print(f"  {len(p3)} combos, {int(p3['rejected'].sum())} rejected, {time.time()-t3:.1f}s")

    print("\n" + "=" * 60)
    print("SYNTHESIS REPORT")
    print("=" * 60)
    h = 10
    report = generate_synthesis_report(phase1[h]["features"], phase1[h]["conditional"], phase2[h], phase3[h], h)
    print(report)
    (OUT_DIR / "synthesis_report.txt").write_text(report)
    (OUT_DIR / "synthesis_summary.json").write_text(json.dumps({
        "phase1": {str(h): {"n": len(phase1[h]["features"]), "cls": phase1[h]["features"]["classification"].value_counts().to_dict()} for h in HORIZONS},
        "phase2": {str(h): {"n": len(phase2[h]), "best": phase2[h].loc[phase2[h]["full_universe_ic"].idxmax(), "target"] if not phase2[h].empty else None} for h in HORIZONS},
        "phase3": {str(h): {"rej": int(phase3[h]["rejected"].sum()), "rate": float(phase3[h]["rejected"].mean())} for h in HORIZONS},
    }, indent=2, default=str))
    print(f"\nTotal: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
