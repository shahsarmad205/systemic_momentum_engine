"""
WRDS Migration Validation — Phase 8
=====================================
Run D1–D6 diagnostic checks to verify the CRSP data migration is correct
and quantify any phantom alpha from the old Yahoo pipeline.

Usage:
    WRDS_USERNAME=your_username python scripts/validate_wrds_migration.py

    # Run specific diagnostics only
    WRDS_USERNAME=your_username python scripts/validate_wrds_migration.py \\
        --diagnostics D1 D2 D3 D5

Diagnostics:
    D1  Return distribution sanity: CRSP ret vs Yahoo AdjClose.pct_change
    D2  Delisting coverage: how many delistings, how many are performance failures
    D3  Point-in-time universe: verify no stock is a hindsight inclusion
    D4  Feature distribution comparison: before/after migration
    D5  IC stability: does signal survive on CRSP data?
    D6  Sharpe decomposition: isolate survivorship vs return quality vs delisting
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("validate_wrds")

SEP = "=" * 70


def parse_args():
    p = argparse.ArgumentParser(description="WRDS migration validation diagnostics")
    p.add_argument("--start", default="2010-01-01")
    p.add_argument("--end", default="2015-12-31")
    p.add_argument(
        "--diagnostics", nargs="+",
        default=["D1", "D2", "D3", "D4", "D5", "D6"],
        help="Which diagnostics to run (default: all)"
    )
    p.add_argument(
        "--sample-tickers", nargs="+",
        default=["AAPL", "MSFT", "JPM", "XOM", "GE"],
        help="Sample tickers for D1/D4/D5 (must be in Yahoo cache and WRDS)"
    )
    p.add_argument(
        "--sample-permnos", nargs="+", type=int, default=None,
        help="Override sample PERMNOs for D2/D3 (looked up from tickers if not given)"
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _connect_wrds():
    from utils.wrds_universe import connect_wrds
    user = os.environ.get("WRDS_USERNAME")
    if not user:
        raise RuntimeError("Set WRDS_USERNAME environment variable.")
    return connect_wrds(user)


def _load_yahoo_ret(ticker: str, start: str, end: str) -> pd.Series:
    """Load AdjClose.pct_change from Yahoo Parquet cache."""
    path = Path("data/cache") / f"{ticker}.parquet"
    if not path.exists():
        return pd.Series(dtype=float)
    df = pd.read_parquet(path)
    df.index = pd.to_datetime(df.index)
    df = df.sort_index().loc[start:end]
    col = "AdjClose" if "AdjClose" in df.columns else "Close"
    return df[col].pct_change().dropna().rename(ticker)


def _load_crsp_ret(db, permno: int, start: str, end: str) -> pd.Series:
    """Load CRSP ret from wrds_loader (cache or live query)."""
    from utils.wrds_loader import WRDSLoader
    loader = WRDSLoader(db, cache_ttl_days=30)
    df = loader.load_single(
        permno, f"PERMNO_{permno}",
        (pd.Timestamp(start) - pd.Timedelta(days=5)).strftime("%Y-%m-%d"),
        end,
    )
    if df.empty or "ret" not in df.columns:
        return pd.Series(dtype=float)
    return df.loc[start:end, "ret"].dropna().rename(permno)


def header(title: str):
    print(f"\n{SEP}")
    print(f"  {title}")
    print(SEP)


# ---------------------------------------------------------------------------
# D1: Return Distribution Sanity Check
# ---------------------------------------------------------------------------

def d1_return_distribution(db, tickers: list[str], permno_map: dict[str, int],
                            start: str, end: str):
    header("D1: Return Distribution Sanity — CRSP ret vs Yahoo AdjClose.pct_change")

    results = []
    for ticker in tickers:
        permno = permno_map.get(ticker)
        if permno is None:
            print(f"  {ticker}: no PERMNO mapping — skipping")
            continue

        yahoo = _load_yahoo_ret(ticker, start, end)
        crsp = _load_crsp_ret(db, permno, start, end)

        if yahoo.empty or crsp.empty:
            print(f"  {ticker}: missing data (yahoo={len(yahoo)}, crsp={len(crsp)})")
            continue

        aligned = pd.concat([yahoo.rename("yahoo"), crsp.rename("crsp")], axis=1).dropna()
        diff = aligned["crsp"] - aligned["yahoo"]

        mean_diff = diff.mean()
        p99_diff = diff.abs().quantile(0.99)
        corr = aligned.corr().iloc[0, 1]

        status = "PASS" if abs(mean_diff) < 0.0005 and p99_diff < 0.005 else "WARN"
        print(
            f"  {ticker:>6} ({permno}):  mean_diff={mean_diff:+.6f}  "
            f"p99|diff|={p99_diff:.5f}  corr={corr:.4f}  [{status}]"
        )
        results.append({"ticker": ticker, "mean_diff": mean_diff, "p99_diff": p99_diff, "status": status})

    if results:
        df = pd.DataFrame(results)
        overall_mean = df["mean_diff"].abs().mean()
        n_warn = (df["status"] == "WARN").sum()
        print(f"\n  Summary: mean |bias|={overall_mean:.6f}  warnings={n_warn}/{len(df)}")
        if overall_mean < 0.0005:
            print("  → PASS: systematic bias within tolerance.")
        else:
            print("  → WARN: systematic bias exceeds 5 bps — investigate corporate action handling.")


# ---------------------------------------------------------------------------
# D2: Delisting Coverage Check
# ---------------------------------------------------------------------------

def d2_delisting_coverage(db, permnos: list[int], start: str, end: str):
    header("D2: Delisting Coverage — splice completeness check")

    permno_str = ",".join(str(p) for p in permnos)
    sql = f"""
        SELECT permno, dlstdt, dlret, dlstcd
        FROM crsp.dsedelist
        WHERE permno IN ({permno_str})
          AND dlstdt BETWEEN '{start}' AND '{end}'
        ORDER BY dlstdt
    """
    delistings = db.raw_sql(sql, date_cols=["dlstdt"])

    if delistings.empty:
        print("  WARNING: 0 delisting events found — splice may not be working!")
        print("  Expected ~2–5% of the universe over a multi-year window.")
        return

    n_total = len(delistings)
    n_perf = delistings["dlstcd"].between(500, 591).sum()
    n_missing_dlret = delistings["dlret"].isna().sum()
    n_imputed = (delistings["dlret"].isna() & delistings["dlstcd"].between(500, 591)).sum()

    print(f"  Total delisting events    : {n_total}")
    print(f"  Performance delistings    : {n_perf}  (dlstcd 500–591, imputed at -30%/-55%)")
    print(f"  Missing dlret             : {n_missing_dlret}")
    print(f"  Will be imputed           : {n_imputed}")
    print(f"  Coverage                  : {n_total / len(permnos):.1%} of universe PERMNOs")

    if n_total >= len(permnos) * 0.01:
        print("  → PASS: delistings found within expected range (≥1% of universe).")
    else:
        print("  → WARN: fewer delistings than expected — check dlstdt date range.")

    # Show worst delistings
    worst = delistings[delistings["dlstcd"].between(500, 591)].sort_values("dlstdt")
    if not worst.empty:
        print(f"\n  Performance delisting sample (first 5):")
        print(worst[["permno", "dlstdt", "dlret", "dlstcd"]].head().to_string(index=False))


# ---------------------------------------------------------------------------
# D3: Point-in-Time Universe Check
# ---------------------------------------------------------------------------

def d3_pit_universe(db, start: str, end: str):
    header("D3: Point-in-Time Universe — no hindsight inclusions")

    from utils.wrds_universe import build_backtest_universe

    # Sample 5 random dates across the backtest window
    date_range = pd.bdate_range(start=start, end=end)
    sample_dates = date_range[::max(1, len(date_range) // 5)][:5]

    all_passed = True
    for date in sample_dates:
        date_str = date.strftime("%Y-%m-%d")
        investable = build_backtest_universe(db, date_str)

        # Verify each PERMNO was in dsp500list on this date
        if not investable:
            continue
        permno_str = ",".join(str(p) for p in investable)
        sql = f"""
            SELECT DISTINCT permno
            FROM crsp.dsp500list
            WHERE permno IN ({permno_str})
              AND start <= '{date_str}'
              AND (ending IS NULL OR ending >= '{date_str}')
        """
        valid = db.raw_sql(sql)
        valid_set = set(valid["permno"].astype(int).tolist())
        leakage = [p for p in investable if p not in valid_set]

        status = "PASS" if not leakage else f"FAIL ({len(leakage)} look-ahead permnos)"
        print(f"  {date_str}: {len(investable)} in universe — {status}")
        if leakage:
            all_passed = False
            print(f"    Look-ahead PERMNOs: {leakage[:5]}{'...' if len(leakage) > 5 else ''}")

    print(f"\n  → {'PASS: no hindsight inclusions detected.' if all_passed else 'FAIL: look-ahead bias found — investigate build_backtest_universe.'}")


# ---------------------------------------------------------------------------
# D4: Feature Distribution Comparison
# ---------------------------------------------------------------------------

def d4_feature_distribution(db, tickers: list[str], permno_map: dict[str, int],
                              start: str, end: str):
    header("D4: Feature Distribution — before/after migration")

    from features.feature_pipeline import build_feature_matrix
    from utils.wrds_loader import WRDSLoader

    loader = WRDSLoader(db, cache_ttl_days=30)
    check_features = ["ret_5d", "ret_20d", "momentum_12m_skip1", "momentum_3m"]

    print(f"  {'Feature':<22}  {'Yahoo mean':>11}  {'Yahoo std':>10}  {'CRSP mean':>10}  {'CRSP std':>10}  {'Ratio std':>10}")
    print("  " + "-" * 80)

    for ticker in tickers:
        permno = permno_map.get(ticker)

        # Yahoo features
        yahoo_path = Path("data/cache") / f"{ticker}.parquet"
        yahoo_feats = {}
        if yahoo_path.exists():
            ydf = pd.read_parquet(yahoo_path)
            ydf.index = pd.to_datetime(ydf.index)
            ydf = ydf.sort_index().loc[start:end]
            yfm = build_feature_matrix(ydf)
            for feat in check_features:
                if feat in yfm.columns:
                    yahoo_feats[feat] = yfm[feat].dropna()

        # CRSP features
        crsp_feats = {}
        if permno is not None:
            crsp_df = loader.load_single(permno, ticker, start, end)
            if not crsp_df.empty:
                cfm = build_feature_matrix(crsp_df)
                for feat in check_features:
                    if feat in cfm.columns:
                        crsp_feats[feat] = cfm[feat].dropna()

        for feat in check_features:
            y = yahoo_feats.get(feat, pd.Series(dtype=float))
            c = crsp_feats.get(feat, pd.Series(dtype=float))
            if y.empty and c.empty:
                continue
            ym = y.mean() if not y.empty else float("nan")
            ys = y.std() if not y.empty else float("nan")
            cm = c.mean() if not c.empty else float("nan")
            cs = c.std() if not c.empty else float("nan")
            ratio = cs / ys if ys and ys > 0 else float("nan")
            print(
                f"  {ticker}/{feat:<16}  {ym:>+11.4f}  {ys:>10.4f}  {cm:>+10.4f}  {cs:>10.4f}  {ratio:>10.3f}"
            )

    print("\n  Interpretation:")
    print("  ratio≈1.0 → feature distributions match (data quality OK)")
    print("  ratio<0.9 → CRSP has less cross-sectional variation (expected — fewer Yahoo artifacts)")
    print("  ratio>1.2 → unexpected; investigate")


# ---------------------------------------------------------------------------
# D5: IC Stability Check
# ---------------------------------------------------------------------------

def d5_ic_stability(db, tickers: list[str], permno_map: dict[str, int],
                     start: str, end: str):
    header("D5: IC Stability — signal on CRSP vs Yahoo")

    from utils.wrds_loader import WRDSLoader
    from features.feature_pipeline import build_feature_matrix

    loader = WRDSLoader(db, cache_ttl_days=30)

    def _compute_ic(price_df: pd.DataFrame, feature_col: str = "momentum_3m") -> float:
        """Rank IC between feature at t and 1-day forward return at t+1."""
        fm = build_feature_matrix(price_df)
        if feature_col not in fm.columns or "ret" not in fm.columns:
            fwd = (
                fm["Close"].pct_change() if "Close" in fm.columns
                else fm.get("daily_return", pd.Series(dtype=float))
            )
        else:
            fwd = fm["ret"]
        fwd = fwd.shift(-1)
        feat = fm[feature_col] if feature_col in fm.columns else pd.Series(dtype=float)
        aligned = pd.concat([feat, fwd.rename("fwd")], axis=1).dropna()
        if len(aligned) < 30:
            return float("nan")
        return aligned.corr(method="spearman").iloc[0, 1]

    print(f"  {'Ticker':<8}  {'IC Yahoo':>9}  {'IC CRSP':>9}  {'Ratio':>7}  {'Status'}")
    print("  " + "-" * 50)

    for ticker in tickers:
        permno = permno_map.get(ticker)

        ic_yahoo = float("nan")
        yahoo_path = Path("data/cache") / f"{ticker}.parquet"
        if yahoo_path.exists():
            ydf = pd.read_parquet(yahoo_path)
            ydf.index = pd.to_datetime(ydf.index)
            ydf = ydf.sort_index().loc[start:end]
            ic_yahoo = _compute_ic(ydf)

        ic_crsp = float("nan")
        if permno is not None:
            crsp_df = loader.load_single(permno, ticker, start, end)
            if not crsp_df.empty:
                ic_crsp = _compute_ic(crsp_df)

        ratio = (ic_crsp / ic_yahoo) if (not np.isnan(ic_yahoo) and abs(ic_yahoo) > 0.001) else float("nan")
        if np.isnan(ratio):
            status = "N/A"
        elif ratio >= 0.85:
            status = "PASS — signal survives"
        elif ratio >= 0.70:
            status = "WARN — partial Yahoo artifact"
        else:
            status = "FAIL — signal is largely Yahoo artifact"

        print(f"  {ticker:<8}  {ic_yahoo:>+9.4f}  {ic_crsp:>+9.4f}  {ratio:>7.3f}  {status}")

    print("\n  Interpretation:")
    print("  ratio 0.85–1.05: signal is real, migrate with confidence")
    print("  ratio 0.70–0.85: partial Yahoo artifact, retrain model on CRSP")
    print("  ratio <0.70    : signal was largely Yahoo-specific; do NOT migrate without retraining")


# ---------------------------------------------------------------------------
# D6: Sharpe Decomposition
# ---------------------------------------------------------------------------

def d6_sharpe_decomposition(db, tickers: list[str], permno_map: dict[str, int],
                              start: str, end: str):
    header("D6: Sharpe Decomposition — isolating phantom alpha sources")

    from utils.wrds_loader import WRDSLoader

    loader = WRDSLoader(db, cache_ttl_days=30)

    # Compute equal-weight portfolio returns under three scenarios
    # A: Yahoo returns (baseline)
    # B: CRSP returns, no delisting splice (isolates return quality)
    # C: CRSP returns, with delisting splice (full CRSP — gold standard)
    #    Difference A→B = survivorship + retroactive adjustment
    #    Difference B→C = isolated delisting return effect

    def _portfolio_sharpe(returns_matrix: pd.DataFrame) -> float:
        ew = returns_matrix.mean(axis=1)
        if ew.std() < 1e-9:
            return 0.0
        return float(ew.mean() / ew.std() * np.sqrt(252))

    yahoo_rets, crsp_rets_no_delist, crsp_rets_with_delist = {}, {}, {}

    for ticker in tickers:
        permno = permno_map.get(ticker)

        # Scenario A: Yahoo
        yahoo = _load_yahoo_ret(ticker, start, end)
        if not yahoo.empty:
            yahoo_rets[ticker] = yahoo

        if permno is None:
            continue

        # Scenarios B and C: CRSP
        crsp_df = loader.load_single(permno, ticker, start, end)
        if crsp_df.empty or "ret" not in crsp_df.columns:
            continue
        crsp = crsp_df.loc[start:end, "ret"].dropna()
        crsp_rets_with_delist[ticker] = crsp

        # B: no delisting = clip extreme negative returns (proxy for Yahoo survivorship)
        # Real decomposition: use CRSP without dlret splice by re-running WRDSLoader
        # with _splice_delisting_returns disabled.  This proxy clips -30% tails.
        crsp_no_delist = crsp.clip(lower=-0.30)
        crsp_rets_no_delist[ticker] = crsp_no_delist

    if not yahoo_rets:
        print("  No Yahoo data available — skipping D6 (delete Yahoo cache after D1-D5).")
        return

    common = sorted(
        set(yahoo_rets) & set(crsp_rets_with_delist) & set(crsp_rets_no_delist)
    )
    if len(common) < 2:
        print(f"  Only {len(common)} overlapping tickers — D6 needs ≥2. Skipping.")
        return

    dates = sorted(set.intersection(*[set(yahoo_rets[t].index) for t in common]))
    yahoo_mat = pd.DataFrame({t: yahoo_rets[t] for t in common}).reindex(dates).dropna()
    crsp_no_mat = pd.DataFrame({t: crsp_rets_no_delist[t] for t in common}).reindex(dates).dropna()
    crsp_full_mat = pd.DataFrame({t: crsp_rets_with_delist[t] for t in common}).reindex(dates).dropna()

    sharpe_yahoo = _portfolio_sharpe(yahoo_mat)
    sharpe_crsp_no = _portfolio_sharpe(crsp_no_mat)
    sharpe_crsp_full = _portfolio_sharpe(crsp_full_mat)

    delta_survivorship = sharpe_yahoo - sharpe_crsp_no
    delta_delisting = sharpe_crsp_no - sharpe_crsp_full
    delta_total = sharpe_yahoo - sharpe_crsp_full

    print(f"  Tickers in decomposition : {len(common)}")
    print(f"  Date range               : {dates[0] if dates else 'N/A'}  →  {dates[-1] if dates else 'N/A'}")
    print()
    print(f"  Sharpe (Yahoo, baseline)      : {sharpe_yahoo:+.3f}")
    print(f"  Sharpe (CRSP, no delist)      : {sharpe_crsp_no:+.3f}")
    print(f"  Sharpe (CRSP, full gold std)  : {sharpe_crsp_full:+.3f}")
    print()
    print(f"  Δ Survivorship + retro-adj    : {delta_survivorship:+.3f}  (Yahoo − CRSP_no_delist)")
    print(f"  Δ Delisting returns           : {delta_delisting:+.3f}  (CRSP_no_delist − CRSP_full)")
    print(f"  Δ Total phantom alpha         : {delta_total:+.3f}  (Yahoo − CRSP_full)")
    print()
    if delta_total > 0.05:
        print(f"  → WARN: {delta_total:.3f} Sharpe phantom alpha detected. Expect Sharpe to drop after migration.")
    elif delta_total > 0:
        print(f"  → INFO: small phantom alpha ({delta_total:.3f}). Migration should be largely neutral.")
    else:
        print(f"  → PASS: no phantom alpha — CRSP signal is at least as strong as Yahoo.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    run = set(args.diagnostics)

    db = _connect_wrds()

    from utils.wrds_universe import WRDSUniverse, build_backtest_universe

    universe = WRDSUniverse(db)
    logger.info("Building sample PERMNO map for tickers: %s", args.sample_tickers)

    # Get PERMNOs for sample tickers
    all_permnos = universe.get_unique_permnos(args.start, args.end)
    permno_to_tick = universe.permno_to_ticker_map(all_permnos, args.start)
    ticker_to_permno = {v: k for k, v in permno_to_tick.items()}
    permno_map = {t: ticker_to_permno.get(t) for t in args.sample_tickers}

    if args.sample_permnos:
        sample_permnos = args.sample_permnos
    else:
        investable = build_backtest_universe(db, args.start)
        sample_permnos = investable[:50]  # first 50 for D2/D3

    if "D1" in run:
        d1_return_distribution(db, args.sample_tickers, permno_map, args.start, args.end)

    if "D2" in run:
        d2_delisting_coverage(db, sample_permnos, args.start, args.end)

    if "D3" in run:
        d3_pit_universe(db, args.start, args.end)

    if "D4" in run:
        d4_feature_distribution(db, args.sample_tickers, permno_map, args.start, args.end)

    if "D5" in run:
        d5_ic_stability(db, args.sample_tickers, permno_map, args.start, args.end)

    if "D6" in run:
        d6_sharpe_decomposition(db, args.sample_tickers, permno_map, args.start, args.end)

    print(f"\n{SEP}")
    print("  Validation complete.")
    print(SEP)


if __name__ == "__main__":
    main()
