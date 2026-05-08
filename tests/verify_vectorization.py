import numpy as np
import pandas as pd
import time
from scipy.stats import rankdata

def create_test_data(n_dates=1500, n_tickers=2000):
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=n_dates)
    tickers = [f"T{i}" for i in range(n_tickers)]
    
    dfs = []
    for d in dates:
        active = np.random.choice(tickers, size=int(n_tickers * 0.8), replace=False)
        scores = np.random.randn(len(active))
        returns = scores * 0.1 + np.random.randn(len(active)) * 0.5
        dfs.append(pd.DataFrame({"date": d, "ticker": active, "score": scores, "target_return": returns}))
    return pd.concat(dfs, ignore_index=True)


def old_turnover(scored: pd.DataFrame, primary_path: str) -> float:
    work = scored.copy()
    n_per_date = work.groupby("date")["score"].transform("count")
    uniq_per_date = work.groupby("date")["score"].transform("nunique")
    work = work[(n_per_date >= 5) & (uniq_per_date >= 2)].copy()
    if work.empty: return float("nan")
    work["_rank"] = work.groupby("date")["score"].rank(method="average", pct=True)
    if primary_path == "long_short_spread":
        grp_mean = work.groupby("date")["_rank"].transform("mean")
        work["_w_raw"] = work["_rank"] - grp_mean
    abs_sum = work.groupby("date")["_w_raw"].transform(lambda x: np.abs(x).sum())
    valid_gross = abs_sum > 1e-12
    work = work[valid_gross].copy()
    work["_w"] = work["_w_raw"] / abs_sum[valid_gross]
    wide = work.pivot_table(index="date", columns="ticker", values="_w", fill_value=0.0)
    wide.sort_index(inplace=True)
    delta = wide.diff().abs().sum(axis=1).iloc[1:]
    return float(np.nanmean(delta)) if len(delta) > 0 else float("nan")

def old_ic(scored: pd.DataFrame, target_col="target_return"):
    df = scored[["date", "score", target_col]].dropna().copy()
    pearson, spearman = [], []
    for _, g in df.groupby("date", sort=True):
        if len(g) < 5 or g["score"].nunique() < 2 or g[target_col].nunique() < 2: continue
        pearson.append(float(g["score"].corr(g[target_col], method="pearson")))
        spearman.append(float(g["score"].corr(g[target_col], method="spearman")))
    return np.nanmean(pearson), np.nanmean(spearman)

# --- PURE NUMPY VECTORIZATION ---
def get_group_boundaries(group_ids):
    # Assumes group_ids is sorted
    boundaries = np.where(group_ids[1:] != group_ids[:-1])[0] + 1
    return np.concatenate(([0], boundaries, [len(group_ids)]))

def numpy_turnover(scored: pd.DataFrame, primary_path: str) -> float:
    df = scored[["date", "ticker", "score"]].dropna().copy()
    df["date_id"] = df["date"].astype("category").cat.codes
    df["ticker_id"] = df["ticker"].astype("category").cat.codes
    df.sort_values(["date_id", "score"], inplace=True)
    
    dates = df["date_id"].values
    tickers = df["ticker_id"].values
    scores = df["score"].values
    
    bounds = get_group_boundaries(dates)
    
    w_raw = np.zeros(len(scores), dtype=np.float64)
    valid_mask = np.zeros(len(scores), dtype=bool)
    
    # Process group by group in cython-like numpy loop (still faster than pandas lambda)
    for i in range(len(bounds) - 1):
        s, e = bounds[i], bounds[i+1]
        n = e - s
        if n < 5 or scores[e-1] == scores[s]:
            continue
            
        valid_mask[s:e] = True
        
        # Rank logic (using pre-sorted property)
        # Unique elements and counts
        unq, inv, counts = np.unique(scores[s:e], return_inverse=True, return_counts=True)
        # Average rank calculation
        cs = np.cumsum(counts)
        ranks = (cs - counts/2.0 + 0.5)
        r_pct = ranks[inv] / float(n)
        
        if primary_path == "long_only_overlay":
            w = np.clip(r_pct - 0.5, 0.0, None)
        elif primary_path == "short_side":
            w = -np.clip(0.5 - r_pct, 0.0, None)
        else:
            w = r_pct - np.mean(r_pct)
            
        w_raw[s:e] = w
        
    df["_w_raw"] = w_raw
    df = df[valid_mask]
    if df.empty: return float("nan")
    
    # Natively vectorized
    dates_valid = df["date_id"].values
    w_raw_valid = df["_w_raw"].values
    bounds_valid = get_group_boundaries(dates_valid)
    
    abs_sums = np.add.reduceat(np.abs(w_raw_valid), bounds_valid[:-1])
    # Broadcast sum back to rows
    abs_sum_full = np.repeat(abs_sums, np.diff(bounds_valid))
    
    valid_gross = abs_sum_full > 1e-12
    df = df[valid_gross].copy()
    if df.empty: return float("nan")
    
    df["_w"] = df["_w_raw"] / abs_sum_full[valid_gross]
    
    # Still need pivot for ticker diffs, but scipy sparse is faster:
    # Actually, pivot_table is fine now that we've filtered
    wide = df.set_index(["date_id", "ticker_id"])["_w"].unstack(fill_value=0.0)
    delta = wide.diff().abs().sum(axis=1).iloc[1:]
    return float(np.nanmean(delta)) if len(delta) > 0 else float("nan")

def numpy_ic(scored: pd.DataFrame, target_col="target_return"):
    df = scored[["date", "score", target_col]].dropna().copy()
    df["date_id"] = df["date"].astype("category").cat.codes
    df.sort_values("date_id", inplace=True)
    
    dates = df["date_id"].values
    scores = df["score"].values
    targets = df[target_col].values
    
    bounds = get_group_boundaries(dates)
    pearson = []
    spearman = []
    
    for i in range(len(bounds)-1):
        s, e = bounds[i], bounds[i+1]
        n = e - s
        if n < 5: continue
        sc = scores[s:e]
        tc = targets[s:e]
        
        if np.max(sc) == np.min(sc) or np.max(tc) == np.min(tc):
            continue
            
        # Pearson
        sc_m = sc - np.mean(sc)
        tc_m = tc - np.mean(tc)
        cov = np.sum(sc_m * tc_m)
        var = np.sqrt(np.sum(sc_m**2) * np.sum(tc_m**2))
        pearson.append(cov / var if var > 1e-12 else np.nan)
        
        # Spearman
        # rankdata is compiled C from scipy
        sc_rank = rankdata(sc)
        tc_rank = rankdata(tc)
        sc_rm = sc_rank - np.mean(sc_rank)
        tc_rm = tc_rank - np.mean(tc_rank)
        scov = np.sum(sc_rm * tc_rm)
        svar = np.sqrt(np.sum(sc_rm**2) * np.sum(tc_rm**2))
        spearman.append(scov / svar if svar > 1e-12 else np.nan)
        
    return np.nanmean(pearson), np.nanmean(spearman)

def main():
    df = create_test_data(n_dates=1500, n_tickers=2000)
    t0 = time.time()
    old_to = old_turnover(df, "long_short_spread")
    t1 = time.time()
    new_to = numpy_turnover(df, "long_short_spread")
    t2 = time.time()
    print(f'Old Turnover: {t1-t0:.4f}s | New Turnover: {t2-t1:.4f}s')
    assert np.isclose(old_to, new_to, atol=1e-8)
    
    t0 = time.time()
    old_p, old_s = old_ic(df)
    t1 = time.time()
    new_p, new_s = numpy_ic(df)
    t2 = time.time()
    print(f'Old IC: {t1-t0:.4f}s | New IC: {t2-t1:.4f}s')
    assert np.isclose(old_p, new_p, atol=1e-8)
    assert np.isclose(old_s, new_s, atol=1e-8)

if __name__ == "__main__":
    main()
