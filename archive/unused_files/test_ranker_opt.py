import numpy as np
import time

y = np.random.randn(500_000)
date_groups = np.repeat(np.arange(1000), 500)
np.random.shuffle(date_groups) # Unsorted

def old_way(y, date_groups):
    labels = np.zeros(len(y), dtype=np.int32)
    n_bins = 10
    for g in np.unique(date_groups):
        mask = date_groups == g
        if mask.sum() < 2:
            continue
        ranks = y[mask].argsort().argsort()
        n = int(mask.sum())
        labels[mask] = np.minimum((ranks * n_bins) // n, n_bins - 1).astype(np.int32)
    return labels

def new_way(y, date_groups):
    sort_idx = np.argsort(date_groups, kind="stable")
    sorted_groups = date_groups[sort_idx]
    sorted_y = y[sort_idx]
    
    diffs = np.concatenate(([True], sorted_groups[1:] != sorted_groups[:-1], [True]))
    boundaries = np.where(diffs)[0]
    
    labels = np.zeros(len(y), dtype=np.int32)
    sorted_labels = np.zeros(len(y), dtype=np.int32)
    n_bins = 10
    
    for i in range(len(boundaries) - 1):
        start, end = boundaries[i], boundaries[i+1]
        n = end - start
        if n < 2:
            continue
        y_group = sorted_y[start:end]
        ranks = y_group.argsort().argsort()
        sorted_labels[start:end] = np.minimum((ranks * n_bins) // n, n_bins - 1)
        
    labels[sort_idx] = sorted_labels
    return labels

t0 = time.time()
old_res = old_way(y, date_groups)
t1 = time.time()
print(f"Old way: {t1-t0:.4f}s")

t0 = time.time()
new_res = new_way(y, date_groups)
t1 = time.time()
print(f"New way: {t1-t0:.4f}s")

print(f"Equal? {np.array_equal(old_res, new_res)}")
