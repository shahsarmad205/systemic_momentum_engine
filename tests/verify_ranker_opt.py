import numpy as np
import time

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
    _, first_idx, counts = np.unique(date_groups, return_index=True, return_counts=True)
    group_sizes = counts[np.argsort(first_idx)].tolist()
    return labels, group_sizes

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
        sorted_labels[start:end] = np.minimum((ranks * n_bins) // n, n_bins - 1).astype(np.int32)

    labels[sort_idx] = sorted_labels
    _, first_idx, counts = np.unique(date_groups, return_index=True, return_counts=True)
    group_sizes = counts[np.argsort(first_idx)].tolist()
    return labels, group_sizes

def main():
    print("Generating mock data (N=500,000, M=1,000)...")
    y = np.random.randn(500_000)
    date_groups = np.repeat(np.arange(1000), 500)
    np.random.shuffle(date_groups)

    t0 = time.time()
    old_labels, old_sizes = old_way(y, date_groups)
    t1 = time.time()
    print(f"Old O(N*M) Execution: {t1-t0:.4f}s")

    t0 = time.time()
    new_labels, new_sizes = new_way(y, date_groups)
    t1 = time.time()
    print(f"New O(N log N) Execution: {t1-t0:.4f}s")

    assert np.array_equal(old_labels, new_labels), "CRITICAL FAILURE: Labels do not match!"
    assert old_sizes == new_sizes, "CRITICAL FAILURE: Group sizes do not match!"
    
    # Assert per-group ordering
    for g in np.unique(date_groups):
        mask = date_groups == g
        assert np.array_equal(old_labels[mask], new_labels[mask]), f"Mismatch in group {g}"

    print("VERIFICATION SUCCESS: New algorithm produces mathematically identical output.")

if __name__ == "__main__":
    main()
