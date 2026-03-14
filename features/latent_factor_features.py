"""
Latent factor extraction from a numeric feature matrix.

This module applies PCA to compress a broad set of standardised
features into a small number of latent factors, which can be used
as higher-level risk or style dimensions.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def extract_latent_factors(feature_matrix: pd.DataFrame) -> pd.DataFrame:
    """
    Extract three latent factors from a numeric feature matrix using PCA.

    Financial intuition
    -------------------
    - Many raw signals are correlated; PCA summarises them into a
      few orthogonal components that often behave like style factors
      (e.g. growth vs value, momentum vs mean-reversion).
    - Re-standardising the components makes their magnitudes
      comparable across time and datasets.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Input DataFrame of candidate features. May contain non-numeric
        columns and NaNs.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same index as ``feature_matrix`` and three
        columns: ``latent_factor_1``, ``latent_factor_2``,
        ``latent_factor_3``. Rows that could not be used for PCA are
        filled with NaN.

    Notes
    -----
    Emits a ``UserWarning`` (but does not raise) if there are
    insufficient numeric columns or rows to run PCA reliably.
    """
    if feature_matrix is None or feature_matrix.empty:
        warnings.warn("extract_latent_factors: empty feature matrix.", UserWarning)
        return pd.DataFrame(
            index=feature_matrix.index if feature_matrix is not None else None,
            columns=["latent_factor_1", "latent_factor_2", "latent_factor_3"],
            dtype=float,
        )

    numeric = feature_matrix.select_dtypes(include=[np.number])
    numeric = numeric.dropna(axis=1, how="all")

    if numeric.shape[1] < 3:
        warnings.warn(
            "extract_latent_factors: need at least 3 numeric feature columns for PCA.",
            UserWarning,
        )
        return pd.DataFrame(
            index=feature_matrix.index,
            columns=["latent_factor_1", "latent_factor_2", "latent_factor_3"],
            dtype=float,
        )

    clean = numeric.dropna(axis=0, how="any")
    if clean.shape[0] < 10:
        warnings.warn(
            "extract_latent_factors: need at least 10 complete rows for PCA.",
            UserWarning,
        )
        return pd.DataFrame(
            index=feature_matrix.index,
            columns=["latent_factor_1", "latent_factor_2", "latent_factor_3"],
            dtype=float,
        )

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(clean.values)

    pca = PCA(n_components=3)
    comps = pca.fit_transform(X_scaled)

    comps_std = comps.std(axis=0, ddof=0)
    comps_std[comps_std == 0] = 1.0
    comps_norm = (comps - comps.mean(axis=0)) / comps_std

    factors_local = pd.DataFrame(
        comps_norm,
        index=clean.index,
        columns=["latent_factor_1", "latent_factor_2", "latent_factor_3"],
    )

    factors = pd.DataFrame(
        index=feature_matrix.index,
        columns=["latent_factor_1", "latent_factor_2", "latent_factor_3"],
        dtype=float,
    )
    factors.loc[factors_local.index, :] = factors_local

    return factors


def extract_latent_factors_walk_forward(
    feature_matrix: pd.DataFrame,
    train_end: int,
    n_components: int = 3,
) -> pd.DataFrame:
    """
    Extract latent factors using a single in-sample training window.

    Why the original function is unsafe
    -----------------------------------
    ``extract_latent_factors`` fits both ``StandardScaler`` and ``PCA``
    on the *entire* feature matrix.  In a backtest this means that
    future observations influence the scale and component directions
    used for past rows, creating look-ahead bias and artificially
    inflating performance.

    This helper instead:
      - fits the scaler and PCA using only rows ``iloc[:train_end]``
        of the cleaned numeric matrix, and
      - applies the fitted transforms to *all* available rows.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Input DataFrame of candidate features (may include non-numeric
        columns and NaNs).
    train_end : int
        Integer iloc index; only rows with ``iloc < train_end`` are
        used to fit the scaler and PCA.
    n_components : int, default 3
        Number of principal components / latent factors to extract.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same index as ``feature_matrix`` and
        columns ``latent_factor_1`` .. ``latent_factor_n`` (up to
        ``n_components``). Rows excluded from the numeric clean subset
        are filled with NaN.
    """
    if feature_matrix is None or feature_matrix.empty:
        warnings.warn(
            "extract_latent_factors_walk_forward: empty feature matrix.",
            UserWarning,
        )
        return pd.DataFrame(
            index=feature_matrix.index if feature_matrix is not None else None,
            columns=[f"latent_factor_{i+1}" for i in range(n_components)],
            dtype=float,
        )

    numeric = feature_matrix.select_dtypes(include=[np.number])
    numeric = numeric.dropna(axis=1, how="all")

    if numeric.shape[1] < n_components:
        warnings.warn(
            "extract_latent_factors_walk_forward: insufficient numeric feature columns for PCA.",
            UserWarning,
        )
        return pd.DataFrame(
            index=feature_matrix.index,
            columns=[f"latent_factor_{i+1}" for i in range(n_components)],
            dtype=float,
        )

    clean = numeric.dropna(axis=0, how="any")
    if clean.shape[0] < max(10, n_components * 3):
        warnings.warn(
            "extract_latent_factors_walk_forward: need more complete rows for PCA.",
            UserWarning,
        )
        return pd.DataFrame(
            index=feature_matrix.index,
            columns=[f"latent_factor_{i+1}" for i in range(n_components)],
            dtype=float,
        )

    train_end = max(1, min(int(train_end), len(clean)))
    train_df = clean.iloc[:train_end]
    full_df = clean

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train_df.values)

    pca = PCA(n_components=n_components)
    train_comps = pca.fit_transform(X_train_scaled)

    X_full_scaled = scaler.transform(full_df.values)
    full_comps = pca.transform(X_full_scaled)

    comp_scaler = StandardScaler()
    comps_train_scaled = comp_scaler.fit_transform(train_comps)
    comps_full_scaled = comp_scaler.transform(full_comps)

    columns = [f"latent_factor_{i+1}" for i in range(n_components)]
    factors_local = pd.DataFrame(
        comps_full_scaled,
        index=full_df.index,
        columns=columns,
    )

    factors = pd.DataFrame(
        index=feature_matrix.index,
        columns=columns,
        dtype=float,
    )
    factors.loc[factors_local.index, :] = factors_local

    return factors


def extract_latent_factors_expanding(
    feature_matrix: pd.DataFrame,
    min_train_rows: int = 60,
    n_components: int = 3,
) -> pd.DataFrame:
    """
    Extract latent factors with an expanding, strictly causal window.

    Why the original function is unsafe
    -----------------------------------
    ``extract_latent_factors`` fits scaler and PCA on all rows at
    once, letting information from the future affect earlier
    components.  This is inappropriate for live signal generation and
    can bias backtests.

    This expanding variant:
      - for each row ``i >= min_train_rows``, fits scaler and PCA on
        the history ``iloc[:i]`` only, and
      - transforms **only row i** with that in-sample fit.
    No row ever sees information from the future.

    Window and complexity
    ---------------------
    - The training window grows from ``min_train_rows`` up to the full
      history (expanding window).
    - Computational complexity is roughly O(n²) in the number of
      usable rows, since a full PCA is refit for each new observation.
      This is acceptable for research and low-frequency signals but
      should be used with caution on very long histories.

    Parameters
    ----------
    feature_matrix : pd.DataFrame
        Input DataFrame of candidate features (numeric and non-numeric
        allowed; NaNs allowed).
    min_train_rows : int, default 60
        Minimum number of complete rows required before the first
        factor estimate is produced.
    n_components : int, default 3
        Number of principal components / latent factors to compute.

    Returns
    -------
    pd.DataFrame
        DataFrame with the same index as ``feature_matrix`` and
        columns ``latent_factor_1`` .. ``latent_factor_n``.  Rows
        before ``min_train_rows`` or that cannot be used because of
        missing data are filled with NaN.
    """
    if feature_matrix is None or feature_matrix.empty:
        warnings.warn(
            "extract_latent_factors_expanding: empty feature matrix.",
            UserWarning,
        )
        return pd.DataFrame(
            index=feature_matrix.index if feature_matrix is not None else None,
            columns=[f"latent_factor_{i+1}" for i in range(n_components)],
            dtype=float,
        )

    numeric = feature_matrix.select_dtypes(include=[np.number])
    numeric = numeric.dropna(axis=1, how="all")

    if numeric.shape[1] < n_components:
        warnings.warn(
            "extract_latent_factors_expanding: insufficient numeric feature columns for PCA.",
            UserWarning,
        )
        return pd.DataFrame(
            index=feature_matrix.index,
            columns=[f"latent_factor_{i+1}" for i in range(n_components)],
            dtype=float,
        )

    clean = numeric.dropna(axis=0, how="any")
    if clean.shape[0] < max(min_train_rows, n_components * 3):
        warnings.warn(
            "extract_latent_factors_expanding: not enough complete rows for expanding PCA.",
            UserWarning,
        )
        return pd.DataFrame(
            index=feature_matrix.index,
            columns=[f"latent_factor_{i+1}" for i in range(n_components)],
            dtype=float,
        )

    columns = [f"latent_factor_{i+1}" for i in range(n_components)]
    factors = pd.DataFrame(
        index=feature_matrix.index,
        columns=columns,
        dtype=float,
    )

    n_rows = len(clean)
    start_idx = max(int(min_train_rows), n_components * 3)

    for i in range(start_idx, n_rows):
        train_df = clean.iloc[:i]
        row_df = clean.iloc[[i]]

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(train_df.values)

        pca = PCA(n_components=n_components)
        train_comps = pca.fit_transform(X_train_scaled)

        X_row_scaled = scaler.transform(row_df.values)
        row_comp = pca.transform(X_row_scaled)

        comp_scaler = StandardScaler()
        train_comps_scaled = comp_scaler.fit_transform(train_comps)
        row_scaled = comp_scaler.transform(row_comp)

        idx = row_df.index[0]
        factors.loc[idx, :] = row_scaled[0]

    return factors


