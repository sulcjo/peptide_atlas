from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd


def zscore(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Z-score each column: (X - mean) / std, ignoring NaNs."""
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sigma = np.nanstd(X, axis=0, ddof=0)
    sigma = np.where(sigma == 0, 1.0, sigma)
    Xz = (X - mu) / sigma
    return Xz, mu, sigma


def pca_svd(X: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    PCA via SVD.

    Returns
    -------
    scores : (n_samples, n_components)
    components : (n_components, n_features)
    explained_var_ratio : (n_components,)
    """
    X = np.asarray(X, dtype=float)
    Xz, _, _ = zscore(X)
    Xz = np.nan_to_num(Xz, nan=0.0, posinf=0.0, neginf=0.0)

    U, S, Vt = np.linalg.svd(Xz, full_matrices=False)
    comps = Vt[:n_components, :]
    scores = U[:, :n_components] * S[:n_components]

    n = max(1, Xz.shape[0] - 1)
    ev = (S**2) / n
    evr = ev / max(np.sum(ev), 1e-12)
    return scores, comps, evr[:n_components]


def prepare_matrix(df: pd.DataFrame, cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    """
    Extract numeric matrix from dataframe.

    - Coerces to numeric
    - Drops columns that are all-NaN
    - Fills remaining NaNs with column means
    """
    if df is None or df.empty or not cols:
        return np.zeros((0, 0)), []

    keep: List[str] = []
    mat_cols = []
    for c in cols:
        if c in df.columns:
            s = pd.to_numeric(df[c], errors="coerce")
            if not np.all(np.isnan(s.to_numpy(dtype=float))):
                keep.append(c)
                mat_cols.append(s)

    if not keep:
        return np.zeros((len(df), 0)), []

    X = np.vstack([c.to_numpy(dtype=float) for c in mat_cols]).T
    mu = np.nanmean(X, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    inds = np.where(np.isnan(X))
    X[inds] = np.take(mu, inds[1])
    return X, keep
