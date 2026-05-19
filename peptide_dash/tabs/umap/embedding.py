from .shared import *
from .input_matrix import _umap_source_mode, _prepare_feature_design_matrix, _build_pmf_shape_matrix_for_umap
from .selection import _dedupe_preserve_order
from .analytics import _parse_family


def _umap_defaults(preset: str, dims: int, source_mode: str = UMAP_SOURCE_BASIC) -> Tuple[Optional[int], Optional[float], Optional[str]]:
    """Basic preset defaults for UMAP."""
    preset = (preset or "").lower()
    if preset == "local":
        nn = 40 if dims == 2 else 70
        md = 0.10 if dims == 2 else 0.08
        metric = "hellinger" if _umap_source_mode(source_mode) == UMAP_SOURCE_PMF else "cosine"
    elif preset == "global":
        nn = 60 if dims == 2 else 100
        md = 0.12 if dims == 2 else 0.08
        metric = "hellinger" if _umap_source_mode(source_mode) == UMAP_SOURCE_PMF else "cosine"
    else:
        nn = None
        md = None
        metric = None
    return nn, md, metric




def _knn_indices(X: np.ndarray, k: int) -> Optional[np.ndarray]:
    """Return kNN indices for each row of X (excluding self), or None if too large without sklearn."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.shape[0] < (k + 2):
        return None
    k = int(max(1, k))

    # Prefer scikit-learn if available
    try:
        from sklearn.neighbors import NearestNeighbors  # type: ignore
        nn = NearestNeighbors(n_neighbors=min(k + 1, X.shape[0]), algorithm="auto")
        nn.fit(X)
        ind = nn.kneighbors(return_distance=False)
        if ind.shape[1] >= 2:
            return ind[:, 1 : (k + 1)]
        return None
    except Exception:
        pass

    # Brute force fallback (cap size to avoid O(N^2) blowups)
    if X.shape[0] > 1500:
        return None
    d2 = np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1)
    ind = np.argsort(d2, axis=1)[:, 1 : (k + 1)]
    return ind


def _procrustes_rms(X: np.ndarray, Y: np.ndarray) -> float:
    """Orthogonal Procrustes alignment RMS error (scale+rotation), center-normalized."""
    X = np.asarray(X, dtype=float)
    Y = np.asarray(Y, dtype=float)
    if X.shape != Y.shape or X.ndim != 2 or X.shape[0] < 3:
        return float("nan")

    X0 = X - np.mean(X, axis=0, keepdims=True)
    Y0 = Y - np.mean(Y, axis=0, keepdims=True)
    nx = float(np.linalg.norm(X0))
    ny = float(np.linalg.norm(Y0))
    if nx <= 0 or ny <= 0:
        return float("nan")
    X0 = X0 / nx
    Y0 = Y0 / ny

    try:
        U, _, Vt = np.linalg.svd(Y0.T @ X0, full_matrices=False)
        R = U @ Vt
        Y_al = Y0 @ R
        return float(np.sqrt(np.mean((X0 - Y_al) ** 2)))
    except Exception:
        return float("nan")


def _pairwise_distances_dense(X: np.ndarray, metric: str = "euclidean") -> np.ndarray:
    """Small dependency-light pairwise distances for graph-based DR."""
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        return np.zeros((0, 0), dtype=float)
    metric = (metric or "euclidean").lower().strip()
    try:
        from sklearn.metrics import pairwise_distances  # type: ignore
        metric_skl = "euclidean" if metric == "hellinger" else metric
        return np.asarray(pairwise_distances(X, metric=metric_skl), dtype=float)
    except Exception:
        pass
    if metric in {"manhattan", "cityblock", "l1"}:
        D = np.sum(np.abs(X[:, None, :] - X[None, :, :]), axis=-1)
    elif metric in {"cosine", "correlation"}:
        Y = X - np.mean(X, axis=1, keepdims=True) if metric == "correlation" else X.copy()
        norms = np.linalg.norm(Y, axis=1, keepdims=True)
        Y = Y / np.maximum(norms, 1e-12)
        D = 1.0 - np.clip(Y @ Y.T, -1.0, 1.0)
        D = np.maximum(D, 0.0)
    else:
        D = np.sqrt(np.maximum(np.sum((X[:, None, :] - X[None, :, :]) ** 2, axis=-1), 0.0))
    D = np.asarray(D, dtype=float)
    np.fill_diagonal(D, 0.0)
    return D


def _adaptive_diffusion_kernel(D: np.ndarray, knn: int, alpha: float = 1.0) -> tuple[np.ndarray, list[str]]:
    """Return row-stochastic diffusion operator from a dense distance matrix."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    notes: list[str] = []
    if n == 0:
        return np.zeros((0, 0), dtype=float), ["empty graph"]
    knn = max(2, min(int(knn or 15), n - 1))
    alpha = float(alpha)
    notes.append(f"knn={knn}")
    notes.append(f"alpha={alpha:g}")
    idx = np.argsort(D, axis=1)[:, : knn + 1]
    eps = np.take_along_axis(D, idx[:, -1:], axis=1).reshape(-1)
    tri = D[np.triu_indices(n, 1)] if n > 1 else np.asarray([1.0])
    gmed = float(np.nanmedian(tri[np.isfinite(tri) & (tri > 0)])) if np.any(np.isfinite(tri) & (tri > 0)) else 1.0
    eps = np.where(np.isfinite(eps) & (eps > 0), eps, gmed)
    eps = np.maximum(eps, 1e-12)
    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        js = idx[i]
        dij = D[i, js]
        W[i, js] = np.exp(-np.clip((dij * dij) / np.maximum(eps[i] * eps[js], 1e-12), 0.0, 700.0))
    W = 0.5 * (W + W.T)
    q = np.maximum(W.sum(axis=1), 1e-12)
    K = W / ((q ** alpha)[:, None] * (q ** alpha)[None, :])
    d = np.maximum(K.sum(axis=1), 1e-12)
    P = K / d[:, None]
    return P, notes


def _diffusion_map_embedding(
    X: np.ndarray,
    dims: int,
    nn: int,
    metric: str,
    *,
    alpha: float = 1.0,
    t: int = 1,
    n_components: int = 10,
) -> tuple[np.ndarray, str, dict]:
    """Diffusion-map coordinates from preprocessed feature/PCA rows.

    Returns coords of shape (n_samples, n_comp) where n_comp = max(dims, n_components).
    All diffusion components are returned so callers can choose which to visualise.
    """
    X = np.asarray(X, dtype=float)
    dims = int(dims or 2)
    n_comp = max(dims, max(2, int(n_components or 10)))
    if X.ndim != 2 or X.shape[0] < 3:
        raise ValueError("Diffusion map needs at least 3 rows.")
    D = _pairwise_distances_dense(X, metric=metric)
    P, notes = _adaptive_diffusion_kernel(D, nn, alpha=alpha)
    A = 0.5 * (P + P.T)
    vals, vecs = np.linalg.eigh(A)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    lam_all = np.clip(vals[1:], 0.0, 1.0)
    # Extract n_comp non-trivial eigenvectors (skip index 0 = trivial)
    n_avail = min(n_comp, vecs.shape[1] - 1)
    lam = np.asarray(vals[1 : n_avail + 1], dtype=float)
    psi = np.asarray(vecs[:, 1 : n_avail + 1], dtype=float)
    if psi.shape[1] < n_comp:
        psi = np.hstack([psi, np.zeros((psi.shape[0], n_comp - psi.shape[1]), dtype=float)])
        lam = np.pad(lam, (0, n_comp - lam.size), constant_values=0.0)
    t = max(0, int(t or 1))
    coords = psi * (np.maximum(lam, 0.0)[None, :] ** float(t))  # (n_samples, n_comp)
    if not np.isfinite(coords).all():
        coords = np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)
    near1 = int(np.sum(vals[1:] > 0.999)) if vals.size > 1 else 0
    extra = "; ".join(notes + [f"t={t}", f"lambda2={lam[0]:.4g}" if lam.size else "lambda2=NA"])
    if near1 >= 2:
        extra += "; warning: graph may be disconnected"
    t_range = list(range(1, 21))
    vne_list: list[float] = []
    if lam_all.size > 0:
        for t_vne in t_range:
            q = lam_all ** t_vne
            q_sum = float(q.sum())
            if q_sum > 0:
                q = q / q_sum
                vne_list.append(float(-np.sum(q * np.log(q + 1e-12))))
            else:
                vne_list.append(0.0)
    dm_data = {
        "eigenvalues": lam_all[:30].tolist(),
        "vne_curve": list(zip(t_range, vne_list)),
        "t_used": int(t),
        "n_components": n_comp,
        "dc_labels": [f"DC{i + 1}" for i in range(n_comp)],
    }
    return coords, extra, dm_data


def _classical_mds(D: np.ndarray, dims: int) -> np.ndarray:
    """Classical metric MDS/PCoA embedding from a dense distance matrix."""
    D = np.asarray(D, dtype=float)
    n = D.shape[0]
    if n == 0:
        return np.zeros((0, int(dims or 2)), dtype=float)
    D2 = D ** 2
    J = np.eye(n) - np.ones((n, n), dtype=float) / float(n)
    B = -0.5 * (J @ D2 @ J)
    vals, vecs = np.linalg.eigh(B)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    pos = np.maximum(vals[: int(dims or 2)], 0.0)
    coords = vecs[:, : int(dims or 2)] * np.sqrt(pos[None, :])
    if coords.shape[1] < int(dims or 2):
        coords = np.hstack([coords, np.zeros((n, int(dims or 2) - coords.shape[1]), dtype=float)])
    return np.nan_to_num(coords, nan=0.0, posinf=0.0, neginf=0.0)


def _phate_like_embedding(X: np.ndarray, dims: int, nn: int, metric: str, *, alpha: float = 1.0, t: int = 3) -> tuple[np.ndarray, str]:
    """Dependency-free PHATE-style map using diffusion potential + classical MDS.

    This is a PHATE-compatible visualization path for environments where the
    optional phate package is not installed: build a diffusion operator, diffuse
    it for t steps, transform probabilities to potentials (-log P_t), and embed
    potential distances with classical MDS.
    """
    X = np.asarray(X, dtype=float)
    dims = int(dims or 2)
    if X.ndim != 2 or X.shape[0] < 3:
        raise ValueError("PHATE needs at least 3 rows.")
    D = _pairwise_distances_dense(X, metric=metric)
    P, notes = _adaptive_diffusion_kernel(D, nn, alpha=alpha)
    t = max(1, int(t or 3))
    Pt = np.linalg.matrix_power(P, t)
    Pt = np.maximum(Pt, 1e-12)
    U = -np.log(Pt)
    U = U - np.mean(U, axis=1, keepdims=True)
    DU = _pairwise_distances_dense(U, metric="euclidean")
    coords = _classical_mds(DU, dims)
    extra = "; ".join(notes + [f"t={t}", "potential=-log(P^t)", "embedding=classical MDS"])
    return coords, extra


def _compute_umap_embedding(
    pmf_df: pd.DataFrame,
    feats_df: pd.DataFrame,
    preset: str,
    dims: int,
    nn: Optional[int],
    min_dist: Optional[float],
    metric: str,
    input_source: str,
    pmf_metrics_sel: List[str],
    pmf_repr: str,
    embedding_method: str = "umap",
    dens_lambda: Optional[float] = None,
    dens_frac: Optional[float] = None,
    dens_var_shift: Optional[float] = None,
    fit_use_all: bool = True,
    plot_use_all: bool = True,
    fit_variants: Optional[List[str]] = None,
    plot_variants: Optional[List[str]] = None,
    plot_union: bool = False,
    stability_runs: int = 1,
    stability_k: int = 10,
    random_state: int = 42,
    pca_cap: int = 64,
    dm_alpha: float = 1.0,
    dm_n_components: int = 10,
) -> Tuple[Optional[pd.DataFrame], np.ndarray, Optional[pd.DataFrame], List[str], str, Optional[np.ndarray], dict]:
    """
    Heavy lifting: fit UMAP (or DensMAP) on a *fit* subset, then (optionally) project other variants.

    Variant selection:
      - fit_use_all=True  => fit on all variants available in the chosen design matrix.
      - plot_use_all=True => plot all variants available in the chosen design matrix.
      - When plot_use_all=False, only `plot_variants` are plotted (and those not in fit-set are projected).

    Returns
    -------
    plot_df      : DataFrame with embedding coords + metadata (or None on error)
    emb          : ndarray (n_plotted, dims) of embedding
    Xdf          : design matrix rows for plotted points (for analytics) or None
    colnames     : list of feature/metric names for Xdf
    info         : info / error string
    pca_loadings : Vt[:k_pca, :] array (PC loadings; fitted on fit-set) or None
    """
    dims = int(dims or 2)
    preset = preset or "global"
    metric = metric or "cosine"
    embedding_method = (embedding_method or "umap").lower().strip()
    source_mode = _umap_source_mode(input_source)
    # PCA/t-SNE are intentionally not exposed in this tab. If stale browser
    # state still sends those values, fall back to UMAP instead of branching
    # into old experimental code paths.
    if embedding_method in {"pca", "linear-pca", "linear_pca", "linear", "tsne", "t-sne", "t_sne"}:
        embedding_method = "umap"
    use_linear_pca = False
    use_tsne = False
    use_isomap = embedding_method in {"isomap", "iso-map", "iso_map"}
    use_densmap = embedding_method in {"densmap", "dens-map", "dens_map", "dens"}
    use_diffusion = embedding_method in {"diffusion", "diffusion-map", "diffusion_map", "diffmap", "dm"}
    use_phate = embedding_method in {"phate", "phate-like", "phate_like"}

    if use_isomap and not HAVE_ISOMAP:
        return None, np.zeros((0, 0)), None, [], "Isomap requested, but scikit-learn Isomap is not available.", None, {}
    if (not use_isomap) and (not use_diffusion) and (not use_phate) and not HAVE_UMAP:
        return None, np.zeros((0, 0)), None, [], "umap-learn not installed (pip install umap-learn), unless using Diffusion Map or PHATE.", None, {}

    # Apply preset defaults
    if preset in ("local", "global"):
        dnn, dmd, dmetric = _umap_defaults(preset, dims, source_mode)
        if dnn is not None:
            nn = int(dnn)
        if dmd is not None:
            min_dist = float(dmd)
        if dmetric is not None:
            metric = dmetric

    nn = int(nn or 50)
    min_dist = float(min_dist or 0.12)

    metrics = list(pmf_metrics_sel or [])
    source_note = ""

    if source_mode == UMAP_SOURCE_PMF:
        X_raw, colnames_raw, metaX, source_note = _build_pmf_shape_matrix_for_umap(
            pmf_df,
            metrics,
            use_repr=str(pmf_repr or "P"),
        )
        matrix_type = "PMF-shape log(P)" if str(pmf_repr or "P") == "log_P" else "PMF-shape sqrt(P_mass)"
        if X_raw.size == 0 or metaX.empty:
            return None, np.zeros((0, 0)), None, [], source_note or "Could not build PMF-shape UMAP matrix.", None, {}
        # PMF matrix is already sqrt(P) and family-balanced.  Force Euclidean
        # unless the user chose a non-Hellinger metric deliberately.  The
        # 'hellinger' selector is normalized to euclidean below.
    else:
        X_raw, colnames_raw, metaX, source_note = _prepare_feature_design_matrix(
            feats_df,
            source_mode,
            selected_metrics=metrics,
        )
        if source_mode == UMAP_SOURCE_THERMO:
            matrix_type = "thermodynamic basin-corrected scalar features"
        elif source_mode == UMAP_SOURCE_SEQUENCE:
            matrix_type = "sequence/letter-space descriptors"
        else:
            matrix_type = "basic/native scalar features"
        if X_raw.size == 0 or metaX.empty:
            return None, np.zeros((0, 0)), None, [], source_note or "Could not build feature-based UMAP matrix.", None, {}


    # Metric / transformation adjustments for PMF probability vectors
    #
    # Hellinger mode preserves distributional geometry end-to-end:
    #   (a) x |-> sqrt(x)  puts each PMF on the unit Hellinger sphere
    #   (b) centering (subtracting fit-set mean) preserves pairwise Euclidean
    #       distances (a rigid translation), so it's safe.
    #   (c) per-column STD-SCALING would break Hellinger - we deliberately
    #       skip it below when hellinger_mode is True.
    #   (d) family balancing 1/sqrt(n_bins) is a uniform per-family scale
    #       on the sqrt(P) block, which reweights each family's contribution
    #       to Hellinger^2 by 1/n_bins. That's the whole point - it prevents
    #       a finely binned family from dominating just because it has more
    #       columns. The Hellinger interpretation survives this as a weighted
    #       sum of per-family Hellinger distances.
    #   (e) PCA truncation is a linear projection; Euclidean distances in
    #       the PC space approach the full Hellinger distance as k grows,
    #       with error bounded by the dropped variance (~1% at EVR=0.99).
    #   (f) UMAP is called with metric='euclidean' on the sqrt(P)-space
    #       vectors, which is mathematically equivalent to metric='hellinger'
    #       on the original P-space (up to the sqrt(2) factor that UMAP
    #       absorbs into its scale-invariant neighbor graph).
    #
    # Outside Hellinger mode (metric='euclidean','cosine','manhattan',...):
    # we keep the original per-column z-score + family balancing + PCA flow.
    pmf_metric_note = ""
    is_pmf_matrix = matrix_type.startswith("PMF")
    hellinger_mode = False

    if is_pmf_matrix:
        # PMF-shape source is already sqrt(P_mass) and family-balanced by the
        # shared PCA/PMF vectorization logic.  For hellinger/euclidean requests,
        # center-only preprocessing below preserves Hellinger-like pairwise
        # distances; standard z-scaling would not. Other user-selected metrics
        # keep the normal z-scored pipeline.
        metric_req = (metric or "").lower().strip()
        if metric_req == "hellinger":
            metric = "euclidean"
            metric_req = "euclidean"
        hellinger_mode = metric_req == "euclidean" and str(pmf_repr or "P") != "log_P"
        if hellinger_mode:
            pmf_metric_note = "PMF-shape mode: sqrt(P_mass) + UMAP euclidean input; per-column std-scaling SKIPPED"
        else:
            pmf_metric_note = f"PMF-shape mode: sqrt(P_mass) + UMAP metric={metric}; z-scored before PCA"
    elif (metric or "").lower().strip() == "hellinger":
        metric = "euclidean"
        pmf_metric_note = "metric=hellinger requested but matrix is not PMF-based; using euclidean instead"

    # Variant universe in this matrix (important: PMF mode may already be an intersection set)
    variants_avail = metaX["variant"].astype(str).tolist()
    variants_avail = _dedupe_preserve_order(variants_avail)
    vset = set(variants_avail)

    # Normalize user selections against available variants
    fit_variants = [str(v) for v in (fit_variants or [])]
    plot_variants = [str(v) for v in (plot_variants or [])]

    if plot_use_all:
        plot_sel = list(variants_avail)
    else:
        plot_sel = [v for v in _dedupe_preserve_order(plot_variants) if v in vset]
        if not plot_sel:
            return None, np.zeros((0, 0)), None, [], "No plot variants selected (or none matched the available matrix).", None, {}

    if fit_use_all:
        fit_sel = list(variants_avail)
    else:
        fit_sel = [v for v in _dedupe_preserve_order(fit_variants) if v in vset]
        if not fit_sel:
            return None, np.zeros((0, 0)), None, [], "No fit variants selected (or none matched the available matrix).", None, {}

    # UMAP needs a small but nontrivial number of training points
    if len(fit_sel) < 3:
        return None, np.zeros((0, 0)), None, [], "Fit set too small: select at least 3 variants for fitting.", None, {}

    # Indices for rows in X_raw
    v2row = {v: i for i, v in enumerate(variants_avail)}

    # Union of variants required (fit + plotted), preserving matrix order
    union_set = set(fit_sel) | set(plot_sel)
    union_sel = [v for v in variants_avail if v in union_set]
    union_pos = {v: i for i, v in enumerate(union_sel)}

    rows_union = np.asarray([v2row[v] for v in union_sel], dtype=int)
    X_union = X_raw[rows_union, :]

    # Fit rows (in union coordinates)
    fit_union_idx = np.asarray([union_pos[v] for v in fit_sel if v in union_pos], dtype=int)
    if fit_union_idx.size < 3:
        return None, np.zeros((0, 0)), None, [], "Fit set collapsed after filtering to the available matrix; need ≥3.", None, {}

    X_fit = X_union[fit_union_idx, :]

    # Standardization parameters from FIT set only.
    # Hellinger mode: center only (sd := 1). Centering is a rigid translation,
    # so pairwise Euclidean distances on sqrt(P) rows are preserved, which is
    # exactly what we need for 'metric=hellinger' semantics.
    # Other modes: classical z-score (mean-zero, unit-std per column).
    mu = np.mean(X_fit, axis=0)
    if hellinger_mode:
        sd = np.ones_like(mu)
    else:
        sd = np.std(X_fit, axis=0, ddof=1)
        sd = np.where(sd == 0, 1.0, sd)

    Xz_fit = (X_fit - mu) / sd
    Xz_union = (X_union - mu) / sd

    # PMF family balancing: equalize total variance per metric family so
    # "more bins" doesn't dominate. In Hellinger mode this reweights each
    # family's contribution to Hellinger^2 uniformly by 1/n_bins; outside
    # Hellinger mode it's a variance-balancing heuristic on z-scored bins.
    pmf_balance_note = ""
    if is_pmf_matrix and colnames_raw and source_mode != UMAP_SOURCE_PMF:
        fams = np.asarray([_parse_family(c) for c in colnames_raw], dtype=object)
        uniq, counts = np.unique(fams, return_counts=True)
        fam2cnt = dict(zip(uniq.tolist(), counts.tolist()))
        scales = np.asarray([1.0 / np.sqrt(float(fam2cnt[f])) for f in fams], dtype=float)
        Xz_fit = Xz_fit * scales
        Xz_union = Xz_union * scales
        cmin = int(np.min(counts)) if counts.size else 0
        cmax = int(np.max(counts)) if counts.size else 0
        pmf_balance_note = f"pmf_family_balance=1/sqrt(n_bins); families={len(fam2cnt)}; bins_range={cmin}..{cmax}"


    pca_loadings: Optional[np.ndarray] = None

    # PCA fit on FIT set only; project UNION with the same loadings.
    # In Hellinger mode this is PCA on centered sqrt(P) (a legitimate
    # distance-preserving-up-to-truncation linear projection).
    try:
        U, s, Vt = np.linalg.svd(Xz_fit, full_matrices=False)
        if s.size:
            evr = (s ** 2) / np.sum(s ** 2)
            cum = np.cumsum(evr)
            k_pca = int(np.searchsorted(cum, 0.99) + 1)
            k_pca = min(k_pca, int(pca_cap or 64), Xz_fit.shape[1])
        else:
            k_pca = min(int(pca_cap or 64), Xz_fit.shape[1])

        Vt_k = Vt[:k_pca, :]
        X_pca_union = Xz_union @ Vt_k.T
        X_pca_fit = X_pca_union[fit_union_idx, :]
        pca_space_label = "centered sqrt(P)" if hellinger_mode else "z-scored features"
        pca_note = f"PCA on {pca_space_label} (fit-set): k={k_pca} (≈99% EVR, cap {int(pca_cap or 64)})"
        pca_loadings = Vt_k
    except Exception:
        X_pca_union = Xz_union
        X_pca_fit = X_pca_union[fit_union_idx, :]
        pca_note = "PCA failed; used preprocessed matrix directly."
        pca_loadings = None

    # Embedding method (UMAP / DensMAP / PCA / t-SNE)
    dl = df_ = dv = None  # densmap params (for info)
    nonfit_idx = np.asarray([i for i in range(len(union_sel)) if i not in set(fit_union_idx.tolist())], dtype=int)
    proj_note = ""
    method_note = ""
    extra_data: dict = {}
    umap_kwargs = {}

    if use_linear_pca:
        # Cheap deterministic baseline: use the fit-set PCA basis already computed
        # above and plot the first 2/3 PC scores for all requested variants.
        emb_union = X_pca_union[:, :dims]
        if emb_union.shape[1] < dims:
            pad = np.zeros((emb_union.shape[0], dims - emb_union.shape[1]), dtype=float)
            emb_union = np.hstack([emb_union, pad])
        method_note = "; method=PCA(linear, fit-set basis)"
    elif use_tsne:
        # Defensive dead branch: stale values are normalized to UMAP above.
        return None, np.zeros((0, 0)), None, [], "t-SNE is not exposed in this tab; using UMAP requires clearing stale browser state.", None, {}
    elif use_isomap:
        if X_pca_union.shape[0] < 3:
            return None, np.zeros((0, 0)), None, [], "Isomap needs at least 3 variants.", None, {}
        iso_nn = int(max(2, min(int(nn or 10), X_pca_union.shape[0] - 1)))
        try:
            try:
                reducer = Isomap(n_neighbors=iso_nn, n_components=dims, metric=metric)
            except TypeError:
                reducer = Isomap(n_neighbors=iso_nn, n_components=dims)
            emb_union = np.asarray(reducer.fit_transform(X_pca_union), dtype=float)
            if emb_union.shape[1] < dims:
                pad = np.full((emb_union.shape[0], dims - emb_union.shape[1]), np.nan, dtype=float)
                emb_union = np.hstack([emb_union, pad])
            proj_note = "; Isomap fit on (fit ∪ plotted)"
            method_note = f"; method=Isomap; n_neighbors={iso_nn}; min_dist ignored"
        except Exception as e:
            return None, np.zeros((0, 0)), None, [], f"Isomap fitting failed: {e}", None, {}
    elif use_diffusion:
        if X_pca_union.shape[0] < 3:
            return None, np.zeros((0, 0)), None, [], "Diffusion Map needs at least 3 variants.", None, {}
        dm_nn = int(max(2, min(int(nn or 25), X_pca_union.shape[0] - 1)))
        try:
            try:
                dm_t = int(round(float(min_dist)))
            except Exception:
                dm_t = 1
            dm_t = max(0, min(10, dm_t))
            emb_union, dm_note, extra_data = _diffusion_map_embedding(
                X_pca_union, dims, dm_nn, metric,
                alpha=float(dm_alpha if dm_alpha is not None else 1.0),
                t=dm_t,
                n_components=int(dm_n_components or 10),
            )
            proj_note = "; Diffusion Map fit on (fit ∪ plotted); no out-of-sample projection"
            method_note = f"; method=Diffusion Map; {dm_note}; parameter=time t={dm_t}"
        except Exception as e:
            return None, np.zeros((0, 0)), None, [], f"Diffusion Map fitting failed: {e}", None, {}
    elif use_phate:
        if X_pca_union.shape[0] < 3:
            return None, np.zeros((0, 0)), None, [], "PHATE needs at least 3 variants.", None, {}
        ph_nn = int(max(2, min(int(nn or 25), X_pca_union.shape[0] - 1)))
        try:
            try:
                ph_t = int(round(float(min_dist)))
            except Exception:
                ph_t = 3
            ph_t = max(1, min(20, ph_t))
            emb_union, phate_note = _phate_like_embedding(X_pca_union, dims, ph_nn, metric, alpha=1.0, t=ph_t)
            proj_note = "; PHATE-style fit on (fit ∪ plotted); no out-of-sample projection"
            method_note = f"; method=PHATE-style; {phate_note}; parameter=time t={ph_t}"
        except Exception as e:
            return None, np.zeros((0, 0)), None, [], f"PHATE fitting failed: {e}", None, {}
    else:
        umap_kwargs = dict(
            n_neighbors=nn,
            min_dist=min_dist,
            n_components=dims,
            metric=metric,
            random_state=int(random_state),
        )

        if use_densmap:
            dl = 2.0 if dens_lambda is None else float(dens_lambda)
            df_ = 0.30 if dens_frac is None else float(dens_frac)
            dv = 0.10 if dens_var_shift is None else float(dens_var_shift)
            df_ = float(np.clip(df_, 0.0, 1.0))
            dv = float(np.clip(dv, 0.0, 1.0))
            umap_kwargs.update(
                densmap=True,
                dens_lambda=dl,
                dens_frac=df_,
                dens_var_shift=dv,
                output_dens=False,
            )

        try:
            reducer = umap.UMAP(**umap_kwargs)
        except TypeError as e:
            if use_densmap and ("densmap" in str(e) or "dens_" in str(e)):
                return (
                    None,
                    np.zeros((0, 0)),
                    None,
                    [],
                    "DensMAP requested, but your installed umap-learn does not support densmap parameters. "
                    "Upgrade umap-learn to a newer version that includes DensMAP.",
                    None,
                    {},
                )
            raise

        # Fit on FIT set; then project other points (if requested/possible).
        try:
            reducer.fit(X_pca_fit)
            emb_fit = np.asarray(getattr(reducer, "embedding_", None))
            if emb_fit is None or emb_fit.size == 0:
                emb_fit = reducer.fit_transform(X_pca_fit)
            emb_union = np.full((X_pca_union.shape[0], dims), np.nan, dtype=float)
            emb_union[fit_union_idx, :] = emb_fit

            if nonfit_idx.size:
                try:
                    emb_union[nonfit_idx, :] = reducer.transform(X_pca_union[nonfit_idx, :])
                except Exception:
                    # Fallback: re-fit on union (loses strict separation, but at least shows points).
                    emb_union = reducer.fit_transform(X_pca_union)
                    proj_note = "; projection unsupported → re-fit on (fit ∪ plotted)"
        except Exception as e:
            return None, np.zeros((0, 0)), None, [], f"UMAP/DensMAP fitting failed: {e}", None, {}


    # Optional stability diagnostics (rerun with different random seeds; compare neighborhoods + Procrustes)
    stability_note = ""
    try:
        runs = int(stability_runs) if stability_runs is not None else 1
    except Exception:
        runs = 1
    try:
        k_nn = int(stability_k) if stability_k is not None else 10
    except Exception:
        k_nn = 10
    runs = max(1, min(runs, 8))
    k_nn = max(3, min(k_nn, 50))

    if HAVE_UMAP and (not use_linear_pca) and (not use_tsne) and (not use_isomap) and (not use_diffusion) and (not use_phate) and runs > 1 and emb_union.shape[0] >= (k_nn + 2):
        try:
            union_pos = {str(v): i for i, v in enumerate(union_sel)}
            plot_sel_present = union_sel if plot_union else [v for v in plot_sel if str(v) in union_pos]
            sel_idx = np.asarray([union_pos[str(v)] for v in plot_sel_present], dtype=int)
            E0 = emb_union[sel_idx, :]
            knn0 = _knn_indices(E0, k_nn)
            overlaps: List[float] = []
            procs: List[float] = []
            if knn0 is not None:
                for s in range(1, runs):
                    umap_kwargs_s = dict(umap_kwargs)
                    umap_kwargs_s["random_state"] = int(random_state) + int(s)
                    try:
                        reducer_s = umap.UMAP(**umap_kwargs_s)
                        reducer_s.fit(X_pca_fit)
                        emb_fit_s = np.asarray(getattr(reducer_s, "embedding_", None))
                        if emb_fit_s is None or emb_fit_s.size == 0:
                            emb_fit_s = reducer_s.fit_transform(X_pca_fit)

                        emb_union_s = np.full((X_pca_union.shape[0], dims), np.nan, dtype=float)
                        emb_union_s[fit_union_idx, :] = emb_fit_s

                        if nonfit_idx.size:
                            try:
                                emb_union_s[nonfit_idx, :] = reducer_s.transform(X_pca_union[nonfit_idx, :])
                            except Exception:
                                emb_union_s = reducer_s.fit_transform(X_pca_union)

                        Es = emb_union_s[sel_idx, :]
                        knns = _knn_indices(Es, k_nn)
                        if knns is None:
                            continue

                        ov = float(
                            np.mean(
                                [
                                    len(set(knn0[i]).intersection(set(knns[i]))) / float(k_nn)
                                    for i in range(knn0.shape[0])
                                ]
                            )
                        )
                        overlaps.append(ov)
                        procs.append(_procrustes_rms(E0, Es))
                    except Exception:
                        continue

                if overlaps:
                    stability_note = (
                        f"; stability(runs={runs}, k={k_nn}): "
                        f"mean kNN overlap={float(np.mean(overlaps)):.2f}, "
                        f"mean Procrustes RMS={float(np.nanmean(procs)):.3g}"
                    )
        except Exception:
            stability_note = ""
    # Build metadata (union), then subset to plotted
    base = feats_df.loc[:, ~feats_df.columns.duplicated()].copy()
    if "variant" in base.columns:
        base = base.drop_duplicates(subset=["variant"], keep="first")

    meta_union = metaX.copy()
    meta_union["variant"] = meta_union["variant"].astype(str)
    meta_union = meta_union.drop_duplicates(subset=["variant"], keep="first").set_index("variant")
    meta_union = meta_union.reindex(union_sel).reset_index()

    meta_union = meta_union.merge(base, on="variant", how="left")
    if len(meta_union) != len(union_sel):
        meta_union = pd.DataFrame({"variant": union_sel})

    plot_union_df = meta_union.copy()
    plot_union_df["UMAP1"] = emb_union[:, 0]
    plot_union_df["UMAP2"] = emb_union[:, 1]
    if dims == 3 and emb_union.shape[1] >= 3:
        plot_union_df["UMAP3"] = emb_union[:, 2]

    if use_diffusion:
        for i in range(emb_union.shape[1]):
            plot_union_df[f"DC{i+1}"] = emb_union[:, i]

    fit_set = set(map(str, fit_sel))
    plot_set = set(map(str, plot_sel))

    def _role(v: str) -> str:
        in_fit = v in fit_set
        in_plot = v in plot_set
        if in_fit and in_plot:
            return "fit+plot"
        if in_fit:
            return "fit"
        return "plot-only"

    plot_union_df["role"] = [_role(str(v)) for v in plot_union_df["variant"].astype(str)]

    # Subset to plot_sel, preserving requested order
    plot_sel_present = union_sel if plot_union else [v for v in plot_sel if v in union_pos]
    plot_idx = np.asarray([union_pos[v] for v in plot_sel_present], dtype=int)

    plot_df = plot_union_df.iloc[plot_idx, :].reset_index(drop=True)
    emb_plot = emb_union[plot_idx, :]

    # Xdf rows aligned to plotted points (for analytics)
    X_plot = X_union[plot_idx, :]
    Xdf = pd.DataFrame(X_plot, columns=colnames_raw)

    if use_linear_pca:
        algo = "PCA"
    elif use_tsne:
        algo = "t-SNE"
    elif use_isomap:
        algo = "Isomap"
    elif use_diffusion:
        algo = "Diffusion Map"
    elif use_phate:
        algo = "PHATE"
    elif use_densmap:
        algo = "DensMAP"
    else:
        algo = "UMAP"
    dens_note = ""
    if use_densmap:
        dens_note = f"; dens_lambda={dl:g}; dens_frac={df_:g}; dens_var_shift={dv:g}"

    fit_in_plot = sum(v in fit_set for v in plot_sel_present)
    proj_in_plot = len(plot_sel_present) - fit_in_plot
    missing_fit = sorted(set(fit_variants) - vset) if (not fit_use_all and fit_variants) else []
    missing_plot = sorted(set(plot_variants) - vset) if (not plot_use_all and plot_variants) else []

    miss_note = ""
    if missing_fit or missing_plot:
        parts = []
        if missing_fit:
            parts.append(f"fit-missing={len(missing_fit)}")
        if missing_plot:
            parts.append(f"plot-missing={len(missing_plot)}")
        miss_note = "; " + ", ".join(parts)


    extra_notes_list = [n for n in [source_note, pmf_metric_note, pmf_balance_note] if n]
    extra_notes = ("; " + "; ".join(extra_notes_list)) if extra_notes_list else ""

    second_param_label = "min_dist"
    if use_diffusion or use_phate:
        second_param_label = "time_t"
    elif use_isomap:
        second_param_label = "unused_param"

    info = (
        f"{len(plot_sel_present)} plotted (fit={fit_in_plot}, projected={proj_in_plot}); "
        f"fit-set={len(fit_sel)}; algo={algo}; dims={dims}; metric={metric}; "
        f"n_neighbors={nn}; {second_param_label}={min_dist}{dens_note}; matrix={matrix_type}; {pca_note}{method_note}{proj_note}{miss_note}{extra_notes}{stability_note}"
    )

    return plot_df, emb_plot, Xdf, colnames_raw, info, pca_loadings, extra_data
