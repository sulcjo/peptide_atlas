from .shared import *


def _add_dbscan_clusters(
    plot_df: pd.DataFrame,
    dims: int,
    enabled: bool,
    eps: Optional[float],
    min_samples: Optional[int],
    cluster_on: str = "plotted",
) -> Tuple[pd.DataFrame, str, Optional[pd.DataFrame]]:
    """Add DBSCAN cluster labels to plot_df and return a summary table."""
    if plot_df is None or plot_df.empty:
        return plot_df, "DBSCAN: no points.", None

    plot_df2 = plot_df.copy()
    dims = int(dims or 2)

    if not enabled:
        return plot_df2, "", None

    if not HAVE_SKLEARN:
        plot_df2["dbscan_cluster"] = "unavailable"
        return plot_df2, "DBSCAN requested but scikit-learn is not available.", None

    coord_cols = ["UMAP1", "UMAP2"] + (["UMAP3"] if (dims == 3 and "UMAP3" in plot_df2.columns) else [])
    if any(c not in plot_df2.columns for c in coord_cols):
        plot_df2["dbscan_cluster"] = "unavailable"
        return plot_df2, "DBSCAN: missing embedding coordinates.", None

    cluster_on = (cluster_on or "plotted").lower().strip()
    mask = np.ones(len(plot_df2), dtype=bool)
    if cluster_on in {"fit", "fit-only", "fit_only"} and "role" in plot_df2.columns:
        mask = plot_df2["role"].astype(str).isin(["fit", "fit+plot"]).to_numpy()

    # parse params
    try:
        eps_f = float(eps) if eps is not None else 0.5
    except Exception:
        eps_f = 0.5
    eps_f = max(eps_f, 1e-6)

    try:
        ms = int(min_samples) if min_samples is not None else 5
    except Exception:
        ms = 5
    ms = max(ms, 1)

    X = plot_df2.loc[mask, coord_cols].to_numpy(dtype=float, copy=False)
    if X.shape[0] < max(3, ms):
        plot_df2["dbscan_cluster"] = "unclustered" if (mask.sum() != len(plot_df2)) else "noise"
        return plot_df2, f"DBSCAN skipped (too few points: {X.shape[0]} for min_samples={ms}).", None

    model = DBSCAN(eps=eps_f, min_samples=ms)
    labels = model.fit_predict(X)

    def _lab_to_str(l: int) -> str:
        if int(l) == -1:
            return "noise"
        return f"C{int(l)}"

    lab_str = np.array([_lab_to_str(l) for l in labels], dtype=object)

    if mask.sum() != len(plot_df2):
        plot_df2["dbscan_cluster"] = "unclustered"
        plot_df2.loc[mask, "dbscan_cluster"] = lab_str
    else:
        plot_df2["dbscan_cluster"] = lab_str

    vc = plot_df2["dbscan_cluster"].value_counts(dropna=False)
    summary = (
        vc.rename_axis("cluster")
          .reset_index(name="n_points")
          .assign(frac=lambda d: d["n_points"] / float(len(plot_df2)))
    )

    def _sort_key(x: str):
        s = str(x)
        if s.startswith("C"):
            try:
                return (0, int(s[1:]))
            except Exception:
                return (0, 10**9)
        if s == "noise":
            return (1, 0)
        if s == "unclustered":
            return (2, 0)
        return (3, 0)

    summary["__k"] = summary["cluster"].map(_sort_key)
    summary = summary.sort_values("__k").drop(columns="__k")

    n_noise = int((plot_df2["dbscan_cluster"] == "noise").sum())
    n_uncl = int((plot_df2["dbscan_cluster"] == "unclustered").sum())
    n_clusters = len({c for c in plot_df2["dbscan_cluster"].unique().tolist() if str(c).startswith("C")})

    sil_note = ""
    if silhouette_score is not None:
        try:
            sub = plot_df2.loc[mask, coord_cols + ["dbscan_cluster"]].copy()
            sub = sub[sub["dbscan_cluster"].astype(str).str.startswith("C")]
            if sub["dbscan_cluster"].nunique() >= 2 and len(sub) >= 3:
                labs = sub["dbscan_cluster"].astype(str).str[1:].astype(int).to_numpy()
                XX = sub[coord_cols].to_numpy(dtype=float)
                sil = float(silhouette_score(XX, labs))
                sil_note = f"; silhouette={sil:.3f} (clustered points only)"
        except Exception:
            pass

    note = f"DBSCAN: eps={eps_f:g}, min_samples={ms}; clusters={n_clusters}, noise={n_noise}"
    if n_uncl:
        note += f", unclustered={n_uncl}"
    note += sil_note

    return plot_df2, note, summary

def _parse_family(colname: str) -> str:
    """e.g. 'metric|x=0.1' -> 'metric'."""
    if "|" in colname:
        return colname.split("|", 1)[0]
    return colname


def _family_groups(colnames: List[str]) -> Dict[str, np.ndarray]:
    fam_to_idx: Dict[str, List[int]] = {}
    for j, c in enumerate(colnames):
        fam = _parse_family(c)
        fam_to_idx.setdefault(fam, []).append(j)
    return {k: np.asarray(v, int) for k, v in fam_to_idx.items()}


def _corr_tables_for_axes(
    emb: np.ndarray,
    colnames: List[str],
    Xmatrix_df: pd.DataFrame,
    per_family_top_n: int = 5,
) -> List[pd.DataFrame]:
    """
    For each UMAP axis (up to 3), and for each family:

    - Compute Spearman ρ between that axis and **each bin** in the family.
    - Take up to `per_family_top_n` strongest |ρ|.
    - Return a table with one row per family:

        [Family, Bin1, ρ1, Bin2, ρ2, ..., Bin5, ρ5]
    """
    fam_idx = _family_groups(colnames)
    out: List[pd.DataFrame] = []

    n_axes = min(emb.shape[1], 3)
    for i in range(n_axes):
        axis = emb[:, i]
        axis_series = pd.Series(axis)
        rows = []

        for fam, idxs in fam_idx.items():
            contribs: List[Tuple[str, float]] = []
            for j in idxs:
                s = pd.to_numeric(Xmatrix_df.iloc[:, j], errors="coerce")
                ok = np.isfinite(s) & np.isfinite(axis_series)
                if ok.sum() < 3:
                    continue
                sr = s[ok].rank()
                ar = axis_series[ok].rank()
                if sr.std(ddof=1) == 0 or ar.std(ddof=1) == 0:
                    continue
                rho = float(np.corrcoef(sr.to_numpy(), ar.to_numpy())[0, 1])
                if not np.isfinite(rho):
                    continue
                bin_label = re.sub(r"^.+\|x=", "x=", colnames[j])
                contribs.append((bin_label, rho))

            if not contribs:
                continue

            contribs.sort(key=lambda x: abs(x[1]), reverse=True)
            top = contribs[:per_family_top_n]

            row: List[object] = [fam]
            for bin_label, rho in top:
                row.extend([bin_label, rho])

            while len(row) < 1 + 2 * per_family_top_n:
                row.append("")
            rows.append(row)

        cols = [f"Family (axis {i+1})"]
        for k in range(1, per_family_top_n + 1):
            cols.extend([f"Bin {k}", f"ρ{k}"])

        df_axis = pd.DataFrame(rows, columns=cols)
        out.append(df_axis)

    return out


def _global_top_corr_for_axes(
    emb: np.ndarray,
    colnames: List[str],
    Xmatrix_df: pd.DataFrame,
    top_n: int = 5,
) -> List[pd.DataFrame]:
    """
    For each UMAP axis (up to 3):
      Across all columns, compute Spearman ρ vs axis.
      Return top_n strongest (by |ρ|) bin-level contributors.

    Output per axis:
      columns: ['Family', 'Bin', 'Spearman ρ']
    """
    out: List[pd.DataFrame] = []
    n_axes = min(emb.shape[1], 3)

    for i in range(n_axes):
        axis = emb[:, i]
        axis_series = pd.Series(axis)
        candidates: List[Tuple[str, str, float]] = []

        for j, colname in enumerate(colnames):
            s = pd.to_numeric(Xmatrix_df.iloc[:, j], errors="coerce")
            ok = np.isfinite(s) & np.isfinite(axis_series)
            if ok.sum() < 3:
                continue
            sr = s[ok].rank()
            ar = axis_series[ok].rank()
            if sr.std(ddof=1) == 0 or ar.std(ddof=1) == 0:
                continue
            rho = float(np.corrcoef(sr.to_numpy(), ar.to_numpy())[0, 1])
            if not np.isfinite(rho):
                continue

            fam = _parse_family(colname)
            xlab = re.sub(r"^.+\|x=", "x=", colname)
            candidates.append((fam, xlab, rho))

        candidates.sort(key=lambda x: abs(x[2]), reverse=True)
        top = candidates[:top_n]
        df_axis = pd.DataFrame(top, columns=[f"Family (axis {i+1})", "Bin", "Spearman ρ"])
        out.append(df_axis)

    return out


# ----------------------------------------------------------------------
# PCA loading helpers
# ----------------------------------------------------------------------

def _pca_loading_tables(
    pca_loadings: Optional[np.ndarray],
    colnames: List[str],
    top_n: int = 10,
) -> List[pd.DataFrame]:
    """
    From PCA loadings (Vt) and feature names (colnames):

    For each of first up to 3 PCs:
      - Sort features by |loading|.
      - Take top_n.
      - Return one DataFrame per PC:

        ['Feature', 'Loading']
    """
    if pca_loadings is None or pca_loadings.size == 0:
        return []

    n_pcs = min(3, pca_loadings.shape[0])
    n_features = pca_loadings.shape[1]
    if n_features != len(colnames):
        return []

    out: List[pd.DataFrame] = []
    for i in range(n_pcs):
        load = pca_loadings[i, :]
        idx_sorted = np.argsort(-np.abs(load))  # descending by |loading|
        idx_top = idx_sorted[:top_n]
        rows = [(colnames[j], float(load[j])) for j in idx_top]
        df_pc = pd.DataFrame(rows, columns=[f"PC{i+1} Feature", "Loading"])
        out.append(df_pc)

    return out


def _family_pca_loading_table_from_Xdf(
    Xdf: Optional[pd.DataFrame],
    colnames: List[str],
    top_n: int = 5,
    max_pcs_per_family: int = 1,
) -> Optional[pd.DataFrame]:
    """
    For each family (group of columns sharing the same prefix), do a *local* PCA:

      - Take all columns in that family (original metrics, e.g. dist_term_min_1_x, dist_term_min_2_x, ...).
      - Z-score within the family.
      - PCA in that family alone.
      - For the first `max_pcs_per_family` PCs, list top `top_n` original metrics by |loading|.

    Returns one big table:

        Family | PC | Feature | Loading

    This tells you: which original metrics build each family's 'family vector'.
    """
    if Xdf is None or not len(colnames):
        return None
    if Xdf.shape[1] != len(colnames):
        return None

    X_raw = Xdf.to_numpy(dtype=float)
    fam_idx = _family_groups(colnames)

    rows: List[tuple] = []

    for fam, idxs in fam_idx.items():
        if len(idxs) == 0:
            continue

        Xf = X_raw[:, idxs]
        # Z-score within family
        mu = np.nanmean(Xf, axis=0)
        sd = np.nanstd(Xf, axis=0, ddof=1)
        sd = np.where(sd == 0, 1.0, sd)
        Zf = (Xf - mu) / sd

        # PCA within this family
        try:
            U, s, Vt = np.linalg.svd(np.nan_to_num(Zf), full_matrices=False)
        except Exception:
            continue
        if not s.size:
            continue

        evr = (s ** 2) / np.sum(s ** 2)
        # keep PCs with EVR >= 0.1%, but at least 1
        mask = evr >= 1e-3
        kpcs = int(np.sum(mask)) or 1
        kpcs = min(kpcs, max_pcs_per_family, Vt.shape[0])

        fam_colnames = [colnames[j] for j in idxs]

        for p in range(kpcs):
            load = Vt[p, :]
            # sort metrics in this family by |loading|
            idx_sorted = np.argsort(-np.abs(load))
            for j in idx_sorted[:top_n]:
                feat = fam_colnames[j]
                rows.append((fam, f"PC{p+1}", feat, float(load[j])))

    if not rows:
        return None

    df = pd.DataFrame(rows, columns=["Family", "PC", "Feature", "Loading"])
    return df
