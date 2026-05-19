from .shared import *
from .figures import _error_fig


def _embedding_coord_columns(df: pd.DataFrame, dims: int | None = None) -> list[str]:
    cols = [c for c in ("UMAP1", "UMAP2", "UMAP3") if c in getattr(df, "columns", [])]
    if dims is not None:
        cols = cols[: max(2, min(int(dims or 2), 3))]
    return cols


def _composite_path_coordinate(df: pd.DataFrame, mode: str, dims: int | None = None) -> tuple[pd.Series, str]:
    """Return a scalar path coordinate through the embedding.

    Axis modes use raw displayed coordinates. Composite modes (XY/XZ/YZ/XYZ)
    standardize requested coordinates and use PC1, with deterministic sign
    aligned to the first requested coordinate.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.Series(dtype=float), "no embedding"
    mode = str(mode or "axis1").strip().lower()
    axis_map = {"axis1": ["UMAP1"], "x": ["UMAP1"], "axis2": ["UMAP2"], "y": ["UMAP2"], "axis3": ["UMAP3"], "z": ["UMAP3"]}
    comp_map = {"xy": ["UMAP1", "UMAP2"], "xz": ["UMAP1", "UMAP3"], "yz": ["UMAP2", "UMAP3"], "xyz": ["UMAP1", "UMAP2", "UMAP3"]}
    if mode in axis_map:
        col = axis_map[mode][0]
        if col not in df.columns:
            return pd.Series(np.nan, index=df.index, dtype=float), f"{col} unavailable"
        return pd.to_numeric(df[col], errors="coerce"), col.replace("UMAP", "Axis ")
    cols = comp_map.get(mode, ["UMAP1"])
    cols = [c for c in cols if c in df.columns]
    if len(cols) == 1:
        return pd.to_numeric(df[cols[0]], errors="coerce"), cols[0].replace("UMAP", "Axis ")
    if len(cols) < 1:
        return pd.Series(np.nan, index=df.index, dtype=float), "no composite coordinates"
    X = df[cols].apply(pd.to_numeric, errors="coerce")
    good = X.notna().all(axis=1)
    coord = pd.Series(np.nan, index=df.index, dtype=float)
    if good.sum() < 2:
        return coord, "+".join(c.replace("UMAP", "") for c in cols)
    Xg = X.loc[good].to_numpy(dtype=float)
    mu = np.nanmean(Xg, axis=0)
    sd = np.nanstd(Xg, axis=0, ddof=1)
    sd = np.where(np.isfinite(sd) & (sd > 0), sd, 1.0)
    Z = (Xg - mu) / sd
    try:
        _, _, Vt = np.linalg.svd(Z, full_matrices=False)
        pc = Z @ Vt[0, :]
        # deterministic orientation: positive correlation with first requested coordinate
        if np.corrcoef(pc, Z[:, 0])[0, 1] < 0:
            pc = -pc
        coord.loc[good] = pc
    except Exception:
        coord.loc[good] = Z[:, 0]
    return coord, f"PC1({'+'.join(c.replace('UMAP','') for c in cols)})"




def _embedding_coords_matrix(df: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    cols = [c for c in ("UMAP1", "UMAP2", "UMAP3") if c in getattr(df, "columns", [])]
    if len(cols) < 2:
        return np.zeros((0, 0), dtype=float), []
    X = df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    return X, cols


def _project_to_polyline(points: np.ndarray, anchors_xyz: np.ndarray) -> np.ndarray:
    """Project each point to nearest segment of an ordered polyline; return cumulative arc coordinate."""
    P = np.asarray(points, dtype=float)
    A = np.asarray(anchors_xyz, dtype=float)
    out = np.full(P.shape[0], np.nan, dtype=float)
    if P.ndim != 2 or A.ndim != 2 or P.shape[1] != A.shape[1] or A.shape[0] < 2:
        return out
    seg = A[1:] - A[:-1]
    lens = np.linalg.norm(seg, axis=1)
    cum = np.concatenate([[0.0], np.cumsum(lens)])
    good_seg = lens > 0
    if not np.any(good_seg):
        return out
    for i, pnt in enumerate(P):
        if not np.all(np.isfinite(pnt)):
            continue
        best_d2 = np.inf
        best_s = np.nan
        for j in range(seg.shape[0]):
            if not good_seg[j]:
                continue
            v = seg[j]
            t = float(np.dot(pnt - A[j], v) / max(np.dot(v, v), 1e-12))
            t = min(1.0, max(0.0, t))
            q = A[j] + t * v
            d2 = float(np.sum((pnt - q) ** 2))
            if d2 < best_d2:
                best_d2 = d2
                best_s = cum[j] + t * lens[j]
        out[i] = best_s
    return out


def _shortest_knn_path(coords: np.ndarray, start_i: int, end_i: int, k: int = 12) -> list[int]:
    """Tiny dependency-free Dijkstra over a symmetric kNN graph."""
    import heapq
    X = np.asarray(coords, dtype=float)
    n = X.shape[0]
    if n < 2 or start_i < 0 or end_i < 0 or start_i >= n or end_i >= n:
        return []
    finite_rows = np.isfinite(X).all(axis=1)
    if not finite_rows[start_i] or not finite_rows[end_i]:
        return []
    k = max(2, min(int(k or 12), n - 1))
    # Brute-force distances are acceptable for dashboard pathing; cap fallback by using vectorized rows.
    D = np.linalg.norm(X[:, None, :] - X[None, :, :], axis=2)
    D[~np.isfinite(D)] = np.inf
    neigh = np.argsort(D, axis=1)[:, 1:k+1]
    adj = [[] for _ in range(n)]
    for i in range(n):
        for j in neigh[i]:
            w = float(D[i, j])
            if np.isfinite(w):
                adj[i].append((int(j), w))
                adj[int(j)].append((i, w))
    dist = [float('inf')] * n
    prev = [-1] * n
    dist[start_i] = 0.0
    pq = [(0.0, start_i)]
    seen = set()
    while pq:
        di, i = heapq.heappop(pq)
        if i in seen:
            continue
        seen.add(i)
        if i == end_i:
            break
        for j, w in adj[i]:
            nd = di + w
            if nd < dist[j]:
                dist[j] = nd
                prev[j] = i
                heapq.heappush(pq, (nd, j))
    if not np.isfinite(dist[end_i]):
        return []
    path = []
    cur = end_i
    while cur != -1:
        path.append(cur)
        if cur == start_i:
            break
        cur = prev[cur]
    return list(reversed(path)) if path and path[-1] == end_i else []


def _trajectory_path_coordinate(df: pd.DataFrame, mode: str, anchors: Optional[Sequence[str]] = None, feature_col: Optional[str] = None) -> tuple[pd.Series, str]:
    mode = str(mode or "axis1").strip().lower()
    if mode in {"feature", "physical", "physical_feature", "scalar", "value"}:
        col = str(feature_col or "").strip()
        if not col or col not in getattr(df, "columns", []):
            return pd.Series(np.nan, index=getattr(df, "index", None), dtype=float), "feature unavailable"
        coord = pd.to_numeric(df[col], errors="coerce")
        return coord, f"feature: {prettify_column_label(col)}"
    if mode not in {"line", "custom_line", "polyline", "manual_polyline", "geodesic", "knn_geodesic"}:
        return _composite_path_coordinate(df, mode, None)
    if not isinstance(df, pd.DataFrame) or df.empty or "variant" not in df.columns:
        return pd.Series(dtype=float), "no embedding"
    X, cols = _embedding_coords_matrix(df)
    coord = pd.Series(np.nan, index=df.index, dtype=float)
    if X.size == 0 or len(cols) < 2:
        return coord, "missing embedding coordinates"
    variants = df["variant"].astype(str).tolist()
    v_to_pos = {v: i for i, v in enumerate(variants)}
    anchor_vars = [str(a) for a in (anchors or []) if str(a) in v_to_pos]
    if len(anchor_vars) < 2:
        # Graceful fallback so the panel remains usable without anchors.
        return _composite_path_coordinate(df, "axis1", None)
    anchor_idx = [v_to_pos[a] for a in anchor_vars]
    if mode in {"geodesic", "knn_geodesic"}:
        path_idx = _shortest_knn_path(X, anchor_idx[0], anchor_idx[-1], k=12)
        if len(path_idx) >= 2:
            anchor_idx = path_idx
            anchor_vars = [variants[i] for i in path_idx]
        else:
            anchor_idx = [anchor_idx[0], anchor_idx[-1]]
    elif mode in {"line", "custom_line"}:
        anchor_idx = [anchor_idx[0], anchor_idx[1]]
        anchor_vars = [anchor_vars[0], anchor_vars[1]]
    # polyline uses provided ordered anchors
    A = X[anchor_idx, :]
    if A.shape[0] < 2 or not np.isfinite(A).all():
        return _composite_path_coordinate(df, "axis1", None)
    s = _project_to_polyline(X, A)
    coord.loc[:] = s
    label = {
        "line": "line anchors",
        "custom_line": "line anchors",
        "polyline": "polyline anchors",
        "manual_polyline": "polyline anchors",
        "geodesic": "kNN geodesic anchors",
        "knn_geodesic": "kNN geodesic anchors",
    }.get(mode, "anchor path")
    return coord, f"{label}: {anchor_vars[0]}→{anchor_vars[-1]}"

def _axis_bins_for_values(values: pd.Series, n_bins: int) -> list[np.ndarray]:
    vals = pd.to_numeric(values, errors="coerce")
    valid_idx = np.asarray(vals.dropna().sort_values(kind="mergesort").index.tolist())
    if valid_idx.size == 0:
        return []
    n_bins = max(2, min(int(n_bins or 8), min(80, int(valid_idx.size))))
    return [np.asarray(x, dtype=object) for x in np.array_split(valid_idx, n_bins) if len(x)]
