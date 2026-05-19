"""
peptide_dash/tabs/pmf_dendrogram_tab.py

PMF Dendrogram (no UMAP) + Typical PMF reconstructed per selected cluster

Key behaviors
-------------
- Cluster variants by PMF-shape similarity (Jensen–Shannon distance on probability mass vectors).
- Average distances across selected metrics (distance metrics).
- Hierarchical linkage + cut height → clusters.
- For the selected cluster:
    (A) Cluster-conditioned PMF for user-selected plot metrics.
    (B) Typical PMF reconstructed for the *distance metrics* (NEW).
- Dendrogram always shows variant names on hover (leaf markers), and axis labels when feasible.

Important fix
-------------
dcc.Store JSON-serializes dict keys as strings.
We store clusters with string keys and always look up using str(cluster_id).
This prevents "Cluster has no members" when cluster_id is an int.
"""
from __future__ import annotations

from ..metrics import metric_display_label, torsion_sort_key

from typing import Any, Dict, List, Optional, Sequence, Tuple
from ..analysis.pmf_vectorize import stable_grid_from_xs, density_to_mass_on_grid
from .shared import (
    PmfCols, pmf_panel as _panel, R_GAS,
    kT as _kT, bin_dx as _bin_dx, normalize_density as _normalize_density,
    infer_features_df as _infer_features_df, infer_weight_columns as _infer_weight_columns,
    metric_options,
)
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, dash_table

from .pmf_plot_ci import (
    pmf_error_fig as _error_fig,
    pmf_overlay_fig as _shared_pmf_overlay_fig,
    per_variant_raw_pmf_with_ci as _shared_per_variant_raw_pmf_with_ci,
    pmf_has_ci_columns as _shared_pmf_has_ci_columns,
)

try:
    from scipy.cluster.hierarchy import linkage as scipy_linkage  # type: ignore
    from scipy.spatial.distance import pdist as scipy_pdist  # type: ignore

    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False

try:
    from sklearn.cluster import AgglomerativeClustering  # type: ignore

    HAVE_SKLEARN = True
except Exception:
    HAVE_SKLEARN = False

TAB_LABEL = "PMF Dendrogram"
TAB_VALUE = "pmf_dendrogram"

def _to_prob_density(
    df_variant_metric: pd.DataFrame,
    cols: PmfCols,
    use_repr: str,
    energy_units: str,
    T_K: float,
    preserve_norm: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    dv = df_variant_metric.copy()
    dv[cols.x] = pd.to_numeric(dv[cols.x], errors="coerce")
    dv = dv.dropna(subset=[cols.x]).sort_values(cols.x)
    if dv.empty:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=float)

    x = dv[cols.x].to_numpy(dtype=float)

    if use_repr == "P" and cols.p in dv.columns:
        p = pd.to_numeric(dv[cols.p], errors="coerce").to_numpy(dtype=float)
        p = np.clip(p, 0.0, None)
    else:
        if cols.f in dv.columns:
            F = pd.to_numeric(dv[cols.f], errors="coerce").to_numpy(dtype=float)
            F0 = np.nanmin(F) if np.isfinite(np.nanmin(F)) else 0.0
            kt = max(_kT(energy_units, T_K, None), 1e-12)
            p = np.exp(-(F - F0) / kt)
        elif cols.p in dv.columns:
            p = pd.to_numeric(dv[cols.p], errors="coerce").to_numpy(dtype=float)
            p = np.clip(p, 0.0, None)
        else:
            return x, np.zeros_like(x, dtype=float)

    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    if preserve_norm:
        p = _normalize_density(x, p)
    return x, p

def _stable_grid_from_xs(xs: np.ndarray, max_bins: int, decimals: int = 8) -> np.ndarray:
    """
    Build a stable x-grid for a metric.

    Why:
    - Raw float x bin centers can differ by tiny jitters across variants, causing
      misalignment and artificial distances.
    - We round, infer a representative dx, and construct a regularized grid.
    """
    xs = np.asarray(xs, dtype=float)
    xs = xs[np.isfinite(xs)]
    if xs.size == 0:
        return np.linspace(0.0, 1.0, max(2, int(max_bins)))

    xs = np.round(xs, int(decimals))
    u = np.unique(np.sort(xs))
    if u.size <= 1:
        return u

    diffs = np.diff(u)
    diffs = diffs[diffs > 0]
    if diffs.size == 0:
        return u[: int(max_bins)]

    dx = float(np.median(diffs))
    if not np.isfinite(dx) or dx <= 0:
        return u[: int(max_bins)]

    x0 = float(u.min())
    x1 = float(u.max())
    n = int(round((x1 - x0) / dx)) + 1
    n = max(2, n)

    if n > int(max_bins):
        # Too many bins for a strict dx grid → regularize to max_bins points.
        return np.linspace(x0, x1, int(max_bins))

    grid = x0 + dx * np.arange(n, dtype=float)
    return np.round(grid, int(decimals))

def _metric_grid(pmf_df: pd.DataFrame, cols: PmfCols, metric: str, max_bins: int) -> np.ndarray:
    sub = pmf_df[pmf_df[cols.metric].astype(str) == str(metric)]
    xs = (
        pd.to_numeric(sub[cols.x], errors="coerce").dropna().to_numpy(dtype=float)
        if not sub.empty
        else np.array([], dtype=float)
    )
    # Shared stable grid (collapses float jitter + caps bin explosions).
    return stable_grid_from_xs(xs, max_bins=int(max_bins), round_decimals=None)

def _density_to_mass_on_grid(x_grid: np.ndarray, x: np.ndarray, p: np.ndarray) -> np.ndarray:
    # Shared conversion: interpolate density, convert to mass via dx, normalize to sum=1.
    return density_to_mass_on_grid(x_grid, x, p, dirichlet_alpha=0.0)

def _js_distance_condensed_numpy(P: np.ndarray) -> np.ndarray:
    N, _B = P.shape
    out = np.empty(N * (N - 1) // 2, dtype=np.float64)
    eps = 1e-12
    k = 0
    for i in range(N - 1):
        Pi = np.clip(P[i], eps, None)
        for j in range(i + 1, N):
            Qj = np.clip(P[j], eps, None)
            M = 0.5 * (Pi + Qj)
            js = 0.5 * np.sum(Pi * np.log(Pi / M)) + 0.5 * np.sum(Qj * np.log(Qj / M))
            out[k] = float(np.sqrt(max(js, 0.0)))
            k += 1
    return out

def _condensed_to_square(d: np.ndarray, n: int) -> np.ndarray:
    D = np.zeros((n, n), dtype=np.float64)
    k = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            D[i, j] = d[k]
            D[j, i] = d[k]
            k += 1
    return D

def _build_distributions(
    pmf_df: pd.DataFrame,
    cols: PmfCols,
    metrics: Sequence[str],
    use_repr: str,
    energy_units: str,
    T_K: float,
    max_bins: int,
    max_variants: int,
    variant_policy: str = "intersection",
) -> Tuple[List[str], Dict[str, np.ndarray], Dict[str, int]]:
    if pmf_df.empty or not metrics:
        return [], {}, {}

    variant_sets: list[set[str]] = []
    variant_union: set[str] = set()
    for m in metrics:
        sub = pmf_df[pmf_df[cols.metric].astype(str) == str(m)]
        if sub.empty:
            continue
        vs = set(sub[cols.variant].astype(str).dropna().tolist())
        if vs:
            variant_sets.append(vs)
            variant_union |= vs

    policy = str(variant_policy or "intersection").lower().strip()

    if policy == "intersection" and variant_sets:
        variants = sorted(set.intersection(*variant_sets))
    else:
        variants = sorted(variant_union)

    if not variants:
        return [], {}, {}

    if len(variants) > max_variants:
        variants = variants[:max_variants]

    P_by_metric: Dict[str, np.ndarray] = {}
    coverage: Dict[str, int] = {}

    for m in metrics:
        x_grid = _metric_grid(pmf_df, cols, str(m), max_bins=max_bins)
        P = np.zeros((len(variants), x_grid.size), dtype=np.float64)
        present_mask = np.zeros(len(variants), dtype=bool)
        cover = 0

        sub = pmf_df[pmf_df[cols.metric].astype(str) == str(m)].copy()
        if not sub.empty:
            sub[cols.variant] = sub[cols.variant].astype(str)
            groups = {k: g for k, g in sub.groupby(cols.variant)}
            for i, v in enumerate(variants):
                g = groups.get(v)
                if g is None or g.empty:
                    P[i, :] = 1.0 / max(1, x_grid.size)
                    continue
                x, p = _to_prob_density(g, cols, use_repr, energy_units, T_K, preserve_norm=True)
                if x.size == 0:
                    P[i, :] = 1.0 / max(1, x_grid.size)
                    continue
                P[i, :] = _density_to_mass_on_grid(x_grid, x, p)
                present_mask[i] = True
                cover += 1

        if policy in {"union_mean", "union-mean", "unionmean"} and present_mask.any() and (~present_mask).any():
            mean_vec = np.mean(P[present_mask], axis=0)
            P[~present_mask, :] = mean_vec

        eps = 1e-12
        P = np.clip(P, eps, None)
        P = P / np.sum(P, axis=1, keepdims=True)

        P_by_metric[str(m)] = P
        coverage[str(m)] = int(cover)

    return variants, P_by_metric, coverage

def _avg_js_distance_condensed(P_by_metric: Dict[str, np.ndarray]) -> np.ndarray:
    metrics = [k for k, v in P_by_metric.items() if isinstance(v, np.ndarray) and v.size > 0]
    if not metrics:
        return np.zeros((0,), dtype=float)

    if HAVE_SCIPY:
        acc = None
        for m in metrics:
            d = scipy_pdist(P_by_metric[m], metric="jensenshannon")
            acc = d if acc is None else (acc + d)
        return acc / float(len(metrics))

    acc = None
    for m in metrics:
        d = _js_distance_condensed_numpy(P_by_metric[m])
        acc = d if acc is None else (acc + d)
    return acc / float(len(metrics))

def _linkage_from_dist(d_condensed: np.ndarray, n: int, method: str) -> np.ndarray:
    if HAVE_SCIPY:
        Z = scipy_linkage(d_condensed, method=method)
        return np.asarray(Z, dtype=float)

    if not HAVE_SKLEARN:
        raise RuntimeError("Need scipy or scikit-learn to compute linkage.")

    D = _condensed_to_square(d_condensed, n)
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.0,
        metric="precomputed",
        linkage=method,
        compute_distances=True,
    )
    model.fit(D)

    children = np.asarray(model.children_, dtype=int)
    distances = np.asarray(getattr(model, "distances_", None), dtype=float)
    if distances is None or distances.size == 0:
        raise RuntimeError("AgglomerativeClustering did not provide distances_ (upgrade scikit-learn).")

    counts = np.zeros(children.shape[0], dtype=float)
    for i, (a, b) in enumerate(children):
        ca = 1.0 if a < n else counts[a - n]
        cb = 1.0 if b < n else counts[b - n]
        counts[i] = ca + cb

    return np.column_stack([children, distances, counts]).astype(float)

def _leaf_order_from_Z(Z: np.ndarray, n: int) -> List[int]:
    if n <= 1:
        return [0]
    root = n + (n - 2)

    def rec(node: int) -> List[int]:
        if node < n:
            return [node]
        i = node - n
        return rec(int(Z[i, 0])) + rec(int(Z[i, 1]))

    return rec(root)

def _truncate(s: str, n: int) -> str:
    s = str(s)
    if n <= 0 or len(s) <= n:
        return s
    if n <= 1:
        return "…"
    return s[: n - 1] + "…"

def _dendrogram_fig(
    Z: np.ndarray,
    labels: List[str],
    show_axis_labels: bool,
    max_axis_labels: int,
    truncate_to: int,
    search: str,
) -> go.Figure:
    if Z.size == 0 or not labels:
        return _error_fig("No dendrogram available.")

    n = len(labels)
    order = _leaf_order_from_Z(Z, n)

    node_x: Dict[int, float] = {leaf: float(i) for i, leaf in enumerate(order)}
    node_y: Dict[int, float] = {leaf: 0.0 for leaf in range(n)}

    xs: List[Optional[float]] = []
    ys: List[Optional[float]] = []

    for i in range(n - 1):
        a = int(Z[i, 0])
        b = int(Z[i, 1])
        h = float(Z[i, 2])
        node = n + i

        xa, ya = node_x[a], node_y[a]
        xb, yb = node_x[b], node_y[b]

        xs += [xa, xa, None]
        ys += [ya, h, None]
        xs += [xb, xb, None]
        ys += [yb, h, None]
        xs += [xa, xb, None]
        ys += [h, h, None]

        node_x[node] = 0.5 * (xa + xb)
        node_y[node] = h

    leaf_labels_full = [labels[i] for i in order]
    q = (search or "").strip().lower()
    hits = [(q in str(lbl).lower()) if q else False for lbl in leaf_labels_full]
    marker_sizes = [10 if h else 6 for h in hits]

    leaf_x = list(range(n))
    leaf_y = [0.0] * n

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="linkage"))
    fig.add_trace(
        go.Scatter(
            x=leaf_x,
            y=leaf_y,
            mode="markers",
            name="variants",
            marker=dict(size=marker_sizes, opacity=0.85),
            customdata=leaf_labels_full,
            hovertemplate="variant=%{customdata}<br>leaf=%{x}<extra></extra>",
        )
    )

    show_ticks = bool(show_axis_labels) and (n <= int(max_axis_labels))
    ticktext = [_truncate(lbl, int(truncate_to)) for lbl in leaf_labels_full]

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=20, t=40, b=140 if show_ticks else 60),
        title="PMF-space dendrogram (Jensen–Shannon distance)",
        legend=dict(orientation="h"),
    )
    fig.update_xaxes(
        title="variant order",
        tickmode="array",
        tickvals=leaf_x if show_ticks else [],
        ticktext=ticktext if show_ticks else [],
        tickangle=90 if show_ticks else 0,
        showticklabels=show_ticks,
    )
    fig.update_yaxes(title="distance")
    return fig

def _clusters_at_cut(Z: np.ndarray, n: int, cut: float) -> List[List[int]]:
    if n <= 1:
        return [[0]]
    root = n + (n - 2)

    def leaves(node: int) -> List[int]:
        if node < n:
            return [node]
        i = node - n
        return leaves(int(Z[i, 0])) + leaves(int(Z[i, 1]))

    def rec(node: int) -> List[List[int]]:
        if node < n:
            return [[node]]
        i = node - n
        dist = float(Z[i, 2])
        if dist <= cut:
            return [leaves(node)]
        return rec(int(Z[i, 0])) + rec(int(Z[i, 1]))

    clusters = [sorted(set(c)) for c in rec(root)]
    clusters.sort(key=len, reverse=True)
    return clusters

def _pmf_fig(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    output_kind: str,
    title: str,
    *,
    ci_bands: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
) -> go.Figure:
    """Thin shim over the shared overlay helper; kept for call-site stability."""
    y_title = "P" if output_kind == "P" else "F"
    return _shared_pmf_overlay_fig(
        curves,
        title=title,
        y_title=y_title,
        ci_bands=ci_bands,
        error_text="No PMF to plot.",
    )

def _per_variant_raw_pmf_with_ci(
    pmf_df: pd.DataFrame,
    cols: PmfCols,
    members: Sequence[str],
    metric: str,
    max_variants: int = 10,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """Thin shim over the shared helper, preserving the original signature."""
    return _shared_per_variant_raw_pmf_with_ci(
        pmf_df,
        variants=members,
        metric=metric,
        variant_col=cols.variant,
        metric_col=cols.metric,
        x_col=cols.x,
        f_col=cols.f,
        max_variants=max_variants,
    )

def _conditional_pmf(
    pmf_df: pd.DataFrame,
    cols: PmfCols,
    members: Sequence[str],
    metrics: Sequence[str],
    weights: Dict[str, float],
    use_repr: str,
    energy_units: str,
    T_K: float,
    kT_override: Optional[float],
    preserve_variant_norm: bool,
    output_kind: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if pmf_df.empty or not members or not metrics:
        return {}

    kt = max(_kT(energy_units, float(T_K), kT_override), 1e-12)

    df = pmf_df[
        pmf_df[cols.variant].astype(str).isin(list(map(str, members)))
        & pmf_df[cols.metric].astype(str).isin(list(map(str, metrics)))
    ].copy()
    if df.empty:
        return {}

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for m in metrics:
        dm = df[df[cols.metric].astype(str) == str(m)]
        if dm.empty:
            continue

        per_variant: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        grids: list[np.ndarray] = []

        for v, dv in dm.groupby(cols.variant):
            x, p = _to_prob_density(dv, cols, use_repr, energy_units, float(T_K), preserve_variant_norm)
            if x.size == 0:
                continue
            per_variant[str(v)] = (x, np.clip(p, 0.0, None))
            grids.append(x)

        if not per_variant:
            continue

        x_grid = np.unique(np.concatenate(grids))
        if x_grid.size < 2:
            continue

        keys = list(per_variant.keys())
        w = np.array([max(0.0, float(weights.get(v, 1.0))) for v in keys], dtype=float)
        if not np.isfinite(w).all() or float(np.sum(w)) <= 0:
            w = np.ones_like(w)
        w = w / float(np.sum(w))

        Pmass = np.zeros_like(x_grid, dtype=float)
        for (v, (xv, pv)), wi in zip(per_variant.items(), w):
            Pmass += float(wi) * _density_to_mass_on_grid(x_grid, xv, pv)

        dx = float(np.median(np.diff(x_grid))) if x_grid.size >= 2 else 1.0
        Pdens = Pmass / max(dx, 1e-12)

        if output_kind == "P":
            out[str(m)] = (x_grid, Pdens)
        else:
            p = np.clip(Pmass, 1e-300, None)
            F = -kt * np.log(p)
            F = F - np.nanmin(F)
            out[str(m)] = (x_grid, F)

    return out

def _cluster_cohesion_and_medoid(
    pmf_df: pd.DataFrame,
    cols: PmfCols,
    members: List[str],
    dist_metrics: List[str],
    use_repr: str,
    energy_units: str,
    T_K: float,
    max_bins: int,
    variant_policy: str,
) -> Tuple[Optional[str], Optional[float]]:
    if len(members) == 0:
        return None, None
    if len(members) == 1:
        return members[0], 0.0
    if len(members) > 250:
        return None, None

    sub = pmf_df[pmf_df[cols.variant].astype(str).isin(members)].copy()
    if sub.empty:
        return None, None

    variants, P_by_metric, _cov = _build_distributions(
        pmf_df=sub,
        cols=cols,
        metrics=dist_metrics,
        use_repr=use_repr,
        energy_units=energy_units,
        T_K=T_K,
        max_bins=max_bins,
        max_variants=len(members),
        variant_policy=str(variant_policy or 'intersection'),
    )
    if not variants:
        return None, None

    d = _avg_js_distance_condensed(P_by_metric)
    if d.size == 0:
        return None, None

    D = _condensed_to_square(d, len(variants))
    cohesion = float(np.sum(D) / (len(variants) * (len(variants) - 1)))
    avg_to_others = np.sum(D, axis=1) / np.maximum(1, (len(variants) - 1))
    medoid_idx = int(np.argmin(avg_to_others))
    return variants[medoid_idx], cohesion

def layout(ctx: Any) -> html.Div:
    cols = PmfCols()
    pmf_df = pd.DataFrame()  # loaded lazily by callbacks
    features_df = _infer_features_df(ctx)
    weight_cols = _infer_weight_columns(features_df)

    metrics: list[str] = []
    if isinstance(pmf_df, pd.DataFrame) and not pmf_df.empty and cols.metric in pmf_df.columns:
        metrics = sorted(pmf_df[cols.metric].astype(str).dropna().unique().tolist(), key=torsion_sort_key)

    dep_msg = None
    if not HAVE_SCIPY and not HAVE_SKLEARN:
        dep_msg = "Install: `pip install scipy` (preferred) or `pip install scikit-learn`."

    method_text = dcc.Markdown(
        """
**Method**
- Normalize each per-variant PMF into a probability distribution (per metric).
- Compute pairwise Jensen–Shannon distances on distributions; average across distance metrics.
- Build hierarchical clustering; cut height defines clusters.
- For a cluster: report cohesion (mean within-cluster JS) + medoid (representative).
- **Typical PMF reconstructed**: cluster-conditioned PMF using the *distance metrics*.
        """.strip()
    )

    return html.Div(
        className="tab-body-inner",
        children=[
            _panel(
                "PMF Dendrogram (no UMAP)",
                dep_msg,
                [
                    html.Div(f"pmf_df rows: {len(pmf_df) if isinstance(pmf_df, pd.DataFrame) else 0}"),
                    html.Div(f"pmf metrics: {len(metrics)}"),
                    html.Div(f"backend: {'scipy' if HAVE_SCIPY else ('sklearn' if HAVE_SKLEARN else 'none')}"),
                    html.Hr(),
                    method_text,
                ],
            ),
            html.Div(
                style={"display": "flex", "gap": "14px", "flexWrap": "wrap"},
                children=[
                    html.Div(
                        style={"minWidth": "320px", "flex": "0 0 360px"},
                        children=[
                            _panel(
                                "Distance & dendrogram",
                                "Variant names: hover always; axis labels when feasible.",
                                [
                                    html.Label("Metrics for dendrogram distance"),
                                    dcc.Dropdown(
                                        id="pdend-metrics",
                                        options=[{"label": metric_display_label(m), "value": m} for m in metrics],
                                        value=metrics[:4],
                                        multi=True,
                                    ),

html.Div(style={"height": "10px"}),
html.Label("Variants missing distance metrics"),
dcc.Dropdown(
    id="pdend-variant-policy",
    options=[
        {"label": "Intersection (drop incomplete variants)", "value": "intersection"},
        {"label": "Union (fill missing as uniform)", "value": "union"},
        {"label": "Union (impute missing as mean)", "value": "union_mean"},
    ],
    value="intersection",
    clearable=False,
),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Representation"),
                                    dcc.Dropdown(
                                        id="pdend-repr",
                                        options=[
                                            {"label": "Use P when available", "value": "P"},
                                            {"label": "Use F→P when needed", "value": "F"},
                                        ],
                                        value="P" if (isinstance(pmf_df, pd.DataFrame) and cols.p in pmf_df.columns) else "F",
                                        clearable=False,
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Energy units (if using F)"),
                                    dcc.Dropdown(
                                        id="pdend-energy-units",
                                        options=[{"label": "kJ/mol", "value": "kJ/mol"}, {"label": "kT", "value": "kT"}],
                                        value="kJ/mol" if (isinstance(pmf_df, pd.DataFrame) and cols.f in pmf_df.columns) else "kT",
                                        clearable=False,
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Temperature (K)"),
                                    dcc.Input(id="pdend-temp-k", type="number", value=300.0, min=1, step=1),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Max bins per metric"),
                                    dcc.Input(id="pdend-max-bins", type="number", value=128, min=16, step=8),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Max variants (cap)"),
                                    dcc.Input(id="pdend-max-variants", type="number", value=400, min=50, step=50),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Linkage method"),
                                    dcc.Dropdown(
                                        id="pdend-linkage",
                                        options=[
                                            {"label": "average", "value": "average"},
                                            {"label": "complete", "value": "complete"},
                                            {"label": "single", "value": "single"},
                                        ],
                                        value="average",
                                        clearable=False,
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Leaf labels"),
                                    dcc.Checklist(
                                        id="pdend-show-axis-labels",
                                        options=[{"label": "Show axis labels", "value": "yes"}],
                                        value=["yes"],
                                    ),
                                    html.Div(style={"height": "8px"}),
                                    html.Label("Max axis labels"),
                                    dcc.Input(id="pdend-max-axis-labels", type="number", value=80, min=10, step=10),
                                    html.Div(style={"height": "8px"}),
                                    html.Label("Truncate labels to (chars)"),
                                    dcc.Input(id="pdend-truncate", type="number", value=28, min=5, step=1),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Search (highlights leaves)"),
                                    dcc.Input(id="pdend-search", type="text", value="", placeholder="e.g., variant_42"),
                                    html.Div(style={"height": "12px"}),
                                    html.Button("Compute dendrogram", id="pdend-compute", className="btn-ghost"),
                                    html.Div(id="pdend-status", className="panel-subtitle", style={"marginTop": "10px"}),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Cut height"),
                                    dcc.Slider(
                                        id="pdend-cut",
                                        min=0.0,
                                        max=1.0,
                                        step=0.01,
                                        value=0.5,
                                        tooltip={"placement": "bottom", "always_visible": False},
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Cluster"),
                                    dcc.Dropdown(id="pdend-cluster", options=[], value=None, clearable=False),
                                ],
                            ),
                            _panel(
                                "Cluster-conditioned PMF",
                                None,
                                [
                                    html.Label("PMF metric(s) to plot"),
                                    dcc.Dropdown(
                                        id="pdend-pmf-metrics",
                                        options=[{"label": metric_display_label(m), "value": m} for m in metrics],
                                        value=metrics[:2],
                                        multi=True,
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Output"),
                                    dcc.RadioItems(
                                        id="pdend-output",
                                        options=[{"label": "F(x|cluster)", "value": "F"}, {"label": "P(x|cluster)", "value": "P"}],
                                        value="F",
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    dcc.Checklist(
                                        id="pdend-preserve-norm",
                                        options=[{"label": "Preserve per-variant normalization", "value": "yes"}],
                                        value=["yes"],
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Weights"),
                                    dcc.Dropdown(
                                        id="pdend-weight-mode",
                                        options=[{"label": "Uniform", "value": "uniform"}]
                                        + [{"label": f"Column: {c}", "value": f"col::{c}"} for c in weight_cols],
                                        value="uniform",
                                        clearable=False,
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("kT override (optional)"),
                                    dcc.Input(id="pdend-kt-override", type="number", value=None),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Typical PMF reconstructed (distance metrics) output"),
                                    dcc.RadioItems(
                                        id="pdend-typical-output",
                                        options=[{"label": "F(x|cluster)", "value": "F"}, {"label": "P(x|cluster)", "value": "P"}],
                                        value="F",
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        style={"flex": "1 1 720px", "minWidth": "360px"},
                        children=[
                            _panel("Dendrogram", None, [dcc.Graph(id="pdend-graph", figure=_error_fig("Compute dendrogram to begin."))]),
                            _panel("Cluster diagnostics", None, [html.Div(id="pdend-cluster-diag", className="panel-subtitle")]),
                            _panel("Conditional PMF (selected metrics)", None, [dcc.Graph(id="pdend-pmf-graph", figure=_error_fig("Pick a cluster."))]),
                            _panel("Typical PMF reconstructed (distance metrics)", None, [dcc.Graph(id="pdend-typical-graph", figure=_error_fig("Pick a cluster."))]),
                            _panel(
                                "Per-variant raw PMF with 95% CI",
                                "Shows each member's raw F(x) with replica-block bootstrap CI bands. "
                                "Limited to 10 variants and one metric for legibility. "
                                "Requires F_ci_lo_kJ_mol / F_ci_hi_kJ_mol columns produced by the batch pipeline.",
                                [
                                    html.Div(
                                        style={"marginBottom": "6px"},
                                        children=[
                                            html.Label("Metric for CI plot: ",
                                                       style={"marginRight": "6px", "fontSize": "12px"}),
                                            dcc.Dropdown(
                                                id="pdend-ci-metric",
                                                options=[],
                                                value=None,
                                                clearable=True,
                                                style={"maxWidth": "260px", "display": "inline-block",
                                                       "verticalAlign": "middle"},
                                            ),
                                        ],
                                    ),
                                    dcc.Graph(id="pdend-raw-ci-graph",
                                              figure=_error_fig("Pick a cluster and a metric.")),
                                ],
                            ),
                            _panel(
                                "Members",
                                None,
                                [
                                    dash_table.DataTable(
                                        id="pdend-members",
                                        columns=[{"name": "variant", "id": "variant"}, {"name": "weight", "id": "weight"}],
                                        data=[],
                                        page_size=12,
                                        style_table={"overflowX": "auto"},
                                        style_cell={"padding": "6px", "fontFamily": "var(--font-mono)", "fontSize": 12},
                                    )
                                ],
                            ),
                        ],
                    ),
                ],
            ),
            dcc.Store(id="pdend-store", data=None),
            dcc.Store(id="pdend-clusters-store", data=None),
        ],
    )

def register_callbacks(app: Any, ctx: Any) -> None:
    from .shared import apply_theme
    cols = PmfCols()
    features_df = _infer_features_df(ctx)

    @app.callback(
        Output("pdend-store", "data"),
        Output("pdend-cut", "min"),
        Output("pdend-cut", "max"),
        Output("pdend-cut", "value"),
        Output("pdend-cut", "step"),
        Output("pdend-status", "children"),
        Input("pdend-compute", "n_clicks"),
        State("pdend-metrics", "value"),
        State("pdend-variant-policy", "value"),
        State("pdend-repr", "value"),
        State("pdend-energy-units", "value"),
        State("pdend-temp-k", "value"),
        State("pdend-max-bins", "value"),
        State("pdend-max-variants", "value"),
        State("pdend-linkage", "value"),
        prevent_initial_call=True,
    )
    def _compute(
        _n: Optional[int],
        metrics: Optional[list[str]],
        variant_policy: str,
        use_repr: str,
        energy_units: str,
        temp_k: Optional[float],
        max_bins: Optional[int],
        max_variants: Optional[int],
        linkage_method: str,
    ):
        pmf_df = ctx.pmf_df  # lazy load
        if not isinstance(pmf_df, pd.DataFrame) or pmf_df.empty:
            msg = "pmf_df is empty; cannot compute dendrogram."
            return None, 0.0, 1.0, 0.5, 0.01, msg

        if not HAVE_SCIPY and not HAVE_SKLEARN:
            msg = "Install `scipy` (preferred) or `scikit-learn`."
            return None, 0.0, 1.0, 0.5, 0.01, msg

        metrics = list(map(str, metrics or []))
        if not metrics:
            msg = "Pick at least one metric for dendrogram distance."
            return None, 0.0, 1.0, 0.5, 0.01, msg

        T = float(temp_k) if temp_k is not None else 300.0
        mb = int(max_bins) if max_bins is not None else 128
        mv = int(max_variants) if max_variants is not None else 400
        method = str(linkage_method)

        try:
            variants, P_by_metric, _coverage = _build_distributions(
                pmf_df=pmf_df,
                cols=cols,
                metrics=metrics,
                use_repr=use_repr,
                energy_units=energy_units,
                T_K=T,
                max_bins=mb,
                max_variants=mv,
                variant_policy=str(variant_policy or 'intersection'),
            )
            if not variants:
                msg = "No variants found for selected metrics."
                return None, 0.0, 1.0, 0.5, 0.01, msg

            d = _avg_js_distance_condensed(P_by_metric)
            if d.size == 0:
                msg = "Distance array empty."
                return None, 0.0, 1.0, 0.5, 0.01, msg

            Z = _linkage_from_dist(d, n=len(variants), method=method)
            dist_max = float(np.nanmax(Z[:, 2])) if Z.size else 1.0
            dist_max = dist_max if np.isfinite(dist_max) and dist_max > 0 else 1.0
            step = dist_max / 200.0
            cut_default = dist_max * 0.5

            cov_txt = ', '.join([f"{k}:{v}" for k, v in sorted(_coverage.items())])
            pol = str(variant_policy or 'intersection')
            msg = f"OK: n={len(variants)} metrics={len(metrics)} policy={pol} dist_max={dist_max:.3g} coverage=[{cov_txt}]"
            store = {
                "variants": variants,
                "labels": variants,  # variant names
                "Z": Z.tolist(),
                "dist_max": dist_max,
                "dist_metrics": metrics,
                "variant_policy": str(variant_policy or "intersection"),
                "coverage": dict(_coverage),
                "repr": use_repr,
                "energy_units": energy_units,
                "temp_k": T,
                "max_bins": mb,
            }
            return store, 0.0, dist_max, cut_default, step, msg
        except Exception as e:
            msg = f"Compute error: {e}"
            return None, 0.0, 1.0, 0.5, 0.01, msg

    @app.callback(
        Output("pdend-graph", "figure"),
        Input("pdend-store", "data"),
        Input("pdend-show-axis-labels", "value"),
        Input("pdend-max-axis-labels", "value"),
        Input("pdend-truncate", "value"),
        Input("pdend-search", "value"),
        Input("theme-store", "data"),
    )
    def _render_dendrogram(store, show_axis_labels, max_axis_labels, truncate_to, search, theme):
        if not store or "Z" not in store or "labels" not in store:
            return apply_theme(_error_fig("Compute dendrogram to begin."), theme)

        Z = np.asarray(store["Z"], dtype=float)
        labels = list(map(str, store["labels"]))
        return apply_theme(_dendrogram_fig(
            Z=Z,
            labels=labels,
            show_axis_labels=("yes" in (show_axis_labels or [])),
            max_axis_labels=int(max_axis_labels) if max_axis_labels not in (None, "") else 80,
            truncate_to=int(truncate_to) if truncate_to not in (None, "") else 28,
            search=str(search or ""),
        ), theme)

    @app.callback(
        Output("pdend-clusters-store", "data"),
        Output("pdend-cluster", "options"),
        Output("pdend-cluster", "value"),
        Input("pdend-store", "data"),
        Input("pdend-cut", "value"),
    )
    def _clusters(store: Optional[Dict[str, Any]], cut: Optional[float]):
        if not store or "variants" not in store or "Z" not in store:
            return None, [], None

        variants = list(map(str, store["variants"]))
        Z = np.asarray(store["Z"], dtype=float)
        if not variants or Z.size == 0:
            return None, [], None

        t = float(cut) if cut is not None else float(store.get("dist_max", 1.0)) * 0.5
        clusters_idx = _clusters_at_cut(Z, n=len(variants), cut=t)

        clusters: Dict[int, List[str]] = {}
        for i, leaf_list in enumerate(clusters_idx, start=1):
            clusters[i] = [variants[j] for j in leaf_list]

        items = sorted(clusters.items(), key=lambda kv: (-len(kv[1]), kv[0]))
        options = [{"label": f"Cluster {cid} (n={len(vs)})", "value": cid} for cid, vs in items]
        default = items[0][0] if items else None

        # IMPORTANT: store keys as strings (JSON-safe)
        clusters_json = {str(cid): vs for cid, vs in clusters.items()}
        return {"clusters": clusters_json, "cut": t}, options, default

    @app.callback(
        Output("pdend-pmf-graph", "figure"),
        Output("pdend-typical-graph", "figure"),
        Output("pdend-members", "data"),
        Output("pdend-cluster-diag", "children"),
        Input("pdend-store", "data"),
        Input("pdend-clusters-store", "data"),
        Input("pdend-cluster", "value"),
        Input("pdend-pmf-metrics", "value"),
        Input("pdend-weight-mode", "value"),
        Input("pdend-repr", "value"),
        Input("pdend-energy-units", "value"),
        Input("pdend-temp-k", "value"),
        Input("pdend-kt-override", "value"),
        Input("pdend-preserve-norm", "value"),
        Input("pdend-output", "value"),
        Input("pdend-typical-output", "value"),
        Input("theme-store", "data"),
        prevent_initial_call=True,
    )
    def _pmf(
        store: Optional[Dict[str, Any]],
        clusters_store: Optional[Dict[str, Any]],
        cluster_id: Optional[int],
        pmf_metrics: Optional[list[str]],
        weight_mode: str,
        use_repr: str,
        energy_units: str,
        temp_k: Optional[float],
        kt_override: Optional[float],
        preserve_norm: list[str],
        output_kind: str,
        typical_output: str,
        theme,
    ):
        pmf_df = ctx.pmf_df  # lazy load
        if not isinstance(pmf_df, pd.DataFrame) or pmf_df.empty:
            return apply_theme(_error_fig("pmf_df is empty."), theme), apply_theme(_error_fig("pmf_df is empty."), theme), [], ""

        if not clusters_store or "clusters" not in clusters_store or cluster_id is None:
            return apply_theme(_error_fig("Pick a cluster."), theme), apply_theme(_error_fig("Pick a cluster."), theme), [], ""

        # IMPORTANT: clusters stored with string keys
        key = str(int(cluster_id))
        members = clusters_store["clusters"].get(key, [])
        members = list(map(str, members))
        if not members:
            return apply_theme(_error_fig("Cluster has no members."), theme), apply_theme(_error_fig("Cluster has no members."), theme), [], ""

        metrics = list(map(str, pmf_metrics or []))

        weights: Dict[str, float] = {v: 1.0 for v in members}
        if weight_mode and weight_mode.startswith("col::"):
            col = weight_mode.split("::", 1)[1]
            if (
                isinstance(features_df, pd.DataFrame)
                and not features_df.empty
                and "variant" in features_df.columns
                and col in features_df.columns
            ):
                tmp = features_df[["variant", col]].copy()
                tmp["variant"] = tmp["variant"].astype(str)
                tmp[col] = pd.to_numeric(tmp[col], errors="coerce")
                tmp = tmp.dropna(subset=[col])
                wmap = tmp.set_index("variant")[col].to_dict()
                for v in members:
                    w = wmap.get(v)
                    if w is not None and np.isfinite(w):
                        weights[v] = float(w)

        preserve_variant_norm = "yes" in (preserve_norm or [])
        T = float(temp_k) if temp_k is not None else 300.0
        kto = float(kt_override) if kt_override not in (None, "") else None

        # Selected-metrics conditional PMF
        if metrics:
            curves = _conditional_pmf(
                pmf_df=pmf_df,
                cols=cols,
                members=members,
                metrics=metrics,
                weights=weights,
                use_repr=use_repr,
                energy_units=energy_units,
                T_K=T,
                kT_override=kto,
                preserve_variant_norm=preserve_variant_norm,
                output_kind=output_kind,
            )
            pmf_fig = apply_theme(_pmf_fig(curves, output_kind, "Cluster-conditioned PMF (selected metrics)"), theme)
        else:
            pmf_fig = apply_theme(_error_fig("Pick PMF metric(s) to plot."), theme)

        # Typical reconstructed using dendrogram distance metrics
        dist_metrics = list(map(str, store.get("dist_metrics", []))) if store else []
        if dist_metrics:
            typ_curves = _conditional_pmf(
                pmf_df=pmf_df,
                cols=cols,
                members=members,
                metrics=dist_metrics,
                weights=weights,
                use_repr=use_repr,
                energy_units=energy_units,
                T_K=T,
                kT_override=kto,
                preserve_variant_norm=preserve_variant_norm,
                output_kind=typical_output,
            )
            typical_fig = apply_theme(_pmf_fig(typ_curves, typical_output, "Typical PMF reconstructed (distance metrics)"), theme)
        else:
            typical_fig = apply_theme(_error_fig("Compute dendrogram first (distance metrics unknown)."), theme)

        # Diagnostics: medoid + cohesion
        diag = ""
        if store and dist_metrics:
            mb = int(store.get("max_bins", 128))
            medoid, cohesion = _cluster_cohesion_and_medoid(
                pmf_df=pmf_df,
                cols=cols,
                members=members,
                dist_metrics=dist_metrics,
                use_repr=str(store.get("repr", use_repr)),
                energy_units=str(store.get("energy_units", energy_units)),
                T_K=float(store.get("temp_k", T)),
                max_bins=mb,
                variant_policy=str(store.get("variant_policy", "intersection")),
            )
            if medoid is not None and cohesion is not None:
                diag = f"Cluster {cluster_id}: n={len(members)} | cohesion(mean JS)={cohesion:.4g} | medoid={medoid}"
            else:
                diag = f"Cluster {cluster_id}: n={len(members)} | cohesion/medoid skipped (cluster too large)."

        # Member weights table
        w = np.array([max(0.0, float(weights.get(v, 1.0))) for v in members], dtype=float)
        if not np.isfinite(w).all() or float(np.sum(w)) <= 0:
            w = np.ones_like(w)
        w = w / float(np.sum(w))
        table = [{"variant": v, "weight": float(wi)} for v, wi in zip(members[:800], w[:800])]

        return pmf_fig, typical_fig, table, diag

    # --- CI metric dropdown: populate from the currently selected cluster's
    # metrics that actually have bootstrap CI columns in pmf_df. This is
    # separate from the main PMF callback so the user can explore CI per-metric
    # without reclustering, and so that adding this feature doesn't touch
    # the return signature of _pmf.
    @app.callback(
        Output("pdend-ci-metric", "options"),
        Output("pdend-ci-metric", "value"),
        Input("pdend-clusters-store", "data"),
        Input("pdend-cluster", "value"),
        Input("pdend-pmf-metrics", "value"),
        State("pdend-ci-metric", "value"),
        prevent_initial_call=True,
    )
    def _populate_ci_metric(clusters_store, cluster_id, pmf_metrics, current):
        pmf_df = ctx.pmf_df  # lazy load
        if not isinstance(pmf_df, pd.DataFrame) or pmf_df.empty:
            return [], None
        if not clusters_store or cluster_id is None:
            return [], None

        has_ci = (
            "F_ci_lo_kJ_mol" in pmf_df.columns
            and "F_ci_hi_kJ_mol" in pmf_df.columns
        )
        if not has_ci:
            # Backward-compat: old pmf files without the CI columns just show
            # a disabled dropdown. The graph callback will render a notice.
            return [], None

        key = str(int(cluster_id))
        members = clusters_store["clusters"].get(key, []) if clusters_store else []
        if not members:
            return [], None

        # Only offer metrics that (a) are in the user's selected PMF metrics
        # (if any) and (b) have at least one finite F_ci_* value among the
        # cluster's members. This avoids promising CI data that turns out to
        # all be NaN (single-replica variants).
        cluster_sub = pmf_df[pmf_df[cols.variant].astype(str).isin(list(map(str, members)))]
        if cluster_sub.empty:
            return [], None

        candidate_metrics = cluster_sub[cols.metric].astype(str).dropna().unique().tolist()
        if pmf_metrics:
            sel = set(map(str, pmf_metrics))
            candidate_metrics = [m for m in candidate_metrics if m in sel]

        good: list[str] = []
        for m in candidate_metrics:
            dm = cluster_sub[cluster_sub[cols.metric].astype(str) == m]
            lo = pd.to_numeric(dm["F_ci_lo_kJ_mol"], errors="coerce")
            if lo.notna().any():
                good.append(m)
        good = sorted(good, key=torsion_sort_key)
        options = [{"label": m, "value": m} for m in good]
        value = current if current in good else (good[0] if good else None)
        return options, value

    @app.callback(
        Output("pdend-raw-ci-graph", "figure"),
        Input("pdend-clusters-store", "data"),
        Input("pdend-cluster", "value"),
        Input("pdend-ci-metric", "value"),
        Input("pdend-energy-units", "value"),
        Input("theme-store", "data"),
        prevent_initial_call=True,
    )
    def _raw_ci_graph(clusters_store, cluster_id, ci_metric, energy_units, theme):
        pmf_df = ctx.pmf_df  # lazy load
        if not isinstance(pmf_df, pd.DataFrame) or pmf_df.empty:
            return apply_theme(_error_fig("pmf_df is empty."), theme)

        has_ci = (
            "F_ci_lo_kJ_mol" in pmf_df.columns
            and "F_ci_hi_kJ_mol" in pmf_df.columns
        )
        if not has_ci:
            return apply_theme(_error_fig(
                "F_ci_lo_kJ_mol / F_ci_hi_kJ_mol columns not found in pmf_df. "
                "Re-run the batch pipeline with --pmf-bootstrap > 0."
            ), theme)

        if not clusters_store or cluster_id is None:
            return apply_theme(_error_fig("Pick a cluster."), theme)
        if not ci_metric:
            return apply_theme(_error_fig("Pick a metric that has bootstrap CI data."), theme)

        key = str(int(cluster_id))
        members = clusters_store["clusters"].get(key, [])
        members = list(map(str, members))
        if not members:
            return apply_theme(_error_fig("Cluster has no members."), theme)

        curves, bands = _per_variant_raw_pmf_with_ci(
            pmf_df=pmf_df,
            cols=cols,
            members=members,
            metric=str(ci_metric),
            max_variants=10,
        )
        if not curves:
            return apply_theme(_error_fig(f"No PMF data for metric '{ci_metric}' in this cluster."), theme)

        # Note: raw F is always in kJ/mol (it's what the batch pipeline writes).
        # We do not convert to kT here because the CI bands are in kJ/mol too
        # and a conversion would need to be applied consistently; the honest
        # and simplest thing is to tell the user the units.
        title = (
            f"Per-variant raw PMF + replica-block 95% CI - metric: {ci_metric}"
            + (f" (showing first 10 of {len(members)} members)" if len(members) > 10 else "")
        )
        return apply_theme(_pmf_fig(curves, "F", title, ci_bands=bands), theme)
