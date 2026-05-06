from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html


@dataclass(frozen=True)
class CurveCols:
    variant: str = "variant"
    metric: str = "metric"
    x: str = "x"
    p: str = "P"
    replica: str = "replica"
    repl: str = "repl"
    rep: str = "rep"
    replica_id: str = "replica_id"
    rep_id: str = "rep_id"
    replicate: str = "replicate"
    source: str = "source"


@dataclass(frozen=True)
class MetricBlock:
    """
    Replica-mean PMF representation for one metric on a common x-grid.

    mass[i, k] is probability mass in bin k for variant i, and sums to 1 across k.
    """
    metric: str
    x: np.ndarray
    dx: np.ndarray
    mass: np.ndarray


_DISTANCE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("Wasserstein-1 (physical x)", "w1"),
    ("Wasserstein-2 (physical x)", "w2"),
    ("Jensen–Shannon distance", "js"),
    ("Hellinger distance", "hellinger"),
)

_ALPHA_OPTIONS: tuple[tuple[str, float], ...] = (
    ("0.0 (density preserved)", 0.0),
    ("0.5 (partially density-corrected)", 0.5),
    ("1.0 (density corrected)", 1.0),
)


def layout(ctx: Any) -> list[Any]:
    cols = CurveCols()
    pmf_rep = getattr(ctx, "pmf_replica_df", pd.DataFrame())
    if not isinstance(pmf_rep, pd.DataFrame) or pmf_rep.empty:
        return [
            html.Div(
                [
                    html.H3("Diffusion Map (PMF replica means)"),
                    html.Div(
                        "Replica-resolved PMF tables were not found. "
                        "This tab requires pmf_replica (pmf_replica_df) produced by the updated analysis pipeline.",
                        className="note",
                    ),
                ]
            )
        ]

    metrics = sorted(map(str, pmf_rep.get(cols.metric, pd.Series(dtype=str)).dropna().unique().tolist()))
    default_metrics = metrics[: min(6, len(metrics))]

    numeric_cols: list[str] = []
    try:
        numeric_cols = list(getattr(ctx, "numeric_cols", []) or [])
    except Exception:
        numeric_cols = []
    color_options = [{"label": "None", "value": ""}] + [{"label": c, "value": c} for c in numeric_cols]

    return [
        html.Div(
            [
                html.H3("Diffusion Map (PMF replica means)"),
                html.Div(
                    "Diffusion maps emphasize smooth continua/branches in variant behavior-space. "
                    "This tab embeds variants using replica-mean PMFs (not pooled replicas).",
                    className="note",
                ),
                html.Div(
                    [
                        html.Div(
                            [
                                html.Label("Metrics (PMFs)"),
                                dcc.Dropdown(
                                    id="dm-metrics",
                                    options=[{"label": m, "value": m} for m in metrics],
                                    value=default_metrics,
                                    multi=True,
                                    placeholder="Select one or more metrics",
                                ),
                            ],
                            className="control",
                        ),
                        html.Div(
                            [
                                html.Label("Distance (switcher)"),
                                dcc.Dropdown(
                                    id="dm-distance",
                                    options=[{"label": lab, "value": val} for lab, val in _DISTANCE_OPTIONS],
                                    value="w1",
                                    clearable=False,
                                ),
                            ],
                            className="control",
                        ),
                        html.Div(
                            [
                                html.Label("kNN (graph)"),
                                dcc.Slider(
                                    id="dm-knn",
                                    min=5,
                                    max=80,
                                    step=1,
                                    value=25,
                                    marks={5: "5", 25: "25", 50: "50", 80: "80"},
                                ),
                            ],
                            className="control",
                        ),
                        html.Div(
                            [
                                html.Label("α (density correction)"),
                                dcc.Dropdown(
                                    id="dm-alpha",
                                    options=[{"label": lab, "value": val} for lab, val in _ALPHA_OPTIONS],
                                    value=1.0,
                                    clearable=False,
                                ),
                            ],
                            className="control",
                        ),
                        html.Div(
                            [
                                html.Label("Diffusion time (t)"),
                                dcc.Slider(
                                    id="dm-t",
                                    min=0,
                                    max=10,
                                    step=1,
                                    value=1,
                                    marks={0: "0", 1: "1", 3: "3", 5: "5", 10: "10"},
                                ),
                            ],
                            className="control",
                        ),
                        html.Div(
                            [
                                html.Label("Color by (features)"),
                                dcc.Dropdown(
                                    id="dm-colorby",
                                    options=color_options,
                                    value="",
                                    clearable=False,
                                ),
                            ],
                            className="control",
                        ),
                        html.Div(
                            [
                                html.Button("Compute diffusion map", id="dm-run", n_clicks=0, className="btn"),
                                html.Div(id="dm-info", className="note", style={"marginTop": "6px"}),
                            ],
                            className="control",
                        ),
                    ],
                    className="controls-grid",
                ),
                dcc.Store(id="dm-store"),
                html.Div(
                    [
                        dcc.Graph(id="dm-embed-graph", figure=_empty_fig("Run diffusion map to see embedding.")),
                        dcc.Graph(id="dm-eigs-graph", figure=_empty_fig("Eigenvalue spectrum will appear here.")),
                    ],
                    className="two-col",
                ),
            ]
        )
    ]


def register_callbacks(app, ctx: Any) -> None:
    from .shared import apply_theme
    feats = getattr(ctx, "features", pd.DataFrame())

    @app.callback(
        Output("dm-store", "data"),
        Output("dm-info", "children"),
        Input("dm-run", "n_clicks"),
        State("dm-metrics", "value"),
        State("dm-distance", "value"),
        State("dm-knn", "value"),
        State("dm-alpha", "value"),
        State("dm-t", "value"),
        prevent_initial_call=True,
    )
    def _compute_dm(n_clicks: int, metrics: list[str], dist_kind: str, knn: int, alpha: float, t: int):
        pmf_rep = ctx.pmf_replica_df  # lazy: loaded only when this callback fires
        del n_clicks
        try:
            if not isinstance(pmf_rep, pd.DataFrame) or pmf_rep.empty:
                return {}, "pmf_replica_df is empty."

            metrics = list(metrics or [])
            if not metrics:
                return {}, "Select at least one metric."

            blocks, meta, notes = _build_replica_mean_blocks(pmf_rep, metrics)
            if not blocks or meta.empty:
                return {}, "No replica-mean PMFs could be built for the selected metrics."

            D, dnotes = _pairwise_distance(blocks, dist_kind=str(dist_kind or "w1"))
            notes.extend(dnotes)

            eigvals, coords, gnotes = _diffusion_map(
                D,
                knn=int(knn or 25),
                alpha=float(alpha if alpha is not None else 1.0),
                t=int(t if t is not None else 1),
                n_components=6,
            )
            notes.extend(gnotes)

            store = {
                "meta": meta.to_dict("records"),
                "eigvals": eigvals.tolist(),
                "coords": coords.tolist(),
                "params": {
                    "metrics": metrics,
                    "dist": str(dist_kind or "w1"),
                    "knn": int(knn or 25),
                    "alpha": float(alpha if alpha is not None else 1.0),
                    "t": int(t if t is not None else 1),
                },
            }
            return store, " | ".join(notes[:12])
        except Exception as e:
            return {}, f"Error: {type(e).__name__}: {e}"

    @app.callback(
        Output("dm-embed-graph", "figure"),
        Output("dm-eigs-graph", "figure"),
        Input("dm-store", "data"),
        Input("dm-colorby", "value"),
        Input("theme-store", "data"),
    )
    def _render(store: Optional[dict], colorby: str, theme):
        if not store or "meta" not in store:
            return apply_theme(_empty_fig("Run diffusion map to see embedding."), theme), apply_theme(_empty_fig("Eigenvalue spectrum will appear here."), theme)

        meta = pd.DataFrame(store.get("meta") or [])
        coords = np.asarray(store.get("coords") or [], dtype=float)
        eigvals = np.asarray(store.get("eigvals") or [], dtype=float)

        if meta.empty or coords.size == 0:
            return apply_theme(_empty_fig("No embedding data."), theme), apply_theme(_empty_fig("No eigenvalue data."), theme)

        df = meta.copy()
        df["dc1"] = coords[:, 0] if coords.shape[1] > 0 else np.nan
        df["dc2"] = coords[:, 1] if coords.shape[1] > 1 else np.nan

        cvals = None
        ctitle = ""
        if colorby and isinstance(feats, pd.DataFrame) and (not feats.empty) and ("variant" in feats.columns) and (colorby in feats.columns):
            tmp = feats[["variant", colorby]].copy()
            tmp["variant"] = tmp["variant"].astype(str)
            df["variant"] = df["variant"].astype(str)
            df = df.merge(tmp, on="variant", how="left")
            cvals = df[colorby]
            ctitle = str(colorby)

        fig = go.Figure()
        marker = {"size": 9, "opacity": 0.85}
        if cvals is not None:
            marker["color"] = cvals
            marker["colorbar"] = {"title": ctitle}
        fig.add_trace(
            go.Scatter(
                x=df["dc1"],
                y=df["dc2"],
                mode="markers",
                marker=marker,
                customdata=df["variant"].astype(str),
                hovertemplate="variant=%{customdata}<br>DC1=%{x:.4g}<br>DC2=%{y:.4g}<extra></extra>",
                name="variants",
            )
        )
        fig.update_layout(
            title="Diffusion Map embedding (replica means)",
            xaxis_title="DC1",
            yaxis_title="DC2",
            margin=dict(l=40, r=20, t=40, b=40),
        )

        eig_fig = go.Figure()
        if eigvals.size:
            xs = np.arange(len(eigvals))
            eig_fig.add_trace(go.Scatter(x=xs, y=eigvals, mode="markers+lines"))
        eig_fig.update_layout(
            title="Diffusion eigenvalues",
            xaxis_title="Index",
            yaxis_title="λ",
            margin=dict(l=40, r=20, t=40, b=40),
        )
        apply_theme(fig, theme)
        apply_theme(eig_fig, theme)
        return fig, eig_fig


def _empty_fig(msg: str) -> go.Figure:
    fig = go.Figure()
    fig.update_layout(
        template="plotly_white",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        annotations=[dict(text=msg, x=0.5, y=0.5, xref="paper", yref="paper", showarrow=False)],
        margin=dict(l=20, r=20, t=40, b=20),
    )
    return fig


def _replica_col(df: pd.DataFrame) -> Optional[str]:
    cols = CurveCols()
    candidates = [cols.replica, cols.repl, cols.rep, cols.replica_id, cols.rep_id, cols.replicate, cols.source]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def _looks_periodic_metric(metric: str) -> bool:
    m = str(metric).lower()
    return any(tok in m for tok in ("phi", "psi", "chi", "torsion", "dihed", "dihedral"))


def _build_replica_mean_blocks(pmf_rep: pd.DataFrame, metrics: list[str]) -> tuple[list[MetricBlock], pd.DataFrame, list[str]]:
    """
    Build per-metric replica-mean PMF blocks.

    For each metric:
      1) interpolate each replica density P(x) onto a common x-grid,
      2) mean over replicas (per variant),
      3) convert to probability mass using dx,
      4) normalize to sum to 1.

    Variants are restricted to those with data for *all* selected metrics (intersection),
    to keep distances well-defined.
    """
    cols = CurveCols()
    df = pmf_rep.copy()
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[cols.variant, cols.metric, cols.x, cols.p])

    df[cols.variant] = df[cols.variant].astype(str)
    df[cols.metric] = df[cols.metric].astype(str)
    df[cols.x] = df[cols.x].astype(float)
    df[cols.p] = df[cols.p].astype(float)

    metrics = list(map(str, metrics))
    df = df[df[cols.metric].isin(metrics)]
    if df.empty:
        return [], pd.DataFrame(), ["no rows for selected metrics"]

    repc = _replica_col(df)
    if repc is None:
        return [], pd.DataFrame(), ["no replica identifier column found"]

    notes: list[str] = [f"replica_col={repc}"]

    # per-metric common grids from rounded x
    grids: dict[str, np.ndarray] = {}
    for m, g in df.groupby(cols.metric):
        xs = np.round(np.asarray(g[cols.x], dtype=float), 6)
        grid = np.unique(xs)
        grid.sort()
        if grid.size >= 5:
            grids[str(m)] = grid

    variants_per_metric: dict[str, set[str]] = {}

    for m in metrics:
        if m not in grids:
            continue
        g = df[df[cols.metric] == m]
        if g.empty:
            continue
        v_ok: set[str] = set()
        for (v, r), gr in g.groupby([cols.variant, repc]):
            xs = np.asarray(gr[cols.x], dtype=float)
            ps = np.asarray(gr[cols.p], dtype=float)
            if np.isfinite(xs).sum() < 3 or np.isfinite(ps).sum() < 3:
                continue
            v_ok.add(str(v))
        if v_ok:
            variants_per_metric[m] = v_ok

    if not variants_per_metric:
        return [], pd.DataFrame(), ["no usable metrics with replicas"]

    # intersection of variants across selected metrics (only those with all blocks)
    common: Optional[set[str]] = None
    usable_metrics: list[str] = []
    for m in metrics:
        if m not in variants_per_metric:
            continue
        if common is None:
            common = set(variants_per_metric[m])
        else:
            common &= set(variants_per_metric[m])
        usable_metrics.append(m)

    common = common or set()
    if not common:
        return [], pd.DataFrame(), ["no variants have all selected metrics"]

    variants = sorted(common)
    dropped = int(df[cols.variant].nunique() - len(variants))
    if dropped:
        notes.append(f"dropped_variants_missing_metrics={dropped}")

    blocks: list[MetricBlock] = []
    for m in usable_metrics:
        if m not in grids:
            continue
        grid = grids[m]
        dx = np.diff(grid)
        dx = np.concatenate([dx, dx[-1:]])
        g = df[df[cols.metric] == m]
        repc = _replica_col(g) or repc

        by_variant: dict[str, list[np.ndarray]] = {}
        for (v, r), gr in g.groupby([cols.variant, repc]):
            v = str(v)
            if v not in common:
                continue
            xs = np.round(np.asarray(gr[cols.x], dtype=float), 6)
            ps = np.asarray(gr[cols.p], dtype=float)
            msk = np.isfinite(xs) & np.isfinite(ps)
            xs, ps = xs[msk], ps[msk]
            if xs.size < 3:
                continue
            o = np.argsort(xs)
            xs, ps = xs[o], ps[o]
            p_interp = np.interp(grid, xs, ps, left=0.0, right=0.0)
            p_interp = np.clip(p_interp, 0.0, np.inf)
            by_variant.setdefault(v, []).append(p_interp)

        mass = np.full((len(variants), grid.size), np.nan, dtype=float)
        for i, v in enumerate(variants):
            reps = by_variant.get(v) or []
            if not reps:
                continue
            mean_p = np.nanmean(np.vstack(reps), axis=0)
            mean_p = np.clip(mean_p, 0.0, np.inf)
            mvec = mean_p * dx
            mvec = np.clip(mvec, 0.0, np.inf) + 1e-12
            mvec = mvec / float(np.sum(mvec))
            mass[i, :] = mvec

        if np.isnan(mass).any():
            # Should not happen given intersection, but keep robust
            bad = int(np.isnan(mass).any(axis=1).sum())
            notes.append(f"warning:missing_rows_metric={m}:{bad}")

        blocks.append(MetricBlock(metric=m, x=grid, dx=dx, mass=mass))

        if _looks_periodic_metric(m):
            notes.append(f"warning:periodic_metric={m}")

    meta = pd.DataFrame({"variant": variants})
    notes.append(f"metrics_used={len(blocks)}")
    notes.append(f"variants={len(variants)}")
    return blocks, meta, notes


def _pairwise_distance(blocks: list[MetricBlock], *, dist_kind: str) -> tuple[np.ndarray, list[str]]:
    """
    Combine per-metric distances into a single NxN matrix.

    Combination uses per-metric scale normalization:
      D = sum_m D_m / median(D_m)

    This keeps metrics with different units/ranges from dominating the graph.
    """
    notes: list[str] = [f"distance={dist_kind}"]
    if not blocks:
        return np.zeros((0, 0), dtype=float), notes + ["no_blocks"]

    n = blocks[0].mass.shape[0]
    D = np.zeros((n, n), dtype=float)

    for blk in blocks:
        M = np.asarray(blk.mass, dtype=float)
        M = np.clip(M, 1e-15, np.inf)
        M = M / M.sum(axis=1, keepdims=True)

        Dm = np.zeros((n, n), dtype=float)

        if dist_kind == "js":
            for i in range(n):
                pi = M[i]
                for j in range(i + 1, n):
                    pj = M[j]
                    m = 0.5 * (pi + pj)
                    js = 0.5 * (np.sum(pi * np.log(pi / m)) + np.sum(pj * np.log(pj / m)))
                    Dm[i, j] = Dm[j, i] = float(np.sqrt(max(js, 0.0)))

        elif dist_kind == "hellinger":
            sM = np.sqrt(M)
            for i in range(n):
                for j in range(i + 1, n):
                    Dm[i, j] = Dm[j, i] = float(np.linalg.norm(sM[i] - sM[j]) / np.sqrt(2.0))

        elif dist_kind == "w2":
            # quantile function on uniform u-grid
            u = np.linspace(0.0, 1.0, 256)
            cdf = np.cumsum(M, axis=1)
            Q = np.zeros((n, u.size), dtype=float)
            for i in range(n):
                Q[i] = np.interp(u, cdf[i], blk.x)
            for i in range(n):
                for j in range(i + 1, n):
                    Dm[i, j] = Dm[j, i] = float(np.sqrt(np.mean((Q[i] - Q[j]) ** 2)))

        else:
            # default w1
            cdf = np.cumsum(M, axis=1)
            dx = blk.dx
            for i in range(n):
                for j in range(i + 1, n):
                    Dm[i, j] = Dm[j, i] = float(np.sum(np.abs(cdf[i] - cdf[j]) * dx))

        # normalize scale
        tri = Dm[np.triu_indices(n, 1)]
        s = float(np.median(tri[tri > 0])) if np.any(tri > 0) else 1.0
        s = max(s, 1e-12)
        D += Dm / s
        notes.append(f"{blk.metric}:scale={s:.3g}")

    return D, notes


def _diffusion_map(D: np.ndarray, *, knn: int, alpha: float, t: int, n_components: int = 6) -> tuple[np.ndarray, np.ndarray, list[str]]:
    notes: list[str] = []
    n = D.shape[0]
    if n == 0:
        return np.array([]), np.zeros((0, 0)), ["empty"]

    knn = max(2, min(int(knn), n - 1))
    alpha = float(alpha)
    t = int(t)

    notes.append(f"knn={knn}")
    notes.append(f"alpha={alpha:g}")
    notes.append(f"t={t}")

    idx = np.argsort(D, axis=1)[:, : knn + 1]  # include self

    eps = np.take_along_axis(D, idx[:, -1:], axis=1).reshape(-1)
    eps = np.where(eps > 0, eps, np.nan)
    gmed = float(np.nanmedian(D[np.triu_indices(n, 1)])) if n > 1 else 1.0
    eps = np.where(np.isfinite(eps), eps, gmed)
    eps = np.maximum(eps, 1e-12)

    W = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in idx[i]:
            dij = float(D[i, j])
            W[i, j] = math.exp(-(dij * dij) / float(eps[i] * eps[j]))
    W = 0.5 * (W + W.T)

    q = np.maximum(W.sum(axis=1), 1e-12)
    K = W / ((q ** alpha)[:, None] * (q ** alpha)[None, :])

    d = np.maximum(K.sum(axis=1), 1e-12)
    d_sqrt_inv = 1.0 / np.sqrt(d)
    A = (d_sqrt_inv[:, None] * K) * d_sqrt_inv[None, :]

    vals, vecs = np.linalg.eigh(A)
    order = np.argsort(vals)[::-1]
    vals = vals[order]
    vecs = vecs[:, order]

    phi = vecs * d_sqrt_inv[:, None]

    keep = max(2, n_components + 1)
    vals = vals[:keep]
    phi = phi[:, :keep]

    lam = vals[1:]
    psi = phi[:, 1:]

    t = max(0, t)
    coords = psi * (lam[None, :] ** float(t))
    coords = coords[:, : min(n_components, coords.shape[1])]

    notes.append(f"components={coords.shape[1]}")
    if lam.size:
        notes.append(f"lambda2={lam[0]:.4g}")

    near1 = int(np.sum(lam > 0.999))
    if near1 >= 2:
        notes.append("warning:graph_may_be_disconnected")

    return vals, coords, notes
