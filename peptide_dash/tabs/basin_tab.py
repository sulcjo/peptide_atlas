from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, dcc, html

from ..theming.errors import error_fig

TAB_LABEL = "Basin Landscape"

_STAT_OPTIONS = [
    {"label": "Global min position (nm / °)",     "value": "global_basin_min_x"},
    {"label": "Secondary min position (nm / °)",   "value": "secondary_basin_min_x"},
    {"label": "Secondary persistence (kT)",         "value": "secondary_basin_persistence_kT"},
    {"label": "Basin width at 1 kT (nm / °)",      "value": "global_basin_width_1kT"},
    {"label": "Equilibrium-population position",   "value": "x_eqpop"},
    {"label": "Ensemble mean",                     "value": "mean"},
    {"label": "Circular mean (°) — torsions",      "value": "circular_mean_deg"},
]

_STAT_LABEL = {o["value"]: o["label"] for o in _STAT_OPTIONS}

_CARD = {
    "padding": "10px",
    "border": "1px solid #ccc",
    "borderRadius": "8px",
    "marginBottom": "12px",
    "display": "flex",
    "flexWrap": "wrap",
    "gap": "16px",
    "alignItems": "flex-end",
}


def _annotation_prefixes(ctx) -> list[str]:
    cols: set[str] = getattr(ctx, "_annotation_cols", set())
    prefixes: set[str] = set()
    for c in cols:
        if "__" in c:
            prefixes.add(c.split("__")[0])
    return sorted(prefixes)


def layout(ctx) -> html.Div:
    prefixes = _annotation_prefixes(ctx)
    metric_opts = [{"label": p, "value": p} for p in prefixes]

    return html.Div([
        html.H2("Basin Landscape"),
        html.Div(
            [
                html.Div([
                    html.Label("Statistic"),
                    dcc.Dropdown(
                        id="basin-stat",
                        options=_STAT_OPTIONS,
                        value="global_basin_min_x",
                        clearable=False,
                        style={"minWidth": "280px"},
                    ),
                ]),
                html.Div([
                    html.Label("Metrics"),
                    dcc.Dropdown(
                        id="basin-metrics",
                        options=metric_opts,
                        value=prefixes,
                        multi=True,
                        placeholder="All metrics",
                        style={"minWidth": "320px"},
                    ),
                ]),
                html.Div([
                    html.Label("Options"),
                    dcc.Checklist(
                        id="basin-opts",
                        options=[
                            {"label": " Cluster variants (rows)", "value": "cluster_rows"},
                            {"label": " Cluster metrics (cols)",  "value": "cluster_cols"},
                            {"label": " Z-score columns",         "value": "zscore"},
                        ],
                        value=["cluster_rows"],
                        inline=False,
                    ),
                ]),
            ],
            style=_CARD,
        ),
        dcc.Graph(
            id="basin-heatmap",
            style={"height": "78vh"},
            config={"displaylogo": False, "scrollZoom": True},
        ),
    ])


def register_callbacks(app, ctx):
    from .shared import apply_theme

    @app.callback(
        Output("basin-heatmap", "figure"),
        Input("basin-stat", "value"),
        Input("basin-metrics", "value"),
        Input("basin-opts", "value"),
        Input("theme-store", "data"),
    )
    def update_heatmap(stat, selected_metrics, opts, theme):
        opts = opts or []

        df: pd.DataFrame = getattr(ctx, "df", pd.DataFrame())
        ann_cols: set[str] = getattr(ctx, "_annotation_cols", set())

        if df is None or df.empty or not stat:
            return apply_theme(error_fig("No data available."), theme)

        suffix = f"__{stat}"
        matching = [c for c in ann_cols if c.endswith(suffix)]

        if selected_metrics:
            matching = [c for c in matching if c.split("__")[0] in selected_metrics]

        if not matching:
            return apply_theme(error_fig(f"No annotation columns found for suffix '{stat}'."), theme)

        variant_col = "variant" if "variant" in df.columns else df.index.name or None
        if variant_col and variant_col in df.columns:
            sub = df[[variant_col] + [c for c in matching if c in df.columns]].copy()
            sub = sub.set_index(variant_col)
        else:
            sub = df[[c for c in matching if c in df.columns]].copy()

        sub.columns = [c.split("__")[0] for c in sub.columns]
        sub = sub.dropna(axis=0, how="all").dropna(axis=1, how="all")

        if sub.empty:
            return apply_theme(error_fig("Matrix is empty after dropping all-NaN rows/cols."), theme)

        do_zscore = "zscore" in opts
        do_cluster_rows = "cluster_rows" in opts
        do_cluster_cols = "cluster_cols" in opts

        if do_zscore:
            col_mean = sub.mean(skipna=True)
            col_std = sub.std(skipna=True).replace(0, np.nan)
            sub = (sub - col_mean) / col_std

        def _ward_order(mat2d: np.ndarray) -> list[int]:
            from scipy.cluster.hierarchy import linkage, leaves_list
            filled = np.where(np.isnan(mat2d), np.nanmedian(mat2d, axis=0, keepdims=True) * np.ones_like(mat2d), mat2d)
            filled = np.nan_to_num(filled, nan=0.0)
            Z = linkage(filled, method="ward", metric="euclidean")
            return list(leaves_list(Z))

        if do_cluster_rows and len(sub) >= 3:
            order = _ward_order(sub.values)
            sub = sub.iloc[order]

        if do_cluster_cols and len(sub.columns) >= 3:
            order = _ward_order(sub.values.T)
            sub = sub.iloc[:, order]

        variants = list(sub.index.astype(str))
        metrics = list(sub.columns.astype(str))
        z = sub.values

        fig_height = max(500, min(len(sub) * 14, 3000))
        stat_label = _STAT_LABEL.get(stat, stat)
        title = f"{stat_label} — {len(sub)} variants × {len(sub.columns)} metrics"

        fig = go.Figure(go.Heatmap(
            z=z,
            x=metrics,
            y=variants,
            colorscale="RdBu_r",
            zmid=0 if do_zscore else None,
            hoverongaps=False,
            hovertemplate="variant=%{y}<br>metric=%{x}<br>value=%{z:.4g}<extra></extra>",
        ))

        fig.update_layout(
            title=title,
            height=fig_height,
            margin=dict(l=160, r=40, t=60, b=80),
            xaxis=dict(tickangle=-45, tickfont=dict(size=11)),
            yaxis=dict(tickfont=dict(size=10)),
        )

        apply_theme(fig, theme)
        return fig
