from __future__ import annotations

from typing import List

import pandas as pd
import plotly.express as px
from dash import Input, Output, dcc, html, dash_table

from ..theming.errors import error_fig
from ..metrics import prettify_column_label, torsion_sort_key

TAB_LABEL = "Stat. Moments"


def _numeric_columns(df: pd.DataFrame) -> List[str]:
    if df is None or df.empty:
        return []
    cols = df.select_dtypes(include="number").columns.astype(str).tolist()
    return [c for c in cols if c != "variant"]


def _describe_table(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if df.empty or not cols:
        return pd.DataFrame()
    desc = df[cols].describe().T
    desc.insert(0, "column", desc.index.astype(str))
    desc = desc.reset_index(drop=True)
    keep = ["column", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    return desc[[c for c in keep if c in desc.columns]]


def layout(ctx) -> html.Div:
    df = getattr(ctx, "features", getattr(ctx, "df", pd.DataFrame()))
    cols = sorted(_numeric_columns(df), key=torsion_sort_key)
    default_cols = cols[:5]

    return html.Div(
        [
            html.H2("Stats"),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Columns"),
                            dcc.Dropdown(
                                id="stats-cols",
                                options=[{"label": prettify_column_label(c), "value": c} for c in cols],
                                value=default_cols,
                                multi=True,
                                placeholder="Select numeric columns...",
                            ),
                        ],
                        style={"minWidth": "320px", "flex": "1"},
                    ),
                    html.Div(
                        [
                            html.Label("Histogram column"),
                            dcc.Dropdown(
                                id="stats-hist-col",
                                options=[{"label": prettify_column_label(c), "value": c} for c in cols],
                                value=(default_cols[0] if default_cols else None),
                                clearable=True,
                            ),
                        ],
                        style={"minWidth": "260px", "flex": "1"},
                    ),
                ],
                style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
            ),
            html.Hr(),
            dash_table.DataTable(
                id="stats-table",
                columns=[],
                data=[],
                page_size=20,
                style_table={"overflowX": "auto"},
                style_cell={"padding": "6px", "fontFamily": "ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace", "fontSize": 12},
            ),
            html.Hr(),
            dcc.Graph(id="stats-hist"),
        ],
        className="tab-body-inner",
    )


def register_callbacks(app, ctx) -> None:
    from .shared import apply_theme
    df = getattr(ctx, "features", getattr(ctx, "df", pd.DataFrame()))

    @app.callback(
        Output("stats-table", "columns"),
        Output("stats-table", "data"),
        Input("stats-cols", "value"),
    )
    def _update_table(selected: List[str] | None):
        if df is None or df.empty:
            out = pd.DataFrame({"message": ["No features dataframe loaded (ctx.features is empty)."]})
            return [{"name": c, "id": c} for c in out.columns], out.to_dict("records")

        cols = [c for c in (selected or []) if c in df.columns]
        desc = _describe_table(df, cols)
        if desc.empty:
            out = pd.DataFrame({"message": ["Select at least one numeric column."]})
            return [{"name": c, "id": c} for c in out.columns], out.to_dict("records")

        return [{"name": c, "id": c} for c in desc.columns], desc.to_dict("records")

    @app.callback(
        Output("stats-hist", "figure"),
        Input("stats-hist-col", "value"),
        Input("theme-store", "data"),
    )
    def _update_hist(col: str | None, theme):
        if df is None or df.empty:
            return apply_theme(error_fig("No features dataframe loaded."), theme)
        if not col or col not in df.columns:
            return apply_theme(error_fig("Pick a histogram column."), theme)
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            return apply_theme(error_fig(f"No numeric data in column: {col}"), theme)
        fig = px.histogram(s, nbins=50, title=f"Histogram: {prettify_column_label(col)}")
        fig.update_layout(margin=dict(l=40, r=20, t=50, b=40))
        apply_theme(fig, theme)
        return fig
