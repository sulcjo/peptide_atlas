from __future__ import annotations

import plotly.graph_objs as go
from dash import html


def apply_theme(fig: go.Figure, theme: str | None) -> go.Figure:
    """Apply light/dark Plotly template and transparent background to a figure."""
    template = "plotly_dark" if theme == "dark" else "plotly_white"
    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def panel(children, title: str | None = None, subtitle: str | None = None) -> html.Div:
    """Wrap a section in a consistent modern panel."""
    header = []
    if title:
        header.append(html.H3(title, className="panel-title"))
    if subtitle:
        header.append(html.Div(subtitle, className="panel-subtitle"))
    if header:
        header.append(html.Hr(className="panel-divider"))
    return html.Div(
        [
            html.Div(header, className="panel-header") if header else None,
            html.Div(children, className="panel-body"),
        ],
        className="panel",
    )
