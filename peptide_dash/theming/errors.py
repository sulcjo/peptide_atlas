from __future__ import annotations

import plotly.graph_objects as go


def error_fig(msg: str, theme: str | None = None) -> go.Figure:
    """Small, dependency-free error figure used across tabs."""
    template = "plotly_dark" if theme == "dark" else "plotly_white"
    fig = go.Figure()
    fig.add_annotation(
        text=str(msg),
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=16),
    )
    fig.update_layout(
        template=template,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        margin=dict(l=20, r=20, t=40, b=20),
        title="Error",
    )
    return fig
