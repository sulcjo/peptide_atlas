from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go


def add_error_band(fig: go.Figure, x, y, yerr, **style: Any) -> go.Figure:
    """
    Add a simple symmetric error band (y ± yerr) to an existing figure.
    """
    x = np.asarray(x)
    y = np.asarray(y, dtype=float)
    yerr = np.asarray(yerr, dtype=float)

    upper = y + yerr
    lower = y - yerr

    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([upper, lower[::-1]]),
            fill="toself",
            hoverinfo="skip",
            line=dict(width=0),
            showlegend=False,
            **style,
        )
    )
    return fig
