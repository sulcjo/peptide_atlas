from __future__ import annotations

from typing import Any, Dict, List, Tuple

import pandas as pd
import plotly.graph_objects as go


def conv_default_columns() -> Tuple[List[str], List[str]]:
    """Default convergence column preferences (generic, non-legacy)."""
    x = ["frame", "time_ps", "time_ns", "step"]
    y = ["value", "mean", "metric_value", "rmsd", "rg", "energy"]
    return x, y


def conv_init(df: pd.DataFrame, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    """Lightweight initializer retained for API compatibility."""
    return {"df": df}


def conv_plot(
    state: Dict[str, Any],
    x: str,
    y: str,
    title: str = "Convergence",
) -> go.Figure:
    """Plot a simple line chart for convergence data."""
    df: pd.DataFrame | None = state.get("df") if isinstance(state, dict) else None
    if df is None or df.empty or x not in df.columns or y not in df.columns:
        fig = go.Figure()
        fig.update_layout(title=f"{title}: missing data")
        return fig

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df[x], y=df[y], mode="lines", name=y))
    fig.update_layout(title=title, margin=dict(l=40, r=20, t=50, b=40))
    return fig
