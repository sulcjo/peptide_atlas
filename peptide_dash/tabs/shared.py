from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from dash import html

from ..metrics import metric_display_label


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


def pmf_panel(title: str, subtitle: str | None, body: list) -> html.Div:
    """panel() with (title, subtitle, body) arg order used by PMF tabs."""
    return panel(body, title=title, subtitle=subtitle)


# ---------------------------------------------------------------------------
# PMF column layout
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class PmfCols:
    variant: str = "variant"
    metric: str = "metric"
    x: str = "x"
    p: str = "P"
    f: str = "F_kJ_mol"


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

R_GAS: float = 0.008314462618  # kJ mol⁻¹ K⁻¹


def kT(energy_units: str, T_K: float, kT_override: float | None = None) -> float:
    if kT_override is not None and np.isfinite(kT_override) and float(kT_override) > 0:
        return float(kT_override)
    if str(energy_units) == "kT":
        return 1.0
    return float(R_GAS * max(1.0, float(T_K)))


def bin_dx(x: np.ndarray) -> float:
    u = np.unique(x[np.isfinite(x)])
    if u.size < 2:
        return 1.0
    return float(np.median(np.diff(u)))


def normalize_density(x: np.ndarray, p: np.ndarray, dx: float | None = None) -> np.ndarray:
    if dx is None:
        dx = bin_dx(x)
    Z = float(np.sum(np.clip(p, 0.0, None)) * dx)
    if not np.isfinite(Z) or Z <= 0:
        return p
    return p / Z


# ---------------------------------------------------------------------------
# Context helpers
# ---------------------------------------------------------------------------

def infer_features_df(ctx) -> pd.DataFrame:
    df = getattr(ctx, "features", None)
    if isinstance(df, pd.DataFrame):
        return df
    df = getattr(ctx, "df", None)
    if isinstance(df, pd.DataFrame):
        return df
    return pd.DataFrame()


def infer_weight_columns(features_df: pd.DataFrame) -> list[str]:
    if not isinstance(features_df, pd.DataFrame) or features_df.empty:
        return []
    if "variant" not in features_df.columns:
        return []
    num_cols = features_df.select_dtypes(include="number").columns.astype(str).tolist()
    preferred = [c for c in ["n_frames", "n", "n_samples", "weight", "quality"] if c in num_cols]
    rest = [c for c in num_cols if c not in preferred]
    return preferred + rest[:24]


# ---------------------------------------------------------------------------
# Dropdown helpers
# ---------------------------------------------------------------------------

def metric_options(metrics) -> list[dict]:
    return [{"label": metric_display_label(m), "value": m} for m in metrics]
