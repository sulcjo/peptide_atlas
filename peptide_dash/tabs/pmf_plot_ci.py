"""
Shared PMF plotting helpers with replica-block bootstrap CI bands.

Used by `tabs/pmf_dendrogram_tab.py` and `tabs/umap_region_pmf_tab.py`.
Kept separate from those files so the two tabs cannot drift apart in
how they render uncertainty.

Rendering convention:
    - CI band drawn first (under the line) using a filled Scatter
      polygon at 18% alpha of the line's color.
    - NaN values in lo/hi break the polygon at that bin (Plotly handles
      this correctly without extra masking work).
    - Bands and lines share a legendgroup and use showlegend=False on
      the band so the legend stays clean.
    - Band is always 'hoverinfo=skip' so the hover shows the line's y
      value, not confusing polygon coordinates.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go


# Plotly's default qualitative palette. Reused for band + line so the
# color pairing is visually obvious.
_PALETTE = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)


def pmf_error_fig(msg: str) -> go.Figure:
    """Compact error figure for PMF panels (height=520, no title)."""
    fig = go.Figure()
    fig.add_annotation(text=str(msg), x=0.5, y=0.5, xref="paper", yref="paper",
                       showarrow=False, align="center", font=dict(size=14))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template="plotly_white", height=520, margin=dict(l=20, r=20, t=20, b=20))
    return fig


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    """Convert '#rrggbb' to 'rgba(r, g, b, alpha)' for Plotly fillcolor."""
    h = str(hex_color).lstrip("#")
    if len(h) != 6:
        return f"rgba(100, 100, 100, {float(alpha)})"
    try:
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    except ValueError:
        return f"rgba(100, 100, 100, {float(alpha)})"
    return f"rgba({r}, {g}, {b}, {float(alpha)})"


def pmf_overlay_fig(
    curves: Dict[str, Tuple[np.ndarray, np.ndarray]],
    *,
    title: str,
    y_title: str = "F (kJ/mol)",
    x_title: str = "x",
    height: int = 520,
    ci_bands: Optional[Dict[str, Tuple[np.ndarray, np.ndarray]]] = None,
    error_text: str = "No PMF to plot.",
) -> go.Figure:
    """
    Draw an overlay of 1D curves with optional semi-transparent CI bands.

    Parameters
    ----------
    curves : {name: (x, y)}
        One 1D curve per name; x and y are same-length arrays.
    title, x_title, y_title, height : plot cosmetics.
    ci_bands : optional {name: (y_lo, y_hi)}
        Bands keyed by the same names as curves. Each (y_lo, y_hi) must
        match the length of the corresponding y. Bins where either bound
        is NaN break the polygon at that bin, which is what we want for
        unsampled bootstrap bins.
    error_text : str
        Annotation to render if curves is empty.
    """
    if not curves:
        fig = go.Figure()
        fig.add_annotation(text=error_text, x=0.5, y=0.5, xref="paper", yref="paper",
                           showarrow=False, font=dict(size=14))
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        fig.update_layout(template="plotly_white", height=height, margin=dict(l=20, r=20, t=20, b=20))
        return fig

    fig = go.Figure()
    for i, (name, (x, y)) in enumerate(curves.items()):
        color = _PALETTE[i % len(_PALETTE)]

        # CI band first (under the line).
        if ci_bands is not None and str(name) in ci_bands:
            lo, hi = ci_bands[str(name)]
            lo = np.asarray(lo, dtype=float)
            hi = np.asarray(hi, dtype=float)
            y_arr = np.asarray(y, dtype=float)
            if lo.size == y_arr.size and hi.size == y_arr.size \
                    and (np.isfinite(lo).any() and np.isfinite(hi).any()):
                xs_band = np.concatenate([np.asarray(x, float), np.asarray(x, float)[::-1]])
                ys_band = np.concatenate([hi, lo[::-1]])
                fig.add_trace(
                    go.Scatter(
                        x=xs_band, y=ys_band,
                        mode="lines",
                        fill="toself",
                        fillcolor=hex_to_rgba(color, 0.18),
                        line=dict(width=0),
                        name=f"{name} CI",
                        legendgroup=str(name),
                        showlegend=False,
                        hoverinfo="skip",
                    )
                )

        fig.add_trace(
            go.Scatter(
                x=x, y=y,
                mode="lines",
                line=dict(color=color, width=2),
                name=str(name),
                legendgroup=str(name),
                hovertemplate="%{fullData.name}<br>x=%{x:.3f}<br>y=%{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=height,
        margin=dict(l=40, r=20, t=40, b=60),
        title=title,
        legend=dict(orientation="h", x=0, y=-0.18, xanchor="left", yanchor="top"),
    )
    fig.update_xaxes(title=x_title)
    fig.update_yaxes(title=y_title)
    return fig


def per_variant_raw_pmf_with_ci(
    pmf_df: pd.DataFrame,
    *,
    variants: Sequence[str],
    metric: str,
    variant_col: str = "variant",
    metric_col: str = "metric",
    x_col: str = "x",
    f_col: str = "F_kJ_mol",
    lo_col: str = "F_ci_lo_kJ_mol",
    hi_col: str = "F_ci_hi_kJ_mol",
    max_variants: int = 10,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], Dict[str, Tuple[np.ndarray, np.ndarray]]]:
    """
    Return unmixed per-variant F(x) curves and their bootstrap CI bands
    for a single metric, pulled straight from the batch-pipeline pmf table.

    This deliberately does NOT mix variants (no weighted averaging).
    Cluster-level averaging with bootstrap CIs would require resampling
    inside the mixture - a different, more expensive, and more
    statistically fraught calculation. The honest plot is the raw one.

    Returns
    -------
    curves : {variant: (x, F_kJ_mol)}
    bands  : {variant: (F_ci_lo, F_ci_hi)}   (empty if CI columns missing)
    """
    empty: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    if pmf_df is None or pmf_df.empty or not variants:
        return empty, empty

    has_ci = (lo_col in pmf_df.columns) and (hi_col in pmf_df.columns)

    variants_use = list(map(str, variants))[: int(max_variants)]
    sub = pmf_df[
        (pmf_df[metric_col].astype(str) == str(metric))
        & (pmf_df[variant_col].astype(str).isin(variants_use))
    ].copy()
    if sub.empty:
        return empty, empty

    curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    bands: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for v, g in sub.groupby(variant_col):
        g = g.sort_values(x_col)
        x = pd.to_numeric(g[x_col], errors="coerce").to_numpy(dtype=float)
        F = pd.to_numeric(g[f_col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(F)
        if not mask.any():
            continue
        curves[str(v)] = (x[mask], F[mask])
        if has_ci:
            lo = pd.to_numeric(g[lo_col], errors="coerce").to_numpy(dtype=float)
            hi = pd.to_numeric(g[hi_col], errors="coerce").to_numpy(dtype=float)
            if lo.size == mask.size and hi.size == mask.size:
                bands[str(v)] = (lo[mask], hi[mask])

    return curves, bands


def pmf_has_ci_columns(
    pmf_df: pd.DataFrame,
    lo_col: str = "F_ci_lo_kJ_mol",
    hi_col: str = "F_ci_hi_kJ_mol",
) -> bool:
    """Quick check that the batch pipeline produced bootstrap CI columns."""
    if pmf_df is None or pmf_df.empty:
        return False
    return (lo_col in pmf_df.columns) and (hi_col in pmf_df.columns)
