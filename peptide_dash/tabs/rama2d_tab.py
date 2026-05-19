#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
rama2d_tab.py — 2D Ramachandran plot tab for peptide_dash

Data layout (legacy structure):

  ${BASE_DIR}/${VARIANT}/md_cluster_pbc_${REPLICA_NO}_phi_hist.xvg
  ${BASE_DIR}/${VARIANT}/md_cluster_pbc_${REPLICA_NO}_psi_hist.xvg

Each *_hist.xvg is a gmx gangle histogram:
  angle  h_1  h_2  ... h_N

Where each column h_k is the histogram for dihedral k
(typically one per residue along the chain, in selection order).

Public API (matches other modular tabs):

  • layout(ctx)                -> Dash layout (for dcc.Tab children)
  • register_callbacks(app, ctx)

ctx is the global context from CLI, expected to contain at least:
  - ctx.timeseries_dir  (from --timeseries-dir; used as BASE_DIR)
  - optionally ctx.theme (defaults to "dark" if missing)
"""

from __future__ import annotations

import os
import glob
from functools import lru_cache
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
import pandas as pd

from dash import dcc, html, Input, Output
import plotly.graph_objs as go
from ..metrics import residue_display
from .shared import R_GAS


# -------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------

R_GAS_KJ_MOLK = R_GAS  # alias for local usage


# -------------------------------------------------------------------
# Context helpers
# -------------------------------------------------------------------

def _get_data_root(ctx: Any) -> str:
    """
    Extract data root from ctx.

    Prefer ctx.timeseries_dir (from --timeseries-dir),
    fall back to ctx.base_dir if present, otherwise ".".
    """
    # Attribute-style ctx
    if hasattr(ctx, "timeseries_dir"):
        v = getattr(ctx, "timeseries_dir")
        if v:
            return v
    if hasattr(ctx, "base_dir"):
        v = getattr(ctx, "base_dir")
        if v:
            return v

    # Dict-style ctx
    if isinstance(ctx, dict):
        if ctx.get("timeseries_dir"):
            return ctx["timeseries_dir"]
        if ctx.get("base_dir"):
            return ctx["base_dir"]

    return "."


def _get_theme(ctx: Any, default: str = "dark") -> str:
    """Extract theme from ctx if present."""
    if hasattr(ctx, "theme"):
        return getattr(ctx, "theme") or default
    if isinstance(ctx, dict) and "theme" in ctx:
        return ctx["theme"] or default
    return default


from ..theming.errors import error_fig as _error_fig_base

def _template(theme: str = "dark") -> str:
    return "plotly_dark" if (theme or "dark") == "dark" else "plotly_white"

_empty_fig = _error_fig_base




def _periodic_gaussian_smooth_2d(Z: np.ndarray, sigma_bins: float) -> np.ndarray:
    """Periodic (wrap-around) Gaussian smoothing on a 2D grid.

    σ is expressed in *bins* (grid index units). σ=0 returns Z unchanged.
    """
    Z = np.asarray(Z, dtype=float)
    if Z.size == 0:
        return Z
    sigma = float(sigma_bins or 0.0)
    if sigma <= 0.0:
        return Z

    n, m = Z.shape
    # Build a cyclic Gaussian kernel centered at (0,0) in index space.
    iy = np.arange(n)
    ix = np.arange(m)
    dy = np.minimum(iy, n - iy).astype(float)
    dx = np.minimum(ix, m - ix).astype(float)
    Dy2 = dy[:, None] ** 2
    Dx2 = dx[None, :] ** 2
    K = np.exp(-0.5 * (Dy2 + Dx2) / (sigma ** 2))
    s = float(np.sum(K))
    if s > 0:
        K = K / s

    # Cyclic convolution via FFT (periodic BCs by construction)
    Zf = np.fft.rfftn(Z)
    Kf = np.fft.rfftn(K)
    out = np.fft.irfftn(Zf * Kf, s=Z.shape)
    return np.asarray(out.real, dtype=float)
# -------------------------------------------------------------------
# CSV-based helpers (cgrama2d / cgrama2d_perres DataFrames)
# -------------------------------------------------------------------

def _df_to_grid(
    df: pd.DataFrame,
    variant: str,
    residue: Optional[int] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """Pivot a (variant[, residue]) slice of a cgrama2d DataFrame into a 2D grid.

    Returns (phi_vals, psi_vals, Z) where Z.shape == (n_psi, n_phi).
    Returns (None, None, None) on any failure.
    """
    if df is None or df.empty:
        return None, None, None
    try:
        sub = df[df["variant"].astype(str) == str(variant)]
        if residue is not None and "residue" in sub.columns:
            sub = sub[pd.to_numeric(sub["residue"], errors="coerce") == int(residue)]
        if sub.empty:
            return None, None, None
        phi_vals = np.sort(pd.to_numeric(sub["x"], errors="coerce").dropna().unique())
        psi_vals = np.sort(pd.to_numeric(sub["y"], errors="coerce").dropna().unique())
        piv = sub.pivot_table(index="y", columns="x", values="density", aggfunc="sum")
        Z = piv.reindex(index=psi_vals, columns=phi_vals).fillna(0.0).to_numpy(dtype=float)
        return phi_vals, psi_vals, Z
    except Exception:
        return None, None, None


def _rama_2d_from_grid(
    phi_vals: np.ndarray,
    psi_vals: np.ndarray,
    Z: np.ndarray,
    theme: str,
    title: str,
    mode: str = "prob",
    gamma: float = 1.0,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    temp_K: float = 300.0,
    smooth_sigma: float = 0.0,
) -> go.Figure:
    """Build a Ramachandran heatmap from a pre-computed 2D density grid Z (psi × phi)."""
    if Z is None or Z.size == 0:
        return _empty_fig("No density grid available", theme)
    Z = np.maximum(np.asarray(Z, dtype=float), 0.0)
    try:
        Z = _periodic_gaussian_smooth_2d(Z, float(smooth_sigma or 0.0))
    except Exception:
        pass
    mode = (mode or "prob").lower()
    if mode == "pmf":
        if not temp_K or temp_K <= 0:
            temp_K = 300.0
        P = Z.copy()
        tot = float(P.sum())
        if tot <= 0:
            return _empty_fig("No probability mass", theme)
        P /= tot
        zero_mask = P <= 0.0
        with np.errstate(divide="ignore", invalid="ignore"):
            E = -R_GAS_KJ_MOLK * float(temp_K) * np.log(P)
        E[zero_mask] = np.nan
        finite = E[np.isfinite(E)]
        if finite.size == 0:
            return _empty_fig("PMF undefined", theme)
        E -= np.nanmin(E)
        Z_plot = E
        cb_title = f"PMF (kJ/mol, {int(round(temp_K))} K)"
        if zmin is None and zmax is None:
            zmin_used: Optional[float] = 0.0
            zmax_used: Optional[float] = float(np.percentile(finite, 95.0))
        elif zmin is not None and zmax is not None and zmax > zmin:
            zmin_used, zmax_used = float(zmin), float(zmax)
        else:
            zmin_used, zmax_used = None, None
    else:
        if not gamma or gamma <= 0:
            gamma = 1.0
        Z_plot = Z.copy()
        if abs(gamma - 1.0) > 1e-6:
            with np.errstate(invalid="ignore"):
                Z_plot = np.power(Z_plot, gamma)
        cb_title = "P(φ,ψ)"
        if zmin is not None and zmax is not None and zmax > zmin:
            zmin_used, zmax_used = float(zmin), float(zmax)
        else:
            zmin_used, zmax_used = None, None
    fig = go.Figure(data=[go.Heatmap(
        x=phi_vals, y=psi_vals, z=Z_plot,
        zmin=zmin_used, zmax=zmax_used,
        colorbar=dict(title=cb_title),
    )])
    fig.update_layout(
        template=_template(theme),
        title=title,
        xaxis=dict(title="φ (deg)", range=[-180, 180],
                   zeroline=True, zerolinewidth=1, zerolinecolor="rgba(255,255,255,0.4)"),
        yaxis=dict(title="ψ (deg)", range=[-180, 180],
                   scaleanchor="x", scaleratio=1.0,
                   zeroline=True, zerolinewidth=1, zerolinecolor="rgba(255,255,255,0.4)"),
        margin=dict(l=60, r=20, t=60, b=60),
    )
    return fig


def _get_variants_from_ctx(ctx: Any) -> List[str]:
    """Get sorted variant list from ctx, preferring the already-loaded features table."""
    # Fast path: features is loaded eagerly at startup — no lazy I/O
    try:
        feats = getattr(ctx, "features", None)
        if feats is not None and isinstance(feats, pd.DataFrame) and not feats.empty:
            if "variant" in feats.columns:
                return sorted(feats["variant"].dropna().astype(str).unique().tolist())
    except Exception:
        pass
    # Fallback: rama2d_pooled_df (lazy, loads on first access)
    try:
        df = getattr(ctx, "rama2d_pooled_df", None)
        if df is not None and isinstance(df, pd.DataFrame) and not df.empty and "variant" in df.columns:
            return sorted(df["variant"].dropna().astype(str).unique().tolist())
    except Exception:
        pass
    # Legacy XVG subdirectory scan
    return discover_variants(_get_data_root(ctx))


# -------------------------------------------------------------------
# Data discovery & loading
# -------------------------------------------------------------------

def discover_variants(base_dir: str) -> List[str]:
    """
    Discover variants as direct subdirectories under base_dir.

    Assumes layout:
        base_dir/
          VARIANT_A/
            *.xvg
          VARIANT_B/
            *.xvg
          ...
    """
    if not os.path.isdir(base_dir):
        return []

    variants: List[str] = []
    for entry in os.listdir(base_dir):
        vdir = os.path.join(base_dir, entry)
        if os.path.isdir(vdir):
            variants.append(entry)

    return sorted(variants)


def _parse_hist_xvg(path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Parse a gmx gangle histogram xvg file:

        angle  h_1  h_2  ... h_N

    Returns:
        angles: shape (nbins,)
        hist:   shape (nbins, n_dihedrals)
    """
    angles: List[float] = []
    rows: List[List[float]] = []

    with open(path, "r", errors="ignore") as fh:
        for ln in fh:
            if not ln:
                continue
            c0 = ln[0]
            if c0 in ("@", "#"):
                continue
            parts = ln.split()
            if len(parts) < 2:
                continue
            try:
                ang = float(parts[0])
                vals = [float(x) for x in parts[1:]]
            except Exception:
                continue
            angles.append(ang)
            rows.append(vals)

    if not rows:
        return np.array([]), np.zeros((0, 0))

    angles_arr = np.asarray(angles, float)
    hist_arr = np.asarray(rows, float)  # (nbins, n_dihedrals)

    return angles_arr, hist_arr


@lru_cache(maxsize=64)
def load_phi_psi_histograms(
    base_dir: str,
    variant: str,
) -> Dict[str, Optional[np.ndarray]]:
    """
    Load and pool φ/ψ histograms for a given variant across all replicas.

    Returns dict with:
        "phi_angles": np.ndarray (nbins,)
        "psi_angles": np.ndarray (nbins,)
        "phi_hist":   np.ndarray (n_res, nbins)  [pooled over replicas]
        "psi_hist":   np.ndarray (n_res, nbins)  [pooled over replicas]
    Or None/empty if nothing is found.
    """
    vdir = os.path.join(base_dir, variant)
    if not os.path.isdir(vdir):
        return dict(phi_angles=None, psi_angles=None,
                    phi_hist=None, psi_hist=None)

    phi_files = sorted(glob.glob(os.path.join(vdir, "md_cluster_pbc_*_phi_hist.xvg")))
    psi_files = sorted(glob.glob(os.path.join(vdir, "md_cluster_pbc_*_psi_hist.xvg")))

    phi_angles = None
    psi_angles = None
    phi_hist_sum = None  # shape (n_res, nbins)
    psi_hist_sum = None  # shape (n_res, nbins)

    # Pool φ over replicas
    for f in phi_files:
        ang, hist = _parse_hist_xvg(f)  # hist: (nbins, n_dihedrals)
        if hist.size == 0:
            continue
        if phi_angles is None:
            phi_angles = ang
        else:
            if len(ang) != len(phi_angles):
                # Different binning – skip this replica
                continue
        # transpose -> (n_res, nbins)
        hT = hist.T
        if phi_hist_sum is None:
            phi_hist_sum = hT.copy()
        else:
            n_res = min(phi_hist_sum.shape[0], hT.shape[0])
            n_bins = min(phi_hist_sum.shape[1], hT.shape[1])
            phi_hist_sum[:n_res, :n_bins] += hT[:n_res, :n_bins]

    # Pool ψ over replicas
    for f in psi_files:
        ang, hist = _parse_hist_xvg(f)
        if hist.size == 0:
            continue
        if psi_angles is None:
            psi_angles = ang
        else:
            if len(ang) != len(psi_angles):
                continue
        hT = hist.T
        if psi_hist_sum is None:
            psi_hist_sum = hT.copy()
        else:
            n_res = min(psi_hist_sum.shape[0], hT.shape[0])
            n_bins = min(psi_hist_sum.shape[1], hT.shape[1])
            psi_hist_sum[:n_res, :n_bins] += hT[:n_res, :n_bins]

    # Align number of residues between phi and psi
    if (phi_hist_sum is not None) and (psi_hist_sum is not None):
        n_res = min(phi_hist_sum.shape[0], psi_hist_sum.shape[0])
        phi_hist_sum = phi_hist_sum[:n_res, :]
        psi_hist_sum = psi_hist_sum[:n_res, :]

    return dict(
        phi_angles=phi_angles,
        psi_angles=psi_angles,
        phi_hist=phi_hist_sum,
        psi_hist=psi_hist_sum,
    )


def _rama_2d_from_marginals(
    phi_angles: np.ndarray,
    psi_angles: np.ndarray,
    phi_hist: np.ndarray,
    psi_hist: np.ndarray,
    theme: str,
    title: str,
    mode: str = "prob",      # "prob" or "pmf"
    gamma: float = 1.0,
    zmin: Optional[float] = None,
    zmax: Optional[float] = None,
    temp_K: float = 300.0,
    smooth_sigma: float = 0.0,
) -> go.Figure:
    """
    Construct a 2D Ramachandran map from 1D φ/ψ histograms.

    Steps:
      1) Build probability:
             P(φ,ψ) ≈ P(φ)*P(ψ)
      2) If mode == "pmf":
             E(φ,ψ) = -R * T * ln P_norm(φ,ψ) + const
         with:
             R = 8.314462618e-3 kJ·mol⁻¹·K⁻¹
             T = temp_K (K)
         and const chosen so that min(E) = 0 kJ/mol.
         Bins with zero probability are shown as NaN (no color) and
         do NOT blow up the color range.
      3) If mode == "prob":
             use P(φ,ψ), optionally with gamma:
             P_gamma = P ** gamma
      4) Apply optional zmin/zmax limits for the color scale.
         In PMF mode, if zmin/zmax are not provided, we auto-cap zmax
         at the 95th percentile and set zmin=0.
    """
    if (
        phi_angles is None
        or psi_angles is None
        or phi_hist is None
        or psi_hist is None
        or phi_hist.size == 0
        or psi_hist.size == 0
    ):
        return _empty_fig("No Ramachandran data for this selection", theme)

    phi_hist = np.asarray(phi_hist, float)
    psi_hist = np.asarray(psi_hist, float)

    phi_hist = np.maximum(phi_hist, 0.0)
    psi_hist = np.maximum(psi_hist, 0.0)

    s_phi = phi_hist.sum()
    s_psi = psi_hist.sum()
    if s_phi <= 0:
        phi_pdf = np.ones_like(phi_hist) / len(phi_hist)
    else:
        phi_pdf = phi_hist / s_phi
    if s_psi <= 0:
        psi_pdf = np.ones_like(psi_hist) / len(psi_hist)
    else:
        psi_pdf = psi_hist / s_psi

    # Outer product: rows = ψ bins, cols = φ bins
    Z_prob = np.outer(psi_pdf, phi_pdf)  # (n_psi, n_phi)

    # Optional periodic smoothing (reduces wrap seam artifacts)
    try:
        Z_prob = _periodic_gaussian_smooth_2d(Z_prob, float(smooth_sigma or 0.0))
    except Exception:
        pass

    mode = (mode or "prob").lower()

    # ---- PMF branch: E = -R*T*ln P, finite range, zeros masked ----
    if mode == "pmf":
        # sanitize temperature
        if temp_K is None or temp_K <= 0:
            temp_K = 300.0

        # Normalize & mask zeros -> NaN (no contribution, no crazy energies)
        P = np.maximum(Z_prob, 0.0)
        tot = P.sum()
        if tot <= 0:
            return _empty_fig("No probability mass for PMF", theme)
        P /= tot

        # bins that are exactly zero after normalization
        zero_mask = P <= 0.0

        with np.errstate(divide="ignore", invalid="ignore"):
            E = -R_GAS_KJ_MOLK * float(temp_K) * np.log(P)
        # set zero-probability bins to NaN so they don't stretch the scale
        E[zero_mask] = np.nan

        # shift so min(E) over finite bins is zero
        finite = E[np.isfinite(E)]
        if finite.size == 0:
            return _empty_fig("PMF is undefined (no finite probabilities)", theme)
        E -= np.nanmin(E)
        Z_plot = E
        cb_title = f"PMF (kJ/mol, {int(round(temp_K))} K)"

        # automatic color range for PMF if user didn't specify
        if zmin is None and zmax is None:
            # robust upper cap (95th percentile), lower fixed at 0
            zmin_used = 0.0
            zmax_used = float(np.percentile(finite, 95.0))
        else:
            if zmin is not None and zmax is not None and zmax > zmin:
                zmin_used = float(zmin)
                zmax_used = float(zmax)
            else:
                zmin_used = None
                zmax_used = None

    # ---- Probability branch: P(φ,ψ) with optional gamma ----
    else:
        if gamma is None or gamma <= 0:
            gamma = 1.0
        Z_plot = Z_prob
        if abs(gamma - 1.0) > 1e-6:
            with np.errstate(invalid="ignore"):
                Z_plot = np.power(Z_plot, gamma)
        cb_title = "P(φ,ψ)"

        # For probability mode, we only apply zmin/zmax if user gives them.
        if zmin is not None and zmax is not None and zmax > zmin:
            zmin_used = float(zmin)
            zmax_used = float(zmax)
        else:
            zmin_used = None
            zmax_used = None

    heatmap = go.Heatmap(
        x=phi_angles,
        y=psi_angles,
        z=Z_plot,
        zmin=zmin_used,
        zmax=zmax_used,
        colorbar=dict(title=cb_title),
    )

    fig = go.Figure(data=[heatmap])

    fig.update_layout(
        template=_template(theme),
        title=title,
        xaxis=dict(
            title="φ (deg)",
            range=[-180, 180],
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="rgba(255,255,255,0.4)",
        ),
        yaxis=dict(
            title="ψ (deg)",
            range=[-180, 180],
            scaleanchor="x",
            scaleratio=1.0,
            zeroline=True,
            zerolinewidth=1,
            zerolinecolor="rgba(255,255,255,0.4)",
        ),
        margin=dict(l=60, r=20, t=60, b=60),
    )

    return fig



# -------------------------------------------------------------------
# Internal layout helper
# -------------------------------------------------------------------

def _layout_impl(base_dir: str, theme: str, variants: Optional[List[str]] = None) -> html.Div:
    if variants is None:
        variants = discover_variants(base_dir)

    return html.Div(
        id="rama2d-tab",
        children=[
            # Top control row
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Variant", className="control-label"),
                            dcc.Dropdown(
                                id="rama2d-variant-dropdown",
                                options=[{"label": v, "value": v} for v in variants],
                                value=None,
                                placeholder=(
                                    "No variants found under data root"
                                    if not variants else "Select a variant …"
                                ),
                                clearable=False,
                            ),
                        ],
                        style={"flex": "2", "minWidth": "260px", "marginRight": "1rem"},
                    ),
                    html.Div(
                        [
                            html.Label("View mode", className="control-label"),
                            dcc.RadioItems(
                                id="rama2d-view-mode",
                                options=[
                                    {"label": "Pooled (all residues)", "value": "pooled"},
                                    {"label": "Per residue", "value": "perres"},
                                ],
                                value="perres",
                                labelStyle={
                                    "display": "inline-block",
                                    "marginRight": "1rem",
                                },
                            ),
                        ],
                        style={"flex": "2", "minWidth": "260px", "marginRight": "1rem"},
                    ),
                    html.Div(
                        [
                            html.Label("Residue / dihedral index", className="control-label"),
                            dcc.Dropdown(
                                id="rama2d-residue-dropdown",
                                options=[],
                                value=None,
                                clearable=False,
                            ),
                        ],
                        style={"flex": "1", "minWidth": "220px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "alignItems": "flex-end",
                    "gap": "0.5rem",
                    "marginBottom": "0.5rem",
                },
            ),
            # Value / color controls
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Value mode", className="control-label"),
                            dcc.RadioItems(
                                id="rama2d-value-mode",
                                options=[
                                    {"label": "Probability", "value": "prob"},
                                    {"label": "PMF (kJ/mol)", "value": "pmf"},
                                ],
                                value="prob",
                                labelStyle={
                                    "display": "inline-block",
                                    "marginRight": "1rem",
                                },
                            ),
                        ],
                        style={"flex": "2", "minWidth": "220px", "marginRight": "1rem"},
                    ),
                    html.Div(
                        [
                            html.Label("Temperature (K, PMF)", className="control-label"),
                            dcc.Input(
                                id="rama2d-temp-input",
                                type="number",
                                min=1,
                                step=10,
                                value=300,
                                style={"width": "100%"},
                            ),
                        ],
                        style={"flex": "1", "minWidth": "150px", "marginRight": "1rem"},
                    ),
                    html.Div(
                        [
                            html.Label("Color mapping (γ, prob only)", className="control-label"),
                            dcc.Slider(
                                id="rama2d-gamma-slider",
                                min=0.3,
                                max=3.0,
                                step=0.1,
                                value=1.0,
                                marks={
                                    0.5: "0.5",
                                    1.0: "1.0",
                                    2.0: "2.0",
                                    3.0: "3.0",
                                },
                                tooltip={"always_visible": False, "placement": "bottom"},
                            ),
                        ],
                        style={"flex": "3", "minWidth": "260px", "marginRight": "1rem"},
                    ),
                    html.Div(
                        [
                            html.Label("Color range (z min / z max)", className="control-label"),
                            html.Div(
                                [
                                    
                                    html.Label("Periodic smoothing σ (bins)", className="control-label"),
                                    dcc.Slider(
                                        id="rama2d-smooth-sigma",
                                        min=0.0,
                                        max=3.0,
                                        step=0.25,
                                        value=0.0,
                                        tooltip={"placement": "bottom"},
                                    ),
                                    html.Small(
                                        "σ=0 disables; periodic wrap-around smoothing (avoids seam artifacts).",
                                        style={"opacity": 0.7, "fontSize": "0.75em"},
                                    ),
dcc.Input(
                                        id="rama2d-zmin-input",
                                        type="number",
                                        placeholder="auto min",
                                        style={"width": "45%", "marginRight": "0.5rem"},
                                    ),
                                    dcc.Input(
                                        id="rama2d-zmax-input",
                                        type="number",
                                        placeholder="auto max",
                                        style={"width": "45%"},
                                    ),
                                ],
                                style={"display": "flex"},
                            ),
                        ],
                        style={"flex": "3", "minWidth": "260px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "alignItems": "flex-end",
                    "gap": "0.5rem",
                    "marginBottom": "1rem",
                },
            ),
            # Figure
            dcc.Graph(
                id="rama2d-graph",
                figure=_empty_fig("Select a variant to see Ramachandran plot", theme),
                style={"height": "70vh"},
            ),
        ],
    )


# -------------------------------------------------------------------
# Public API: layout(ctx) & register_callbacks(app, ctx)
# -------------------------------------------------------------------

def layout(ctx: Any) -> html.Div:
    """
    Public layout entry point, matching other tabs:
        dcc.Tab(..., children=rama2d_tab.layout(ctx))

    Prefers ctx.rama2d_pooled_df / ctx.rama2d_perres_df (CSV data from GLOBAL_DATA).
    Falls back to ctx.timeseries_dir XVG files when CSV data is absent.
    """
    base_dir = _get_data_root(ctx)
    theme = _get_theme(ctx, default="light")
    variants = _get_variants_from_ctx(ctx)
    return _layout_impl(base_dir=base_dir, theme=theme, variants=variants)


def register_callbacks(app, ctx: Any):
    """
    Public callback registration, matching other tabs:
        rama2d_tab.register_callbacks(app, ctx)
    """
    base_dir = _get_data_root(ctx)
    theme = _get_theme(ctx, default="light")

    @app.callback(
        Output("rama2d-residue-dropdown", "options"),
        Output("rama2d-residue-dropdown", "value"),
        Input("rama2d-variant-dropdown", "value"),
        Input("rama2d-view-mode", "value"),
        prevent_initial_call=True,
    )
    def _update_residue_options(variant: str, view_mode: str):
        if not variant:
            return [], None

        # CSV path: use per-residue DataFrame
        try:
            df_perres = getattr(ctx, "rama2d_perres_df", None)
            if df_perres is not None and not df_perres.empty and "residue" in df_perres.columns:
                mask = df_perres["variant"].astype(str) == str(variant)
                residues = sorted(
                    pd.to_numeric(df_perres.loc[mask, "residue"], errors="coerce")
                    .dropna().astype(int).unique().tolist()
                )
                if residues:
                    options = [{"label": residue_display(variant, r), "value": r} for r in residues]
                    return options, residues[0]
        except Exception:
            pass

        # XVG fallback
        data = load_phi_psi_histograms(base_dir, variant)
        phi_hist = data.get("phi_hist")
        psi_hist = data.get("psi_hist")
        if phi_hist is None or psi_hist is None:
            return [], None
        n_res = min(phi_hist.shape[0], psi_hist.shape[0])
        if n_res <= 0:
            return [], None
        residues = list(range(1, n_res + 1))
        return [{"label": residue_display(variant, r), "value": r} for r in residues], residues[0]

    @app.callback(
        Output("rama2d-graph", "figure"),
        Input("rama2d-variant-dropdown", "value"),
        Input("rama2d-view-mode", "value"),
        Input("rama2d-residue-dropdown", "value"),
        Input("rama2d-value-mode", "value"),
        Input("rama2d-temp-input", "value"),
        Input("rama2d-gamma-slider", "value"),
        Input("rama2d-smooth-sigma", "value"),
        Input("rama2d-zmin-input", "value"),
        Input("rama2d-zmax-input", "value"),
        Input("theme-store", "data"),
        prevent_initial_call=True,
    )
    def _update_rama_figure(
        variant: str,
        view_mode: str,
        residue_index: Optional[int],
        value_mode: str,
        temp_K: Optional[float],
        gamma: float,
        smooth_sigma: float,
        zmin: Optional[float],
        zmax: Optional[float],
        theme_input,
    ):
        nonlocal theme
        theme = theme_input or theme
        if not variant:
            return _empty_fig("No variant selected", theme)
        if gamma is None or gamma <= 0:
            gamma = 1.0
        if temp_K is None or temp_K <= 0:
            temp_K = 300.0
        mode = (value_mode or "prob").lower()

        # CSV path: prefer pre-computed 2D density grids from GLOBAL_DATA
        try:
            if view_mode == "pooled":
                df = getattr(ctx, "rama2d_pooled_df", None)
                if df is not None and not df.empty:
                    phi_vals, psi_vals, Z = _df_to_grid(df, variant)
                    if Z is not None and Z.size > 0:
                        if mode == "pmf":
                            title = f"Ramachandran PMF — pooled ({variant}, {int(round(temp_K))} K)"
                        else:
                            title = f"Ramachandran 2D — pooled ({variant}), γ={gamma:.1f}"
                        return _rama_2d_from_grid(
                            phi_vals, psi_vals, Z, theme, title,
                            mode=mode, gamma=gamma, zmin=zmin, zmax=zmax,
                            temp_K=temp_K, smooth_sigma=smooth_sigma,
                        )
            else:
                if residue_index is None:
                    residue_index = 1
                df = getattr(ctx, "rama2d_perres_df", None)
                if df is not None and not df.empty:
                    phi_vals, psi_vals, Z = _df_to_grid(df, variant, residue=int(residue_index))
                    if Z is not None and Z.size > 0:
                        if mode == "pmf":
                            title = f"Ramachandran PMF — {variant}, residue {residue_index}, {int(round(temp_K))} K"
                        else:
                            title = f"Ramachandran 2D — {variant}, residue {residue_index}, γ={gamma:.1f}"
                        return _rama_2d_from_grid(
                            phi_vals, psi_vals, Z, theme, title,
                            mode=mode, gamma=gamma, zmin=zmin, zmax=zmax,
                            temp_K=temp_K, smooth_sigma=smooth_sigma,
                        )
        except Exception:
            pass

        # XVG fallback
        data = load_phi_psi_histograms(base_dir, variant)
        phi_angles = data.get("phi_angles")
        psi_angles = data.get("psi_angles")
        phi_hist = data.get("phi_hist")
        psi_hist = data.get("psi_hist")
        if any(x is None for x in [phi_angles, psi_angles, phi_hist, psi_hist]):
            return _empty_fig("No Ramachandran data for this variant", theme)
        n_res = min(phi_hist.shape[0], psi_hist.shape[0])
        if n_res <= 0:
            return _empty_fig("No dihedrals found for this variant", theme)

        if view_mode == "pooled":
            phi_pooled = phi_hist[:n_res, :].sum(axis=0)
            psi_pooled = psi_hist[:n_res, :].sum(axis=0)
            if mode == "pmf":
                title = f"Ramachandran PMF — pooled over residues ({variant}, {int(round(temp_K))} K)"
            else:
                title = f"Ramachandran 2D — pooled over residues ({variant}), γ={gamma:.1f}"
            return _rama_2d_from_marginals(
                phi_angles, psi_angles, phi_pooled, psi_pooled, theme, title,
                mode=mode, gamma=gamma, zmin=zmin, zmax=zmax, temp_K=temp_K, smooth_sigma=smooth_sigma,
            )

        if residue_index is None:
            residue_index = 1
        idx = max(0, min(n_res - 1, int(residue_index) - 1))
        if mode == "pmf":
            title = f"Ramachandran PMF — {variant}, dihedral {residue_index}, {int(round(temp_K))} K"
        else:
            title = f"Ramachandran 2D — {variant}, dihedral {residue_index}, γ={gamma:.1f}"
        return _rama_2d_from_marginals(
            phi_angles, psi_angles, phi_hist[idx, :], psi_hist[idx, :], theme, title,
            mode=mode, gamma=gamma, zmin=zmin, zmax=zmax, temp_K=temp_K, smooth_sigma=smooth_sigma,
        )
