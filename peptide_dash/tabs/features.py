from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objs as go
from plotly.subplots import make_subplots
from dash import html, dcc, Input, Output, State
from dash import dash_table
from ..metrics import prettify_column_label, torsion_sort_key
from ..data.context import filter_numeric_columns
from .pmf_plot_ci import _PALETTE as _COLOR_PALETTE


# ---------------------------------------------------------------------
# Color & numeric helpers
# ---------------------------------------------------------------------


def _distinct_colors(n: int) -> List[str]:
    if n <= len(_COLOR_PALETTE):
        return list(_COLOR_PALETTE[:n])
    return [_COLOR_PALETTE[i % len(_COLOR_PALETTE)] for i in range(n)]


def _nanpercentile(vec: np.ndarray, q_low: float, q_high: float) -> Tuple[float, float]:
    v = np.asarray(vec, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return 0.0, 1.0
    lo, hi = np.nanpercentile(v, [q_low, q_high])
    if not np.isfinite(lo):
        lo = float(np.nanmin(v))
    if not np.isfinite(hi):
        hi = float(np.nanmax(v))
    if lo == hi:
        hi = lo + 1.0
    return float(lo), float(hi)


def _prepare_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Prepare numeric view for plotting.

    - Keeps all rows (no dropna), so short variants (e.g. L=8) are not silently removed
    - Coerces requested columns to numeric where possible
    - If a column is not 1D (e.g. weird object/array-of-arrays), it is skipped
    """
    # Guard against non-DataFrame df
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)

    # Filter out Nones / non-existent columns
    cols = [c for c in cols if c is not None and c in df.columns]
    if not cols:
        return pd.DataFrame()

    sub = df[cols].copy()

    valid_cols: list[str] = []
    for c in cols:
        try:
            # This will fail with TypeError if sub[c] is not 1D (e.g. DataFrame)
            sub[c] = pd.to_numeric(sub[c], errors="coerce")
            valid_cols.append(c)
        except TypeError:
            # Non-1D / weird column – skip it for numeric plotting
            continue

    if not valid_cols:
        return pd.DataFrame()

    sub = sub[valid_cols]
    # Clean infs, but DO NOT drop NaN rows – plotting code handles NaNs
    sub = sub.replace([np.inf, -np.inf], np.nan)
    return sub


def _build_color(
    df: pd.DataFrame,
    base_index: pd.Index,
    color_by: Optional[str],
) -> Tuple[Optional[np.ndarray], bool, Optional[np.ndarray]]:
    """
    Build a color vector aligned to base_index.

    Returns:
        color_values : array or None
        is_numeric : bool
        raw_labels : array or None (string labels for hover, if useful)
    """
    if not color_by or color_by not in df.columns:
        return None, False, None

    s = df[color_by].loc[base_index]
    # Numeric color
    if pd.api.types.is_numeric_dtype(s):
        arr = pd.to_numeric(s, errors="coerce").to_numpy(dtype=float)
        return arr, True, None

    # Categorical color: encode to ints, keep labels
    s_str = s.astype(str)
    codes, uniques = pd.factorize(s_str, sort=True)
    return codes.astype(float), False, s_str.to_numpy()


# ---------------------------------------------------------------------
# Feature family helpers
# ---------------------------------------------------------------------

SEQ_FEATURES = {
    "L",
    "sequence",
    "net_charge",
    "net_charge_per_res",
    "kd_mean",
    "kd_std",
    "KD_mean",
    "KD_std",
    "KD_min",
    "KD_max",
    "KD_Nterm",
    "KD_Cterm",
    "frac_aromatic",
    "frac_pro",
    "frac_gly",
    "frac_polar",
}


def _feature_family_name(col: str) -> str:
    """
    Infer a 'family' from a feature column name.

    - Sequence descriptors      → 'seq'
    - Names with 'metric__...'  → prefix before first '__'
    - Everything else           → 'misc'
    """
    c = str(col)
    if c.startswith("seq_") or c in SEQ_FEATURES:
        return "seq"
    if "__" in c:
        return c.split("__", 1)[0]
    return "misc"


def _compute_family_map(numeric_cols: List[str]) -> dict[str, List[str]]:
    fam_map: dict[str, List[str]] = {}
    for c in numeric_cols:
        fam = _feature_family_name(c)
        fam_map.setdefault(fam, []).append(c)
    # deterministic order inside each family
    for fam in fam_map:
        fam_map[fam] = sorted(fam_map[fam])
    return fam_map


# Mapping: (mean-type suffix, spread suffix, scale applied to spread value).
# IQR covers Q25–Q75, so half-IQR ≈ 0.74σ; we expose it as the semi-axis.
_SPREAD_MAP: List[Tuple[str, str, float]] = [
    ("mean",               "std",               1.0),
    ("median",             "iqr",               0.5),
    ("circular_mean_deg",  "circular_std_deg",  1.0),
]


def _find_spread_col(
    all_cols: set, feature_col: str
) -> Tuple[Optional[str], float]:
    """
    Return (spread_column_name, scale) for a feature column, or (None, 1.0).

    Examples
    --------
    phi__mean   → ("phi__std",  1.0)
    phi__median → ("phi__iqr",  0.5)   # half-IQR as semi-axis
    phi__effective_support_frac → (None, 1.0)   # no spread available
    """
    if "__" not in feature_col:
        return None, 1.0
    prefix, suffix = feature_col.split("__", 1)
    for mean_suf, spread_suf, scale in _SPREAD_MAP:
        if suffix == mean_suf:
            candidate = f"{prefix}__{spread_suf}"
            if candidate in all_cols:
                return candidate, scale
    return None, 1.0


# ---------------------------------------------------------------------
# Plot builders
# ---------------------------------------------------------------------


def _make_scatter_with_histograms(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    marker: Optional[dict],
    text: Optional[np.ndarray],
    hovertemplate: str,
    customdata: Optional[np.ndarray],
    show_density: bool = False,
    include_scatter: bool = True,
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
) -> go.Figure:
    """2D scatter with marginal histograms for X (top) and Y (right)."""
    fig = make_subplots(
        rows=2,
        cols=2,
        column_widths=[0.80, 0.20],
        row_heights=[0.20, 0.80],
        specs=[
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        shared_xaxes=True,
        shared_yaxes=True,
        horizontal_spacing=0.04,
        vertical_spacing=0.04,
    )

    # Optional 2D density contour underlay
    if show_density:
        fig.add_trace(
            go.Histogram2dContour(
                x=df[xcol],
                y=df[ycol],
                nbinsx=90,
                nbinsy=90,
                colorscale="Blues",
                showscale=False,
                contours=dict(showlines=False),
                opacity=0.5,
                name="density",
            ),
            row=2,
            col=1,
        )

    # Main scatter (if requested)
    if include_scatter and marker is not None:
        fig.add_trace(
            go.Scatter(
                x=df[xcol],
                y=df[ycol],
                mode="markers",
                marker=marker,
                name="points",
                text=text,
                hovertemplate=hovertemplate,
                customdata=customdata,
            ),
            row=2,
            col=1,
        )

    # X histogram (top) – thinner bins
    fig.add_trace(
        go.Histogram(
            x=df[xcol],
            nbinsx=90,
            marker=dict(opacity=0.6),
            showlegend=False,
        ),
        row=1,
        col=1,
    )

    # Y histogram (right, horizontal) – thinner bins
    fig.add_trace(
        go.Histogram(
            y=df[ycol],
            nbinsy=90,
            marker=dict(opacity=0.6),
            showlegend=False,
            orientation="h",
        ),
        row=2,
        col=2,
    )

    # Labels & axis cosmetics
    fig.update_xaxes(title_text=xlabel if xlabel is not None else xcol, row=2, col=1)
    fig.update_yaxes(title_text=ylabel if ylabel is not None else ycol, row=2, col=1)

    fig.update_yaxes(showticklabels=False, row=1, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=2)

    fig.update_layout(
        margin=dict(l=50, r=10, t=40, b=50),
        height=650,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def _make_scatter_3d(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    zcol: str,
    marker: dict,
    text: Optional[np.ndarray],
    hovertemplate: str,
    customdata: Optional[np.ndarray],
    xlabel: Optional[str] = None,
    ylabel: Optional[str] = None,
    zlabel: Optional[str] = None,
) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter3d(
            x=df[xcol],
            y=df[ycol],
            z=df[zcol],
            mode="markers",
            marker=marker,
            name="points",
            text=text,
            hovertemplate=hovertemplate,
            customdata=customdata,
        )
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        height=650,
        scene=dict(
            xaxis_title=xlabel if xlabel is not None else xcol,
            yaxis_title=ylabel if ylabel is not None else ycol,
            zaxis_title=zlabel if zlabel is not None else zcol,
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ---------------------------------------------------------------------
# Gradient estimation & arrows
# ---------------------------------------------------------------------


_GRADIENT_MAX_POINTS = 8_000


def _local_gradients_2d(
    x: np.ndarray,
    y: np.ndarray,
    f: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    bandwidth: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Estimate ∂f/∂x, ∂f/∂y on a regular grid via vectorised locally-weighted
    linear regression with Gaussian weights.

    Points are subsampled to _GRADIENT_MAX_POINTS when n is large so that the
    (G, n, 3) working array stays under ~200 MB.
    """
    bw2 = float(bandwidth) ** 2
    n = len(x)

    # Subsample for memory safety; gradient estimation is visualization only
    if n > _GRADIENT_MAX_POINTS:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, _GRADIENT_MAX_POINTS, replace=False)
        x, y, f = x[idx], y[idx], f[idx]
        n = _GRADIENT_MAX_POINTS

    Gy, Gx = gy.size, gx.size
    G = Gy * Gx

    # Design matrix: (n, 3)
    X_design = np.column_stack([x, y, np.ones(n)])

    # Grid centres: (G,)
    cx_grid, cy_grid = np.meshgrid(gx, gy)
    cx_flat = cx_grid.ravel()
    cy_flat = cy_grid.ravel()

    # Weights: (G, n)
    d2 = (x[None, :] - cx_flat[:, None]) ** 2 + (y[None, :] - cy_flat[:, None]) ** 2
    w = np.exp(-0.5 * d2 / (bw2 + 1e-12))
    w[~np.isfinite(w)] = 0.0

    active = w.sum(axis=1) > 1e-9  # (G,) — skip cells with negligible coverage

    # Weighted normal equations via einsum
    XW = w[:, :, None] * X_design[None, :, :]       # (G, n, 3)
    A = np.einsum("gnk,nl->gkl", XW, X_design)      # (G, 3, 3)
    b = np.einsum("gnk,n->gk", XW, f)               # (G, 3)

    # Tiny ridge to keep inactive cells non-singular
    A[:, 0, 0] += 1e-12
    A[:, 1, 1] += 1e-12
    A[:, 2, 2] += 1e-12

    try:
        beta = np.linalg.solve(A, b)  # (G, 3)
    except np.linalg.LinAlgError:
        beta = np.zeros((G, 3))

    beta[~active] = 0.0

    U = beta[:, 0].reshape(Gy, Gx)
    V = beta[:, 1].reshape(Gy, Gx)

    return gx, gy, U, V


def _quiver_trace_2d(
    gx: np.ndarray,
    gy: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    color: str,
    scale: float,
    normalize: bool,
    name: str,
) -> Optional[go.Scatter]:
    """
    Build quiver-like arrows on the (gx, gy) grid with actual arrowheads.
    """
    if U.size == 0 or V.size == 0:
        return None

    Xg, Yg = np.meshgrid(gx, gy)
    U = U.astype(float)
    V = V.astype(float)

    mag = np.sqrt(U**2 + V**2)
    mask = np.isfinite(mag) & (mag > 1e-12)
    if not np.any(mask):
        return None

    Xc = Xg[mask]
    Yc = Yg[mask]
    Uc = U[mask]
    Vc = V[mask]
    magc = mag[mask]

    # Axis span
    x_span = float(gx.max() - gx.min()) if gx.size else 1.0
    y_span = float(gy.max() - gy.min()) if gy.size else 1.0
    span = max(x_span, y_span, 1e-6)

    if normalize:
        Uc = Uc / (magc + 1e-12)
        Vc = Vc / (magc + 1e-12)
        factor = float(scale) * 0.10 * span
        Uc *= factor
        Vc *= factor
    else:
        max_mag = np.nanpercentile(magc, 95)
        if max_mag <= 0 or not np.isfinite(max_mag):
            return None
        factor = float(scale) * 0.10 * span / max_mag
        Uc *= factor
        Vc *= factor

    # Arrowheads: two short segments at the tip
    tips_x = Xc + Uc
    tips_y = Yc + Vc

    dx = Uc
    dy = Vc
    d_norm = np.sqrt(dx**2 + dy**2)
    d_norm[d_norm == 0] = 1.0

    px = -dy / (d_norm + 1e-12)
    py = dx / (d_norm + 1e-12)

    head_len = 0.3
    head_width = 0.25

    hx1 = tips_x - head_len * dx + head_width * px * d_norm
    hy1 = tips_y - head_len * dy + head_width * py * d_norm
    hx2 = tips_x - head_len * dx - head_width * px * d_norm
    hy2 = tips_y - head_len * dy - head_width * py * d_norm

    xs = []
    ys = []

    for x0, y0, x1, y1, ax1, ay1, ax2, ay2 in zip(
        Xc, Yc, tips_x, tips_y, hx1, hy1, hx2, hy2
    ):
        xs.extend([x0, x1, np.nan])
        ys.extend([y0, y1, np.nan])
        xs.extend([x1, ax1, np.nan])
        ys.extend([y1, ay1, np.nan])
        xs.extend([x1, ax2, np.nan])
        ys.extend([y1, ay2, np.nan])

    return go.Scatter(
        x=np.array(xs, dtype=float),
        y=np.array(ys, dtype=float),
        mode="lines",
        line=dict(color=color, width=1.5),
        name=name,
        hoverinfo="skip",
    )


def _quiver_trace_3d_from_2d(
    gx: np.ndarray,
    gy: np.ndarray,
    U: np.ndarray,
    V: np.ndarray,
    z0: float,
    color: str,
    scale: float,
    normalize: bool,
    name: str,
) -> Optional[go.Scatter3d]:
    """
    3D quiver built from 2D gradients, drawn in a plane z = z0.
    """
    if U.size == 0 or V.size == 0:
        return None

    Xg, Yg = np.meshgrid(gx, gy)
    U = U.astype(float)
    V = V.astype(float)

    mag = np.sqrt(U**2 + V**2)
    mask = np.isfinite(mag) & (mag > 1e-12)
    if not np.any(mask):
        return None

    Xc = Xg[mask]
    Yc = Yg[mask]
    Uc = U[mask]
    Vc = V[mask]
    magc = mag[mask]

    # Axis span
    x_span = float(gx.max() - gx.min()) if gx.size else 1.0
    y_span = float(gy.max() - gy.min()) if gy.size else 1.0
    span = max(x_span, y_span, 1e-6)

    if normalize:
        Uc = Uc / (magc + 1e-12)
        Vc = Vc / (magc + 1e-12)
        factor = float(scale) * 0.10 * span
        Uc *= factor
        Vc *= factor
    else:
        max_mag = np.nanpercentile(magc, 95)
        if max_mag <= 0 or not np.isfinite(max_mag):
            return None
        factor = float(scale) * 0.10 * span / max_mag
        Uc *= factor
        Vc *= factor

    tips_x = Xc + Uc
    tips_y = Yc + Vc

    dx = Uc
    dy = Vc
    d_norm = np.sqrt(dx**2 + dy**2)
    d_norm[d_norm == 0] = 1.0

    px = -dy / (d_norm + 1e-12)
    py = dx / (d_norm + 1e-12)

    head_len = 0.3
    head_width = 0.25

    hx1 = tips_x - head_len * dx + head_width * px * d_norm
    hy1 = tips_y - head_len * dy + head_width * py * d_norm
    hx2 = tips_x - head_len * dx - head_width * px * d_norm
    hy2 = tips_y - head_len * dy - head_width * py * d_norm

    xs = []
    ys = []
    zs = []

    for x0, y0, x1, y1, ax1, ay1, ax2, ay2 in zip(
        Xc, Yc, tips_x, tips_y, hx1, hy1, hx2, hy2
    ):
        xs.extend([x0, x1, np.nan])
        ys.extend([y0, y1, np.nan])
        zs.extend([z0, z0, np.nan])
        xs.extend([x1, ax1, np.nan])
        ys.extend([y1, ay1, np.nan])
        zs.extend([z0, z0, np.nan])
        xs.extend([x1, ax2, np.nan])
        ys.extend([y1, ay2, np.nan])
        zs.extend([z0, z0, np.nan])

    return go.Scatter3d(
        x=np.array(xs, dtype=float),
        y=np.array(ys, dtype=float),
        z=np.array(zs, dtype=float),
        mode="lines",
        line=dict(color=color, width=1.5),
        name=name,
        hoverinfo="skip",
    )


# ---------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------


def layout(ctx):
    df = getattr(ctx, "df", pd.DataFrame())

    # Numeric feature columns (globally filtered using data.context.EXCLUDE_SUBSTRINGS)
    numeric_cols: List[str] = list(getattr(ctx, "numeric_cols", []) or [])
    if (not numeric_cols) and isinstance(df, pd.DataFrame) and not df.empty:
        raw = [c for c in df.columns if c != "variant" and pd.api.types.is_numeric_dtype(df[c])]
        numeric_cols = filter_numeric_columns(raw)

    # Variants for selector
    if isinstance(df, pd.DataFrame) and "variant" in df.columns:
        all_variants = (
            df["variant"]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )
    else:
        all_variants = []

    # Feature families
    family_map = _compute_family_map(numeric_cols) if numeric_cols else {}
    family_names = sorted(family_map.keys())
    family_options: List[dict] = []
    default_family = "__ALL__" if numeric_cols else None
    if numeric_cols:
        family_options.append({"label": "All", "value": "__ALL__"})
        for fam in family_names:
            family_options.append({"label": fam, "value": fam})

    # Color-by options
    color_options: List[dict] = []
    if isinstance(df, pd.DataFrame):
        if "variant" in df.columns:
            color_options.append({"label": "variant", "value": "variant"})
        for c in numeric_cols:
            color_options.append({"label": prettify_column_label(c), "value": c})
        for c in df.columns:
            if c == "variant" or c in numeric_cols:
                continue
            color_options.append({"label": prettify_column_label(c), "value": c})
    default_color = "variant" if "variant" in df.columns else (numeric_cols[0] if numeric_cols else None)

    numeric_cols = sorted(numeric_cols, key=torsion_sort_key)
    num_opts = [{"label": prettify_column_label(c), "value": c} for c in numeric_cols]
    default_x: Optional[str] = numeric_cols[0] if numeric_cols else None
    default_y: Optional[str] = numeric_cols[1] if len(numeric_cols) > 1 else default_x
    default_z: Optional[str] = numeric_cols[2] if len(numeric_cols) > 2 else default_x

    # --- Variant controls -------------------------------------------------
    variant_controls = html.Div(
        [
            html.Div(
                "Variants",
                style={"fontWeight": 600, "marginBottom": "4px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Input(
                                id="feat-variant-search",
                                type="text",
                                debounce=True,
                                placeholder="Filter list…",
                                style={"width": "100%"},
                            )
                        ],
                        style={"width": "22%", "paddingRight": "6px"},
                    ),
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="feat-variant-select",
                                options=[{"label": v, "value": v} for v in all_variants],
                                value=[],
                                multi=True,
                                placeholder="All variants (select to restrict)",
                            )
                        ],
                        style={"flex": "1", "minWidth": "0"},
                    ),
                    html.Button(
                        "Select visible",
                        id="feat-variant-select-visible-btn",
                        n_clicks=0,
                        style={
                            "fontSize": "0.8em",
                            "padding": "4px 8px",
                            "whiteSpace": "nowrap",
                            "cursor": "pointer",
                        },
                    ),
                    html.Button(
                        "Clear",
                        id="feat-variant-clear-btn",
                        n_clicks=0,
                        style={
                            "fontSize": "0.8em",
                            "padding": "4px 8px",
                            "cursor": "pointer",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "6px",
                },
            ),
        ],
        id="feat-variant-controls",
        style={
            "marginBottom": "10px",
            "display": "block",
        },
    )

    # --- Axes & color controls -------------------------------------------
    axes_controls = html.Div(
        [
            html.Div("Axes & color", style={"fontWeight": 600, "marginBottom": "4px"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "Feature family",
                                style={"fontSize": "0.85em"},
                            ),
                            dcc.Dropdown(
                                id="feat-family",
                                options=family_options,
                                value=default_family,
                                clearable=False,
                            ),
                        ],
                        style={"width": "25%", "paddingRight": "6px"},
                    ),
                    html.Div(
                        [
                            html.Div("X", style={"fontSize": "0.85em"}),
                            dcc.Dropdown(
                                id="feature-x",
                                options=num_opts,
                                value=default_x,
                                clearable=False,
                            ),
                        ],
                        style={"width": "18%", "paddingRight": "6px"},
                    ),
                    html.Div(
                        [
                            html.Div("Y", style={"fontSize": "0.85em"}),
                            dcc.Dropdown(
                                id="feature-y",
                                options=num_opts,
                                value=default_y,
                                clearable=False,
                            ),
                        ],
                        style={"width": "18%", "paddingRight": "6px"},
                    ),
                    html.Div(
                        [
                            html.Div("Color by", style={"fontSize": "0.85em"}),
                            dcc.Dropdown(
                                id="feat-color-by",
                                options=color_options,
                                value=default_color,
                                clearable=True,
                            ),
                        ],
                        style={"width": "19%", "paddingRight": "6px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Dimensionality",
                                style={"fontSize": "0.85em", "marginBottom": "2px"},
                            ),
                            dcc.RadioItems(
                                id="feat-dim-mode",
                                options=[
                                    {"label": "2D", "value": "2d"},
                                    {"label": "3D", "value": "3d"},
                                ],
                                value="2d",
                                inline=True,
                            ),
                            html.Div(
                                [
                                    html.Div("Z", style={"fontSize": "0.85em"}),
                                    dcc.Dropdown(
                                        id="feat-z-axis",
                                        options=num_opts,
                                        value=default_z,
                                        clearable=False,
                                    ),
                                ],
                                id="feat-z-axis-wrap",
                                style={"marginTop": "4px"},
                            ),
                        ],
                        style={"width": "20%"},
                    ),
                ],
                style={"display": "flex", "gap": "6px", "flexWrap": "wrap"},
            ),
        ],
        style={"marginBottom": "10px"},
    )

    # --- Arrow controls ---------------------------------------------------
    arrow_controls = html.Div(
        [
            html.Div(
                "Gradient arrows (2D / 3D)",
                style={"fontWeight": 600, "marginBottom": "4px"},
            ),
            html.Div(
                [
                    html.Div(
                        [
                            dcc.Dropdown(
                                id="feat-arrow-features",
                                options=num_opts,
                                multi=True,
                                placeholder="Features for ∇f arrows (optional)",
                            )
                        ],
                        style={"width": "40%", "paddingRight": "8px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Arrow density",
                                style={"fontSize": "0.85em", "marginBottom": "2px"},
                            ),
                            dcc.Slider(
                                id="feat-arrow-density",
                                min=5,
                                max=40,
                                step=1,
                                value=16,
                                tooltip={"placement": "bottom"},
                            ),
                        ],
                        style={"width": "20%", "paddingRight": "8px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Bandwidth",
                                style={"fontSize": "0.85em", "marginBottom": "2px"},
                            ),
                            dcc.Slider(
                                id="feat-arrow-bandwidth",
                                min=0.1,
                                max=1.5,
                                step=0.05,
                                value=0.4,
                                tooltip={"placement": "bottom"},
                            ),
                        ],
                        style={"width": "20%", "paddingRight": "8px"},
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Arrow scale",
                                style={"fontSize": "0.85em", "marginBottom": "2px"},
                            ),
                            dcc.Slider(
                                id="feat-arrow-scale",
                                min=0.2,
                                max=2.5,
                                step=0.1,
                                value=0.8,
                                tooltip={"placement": "bottom"},
                            ),
                        ],
                        style={"width": "20%"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "6px",
                    "alignItems": "center",
                },
            ),
            html.Div(
                [
                    dcc.Checklist(
                        id="feat-arrow-normalize",
                        options=[{"label": "Normalize vectors", "value": "norm"}],
                        value=["norm"],
                        style={"marginTop": "4px", "fontSize": "0.85em"},
                    )
                ]
            ),
        ],
        style={"marginBottom": "6px"},
    )

    # --- Analytics controls (density / regression / error circles) -------
    analytics_controls = html.Div(
        [
            html.Div(
                "Analytics (2D)",
                style={"fontWeight": 600, "marginBottom": "4px"},
            ),
            dcc.Checklist(
                id="feat-analytics",
                options=[
                    {"label": "2D density contours", "value": "density2d"},
                    {"label": "Regression line (Y vs X)", "value": "reg2d"},
                    {"label": "Replica spread 1σ (mean→std, median→½IQR)", "value": "error2d"},
                ],
                value=[],
                style={"fontSize": "0.85em"},
            ),
        ],
        style={"marginBottom": "6px"},
    )

    controls_box = html.Div(
        [variant_controls, axes_controls, arrow_controls, analytics_controls],
        style={
            "padding": "10px",
            "border": "1px solid #ccc",
            "borderRadius": "8px",
            "marginBottom": "10px",
            "backgroundColor": "rgba(255,255,255,0.03)",
        },
    )

    graph = dcc.Graph(
        id="features-graph",
        style={"height": "70vh"},
        config={
            "displaylogo": False,
            "modeBarButtonsToAdd": ["lasso2d", "select2d"],
        },
    )

    table = html.Div(
        [
            html.H4("Selected variants", style={"marginBottom": "4px"}),
            html.Div(
                id="feat-selection-summary",
                style={
                    "fontSize": "0.85em",
                    "marginBottom": "6px",
                },
            ),
            dash_table.DataTable(
                id="feat-selected-table",
                columns=[],
                data=[],
                page_size=15,
                style_table={"maxHeight": "70vh", "overflowY": "auto"},
                style_cell={
                    "fontSize": 12,
                    "padding": "3px",
                    "whiteSpace": "normal",
                    "height": "auto",
                },
                sort_action="native",
            ),
        ],
        style={
            "flex": "0 0 32%",
            "borderLeft": "1px solid #ccc",
            "paddingLeft": "10px",
        },
    )

    main_row = html.Div(
        [
            html.Div(
                graph,
                style={"flex": "0 0 68%", "paddingRight": "10px"},
            ),
            table,
        ],
        style={"display": "flex"},
    )

    return html.Div(
        [controls_box, main_row],
        style={"padding": "8px"},
    )


# ---------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------


def register_callbacks(app, ctx):
    from .shared import apply_theme
    df_initial: pd.DataFrame = getattr(ctx, "df", pd.DataFrame())

    # Same logic as layout: prefer ctx.numeric_cols (already globally filtered)
    numeric_cols_initial: List[str] = list(getattr(ctx, "numeric_cols", []) or [])
    if (not numeric_cols_initial) and isinstance(df_initial, pd.DataFrame) and not df_initial.empty:
        raw = [c for c in df_initial.columns if c != "variant" and pd.api.types.is_numeric_dtype(df_initial[c])]
        numeric_cols_initial = filter_numeric_columns(raw)

    def _current_variants() -> List[str]:
        local_df = getattr(ctx, "df", df_initial)
        if isinstance(local_df, pd.DataFrame) and "variant" in local_df.columns:
            return (
                local_df["variant"]
                .dropna()
                .astype(str)
                .sort_values()
                .unique()
                .tolist()
            )
        return []

    @app.callback(
        Output("feat-z-axis-wrap", "style"),
        Input("feat-dim-mode", "value"),
        prevent_initial_call=False,
    )
    def _toggle_z_visibility(dim_mode):
        if dim_mode == "3d":
            return {"marginTop": "4px"}
        return {"display": "none"}

    @app.callback(
        Output("feat-variant-select", "options"),
        Input("feat-variant-search", "value"),
        prevent_initial_call=False,
    )
    def _update_variant_options(search_text):
        all_variants = _current_variants()
        if not all_variants:
            return []
        if not search_text:
            return [{"label": v, "value": v} for v in all_variants]
        s = str(search_text).lower()
        filtered = [v for v in all_variants if s in v.lower()]
        return [{"label": v, "value": v} for v in filtered]

    @app.callback(
        Output("feat-variant-select", "value"),
        Input("feat-variant-select-visible-btn", "n_clicks"),
        Input("feat-variant-clear-btn", "n_clicks"),
        State("feat-variant-select", "options"),
        State("feat-variant-select", "value"),
        prevent_initial_call=True,
    )
    def _handle_variant_buttons(_sel_clicks, _clr_clicks, current_options, current_value):
        from dash import ctx as dash_ctx
        triggered = dash_ctx.triggered_id
        if triggered == "feat-variant-clear-btn":
            return []
        if triggered == "feat-variant-select-visible-btn":
            visible = [o["value"] for o in (current_options or [])]
            already = set(current_value or [])
            merged = list(already) + [v for v in visible if v not in already]
            return merged
        return current_value or []

    # Axis / feature options driven by family ------------------------------
    @app.callback(
        Output("feature-x", "options"),
        Output("feature-x", "value"),
        Output("feature-y", "options"),
        Output("feature-y", "value"),
        Output("feat-z-axis", "options"),
        Output("feat-z-axis", "value"),
        Output("feat-arrow-features", "options"),
        Input("feat-family", "value"),
        State("feature-x", "value"),
        State("feature-y", "value"),
        State("feat-z-axis", "value"),
        prevent_initial_call=False,
    )
    def _update_axis_options(family_value, cur_x, cur_y, cur_z):
        local_df = getattr(ctx, "df", df_initial)

        numeric_cols: List[str] = []
        if isinstance(local_df, pd.DataFrame) and not local_df.empty:
            raw = [
                c for c in local_df.columns
                if c != "variant" and pd.api.types.is_numeric_dtype(local_df[c])
            ]
            numeric_cols = filter_numeric_columns(raw)

        if not numeric_cols:
            return [], None, [], None, [], None, []

        fam_map = _compute_family_map(numeric_cols)
        if family_value and family_value != "__ALL__":
            cols = fam_map.get(family_value, [])
            if not cols:
                cols = numeric_cols
        else:
            cols = numeric_cols

        # keep order but unique
        seen: set = set()
        ordered_cols: List[str] = []
        for c in cols:
            if c not in seen:
                seen.add(c)
                ordered_cols.append(c)
        cols = ordered_cols

        opts = [{"label": prettify_column_label(c), "value": c} for c in cols]

        col_set = set(cols)

        def pick(idx: int, current: Optional[str]) -> Optional[str]:
            if current in col_set:
                return current
            if not cols:
                return None
            return cols[idx] if idx < len(cols) else cols[0]

        x_val = pick(0, cur_x)
        y_val = pick(1, cur_y)
        z_val = pick(2, cur_z) or x_val

        arrow_opts = [{"label": prettify_column_label(c), "value": c} for c in cols]

        return opts, x_val, opts, y_val, opts, z_val, arrow_opts

    # Main figure ---------------------------------------------------------
    @app.callback(
        Output("features-graph", "figure"),
        Input("feature-x", "value"),
        Input("feature-y", "value"),
        Input("feat-dim-mode", "value"),
        Input("feat-z-axis", "value"),
        Input("feat-color-by", "value"),
        Input("feat-arrow-features", "value"),
        Input("feat-arrow-density", "value"),
        Input("feat-arrow-bandwidth", "value"),
        Input("feat-arrow-scale", "value"),
        Input("feat-arrow-normalize", "value"),
        Input("feat-analytics", "value"),
        Input("feat-variant-select", "value"),
        Input("theme-store", "data"),
        prevent_initial_call=False,
    )
    def _update_features_graph(
        xcol,
        ycol,
        dim_mode,
        zcol,
        color_by,
        arrow_feats,
        density,
        bandwidth,
        scale,
        normalize_flags,
        analytics_flags,
        selected_variants,
        theme,
    ):
        local_df = getattr(ctx, "df", df_initial)
        if not isinstance(local_df, pd.DataFrame) or local_df.empty:
            return apply_theme(go.Figure(), theme)

        # Variant filtering
        if "variant" in local_df.columns and selected_variants:
            sel = set(str(v) for v in selected_variants)
            local_df = local_df[local_df["variant"].astype(str).isin(sel)]
            if local_df.empty:
                return apply_theme(go.Figure(), theme)

        if xcol is None or ycol is None:
            return apply_theme(go.Figure(), theme)
        if xcol not in local_df.columns or ycol not in local_df.columns:
            return apply_theme(go.Figure(), theme)

        labels = (
            local_df["variant"].astype(str)
            if "variant" in local_df.columns
            else None
        )

        show_density = "density2d" in (analytics_flags or [])
        show_reg = "reg2d" in (analytics_flags or [])
        show_error = "error2d" in (analytics_flags or [])

        # 3D branch ---------------------------------------------------------
        if dim_mode == "3d":
            if not zcol or zcol not in local_df.columns:
                return apply_theme(go.Figure(), theme)
            base3 = _prepare_numeric(local_df, [xcol, ycol, zcol])
            if base3.empty:
                return apply_theme(go.Figure(), theme)

            # variant names aligned with base3
            if "variant" in local_df.columns:
                var_arr = local_df["variant"].loc[base3.index].astype(str).to_numpy()
            else:
                var_arr = base3.index.astype(str).to_numpy()

            text = labels.loc[base3.index].to_numpy() if labels is not None else var_arr

            color_values, is_numeric_color, _ = _build_color(
                local_df, base3.index, color_by
            )
            marker = dict(size=4, opacity=0.8)
            if color_values is not None:
                marker["color"] = color_values
                if is_numeric_color:
                    marker["colorscale"] = "Viridis"
                    marker["showscale"] = True
                    marker["colorbar"] = dict(title=color_by)
                else:
                    marker["colorscale"] = "Turbo"
                    marker["showscale"] = False

            parts = ["%{text}"]
            parts.append(f"{xcol}: %{{x}}")
            parts.append(f"{ycol}: %{{y}}")
            parts.append(f"{zcol}: %{{z}}")
            if color_by:
                parts.append(f"{color_by}: %{{marker.color}}")
            hovertemplate = "<br>".join(parts) + "<extra></extra>"

            customdata = var_arr  # selection by variant name

            fig = _make_scatter_3d(
                base3,
                xcol,
                ycol,
                zcol,
                marker=marker,
                text=text,
                hovertemplate=hovertemplate,
                customdata=customdata,
                xlabel=prettify_column_label(xcol),
                ylabel=prettify_column_label(ycol),
                zlabel=prettify_column_label(zcol),
            )

            # 3D arrows: project 2D gradients into plane z = median(z)
            if arrow_feats:
                x_arr = base3[xcol].to_numpy(dtype=float)
                y_arr = base3[ycol].to_numpy(dtype=float)
                z_arr = base3[zcol].to_numpy(dtype=float)
                if x_arr.size > 0 and y_arr.size > 0:
                    xmn, xmx = _nanpercentile(x_arr, 1, 99)
                    ymn, ymx = _nanpercentile(y_arr, 1, 99)
                    dens = int(density or 16)
                    dens = max(5, min(40, dens))
                    gx = np.linspace(xmn, xmx, dens)
                    gy = np.linspace(ymn, ymx, dens)
                    z0 = float(np.nanmedian(z_arr)) if np.isfinite(z_arr).any() else 0.0
                    normalize = "norm" in (normalize_flags or [])
                    colors = _distinct_colors(len(arrow_feats))

                    for k, fcol in enumerate(arrow_feats or []):
                        if fcol not in local_df.columns:
                            continue
                        f_series = pd.to_numeric(
                            local_df[fcol], errors="coerce"
                        ).replace([np.inf, -np.inf], np.nan)
                        f = f_series.loc[base3.index].to_numpy(dtype=float)
                        mask_f = np.isfinite(f)
                        if not np.any(mask_f):
                            continue

                        x_f = x_arr[mask_f]
                        y_f = y_arr[mask_f]
                        f_f = f[mask_f]

                        gx2, gy2, U, V = _local_gradients_2d(
                            x_f, y_f, f_f, gx, gy, float(bandwidth or 0.4)
                        )
                        q3d = _quiver_trace_3d_from_2d(
                            gx2,
                            gy2,
                            U,
                            V,
                            z0=z0,
                            color=colors[k],
                            scale=float(scale or 0.8),
                            normalize=normalize,
                            name=f"∇{fcol}",
                        )
                        if q3d is not None:
                            fig.add_trace(q3d)

            apply_theme(fig, theme)
            return fig

        # 2D branch ---------------------------------------------------------
        base = _prepare_numeric(local_df, [xcol, ycol])
        if base.empty:
            return apply_theme(go.Figure(), theme)

        # variant names aligned with base
        if "variant" in local_df.columns:
            var_arr = local_df["variant"].loc[base.index].astype(str).to_numpy()
        else:
            var_arr = base.index.astype(str).to_numpy()

        text = labels.loc[base.index].to_numpy() if labels is not None else var_arr

        color_values, is_numeric_color, cat_labels = _build_color(
            local_df, base.index, color_by
        )

        customdata = var_arr  # selection per variant
        normalize = "norm" in (normalize_flags or [])

        # Decide how to color:
        # - Numeric color or no color: single scatter
        # - Categorical with low cardinality: multiple color layers (separate traces)
        use_multi_layer = False
        series_cat = None
        unique_cats = None
        if color_by and not is_numeric_color and cat_labels is not None:
            series_cat = local_df[color_by].loc[base.index].astype(str)
            unique_cats = series_cat.unique()
            if len(unique_cats) <= 30:
                use_multi_layer = True

        if not use_multi_layer:
            # Single-layer scatter
            marker = dict(size=6, opacity=0.8)
            if color_values is not None:
                marker["color"] = color_values
                if is_numeric_color:
                    marker["colorscale"] = "Viridis"
                    marker["showscale"] = True
                    marker["colorbar"] = dict(title=color_by)
                else:
                    marker["colorscale"] = "Turbo"
                    marker["showscale"] = False

            parts = ["%{text}"]
            parts.append(f"{xcol}: %{{x}}")
            parts.append(f"{ycol}: %{{y}}")
            if color_by:
                parts.append(f"{color_by}: %{{marker.color}}")
            hovertemplate = "<br>".join(parts) + "<extra></extra>"

            fig = _make_scatter_with_histograms(
                base,
                xcol,
                ycol,
                marker=marker,
                text=text,
                hovertemplate=hovertemplate,
                customdata=customdata,
                show_density=show_density,
                include_scatter=True,
                xlabel=prettify_column_label(xcol),
                ylabel=prettify_column_label(ycol),
            )
        else:
            # Multi-layer categorical color (separate traces per category)
            fig = _make_scatter_with_histograms(
                base,
                xcol,
                ycol,
                marker=None,
                text=None,
                hovertemplate="",
                customdata=None,
                show_density=show_density,
                include_scatter=False,
                xlabel=prettify_column_label(xcol),
                ylabel=prettify_column_label(ycol),
            )
            colors = _distinct_colors(len(unique_cats))
            for k, cat in enumerate(unique_cats):
                mask_cat = series_cat == cat
                sub = base[mask_cat]
                if sub.empty:
                    continue
                x_sub = sub[xcol]
                y_sub = sub[ycol]
                idx = sub.index
                var_sub = (
                    local_df["variant"].loc[idx].astype(str).to_numpy()
                    if "variant" in local_df.columns
                    else idx.astype(str)
                )
                text_sub = labels.loc[idx].to_numpy() if labels is not None else var_sub

                parts = ["%{text}"]
                parts.append(f"{xcol}: %{{x}}")
                parts.append(f"{ycol}: %{{y}}")
                parts.append(f"{color_by}: {cat}")
                hovertemplate = "<br>".join(parts) + "<extra></extra>"

                fig.add_trace(
                    go.Scatter(
                        x=x_sub,
                        y=y_sub,
                        mode="markers",
                        marker=dict(size=6, opacity=0.8, color=colors[k]),
                        name=str(cat),
                        text=text_sub,
                        hovertemplate=hovertemplate,
                        customdata=var_sub,
                    ),
                    row=2,
                    col=1,
                )

        # Regression line (2D) + coefficients
        if show_reg:
            x_arr = base[xcol].to_numpy(dtype=float)
            y_arr = base[ycol].to_numpy(dtype=float)
            valid = np.isfinite(x_arr) & np.isfinite(y_arr)
            x_valid = x_arr[valid]
            y_valid = y_arr[valid]
            if x_valid.size > 1:
                xmn, xmx = _nanpercentile(x_valid, 1, 99)
                try:
                    coeffs = np.polyfit(x_valid, y_valid, 1)
                    m, b = float(coeffs[0]), float(coeffs[1])
                    x_line = np.linspace(xmn, xmx, 100)
                    y_line = m * x_line + b

                    # R^2 computed on valid pairs only
                    y_pred = m * x_valid + b
                    ss_res = float(np.sum((y_valid - y_pred) ** 2))
                    ss_tot = float(np.sum((y_valid - np.mean(y_valid)) ** 2))
                    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

                    n_used = int(x_valid.size)
                    n_total = int(x_arr.size)

                    fig.add_trace(
                        go.Scatter(
                            x=x_line,
                            y=y_line,
                            mode="lines",
                            line=dict(color="black", width=2),
                            name=f"reg: y={m:.3g}x+{b:.3g}",
                            hoverinfo="skip",
                        ),
                        row=2,
                        col=1,
                    )

                    # Annotation with coefficients and R^2
                    txt = f"y = {m:.3g}·x + {b:.3g}"
                    if np.isfinite(r2):
                        txt += f"   (R² = {r2:.3f})"
                    if n_used < n_total:
                        txt += f"   [n={n_used}/{n_total}]"
                    fig.add_annotation(
                        xref="paper",
                        yref="paper",
                        x=0.0,
                        y=1.08,
                        text=txt,
                        showarrow=False,
                        font=dict(size=11),
                    )
                except Exception:
                    pass

        # Per-variant spread ellipses (1σ from pre-computed std/IQR columns)
        #
        # ctx.df has ONE ROW PER VARIANT (BATCH_ANA pivots pooled replica stats).
        # The spread lives in sibling columns: phi__mean → phi__std,
        # phi__median → phi__iqr (×0.5), etc.  groupby would always give
        # groups of size 1, so we read the spread columns directly.
        if show_error:
            all_cols_set = set(local_df.columns)
            xcol_spread, sx_scale = _find_spread_col(all_cols_set, xcol)
            ycol_spread, sy_scale = _find_spread_col(all_cols_set, ycol)

            if xcol_spread is None and ycol_spread is None:
                fig.add_annotation(
                    xref="paper",
                    yref="paper",
                    x=0.01,
                    y=0.97,
                    xanchor="left",
                    yanchor="top",
                    text=(
                        "No spread column found for the selected features.<br>"
                        "Select a <i>mean</i> or <i>median</i> feature to see 1σ ellipses."
                    ),
                    showarrow=False,
                    font=dict(size=10, color="gray"),
                    bgcolor="rgba(255,255,255,0.7)",
                )
            else:
                cx_arr = base[xcol].to_numpy(dtype=float)
                cy_arr = base[ycol].to_numpy(dtype=float)

                def _spread_arr(col: Optional[str], scale: float) -> np.ndarray:
                    if col is None or col not in local_df.columns:
                        return np.zeros(len(base))
                    raw = local_df[col].loc[base.index].to_numpy(dtype=float) * scale
                    return np.where(np.isfinite(raw) & (raw > 0), raw, 0.0)

                sx_arr = _spread_arr(xcol_spread, sx_scale)
                sy_arr = _spread_arr(ycol_spread, sy_scale)
                var_names = local_df["variant"].loc[base.index].astype(str).to_numpy() \
                    if "variant" in local_df.columns else base.index.astype(str).to_numpy()

                valid_mask = (
                    np.isfinite(cx_arr) & np.isfinite(cy_arr)
                    & ((sx_arr > 0) | (sy_arr > 0))
                )

                ell_xs: List[float] = []
                ell_ys: List[float] = []
                mean_xs: List[float] = []
                mean_ys: List[float] = []
                mean_texts: List[str] = []
                theta = np.linspace(0, 2 * np.pi, 80)

                for i in np.where(valid_mask)[0]:
                    cx, cy = float(cx_arr[i]), float(cy_arr[i])
                    sx, sy = float(sx_arr[i]), float(sy_arr[i])
                    vname = var_names[i]

                    if sx > 0 and sy > 0:
                        ell_xs.extend((cx + sx * np.cos(theta)).tolist() + [np.nan])
                        ell_ys.extend((cy + sy * np.sin(theta)).tolist() + [np.nan])
                    elif sx > 0:
                        ell_xs.extend([cx - sx, cx + sx, np.nan])
                        ell_ys.extend([cy, cy, np.nan])
                    else:
                        ell_xs.extend([cx, cx, np.nan])
                        ell_ys.extend([cy - sy, cy + sy, np.nan])

                    mean_xs.append(cx)
                    mean_ys.append(cy)
                    parts = [vname]
                    if sx > 0:
                        parts.append(f"σ_x = {sx:.3g}  ({xcol_spread})")
                    if sy > 0:
                        parts.append(f"σ_y = {sy:.3g}  ({ycol_spread})")
                    mean_texts.append("<br>".join(parts))

                if ell_xs:
                    fig.add_trace(
                        go.Scatter(
                            x=np.array(ell_xs),
                            y=np.array(ell_ys),
                            mode="lines",
                            line=dict(color="rgba(0,0,0,0.30)", width=1),
                            name="1σ ellipses",
                            hoverinfo="skip",
                            showlegend=True,
                        ),
                        row=2,
                        col=1,
                    )
                if mean_xs:
                    fig.add_trace(
                        go.Scatter(
                            x=mean_xs,
                            y=mean_ys,
                            mode="markers",
                            marker=dict(
                                symbol="cross",
                                size=8,
                                color="rgba(0,0,0,0.55)",
                                line=dict(width=1, color="rgba(0,0,0,0.55)"),
                            ),
                            text=mean_texts,
                            hovertemplate="%{text}<extra></extra>",
                            name="variant centers",
                            showlegend=True,
                        ),
                        row=2,
                        col=1,
                    )

        # Arrows (2D)
        if arrow_feats:
            x_arr = base[xcol].to_numpy(dtype=float)
            y_arr = base[ycol].to_numpy(dtype=float)
            if x_arr.size > 0 and y_arr.size > 0:
                xmn, xmx = _nanpercentile(x_arr, 1, 99)
                ymn, ymx = _nanpercentile(y_arr, 1, 99)

                dens = int(density or 16)
                dens = max(5, min(40, dens))
                gx = np.linspace(xmn, xmx, dens)
                gy = np.linspace(ymn, ymx, dens)

                normalize = "norm" in (normalize_flags or [])
                colors = _distinct_colors(len(arrow_feats))

                for k, fcol in enumerate(arrow_feats or []):
                    if fcol not in local_df.columns:
                        continue
                    f_series = pd.to_numeric(
                        local_df[fcol], errors="coerce"
                    ).replace([np.inf, -np.inf], np.nan)
                    f = f_series.loc[base.index].to_numpy(dtype=float)
                    mask_f = np.isfinite(f)
                    if not np.any(mask_f):
                        continue

                    x_f = x_arr[mask_f]
                    y_f = y_arr[mask_f]
                    f_f = f[mask_f]

                    gx2, gy2, U, V = _local_gradients_2d(
                        x_f, y_f, f_f, gx, gy, float(bandwidth or 0.4)
                    )
                    q = _quiver_trace_2d(
                        gx2,
                        gy2,
                        U,
                        V,
                        color=colors[k],
                        scale=float(scale or 0.8),
                        normalize=normalize,
                        name=f"∇{fcol}",
                    )
                    if q is not None:
                        fig.add_trace(q, row=2, col=1)

        apply_theme(fig, theme)
        return fig

    # Selection summary + table -------------------------------------------
    @app.callback(
        Output("feat-selection-summary", "children"),
        Output("feat-selected-table", "columns"),
        Output("feat-selected-table", "data"),
        Input("features-graph", "selectedData"),
        Input("features-graph", "clickData"),
        Input("feature-x", "value"),
        Input("feature-y", "value"),
        Input("feat-dim-mode", "value"),
        Input("feat-z-axis", "value"),
        Input("feat-color-by", "value"),
        Input("feat-arrow-features", "value"),
        Input("feat-variant-select", "value"),
        prevent_initial_call=False,
    )
    def _update_selected_table(
        selected_data,
        click_data,
        xcol,
        ycol,
        dim_mode,
        zcol,
        color_by,
        arrow_feats,
        selected_variants,
    ):
        local_df = getattr(ctx, "df", df_initial)
        if not isinstance(local_df, pd.DataFrame) or local_df.empty:
            return "No data", [], []

        # Same variant filter as plot
        if "variant" in local_df.columns and selected_variants:
            sel = set(str(v) for v in selected_variants)
            local_df = local_df[local_df["variant"].astype(str).isin(sel)]
            if local_df.empty:
                return "No data", [], []

        if xcol is None or ycol is None:
            return "No axes selected", [], []

        # Numeric base used for plotting
        if dim_mode == "3d":
            if not zcol or zcol not in local_df.columns:
                return "No data", [], []
            base = _prepare_numeric(local_df, [xcol, ycol, zcol])
        else:
            base = _prepare_numeric(local_df, [xcol, ycol])

        if base.empty:
            return "No data", [], []

        # variant names for base
        if "variant" in local_df.columns:
            var_series = local_df["variant"].loc[base.index].astype(str)
        else:
            var_series = base.index.astype(str)

        # Collect variants from selection + click
        var_set = set()

        def _extract_var(cd):
            if cd is None:
                return None
            if isinstance(cd, (list, tuple)) and cd:
                return str(cd[0])
            return str(cd)

        if selected_data and "points" in selected_data:
            for pt in selected_data["points"]:
                cd = pt.get("customdata", None)
                v = _extract_var(cd)
                if v is not None:
                    var_set.add(v)

        if click_data and "points" in click_data:
            for pt in click_data["points"]:
                cd = pt.get("customdata", None)
                v = _extract_var(cd)
                if v is not None:
                    var_set.add(v)

        # If nothing explicitly selected → all variants currently plotted
        if not var_set:
            var_set = set(var_series.unique().tolist())

        # Build selection dataframe
        if "variant" in local_df.columns:
            local_df = local_df.copy()
            local_df["__variant_str__"] = local_df["variant"].astype(str)
            sel_df = local_df[local_df["__variant_str__"].isin(var_set)]
        else:
            sel_df = local_df.loc[base.index]

        if sel_df.empty:
            return "No data", [], []

        # ---- Summary card -------------------------------------------------
        if "__variant_str__" in sel_df.columns:
            n_variants = sel_df["__variant_str__"].nunique()
        elif "variant" in sel_df.columns:
            n_variants = sel_df["variant"].astype(str).nunique()
        else:
            n_variants = sel_df.index.nunique()

        n_reps = int(len(sel_df))

        def _mean_std_str(col_name: Optional[str]) -> Optional[str]:
            if not col_name or col_name not in sel_df.columns:
                return None
            s = sel_df[col_name]
            if not pd.api.types.is_numeric_dtype(s):
                # try to coerce
                s = pd.to_numeric(s, errors="coerce")
            s = s.replace([np.inf, -np.inf], np.nan).dropna()
            if s.empty:
                return None
            m = float(s.mean())
            sd = float(s.std(ddof=1)) if len(s) > 1 else 0.0
            return f"{m:.3f} ± {sd:.3f}"

        x_summary = _mean_std_str(xcol)
        y_summary = _mean_std_str(ycol)
        color_summary = _mean_std_str(color_by)

        summary_children = [
            html.Div(
                [
                    html.Span("Variants: ", style={"fontWeight": 600}),
                    html.Span(str(n_variants)),
                    html.Span("   ·   "),
                    html.Span("Replicas: ", style={"fontWeight": 600}),
                    html.Span(str(n_reps)),
                ]
            )
        ]

        stat_rows = []
        if x_summary:
            stat_rows.append(
                html.Div(
                    [
                        html.Span(f"X ({prettify_column_label(xcol)}): ", style={"fontWeight": 600}),
                        html.Span(x_summary),
                    ]
                )
            )
        if y_summary:
            stat_rows.append(
                html.Div(
                    [
                        html.Span(f"Y ({prettify_column_label(ycol)}): ", style={"fontWeight": 600}),
                        html.Span(y_summary),
                    ]
                )
            )
        if color_by and color_summary:
            stat_rows.append(
                html.Div(
                    [
                        html.Span(f"Color ({prettify_column_label(color_by)}): ", style={"fontWeight": 600}),
                        html.Span(color_summary),
                    ]
                )
            )
        if stat_rows:
            summary_children.append(
                html.Div(stat_rows, style={"marginTop": "2px"})
            )

        summary_block = html.Div(summary_children)

        # ---- Table content ------------------------------------------------
        # Determine columns we show
        cols = []
        if "variant" in sel_df.columns:
            cols.append("variant")
        for c in [xcol, ycol]:
            if c and c not in cols:
                cols.append(c)
        if dim_mode == "3d" and zcol and zcol not in cols:
            cols.append(zcol)
        if color_by and color_by not in cols:
            cols.append(color_by)
        for c in arrow_feats or []:
            if c and c not in cols:
                cols.append(c)

        cols = [c for c in cols if c in sel_df.columns]
        if not cols:
            return summary_block, [], []

        # Aggregate per variant if variant is present
        if "variant" in sel_df.columns:
            rows = []
            grouped = sel_df.groupby("__variant_str__", sort=True)
            for vname, g in grouped:
                row = {"variant": vname}
                for c in cols:
                    if c == "variant":
                        continue
                    if c not in g.columns:
                        continue
                    s = g[c]
                    if pd.api.types.is_numeric_dtype(s):
                        val = float(pd.to_numeric(s, errors="coerce").replace(
                            [np.inf, -np.inf], np.nan
                        ).mean())
                        row[c] = val
                    else:
                        try:
                            m = (
                                s.dropna()
                                .astype(str)
                                .mode(dropna=True)
                            )
                            row[c] = str(m.iloc[0]) if not m.empty else ""
                        except Exception:
                            row[c] = str(s.iloc[0]) if len(s) else ""
                rows.append(row)
            out_df = pd.DataFrame(rows)
        else:
            out_df = sel_df[cols].copy()

        # Round numeric columns to 3 decimals
        for c in out_df.columns:
            if c == "variant":
                continue
            if pd.api.types.is_numeric_dtype(out_df[c]):
                out_df[c] = out_df[c].round(3)

        columns = [
            {"name": prettify_column_label(c) if c != "variant" else c, "id": c}
            for c in out_df.columns
        ]
        data = out_df.to_dict("records")
        return summary_block, columns, data
