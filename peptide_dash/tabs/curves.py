from __future__ import annotations

from typing import List, Tuple, Dict, Any, Optional

import math
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from dash import Input, Output, State, dcc, html, no_update

from ..theming.errors import error_fig
from ..metrics import all_torsion, metric_display_label, residue_display, torsion_sort_key

TAB_LABEL = "Curves"

# Physical constant (kJ mol^-1 K^-1)
R_GAS = 0.0083144621



_metric_sort_key = torsion_sort_key

# ------------------------ small helpers ------------------------------------


def _bin_width(vals: np.ndarray, fallback: float | None = None) -> float:
    vals = np.asarray(vals, float)
    u = np.unique(vals)
    if u.size < 2:
        return float(fallback if fallback is not None else 5.0)
    return float(np.min(np.diff(u)))


def density_to_pmf(
    Z: np.ndarray,
    xs: np.ndarray,
    ys: np.ndarray,
    T_K: float,
    clip_max: float,
) -> np.ndarray:
    """
    Convert a 2D density on (x,y) into a PMF F(x,y) in kJ/mol.
    F is shifted so min(F) = 0 and clipped at clip_max if provided.
    """
    Z = np.asarray(Z, float)
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)

    if Z.size == 0 or xs.size < 2 or ys.size < 2:
        return np.full_like(Z, np.nan, dtype=float)

    dx = _bin_width(xs, fallback=5.0)
    dy = _bin_width(ys, fallback=5.0)

    P = Z * dx * dy
    total = np.nansum(P)
    if not np.isfinite(total) or total <= 0:
        return np.full_like(Z, np.nan, dtype=float)

    P = P / total
    beta = 1.0 / (R_GAS * float(T_K))
    with np.errstate(divide="ignore"):
        F = -(1.0 / beta) * np.log(np.clip(P, 1e-300, 1.0))

    if np.isfinite(F).any():
        F = F - np.nanmin(F)

    if clip_max is not None and np.isfinite(clip_max):
        F = np.clip(F, 0.0, float(clip_max))

    return F


def _prep_xy(df: pd.DataFrame, xcol: str, ycols: List[str]) -> pd.DataFrame:
    cols = [xcol] + ycols
    cols = [c for c in cols if c in df.columns]
    if not cols or xcol not in cols:
        return pd.DataFrame()
    d = df[cols].dropna()
    if d.empty:
        return d
    return d.sort_values(xcol)


def _rama_region_pop(xs: np.ndarray, ys: np.ndarray, Z: np.ndarray) -> Dict[str, float]:
    """
    Given grid xs (phi), ys (psi) and density Z, compute region populations for
    rough α / β / left-α regions. Returns fractions that sum to ≈1.
    """
    xs = np.asarray(xs, float)
    ys = np.asarray(ys, float)
    Z = np.asarray(Z, float)

    if Z.size == 0 or xs.size < 2 or ys.size < 2:
        return {"alpha": np.nan, "beta": np.nan, "left_alpha": np.nan, "other": np.nan}

    dx = _bin_width(xs, fallback=5.0)
    dy = _bin_width(ys, fallback=5.0)

    P = Z * dx * dy
    total = np.nansum(P)
    if not np.isfinite(total) or total <= 0:
        return {"alpha": np.nan, "beta": np.nan, "left_alpha": np.nan, "other": np.nan}
    P = P / total

    phi_grid, psi_grid = np.meshgrid(xs, ys)

    # crude classical regions
    regions_def = {
        "alpha": ((-100.0, -30.0), (-80.0, -5.0)),
        "beta": ((-180.0, -40.0), (90.0, 180.0)),
        "left_alpha": ((30.0, 100.0), (0.0, 90.0)),
    }

    pops: Dict[str, float] = {}
    used = 0.0
    for name, ((plo, phi), (slo, shi)) in regions_def.items():
        mask = (
            (phi_grid >= plo)
            & (phi_grid < phi)
            & (psi_grid >= slo)
            & (psi_grid < shi)
        )
        mass = float(P[mask].sum())
        pops[name] = mass
        used += mass

    pops["other"] = max(0.0, 1.0 - used)
    return pops


def _add_error_band(fig, x, y, dy, row, col):
    fig.add_trace(
        go.Scatter(
            x=np.concatenate([x, x[::-1]]),
            y=np.concatenate([y - dy, (y + dy)[::-1]]),
            mode="lines",
            line=dict(width=0),
            fill="toself",
            name="±1σ",
            showlegend=False,
        ),
        row=row,
        col=col,
    )

def _round_x_key(x: np.ndarray, decimals: int = 12) -> np.ndarray:
    """Stable float keys for alignment across groupbys."""
    with np.errstate(all="ignore"):
        return np.round(np.asarray(x, dtype=float), decimals=decimals)


def _detect_replica_col(df: pd.DataFrame) -> Optional[str]:
    """Best-effort detection of a replica identifier column."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    cols = list(df.columns)
    lowered = {str(c).lower(): c for c in cols}
    # strict first
    for key in ("replica", "rep", "replicate", "rep_id", "replica_id", "replicate_id", "replica_idx", "replica_num", "replica_number"):
        if key in lowered:
            return str(lowered[key])
    # prefix / regex patterns
    for lc, orig in lowered.items():
        if lc.startswith("replica"):
            return str(orig)
        if re.match(r"^rep(lica)?[_-]?(id|idx|num|number)?$", lc):
            return str(orig)
    return None


def _replica_mean_std_by_x(
    d: pd.DataFrame,
    xcol: str,
    ycol: str,
    repcol: str = "replica",
) -> pd.DataFrame:
    """Mean/std across replicas at each x, with within-replica averaging."""
    if not isinstance(d, pd.DataFrame) or d.empty:
        return pd.DataFrame()
    if xcol not in d.columns or ycol not in d.columns or repcol not in d.columns:
        return pd.DataFrame()

    tmp = d[[repcol, xcol, ycol]].copy()
    tmp[xcol] = pd.to_numeric(tmp[xcol], errors="coerce")
    tmp[ycol] = pd.to_numeric(tmp[ycol], errors="coerce")
    tmp = tmp.dropna(subset=[repcol, xcol, ycol])
    if tmp.empty:
        return pd.DataFrame()

    tmp = (
        tmp.groupby([repcol, xcol], as_index=False, observed=False)[ycol]
        .mean(numeric_only=True)
        .rename(columns={ycol: "y"})
    )

    stats = (
        tmp.groupby(xcol, observed=False)["y"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean", "std": "std", "count": "nrep"})
    )
    return stats


def _replica_normed_curve_stats(
    d: pd.DataFrame,
    xcol: str,
    ycol: str,
    repcol: str = "replica",
) -> tuple[pd.DataFrame, float]:
    """Replica mean/std for a curve after per-replica area normalization.

    Returns (stats_df, mean_raw_area).
    """
    if not isinstance(d, pd.DataFrame) or d.empty:
        return pd.DataFrame(), float("nan")
    if xcol not in d.columns or ycol not in d.columns or repcol not in d.columns:
        return pd.DataFrame(), float("nan")

    parts: list[pd.DataFrame] = []
    raw_areas: list[float] = []

    for rep_id, g in d.groupby(repcol, sort=False, observed=False):
        g2 = _prep_xy(g, xcol, [ycol])
        if g2.empty:
            continue
        x = g2[xcol].to_numpy(dtype=float)
        y = g2[ycol].to_numpy(dtype=float)
        v = np.isfinite(x) & np.isfinite(y)
        if v.sum() < 2:
            continue
        x = x[v]
        y = y[v]
        o = np.argsort(x)
        x = x[o]
        y = y[o]

        area = float(np.trapz(np.clip(y, 0, np.inf), x))
        raw_areas.append(area if np.isfinite(area) else float("nan"))
        if area > 0:
            y = y / area

        parts.append(pd.DataFrame({repcol: rep_id, xcol: x, "y": y}))

    if not parts:
        return pd.DataFrame(), float(np.nanmean(raw_areas)) if raw_areas else float("nan")

    long = pd.concat(parts, ignore_index=True)
    long = (
        long.groupby([repcol, xcol], as_index=False, observed=False)["y"]
        .mean(numeric_only=True)
        .rename(columns={"y": "y"})
    )

    stats = (
        long.groupby(xcol, observed=False)["y"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "mean", "std": "std", "count": "nrep"})
    )
    return stats, float(np.nanmean(raw_areas)) if raw_areas else float("nan")


def _align_std_to_x(x: np.ndarray, stats: pd.DataFrame, xcol: str = "x") -> np.ndarray:
    """Align stats['std'] to x using rounded keys."""
    if not isinstance(stats, pd.DataFrame) or stats.empty:
        return np.full_like(np.asarray(x, dtype=float), np.nan, dtype=float)
    if xcol not in stats.columns or "std" not in stats.columns:
        return np.full_like(np.asarray(x, dtype=float), np.nan, dtype=float)

    xk = _round_x_key(np.asarray(x, dtype=float))
    sk = _round_x_key(stats[xcol].to_numpy(dtype=float))
    s = pd.Series(stats["std"].to_numpy(dtype=float), index=sk)
    return s.reindex(xk).to_numpy(dtype=float)



# ------------------------------ layout -------------------------------------


def layout(ctx):
    df: pd.DataFrame = getattr(ctx, "df", pd.DataFrame())
    # pmf_df / cum_df / rmsf_df are intentionally not accessed here (lazy loading).

    # metrics from PMF + cumulative
    # Metrics and x-range are intentionally NOT pre-loaded here.
    # Accessing ctx.pmf_df / ctx.cum_df at layout-build time would trigger their
    # lazy loads at startup. Instead we start with safe defaults and let the
    # `_populate_curve_controls` callback fill in real values on first render.
    metric_options: List[str] = []

    variants: List[str] = []
    if isinstance(df, pd.DataFrame) and not df.empty and "variant" in df.columns:
        variants = sorted(df["variant"].dropna().astype(str).unique().tolist())

    x_min, x_max = -180.0, 180.0   # sensible default for torsion angles

    if x_max <= x_min:
        x_min, x_max = 0.0, 1.0

    x_mid = 0.5 * (x_min + x_max)
    span = max(1e-9, x_max - x_min)
    step = span / 200.0

    # slider marks
    marks = {
        float(f"{x_min:.1f}"): f"{x_min:.1f}",
        float(f"{x_mid:.1f}"): f"{x_mid:.1f}",
        float(f"{x_max:.1f}"): f"{x_max:.1f}",
    }

    # nice default regions: middle and right chunk
    default_region1 = [x_min + 0.2 * span, x_min + 0.45 * span]
    default_region2 = [x_min + 0.55 * span, x_min + 0.8 * span]

    # --- controls card ---
    controls = html.Div(
        [
            html.Div(
                "Curves & Distributions",
                style={
                    "fontWeight": 700,
                    "marginBottom": "4px",
                    "fontSize": "1.08em",
                },
            ),
            html.Div(
                "Compare PMFs, cumulative FE, RMSF and Ramachandran statistics across variants.",
                style={"fontSize": "0.78em", "opacity": 0.75, "marginBottom": "6px"},
            ),
            html.Div(
                [
                    # column 1: curve type
                    html.Div(
                        [
                            html.Div(
                                "Curve type",
                                style={"fontSize": "0.83em", "marginBottom": "2px"},
                            ),
                            dcc.Dropdown(
                                id="curve-type",
                                clearable=False,
                                options=[
                                    {"label": "PMF: F(x)", "value": "pmf_F"},
                                    {"label": "PMF: P(x)", "value": "pmf_P"},
                                    {"label": "Cumulative FE (L/R)", "value": "cum_FE"},
                                    {
                                        "label": "ΔF (left/right)",
                                        "value": "cum_DeltaF",
                                    },
                                    {
                                        "label": "RMSF (per replica mean)",
                                        "value": "rmsf",
                                    },
                                    {
                                        "label": "Ramachandran 1D (R order)",
                                        "value": "rama",
                                    },
                                    {
                                        "label": "Ramachandran 2D (PMF)",
                                        "value": "rama2d",
                                    },
                                ],
                                value=("pmf_F" if metric_options else "rmsf"),
                                style={"fontSize": "0.9em"},
                            ),
                        ],
                        style={
                            "flex": "1 1 22%",
                            "minWidth": "210px",
                            "paddingRight": "10px",
                        },
                    ),
                    # column 2: metrics + panels
                    html.Div(
                        [
                            html.Div(
                                "Metric(s)",
                                style={"fontSize": "0.83em", "marginBottom": "2px"},
                            ),
                            dcc.Dropdown(
                                id="curve-metrics",
                                multi=True,
                                placeholder="Pick 1–3 metrics",
                                options=[{"label": metric_display_label(m), "value": m} for m in metric_options],
                                value=([metric_options[0]] if metric_options else []),
                                style={"fontSize": "0.9em"},
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Panels (max)",
                                        style={
                                            "fontSize": "0.78em",
                                            "marginTop": "6px",
                                            "marginBottom": "2px",
                                        },
                                    ),
                                    dcc.Slider(
                                        id="curve-maxpanels",
                                        min=1,
                                        max=6,
                                        step=1,
                                        value=3,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ],
                                style={"marginTop": "2px"},
                            ),
                        ],
                        style={
                            "flex": "1 1 38%",
                            "minWidth": "260px",
                            "paddingRight": "10px",
                        },
                    ),
                    # column 3: options + download
                    html.Div(
                        [
                            html.Div(
                                "Plot options",
                                style={"fontSize": "0.83em", "marginBottom": "2px"},
                            ),
                            dcc.Checklist(
                                id="curve-options",
                                value=["sharey"],
                                options=[
                                    {"label": "Share Y axis", "value": "sharey"},
                                    {"label": "Error band (±1σ, if available)", "value": "err"},
                                    {"label": "Show replicas (if available)", "value": "reps"},
                                    {
                                        "label": "Show dF/dx & d²F/dx² (PMF F)",
                                        "value": "deriv",
                                    },
                                ],
                                style={"fontSize": "0.8em"},
                                inline=False,
                            ),
                            html.Button(
                                "Download curves as CSV",
                                id="curves-download-btn",
                                n_clicks=0,
                                style={
                                    "marginTop": "8px",
                                    "fontSize": "0.8em",
                                    "padding": "4px 10px",
                                    "borderRadius": "4px",
                                },
                            ),
                            dcc.Download(id="curves-download"),
                        ],
                        style={"flex": "1 1 22%", "minWidth": "220px"},
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "14px",
                    "flexWrap": "wrap",
                    "alignItems": "flex-start",
                },
            ),
            html.Hr(style={"margin": "10px 0"}),
            # variants + reference
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "Peptides (variants)",
                                style={"fontSize": "0.83em", "marginBottom": "2px"},
                            ),
                            dcc.Dropdown(
                                id="curve-variants",
                                multi=True,
                                options=[{"label": v, "value": v} for v in variants],
                                value=variants,
                                style={"fontSize": "0.9em"},
                            ),
                        ],
                        style={
                            "flex": "1 1 60%",
                            "minWidth": "260px",
                            "paddingRight": "10px",
                        },
                    ),
                    html.Div(
                        [
                            html.Div(
                                "Reference variant (optional)",
                                style={"fontSize": "0.83em", "marginBottom": "2px"},
                            ),
                            dcc.Dropdown(
                                id="curve-ref-variant",
                                multi=False,
                                options=[{"label": v, "value": v} for v in variants],
                                value=None,
                                placeholder="For qualitative comparison",
                                style={"fontSize": "0.9em"},
                                clearable=True,
                            ),
                        ],
                        style={
                            "flex": "1 1 40%",
                            "minWidth": "220px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "14px",
                    "flexWrap": "wrap",
                    "alignItems": "center",
                    "marginBottom": "6px",
                },
            ),
            # draggable integration regions + numeric inputs
            html.Div(
                [
                    html.Div(
                        [
                            html.Div(
                                "Integration regions for PMF P(x) / PMF F(x)",
                                style={
                                    "fontSize": "0.83em",
                                    "marginBottom": "4px",
                                    "fontWeight": 600,
                                },
                            ),
                            html.Div(
                                "These two draggable ranges define State 1 and State 2. "
                                "For PMF P(x), state populations are computed. "
                                "For PMF F(x), state minima and ΔF are computed.",
                                style={
                                    "fontSize": "0.75em",
                                    "opacity": 0.8,
                                    "marginBottom": "4px",
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "State 1 region",
                                        style={
                                            "fontSize": "0.78em",
                                            "marginBottom": "2px",
                                        },
                                    ),
                                    dcc.RangeSlider(
                                        id="curve-region1",
                                        min=x_min,
                                        max=x_max,
                                        step=step,
                                        value=default_region1,
                                        marks=marks,
                                        tooltip={"placement": "bottom"},
                                        allowCross=False,
                                    ),
                                    html.Div(
                                        [
                                            html.Span(
                                                "From:",
                                                style={
                                                    "fontSize": "0.75em",
                                                    "marginRight": "4px",
                                                },
                                            ),
                                            dcc.Input(
                                                id="curve-region1-min-input",
                                                type="number",
                                                value=default_region1[0],
                                                style={
                                                    "width": "80px",
                                                    "fontSize": "0.8em",
                                                    "marginRight": "8px",
                                                },
                                            ),
                                            html.Span(
                                                "To:",
                                                style={
                                                    "fontSize": "0.75em",
                                                    "marginRight": "4px",
                                                },
                                            ),
                                            dcc.Input(
                                                id="curve-region1-max-input",
                                                type="number",
                                                value=default_region1[1],
                                                style={
                                                    "width": "80px",
                                                    "fontSize": "0.8em",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "marginTop": "4px",
                                        },
                                    ),
                                ],
                                style={"marginBottom": "10px"},
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "State 2 region",
                                        style={
                                            "fontSize": "0.78em",
                                            "marginBottom": "2px",
                                        },
                                    ),
                                    dcc.RangeSlider(
                                        id="curve-region2",
                                        min=x_min,
                                        max=x_max,
                                        step=step,
                                        value=default_region2,
                                        marks=marks,
                                        tooltip={"placement": "bottom"},
                                        allowCross=False,
                                    ),
                                    html.Div(
                                        [
                                            html.Span(
                                                "From:",
                                                style={
                                                    "fontSize": "0.75em",
                                                    "marginRight": "4px",
                                                },
                                            ),
                                            dcc.Input(
                                                id="curve-region2-min-input",
                                                type="number",
                                                value=default_region2[0],
                                                style={
                                                    "width": "80px",
                                                    "fontSize": "0.8em",
                                                    "marginRight": "8px",
                                                },
                                            ),
                                            html.Span(
                                                "To:",
                                                style={
                                                    "fontSize": "0.75em",
                                                    "marginRight": "4px",
                                                },
                                            ),
                                            dcc.Input(
                                                id="curve-region2-max-input",
                                                type="number",
                                                value=default_region2[1],
                                                style={
                                                    "width": "80px",
                                                    "fontSize": "0.8em",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "alignItems": "center",
                                            "marginTop": "4px",
                                        },
                                    ),
                                ]
                            ),
                            html.Div(
                                f"Initial slider range is based on overall PMF x-span: [{x_min:.2f}, {x_max:.2f}]. "
                                f"For PMF F/P, min/max are updated to match the current data subset.",
                                style={
                                    "fontSize": "0.72em",
                                    "opacity": 0.7,
                                    "marginTop": "4px",
                                },
                            ),
                        ],
                        style={
                            "flex": "1 1 100%",
                            "minWidth": "260px",
                        },
                    )
                ],
                style={
                    "display": "flex",
                    "flexWrap": "wrap",
                    "gap": "10px",
                    "marginBottom": "8px",
                },
            ),
            # rama2d controls (hidden unless Rama2D)
            html.Div(
                id="rama2d-controls-wrapper",
                children=[
                    html.Hr(style={"margin": "8px 0"}),
                    html.Div(
                        [
                            html.Div(
                                [
                                    html.Div(
                                        "Residues (per-residue PMF)",
                                        style={
                                            "fontSize": "0.83em",
                                            "marginBottom": "2px",
                                        },
                                    ),
                                    dcc.Dropdown(
                                        id="rama2d-residues",
                                        options=[],
                                        value=[],
                                        multi=True,
                                        style={"fontSize": "0.9em"},
                                    ),
                                ],
                                style={
                                    "flex": "1 1 50%",
                                    "minWidth": "180px",
                                    "paddingRight": "10px",
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Temperature (K)",
                                        style={
                                            "fontSize": "0.83em",
                                            "marginBottom": "2px",
                                        },
                                    ),
                                    dcc.Input(
                                        id="rama2d-T-curves",
                                        type="number",
                                        min=200,
                                        step=5,
                                        value=300,
                                        style={"width": "100%", "fontSize": "0.9em"},
                                    ),
                                ],
                                style={
                                    "flex": "0 0 16%",
                                    "minWidth": "120px",
                                    "paddingRight": "10px",
                                },
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "PMF max (kJ/mol)",
                                        style={
                                            "fontSize": "0.83em",
                                            "marginBottom": "2px",
                                        },
                                    ),
                                    dcc.Slider(
                                        id="rama2d-pmfmax-curves",
                                        min=5,
                                        max=40,
                                        step=1,
                                        value=20,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ],
                                style={"flex": "0 0 30%", "minWidth": "140px"},
                            ),
                        ],
                        style={
                            "display": "flex",
                            "gap": "12px",
                            "flexWrap": "wrap",
                            "alignItems": "center",
                        },
                    ),
                ],
                style={"display": "none"},
            ),
            html.Div(
                "Data sources: PMF/cumulative from *_pmf.* and *_cumulative.*; "
                "RMSF from *_replica; Ramachandran from phi/psi density tables.",
                style={"fontSize": "0.72em", "opacity": 0.7, "marginTop": "6px"},
            ),
        ],
        style={
            "padding": "10px",
            "border": "1px solid rgba(180,180,180,0.7)",
            "borderRadius": "10px",
            "marginBottom": "10px",
            "backgroundColor": "rgba(255,255,255,0.03)",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.08)",
        },
    )

    # Summary ABOVE the main graph
    summary_div = html.Div(
        id="curves-summary",
        style={
            "marginBottom": "8px",
            "fontSize": "0.78em",
            "border": "1px solid rgba(200,200,200,0.6)",
            "borderRadius": "6px",
            "padding": "6px 8px",
            "backgroundColor": "rgba(0,0,0,0.02)",
            "maxHeight": "260px",
            "overflowY": "auto",
        },
    )

    graph = dcc.Graph(
        id="curves-graph",
        style={"height": "75vh"},
        config={"displaylogo": False},
        figure=px.scatter(title="Select a curve type or adjust filters…"),
    )

    return html.Div(
        [controls, summary_div, graph],
        style={"padding": "8px", "maxWidth": "1400px", "margin": "0 auto"},
    )


# ---------------------------- callbacks -------------------------------------


def register_callbacks(app, ctx):
    # NOTE: Do NOT capture ctx.pmf_df / ctx.cum_df / etc. here.
    # Doing so would trigger their lazy loads at startup, before any tab is rendered.
    # Each callback reads from ctx inside its body so loading is deferred until
    # the user first visits the Curves tab.
    from .shared import apply_theme
    df_features: pd.DataFrame = getattr(ctx, "df", pd.DataFrame())  # features are pre-loaded

    @app.callback(
        Output("curve-metrics", "options"),
        Output("curve-metrics", "value"),
        Input("curve-type", "value"),
        prevent_initial_call=False,
    )
    def _populate_curve_controls(curve_type):
        """Populate metric options lazily — runs on first render, not at startup."""
        pmf_df = ctx.pmf_df
        cum_df = ctx.cum_df
        pmf_metrics: List[str] = []
        if isinstance(pmf_df, pd.DataFrame) and not pmf_df.empty and "metric" in pmf_df.columns:
            pmf_metrics = sorted(pmf_df["metric"].dropna().unique().tolist(), key=_metric_sort_key)
        cum_metrics: List[str] = []
        if isinstance(cum_df, pd.DataFrame) and not cum_df.empty and "metric" in cum_df.columns:
            cum_metrics = sorted(cum_df["metric"].dropna().unique().tolist(), key=_metric_sort_key)
        metric_options = sorted(set(pmf_metrics + cum_metrics), key=_metric_sort_key)
        opts = [{"label": metric_display_label(m), "value": m} for m in metric_options]
        default_val = [metric_options[0]] if metric_options else []
        return opts, default_val

    @app.callback(
        Output("rama2d-controls-wrapper", "style"),
        Input("curve-type", "value"),
        prevent_initial_call=False,
    )
    def _toggle_rama2d_controls(curve_type):
        return {"display": "block"} if curve_type == "rama2d" else {"display": "none"}

    @app.callback(
        Output("rama2d-residues", "options"),
        Output("rama2d-residues", "value"),
        Input("curve-type", "value"),
        Input("curve-variants", "value"),
        prevent_initial_call=False,
    )
    def _populate_rama2d_residues(curve_type, vlist):
        rama2d_perres_df = ctx.rama2d_perres_df
        if (
            curve_type != "rama2d"
            or not isinstance(rama2d_perres_df, pd.DataFrame)
            or rama2d_perres_df.empty
        ):
            return [], []
        vset = set(vlist or [])
        d = rama2d_perres_df
        if vset and "variant" in d.columns:
            d = d[d["variant"].isin(vset)]
        if d.empty or "residue" not in d.columns:
            return [], []
        idxs = sorted({int(i) for i in d["residue"].dropna().tolist()})
        base_v = str((vlist or [None])[0]) if (vlist or []) else ""
        opts = [{"label": residue_display(base_v, int(i)), "value": int(i)} for i in idxs]
        return opts, idxs[: min(6, len(idxs))]

    # keep slider min/max synced with *current* PMF data subset,
    # but DO NOT touch slider values (so user can freely drag).
    @app.callback(
        Output("curve-region1", "min"),
        Output("curve-region1", "max"),
        Output("curve-region1", "step"),
        Output("curve-region1", "marks"),
        Output("curve-region2", "min"),
        Output("curve-region2", "max"),
        Output("curve-region2", "step"),
        Output("curve-region2", "marks"),
        Input("curve-type", "value"),
        Input("curve-metrics", "value"),
        Input("curve-variants", "value"),
        prevent_initial_call=False,
    )
    def _update_region_slider_ranges(curve_type, metrics_sel, vlist):
        # Only meaningful for PMF-based curves
        if curve_type not in ("pmf_F", "pmf_P"):
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        pmf_df = ctx.pmf_df
        if not isinstance(pmf_df, pd.DataFrame) or pmf_df.empty or "x" not in pmf_df.columns:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        d = pmf_df.copy()

        # Filter by metrics
        if metrics_sel and "metric" in d.columns:
            d = d[d["metric"].isin(metrics_sel)]

        # Filter by selected variants
        if vlist and "variant" in d.columns:
            d = d[d["variant"].isin(vlist)]

        # If filters yield nothing, don't mess with user sliders
        if d.empty:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        
        # If torsion PMFs selected, keep a fixed [-180, 180] range for comparability
        if metrics_sel and all_torsion(metrics_sel):
            xmin, xmax = -180.0, 180.0
            step = 1.0
            marks = {
                -180.0: "-180",
                -120.0: "-120",
                -60.0: "-60",
                0.0: "0",
                60.0: "60",
                120.0: "120",
                180.0: "180",
            }
            return xmin, xmax, step, marks, xmin, xmax, step, marks

        xs = pd.to_numeric(d["x"], errors="coerce").dropna()
        if xs.empty:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        xmin = float(xs.min())
        xmax = float(xs.max())
        if xmax <= xmin:
            return no_update, no_update, no_update, no_update, no_update, no_update, no_update, no_update

        span = max(1e-9, xmax - xmin)
        step = span / 200.0
        xmid = 0.5 * (xmin + xmax)
        marks = {
            float(f"{xmin:.1f}"): f"{xmin:.1f}",
            float(f"{xmid:.1f}"): f"{xmid:.1f}",
            float(f"{xmax:.1f}"): f"{xmax:.1f}",
        }

        return xmin, xmax, step, marks, xmin, xmax, step, marks

    # sync numeric inputs with sliders (slider -> inputs)
    @app.callback(
        Output("curve-region1-min-input", "value"),
        Output("curve-region1-max-input", "value"),
        Output("curve-region2-min-input", "value"),
        Output("curve-region2-max-input", "value"),
        Input("curve-region1", "value"),
        Input("curve-region2", "value"),
        prevent_initial_call=False,
    )
    def _sync_inputs_from_sliders(region1_val, region2_val):
        r1_min = region1_val[0] if isinstance(region1_val, (list, tuple)) and len(region1_val) == 2 else no_update
        r1_max = region1_val[1] if isinstance(region1_val, (list, tuple)) and len(region1_val) == 2 else no_update
        r2_min = region2_val[0] if isinstance(region2_val, (list, tuple)) and len(region2_val) == 2 else no_update
        r2_max = region2_val[1] if isinstance(region2_val, (list, tuple)) and len(region2_val) == 2 else no_update
        return r1_min, r1_max, r2_min, r2_max

    # numeric inputs -> sliders, with clamping to current slider min/max
    @app.callback(
        Output("curve-region1", "value"),
        Output("curve-region2", "value"),
        Input("curve-region1-min-input", "value"),
        Input("curve-region1-max-input", "value"),
        Input("curve-region2-min-input", "value"),
        Input("curve-region2-max-input", "value"),
        State("curve-region1", "min"),
        State("curve-region1", "max"),
        State("curve-region1", "value"),
        State("curve-region2", "min"),
        State("curve-region2", "max"),
        State("curve-region2", "value"),
        prevent_initial_call=True,
    )
    def _sync_sliders_from_inputs(
        r1_min_in,
        r1_max_in,
        r2_min_in,
        r2_max_in,
        r1_min_slider,
        r1_max_slider,
        r1_val,
        r2_min_slider,
        r2_max_slider,
        r2_val,
    ):
        # Helper: clamp a pair to slider range and ensure lo < hi
        def clamp_pair(min_in, max_in, s_min, s_max, current):
            if current is None or not isinstance(current, (list, tuple)) or len(current) != 2:
                current = [s_min, s_max]
            lo, hi = current
            if min_in is not None:
                lo = float(min_in)
            if max_in is not None:
                hi = float(max_in)
            # fall back if slider bounds are None
            if s_min is None or s_max is None:
                return current
            s_min_f = float(s_min)
            s_max_f = float(s_max)
            # clamp
            lo = max(s_min_f, min(lo, s_max_f))
            hi = max(s_min_f, min(hi, s_max_f))
            if hi <= lo:
                # invalid range -> keep old
                return current
            return [lo, hi]

        r1_new = clamp_pair(r1_min_in, r1_max_in, r1_min_slider, r1_max_slider, r1_val)
        r2_new = clamp_pair(r2_min_in, r2_max_in, r2_min_slider, r2_max_slider, r2_val)
        return r1_new, r2_new

    @app.callback(
        Output("curves-graph", "figure"),
        Output("curves-summary", "children"),
        Input("curve-type", "value"),
        Input("curve-metrics", "value"),
        Input("curve-maxpanels", "value"),
        Input("curve-options", "value"),
        Input("curve-variants", "value"),
        Input("curve-ref-variant", "value"),
        Input("curve-region1", "value"),
        Input("curve-region2", "value"),
        Input("rama2d-residues", "value"),
        Input("rama2d-T-curves", "value"),
        Input("rama2d-pmfmax-curves", "value"),
        Input("theme-store", "data"),
        prevent_initial_call=False,
    )
    def update_curves(  # noqa: C901
        curve_type,
        metrics_sel,
        maxpanels,
        opts,
        vlist,
        ref_variant,
        region1,
        region2,
        rama_residues,
        Tcurves,
        pmfmax_curves,
        theme,
    ):
        # Read curve data lazily — first access triggers file/DB load; subsequent
        # accesses return the cached DataFrame from LazyCurvesLoader.
        pmf_df: pd.DataFrame = ctx.pmf_df
        pmf_replica_df: pd.DataFrame = ctx.pmf_replica_df
        cum_df: pd.DataFrame = ctx.cum_df
        cum_replica_df: pd.DataFrame = ctx.cum_replica_df
        rmsf_df: pd.DataFrame = ctx.rmsf_df
        rama2d_perres_df: pd.DataFrame = ctx.rama2d_perres_df
        rama2d_pooled_df: pd.DataFrame = ctx.rama2d_pooled_df

        opts = opts or []
        sharey = "sharey" in opts
        showerr = "err" in opts
        show_reps = "reps" in opts
        show_deriv = "deriv" in opts
        vset = set(vlist or [])

        # parse regions from sliders
        regions: List[Tuple[float, float]] = []
        if isinstance(region1, (list, tuple)) and len(region1) == 2:
            lo, hi = float(region1[0]), float(region1[1])
            if hi > lo:
                regions.append((lo, hi))
        if isinstance(region2, (list, tuple)) and len(region2) == 2:
            lo, hi = float(region2[0]), float(region2[1])
            if hi > lo:
                regions.append((lo, hi))

        # --- PMF F/P ---
        if curve_type in ("pmf_F", "pmf_P"):
            if not isinstance(pmf_df, pd.DataFrame) or pmf_df.empty:
                return apply_theme(error_fig("No PMF data found (place *_pmf.* in data-dir)."), theme), html.Div(
                    "No PMF data found."
                )
            if "metric" not in pmf_df.columns:
                return apply_theme(error_fig("PMF table has no 'metric' column."), theme), html.Div(
                    "PMF table has no 'metric' column."
                )
            avail_metrics = sorted(pmf_df["metric"].dropna().unique().tolist())
            metrics = [m for m in (metrics_sel or []) if m in avail_metrics] or (
                avail_metrics[:1] if avail_metrics else []
            )
            metrics = metrics[: int(maxpanels or 3)]
            n = len(metrics)
            if n == 0:
                return apply_theme(error_fig("No metrics available in PMF table."), theme), html.Div(
                    "No metrics available in PMF table."
                )

            cols = 1 if n == 1 else min(3, n)
            rows_main = int(math.ceil(n / cols))

            # for pmf_F + derivatives, double the number of rows (F on top, derivatives below)
            if curve_type == "pmf_F" and show_deriv:
                rows = rows_main * 2
                row_heights: Optional[List[float]] = []
                for _ in range(rows_main):
                    row_heights.extend([0.72, 0.28])  # F taller, deriv shorter
            else:
                rows = rows_main
                row_heights = None

            base_titles = [
                f"{m} — {'F(x)' if curve_type == 'pmf_F' else 'P(x)'}" for m in metrics
            ]
            n_subplots = rows * cols
            if len(base_titles) < n_subplots:
                titles = base_titles + [""] * (n_subplots - len(base_titles))
            else:
                titles = base_titles

            fig = make_subplots(
                rows=rows,
                cols=cols,
                shared_yaxes=sharey,
                subplot_titles=titles,
                row_heights=row_heights,
            )

            summary_children: List[Any] = []
            fmins: List[float] = []
            area_raw_vals: List[float] = []
            summary_rows_P: List[Dict[str, Any]] = []
            summary_rows_F: List[Dict[str, Any]] = []
            shapes: List[Dict[str, Any]] = []
            panel_ranges: Dict[int, Tuple[float, float]] = {}

            for k, m in enumerate(metrics):
                col_idx = (k % cols) + 1

                if curve_type == "pmf_F":
                    row_group = k // cols
                    if show_deriv:
                        rF = 2 * row_group + 1
                        rD = 2 * row_group + 2
                    else:
                        rF = row_group + 1
                        rD = None
                else:
                    r = (k // cols) + 1
                    rF = r  # for pmf_P we only have one row
                    rD = None

                sub = pmf_df[pmf_df["metric"] == m]
                sub_rep = (
                    pmf_replica_df[pmf_replica_df["metric"] == m]
                    if isinstance(pmf_replica_df, pd.DataFrame)
                    and (not pmf_replica_df.empty)
                    and ("metric" in pmf_replica_df.columns)
                    else pd.DataFrame()
                )
                if vset and "variant" in sub.columns:
                    sub = sub[sub["variant"].isin(vset)]
                if vset and "variant" in sub_rep.columns:
                    sub_rep = sub_rep[sub_rep["variant"].isin(vset)]
                if sub.empty:
                    axidx = k + 1
                    xref = "x domain" if axidx == 1 else f"x{axidx} domain"
                    yref = "y domain" if axidx == 1 else f"y{axidx} domain"
                    fig.add_annotation(
                        text="No rows",
                        xref=xref,
                        yref=yref,
                        x=0.5,
                        y=0.5,
                        showarrow=False,
                    )
                    continue

                # Track x-range for this panel for region shading
                panel_xmin: Optional[float] = None
                panel_xmax: Optional[float] = None

                for var, dd in sub.groupby("variant", sort=False, observed=False):
                    dd_rep = dd
                    if isinstance(sub_rep, pd.DataFrame) and (not sub_rep.empty) and ("variant" in sub_rep.columns):
                        try:
                            dd_rep = sub_rep[sub_rep["variant"].astype(str) == str(var)]
                        except Exception:
                            dd_rep = sub_rep[sub_rep["variant"] == var]
                        if dd_rep.empty:
                            dd_rep = dd
                    repcol = _detect_replica_col(dd_rep)
                    is_ref = ref_variant is not None and str(var) == str(ref_variant)
                    line_width = 3 if is_ref else 1.5
                    opacity = 1.0 if is_ref else 0.4
                    name_main = f"{var} (ref)" if is_ref else str(var)

                    # ---- PMF F(x) ----
                    if curve_type == "pmf_F":
                        want_err = bool(showerr)
                        if "x" not in dd.columns or "F_kJ_mol" not in dd.columns:
                            continue

                        cols_y = ["F_kJ_mol"] + (["dF"] if "dF" in dd.columns else [])
                        dds = _prep_xy(dd, "x", cols_y)
                        if dds.empty:
                            continue

                        # If replica column exists, optionally plot per-replica traces and collapse to mean for the main trace.
                        dds_main = dds
                        if show_reps and (repcol is not None):
                            try:
                                for rep_id, ddr in dd_rep.groupby(repcol, sort=False, observed=False):
                                    ddrs = _prep_xy(ddr, "x", ["F_kJ_mol"])
                                    if ddrs.empty:
                                        continue
                                    xr = ddrs["x"].to_numpy(dtype=float)
                                    yr = ddrs["F_kJ_mol"].to_numpy(dtype=float)
                                    vv = np.isfinite(xr) & np.isfinite(yr)
                                    if vv.sum() < 2:
                                        continue
                                    xr = xr[vv]
                                    yr = yr[vv]
                                    ord_r = np.argsort(xr)
                                    xr = xr[ord_r]
                                    yr = yr[ord_r]
                                    fig.add_trace(
                                        go.Scatter(
                                            x=xr,
                                            y=yr,
                                            mode="lines",
                                            name=f"{var} rep {rep_id}",
                                            legendgroup=str(var),
                                            connectgaps=False,
                                            opacity=0.20,
                                            line=dict(width=1),
                                            showlegend=False,
                                        ),
                                        row=rF,
                                        col=col_idx,
                                    )
                            except Exception:
                                pass
                            try:
                                dds_main = dds.groupby("x", as_index=False, observed=False).mean(numeric_only=True)
                            except Exception:
                                dds_main = dds

                        # Main trace is collapsed to x-binned mean (averaging across replicas if present).
                        try:
                            dds_main = dds.groupby("x", as_index=False, observed=False).mean(numeric_only=True)
                        except Exception:
                            dds_main = dds

                        x_raw = dds_main["x"].to_numpy(dtype=float)
                        y_raw = dds_main["F_kJ_mol"].to_numpy(dtype=float)

                        valid = np.isfinite(x_raw) & np.isfinite(y_raw)
                        if valid.sum() < 2:
                            continue
                        x_raw = x_raw[valid]
                        y_raw = y_raw[valid]
                        order = np.argsort(x_raw)
                        x_raw = x_raw[order]
                        y_raw = y_raw[order]

                        xmin_local = float(np.nanmin(x_raw))
                        xmax_local = float(np.nanmax(x_raw))
                        if panel_xmin is None or xmin_local < panel_xmin:
                            panel_xmin = xmin_local
                        if panel_xmax is None or xmax_local > panel_xmax:
                            panel_xmax = xmax_local

                        fmins.append(float(np.nanmin(y_raw)))

                        # Plot main F(x) in the F-row
                        fig.add_trace(
                            go.Scatter(
                                x=x_raw,
                                y=y_raw,
                                mode="lines",
                                name=name_main,
                                legendgroup=str(var),
                                connectgaps=False,
                                opacity=opacity,
                                line=dict(width=line_width),
                            ),
                            row=rF,
                            col=col_idx,
                        )
                        # Error band (±1σ): prefer provided dF; otherwise fall back to replica variability.
                        if want_err:
                            dy_raw: Optional[np.ndarray] = None
                            if "dF" in dds_main.columns:
                                try:
                                    dy_raw = dds_main["dF"].to_numpy(dtype=float)
                                    dy_raw = dy_raw[valid][order]
                                except Exception:
                                    dy_raw = None
                            elif repcol is not None:
                                try:
                                    stats_r = _replica_mean_std_by_x(dd_rep, "x", "F_kJ_mol", repcol=repcol)
                                    dy_raw = _align_std_to_x(x_raw, stats_r, xcol="x")
                                except Exception:
                                    dy_raw = None
                            if dy_raw is not None and np.isfinite(dy_raw).sum() >= 2:
                                _add_error_band(fig, x_raw, y_raw, dy_raw, row=rF, col=col_idx)

                        # Derivatives - use x-unique averaged F to avoid zero Δx issues
                        if show_deriv and rD is not None:
                            uniq_x, inv = np.unique(x_raw, return_inverse=True)
                            if uniq_x.size > 1:
                                F_collapsed = np.zeros_like(uniq_x, dtype=float)
                                counts = np.zeros_like(uniq_x, dtype=int)
                                for idx_point, idx_bin in enumerate(inv):
                                    F_collapsed[idx_bin] += y_raw[idx_point]
                                    counts[idx_bin] += 1
                                counts[counts == 0] = 1
                                F_collapsed = F_collapsed / counts

                                with np.errstate(all="ignore"):
                                    dFdx = np.gradient(F_collapsed, uniq_x)

                                fig.add_trace(
                                    go.Scatter(
                                        x=uniq_x,
                                        y=dFdx,
                                        mode="lines",
                                        line=dict(dash="dot", width=1),
                                        name=f"{var} dF/dx",
                                        legendgroup=f"{var}_deriv1",
                                        showlegend=is_ref,
                                        opacity=opacity,
                                    ),
                                    row=rD,
                                    col=col_idx,
                                )

                                if uniq_x.size > 2:
                                    with np.errstate(all="ignore"):
                                        d2Fdx2 = np.gradient(dFdx, uniq_x)
                                    fig.add_trace(
                                        go.Scatter(
                                            x=uniq_x,
                                            y=d2Fdx2,
                                            mode="lines",
                                            line=dict(dash="dashdot", width=1),
                                            name=f"{var} d²F/dx²",
                                            legendgroup=f"{var}_deriv2",
                                            showlegend=is_ref,
                                            opacity=opacity * 0.9,
                                        ),
                                        row=rD,
                                        col=col_idx,
                                    )

                        fig.update_yaxes(title_text="F (kJ/mol)", row=rF, col=col_idx)
                        if show_deriv and rD is not None:
                            fig.update_yaxes(
                                title_text="dF/dx, d²F/dx²",
                                row=rD,
                                col=col_idx,
                            )

                        # region minima & ΔF from F(x)
                        if regions:
                            region_Fmins: List[float] = []
                            for (lo, hi) in regions:
                                lo_eff = max(lo, xmin_local)
                                hi_eff = min(hi, xmax_local)
                                if hi_eff <= lo_eff:
                                    region_Fmins.append(float("nan"))
                                    continue
                                mask_reg = (x_raw >= lo_eff) & (x_raw <= hi_eff)
                                if np.any(mask_reg):
                                    Fmin = float(np.nanmin(y_raw[mask_reg]))
                                else:
                                    Fmin = float("nan")
                                region_Fmins.append(Fmin)

                            row_info_F: Dict[str, Any] = {
                                "metric": m,
                                "variant": str(var),
                                "is_ref": bool(is_ref),
                            }
                            for idx_reg, Fmin in enumerate(region_Fmins):
                                row_info_F[f"state{idx_reg+1}_Fmin_kJmol"] = Fmin
                            if len(region_Fmins) >= 2:
                                F1, F2 = region_Fmins[0], region_Fmins[1]
                                if np.isfinite(F1) and np.isfinite(F2):
                                    row_info_F["DeltaF_state2-1_kJmol"] = F2 - F1
                                else:
                                    row_info_F["DeltaF_state2-1_kJmol"] = float("nan")
                            summary_rows_F.append(row_info_F)

                    # ---- PMF P(x) ----
                    else:
                        want_err = bool(showerr)
                        if "x" not in dd.columns or "P" not in dd.columns:
                            continue
                        dds = _prep_xy(dd, "x", ["P"])
                        if dds.empty:
                            continue

                        dds_main = dds
                        if show_reps and (repcol is not None):
                            try:
                                for rep_id, ddr in dd_rep.groupby(repcol, sort=False, observed=False):
                                    ddrs = _prep_xy(ddr, "x", ["P"])
                                    if ddrs.empty:
                                        continue
                                    xr = ddrs["x"].to_numpy(dtype=float)
                                    pr = ddrs["P"].to_numpy(dtype=float)
                                    vv = np.isfinite(xr) & np.isfinite(pr)
                                    if vv.sum() < 2:
                                        continue
                                    xr = xr[vv]
                                    pr = pr[vv]
                                    ord_r = np.argsort(xr)
                                    xr = xr[ord_r]
                                    pr = pr[ord_r]
                                    # Normalize each replica curve to unit area for comparability
                                    area_r = float(np.trapz(np.clip(pr, 0, np.inf), xr))
                                    if area_r > 0:
                                        pr = pr / area_r
                                    fig.add_trace(
                                        go.Scatter(
                                            x=xr,
                                            y=pr,
                                            mode="lines",
                                            name=f"{var} rep {rep_id}",
                                            legendgroup=str(var),
                                            connectgaps=False,
                                            opacity=0.20,
                                            line=dict(width=1),
                                            showlegend=False,
                                        ),
                                        row=rF,
                                        col=col_idx,
                                    )
                            except Exception:
                                pass
                            try:
                                dds_main = dds.groupby("x", as_index=False, observed=False).mean(numeric_only=True)
                            except Exception:
                                dds_main = dds

                        # Main trace is collapsed to x-binned mean (averaging across replicas if present).
                        try:
                            dds_main = dds.groupby("x", as_index=False, observed=False).mean(numeric_only=True)
                        except Exception:
                            dds_main = dds

                        x = dds_main["x"].to_numpy(dtype=float)
                        P_raw = dds_main["P"].to_numpy(dtype=float)

                        valid = np.isfinite(x) & np.isfinite(P_raw)
                        if valid.sum() < 2:
                            continue
                        x = x[valid]
                        P_raw = P_raw[valid]
                        order = np.argsort(x)
                        x = x[order]
                        P_raw = P_raw[order]

                        xmin_local = float(np.nanmin(x))
                        xmax_local = float(np.nanmax(x))
                        if panel_xmin is None or xmin_local < panel_xmin:
                            panel_xmin = xmin_local
                        if panel_xmax is None or xmax_local > panel_xmax:
                            panel_xmax = xmax_local

                        area_raw = float(np.trapz(np.clip(P_raw, 0, np.inf), x))
                        area_raw_vals.append(area_raw if np.isfinite(area_raw) else np.nan)
                        if area_raw > 0:
                            P = P_raw / area_raw
                        else:
                            P = P_raw.copy()

                        # region integrals for P(x) using sliders & local x-range
                        region_vals: List[float] = []
                        for (lo, hi) in regions:
                            lo_eff = max(lo, xmin_local)
                            hi_eff = min(hi, xmax_local)
                            if hi_eff <= lo_eff:
                                region_vals.append(0.0)
                                continue
                            mask = (x >= lo_eff) & (x <= hi_eff)
                            if np.count_nonzero(mask) >= 2:
                                val = float(np.trapz(P[mask], x[mask]))
                            else:
                                val = 0.0
                            region_vals.append(val)

                        row_info: Dict[str, Any] = {
                            "metric": m,
                            "variant": str(var),
                            "is_ref": bool(is_ref),
                            "area_raw": area_raw,
                        }
                        for idx_reg, val in enumerate(region_vals):
                            row_info[f"state{idx_reg+1}_pop"] = val
                        # NOTE: no ΔG/ΔF here – probabilities only
                        summary_rows_P.append(row_info)

                        fig.add_trace(
                            go.Scatter(
                                x=x,
                                y=P,
                                mode="lines",
                                name=name_main,
                                legendgroup=str(var),
                                connectgaps=False,
                                opacity=opacity,
                                line=dict(width=line_width),
                            ),
                            row=rF,
                            col=col_idx,
                        )
                                                # Error band (±1σ) from replica variability (after per-replica normalization).
                        if want_err and (repcol is not None):
                            try:
                                stats_p, _area_mean = _replica_normed_curve_stats(dd_rep, "x", "P", repcol=repcol)
                                dy_p = _align_std_to_x(x, stats_p, xcol="x")
                                if np.isfinite(dy_p).sum() >= 2:
                                    _add_error_band(fig, x, P, dy_p, row=rF, col=col_idx)
                            except Exception:
                                pass

                        fig.update_yaxes(title_text="P(x)", row=rF, col=col_idx)

                    # x-axis label: put it on the lowest row associated with this panel
                    if curve_type == "pmf_F" and show_deriv and rD is not None:
                        fig.update_xaxes(title_text="x", row=rD, col=col_idx)
                    else:
                        fig.update_xaxes(title_text="x", row=rF, col=col_idx)

                # store panel x-range
                if panel_xmin is not None and panel_xmax is not None:
                    panel_ranges[k] = (panel_xmin, panel_xmax)

                # draw region markers on F(x) and P(x) panels if we have regions
                if curve_type in ("pmf_F", "pmf_P") and regions and k in panel_ranges:
                    panel_xmin, panel_xmax = panel_ranges[k]
                    for (lo, hi) in regions:
                        x0 = max(lo, panel_xmin)
                        x1 = min(hi, panel_xmax)
                        if x1 <= x0:
                            continue
                        shapes.append(
                            dict(
                                type="rect",
                                xref=f"x{k+1}",
                                yref="paper",
                                x0=x0,
                                x1=x1,
                                y0=0.0,
                                y1=1.0,
                                fillcolor="rgba(120,160,255,0.30)",
                                line=dict(width=1, color="rgba(120,160,255,0.9)"),
                                layer="below",
                            )
                        )

            if shapes:
                fig.update_layout(shapes=tuple(shapes))

            fig.update_layout(
                height=780,
                margin=dict(l=60, r=20, t=60, b=60),
                legend=dict(
                    orientation="h",
                    y=1.04,
                    x=1.0,
                    xanchor="right",
                    yanchor="bottom",
                    font=dict(size=9),
                ),
                hovermode="x unified",
            )

            # ---- summary text / tables ----
            if curve_type == "pmf_F":
                if fmins:
                    fmins_arr = np.array(fmins, float)
                    summary_children.append(
                        html.Div(
                            f"PMF F(x): min(F) across curves ranges from "
                            f"{np.nanmin(fmins_arr):.3f} to {np.nanmax(fmins_arr):.3f} kJ/mol. "
                            "Ideally min(F) ≈ 0; offsets indicate arbitrary zero.",
                            style={"marginBottom": "4px"},
                        )
                    )
                if show_deriv:
                    summary_children.append(
                        html.Div(
                            "For each metric, the top panel shows F(x); "
                            "the bottom panel shows dF/dx (dotted) and d²F/dx² (dash-dot) "
                            "computed on an x-unique, averaged F(x).",
                            style={"marginBottom": "4px"},
                        )
                    )
                if regions and summary_rows_F:
                    df_sumF = pd.DataFrame(summary_rows_F)
                    num_cols = [c for c in df_sumF.columns if c not in ("metric", "variant")]
                    df_sumF[num_cols] = df_sumF[num_cols].astype(float).round(3)
                    if "is_ref" in df_sumF.columns:
                        df_sumF = df_sumF.sort_values(
                            by=["metric", "is_ref", "variant"],
                            ascending=[True, False, True],
                        )

                    def _pretty_header(col: str) -> str:
                        if col == "metric":
                            return "metric"
                        if col == "variant":
                            return "variant"
                        if col == "is_ref":
                            return "ref?"
                        if col.startswith("state") and col.endswith("_Fmin_kJmol"):
                            idx = col[len("state") : col.find("_")]
                            return f"Fmin{idx} (kJ/mol)"
                        if col == "DeltaF_state2-1_kJmol":
                            return "ΔF₂₋₁ (kJ/mol)"
                        return col

                    header = html.Tr(
                        [
                            html.Th(
                                _pretty_header(c),
                                style={
                                    "border": "1px solid #ccc",
                                    "padding": "2px 4px",
                                    "fontWeight": "600",
                                    "whiteSpace": "nowrap",
                                },
                            )
                            for c in df_sumF.columns
                        ]
                    )
                    rows_html = []
                    for _, row in df_sumF.iterrows():
                        rows_html.append(
                            html.Tr(
                                [
                                    html.Td(
                                        str(row[c]),
                                        style={
                                            "border": "1px solid #eee",
                                            "padding": "2px 4px",
                                            "whiteSpace": "nowrap",
                                        },
                                    )
                                    for c in df_sumF.columns
                                ],
                                style={"fontSize": "0.78em"},
                            )
                        )
                    summary_children.append(
                        html.Div(
                            [
                                html.Div(
                                    "State minima & ΔF from PMF F(x). "
                                    "ΔF₂₋₁ = F₂,min − F₁,min.",
                                    style={"marginBottom": "2px"},
                                ),
                                html.Table(
                                    [header] + rows_html,
                                    style={
                                        "width": "100%",
                                        "borderCollapse": "collapse",
                                        "border": "1px solid #ccc",
                                    },
                                ),
                            ]
                        )
                    )
                elif regions:
                    summary_children.append(
                        html.Div(
                            "Regions defined but no valid PMF F(x) data within those ranges.",
                            style={"marginBottom": "4px"},
                        )
                    )

            else:  # pmf_P
                summary_children.append(
                    html.Div(
                        "PMF P(x) curves are normalized so ∫P(x)dx ≈ 1 in each panel.",
                        style={"marginBottom": "4px"},
                    )
                )
                if area_raw_vals:
                    arr = np.array(area_raw_vals, float)
                    summary_children.append(
                        html.Div(
                            f"Raw areas before normalization (trapz(P_raw dx)) "
                            f"range from {np.nanmin(arr):.3f} to {np.nanmax(arr):.3f}.",
                            style={"marginBottom": "4px"},
                        )
                    )
                if regions and summary_rows_P:
                    df_sum = pd.DataFrame(summary_rows_P)
                    num_cols = [c for c in df_sum.columns if c not in ("metric", "variant")]
                    df_sum[num_cols] = df_sum[num_cols].astype(float).round(3)
                    if "is_ref" in df_sum.columns:
                        df_sum = df_sum.sort_values(
                            by=["metric", "is_ref", "variant"],
                            ascending=[True, False, True],
                        )

                    def _pretty_header_p(col: str) -> str:
                        if col == "metric":
                            return "metric"
                        if col == "variant":
                            return "variant"
                        if col == "is_ref":
                            return "ref?"
                        if col == "area_raw":
                            return "raw ∫P dx"
                        if col.startswith("state") and col.endswith("_pop"):
                            idx = col[len("state") : col.find("_")]
                            return f"state {idx} pop"
                        return col

                    header = html.Tr(
                        [
                            html.Th(
                                _pretty_header_p(c),
                                style={
                                    "border": "1px solid #ccc",
                                    "padding": "2px 4px",
                                    "fontWeight": "600",
                                    "whiteSpace": "nowrap",
                                },
                            )
                            for c in df_sum.columns
                        ]
                    )
                    rows_html = []
                    for _, row in df_sum.iterrows():
                        rows_html.append(
                            html.Tr(
                                [
                                    html.Td(
                                        str(row[c]),
                                        style={
                                            "border": "1px solid #eee",
                                            "padding": "2px 4px",
                                            "whiteSpace": "nowrap",
                                        },
                                    )
                                    for c in df_sum.columns
                                ],
                                style={"fontSize": "0.78em"},
                            )
                        )
                    summary_children.append(
                        html.Div(
                            [
                                html.Div(
                                    "State populations from PMF P(x) "
                                    "(State 1 = region 1, State 2 = region 2). "
                                    "No ΔF is computed here; this is purely probabilistic.",
                                    style={"marginBottom": "2px"},
                                ),
                                html.Table(
                                    [header] + rows_html,
                                    style={
                                        "width": "100%",
                                        "borderCollapse": "collapse",
                                        "border": "1px solid #ccc",
                                    },
                                ),
                            ]
                        )
                    )
                elif regions:
                    summary_children.append(
                        html.Div(
                            "Integration regions defined but no data rows passed the filters.",
                            style={"marginBottom": "4px"},
                        )
                    )

            return apply_theme(fig, theme), summary_children

        # --- Cumulative FE / ΔF ---
        if curve_type in ("cum_FE", "cum_DeltaF"):
            if not isinstance(cum_df, pd.DataFrame) or cum_df.empty:
                return apply_theme(error_fig("No cumulative data found (place *_cumulative.*)."), theme), html.Div(
                    "No cumulative data found."
                )
            if "metric" not in cum_df.columns:
                return apply_theme(error_fig("Cumulative table has no 'metric' column."), theme), html.Div(
                    "Cumulative table has no 'metric' column."
                )

            avail_metrics = sorted(cum_df["metric"].dropna().unique().tolist())
            metrics = [m for m in (metrics_sel or []) if m in avail_metrics] or (
                avail_metrics[:1] if avail_metrics else []
            )
            metrics = metrics[: int(maxpanels or 3)]
            n = len(metrics)
            if n == 0:
                return apply_theme(error_fig("No metrics available in cumulative table."), theme), html.Div(
                    "No metrics available in cumulative table."
                )
            cols = 1 if n == 1 else min(3, n)
            rows = int(math.ceil(n / cols))
            base_titles = [
                f"{m} — {'Fcum (L/R)' if curve_type == 'cum_FE' else 'ΔF(L/R)'}"
                for m in metrics
            ]
            n_subplots = rows * cols
            if len(base_titles) < n_subplots:
                titles = base_titles + [""] * (n_subplots - len(base_titles))
            else:
                titles = base_titles

            fig = make_subplots(
                rows=rows,
                cols=cols,
                shared_yaxes=sharey,
                subplot_titles=titles,
            )

            for k, m in enumerate(metrics):
                r = (k // cols) + 1
                c = (k % cols) + 1
                sub = cum_df[cum_df["metric"] == m]
                sub_rep = (
                    cum_replica_df[cum_replica_df["metric"] == m]
                    if isinstance(cum_replica_df, pd.DataFrame)
                    and (not cum_replica_df.empty)
                    and ("metric" in cum_replica_df.columns)
                    else pd.DataFrame()
                )
                if vset and "variant" in sub.columns:
                    sub = sub[sub["variant"].isin(vset)]
                if vset and "variant" in sub_rep.columns:
                    sub_rep = sub_rep[sub_rep["variant"].isin(vset)]
                if sub.empty:
                    continue
                for var, dd in sub.groupby("variant", sort=False, observed=False):
                    dd_rep = dd
                    if isinstance(sub_rep, pd.DataFrame) and (not sub_rep.empty) and ("variant" in sub_rep.columns):
                        try:
                            dd_rep = sub_rep[sub_rep["variant"].astype(str) == str(var)]
                        except Exception:
                            dd_rep = sub_rep[sub_rep["variant"] == var]
                        if dd_rep.empty:
                            dd_rep = dd
                    repcol = _detect_replica_col(dd_rep)
                    is_ref = ref_variant is not None and str(var) == str(ref_variant)
                    line_width = 3 if is_ref else 1.5
                    opacity = 1.0 if is_ref else 0.4
                    name_main = f"{var} (ref)" if is_ref else str(var)

                    if curve_type == "cum_FE":
                        needed = [
                            col
                            for col in ["x", "Fcum_left", "Fcum_right"]
                            if col in dd.columns
                        ]
                        if len(needed) < 3:
                            continue
                        dds = _prep_xy(dd, "x", ["Fcum_left", "Fcum_right"])
                        if dds.empty:
                            continue

                        if show_reps and (repcol is not None):
                            try:
                                for rep_id, ddr in dd_rep.groupby(repcol, sort=False, observed=False):
                                    ddrs = _prep_xy(ddr, "x", ["Fcum_left", "Fcum_right"])
                                    if ddrs.empty:
                                        continue
                                    for side, coly in (("L", "Fcum_left"), ("R", "Fcum_right")):
                                        xr = ddrs["x"].to_numpy(dtype=float)
                                        yr = ddrs[coly].to_numpy(dtype=float)
                                        vv = np.isfinite(xr) & np.isfinite(yr)
                                        if vv.sum() < 2:
                                            continue
                                        xr = xr[vv]
                                        yr = yr[vv]
                                        o = np.argsort(xr)
                                        xr = xr[o]
                                        yr = yr[o]
                                        fig.add_trace(
                                            go.Scatter(
                                                x=xr,
                                                y=yr,
                                                mode="lines",
                                                name=f"{name_main} rep {rep_id} ({side})",
                                                legendgroup=str(var),
                                                connectgaps=False,
                                                opacity=0.18,
                                                line=dict(width=1),
                                                showlegend=False,
                                            ),
                                            row=r,
                                            col=c,
                                        )
                            except Exception:
                                pass

                        try:
                            dds = dds.groupby("x", as_index=False, observed=False).mean(numeric_only=True)
                        except Exception:
                            pass
                        fig.add_trace(
                            go.Scatter(
                                x=dds["x"],
                                y=dds["Fcum_left"],
                                mode="lines",
                                name=f"{name_main} (L)",
                                legendgroup=str(var),
                                connectgaps=False,
                                opacity=opacity,
                                line=dict(width=line_width),
                            ),
                            row=r,
                            col=c,
                        )
                        fig.add_trace(
                            go.Scatter(
                                x=dds["x"],
                                y=dds["Fcum_right"],
                                mode="lines",
                                name=f"{name_main} (R)",
                                legendgroup=str(var),
                                connectgaps=False,
                                opacity=opacity,
                                line=dict(width=line_width),
                            ),
                            row=r,
                            col=c,
                        )
                                                # Error bands from replica variability (±1σ)
                        if showerr and (repcol is not None):
                            try:
                                stats_l = _replica_mean_std_by_x(dd_rep, "x", "Fcum_left", repcol=repcol)
                                stats_r = _replica_mean_std_by_x(dd_rep, "x", "Fcum_right", repcol=repcol)
                                x_main = dds["x"].to_numpy(dtype=float)
                                y_l = dds["Fcum_left"].to_numpy(dtype=float)
                                y_r = dds["Fcum_right"].to_numpy(dtype=float)
                                dy_l = _align_std_to_x(x_main, stats_l, xcol="x")
                                dy_r = _align_std_to_x(x_main, stats_r, xcol="x")
                                if np.isfinite(dy_l).sum() >= 2:
                                    _add_error_band(fig, x_main, y_l, dy_l, row=r, col=c)
                                if np.isfinite(dy_r).sum() >= 2:
                                    _add_error_band(fig, x_main, y_r, dy_r, row=r, col=c)
                            except Exception:
                                pass

                        fig.update_yaxes(title_text="Fcum (kJ/mol)", row=r, col=c)
                    else:
                        if "x" not in dd.columns:
                            continue
                        yname = (
                            "DeltaF_left_right"
                            if "DeltaF_left_right" in dd.columns
                            else None
                        )
                        if not yname:
                            continue
                        dds = _prep_xy(dd, "x", [yname])
                        if dds.empty:
                            continue

                        if show_reps and (repcol is not None):
                            try:
                                for rep_id, ddr in dd_rep.groupby(repcol, sort=False, observed=False):
                                    ddrs = _prep_xy(ddr, "x", [yname])
                                    if ddrs.empty:
                                        continue
                                    xr = ddrs["x"].to_numpy(dtype=float)
                                    yr = ddrs[yname].to_numpy(dtype=float)
                                    vv = np.isfinite(xr) & np.isfinite(yr)
                                    if vv.sum() < 2:
                                        continue
                                    xr = xr[vv]
                                    yr = yr[vv]
                                    o = np.argsort(xr)
                                    xr = xr[o]
                                    yr = yr[o]
                                    fig.add_trace(
                                        go.Scatter(
                                            x=xr,
                                            y=yr,
                                            mode="lines",
                                            name=f"{name_main} rep {rep_id}",
                                            legendgroup=str(var),
                                            connectgaps=False,
                                            opacity=0.18,
                                            line=dict(width=1),
                                            showlegend=False,
                                        ),
                                        row=r,
                                        col=c,
                                    )
                            except Exception:
                                pass

                        try:
                            dds = dds.groupby("x", as_index=False, observed=False).mean(numeric_only=True)
                        except Exception:
                            pass
                        fig.add_trace(
                            go.Scatter(
                                x=dds["x"],
                                y=dds[yname],
                                mode="lines",
                                name=name_main,
                                legendgroup=str(var),
                                connectgaps=False,
                                opacity=opacity,
                                line=dict(width=line_width),
                            ),
                            row=r,
                            col=c,
                        )
                                                # Error band from replica variability (±1σ)
                        if showerr and (repcol is not None):
                            try:
                                stats_d = _replica_mean_std_by_x(dd_rep, "x", yname, repcol=repcol)
                                x_main = dds["x"].to_numpy(dtype=float)
                                y_main = dds[yname].to_numpy(dtype=float)
                                dy = _align_std_to_x(x_main, stats_d, xcol="x")
                                if np.isfinite(dy).sum() >= 2:
                                    _add_error_band(fig, x_main, y_main, dy, row=r, col=c)
                            except Exception:
                                pass

                        fig.update_yaxes(title_text="ΔF(L/R) (kJ/mol)", row=r, col=c)
                    fig.update_xaxes(title_text="x", row=r, col=c)

            fig.update_layout(
                height=750,
                margin=dict(l=60, r=20, t=60, b=60),
                legend=dict(
                    orientation="h",
                    y=1.03,
                    x=1.0,
                    xanchor="right",
                    yanchor="bottom",
                    font=dict(size=9),
                ),
                hovermode="x unified",
            )
            summary_children = html.Div(
                "Cumulative FE/ΔF is shown per metric and variant. "
                "State-region analysis is based on PMF F(x)/P(x) in the PMF modes.",
            )
            return apply_theme(fig, theme), summary_children

        # --- RMSF per-replica box ---
        if curve_type == "rmsf":
            if not isinstance(rmsf_df, pd.DataFrame) or rmsf_df.empty:
                return error_fig("No *_replica tables with RMSF metrics found."), html.Div(
                    "No RMSF data found."
                )
            sub = rmsf_df.copy()
            if vset and "variant" in sub.columns:
                sub = sub[sub["variant"].isin(vset)]
            if "metric" in sub.columns:
                sub = sub[sub["metric"] == "rmsf"]
            if sub.empty or "mean" not in sub.columns:
                return error_fig("No RMSF rows present in replica tables."), html.Div(
                    "No RMSF rows present in replica tables."
                )

            x_arg = "variant" if "variant" in sub.columns else None
            fig = px.box(
                sub,
                x=x_arg,
                y="mean",
                points="all",
                title="Per-replica RMSF (mean over residues)",
            )
            fig.update_layout(
                height=700,
                margin=dict(l=60, r=20, t=60, b=60),
                showlegend=False,
            )

            # Overlay mean ±1σ across replicas
            if "variant" in sub.columns:
                try:
                    stats = (
                        sub.groupby("variant", observed=False)["mean"]
                        .agg(["mean", "std", "count"])
                        .reset_index()
                    )
                    fig.add_trace(
                        go.Scatter(
                            x=stats["variant"].astype(str),
                            y=stats["mean"].to_numpy(dtype=float),
                            mode="markers",
                            name="mean ±1σ",
                            error_y=dict(type="data", array=stats["std"].fillna(0.0).to_numpy(dtype=float), visible=True),
                        )
                    )
                    fig.update_layout(showlegend=True)
                except Exception:
                    pass

            summary_children: List[Any] = []
            if "variant" in sub.columns:
                stats = (
                    sub.groupby("variant", observed=False)["mean"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                )
                stats["mean"] = stats["mean"].round(3)
                stats["std"] = stats["std"].round(3)
                header = html.Tr(
                    [
                        html.Th(
                            c,
                            style={
                                "border": "1px solid #ccc",
                                "padding": "2px 4px",
                                "fontWeight": "600",
                            },
                        )
                        for c in stats.columns
                    ]
                )
                rows_html = []
                for _, row in stats.iterrows():
                    rows_html.append(
                        html.Tr(
                            [
                                html.Td(
                                    str(row[c]),
                                    style={
                                        "border": "1px solid #eee",
                                        "padding": "2px 4px",
                                    },
                                )
                                for c in stats.columns
                            ],
                            style={"fontSize": "0.78em"},
                        )
                    )
                summary_children.append(
                    html.Div(
                        [
                            html.Div(
                                "RMSF per-replica statistics (mean over residues).",
                                style={"marginBottom": "2px"},
                            ),
                            html.Table(
                                [header] + rows_html,
                                style={
                                    "width": "100%",
                                    "borderCollapse": "collapse",
                                    "border": "1px solid #ccc",
                                },
                            ),
                        ]
                    )
                )
                if ref_variant and ref_variant in stats["variant"].values:
                    ref_mean = float(
                        stats.loc[stats["variant"] == ref_variant, "mean"].iloc[0]
                    )
                    summary_children.append(
                        html.Div(
                            f"Reference variant {ref_variant}: mean RMSF = {ref_mean:.3f}. "
                            "Differences vs reference can be read directly from the table.",
                            style={"marginTop": "4px"},
                        )
                    )
            return apply_theme(fig, theme), summary_children

        # --- Ramachandran 1D circular R ---
        if curve_type == "rama":
            df = df_features if isinstance(df_features, pd.DataFrame) else pd.DataFrame()
            if df.empty:
                return error_fig(
                    "No feature table available for Ramachandran statistics."
                ), html.Div("No feature table available for Ramachandran statistics.")
            corr_cols = [
                c for c in df.columns if "__circular_R" in c or c.endswith("_circular_R")
            ]
            if not corr_cols:
                return error_fig(
                    "No circular statistics columns (__circular_R) present."
                ), html.Div("No circular statistics columns (__circular_R) present.")
            if "variant" in df.columns:
                sub = df[["variant"] + corr_cols]
                if vset:
                    sub = sub[sub["variant"].astype(str).isin(vset)]
            else:
                sub = df[corr_cols].copy()
                sub.insert(0, "variant", "all")

            if sub.empty:
                return error_fig("No Ramachandran features after filtering."), html.Div(
                    "No Ramachandran features after filtering."
                )

            long = sub.melt(
                id_vars="variant", var_name="feature", value_name="R"
            ).dropna()

            def _parse_res_idx(name: str) -> Optional[int]:
                import re

                m = re.search(r"res(\d+)", name)
                if m:
                    try:
                        return int(m.group(1))
                    except Exception:
                        return None
                return None

            long["residx"] = long["feature"].map(_parse_res_idx)
            if long["residx"].notna().any():
                xcol = "residx"
                xlabel = "residue index"
            else:
                cats = {f: i for i, f in enumerate(sorted(long["feature"].unique()))}
                long["residx"] = long["feature"].map(cats)
                xcol = "residx"
                xlabel = "feature (index)"

            fig = px.line(
                long.sort_values([xcol]),
                x=xcol,
                y="R",
                color="variant",
                markers=True,
                hover_data=["feature"],
                title="Ramachandran 1D circular R order parameter",
            )
            fig.update_layout(
                height=700,
                margin=dict(l=60, r=20, t=60, b=60),
                legend=dict(
                    orientation="h",
                    y=1.03,
                    x=1.0,
                    xanchor="right",
                    yanchor="bottom",
                    font=dict(size=9),
                ),
            )
            fig.update_yaxes(range=[0, 1], title="R (0–1)")
            fig.update_xaxes(title=xlabel)

            summary_children = html.Div(
                "Circular R order parameter per feature/residue: "
                "R ≈ 1 means highly localized angles; R ≈ 0 means very broad / disordered.",
            )
            return apply_theme(fig, theme), summary_children

        # --- Ramachandran 2D PMF ---
        if curve_type == "rama2d":
            T = float(Tcurves or 300.0)
            pmfmax = float(pmfmax_curves or 20.0)

            # per-residue density
            if (
                isinstance(rama2d_perres_df, pd.DataFrame)
                and not rama2d_perres_df.empty
                and vset
            ):
                d = rama2d_perres_df.copy()
                if "variant" in d.columns:
                    d = d[d["variant"].isin(vset)]
                if rama_residues:
                    d = d[d["residue"].isin(rama_residues)]
                if d.empty:
                    return error_fig(
                        "No residue-level density for current selection."
                    ), html.Div("No residue-level density for current selection.")

                if "variant" not in d.columns or "residue" not in d.columns:
                    return error_fig(
                        "Rama2D per-residue table missing 'variant' or 'residue'."
                    ), html.Div(
                        "Rama2D per-residue table missing 'variant' or 'residue'."
                    )

                panels = (
                    d[["variant", "residue"]]
                    .drop_duplicates()
                    .head(int(maxpanels or 3))
                )
                n = len(panels)
                cols = min(3, max(1, n))
                rows = int(math.ceil(n / cols))
                titles = [
                    f"Res {residue_display(v, int(r))} — {v}" for v, r in panels.to_records(index=False)
                ]
                fig = make_subplots(
                    rows=rows,
                    cols=cols,
                    subplot_titles=titles,
                )

                pop_rows: List[Dict[str, Any]] = []

                for i, (v, r) in enumerate(panels.to_records(index=False)):
                    sub = d[(d["variant"] == v) & (d["residue"] == r)].copy()
                    if sub.empty:
                        continue
                    for col in ("x", "y", "density"):
                        if col in sub.columns:
                            sub[col] = pd.to_numeric(sub[col], errors="coerce")
                    sub = sub.dropna(subset=["x", "y", "density"])
                    if sub.empty:
                        continue
                    pivot = sub.pivot_table(
                        index="y",
                        columns="x",
                        values="density",
                        aggfunc="sum",
                        fill_value=0.0,
                    )
                    pivot = pivot.sort_index(axis=0).sort_index(axis=1)
                    xs = pivot.columns.to_numpy()
                    ys = pivot.index.to_numpy()
                    Z = pivot.to_numpy()
                    F = density_to_pmf(Z, xs, ys, T_K=T, clip_max=pmfmax)

                    pops = _rama_region_pop(xs, ys, Z)
                    pop_rows.append(
                        {
                            "variant": v,
                            "residue": int(r),
                            **{k: float(vv) for k, vv in pops.items()},
                        }
                    )

                    rr = (i // cols) + 1
                    cc = (i % cols) + 1
                    fig.add_trace(
                        go.Heatmap(
                            x=xs,
                            y=ys,
                            z=F,
                            colorscale="Viridis",
                            zmin=0,
                            zmax=pmfmax,
                            colorbar=dict(title="F (kJ/mol)"),
                        ),
                        row=rr,
                        col=cc,
                    )
                    fig.update_xaxes(
                        title_text="φ (deg)",
                        range=[-180, 180],
                        row=rr,
                        col=cc,
                        constrain="domain",
                    )
                    fig.update_yaxes(
                        title_text="ψ (deg)",
                        range=[-180, 180],
                        row=rr,
                        col=cc,
                        constrain="domain",
                    )
                    txt = (
                        f"α={pops['alpha']:.2f}, β={pops['beta']:.2f}, "
                        f"Lα={pops['left_alpha']:.2f}"
                    )
                    axidx = i + 1
                    # Plotly axis name: 'x' / 'y' for first, 'x2', 'x3', ... for others
                    suffix = "" if axidx == 1 else str(axidx)

                    fig.add_annotation(
                        text=txt,
                        xref=f"x{suffix} domain",
                        yref=f"y{suffix} domain",
                        x=0.5,
                        y=-0.15,
                        showarrow=False,
                        font=dict(size=9),
                    )


                fig.update_layout(
                    height=750,
                    margin=dict(l=60, r=20, t=60, b=80),
                    showlegend=False,
                    title=f"Ramachandran 2D PMF (per-residue) — T={T:.0f} K; Fmax={pmfmax:.0f}",
                )

                if pop_rows:
                    df_pop = pd.DataFrame(pop_rows)
                    for col in ["alpha", "beta", "left_alpha", "other"]:
                        if col in df_pop.columns:
                            df_pop[col] = df_pop[col].astype(float).round(3)
                    header = html.Tr(
                        [
                            html.Th(
                                c,
                                style={
                                    "border": "1px solid #ccc",
                                    "padding": "2px 4px",
                                    "fontWeight": "600",
                                },
                            )
                            for c in df_pop.columns
                        ]
                    )
                    rows_html = []
                    for _, row in df_pop.iterrows():
                        rows_html.append(
                            html.Tr(
                                [
                                    html.Td(
                                        str(row[c]),
                                        style={
                                            "border": "1px solid #eee",
                                            "padding": "2px 4px",
                                        },
                                    )
                                    for c in df_pop.columns
                                ],
                                style={"fontSize": "0.78em"},
                            )
                        )
                    summary_children = html.Div(
                        [
                            html.Div(
                                "Rama2D region populations per (variant, residue).",
                                style={"marginBottom": "2px"},
                            ),
                            html.Table(
                                [header] + rows_html,
                                style={
                                    "width": "100%",
                                    "borderCollapse": "collapse",
                                    "border": "1px solid #ccc",
                                },
                            ),
                        ]
                    )
                else:
                    summary_children = html.Div(
                        "No valid density to compute region populations."
                    )
                return apply_theme(fig, theme), summary_children

            # pooled density
            if not isinstance(rama2d_pooled_df, pd.DataFrame) or rama2d_pooled_df.empty:
                return error_fig(
                    "No pooled Ramachandran 2D density available."
                ), html.Div("No pooled Ramachandran 2D density available.")

            d = rama2d_pooled_df.copy()
            if vset and "variant" in d.columns:
                d = d[d["variant"].isin(vset)]
            if d.empty:
                return error_fig(
                    "No pooled density for selected variants."
                ), html.Div("No pooled density for selected variants.")

            figs = []
            pop_rows: List[Dict[str, Any]] = []
            for var, dd in d.groupby("variant", sort=False, observed=False):
                sub = dd.copy()
                for col in ("x", "y", "density"):
                    if col in sub.columns:
                        sub[col] = pd.to_numeric(sub[col], errors="coerce")
                sub = sub.dropna(subset=["x", "y", "density"])
                if sub.empty:
                    continue
                pivot = sub.pivot_table(
                    index="y",
                    columns="x",
                    values="density",
                    aggfunc="sum",
                    fill_value=0.0,
                )
                pivot = pivot.sort_index(axis=0).sort_index(axis=1)
                xs = pivot.columns.to_numpy()
                ys = pivot.index.to_numpy()
                Z = pivot.to_numpy()
                F = density_to_pmf(Z, xs, ys, T_K=T, clip_max=pmfmax)
                pops = _rama_region_pop(xs, ys, Z)
                pop_rows.append(
                    {"variant": var, **{k: float(vv) for k, vv in pops.items()}}
                )
                fig1 = go.Figure(
                    data=go.Heatmap(
                        x=xs,
                        y=ys,
                        z=F,
                        colorscale="Viridis",
                        zmin=0,
                        zmax=pmfmax,
                        colorbar=dict(title="F (kJ/mol)"),
                    )
                )
                fig1.update_layout(title=f"{var} — φ/ψ PMF (pooled)")
                fig1.update_xaxes(
                    title="φ (deg)",
                    range=[-180, 180],
                    constrain="domain",
                )
                fig1.update_yaxes(
                    title="ψ (deg)",
                    range=[-180, 180],
                    constrain="domain",
                    scaleanchor="x",
                )
                figs.append(fig1)

            if not figs:
                return error_fig(
                    "No valid pooled Ramachandran 2D panels."
                ), html.Div("No valid pooled Ramachandran 2D panels.")

            if len(figs) == 1:
                figs[0].update_layout(
                    height=700,
                    title=f"Ramachandran 2D PMF (pooled) — T={T:.0f} K; Fmax={pmfmax:.0f}",
                )
                if pop_rows:
                    df_pop = pd.DataFrame(pop_rows)
                    for col in ["alpha", "beta", "left_alpha", "other"]:
                        if col in df_pop.columns:
                            df_pop[col] = df_pop[col].astype(float).round(3)
                    header = html.Tr(
                        [
                            html.Th(
                                c,
                                style={
                                    "border": "1px solid #ccc",
                                    "padding": "2px 4px",
                                    "fontWeight": "600",
                                },
                            )
                            for c in df_pop.columns
                        ]
                    )
                    rows_html = []
                    for _, row in df_pop.iterrows():
                        rows_html.append(
                            html.Tr(
                                [
                                    html.Td(
                                        str(row[c]),
                                        style={
                                            "border": "1px solid #eee",
                                            "padding": "2px 4px",
                                        },
                                    )
                                    for c in df_pop.columns
                                ],
                                style={"fontSize": "0.78em"},
                            )
                        )
                    summary_children = html.Div(
                        [
                            html.Div(
                                "Rama2D pooled region populations per variant.",
                                style={"marginBottom": "2px"},
                            ),
                            html.Table(
                                [header] + rows_html,
                                style={
                                    "width": "100%",
                                    "borderCollapse": "collapse",
                                    "border": "1px solid #ccc",
                                },
                            ),
                        ]
                    )
                else:
                    summary_children = html.Div(
                        "No valid density to compute region populations."
                    )
                return apply_theme(figs[0], theme), summary_children

            cols = min(3, len(figs))
            rows = int(math.ceil(len(figs) / cols))
            canvas = make_subplots(
                rows=rows,
                cols=cols,
                subplot_titles=[f.layout.title.text for f in figs],
            )
            for i, fsub in enumerate(figs):
                r = (i // cols) + 1
                c = (i % cols) + 1
                for tr in fsub.data:
                    canvas.add_trace(tr, row=r, col=c)
                canvas.update_xaxes(
                    title_text="φ (deg)",
                    range=[-180, 180],
                    row=r,
                    col=c,
                    constrain="domain",
                )
                canvas.update_yaxes(
                    title_text="ψ (deg)",
                    range=[-180, 180],
                    row=r,
                    col=c,
                    constrain="domain",
                )
            canvas.update_layout(
                height=750,
                showlegend=False,
                title=f"Ramachandran 2D PMF (pooled) — T={T:.0f} K; Fmax={pmfmax:.0f}",
            )

            if pop_rows:
                df_pop = pd.DataFrame(pop_rows)
                for col in ["alpha", "beta", "left_alpha", "other"]:
                    if col in df_pop.columns:
                        df_pop[col] = df_pop[col].astype(float).round(3)
                header = html.Tr(
                    [
                        html.Th(
                            c,
                            style={
                                "border": "1px solid #ccc",
                                "padding": "2px 4px",
                                "fontWeight": "600",
                            },
                        )
                        for c in df_pop.columns
                    ]
                )
                rows_html = []
                for _, row in df_pop.iterrows():
                    rows_html.append(
                        html.Tr(
                            [
                                html.Td(
                                    str(row[c]),
                                    style={
                                        "border": "1px solid #eee",
                                        "padding": "2px 4px",
                                    },
                                )
                                for c in df_pop.columns
                            ],
                            style={"fontSize": "0.78em"},
                        )
                    )
                summary_children = html.Div(
                    [
                        html.Div(
                            "Rama2D pooled region populations per variant.",
                            style={"marginBottom": "2px"},
                        ),
                        html.Table(
                            [header] + rows_html,
                            style={
                                "width": "100%",
                                "borderCollapse": "collapse",
                                "border": "1px solid #ccc",
                            },
                        ),
                    ]
                )
            else:
                summary_children = html.Div(
                    "No valid density to compute region populations."
                )
            return apply_theme(canvas, theme), summary_children

        # fallback
        return apply_theme(error_fig("Unknown curve type or no data available."), theme), html.Div(
            "Unknown curve type or no data available."
        )

    # ---- CSV download of current selection ----
    @app.callback(
        Output("curves-download", "data"),
        Input("curves-download-btn", "n_clicks"),
        State("curve-type", "value"),
        State("curve-metrics", "value"),
        State("curve-variants", "value"),
        prevent_initial_call=True,
    )
    def download_curves(n_clicks, curve_type, metrics_sel, vlist):
        if not n_clicks:
            return no_update
        pmf_df = ctx.pmf_df
        cum_df = ctx.cum_df
        rmsf_df = ctx.rmsf_df
        rama2d_pooled_df = ctx.rama2d_pooled_df
        vset = set(vlist or [])

        if curve_type in ("pmf_F", "pmf_P"):
            if not isinstance(pmf_df, pd.DataFrame) or pmf_df.empty:
                return no_update
            sub = pmf_df.copy()
            if metrics_sel and "metric" in sub.columns:
                sub = sub[sub["metric"].isin(metrics_sel)]
            if vset and "variant" in sub.columns:
                sub = sub[sub["variant"].isin(vset)]
            if sub.empty:
                return no_update
            csv = sub.to_csv(index=False)
            return dict(content=csv, filename="curves_pmf.csv")

        if curve_type in ("cum_FE", "cum_DeltaF"):
            if not isinstance(cum_df, pd.DataFrame) or cum_df.empty:
                return no_update
            sub = cum_df.copy()
            if metrics_sel and "metric" in sub.columns:
                sub = sub[sub["metric"].isin(metrics_sel)]
            if vset and "variant" in sub.columns:
                sub = sub[sub["variant"].isin(vset)]
            if sub.empty:
                return no_update
            csv = sub.to_csv(index=False)
            return dict(content=csv, filename="curves_cumulative.csv")

        if curve_type == "rmsf":
            if not isinstance(rmsf_df, pd.DataFrame) or rmsf_df.empty:
                return no_update
            sub = rmsf_df.copy()
            if vset and "variant" in sub.columns:
                sub = sub[sub["variant"].isin(vset)]
            if "metric" in sub.columns:
                sub = sub[sub["metric"] == "rmsf"]
            if sub.empty:
                return no_update
            csv = sub.to_csv(index=False)
            return dict(content=csv, filename="curves_rmsf.csv")

        if curve_type == "rama2d":
            if not isinstance(rama2d_pooled_df, pd.DataFrame) or rama2d_pooled_df.empty:
                return no_update
            sub = rama2d_pooled_df.copy()
            if vset and "variant" in sub.columns:
                sub = sub[sub["variant"].isin(vset)]
            if sub.empty:
                return no_update
            csv = sub.to_csv(index=False)
            return dict(content=csv, filename="curves_rama2d_pooled.csv")

        if curve_type == "rama":
            if not isinstance(df_features, pd.DataFrame) or df_features.empty:
                return no_update
            cols = [
                c
                for c in df_features.columns
                if "__circular_R" in c or c.endswith("_circular_R")
            ]
            if not cols:
                return no_update
            if "variant" in df_features.columns:
                sub = df_features[["variant"] + cols]
                if vset:
                    sub = sub[sub["variant"].astype(str).isin(vset)]
            else:
                sub = df_features[cols].copy()
            if sub.empty:
                return no_update
            csv = sub.to_csv(index=False)
            return dict(content=csv, filename="curves_rama1d.csv")

        return no_update
