from __future__ import annotations

from ..metrics import metric_display_label, torsion_sort_key, is_torsion_metric

from ..analysis.stats import (
    integrated_autocorr_time,
    effective_sample_size,
    normal_ci_for_prob,
    prob_mass_from_counts,
    js_divergence,
)

import os
import re
from typing import Any, Iterable, List, Tuple, Dict, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
from dash import dcc, html, Input, Output, State, no_update
import plotly.graph_objs as go
from plotly.subplots import make_subplots


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _template(theme: str = "dark") -> str:
    return "plotly_dark" if (theme or "dark") == "dark" else "plotly_white"


def _error_fig(msg: str, theme: str = "dark") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        text=msg,
        showarrow=False,
        xref="paper",
        yref="paper",
        font=dict(size=14),
    )
    fig.update_layout(
        template=_template(theme),
        xaxis={"visible": False},
        yaxis={"visible": False},
        margin=dict(l=40, r=40, t=40, b=40),
    )
    return fig


_metric_sort_key = torsion_sort_key

# -------------------------------------------------------------------
# Filename parsing / discovery
# -------------------------------------------------------------------


_TORSION_PERRES_RE = re.compile(
    r"^(?P<prefix>.+?)(?:__|_)(?P<torsion>phi|psi)(?:[_-](?P<resname>[A-Za-z]{1,6}))?_repl_?(?P<repl>\d+)_(?P<resid>\d+)$",
    flags=re.IGNORECASE,
)


def _xvg_is_multi(path: str) -> bool:
    """
    Heuristic: multi-dataset XVG often uses '&' separators or block info.
    We just peek for '&' to label as 'multi'.
    """
    try:
        with open(path, "rb") as fh:
            for _ in range(200):  # don't read the whole file
                line = fh.readline()
                if not line:
                    break
                if line.startswith(b"&"):
                    return True
    except Exception:
        return False
    return False


def _parse_variant_metric_replica(
    filename_noext: str,
    variant_dir: str | None = None,
) -> Tuple[str, str, int | None]:
    """
    Parse (variant, metric, replica) from an XVG filename (without extension).

    Heuristics tuned for your zoo of files:

    - Any ALL-UPPER alpha token (len >= 5) is treated as a variant code.
      If present, that wins.
    - Else, we use the directory name (VARIANT folder).
    - The last purely numeric token is `replica`.
    - The remaining tokens (excluding the variant) joined by "_" are the metric.
    """
    base = str(filename_noext)
    # ANA torsion-per-residue naming:
    #   SAFEXXX__phi_repl32_10.xvg
    #   SAFEXXX__psi_repl143_9.xvg
    m = _TORSION_PERRES_RE.match(base)
    if m:
        tors = m.group("torsion").lower()
        repl = int(m.group("repl"))
        resid = int(m.group("resid"))
        resname = m.groupdict().get("resname")
        resname = resname.upper() if resname else None
        metric = f"{tors}_{resname}{resid}" if resname else f"{tors}_res{resid}"
        variant = variant_dir or (m.group("prefix") or "unknown")
        return variant, metric, repl

    parts = base.split("_")

    # 1) find possible variant tokens in the name
    upper_tokens = [
        p for p in parts
        if p.isalpha() and p.upper() == p and len(p) >= 5
    ]

    variant: str | None = None
    if upper_tokens:
        variant = upper_tokens[0]
    elif variant_dir is not None:
        variant = variant_dir

    if variant is None and parts:
        # absolute last fallback
        variant = parts[0]

    # strip the variant token from metric parts (only once)
    metric_tokens = parts[:]
    if variant in metric_tokens:
        metric_tokens.remove(variant)

    # replica is last numeric token, if present
    replica = None
    numeric_indices = [i for i, p in enumerate(metric_tokens) if p.isdigit()]
    if numeric_indices:
        idx = numeric_indices[-1]
        try:
            replica = int(metric_tokens[idx])
        except ValueError:
            replica = None
        else:
            metric_tokens.pop(idx)

    metric = "_".join(metric_tokens) if metric_tokens else filename_noext
    return variant, metric, replica


def discover_timeseries_files(
    data_dir: str | None,
    timeseries_dir: str | None = None,
) -> pd.DataFrame:
    """
    Discover timeseries *.xvg files.

    Priority:
      1. If timeseries_dir is provided and exists, treat it as ROOT:
         ROOT/<variant>/*.xvg (or ROOT/*.xvg if ROOT is already a variant dir)
      2. Else, use data_dir/TIMESERIES if it exists.

    Returns
    -------
    DataFrame with columns:
      variant, metric, path, kind, replica
    """
    ts_dir: str | None = None

    if timeseries_dir:
        print(f"[TIMESERIES] requested timeseries_dir={timeseries_dir}")
        if os.path.isdir(timeseries_dir):
            ts_dir = timeseries_dir
            print(f"[TIMESERIES] using explicit timeseries_dir={timeseries_dir}")
        else:
            print(
                f"[TIMESERIES] WARN: timeseries_dir does not exist or is not a directory: {timeseries_dir}"
            )

    if ts_dir is None and data_dir:
        candidate = os.path.join(data_dir, "TIMESERIES")
        print(f"[TIMESERIES] probing fallback {candidate}")
        if os.path.isdir(candidate):
            ts_dir = candidate
            print(f"[TIMESERIES] using {candidate} (data_dir/TIMESERIES)")

    rows: List[Dict[str, Any]] = []

    if ts_dir and os.path.isdir(ts_dir):
        print(f"[TIMESERIES] walking {ts_dir}")
        for root, _, files in os.walk(ts_dir):
            xvg_files = [f for f in files if f.endswith(".xvg")]
            if not xvg_files:
                continue
            print(f"[TIMESERIES] root={root}, xvg_here={len(xvg_files)}")

            for fn in xvg_files:
                path = os.path.join(root, fn)
                rel = os.path.relpath(path, ts_dir)
                parts = rel.split(os.sep)

                # if ROOT/VARIANT/file.xvg → parts[0] is variant folder
                variant_dir = parts[0] if len(parts) >= 2 else None

                # If timeseries_dir points directly to a VARIANT folder, infer variant from the folder name
                if variant_dir is None:
                    base_name = os.path.basename(os.path.normpath(ts_dir))
                    if base_name and base_name not in ("TIMESERIES", "timeseries"):
                        variant_dir = base_name
                base = os.path.splitext(os.path.basename(fn))[0]

                variant, metric, replica = _parse_variant_metric_replica(
                    filename_noext=base,
                    variant_dir=variant_dir,
                )

                kind = "multi" if _xvg_is_multi(path) else "single"

                rows.append(
                    dict(
                        variant=variant,
                        metric=metric,
                        path=path,
                        kind=kind,
                        replica=replica,
                    )
                )

    if not rows:
        print("[TIMESERIES] no XVG found under timeseries_dir/data_dir")
        return pd.DataFrame(
            columns=["variant", "metric", "path", "kind", "replica"]
        )

    df = pd.DataFrame(rows)
    print(
        f"[TIMESERIES] indexed {len(df)} xvg files across "
        f"{df['variant'].nunique()} variants and {df['metric'].nunique()} metrics"
    )
    return df


# -------------------------------------------------------------------
# XVG parsing – IO-level downsampling
# -------------------------------------------------------------------


def _xvg_to_dataframe(path: str, stride: int = 1) -> pd.DataFrame:
    """
    Parse an XVG file into a DataFrame with columns:
      block, time, value

    `stride` is applied at IO level: only every `stride`-th **data** line
    (per block) is parsed and stored.

    Notes
    -----
    - Lines starting with '@' or '#' are treated as header/meta.
    - Lines starting with '&' start a new block (multi-dataset XVG).
    - For data lines, we parse the first two numeric columns as (time, value).
    """
    rows: List[Tuple[int, float, float]] = []
    stride = max(1, int(stride or 1))

    try:
        with open(path, "rb") as fh:
            block_idx = 0
            data_idx_in_block = 0

            for raw in fh:
                if not raw:
                    break

                # Meta / header
                if raw.startswith(b"@") or raw.startswith(b"#"):
                    continue

                # New dataset / block separator
                if raw.startswith(b"&"):
                    block_idx += 1
                    data_idx_in_block = 0
                    continue

                # Strip and skip empty lines early
                s = raw.strip()
                if not s:
                    continue

                # Apply IO-level stride on *data* lines per block,
                # before any expensive parsing
                if stride > 1 and (data_idx_in_block % stride) != 0:
                    data_idx_in_block += 1
                    continue

                parts = s.split()
                if len(parts) < 2:
                    data_idx_in_block += 1
                    continue

                try:
                    t = float(parts[0])
                    y = float(parts[1])
                except ValueError:
                    data_idx_in_block += 1
                    continue

                rows.append((block_idx, t, y))
                data_idx_in_block += 1

    except Exception as exc:
        print(f"[TIMESERIES] WARN: failed to parse {path}: {exc}")
        return pd.DataFrame(columns=["block", "time", "value"])

    if not rows:
        return pd.DataFrame(columns=["block", "time", "value"])

    return pd.DataFrame(rows, columns=["block", "time", "value"])


def iter_xvg_datasets_with_time(
    path: str,
    stride: int = 1,
) -> Iterable[Tuple[np.ndarray, np.ndarray]]:
    """
    Yield (t, y) datasets from an XVG file, respecting 'block' if present.

    Stride is applied at IO level in `_xvg_to_dataframe`.
    """
    df = _xvg_to_dataframe(path, stride=stride)
    if df.empty:
        return
    for _, sub in df.groupby("block"):
        t = sub["time"].to_numpy()
        y = sub["value"].to_numpy()
        yield t, y


def read_xvg_timeseries_with_time(
    path: str,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience wrapper: return (t, y) for the first dataset (block 0).
    """
    df = _xvg_to_dataframe(path, stride=stride)
    if df.empty:
        return np.array([]), np.array([])

    if "block" in df.columns:
        sub = df[df["block"] == 0]
    else:
        sub = df

    t = sub["time"].to_numpy()
    y = sub["value"].to_numpy()
    return t, y


# -------------------------------------------------------------------
# Math helpers
# -------------------------------------------------------------------


def _moving_average(y: np.ndarray, w: int) -> np.ndarray:
    if w is None or w <= 1:
        return y
    w = int(w)
    if w >= len(y):
        return np.full_like(y, float(np.nanmean(y)))
    kern = np.ones(w, dtype=float) / float(w)
    return np.convolve(y, kern, mode="same")


def _compute_acf(
    t: np.ndarray,
    y: np.ndarray,
    max_points: int = 300,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute an autocorrelation function for a single timeseries.

    Returns (lags, acf) in the same time units as `t`.
    """
    mask = np.isfinite(y) & np.isfinite(t)
    t = t[mask]
    y = y[mask]
    if y.size < 3:
        return np.array([]), np.array([])
    order = np.argsort(t)
    t = t[order]
    y = y[order]
    dt = np.median(np.diff(t)) if t.size > 1 else 1.0
    y0 = y - y.mean()
    full = np.correlate(y0, y0, mode="full")
    acf = full[full.size // 2 :]
    if acf[0] == 0:
        return np.array([]), np.array([])
    acf = acf / acf[0]
    if max_points and acf.size > max_points:
        acf = acf[:max_points]
    lags = np.arange(acf.size) * dt
    return lags, acf


def _interp_2d_regular(
    z: np.ndarray,
    x_old: np.ndarray,
    y_old: np.ndarray,
    x_new: np.ndarray,
    y_new: np.ndarray,
) -> np.ndarray:
    """
    Bilinear interpolation on a regular grid.

    z shape: (nx_old, ny_old), with x_old (nx_old,), y_old (ny_old,).
    Returns array of shape (len(x_new), len(y_new)).
    """
    z = np.asarray(z, dtype=float)
    x_old = np.asarray(x_old, dtype=float)
    y_old = np.asarray(y_old, dtype=float)
    x_new = np.asarray(x_new, dtype=float)
    y_new = np.asarray(y_new, dtype=float)

    nx_new = x_new.size
    ny_old = y_old.size
    ny_new = y_new.size

    # First interpolate along x (for each y)
    tmp = np.empty((nx_new, ny_old), dtype=float)
    for j in range(ny_old):
        tmp[:, j] = np.interp(x_new, x_old, z[:, j])

    # Then interpolate along y (for each x)
    out = np.empty((nx_new, ny_new), dtype=float)
    for i in range(nx_new):
        out[i, :] = np.interp(y_new, y_old, tmp[i, :])

    return out


# -------------------------------------------------------------------
# Layout
# -------------------------------------------------------------------


def layout(ctx: Any) -> html.Div:
    """
    Timeseries tab layout.

    Expects DataContext attributes:
      - ctx.timeseries_dir : path to folder that contains <variant>/*.xvg
      - ctx.data_dir       : fallback root for old TIMESERIES search
      - ctx.ts_index_df    : (optional) precomputed index DataFrame

    Discovery is intentionally lazy: the XVG index is built on first entry to
    the tab instead of during app startup.
    """
    metric_opts: List[Dict[str, str]] = []
    variant_opts: List[Dict[str, str]] = []
    metric_val: List[str] = []
    variant_val: List[str] = []

    controls_row1 = html.Div(
        className="row",
        children=[
            html.Div(
                className="three columns",
                children=[
                    html.Label("Metric(s)"),
                    dcc.Dropdown(
                        id="ts-metrics",
                        multi=True,
                        options=metric_opts,
                        value=metric_val,
                    ),
                ],
            ),
            html.Div(
                className="three columns",
                children=[
                    html.Label("Variant(s)"),
                    dcc.Dropdown(
                        id="ts-variants",
                        multi=True,
                        options=variant_opts,
                        value=variant_val,
                    ),
                ],
            ),
            html.Div(
                className="two columns",
                children=[
                    html.Label("Downsample (stride)"),
                    dcc.Input(
                        id="ts-stride",
                        type="number",
                        min=1,
                        step=1,
                        value=10,
                    ),
                ],
            ),
            html.Div(
                className="two columns",
                children=[
                    html.Label("Smooth (moving average window)"),
                    dcc.Input(
                        id="ts-smooth",
                        type="number",
                        min=0,
                        step=1,
                        value=0,
                    ),
                ],
            ),
            html.Div(
                className="two columns",
                children=[
                    html.Label("Layout"),
                    dcc.Dropdown(
                        id="ts-layout",
                        clearable=False,
                        value="facet_metric",
                        options=[
                            {"label": "Facet by metric", "value": "facet_metric"},
                            {"label": "Facet by variant", "value": "facet_variant"},
                            {"label": "Overlay", "value": "overlay"},
                        ],
                    ),
                ],
            ),
        ],
    )

    controls_row2 = html.Div(
        className="row",
        children=[
            html.Div(
                className="four columns",
                children=[
                    html.Label("Options"),
                    dcc.Checklist(
                        id="ts-opts",
                        options=[
                            {"label": "Include multi-block XVGs", "value": "multi"},
                            {"label": "Show replicas separately", "value": "replicas"},
                        ],
                        value=[],
                        labelStyle={"display": "block"},
                    ),
                    dcc.Checklist(
                        id="ts-logy",
                        options=[{"label": "Log Y (timeseries only)", "value": "log"}],
                        value=[],
                        style={"marginTop": "0.5rem"},
                    ),
                    html.Label("Distribution / PMF", style={"marginTop": "0.75rem"}),
                    dcc.Dropdown(
                        id="ts-dist-mode",
                        clearable=False,
                        value="none",
                        options=[
                            {"label": "Off", "value": "none"},
                            {"label": "Probability density", "value": "pdf"},
                            {"label": "PMF (-ln P)", "value": "pmf"},
                        ],
                        style={"marginTop": "0.25rem"},
                    ),
                    html.Div(
                        children=[
                            html.Label(
                                "Histogram bins", style={"marginTop": "0.5rem"}
                            ),
                            dcc.Input(
                                id="ts-dist-bins",
                                type="number",
                                min=5,
                                step=1,
                                value=50,
                                style={"width": "6rem", "marginLeft": "0.5rem"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),

                    html.Div(
                        children=[
                            html.Label("PMF units", style={"marginTop": "0.5rem"}),
                            dcc.Dropdown(
                                id="ts-pmf-units",
                                clearable=False,
                                value="kT",
                                options=[
                                    {"label": "kT (dimensionless)", "value": "kT"},
                                    {"label": "kJ/mol", "value": "kJmol"},
                                ],
                                style={"marginTop": "0.25rem"},
                            ),
                        ],
                    ),
                    html.Div(
                        children=[
                            html.Label("Temperature (K)", style={"marginTop": "0.5rem"}),
                            dcc.Input(
                                id="ts-tempK",
                                type="number",
                                min=1,
                                step=1,
                                value=300,
                                style={"width": "6rem", "marginLeft": "0.5rem"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    html.Div(
                        children=[
                            html.Label("PMF pseudocount α", style={"marginTop": "0.5rem"}),
                            dcc.Input(
                                id="ts-pmf-alpha",
                                type="number",
                                min=0,
                                step=0.1,
                                value=0.5,
                                style={"width": "6rem", "marginLeft": "0.5rem"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    dcc.Checklist(
                        id="ts-dist-ci",
                        options=[{"label": "Show approx. 95% CI (ESS-aware)", "value": "ci"}],
                        value=[],
                        labelStyle={"display": "block"},
                        style={"marginTop": "0.5rem"},
                    ),
                    dcc.Checklist(
                        id="ts-torsion-wrap",
                        options=[{"label": "Wrap torsions to (-180, 180]", "value": "wrap"}],
                        value=["wrap"],
                        labelStyle={"display": "block"},
                        style={"marginTop": "0.25rem"},
                    ),
                    html.Label("2D PMF mode", style={"marginTop": "0.75rem"}),
                    dcc.Dropdown(
                        id="ts-2dpmf-mode",
                        clearable=False,
                        value="all",
                        options=[
                            {"label": "Aggregate all variants", "value": "all"},
                            {"label": "Per-variant subplots", "value": "per_variant"},
                        ],
                        style={"marginTop": "0.25rem"},
                    ),
                    html.Div(
                        children=[
                            dcc.Checklist(
                                id="ts-2dpmf-interp",
                                options=[
                                    {
                                        "label": "Interpolate 2D PMF",
                                        "value": "interp",
                                    }
                                ],
                                value=[],
                                style={"marginTop": "0.5rem"},
                            )
                        ]
                    ),
                    html.Div(
                        children=[
                            html.Label(
                                "2D PMF grid N", style={"marginTop": "0.5rem"}
                            ),
                            dcc.Input(
                                id="ts-2dpmf-n",
                                type="number",
                                min=20,
                                step=10,
                                value=80,
                                style={"width": "6rem", "marginLeft": "0.5rem"},
                            ),
                        ],
                        style={"display": "flex", "alignItems": "center"},
                    ),
                    html.Div(
                        children=[
                            html.Small(
                                "2D PMF uses first two selected metrics.",
                                style={"display": "block", "marginTop": "0.5rem"},
                            )
                        ]
                    ),
                ],
            ),
            html.Div(
                className="eight columns",
                children=[
                    html.Label("Axis limits"),
                    html.Div(
                        className="row",
                        children=[
                            html.Div(
                                className="six columns",
                                children=[
                                    dcc.Input(
                                        id="ts-xmin",
                                        type="number",
                                        step="any",
                                        placeholder="xmin",
                                    ),
                                    dcc.Input(
                                        id="ts-xmax",
                                        type="number",
                                        step="any",
                                        placeholder="xmax",
                                    ),
                                ],
                            ),
                            html.Div(
                                className="six columns",
                                children=[
                                    dcc.Input(
                                        id="ts-ymin",
                                        type="number",
                                        step="any",
                                        placeholder="ymin",
                                    ),
                                    dcc.Input(
                                        id="ts-ymax",
                                        type="number",
                                        step="any",
                                        placeholder="ymax",
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            ),
        ],
    )

    graphs_row = html.Div(
        className="row",
        children=[
            html.Div(
                className="eight columns",
                children=[
                    dcc.Graph(
                        id="timeseries-graph",
                        figure=_error_fig(
                            "Select at least one metric and variant.", theme="dark"
                        ),
                    )
                ],
            ),
            html.Div(
                className="four columns",
                children=[
                    dcc.Graph(
                        id="timeseries-acf-graph",
                        figure=_error_fig(
                            "Autocorrelation will appear here.", theme="dark"
                        ),
                    ),
                    dcc.Graph(
                        id="timeseries-pmf-graph",
                        figure=_error_fig(
                            "Distribution / PMF will appear here.", theme="dark"
                        ),
                        style={"marginTop": "1rem"},
                    ),
                    html.Div(
                        id="timeseries-stats",
                        style={
                            "marginTop": "0.5rem",
                            "fontSize": "0.75em",
                            "opacity": 0.85,
                            "lineHeight": "1.3",
                        },
                    ),
                    dcc.Graph(
                        id="timeseries-2dpmf-graph",
                        figure=_error_fig(
                            "2D PMF will appear here (needs PMF mode and ≥2 metrics).",
                            theme="dark",
                        ),
                        style={"marginTop": "1rem"},
                    ),
                ],
            ),
        ],
    )

    return html.Div([dcc.Store(id="ts-index-store", data=None), controls_row1, html.Hr(), controls_row2, html.Hr(), graphs_row])


# -------------------------------------------------------------------
# Callbacks
# -------------------------------------------------------------------


def register_callbacks(app, ctx: Any) -> None:
    """
    Register Dash callbacks for the timeseries tab.
    """
    theme_default = "dark"

    def _ts_index_records() -> List[Dict[str, Any]]:
        ts_index_df: pd.DataFrame | None = getattr(ctx, "ts_index_df", None)
        if ts_index_df is None:
            timeseries_dir = getattr(ctx, "timeseries_dir", None)
            data_dir = getattr(ctx, "data_dir", None)
            ts_index_df = discover_timeseries_files(data_dir, timeseries_dir)
            setattr(ctx, "ts_index_df", ts_index_df)
        if ts_index_df is None or ts_index_df.empty:
            return []
        return ts_index_df.to_dict("records")

    def _ts_index_from_store(data: List[Dict[str, Any]] | None) -> pd.DataFrame:
        if not data:
            return pd.DataFrame(columns=["variant", "metric", "path", "kind", "replica"])
        return pd.DataFrame(data)

    @app.callback(
        Output("ts-index-store", "data"),
        Input("tabs", "value"),
        State("ts-index-store", "data"),
        prevent_initial_call=False,
    )
    def _ensure_ts_index(active_tab, current_data):
        if active_tab != "timeseries":
            return no_update
        if current_data:
            return no_update
        return _ts_index_records()

    @app.callback(
        Output("ts-variants", "options"),
        Output("ts-variants", "value"),
        Output("ts-metrics", "options"),
        Output("ts-metrics", "value"),
        Input("tabs", "value"),
        Input("ts-index-store", "data"),
        Input("ts-metrics", "value"),
        Input("ts-variants", "value"),
        prevent_initial_call=False,
    )
    def _ts_populate(active_tab, ts_index_data, metric_sel, variant_sel):
        if active_tab != "timeseries":
            return no_update, no_update, no_update, no_update

        ts_index_df = _ts_index_from_store(ts_index_data)
        if ts_index_df.empty:
            return [], [], [], []

        all_metrics = sorted(ts_index_df["metric"].dropna().unique().tolist(), key=_metric_sort_key)
        all_variants = sorted(ts_index_df["variant"].dropna().unique().tolist())

        metric_sel = [m for m in (metric_sel or []) if m in all_metrics] or (
            all_metrics[:1] if all_metrics else []
        )

        if metric_sel:
            v_uni = sorted(
                ts_index_df[ts_index_df["metric"].isin(metric_sel)][
                    "variant"
                ].unique()
            )
        else:
            v_uni = all_variants

        var_opts = [{"label": v, "value": v} for v in v_uni]
        variant_sel = [v for v in (variant_sel or []) if v in v_uni] or v_uni[
            : min(2, len(v_uni))
        ]

        met_opts = [{"label": metric_display_label(m), "value": m} for m in all_metrics]

        return var_opts, variant_sel, met_opts, metric_sel

    @app.callback(
        Output("timeseries-graph", "figure"),
        Output("timeseries-acf-graph", "figure"),
        Output("timeseries-pmf-graph", "figure"),
        Output("timeseries-stats", "children"),
        Output("timeseries-2dpmf-graph", "figure"),
        Input("tabs", "value"),
        Input("ts-index-store", "data"),
        Input("ts-metrics", "value"),
        Input("ts-variants", "value"),
        Input("ts-stride", "value"),
        Input("ts-smooth", "value"),
        Input("ts-opts", "value"),
        Input("ts-logy", "value"),
        Input("ts-layout", "value"),
        Input("ts-xmin", "value"),
        Input("ts-xmax", "value"),
        Input("ts-ymin", "value"),
        Input("ts-ymax", "value"),
        Input("ts-dist-mode", "value"),
        Input("ts-dist-bins", "value"),
        Input("ts-pmf-units", "value"),
        Input("ts-tempK", "value"),
        Input("ts-pmf-alpha", "value"),
        Input("ts-dist-ci", "value"),
        Input("ts-torsion-wrap", "value"),
        Input("ts-2dpmf-mode", "value"),
        Input("ts-2dpmf-interp", "value"),
        Input("ts-2dpmf-n", "value"),
        Input("theme-store", "data"),
        prevent_initial_call=False,
    )
    def _update_curves(
        active_tab,
        ts_index_data,
        metrics,
        variants_,
        stride,
        smooth,
        opts,
        logy,
        layout_mode,
        xmin,
        xmax,
        ymin,
        ymax,
        dist_mode,
        dist_bins,
        pmf_units,
        temp_K,
        pmf_alpha,
        dist_ci_flags,
        torsion_wrap_flags,
        pmf2d_mode,
        interp_flags,
        n2d_grid,
        theme_input,
    ):
        theme = theme_input or theme_default

        stats_children = html.Div()

        if active_tab != "timeseries":
            fig = _error_fig(
                "Open the Timeseries tab to build the XVG index and render plots.",
                theme=theme,
            )
            err_acf = _error_fig("Autocorrelation will appear here.", theme=theme)
            err_pmf = _error_fig("Distribution / PMF will appear here.", theme=theme)
            err_2d = _error_fig(
                "2D PMF will appear here (needs PMF mode and ≥2 metrics).",
                theme=theme,
            )
            return fig, err_acf, err_pmf, stats_children, err_2d

        ts_index_df = _ts_index_from_store(ts_index_data)
        if ts_index_df.empty:
            fig = _error_fig(
                "No timeseries index found. Check --timeseries-dir or data_dir/TIMESERIES.",
                theme=theme,
            )
            err_acf = _error_fig("ACF not available.", theme=theme)
            err_pmf = _error_fig("Distribution / PMF not available.", theme=theme)
            err_2d = _error_fig("2D PMF not available (no index).", theme=theme)
            return fig, err_acf, err_pmf, stats_children, err_2d

        metrics = [m for m in (metrics or [])]
        variants_ = [v for v in (variants_ or [])]
        if not metrics or not variants_:
            fig = _error_fig("Pick at least one metric and one variant.", theme=theme)
            err_acf = _error_fig("ACF not available.", theme=theme)
            err_pmf = _error_fig("Distribution / PMF not available.", theme=theme)
            err_2d = _error_fig(
                "2D PMF not available (need ≥1 metric and variant).", theme=theme
            )
            return fig, err_acf, err_pmf, stats_children, err_2d

        show_replicas = "replicas" in (opts or [])
        stride = max(1, int(stride or 1))  # IO-level stride
        smooth = int(smooth or 0)
        dist_mode = dist_mode or "none"
        pmf2d_mode = pmf2d_mode or "all"
        interp2d = "interp" in (interp_flags or [])

        try:
            bins = int(dist_bins or 50)
        except Exception:
            bins = 50
        bins = max(5, bins)

        try:
            n2d = int(n2d_grid or 80)
        except Exception:
            n2d = 80
        n2d = max(20, n2d)

        idx = ts_index_df[
            ts_index_df["metric"].isin(metrics)
            & ts_index_df["variant"].isin(variants_)
        ].copy()

        if idx.empty:
            fig = _error_fig(
                "No matching *.xvg files for this selection.", theme=theme
            )
            err_acf = _error_fig("ACF not available.", theme=theme)
            err_pmf = _error_fig("Distribution / PMF not available.", theme=theme)
            err_2d = _error_fig(
                "2D PMF not available (no curves).", theme=theme
            )
            return fig, err_acf, err_pmf, stats_children, err_2d

        # On-demand parallel loading
        records = idx.to_dict("records")

        def _load_one(row: dict) -> dict:
            path = row["path"]
            kind = row.get("kind", "single")
            try:
                if kind == "multi":
                    datasets = list(iter_xvg_datasets_with_time(path, stride=stride))
                    # just take first dataset for now
                    if not datasets:
                        t, y = np.array([]), np.array([])
                    else:
                        t, y = datasets[0]
                else:
                    t, y = read_xvg_timeseries_with_time(path, stride=stride)
            except Exception as exc:
                print(f"[TIMESERIES] error reading {path}: {exc}")
                t, y = np.array([]), np.array([])

            return {
                "variant": row["variant"],
                "metric": row["metric"],
                "replica": row.get("replica"),
                "path": path,
                "t": t,
                "y": y,
            }

        loaded: List[Dict[str, Any]] = []
        max_workers = min(8, len(records)) if records else 1
        if max_workers < 1:
            max_workers = 1

        with ThreadPoolExecutor(max_workers=max_workers) as ex:
            futures = [ex.submit(_load_one, r) for r in records]
            for fut in as_completed(futures):
                loaded.append(fut.result())

        loaded = [d for d in loaded if d["y"].size > 0]

        if not loaded:
            fig = _error_fig(
                "Failed to read any timeseries for this selection.", theme=theme
            )
            err_acf = _error_fig("ACF not available.", theme=theme)
            err_pmf = _error_fig("Distribution / PMF not available.", theme=theme)
            err_2d = _error_fig(
                "2D PMF not available (no usable curves).", theme=theme
            )
            return fig, err_acf, err_pmf, stats_children, err_2d

        # Main figure layout
        layout_mode = layout_mode or "facet_metric"
        if layout_mode == "facet_metric":
            keys = metrics
        elif layout_mode == "facet_variant":
            keys = variants_
        else:
            keys = ["overlay"]

        n = max(1, len(keys))
        fig = make_subplots(
            rows=n,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.02,
        )

        def panel_rc(metric_name: str, variant_name: str) -> Tuple[int, int]:
            if layout_mode == "facet_metric":
                try:
                    r = keys.index(metric_name) + 1
                except ValueError:
                    r = 1
                return r, 1
            elif layout_mode == "facet_variant":
                try:
                    r = keys.index(variant_name) + 1
                except ValueError:
                    r = 1
                return r, 1
            else:
                return 1, 1

        # Add timeseries traces
        for d in loaded:
            v = d["variant"]
            m = d["metric"]
            replica = d["replica"]
            path = d["path"]
            t = d["t"]
            y = d["y"]

            if y.size == 0:
                continue

            y_plot = y
            if smooth and smooth > 1:
                y_plot = _moving_average(y_plot, int(smooth))

            r, c = panel_rc(m, v)

            if layout_mode == "facet_metric":
                if show_replicas and replica is not None:
                    name = f"{v} (rep {replica})"
                else:
                    name = f"{v}"
            elif layout_mode == "facet_variant":
                if show_replicas and replica is not None:
                    name = f"{metric_display_label(m, variant=v)} (rep {replica})"
                else:
                    name = f"{metric_display_label(m, variant=v)}"
            else:
                base = os.path.basename(path)
                if show_replicas and replica is not None:
                    name = f"{v}/{metric_display_label(m, variant=v)} (rep {replica})"
                elif show_replicas:
                    name = f"{v}/{base}"
                else:
                    name = f"{v}/{metric_display_label(m, variant=v)}"

            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=y_plot,
                    mode="lines",
                    name=name,
                ),
                row=r,
                col=c,
            )

        fig.update_layout(
            template=_template(theme),
            hovermode="x",
            legend=dict(
                orientation="h",
                y=1.02,
                x=1.0,
                xanchor="right",
                yanchor="bottom",
            ),
            margin=dict(l=60, r=20, t=40, b=60),
            showlegend=True,
        )

        if "log" in (logy or []):
            fig.update_yaxes(type="log")
        else:
            fig.update_yaxes(type="linear")

        if xmin is not None or xmax is not None:
            fig.update_xaxes(range=[xmin, xmax])
        if ymin is not None or ymax is not None:
            fig.update_yaxes(range=[ymin, ymax])

        # ---------------------------------------------------------------
        # ACF figure, reusing loaded data
        # ---------------------------------------------------------------
        ess_map: Dict[Tuple[str, str, Optional[str]], Dict[str, float]] = {}
        acf_fig = go.Figure()
        for d in loaded:
            v = d["variant"]
            m = d["metric"]
            replica = d["replica"]
            t = d["t"]
            y = d["y"]

            if y.size < 4:
                continue

            y_acf = y
            if smooth and smooth > 1:
                y_acf = _moving_average(y_acf, int(smooth))

            lags, acf = _compute_acf(t, y_acf)
            if lags.size == 0:
                continue

            tau = integrated_autocorr_time(acf)
            ess = effective_sample_size(int(y_acf.size), tau)
            key = (str(v), str(m), str(replica) if replica is not None else None)
            ess_map[key] = {"tau_int": float(tau), "ess": float(ess), "n": float(y_acf.size)}

            if show_replicas and replica is not None:
                name = f"{v}/{metric_display_label(m, variant=v)} (rep {replica})"
            else:
                name = f"{v}/{metric_display_label(m, variant=v)}"

            acf_fig.add_trace(
                go.Scatter(
                    x=lags,
                    y=acf,
                    mode="lines",
                    name=name,
                )
            )

        if not acf_fig.data:
            acf_fig = _error_fig(
                "ACF not available (timeseries too short or constant).", theme=theme
            )
        else:
            acf_fig.update_layout(
                template=_template(theme),
                xaxis_title="Lag",
                yaxis_title="Autocorrelation",
                hovermode="x",
                legend=dict(
                    orientation="h",
                    y=1.03,
                    x=1.0,
                    xanchor="right",
                    yanchor="bottom",
                ),
                margin=dict(l=60, r=20, t=60, b=60),
                title="Autocorrelation of shown curves",
            )
            acf_fig.update_yaxes(range=[-0.2, 1.05])

        # ---------------------------------------------------------------
        # 1D Distribution / PMF figure
        # ---------------------------------------------------------------
        pmf_fig = go.Figure()

        if dist_mode == "none":
            pmf_fig = _error_fig(
                "Distribution / PMF disabled. Select a mode in controls.",
                theme=theme,
            )
        else:
            for d in loaded:
                v = d["variant"]
                m = d["metric"]
                replica = d["replica"]
                y = d["y"]

                vals = y[np.isfinite(y)]
                if vals.size < 8:
                    continue

                y_dist = vals
                if smooth and smooth > 1:
                    y_dist = _moving_average(y_dist, int(smooth))
                    y_dist = y_dist[np.isfinite(y_dist)]

                if y_dist.size < 8:
                    continue

                y_lo = None
                y_hi = None
                show_ci = "ci" in (dist_ci_flags or [])
                wrap_torsion = ("wrap" in (torsion_wrap_flags or [])) and is_torsion_metric(m)
                key = (str(v), str(m), str(replica) if replica is not None else None)

                y_use = y_dist
                hist_range = None
                if wrap_torsion:
                    y_use = ((y_use + 180.0) % 360.0) - 180.0
                    hist_range = (-180.0, 180.0)

                # Circular summary (torsions)
                if wrap_torsion and y_use.size >= 3:
                    ang = np.deg2rad(y_use)
                    C = float(np.nanmean(np.cos(ang)))
                    S = float(np.nanmean(np.sin(ang)))
                    R = float(np.sqrt(C * C + S * S))
                    mean_deg = float((np.rad2deg(np.arctan2(S, C)) + 360.0) % 360.0)
                    info = ess_map.get(key, {})
                    info.update({"circ_mean_deg": mean_deg, "circ_var": float(1.0 - R)})
                    ess_map[key] = info

                counts, edges = np.histogram(y_use, bins=bins, range=hist_range, density=False)
                centers = 0.5 * (edges[:-1] + edges[1:])
                dx = np.diff(edges)
                dx = np.where(dx == 0, 1.0, dx)

                p_mass = prob_mass_from_counts(counts, alpha=float(pmf_alpha or 0.0))
                n_eff = np.nan
                if key in ess_map:
                    n_eff = float(ess_map[key]["ess"])
                if not np.isfinite(n_eff) or n_eff <= 1.0:
                    n_eff = float(max(np.sum(counts), 1.0))

                if dist_mode == "pdf":
                    y_plot = p_mass / dx
                    y_label = "Probability density"
                    title = "Distribution of observable values"
                    if show_ci:
                        p_lo, p_hi = normal_ci_for_prob(p_mass, n_eff)
                        y_lo = p_lo / dx
                        y_hi = p_hi / dx
                else:  # "pmf"
                    eps = 1e-300
                    with np.errstate(divide="ignore", invalid="ignore"):
                        F = -np.log(np.clip(p_mass, eps, 1.0))
                    unit_mode = str(pmf_units or "kT")
                    if unit_mode == "kJmol":
                        T = float(temp_K or 300.0)
                        F = F * (8.314462618e-3 * T)  # R*T in kJ/mol
                        y_label = "PMF (kJ/mol)"
                    else:
                        y_label = "PMF (kT)"
                    finite = np.isfinite(F)
                    shift = float(np.nanmin(F[finite])) if finite.any() else 0.0
                    F = F - shift
                    y_plot = F
                    title = "PMF from time-distribution"
                    if show_ci:
                        p_lo, p_hi = normal_ci_for_prob(p_mass, n_eff)
                        with np.errstate(divide="ignore", invalid="ignore"):
                            F_lo = -np.log(np.clip(p_hi, eps, 1.0))  # more probable => lower F
                            F_hi = -np.log(np.clip(p_lo, eps, 1.0))
                        if unit_mode == "kJmol":
                            F_lo = F_lo * (8.314462618e-3 * T)
                            F_hi = F_hi * (8.314462618e-3 * T)
                        F_lo = F_lo - shift
                        F_hi = F_hi - shift
                        y_lo = F_lo
                        y_hi = F_hi

                if show_replicas and replica is not None:
                    name = f"{v}/{metric_display_label(m, variant=v)} (rep {replica})"
                else:
                    name = f"{v}/{metric_display_label(m, variant=v)}"

                pmf_fig.add_trace(
                    go.Scatter(
                        x=centers,
                        y=y_plot,
                        mode="lines",
                        name=name,
                    )
                )

                if dist_mode in {"pdf", "pmf"} and ("ci" in (dist_ci_flags or [])) and (y_lo is not None) and (y_hi is not None):
                    pmf_fig.add_trace(
                        go.Scatter(
                            x=centers,
                            y=y_hi,
                            mode="lines",
                            line=dict(width=0),
                            showlegend=False,
                            hoverinfo="skip",
                            opacity=0.2,
                        )
                    )
                    pmf_fig.add_trace(
                        go.Scatter(
                            x=centers,
                            y=y_lo,
                            mode="lines",
                            line=dict(width=0),
                            fill="tonexty",
                            fillcolor="rgba(0,0,0,0.15)",
                            showlegend=False,
                            hoverinfo="skip",
                        )
                    )

            if not pmf_fig.data:
                pmf_fig = _error_fig(
                    "Distribution / PMF not available (curves too short or constant).",
                    theme=theme,
                )
            else:
                pmf_fig.update_layout(
                    template=_template(theme),
                    xaxis_title="Observable value",
                    yaxis_title=y_label,
                    hovermode="x",
                    legend=dict(
                        orientation="h",
                        y=1.03,
                        x=1.0,
                        xanchor="right",
                        yanchor="bottom",
                    ),
                    margin=dict(l=60, r=20, t=60, b=60),
                    title=title,
                )

        # ---------------------------------------------------------------
        # 2D PMF figure (heatmaps from two timeseries)
        # ---------------------------------------------------------------
        pmf2d_fig = go.Figure()

        if dist_mode != "pmf" or len(metrics) < 2:
            pmf2d_fig = _error_fig(
                "2D PMF requires PMF mode and at least two selected metrics.",
                theme=theme,
            )
        else:
            metric_x = metrics[0]
            metric_y = metrics[1]

            # Index loaded curves by (variant, replica, metric)
            by_key: Dict[Tuple[str, int | None, str], Dict[str, Any]] = {}
            for d in loaded:
                key = (d["variant"], d.get("replica"), d["metric"])
                by_key[key] = d

            xs_all: List[np.ndarray] = []
            ys_all: List[np.ndarray] = []
            per_variant_samples: Dict[str, Tuple[List[np.ndarray], List[np.ndarray]]] = {}

            for v in variants_:
                v_xs: List[np.ndarray] = []
                v_ys: List[np.ndarray] = []

                # collect replicas for which we have both metrics
                replica_ids = set()
                for k in by_key.keys():
                    var_k, rep_k, met_k = k
                    if var_k == v and met_k in (metric_x, metric_y):
                        replica_ids.add(rep_k)

                for rep in replica_ids:
                    dx = by_key.get((v, rep, metric_x))
                    dy = by_key.get((v, rep, metric_y))
                    if dx is None or dy is None:
                        continue

                    yx = dx["y"]
                    yy = dy["y"]

                    if yx.size < 8 or yy.size < 8:
                        continue

                    # align by index (assumes same stride and sampling)
                    n_min = min(len(yx), len(yy))
                    if n_min < 8:
                        continue

                    yx = yx[:n_min]
                    yy = yy[:n_min]

                    if smooth and smooth > 1:
                        yx = _moving_average(yx, int(smooth))
                        yy = _moving_average(yy, int(smooth))

                    mask = np.isfinite(yx) & np.isfinite(yy)
                    yx = yx[mask]
                    yy = yy[mask]

                    if yx.size < 8:
                        continue

                    v_xs.append(yx)
                    v_ys.append(yy)
                    xs_all.append(yx)
                    ys_all.append(yy)

                if v_xs:
                    per_variant_samples[v] = (v_xs, v_ys)

            if not xs_all:
                pmf2d_fig = _error_fig(
                    "2D PMF not available (no overlapping curves with both metrics).",
                    theme=theme,
                )
            else:
                # Global samples for edges (shared grid for all variants)
                X_all = np.concatenate(xs_all)
                Y_all = np.concatenate(ys_all)

                if X_all.size < 16:
                    pmf2d_fig = _error_fig(
                        "2D PMF not available (too few paired samples).",
                        theme=theme,
                    )
                    return fig, acf_fig, pmf_fig, stats_children, pmf2d_fig
            

                # Optional torsion wrapping for 2D PMF (avoid seam artifacts)
                wrap_x = ("wrap" in (torsion_wrap_flags or [])) and is_torsion_metric(metric_x)
                wrap_y = ("wrap" in (torsion_wrap_flags or [])) and is_torsion_metric(metric_y)
                range2d = None
                if wrap_x:
                    X_all = ((X_all + 180.0) % 360.0) - 180.0
                if wrap_y:
                    Y_all = ((Y_all + 180.0) % 360.0) - 180.0
                if wrap_x or wrap_y:
                    rx = (-180.0, 180.0) if wrap_x else (float(np.nanmin(X_all)), float(np.nanmax(X_all)))
                    ry = (-180.0, 180.0) if wrap_y else (float(np.nanmin(Y_all)), float(np.nanmax(Y_all)))
                    range2d = [rx, ry]

                # Compute global edges
                hist_all, xedges, yedges = np.histogram2d(
                    X_all, Y_all, bins=bins, range=range2d, density=False
                )

                entries: List[Dict[str, Any]] = []

                if pmf2d_mode == "all":
                    entries.append(
                        dict(
                            label="All variants",
                            hist2d=hist_all,
                            xedges=xedges,
                            yedges=yedges,
                        )
                    )
                else:  # per_variant
                    for v in variants_:
                        if v not in per_variant_samples:
                            continue
                        v_xs, v_ys = per_variant_samples[v]
                        X_v = np.concatenate(v_xs)
                        Y_v = np.concatenate(v_ys)
                        if X_v.size < 16:
                            continue
                        if wrap_x:
                            X_v = ((X_v + 180.0) % 360.0) - 180.0
                        if wrap_y:
                            Y_v = ((Y_v + 180.0) % 360.0) - 180.0
                        hist_v, _, _ = np.histogram2d(
                            X_v, Y_v, bins=[xedges, yedges], density=False
                        )
                        entries.append(
                            dict(
                                label=v,
                                hist2d=hist_v,
                                xedges=xedges,
                                yedges=yedges,
                            )
                        )

                if not entries:
                    pmf2d_fig = _error_fig(
                        "2D PMF not available (no variant had enough paired samples).",
                        theme=theme,
                    )
                else:
                    # Build subplot figure (even if only one entry)
                    n_rows = len(entries)
                    pmf2d_fig = make_subplots(
                        rows=n_rows,
                        cols=1,
                        shared_xaxes=False,
                        shared_yaxes=False,
                        subplot_titles=[e["label"] for e in entries],
                        vertical_spacing=0.08,
                    )

                    for i, ent in enumerate(entries, start=1):
                        hist2d = ent["hist2d"]
                        xedges = ent["xedges"]
                        yedges = ent["yedges"]

                        # Convert counts -> probability mass; then F = -ln P (optionally scaled)
                        P2 = np.asarray(hist2d, dtype=float)
                        alpha2 = float(pmf_alpha or 0.0)
                        if alpha2 > 0.0:
                            P2 = P2 + alpha2
                        denom2 = float(np.sum(P2))
                        if denom2 <= 0.0:
                            continue
                        P2 = P2 / denom2

                        eps = 1e-300
                        with np.errstate(divide="ignore", invalid="ignore"):
                            pmf2 = -np.log(np.clip(P2, eps, 1.0))

                        unit_mode = str(pmf_units or "kT")
                        if unit_mode == "kJmol":
                            T = float(temp_K or 300.0)
                            pmf2 = pmf2 * (8.314462618e-3 * T)

                        finite = np.isfinite(pmf2)
                        if finite.any():
                            pmf2[finite] -= np.nanmin(pmf2[finite])
                            # Fill non-finite with a large positive value
                            fill_val = np.nanmax(pmf2[finite]) + 5.0
                            pmf2[~finite] = fill_val
                        else:
                            # This particular entry has no finite probabilities; skip it
                            continue

                        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
                        ycenters = 0.5 * (yedges[:-1] + yedges[1:])

                        # Optional interpolation
                        if interp2d:
                            xnew = np.linspace(xcenters[0], xcenters[-1], n2d)
                            ynew = np.linspace(ycenters[0], ycenters[-1], n2d)
                            pmf2_interp = _interp_2d_regular(
                                pmf2, xcenters, ycenters, xnew, ynew
                            )
                            z_plot = pmf2_interp
                            x_plot = xnew
                            y_plot = ynew
                        else:
                            z_plot = pmf2
                            x_plot = xcenters
                            y_plot = ycenters

                        pmf2d_fig.add_trace(
                            go.Heatmap(
                                x=x_plot,
                                y=y_plot,
                                z=z_plot.T,  # transpose so axes align visually
                                colorbar=dict(
                                    title="PMF (arb.)",
                                ),
                            ),
                            row=i,
                            col=1,
                        )

                    pmf2d_fig.update_layout(
                        template=_template(theme),
                        margin=dict(l=60, r=20, t=60, b=60),
                        title=(
                            f"2D PMF: {metric_display_label(metric_x)} vs {metric_display_label(metric_y)} "
                            + (
                                "(all variants)"
                                if pmf2d_mode == "all"
                                else "(per variant)"
                            )
                            + (" (interpolated)" if interp2d else "")
                        ),
                    )
                    # Axes labels on the bottom-most subplot
                    pmf2d_fig.update_xaxes(
                        title_text=f"{metric_display_label(metric_x)} value", row=n_rows, col=1
                    )
                    pmf2d_fig.update_yaxes(
                        title_text=f"{metric_display_label(metric_y)} value", row=1, col=1
                    )

        # ---------------------------------------------------------------
        # Sampling diagnostics summary (ESS-aware; approximate)
        # ---------------------------------------------------------------
        try:
            items = []
            for d in loaded:
                v = str(d.get("variant"))
                m = str(d.get("metric"))
                rep = d.get("replica")
                key = (v, m, str(rep) if rep is not None else None)
                info = ess_map.get(key)
                if not info:
                    continue
                label = f"{v}/{metric_display_label(m, variant=v)}"
                if show_replicas and rep is not None:
                    label += f" (rep {rep})"
                extra = ""
                if "circ_mean_deg" in info and np.isfinite(info["circ_mean_deg"]):
                    extra = f", circ-mean={info['circ_mean_deg']:.1f}°, circ-var={info.get('circ_var', float('nan')):.3f}"
                items.append(html.Li(f"{label}: N={int(info.get('n', 0))}, τ_int≈{info.get('tau_int', float('nan')):.2f}, ESS≈{info.get('ess', float('nan')):.1f}{extra}"))
            if items:
                stats_children = html.Div([html.Div("Sampling diagnostics (approx.)", style={"fontWeight": 600}), html.Ul(items, style={"margin": "4px 0 0 18px"})])
            else:
                stats_children = html.Div()
        except Exception:
            stats_children = html.Div()

        return fig, acf_fig, pmf_fig, stats_children, pmf2d_fig
