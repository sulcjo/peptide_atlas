from __future__ import annotations

from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path as _Path

from dash import Input, Output, State, dcc, html, dash_table

from ..theming.errors import error_fig
from ..metrics import prettify_column_label, torsion_sort_key, parse_torsion_metric
from ..data.context import filter_numeric_columns

try:
    from scipy.stats import t as _t_dist
    _SCIPY_AVAILABLE = True
except ImportError:
    _SCIPY_AVAILABLE = False

try:
    from sklearn.feature_selection import mutual_info_regression as _mi_regression
    _SKLEARN_AVAILABLE = True
except ImportError:
    _SKLEARN_AVAILABLE = False


TAB_LABEL = "Correlation Matrix"


def _group_columns_by_prefix(cols: List[str]) -> Dict[str, List[str]]:
    groups: Dict[str, List[str]] = {}
    for c in cols:
        prefix = c.split("__", 1)[0] if "__" in c else "misc"
        groups.setdefault(prefix, []).append(c)
    return groups


def _reorder_corr_matrix(cm: pd.DataFrame) -> pd.DataFrame:
    if cm.shape[0] <= 2:
        return cm
    cols = list(cm.columns)
    remaining = cols.copy()
    ordered = [remaining.pop(0)]
    while remaining:
        best_idx, best_score = 0, -np.inf
        for i, name in enumerate(remaining):
            score = float(np.mean(np.abs(cm.loc[name, ordered])))
            if score > best_score:
                best_score = score
                best_idx = i
        ordered.append(remaining.pop(best_idx))
    return cm.loc[ordered, ordered]


def _compute_nmi_matrix(df_sub: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """Normalized Mutual Information matrix via sklearn k-NN estimator.

    Uses listwise-complete rows (no per-pair imputation).
    NMI_ij = MI_ij / sqrt(H_i * H_j) where H estimated from the diagonal
    (MI(X_i, X_i) ≈ H(X_i) under the k-NN estimator).
    Values in [0, 1]; 1 means one variable is a deterministic function of the
    other; 0 means statistical independence.
    """
    if not _SKLEARN_AVAILABLE:
        return pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    X = df_sub[cols].replace([np.inf, -np.inf], np.nan).dropna()
    if X.shape[0] < 3 or len(cols) < 2:
        return pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    X_arr = X.values.astype(float)
    k = len(cols)
    mi_mat = np.zeros((k, k))
    for j in range(k):
        mi_mat[:, j] = _mi_regression(X_arr, X_arr[:, j], random_state=0, n_neighbors=3)
    mi_mat = (mi_mat + mi_mat.T) / 2.0
    h_diag = np.maximum(np.diag(mi_mat), 1e-10)
    norm_outer = np.sqrt(np.outer(h_diag, h_diag))
    nmi_mat = np.clip(np.where(norm_outer > 0, mi_mat / norm_outer, 0.0), 0.0, 1.0)
    np.fill_diagonal(nmi_mat, 1.0)
    return pd.DataFrame(nmi_mat, index=cols, columns=cols)


def _pairwise_n(df: pd.DataFrame) -> pd.DataFrame:
    valid = df.notna().astype(np.int8)
    n_mat = valid.T @ valid
    n_mat.index = df.columns
    n_mat.columns = df.columns
    return n_mat


def _pvalues_from_corr(cm: pd.DataFrame, n_mat: pd.DataFrame) -> pd.DataFrame:
    """Two-sided p-value from correlation + n using t-approximation."""
    if not _SCIPY_AVAILABLE:
        return pd.DataFrame(np.ones(cm.shape), index=cm.index, columns=cm.columns)
    r = cm.values.astype(float)
    n = n_mat.values.astype(float)
    df_arr = np.maximum(n - 2.0, 1.0)
    r_c = np.clip(r, -1 + 1e-12, 1 - 1e-12)
    t_stat = r_c * np.sqrt(df_arr) / np.sqrt(1.0 - r_c ** 2)
    p_vals = 2.0 * _t_dist.sf(np.abs(t_stat), df=df_arr)
    return pd.DataFrame(p_vals, index=cm.index, columns=cm.columns)


def _bh_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment."""
    n = len(pvals)
    if n == 0:
        return np.array([], dtype=float)
    order = np.argsort(pvals)
    adj = pvals[order] * n / (np.arange(n, dtype=float) + 1.0)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    result = np.empty(n)
    result[order] = np.minimum(adj, 1.0)
    return result


def _sig_star(p: float) -> str:
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def _build_pairs_from_corr(
    cm: pd.DataFrame,
    n_mat: Optional[pd.DataFrame] = None,
    adj_p_mat: Optional[pd.DataFrame] = None,
) -> List[Dict]:
    pairs: List[Dict] = []
    cols = list(cm.columns)
    n = len(cols)
    for i in range(n):
        for j in range(i + 1, n):
            f1, f2 = cols[i], cols[j]
            val = float(cm.iloc[i, j])
            n_obs = int(n_mat.loc[f1, f2]) if n_mat is not None else 0
            adj_p = float(adj_p_mat.iloc[i, j]) if adj_p_mat is not None else 1.0
            pairs.append(
                {
                    "feat1": prettify_column_label(f1),
                    "feat2": prettify_column_label(f2),
                    "feat1_raw": f1,
                    "feat2_raw": f2,
                    "n": n_obs,
                    "corr": val,
                    "abs_corr": abs(val),
                    "sig": _sig_star(adj_p) if np.isfinite(adj_p) else "",
                }
            )
    pairs.sort(key=lambda d: d["abs_corr"], reverse=True)
    return pairs


def _effective_dim_text(cm: pd.DataFrame, n_variants: int) -> str:
    cm_arr = np.nan_to_num(cm.values.astype(float), nan=0.0)
    eigvals = np.linalg.eigvalsh(cm_arr)[::-1]
    eigvals = np.maximum(eigvals, 0.0)
    total = eigvals.sum()
    if total < 1e-12:
        return ""
    n_eff = total ** 2 / (eigvals ** 2).sum()
    cumvar = np.cumsum(eigvals) / total
    n_90 = int(np.searchsorted(cumvar, 0.90)) + 1
    n_feat = len(eigvals)
    return (
        f"Effective dimensionality: {n_eff:.1f} / {n_feat} features  ·  "
        f"PCs to 90% variance: {n_90}  ·  "
        f"n variants: {n_variants}"
    )


def _residue_coupling_fig(cm: pd.DataFrame, metric_label: str = "|r|") -> go.Figure:
    cols = list(cm.columns)
    col_res: Dict[str, int] = {}
    for c in cols:
        prefix = c.split("__", 1)[0] if "__" in c else c
        t = parse_torsion_metric(prefix)
        if t is not None:
            col_res[c] = t.resid

    if len(set(col_res.values())) < 2:
        return error_fig("No residue-resolved torsion features in current selection")

    residues = sorted(set(col_res.values()))
    n_res = len(residues)
    res_idx = {r: i for i, r in enumerate(residues)}

    coupling = np.zeros((n_res, n_res))
    counts = np.zeros((n_res, n_res), dtype=int)
    for i, ci in enumerate(cols):
        ri = col_res.get(ci)
        if ri is None:
            continue
        for j, cj in enumerate(cols):
            if j <= i:
                continue
            rj = col_res.get(cj)
            if rj is None:
                continue
            val = float(cm.iloc[i, j])
            if np.isfinite(val):
                a, b = res_idx[ri], res_idx[rj]
                coupling[a, b] += abs(val)
                coupling[b, a] += abs(val)
                counts[a, b] += 1
                counts[b, a] += 1

    with np.errstate(invalid="ignore"):
        mean_c = np.where(counts > 0, coupling / counts, np.nan)

    labels = [str(r) for r in residues]
    fig = go.Figure(
        go.Heatmap(
            z=mean_c,
            x=labels,
            y=labels,
            colorscale="Blues",
            zmin=0,
            zmax=1,
            colorbar=dict(title=f"mean {metric_label}"),
            hovertemplate=f"res %{{y}} — res %{{x}}<br>mean {metric_label} = %{{z:.3f}}<extra></extra>",
        )
    )
    fig.update_layout(
        title=f"Residue–residue coupling (mean {metric_label} of torsion features)",
        height=360,
        margin=dict(l=50, r=20, t=40, b=50),
        xaxis=dict(title="Residue"),
        yaxis=dict(title="Residue", autorange="reversed"),
    )
    return fig



def layout(ctx):
    df: pd.DataFrame = getattr(ctx, "df", pd.DataFrame())
    numeric_cols: List[str] = list(getattr(ctx, "numeric_cols", []) or [])

    if (not numeric_cols) and isinstance(df, pd.DataFrame) and not df.empty:
        raw = [c for c in df.columns if c != "variant" and pd.api.types.is_numeric_dtype(df[c])]
        numeric_cols = filter_numeric_columns(raw)

    groups = _group_columns_by_prefix(numeric_cols) if numeric_cols else {}
    group_options = [{"label": k, "value": k} for k in groups.keys()]
    target_options = [{"label": prettify_column_label(c), "value": c} for c in numeric_cols[:300]]

    try:
        _help_txt = (_Path(__file__).parent / "correlation_features.txt").read_text()
    except Exception:
        _help_txt = "Feature documentation not available."

    controls = html.Div(
        [
            html.Div(
                [
                    html.Div("Correlation matrix", style={"fontWeight": 600}),
                    html.Button(
                        "? Help",
                        id="corr-help-btn",
                        n_clicks=0,
                        style={
                            "fontSize": "0.8em",
                            "padding": "2px 10px",
                            "cursor": "pointer",
                            "borderRadius": "4px",
                            "border": "1px solid #aaa",
                            "background": "transparent",
                        },
                    ),
                ],
                style={"display": "flex", "justifyContent": "space-between", "alignItems": "center", "marginBottom": "4px"},
            ),
            # Row 1: kind / group / max / cluster
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Correlation", style={"fontSize": "0.85em"}),
                            dcc.RadioItems(
                                id="corr-kind",
                                options=[
                                    {"label": "MI (NMI)", "value": "mi"},
                                    {"label": "Pearson", "value": "pearson"},
                                    {"label": "Spearman", "value": "spearman"},
                                ],
                                value="mi",
                                inline=True,
                            ),
                        ],
                        style={"flex": "0 0 30%", "paddingRight": "8px"},
                    ),
                    html.Div(
                        [
                            html.Div("Feature group (optional)", style={"fontSize": "0.85em"}),
                            dcc.Dropdown(
                                id="corr-group",
                                options=group_options,
                                value=None,
                                placeholder="All groups",
                                clearable=True,
                            ),
                        ],
                        style={"flex": "0 0 38%", "paddingRight": "8px"},
                    ),
                    html.Div(
                        [
                            html.Div("Max features", style={"fontSize": "0.85em"}),
                            dcc.Slider(
                                id="corr-max",
                                min=6,
                                max=40,
                                step=1,
                                value=16,
                                tooltip={"placement": "bottom"},
                            ),
                        ],
                        style={"flex": "0 0 25%", "paddingRight": "8px"},
                    ),
                    html.Div(
                        [
                            html.Div("Ordering", style={"fontSize": "0.85em", "marginBottom": "2px"}),
                            dcc.Checklist(
                                id="corr-cluster",
                                options=[{"label": "Cluster / reorder", "value": "cluster"}],
                                value=[],
                                inline=True,
                                style={"fontSize": "0.85em"},
                            ),
                        ],
                        style={"flex": "0 0 15%"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "gap": "8px", "flexWrap": "wrap"},
            ),
            # Row 2: redundancy threshold
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("|corr| threshold for redundancy", style={"fontSize": "0.85em"}),
                            dcc.Slider(
                                id="corr-redundant-threshold",
                                min=0.5,
                                max=0.99,
                                step=0.01,
                                value=0.9,
                                tooltip={"placement": "bottom"},
                            ),
                        ],
                        style={"flex": "1 1 100%", "marginTop": "6px"},
                    ),
                ],
                style={"display": "flex", "gap": "8px", "flexWrap": "wrap"},
            ),
            # Row 3: bootstrap
            html.Div(
                [
                    html.Div("Bootstrap CI for r", style={"fontSize": "0.85em", "marginTop": "6px"}),
                    dcc.Checklist(
                        id="corr-bootstrap",
                        options=[{"label": "Show 95% CI", "value": "boot"}],
                        value=[],
                        inline=True,
                        style={"fontSize": "0.85em"},
                    ),
                    html.Div(
                        [
                            html.Div("Bootstrap draws", style={"fontSize": "0.8em"}),
                            dcc.Slider(
                                id="corr-bootstrap-n",
                                min=100,
                                max=2000,
                                step=100,
                                value=500,
                                tooltip={"placement": "bottom"},
                            ),
                        ],
                        style={"marginTop": "4px"},
                    ),
                ],
                style={"marginTop": "4px"},
            ),
            # Row 4: significance + target feature
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Significance (BH-corrected)", style={"fontSize": "0.85em"}),
                            dcc.Checklist(
                                id="corr-sig-check",
                                options=[{"label": "Show p-value stars in table + hover", "value": "sig"}],
                                value=[],
                                inline=True,
                                style={"fontSize": "0.85em"},
                            ),
                        ],
                        style={"flex": "0 0 50%", "paddingRight": "12px"},
                    ),
                    html.Div(
                        [
                            html.Div("Target feature (for bar chart)", style={"fontSize": "0.85em"}),
                            dcc.Dropdown(
                                id="corr-target-feat",
                                options=target_options,
                                value=None,
                                placeholder="Select target…",
                                clearable=True,
                            ),
                        ],
                        style={"flex": "0 0 50%"},
                    ),
                ],
                style={"display": "flex", "alignItems": "flex-start", "gap": "8px", "marginTop": "8px", "flexWrap": "wrap"},
            ),
            html.Div(
                "Uses the same variant selection as the Features tab. "
                "Click a cell or a table row to inspect the scatter.",
                style={"fontSize": "0.8em", "marginTop": "4px", "opacity": 0.7},
            ),
        ],
        style={
            "padding": "10px",
            "border": "1px solid #ccc",
            "borderRadius": "8px",
            "marginBottom": "6px",
            "backgroundColor": "rgba(255,255,255,0.03)",
        },
    )

    dim_row = html.Div(
        id="corr-dim-text",
        style={"fontSize": "0.82em", "opacity": 0.75, "padding": "2px 4px", "marginBottom": "6px"},
    )

    corr_graph = dcc.Graph(
        id="corr-graph",
        style={"height": "70vh"},
        config={"displaylogo": False},
    )
    scatter_graph = dcc.Graph(
        id="corr-scatter",
        style={"height": "35vh"},
        config={"displaylogo": False},
    )

    _pair_cols = [
        {"name": "Feature 1", "id": "feat1"},
        {"name": "Feature 2", "id": "feat2"},
        {"name": "n", "id": "n"},
        {"name": "corr", "id": "corr"},
        {"name": "|corr|", "id": "abs_corr"},
        {"name": "sig", "id": "sig"},
        {"name": "_f1", "id": "feat1_raw"},
        {"name": "_f2", "id": "feat2_raw"},
    ]
    _pair_style = {"fontSize": "11px", "padding": "4px", "whiteSpace": "normal", "height": "auto"}

    top_pairs_table = dash_table.DataTable(
        id="corr-top-pairs",
        columns=_pair_cols,
        hidden_columns=["feat1_raw", "feat2_raw"],
        data=[],
        page_size=10,
        sort_action="native",
        style_table={"maxHeight": "260px", "overflowY": "auto"},
        style_cell=_pair_style,
        style_header={"fontWeight": "600"},
    )
    redundant_table = dash_table.DataTable(
        id="corr-redundant-table",
        columns=_pair_cols,
        hidden_columns=["feat1_raw", "feat2_raw"],
        data=[],
        page_size=10,
        sort_action="native",
        style_table={"maxHeight": "260px", "overflowY": "auto"},
        style_cell=_pair_style,
        style_header={"fontWeight": "600"},
    )

    right_panel = html.Div(
        [
            html.Div("Feature pair scatter & regression", style={"fontWeight": 500, "fontSize": "0.9em", "marginBottom": "2px"}),
            scatter_graph,
            html.Div("Top correlated pairs (click a row to inspect):", style={"fontWeight": 500, "fontSize": "0.9em", "marginTop": "8px", "marginBottom": "2px"}),
            top_pairs_table,
            html.Div("Redundant feature pairs (|corr| ≥ threshold):", style={"fontWeight": 500, "fontSize": "0.9em", "marginTop": "10px", "marginBottom": "2px"}),
            redundant_table,
        ],
        style={"flex": "0 0 32%", "minWidth": "280px", "paddingLeft": "12px"},
    )

    main_panel = html.Div(
        [
            html.Div(corr_graph, style={"flex": "1 1 0", "minWidth": "0"}),
            right_panel,
        ],
        style={"display": "flex", "alignItems": "stretch", "gap": "12px", "flexWrap": "wrap"},
    )

    # Analytics row: residue coupling + target bar
    analytics_row = html.Div(
        [
            html.Div(
                [
                    html.Div("Residue–residue coupling", style={"fontWeight": 500, "fontSize": "0.9em", "marginBottom": "2px"}),
                    dcc.Graph(id="corr-residue-heatmap", style={"height": "360px"}, config={"displaylogo": False}),
                ],
                style={"flex": "1 1 0", "minWidth": "260px"},
            ),
            html.Div(
                [
                    html.Div("Target feature correlations", style={"fontWeight": 500, "fontSize": "0.9em", "marginBottom": "2px"}),
                    dcc.Graph(id="corr-target-bar", style={"height": "360px"}, config={"displaylogo": False}),
                ],
                style={"flex": "1 1 0", "minWidth": "260px"},
            ),
        ],
        style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginTop": "14px"},
    )

    help_panel = html.Div(
        html.Pre(
            _help_txt,
            style={"fontSize": "0.8em", "whiteSpace": "pre-wrap", "margin": 0, "lineHeight": "1.5"},
        ),
        id="corr-help-panel",
        style={
            "display": "none",
            "padding": "10px 14px",
            "border": "1px solid #ccc",
            "borderRadius": "6px",
            "marginBottom": "8px",
            "backgroundColor": "rgba(255,255,255,0.04)",
            "maxHeight": "420px",
            "overflowY": "auto",
        },
    )

    return html.Div([controls, help_panel, dim_row, main_panel, analytics_row], style={"padding": "8px"})


_HELP_HIDDEN = {"display": "none"}
_HELP_SHOWN = {
    "display": "block",
    "padding": "10px 14px",
    "border": "1px solid #ccc",
    "borderRadius": "6px",
    "marginBottom": "8px",
    "backgroundColor": "rgba(255,255,255,0.04)",
    "maxHeight": "420px",
    "overflowY": "auto",
}


def register_callbacks(app, ctx):
    from .shared import apply_theme
    df_initial: pd.DataFrame = getattr(ctx, "df", pd.DataFrame())
    numeric_initial: List[str] = list(getattr(ctx, "numeric_cols", []) or [])

    @app.callback(
        Output("corr-help-panel", "style"),
        Input("corr-help-btn", "n_clicks"),
        State("corr-help-panel", "style"),
        prevent_initial_call=True,
    )
    def _toggle_help(n_clicks, current_style):
        if current_style and current_style.get("display") == "block":
            return _HELP_HIDDEN
        return _HELP_SHOWN

    # ------------------------------------------------------------------ #
    # Main callback: heatmap, scatter, tables, dim text, residue
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("corr-graph", "figure"),
        Output("corr-scatter", "figure"),
        Output("corr-top-pairs", "data"),
        Output("corr-redundant-table", "data"),
        Output("corr-dim-text", "children"),
        Output("corr-residue-heatmap", "figure"),
        Input("corr-kind", "value"),
        Input("corr-group", "value"),
        Input("corr-max", "value"),
        Input("feat-variant-select", "value"),
        Input("feat-selected-table", "data"),
        Input("corr-cluster", "value"),
        Input("corr-graph", "clickData"),
        Input("corr-top-pairs", "active_cell"),
        Input("corr-redundant-threshold", "value"),
        Input("corr-bootstrap", "value"),
        Input("corr-bootstrap-n", "value"),
        Input("corr-sig-check", "value"),
        Input("theme-store", "data"),
        prevent_initial_call=False,
    )
    def _update_corr(
        kind,
        group_sel,
        kmax,
        variant_filter,
        selected_table,
        cluster_value,
        click_data,
        top_active_cell,
        redundant_thr,
        boot_flags,
        boot_n,
        sig_flags,
        theme,
    ):
        scatter_empty = apply_theme(error_fig("Click a matrix cell or table row to see scatter"), theme)
        no_data = apply_theme(error_fig("No data for correlation matrix"), theme)
        no_res = apply_theme(error_fig("No residue-resolved torsion features in current selection"), theme)
        _err6 = (no_data, scatter_empty, [], [], "", no_res)

        df: pd.DataFrame = getattr(ctx, "df", df_initial)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return _err6

        numeric_cols: List[str] = list(getattr(ctx, "numeric_cols", []) or numeric_initial)
        if (not numeric_cols) and not df.empty:
            numeric_cols = [
                c for c in df.columns
                if c != "variant" and pd.api.types.is_numeric_dtype(df[c])
            ]
        if not numeric_cols:
            return (apply_theme(error_fig("No numeric columns"), theme), scatter_empty, [], [], "", no_res)

        numeric_cols = filter_numeric_columns(numeric_cols)
        groups = _group_columns_by_prefix(numeric_cols)

        if group_sel and group_sel in groups:
            cols = [c for c in groups[group_sel] if c in df.columns]
        else:
            cols = [c for c in numeric_cols if c in df.columns]

        if not cols:
            return (apply_theme(error_fig("No columns selected"), theme), scatter_empty, [], [], "", no_res)

        try:
            kmax_int = int(kmax) if kmax is not None else 16
        except Exception:
            kmax_int = 16
        kmax_int = max(1, min(kmax_int, len(cols)))
        cols = cols[:kmax_int]

        # Variant subset
        var_set: Set[str] = set()
        if selected_table:
            for row in selected_table:
                v = row.get("variant")
                if v is not None:
                    var_set.add(str(v))
        if (not var_set) and variant_filter:
            for v in variant_filter:
                if v is not None:
                    var_set.add(str(v))

        df_sub = df
        if var_set and "variant" in df_sub.columns:
            df_sub = df_sub[df_sub["variant"].astype(str).isin(var_set)]
        if df_sub.empty:
            return (apply_theme(error_fig("No variants in current selection"), theme), scatter_empty, [], [], "", no_res)

        use_cols = [c for c in cols if c in df_sub.columns]
        if not use_cols:
            return (apply_theme(error_fig("No numeric columns after filtering"), theme), scatter_empty, [], [], "", no_res)

        numeric_sub = df_sub[use_cols].replace([np.inf, -np.inf], np.nan)
        finite_counts = numeric_sub.notna().sum(axis=0)
        viable_cols = [c for c in use_cols if int(finite_counts.get(c, 0)) >= 2]
        if len(viable_cols) < 2:
            return (apply_theme(error_fig("Need at least two features with finite values"), theme), scatter_empty, [], [], "", no_res)

        sub_for_corr = numeric_sub[viable_cols]

        is_mi = kind == "mi"

        if is_mi:
            try:
                cm = _compute_nmi_matrix(sub_for_corr, viable_cols)
            except Exception as exc:
                return (apply_theme(error_fig(f"Failed to compute MI: {exc}"), theme), scatter_empty, [], [], "", no_res)
            n_complete = int(sub_for_corr.dropna().shape[0])
            n_mat = pd.DataFrame(
                np.full((len(viable_cols), len(viable_cols)), n_complete, dtype=int),
                index=viable_cols, columns=viable_cols,
            )
        else:
            n_mat = _pairwise_n(sub_for_corr)
            method = "pearson" if kind == "pearson" else "spearman"
            try:
                cm = sub_for_corr.corr(method=method, min_periods=2)
            except Exception as exc:
                return (apply_theme(error_fig(f"Failed to compute correlation: {exc}"), theme), scatter_empty, [], [], "", no_res)

        cm = cm.dropna(axis=0, how="all").dropna(axis=1, how="all")
        if cm.shape[0] < 2 or cm.empty:
            return (apply_theme(error_fig("No overlapping finite values"), theme), scatter_empty, [], [], "", no_res)

        shared = [c for c in cm.columns if c in n_mat.columns]
        n_mat = n_mat.loc[shared, shared]

        # p-values only for Pearson/Spearman (t-approximation not valid for NMI)
        sig_on = (not is_mi) and ("sig" in (sig_flags or []))
        adj_p_mat: Optional[pd.DataFrame] = None
        if sig_on and _SCIPY_AVAILABLE:
            raw_p = _pvalues_from_corr(cm, n_mat)
            cols_cm = list(cm.columns)
            n_pairs = len(cols_cm) * (len(cols_cm) - 1) // 2
            if n_pairs > 0:
                upper_idx = [(i, j) for i in range(len(cols_cm)) for j in range(i + 1, len(cols_cm))]
                pvals_upper = np.array([raw_p.iloc[i, j] for i, j in upper_idx])
                adj_upper = _bh_adjust(pvals_upper)
                adj_arr = np.ones(raw_p.shape)
                for k, (i, j) in enumerate(upper_idx):
                    adj_arr[i, j] = adj_arr[j, i] = adj_upper[k]
                adj_p_mat = pd.DataFrame(adj_arr, index=cm.index, columns=cm.columns)

        cluster_on = cluster_value and ("cluster" in cluster_value)
        if cluster_on:
            cm = _reorder_corr_matrix(cm)
            n_mat = n_mat.loc[cm.index, cm.columns]
            if adj_p_mat is not None:
                adj_p_mat = adj_p_mat.loc[cm.index, cm.columns]

        # Pairs tables
        all_pairs = _build_pairs_from_corr(cm, n_mat, adj_p_mat)
        top_pairs = all_pairs[:100]

        try:
            thr = float(redundant_thr) if redundant_thr is not None else 0.9
        except Exception:
            thr = 0.9
        thr = max(0.0, min(thr, 0.9999))
        redundant_pairs = [p for p in all_pairs if p["abs_corr"] >= thr][:200]

        def _round_pairs(lst: List[Dict]) -> List[Dict]:
            return [
                {
                    "feat1": p["feat1"],
                    "feat2": p["feat2"],
                    "feat1_raw": p["feat1_raw"],
                    "feat2_raw": p["feat2_raw"],
                    "n": p.get("n", ""),
                    "corr": round(float(p["corr"]), 3),
                    "abs_corr": round(float(p["abs_corr"]), 3),
                    "sig": p.get("sig", "") if sig_on else "",
                }
                for p in lst
            ]

        top_pairs_display = _round_pairs(top_pairs)
        redundant_display = _round_pairs(redundant_pairs)

        # Selected pair
        selected_f1: Optional[str] = None
        selected_f2: Optional[str] = None

        if top_active_cell and isinstance(top_active_cell, dict):
            row_idx = top_active_cell.get("row")
            if isinstance(row_idx, int) and 0 <= row_idx < len(top_pairs_display):
                row = top_pairs_display[row_idx]
                selected_f1 = row.get("feat1_raw")
                selected_f2 = row.get("feat2_raw")

        if (selected_f1 is None or selected_f2 is None) and click_data:
            try:
                pt = click_data["points"][0]
                x_name = pt.get("x")
                y_name = pt.get("y")
                if isinstance(x_name, str) and isinstance(y_name, str):
                    label_to_raw = {prettify_column_label(c): c for c in cm.columns}
                    selected_f1 = label_to_raw.get(y_name, y_name)
                    selected_f2 = label_to_raw.get(x_name, x_name)
            except Exception:
                pass

        # Build heatmap
        _cols = list(cm.columns)
        _labels = [prettify_column_label(c) for c in _cols]
        _n = len(_cols)
        _val_label = "NMI" if is_mi else "r"

        cell_text = np.full((_n, _n), "", dtype=object)
        hover_text = np.full((_n, _n), "", dtype=object)
        for i in range(_n):
            for j in range(_n):
                r_val = float(cm.iloc[i, j])
                r_str = f"{r_val:.2f}" if np.isfinite(r_val) else ""
                n_ij = int(n_mat.iloc[i, j]) if i < n_mat.shape[0] and j < n_mat.shape[1] else 0
                p_ij = float(adj_p_mat.iloc[i, j]) if adj_p_mat is not None else 1.0
                star = _sig_star(p_ij) if sig_on and np.isfinite(p_ij) else ""
                cell_text[i, j] = f"{r_str}{star}"
                hover_parts = [f"{_labels[i]} vs {_labels[j]}", f"{_val_label} = {r_val:.3f}", f"n = {n_ij}"]
                if sig_on and adj_p_mat is not None and np.isfinite(p_ij):
                    hover_parts.append(f"adj-p = {p_ij:.3e} {star}")
                hover_text[i, j] = "<br>".join(hover_parts) if np.isfinite(r_val) else f"{_labels[i]} vs {_labels[j]}<br>{_val_label} = NaN"

        if is_mi:
            _colorscale, _zmin, _zmax, _cbar_title = "YlOrRd", 0.0, 1.0, "NMI"
        else:
            _colorscale, _zmin, _zmax, _cbar_title = "RdBu", -1.0, 1.0, "r"

        fig_corr = go.Figure(
            go.Heatmap(
                z=cm.values,
                x=_labels,
                y=_labels,
                text=cell_text,
                texttemplate="%{text}",
                hovertext=hover_text,
                hovertemplate="%{hovertext}<extra></extra>",
                colorscale=_colorscale,
                zmin=_zmin,
                zmax=_zmax,
                colorbar=dict(title=_cbar_title),
            )
        )
        fig_corr.update_layout(
            height=700,
            margin=dict(l=80, r=20, t=40, b=40),
            xaxis=dict(title=None, tickangle=-40),
            yaxis=dict(title=None, autorange="reversed"),
        )

        if selected_f1 and selected_f2 and selected_f1 in cm.index and selected_f2 in cm.columns:
            fig_corr.add_trace(
                go.Scatter(
                    x=[prettify_column_label(selected_f2)],
                    y=[prettify_column_label(selected_f1)],
                    mode="markers",
                    marker=dict(size=18, color="rgba(0,0,0,0)", line=dict(color="black", width=2)),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )

        # Scatter for selected pair
        scatter_fig = scatter_empty
        if selected_f1 and selected_f2:
            if selected_f1 in numeric_sub.columns and selected_f2 in numeric_sub.columns:
                _hover_cols = [selected_f1, selected_f2]
                if "variant" in df_sub.columns:
                    _hover_cols = ["variant"] + _hover_cols
                xy = (
                    df_sub[_hover_cols]
                    .replace([np.inf, -np.inf], np.nan)
                    .dropna(subset=[selected_f1, selected_f2])
                )
                if xy.shape[0] >= 2:
                    x_vals = xy[selected_f1].values.astype(float)
                    y_vals = xy[selected_f2].values.astype(float)

                    try:
                        a, b = np.polyfit(x_vals, y_vals, 1)
                        r = float(np.corrcoef(x_vals, y_vals)[0, 1])
                    except Exception:
                        a, b, r = np.nan, np.nan, np.nan

                    scatter_fig = px.scatter(
                        xy,
                        x=selected_f1,
                        y=selected_f2,
                        hover_name="variant" if "variant" in xy.columns else None,
                        opacity=0.7,
                    )
                    try:
                        scatter_fig.update_layout(
                            xaxis_title=prettify_column_label(selected_f1),
                            yaxis_title=prettify_column_label(selected_f2),
                            title=f"{prettify_column_label(selected_f1)} vs {prettify_column_label(selected_f2)}",
                        )
                    except Exception:
                        pass

                    if np.isfinite(a) and np.isfinite(b):
                        x_line = np.linspace(x_vals.min(), x_vals.max(), 50)
                        y_line = a * x_line + b
                        scatter_fig.add_trace(
                            go.Scatter(x=x_line, y=y_line, mode="lines", name="Fit")
                        )
                        _pf1 = prettify_column_label(selected_f1)
                        _pf2 = prettify_column_label(selected_f2)
                        title_txt = (
                            f"{_pf2} vs {_pf1} "
                            f"(n={xy.shape[0]}, r={r:.3f}, y = {a:.3f}·x + {b:.3f})"
                        )
                    else:
                        _pf1 = prettify_column_label(selected_f1)
                        _pf2 = prettify_column_label(selected_f2)
                        title_txt = f"{_pf2} vs {_pf1} (n={xy.shape[0]})"

                    try:
                        boot_on = "boot" in (boot_flags or [])
                    except Exception:
                        boot_on = False
                    if boot_on and xy.shape[0] >= 6:
                        try:
                            B = int(boot_n) if boot_n is not None else 500
                        except Exception:
                            B = 500
                        B = max(100, min(B, 5000))
                        rng = np.random.default_rng(0)
                        npts = int(x_vals.size)
                        idx_mat = rng.integers(0, npts, size=(B, npts))
                        xb_mat = x_vals[idx_mat]
                        yb_mat = y_vals[idx_mat]
                        if kind == "spearman":
                            from scipy.stats import rankdata as _rankdata
                            xb_mat = np.apply_along_axis(_rankdata, 1, xb_mat).astype(float)
                            yb_mat = np.apply_along_axis(_rankdata, 1, yb_mat).astype(float)
                        xb_c = xb_mat - xb_mat.mean(axis=1, keepdims=True)
                        yb_c = yb_mat - yb_mat.mean(axis=1, keepdims=True)
                        num = (xb_c * yb_c).sum(axis=1)
                        denom = np.sqrt((xb_c ** 2).sum(axis=1) * (yb_c ** 2).sum(axis=1))
                        rs_arr = np.where(denom > 0, num / denom, np.nan)
                        rs = rs_arr[np.isfinite(rs_arr)]
                        if len(rs) >= 20:
                            lo, hi = np.percentile(rs, [2.5, 97.5])
                            title_txt = f"{title_txt}, 95% CI [{lo:.3f}, {hi:.3f}]"

                    scatter_fig.update_layout(
                        height=350,
                        margin=dict(l=60, r=20, t=40, b=50),
                        title=title_txt,
                        showlegend=False,
                    )
                else:
                    scatter_fig = apply_theme(error_fig("Not enough finite data points for this feature pair"), theme)

        # Effective dimensionality text
        dim_text = _effective_dim_text(cm, n_variants=int(df_sub.shape[0]))

        # Residue coupling heatmap
        residue_fig = _residue_coupling_fig(cm, metric_label="NMI" if is_mi else "|r|")

        return (
            apply_theme(fig_corr, theme), apply_theme(scatter_fig, theme),
            top_pairs_display, redundant_display,
            dim_text, apply_theme(residue_fig, theme),
        )

    # ------------------------------------------------------------------ #
    # Separate callback: target-feature correlation bar chart
    # ------------------------------------------------------------------ #
    @app.callback(
        Output("corr-target-bar", "figure"),
        Input("corr-target-feat", "value"),
        Input("corr-kind", "value"),
        Input("feat-variant-select", "value"),
        Input("feat-selected-table", "data"),
        Input("theme-store", "data"),
        prevent_initial_call=False,
    )
    def _update_target_bar(target_feat, kind, variant_filter, selected_table, theme):
        if not target_feat:
            return apply_theme(error_fig("Select a target feature to see ranked correlations"), theme)

        df: pd.DataFrame = getattr(ctx, "df", df_initial)
        if not isinstance(df, pd.DataFrame) or df.empty:
            return apply_theme(error_fig("No data"), theme)

        var_set: Set[str] = set()
        if selected_table:
            for row in selected_table:
                v = row.get("variant")
                if v is not None:
                    var_set.add(str(v))
        if (not var_set) and variant_filter:
            for v in variant_filter:
                if v is not None:
                    var_set.add(str(v))

        df_sub = df
        if var_set and "variant" in df_sub.columns:
            df_sub = df_sub[df_sub["variant"].astype(str).isin(var_set)]

        numeric_cols: List[str] = list(getattr(ctx, "numeric_cols", []) or numeric_initial)
        if not numeric_cols and not df_sub.empty:
            numeric_cols = [
                c for c in df_sub.columns
                if c != "variant" and pd.api.types.is_numeric_dtype(df_sub[c])
            ]
        numeric_cols = filter_numeric_columns(numeric_cols)

        if target_feat not in df_sub.columns:
            return apply_theme(error_fig(f"Target feature '{target_feat}' not in data"), theme)

        target_vals = df_sub[target_feat].replace([np.inf, -np.inf], np.nan)
        other_cols = [c for c in numeric_cols if c != target_feat and c in df_sub.columns]
        if not other_cols:
            return apply_theme(error_fig("No other features to compare"), theme)

        is_mi = kind == "mi"
        results: List[Dict] = []

        if is_mi and _SKLEARN_AVAILABLE:
            candidates = [c for c in other_cols if c in df_sub.columns][:60]
            combined = (
                pd.concat([target_vals.rename("__target__")] + [
                    df_sub[c].replace([np.inf, -np.inf], np.nan) for c in candidates
                ], axis=1)
                .dropna()
            )
            if len(combined) < 3:
                return apply_theme(error_fig("Not enough complete rows for MI computation"), theme)
            y_arr = combined["__target__"].values.astype(float)
            X_arr = combined.drop(columns=["__target__"]).values.astype(float)
            mi_raw = _mi_regression(X_arr, y_arr, random_state=0, n_neighbors=3)
            h_target = float(_mi_regression(y_arr.reshape(-1, 1), y_arr, random_state=0, n_neighbors=3)[0])
            h_target = max(h_target, 1e-10)
            col_names = [c for c in combined.columns if c != "__target__"]
            for c, mi_val in zip(col_names, mi_raw):
                nmi_val = float(np.clip(mi_val / h_target, 0.0, 1.0))
                results.append({"col": c, "r": nmi_val, "abs_r": nmi_val})
        else:
            method = "pearson" if kind == "pearson" else "spearman"
            for c in other_cols:
                other_vals = df_sub[c].replace([np.inf, -np.inf], np.nan)
                xy = pd.concat([target_vals, other_vals], axis=1).dropna()
                if len(xy) < 4:
                    continue
                try:
                    x_ = xy.iloc[:, 0].values.astype(float)
                    y_ = xy.iloc[:, 1].values.astype(float)
                    if method == "pearson":
                        r_val = float(np.corrcoef(x_, y_)[0, 1])
                    else:
                        from scipy.stats import spearmanr as _spearmanr
                        r_val = float(_spearmanr(x_, y_).statistic)
                    if np.isfinite(r_val):
                        results.append({"col": c, "r": r_val, "abs_r": abs(r_val)})
                except Exception:
                    pass

        if not results:
            return apply_theme(error_fig("No valid correlations computed"), theme)

        results.sort(key=lambda d: d["abs_r"], reverse=True)
        results = results[:60]

        labels = [prettify_column_label(d["col"]) for d in results]
        r_vals = [d["r"] for d in results]
        if is_mi:
            colors = ["#e6550d"] * len(r_vals)
        else:
            colors = ["#d62728" if r < 0 else "#1f77b4" for r in r_vals]

        fig = go.Figure(
            go.Bar(
                x=r_vals,
                y=labels,
                orientation="h",
                marker_color=colors,
                hovertemplate="%{y}<br>r = %{x:.3f}<extra></extra>",
            )
        )
        _x_label = "NMI" if is_mi else "r"
        _x_range = [0.0, 1.0] if is_mi else [-1.0, 1.0]
        n_label = int(df_sub.shape[0])
        fig.update_layout(
            title=f"{_x_label} with {prettify_column_label(target_feat)} (n={n_label})",
            xaxis=dict(title=_x_label, range=_x_range),
            yaxis=dict(title=None, autorange="reversed"),
            height=max(300, min(len(results) * 22 + 80, 600)),
            margin=dict(l=220, r=20, t=40, b=40),
        )
        return apply_theme(fig, theme)
