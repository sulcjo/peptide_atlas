from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..data.timeseries import discover_timeseries_files
from dash import Input, Output, State, dcc, html, no_update
from dash.dash_table import DataTable

from .shared import panel

TAB_LABEL = "Diagnosis"


def _df_overview(name: str, df: pd.DataFrame) -> dict:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return {
            "table": name,
            "rows": 0,
            "cols": 0,
            "variants": 0,
            "metrics": 0,
            "replicas": 0,
            "na_cells": 0,
            "na_pct": 0.0,
        }

    variants = int(df["variant"].nunique()) if "variant" in df.columns else 0
    metrics = int(df["metric"].nunique()) if "metric" in df.columns else 0

    repl_col = None
    for c in ("repl", "replica"):
        if c in df.columns:
            repl_col = c
            break
    replicas = int(df[repl_col].nunique()) if repl_col else 0

    na_cells = int(df.isna().to_numpy().sum())
    total_cells = int(df.shape[0] * df.shape[1])
    na_pct = float(100.0 * na_cells / total_cells) if total_cells else 0.0

    return {
        "table": name,
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "variants": variants,
        "metrics": metrics,
        "replicas": replicas,
        "na_cells": na_cells,
        "na_pct": round(na_pct, 4),
    }


def _set_diff(a: set, b: set) -> Tuple[List[str], List[str]]:
    return sorted(list(a - b)), sorted(list(b - a))


def _features_missing_tables(features: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
    if features is None or not isinstance(features, pd.DataFrame) or features.empty:
        return [], []

    miss = features.isna().sum()
    total = int(features.shape[0])
    rows = []
    for col, nmiss in miss.items():
        nmiss_i = int(nmiss)
        pct = float(100.0 * nmiss_i / total) if total else 0.0
        rows.append(
            {
                "column": str(col),
                "dtype": str(features[col].dtype),
                "missing": nmiss_i,
                "missing_pct": round(pct, 4),
            }
        )
    rows.sort(key=lambda r: (r["missing"], r["column"]), reverse=True)

    # Variant-level completeness (numeric columns only to avoid huge object cols)
    num_cols = features.select_dtypes(include="number").columns.tolist()
    if "variant" in num_cols:
        num_cols.remove("variant")

    vrows = []
    if "variant" in features.columns and num_cols:
        x = features[["variant"] + num_cols].copy()
        miss_frac = x[num_cols].isna().mean(axis=1)
        for v, frac in zip(x["variant"].astype(str).tolist(), miss_frac.tolist()):
            vrows.append({"variant": v, "missing_pct_numeric": round(float(frac) * 100.0, 4)})
        vrows.sort(key=lambda r: r["missing_pct_numeric"], reverse=True)

    return rows[:200], vrows[:200]


def _coverage_by_variant_metric(df: pd.DataFrame) -> Tuple[List[dict], List[dict]]:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return [], []

    if "variant" not in df.columns or "metric" not in df.columns:
        return [], []

    all_metrics = sorted(df["metric"].dropna().astype(str).unique().tolist())
    if not all_metrics:
        return [], []

    rows = []
    missing_rows = []
    for v, g in df.groupby("variant"):
        metrics = set(g["metric"].dropna().astype(str).tolist())
        n_present = len(metrics)
        n_missing = len(all_metrics) - n_present
        rows.append(
            {
                "variant": str(v),
                "metrics_present": n_present,
                "metrics_missing": n_missing,
                "metrics_total": len(all_metrics),
            }
        )
        if n_missing:
            missing = [m for m in all_metrics if m not in metrics]
            missing_rows.append(
                {
                    "variant": str(v),
                    "missing_metrics": ", ".join(missing[:25]) + (" …" if len(missing) > 25 else ""),
                    "missing_count": n_missing,
                }
            )

    rows.sort(key=lambda r: (r["metrics_missing"], r["variant"]), reverse=True)
    missing_rows.sort(key=lambda r: (r["missing_count"], r["variant"]), reverse=True)
    return rows[:200], missing_rows[:200]


def _make_table(rows: List[dict], title: str, *, height: str = "320px") -> html.Div:
    if not rows:
        return panel([html.Div("No data.", className="muted")], title=title)

    cols = [{"name": k, "id": k} for k in rows[0].keys()]
    dt = DataTable(
        columns=cols,
        data=rows,
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_table={"overflowX": "auto", "maxHeight": height, "overflowY": "auto"},
        style_cell={"fontFamily": "system-ui", "fontSize": "12px", "padding": "6px", "textAlign": "left"},
        style_header={"fontWeight": "600"},
    )
    return panel([dt], title=title)


def _ensure_df(obj: Any) -> pd.DataFrame:
    if obj is None or not isinstance(obj, pd.DataFrame):
        return pd.DataFrame()
    return obj


def _compute(ctx) -> Dict[str, Any]:
    feats = _ensure_df(getattr(ctx, "features", None))
    pmf = _ensure_df(getattr(ctx, "pmf_df", None))
    cum = _ensure_df(getattr(ctx, "cum_df", None))
    replica = _ensure_df(getattr(ctx, "rmsf_df", None))
    conv = _ensure_df(getattr(ctx, "conv_df", None))
    r2p = _ensure_df(getattr(ctx, "rama2d_pooled_df", None))
    r2r = _ensure_df(getattr(ctx, "rama2d_perres_df", None))

    ts_err: str = ""
    ts_idx = _ensure_df(getattr(ctx, "ts_index_df", None))
    if ts_idx.empty:
        try:
            timeseries_dir = getattr(ctx, "timeseries_dir", None)
            data_dir = getattr(ctx, "data_dir", None)
            ts_idx = _ensure_df(discover_timeseries_files(data_dir, timeseries_dir))
            setattr(ctx, "ts_index_df", ts_idx)
        except Exception as e:
            ts_err = str(e)
            ts_idx = pd.DataFrame()

    overview = [
        _df_overview("features", feats),
        _df_overview("pmf", pmf),
        _df_overview("cumulative", cum),
        _df_overview("replica (rmsf_df)", replica),
        _df_overview("convergence", conv),
        _df_overview("rama2d_pooled", r2p),
        _df_overview("rama2d_perres", r2r),
        _df_overview("timeseries_index", ts_idx),
    ]

    feat_vars = set(feats["variant"].dropna().astype(str).tolist()) if "variant" in feats.columns else set()

    coverage = []
    for name, df in (
        ("pmf", pmf),
        ("cumulative", cum),
        ("replica", replica),
        ("convergence", conv),
        ("rama2d_pooled", r2p),
        ("rama2d_perres", r2r),
        ("timeseries_index", ts_idx),
    ):
        if df.empty or "variant" not in df.columns:
            missing = sorted(list(feat_vars)) if feat_vars else []
            coverage.append(
                {
                    "table": name,
                    "variants_in_features": len(feat_vars),
                    "variants_in_table": 0,
                    "missing_variants": ", ".join(missing[:25]) + (" …" if len(missing) > 25 else ""),
                    "extra_variants": "",
                }
            )
            continue
        vars_df = set(df["variant"].dropna().astype(str).tolist())
        missing, extra = _set_diff(feat_vars, vars_df)
        coverage.append(
            {
                "table": name,
                "variants_in_features": len(feat_vars),
                "variants_in_table": len(vars_df),
                "missing_variants": ", ".join(missing[:25]) + (" …" if len(missing) > 25 else ""),
                "extra_variants": ", ".join(extra[:25]) + (" …" if len(extra) > 25 else ""),
            }
        )

    feat_missing_cols, feat_missing_variants = _features_missing_tables(feats)
    pmf_cov_rows, pmf_missing_rows = _coverage_by_variant_metric(pmf)
    cum_cov_rows, cum_missing_rows = _coverage_by_variant_metric(cum)
    conv_cov_rows, conv_missing_rows = _coverage_by_variant_metric(conv)
    ts_cov_rows, ts_missing_rows = _coverage_by_variant_metric(ts_idx)

    return {
        "overview": overview,
        "variant_coverage": coverage,
        "features_missing_cols": feat_missing_cols,
        "features_missing_variants": feat_missing_variants,
        "pmf_cov": pmf_cov_rows,
        "pmf_missing": pmf_missing_rows,
        "cum_cov": cum_cov_rows,
        "cum_missing": cum_missing_rows,
        "conv_cov": conv_cov_rows,
        "conv_missing": conv_missing_rows,
        "timeseries_error": ts_err,
        "timeseries_cov": ts_cov_rows,
        "timeseries_missing": ts_missing_rows,
    }


def _placeholder_content(message: str) -> html.Div:
    return panel([html.Div(message, className="muted")], title="Status")


def layout(ctx):
    return html.Div(
        [
            html.Div(
                [
                    html.H2("Data diagnosis", className="tab-title"),
                    html.Div(
                        "Quick sanity checks: how much data was loaded, where coverage is missing, and which columns/variants have NaNs.",
                        className="tab-subtitle",
                    ),
                ],
                className="tab-header",
            ),
            html.Div(
                [
                    html.Button("Recompute", id="diag-recompute", n_clicks=0, className="btn"),
                    html.Span(" ", style={"display": "inline-block", "width": "10px"}),
                    html.Span(
                        f"data_dir: {getattr(ctx, 'data_dir', '')} | timeseries_dir: {getattr(ctx, 'timeseries_dir', '')}",
                        className="muted",
                        style={"fontSize": "12px"},
                    ),
                ],
                style={"marginBottom": "10px"},
            ),
            dcc.Store(id="diag-store", data=None),
            dcc.Loading(html.Div(id="diag-content", children=_placeholder_content("Diagnosis is loaded on demand when you open this tab."))),
        ]
    )


def register_callbacks(app, ctx):
    @app.callback(
        Output("diag-store", "data"),
        Input("tabs", "value"),
        Input("diag-recompute", "n_clicks"),
        State("diag-store", "data"),
        prevent_initial_call=False,
    )
    def _ensure_or_recompute(active_tab, _n_clicks, current_data):
        should_recompute = active_tab == "diagnosis" and current_data is None
        if active_tab != "diagnosis" and not should_recompute:
            return no_update
        if active_tab != "diagnosis":
            return no_update
        return _compute(ctx)

    @app.callback(
        Output("diag-content", "children"),
        Input("tabs", "value"),
        Input("diag-store", "data"),
    )
    def _render(active_tab, data: Dict[str, Any]):
        if active_tab != "diagnosis":
            return _placeholder_content("Diagnosis is loaded on demand when you open this tab.")

        if not data:
            return _placeholder_content("Computing diagnosis …")

        data = data or {}
        return html.Div(
            [
                _make_table(data.get("overview", []), "Overview"),
                _make_table(data.get("variant_coverage", []), "Variant coverage (vs features)"),
                html.Div(
                    [
                        _make_table(data.get("features_missing_cols", []), "Features: columns with missing values (top 200)", height="420px"),
                        _make_table(data.get("features_missing_variants", []), "Features: variants with missing numeric values (top 200)", height="420px"),
                    ],
                    style={"display":"grid","gridTemplateColumns":"repeat(2, minmax(0, 1fr))","gap":"12px"},
                ),
                html.Div(
                    [
                        _make_table(data.get("pmf_cov", []), "PMF: metrics per variant (top 200)", height="420px"),
                        _make_table(data.get("pmf_missing", []), "PMF: missing metric names by variant (top 200)", height="420px"),
                    ],
                    style={"display":"grid","gridTemplateColumns":"repeat(2, minmax(0, 1fr))","gap":"12px"},
                ),
                html.Div(
                    [
                        _make_table(data.get("cum_cov", []), "Cumulative: metrics per variant (top 200)", height="420px"),
                        _make_table(data.get("cum_missing", []), "Cumulative: missing metric names by variant (top 200)", height="420px"),
                    ],
                    style={"display":"grid","gridTemplateColumns":"repeat(2, minmax(0, 1fr))","gap":"12px"},
                ),
                html.Div(
                    [
                        _make_table(data.get("conv_cov", []), "Convergence: metrics per variant (top 200)", height="420px"),
                        _make_table(data.get("conv_missing", []), "Convergence: missing metric names by variant (top 200)", height="420px"),
                    ],
                    style={"display":"grid","gridTemplateColumns":"repeat(2, minmax(0, 1fr))","gap":"12px"},
                ),
                html.Div(
                    [
                        panel(
                            [html.Div(data.get("timeseries_error") or "OK", className="muted" if not data.get("timeseries_error") else "error")],
                            title="Timeseries discovery",
                        ),
                    ],
                    style={"marginTop": "8px"},
                ),
                html.Div(
                    [
                        _make_table(data.get("timeseries_cov", []), "Timeseries index: metrics per variant (top 200)", height="420px"),
                        _make_table(data.get("timeseries_missing", []), "Timeseries index: missing metric names by variant (top 200)", height="420px"),
                    ],
                    style={"display":"grid","gridTemplateColumns":"repeat(2, minmax(0, 1fr))","gap":"12px"},
                ),

            ]
        )
