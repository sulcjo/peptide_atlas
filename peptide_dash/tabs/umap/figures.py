from .shared import *


def _template() -> str:
    """Simple template; no theme-mode dependency."""
    return "plotly_white"


def _error_fig(msg: str, details: Optional[str] = None) -> go.Figure:
    """Figure with centered message instead of silently failing."""
    txt = msg
    if details:
        txt = f"{msg}<br><span style='font-size:10px'>{details}</span>"
    fig = go.Figure()
    fig.add_annotation(
        text=txt,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        showarrow=False,
        align="center",
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(template=_template(), height=600, margin=dict(l=20, r=20, t=20, b=20))
    return fig




def _embedding_method_label(method: Optional[str]) -> str:
    m = str(method or "umap").strip().lower()
    if m in {"densmap", "dens-map", "dens_map", "dens"}:
        return "DensMAP"
    if m in {"isomap", "iso-map", "iso_map"}:
        return "Isomap"
    if m in {"diffusion", "diffusion-map", "diffusion_map", "diffmap", "dm"}:
        return "Diffusion Map"
    if m in {"phate", "phate-like", "phate_like"}:
        return "PHATE"
    return "UMAP"

def _dm_spectral_fig(dm_data: dict, theme: str = "plotly_white") -> html.Div:
    """Eigenvalue spectrum + Von Neumann Entropy curve for a diffusion map computation."""
    if not isinstance(dm_data, dict) or not dm_data:
        return html.Div()
    eigenvalues = dm_data.get("eigenvalues", [])
    vne_curve = dm_data.get("vne_curve", [])
    t_used = int(dm_data.get("t_used", 1))
    if not eigenvalues and not vne_curve:
        return html.Div()

    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Non-trivial eigenvalues λ", "Von Neumann Entropy vs t"),
    )
    if eigenvalues:
        idx = list(range(1, len(eigenvalues) + 1))
        fig.add_trace(go.Bar(x=idx, y=eigenvalues, name="λ", marker_color="steelblue", showlegend=False), row=1, col=1)
        fig.update_xaxes(title_text="Index", row=1, col=1)
        fig.update_yaxes(title_text="λ", row=1, col=1)
    if vne_curve:
        ts = [p[0] for p in vne_curve]
        vnes = [p[1] for p in vne_curve]
        fig.add_trace(go.Scatter(x=ts, y=vnes, mode="lines+markers", name="VNE", line=dict(color="darkorange"), showlegend=False), row=1, col=2)
        if t_used in ts:
            fig.add_vline(x=t_used, line_dash="dash", line_color="firebrick", annotation_text=f"t={t_used}", row=1, col=2)
        fig.update_xaxes(title_text="t", row=1, col=2)
        fig.update_yaxes(title_text="VNE(t)", row=1, col=2)
    fig.update_layout(
        template=theme or "plotly_white",
        height=220,
        margin=dict(l=45, r=20, t=40, b=30),
        title_text=f"Diffusion Map spectral diagnostics (t used = {t_used})",
        title_font_size=12,
    )
    return html.Div(dcc.Graph(figure=fig, config={"displaylogo": False}))


def _build_umap_figure(
    plot_df: pd.DataFrame,
    dims: int,
    colorby: Optional[str],
    label_col: Optional[str],
    gradient_data: Optional[dict] = None,
    uirevision: Optional[str] = None,
) -> go.Figure:
    """Scatter (2D/3D); plot-only points are overlaid, plot-selection is highlighted."""
    if plot_df is None or plot_df.empty:
        return _error_fig("UMAP produced no data.")

    if "UMAP1" not in plot_df.columns or "UMAP2" not in plot_df.columns:
        return _error_fig("UMAP coordinates missing.")

    dims = int(dims or 2)

    color_col = None
    if colorby and colorby in plot_df.columns:
        color_col = colorby
    elif label_col and label_col in plot_df.columns:
        color_col = label_col

    has_role = "role" in plot_df.columns
    if has_role:
        df_plot_only = plot_df[plot_df["role"].astype(str).eq("plot-only")].copy()
        df_fit_plot = plot_df[plot_df["role"].astype(str).eq("fit+plot")].copy()
        base_df = plot_df[~plot_df["role"].astype(str).eq("plot-only")].copy()
        if base_df.empty:
            base_df = plot_df.copy()
    else:
        base_df = plot_df
        df_plot_only = plot_df.iloc[0:0].copy()
        df_fit_plot = plot_df.iloc[0:0].copy()

    hover_data: List[str] = []
    if has_role:
        hover_data.append("role")
    if "dbscan_cluster" in plot_df.columns:
        hover_data.append("dbscan_cluster")
    if label_col and label_col in plot_df.columns:
        hover_data.append(label_col)
    if color_col and color_col not in hover_data:
        hover_data.append(color_col)
    hover_data = list(dict.fromkeys([h for h in hover_data if h]))

    category_orders = None
    if color_col == "dbscan_cluster" and "dbscan_cluster" in plot_df.columns:
        cats = plot_df["dbscan_cluster"].astype(str).unique().tolist()

        def _sort_key(s: str):
            if s.startswith("C"):
                try:
                    return (0, int(s[1:]))
                except Exception:
                    return (0, 10**9)
            if s == "noise":
                return (1, 0)
            if s == "unclustered":
                return (2, 0)
            if s == "unavailable":
                return (3, 0)
            return (4, 0)

        cats = sorted(cats, key=_sort_key)
        category_orders = {"dbscan_cluster": cats}

    def _outline_marker(size: int) -> dict:
        # Filled marker without black outline; avoids the distracting ring overlay.
        return dict(size=size, opacity=0.85, line=dict(width=0))

    # The fit+plot overlay is single-color; do not draw it over an active colormap.
    show_fit_highlight = color_col is None

    if dims == 3 and "UMAP3" in plot_df.columns:
        fig = px.scatter_3d(
            base_df,
            x="UMAP1",
            y="UMAP2",
            z="UMAP3",
            color=color_col,
            hover_name="variant" if "variant" in base_df.columns else None,
            hover_data=hover_data or None,
            custom_data=["variant"] if "variant" in base_df.columns else None,
            category_orders=category_orders,
        )
        fig.update_layout(template=_template(), height=720, uirevision=uirevision or "stable")

        if not df_plot_only.empty:
            fig.add_trace(
                go.Scatter3d(
                    x=df_plot_only["UMAP1"],
                    y=df_plot_only["UMAP2"],
                    z=df_plot_only["UMAP3"],
                    mode="markers",
                    name="plot-only",
                    marker=dict(symbol="x", size=9, color="black", opacity=0.95),
                    customdata=df_plot_only.get("variant", pd.Series([""] * len(df_plot_only))).to_numpy(),
                    hovertemplate="variant=%{customdata}<br>role=plot-only<extra></extra>",
                    showlegend=True,
                )
            )

        if show_fit_highlight and not df_fit_plot.empty:
            fig.add_trace(
                go.Scatter3d(
                    x=df_fit_plot["UMAP1"],
                    y=df_fit_plot["UMAP2"],
                    z=df_fit_plot["UMAP3"],
                    mode="markers",
                    name="fit+plot (highlight)",
                    marker=_outline_marker(10),
                    customdata=df_fit_plot.get("variant", pd.Series([""] * len(df_fit_plot))).to_numpy(),
                    hovertemplate="variant=%{customdata}<br>role=fit+plot<extra></extra>",
                    showlegend=True,
                )
            )
    else:
        fig = px.scatter(
            base_df,
            x="UMAP1",
            y="UMAP2",
            color=color_col,
            hover_name="variant" if "variant" in base_df.columns else None,
            hover_data=hover_data or None,
            custom_data=["variant"] if "variant" in base_df.columns else None,
            category_orders=category_orders,
        )
        fig.update_layout(template=_template(), height=720, uirevision=uirevision or "stable")

        if not df_plot_only.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_plot_only["UMAP1"],
                    y=df_plot_only["UMAP2"],
                    mode="markers",
                    name="plot-only",
                    marker=dict(symbol="x", size=11, color="black", opacity=0.95),
                    customdata=df_plot_only.get("variant", pd.Series([""] * len(df_plot_only))).to_numpy(),
                    hovertemplate="variant=%{customdata}<br>role=plot-only<extra></extra>",
                    showlegend=True,
                )
            )

        if show_fit_highlight and not df_fit_plot.empty:
            fig.add_trace(
                go.Scatter(
                    x=df_fit_plot["UMAP1"],
                    y=df_fit_plot["UMAP2"],
                    mode="markers",
                    name="fit+plot (highlight)",
                    marker=_outline_marker(12),
                    customdata=df_fit_plot.get("variant", pd.Series([""] * len(df_fit_plot))).to_numpy(),
                    hovertemplate="variant=%{customdata}<br>role=fit+plot<extra></extra>",
                    showlegend=True,
                )
            )

        if gradient_data and isinstance(gradient_data, dict) and gradient_data.get("arrows"):
            cx = float(plot_df["UMAP1"].mean())
            cy = float(plot_df["UMAP2"].mean())
            u1_range = float(plot_df["UMAP1"].max() - plot_df["UMAP1"].min())
            u2_range = float(plot_df["UMAP2"].max() - plot_df["UMAP2"].min())
            rx = u1_range * 0.30 if u1_range > 0 else 1.0
            ry = u2_range * 0.30 if u2_range > 0 else 1.0
            for arrow in gradient_data["arrows"]:
                feat = str(arrow.get("feature", ""))
                ex = cx + float(arrow.get("rho_x", 0)) * rx
                ey = cy + float(arrow.get("rho_y", 0)) * ry
                fig.add_annotation(
                    x=ex, y=ey, ax=cx, ay=cy,
                    xref="x", yref="y", axref="x", ayref="y",
                    text=feat,
                    showarrow=True,
                    arrowhead=2,
                    arrowwidth=1.5,
                    arrowcolor="rgba(80,80,80,0.75)",
                    font=dict(size=9),
                    bgcolor="rgba(255,255,255,0.55)",
                )

    return fig


def _color_options_from_features(feats_df: pd.DataFrame) -> list[dict]:
    """Build robust color dropdown options from the current feature table."""
    opts: list[dict] = []
    if isinstance(feats_df, pd.DataFrame) and not feats_df.empty:
        for c in feats_df.columns:
            if c == "variant" or is_excluded_feature_column(str(c)):
                continue
            opts.append({"label": prettify_column_label(str(c)), "value": str(c)})
    if not any(o.get("value") == "dbscan_cluster" for o in opts):
        opts.append({"label": "DBSCAN cluster", "value": "dbscan_cluster"})
    return opts


def _trajectory_feature_options_from_features(feats_df: pd.DataFrame) -> list[dict]:
    """Numeric scalar features usable as physical-value trajectory coordinates."""
    opts: list[dict] = []
    if isinstance(feats_df, pd.DataFrame) and not feats_df.empty:
        for c in feats_df.columns:
            if c == "variant" or is_excluded_feature_column(str(c)):
                continue
            try:
                if pd.api.types.is_numeric_dtype(feats_df[c]):
                    opts.append({"label": prettify_column_label(str(c)), "value": str(c)})
            except Exception:
                pass
    def _rank(o: dict):
        v = str(o.get("value", ""))
        if v.endswith("__global_basin_min_x"):
            return (0, v)
        if "global_basin_min_x" in v:
            return (1, v)
        if v.endswith("__secondary_basin_min_x"):
            return (2, v)
        if "basin" in v:
            return (3, v)
        return (4, v)
    return sorted(opts, key=_rank)


def _merge_feature_columns_for_embedding(emb_df: pd.DataFrame, feats_df: pd.DataFrame) -> pd.DataFrame:
    """Attach current scalar/categorical feature columns to an embedding table by variant."""
    if not isinstance(emb_df, pd.DataFrame) or emb_df.empty or "variant" not in emb_df.columns:
        return emb_df
    if not isinstance(feats_df, pd.DataFrame) or feats_df.empty or "variant" not in feats_df.columns:
        return emb_df
    out = emb_df.copy()
    out["variant"] = out["variant"].astype(str)
    base = feats_df.loc[:, ~feats_df.columns.duplicated()].copy()
    base["variant"] = base["variant"].astype(str)
    base = base.drop_duplicates(subset=["variant"], keep="first")
    add_cols = [c for c in base.columns if c == "variant" or c not in out.columns]
    if len(add_cols) <= 1:
        return out
    try:
        return out.merge(base[add_cols], on="variant", how="left")
    except Exception:
        return out
