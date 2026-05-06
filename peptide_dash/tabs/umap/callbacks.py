from .shared import *
from .analytics import (_add_dbscan_clusters, _pca_loading_tables, _family_pca_loading_table_from_Xdf, _corr_tables_for_axes, _global_top_corr_for_axes)
from .data_access import (_ctx_features_df, _ctx_pmf_df, _ctx_pmf_metrics, _ctx_pmf_variants, _variant_values, _variant_dropdown_options, _umap_data_status_children)
from .embedding import _compute_umap_embedding
from .figures import (_template, _error_fig, _build_umap_figure, _merge_feature_columns_for_embedding, _color_options_from_features, _trajectory_feature_options_from_features)
from .input_matrix import _umap_source_mode
from .layout import make_layout
from .pmf_landscape import _pmf_temperature_from_frame, _pmf_unit_factor_label, _pmf_y_column
from .selection import _split_variant_tokens, _match_variants, _dedupe_preserve_order
from .sequence_grammar import (_variants_from_plot_points, _variant_from_plot_point, _dataset_background_variants, _build_region_sequence_grammar_figure, _build_trajectory_sequence_grammar_figure, _build_sequence_language_cluster_outputs)
from .stores import (_embedding_cache_params, _embedding_cache_key, _load_embedding_cache_payload, _plot_df_from_embedding_cache, _embedding_array_from_plot_df, _save_embedding_cache, _cache_summary_text, _plot_df_store_payload, _plot_df_from_store)
from .trajectory import _build_trajectory_figure, _build_residue_composition_figure



def _embedding_dimensionality_diagnostics(Xdf: Optional[pd.DataFrame], emb: Optional[np.ndarray], dims: int) -> html.Div:
    """Compact diagnostics for intrinsic dimensionality and displayed-map information.

    Effective dimensionality is the participation ratio of the standardized
    input matrix singular spectrum.  The displayed-map percentage is the
    fraction of standardized input variance linearly reconstructable from the
    current displayed coordinates.  For nonlinear embeddings such as UMAP this
    is not a formal UMAP explained-variance ratio; it is an interpretable
    low-dimensional information diagnostic.
    """
    try:
        if Xdf is None or not isinstance(Xdf, pd.DataFrame) or Xdf.empty:
            return html.Div(
                html.B("Dimensionality: unavailable for cached embedding; press Recalculate to recompute input-matrix diagnostics."),
                className="text-muted",
            )
        if emb is None:
            return html.Div()
        E = np.asarray(emb, dtype=float)
        if E.ndim != 2 or E.shape[0] != Xdf.shape[0] or E.shape[0] < 3:
            return html.Div()
        X = Xdf.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
        finite_counts = X.notna().sum(axis=0)
        keep = finite_counts[finite_counts >= max(3, min(5, len(X)))].index.tolist()
        if len(keep) < 2:
            keep = finite_counts[finite_counts >= 3].index.tolist()
        if len(keep) < 2:
            return html.Div(html.B("Dimensionality: not enough finite input columns for diagnostics."), className="text-muted")
        X = X[keep].copy()
        med = X.median(axis=0, skipna=True).fillna(0.0)
        X = X.fillna(med).fillna(0.0)
        sd = X.std(axis=0, ddof=1).replace(0.0, np.nan)
        keep = sd[sd.notna() & np.isfinite(sd) & (sd > 0)].index.tolist()
        if len(keep) < 2:
            return html.Div(html.B("Dimensionality: input columns are constant after filtering."), className="text-muted")
        X = X[keep]
        Z = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1).replace(0.0, 1.0)
        Z = Z.replace([np.inf, -np.inf], np.nan).fillna(0.0).to_numpy(dtype=float)
        Z = Z - np.mean(Z, axis=0, keepdims=True)
        if not np.isfinite(Z).all() or Z.shape[0] < 3 or Z.shape[1] < 2:
            return html.Div()
        s = np.linalg.svd(Z, full_matrices=False, compute_uv=False)
        eig = np.asarray(s ** 2, dtype=float)
        eig = eig[np.isfinite(eig) & (eig > 1e-12)]
        if eig.size == 0:
            return html.Div()
        total = float(np.sum(eig))
        eff_dim = float((total ** 2) / np.sum(eig ** 2))
        cum = np.cumsum(eig) / max(total, 1e-12)
        pca90 = int(np.searchsorted(cum, 0.90) + 1)
        pca95 = int(np.searchsorted(cum, 0.95) + 1)
        dshown = max(1, min(int(dims or E.shape[1] or 2), E.shape[1]))
        pca_disp = float(np.sum(eig[: min(dshown, eig.size)]) / max(total, 1e-12))

        # How much standardized input variance is linearly recoverable from the
        # displayed map coordinates?  This is a conservative, map-specific R^2,
        # not a formal nonlinear UMAP explained-variance ratio.
        G = E[:, :dshown].astype(float, copy=False)
        good = np.isfinite(G).all(axis=1) & np.isfinite(Z).all(axis=1)
        map_r2 = float("nan")
        if int(np.sum(good)) >= max(3, dshown + 2):
            Gg = G[good]
            Zg = Z[good]
            Gg = Gg - np.mean(Gg, axis=0, keepdims=True)
            if np.linalg.matrix_rank(Gg) >= 1:
                A = np.column_stack([np.ones(Gg.shape[0]), Gg])
                coef, *_ = np.linalg.lstsq(A, Zg, rcond=None)
                pred = A @ coef
                sse = float(np.sum((Zg - pred) ** 2))
                sst = float(np.sum((Zg - np.mean(Zg, axis=0, keepdims=True)) ** 2))
                if sst > 0:
                    map_r2 = max(0.0, min(1.0, 1.0 - sse / sst))
        map_txt = "n/a" if not np.isfinite(map_r2) else f"{100.0 * map_r2:.1f}%"
        txt = (
            f"Effective input dimensionality: {eff_dim:.1f} "
            f"(90% PCA: {pca90} dims; 95% PCA: {pca95} dims). "
            f"Displayed {dshown}D map linear variance: {map_txt} "
            f"(PCA {dshown}D upper bound: {100.0 * pca_disp:.1f}%)."
        )
        return html.Div(
            [
                html.B(txt),
                html.Div(
                    "Map variance is a linear reconstruction diagnostic from the displayed coordinates, not a formal nonlinear UMAP explained-variance ratio.",
                    className="text-muted",
                    style={"fontSize": "0.78em"},
                ),
            ],
            className="mb-2",
        )
    except Exception as e:
        return html.Div(html.B(f"Dimensionality diagnostics failed: {e}"), className="text-muted")

def _register_callbacks_with_data(app: dash.Dash, ctx: Any, feats_df: pd.DataFrame) -> None:
    """
    Callbacks:
      1) Bulk-selection helpers for fit/plot variant selectors.
      2) Main compute callback (fit on subset, project others).
    """
    from ..shared import apply_theme

    @app.callback(
        Output("umap-help-modal", "is_open"),
        Input("umap-help-open", "n_clicks"),
        Input("umap-help-close", "n_clicks"),
        State("umap-help-modal", "is_open"),
        prevent_initial_call=True,
    )
    def _toggle_umap_help(open_clicks, close_clicks, is_open):
        from dash import callback_context as _dcc_ctx
        if not _dcc_ctx.triggered:
            return bool(is_open)
        trig = _dcc_ctx.triggered[0]["prop_id"].split(".")[0]
        if trig == "umap-help-open" and open_clicks:
            return True
        if trig == "umap-help-close" and close_clicks:
            return False
        return bool(is_open)


    # --- Small UI helpers: disable dropdowns when "Use all" is enabled ---
    @app.callback(
        Output("umap-fit-variants", "disabled"),
        Input("umap-fit-use-all", "value"),
    )
    def _toggle_fit_disabled(v):
        return "all" in (v or [])

    @app.callback(
        Output("umap-plot-variants", "disabled"),
        Input("umap-plot-use-all", "value"),
    )
    def _toggle_plot_disabled(v):
        return "all" in (v or [])

    # --- Bulk selection: FIT variants ---
    @app.callback(
        Output("umap-fit-variants", "value"),
        Input("umap-fit-filter-set", "n_clicks"),
        Input("umap-fit-filter-add", "n_clicks"),
        Input("umap-fit-filter-remove", "n_clicks"),
        Input("umap-fit-invert", "n_clicks"),
        Input("umap-fit-clear", "n_clicks"),
        Input("umap-fit-paste-set", "n_clicks"),
        Input("umap-fit-paste-add", "n_clicks"),
        State("umap-fit-filter", "value"),
        State("umap-fit-paste", "value"),
        State("umap-fit-variants", "value"),
        State("umap-variants-universe", "data"),
        prevent_initial_call=True,
    )
    def _bulk_fit(
        n_set, n_add, n_rem, n_inv, n_clear, n_p_set, n_p_add,
        filt, pasted, current, universe
    ):
        ctx = dash.callback_context
        trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        universe = [str(v) for v in (universe or [])]
        u_set = set(universe)
        current = [str(v) for v in (current or []) if str(v) in u_set]

        if trig == "umap-fit-clear":
            return []

        if trig == "umap-fit-invert":
            cur_set = set(current)
            return [v for v in universe if v not in cur_set]

        if trig in {"umap-fit-filter-set", "umap-fit-filter-add", "umap-fit-filter-remove"}:
            matches = [v for v in _match_variants(filt, universe) if v in u_set]
            mset = set(matches)
            if trig == "umap-fit-filter-set":
                return _dedupe_preserve_order(matches)
            if trig == "umap-fit-filter-add":
                return _dedupe_preserve_order(current + matches)
            if trig == "umap-fit-filter-remove":
                return [v for v in current if v not in mset]

        if trig in {"umap-fit-paste-set", "umap-fit-paste-add"}:
            toks = _split_variant_tokens(pasted)
            toks = [t for t in toks if t in u_set]
            if trig == "umap-fit-paste-set":
                return _dedupe_preserve_order(toks)
            return _dedupe_preserve_order(current + toks)

        return current

    # --- Bulk selection: PLOT variants ---
    @app.callback(
        Output("umap-plot-variants", "value"),
        Input("umap-plot-filter-set", "n_clicks"),
        Input("umap-plot-filter-add", "n_clicks"),
        Input("umap-plot-filter-remove", "n_clicks"),
        Input("umap-plot-invert", "n_clicks"),
        Input("umap-plot-clear", "n_clicks"),
        Input("umap-plot-paste-set", "n_clicks"),
        Input("umap-plot-paste-add", "n_clicks"),
        State("umap-plot-filter", "value"),
        State("umap-plot-paste", "value"),
        State("umap-plot-variants", "value"),
        State("umap-variants-universe", "data"),
        prevent_initial_call=True,
    )
    def _bulk_plot(
        n_set, n_add, n_rem, n_inv, n_clear, n_p_set, n_p_add,
        filt, pasted, current, universe
    ):
        ctx = dash.callback_context
        trig = ctx.triggered[0]["prop_id"].split(".")[0] if ctx.triggered else None

        universe = [str(v) for v in (universe or [])]
        u_set = set(universe)
        current = [str(v) for v in (current or []) if str(v) in u_set]

        if trig == "umap-plot-clear":
            return []

        if trig == "umap-plot-invert":
            cur_set = set(current)
            return [v for v in universe if v not in cur_set]

        if trig in {"umap-plot-filter-set", "umap-plot-filter-add", "umap-plot-filter-remove"}:
            matches = [v for v in _match_variants(filt, universe) if v in u_set]
            mset = set(matches)
            if trig == "umap-plot-filter-set":
                return _dedupe_preserve_order(matches)
            if trig == "umap-plot-filter-add":
                return _dedupe_preserve_order(current + matches)
            if trig == "umap-plot-filter-remove":
                return [v for v in current if v not in mset]

        if trig in {"umap-plot-paste-set", "umap-plot-paste-add"}:
            toks = _split_variant_tokens(pasted)
            toks = [t for t in toks if t in u_set]
            if trig == "umap-plot-paste-set":
                return _dedupe_preserve_order(toks)
            return _dedupe_preserve_order(current + toks)

        return current

    @app.callback(
        Output("umap-pmf-metrics", "options"),
        Output("umap-pmf-metrics", "value"),
        Output("umap-variants-universe", "data"),
        Output("umap-fit-variants", "options"),
        Output("umap-plot-variants", "options"),
        Output("umap-trajectory-anchors", "options"),
        Output("umap-color", "options"),
        Output("umap-trajectory-feature", "options"),
        Output("umap-data-status", "children"),
        Input("tabs", "value"),
        State("umap-pmf-metrics", "value"),
        prevent_initial_call=False,
    )
    def _populate_umap_data_controls(active_tab, current_metrics):
        if str(active_tab) != "umap":
            raise PreventUpdate

        feats_now = _ctx_features_df(ctx, feats_df)

        metrics = _ctx_pmf_metrics(ctx)
        metric_options = [{"label": m, "value": m} for m in metrics]

        current = [str(v) for v in (current_metrics or [])]
        metric_value = [v for v in current if v in metrics]
        # Empty selection intentionally means "all PMF metrics" (same UX as PCA).

        variants = _variant_values(feats_now, pd.DataFrame())
        variants = sorted(set(variants).union(_ctx_pmf_variants(ctx)))
        variant_options = _variant_dropdown_options(variants)

        n_feat_rows = int(len(feats_now)) if isinstance(feats_now, pd.DataFrame) else 0
        n_feat_vars = int(feats_now["variant"].astype(str).nunique()) if isinstance(feats_now, pd.DataFrame) and (not feats_now.empty) and ("variant" in feats_now.columns) else 0
        status = html.Small(
            f"features: {n_feat_rows} rows / {n_feat_vars} variants; "
            f"pmf variants indexed: {len(_ctx_pmf_variants(ctx))}; "
            f"pmf metrics: {len(metrics)}",
            className="text-muted",
        )

        return metric_options, metric_value, variants, variant_options, variant_options, variant_options, _color_options_from_features(feats_now), _trajectory_feature_options_from_features(feats_now), status


    @app.callback(
        Output("umap-nn-label", "children"),
        Output("umap-min-dist-label", "children"),
        Output("umap-method-param-help", "children"),
        Output("umap-min-dist", "min"),
        Output("umap-min-dist", "max"),
        Output("umap-min-dist", "step"),
        Output("umap-min-dist", "value"),
        Input("umap-embedding", "value"),
        State("umap-min-dist", "value"),
        prevent_initial_call=False,
    )
    def _update_embedding_parameter_labels(embedding_method, current_second_param):
        method = str(embedding_method or "umap").lower()
        try:
            cur = float(current_second_param)
        except Exception:
            cur = float("nan")

        if method == "densmap":
            val = cur if np.isfinite(cur) and 0.0 <= cur <= 1.0 else 0.12
            return (
                "neighbors",
                "min_dist",
                "DensMAP uses the UMAP graph plus density preservation; min_dist controls visual packing. DensMAP-specific density controls are below.",
                0.0, 1.0, 0.01, val,
            )
        if method == "isomap":
            val = cur if np.isfinite(cur) and 0.0 <= cur <= 1.0 else 0.0
            return (
                "geodesic kNN",
                "unused",
                "Isomap uses the neighbor count to build a geodesic-distance graph. The second parameter is ignored for Isomap.",
                0.0, 1.0, 0.01, val,
            )
        if method in {"diffusion", "diffusion-map", "diffusion_map", "diffmap", "dm"}:
            val = int(round(cur)) if np.isfinite(cur) else 1
            val = max(0, min(10, val))
            return (
                "diffusion kNN",
                "time t",
                "Diffusion Map uses kNN to build a random-walk graph; time t controls how many diffusion steps are encoded in the eigenvector coordinates.",
                0, 10, 1, val,
            )
        if method in {"phate", "phate-like", "phate_like"}:
            val = int(round(cur)) if np.isfinite(cur) else 3
            val = max(1, min(20, val))
            return (
                "diffusion kNN",
                "time t",
                "PHATE uses a diffusion graph and potential distance -log(P^t); time t controls smoothing along progressions/branches.",
                1, 20, 1, val,
            )
        val = cur if np.isfinite(cur) and 0.0 <= cur <= 1.0 else 0.12
        return (
            "neighbors",
            "min_dist",
            "UMAP uses neighbors to build the local graph; min_dist controls how tightly points may pack in the displayed map.",
            0.0, 1.0, 0.01, val,
        )


    # --- Main compute callback ---
    @app.callback(
        Output("umap-graph", "figure"),
        Output("umap-corr-table", "children"),
        Output("umap-metrics", "children"),
        Output("umap-embedding-data", "data"),
        Input("umap-recalc", "n_clicks"),
        Input("umap-cache-quickload", "n_clicks"),
        State("umap-preset", "value"),
        State("umap-dims", "value"),
        State("umap-nn", "value"),
        State("umap-min-dist", "value"),
        State("umap-metric", "value"),
        State("umap-input-source", "value"),
        State("umap-embedding", "value"),
        State("umap-dens-lambda", "value"),
        State("umap-dens-frac", "value"),
        State("umap-dens-var-shift", "value"),
        State("umap-pmf-metrics", "value"),
        State("umap-pmf-repr", "value"),
        State("umap-fit-use-all", "value"),
        State("umap-fit-variants", "value"),
        State("umap-plot-use-all", "value"),
        State("umap-plot-variants", "value"),
        State("umap-plot-union", "value"),
        State("umap-dbscan-enable", "value"),
        State("umap-dbscan-on", "value"),
        State("umap-dbscan-eps", "value"),
        State("umap-dbscan-min-samples", "value"),
        State("umap-dbscan-color", "value"),
        State("umap-color", "value"),
        State("umap-label", "value"),
        State("umap-stability-runs", "value"),
        State("umap-stability-k", "value"),
        Input("theme-store", "data"),
    )
    def _umap_callback(
        n_clicks,
        quickload_clicks,
        preset,
        dims,
        nn,
        min_dist,
        metric,
        input_source,
        embedding_method,
        dens_lambda,
        dens_frac,
        dens_var_shift,
        pmf_metrics_sel,
        pmf_repr,
        fit_use_all_val,
        fit_variants,
        plot_use_all_val,
        plot_variants,
        plot_union_val,
        dbscan_enable,
        dbscan_on,
        dbscan_eps,
        dbscan_min_samples,
        dbscan_color,
        colorby,
        label_col,
        stability_runs=None,
        stability_k=None,
        theme=None,
    ):
        from dash import callback_context as _dcc_ctx
        triggered = {t.get("prop_id", "") for t in (_dcc_ctx.triggered or [])}
        quickload_requested = "umap-cache-quickload.n_clicks" in triggered

        # Before first click
        if not n_clicks and not quickload_requested:
            fig = _error_fig("Press 'Recalculate UMAP' to compute embedding, or Quickload last to restore the previous cached map.")
            return apply_theme(fig, theme), html.Div(), html.Div(), {}

        fit_use_all = "all" in (fit_use_all_val or [])
        plot_use_all = "all" in (plot_use_all_val or [])
        plot_union = "on" in (plot_union_val or [])

        cached_payload = None
        cache_note = ""
        cache_params = None
        cache_key = None
        pmf_now = pd.DataFrame()

        # Fast path: Quickload must not load/fingerprint PMF data or compute a
        # parameter cache key. It should only read last_embedding.json + cached
        # embedding coordinates, then optionally merge lightweight feature columns
        # for coloring. The trajectory PMFs are loaded only after pressing
        # "Update trajectory" below.
        if quickload_requested:
            cached_payload, cache_note = _load_embedding_cache_payload(ctx, last=True)
            feats_now = _ctx_features_df(ctx, feats_df)
        else:
            feats_now = _ctx_features_df(ctx, feats_df)
            pmf_now = _ctx_pmf_df(ctx)
            cache_params = _embedding_cache_params(
                preset=preset, dims=int(dims or 2), nn=nn, min_dist=min_dist, metric=metric,
                input_source=input_source, embedding_method=embedding_method, dens_lambda=dens_lambda,
                dens_frac=dens_frac, dens_var_shift=dens_var_shift, pmf_metrics_sel=pmf_metrics_sel or [],
                pmf_repr=pmf_repr, fit_use_all=fit_use_all, plot_use_all=plot_use_all,
                fit_variants=fit_variants or [], plot_variants=plot_variants or [], plot_union=plot_union,
                stability_runs=stability_runs, stability_k=stability_k,
            )
            cache_key = _embedding_cache_key(cache_params, feats_now, pmf_now)

        if cached_payload is not None:
            plot_df = _plot_df_from_embedding_cache(cached_payload)
            if plot_df.empty:
                fig = _error_fig("Cached embedding is empty.", cache_note)
                return apply_theme(fig, theme), html.Div(), html.Div(html.Small(cache_note)), {}
            cached_params = cached_payload.get("params", {}) if isinstance(cached_payload, dict) else {}
            try:
                dims = int(cached_params.get("dims", dims or 2) or 2)
            except Exception:
                dims = int(dims or 2)
            embedding_method = str(cached_params.get("embedding_method", embedding_method or "umap"))
            emb = _embedding_array_from_plot_df(plot_df, int(dims or 2))
            Xdf = None
            colnames = []
            pca_loadings = None
            info = str(cached_payload.get("info", "cached embedding"))
            info = f"{info}; cache: {cache_note}"
        else:
            try:
                (
                    plot_df,
                    emb,
                    Xdf,
                    colnames,
                    info,
                    pca_loadings,
                ) = _compute_umap_embedding(
                pmf_df=pmf_now,
                feats_df=feats_now,
                preset=preset,
                dims=int(dims or 2),
                nn=nn,
                min_dist=min_dist,
                metric=metric,
                input_source=str(input_source or UMAP_SOURCE_BASIC),
                embedding_method=str(embedding_method or "umap"),
                dens_lambda=dens_lambda,
                dens_frac=dens_frac,
                dens_var_shift=dens_var_shift,
                pmf_metrics_sel=list(pmf_metrics_sel or []),
                pmf_repr=str(pmf_repr or "P"),
                fit_use_all=fit_use_all,
                plot_use_all=plot_use_all,
                fit_variants=list(fit_variants or []),
                plot_variants=list(plot_variants or []),
                plot_union=plot_union,
                stability_runs=stability_runs,
                stability_k=stability_k,
            )
            except Exception as e:
                fig = _error_fig("UMAP error (exception)", str(e))
                return apply_theme(fig, theme), html.Div(html.Small("UMAP raised an exception.")), html.Div(), {}

            if plot_df is None:
                fig = _error_fig("UMAP could not be computed.", info)
                return apply_theme(fig, theme), html.Div(), html.Div(html.Small(info)), {}

            saved_path = _save_embedding_cache(ctx, cache_key, plot_df, str(info or ""), cache_params)
            info = f"{info}; {_cache_summary_text(saved_path, cache_key)}"


        # Attach current feature columns after recompute/quickload so scalar color-by
        # (e.g. metric__global_basin_min_x) works reliably.
        plot_df = _merge_feature_columns_for_embedding(plot_df, feats_now)

        # Optional DBSCAN clustering on the embedding
        dbscan_enabled = "on" in (dbscan_enable or [])
        cluster_summary_df = None
        if dbscan_enabled:
            plot_df, db_note, cluster_summary_df = _add_dbscan_clusters(
                plot_df=plot_df,
                dims=int(dims or 2),
                enabled=True,
                eps=dbscan_eps,
                min_samples=dbscan_min_samples,
                cluster_on=str(dbscan_on or "plotted"),
            )
            if db_note:
                info = f"{info}; {db_note}"

        # Cluster-based coloring (optional override)
        if dbscan_enabled and ("on" in (dbscan_color or [])) and ("dbscan_cluster" in plot_df.columns):
            colorby = "dbscan_cluster"

        # Main figure
        fig = _build_umap_figure(plot_df, dims=int(dims or 2), colorby=colorby, label_col=label_col)

        # Analytics
        corr_div = html.Div()
        if Xdf is not None and len(colnames) > 0 and emb.shape[0] == Xdf.shape[0]:
            try:
                # PCA feature loadings (PCA is fitted on the fit-set)
                pca_tables = _pca_loading_tables(pca_loadings, colnames, top_n=10)
                pca_blocks = []
                for i, df_pc in enumerate(pca_tables, start=1):
                    for col in df_pc.columns:
                        if "Feature" in str(col):
                            df_pc[col] = df_pc[col].map(prettify_column_label)

                    pca_blocks.append(
                        html.Div(
                            [
                                html.H6(f"PCA feature loadings – PC{i}"),
                                dbc.Table.from_dataframe(
                                    df_pc.round(4),
                                    striped=True,
                                    bordered=True,
                                    hover=True,
                                    size="sm",
                                ),
                            ],
                            className="mb-3",
                        )
                    )

                fam_pca_df = _family_pca_loading_table_from_Xdf(
                    Xdf, colnames, top_n=5, max_pcs_per_family=1
                )
                fam_pca_block = html.Div()
                if fam_pca_df is not None and not fam_pca_df.empty:
                    for col in fam_pca_df.columns:
                        if "Feature" in str(col):
                            fam_pca_df[col] = fam_pca_df[col].map(prettify_column_label)
                    fam_pca_block = html.Div(
                        [
                            html.H5("Family vectors – original metric contributions (per-family PCA)"),
                            html.Small(
                                "For each family, PC1 is a 'family vector'; this table shows which original "
                                "metrics contribute most to it."
                            ),
                                                        dbc.Table.from_dataframe(
                                fam_pca_df.round(4),
                                striped=True,
                                bordered=True,
                                hover=True,
                                size="sm",
                            ),
                        ],
                        className="mb-3",
                    )

                fam_tables = _corr_tables_for_axes(emb, colnames, Xdf, per_family_top_n=5)
                fam_blocks = []
                for i, t in enumerate(fam_tables, start=1):
                    fam_blocks.append(
                        html.Div(
                            [
                                html.H6(f"UMAP axis {i}: per-family strongest bins (top 5 per family)"),
                                dbc.Table.from_dataframe(
                                    t.round(3),
                                    striped=True,
                                    bordered=True,
                                    hover=True,
                                    size="sm",
                                ),
                            ],
                            className="mb-3",
                        )
                    )

                global_tables = _global_top_corr_for_axes(emb, colnames, Xdf, top_n=5)
                global_cols = [
                    dbc.Col(
                        dbc.Table.from_dataframe(
                            t.round(3),
                            striped=True,
                            bordered=True,
                            hover=True,
                            size="sm",
                        ),
                        md=4,
                    )
                    for t in global_tables
                ]

                corr_div = html.Div(
                    [
                        html.H5("Original PCA – feature loadings (fit-set PCA)"),
                        *pca_blocks,
                        html.Hr(),
                        fam_pca_block,
                        html.Hr(),
                        html.H5("UMAP axes – PMF bin contributions (plotted set)"),
                        *fam_blocks,
                        html.Hr(),
                        html.H6("UMAP axes – global top 5 bin-level contributors"),
                        dbc.Row(global_cols),
                    ]
                )
            except Exception as e:
                corr_div = html.Div(html.Small(f"Analytics computation failed: {e}"))

        dim_diag = _embedding_dimensionality_diagnostics(Xdf, emb, int(dims or 2))
        metrics_children = [dim_diag, html.Small(info)]
        if cluster_summary_df is not None and not cluster_summary_df.empty:
            metrics_children.insert(
                0,
                html.Div(
                    [
                        html.H6("DBSCAN cluster sizes"),
                        dbc.Table.from_dataframe(
                            cluster_summary_df.round(3),
                            striped=True,
                            bordered=True,
                            hover=True,
                            size="sm",
                        ),
                    ],
                    className="mb-2",
                ),
            )
        metrics_div = html.Div(metrics_children)
        embedding_store = _plot_df_store_payload(plot_df, embedding_method, int(dims or 2))
        return apply_theme(fig, theme), corr_div, metrics_div, embedding_store

    # --- Fast recolor: update point colors without recomputing the embedding ---
    @app.callback(
        Output("umap-graph", "figure", allow_duplicate=True),
        Input("umap-color", "value"),
        Input("umap-label", "value"),
        State("umap-dims", "value"),
        State("umap-embedding-data", "data"),
        Input("theme-store", "data"),
        prevent_initial_call=True,
    )
    def _recolor_umap_from_store(colorby, label_col, dims, embedding_data, theme=None):
        plot_df = _plot_df_from_store(embedding_data)
        plot_df = _merge_feature_columns_for_embedding(plot_df, _ctx_features_df(ctx, feats_df))
        if plot_df.empty:
            raise PreventUpdate
        fig = _build_umap_figure(plot_df, dims=int(dims or 2), colorby=colorby, label_col=label_col)
        return apply_theme(fig, theme)


    @app.callback(
        Output("umap-sequence-region-graph", "figure"),
        Output("umap-sequence-region-status", "children"),
        Input("umap-graph", "selectedData"),
        Input("umap-graph", "clickData"),
        State("umap-embedding-data", "data"),
        Input("theme-store", "data"),
        prevent_initial_call=False,
    )
    def _update_umap_sequence_region(selected_data, click_data, embedding_data, theme=None):
        emb_df = _plot_df_from_store(embedding_data)
        emb_df = _merge_feature_columns_for_embedding(emb_df, _ctx_features_df(ctx, feats_df))
        all_variants = emb_df["variant"].dropna().astype(str).tolist() if isinstance(emb_df, pd.DataFrame) and not emb_df.empty and "variant" in emb_df.columns else []
        pts = (selected_data or {}).get("points", []) or []
        if pts:
            selected = _variants_from_plot_points(pts)
            source = "lasso/box selection"
        else:
            pts = (click_data or {}).get("points", [])[:1] if click_data else []
            selected = _variants_from_plot_points(pts)
            source = "click" if selected else "idle"
        if not selected:
            fig = _error_fig("Click or lasso-select points in the UMAP scatter above.")
            return apply_theme(fig, theme), "Sequence grammar idle: select a manifold region in the UMAP plot."
        feats_now = _ctx_features_df(ctx, feats_df)
        background = _dataset_background_variants(ctx, feats_now, fallback=all_variants)
        fig, status = _build_region_sequence_grammar_figure(selected, background, feats_now)
        return apply_theme(fig, theme), f"selection={source}; variants={len(selected)}. {status}"


    @app.callback(
        Output("umap-seq-cluster-graph", "figure"),
        Output("umap-seq-cluster-table", "children"),
        Output("umap-seq-cluster-status", "children"),
        Input("umap-seq-cluster-run", "n_clicks"),
        State("umap-embedding-data", "data"),
        State("umap-seq-cluster-mode", "value"),
        State("umap-seq-cluster-max-k", "value"),
        Input("theme-store", "data"),
        prevent_initial_call=False,
    )
    def _update_sequence_language_clusters(n_clicks, embedding_data, mode, max_k, theme=None):
        if not n_clicks:
            fig = _error_fig("Run sequence-language clustering after computing an embedding.")
            return apply_theme(fig, theme), html.Div(), "Sequence-language clustering idle: compute/quickload an embedding, then click Detect clusters."
        emb_df = _plot_df_from_store(embedding_data)
        emb_df = _merge_feature_columns_for_embedding(emb_df, _ctx_features_df(ctx, feats_df))
        feats_now = _ctx_features_df(ctx, feats_df)
        fig, table, status = _build_sequence_language_cluster_outputs(
            emb_df,
            feats_now,
            ctx,
            mode=str(mode or "all"),
            max_clusters=int(max_k or 8),
        )
        return apply_theme(fig, theme), table, status


    # --- PMF curve viewer callbacks (mirrors PCA tab behavior) ---

    def _curve_variant_list_text(variants):
        if not variants:
            return "No variants selected. Click points in the UMAP scatter above."
        shown = ", ".join(str(v) for v in variants[:12])
        suffix = f" ... +{len(variants) - 12} more" if len(variants) > 12 else ""
        return f"Selected ({len(variants)}): {shown}{suffix}"

    def _curve_variant_from_point(pt: dict) -> Optional[str]:
        return _variant_from_plot_point(pt)

    def _curve_variants_from_points(pts) -> list:
        return _variants_from_plot_points(pts)

    @app.callback(
        Output("umap-curve-variants", "data"),
        Output("umap-curve-variant-display", "children"),
        Input("umap-graph", "clickData"),
        Input("umap-graph", "selectedData"),
        Input("umap-curve-clear", "n_clicks"),
        State("umap-curve-variants", "data"),
        prevent_initial_call=True,
    )
    def _accumulate_umap_curve_variants(click_data, selected_data, _n_clear, current):
        from dash import callback_context as _dcc_ctx
        triggered_ids = {t["prop_id"] for t in _dcc_ctx.triggered} if _dcc_ctx.triggered else set()
        if "umap-curve-clear.n_clicks" in triggered_ids:
            return [], _curve_variant_list_text([])
        if "umap-graph.selectedData" in triggered_ids:
            pts = (selected_data or {}).get("points", [])
            if pts:
                lst = _curve_variants_from_points(pts)
                return lst, _curve_variant_list_text(lst)
        if "umap-graph.clickData" in triggered_ids and click_data:
            pts = click_data.get("points", [])[:1]
            v = _curve_variant_from_point(pts[0]) if pts else None
            if v is not None:
                lst = list(current or [])
                if v in lst:
                    lst.remove(v)
                else:
                    lst.append(v)
                return lst, _curve_variant_list_text(lst)
        lst = list(current or [])
        return lst, _curve_variant_list_text(lst)

    def _load_umap_pmf_targeted(variants: list, metric: Optional[str] = None) -> pd.DataFrame:
        """Load selected PMF curves via the shared cached variant-PMF store."""
        try:
            max_pts = int(os.environ.get("PEPTIDE_DASH_PMF_VIEWER_MAX_POINTS", "1200"))
            return load_variant_pmfs(
                ctx,
                variants,
                metric=metric,
                columns=PMF_PLOT_COLS,
                max_points_per_curve=max_pts,
            )
        except Exception:
            return pd.DataFrame()


    def _umap_pmf_loader_status(df: pd.DataFrame, variants: list, metric: Optional[str]) -> str:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return f"No PMF curves loaded for metric={metric!r}; requested variants={len(variants or [])}."
        backend = df.attrs.get("pmf_loader_backend", "unknown")
        workers = df.attrs.get("pmf_loader_workers", "?")
        max_pts = os.environ.get("PEPTIDE_DASH_PMF_VIEWER_MAX_POINTS", "1200")
        if "variant" in df.columns and "metric" in df.columns:
            n_curves = int(df[["variant", "metric"]].drop_duplicates().shape[0])
        elif "variant" in df.columns:
            n_curves = int(df["variant"].nunique())
        else:
            n_curves = 1
        return (
            f"Loaded {len(df):,} plotted point(s) across {n_curves} curve(s); "
            f"backend={backend}, workers={workers}, max_points_per_curve={max_pts}."
        )


    @app.callback(
        Output("umap-curve-metric", "options"),
        Output("umap-curve-metric", "value"),
        Input("umap-curve-variants", "data"),
        State("umap-pmf-metrics", "value"),
        prevent_initial_call=True,
    )
    def _populate_umap_curve_metrics(variants, selected_metrics):
        if not variants:
            return [], None
        try:
            metrics = available_pmf_metrics(ctx, variants=variants)
        except Exception:
            metrics = []
        selected_metrics = [str(m) for m in (selected_metrics or [])]
        if selected_metrics:
            selected = set(selected_metrics)
            metrics = [m for m in metrics if m in selected] or metrics
        if not metrics:
            return [{"label": "PMF data not found", "value": ""}], None
        opts = [{"label": m, "value": m} for m in metrics]
        return opts, metrics[0]


    @app.callback(
        Output("umap-curve-overlay", "figure"),
        Output("umap-curve-loader-status", "children"),
        Input("umap-curve-metric", "value"),
        Input("umap-curve-variants", "data"),
        Input("umap-pmf-unit", "value"),
        Input("theme-store", "data"),
        prevent_initial_call=True,
    )
    def _update_umap_curve_overlay(metric, variants, pmf_unit, theme=None):
        if not variants:
            return apply_theme(_error_fig("Click variants in the UMAP scatter to overlay PMFs."), theme), "PMF viewer idle. Select variants and a metric to load curves."
        if not metric:
            return apply_theme(_error_fig("Select a PMF metric from the dropdown."), theme), "Waiting for a valid PMF metric."
        d = _load_umap_pmf_targeted(list(variants or []), metric=str(metric))
        status = _umap_pmf_loader_status(d, list(variants or []), str(metric))
        if d.empty:
            return apply_theme(_error_fig("No PMF data found for selected variants."), theme), status
        if "metric" in d.columns:
            d = d[d["metric"].astype(str) == str(metric)]
        if d.empty:
            return apply_theme(_error_fig(f"No PMF data for metric {metric!r} in selected variants."), theme), status
        if "x" not in d.columns:
            return apply_theme(_error_fig("PMF data has no x column."), theme), status
        y_col = next((c for c in ("F_kJ_mol", "pmf_F_kJmol", "F", "y", "free_energy") if c in d.columns), None)
        if y_col is None:
            return apply_theme(_error_fig("Cannot identify PMF free-energy column."), theme), status
        d = d.copy()
        d["variant"] = d["variant"].astype(str)
        d["x"] = pd.to_numeric(d["x"], errors="coerce")
        d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
        d = d.dropna(subset=["x", y_col]).sort_values(["variant", "x"])
        if d.empty:
            return apply_theme(_error_fig("Selected PMF curves contain no finite x/F values."), theme), status
        factor, unit_label = _pmf_unit_factor_label(pmf_unit, T_K=_pmf_temperature_from_frame(d))
        d["F_display"] = d[y_col] * factor
        fig = px.line(d, x="x", y="F_display", color="variant", markers=False, title=f"PMF - {metric}")
        color_seq = px.colors.qualitative.Plotly
        var_order = list(dict.fromkeys(d["variant"].astype(str)))
        var_color = {v: color_seq[i % len(color_seq)] for i, v in enumerate(var_order)}
        for var, grp in d.groupby("variant", observed=True, sort=False):
            grp = grp.dropna(subset=["F_display"])
            if grp.empty:
                continue
            row = grp.loc[grp["F_display"].idxmin()]
            fig.add_scatter(
                x=[row["x"]],
                y=[row["F_display"]],
                mode="markers",
                marker=dict(size=14, symbol="star", color=var_color.get(str(var), "#888"), line=dict(width=0)),
                showlegend=False,
                hovertemplate=(f"<b>{var}</b><br>{metric} = {float(row['x']):.4g}<br>F = {float(row["F_display"]):.2f} {unit_label}<extra></extra>"),
            )
        feat_df = getattr(ctx, "df", pd.DataFrame())
        if isinstance(feat_df, pd.DataFrame) and not feat_df.empty and "variant" in feat_df.columns:
            gcol = f"{metric}__global_basin_min_x"
            scol = f"{metric}__secondary_basin_min_x"
            for var in variants:
                row = feat_df[feat_df["variant"].astype(str) == str(var)]
                if row.empty:
                    continue
                clr = var_color.get(str(var), "#888")
                if gcol in feat_df.columns:
                    gx = pd.to_numeric(row[gcol].iloc[:1], errors="coerce").iloc[0]
                    if np.isfinite(gx):
                        fig.add_vline(x=float(gx), line=dict(color=clr, width=1.5, dash="dash"), annotation_text=f"{var} G", annotation_font_size=10, annotation_position="top right")
                if scol in feat_df.columns:
                    sx = pd.to_numeric(row[scol].iloc[:1], errors="coerce").iloc[0]
                    if np.isfinite(sx):
                        fig.add_vline(x=float(sx), line=dict(color=clr, width=1.0, dash="dot"), annotation_text=f"{var} S", annotation_font_size=10, annotation_position="top left")
        fig.update_layout(
            title=f"PMF - {metric}",
            xaxis_title=str(metric),
            yaxis_title=f"F ({unit_label})",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            margin=dict(l=70, r=20, t=55, b=60),
            height=480,
            template=_template(),
        )
        return apply_theme(fig, theme), status



    # --- Trajectory/path PMF viewer callbacks ---

    @app.callback(
        Output("umap-trajectory-metric", "options"),
        Output("umap-trajectory-metric", "value"),
        Input("umap-embedding-data", "data"),
        State("umap-pmf-metrics", "value"),
        prevent_initial_call=False,
    )
    def _populate_trajectory_metric_options(_embedding_data, selected_metrics):
        try:
            metrics = available_pmf_metrics(ctx)
        except Exception:
            metrics = []
        selected_metrics = [str(m) for m in (selected_metrics or [])]
        if selected_metrics:
            metrics = [m for m in metrics if m in set(selected_metrics)] or metrics
        opts = [{"label": m, "value": m} for m in metrics]
        return opts, (metrics[0] if metrics else None)

    @app.callback(
        Output("umap-trajectory-graph", "figure"),
        Output("umap-trajectory-status", "children"),
        Output("umap-trajectory-bins-data", "data"),
        Output("umap-residue-bin-slider", "max"),
        Output("umap-residue-bin-slider", "value"),
        Output("umap-residue-bin-slider", "marks"),
        Input("umap-trajectory-update", "n_clicks"),
        State("umap-embedding-data", "data"),
        State("umap-trajectory-metric", "value"),
        State("umap-trajectory-mode", "value"),
        State("umap-trajectory-bins", "value"),
        State("umap-trajectory-unit", "value"),
        State("umap-trajectory-anchors", "value"),
        State("umap-trajectory-feature", "value"),
        Input("theme-store", "data"),
        prevent_initial_call=False,
    )
    def _update_umap_trajectory(update_clicks, embedding_data, metric, mode, n_bins, unit, anchors, feature_col, theme):
        if not update_clicks:
            fig = apply_theme(_error_fig("Press Update trajectory to load PMF curves for the current embedding."), theme)
            return fig, "Trajectory idle: quickload/recalculate only restores coordinates; press Update trajectory to load PMFs.", {"bins": []}, 1, 1, {1: "1"}
        emb_df = _plot_df_from_store(embedding_data)
        emb_df = _merge_feature_columns_for_embedding(emb_df, _ctx_features_df(ctx, feats_df))
        if emb_df.empty:
            fig = apply_theme(_error_fig("Compute embedding first."), theme)
            return fig, "Trajectory idle: compute or quickload an embedding first.", {"bins": []}, 1, 1, {1: "1"}
        if not metric:
            fig = apply_theme(_error_fig("Select a PMF metric for trajectory."), theme)
            return fig, "Trajectory idle: choose PMF metric, then press Update trajectory.", {"bins": []}, 1, 1, {1: "1"}
        variants = emb_df["variant"].dropna().astype(str).tolist() if "variant" in emb_df.columns else []
        curves = _load_umap_pmf_targeted(variants, metric=str(metric))
        method_label = str((embedding_data or {}).get("method", "UMAP")) if isinstance(embedding_data, dict) else "UMAP"
        fig, status, bins_data = _build_trajectory_figure(emb_df, curves, str(metric), str(mode or "axis1"), int(n_bins or 8), str(unit or "kJ/mol"), method_label=method_label, anchors=list(anchors or []), feature_col=str(feature_col or ""))
        n = max(1, len((bins_data or {}).get("bins", [])))
        marks = {1: "1", n: str(n)} if n > 1 else {1: "1"}
        apply_theme(fig, theme)
        return fig, status, bins_data, n, 1, marks

    @app.callback(
        Output("umap-trajectory-selected-bin", "data"),
        Input("umap-residue-bin-slider", "value"),
        Input("umap-trajectory-graph", "clickData"),
        State("umap-trajectory-selected-bin", "data"),
        prevent_initial_call=True,
    )
    def _select_trajectory_bin(slider_value, click_data, current):
        from dash import callback_context as _dcc_ctx
        trig = _dcc_ctx.triggered[0]["prop_id"].split(".")[0] if _dcc_ctx.triggered else ""
        if trig == "umap-residue-bin-slider":
            try:
                return int(slider_value or 1)
            except Exception:
                return 1
        if trig == "umap-trajectory-graph" and click_data:
            pts = click_data.get("points", []) or []
            for pt in pts:
                cd = pt.get("customdata")
                if isinstance(cd, (list, tuple, np.ndarray)) and len(cd):
                    try:
                        return int(cd[0])
                    except Exception:
                        pass
                elif cd is not None:
                    try:
                        return int(cd)
                    except Exception:
                        pass
                for key in ("x", "pointNumber"):
                    try:
                        val = int(round(float(pt.get(key))))
                        if val >= 1:
                            return val
                    except Exception:
                        pass
        try:
            return int(current or 1)
        except Exception:
            return 1

    @app.callback(
        Output("umap-residue-composition", "figure"),
        Output("umap-residue-status", "children"),
        Input("umap-trajectory-bins-data", "data"),
        Input("umap-trajectory-selected-bin", "data"),
        Input("theme-store", "data"),
        prevent_initial_call=False,
    )
    def _update_residue_composition(bins_data, selected_bin, theme):
        bins = (bins_data or {}).get("bins", []) if isinstance(bins_data, dict) else []
        if not bins:
            return apply_theme(_error_fig("No trajectory bins available."), theme), "Residue composition idle: compute trajectory first."
        try:
            bnum = int(selected_bin or 1)
        except Exception:
            bnum = 1
        # Clamp to available bin numbers.
        available = [int(b.get("bin", i+1)) for i, b in enumerate(bins)]
        if bnum not in available:
            bnum = available[0]
        selected = []
        trajectory_background = []
        for b in bins:
            vs = [str(v) for v in b.get("variants", [])]
            trajectory_background.extend(vs)
            if int(b.get("bin", -1)) == bnum:
                selected = vs
        feats_now = _ctx_features_df(ctx, feats_df)
        dataset_background: list[str] = []
        if isinstance(feats_now, pd.DataFrame) and not feats_now.empty and "variant" in feats_now.columns:
            dataset_background.extend(feats_now["variant"].dropna().astype(str).tolist())
        try:
            dataset_background.extend(_ctx_pmf_variants(ctx))
        except Exception:
            pass
        dataset_background = sorted(set(v for v in dataset_background if str(v).strip())) or sorted(set(trajectory_background))
        fig, status = _build_residue_composition_figure(selected, dataset_background)
        apply_theme(fig, theme)
        return fig, f"bin={bnum}; {status}"

    @app.callback(
        Output("umap-trajectory-sequence-graph", "figure"),
        Output("umap-trajectory-sequence-status", "children"),
        Input("umap-trajectory-bins-data", "data"),
        Input("theme-store", "data"),
        prevent_initial_call=False,
    )
    def _update_trajectory_sequence_grammar(bins_data, theme):
        bins = (bins_data or {}).get("bins", []) if isinstance(bins_data, dict) else []
        if not bins:
            return apply_theme(_error_fig("No trajectory grammar available."), theme), "Trajectory sequence grammar idle: compute trajectory first."
        feats_now = _ctx_features_df(ctx, feats_df)
        trajectory_background = []
        for b in bins:
            trajectory_background.extend([str(v) for v in b.get("variants", [])])
        background = _dataset_background_variants(ctx, feats_now, fallback=trajectory_background)
        fig, status = _build_trajectory_sequence_grammar_figure(bins_data, background, feats_now)
        apply_theme(fig, theme)
        return fig, status

# ----------------------------------------------------------------------
# Public API used by tabs/__init__.py
# ----------------------------------------------------------------------

def layout(ctx) -> html.Div:
    pmf_df = pd.DataFrame()  # not loaded at layout time
    feats_df = getattr(ctx, "df", getattr(ctx, "features_df", pd.DataFrame()))
    return make_layout(pmf_df, feats_df)


def register_callbacks(app: dash.Dash, ctx) -> None:
    feats_df = getattr(ctx, "df", getattr(ctx, "features_df", pd.DataFrame()))
    _register_callbacks_with_data(app, ctx, feats_df)
