from .shared import *
from .data_access import _umap_data_status_children
from .figures import _error_fig, _color_options_from_features, _trajectory_feature_options_from_features


def make_layout(pmf_df: pd.DataFrame, feats_df: pd.DataFrame) -> html.Div:
    # PMF metrics for dropdown
    if not pmf_df.empty and "metric" in pmf_df.columns:
        pmf_metrics_for_umap = sorted(pmf_df["metric"].dropna().unique().tolist())
    else:
        pmf_metrics_for_umap = []

    # Variant universe for selectors (union of whatever we can see)
    variants_all: List[str] = []
    if not feats_df.empty and "variant" in feats_df.columns:
        variants_all.extend(feats_df["variant"].dropna().astype(str).tolist())
    if not pmf_df.empty and "variant" in pmf_df.columns:
        variants_all.extend(pmf_df["variant"].dropna().astype(str).tolist())
    variants_all = sorted(set(variants_all))

    variant_options = [{"label": v, "value": v} for v in variants_all]

    # Features for color/label
    color_options: List[dict] = _color_options_from_features(feats_df)
    trajectory_feature_options: List[dict] = _trajectory_feature_options_from_features(feats_df)
    label_options: List[dict] = []
    if not feats_df.empty:
        for c in feats_df.columns:
            if c == "variant" or is_excluded_feature_column(str(c)):
                continue
            if not pd.api.types.is_numeric_dtype(feats_df[c]):
                label_options.append({"label": str(c), "value": str(c)})

    placeholder_fig = _error_fig("Press 'Recalculate UMAP' to compute embedding.")

    def _control_section(title: str, children: list, note: str | None = None) -> html.Div:
        body = [html.Div(title, style={"fontWeight": 600, "fontSize": "0.86em", "marginBottom": "6px"})]
        if note:
            body.append(html.Div(note, style={"fontSize": "0.68em", "opacity": 0.70, "lineHeight": "1.25", "marginBottom": "6px"}))
        body.extend(children)
        return html.Div(
            body,
            style={
                "borderTop": "1px solid rgba(180,180,180,0.35)",
                "paddingTop": "8px",
                "marginTop": "8px",
            },
        )

    controls = dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(
                            dbc.Button("Recalculate", id="umap-recalc", color="primary", size="sm"),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.Button("Quickload last", id="umap-cache-quickload", color="secondary", size="sm", outline=True),
                            width="auto",
                        ),
                        dbc.Col(
                            dbc.InputGroup(
                                [dbc.InputGroupText("runs"), dbc.Input(id="umap-stability-runs", type="number", min=1, step=1, value=1)],
                                size="sm",
                            ),
                            width=True,
                        ),
                        dbc.Col(
                            dbc.InputGroup(
                                [dbc.InputGroupText("k"), dbc.Input(id="umap-stability-k", type="number", min=3, step=1, value=10)],
                                size="sm",
                            ),
                            width=True,
                        ),
                    ],
                    className="g-1 align-items-center",
                ),
                html.Div(id="umap-data-status", className="mt-2 mb-1"),

                _control_section(
                    "Input space",
                    [
                        dcc.Dropdown(
                            id="umap-input-source",
                            options=[
                                {"label": "Basic / native scalar features", "value": UMAP_SOURCE_BASIC},
                                {"label": "Thermodynamic / basin geometry corrected", "value": UMAP_SOURCE_THERMO},
                                {"label": "PMFs themselves", "value": UMAP_SOURCE_PMF},
                                {"label": "Sequence / letter-space descriptors", "value": UMAP_SOURCE_SEQUENCE},
                            ],
                            value=UMAP_SOURCE_BASIC,
                            clearable=False,
                            style={"fontSize": "0.78em"},
                        ),
                        html.Div(
                            "Same source logic as PCA plus sequence space: native values, corrected basin descriptors, sqrt(P)-based PMF shapes, or interpretable batch_ana seq_* descriptors.",
                            style={"fontSize": "0.68em", "opacity": 0.70, "lineHeight": "1.25", "marginTop": "3px"},
                        ),
                        html.Div("PMF metrics", style={"fontSize": "0.76em", "marginTop": "7px"}),
                        dcc.Dropdown(
                            id="umap-pmf-metrics",
                            multi=True,
                            clearable=True,
                            options=[{"label": m, "value": m} for m in pmf_metrics_for_umap],
                            value=[],
                            placeholder="Empty = all PMF metrics",
                            style={"fontSize": "0.76em"},
                        ),
                        html.Div("PMF representation", style={"fontSize": "0.76em", "marginTop": "7px"}),
                        dcc.RadioItems(
                            id="umap-pmf-repr",
                            options=[
                                {"label": "sqrt(P)", "value": "P"},
                                {"label": "F → P", "value": "F"},
                                {"label": "log(P)", "value": "log_P"},
                            ],
                            value="P",
                            inline=True,
                            style={"fontSize": "0.75em"},
                        ),
                    ],
                ),

                _control_section(
                    "Embedding",
                    [
                        dcc.Dropdown(
                            id="umap-embedding",
                            options=[
                                {"label": "UMAP", "value": "umap"},
                                {"label": "DensMAP", "value": "densmap"},
                                {"label": "Isomap", "value": "isomap"},
                                {"label": "Diffusion Map", "value": "diffusion"},
                                {"label": "PHATE", "value": "phate"},
                            ],
                            value="umap",
                            clearable=False,
                            style={"fontSize": "0.78em"},
                        ),
                        dbc.Row(
                            [
                                dbc.Col(
                                    [html.Label("Preset", style={"fontSize": "0.72em"}), dcc.Dropdown(id="umap-preset", options=[{"label": "Global", "value": "global"}, {"label": "Local", "value": "local"}, {"label": "Custom", "value": "custom"}], value="global", clearable=False, style={"fontSize": "0.76em"})],
                                    md=6,
                                ),
                                dbc.Col(
                                    [html.Label("Dims", style={"fontSize": "0.72em"}), dcc.Dropdown(id="umap-dims", options=[{"label": "2D", "value": 2}, {"label": "3D", "value": 3}], value=2, clearable=False, style={"fontSize": "0.76em"})],
                                    md=6,
                                ),
                            ],
                            className="g-1 mt-1",
                        ),
                        dbc.Row(
                            [
                                dbc.Col([html.Label("neighbors", id="umap-nn-label", style={"fontSize": "0.72em"}), dcc.Input(id="umap-nn", type="number", min=2, step=1, value=60, className="form-control form-control-sm")], md=6),
                                dbc.Col([html.Label("min_dist", id="umap-min-dist-label", style={"fontSize": "0.72em"}), dcc.Input(id="umap-min-dist", type="number", min=0, max=1, step=0.01, value=0.12, className="form-control form-control-sm")], md=6),
                            ],
                            className="g-1 mt-1",
                        ),
                        html.Div("Distance metric", style={"fontSize": "0.72em", "marginTop": "5px"}),
                        dcc.Dropdown(
                            id="umap-metric",
                            options=[
                                {"label": "cosine", "value": "cosine"},
                                {"label": "euclidean", "value": "euclidean"},
                                {"label": "manhattan", "value": "manhattan"},
                                {"label": "correlation", "value": "correlation"},
                                {"label": "hellinger (PMF probs)", "value": "hellinger"},
                            ],
                            value="cosine",
                            clearable=False,
                            style={"fontSize": "0.76em"},
                        ),
                        html.Div(
                            "UMAP/DensMAP: neighbors + min_dist. Diffusion Map/PHATE: kNN graph + diffusion time t. Isomap: geodesic kNN; second parameter is ignored.",
                            id="umap-method-param-help",
                            style={"fontSize": "0.68em", "opacity": 0.70, "lineHeight": "1.25", "marginTop": "4px"},
                        ),
                        html.Details(
                            [
                                html.Summary("DensMAP parameters", style={"fontSize": "0.75em", "cursor": "pointer"}),
                                dbc.Row(
                                    [
                                        dbc.Col([html.Label("lambda", style={"fontSize": "0.7em"}), dcc.Input(id="umap-dens-lambda", type="number", step=0.1, value=2.0, className="form-control form-control-sm")], md=4),
                                        dbc.Col([html.Label("frac", style={"fontSize": "0.7em"}), dcc.Input(id="umap-dens-frac", type="number", min=0, max=1, step=0.05, value=0.30, className="form-control form-control-sm")], md=4),
                                        dbc.Col([html.Label("var_shift", style={"fontSize": "0.7em"}), dcc.Input(id="umap-dens-var-shift", type="number", min=0, max=1, step=0.05, value=0.10, className="form-control form-control-sm")], md=4),
                                    ],
                                    className="g-1 mt-1",
                                ),
                            ],
                            style={"marginTop": "6px"},
                        ),
                        dbc.Row(
                            [
                                dbc.Col(html.Label("PCA cap", style={"fontSize": "0.72em"}), md="auto"),
                                dbc.Col(dcc.Input(id="umap-pca-cap", type="number", min=2, max=256, step=1, value=64, className="form-control form-control-sm"), md=True),
                            ],
                            className="g-1 mt-2 align-items-center",
                        ),
                        html.Div("Max PCs used before UMAP/DM/Isomap. Higher = more variance captured, slower.", style={"fontSize": "0.65em", "opacity": 0.65, "lineHeight": "1.2", "marginTop": "2px"}),
                        # Diffusion-map-only controls (hidden unless method == diffusion)
                        html.Div(
                            id="umap-dm-controls",
                            style={"display": "none"},
                            children=[
                                html.Hr(style={"margin": "6px 0"}),
                                html.Div("Diffusion Map settings", style={"fontSize": "0.72em", "fontWeight": 600, "marginBottom": "4px"}),
                                dbc.Row(
                                    [
                                        dbc.Col([html.Label("Alpha (dens. corr.)", style={"fontSize": "0.72em"}), dcc.Dropdown(id="umap-dm-alpha", options=[{"label": "0.0 (none)", "value": 0.0}, {"label": "0.5", "value": 0.5}, {"label": "1.0 (full)", "value": 1.0}], value=1.0, clearable=False, style={"fontSize": "0.76em"})], md=6),
                                        dbc.Col([html.Label("# components", style={"fontSize": "0.72em"}), dcc.Input(id="umap-dm-n-components", type="number", min=2, max=20, step=1, value=10, className="form-control form-control-sm")], md=6),
                                    ],
                                    className="g-1 mt-1",
                                ),
                                html.Div("Axis selection", style={"fontSize": "0.72em", "marginTop": "6px", "marginBottom": "2px"}),
                                dbc.Row(
                                    [
                                        dbc.Col([html.Label("X", style={"fontSize": "0.72em"}), dcc.Dropdown(id="umap-dm-dc-x", options=[{"label": "DC1", "value": 1}], value=1, clearable=False, style={"fontSize": "0.76em"})], md=4),
                                        dbc.Col([html.Label("Y", style={"fontSize": "0.72em"}), dcc.Dropdown(id="umap-dm-dc-y", options=[{"label": "DC2", "value": 2}], value=2, clearable=False, style={"fontSize": "0.76em"})], md=4),
                                        dbc.Col([html.Label("Z", style={"fontSize": "0.72em"}), dcc.Dropdown(id="umap-dm-dc-z", options=[{"label": "DC3", "value": 3}], value=3, clearable=False, style={"fontSize": "0.76em"})], md=4),
                                    ],
                                    className="g-1 mt-1",
                                ),
                                html.Div("Axis selectors populate after Compute. Changing axes re-renders without recomputing.", style={"fontSize": "0.65em", "opacity": 0.65, "lineHeight": "1.2", "marginTop": "3px"}),
                            ],
                        ),
                    ],
                ),

                _control_section(
                    "Variants",
                    [
                        dcc.Store(id="umap-variants-universe", data=variants_all),
                        dbc.Checklist(id="umap-plot-union", options=[{"label": "Plot union (fit ∪ plot)", "value": "on"}], value=["on"], switch=True, style={"fontSize": "0.75em"}),
                        html.Details(
                            [
                                html.Summary("Fit set", style={"fontSize": "0.78em", "fontWeight": 600, "cursor": "pointer"}),
                                dbc.Checklist(id="umap-fit-use-all", options=[{"label": "Use all variants for fitting", "value": "all"}], value=["all"], switch=True, style={"fontSize": "0.74em"}),
                                dcc.Dropdown(id="umap-fit-variants", options=variant_options, value=[], multi=True, placeholder="Fit variants", style={"fontSize": "0.74em"}),
                                dbc.Input(id="umap-fit-filter", placeholder="substring or re:<regex>", size="sm", className="mt-1"),
                                dbc.ButtonGroup([dbc.Button("Set", id="umap-fit-filter-set", size="sm", outline=True), dbc.Button("Add", id="umap-fit-filter-add", size="sm", outline=True), dbc.Button("Remove", id="umap-fit-filter-remove", size="sm", outline=True), dbc.Button("Invert", id="umap-fit-invert", size="sm", outline=True), dbc.Button("All", id="umap-fit-clear", size="sm", outline=True)], className="mt-1"),
                                dcc.Textarea(id="umap-fit-paste", placeholder="paste variants…", style={"width": "100%", "height": "4.2em"}, className="mt-1"),
                                dbc.ButtonGroup([dbc.Button("Set=paste", id="umap-fit-paste-set", size="sm", outline=True), dbc.Button("Add", id="umap-fit-paste-add", size="sm", outline=True)], className="mt-1"),
                            ],
                            open=False,
                            style={"marginTop": "5px"},
                        ),
                        html.Details(
                            [
                                html.Summary("Plot set", style={"fontSize": "0.78em", "fontWeight": 600, "cursor": "pointer"}),
                                dbc.Checklist(id="umap-plot-use-all", options=[{"label": "Plot all variants", "value": "all"}], value=["all"], switch=True, style={"fontSize": "0.74em"}),
                                dcc.Dropdown(id="umap-plot-variants", options=variant_options, value=[], multi=True, placeholder="Plot variants", style={"fontSize": "0.74em"}),
                                dbc.Input(id="umap-plot-filter", placeholder="substring or re:<regex>", size="sm", className="mt-1"),
                                dbc.ButtonGroup([dbc.Button("Set", id="umap-plot-filter-set", size="sm", outline=True), dbc.Button("Add", id="umap-plot-filter-add", size="sm", outline=True), dbc.Button("Remove", id="umap-plot-filter-remove", size="sm", outline=True), dbc.Button("Invert", id="umap-plot-invert", size="sm", outline=True), dbc.Button("All", id="umap-plot-clear", size="sm", outline=True)], className="mt-1"),
                                dcc.Textarea(id="umap-plot-paste", placeholder="paste variants…", style={"width": "100%", "height": "4.2em"}, className="mt-1"),
                                dbc.ButtonGroup([dbc.Button("Set=paste", id="umap-plot-paste-set", size="sm", outline=True), dbc.Button("Add", id="umap-plot-paste-add", size="sm", outline=True)], className="mt-1"),
                            ],
                            open=False,
                            style={"marginTop": "5px"},
                        ),
                    ],
                ),

                _control_section(
                    "Display / clustering",
                    [
                        html.Div("Color by", style={"fontSize": "0.72em"}),
                        dcc.Dropdown(id="umap-color", options=color_options, value=(color_options[0]["value"] if color_options else None), placeholder="Feature / cluster", style={"fontSize": "0.76em"}),
                        html.Div("Label column", style={"fontSize": "0.72em", "marginTop": "5px"}),
                        dcc.Dropdown(id="umap-label", options=label_options, value=None, placeholder="Optional categorical label", style={"fontSize": "0.76em"}),
                        dbc.Checklist(
                            id="umap-feature-arrows",
                            options=[{"label": "Feature gradient arrows (biplot, top 5)", "value": "on"}],
                            value=[],
                            switch=True,
                            style={"fontSize": "0.74em", "marginTop": "5px"},
                        ),
                        html.Details(
                            [
                                html.Summary("DBSCAN", style={"fontSize": "0.78em", "fontWeight": 600, "cursor": "pointer"}),
                                dbc.Checklist(id="umap-dbscan-enable", options=[{"label": "Enable DBSCAN", "value": "on"}], value=[], switch=True, style={"fontSize": "0.74em"}),
                                dcc.Dropdown(id="umap-dbscan-on", options=[{"label": "All plotted points", "value": "plotted"}, {"label": "Fit points only", "value": "fit"}], value="plotted", clearable=False, style={"fontSize": "0.74em"}),
                                dbc.Row([dbc.Col([html.Label("eps", style={"fontSize": "0.7em"}), dcc.Input(id="umap-dbscan-eps", type="number", min=0, step=0.05, value=0.50, className="form-control form-control-sm")], md=6), dbc.Col([html.Label("min_samples", style={"fontSize": "0.7em"}), dcc.Input(id="umap-dbscan-min-samples", type="number", min=1, step=1, value=5, className="form-control form-control-sm")], md=6)], className="g-1 mt-1"),
                                dbc.Checklist(id="umap-dbscan-color", options=[{"label": "Color by DBSCAN", "value": "on"}], value=["on"], switch=True, style={"fontSize": "0.74em"}),
                            ],
                            open=False,
                            style={"marginTop": "5px"},
                        ),
                    ],
                ),
            ]
        ),
        style={"height": "76vh", "overflowY": "auto"},
    )

    pmf_viewer = dbc.Card(
        dbc.CardBody(
            [
                dcc.Store(id="umap-curve-variants", data=[]),
                dbc.Row(
                    [
                        dbc.Col(html.Strong("PMF curve viewer"), md="auto"),
                        dbc.Col(
                            dcc.Dropdown(
                                id="umap-curve-metric",
                                options=[],
                                value=None,
                                placeholder="Click/lasso points, then choose PMF metric",
                                clearable=False,
                            ),
                            md=4,
                        ),
                        dbc.Col(
                            dcc.Dropdown(
                                id="umap-pmf-unit",
                                options=[
                                    {"label": "kJ/mol", "value": "kJ/mol"},
                                    {"label": "kcal/mol", "value": "kcal/mol"},
                                    {"label": "kT", "value": "kT"},
                                ],
                                value="kJ/mol",
                                clearable=False,
                            ),
                            md=2,
                        ),
                        dbc.Col(
                            dbc.Button("Clear", id="umap-curve-clear", n_clicks=0, size="sm", outline=True),
                            md="auto",
                        ),
                        dbc.Col(
                            html.Small(
                                id="umap-curve-variant-display",
                                children="Click or lasso-select points in the UMAP scatter above.",
                                className="text-muted",
                            ),
                            md=True,
                        ),
                    ],
                    className="g-2 align-items-center",
                ),
                html.Div(
                    id="umap-curve-loader-status",
                    children="PMF viewer idle. Select variants and a metric to load curves.",
                    className="pmf-loader-status",
                ),
                dcc.Loading(
                    html.Div(
                        [
                            html.Div(
                                html.Div(className="pmf-progress-bar"),
                                className="pmf-progress-track",
                                title="Indeterminate PMF-loading progress indicator",
                            ),
                            dcc.Graph(
                                id="umap-curve-overlay",
                                figure=_error_fig("Click variants in the UMAP scatter to overlay PMFs."),
                                style={"height": "38vh"},
                            ),
                        ],
                        className="pmf-curve-loading-shell",
                    ),
                    type="circle",
                ),
            ]
        ),
        className="mt-2",
    )


    sequence_region_viewer = dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Strong("Sequence grammar for selected manifold region"), md="auto"),
                        dbc.Col(html.Small(id="umap-sequence-region-status", children="Click or lasso-select points in the UMAP scatter to inspect sequence grammar.", className="text-muted"), md=True),
                    ],
                    className="g-2 align-items-center",
                ),
                dcc.Graph(id="umap-sequence-region-graph", figure=_error_fig("Click or lasso-select points in the UMAP scatter above."), style={"height": "52vh"}, config={"displaylogo": False}),
            ]
        ),
        className="mt-2",
    )



    sequence_cluster_viewer = dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Strong("Auto sequence-language clusters"), md="auto"),
                        dbc.Col(
                            dcc.Dropdown(
                                id="umap-seq-cluster-mode",
                                options=[
                                    {"label": "All sequence-language features", "value": "all"},
                                    {"label": "Composition only", "value": "composition"},
                                    {"label": "Aligned positions", "value": "position"},
                                    {"label": "Motifs / k-mers", "value": "motif"},
                                ],
                                value="all",
                                clearable=False,
                                style={"fontSize": "0.76em"},
                            ),
                            md=3,
                        ),
                        dbc.Col(dbc.InputGroup([dbc.InputGroupText("max k"), dbc.Input(id="umap-seq-cluster-max-k", type="number", min=2, max=12, step=1, value=8)], size="sm"), md=2),
                        dbc.Col(dbc.Button("Detect clusters", id="umap-seq-cluster-run", n_clicks=0, color="secondary", size="sm", outline=True), md="auto"),
                        dbc.Col(html.Small(id="umap-seq-cluster-status", children="Compute an embedding, then run automatic sequence-language clustering.", className="text-muted"), md=True),
                    ],
                    className="g-2 align-items-center",
                ),
                dcc.Loading(
                    html.Div(
                        [
                            dcc.Graph(id="umap-seq-cluster-graph", figure=_error_fig("Run sequence-language clustering after computing an embedding."), style={"height": "56vh"}, config={"displaylogo": False}),
                            html.Div(id="umap-seq-cluster-table", className="mt-2"),
                        ]
                    ),
                    type="dot",
                ),
            ]
        ),
        className="mt-2",
    )


    trajectory_viewer = dbc.Card(
        dbc.CardBody(
            [
                dbc.Row(
                    [
                        dbc.Col(html.Strong("PMF trajectory / path viewer"), md="auto"),
                        dbc.Col(dcc.Dropdown(id="umap-trajectory-metric", options=[], value=None, placeholder="PMF metric", clearable=False), md=3),
                        dbc.Col(
                            dcc.Dropdown(
                                id="umap-trajectory-mode",
                                options=[
                                    {"label": "Axis 1 / X", "value": "axis1"},
                                    {"label": "Axis 2 / Y", "value": "axis2"},
                                    {"label": "Axis 3 / Z", "value": "axis3"},
                                    {"label": "XY composite", "value": "xy"},
                                    {"label": "XZ composite", "value": "xz"},
                                    {"label": "YZ composite", "value": "yz"},
                                    {"label": "XYZ composite", "value": "xyz"},
                                    {"label": "Follow scalar feature", "value": "feature"},
                                    {"label": "Custom line anchors", "value": "line"},
                                    {"label": "kNN geodesic anchors", "value": "geodesic"},
                                    {"label": "Manual polyline anchors", "value": "polyline"},
                                ],
                                value="axis1",
                                clearable=False,
                            ),
                            md=2,
                        ),
                        dbc.Col(dbc.InputGroup([dbc.InputGroupText("bins"), dbc.Input(id="umap-trajectory-bins", type="number", min=2, max=80, step=1, value=8)], size="sm"), md=2),
                        dbc.Col(
                            dcc.Dropdown(
                                id="umap-trajectory-unit",
                                options=[
                                    {"label": "kJ/mol", "value": "kJ/mol"},
                                    {"label": "kcal/mol", "value": "kcal/mol"},
                                    {"label": "kT", "value": "kT"},
                                ],
                                value="kJ/mol",
                                clearable=False,
                            ),
                            md=2,
                        ),
                        dbc.Col(
                            dbc.Button("Update trajectory", id="umap-trajectory-update", n_clicks=0, color="secondary", size="sm", outline=True),
                            md="auto",
                        ),
                    ],
                    className="g-2 align-items-center",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Small("Anchors for line/geodesic/polyline", className="text-muted"), md="auto"),
                        dbc.Col(dcc.Dropdown(id="umap-trajectory-anchors", options=variant_options, value=[], multi=True, placeholder="ordered anchor variants", style={"fontSize": "0.76em"}), md=True),
                    ],
                    className="g-2 align-items-center mt-1",
                ),
                dbc.Row(
                    [
                        dbc.Col(html.Small("Follow scalar feature", className="text-muted"), md="auto"),
                        dbc.Col(dcc.Dropdown(id="umap-trajectory-feature", options=trajectory_feature_options, value=(trajectory_feature_options[0]["value"] if trajectory_feature_options else None), clearable=True, placeholder="e.g. metric__global_basin_min_x", style={"fontSize": "0.76em"}), md=True),
                    ],
                    className="g-2 align-items-center mt-1",
                ),
                html.Small(id="umap-trajectory-status", className="text-muted", children="Compute an embedding, then choose a PMF metric."),
                dcc.Loading(
                    dcc.Graph(
                        id="umap-trajectory-graph",
                        figure=_error_fig("Compute embedding first, then choose trajectory metric."),
                        style={"height": "78vh"},
                        config={"displaylogo": False},
                    ),
                    type="dot",
                ),
                html.Div(
                    [
                        html.Div("Residue composition bin", style={"fontWeight": 600, "fontSize": "0.86em"}),
                        dcc.Slider(id="umap-residue-bin-slider", min=1, max=1, step=1, value=1, marks={1: "1"}, tooltip={"placement": "bottom", "always_visible": False}),
                        html.Small(id="umap-residue-status", className="text-muted"),
                        dcc.Graph(id="umap-residue-composition", figure=_error_fig("No trajectory bin selected."), style={"height": "44vh"}),
                    ],
                    className="mt-2",
                ),
                html.Div(
                    [
                        html.Div("Trajectory sequence grammar", style={"fontWeight": 600, "fontSize": "0.86em"}),
                        html.Small(id="umap-trajectory-sequence-status", className="text-muted"),
                        dcc.Graph(id="umap-trajectory-sequence-graph", figure=_error_fig("No trajectory grammar available."), style={"height": "64vh"}, config={"displaylogo": False}),
                    ],
                    className="mt-2",
                ),
            ]
        ),
        className="mt-2",
    )

    help_modal = dbc.Modal(
        [
            dbc.ModalHeader(dbc.ModalTitle("UMAP / manifold tab help")),
            dbc.ModalBody(
                [
                    html.H5("Embedding methods"),
                    html.P("UMAP preserves local neighborhoods and is useful as the default visual atlas. DensMAP is UMAP with local-density preservation. Isomap preserves geodesic distances on a kNN graph."),
                    html.P([html.B("Diffusion Map: "), "builds a kNN diffusion/random-walk graph and uses its smooth eigenvectors as intrinsic coordinates. Use it when you want reaction-coordinate-like latent axes, smooth continua, or trajectory/path coordinates from PMF-shape or sequence-language space."]),
                    html.P([html.B("PHATE: "), "uses diffusion probabilities transformed into a potential-distance geometry. In this tab it is implemented as a dependency-free PHATE-style map: adaptive diffusion graph → P^t → -log(P^t) potential → classical MDS. Use it for visually readable progressions, branches, and continua."]),
                    html.H5("Scientific interpretation"),
                    html.P("For PMF-shape input, Diffusion Map/PHATE act on the same preprocessed sqrt(P)-style representation used by UMAP, so nearby points mean similar free-energy/probability landscapes. For sequence input, they describe the geometry induced by residue composition, motif, and sequence descriptors."),
                    html.P("Neither method creates a new peptide. They show the sampled manifold geometry; paths and clusters are hypotheses to validate against PMFs, residue grammar, and physical descriptors."),
                    html.H5("PMF representation"),
                    html.Ul([
                        html.Li([html.B("sqrt(P): "), "default; each PMF on the Hellinger sphere. Euclidean distances in sqrt(P)-space equal Hellinger distances in P-space. Geometrically sound and the recommended choice."]),
                        html.Li([html.B("log(P): "), "amplifies rare/tail conformations suppressed in probability space. Useful for visualising fine differences between near-zero-probability states, but breaks Hellinger geometry — use with non-Hellinger metrics."]),
                        html.Li([html.B("F → P: "), "converts free-energy F to probability P via Boltzmann; same as sqrt(P) downstream."]),
                    ]),
                    html.H5("Controls"),
                    html.Ul([
                        html.Li("For UMAP/DensMAP, neighbors controls the local graph and min_dist controls how tightly points may pack."),
                        html.Li("For Diffusion Map and PHATE, neighbors controls the kNN diffusion graph and the second parameter becomes diffusion time t."),
                        html.Li("For Isomap, neighbors controls the geodesic graph; the second parameter is ignored."),
                        html.Li("The distance metric still controls pairwise distances before graph construction."),
                        html.Li("The dimensionality diagnostics report effective input dimension and how much input variance is linearly recoverable from the displayed map; this is not formal nonlinear explained variance."),
                        html.Li([html.B("PCA cap: "), "maximum number of PCs fed to the embedding method. Default 64 (≈99% EVR). Raising it rarely improves quality; lowering it is useful for very small datasets."]),
                        html.Li([html.B("Feature gradient arrows: "), "biplot arrows showing the top 5 features with strongest Spearman correlation to UMAP axes. Arrow direction indicates feature gradient; length indicates correlation magnitude."]),
                        html.Li([html.B("Out-of-sample projection (note): "), "Diffusion Map, PHATE, and Isomap are transductive — they fit on the entire (fit ∪ plot) set at once. There is no mathematical out-of-sample projection for new variants added after the fact; a new computation is required. Only UMAP/DensMAP support true out-of-sample transform."]),
                    ]),
                ]
            ),
            dbc.ModalFooter(dbc.Button("Close", id="umap-help-close", className="ms-auto", n_clicks=0)),
        ],
        id="umap-help-modal",
        is_open=False,
        size="lg",
    )

    layout = html.Div(
        [
            dbc.Row(
                [dbc.Col(dbc.Button("Help", id="umap-help-open", n_clicks=0, size="sm", color="secondary", outline=True), width="auto")],
                justify="end",
                className="mb-1",
            ),
            help_modal,
            dcc.Store(id="umap-embedding-data", data={}),
            dcc.Store(id="umap-trajectory-bins-data", data={"bins": []}),
            dcc.Store(id="umap-trajectory-selected-bin", data=1),
            dbc.Row(
                [
                    dbc.Col(controls, md=3),
                    dbc.Col(
                        [
                            dcc.Loading(
                                dcc.Graph(
                                    id="umap-graph",
                                    figure=placeholder_fig,
                                    style={"height": "76vh"},
                                    config={"displaylogo": False},
                                ),
                                type="dot",
                            ),
                            html.Div(id="umap-hover-info", className="text-muted mt-1",
                                     style={"fontSize": "0.8em", "minHeight": "1.2em"}),
                            html.Div(id="umap-metrics", className="mt-2"),
                            html.Div(id="umap-dm-spectral", className="mt-2"),
                        ],
                        md=9,
                    ),
                ],
                className="g-2",
            ),
            pmf_viewer,
            sequence_region_viewer,
            sequence_cluster_viewer,
            trajectory_viewer,
            html.Hr(),
            dcc.Loading(
                html.Div(id="umap-corr-table", className="mt-2"),
                type="dot",
            ),
        ]
    )

    return layout
