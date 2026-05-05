from __future__ import annotations

from dash import dcc, html

from . import (
    features,
    diagnostics_tab,
    correlation,
    pca_tab,
    diffusion_map_tab,
    umap_tab,
    umap_region_pmf_tab,
    pmf_dendrogram_tab,
    curves,
    timeseries_tab,
    convergence,
    rama2d_tab,
    readme_tab,
    stats_tab,
    basin_tab,
)


def _wrap(children):
    return html.Div(children=children, className="tab-body")


def build_tabs_layout(ctx):
    tabs = [
        dcc.Tab(
            label="Features",
            value="features",
            className="app-tab",
            selected_className="app-tab-selected",
            children=_wrap(features.layout(ctx)),
        ),

        dcc.Tab(
            label="Diagnosis",
            value="diagnosis",
            className="app-tab",
            selected_className="app-tab-selected",
            children=_wrap(diagnostics_tab.layout(ctx)),
        ),
        dcc.Tab(
            label="Correlation",
            value="corr",
            className="app-tab",
            selected_className="app-tab-selected",
            children=_wrap(correlation.layout(ctx)),
        ),
        dcc.Tab(
            label="Variance & PCA",
            value="pca",
            className="app-tab",
            selected_className="app-tab-selected",
            children=_wrap(pca_tab.layout(ctx)),
        ),

        dcc.Tab(
            label="Diffusion Map",
            value="diffusion_map",
            className="app-tab",
            selected_className="app-tab-selected",
            children=_wrap(diffusion_map_tab.layout(ctx)),
        ),
        dcc.Tab(
            label="UMAP",
            value="umap",
            className="app-tab",
            selected_className="app-tab-selected",
            children=_wrap(umap_tab.layout(ctx)),
        ),
        dcc.Tab(
            label="UMAP → PMF",
            value="umap_region_pmf",
            className="app-tab",
            selected_className="app-tab-selected",
            children=_wrap(umap_region_pmf_tab.layout(ctx)),
        ),
        dcc.Tab(
            label="PMF-based Dendrogram",
            value="pmf_dendrogram_tab",
            className="app-tab",
            selected_className="app-tab-selected",
            children=_wrap(pmf_dendrogram_tab.layout(ctx)),
        ),
        dcc.Tab(
            label="Curves",
            value="curves",
            className="app-tab",
            selected_className="app-tab-selected",
            children=_wrap(curves.layout(ctx)),
        ),
        dcc.Tab(
            label="Timeseries",
            value="timeseries",
            className="app-tab",
            selected_className="app-tab-selected",
            children=_wrap(timeseries_tab.layout(ctx)),
        ),
        dcc.Tab(
            label="Convergence",
            value="convergence",
            className="app-tab",
            selected_className="app-tab-selected",
            children=_wrap(convergence.layout(ctx)),
        ),
        dcc.Tab(
            label="Rama 2D",
            value="rama2d_tab",
            className="app-tab",
            selected_className="app-tab-selected",
            children=_wrap(rama2d_tab.layout(ctx)),
        ),
        dcc.Tab(
            label="README",
            value="readme",
            className="app-tab",
            selected_className="app-tab-selected",
            children=_wrap(readme_tab.layout(ctx)),
        ),
        dcc.Tab(
            label="Stats",
            value="stats",
            className="app-tab",
            selected_className="app-tab-selected",
            children=_wrap(stats_tab.layout(ctx)),
        ),
        dcc.Tab(
            label="Basin Landscape",
            value="basin_landscape",
            className="app-tab",
            selected_className="app-tab-selected",
            children=_wrap(basin_tab.layout(ctx)),
        ),
    ]

    return dcc.Tabs(
        id="tabs",
        value="features",
        parent_className="app-tabs-container app-tabs-hidden",
        className="app-tabs",
        children=tabs,
    )


def register_all_callbacks(app, ctx):
    features.register_callbacks(app, ctx)
    diagnostics_tab.register_callbacks(app, ctx)
    correlation.register_callbacks(app, ctx)
    pca_tab.register_callbacks(app, ctx)
    diffusion_map_tab.register_callbacks(app, ctx)
    umap_tab.register_callbacks(app, ctx)
    umap_region_pmf_tab.register_callbacks(app, ctx)
    pmf_dendrogram_tab.register_callbacks(app, ctx)
    curves.register_callbacks(app, ctx)
    timeseries_tab.register_callbacks(app, ctx)
    convergence.register_callbacks(app, ctx)
    rama2d_tab.register_callbacks(app, ctx)
    readme_tab.register_callbacks(app, ctx)
    stats_tab.register_callbacks(app, ctx)
    basin_tab.register_callbacks(app, ctx)
