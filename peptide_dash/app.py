# peptide_dash/app.py
from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

from dash import Dash, Input, Output, State, dcc, html, no_update

from .tabs import build_tabs_layout, register_all_callbacks

if TYPE_CHECKING:
    from .data.loader import BackgroundLoader

# ── Tab group definitions ──────────────────────────────────────────────────

_TAB_LABELS: dict[str, str] = {
    "features": "Features",
    "stats": "Stats",
    "pca": "Variance & PCA",
    "diffusion_map": "Diffusion Map",
    "umap": "UMAP",
    "umap_region_pmf": "UMAP → PMF",
    "basin_landscape": "Basin Landscape",
    "pmf_dendrogram_tab": "PMF Dendrogram",
    "curves": "Curves",
    "timeseries": "Timeseries",
    "rama2d_tab": "Rama 2D",
    "convergence": "Convergence",
    "diagnosis": "Diagnosis",
    "corr": "Correlation",
    "readme": "README",
}

_GROUPS: List[Tuple[str, str, List[str]]] = [
    ("features",    "Features",       ["features", "stats"]),
    ("dimred",      "Dim. Reduction", ["pca", "diffusion_map", "umap", "umap_region_pmf", "basin_landscape"]),
    ("thermo",      "Thermodynamics", ["pmf_dendrogram_tab", "curves", "timeseries", "rama2d_tab"]),
    ("convergence", "Convergence",    ["convergence", "diagnosis", "corr"]),
    ("info",        "Info",           ["readme"]),
]

# group_id → [tab_value, ...]
_GROUPS_MAP: dict[str, List[str]] = {g[0]: g[2] for g in _GROUPS}

# tab_value → group_id
_TAB_TO_GROUP: dict[str, str] = {
    tab: gid for gid, _label, tabs in _GROUPS for tab in tabs
}


# ── Loading overlay CSS (unchanged) ───────────────────────────────────────

_LOADING_CSS = """
#loading-overlay {
    position: fixed;
    inset: 0;
    background: #0f1117;
    color: #e0e0e0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    z-index: 9999;
    font-family: "Inter", "Segoe UI", system-ui, sans-serif;
}
#loading-overlay h1 {
    font-size: 1.6rem;
    font-weight: 700;
    margin-bottom: 0.5rem;
    color: #fff;
}
#loading-overlay .sub {
    font-size: 0.95rem;
    color: #aaa;
    margin-bottom: 2rem;
}
.progress-bar-track {
    width: 420px;
    max-width: 90vw;
    height: 10px;
    background: #2a2d3a;
    border-radius: 6px;
    overflow: hidden;
    margin-bottom: 0.75rem;
}
.progress-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    border-radius: 6px;
    transition: width 0.4s ease;
    min-width: 4px;
}
.progress-label {
    font-size: 0.82rem;
    color: #888;
    text-align: center;
    min-height: 1.2em;
}
.error-box {
    background: #2d1515;
    border: 1px solid #a33;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin-top: 1rem;
    max-width: 640px;
    font-size: 0.82rem;
    font-family: monospace;
    white-space: pre-wrap;
    color: #f88;
}
"""


# ── Component builders ─────────────────────────────────────────────────────

def _loading_overlay() -> html.Div:
    return html.Div(
        id="loading-overlay",
        children=[
            html.H1("The Peptideverse"),
            html.Div(
                "Loading features – the dashboard will appear automatically.",
                className="sub",
            ),
            html.Div(
                html.Div(
                    id="progress-bar-fill",
                    className="progress-bar-fill",
                    style={"width": "0%"},
                ),
                className="progress-bar-track",
            ),
            html.Div(
                id="progress-label",
                className="progress-label",
                children="Initialising …",
            ),
            html.Div(
                id="progress-error",
                className="error-box",
                style={"display": "none"},
            ),
        ],
    )


def _header() -> html.Div:
    return html.Div(
        className="app-header",
        children=[
            html.Span("The Peptideverse", className="brand-title"),
            html.Button(
                "🌙 Dark",
                id="theme-toggle-btn",
                className="theme-toggle-btn",
                n_clicks=0,
            ),
        ],
    )


def _build_nav_children(active_group: str, active_tab: str) -> list:
    group_buttons = []
    for gid, glabel, _ in _GROUPS:
        cls = "nav-group-btn nav-group-btn-active" if gid == active_group else "nav-group-btn"
        group_buttons.append(
            html.Button(
                glabel,
                id={"type": "nav-group-btn", "index": gid},
                className=cls,
                n_clicks=0,
            )
        )

    tab_buttons = []
    for tab_value in _GROUPS_MAP.get(active_group, []):
        cls = "nav-tab-btn nav-tab-btn-active" if tab_value == active_tab else "nav-tab-btn"
        tab_buttons.append(
            html.Button(
                _TAB_LABELS.get(tab_value, tab_value),
                id={"type": "nav-tab-btn", "index": tab_value},
                className=cls,
                n_clicks=0,
            )
        )

    return [
        html.Div(group_buttons, className="nav-group-bar"),
        html.Div(tab_buttons, className="nav-tab-bar"),
    ]


# ── App factory ────────────────────────────────────────────────────────────

def create_app(ctx, loader=None):
    app = Dash(__name__, suppress_callback_exceptions=True)
    app.index_string = app.index_string.replace(
        "</head>", f"<style>{_LOADING_CSS}</style></head>"
    )

    register_all_callbacks(app, ctx)

    # ── Static layout (stores + shell always present) ──────────────────
    static_stores = [
        dcc.Store(id="app-ready-store", data={"ready": False}),
        dcc.Store(id="theme-store", data="light", storage_type="local"),
        dcc.Store(id="active-group-store", data="features"),
        dcc.Store(id="active-tab-store", data="features"),
    ]

    app_shell = html.Div(
        id="app-shell",
        className="theme-light",
        style={"display": "none"},
        children=[
            _header(),
            html.Div(id="grouped-nav"),
            html.Div(id="app-content"),
            html.Div(id="theme-class-dummy", style={"display": "none"}),
        ],
    )

    if loader is not None:
        app.layout = html.Div(
            id="page-root",
            children=static_stores + [
                dcc.Interval(id="loading-poll", interval=400, n_intervals=0, disabled=False),
                _loading_overlay(),
                app_shell,
            ],
        )

        @app.callback(
            Output("progress-bar-fill", "style"),
            Output("progress-label", "children"),
            Output("progress-error", "children"),
            Output("progress-error", "style"),
            Output("loading-poll", "disabled"),
            Output("app-ready-store", "data"),
            Input("loading-poll", "n_intervals"),
            prevent_initial_call=False,
        )
        def _poll_loading(n_intervals):
            state = loader.state_dict()
            phase = state["phase"]
            pct = min(100.0, state["overall_frac"] * 100)
            bar = {"width": f"{pct:.1f}%"}
            elapsed = state["elapsed_s"]

            if phase == "error":
                label = f"Error after {elapsed:.0f}s"
                err_text = state.get("error") or state.get("message", "unknown error")
                return bar, label, err_text, {"display": "block"}, True, {"ready": False}

            if phase == "done":
                return {"width": "100%"}, "Done – rendering …", "", {"display": "none"}, True, {"ready": True}

            sub_done = state["sub_done"]
            sub_total = state["sub_total"]
            msg = state["message"] or phase
            if sub_total > 0:
                msg += f"  ({sub_done}/{sub_total} files)"
            if elapsed > 2:
                msg += f"  ·  {elapsed:.0f}s elapsed"
            eta = state["eta_s"]
            if eta > 1:
                msg += f"  ·  ~{eta:.0f}s left"

            return bar, msg, "", {"display": "none"}, False, {"ready": False}

        @app.callback(
            Output("loading-overlay", "style"),
            Output("app-shell", "style"),
            Output("app-content", "children"),
            Input("app-ready-store", "data"),
            prevent_initial_call=False,
        )
        def _swap_to_real_content(store_data):
            if not (store_data or {}).get("ready"):
                return no_update, no_update, no_update
            content = html.Div(
                className="app-main",
                children=html.Div(
                    className="content-container",
                    children=build_tabs_layout(ctx),
                ),
            )
            return {"display": "none"}, {"display": "block"}, content

    else:
        content = html.Div(
            className="app-main",
            children=html.Div(
                className="content-container",
                children=build_tabs_layout(ctx),
            ),
        )
        app_shell.style = {"display": "block"}
        # children order: [_header(), grouped-nav, app-content, theme-class-dummy]
        app_shell.children[1].children = _build_nav_children("features", "features")
        app_shell.children[2].children = content
        app.layout = html.Div(
            id="page-root",
            children=static_stores + [app_shell],
        )

    # ── Clientside: apply theme class to #app-shell ────────────────────
    app.clientside_callback(
        """
        function(theme) {
            var shell = document.getElementById('app-shell');
            if (shell) {
                shell.className = 'theme-' + (theme || 'light');
            }
            var btn = document.getElementById('theme-toggle-btn');
            if (btn) {
                btn.textContent = theme === 'dark' ? '☀ Light' : '🌙 Dark';
            }
            return null;
        }
        """,
        Output("theme-class-dummy", "children"),
        Input("theme-store", "data"),
    )

    # ── Clientside: toggle theme store ────────────────────────────────
    app.clientside_callback(
        """
        function(n_clicks, current_theme) {
            return current_theme === 'dark' ? 'light' : 'dark';
        }
        """,
        Output("theme-store", "data"),
        Input("theme-toggle-btn", "n_clicks"),
        State("theme-store", "data"),
        prevent_initial_call=True,
    )

    # ── Server: grouped nav callbacks ─────────────────────────────────
    from dash import ALL

    @app.callback(
        Output("grouped-nav", "children"),
        Input("active-group-store", "data"),
        Input("active-tab-store", "data"),
    )
    def _render_grouped_nav(active_group, active_tab):
        return _build_nav_children(active_group or "features", active_tab or "features")

    @app.callback(
        Output("active-group-store", "data"),
        Output("active-tab-store", "data"),
        Output("tabs", "value"),
        Input({"type": "nav-group-btn", "index": ALL}, "n_clicks"),
        Input({"type": "nav-tab-btn", "index": ALL}, "n_clicks"),
        State("active-group-store", "data"),
        prevent_initial_call=True,
    )
    def _select_nav(group_clicks, tab_clicks, current_group):
        from dash import ctx as dash_ctx
        from dash.exceptions import PreventUpdate
        triggered = dash_ctx.triggered_id
        if triggered is None or not isinstance(triggered, dict):
            raise PreventUpdate
        if triggered.get("type") == "nav-group-btn":
            new_group = triggered["index"]
            first_tab = _GROUPS_MAP[new_group][0]
            return new_group, first_tab, first_tab
        if triggered.get("type") == "nav-tab-btn":
            new_tab = triggered["index"]
            return current_group or "features", new_tab, new_tab
        raise PreventUpdate

    server = app.server
    return app, server
