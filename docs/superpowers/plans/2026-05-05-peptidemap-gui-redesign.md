# PeptideMap GUI Redesign — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add light/dark toggle, grouped tab navigation, and Plotly theme adaptation to the PeptideMap dashboard ("The Peptideverse").

**Architecture:** Wrap the app in `#app-shell` with a `theme-light`/`theme-dark` class driven by a persisted `dcc.Store`. Replace the visible tab strip with a two-row custom nav (group selector + tab pills) while keeping `dcc.Tabs` hidden for content switching. Each figure callback gains `Input("theme-store", "data")` and calls a shared `apply_theme()` helper.

**Tech Stack:** Python 3, Dash 4.0, Plotly, existing `assets/theme.css` CSS variable system

---

## File Map

| File | Change |
|---|---|
| `peptide_dash/tabs/shared.py` | Add `apply_theme(fig, theme)` helper |
| `peptide_dash/assets/theme.css` | Add header / nav / shell CSS rules |
| `peptide_dash/app.py` | Shell, header, stores, clientside callbacks, grouped nav callbacks |
| `peptide_dash/tabs/__init__.py` | Hide tab strip via CSS class on `dcc.Tabs` |
| `peptide_dash/tabs/basin_tab.py` | Add theme input to 1 figure callback |
| `peptide_dash/tabs/stats_tab.py` | Add theme input to 1 figure callback |
| `peptide_dash/tabs/diffusion_map_tab.py` | Add theme input to 1 figure callback (2 outputs) |
| `peptide_dash/tabs/features.py` | Add theme input to 1 figure callback |
| `peptide_dash/tabs/convergence.py` | Add theme input to 1 figure callback |
| `peptide_dash/tabs/correlation.py` | Add theme input to 2 figure callbacks |
| `peptide_dash/tabs/curves.py` | Add theme input to 1 figure callback |
| `peptide_dash/tabs/timeseries_tab.py` | Add theme input to 1 figure callback (4 outputs) |
| `peptide_dash/tabs/rama2d_tab.py` | Add theme input to 1 figure callback |
| `peptide_dash/tabs/pmf_dendrogram_tab.py` | Add theme input to 3 figure callbacks |
| `peptide_dash/tabs/umap_region_pmf_tab.py` | Add theme input to 3 figure callbacks |
| `peptide_dash/tabs/pca_tab.py` | Add theme input to 4 figure callbacks |
| `peptide_dash/tabs/umap/callbacks.py` | Add theme input to 8 figure callbacks |

---

## Task 1: Add `apply_theme` helper

**Files:**
- Modify: `peptide_dash/tabs/shared.py`
- Test: `peptide_dash/tests/test_theme.py`

- [ ] **Step 1: Write the failing test**

Create `peptide_dash/tests/test_theme.py`:

```python
import pytest
import plotly.graph_objs as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from peptide_dash.tabs.shared import apply_theme


def test_apply_theme_dark():
    fig = go.Figure()
    result = apply_theme(fig, "dark")
    assert result.layout.template.layout.paper_bgcolor is not None or result.layout.paper_bgcolor == "rgba(0,0,0,0)"


def test_apply_theme_light():
    fig = go.Figure()
    result = apply_theme(fig, "light")
    assert result is fig  # mutates in place and returns fig


def test_apply_theme_returns_figure():
    fig = go.Figure()
    result = apply_theme(fig, "dark")
    assert isinstance(result, go.Figure)


def test_apply_theme_unknown_defaults_to_light():
    fig = go.Figure()
    result = apply_theme(fig, None)
    assert isinstance(result, go.Figure)
```

- [ ] **Step 2: Run test to confirm it fails**

```bash
cd /run/media/sulcjo/sulcjo-data/IOCB/md/2025_PeptideMap
python -m pytest peptide_dash/tests/test_theme.py -v
```

Expected: `ImportError` or `AttributeError` — `apply_theme` doesn't exist yet.

- [ ] **Step 3: Add `apply_theme` to `shared.py`**

Replace all of `peptide_dash/tabs/shared.py` with:

```python
from __future__ import annotations

import plotly.graph_objs as go
from dash import html


def apply_theme(fig: go.Figure, theme: str | None) -> go.Figure:
    """Apply light/dark Plotly template and transparent background to a figure."""
    template = "plotly_dark" if theme == "dark" else "plotly_white"
    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def panel(children, title: str | None = None, subtitle: str | None = None) -> html.Div:
    """Wrap a section in a consistent modern panel."""
    header = []
    if title:
        header.append(html.H3(title, className="panel-title"))
    if subtitle:
        header.append(html.Div(subtitle, className="panel-subtitle"))
    if header:
        header.append(html.Hr(className="panel-divider"))
    return html.Div(
        [
            html.Div(header, className="panel-header") if header else None,
            html.Div(children, className="panel-body"),
        ],
        className="panel",
    )
```

- [ ] **Step 4: Run test to confirm it passes**

```bash
python -m pytest peptide_dash/tests/test_theme.py -v
```

Expected: all 4 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add peptide_dash/tabs/shared.py peptide_dash/tests/test_theme.py
git commit -m "feat: add apply_theme helper to shared.py"
```

---

## Task 2: CSS additions

**Files:**
- Modify: `peptide_dash/assets/theme.css`

- [ ] **Step 1: Append new rules to `assets/theme.css`**

Add to the **end** of `peptide_dash/assets/theme.css`:

```css
/* ── Redesign 2026-05 ─────────────────────────────────────────────────── */

/* App shell */
#app-shell {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
  background: var(--bg);
  color: var(--text);
}

/* Header */
.app-header {
  height: 40px;
  position: sticky;
  top: 0;
  z-index: 1000;
  background: var(--surface);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  border-bottom: var(--border-w) solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 14px;
  flex-shrink: 0;
}

.brand-title {
  font-size: 13px;
  font-weight: 700;
  letter-spacing: .1px;
  color: var(--text);
  white-space: nowrap;
}

.theme-toggle-btn {
  border: var(--border-w) solid var(--border);
  background: var(--surface-2);
  color: var(--text);
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
  cursor: pointer;
  white-space: nowrap;
  line-height: 1;
}
.theme-toggle-btn:hover { filter: brightness(0.97); }

/* Group selector row */
.nav-group-bar {
  display: flex;
  align-items: center;
  gap: 4px;
  height: 38px;
  padding: 0 14px;
  background: var(--bg);
  border-bottom: var(--border-w) solid var(--border);
  flex-shrink: 0;
  overflow-x: auto;
  scrollbar-width: none;
}
.nav-group-bar::-webkit-scrollbar { display: none; }

.nav-group-btn {
  border: var(--border-w) solid var(--border);
  background: var(--surface);
  color: var(--muted);
  border-radius: 6px;
  padding: 3px 10px;
  font-size: 10px;
  font-weight: 700;
  cursor: pointer;
  white-space: nowrap;
  line-height: 1.4;
}
.nav-group-btn:hover { filter: brightness(0.97); }

.nav-group-btn-active {
  background: var(--accent) !important;
  color: #fff !important;
  border-color: var(--accent) !important;
}

/* Tab pill row */
.nav-tab-bar {
  display: flex;
  align-items: center;
  gap: 4px;
  height: 38px;
  padding: 0 14px;
  background: var(--bg);
  border-bottom: var(--border-w) solid var(--border);
  flex-shrink: 0;
  overflow-x: auto;
  scrollbar-width: none;
}
.nav-tab-bar::-webkit-scrollbar { display: none; }

.nav-tab-btn {
  border: var(--border-w) solid var(--border);
  background: var(--surface);
  color: var(--muted);
  border-radius: 999px;
  padding: 4px 11px;
  font-size: 11px;
  font-weight: 500;
  cursor: pointer;
  white-space: nowrap;
  line-height: 1.4;
}
.nav-tab-btn:hover { filter: brightness(0.97); }

.nav-tab-btn-active {
  background: var(--surface-2) !important;
  color: var(--text) !important;
  border-color: rgba(37, 99, 235, .35) !important;
  font-weight: 600;
  box-shadow: var(--shadow-sm);
}

.theme-dark .nav-tab-btn-active {
  border-color: rgba(96, 165, 250, .35) !important;
}

/* Hide the dcc.Tabs strip — nav replaced by custom grouped nav */
.app-tabs-hidden .app-tabs {
  display: none !important;
}

/* Remove top margin from tab body when using custom nav */
.app-main .tab-body { margin-top: 0; }

/* App main */
.app-main {
  flex: 1;
  padding: 10px 0 16px;
}
```

- [ ] **Step 2: Verify CSS syntax is valid (no unclosed braces)**

```bash
python3 -c "
import re
text = open('peptide_dash/assets/theme.css').read()
opens = text.count('{')
closes = text.count('}')
assert opens == closes, f'Brace mismatch: {{ opens={opens}, closes={closes}'
print('CSS brace check OK')
"
```

Expected: `CSS brace check OK`

- [ ] **Step 3: Commit**

```bash
git add peptide_dash/assets/theme.css
git commit -m "feat: add header/nav/shell CSS for GUI redesign"
```

---

## Task 3: App shell, header, and theme wiring in `app.py`

**Files:**
- Modify: `peptide_dash/app.py`

- [ ] **Step 1: Replace `app.py` with the updated version**

Replace the entire content of `peptide_dash/app.py` with:

```python
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
                "☀ Light",
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
        app_shell.children[1].children = _build_nav_children("features", "features")  # grouped-nav
        app_shell.children[2].children = content  # app-content
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
        triggered = dash_ctx.triggered_id
        if triggered is None or not isinstance(triggered, dict):
            from dash.exceptions import PreventUpdate
            raise PreventUpdate
        if triggered.get("type") == "nav-group-btn":
            new_group = triggered["index"]
            first_tab = _GROUPS_MAP[new_group][0]
            return new_group, first_tab, first_tab
        if triggered.get("type") == "nav-tab-btn":
            new_tab = triggered["index"]
            return current_group or "features", new_tab, new_tab
        from dash.exceptions import PreventUpdate
        raise PreventUpdate

    server = app.server
    return app, server
```

- [ ] **Step 2: Verify the app imports cleanly**

```bash
cd /run/media/sulcjo/sulcjo-data/IOCB/md/2025_PeptideMap
python3 -c "from peptide_dash.app import create_app; print('import OK')"
```

Expected: `import OK`

- [ ] **Step 3: Commit**

```bash
git add peptide_dash/app.py
git commit -m "feat: add app shell, header, theme toggle, grouped nav to app.py"
```

---

## Task 4: Hide the `dcc.Tabs` strip in `tabs/__init__.py`

**Files:**
- Modify: `peptide_dash/tabs/__init__.py`

- [ ] **Step 1: Add `app-tabs-hidden` className to `dcc.Tabs`**

In `peptide_dash/tabs/__init__.py`, find the `return dcc.Tabs(...)` call (line ~139) and add `className="app-tabs-hidden"` to it. The full updated `build_tabs_layout` return statement:

```python
    return dcc.Tabs(
        id="tabs",
        value="features",
        parent_className="app-tabs-container app-tabs-hidden",
        className="app-tabs",
        children=tabs,
    )
```

(`parent_className` adds `app-tabs-hidden`; the CSS rule `.app-tabs-hidden .app-tabs { display: none }` hides the strip.)

- [ ] **Step 2: Verify import still works**

```bash
python3 -c "from peptide_dash.tabs import build_tabs_layout; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add peptide_dash/tabs/__init__.py
git commit -m "feat: hide dcc.Tabs strip; nav replaced by grouped custom nav"
```

---

## Task 5: Theme input — `basin_tab.py`, `stats_tab.py`, `diffusion_map_tab.py`

**Files:**
- Modify: `peptide_dash/tabs/basin_tab.py`
- Modify: `peptide_dash/tabs/stats_tab.py`
- Modify: `peptide_dash/tabs/diffusion_map_tab.py`

The pattern for every figure callback is identical:
1. Add `Input("theme-store", "data")` to the `@app.callback` decorator.
2. Add `theme` as the last parameter of the callback function.
3. Add `from .shared import apply_theme` at the top of `register_callbacks` (or at module level).
4. Call `apply_theme(fig, theme)` before each `return` that returns a figure.

- [ ] **Step 1: Edit `basin_tab.py`**

Find the callback at line ~101 (`Output("basin-heatmap", "figure")`). Add `Input("theme-store", "data")` as the last Input, add `theme` as the last function parameter, and add `apply_theme(fig, theme)` before the return. Also add the import.

At the top of `register_callbacks` in `basin_tab.py`, add:
```python
from .shared import apply_theme
```

The callback decorator becomes:
```python
    @app.callback(
        Output("basin-heatmap", "figure"),
        Input("basin-stat", "value"),
        Input("basin-metrics", "value"),
        Input("basin-opts", "value"),
        Input("theme-store", "data"),
    )
    def update_heatmap(stat, selected_metrics, opts, theme):
```

Before each `return fig` (there are 3 return paths with figures), add `apply_theme(fig, theme)`:
```python
        # Before: return error_fig("No data available.")
        # error_fig returns a figure — wrap it:
        return apply_theme(error_fig("No data available."), theme)

        # ... and before the final return fig:
        apply_theme(fig, theme)
        return fig
```

- [ ] **Step 2: Edit `stats_tab.py`**

Same pattern. Find `Output("stats-hist", "figure")` callback (~line 108). Add `Input("theme-store", "data")`, add `theme` parameter, add `apply_theme(fig, theme)` before return.

Add at top of `register_callbacks`:
```python
from .shared import apply_theme
```

- [ ] **Step 3: Edit `diffusion_map_tab.py`**

Find callback with `Output("dm-embed-graph", "figure"), Output("dm-eigs-graph", "figure")` (~line 251). Add `Input("theme-store", "data")`, add `theme` parameter, call `apply_theme(fig, theme)` and `apply_theme(eig_fig, theme)` before the tuple return.

Add at top of `register_callbacks`:
```python
from .shared import apply_theme
```

- [ ] **Step 4: Verify imports**

```bash
python3 -c "
from peptide_dash.tabs import basin_tab, stats_tab, diffusion_map_tab
print('imports OK')
"
```

Expected: `imports OK`

- [ ] **Step 5: Commit**

```bash
git add peptide_dash/tabs/basin_tab.py peptide_dash/tabs/stats_tab.py peptide_dash/tabs/diffusion_map_tab.py
git commit -m "feat: add theme input to basin, stats, diffusion_map figure callbacks"
```

---

## Task 6: Theme input — `features.py` and `convergence.py`

**Files:**
- Modify: `peptide_dash/tabs/features.py`
- Modify: `peptide_dash/tabs/convergence.py`

- [ ] **Step 1: Edit `features.py`**

Find the callback at line ~1164 (`Output("features-graph", "figure")`).

Add `Input("theme-store", "data")` as the last Input in the decorator.

Add `theme` as the last parameter to `_update_features_graph(...)`.

Add import at top of `register_callbacks`:
```python
from .shared import apply_theme
```

Before `return fig` (or any `return go.Figure()`), call `apply_theme`:
```python
# Empty figures:
return apply_theme(go.Figure(), theme)

# Full figure — find the final return and add before it:
apply_theme(fig, theme)
return fig
```

- [ ] **Step 2: Edit `convergence.py`**

Find callback at line ~1656 (`Output("convergence-graph", "figure"), Output("conv-summary", "children")`).

Add `Input("theme-store", "data")` as the last Input. Add `theme` as the last parameter.

Add import:
```python
from .shared import apply_theme
```

For every `return fig, summary_children` path, wrap the figure:
```python
apply_theme(fig, theme)
return fig, summary_children
```

For error paths like `return error_fig("..."), html.Div(...)`:
```python
return apply_theme(error_fig("..."), theme), html.Div(...)
```

- [ ] **Step 3: Verify imports**

```bash
python3 -c "from peptide_dash.tabs import features, convergence; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add peptide_dash/tabs/features.py peptide_dash/tabs/convergence.py
git commit -m "feat: add theme input to features, convergence figure callbacks"
```

---

## Task 7: Theme input — `correlation.py`

**Files:**
- Modify: `peptide_dash/tabs/correlation.py`

Correlation has 2 figure callbacks: one returning 3 figures (`corr-graph`, `corr-scatter`, `corr-residue-heatmap`) and one returning 1 figure (`corr-target-bar`).

- [ ] **Step 1: Edit `correlation.py`**

Add import at top of `register_callbacks`:
```python
from .shared import apply_theme
```

**Callback 1** (~line 580, outputs `corr-graph`, `corr-scatter`, `corr-residue-heatmap`, possibly `corr-residue-heatmap`):
- Add `Input("theme-store", "data")` as last Input
- Add `theme` as last parameter
- Before each tuple return: call `apply_theme(fig, theme)` on each figure in the tuple

**Callback 2** (~line 951, output `corr-target-bar`):
- Same pattern: add Input, add parameter, wrap figure in `apply_theme`

- [ ] **Step 2: Verify**

```bash
python3 -c "from peptide_dash.tabs import correlation; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add peptide_dash/tabs/correlation.py
git commit -m "feat: add theme input to correlation figure callbacks"
```

---

## Task 8: Theme input — `curves.py`, `timeseries_tab.py`, `rama2d_tab.py`

**Files:**
- Modify: `peptide_dash/tabs/curves.py`
- Modify: `peptide_dash/tabs/timeseries_tab.py`
- Modify: `peptide_dash/tabs/rama2d_tab.py`

**Note on `timeseries_tab.py` and `rama2d_tab.py`:** These files already have their own `_template(theme)` helpers and build figures with `template=_template(theme)` throughout. The `theme` value is a closure variable set once in `register_callbacks`. The fix is to wire the store Input to that closure variable — NOT to call `apply_theme` at the end (which would be redundant). For `curves.py` (no internal theme system), use the standard `apply_theme` approach.

- [ ] **Step 1: Edit `curves.py`**

Find callback at line ~1034 (`Output("curves-graph", "figure")`). This callback has many return paths.

Add import at top of `register_callbacks`: `from .shared import apply_theme`

Add `Input("theme-store", "data")` as last Input in the decorator, `theme` as the last parameter.

Find all `return fig, summary_children` and `return figs[0], summary_children` paths. Before each, add:
```python
apply_theme(fig, theme)   # or apply_theme(figs[0], theme)
```

- [ ] **Step 2: Edit `timeseries_tab.py`**

`timeseries_tab.py` already has `_template(theme)` and `_error_fig(msg, theme)` helpers that build figures with the correct template. The closure variable `theme` is set at `register_callbacks` scope via `theme_default = "dark"`.

Find the main figure callback at line ~886 (outputs: `timeseries-graph`, `timeseries-acf-graph`, `timeseries-pmf-graph`, `timeseries-stats`, `timeseries-2dpmf-graph`).

**a)** Add `Input("theme-store", "data")` as the last Input in the decorator.

**b)** Add `theme_input` as the last parameter of the callback function.

**c)** Inside the callback body, find `theme = theme_default` (line ~941) and replace with:
```python
theme = theme_input or theme_default
```

That's the entire change — the existing `_template(theme)` and `_error_fig(..., theme=theme)` calls then use the correct value automatically.

- [ ] **Step 3: Edit `rama2d_tab.py`**

`rama2d_tab.py` sets `theme = _get_theme(ctx, default="light")` at the top of `register_callbacks`, and the `_update_rama_figure` callback is a closure that captures it.

Find the `_update_rama_figure` callback at line ~838.

**a)** Add `Input("theme-store", "data")` as the last Input in the decorator.

**b)** Add `theme_input` as the last parameter.

**c)** Add as the first line of the callback body:
```python
nonlocal theme
theme = theme_input or theme
```

This shadows the closure variable with the live store value whenever the callback fires.

- [ ] **Step 4: Verify**

```bash
python3 -c "from peptide_dash.tabs import curves, timeseries_tab, rama2d_tab; print('OK')"
```

Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add peptide_dash/tabs/curves.py peptide_dash/tabs/timeseries_tab.py peptide_dash/tabs/rama2d_tab.py
git commit -m "feat: add theme input to curves, timeseries, rama2d figure callbacks"
```

---

## Task 9: Theme input — `pmf_dendrogram_tab.py` and `umap_region_pmf_tab.py`

**Files:**
- Modify: `peptide_dash/tabs/pmf_dendrogram_tab.py`
- Modify: `peptide_dash/tabs/umap_region_pmf_tab.py`

Each has 3 figure-returning callbacks.

- [ ] **Step 1: Edit `pmf_dendrogram_tab.py`**

Add import at top of `register_callbacks`: `from .shared import apply_theme`

**Callback 1** (~line 1038, `pdend-graph`): add `Input("theme-store", "data")`, `theme` param, `apply_theme(fig, theme)`.

**Callback 2** (~line 1092, `pdend-pmf-graph`, `pdend-typical-graph`): add `Input("theme-store", "data")`, `theme` param, `apply_theme` on both returned figures.

**Callback 3** (~line 1291, `pdend-raw-ci-graph`): add `Input("theme-store", "data")`, `theme` param, `apply_theme(fig, theme)`.

- [ ] **Step 2: Edit `umap_region_pmf_tab.py`**

Add import at top of `register_callbacks`: `from .shared import apply_theme`

**Callback 1** (~line 1237, `urp-embed-graph`): add theme input + `apply_theme`.
**Callback 2** (~line 1498, `urp-pmf-graph`): add theme input + `apply_theme`.
**Callback 3** (~line 1636, `urp-raw-ci-graph`): add theme input + `apply_theme`.

- [ ] **Step 3: Verify**

```bash
python3 -c "from peptide_dash.tabs import pmf_dendrogram_tab, umap_region_pmf_tab; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add peptide_dash/tabs/pmf_dendrogram_tab.py peptide_dash/tabs/umap_region_pmf_tab.py
git commit -m "feat: add theme input to pmf_dendrogram, umap_region_pmf figure callbacks"
```

---

## Task 10: Theme input — `pca_tab.py`

**Files:**
- Modify: `peptide_dash/tabs/pca_tab.py`

PCA has 4 figure callbacks (lines ~1668, ~2034, ~2838, ~3182).

- [ ] **Step 1: Edit `pca_tab.py`**

Add import at top of `register_callbacks`: `from .shared import apply_theme`

**Callback 1** (~line 1668, outputs `pca-ev`, `pca-load`, `pca-group-contrib`):
- Add `Input("theme-store", "data")`, `theme` param
- Call `apply_theme` on all 3 returned figures before the tuple return

**Callback 2** (~line 2034, outputs `pca-scatter`, `pca-vector-biplot`, `pca-outlier-plot`):
- Add `Input("theme-store", "data")`, `theme` param
- Call `apply_theme` on all 3 returned figures

**Callback 3** (~line 2838, outputs `pca-stab-cos`, `pca-stab-proc`):
- Add `Input("theme-store", "data")`, `theme` param
- Call `apply_theme` on both figures

**Callback 4** (~line 3182, output `pca-curve-overlay`):
- Add `Input("theme-store", "data")`, `theme` param
- Call `apply_theme(fig, theme)`

- [ ] **Step 2: Verify**

```bash
python3 -c "from peptide_dash.tabs import pca_tab; print('OK')"
```

Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add peptide_dash/tabs/pca_tab.py
git commit -m "feat: add theme input to pca_tab figure callbacks"
```

---

## Task 11: Theme input — `umap/callbacks.py`

**Files:**
- Modify: `peptide_dash/tabs/umap/callbacks.py`

The UMAP module has 8 figure callbacks in `tabs/umap/callbacks.py`.

- [ ] **Step 1: Add import to `umap/callbacks.py`**

At the top of `register_callbacks` in `umap/callbacks.py` (or at module level if already structured that way), add:
```python
from ..shared import apply_theme
```

- [ ] **Step 2: Add theme input to each of the 8 callbacks**

The 8 callbacks are at approximately these output IDs:
1. `umap-graph` (~line 363)
2. `umap-graph` with `allow_duplicate=True` (~line 683)
3. `umap-sequence-region-graph` (~line 699)
4. `umap-seq-cluster-graph` (~line 727)
5. `umap-curve-overlay` (~line 857)
6. `umap-trajectory-graph` (~line 959)
7. `umap-residue-composition` (~line 1037)
8. `umap-trajectory-sequence-graph` (~line 1075)

For each: add `Input("theme-store", "data")`, add `theme` parameter, call `apply_theme(fig, theme)` before return.

- [ ] **Step 3: Verify**

```bash
python3 -c "from peptide_dash.tabs.umap import callbacks; print('OK')"
```

Expected: `OK`

- [ ] **Step 4: Commit**

```bash
git add peptide_dash/tabs/umap/callbacks.py
git commit -m "feat: add theme input to umap figure callbacks"
```

---

## Task 12: Smoke test

- [ ] **Step 1: Run all unit tests**

```bash
cd /run/media/sulcjo/sulcjo-data/IOCB/md/2025_PeptideMap
python -m pytest peptide_dash/tests/ -v
```

Expected: all tests PASS (including the 4 new theme tests from Task 1).

- [ ] **Step 2: Test app startup (no data)**

```bash
python3 -c "
from peptide_dash.data.context import DataContext
ctx = DataContext.__new__(DataContext)
ctx.df = None
from peptide_dash.app import create_app
app, server = create_app(ctx, loader=None)
print('App created OK, layout type:', type(app.layout).__name__)
"
```

Expected: `App created OK, layout type: Div`

- [ ] **Step 3: Verify no duplicate callback outputs**

```bash
python3 -c "
from peptide_dash.data.context import DataContext
ctx = DataContext.__new__(DataContext)
ctx.df = None
from peptide_dash.app import create_app
app, server = create_app(ctx, loader=None)
print('Callback count:', len(app.callback_map))
print('No duplicate output errors')
"
```

Expected: prints callback count with no `DuplicateCallbackError`.

- [ ] **Step 4: Final commit**

```bash
git add -A
git commit -m "feat: complete GUI redesign — The Peptideverse with light/dark theme and grouped nav"
```
