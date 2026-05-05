# PeptideMap GUI Redesign — Design Spec
**Date:** 2026-05-05  
**Status:** Approved

---

## Overview

Modernise the PeptideMap Dash dashboard (The Peptideverse) with a light/dark theme toggle, grouped tab navigation that maximises plot space, and Plotly charts that adapt to the active theme.

---

## Design Decisions

| Decision | Choice |
|---|---|
| Theme | Light/dark toggle, default light, persisted via localStorage |
| Tab navigation | Group selector (row 1) + filtered tab pills (row 2) |
| Plot space | Header ≈ 40px, nav ≈ 76px total; zero wasted vertical space |
| Plotly theming | Charts adapt to active theme via shared helper |

---

## Tab Groupings

| Group | Tabs |
|---|---|
| Features | Features, Stats |
| Dim. Reduction | PCA, Diffusion Map, UMAP, UMAP → PMF, Basin Landscape |
| Thermodynamics | PMF Dendrogram, Curves, Timeseries, Rama 2D |
| Convergence | Convergence, Diagnosis, Correlation |
| Info | README |

---

## Architecture

### Layout structure (`app.py`)

- Wrap entire app in `html.Div(id="app-shell", className="theme-light")` 
- Add `_header()` component: sticky 40px bar with title + theme toggle button
- Add `_grouped_nav(ctx)` component: two-row nav (group selector + tab pills)
- Keep existing `dcc.Tabs` but set `style={"display": "none"}` — it still drives content rendering
- Add stores:
  - `dcc.Store(id="theme-store", data="light", storage_type="local")`
  - `dcc.Store(id="active-group-store", data="features")`
  - `dcc.Store(id="active-tab-store", data="features")`

### Header component

```
[ The Peptideverse ]                              [ ☀/🌙 ]
```

- Title: "The Peptideverse", 13px bold, `#0b1220` / `#e5e7eb`
- Toggle button: shows ☀ in light mode, 🌙 in dark mode
- 40px height, `position: sticky; top: 0; z-index: 1000`
- Background: `var(--surface)` with `backdrop-filter: blur(10px)`

### Grouped nav component

**Row 1 — Group selector (38px):**
Five rectangular buttons: Features / Dim. Reduction / Thermodynamics / Convergence / Info  
Active group: accent background (`var(--accent)`, white text). Inactive: ghost style.

**Row 2 — Tab pills (38px):**
Shows only tabs in the active group. Active tab: selected pill style. Inactive: ghost pill.

Both rows live in a `<div class="nav-group-bar">` and `<div class="nav-tab-bar">` respectively.

### Clientside callbacks (no server round-trips)

1. **Theme toggle (clientside):** `Input("theme-toggle-btn", "n_clicks")` + `State("theme-store", "data")` → `Output("theme-store", "data")`. A second clientside callback on `Input("theme-store", "data")` sets `document.getElementById("app-shell").className` — fires on every page load too, picking up the persisted localStorage value.

2. **Group selector (server):** `Input({"type": "group-btn", "index": ALL}, "n_clicks")` → `Output("active-group-store", "data")` — records active group name.

3. **Tab pills (server):** `Input({"type": "tab-btn", "index": ALL}, "n_clicks")` → `Output("active-tab-store", "data")` + `Output("tabs", "value")` — syncs hidden `dcc.Tabs`.

4. **Nav re-render (server):** `Input("active-group-store", "data")` + `Input("active-tab-store", "data")` → `Output("grouped-nav", "children")` — returns updated group buttons + tab pills with correct active classes. `"grouped-nav"` is a placeholder `html.Div` in the layout.

Note: pattern-matching IDs (`{"type": "group-btn", "index": ...}`) must not collide with any existing callback IDs — a quick grep of the codebase confirms no existing use of these type keys.

---

## Plotly Theme Integration

### Shared helper (`tabs/shared.py`)

```python
def apply_theme(fig: go.Figure, theme: str) -> go.Figure:
    template = "plotly_dark" if theme == "dark" else "plotly_white"
    fig.update_layout(
        template=template,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig
```

### Callback changes

Every **registered callback** (i.e. decorated with `@app.callback`) whose `Output` is a Plotly figure gains:
- `Input("theme-store", "data")` as an additional input parameter `theme`
- `apply_theme(fig, theme)` call before returning

Helper functions that build figures (e.g. `pmf_plot_ci.pmf_overlay_fig`) are **not** modified — the callback that calls them applies `apply_theme` to the returned figure.

Affected files (callbacks with figure Outputs):
- `tabs/features.py`
- `tabs/diagnostics_tab.py`
- `tabs/correlation.py`
- `tabs/pca_tab.py`
- `tabs/diffusion_map_tab.py`
- `tabs/umap_tab.py`
- `tabs/umap_region_pmf_tab.py`
- `tabs/pmf_dendrogram_tab.py` (calls `pmf_plot_ci` helpers)
- `tabs/curves.py`
- `tabs/timeseries_tab.py`
- `tabs/convergence.py`
- `tabs/rama2d_tab.py`
- `tabs/stats_tab.py`
- `tabs/basin_tab.py`

---

## CSS Changes (`assets/theme.css`)

Additions only — no existing rules removed:

```css
/* App shell */
#app-shell {
  display: flex;
  flex-direction: column;
  min-height: 100vh;
}

/* Header */
.app-header {
  height: 40px;
  position: sticky;
  top: 0;
  z-index: 1000;
  background: var(--surface);
  backdrop-filter: blur(10px);
  border-bottom: var(--border-w) solid var(--border);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 14px;
}

/* Group selector row */
.nav-group-bar {
  display: flex;
  align-items: center;
  gap: 4px;
  height: 38px;
  padding: 0 14px;
  background: var(--bg);
  border-bottom: var(--border-w) solid var(--border);
}

.nav-group-btn {
  border: var(--border-w) solid var(--border);
  background: var(--surface);
  color: var(--muted);
  border-radius: 6px;
  padding: 3px 10px;
  font-size: 10px;
  font-weight: 700;
  cursor: pointer;
}

.nav-group-btn-active {
  background: var(--accent);
  color: #fff;
  border-color: var(--accent);
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
  overflow-x: auto;
  scrollbar-width: none;
}

/* Remove top margin from tab body */
.tab-body { margin-top: 0; }
```

---

## Files Modified

| File | Change |
|---|---|
| `peptide_dash/app.py` | Add shell, header, nav, stores, clientside callbacks |
| `peptide_dash/tabs/__init__.py` | Hide dcc.Tabs; keep callback registration unchanged |
| `peptide_dash/tabs/shared.py` | Add `apply_theme()` helper |
| `peptide_dash/assets/theme.css` | Add nav/header CSS rules |
| `peptide_dash/tabs/*.py` (14 files) | Add theme Input + `apply_theme()` to figure callbacks |

---

## Out of Scope

- Changes to any callback logic, data loading, or analysis code
- Plotly colorscale changes (only template/background, not trace colors)
- Mobile-specific layout changes
- Any new analysis features
