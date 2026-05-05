import plotly.graph_objs as go
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from peptide_dash.tabs.shared import apply_theme


def test_apply_theme_dark_sets_bgcolor():
    fig = go.Figure()
    result = apply_theme(fig, "dark")
    assert result.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert result.layout.plot_bgcolor == "rgba(0,0,0,0)"


def test_apply_theme_light_sets_bgcolor():
    fig = go.Figure()
    result = apply_theme(fig, "light")
    assert result.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert result.layout.plot_bgcolor == "rgba(0,0,0,0)"


def test_apply_theme_returns_same_figure():
    fig = go.Figure()
    result = apply_theme(fig, "light")
    assert result is fig


def test_apply_theme_none_defaults_to_light():
    fig = go.Figure()
    result = apply_theme(fig, None)
    assert result.layout.paper_bgcolor == "rgba(0,0,0,0)"
    assert isinstance(result, go.Figure)


def test_apply_theme_dark_vs_light_differ():
    fig_dark = go.Figure()
    fig_light = go.Figure()
    apply_theme(fig_dark, "dark")
    apply_theme(fig_light, "light")
    assert fig_dark.layout.template != fig_light.layout.template
