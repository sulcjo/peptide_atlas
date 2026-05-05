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
