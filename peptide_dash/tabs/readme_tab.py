from __future__ import annotations

from pathlib import Path
from dash import dcc, html

TAB_LABEL = "README"


def _repo_root() -> Path:
    """Best-effort repo root resolver: tabs/ -> repo root."""
    return Path(__file__).resolve().parents[1]


def _load_readme_text() -> str:
    """
    Prefer a repo-provided markdown file. Falls back to a small inline blurb.
    """
    candidates = [
        _repo_root() / "README_modularization.md",
        _repo_root() / "README.md",
    ]
    for p in candidates:
        try:
            if p.is_file():
                return p.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue

    return (
        "# Peptide Dash\n\n"
        "This is the modular Dash app.\n\n"
        "Legacy (monolith) modules are not loaded at runtime.\n"
    )


def layout(ctx) -> html.Div:
    data_dir = getattr(ctx, "data_dir", None)
    timeseries_dir = getattr(ctx, "timeseries_dir", None)

    hdr = html.Div(
        [
            html.H2("Peptide Dash"),
            html.Div(
                [
                    html.Div(["data_dir: ", html.Code(str(data_dir))]) if data_dir else None,
                    html.Div(["timeseries_dir: ", html.Code(str(timeseries_dir))]) if timeseries_dir else None,
                ]
            ),
            html.Hr(),
        ]
    )

    return html.Div(
        [
            hdr,
            dcc.Markdown(_load_readme_text(), link_target="_blank"),
        ],
        className="tab-body-inner",
    )


def register_callbacks(app, ctx) -> None:
    return
