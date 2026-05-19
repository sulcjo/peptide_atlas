#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compatibility wrapper for the modular UMAP tab implementation."""

from .umap.embedding import _compute_umap_embedding, _umap_defaults
from .umap.figures import _build_umap_figure
from .umap.input_matrix import _prepare_feature_design_matrix, _sequence_umap_feature_cols, _umap_source_mode
from .umap.shared import UMAP_SOURCE_BASIC, UMAP_SOURCE_PMF, UMAP_SOURCE_SEQUENCE, UMAP_SOURCE_THERMO
from .umap.callbacks import layout, register_callbacks

__all__ = [
    "UMAP_SOURCE_BASIC",
    "UMAP_SOURCE_THERMO",
    "UMAP_SOURCE_PMF",
    "UMAP_SOURCE_SEQUENCE",
    "_build_umap_figure",
    "_compute_umap_embedding",
    "_prepare_feature_design_matrix",
    "_sequence_umap_feature_cols",
    "_umap_defaults",
    "_umap_source_mode",
    "layout",
    "register_callbacks",
]
