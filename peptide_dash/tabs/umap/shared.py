from __future__ import annotations

from typing import Any, List, Tuple, Optional, Dict, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import re
import os
import json
import hashlib
import time

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import dash
from dash import html, dcc
from dash.dependencies import Input, Output, State
from dash.exceptions import PreventUpdate
import dash_bootstrap_components as dbc

from ...metrics import prettify_column_label
from ...data.context import filter_numeric_columns, is_excluded_feature_column
from ...data.io import _resolve_layout
from ...data.pmf_input import PMF_CORE_FEATURES, PMF_PHYSICAL_ANNOTATIONS, PMF_TRANSFORM_RULES, transform_features
from ...analysis.pmf_vectorize import build_pmf_design_matrix, parse_family as _pmf_parse_family
from ...data.variant_pmf import PMF_PLOT_COLS, available_pmf_metrics, available_pmf_variants, load_variant_pmfs

try:
    import umap
    HAVE_UMAP = True
except Exception:
    umap = None
    HAVE_UMAP = False

try:
    from sklearn.cluster import DBSCAN
    from sklearn.metrics import silhouette_score
    HAVE_SKLEARN = True
except Exception:
    DBSCAN = None
    silhouette_score = None
    HAVE_SKLEARN = False

try:
    from sklearn.manifold import Isomap
    HAVE_ISOMAP = True
except Exception:
    Isomap = None
    HAVE_ISOMAP = False

from ..shared import R_GAS

UMAP_SOURCE_BASIC = "basic"
UMAP_SOURCE_THERMO = "thermodynamic"
UMAP_SOURCE_PMF = "pmf_shape"
UMAP_SOURCE_SEQUENCE = "sequence"
