from __future__ import annotations

import pandas as pd


def pmf_matrix_for_umap(df: pd.DataFrame, cols, **kw):
    raise NotImplementedError(
        "Legacy UMAP helpers were removed. Use tabs/umap_tab.py (modular implementation)."
    )


def pmf_matrix_by_replica_for_umap(df: pd.DataFrame, cols, **kw):
    raise NotImplementedError(
        "Legacy UMAP helpers were removed. Use tabs/umap_tab.py (modular implementation)."
    )


def build_family_balanced_design_matrix(df: pd.DataFrame, cols, **kw):
    raise NotImplementedError("Legacy family-balanced helpers were removed.")


def project_replicas_family_balanced(df: pd.DataFrame, cols, **kw):
    raise NotImplementedError("Legacy family-balanced helpers were removed.")
