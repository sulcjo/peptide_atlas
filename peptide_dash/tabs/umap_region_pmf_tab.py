"""
peptide_dash/tabs/umap_region_pmf_tab.py

UMAP → Axis-conditioned PMF tab (Region-conditioned ensemble curves)

Core:
    P_m(x | R) = Σ_{v ∈ R} w_v * P_m(x | v)

If only free energies F are available:
    P ∝ exp(-F / kT)
Optionally convert back to free energy:
    F(x|R) = -kT ln P(x|R), shifted so min(F)=0

Region R:
- Box/Lasso selection on the embedding scatter, OR
- Axis range filters

Data expectations (matches the existing app):
- ctx.features: feature table with column "variant" (required for feature embedding / weights)
- ctx.pmf_df: PMF table with columns:
    variant, metric, x, and either P or F_kJ_mol

Embedding methodology (kept consistent with umap_tab.py):
- PMF design matrix:
    - rounds x bins to collapse float jitter,
    - interpolates each variant PMF onto a per-metric common x-grid,
    - optionally caps bins per metric (prevents "tiny dx" exploding feature count),
    - normalizes each variant distribution by ∫ p(x) dx (dx estimated from the grid),
    - concatenates metrics and (by default) intersects variants across metrics.
- UMAP preprocessing:
    1) non-finite → 0
    2) optional Hellinger transform for PMFs: sqrt(P) + euclidean
    3) z-score (ddof=1)
    4) PMF family balancing: scale columns by 1/sqrt(n_bins_per_metric) so "more bins" doesn't dominate
    5) SVD-PCA (~99% EVR, cap 64), then UMAP/DensMAP

Notes:
- Region selection only affects conditional PMF outputs; embedding is global for selected data source.
"""
from __future__ import annotations

from ..metrics import metric_display_label, torsion_sort_key

from ..analysis.pmf_vectorize import build_pmf_design_matrix
from ..data import io as data_io
from typing import Any, Dict, Optional, Sequence, Tuple, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html, dash_table, no_update

from .pmf_plot_ci import (
    pmf_overlay_fig as _shared_pmf_overlay_fig,
    per_variant_raw_pmf_with_ci as _shared_per_variant_raw_pmf_with_ci,
    pmf_has_ci_columns as _shared_pmf_has_ci_columns,
    pmf_error_fig as _error_fig,
)
from .shared import (
    PmfCols as PmfColumns, pmf_panel as _panel, R_GAS,
    kT as _kT, bin_dx as _bin_dx, normalize_density as _normalize_prob,
    infer_features_df as _infer_features_df, infer_weight_columns as _infer_weight_columns,
    metric_options,
)

try:
    import umap  # type: ignore

    HAVE_UMAP = True
except Exception:
    umap = None
    HAVE_UMAP = False

TAB_LABEL = "UMAP → PMF (Region)"
TAB_VALUE = "umap_region_pmf"

# ----------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------

def _live_pmf_metrics(pmf_df: pd.DataFrame, cols: PmfColumns) -> list[str]:
    if not isinstance(pmf_df, pd.DataFrame) or pmf_df.empty or cols.metric not in pmf_df.columns:
        return []
    vals = pmf_df[cols.metric].astype(str).dropna().unique().tolist()
    return sorted(vals, key=torsion_sort_key)

def _pick_metric_defaults(metrics: Sequence[str], current: Optional[Sequence[str]], n_default: int) -> list[str]:
    avail = [str(m) for m in metrics]
    current_list = [str(m) for m in (current or []) if str(m) in avail]
    return current_list if current_list else avail[: max(0, int(n_default))]

def _is_nonempty_df(df: Any) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty

def _prime_lazy_curve_cache(ctx: Any, **frames: pd.DataFrame) -> None:
    lazy = getattr(ctx, "_lazy_curves", None)
    if lazy is None or not hasattr(lazy, "_cache"):
        try:
            lazy = data_io.load_curves_tables_lazy(getattr(ctx, "data_dir", "."))
            setattr(ctx, "_lazy_curves", lazy)
        except Exception:
            return
    cache = getattr(lazy, "_cache", None)
    if not isinstance(cache, dict):
        return
    for key, df in frames.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            cache[key] = df

def _load_curve_df_recursive(data_dir: str, key: str) -> pd.DataFrame:
    try:
        base_dir, _ = data_io._resolve_layout(data_dir)
    except Exception:
        return pd.DataFrame()
    patterns = list(getattr(data_io.LazyCurvesLoader, "_FILE_PATTERNS", {}).get(key, []))
    if not patterns:
        return pd.DataFrame()

    seen: set[str] = set()
    frames: list[pd.DataFrame] = []
    for pat in patterns:
        for p in Path(base_dir).rglob(pat):
            rp = str(p.resolve())
            if rp in seen:
                continue
            seen.add(rp)
            try:
                df = data_io._read_any_df(Path(rp))
            except Exception:
                continue
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df)
    if not frames:
        return pd.DataFrame()
    try:
        df = data_io._concat_nonempty(frames)
    except Exception:
        df = pd.concat(frames, ignore_index=True, sort=False)
    try:
        df = data_io._drop_all_na_cols(df)
    except Exception:
        pass
    try:
        df = data_io._optimize_curves_df(df)
    except Exception:
        pass
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()

def _get_live_pmf_df(ctx: Any, *, force_refresh: bool = False) -> pd.DataFrame:
    if not force_refresh:
        try:
            df = getattr(ctx, "pmf_df")
            if _is_nonempty_df(df):
                return df
        except Exception:
            pass

    lazy = getattr(ctx, "_lazy_curves", None)
    if lazy is not None and hasattr(lazy, "_cache"):
        try:
            getattr(lazy, "_cache").pop("pmf", None)
        except Exception:
            pass

    try:
        if lazy is None:
            lazy = data_io.load_curves_tables_lazy(getattr(ctx, "data_dir", "."))
            setattr(ctx, "_lazy_curves", lazy)
        df = lazy.pmf_df
        if _is_nonempty_df(df):
            return df
    except Exception:
        pass

    try:
        loaded = data_io.load_curves_tables(getattr(ctx, "data_dir", "."))
        pmf_df = loaded[0] if len(loaded) > 0 and isinstance(loaded[0], pd.DataFrame) else pd.DataFrame()
        if _is_nonempty_df(pmf_df):
            _prime_lazy_curve_cache(
                ctx,
                pmf=pmf_df,
                cum=loaded[1] if len(loaded) > 1 else pd.DataFrame(),
                rmsf=loaded[2] if len(loaded) > 2 else pd.DataFrame(),
                rama2d_pooled=loaded[3] if len(loaded) > 3 else pd.DataFrame(),
                rama2d_perres=loaded[4] if len(loaded) > 4 else pd.DataFrame(),
                conv=loaded[5] if len(loaded) > 5 else pd.DataFrame(),
                pmf_replica=loaded[6] if len(loaded) > 6 else pd.DataFrame(),
                cum_replica=loaded[7] if len(loaded) > 7 else pd.DataFrame(),
                conv_replica=loaded[8] if len(loaded) > 8 else pd.DataFrame(),
            )
            return pmf_df
    except Exception:
        pass

    try:
        pmf_df = _load_curve_df_recursive(getattr(ctx, "data_dir", "."), "pmf")
        if _is_nonempty_df(pmf_df):
            _prime_lazy_curve_cache(ctx, pmf=pmf_df)
            return pmf_df
    except Exception:
        pass

    return pd.DataFrame()

def _get_live_features_df(ctx: Any) -> pd.DataFrame:
    return _infer_features_df(ctx)

def _prepare_feature_embedding_matrix(features_df: pd.DataFrame, requested_cols: Sequence[str]) -> Tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Robust feature matrix builder for sparse numeric feature tables."""
    if not isinstance(features_df, pd.DataFrame) or features_df.empty or "variant" not in features_df.columns:
        return np.zeros((0, 0), dtype=float), pd.DataFrame(columns=["variant"]), []

    feats = [str(c) for c in (requested_cols or []) if str(c) in features_df.columns and str(c) != "variant"]
    if not feats:
        return np.zeros((0, 0), dtype=float), pd.DataFrame(columns=["variant"]), []

    df_num = features_df[["variant"] + feats].copy()
    for c in feats:
        df_num[c] = pd.to_numeric(df_num[c], errors="coerce")

    valid_cols: list[str] = []
    for c in feats:
        s = df_num[c]
        finite = np.isfinite(s.to_numpy(dtype=float, na_value=np.nan))
        if int(finite.sum()) < 2:
            continue
        vals = s[finite].to_numpy(dtype=float)
        if vals.size < 2:
            continue
        if np.nanmax(vals) == np.nanmin(vals):
            continue
        valid_cols.append(c)

    if not valid_cols:
        return np.zeros((0, 0), dtype=float), pd.DataFrame(columns=["variant"]), []

    df_num = df_num[["variant"] + valid_cols].copy()
    X = df_num[valid_cols].to_numpy(dtype=float)
    row_keep = np.isfinite(X).any(axis=1)
    if not np.any(row_keep):
        return np.zeros((0, len(valid_cols)), dtype=float), pd.DataFrame(columns=["variant"]), valid_cols

    df_num = df_num.loc[row_keep].reset_index(drop=True)
    X = df_num[valid_cols].to_numpy(dtype=float)

    med = np.nanmedian(X, axis=0)
    med = np.where(np.isfinite(med), med, 0.0)
    inds = np.where(~np.isfinite(X))
    if inds[0].size:
        X[inds] = med[inds[1]]

    return X, df_num[["variant"]].copy(), valid_cols

def _prob_to_free_energy(p: np.ndarray, kt: float) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, 1e-300, None)
    F = -float(kt) * np.log(p)
    F = F - np.nanmin(F)
    return F

def _parse_family(colname: str) -> str:
    """e.g. 'metric|x=0.1' -> 'metric'."""
    if "|" in colname:
        return colname.split("|", 1)[0]
    return colname

def _existing_embedding_from_features(features_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Accept common naming patterns.
    Requires 'variant' column.
    """
    if not isinstance(features_df, pd.DataFrame) or features_df.empty:
        return None
    if "variant" not in features_df.columns:
        return None

    cols = {str(c).lower(): str(c) for c in features_df.columns}
    u_cands = ["umap_1", "umap1", "u", "embedding_1", "x_umap", "umapx", "umap_x", "x"]
    v_cands = ["umap_2", "umap2", "v", "embedding_2", "y_umap", "umapy", "umap_y", "y"]

    ucol = next((cols[c] for c in u_cands if c in cols), None)
    vcol = next((cols[c] for c in v_cands if c in cols), None)
    if not ucol or not vcol:
        return None

    emb = features_df[["variant", ucol, vcol]].copy()
    emb.columns = ["variant", "u", "v"]
    emb["variant"] = emb["variant"].astype(str)
    emb["u"] = pd.to_numeric(emb["u"], errors="coerce")
    emb["v"] = pd.to_numeric(emb["v"], errors="coerce")
    emb = emb.dropna(subset=["u", "v"])
    return None if emb.empty else emb

# ----------------------------------------------------------------------
# PMF matrix builder (consistent with umap_tab.py)
# ----------------------------------------------------------------------

def _pmf_matrix_for_umap(
    pmf_df: pd.DataFrame,
    cols: PmfColumns,
    metrics: List[str],
    use_repr: str,
    energy_units: str,
    T_K: float,
    *,
    x_round_decimals: int = 4,
    max_bins_per_metric: int = 256,
    normalize_by_integral: bool = True,
    variant_set_mode: str = "intersection",
) -> Tuple[np.ndarray, List[str], pd.DataFrame]:
    """
    Shared PMF design matrix builder (kept identical to umap_tab.py).

    Notes:
    - We always normalize by ∫ p(x) dx (probability mass on grid). The `normalize_by_integral`
      argument is kept for API compatibility.
    - Variant policies:
        * "intersection" (default)
        * "union" (fills missing with zeros)
        * "union_mean" / "union-mean" (fills missing with per-metric mean distribution)
    """
    mode = str(variant_set_mode or "intersection").lower().strip()
    if mode in {"union_mean", "union-mean", "unionmean"}:
        policy = "union_mean"
        impute = "mean"
    elif mode.startswith("uni"):
        policy = "union"
        impute = "zeros"
    else:
        policy = "intersection"
        impute = "zeros"

    X, colnames, meta, _grids = build_pmf_design_matrix(
        pmf_df=pmf_df,
        metrics=metrics,
        cols=cols,  # compatible dataclass fields
        use_repr=use_repr,
        energy_units=energy_units,
        T_K=float(T_K),
        max_bins_per_metric=int(max_bins_per_metric),
        round_decimals=int(x_round_decimals) if x_round_decimals is not None else None,
        variant_policy=policy,
        missing_impute=impute,
        dirichlet_alpha=0.0,
    )
    return X, colnames, meta

# ----------------------------------------------------------------------
# Embedding computation (UMAP / DensMAP)
# ----------------------------------------------------------------------

def _compute_umap_embedding(
    X_raw: np.ndarray,
    *,
    colnames_raw: Optional[List[str]],
    matrix_type: str,
    n_neighbors: int,
    min_dist: float,
    metric: str,
    seed: int,
    embedding_method: str = "umap",
    dens_lambda: Optional[float] = None,
    dens_frac: Optional[float] = None,
    dens_var_shift: Optional[float] = None,
) -> np.ndarray:
    """
    Compute a 2D embedding using the same preprocessing + UMAP/DensMAP pattern as `umap_tab.py`.
    """
    if not HAVE_UMAP:
        raise RuntimeError("UMAP is not installed (pip install umap-learn).")

    X_raw = np.asarray(X_raw, dtype=float)
    if X_raw.size == 0:
        return np.zeros((0, 2))

    X_raw = np.nan_to_num(X_raw, nan=0.0, posinf=0.0, neginf=0.0)

    matrix_type = (matrix_type or "features").lower().strip()
    is_pmf_matrix = matrix_type.startswith("pmf")
    hellinger_mode = False

    # Hellinger mode preserves distributional geometry end-to-end. See the
    # detailed rationale block in umap_tab.py::_compute_embedding. In short:
    # sqrt(P) transforms each PMF to the unit Hellinger sphere; centering is
    # fine (rigid translation preserves Euclidean distances); per-column
    # std-scaling is SKIPPED because it would destroy the Hellinger metric;
    # family balancing is kept; PCA on centered sqrt(P) is a legitimate
    # distance-preserving-up-to-truncation projection; UMAP with
    # metric='euclidean' on sqrt(P)-space is equivalent to Hellinger on P.
    metric = (metric or "cosine").lower().strip()
    if metric == "hellinger":
        if is_pmf_matrix:
            X_raw = np.sqrt(np.clip(X_raw, 0.0, None))
            # Drop near-empty histogram tails (contribute only noise under
            # any scaling and hurt k-NN stability).
            min_support = max(3, int(0.10 * X_raw.shape[0]))
            support = np.sum(X_raw > 1e-6, axis=0)
            keep_cols = support >= min_support
            if (not keep_cols.all()) and keep_cols.any():
                X_raw = X_raw[:, keep_cols]
                if colnames_raw is not None:
                    colnames_raw = [c for c, k in zip(colnames_raw, keep_cols.tolist()) if k]
            metric = "euclidean"
            hellinger_mode = True
        else:
            metric = "euclidean"

    # Standardization: in Hellinger mode, center only (skip std-scaling).
    mu = np.mean(X_raw, axis=0)
    if hellinger_mode:
        sd = np.ones_like(mu)
    else:
        sd = np.std(X_raw, axis=0, ddof=1)
        sd = np.where(sd == 0, 1.0, sd)
    Xz = (X_raw - mu) / sd

    # PMF family balancing: equalize total variance per metric family so "more bins" doesn't dominate.
    if is_pmf_matrix and colnames_raw:
        fams = np.asarray([_parse_family(c) for c in colnames_raw], dtype=object)
        uniq, counts = np.unique(fams, return_counts=True)
        fam2cnt = dict(zip(uniq.tolist(), counts.tolist()))
        scales = np.asarray([1.0 / np.sqrt(float(fam2cnt[f])) for f in fams], dtype=float)
        Xz = Xz * scales

    # PCA via SVD to ~99% EVR (cap 64). In Hellinger mode this is PCA on
    # centered sqrt(P) and the Euclidean distances in PC space approach the
    # full Hellinger distance as k grows.
    X_pca = Xz
    try:
        _, s, Vt = np.linalg.svd(Xz, full_matrices=False)
        if s.size:
            evr = (s**2) / np.sum(s**2)
            cum = np.cumsum(evr)
            k = int(np.searchsorted(cum, 0.99) + 1)
            k = min(k, 64, Xz.shape[1])
        else:
            k = min(64, Xz.shape[1])

        if 1 < k < Xz.shape[1]:
            Vt_k = Vt[:k, :]
            X_pca = Xz @ Vt_k.T
    except Exception:
        X_pca = Xz

    n_samples = int(X_pca.shape[0])
    nn = int(max(2, n_neighbors))
    if n_samples > 2:
        nn = int(min(nn, n_samples - 1))

    method = (embedding_method or "umap").lower().strip()
    use_densmap = method in {"densmap", "dens-map", "dens_map", "dens"}

    umap_kwargs: Dict[str, Any] = dict(
        n_neighbors=nn,
        min_dist=float(min_dist),
        metric=str(metric),
        n_components=2,
        random_state=int(seed),
    )

    if use_densmap:
        dl = 2.0 if dens_lambda is None else float(dens_lambda)
        df_ = 0.30 if dens_frac is None else float(dens_frac)
        dv = 0.10 if dens_var_shift is None else float(dens_var_shift)
        df_ = float(np.clip(df_, 0.0, 1.0))
        dv = float(np.clip(dv, 0.0, 1.0))
        umap_kwargs.update(
            densmap=True,
            dens_lambda=dl,
            dens_frac=df_,
            dens_var_shift=dv,
            output_dens=False,
        )

    try:
        reducer = umap.UMAP(**umap_kwargs)
    except TypeError as e:
        if use_densmap and ("densmap" in str(e) or "dens_" in str(e)):
            raise RuntimeError(
                "DensMAP requested, but the installed umap-learn does not support densmap parameters. "
                "Upgrade umap-learn to a newer version that includes DensMAP."
            ) from e
        raise

    emb = reducer.fit_transform(X_pca)
    emb = np.asarray(emb, dtype=float)
    if emb.ndim != 2 or emb.shape[1] != 2:
        return np.zeros((X_pca.shape[0], 2))
    return emb

def _add_has_pmf_flag(emb: pd.DataFrame, pmf_variants: set[str]) -> pd.DataFrame:
    if (not isinstance(emb, pd.DataFrame)) or emb.empty or ("variant" not in emb.columns) or (not pmf_variants):
        return emb
    out = emb.copy()
    out["has_pmf"] = out["variant"].astype(str).isin(pmf_variants)
    return out

# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def _as_variant_from_point(p: Dict[str, Any]) -> Optional[str]:
    cd = p.get("customdata")
    if isinstance(cd, (list, tuple)) and cd:
        cd = cd[0]
    if cd is not None:
        return str(cd)
    for k in ("text", "hovertext", "label"):
        v = p.get(k)
        if v is not None:
            return str(v)
    return None

def _scatter_fig(emb: pd.DataFrame) -> go.Figure:
    if emb.empty:
        return _error_fig("No embedding points to display.")

    fig = go.Figure()

    if "has_pmf" in emb.columns:
        has = emb["has_pmf"].fillna(False).astype(bool)
        for name, sub, opacity in (
            ("PMF available", emb.loc[has], 0.75),
            ("No PMF", emb.loc[~has], 0.18),
        ):
            if sub.empty:
                continue
            fig.add_trace(
                go.Scatter(
                    x=sub["u"],
                    y=sub["v"],
                    mode="markers",
                    name=name,
                    marker=dict(size=7, opacity=opacity),
                    customdata=sub["variant"].astype(str),
                    hovertemplate="variant=%{customdata}<br>u=%{x:.3f}<br>v=%{y:.3f}<extra></extra>",
                )
            )
    else:
        fig.add_trace(
            go.Scatter(
                x=emb["u"],
                y=emb["v"],
                mode="markers",
                name="variants",
                marker=dict(size=7, opacity=0.7),
                customdata=emb["variant"].astype(str),
                hovertemplate="variant=%{customdata}<br>u=%{x:.3f}<br>v=%{y:.3f}<extra></extra>",
            )
        )

    fig.update_layout(
        template="plotly_white",
        height=520,
        margin=dict(l=40, r=20, t=40, b=40),
        title="Embedding (box/lasso select a region; click selects one)",
        dragmode="lasso",
        clickmode="event+select",
        legend=dict(orientation="h", x=0, y=-0.12, xanchor="left", yanchor="top"),
        uirevision="urp-embed",
    )
    fig.update_xaxes(title="UMAP-1")
    fig.update_yaxes(title="UMAP-2")
    return fig

# ----------------------------------------------------------------------
# Conditional PMF computation (unchanged)
# ----------------------------------------------------------------------

def _conditional_pmf(
    pmf_df: pd.DataFrame,
    cols: PmfColumns,
    variants: Sequence[str],
    metrics: Sequence[str],
    weights: Dict[str, float],
    use_repr: str,
    energy_units: str,
    T_K: float,
    kT_override: Optional[float],
    preserve_variant_norm: bool,
    output_kind: str,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    if not isinstance(pmf_df, pd.DataFrame) or pmf_df.empty or not variants or not metrics:
        return {}

    kt = _kT(energy_units, T_K, kT_override)
    df = pmf_df[
        pmf_df[cols.variant].astype(str).isin(list(map(str, variants)))
        & pmf_df[cols.metric].astype(str).isin(list(map(str, metrics)))
    ].copy()
    if df.empty:
        return {}

    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for m in metrics:
        sub = df[df[cols.metric].astype(str) == str(m)].copy()
        if sub.empty:
            continue

        sub[cols.variant] = sub[cols.variant].astype(str)
        sub[cols.x] = pd.to_numeric(sub[cols.x], errors="coerce")
        sub = sub.dropna(subset=[cols.x])

        use_P = (use_repr == "P") and (cols.p in sub.columns)

        if use_P:
            sub["p"] = pd.to_numeric(sub[cols.p], errors="coerce").clip(lower=0.0)
        else:
            if cols.f not in sub.columns:
                continue
            F = pd.to_numeric(sub[cols.f], errors="coerce").to_numpy(dtype=float)
            F = np.asarray(F, dtype=float)
            F = F - np.nanmin(F)
            p = np.exp(-F / max(kt, 1e-12))
            sub["p"] = p

        sub = sub.dropna(subset=["p"])
        if sub.empty:
            continue

        curves: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for v, g in sub.groupby(cols.variant):
            x = pd.to_numeric(g[cols.x], errors="coerce").to_numpy(dtype=float)
            p = pd.to_numeric(g["p"], errors="coerce").to_numpy(dtype=float)
            mask = np.isfinite(x) & np.isfinite(p)
            x, p = x[mask], p[mask]
            if x.size == 0:
                continue
            order = np.argsort(x)
            x, p = x[order], p[order]
            if preserve_variant_norm:
                p = _normalize_prob(x, p)
            curves[str(v)] = (x, p)

        if not curves:
            continue

        x_grid = np.unique(np.concatenate([curves[v][0] for v in curves], axis=0))
        if x_grid.size == 0:
            continue

        Pmix = np.zeros_like(x_grid, dtype=float)
        for v, (xv, pv) in curves.items():
            w = float(weights.get(str(v), 0.0))
            if w <= 0:
                continue
            p_i = np.interp(x_grid, xv, pv, left=0.0, right=0.0)
            Pmix += w * p_i

        if preserve_variant_norm:
            Pmix = _normalize_prob(x_grid, Pmix)

        if output_kind == "F":
            y = _prob_to_free_energy(Pmix, kt)
        else:
            y = Pmix

        out[str(m)] = (x_grid, y)

    return out

def _pmf_fig(curves: Dict[str, Tuple[np.ndarray, np.ndarray]], output_kind: str) -> go.Figure:
    if not curves:
        return _error_fig("No conditional PMF data (try selecting a region + metrics).")

    fig = go.Figure()
    for m, (x, y) in curves.items():
        fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=metric_display_label(m)))

    fig.update_layout(
        template="plotly_white",
        height=460,
        margin=dict(l=40, r=20, t=40, b=40),
        title="Conditional PMF over selected region",
        legend=dict(orientation="h", x=0, y=-0.18, xanchor="left", yanchor="top"),
    )
    fig.update_xaxes(title="Reaction coordinate")
    fig.update_yaxes(title="F (shifted to min=0)" if output_kind == "F" else "P (normalized)")
    return fig

# ----------------------------------------------------------------------
# Layout
# ----------------------------------------------------------------------

def layout(ctx: Any) -> html.Div:
    cols = PmfColumns()
    pmf_df = pd.DataFrame()  # not loaded at layout time
    features_df = _infer_features_df(ctx)

    metrics: list[str] = []
    if isinstance(pmf_df, pd.DataFrame) and (not pmf_df.empty) and (cols.metric in pmf_df.columns):
        metrics = sorted(pmf_df[cols.metric].astype(str).dropna().unique().tolist(), key=torsion_sort_key)

    feat_cols: list[str] = []
    if isinstance(features_df, pd.DataFrame) and (not features_df.empty):
        feat_cols = [c for c in features_df.select_dtypes(include="number").columns.astype(str).tolist() if c != "variant"]

    weight_cols = _infer_weight_columns(features_df)

    default_embed_source = "existing" if _existing_embedding_from_features(features_df) is not None else "pmf"
    if default_embed_source == "pmf" and not metrics and feat_cols:
        default_embed_source = "features"

    default_embed_metrics = metrics[:4]
    default_pmf_metrics = metrics[:2]

    embed_config = {"displaylogo": False, "modeBarButtonsToAdd": ["lasso2d", "select2d"]}

    return html.Div(
        className="tab-body-inner",
        children=[
            html.Div(
                style={"display": "flex", "gap": "14px", "flexWrap": "wrap"},
                children=[
                    html.Div(
                        style={"minWidth": "320px", "flex": "0 0 360px"},
                        children=[
                            _panel(
                                "Data status",
                                None,
                                [
                                    html.Div(
                                        f"features rows: {len(features_df) if isinstance(features_df, pd.DataFrame) else 0}",
                                        id="urp-status-features-rows",
                                    ),
                                    html.Div("pmf_df rows: loading on tab open…", id="urp-status-pmf-rows"),
                                    html.Div("pmf metrics: loading on tab open…", id="urp-status-pmf-metrics"),
                                    html.Div("pmf variants: loading on tab open…", id="urp-status-pmf-variants"),
                                ],
                            ),
                            _panel(
                                "Embedding",
                                "Compute / load embedding. PMF-mode uses bin-aligned, family-balanced vectors (consistent with umap_tab.py).",
                                [
                                    html.Label("Embedding source"),
                                    dcc.Dropdown(
                                        id="urp-embed-source",
                                        options=[
                                            {"label": "Use existing embedding from ctx.features", "value": "existing"},
                                            {"label": "Compute from PMF probability matrix", "value": "pmf"},
                                            {"label": "Compute from feature matrix (numeric)", "value": "features"},
                                        ],
                                        value=default_embed_source,
                                        clearable=False,
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Embedding PMF metrics (if computing from PMFs)"),
                                    dcc.Dropdown(
                                        id="urp-embed-metrics",
                                        options=[],
                                        value=[],
                                        multi=True,
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Embedding feature columns (if computing from features)"),
                                    dcc.Dropdown(
                                        id="urp-embed-features",
                                        options=[{"label": c, "value": c} for c in feat_cols],
                                        value=feat_cols[:12],
                                        multi=True,
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("PMF representation"),
                                    dcc.Dropdown(
                                        id="urp-pmf-repr",
                                        options=[
                                            {"label": "Use P (probability/density)", "value": "P"},
                                            {"label": "Use F (free energy) if needed", "value": "F"},
                                        ],
                                        value="P" if (isinstance(pmf_df, pd.DataFrame) and (cols.p in pmf_df.columns)) else "F",
                                        clearable=False,
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Energy units (only if using F)"),
                                    dcc.Dropdown(
                                        id="urp-energy-units",
                                        options=[
                                            {"label": "kJ/mol", "value": "kJ/mol"},
                                            {"label": "kT (dimensionless)", "value": "kT"},
                                        ],
                                        value="kJ/mol" if (isinstance(pmf_df, pd.DataFrame) and (cols.f in pmf_df.columns)) else "kT",
                                        clearable=False,
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Temperature (K) (only if kJ/mol)"),
                                    dcc.Input(id="urp-temp-k", type="number", value=300.0, min=1, step=1),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("PMF variant set across metrics"),
                                    dcc.Dropdown(
                                        id="urp-pmf-variant-mode",
                                        options=[
                                            {"label": "intersection (same as umap_tab.py)", "value": "intersection"},
                                            {"label": "union (fill missing bins with 0)", "value": "union"},
                                        ],
                                        value="intersection",
                                        clearable=False,
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("PMF x rounding (decimals)"),
                                    dcc.Input(id="urp-pmf-x-round", type="number", value=4, min=0, step=1),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("PMF max bins per metric (cap)"),
                                    dcc.Input(id="urp-pmf-max-bins", type="number", value=256, min=16, step=16),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Normalize PMF by integral ∫p(x)dx"),
                                    dcc.Dropdown(
                                        id="urp-pmf-integral-norm",
                                        options=[{"label": "yes", "value": "yes"}, {"label": "no", "value": "no"}],
                                        value="yes",
                                        clearable=False,
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Embedding method"),
                                    dcc.Dropdown(
                                        id="urp-embed-method",
                                        options=[
                                            {"label": "UMAP", "value": "umap"},
                                            {"label": "DensMAP (density-preserving)", "value": "densmap"},
                                        ],
                                        value="umap",
                                        clearable=False,
                                    ),
                                    html.Div(
                                        "DensMAP preserves local density; dens_* params below are only used when DensMAP is selected.",
                                        className="panel-subtitle",
                                        style={"marginTop": "6px"},
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("dens_lambda (DensMAP)"),
                                    dcc.Input(id="urp-dens-lambda", type="number", value=2.0, step=0.1),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("dens_frac (DensMAP)"),
                                    dcc.Input(id="urp-dens-frac", type="number", value=0.30, min=0.0, max=1.0, step=0.05),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("dens_var_shift (DensMAP)"),
                                    dcc.Input(id="urp-dens-var-shift", type="number", value=0.10, min=0.0, max=1.0, step=0.05),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("UMAP neighbors"),
                                    dcc.Input(id="urp-umap-nn", type="number", value=60, min=2, step=1),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("UMAP min_dist"),
                                    dcc.Input(id="urp-umap-min-dist", type="number", value=0.12, min=0.0, step=0.01),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("UMAP metric"),
                                    dcc.Dropdown(
                                        id="urp-umap-metric",
                                        options=[
                                            {"label": "cosine", "value": "cosine"},
                                            {"label": "euclidean", "value": "euclidean"},
                                            {"label": "manhattan", "value": "manhattan"},
                                            {"label": "hellinger (PMF probs)", "value": "hellinger"},
                                        ],
                                        value="cosine",
                                        clearable=False,
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("Random seed"),
                                    dcc.Input(id="urp-umap-seed", type="number", value=42, step=1),
                                    html.Div(style={"height": "12px"}),
                                    html.Button("Load / Compute embedding", id="urp-compute-embed", className="btn-ghost"),
                                    html.Div(id="urp-embed-status", className="panel-subtitle", style={"marginTop": "10px"}),
                                ],
                            ),
                            _panel(
                                "Region selection",
                                "Select via lasso/box OR use axis ranges.",
                                [
                                    html.Label("Selection mode"),
                                    dcc.RadioItems(
                                        id="urp-select-mode",
                                        options=[
                                            {"label": "Use plot selection", "value": "select"},
                                            {"label": "Use axis ranges", "value": "ranges"},
                                        ],
                                        value="select",
                                        labelStyle={"display": "block"},
                                    ),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("U range (only if axis ranges)"),
                                    dcc.RangeSlider(id="urp-u-range", min=-1, max=1, value=[-1, 1], allowCross=False),
                                    html.Div(style={"height": "10px"}),
                                    html.Label("V range (only if axis ranges)"),
                                    dcc.RangeSlider(id="urp-v-range", min=-1, max=1, value=[-1, 1], allowCross=False),
                                    html.Div(style={"height": "10px"}),
                                    html.Div(id="urp-selected-count", className="panel-subtitle"),
                                ],
                            ),
                        ],
                    ),
                    html.Div(
                        style={"minWidth": "520px", "flex": "1 1 720px"},
                        children=[
                            _panel(
                                "Embedding scatter",
                                "Use box/lasso; selection drives conditional PMF below.",
                                [
                                    dcc.Store(id="urp-embed-store"),
                                    dcc.Store(id="urp-selected-store"),
                                    dcc.Graph(id="urp-embed-graph", figure=_error_fig("Click 'Load / Compute embedding'."), config=embed_config),
                                ],
                            ),
                            _panel(
                                "Conditional PMF over region",
                                "Compute P(x|R) (or F(x|R)) for selected variants.",
                                [
                                    html.Div(
                                        style={"display": "flex", "gap": "14px", "flexWrap": "wrap", "alignItems": "flex-end"},
                                        children=[
                                            html.Div(
                                                style={"minWidth": "260px"},
                                                children=[
                                                    html.Label("PMF metric(s)"),
                                                    dcc.Dropdown(
                                                        id="urp-pmf-metrics",
                                                        options=[],
                                                        value=[],
                                                        multi=True,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"minWidth": "240px"},
                                                children=[
                                                    html.Label("Output kind"),
                                                    dcc.Dropdown(
                                                        id="urp-output-kind",
                                                        options=[
                                                            {"label": "P(x | R)", "value": "P"},
                                                            {"label": "F(x | R)", "value": "F"},
                                                        ],
                                                        value="F",
                                                        clearable=False,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"minWidth": "240px"},
                                                children=[
                                                    html.Label("Weights"),
                                                    dcc.Dropdown(
                                                        id="urp-weight-mode",
                                                        options=[{"label": "uniform", "value": "uniform"}]
                                                        + [{"label": f"col::{c}", "value": f"col::{c}"} for c in weight_cols],
                                                        value="uniform",
                                                        clearable=False,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"minWidth": "240px"},
                                                children=[
                                                    html.Label("Preserve per-variant normalization"),
                                                    dcc.Dropdown(
                                                        id="urp-preserve-norm",
                                                        options=[
                                                            {"label": "yes", "value": "yes"},
                                                            {"label": "no", "value": "no"},
                                                        ],
                                                        value="yes",
                                                        clearable=False,
                                                    ),
                                                ],
                                            ),
                                            html.Div(
                                                style={"minWidth": "220px"},
                                                children=[
                                                    html.Label("kT override (optional)"),
                                                    dcc.Input(id="urp-kt-override", type="number", value=None, step=0.05),
                                                ],
                                            ),
                                        ],
                                    ),
                                    html.Div(style={"height": "12px"}),
                                    dcc.Graph(id="urp-pmf-graph", figure=_error_fig("Select a region on the embedding.")),

                                    # --- Per-variant raw PMF with bootstrap CI bands ---
                                    # Shows individual member PMFs + their replica-block
                                    # bootstrap CIs for a chosen metric. Deliberately
                                    # separate from the aggregated conditional plot
                                    # above because aggregating CIs through the
                                    # weighted-mixture path would require resampling
                                    # inside the mixture.
                                    html.Div(style={"height": "12px"}),
                                    html.H4("Per-variant raw PMF with 95% CI", className="panel-title"),
                                    html.Div(
                                        "Plots each selected variant's raw F(x) with replica-block "
                                        "bootstrap 95% CI bands (kJ/mol). Capped at 10 variants for "
                                        "legibility. Requires F_ci_lo_kJ_mol / F_ci_hi_kJ_mol columns "
                                        "produced by the batch pipeline.",
                                        className="panel-subtitle",
                                    ),
                                    html.Div(
                                        style={"marginBottom": "6px"},
                                        children=[
                                            html.Label("Metric for CI plot: ",
                                                       style={"marginRight": "6px", "fontSize": "12px"}),
                                            dcc.Dropdown(
                                                id="urp-ci-metric",
                                                options=[],
                                                value=None,
                                                clearable=True,
                                                style={"maxWidth": "260px", "display": "inline-block",
                                                       "verticalAlign": "middle"},
                                            ),
                                        ],
                                    ),
                                    dcc.Graph(id="urp-raw-ci-graph",
                                              figure=_error_fig("Select a region and a metric.")),

                                    html.Div(style={"height": "10px"}),
                                    html.H4("Selected variants & weights", className="panel-title"),
                                    dash_table.DataTable(
                                        id="urp-weights-table",
                                        columns=[
                                            {"name": "variant", "id": "variant"},
                                            {"name": "weight", "id": "weight"},
                                        ],
                                        page_size=12,
                                        style_table={"overflowX": "auto"},
                                        style_cell={"fontFamily": "monospace", "fontSize": "12px", "padding": "6px"},
                                    ),
                                ],
                            ),
                        ],
                    ),
                ],
            )
        ],
    )

# ----------------------------------------------------------------------
# Callbacks
# ----------------------------------------------------------------------

def register_callbacks(app, ctx: Any) -> None:
    from .shared import apply_theme
    cols = PmfColumns()
    features_df = _infer_features_df(ctx)

    @app.callback(
        Output("urp-status-features-rows", "children"),
        Output("urp-status-pmf-rows", "children"),
        Output("urp-status-pmf-metrics", "children"),
        Output("urp-status-pmf-variants", "children"),
        Output("urp-embed-metrics", "options"),
        Output("urp-embed-metrics", "value"),
        Output("urp-pmf-metrics", "options"),
        Output("urp-pmf-metrics", "value"),
        Output("urp-pmf-repr", "value"),
        Output("urp-energy-units", "value"),
        Input("tabs", "value"),
        State("urp-embed-metrics", "value"),
        State("urp-pmf-metrics", "value"),
        State("urp-pmf-repr", "value"),
        State("urp-energy-units", "value"),
    )
    def _populate_live_pmf_controls(active_tab, current_embed_metrics, current_pmf_metrics, current_repr, current_units):
        features_df = _get_live_features_df(ctx)
        feat_rows = len(features_df) if isinstance(features_df, pd.DataFrame) else 0
        feat_text = f"features rows: {feat_rows}"
        if active_tab != TAB_VALUE:
            return (
                feat_text,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
                no_update,
            )

        pmf_df = _get_live_pmf_df(ctx)
        metrics = _live_pmf_metrics(pmf_df, cols)
        options = [{"label": metric_display_label(m), "value": m} for m in metrics]
        embed_value = _pick_metric_defaults(metrics, current_embed_metrics, 4)
        pmf_value = _pick_metric_defaults(metrics, current_pmf_metrics, 2)
        pmf_rows = len(pmf_df) if isinstance(pmf_df, pd.DataFrame) else 0
        pmf_variants = (
            len(pmf_df[cols.variant].astype(str).dropna().unique())
            if isinstance(pmf_df, pd.DataFrame) and not pmf_df.empty and cols.variant in pmf_df.columns
            else 0
        )
        has_p = isinstance(pmf_df, pd.DataFrame) and (cols.p in pmf_df.columns)
        has_f = isinstance(pmf_df, pd.DataFrame) and (cols.f in pmf_df.columns)
        repr_value = current_repr if current_repr in {"P", "F"} else ("P" if has_p else "F")
        if repr_value == "P" and not has_p and has_f:
            repr_value = "F"
        if repr_value == "F" and not has_f and has_p:
            repr_value = "P"
        units_value = current_units if current_units in {"kJ/mol", "kT"} else ("kJ/mol" if has_f else "kT")
        return (
            feat_text,
            f"pmf_df rows: {pmf_rows}",
            f"pmf metrics: {len(metrics)}",
            f"pmf variants: {pmf_variants}",
            options,
            embed_value,
            options,
            pmf_value,
            repr_value,
            units_value,
        )

    @app.callback(
        Output("urp-embed-store", "data"),
        Output("urp-embed-graph", "figure"),
        Output("urp-u-range", "min"),
        Output("urp-u-range", "max"),
        Output("urp-u-range", "value"),
        Output("urp-v-range", "min"),
        Output("urp-v-range", "max"),
        Output("urp-v-range", "value"),
        Output("urp-embed-status", "children"),
        Input("urp-compute-embed", "n_clicks"),
        State("urp-embed-source", "value"),
        State("urp-embed-metrics", "value"),
        State("urp-embed-features", "value"),
        State("urp-pmf-repr", "value"),
        State("urp-energy-units", "value"),
        State("urp-temp-k", "value"),
        State("urp-pmf-variant-mode", "value"),
        State("urp-pmf-x-round", "value"),
        State("urp-pmf-max-bins", "value"),
        State("urp-pmf-integral-norm", "value"),
        State("urp-umap-nn", "value"),
        State("urp-umap-min-dist", "value"),
        State("urp-umap-metric", "value"),
        State("urp-embed-method", "value"),
        State("urp-dens-lambda", "value"),
        State("urp-dens-frac", "value"),
        State("urp-dens-var-shift", "value"),
        State("urp-umap-seed", "value"),
        Input("theme-store", "data"),
        prevent_initial_call=True,
    )
    def _compute_embedding(
        _n_clicks: Optional[int],
        embed_source: str,
        embed_metrics: Optional[list[str]],
        embed_features: Optional[list[str]],
        pmf_repr: str,
        energy_units: str,
        temp_k: Optional[float],
        pmf_variant_mode: str,
        pmf_x_round: Optional[int],
        pmf_max_bins: Optional[int],
        pmf_integral_norm: str,
        nn: Optional[int],
        min_dist: Optional[float],
        umap_metric: str,
        embed_method: str,
        dens_lambda: Optional[float],
        dens_frac: Optional[float],
        dens_var_shift: Optional[float],
        seed: Optional[int],
        theme,
    ):
        features_df = _get_live_features_df(ctx)
        pmf_df = _get_live_pmf_df(ctx)
        pmf_variants = (
            set(pmf_df[cols.variant].astype(str).dropna().unique())
            if isinstance(pmf_df, pd.DataFrame) and not pmf_df.empty
            and cols.variant in pmf_df.columns else set()
        )
        temp_k = float(temp_k) if temp_k is not None else 300.0
        seed_i = int(seed) if seed is not None else 42

        # 1) Existing embedding
        if embed_source == "existing":
            emb = _existing_embedding_from_features(features_df)
            if emb is not None:
                emb = _add_has_pmf_flag(emb, pmf_variants)
                umin, umax = float(np.nanmin(emb["u"])), float(np.nanmax(emb["u"]))
                vmin, vmax = float(np.nanmin(emb["v"])), float(np.nanmax(emb["v"]))
                store = {"cfg": {"embed_source": "existing"}, "embedding": emb.to_dict("records")}
                fig = apply_theme(_scatter_fig(emb), theme)
                return store, fig, umin, umax, [umin, umax], vmin, vmax, [vmin, vmax], f"Loaded existing embedding: {len(emb)} variants."
            embed_source = "pmf"

        # 2) PMF embedding
        if embed_source == "pmf":
            if not HAVE_UMAP:
                fig = apply_theme(_error_fig("UMAP not installed. Install `umap-learn`."), theme)
                return None, fig, -1, 1, [-1, 1], -1, 1, [-1, 1], "UMAP missing."
            if not isinstance(pmf_df, pd.DataFrame) or pmf_df.empty:
                fig = apply_theme(_error_fig("pmf_df is empty."), theme)
                return None, fig, -1, 1, [-1, 1], -1, 1, [-1, 1], "pmf_df empty."

            metrics = list(embed_metrics or [])
            if not metrics:
                fig = apply_theme(_error_fig("Select at least one PMF metric for embedding."), theme)
                return None, fig, -1, 1, [-1, 1], -1, 1, [-1, 1], "No PMF metrics selected."

            X_raw, colnames_raw, meta = _pmf_matrix_for_umap(
                pmf_df=pmf_df,
                cols=cols,
                metrics=metrics,
                use_repr="P" if (pmf_repr or "P") == "P" else "F",
                energy_units=energy_units or "kJ/mol",
                T_K=temp_k,
                x_round_decimals=int(pmf_x_round) if pmf_x_round is not None else 4,
                max_bins_per_metric=int(pmf_max_bins) if pmf_max_bins is not None else 256,
                normalize_by_integral=(pmf_integral_norm or "yes") == "yes",
                variant_set_mode=pmf_variant_mode or "intersection",
            )
            if meta.empty or X_raw.size == 0:
                fig = apply_theme(_error_fig("PMF design matrix empty (try fewer metrics or switch to union mode)."), theme)
                return None, fig, -1, 1, [-1, 1], -1, 1, [-1, 1], "PMF matrix empty."

            try:
                emb_xy = _compute_umap_embedding(
                    X_raw,
                    colnames_raw=colnames_raw,
                    matrix_type="pmf",
                    n_neighbors=int(nn) if nn is not None else 60,
                    min_dist=float(min_dist) if min_dist is not None else 0.12,
                    metric=str(umap_metric),
                    seed=seed_i,
                    embedding_method=str(embed_method),
                    dens_lambda=dens_lambda,
                    dens_frac=dens_frac,
                    dens_var_shift=dens_var_shift,
                )
                emb_df = pd.DataFrame({"variant": meta["variant"].astype(str).tolist(), "u": emb_xy[:, 0], "v": emb_xy[:, 1]})
                emb_df = _add_has_pmf_flag(emb_df, pmf_variants)
                umin, umax = float(np.nanmin(emb_df["u"])), float(np.nanmax(emb_df["u"]))
                vmin, vmax = float(np.nanmin(emb_df["v"])), float(np.nanmax(emb_df["v"]))
                store = {
                    "cfg": {
                        "embed_source": "pmf",
                        "embed_method": str(embed_method),
                        "n_neighbors": int(nn) if nn is not None else 60,
                        "min_dist": float(min_dist) if min_dist is not None else 0.12,
                        "metric": str(umap_metric),
                        "seed": seed_i,
                        "dens_lambda": float(dens_lambda) if dens_lambda is not None else None,
                        "dens_frac": float(dens_frac) if dens_frac is not None else None,
                        "dens_var_shift": float(dens_var_shift) if dens_var_shift is not None else None,
                        "pmf_variant_mode": str(pmf_variant_mode or "intersection"),
                        "pmf_x_round": int(pmf_x_round) if pmf_x_round is not None else 4,
                        "pmf_max_bins": int(pmf_max_bins) if pmf_max_bins is not None else 256,
                        "pmf_integral_norm": (pmf_integral_norm or "yes"),
                    },
                    "embedding": emb_df.to_dict("records"),
                }
                fig = apply_theme(_scatter_fig(emb_df), theme)
                return store, fig, umin, umax, [umin, umax], vmin, vmax, [vmin, vmax], f"Computed PMF embedding ({embed_method}): {len(emb_df)} variants."
            except Exception as e:
                fig = apply_theme(_error_fig(f"PMF embedding error: {e}"), theme)
                return None, fig, -1, 1, [-1, 1], -1, 1, [-1, 1], f"PMF embedding error: {e}"

        # 3) Feature embedding
        if embed_source == "features":
            if not HAVE_UMAP:
                fig = apply_theme(_error_fig("UMAP not installed. Install `umap-learn`."), theme)
                return None, fig, -1, 1, [-1, 1], -1, 1, [-1, 1], "UMAP missing."
            if not isinstance(features_df, pd.DataFrame) or features_df.empty or "variant" not in features_df.columns:
                fig = apply_theme(_error_fig("features missing or missing 'variant'."), theme)
                return None, fig, -1, 1, [-1, 1], -1, 1, [-1, 1], "features missing."

            feats = [c for c in (embed_features or []) if c in features_df.columns]
            if not feats:
                fig = apply_theme(_error_fig("Select at least one numeric feature column."), theme)
                return None, fig, -1, 1, [-1, 1], -1, 1, [-1, 1], "No features selected."

            X_raw, meta, feats_used = _prepare_feature_embedding_matrix(features_df, feats)
            if X_raw.size == 0 or meta.empty or not feats_used:
                fig = apply_theme(_error_fig("Selected numeric features do not contain enough finite, non-constant data for embedding."), theme)
                return None, fig, -1, 1, [-1, 1], -1, 1, [-1, 1], "No usable feature rows."

            feats = feats_used

            try:
                emb_xy = _compute_umap_embedding(
                    X_raw,
                    colnames_raw=feats,
                    matrix_type="features",
                    n_neighbors=int(nn) if nn is not None else 60,
                    min_dist=float(min_dist) if min_dist is not None else 0.12,
                    metric=str(umap_metric),
                    seed=seed_i,
                    embedding_method=str(embed_method),
                    dens_lambda=dens_lambda,
                    dens_frac=dens_frac,
                    dens_var_shift=dens_var_shift,
                )
                emb_df = pd.DataFrame({"variant": meta["variant"].astype(str).tolist(), "u": emb_xy[:, 0], "v": emb_xy[:, 1]})
                emb_df = _add_has_pmf_flag(emb_df, pmf_variants)
                umin, umax = float(np.nanmin(emb_df["u"])), float(np.nanmax(emb_df["u"]))
                vmin, vmax = float(np.nanmin(emb_df["v"])), float(np.nanmax(emb_df["v"]))
                store = {
                    "cfg": {
                        "embed_source": "features",
                        "embed_method": str(embed_method),
                        "n_neighbors": int(nn) if nn is not None else 60,
                        "min_dist": float(min_dist) if min_dist is not None else 0.12,
                        "metric": str(umap_metric),
                        "seed": seed_i,
                        "dens_lambda": float(dens_lambda) if dens_lambda is not None else None,
                        "dens_frac": float(dens_frac) if dens_frac is not None else None,
                        "dens_var_shift": float(dens_var_shift) if dens_var_shift is not None else None,
                    },
                    "embedding": emb_df.to_dict("records"),
                }
                fig = apply_theme(_scatter_fig(emb_df), theme)
                return store, fig, umin, umax, [umin, umax], vmin, vmax, [vmin, vmax], f"Computed feature embedding ({embed_method}): {len(emb_df)} variants."
            except Exception as e:
                fig = apply_theme(_error_fig(f"Feature embedding error: {e}"), theme)
                return None, fig, -1, 1, [-1, 1], -1, 1, [-1, 1], f"Feature embedding error: {e}"

        fig = apply_theme(_error_fig("Could not load/compute embedding."), theme)
        return None, fig, -1, 1, [-1, 1], -1, 1, [-1, 1], "No embedding."

    @app.callback(
        Output("urp-selected-store", "data"),
        Output("urp-selected-count", "children"),
        Input("urp-embed-graph", "selectedData"),
        Input("urp-embed-graph", "clickData"),
        Input("urp-select-mode", "value"),
        Input("urp-u-range", "value"),
        Input("urp-v-range", "value"),
        State("urp-embed-store", "data"),
    )
    def _select_region(selectedData: Optional[dict], clickData: Optional[dict], mode: str, u_range: list[float], v_range: list[float], store: Optional[dict]):
        pmf_df = _get_live_pmf_df(ctx)
        pmf_variants = (
            set(pmf_df[cols.variant].astype(str).dropna().unique())
            if isinstance(pmf_df, pd.DataFrame) and not pmf_df.empty
            and cols.variant in pmf_df.columns else set()
        )
        if not store or "embedding" not in store:
            return [], "Selected: 0"

        emb = pd.DataFrame(store["embedding"])
        if emb.empty or "variant" not in emb.columns:
            return [], "Selected: 0"

        mode = (mode or "select").lower().strip()

        if mode == "ranges":
            try:
                u0, u1 = float(u_range[0]), float(u_range[1])
                v0, v1 = float(v_range[0]), float(v_range[1])
            except Exception:
                return [], "Selected: 0"
            m = emb["u"].between(min(u0, u1), max(u0, u1)) & emb["v"].between(min(v0, v1), max(v0, v1))
            sel = emb.loc[m, "variant"].astype(str).tolist()
        else:
            pts = (selectedData or {}).get("points") or []
            if not pts:
                pts = (clickData or {}).get("points") or []

            sel = []
            for p in pts:
                v = _as_variant_from_point(p) if isinstance(p, dict) else None
                if v is not None:
                    sel.append(str(v))

        sel = sorted(set(map(str, sel)))
        raw_n = len(sel)
        if pmf_variants:
            sel = [v for v in sel if v in pmf_variants]
        msg = f"Selected: {len(sel)}"
        if pmf_variants and raw_n != len(sel):
            msg += f" (PMF: {len(sel)}/{raw_n})"
        return sel, msg

    @app.callback(
        Output("urp-pmf-graph", "figure"),
        Output("urp-weights-table", "data"),
        Input("urp-selected-store", "data"),
        Input("urp-pmf-metrics", "value"),
        Input("urp-weight-mode", "value"),
        Input("urp-pmf-repr", "value"),
        Input("urp-energy-units", "value"),
        Input("urp-temp-k", "value"),
        Input("urp-kt-override", "value"),
        Input("urp-preserve-norm", "value"),
        Input("urp-output-kind", "value"),
        Input("theme-store", "data"),
    )
    def _update_conditional(
        selected: Optional[list[str]],
        pmf_metrics: Optional[list[str]],
        weight_mode: str,
        pmf_repr: str,
        energy_units: str,
        temp_k: Optional[float],
        kt_override: Optional[float],
        preserve_norm: str,
        output_kind: str,
        theme,
    ):
        pmf_df = _get_live_pmf_df(ctx)
        selected = list(map(str, selected or []))
        metrics = list(map(str, pmf_metrics or []))
        if not selected:
            return apply_theme(_error_fig("Select a region on the embedding first."), theme), []
        if not metrics:
            return apply_theme(_error_fig("Select at least one PMF metric."), theme), []

        features_df = _get_live_features_df(ctx)
        temp_k = float(temp_k) if temp_k is not None else 300.0
        preserve_variant_norm = (preserve_norm or "yes").lower().strip() == "yes"

        # weights
        weights: Dict[str, float] = {}
        if (weight_mode or "uniform") == "uniform":
            w = 1.0 / max(1, len(selected))
            weights = {v: w for v in selected}
        elif weight_mode.startswith("col::") and isinstance(features_df, pd.DataFrame) and (not features_df.empty) and ("variant" in features_df.columns):
            col = weight_mode.split("col::", 1)[1]
            if col in features_df.columns:
                s = features_df.set_index("variant")[col]
                for v in selected:
                    try:
                        weights[v] = float(s.get(v, 0.0))
                    except Exception:
                        weights[v] = 0.0
                Z = float(np.sum(list(weights.values())))
                if not np.isfinite(Z) or Z <= 0:
                    w = 1.0 / max(1, len(selected))
                    weights = {v: w for v in selected}
                else:
                    weights = {v: float(wv) / Z for v, wv in weights.items()}
            else:
                w = 1.0 / max(1, len(selected))
                weights = {v: w for v in selected}
        else:
            w = 1.0 / max(1, len(selected))
            weights = {v: w for v in selected}

        curves = _conditional_pmf(
            pmf_df=pmf_df,
            cols=cols,
            variants=selected,
            metrics=metrics,
            weights=weights,
            use_repr="P" if (pmf_repr or "P") == "P" else "F",
            energy_units=energy_units or "kJ/mol",
            T_K=temp_k,
            kT_override=float(kt_override) if (kt_override is not None and np.isfinite(float(kt_override))) else None,
            preserve_variant_norm=preserve_variant_norm,
            output_kind="F" if (output_kind or "F") == "F" else "P",
        )

        table = [{"variant": v, "weight": f"{weights.get(v, 0.0):.6g}"} for v in selected[:500]]

        if not curves:
            try:
                m = pmf_df[cols.variant].astype(str).isin(selected) & pmf_df[cols.metric].astype(str).isin(metrics)
                matched_rows = int(m.sum())
            except Exception:
                matched_rows = 0
            msg = (
                f"No conditional PMF data for this selection (selected_variants={len(selected)}, "
                f"metrics={len(metrics)}, matched_rows={matched_rows}). "
                "Tip: use the lasso/box tools (or click) on points labeled 'PMF available', "
                "or set Embedding source='Compute from PMF probability matrix' to guarantee overlap."
            )
            return apply_theme(_error_fig(msg), theme), table

        fig = apply_theme(_pmf_fig(curves, output_kind="F" if (output_kind or "F") == "F" else "P"), theme)
        return fig, table

    # --- Per-variant CI panel callbacks --------------------------------------
    # Populate the CI metric dropdown with metrics that (a) the user has
    # selected in urp-pmf-metrics, (b) have actual data among the selected
    # variants, and (c) have at least one finite F_ci_* value there.
    @app.callback(
        Output("urp-ci-metric", "options"),
        Output("urp-ci-metric", "value"),
        Input("urp-selected-store", "data"),
        Input("urp-pmf-metrics", "value"),
        State("urp-ci-metric", "value"),
    )
    def _populate_urp_ci_metric(selected, pmf_metrics, current):
        pmf_df = _get_live_pmf_df(ctx)
        if not isinstance(pmf_df, pd.DataFrame) or pmf_df.empty:
            return [], None
        if not _shared_pmf_has_ci_columns(pmf_df):
            return [], None

        selected = list(map(str, selected or []))
        if not selected:
            return [], None

        region_sub = pmf_df[pmf_df[cols.variant].astype(str).isin(selected)]
        if region_sub.empty:
            return [], None

        candidate_metrics = region_sub[cols.metric].astype(str).dropna().unique().tolist()
        if pmf_metrics:
            sel = set(map(str, pmf_metrics))
            candidate_metrics = [m for m in candidate_metrics if m in sel]

        good: list[str] = []
        for m in candidate_metrics:
            dm = region_sub[region_sub[cols.metric].astype(str) == m]
            lo = pd.to_numeric(dm["F_ci_lo_kJ_mol"], errors="coerce")
            if lo.notna().any():
                good.append(m)
        good = sorted(good, key=torsion_sort_key)
        options = [{"label": metric_display_label(m), "value": m} for m in good]
        value = current if current in good else (good[0] if good else None)
        return options, value

    @app.callback(
        Output("urp-raw-ci-graph", "figure"),
        Input("urp-selected-store", "data"),
        Input("urp-ci-metric", "value"),
        Input("theme-store", "data"),
    )
    def _urp_raw_ci_graph(selected, ci_metric, theme):
        pmf_df = _get_live_pmf_df(ctx)
        if not isinstance(pmf_df, pd.DataFrame) or pmf_df.empty:
            return apply_theme(_error_fig("pmf_df is empty."), theme)
        if not _shared_pmf_has_ci_columns(pmf_df):
            return apply_theme(_error_fig(
                "F_ci_lo_kJ_mol / F_ci_hi_kJ_mol columns not found in pmf_df. "
                "Re-run the batch pipeline with --pmf-bootstrap > 0."
            ), theme)

        selected = list(map(str, selected or []))
        if not selected:
            return apply_theme(_error_fig("Select a region on the embedding first."), theme)
        if not ci_metric:
            return apply_theme(_error_fig("Pick a metric that has bootstrap CI data."), theme)

        curves, bands = _shared_per_variant_raw_pmf_with_ci(
            pmf_df,
            variants=selected,
            metric=str(ci_metric),
            variant_col=cols.variant,
            metric_col=cols.metric,
            x_col=cols.x,
            f_col=cols.f,
            max_variants=10,
        )
        if not curves:
            return apply_theme(_error_fig(f"No PMF data for metric '{ci_metric}' in this region."), theme)

        extra = f" (showing first 10 of {len(selected)})" if len(selected) > 10 else ""
        title = f"Per-variant raw PMF + replica-block 95% CI - {metric_display_label(str(ci_metric))}{extra}"
        return apply_theme(_shared_pmf_overlay_fig(
            curves,
            title=title,
            y_title="F (kJ/mol)",
            ci_bands=bands,
        ), theme)

def tab_spec(ctx: Any) -> Dict[str, Any]:
    return {"label": TAB_LABEL, "value": TAB_VALUE, "layout": layout(ctx), "register": register_callbacks}
