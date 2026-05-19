from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Sequence, Set, Tuple

from itertools import permutations

import re
import os

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from pathlib import Path

from ..theming.errors import error_fig
from ..data.context import filter_numeric_columns
from ..data.io import _resolve_layout
from ..data.pmf_input import PMF_CORE_FEATURES, PMF_TRANSFORM_RULES, transform_features
from ..analysis.pmf_vectorize import build_pmf_design_matrix, parse_family
from ..data.variant_pmf import PMF_PLOT_COLS, available_pmf_metrics, load_variant_pmfs
from ..metrics import (prettify_column_label, torsion_sort_key, metric_display_label, is_torsion_feature_column, circular_encode_torsion_angles)

TAB_LABEL = "Variance & PCA"


PCA_PRESET_NATIVE = "native"
PCA_PRESET_BASIN = "basin_geometry"
PCA_PRESET_PMF_SHAPE = "pmf_shape"
PCA_PRESET_SEQUENCE = "sequence"


def _pca_preset_mode(value) -> str:
    """Normalize the PCA input-space preset.

    Backward compatibility: older code used a boolean thermodynamic toggle.
    True maps to the corrected basin-geometry preset; False maps to native.
    """
    if value is True:
        return PCA_PRESET_BASIN
    if value is False or value is None:
        return PCA_PRESET_NATIVE
    v = str(value).strip().lower()
    aliases = {
        "thermo": PCA_PRESET_BASIN,
        "thermodynamic": PCA_PRESET_BASIN,
        "basin": PCA_PRESET_BASIN,
        "corrected": PCA_PRESET_BASIN,
        "pmf": PCA_PRESET_PMF_SHAPE,
        "shape": PCA_PRESET_PMF_SHAPE,
        "pmf_vector": PCA_PRESET_PMF_SHAPE,
        "pmf_vectors": PCA_PRESET_PMF_SHAPE,
        "seq": PCA_PRESET_SEQUENCE,
        "sequence": PCA_PRESET_SEQUENCE,
        "letters": PCA_PRESET_SEQUENCE,
        "letter": PCA_PRESET_SEQUENCE,
        "aminoacid": PCA_PRESET_SEQUENCE,
        "amino_acid": PCA_PRESET_SEQUENCE,
        "aa": PCA_PRESET_SEQUENCE,
    }
    valid = {PCA_PRESET_NATIVE, PCA_PRESET_BASIN, PCA_PRESET_PMF_SHAPE, PCA_PRESET_SEQUENCE}
    return aliases.get(v, v if v in valid else PCA_PRESET_NATIVE)


def _is_basin_preset(value) -> bool:
    return _pca_preset_mode(value) == PCA_PRESET_BASIN


def _is_pmf_shape_preset(value) -> bool:
    return _pca_preset_mode(value) == PCA_PRESET_PMF_SHAPE


def _is_sequence_preset(value) -> bool:
    return _pca_preset_mode(value) == PCA_PRESET_SEQUENCE


def _available_pmf_metrics(ctx) -> List[str]:
    """Return available PMF metric names using the fast shared PMF store."""
    try:
        return available_pmf_metrics(ctx)
    except Exception:
        return []


from ..data.variant_pmf import normalize_pmf_curve_columns as _normalize_pmf_curve_columns


def _coerce_pmf_metric_selection(selected: Optional[Sequence[str]], available: Sequence[str]) -> List[str]:
    """Keep selected PMF metrics if possible; otherwise default to all available metrics."""
    avail = [str(m) for m in available if m is not None]
    if not avail:
        return []
    if selected:
        sel_set = {str(m) for m in selected if m is not None}
        kept = [m for m in avail if m in sel_set]
        if kept:
            return kept
    return avail



# -------------------------------------------------------------------
# Figure sizing / readability defaults
# -------------------------------------------------------------------

_FONT_BASE = 14
_FONT_TITLE = 18
_FONT_AXIS_TITLE = 16
_FONT_TICKS = 12


def _pca_scatter_square_range(df: pd.DataFrame, cx: str, cy: str, pad: float = 1.05) -> float:
    """Return symmetric half-range so both PC axes use identical limits."""
    try:
        vals = np.concatenate([
            df[cx].dropna().values if cx in df.columns else np.array([]),
            df[cy].dropna().values if cy in df.columns else np.array([]),
        ])
        return float(np.nanmax(np.abs(vals)) * pad) if vals.size else 5.0
    except Exception:
        return 5.0


def _apply_readable_layout(
    fig: go.Figure,
    *,
    height: int | None = None,
    square_2d: bool = False,
    tight_margins: bool = False,
):
    """Apply consistent sizing + readable typography.

    square_2d enforces 1:1 aspect for x/y axes (useful for PC scatter & biplots).
    """
    try:
        fig.update_layout(
            font=dict(size=_FONT_BASE),
            title=dict(font=dict(size=_FONT_TITLE)),
        )
        if height is not None:
            fig.update_layout(height=int(height))

        # Axes readability
        if hasattr(fig.layout, "xaxis"):
            fig.update_xaxes(
                title_font=dict(size=_FONT_AXIS_TITLE),
                tickfont=dict(size=_FONT_TICKS),
            )
        if hasattr(fig.layout, "yaxis"):
            fig.update_yaxes(
                title_font=dict(size=_FONT_AXIS_TITLE),
                tickfont=dict(size=_FONT_TICKS),
            )

        # 3D axes readability (if present)
        if hasattr(fig.layout, "scene") and fig.layout.scene is not None:
            fig.update_layout(
                scene=dict(
                    xaxis=dict(title_font=dict(size=_FONT_AXIS_TITLE), tickfont=dict(size=_FONT_TICKS)),
                    yaxis=dict(title_font=dict(size=_FONT_AXIS_TITLE), tickfont=dict(size=_FONT_TICKS)),
                    zaxis=dict(title_font=dict(size=_FONT_AXIS_TITLE), tickfont=dict(size=_FONT_TICKS)),
                )
            )

        if square_2d and hasattr(fig.layout, "xaxis") and hasattr(fig.layout, "yaxis"):
            # keep a 1:1 aspect ratio in pixels
            fig.update_yaxes(scaleanchor="x", scaleratio=1)
            fig.update_xaxes(constrain="domain")

        if tight_margins:
            fig.update_layout(margin=dict(l=10, r=10, t=50, b=10))
    except Exception:
        # never fail a callback due to styling
        return fig

    return fig


TECHNICAL_COL_REGEXES = [
    # Replica-vs-pooled divergence diagnostics (batch_ana "js_reps_to_pooled_*", "L1_reps_to_pooled_*")
    re.compile(r"(^|__)js_reps_to_pooled", re.IGNORECASE),
    re.compile(r"(^|__)l1_reps_to_pooled", re.IGNORECASE),
    # Autocorrelation / effective-sample-size diagnostics
    re.compile(r"(^|__)tau_int_frames", re.IGNORECASE),
    re.compile(r"(^|__)ess(_|$)", re.IGNORECASE),
    re.compile(r"(^|__)mbar(_|$)", re.IGNORECASE),
    # Convergence-threshold summary columns (e.g. metric__n_frames_JS_lt_0.01)
    re.compile(r"(^|__)n_frames_(js|rmsef)_lt_", re.IGNORECASE),
    re.compile(r"(^|__)frac_(js|rmsef)_lt_", re.IGNORECASE),
    re.compile(r"(^|__)converged_bool$", re.IGNORECASE),
]

def _is_technical(col: str) -> bool:
    """Drop pipeline-diagnostic columns (JS/L1/tau/convergence flags), keep scientific stats."""
    if col == "variant":
        return False
    s = str(col)
    return any(rx.search(s) for rx in TECHNICAL_COL_REGEXES)


def _group_columns_by_prefix(cols: List[str]) -> dict:
    groups: dict = {}
    for c in cols:
        if "__" in c:
            prefix = c.split("__", 1)[0]
        else:
            prefix = "misc"
        groups.setdefault(prefix, []).append(c)
    return groups


def _wide_feature_stat(col: object) -> Optional[str]:
    """Return the suffix after ``__`` for wide ``metric__feature`` columns."""
    c = str(col)
    if "__" not in c:
        return None
    return c.split("__", 1)[1]


def _is_pmf_core_wide_col(col: object) -> bool:
    """True for unit-safe PMF landscape metrics in wide form."""
    return _wide_feature_stat(col) in set(PMF_CORE_FEATURES)


def _pmf_core_wide_cols(df: pd.DataFrame, cols: Optional[List[str]] = None) -> List[str]:
    """Find numeric PMF intrinsic landscape columns in a wide feature table."""
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    base_cols = cols if cols is not None else list(df.columns)
    return [
        c for c in base_cols
        if c in df.columns
        and c != "variant"
        and pd.api.types.is_numeric_dtype(df[c])
        and _is_pmf_core_wide_col(c)
    ]


def _is_sequence_feature_col(col: object) -> bool:
    """True for sequence/letter-space descriptors in feature tables."""
    s = str(col)
    return (
        s.startswith("seq_")
        or s == "sequence"
        or s in {"L", "KD_mean", "KD_std", "KD_min", "KD_max", "KD_Nterm", "KD_Cterm"}
    )


def _sequence_pca_feature_cols(df: pd.DataFrame, raw_cols: Optional[Sequence[str]] = None) -> List[str]:
    """Return numeric sequence descriptors for sequence-space PCA.

    The dedicated sequence mode uses normalized composition/position descriptors
    and deliberately omits raw ``*_count`` columns, which mostly re-encode length
    and can dominate PCA when mixed with fractions.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []

    raw = list(raw_cols) if raw_cols is not None else [
        c for c in df.columns if c != "variant" and pd.api.types.is_numeric_dtype(df[c])
    ]
    raw_set = {str(c) for c in raw}
    aa_order = "ACDEFGHIKLMNPQRSTVWY"

    preferred: List[str] = []
    explicit = [
        "seq_length",
        "seq_known_frac",
        "seq_unknown_frac",
        "seq_net_charge_sidechain_pH7_approx",
        "seq_net_charge_pH7_approx",
        "seq_abs_net_charge_pH7_approx",
        "seq_charge_density_pH7_approx",
        "seq_abs_charge_density_pH7_approx",
        "seq_kd_mean",
        "seq_kd_std",
        "seq_kd_min",
        "seq_kd_max",
        "seq_kd_range",
        "seq_kd_nterm",
        "seq_kd_cterm",
        "seq_composition_entropy_bits",
        "seq_composition_entropy_norm20",
        "seq_unique_aa_count",
        "seq_unique_aa_frac20",
        "seq_max_run_len",
        "seq_max_run_frac",
    ]
    preferred.extend([c for c in explicit if c in raw_set])
    preferred.extend([f"seq_aa_{aa}_frac" for aa in aa_order if f"seq_aa_{aa}_frac" in raw_set])

    for prefix in (
        "seq_frac_",
        "seq_nterm_",
        "seq_cterm_",
        "seq_npos",
        "seq_cpos",
        "seq_dipep_",
        "seq_class_transition_",
    ):
        for c in sorted(c for c in raw_set if c.startswith(prefix)):
            if c.endswith("_count"):
                continue
            if prefix in {"seq_dipep_", "seq_class_transition_"} and not c.endswith("_frac"):
                continue
            preferred.append(c)

    if not preferred:
        legacy = ["L", "KD_mean", "KD_std", "KD_min", "KD_max", "KD_Nterm", "KD_Cterm"]
        preferred.extend([c for c in legacy if c in raw_set])

    out: List[str] = []
    seen: set[str] = set()
    for c in preferred:
        if c in seen or c not in df.columns:
            continue
        if not pd.api.types.is_numeric_dtype(df[c]):
            continue
        out.append(c)
        seen.add(c)
    return out


def _transform_wide_pmf_core(df_num: pd.DataFrame) -> Tuple[pd.DataFrame, str]:
    """Apply PMF intrinsic transforms to wide ``metric__feature`` columns.

    ``data.pmf_input.transform_features`` expects bare feature names, so this
    wrapper maps each wide column to its suffix-specific transform rule, then
    restores the original wide names.  NaNs are preserved here; the normal PCA
    preparation step still handles missingness/imputation consistently.
    """
    if not isinstance(df_num, pd.DataFrame) or df_num.empty:
        return pd.DataFrame(index=getattr(df_num, "index", None)), ""

    tmp = df_num.copy()
    rename: dict[str, str] = {}
    rev: dict[str, str] = {}
    rules: dict[str, str] = {}

    for i, c in enumerate(tmp.columns.astype(str)):
        stat = _wide_feature_stat(c) or c
        tmp_name = f"{stat}__wide{i}"
        rename[c] = tmp_name
        rev[tmp_name] = c
        rules[tmp_name] = PMF_TRANSFORM_RULES.get(stat, "zscore")

    tmp = tmp.rename(columns=rename)
    Xt, meta = transform_features(tmp, rules, scaler_scope="current_selection")
    Xt = Xt.rename(columns=rev)

    used = sorted({str(v) for v in meta.get("columns", {}).values() if v != "none"})
    note = "PMF intrinsic transforms applied by metric suffix"
    if used:
        note += f" ({', '.join(used)})."
    else:
        note += "."
    return Xt, note




def _build_pmf_shape_feature_frame(ctx, variant_subset: Optional[Set[str]] = None, pmf_metrics: Optional[Sequence[str]] = None) -> Tuple[pd.DataFrame, str]:
    """Build a PCA-ready feature frame from the PMFs themselves.

    Rows are variants. Columns are PMF probability-mass bins named
    ``metric|x=<bin>``. Values are transformed to ``sqrt(P_mass)`` so
    Euclidean PCA operates in the Hellinger geometry of distributions.
    Metric blocks are additionally scaled by ``1/sqrt(n_bins)`` so a PMF
    family with many bins does not dominate purely by dimensionality.
    """
    try:
        pmf_df = ctx.pmf_df
    except Exception as exc:
        return pd.DataFrame(), f"Could not load PMF table for PMF-shape PCA: {exc}"

    if not isinstance(pmf_df, pd.DataFrame) or pmf_df.empty:
        return pd.DataFrame(), "No PMF table available for PMF-shape PCA."
    required = {"variant", "metric", "x"}
    if not required.issubset(set(pmf_df.columns)):
        missing = ", ".join(sorted(required - set(pmf_df.columns)))
        return pd.DataFrame(), f"PMF table lacks required columns for PMF-shape PCA: {missing}."

    pmf_df = _normalize_pmf_curve_columns(pmf_df)
    available_metrics = sorted(pmf_df["metric"].dropna().astype(str).unique().tolist(), key=str.lower)
    metrics = _coerce_pmf_metric_selection(pmf_metrics, available_metrics)
    if not metrics:
        return pd.DataFrame(), "PMF table has no selected metric values for PMF-shape PCA."
    dsel = pmf_df[pmf_df["metric"].astype(str).isin(metrics)].copy()
    if dsel.empty:
        return pd.DataFrame(), "Selected PMF metrics are not present in the PMF table."
    if "P" not in dsel.columns and "F_kJ_mol" not in dsel.columns:
        return pd.DataFrame(), "PMF table has no usable probability or free-energy column (expected P or F_kJ_mol/aliases)."

    X, colnames, meta, grids = build_pmf_design_matrix(
        dsel,
        metrics,
        use_repr="P" if "P" in dsel.columns else "F",
        energy_units="kJ/mol",
        T_K=300.0,
        max_bins_per_metric=192,
        variant_policy="union_mean",
        missing_impute="mean",
        dirichlet_alpha=0.0,
    )
    if X.size == 0 or not colnames or meta.empty or "variant" not in meta.columns:
        return pd.DataFrame(), "Could not construct PMF-shape design matrix."

    X = np.sqrt(np.clip(X, 0.0, None))

    families = [parse_family(c) for c in colnames]
    fam_counts = pd.Series(families).value_counts().to_dict()
    weights = np.array([1.0 / np.sqrt(max(1, int(fam_counts.get(f, 1)))) for f in families], dtype=float)
    X = X * weights[None, :]

    out = pd.DataFrame(X, columns=colnames)
    out.insert(0, "variant", meta["variant"].astype(str).to_numpy())

    if variant_subset:
        keep = {str(v) for v in variant_subset}
        out = out[out["variant"].astype(str).isin(keep)].reset_index(drop=True)

    note = (
        f"PMF-shape PCA: sqrt(P_mass) design matrix with Hellinger geometry; "
        f"{len(metrics)} selected PMF metric block(s), {len(colnames)} bins/features, "
        f"{out.shape[0]} variants. Missing metric blocks use mean-imputation. "
        "Metric blocks scaled by 1/sqrt(n_bins)."
    )
    return out, note

def _zscore(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1)
    sd = np.where(sd == 0, 1.0, sd)
    return (X - mu) / sd, mu, sd



DEFAULT_DROP_COL_MISS_FRAC = 0.30


def _prepare_numeric_frame(
    df: pd.DataFrame,
    numeric_cols: List[str],
    *,
    drop_col_missing_frac: float = DEFAULT_DROP_COL_MISS_FRAC,
) -> pd.DataFrame:
    """Prepare numeric feature frame for PCA (drop very-missing cols + median-impute)."""
    cols = [c for c in (numeric_cols or []) if c in df.columns and c != "variant"]
    if not cols:
        return pd.DataFrame()

    sub = df[cols].copy()
    sub = sub.replace([np.inf, -np.inf], np.nan)

    miss = sub.isna().mean()
    keep = miss[miss <= float(drop_col_missing_frac)].index.tolist()
    if not keep:
        return pd.DataFrame()
    sub = sub[keep]

    keep2: List[str] = []
    for c in sub.columns:
        arr = pd.to_numeric(sub[c], errors="coerce").to_numpy(float)
        if np.isfinite(arr).any():
            keep2.append(c)
    if not keep2:
        return pd.DataFrame()
    sub = sub[keep2]

    meds = sub.median(numeric_only=True)
    sub = sub.fillna(meds).fillna(0.0)

    return sub


def _prepare_matrix(df: pd.DataFrame, numeric_cols: List[str]) -> Tuple[np.ndarray, List[str]]:
    sub = _prepare_numeric_frame(df, numeric_cols)
    if sub.empty:
        return np.zeros((0, 0), float), []
    return sub.to_numpy(float), sub.columns.astype(str).tolist()


def _apply_torsion_handling(
    Xz: np.ndarray,
    cols: List[str],
    torsion_mode: Optional[List[str]],
    *,
    n_torsion_concepts: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], str]:
    """Optionally exclude / downweight torsion-derived feature columns.

    n_torsion_concepts: number of original torsion angles before circular
    encoding (each concept → 2 columns after sin/cos split). When provided,
    balance weight is computed per-concept rather than per-encoded-column,
    preventing circular-encoding from halving the effective torsion weight.
    """
    mode = set(torsion_mode or [])
    if Xz.size == 0 or not cols:
        return Xz, cols, ""

    tors_idx = [i for i, c in enumerate(cols) if _is_torsion_feature_col_local(c)]
    if not tors_idx:
        return Xz, cols, ""

    if "exclude" in mode:
        keep = [i for i in range(len(cols)) if i not in set(tors_idx)]
        cols2 = [cols[i] for i in keep]
        X2 = Xz[:, keep] if keep else np.zeros((Xz.shape[0], 0), float)
        return X2, cols2, f"Excluded torsion columns: {len(tors_idx)}"

    if "balance" in mode:
        n_t = len(tors_idx)
        n_eff = n_torsion_concepts if (n_torsion_concepts is not None and n_torsion_concepts > 0) else n_t
        n_all = len(cols)
        n_non = n_all - n_t
        if n_eff > 0 and n_non > 0:
            w = float(np.sqrt(n_non / n_eff))
            X2 = Xz.copy()
            X2[:, tors_idx] *= w
            return X2, cols, f"Balanced torsions (×{w:.3g}) — concepts={n_eff}, encoded={n_t}, non-torsion={n_non}"

    return Xz, cols, ""


def _pca_numpy(X: np.ndarray, n_components: int = 4):
    if X.size == 0:
        return np.array([]), np.zeros((0, 0)), np.zeros((0, 0))
    U, s, Vt = np.linalg.svd(np.nan_to_num(X), full_matrices=False)
    comps = Vt[:n_components]
    scores = (U * s)[:, :n_components]
    evr = (s ** 2) / np.sum(s ** 2) if s.size else np.array([])
    return evr, comps, scores




# -------------------------------------------------------------------
# Stability helpers (split-half / bootstrap + Procrustes)
# -------------------------------------------------------------------

def _fit_pca_loadings(X: np.ndarray, n_components: int) -> np.ndarray:
    """Return loading matrix (features × components) via SVD."""
    if X.size == 0:
        return np.zeros((0, 0), float)
    n_components = int(max(1, n_components))
    U, s, Vt = np.linalg.svd(np.nan_to_num(X), full_matrices=False)
    k = min(n_components, Vt.shape[0])
    return Vt[:k].T


def _best_perm_bruteforce(sim: np.ndarray) -> List[int]:
    """Max-trace assignment for small k (k<=6)."""
    k = int(sim.shape[0])
    best_score = -1.0
    best = list(range(k))
    for perm in permutations(range(k)):
        score = 0.0
        for i in range(k):
            score += float(sim[i, perm[i]])
        if score > best_score:
            best_score = score
            best = list(perm)
    return best


def _align_loadings(ref: np.ndarray, cur: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Align current loadings to reference: permute + flip signs.

    Returns (cur_aligned, cosine_per_pc).
    """
    if ref.size == 0 or cur.size == 0:
        return cur, np.array([], float)
    p, k = ref.shape
    k2 = cur.shape[1]
    k = min(k, k2)
    refk = ref[:, :k]
    curk = cur[:, :k]

    sim = np.abs(refk.T @ curk)
    perm = _best_perm_bruteforce(sim)
    curp = curk[:, perm]

    cos = np.zeros(k, float)
    for i in range(k):
        dot = float(refk[:, i].T @ curp[:, i])
        sgn = 1.0 if dot >= 0 else -1.0
        curp[:, i] *= sgn
        cos[i] = abs(dot)
    return curp, cos


def _procrustes_disparity(A: np.ndarray, B: np.ndarray, *, allow_scale: bool = True) -> float:
    """Procrustes disparity between two score matrices (n×k).

    Returns a normalized Frobenius residual after optimal rotation/reflection,
    (and optional scaling). Lower is better; 0 means identical up to transform.
    """
    if A.size == 0 or B.size == 0:
        return float("nan")
    A = np.asarray(A, float)
    B = np.asarray(B, float)
    n = min(A.shape[0], B.shape[0])
    k = min(A.shape[1], B.shape[1])
    if n < 3 or k < 1:
        return float("nan")
    A = A[:n, :k]
    B = B[:n, :k]

    A = A - np.mean(A, axis=0, keepdims=True)
    B = B - np.mean(B, axis=0, keepdims=True)

    normA = float(np.linalg.norm(A))
    normB = float(np.linalg.norm(B))
    if normA == 0 or normB == 0:
        return float("nan")

    if allow_scale:
        A0 = A / normA
        B0 = B / normB
    else:
        A0 = A
        B0 = B

    M = A0.T @ B0
    U, _, Vt = np.linalg.svd(M, full_matrices=False)
    R = U @ Vt
    B_aligned = B0 @ R
    resid = float(np.linalg.norm(A0 - B_aligned))
    denom = float(np.linalg.norm(A0)) or 1.0
    return float(resid / denom)


def _is_torsion_feature_col_local(col: str) -> bool:
    """Local torsion detector (does not depend on metrics.py internals)."""
    base = str(col).split("__", 1)[0]
    b = base.lower()
    for suf in ("_sin", "_cos"):
        if b.endswith(suf):
            b = b[: -len(suf)]
    return bool(re.match(r"^(phi|psi)(?:[_-]?res)?[a-z]{0,6}\d+$", b) or re.match(r"^(phi|psi)_res\d+$", b))


def _filter_impute_and_zscore(
    X: np.ndarray,
    cols: List[str],
    *,
    max_missing_frac: float = 0.30,
) -> Tuple[np.ndarray, List[str], str]:
    """Drop very-missing columns, median-impute rest, then z-score.

    Uses full-set stats for stability comparisons (consistent feature set).
    """
    if X.size == 0 or not cols:
        return np.zeros((0, 0), float), [], "Empty matrix"

    X = np.asarray(X, float)
    finite = np.isfinite(X)
    miss_frac = 1.0 - finite.mean(axis=0)
    keep_mask = (miss_frac <= float(max_missing_frac)) & finite.any(axis=0)
    keep_idx = np.where(keep_mask)[0].tolist()
    if not keep_idx:
        return np.zeros((0, 0), float), [], "All columns dropped by missingness"

    Xk = X[:, keep_idx].copy()
    colsk = [cols[i] for i in keep_idx]

    for j in range(Xk.shape[1]):
        col = Xk[:, j]
        m = np.isfinite(col)
        if not np.any(m):
            Xk[:, j] = 0.0
            continue
        med = float(np.nanmedian(col[m]))
        col[~m] = med
        Xk[:, j] = col

    Xz, _, _ = _zscore(Xk)
    dropped = int(len(cols) - len(colsk))
    note = f"Missingness: dropped {dropped} cols (> {max_missing_frac*100:.0f}% NaN), imputed rest (median)."
    return Xz, colsk, note


def _apply_torsion_handling_local(
    Xz: np.ndarray,
    cols: List[str],
    torsion_mode: Optional[List[str]],
    *,
    n_torsion_concepts: Optional[int] = None,
) -> Tuple[np.ndarray, List[str], str]:
    """Exclude / downweight torsion-derived cols, using local detector."""
    mode = set(torsion_mode or [])
    if Xz.size == 0 or not cols:
        return Xz, cols, ""
    tors_idx = [i for i, c in enumerate(cols) if _is_torsion_feature_col_local(c)]
    if not tors_idx:
        return Xz, cols, ""
    if "exclude" in mode:
        keep = [i for i in range(len(cols)) if i not in set(tors_idx)]
        cols2 = [cols[i] for i in keep]
        X2 = Xz[:, keep] if keep else np.zeros((Xz.shape[0], 0), float)
        return X2, cols2, f"Excluded torsion cols: {len(tors_idx)}"
    if "balance" in mode:
        n_t = len(tors_idx)
        n_eff = n_torsion_concepts if (n_torsion_concepts is not None and n_torsion_concepts > 0) else n_t
        n_non = len(cols) - n_t
        if n_eff > 0 and n_non > 0:
            w = float(np.sqrt(n_non / n_eff))
            X2 = Xz.copy()
            X2[:, tors_idx] *= w
            return X2, cols, f"Balanced torsions (×{w:.3g})"
    return Xz, cols, ""
def layout(ctx):
    df: pd.DataFrame = getattr(ctx, "df", pd.DataFrame())

    color_options = []
    numeric_cols_for_groups: List[str] = []
    if isinstance(df, pd.DataFrame) and not df.empty:
        for c in df.columns:
            if c == "variant":
                continue
            if pd.api.types.is_numeric_dtype(df[c]):
                numeric_cols_for_groups.append(c)
            color_options.append({"label": prettify_column_label(c), "value": c})

    # Apply global exclusions (e.g. "_n", "rms", "js", etc.)
    numeric_cols_for_groups = filter_numeric_columns(numeric_cols_for_groups)

    groups = _group_columns_by_prefix(numeric_cols_for_groups) if numeric_cols_for_groups else {}
    group_options = [{"label": k, "value": k} for k in sorted(groups.keys())]

    # PMF intrinsic landscape groups (for thermodynamic preset).
    # Prefer the unit-safe PMF core columns over the broader annotation set.
    core_cols_for_groups = _pmf_core_wide_cols(df, numeric_cols_for_groups)
    ann_set: Set[str] = getattr(ctx, "_annotation_cols", set())
    ann_groups = sorted({
        str(c).split("__", 1)[0]
        for c in core_cols_for_groups
        if "__" in str(c) and str(c).split("__", 1)[0] in groups
    })
    if not ann_groups:
        ann_groups = sorted({
            str(c).split("__", 1)[0]
            for c in ann_set
            if "__" in str(c) and str(c).split("__", 1)[0] in groups
        })

    # PMF-shape PCA metric selector is populated at layout time.  Keeping this
    # static avoids an initial callback dependency where the PCA callbacks wait
    # on the dropdown value callback before drawing anything.
    pmf_metrics_available = _available_pmf_metrics(ctx)
    pmf_metric_options = [
        {"label": metric_display_label(m), "value": m}
        for m in pmf_metrics_available
    ]
    pmf_metric_status_initial = (
        f"{len(pmf_metrics_available)} PMF metric(s) available; empty selection means all."
        if pmf_metrics_available else
        "No PMF metrics found in ctx.pmf_df."
    )

    # ---- top controls ----
    controls = html.Div(
        [
            html.Div(
                "Variance & PCA",
                style={
                    "fontWeight": 700,
                    "marginBottom": "6px",
                    "fontSize": "1.05em",
                },
            ),
            html.Div(
                [
                    # left: local PCA config
                    html.Div(
                        [
                            html.Div(
                                "Local PCA: top-K variance features",
                                style={"fontSize": "0.8em", "marginBottom": "2px"},
                            ),
                            dcc.Slider(
                                id="var-k",
                                min=6,
                                max=60,
                                step=2,
                                value=20,
                                tooltip={"placement": "bottom"},
                            ),

                            html.Div(
                                [
                                    html.Div(
                                        "Rank features by",
                                        style={"fontSize": "0.8em", "marginTop": "6px"},
                                    ),
                                    dcc.RadioItems(
                                        id="pca-rank-method",
                                        options=[
                                            {"label": "Variance", "value": "var"},
                                            {"label": "SNR (variance / MAD)", "value": "snr"},
                                        ],
                                        value="var",
                                        inline=False,
                                        style={"fontSize": "0.75em"},
                                    ),
                                ],
                                style={"marginTop": "2px"},
                            ),
                            html.Div(
                                [
                                    dcc.Checklist(
                                        id="pca-drop-correlated",
                                        options=[
                                            {
                                                "label": "Drop redundant features (|corr| > threshold)",
                                                "value": "drop",
                                            }
                                        ],
                                        value=["drop"],
                                        style={"fontSize": "0.75em", "marginTop": "4px"},
                                    ),
                                    dcc.Slider(
                                        id="pca-corr-thresh",
                                        min=0.70,
                                        max=0.99,
                                        step=0.01,
                                        value=0.95,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ],
                                style={"marginTop": "2px"},
                            ),
                            html.Div(
                                [
                                    dcc.Checklist(
                                        id="hide-tech",
                                        options=[
                                            {
                                                "label": "Hide technical columns (JS/L1 divergence, tau/ESS, convergence flags)",
                                                "value": "yes",
                                            }
                                        ],
                                        value=["yes"],
                                        style={"fontSize": "0.75em", "marginTop": "4px"},
                                    )
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Feature groups (for local PCA input)",
                                        style={"fontSize": "0.8em", "marginTop": "6px"},
                                    ),
                                    dcc.Dropdown(
                                        id="pca-feature-groups",
                                        options=group_options,
                                        value=[g["value"] for g in group_options] if group_options else [],
                                        multi=True,
                                        placeholder="Select groups (default: all)",
                                        style={"fontSize": "0.8em", "marginTop": "2px"},
                                    ),
                                    html.Div(
                                        "Grouping uses prefix before '__' in column names.",
                                        style={"fontSize": "0.7em", "opacity": 0.7, "marginTop": "2px"},
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                "PCA input preset",
                                                style={"fontSize": "0.8em", "marginTop": "8px"},
                                            ),
                                            html.Div(
                                                [
                                                    html.Button("Native values", id="pca-preset-native", n_clicks=0, style={"fontSize": "0.72em", "padding": "4px 7px", "borderRadius": "6px", "cursor": "pointer"}),
                                                    html.Button("Basin geometry corrected", id="pca-preset-basin", n_clicks=0, style={"fontSize": "0.72em", "padding": "4px 7px", "borderRadius": "6px", "cursor": "pointer", "marginLeft": "4px"}),
                                                    html.Button("PMFs themselves", id="pca-preset-pmf-shape", n_clicks=0, style={"fontSize": "0.72em", "padding": "4px 7px", "borderRadius": "6px", "cursor": "pointer", "marginLeft": "4px"}),
                                                    html.Button("Sequence descriptors", id="pca-preset-sequence", n_clicks=0, style={"fontSize": "0.72em", "padding": "4px 7px", "borderRadius": "6px", "cursor": "pointer", "marginLeft": "4px"}),
                                                ],
                                                style={"marginTop": "4px"},
                                            ),
                                            html.Div(
                                                "Native keeps scalar physical values; basin corrected uses unit-safe PMF landscape descriptors; PMFs themselves uses sqrt(P_mass) vectors; sequence uses seq_* letter-space descriptors.",
                                                style={"fontSize": "0.68em", "opacity": 0.72, "marginTop": "3px", "lineHeight": "1.25"},
                                            ),
                                            dcc.Store(id="pca-thermo-preset-active", data=PCA_PRESET_NATIVE),
                                            html.Div(
                                                [
                                                    html.Div(
                                                        "PMF metric selector (PMFs themselves only)",
                                                        style={"fontSize": "0.78em", "marginTop": "7px"},
                                                    ),
                                                    dcc.Dropdown(
                                                        id="pca-pmf-metrics",
                                                        options=pmf_metric_options,
                                                        value=[],
                                                        multi=True,
                                                        placeholder="Default: all PMF metrics",
                                                        style={"fontSize": "0.76em", "marginTop": "2px"},
                                                    ),
                                                    html.Div(
                                                        pmf_metric_status_initial,
                                                        id="pca-pmf-metric-status",
                                                        style={"fontSize": "0.68em", "opacity": 0.72, "marginTop": "3px", "lineHeight": "1.25"},
                                                    ),
                                                ]
                                            ),
                                        ]
                                    ),
                                    html.Div(
                                        [
                                            html.Div(
                                                "Torsion (φ/ψ) feature handling",
                                                style={"fontSize": "0.8em", "marginTop": "8px"},
                                            ),
                                            dcc.Checklist(
                                                id="pca-torsion-mode",
                                                options=[
                                                    {"label": "Exclude torsion-derived columns", "value": "exclude"},
                                                    {
                                                        "label": "Balance torsions vs non-torsions (downweight if many)",
                                                        "value": "balance",
                                                    },
                                                ],
                                                value=["balance"],
                                                style={"fontSize": "0.75em", "marginTop": "4px"},
                                            ),

                                            html.Div(
                                                "Angle encoding (torsions)",
                                                style={"fontSize": "0.8em", "marginTop": "8px"},
                                            ),
                                            dcc.Checklist(
                                                id="pca-angle-encoding",
                                                options=[
                                                    {"label": "Circular-encode torsion angles (sin/cos)", "value": "encode"},
                                                ],
                                                value=["encode"],
                                                style={"fontSize": "0.75em", "marginTop": "4px"},
                                            ),
                                            html.Div(
                                                "Applied before z-scoring; torsion balancing prevents φ/ψ blocks from dominating due to dimensionality.",
                                                style={"fontSize": "0.7em", "opacity": 0.7, "marginTop": "2px"},
                                            ),
                                        ]
                                    ),
                                ]
                            ),
                        ],
                        style={
                            "flex": "1 1 35%",
                            "minWidth": "280px",
                            "paddingRight": "12px",
                        },
                    ),
                    # middle: PCA scatter config
                    html.Div(
                        [
                            html.Div(
                                "PCA scatter configuration",
                                style={"fontSize": "0.8em", "marginBottom": "4px"},
                            ),
                            html.Div(
                                [
                                    html.Span("Source: ", style={"fontSize": "0.75em"}),
                                    dcc.RadioItems(
                                        id="pca-scatter-source",
                                        options=[
                                            {"label": "Global", "value": "global"},
                                            {"label": "Local (top-K)", "value": "local"},
                                        ],
                                        value="global",
                                        inline=True,
                                        style={"fontSize": "0.8em"},
                                    ),
                                ],
                            ),
                            html.Div(
                                [
                                    html.Span("Dimensions: ", style={"fontSize": "0.75em"}),
                                    dcc.RadioItems(
                                        id="pca-scatter-dims",
                                        options=[
                                            {"label": "2D", "value": 2},
                                            {"label": "3D", "value": 3},
                                        ],
                                        value=2,
                                        inline=True,
                                        style={"fontSize": "0.8em"},
                                    ),
                                ],
                                style={"marginTop": "2px"},
                            ),
                            html.Div(
                                [
                                    html.Span(
                                        "Axes (PC indices):",
                                        style={"fontSize": "0.75em", "display": "block"},
                                    ),
                                    html.Div(
                                        [
                                            dcc.Dropdown(
                                                id="pca-x-axis",
                                                options=[
                                                    {"label": f"PC{i}", "value": i}
                                                    for i in range(1, 7)
                                                ],
                                                value=1,
                                                clearable=False,
                                                style={"width": "30%", "fontSize": "0.8em"},
                                            ),
                                            dcc.Dropdown(
                                                id="pca-y-axis",
                                                options=[
                                                    {"label": f"PC{i}", "value": i}
                                                    for i in range(1, 7)
                                                ],
                                                value=2,
                                                clearable=False,
                                                style={
                                                    "width": "30%",
                                                    "marginLeft": "4px",
                                                    "fontSize": "0.8em",
                                                },
                                            ),
                                            dcc.Dropdown(
                                                id="pca-z-axis",
                                                options=[
                                                    {"label": f"PC{i}", "value": i}
                                                    for i in range(1, 7)
                                                ],
                                                value=3,
                                                clearable=False,
                                                style={
                                                    "width": "30%",
                                                    "marginLeft": "4px",
                                                    "fontSize": "0.8em",
                                                },
                                            ),
                                        ],
                                        style={
                                            "display": "flex",
                                            "marginTop": "2px",
                                        },
                                    ),
                                ],
                                style={"marginTop": "4px"},
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Clustering (PC space)",
                                        style={"fontSize": "0.8em", "marginTop": "6px"},
                                    ),
                                    dcc.Dropdown(
                                        id="pca-kmeans",
                                        options=[
                                            {"label": "None", "value": 0},
                                            {"label": "k = 2", "value": 2},
                                            {"label": "k = 3", "value": 3},
                                            {"label": "k = 4", "value": 4},
                                            {"label": "k = 5", "value": 5},
                                        ],
                                        value=0,
                                        clearable=False,
                                        style={"fontSize": "0.8em", "marginTop": "2px"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        "Biplot: number of vectors",
                                        style={"fontSize": "0.8em", "marginTop": "6px"},
                                    ),
                                    dcc.Slider(
                                        id="pca-biplot-nvec",
                                        min=3,
                                        max=30,
                                        step=1,
                                        value=12,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ]
                            ),
                        ],
                        style={
                            "flex": "1 1 40%",
                            "minWidth": "320px",
                            "paddingRight": "12px",
                            "borderLeft": "1px solid rgba(200,200,200,0.4)",
                            "paddingLeft": "12px",
                        },
                    ),
                    # right: coloring
                    html.Div(
                        [
                            html.Div(
                                "Coloring",
                                style={"fontSize": "0.8em", "marginBottom": "4px"},
                            ),
                            dcc.Dropdown(
                                id="pca-color",
                                options=color_options,
                                value=None,
                                placeholder="Color points by (optional)",
                                clearable=True,
                                style={"fontSize": "0.8em"},
                            ),
                            html.Div(
                                "Supports any column; categorical → discrete palette, numeric → continuous.",
                                style={"fontSize": "0.7em", "opacity": 0.7, "marginTop": "4px"},
                            ),
                        ],
                        style={
                            "flex": "1 1 25%",
                            "minWidth": "240px",
                            "borderLeft": "1px solid rgba(200,200,200,0.4)",
                            "paddingLeft": "12px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "alignItems": "stretch",
                    "gap": "12px",
                    "flexWrap": "wrap",
                },
            ),
            html.Div(
                "Local PCA operates on the strongest-variance scalar features within selected feature groups. "
                "Global PCA is a one-shot PCA over all numeric features (excluding tech columns). "
                "Scatter uses the variant selection from the Features tab.",
                style={
                    "fontSize": "0.75em",
                    "marginTop": "6px",
                    "opacity": 0.75,
                },
            ),
        ],
        style={
            "padding": "12px",
            "border": "1px solid #ccc",
            "borderRadius": "10px",
            "marginBottom": "12px",
            "backgroundColor": "rgba(255,255,255,0.03)",
        },
    )

    # ---- unified PCA row: scree | loadings | heatmap | scatter ----
    _gh = "52vh"  # shared graph height for the main row
    main_card = html.Div(
        [
            dcc.Store(id="pca-curve-variants", data=[]),
            html.Div(
                [
                    # scree plot
                    html.Div(
                        [dcc.Graph(id="pca-ev", style={"height": _gh}, config={"displaylogo": False})],
                        style={"flex": "0 0 12%", "minWidth": "160px"},
                    ),
                    # loadings bar
                    html.Div(
                        [dcc.Graph(id="pca-load", style={"height": _gh}, config={"displaylogo": False})],
                        style={
                            "flex": "0 0 26%", "minWidth": "220px",
                            "borderLeft": "1px solid rgba(200,200,200,0.3)", "paddingLeft": "8px",
                        },
                    ),
                    # group-contribution heatmap
                    html.Div(
                        [dcc.Graph(id="pca-group-contrib", style={"height": _gh}, config={"displaylogo": False})],
                        style={
                            "flex": "0 0 24%", "minWidth": "200px",
                            "borderLeft": "1px solid rgba(200,200,200,0.3)", "paddingLeft": "8px",
                        },
                    ),
                    # scatter (largest)
                    html.Div(
                        [dcc.Graph(id="pca-scatter", style={"height": _gh}, config={"displaylogo": False})],
                        style={
                            "flex": "1 1 38%", "minWidth": "280px",
                            "borderLeft": "1px solid rgba(200,200,200,0.3)", "paddingLeft": "8px",
                        },
                    ),
                ],
                style={"display": "flex", "gap": "0", "alignItems": "stretch", "width": "100%"},
            ),
        ],
        style={
            "padding": "8px",
            "border": "1px solid #ddd",
            "borderRadius": "10px",
            "marginBottom": "8px",
            "backgroundColor": "rgba(255,255,255,0.02)",
        },
    )

    # ---- PMF curve viewer — full width below PCA row ----
    pmf_card = html.Div(
        [
            html.Div(
                [
                    html.Div(
                        "PMF curve viewer",
                        style={"fontWeight": 600, "fontSize": "0.9em", "marginRight": "16px"},
                    ),
                    html.Div("Metric:", style={"fontSize": "0.8em", "alignSelf": "center", "marginRight": "6px"}),
                    html.Div(
                        dcc.Dropdown(
                            id="pca-curve-metric",
                            options=[],
                            value=None,
                            placeholder="populated after first click",
                            clearable=False,
                            style={"fontSize": "0.8em", "width": "280px"},
                        ),
                        style={"flex": "0 0 auto"},
                    ),
                    html.Button(
                        "Clear",
                        id="pca-curve-clear",
                        n_clicks=0,
                        style={
                            "marginLeft": "10px",
                            "padding": "3px 10px",
                            "borderRadius": "6px",
                            "border": "1px solid rgba(180,180,180,0.8)",
                            "cursor": "pointer",
                            "fontSize": "0.8em",
                        },
                    ),
                    html.Div(
                        id="pca-curve-variant-display",
                        children="Click or lasso-select points in the scatter above.",
                        style={"flex": "1 1 auto", "fontSize": "0.73em", "opacity": 0.7,
                               "paddingLeft": "14px", "alignSelf": "center"},
                    ),
                ],
                style={"display": "flex", "alignItems": "center", "flexWrap": "wrap",
                       "gap": "4px", "marginBottom": "6px"},
            ),
            html.Div(
                id="pca-curve-loader-status",
                children="PMF viewer idle. Select variants and a metric to load curves.",
                className="pmf-loader-status",
            ),
            dcc.Loading(
                html.Div(
                    [
                        html.Div(
                            html.Div(className="pmf-progress-bar"),
                            className="pmf-progress-track",
                            title="Indeterminate PMF-loading progress indicator",
                        ),
                        dcc.Graph(
                            id="pca-curve-overlay",
                            style={"height": "38vh"},
                            config={"displaylogo": False},
                        ),
                    ],
                    className="pmf-curve-loading-shell",
                ),
                type="circle",
            ),
        ],
        style={
            "padding": "8px",
            "border": "1px solid #ddd",
            "borderRadius": "10px",
            "marginBottom": "8px",
            "backgroundColor": "rgba(255,255,255,0.02)",
        },
    )

    # ---- secondary details (biplot, outlier, diagnostics) — collapsed by default ----
    secondary_card = html.Div(
        [
            html.Details(
                [
                    html.Summary(
                        "Secondary panels: vector biplot, outliers, top-K table, cluster members",
                        style={"fontSize": "0.82em", "cursor": "pointer", "opacity": 0.75},
                    ),
                    html.Div(
                        [
                            html.Div(
                                [dcc.Graph(id="pca-vector-biplot", style={"height": "32vh"}, config={"displaylogo": False})],
                                style={"flex": "1 1 50%", "minWidth": "260px"},
                            ),
                            html.Div(
                                [dcc.Graph(id="pca-outlier-plot", style={"height": "32vh"}, config={"displaylogo": False})],
                                style={"flex": "1 1 50%", "minWidth": "260px",
                                       "borderLeft": "1px solid rgba(200,200,200,0.3)", "paddingLeft": "10px"},
                            ),
                        ],
                        style={"display": "flex", "gap": "0", "marginTop": "8px"},
                    ),
                    html.Div(
                        [
                            html.Div("Top-K variance features", style={"fontSize": "0.82em", "fontWeight": 500, "marginTop": "8px", "marginBottom": "3px"}),
                            html.Div(id="var-table", style={"maxHeight": "200px", "overflowY": "auto", "fontSize": "0.78em",
                                                             "border": "1px solid rgba(200,200,200,0.4)", "borderRadius": "6px", "padding": "4px"}),
                            html.Div(id="pca-diagnostics", style={"marginTop": "4px", "fontSize": "0.73em", "opacity": 0.85}),
                            html.Div(id="pca-pc-detail", style={"marginTop": "4px", "fontSize": "0.73em", "opacity": 0.85}),
                            html.Div(id="pca-cluster-members", style={"marginTop": "4px", "fontSize": "0.78em", "maxHeight": "200px", "overflowY": "auto"}),
                        ],
                    ),
                ],
            ),
        ],
        style={"padding": "6px 10px", "border": "1px solid #ddd", "borderRadius": "10px",
               "marginBottom": "8px", "backgroundColor": "rgba(255,255,255,0.02)"},
    )

    # ---- stability card ----
    stability_card = html.Div(
        [
            html.Div(
                "PCA stability (split-half / bootstrap) + Procrustes geometry check",
                style={
                    "fontWeight": 600,
                    "fontSize": "0.95em",
                    "marginBottom": "6px",
                },
            ),
            html.Div(
                [
                    html.Div(
                        [
                            html.Div("Settings", style={"fontSize": "0.85em", "fontWeight": 500}),
                            html.Div(
                                [
                                    html.Span("Basis: ", style={"fontSize": "0.75em"}),
                                    dcc.RadioItems(
                                        id="pca-stab-source",
                                        options=[
                                            {"label": "Global", "value": "global"},
                                            {"label": "Local (top-K)", "value": "local"},
                                        ],
                                        value="global",
                                        inline=True,
                                        style={"fontSize": "0.8em"},
                                    ),
                                ],
                                style={"marginTop": "4px"},
                            ),
                            html.Div(
                                [
                                    html.Span("Method: ", style={"fontSize": "0.75em"}),
                                    dcc.RadioItems(
                                        id="pca-stab-method",
                                        options=[
                                            {"label": "Bootstrap (vs reference)", "value": "bootstrap"},
                                            {"label": "Split-half (A vs B)", "value": "split_half"},
                                        ],
                                        value="split_half",
                                        inline=True,
                                        style={"fontSize": "0.8em"},
                                    ),
                                ],
                                style={"marginTop": "4px"},
                            ),
                            html.Div(
                                [
                                    html.Div("Runs", style={"fontSize": "0.75em", "marginTop": "6px"}),
                                    dcc.Slider(
                                        id="pca-stab-runs",
                                        min=10,
                                        max=200,
                                        step=10,
                                        value=60,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div("Bootstrap sample fraction", style={"fontSize": "0.75em", "marginTop": "6px"}),
                                    dcc.Slider(
                                        id="pca-stab-frac",
                                        min=0.4,
                                        max=1.0,
                                        step=0.05,
                                        value=0.8,
                                        tooltip={"placement": "bottom"},
                                    ),
                                    html.Div(
                                        "Used only for Bootstrap; split-half ignores it.",
                                        style={"fontSize": "0.7em", "opacity": 0.7, "marginTop": "2px"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    html.Div("PCs to compare", style={"fontSize": "0.75em", "marginTop": "6px"}),
                                    dcc.Slider(
                                        id="pca-stab-npcs",
                                        min=2,
                                        max=6,
                                        step=1,
                                        value=4,
                                        tooltip={"placement": "bottom"},
                                    ),
                                ]
                            ),
                            html.Div(
                                [
                                    dcc.Checklist(
                                        id="pca-stab-procscale",
                                        options=[{"label": "Procrustes: allow scaling", "value": "scale"}],
                                        value=["scale"],
                                        style={"fontSize": "0.8em", "marginTop": "8px"},
                                    ),
                                    html.Div(
                                        "Procrustes compares score-space geometry after optimal rotation/reflection (and optional scaling).",
                                        style={"fontSize": "0.7em", "opacity": 0.7, "marginTop": "2px"},
                                    ),
                                ]
                            ),
                            html.Button(
                                "Run stability",
                                id="pca-stab-run",
                                n_clicks=0,
                                style={
                                    "marginTop": "10px",
                                    "padding": "6px 10px",
                                    "borderRadius": "8px",
                                    "border": "1px solid rgba(180,180,180,0.8)",
                                    "cursor": "pointer",
                                },
                            ),
                            html.Div(
                                id="pca-stab-summary",
                                style={
                                    "marginTop": "8px",
                                    "fontSize": "0.78em",
                                    "borderTop": "1px solid rgba(200,200,200,0.4)",
                                    "paddingTop": "6px",
                                },
                            ),
                        ],
                        style={"flex": "1 1 30%", "minWidth": "280px", "paddingRight": "10px"},
                    ),
                    html.Div(
                        [
                            dcc.Graph(id="pca-stab-cos", style={"height": "34vh"}, config={"displaylogo": False}),
                            dcc.Graph(id="pca-stab-proc", style={"height": "34vh", "marginTop": "6px"}, config={"displaylogo": False}),
                        ],
                        style={
                            "flex": "1 1 70%",
                            "minWidth": "320px",
                            "borderLeft": "1px solid rgba(200,200,200,0.4)",
                            "paddingLeft": "10px",
                        },
                    ),
                ],
                style={
                    "display": "flex",
                    "gap": "12px",
                    "flexWrap": "wrap",
                    "alignItems": "stretch",
                },
            ),
        ],
        style={
            "padding": "10px",
            "border": "1px solid #ddd",
            "borderRadius": "10px",
            "marginTop": "12px",
            "backgroundColor": "rgba(255,255,255,0.02)",
        },
    )
    return html.Div(
        [controls, main_card, pmf_card, secondary_card, stability_card],
        style={"padding": "8px", "maxWidth": "2400px", "margin": "0 auto"},
    )


def register_callbacks(app, ctx):
    from .shared import apply_theme
    df_initial: pd.DataFrame = getattr(ctx, "df", pd.DataFrame())

    def _current_group_values() -> tuple[list[str], list[str]]:
        """Return (all feature groups, PMF-core groups) from the live context."""
        local_df = getattr(ctx, "df", df_initial)
        if not isinstance(local_df, pd.DataFrame) or local_df.empty:
            return [], []
        numeric = [
            c for c in local_df.columns
            if c != "variant" and pd.api.types.is_numeric_dtype(local_df[c])
        ]
        numeric = filter_numeric_columns(numeric)
        groups = _group_columns_by_prefix(numeric) if numeric else {}
        all_groups = sorted(groups.keys())
        core_cols = _pmf_core_wide_cols(local_df, numeric)
        ann_groups = sorted({
            str(c).split("__", 1)[0]
            for c in core_cols
            if "__" in str(c) and str(c).split("__", 1)[0] in groups
        })
        if not ann_groups:
            ann_set: set = getattr(ctx, "_annotation_cols", set())
            ann_groups = sorted({
                str(c).split("__", 1)[0]
                for c in ann_set
                if "__" in str(c) and str(c).split("__", 1)[0] in groups
            })
        return all_groups, ann_groups

    # ---- PCA input-space preset buttons ----
    @app.callback(
        Output("pca-thermo-preset-active", "data"),
        Output("pca-feature-groups", "value"),
        Input("pca-preset-native", "n_clicks"),
        Input("pca-preset-basin", "n_clicks"),
        Input("pca-preset-pmf-shape", "n_clicks"),
        Input("pca-preset-sequence", "n_clicks"),
        prevent_initial_call=True,
    )
    def _set_pca_preset(n_native, n_basin, n_pmf_shape, n_sequence):
        from dash import callback_context

        trig = callback_context.triggered[0]["prop_id"].split(".")[0] if callback_context.triggered else ""
        all_g, ann_g = _current_group_values()
        if trig == "pca-preset-basin":
            return PCA_PRESET_BASIN, ann_g
        if trig == "pca-preset-pmf-shape":
            # PMF-shape PCA uses PMF bins, not feature-table groups. Keep UI groups unchanged/all.
            return PCA_PRESET_PMF_SHAPE, all_g
        if trig == "pca-preset-sequence":
            # Sequence PCA uses a dedicated seq_* column set, not feature-table groups.
            return PCA_PRESET_SEQUENCE, all_g
        return PCA_PRESET_NATIVE, all_g

    # PMF metric selector is populated in layout(); no callback is needed here.

    # ---- global PCA baseline ----
    def _compute_global_pca(df: pd.DataFrame, hide_tech_values=None, group_sel=None, torsion_mode=None, angle_encoding=None, thermo_preset=False, pmf_metrics=None):
        preset_mode = _pca_preset_mode(thermo_preset)

        if preset_mode == PCA_PRESET_PMF_SHAPE:
            pmf_feat_df, _pmf_note = _build_pmf_shape_feature_frame(ctx, pmf_metrics=pmf_metrics)
            if pmf_feat_df.empty:
                scores_df = pd.DataFrame(columns=["variant"])
                loadings = pd.DataFrame()
                evr_series = pd.Series([], dtype=float)
                return scores_df, loadings, evr_series
            df = pmf_feat_df
            torsion_mode = []
            angle_encoding = []

        if df.empty:
            scores_df = pd.DataFrame(columns=["variant"])
            loadings = pd.DataFrame()
            evr_series = pd.Series([], dtype=float)
            return scores_df, loadings, evr_series

        cols = [
            c
            for c in df.columns
            if c != "variant" and pd.api.types.is_numeric_dtype(df[c])
        ]
        ann_cols = getattr(ctx, "_annotation_cols", set())
        if preset_mode == PCA_PRESET_SEQUENCE:
            cols = _sequence_pca_feature_cols(df, cols)
            torsion_mode = []
            angle_encoding = []
        elif preset_mode == PCA_PRESET_BASIN:
            core_cols = _pmf_core_wide_cols(df, cols)
            cols = core_cols if core_cols else [c for c in cols if c in ann_cols]
            torsion_mode = []
            angle_encoding = []
        elif preset_mode == PCA_PRESET_NATIVE:
            cols = filter_numeric_columns(cols)
            cols = [c for c in cols if not _is_sequence_feature_col(c)]
            if "yes" in (hide_tech_values or []):
                cols = [c for c in cols if not _is_technical(c)]
            # Native preset deliberately keeps PMF annotation / physical-coordinate columns.
            # Sequence descriptors live in their own preset so they do not dominate native PCA.
        else:
            cols = filter_numeric_columns(cols)
            if "yes" in (hide_tech_values or []):
                cols = [c for c in cols if not _is_technical(c)]
            if ann_cols:
                cols = [c for c in cols if c not in ann_cols]
        groups = _group_columns_by_prefix(cols)
        if group_sel and preset_mode not in {PCA_PRESET_BASIN, PCA_PRESET_PMF_SHAPE, PCA_PRESET_SEQUENCE}:
            allowed_cols: List[str] = []
            for g in group_sel:
                allowed_cols.extend(groups.get(g, []))
            cols = [c for c in cols if c in allowed_cols]
        df_num_all = _prepare_numeric_frame(df, cols)
        if preset_mode == PCA_PRESET_BASIN and not df_num_all.empty and any(_is_pmf_core_wide_col(c) for c in df_num_all.columns):
            df_num_all, _ = _transform_wide_pmf_core(df_num_all)
        if df_num_all.empty:
            X_all, good_cols_all = np.zeros((0, 0), float), []
        else:
            X_all = df_num_all.to_numpy(float)
            good_cols_all = df_num_all.columns.astype(str).tolist()
        n_torsion_concepts = sum(1 for c in good_cols_all if _is_torsion_feature_col_local(c))
        encode_on = "encode" in (angle_encoding or [])
        X_all, good_cols_all, _ = circular_encode_torsion_angles(
            X_all, good_cols_all, enabled=encode_on, drop_original=True
        )
        if X_all.size == 0 or not good_cols_all:
            scores_df = pd.DataFrame(columns=["variant"])
            loadings = pd.DataFrame()
            evr_series = pd.Series([], dtype=float)
            return scores_df, loadings, evr_series

        Xz, _, _ = _zscore(X_all)
        Xz, good_cols_all, _ = _apply_torsion_handling(Xz, good_cols_all, torsion_mode, n_torsion_concepts=n_torsion_concepts)
        evr_g, comps_g, scores_g = _pca_numpy(Xz, n_components=6)

        n_keep = min(6, scores_g.shape[1])
        pcs_cols = [f"PC{i+1}" for i in range(n_keep)]
        scores_df = pd.DataFrame(scores_g[:, :n_keep], columns=pcs_cols)
        if "variant" in df.columns:
            scores_df.insert(0, "variant", df["variant"].to_numpy()[: scores_df.shape[0]])
        else:
            scores_df.insert(0, "variant", np.arange(scores_df.shape[0]))

        load_cols = min(n_keep, comps_g.shape[0])
        loadings = pd.DataFrame(
            comps_g[:load_cols].T,
            index=good_cols_all,
            columns=[f"PC{i+1}" for i in range(load_cols)],
        )

        if evr_g.size >= n_keep:
            evr_series = pd.Series(evr_g[:n_keep], index=[f"PC{i+1}" for i in range(n_keep)])
        else:
            vals = list(evr_g[:n_keep])
            evr_series = pd.Series(vals, index=[f"PC{i+1}" for i in range(len(vals))])

        return scores_df, loadings, evr_series

    scores_df_global, loadings_global, evr_g_series = _compute_global_pca(df_initial, hide_tech_values=['yes'], group_sel=None, torsion_mode=['balance'], angle_encoding=[])

    # ---- helper: variance table ----
    def _make_var_table(var_series: pd.Series, k: int) -> html.Table:
        df = (
            var_series.reset_index()
            .rename(columns={"index": "feature", 0: "score"})
            .head(int(k or 20))
            .round(4)
        )
        header = html.Tr(
            [html.Th("feature"), html.Th("score")],
            style={"fontWeight": "600"},
        )
        rows = [
            html.Tr(
                [html.Td(str(row["feature"])), html.Td(str(row.get("score", row.get("variance", ""))))]
            )
            for _, row in df.iterrows()
        ]
        return html.Table(
            [header] + rows,
            style={"width": "100%", "borderCollapse": "collapse"},
        )

    # ---- local PCA EV + loadings + group contributions ----
    @app.callback(
        Output("pca-ev", "figure"),
        Output("pca-load", "figure"),
        Output("var-table", "children"),
        Output("pca-pc-detail", "children"),
        Output("pca-group-contrib", "figure"),
        Input("var-k", "value"),
        Input("pca-rank-method", "value"),
        Input("pca-drop-correlated", "value"),
        Input("pca-corr-thresh", "value"),
        Input("hide-tech", "value"),
        Input("pca-feature-groups", "value"),
        Input("pca-torsion-mode", "value"),
        Input("pca-angle-encoding", "value"),
        Input("pca-ev", "clickData"),
        Input("pca-thermo-preset-active", "data"),
        Input("pca-pmf-metrics", "value"),
        Input("theme-store", "data"),
        prevent_initial_call=False,
    )
    def _update_pca(k, rank_method, drop_corr_flags, corr_thr, hide_tech_values, group_sel, torsion_mode, angle_encoding, ev_click, thermo_preset, pmf_metrics, theme):
        preset_mode = _pca_preset_mode(thermo_preset)
        df = getattr(ctx, "df", df_initial)
        thermo_note = ""

        if preset_mode == PCA_PRESET_PMF_SHAPE:
            df, thermo_note = _build_pmf_shape_feature_frame(ctx, pmf_metrics=pmf_metrics)
            torsion_mode = []
            angle_encoding = []

        if not isinstance(df, pd.DataFrame) or df.empty:
            err = apply_theme(error_fig("No data for PCA"), theme)
            empty_fig = apply_theme(error_fig("No data"), theme)
            msg = thermo_note if thermo_note else "No data"
            return err, empty_fig, html.Div(msg), html.Div(), empty_fig

        cols = [
            c
            for c in df.columns
            if c != "variant" and pd.api.types.is_numeric_dtype(df[c])
        ]
        ann_cols = getattr(ctx, "_annotation_cols", set())
        if preset_mode == PCA_PRESET_SEQUENCE:
            cols = _sequence_pca_feature_cols(df, cols)
            thermo_note = "Sequence preset: using numeric seq_* letter-space descriptors."
            torsion_mode = []
            angle_encoding = []
        elif preset_mode == PCA_PRESET_BASIN:
            # Corrected thermodynamic preset: prefer unit-safe PMF intrinsic landscape metrics.
            # Fall back to the broader annotation set only for older datasets.
            core_cols = _pmf_core_wide_cols(df, cols)
            cols = core_cols if core_cols else [c for c in cols if c in ann_cols]
            thermo_note = (
                "Basin-geometry corrected preset: using unit-safe intrinsic PMF landscape metrics."
                if core_cols else
                "Basin-geometry corrected preset: no PMF core columns found; using legacy annotation columns."
            )
            torsion_mode = []
            angle_encoding = []
        elif preset_mode == PCA_PRESET_NATIVE:
            cols = filter_numeric_columns(cols)
            cols = [c for c in cols if not _is_sequence_feature_col(c)]
            if "yes" in (hide_tech_values or []):
                cols = [c for c in cols if not _is_technical(c)]
            # Native preset deliberately keeps PMF annotation / physical-coordinate columns.
            # Sequence descriptors live in their own preset.
        elif preset_mode != PCA_PRESET_PMF_SHAPE:
            cols = filter_numeric_columns(cols)
            if "yes" in (hide_tech_values or []):
                cols = [c for c in cols if not _is_technical(c)]
            if ann_cols:
                cols = [c for c in cols if c not in ann_cols]

        if not cols:
            err = apply_theme(error_fig("No numeric columns available for PCA"), theme)
            empty_fig = apply_theme(error_fig("No numeric columns"), theme)
            return err, empty_fig, html.Div("No numeric columns"), html.Div(), empty_fig

        groups = _group_columns_by_prefix(cols)
        if group_sel and preset_mode not in {PCA_PRESET_BASIN, PCA_PRESET_PMF_SHAPE, PCA_PRESET_SEQUENCE}:
            allowed_cols: List[str] = []
            for g in group_sel:
                allowed_cols.extend(groups.get(g, []))
            cols = [c for c in cols if c in allowed_cols]
            if not cols:
                err = apply_theme(error_fig("No columns after group filter"), theme)
                empty_fig = apply_theme(error_fig("No columns after group filter"), theme)
                return err, empty_fig, html.Div("No columns after group filter"), html.Div(), empty_fig

        raw_var = df[cols].var(ddof=1)
        df_num = _prepare_numeric_frame(df, cols)
        transform_note = ""
        if preset_mode == PCA_PRESET_BASIN and not df_num.empty and any(_is_pmf_core_wide_col(c) for c in df_num.columns):
            df_num, transform_note = _transform_wide_pmf_core(df_num)
            raw_var = df_num.var(ddof=1)
        if df_num.empty:
            err = apply_theme(error_fig("No usable columns after missingness filtering"), theme)
            empty_fig = apply_theme(error_fig("No usable columns"), theme)
            return err, empty_fig, html.Div("No usable columns after missingness filtering."), html.Div(), empty_fig

        scores = raw_var.reindex(df_num.columns).fillna(0.0)
        rank_method = str(rank_method or "var").lower().strip()

        if rank_method == "snr":
            try:
                mad = df_num.apply(lambda col: (col - col.median()).abs().median())
                snr = df_num.var(ddof=1) / (mad.replace(0, np.nan) + 1e-12)
                snr = snr.replace([np.inf, -np.inf], np.nan).dropna()
                if not snr.empty:
                    scores = snr
            except Exception:
                pass

        scores = scores.replace([np.inf, -np.inf], np.nan).dropna()
        scores = scores.sort_values(ascending=False)

        k_int = int(k) if k is not None else 20
        k_int = max(1, min(k_int, len(scores)))
        top = scores.index[:k_int].tolist()

        # Drop near-duplicate features by correlation (keeps first in ranked order)
        try:
            drop_corr = "drop" in (drop_corr_flags or [])
        except Exception:
            drop_corr = False
        if drop_corr and len(top) >= 3:
            try:
                thr = float(corr_thr) if corr_thr is not None else 0.95
            except Exception:
                thr = 0.95
            thr = max(0.0, min(thr, 0.999))
            cm = df_num[top].corr().abs()
            kept: List[str] = []
            for f in top:
                if all(float(cm.loc[f, k]) <= thr for k in kept):
                    kept.append(f)
            top = kept

        X = df_num[top].to_numpy(float)
        good = top
        n_torsion_concepts = sum(1 for c in good if _is_torsion_feature_col_local(c))
        encode_on = "encode" in (angle_encoding or [])
        X, good, angle_note = circular_encode_torsion_angles(X, good, enabled=encode_on, drop_original=True)
        if not good or X.size == 0:
            err = apply_theme(error_fig("No usable data after variance filter"), theme)
            empty_fig = apply_theme(error_fig("No usable data"), theme)
            return err, empty_fig, html.Div("No usable data"), html.Div(), empty_fig

        Xz, _, _ = _zscore(X)
        Xz, good, tors_note = _apply_torsion_handling(Xz, good, torsion_mode, n_torsion_concepts=n_torsion_concepts)
        ncomp = min(6, len(good))
        evr, comps, _scores = _pca_numpy(Xz, n_components=ncomp)

        pc_labels = [f"PC{i+1}" for i in range(len(evr))]
        selected_idx = None
        if ev_click and "points" in ev_click and ev_click["points"]:
            x_label = ev_click["points"][0].get("x")
            if isinstance(x_label, str) and x_label.startswith("PC"):
                try:
                    idx = int(x_label[2:]) - 1
                    if 0 <= idx < len(evr):
                        selected_idx = idx
                except Exception:
                    selected_idx = None

        colors = ["#7f7f7f"] * len(evr)
        if selected_idx is not None:
            colors[selected_idx] = "#d62728"

        fig_ev = go.Figure(
            data=[
                go.Bar(
                    x=pc_labels,
                    y=evr,
                    marker=dict(color=colors),
                )
            ]
        )
        fig_ev.update_layout(
            height=340,
            margin=dict(l=70, r=20, t=55, b=55),
            yaxis_title="Explained variance ratio",
            xaxis_title="Principal component",
        )
        _apply_readable_layout(fig_ev)

        load_df = pd.DataFrame(
            comps[: len(evr)].T,
            index=good,
            columns=[f"PC{i+1}" for i in range(comps.shape[0])],
        )
        fig_load = px.imshow(
            load_df,
            aspect="auto",
            color_continuous_scale="RdBu",
            zmin=-1,
            zmax=1,
        )
        fig_load.update_layout(
            height=620,
            margin=dict(l=95, r=25, t=55, b=55),
            coloraxis_colorbar=dict(
                title=dict(text="loading", font=dict(size=_FONT_AXIS_TITLE)),
                tickfont=dict(size=_FONT_TICKS),
            ),
        )
        _apply_readable_layout(fig_load)
        # Prettify torsion-like feature labels on loadings heatmap
        try:
            _y = load_df.index.astype(str).tolist()
            fig_load.update_yaxes(tickmode='array', tickvals=_y, ticktext=[prettify_column_label(c) for c in _y])
        except Exception:
            pass


        tbl_raw = _make_var_table(scores, k_int)
        note = f"Local PCA input: top {k_int} / {int(df_num.shape[1])} features by {'SNR' if rank_method=='snr' else 'variance'}; corr-pruned={'drop' in (drop_corr_flags or [])} (after technical + missingness filtering)."
        note_bits = [note]
        if thermo_note:
            note_bits.append(thermo_note)
        if transform_note:
            note_bits.append(transform_note)
        tbl = html.Div([
            html.Div(" ".join(note_bits), style={"fontSize": "0.8em", "marginBottom": "6px", "color": "#555"}),
            tbl_raw,
        ])

        if selected_idx is None:
            selected_idx = 0 if len(evr) > 0 else None

        if selected_idx is not None and selected_idx < load_df.shape[1]:
            pc_name = load_df.columns[selected_idx]
            col = load_df[pc_name]
            top_feats = (
                col.reindex(col.abs().sort_values(ascending=False).index)
                .head(10)
                .round(3)
            )
            detail_children = [
                html.Div(
                    f"Selected {pc_name} — explained variance: {evr[selected_idx]*100:.2f}%",
                    style={"fontWeight": 500, "marginBottom": "3px"},
                ),
                (html.Div(tors_note, style={"fontSize": "0.75em", "opacity": 0.75, "marginBottom": "4px"}) if tors_note else html.Div()),
                (html.Div(angle_note, style={"fontSize": "0.75em", "opacity": 0.75, "marginBottom": "4px"}) if angle_note else html.Div()),
                html.Div(
                    "Top contributing features (|loading|):",
                    style={"fontSize": "0.75em", "marginBottom": "2px"},
                ),
                html.Ul(
                    [
                        html.Li(f"{feat}: {val}")
                        for feat, val in top_feats.items()
                    ],
                    style={"paddingLeft": "16px", "margin": 0},
                ),
            ]
        else:
            detail_children = html.Div(
                "Click a bar in the EV plot to inspect top features for that PC.",
                style={"fontSize": "0.75em", "opacity": 0.7},
            )

        # group contribution chart
        if selected_idx is not None and selected_idx < load_df.shape[1]:
            pc_name = load_df.columns[selected_idx]
            groups_all = _group_columns_by_prefix(good)
            contrib_vals = {}
            total = 0.0
            for gname, feats in groups_all.items():
                vals = load_df.loc[[f for f in feats if f in load_df.index], pc_name]
                if not vals.empty:
                    s = float(np.sum(vals.values ** 2))
                    contrib_vals[gname] = s
                    total += s
            if contrib_vals and total > 0:
                for g in contrib_vals:
                    contrib_vals[g] /= total
                contrib_series = pd.Series(contrib_vals).sort_values(ascending=False)
                fig_group = px.bar(
                    contrib_series,
                    labels={"value": "fraction of loading^2", "index": "group"},
                )
                fig_group.update_layout(
                    height=300,
                    margin=dict(l=70, r=20, t=45, b=85),
                    xaxis_title="Feature group",
                    yaxis_title="Fraction of ∑ loading²",
                )
                _apply_readable_layout(fig_group)
            else:
                fig_group = apply_theme(error_fig("No group contributions for this PC"), theme)
        else:
            fig_group = apply_theme(error_fig("No PC selected"), theme)

        return apply_theme(fig_ev, theme), apply_theme(fig_load, theme), tbl, detail_children, apply_theme(fig_group, theme)

    # ---- optional k-means ----
    try:
        from sklearn.cluster import KMeans  # type: ignore
        HAS_SKLEARN = True
    except Exception:
        HAS_SKLEARN = False

    def _compute_outlier_scores(
        df_ref: pd.DataFrame,
        df_eval: pd.DataFrame,
        pc_cols: List[str],
    ) -> pd.Series:
        """Score df_eval relative to the training distribution in df_ref."""
        if df_eval.empty or not pc_cols:
            return pd.Series([], dtype=float)
        mu = df_ref[pc_cols].mean(axis=0)
        std = df_ref[pc_cols].std(axis=0, ddof=1).replace(0, 1.0)
        Z = (df_eval[pc_cols] - mu) / std
        return (Z ** 2).sum(axis=1)


    def _render_cluster_members(plot_df: pd.DataFrame, kmeans_k: int):
        """Render (cluster -> variants) listing for the current scatter selection."""
        if not isinstance(plot_df, pd.DataFrame) or plot_df.empty:
            return html.Div()
        if int(kmeans_k or 0) < 2 or "cluster" not in plot_df.columns:
            return html.Div()
        if "variant" not in plot_df.columns:
            return html.Div(
                "Cluster members unavailable (no 'variant' column in current plot).",
                style={"opacity": 0.7},
            )

        groups = (
            plot_df.groupby("cluster")["variant"]
            .apply(lambda s: sorted({str(x) for x in s if x is not None}))
        )

        header = html.Div(
            f"Cluster members (k = {int(kmeans_k)})",
            style={"fontWeight": 600, "marginBottom": "4px"},
        )

        blocks = []
        for cid in sorted(groups.index.tolist(), key=lambda x: int(x) if str(x).isdigit() else str(x)):
            members = groups.loc[cid]
            blocks.append(
                html.Div(
                    [
                        html.Div(
                            f"Cluster {int(cid)} ({len(members)})",
                            style={"fontWeight": 600, "marginTop": "6px"},
                        ),
                        html.Div(
                            [html.Div(v) for v in members],
                            style={
                                "columnCount": 2,
                                "columnGap": "18px",
                                "fontFamily": "monospace",
                                "fontSize": "0.85em",
                                "lineHeight": "1.35",
                                "whiteSpace": "nowrap",
                            },
                        ),
                    ]
                )
            )

        return html.Div([header] + blocks)
    # ---- scatter + biplot + outliers + diagnostics ----
    @app.callback(
        Output("pca-scatter", "figure"),
        Output("pca-vector-biplot", "figure"),
        Output("pca-outlier-plot", "figure"),
        Output("pca-diagnostics", "children"),
        Output("pca-cluster-members", "children"),
        Input("feat-variant-select", "value"),
        Input("feat-selected-table", "data"),
        Input("pca-scatter-source", "value"),
        Input("pca-scatter-dims", "value"),
        Input("pca-x-axis", "value"),
        Input("pca-y-axis", "value"),
        Input("pca-z-axis", "value"),
        Input("pca-color", "value"),
        Input("pca-kmeans", "value"),
        Input("var-k", "value"),
        Input("hide-tech", "value"),
        Input("pca-feature-groups", "value"),
        Input("pca-torsion-mode", "value"),
        Input("pca-angle-encoding", "value"),
        Input("pca-biplot-nvec", "value"),
        Input("pca-rank-method", "value"),
        Input("pca-drop-correlated", "value"),
        Input("pca-corr-thresh", "value"),
        Input("pca-thermo-preset-active", "data"),
        Input("pca-pmf-metrics", "value"),
        Input("theme-store", "data"),
        prevent_initial_call=False,
    )
    def _update_pca_scatter(
        variant_filter,
        selected_table,
        source_mode: str,
        n_dims: int,
        pc_x: int,
        pc_y: int,
        pc_z: int,
        color_col: Optional[str],
        kmeans_k: int,
        var_k,
        hide_tech_values,
        group_sel,
        torsion_mode,
        angle_encoding,
        biplot_nvec,
        rank_method,
        drop_corr_flags,
        corr_thr,
        thermo_preset,
        pmf_metrics,
        theme,
    ):
        preset_mode = _pca_preset_mode(thermo_preset)
        df = getattr(ctx, "df", df_initial)

        var_set: Set[str] = set()
        if selected_table:
            for row in selected_table:
                v = row.get("variant")
                if v is not None:
                    var_set.add(str(v))
        if (not var_set) and variant_filter:
            for v in variant_filter:
                if v is not None:
                    var_set.add(str(v))

        n_dims = 3 if int(n_dims) == 3 else 2
        pc_x = max(1, min(int(pc_x or 1), 6))
        pc_y = max(1, min(int(pc_y or 2), 6))
        pc_z = max(1, min(int(pc_z or 3), 6))
        kmeans_k = int(kmeans_k or 0)
        n_vec = max(1, int(biplot_nvec or 12))

        # ===== GLOBAL PCA =====
        if source_mode == "global":
            scores_df_g, loadings_g, evr_g = _compute_global_pca(
                df,
                hide_tech_values=hide_tech_values,
                group_sel=group_sel,
                torsion_mode=torsion_mode,
                angle_encoding=angle_encoding,
                thermo_preset=thermo_preset,
                pmf_metrics=pmf_metrics,
            )
            sub_scores = scores_df_g.copy()
            if var_set and "variant" in sub_scores.columns:
                sub_scores = sub_scores[sub_scores["variant"].astype(str).isin(var_set)]

            if sub_scores.empty:
                err = apply_theme(error_fig("No variants for PCA scatter"), theme)
                err2 = apply_theme(error_fig("No loadings available"), theme)
                err3 = apply_theme(error_fig("No outlier data"), theme)
                return err, err2, err3, html.Div("No variants for PCA scatter."), html.Div()
            available_pcs = [c for c in sub_scores.columns if c.startswith("PC")]
            max_pc_idx = len(available_pcs)
            if max_pc_idx == 0:
                err = apply_theme(error_fig("Global PCA has no components"), theme)
                err2 = apply_theme(error_fig("No loadings available"), theme)
                err3 = apply_theme(error_fig("No outlier data"), theme)
                return err, err2, err3, html.Div("Global PCA has no components."), html.Div()
            pc_x = max(1, min(pc_x, max_pc_idx))
            pc_y = max(1, min(pc_y, max_pc_idx))
            pc_z = max(1, min(pc_z, max_pc_idx))

            pcx_name = f"PC{pc_x}"
            pcy_name = f"PC{pc_y}"
            pcz_name = f"PC{pc_z}"

            use_3d = (n_dims == 3) and (pc_z <= max_pc_idx)

            plot_df = sub_scores.copy()
            if color_col and color_col in df.columns:
                if "variant" in plot_df.columns and "variant" in df.columns:
                    color_map = (
                        df.set_index(df["variant"].astype(str))[color_col]
                        .to_dict()
                    )
                    plot_df[color_col] = plot_df["variant"].astype(str).map(color_map)
                else:
                    plot_df[color_col] = df[color_col].values[: len(plot_df)]

            pc_cols_for_out = [pcx_name, pcy_name]
            if use_3d:
                pc_cols_for_out.append(pcz_name)

            cluster_labels = None
            cluster_note = ""
            if kmeans_k >= 2 and not plot_df.empty and pc_cols_for_out:
                if HAS_SKLEARN:
                    try:
                        coords = plot_df[pc_cols_for_out].values
                        km = KMeans(n_clusters=kmeans_k, n_init=10)
                        cluster_labels = km.fit_predict(coords)
                        plot_df["cluster"] = cluster_labels
                        counts = np.bincount(cluster_labels)
                        cluster_note = "K-means (k={}): cluster sizes = {}".format(
                            kmeans_k, ", ".join(str(int(c)) for c in counts)
                        )
                    except Exception:
                        cluster_note = "K-means requested but failed to run."
                else:
                    cluster_note = "K-means requested but sklearn is not installed."

            title = f"Global PCA scatter (all features={int(loadings_g.shape[0])})"
            ev_pieces = []
            for idx, name in [(pc_x, pcx_name), (pc_y, pcy_name), (pc_z, pcz_name)]:
                if name in evr_g.index:
                    ev_pieces.append(f"{name} {evr_g[name]*100:.1f}%")
            if ev_pieces:
                title += " — " + ", ".join(ev_pieces)

            if use_3d:
                fig_scatter = px.scatter_3d(
                    plot_df,
                    x=pcx_name,
                    y=pcy_name,
                    z=pcz_name,
                    color=(
                        "cluster"
                        if (cluster_labels is not None and color_col is None)
                        else (color_col if color_col in plot_df.columns else None)
                    ),
                    hover_data=["variant"] if "variant" in plot_df.columns else None,
                    opacity=0.85,
                )
                fig_scatter.update_layout(
                    height=720,
                    title=title,
                    scene=dict(
                        xaxis_title=pcx_name,
                        yaxis_title=pcy_name,
                        zaxis_title=pcz_name,
                    ),
                    margin=dict(l=10, r=10, t=60, b=10),
                )
                fig_scatter.update_traces(marker=dict(size=7))
                _apply_readable_layout(fig_scatter, height=720, tight_margins=True)
            else:
                fig_scatter = px.scatter(
                    plot_df,
                    x=pcx_name,
                    y=pcy_name,
                    color=(
                        "cluster"
                        if (cluster_labels is not None and color_col is None)
                        else (color_col if color_col in plot_df.columns else None)
                    ),
                    hover_data=["variant"] if "variant" in plot_df.columns else None,
                    opacity=0.85,
                )
                fig_scatter.update_layout(
                    height=720,
                    title=title,
                    xaxis_title=pcx_name,
                    yaxis_title=pcy_name,
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1.0,
                    ),
                    margin=dict(l=80, r=25, t=60, b=70),
                )
                fig_scatter.update_traces(marker=dict(size=9))
                _half = _pca_scatter_square_range(plot_df, pcx_name, pcy_name)
                fig_scatter.update_layout(
                    xaxis=dict(range=[-_half, _half]),
                    yaxis=dict(range=[-_half, _half]),
                )
                _apply_readable_layout(fig_scatter, height=720, square_2d=True)

            # biplot
            if loadings_g.empty or pcx_name not in loadings_g.columns or pcy_name not in loadings_g.columns:
                fig_vec = apply_theme(error_fig("No loadings available for selected PCs"), theme)
            else:
                L = loadings_g[[pcx_name, pcy_name]].dropna()
                L["norm"] = np.sqrt(L[pcx_name] ** 2 + L[pcy_name] ** 2)
                n_vec_eff = min(n_vec, len(L))
                L = L.sort_values("norm", ascending=False).head(n_vec_eff)

                max_abs = float(
                    max(
                        L[pcx_name].abs().max(),
                        L[pcy_name].abs().max(),
                        1e-6,
                    )
                )
                limit = 1.1 * max_abs

                fig_vec = go.Figure()
                fig_vec.add_shape(
                    type="line",
                    x0=-limit,
                    y0=0,
                    x1=limit,
                    y1=0,
                    line=dict(color="rgba(150,150,150,0.6)", width=1),
                )
                fig_vec.add_shape(
                    type="line",
                    x0=0,
                    y0=-limit,
                    x1=0,
                    y1=limit,
                    line=dict(color="rgba(150,150,150,0.6)", width=1),
                )

                for feat, row in L.iterrows():
                    x2 = row[pcx_name]
                    y2 = row[pcy_name]
                    fig_vec.add_trace(
                        go.Scatter(
                            x=[0, x2],
                            y=[0, y2],
                            mode="lines+markers+text",
                            line=dict(width=2),
                            marker=dict(size=6),
                            textfont=dict(size=11),
                            text=[None, feat],
                            textposition="top center",
                            hovertemplate=(
                                f"{feat}<br>{pcx_name}={x2:.3f}<br>{pcy_name}={y2:.3f}<extra></extra>"
                            ),
                            showlegend=False,
                        )
                    )

                fig_vec.update_layout(
                    height=360,
                    margin=dict(l=70, r=25, t=60, b=65),
                    xaxis_title=pcx_name + " loading",
                    yaxis_title=pcy_name + " loading",
                    xaxis=dict(range=[-limit, limit], zeroline=True, zerolinewidth=1),
                    yaxis=dict(range=[-limit, limit], zeroline=True, zerolinewidth=1),
                    title="Global loadings (top |norm| features)",
                )
                _apply_readable_layout(fig_vec, height=360, square_2d=True)

            # outliers — scored relative to full training distribution, not the filtered subset
            out_scores = _compute_outlier_scores(scores_df_g, plot_df, pc_cols_for_out)
            if out_scores.empty:
                fig_out = apply_theme(error_fig("No outlier scores"), theme)
                out_text = "No outlier scores available."
            else:
                out_df = plot_df.copy()
                out_df["outlier_score"] = out_scores.values
                out_df = out_df.sort_values("outlier_score", ascending=True).reset_index(drop=True)
                out_df["idx"] = np.arange(len(out_df))

                fig_out = px.bar(
                    out_df,
                    x="idx",
                    y="outlier_score",
                    color=(
                        "cluster"
                        if ("cluster" in out_df.columns and kmeans_k >= 2)
                        else None
                    ),
                    labels={"idx": "sorted variants", "outlier_score": "outlier score"},
                )
                q95 = out_df["outlier_score"].quantile(0.95)
                q99 = out_df["outlier_score"].quantile(0.99)
                fig_out.add_hline(y=float(q95), line_width=1, line_dash="dash", line_color="orange")
                fig_out.add_hline(y=float(q99), line_width=1, line_dash="dot", line_color="red")
                fig_out.update_layout(
                    height=360,
                    margin=dict(l=80, r=25, t=55, b=60),
                    xaxis_title="Variants (sorted by score)",
                    yaxis_title="Outlier score (z² sum, ref=all)",
                    showlegend=("cluster" in out_df.columns and kmeans_k >= 2),
                )
                _apply_readable_layout(fig_out, height=360)
                fig_out.update_xaxes(showticklabels=False)
                n95 = int((out_df["outlier_score"] > q95).sum())
                n99 = int((out_df["outlier_score"] > q99).sum())
                out_text = (
                    f"Outliers above 95th percentile: {n95} / {len(out_df)}; "
                    f"above 99th percentile: {n99} / {len(out_df)}."
                )

            n_train = scores_df_g.shape[0]
            n_feat = loadings_g.shape[0]
            n_shown = plot_df.shape[0]
            ratio = float(n_train) / n_feat if n_feat > 0 else np.nan

            cum_2 = float(
                evr_g.get("PC1", 0.0) + evr_g.get("PC2", 0.0)
            ) if not evr_g.empty else 0.0
            cum_3 = cum_2 + float(evr_g.get("PC3", 0.0)) if "PC3" in evr_g.index else cum_2

            notes = []
            if n_feat > 0 and n_train < 2 * n_feat:
                notes.append("Samples < 2×features — PCA directions may be noisy.")
            if cum_2 < 0.3:
                notes.append("PC1+PC2 explain < 30% of variance — structure may be weak in first two PCs.")
            if cluster_note:
                notes.append(cluster_note)
            if preset_mode == PCA_PRESET_NATIVE:
                notes.append("Native preset: PCA uses scalar physical values; sequence descriptors are isolated in the sequence preset.")
            elif preset_mode == PCA_PRESET_BASIN:
                notes.append("Basin-geometry corrected preset: global PCA uses intrinsic PMF landscape metrics when available; legacy annotation fallback otherwise.")
            elif preset_mode == PCA_PRESET_PMF_SHAPE:
                notes.append("PMFs-themselves preset: global PCA uses sqrt(P_mass) PMF vectors; Euclidean geometry approximates Hellinger geometry.")
            elif preset_mode == PCA_PRESET_SEQUENCE:
                notes.append("Sequence preset: global PCA uses numeric seq_* letter-space descriptors.")

            diag_children = [
                html.Div(
                    f"Global PCA basis: {n_train} samples × {n_feat} features "
                    f"(samples/features ≈ {ratio:.2f})." if n_feat > 0 else
                    "Global PCA basis: insufficient features.",
                    style={"marginBottom": "2px"},
                ),
                html.Div(
                    f"Current scatter selection: {n_shown} variants.",
                    style={"marginBottom": "2px"},
                ),
                html.Div(
                    f"Cumulative variance: PC1+PC2 = {cum_2*100:.1f}%, "
                    f"PC1+PC2+PC3 = {cum_3*100:.1f}%",
                    style={"marginBottom": "2px"},
                ),
                html.Div(
                    out_text,
                    style={"marginBottom": "2px"},
                ),
            ]
            if notes:
                diag_children.append(
                    html.Div(
                        "Notes: " + " ".join(notes),
                        style={"color": "#d62728"},
                    )
                )

            cluster_children = _render_cluster_members(plot_df, kmeans_k)

            return apply_theme(fig_scatter, theme), apply_theme(fig_vec, theme), apply_theme(fig_out, theme), diag_children, cluster_children

        # ===== LOCAL PCA =====
        local_thermo_note = ""
        if preset_mode == PCA_PRESET_PMF_SHAPE:
            df_local, local_thermo_note = _build_pmf_shape_feature_frame(ctx, variant_subset=var_set if var_set else None, pmf_metrics=pmf_metrics)
            torsion_mode = []
            angle_encoding = []
        else:
            if not isinstance(df, pd.DataFrame) or df.empty:
                err = apply_theme(error_fig("No data for PCA"), theme)
                err2 = apply_theme(error_fig("No loadings available"), theme)
                err3 = apply_theme(error_fig("No outlier data"), theme)
                return err, err2, err3, html.Div("No data for PCA."), html.Div()
            if var_set and "variant" in df.columns:
                df_local = df[df["variant"].astype(str).isin(var_set)]
            else:
                df_local = df

        if df_local.empty:
            err = apply_theme(error_fig("No variants for PCA scatter"), theme)
            err2 = apply_theme(error_fig("No loadings available"), theme)
            err3 = apply_theme(error_fig("No outlier data"), theme)
            return err, err2, err3, html.Div("No variants for PCA scatter."), html.Div()
        cols = [
            c
            for c in df_local.columns
            if c != "variant" and pd.api.types.is_numeric_dtype(df_local[c])
        ]
        ann_cols = getattr(ctx, "_annotation_cols", set())
        if preset_mode == PCA_PRESET_SEQUENCE:
            cols = _sequence_pca_feature_cols(df_local, cols)
            local_thermo_note = "Sequence preset: using numeric seq_* letter-space descriptors."
            torsion_mode = []
            angle_encoding = []
        elif preset_mode == PCA_PRESET_BASIN:
            core_cols = _pmf_core_wide_cols(df_local, cols)
            cols = core_cols if core_cols else [c for c in cols if c in ann_cols]
            local_thermo_note = (
                "Basin-geometry corrected preset: using unit-safe intrinsic landscape metrics."
                if core_cols else
                "Basin-geometry corrected preset: no PMF core columns found; using legacy annotation columns."
            )
            torsion_mode = []
            angle_encoding = []
        elif preset_mode == PCA_PRESET_NATIVE:
            cols = filter_numeric_columns(cols)
            cols = [c for c in cols if not _is_sequence_feature_col(c)]
            if "yes" in (hide_tech_values or []):
                cols = [c for c in cols if not _is_technical(c)]
            # Native preset deliberately keeps PMF annotation / physical-coordinate columns.
            # Sequence descriptors live in their own preset.
        elif preset_mode != PCA_PRESET_PMF_SHAPE:
            cols = filter_numeric_columns(cols)
            if "yes" in (hide_tech_values or []):
                cols = [c for c in cols if not _is_technical(c)]
            if ann_cols:
                cols = [c for c in cols if c not in ann_cols]

        if not cols:
            err = apply_theme(error_fig("No numeric columns available for local PCA"), theme)
            err2 = apply_theme(error_fig("No loadings available"), theme)
            err3 = apply_theme(error_fig("No outlier data"), theme)
            return err, err2, err3, html.Div("No numeric columns available for local PCA."), html.Div()
        groups = _group_columns_by_prefix(cols)
        if group_sel and preset_mode not in {PCA_PRESET_BASIN, PCA_PRESET_PMF_SHAPE, PCA_PRESET_SEQUENCE}:
            allowed_cols: List[str] = []
            for g in group_sel:
                allowed_cols.extend(groups.get(g, []))
            cols = [c for c in cols if c in allowed_cols]
            if not cols:
                err = apply_theme(error_fig("No columns after group filter"), theme)
                err2 = apply_theme(error_fig("No loadings available"), theme)
                err3 = apply_theme(error_fig("No outlier data"), theme)
                return err, err2, err3, html.Div("No columns after group filter."), html.Div()

        raw_var = df_local[cols].var(ddof=1)
        df_num = _prepare_numeric_frame(df_local, cols)
        local_transform_note = ""
        if preset_mode == PCA_PRESET_BASIN and not df_num.empty and any(_is_pmf_core_wide_col(c) for c in df_num.columns):
            df_num, local_transform_note = _transform_wide_pmf_core(df_num)
            raw_var = df_num.var(ddof=1)
        if df_num.empty:
            err = apply_theme(error_fig("No usable columns after missingness filtering"), theme)
            err2 = apply_theme(error_fig("No loadings available"), theme)
            err3 = apply_theme(error_fig("No outlier data"), theme)
            return err, err2, err3, html.Div("No usable columns after missingness filtering."), html.Div()

        scores_rank = raw_var.reindex(df_num.columns).fillna(0.0)
        rank_method_loc = str(rank_method or "var").lower().strip()
        if rank_method_loc == "snr":
            try:
                mad = df_num.apply(lambda col: (col - col.median()).abs().median())
                snr = df_num.var(ddof=1) / (mad.replace(0, np.nan) + 1e-12)
                snr = snr.replace([np.inf, -np.inf], np.nan).dropna()
                if not snr.empty:
                    scores_rank = snr
            except Exception:
                pass
        scores_rank = scores_rank.replace([np.inf, -np.inf], np.nan).dropna().sort_values(ascending=False)

        k_int = int(var_k) if var_k is not None else 20
        k_int = max(1, min(k_int, len(scores_rank)))
        top = scores_rank.index[:k_int].tolist()

        try:
            drop_corr = "drop" in (drop_corr_flags or [])
        except Exception:
            drop_corr = False
        if drop_corr and len(top) >= 3:
            try:
                thr = float(corr_thr) if corr_thr is not None else 0.95
            except Exception:
                thr = 0.95
            thr = max(0.0, min(thr, 0.999))
            cm = df_num[top].corr().abs()
            kept_loc: List[str] = []
            for f in top:
                if all(float(cm.loc[f, kk]) <= thr for kk in kept_loc):
                    kept_loc.append(f)
            top = kept_loc

        X = df_num[top].to_numpy(float)
        good = top
        raw_n = df_local.shape[0]
        if not good or X.size == 0:
            err = apply_theme(error_fig("No usable data after variance filter"), theme)
            err2 = apply_theme(error_fig("No loadings available"), theme)
            err3 = apply_theme(error_fig("No outlier data"), theme)
            return err, err2, err3, html.Div("No usable data after variance filter."), html.Div()
        n_torsion_concepts = sum(1 for c in good if _is_torsion_feature_col_local(c))
        encode_on = "encode" in (angle_encoding or [])
        X, good, _angle_note = circular_encode_torsion_angles(X, good, enabled=encode_on, drop_original=True)
        if not good or X.size == 0:
            err = apply_theme(error_fig("No usable data after angle encoding"), theme)
            err2 = apply_theme(error_fig("No loadings available"), theme)
            err3 = apply_theme(error_fig("No outlier data"), theme)
            return err, err2, err3, html.Div("No usable data after angle encoding."), html.Div()

        Xz, _, _ = _zscore(X)
        Xz, good, _tors_note = _apply_torsion_handling(Xz, good, torsion_mode, n_torsion_concepts=n_torsion_concepts)
        n_samples = Xz.shape[0]
        n_features = len(good)
        ncomp = min(6, n_features)
        evr, comps, scores = _pca_numpy(Xz, n_components=ncomp)
        if scores.shape[1] < 2:
            err = error_fig("Local PCA has <2 components — cannot plot")
            err2 = error_fig("No loadings available")
            err3 = error_fig("No outlier data")
            return err, err2, err3, html.Div("Local PCA has fewer than 2 components."), html.Div()
        max_pc_idx = scores.shape[1]
        pc_x = max(1, min(pc_x, max_pc_idx))
        pc_y = max(1, min(pc_y, max_pc_idx))
        pc_z = max(1, min(pc_z, max_pc_idx))

        pcx_name = f"PC{pc_x}"
        pcy_name = f"PC{pc_y}"
        pcz_name = f"PC{pc_z}"

        use_3d = (n_dims == 3) and (pc_z <= max_pc_idx)

        pcs_cols = [f"PC{i+1}" for i in range(scores.shape[1])]
        scores_df = pd.DataFrame(scores, columns=pcs_cols)
        if "variant" in df_local.columns:
            scores_df["variant"] = df_local["variant"].to_numpy()[: scores_df.shape[0]]

        plot_df = scores_df.copy()
        if color_col and color_col in df.columns:
            if "variant" in plot_df.columns and "variant" in df.columns:
                color_map = (
                    df.set_index(df["variant"].astype(str))[color_col]
                    .to_dict()
                )
                plot_df[color_col] = plot_df["variant"].astype(str).map(color_map)
            else:
                plot_df[color_col] = df[color_col].values[: len(plot_df)]

        cluster_labels = None
        cluster_note = ""
        pc_cols_for_out = [pcx_name, pcy_name]
        if use_3d:
            pc_cols_for_out.append(pcz_name)

        if kmeans_k >= 2 and not plot_df.empty and pc_cols_for_out:
            if HAS_SKLEARN:
                try:
                    coords = plot_df[pc_cols_for_out].values
                    km = KMeans(n_clusters=kmeans_k, n_init=10)
                    cluster_labels = km.fit_predict(coords)
                    plot_df["cluster"] = cluster_labels
                    counts = np.bincount(cluster_labels)
                    cluster_note = "K-means (k={}): cluster sizes = {}".format(
                        kmeans_k, ", ".join(str(int(c)) for c in counts)
                    )
                except Exception:
                    cluster_note = "K-means requested but failed to run."
            else:
                cluster_note = "K-means requested but sklearn is not installed."

        title = f"Local PCA scatter (top-K features={int(n_features)})"
        ev_pieces = []
        for idx, name in [(pc_x, pcx_name), (pc_y, pcy_name), (pc_z, pcz_name)]:
            if 0 <= idx - 1 < len(evr):
                ev_pieces.append(f"{name} {evr[idx-1]*100:.1f}%")
        if ev_pieces:
            title += " — " + ", ".join(ev_pieces)

        if use_3d:
            fig_scatter = px.scatter_3d(
                plot_df,
                x=pcx_name,
                y=pcy_name,
                z=pcz_name,
                color=(
                    "cluster"
                    if (cluster_labels is not None and color_col is None)
                    else (color_col if color_col in plot_df.columns else None)
                ),
                hover_data=["variant"] if "variant" in plot_df.columns else None,
                opacity=0.85,
            )
            fig_scatter.update_layout(
                height=720,
                title=title,
                scene=dict(
                    xaxis_title=pcx_name,
                    yaxis_title=pcy_name,
                    zaxis_title=pcz_name,
                ),
                margin=dict(l=10, r=10, t=60, b=10),
            )
            fig_scatter.update_traces(marker=dict(size=5))
            _apply_readable_layout(fig_scatter, height=720, tight_margins=True)
        else:
            fig_scatter = px.scatter(
                plot_df,
                x=pcx_name,
                y=pcy_name,
                color=(
                    "cluster"
                    if (cluster_labels is not None and color_col is None)
                    else (color_col if color_col in plot_df.columns else None)
                ),
                hover_data=["variant"] if "variant" in plot_df.columns else None,
                opacity=0.85,
            )
            fig_scatter.update_layout(
                height=720,
                title=title,
                xaxis_title=pcx_name,
                yaxis_title=pcy_name,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1.0,
                ),
                margin=dict(l=80, r=25, t=60, b=70),
            )
            fig_scatter.update_traces(marker=dict(size=9))
            _half = _pca_scatter_square_range(plot_df, pcx_name, pcy_name)
            fig_scatter.update_layout(
                xaxis=dict(range=[-_half, _half]),
                yaxis=dict(range=[-_half, _half]),
            )
            _apply_readable_layout(fig_scatter, height=720, square_2d=True)

        # local biplot
        load_df = pd.DataFrame(
            comps[: scores.shape[1]].T,
            index=good,
            columns=[f"PC{i+1}" for i in range(scores.shape[1])],
        )
        if pcx_name not in load_df.columns or pcy_name not in load_df.columns:
            fig_vec = error_fig("No loadings available for selected PCs")
        else:
            L = load_df[[pcx_name, pcy_name]].dropna()
            L["norm"] = np.sqrt(L[pcx_name] ** 2 + L[pcy_name] ** 2)
            n_vec_eff = min(n_vec, len(L))
            L = L.sort_values("norm", ascending=False).head(n_vec_eff)

            max_abs = float(
                max(
                    L[pcx_name].abs().max(),
                    L[pcy_name].abs().max(),
                    1e-6,
                )
            )
            limit = 1.1 * max_abs

            fig_vec = go.Figure()
            fig_vec.add_shape(
                type="line",
                x0=-limit,
                y0=0,
                x1=limit,
                y1=0,
                line=dict(color="rgba(150,150,150,0.6)", width=1),
            )
            fig_vec.add_shape(
                type="line",
                x0=0,
                y0=-limit,
                x1=0,
                y1=limit,
                line=dict(color="rgba(150,150,150,0.6)", width=1),
            )

            for feat, row in L.iterrows():
                x2 = row[pcx_name]
                y2 = row[pcy_name]
                fig_vec.add_trace(
                    go.Scatter(
                        x=[0, x2],
                        y=[0, y2],
                        mode="lines+markers+text",
                        line=dict(width=2),
                        marker=dict(size=4),
                        textfont=dict(size=11),
                        text=[None, feat],
                        textposition="top center",
                        hovertemplate=(
                            f"{feat}<br>{pcx_name}={x2:.3f}<br>{pcy_name}={y2:.3f}<extra></extra>"
                        ),
                        showlegend=False,
                    )
                )

            fig_vec.update_layout(
                height=360,
                margin=dict(l=70, r=25, t=60, b=65),
                xaxis_title=pcx_name + " loading",
                yaxis_title=pcy_name + " loading",
                xaxis=dict(range=[-limit, limit], zeroline=True, zerolinewidth=1),
                yaxis=dict(range=[-limit, limit], zeroline=True, zerolinewidth=1),
                title="Local loadings (top |norm| features)",
            )
            _apply_readable_layout(fig_vec, height=360, square_2d=True)

        # local outliers — scored relative to the local training distribution
        out_scores = _compute_outlier_scores(scores_df, plot_df, pc_cols_for_out)
        if out_scores.empty:
            fig_out = error_fig("No outlier scores")
            out_text = "No outlier scores available."
        else:
            out_df = plot_df.copy()
            out_df["outlier_score"] = out_scores.values
            out_df = out_df.sort_values("outlier_score", ascending=True).reset_index(drop=True)
            out_df["idx"] = np.arange(len(out_df))

            fig_out = px.bar(
                out_df,
                x="idx",
                y="outlier_score",
                color=(
                    "cluster"
                    if ("cluster" in out_df.columns and kmeans_k >= 2)
                    else None
                ),
                labels={"idx": "sorted variants", "outlier_score": "outlier score"},
            )
            q95 = out_df["outlier_score"].quantile(0.95)
            q99 = out_df["outlier_score"].quantile(0.99)
            fig_out.add_hline(y=float(q95), line_width=1, line_dash="dash", line_color="orange")
            fig_out.add_hline(y=float(q99), line_width=1, line_dash="dot", line_color="red")
            fig_out.update_layout(
                height=360,
                margin=dict(l=80, r=25, t=55, b=60),
                xaxis_title="Variants (sorted by score)",
                yaxis_title="Outlier score (z² sum)",
                showlegend=("cluster" in out_df.columns and kmeans_k >= 2),
            )
            _apply_readable_layout(fig_out, height=360)
            fig_out.update_xaxes(showticklabels=False)
            n95 = int((out_df["outlier_score"] > q95).sum())
            n99 = int((out_df["outlier_score"] > q99).sum())
            out_text = (
                f"Outliers above 95th percentile: {n95} / {len(out_df)}; "
                f"above 99th percentile: {n99} / {len(out_df)}."
            )

        ratio = float(n_samples) / n_features if n_features > 0 else np.nan
        dropped = raw_n - n_samples
        frac_dropped = float(dropped) / raw_n if raw_n > 0 else 0.0

        cum_2 = float(evr[:2].sum()) if len(evr) >= 2 else float(evr[:1].sum())
        cum_3 = float(evr[:3].sum()) if len(evr) >= 3 else cum_2

        notes = []
        if n_features > 0 and n_samples < 2 * n_features:
            notes.append("Samples < 2×features — PCA directions may be noisy.")
        if cum_2 < 0.3:
            notes.append("PC1+PC2 explain < 30% of variance — structure may be weak in first two PCs.")
        if frac_dropped > 0.2:
            notes.append(f"More than 20% of rows dropped due to NaNs (≈ {frac_dropped*100:.1f}%).")
        if cluster_note:
            notes.append(cluster_note)
        if local_thermo_note:
            notes.append(local_thermo_note)
        if local_transform_note:
            notes.append(local_transform_note)

        diag_children = [
            html.Div(
                f"Local PCA basis: {n_samples} samples × {n_features} features "
                f"(samples/features ≈ {ratio:.2f})." if n_features > 0 else
                "Local PCA basis: insufficient features.",
                style={"marginBottom": "2px"},
            ),
            html.Div(
                f"Rows entering local PCA: {raw_n}; row drops after preprocessing: {dropped} "
                f"(≈ {frac_dropped*100:.1f}%). Remaining NaNs are median-imputed after column filtering.",
                style={"marginBottom": "2px"},
            ),
            html.Div(
                f"Cumulative variance: PC1+PC2 = {cum_2*100:.1f}%, "
                f"PC1+PC2+PC3 = {cum_3*100:.1f}%",
                style={"marginBottom": "2px"},
            ),
            html.Div(
                out_text,
                style={"marginBottom": "2px"},
            ),
        ]
        if notes:
            diag_children.append(
                html.Div(
                    "Notes: " + " ".join(notes),
                    style={"color": "#d62728"},
                )
            )

        cluster_children = _render_cluster_members(plot_df, kmeans_k)

        return fig_scatter, fig_vec, fig_out, diag_children, cluster_children

    # ---- stability: split-half / bootstrap ----
    @app.callback(
        Output("pca-stab-cos", "figure"),
        Output("pca-stab-proc", "figure"),
        Output("pca-stab-summary", "children"),
        Input("pca-stab-run", "n_clicks"),
        State("pca-stab-source", "value"),
        State("pca-stab-method", "value"),
        State("pca-stab-runs", "value"),
        State("pca-stab-frac", "value"),
        State("pca-stab-npcs", "value"),
        State("pca-stab-procscale", "value"),
        State("var-k", "value"),
        State("hide-tech", "value"),
        State("pca-feature-groups", "value"),
        State("pca-torsion-mode", "value"),
        State("pca-angle-encoding", "value"),
        State("pca-thermo-preset-active", "data"),
        State("pca-pmf-metrics", "value"),
        prevent_initial_call=True,
    )
    def _run_pca_stability(
        n_clicks,
        source,
        method,
        n_runs,
        sample_frac,
        n_pcs,
        proc_scale,
        k_top,
        hide_tech_values,
        group_sel,
        torsion_mode,
        angle_encoding,
        thermo_preset,
        pmf_metrics,
    ):
        preset_mode = _pca_preset_mode(thermo_preset)
        df = getattr(ctx, "df", df_initial)
        stab_transform_note = ""

        if preset_mode == PCA_PRESET_PMF_SHAPE:
            df, stab_transform_note = _build_pmf_shape_feature_frame(ctx, pmf_metrics=pmf_metrics)
            torsion_mode = []
            angle_encoding = []

        if not isinstance(df, pd.DataFrame) or df.empty:
            err = error_fig("No data")
            msg = stab_transform_note if stab_transform_note else "No data."
            return err, err, msg

        cols = [c for c in df.columns if c != "variant" and pd.api.types.is_numeric_dtype(df[c])]
        ann_cols = getattr(ctx, "_annotation_cols", set())
        if preset_mode == PCA_PRESET_SEQUENCE:
            cols = _sequence_pca_feature_cols(df, cols)
            stab_transform_note = "Sequence preset: using numeric seq_* letter-space descriptors."
            torsion_mode = []
            angle_encoding = []
        elif preset_mode == PCA_PRESET_BASIN:
            core_cols = _pmf_core_wide_cols(df, cols)
            cols = core_cols if core_cols else [c for c in cols if c in ann_cols]
            torsion_mode = []
            angle_encoding = []
        elif preset_mode == PCA_PRESET_NATIVE:
            cols = filter_numeric_columns(cols)
            cols = [c for c in cols if not _is_sequence_feature_col(c)]
            if "yes" in (hide_tech_values or []):
                cols = [c for c in cols if not _is_technical(c)]
            # Native preset deliberately keeps PMF annotation / physical-coordinate columns.
            # Sequence descriptors live in their own preset.
        elif preset_mode != PCA_PRESET_PMF_SHAPE:
            cols = filter_numeric_columns(cols)
            if "yes" in (hide_tech_values or []):
                cols = [c for c in cols if not _is_technical(c)]
            if ann_cols:
                cols = [c for c in cols if c not in ann_cols]
        if not cols:
            err = error_fig("No numeric columns")
            return err, err, "No numeric columns."

        groups = _group_columns_by_prefix(cols)
        if group_sel and preset_mode not in {PCA_PRESET_BASIN, PCA_PRESET_PMF_SHAPE, PCA_PRESET_SEQUENCE}:
            allowed: List[str] = []
            for g in group_sel:
                allowed.extend(groups.get(g, []))
            cols = [c for c in cols if c in allowed]
        if not cols:
            err = error_fig("No columns after group filter")
            return err, err, "No columns after group filter."

        if str(source or "global") == "local":
            var = df[cols].var(numeric_only=True).sort_values(ascending=False)
            k_int = int(k_top) if k_top is not None else 20
            k_int = max(2, min(k_int, len(var)))
            cols = var.index[:k_int].tolist()

        df_num_stab = _prepare_numeric_frame(df, cols)
        if preset_mode == PCA_PRESET_BASIN and not df_num_stab.empty and any(_is_pmf_core_wide_col(c) for c in df_num_stab.columns):
            df_num_stab, stab_transform_note = _transform_wide_pmf_core(df_num_stab)
        if df_num_stab.empty:
            X_raw, good = np.zeros((0, 0), float), []
        else:
            X_raw = df_num_stab.to_numpy(float)
            good = df_num_stab.columns.astype(str).tolist()
        if X_raw.size == 0 or not good:
            err = error_fig("No usable matrix")
            return err, err, "No usable matrix."

        encode_on = "encode" in (angle_encoding or [])
        X_raw, good, angle_note = circular_encode_torsion_angles(X_raw, good, enabled=encode_on, drop_original=True)

        Xz, good, miss_note = _filter_impute_and_zscore(X_raw, good, max_missing_frac=0.30)
        Xz, good, tors_note = _apply_torsion_handling_local(Xz, good, torsion_mode)

        n = Xz.shape[0]
        p = Xz.shape[1]
        if n < 6 or p < 2:
            err = error_fig("Too few samples/features")
            msg = f"Need >=6 variants and >=2 features; got {n}×{p}."
            return err, err, msg

        k = int(n_pcs or 4)
        k = max(2, min(k, 6, p, n - 1))

        ref_load = _fit_pca_loadings(Xz, k)
        ref_scores = Xz @ ref_load

        runs = int(n_runs or 60)
        runs = max(5, min(runs, 500))
        frac = float(sample_frac or 0.8)
        frac = min(1.0, max(0.2, frac))
        allow_scale = "scale" in (proc_scale or [])

        rng = np.random.default_rng(int(n_clicks or 1))

        rows_cos = []
        rows_proc = []

        for r in range(runs):
            if str(method or "split_half") == "bootstrap":
                m = int(max(3, round(frac * n)))
                idx = rng.integers(0, n, size=m)
                load = _fit_pca_loadings(Xz[idx, :], k)
                load_aligned, cos = _align_loadings(ref_load, load)
                scores = Xz @ load_aligned
                disp = _procrustes_disparity(ref_scores, scores, allow_scale=allow_scale)
                label = "vs_ref"
            else:
                perm = rng.permutation(n)
                half = n // 2
                idxA = perm[:half]
                idxB = perm[half : 2 * half]
                LA = _fit_pca_loadings(Xz[idxA, :], k)
                LB = _fit_pca_loadings(Xz[idxB, :], k)
                LB_aligned, cos = _align_loadings(LA, LB)
                scoresA = Xz @ LA
                scoresB = Xz @ LB_aligned
                disp = _procrustes_disparity(scoresA, scoresB, allow_scale=allow_scale)
                label = "A_vs_B"

            for i, cval in enumerate(cos, start=1):
                rows_cos.append({"run": r, "pc": f"PC{i}", "cosine": float(cval), "mode": label})
            rows_proc.append({"run": r, "disparity": float(disp), "mode": label})

        df_cos = pd.DataFrame(rows_cos)
        df_proc = pd.DataFrame(rows_proc)
        if df_cos.empty or df_proc.empty:
            err = error_fig("No stability results")
            return err, err, "No stability results."

        fig_cos = px.box(df_cos, x="pc", y="cosine")
        fig_cos.update_layout(
            title="Loading stability (|cosine|) across runs",
            height=360,
            margin=dict(l=60, r=20, t=55, b=60),
            yaxis_title="|cosine similarity| (aligned PCs)",
            xaxis_title="PC",
        )
        _apply_readable_layout(fig_cos, height=360)
        fig_cos.update_yaxes(range=[0, 1.05])

        fig_proc = px.histogram(df_proc, x="disparity", nbins=30)
        fig_proc.update_layout(
            title="Procrustes disparity of score-space geometry",
            height=360,
            margin=dict(l=60, r=20, t=55, b=60),
            xaxis_title="disparity (lower = more stable)",
            yaxis_title="count",
        )
        _apply_readable_layout(fig_proc, height=360)

        means = df_cos.groupby("pc")["cosine"].mean().to_dict()
        sds = df_cos.groupby("pc")["cosine"].std(ddof=1).fillna(0.0).to_dict()
        disp_mu = float(df_proc["disparity"].mean())
        disp_sd = float(df_proc["disparity"].std(ddof=1)) if len(df_proc) > 1 else 0.0

        summary = [
            html.Div(f"Matrix: {n} variants × {p} features; compared PCs: {k}."),
            html.Div(f"Method: {method}; runs: {runs}; bootstrap frac: {frac:.2f} (ignored for split-half)."),
            html.Div("Preproc: " + " ".join(x for x in [miss_note, stab_transform_note, angle_note, tors_note] if x).strip()),
            html.Div(
                "Loadings stability (mean±sd): "
                + ", ".join(
                    [
                        f"PC{i}={means.get(f'PC{i}', float('nan')):.2f}±{sds.get(f'PC{i}', 0.0):.2f}"
                        for i in range(1, k + 1)
                    ]
                )
            ),
            html.Div(f"Procrustes disparity: {disp_mu:.3g} ± {disp_sd:.3g} (allow_scale={allow_scale})."),
        ]

        return fig_cos, fig_proc, summary

    # ---- PMF curve viewer callbacks ----

    def _variant_list_text(variants):
        if not variants:
            return "No variants selected. Click points in the PCA scatter above."
        shown = ", ".join(str(v) for v in variants[:12])
        suffix = f" … +{len(variants) - 12} more" if len(variants) > 12 else ""
        return f"Selected ({len(variants)}): {shown}{suffix}"

    def _variant_from_point(pt: dict) -> Optional[str]:
        """Extract variant name from a Plotly clickData / selectedData point dict."""
        # Primary: customdata (set via hover_data=["variant"] in px.scatter)
        cd = pt.get("customdata")
        if cd is not None:
            if isinstance(cd, (list, tuple)) and len(cd) > 0 and cd[0] is not None:
                v = str(cd[0]).strip()
                if v:
                    return v
            elif not isinstance(cd, (list, tuple)):
                v = str(cd).strip()
                if v:
                    return v
        # Fallback: hovertext field (present when text= is set on the trace)
        for key in ("hovertext", "text"):
            raw = pt.get(key)
            if raw and str(raw).strip():
                return str(raw).strip()
        return None

    def _variants_from_points(pts) -> list:
        seen: dict = {}
        for pt in pts:
            v = _variant_from_point(pt)
            if v is not None:
                seen[v] = None  # preserve order, dedup
        return list(seen)

    @app.callback(
        Output("pca-curve-variants", "data"),
        Output("pca-curve-variant-display", "children"),
        Input("pca-scatter", "clickData"),
        Input("pca-scatter", "selectedData"),
        Input("pca-curve-clear", "n_clicks"),
        State("pca-curve-variants", "data"),
        prevent_initial_call=True,
    )
    def _accumulate_clicked_variant(click_data, selected_data, _n_clear, current):
        from dash import callback_context as _dcc_ctx
        triggered_ids = (
            {t["prop_id"] for t in _dcc_ctx.triggered}
            if _dcc_ctx.triggered else set()
        )
        if "pca-curve-clear.n_clicks" in triggered_ids:
            return [], _variant_list_text([])

        # Lasso / box select — only handle when selectedData has actual points
        if "pca-scatter.selectedData" in triggered_ids:
            pts = (selected_data or {}).get("points", [])
            if pts:
                lst = _variants_from_points(pts)
                return lst, _variant_list_text(lst)
            # selectedData fired but is empty (deselect event) — fall through to clickData

        # Single click — toggle one variant
        if "pca-scatter.clickData" in triggered_ids and click_data:
            pts = click_data.get("points", [])[:1]
            v = _variant_from_point(pts[0]) if pts else None
            if v is not None:
                lst = list(current or [])
                if v in lst:
                    lst.remove(v)
                else:
                    lst.append(v)
                return lst, _variant_list_text(lst)

        lst = list(current or [])
        return lst, _variant_list_text(lst)

    @app.callback(
        Output("global-selected-variants", "data", allow_duplicate=True),
        Input("pca-curve-variants", "data"),
        prevent_initial_call=True,
    )
    def _pca_write_global_variants(variants):
        return {"variants": list(variants or []), "source": "pca"}

    @app.callback(
        Output("pca-scatter", "figure", allow_duplicate=True),
        Input("global-selected-variants", "data"),
        State("pca-scatter", "figure"),
        prevent_initial_call=True,
    )
    def _pca_cross_highlight(global_sel, current_fig):
        if not global_sel or not isinstance(global_sel, dict):
            raise PreventUpdate
        if global_sel.get("source") == "pca":
            raise PreventUpdate
        variants = list(global_sel.get("variants") or [])
        if not variants or not current_fig:
            raise PreventUpdate
        try:
            fig = go.Figure(current_fig)
            # Remove any previous overlay before rebuilding so selections don't accumulate
            fig.data = tuple(t for t in fig.data if getattr(t, "name", None) != "cross-tab selection")
            variant_set = set(str(v) for v in variants)
            highlight_x, highlight_y, highlight_labels = [], [], []
            for trace in fig.data:
                cd = getattr(trace, "customdata", None)
                xs = getattr(trace, "x", None)
                ys = getattr(trace, "y", None)
                if cd is None or xs is None or ys is None:
                    continue
                for i, (x, y) in enumerate(zip(xs, ys)):
                    try:
                        point_cd = cd[i]
                        v = str(point_cd[0]).strip() if isinstance(point_cd, (list, tuple, np.ndarray)) and len(point_cd) > 0 else str(point_cd).strip()
                        if v in variant_set:
                            highlight_x.append(x)
                            highlight_y.append(y)
                            highlight_labels.append(v)
                    except Exception:
                        continue
            if not highlight_x:
                raise PreventUpdate
            fig.add_trace(go.Scatter(
                x=highlight_x, y=highlight_y,
                mode="markers",
                name="cross-tab selection",
                marker=dict(symbol="circle-open", size=16, color="crimson", line=dict(width=2)),
                text=highlight_labels,
                hovertemplate="%{text}<extra>cross-tab selection</extra>",
                showlegend=True,
            ))
            return fig
        except PreventUpdate:
            raise
        except Exception:
            raise PreventUpdate

    def _load_pmf_targeted(variants: list, metric: Optional[str] = None) -> pd.DataFrame:
        """Load selected PMF curves via the shared cached variant-PMF store."""
        try:
            max_pts = int(os.environ.get("PEPTIDE_DASH_PMF_VIEWER_MAX_POINTS", "1200"))
            return load_variant_pmfs(
                ctx,
                variants,
                metric=metric,
                columns=PMF_PLOT_COLS,
                max_points_per_curve=max_pts,
            )
        except Exception:
            return pd.DataFrame()


    def _pmf_loader_status(df: pd.DataFrame, variants: list, metric: Optional[str]) -> str:
        """Compact post-load diagnostics for the PMF curve viewer."""
        if not isinstance(df, pd.DataFrame) or df.empty:
            return f"No PMF curves loaded for metric={metric!r}; requested variants={len(variants or [])}."
        backend = df.attrs.get("pmf_loader_backend", "unknown")
        workers = df.attrs.get("pmf_loader_workers", "?")
        max_pts = os.environ.get("PEPTIDE_DASH_PMF_VIEWER_MAX_POINTS", "1200")
        if "variant" in df.columns and "metric" in df.columns:
            n_curves = int(df[["variant", "metric"]].drop_duplicates().shape[0])
        elif "variant" in df.columns:
            n_curves = int(df["variant"].nunique())
        else:
            n_curves = 1
        return (
            f"Loaded {len(df):,} plotted point(s) across {n_curves} curve(s); "
            f"backend={backend}, workers={workers}, max_points_per_curve={max_pts}."
        )


    @app.callback(
        Output("pca-curve-metric", "options"),
        Output("pca-curve-metric", "value"),
        Input("pca-curve-variants", "data"),
        prevent_initial_call=True,
    )
    def _populate_curve_metrics(variants):
        if not variants:
            return [], None
        try:
            metrics = available_pmf_metrics(ctx, variants=variants)
        except Exception:
            metrics = []
        if not metrics:
            return [{"label": "⚠ PMF data not found", "value": ""}], None
        opts = [{"label": m, "value": m} for m in metrics]
        return opts, metrics[0]


    @app.callback(
        Output("pca-curve-overlay", "figure"),
        Output("pca-curve-loader-status", "children"),
        Input("pca-curve-metric", "value"),
        Input("pca-curve-variants", "data"),
        prevent_initial_call=True,
    )
    def _update_curve_overlay(metric, variants):
        if not variants:
            return error_fig("Click variants in the PCA scatter to overlay curves."), "PMF viewer idle. Select variants and a metric to load curves."
        if not metric or str(metric).startswith("⚠"):
            return error_fig("Select a metric from the dropdown."), "Waiting for a valid PMF metric."
        d = _load_pmf_targeted(variants, metric=metric)
        status = _pmf_loader_status(d, list(variants or []), metric)
        if d.empty:
            return error_fig("No PMF parquet files found for selected variants."), status
        if "metric" in d.columns:
            d = d[d["metric"] == metric]
        if d.empty:
            return error_fig(f"No PMF data for metric {metric!r} in selected variants."), status
        y_col = next(
            (c for c in ("F_kJ_mol", "pmf_F_kJmol", "F", "y") if c in d.columns), None
        )
        if y_col is None:
            return error_fig("Cannot identify PMF F column."), status
        d = d.sort_values("x")
        fig = px.line(d, x="x", y=y_col, color="variant", markers=False, title=metric)

        # Collect the per-trace colors assigned by Plotly so basin markers match
        color_seq = px.colors.qualitative.Plotly
        var_order = list(dict.fromkeys(d["variant"].astype(str)))
        var_color = {v: color_seq[i % len(color_seq)] for i, v in enumerate(var_order)}

        # Curve minima stars
        for var, grp in d.groupby("variant", observed=True, sort=False):
            grp = grp.dropna(subset=[y_col])
            if grp.empty:
                continue
            row = grp.loc[grp[y_col].idxmin()]
            fig.add_scatter(
                x=[row["x"]],
                y=[row[y_col]],
                mode="markers",
                marker=dict(size=14, symbol="star", color=var_color.get(str(var), "#888"),
                            line=dict(width=1, color="black")),
                showlegend=False,
                hovertemplate=(
                    f"<b>{var}</b><br>"
                    f"{metric} = {float(row['x']):.4g}<br>"
                    f"F = {float(row[y_col]):.2f} kJ/mol<extra></extra>"
                ),
            )

        # Basin positions from pmf_annotations (stored per-metric as {metric}__{suffix})
        feat_df = getattr(ctx, "df", pd.DataFrame())
        if not feat_df.empty and "variant" in feat_df.columns:
            gcol = f"{metric}__global_basin_min_x"
            scol = f"{metric}__secondary_basin_min_x"
            for var in variants:
                row = feat_df[feat_df["variant"].astype(str) == str(var)]
                if row.empty:
                    continue
                clr = var_color.get(str(var), "#888")
                if gcol in feat_df.columns:
                    gx = float(row[gcol].iloc[0])
                    if np.isfinite(gx):
                        fig.add_vline(
                            x=gx,
                            line=dict(color=clr, width=1.5, dash="dash"),
                            annotation_text=f"{var} G",
                            annotation_font_size=10,
                            annotation_position="top right",
                        )
                if scol in feat_df.columns:
                    sx = float(row[scol].iloc[0])
                    if np.isfinite(sx):
                        fig.add_vline(
                            x=sx,
                            line=dict(color=clr, width=1.0, dash="dot"),
                            annotation_text=f"{var} S",
                            annotation_font_size=10,
                            annotation_position="top left",
                        )

        fig.update_layout(
            title=f"PMF — {metric}",
            xaxis_title=metric,
            yaxis_title="F (kJ/mol)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0),
            margin=dict(l=70, r=20, t=55, b=60),
        )
        _apply_readable_layout(fig, height=480)
        return fig, status
