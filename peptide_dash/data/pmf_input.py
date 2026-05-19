from __future__ import annotations

"""
PMF metrics input helper for peptide_dash.

Canonical column names match what 3_BATCH_ANA_replica_curves.py actually writes
to the SQLite summary table.  Where the spec names differ from BATCH_ANA names,
the BATCH_ANA name is authoritative; the spec alias is noted in a comment.

Observable identity: in the summary table the observable / coordinate name lives
in the ``metric`` column.  Pass the summary table directly; ``build_tab_input``
treats ``metric`` as ``observable_name``.

Missing-value convention:
  BATCH_ANA sets max_secondary_persistence_kT = NaN (not 0.0) when no secondary
  basin exists.  Spec §9 recommends 0.0; BATCH_ANA's NaN is more honest because
  0.0 would be indistinguishable from a detected barrier of height 0.  The
  transforms propagate NaN correctly.  Callers that need 0-imputation should
  fill explicitly before calling build_tab_input.
"""

import warnings as _warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 1. Feature groups  (BATCH_ANA actual column names)
# ---------------------------------------------------------------------------

# spec: n_basins_persist_gt_2kT → BATCH_ANA: n_basins_persist_2kT
# spec: basin_population_entropy_norm → BATCH_ANA: basin_pop_entropy_norm
# spec: global_basin_sigma_norm — NOT produced by BATCH_ANA; no proxy included
#   because global_basin_width_1kT is in native coordinate units (nm, deg, …)
#   and cannot be mixed across observables without unit normalisation.
PMF_CORE_FEATURES: List[str] = [
    "effective_support_frac",
    "n_basins_persist_2kT",          # spec: n_basins_persist_gt_2kT
    "global_basin_population",
    "global_basin_escape_barrier_kT",
    "max_secondary_persistence_kT",
    "basin_pop_entropy_norm",         # spec: basin_population_entropy_norm
    "local_ruggedness_kT",
]

# Reference-comparison features — not yet produced by BATCH_ANA; kept as
# forward-compat placeholders.  build_tab_input warns when they are absent.
PMF_REFERENCE_FEATURES: List[str] = [
    "jsd_to_reference_norm",
    "harmonic_hellinger_to_reference",
]

# Physical-coordinate annotations — valid science but unsafe for mixed-observable
# PCA/UMAP.  All values are in native coordinate units (nm, degrees, …).
# global_basin_width_1kT is here (not in PMF_CORE_FEATURES) because width in nm
# and width in degrees are not comparable across observables.
PMF_PHYSICAL_ANNOTATIONS: List[str] = [
    "global_basin_min_x",
    "global_basin_left_boundary_x",
    "global_basin_right_boundary_x",
    "global_basin_width_1kT",
    "secondary_basin_min_x",
    "secondary_basin_left_boundary_x",
    "secondary_basin_right_boundary_x",
    "secondary_basin_population",
    "secondary_basin_persistence_kT",
    "x_eqpop",
    "DeltaF_lr_at_median_kT",
    "barrier_12_kT",
    "DeltaG_state2_vs_state1_kT",
    "auc_Fcum_left_kT",
    "auc_Fcum_right_kT",
]

# Metadata columns — not yet produced by BATCH_ANA in summary table.
# ``metric`` serves as observable_name; other fields are absent.
PMF_METADATA_COLUMNS: List[str] = [
    "observable_name",        # → summary.metric
    "observable_type",
    "coordinate_units",
    "is_periodic",
    "period",
    "pmf_n_bins",
    "pmf_bin_min",
    "pmf_bin_max",
    "pmf_bin_width",
    "pmf_temperature_K",
    "pmf_input_type",
    "pmf_reference_id",
    "pmf_reference_observable",
    "pmf_reference_support_hash",
    "pmf_support_hash",
    "pmf_feature_schema_version",
]

# Quality / validity flags — not yet produced by BATCH_ANA.
PMF_QUALITY_COLUMNS: List[str] = [
    "pmf_has_valid_probability",
    "pmf_has_enough_bins",
    "pmf_has_finite_energy",
    "pmf_has_detectable_basin",
    "pmf_is_multimodal",
    "pmf_harmonic_fit_valid",
    "pmf_jsd_reference_valid",
    "pmf_support_matches_reference",
    "global_basin_touches_boundary",
    "secondary_basin_touches_boundary",
    "barrier_is_boundary_censored",
]

# Legacy wide-format stat columns (designed for wide pivot: metric__stat columns).
# Do NOT use with long-format summary data across multiple observables — mean in nm
# and mean in degrees are not comparable.
LEGACY_STATS_FEATURES: List[str] = [
    "mean", "std", "median", "mad", "iqr", "skew", "kurt_fisher",
    "min", "max", "n",
    "circular_mean_deg", "circular_std_deg", "circular_median_deg",
    "x_eqpop", "DeltaF_lr_at_median_kT",
]

# ---------------------------------------------------------------------------
# 2. Feature presets
# ---------------------------------------------------------------------------

# All list values are copies so that external mutation of PMF_CORE_FEATURES
# etc. does not silently corrupt the preset.
FEATURE_PRESETS: Dict[str, List[str]] = {
    "legacy_stats":             list(LEGACY_STATS_FEATURES),
    "pmf_intrinsic":            list(PMF_CORE_FEATURES),
    "pmf_reference":            list(PMF_CORE_FEATURES) + list(PMF_REFERENCE_FEATURES),
    "pmf_annotations":          list(PMF_PHYSICAL_ANNOTATIONS),
    "pmf_all_numeric_analysis": list(PMF_CORE_FEATURES) + list(PMF_REFERENCE_FEATURES),
}

# Presets whose features are exclusively physical-coordinate annotations.
# build_tab_input bypasses the default embedding exclusion for these.
_ANNOTATION_ONLY_PRESETS: frozenset[str] = frozenset({"pmf_annotations"})

# ---------------------------------------------------------------------------
# 3. Feature roles
# ---------------------------------------------------------------------------

FEATURE_ROLES: Dict[str, str] = {
    "effective_support_frac": "analysis",
    "n_basins_persist_2kT": "analysis",
    "global_basin_population": "analysis",
    "global_basin_escape_barrier_kT": "analysis",
    "max_secondary_persistence_kT": "analysis",
    "basin_pop_entropy_norm": "analysis",
    "local_ruggedness_kT": "analysis",
    "jsd_to_reference_norm": "reference_analysis",
    "harmonic_hellinger_to_reference": "reference_analysis",
    # Physical annotations — coordinate-bearing, native units
    "global_basin_min_x": "physical_annotation",
    "global_basin_left_boundary_x": "physical_annotation",
    "global_basin_right_boundary_x": "physical_annotation",
    "global_basin_width_1kT": "physical_annotation",
    "secondary_basin_min_x": "physical_annotation",
    "secondary_basin_left_boundary_x": "physical_annotation",
    "secondary_basin_right_boundary_x": "physical_annotation",
    "secondary_basin_population": "physical_annotation",
    "secondary_basin_persistence_kT": "physical_annotation",
    "x_eqpop": "physical_annotation",
    "DeltaF_lr_at_median_kT": "physical_annotation",
    "barrier_12_kT": "physical_annotation",
    "DeltaG_state2_vs_state1_kT": "physical_annotation",
    "auc_Fcum_left_kT": "physical_annotation",
    "auc_Fcum_right_kT": "physical_annotation",
}

# ---------------------------------------------------------------------------
# 4. Exclusions from embeddings by default
# ---------------------------------------------------------------------------

EXCLUDE_FROM_EMBEDDING_BY_DEFAULT: List[str] = (
    PMF_PHYSICAL_ANNOTATIONS + PMF_METADATA_COLUMNS + PMF_QUALITY_COLUMNS
)

# ---------------------------------------------------------------------------
# 5. Transform rules
# ---------------------------------------------------------------------------

PMF_TRANSFORM_RULES: Dict[str, str] = {
    # Bounded/fractional — z-score centres and scales; unit-safe across observables.
    "effective_support_frac": "zscore",
    "global_basin_population": "zscore",
    "basin_pop_entropy_norm": "zscore",
    "jsd_to_reference_norm": "zscore",
    "harmonic_hellinger_to_reference": "zscore",
    # Heavy-tailed positive kT quantities — log1p variance-stabilises before z-score.
    # NaN (absent barrier / no secondary basin) propagates as NaN through log1p.
    "global_basin_escape_barrier_kT": "log1p_zscore",
    "max_secondary_persistence_kT": "log1p_zscore",
    "local_ruggedness_kT": "log1p_zscore",
    # Count — sqrt variance-stabilises Poisson-ish counts before z-score.
    # n_basins_persist_2kT >= 1 (global always counted); NaN → no basin detected.
    "n_basins_persist_2kT": "sqrt_zscore",
}

# ---------------------------------------------------------------------------
# 6. Validity filter presets
# ---------------------------------------------------------------------------

_REQUIRED_INTRINSIC: List[str] = [
    "pmf_has_valid_probability",
    "pmf_has_enough_bins",
    "pmf_has_finite_energy",
    "pmf_has_detectable_basin",
]

_REQUIRED_REFERENCE: List[str] = _REQUIRED_INTRINSIC + [
    "pmf_jsd_reference_valid",
    "pmf_support_matches_reference",
]

# Threshold: warn when a feature column has more than this fraction NaN.
_NAN_WARN_THRESHOLD: float = 0.5

# Features where high NaN fractions are scientifically expected (secondary basin
# absent → NaN is correct, not a data quality problem).  Use a looser threshold.
_EXPECTED_SPARSE_FEATURES: frozenset[str] = frozenset({
    "max_secondary_persistence_kT",
    "secondary_basin_min_x",
    "secondary_basin_left_boundary_x",
    "secondary_basin_right_boundary_x",
    "secondary_basin_population",
    "secondary_basin_persistence_kT",
})
_NAN_WARN_THRESHOLD_SPARSE: float = 0.90

# ---------------------------------------------------------------------------
# 7. Core helpers
# ---------------------------------------------------------------------------

def can_include_physical_coordinates(df: pd.DataFrame) -> bool:
    """
    True only when all rows share the same observable and coordinate units.

    Uses the ``metric`` column as the observable name (BATCH_ANA's name for
    the coordinate / observable being measured).
    """
    obs_col = "observable_name" if "observable_name" in df.columns else "metric"
    unit_col = "coordinate_units"

    if obs_col not in df.columns:
        return False

    obs_ok = df[obs_col].dropna().nunique() == 1
    if not obs_ok:
        return False

    if unit_col not in df.columns:
        # Units unknown but observable is uniform — permit with caution.
        return True

    return df[unit_col].dropna().nunique() <= 1


def resolve_feature_preset(
    preset: str,
    df_cols: Optional[Sequence[str]] = None,
) -> Tuple[List[str], List[str]]:
    """
    Return (present_features, missing_features).

    ``missing_features`` lists names that are in the preset but absent from
    ``df_cols``.  Callers should emit a warning for each missing name.
    """
    features = FEATURE_PRESETS.get(preset, [])
    if df_cols is None:
        return list(features), []

    cols_set = set(df_cols)
    present = [f for f in features if f in cols_set]
    missing = [f for f in features if f not in cols_set]
    return present, missing


def apply_pmf_validity_filters(
    df: pd.DataFrame,
    preset: str,
    strict: bool = True,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Return (filtered_df, warnings).

    Missing quality-flag columns are skipped with a warning, not an error.
    If no quality columns are present at all, returns df unchanged.
    """
    warns: List[str] = []
    if not strict:
        return df, warns

    if preset == "pmf_reference":
        required = _REQUIRED_REFERENCE
    elif preset in {"pmf_intrinsic", "pmf_all_numeric_analysis"}:
        required = _REQUIRED_INTRINSIC
    else:
        return df, warns

    existing = [c for c in required if c in df.columns]
    absent = [c for c in required if c not in df.columns]

    if absent:
        warns.append(
            f"Quality flags not present in data (not yet produced by BATCH_ANA): "
            f"{absent}.  Validity filter skipped for these flags."
        )

    if not existing:
        return df, warns

    mask = df[existing].fillna(False).all(axis=1)
    n_dropped = int((~mask).sum())
    if n_dropped:
        warns.append(f"Dropped {n_dropped} rows failing validity flags: {existing}.")

    return df.loc[mask].copy(), warns


def remove_default_exclusions(
    features: List[str],
    df: pd.DataFrame,
    allow_physical_coordinates: bool = False,
    preset: str = "",
) -> Tuple[List[str], List[str]]:
    """
    Return (kept_features, warns).

    Physical-coordinate annotations are excluded from embedding features by
    default.  Pass ``allow_physical_coordinates=True`` AND ensure the selection
    contains a single observable/unit combination (checked via
    ``can_include_physical_coordinates``) to permit them.

    The ``pmf_annotations`` preset bypasses this exclusion automatically,
    because it is exclusively composed of physical annotations and is designed
    for tooltip/detail use, not embedding.
    """
    warns: List[str] = []

    # pmf_annotations preset is intrinsically physical — skip the embedding guard.
    if preset in _ANNOTATION_ONLY_PRESETS:
        excl_set = set(PMF_METADATA_COLUMNS + PMF_QUALITY_COLUMNS)
    elif not allow_physical_coordinates:
        excl_set = set(
            PMF_PHYSICAL_ANNOTATIONS + PMF_METADATA_COLUMNS + PMF_QUALITY_COLUMNS
        )
    else:
        excl_set = set(PMF_METADATA_COLUMNS + PMF_QUALITY_COLUMNS)

    removed = [f for f in features if f in excl_set]
    kept = [f for f in features if f not in excl_set]

    if removed and preset not in _ANNOTATION_ONLY_PRESETS and not allow_physical_coordinates:
        warns.append(
            f"Removed {len(removed)} physical/metadata columns from embedding features "
            f"(allow_physical_coordinates=False): {removed}."
        )

    # For annotation-only presets: always warn when multiple observables/units
    # are present — the physical coordinates will be returned but are not
    # comparable across observables.
    if preset in _ANNOTATION_ONLY_PRESETS:
        if not can_include_physical_coordinates(df):
            warns.append(
                "Returning physical coordinate annotations across multiple observables "
                "or coordinate units.  Values are in native units (nm, degrees, …) and "
                "are not comparable across different observables."
            )
        return kept, warns

    # For embedding presets with allow_physical_coordinates=True: strip physical
    # annotations when the selection is multi-observable or multi-unit.
    if allow_physical_coordinates:
        if not can_include_physical_coordinates(df):
            coord_feats = [f for f in kept if f in set(PMF_PHYSICAL_ANNOTATIONS)]
            if coord_feats:
                kept = [f for f in kept if f not in set(PMF_PHYSICAL_ANNOTATIONS)]
                warns.append(
                    "Removed physical coordinate annotations from embedding because "
                    "multiple observables or coordinate units are present: "
                    f"{coord_feats}."
                )

    return kept, warns


# ---------------------------------------------------------------------------
# 8. Transform
# ---------------------------------------------------------------------------

def _zscore_safe(s: np.ndarray) -> np.ndarray:
    """
    Z-score with NaN-aware mean/std.

    NaN / non-finite values are excluded from mean/std estimation and
    preserved as NaN in the output — they must not silently become 0.

    When there are no finite values (all-NaN column): return all-NaN.
    When all non-NaN values are identical (sd == 0): finite inputs become 0.0,
    NaN inputs stay NaN.
    """
    finite_mask = np.isfinite(s)
    n_finite = int(finite_mask.sum())
    out = np.full_like(s, np.nan, dtype=float)
    if n_finite == 0:
        return out  # all NaN → all NaN
    mu = float(np.mean(s[finite_mask]))
    sd = float(np.std(s[finite_mask], ddof=1)) if n_finite > 1 else 0.0
    if sd == 0.0:
        out[finite_mask] = 0.0
    else:
        out[finite_mask] = (s[finite_mask] - mu) / sd
    return out


def _apply_rule(vals: np.ndarray, rule: str) -> np.ndarray:
    """Apply a single named transform rule to a 1-D float array."""
    if rule == "zscore":
        return _zscore_safe(vals)
    if rule == "log1p_zscore":
        clipped = np.where(np.isfinite(vals), np.clip(vals, 0.0, None), np.nan)
        return _zscore_safe(np.log1p(clipped))
    if rule == "sqrt_zscore":
        clipped = np.where(np.isfinite(vals), np.clip(vals, 0.0, None), np.nan)
        return _zscore_safe(np.sqrt(clipped))
    return vals  # "none" — pass through


def transform_features(
    X_raw: pd.DataFrame,
    rules: Dict[str, str],
    scaler_scope: str = "current_selection",
    groups: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Apply per-column transform rules.

    Returns (X_transformed, transform_metadata).

    Rules:
      "zscore"        – NaN-aware z-score (ddof=1).
      "log1p_zscore"  – clip to [0, ∞), log1p, then z-score.
                        NaN propagates (absent barrier stays absent).
      "sqrt_zscore"   – clip to [0, ∞), sqrt, then z-score.
                        NaN propagates.
      anything else   – pass through unchanged, recorded as "none".

    ``scaler_scope``:
      "current_selection" – z-score across all rows (default).
      "per_observable"    – z-score within each group defined by ``groups``
                            (a Series aligned with X_raw, typically metric/observable).
                            Falls back to "current_selection" with a warning if
                            ``groups`` is None or misaligned.

    Clipping to [0, ∞) before log1p/sqrt is safe for all current PMF metrics:
      - escape_barrier_kT, persistence_kT, ruggedness_kT cannot be negative.
      - Any rare negative value from floating-point noise is clipped to 0, not
        silently set to NaN, because 0 is a meaningful floor for these metrics.
    NaN is preserved through np.clip so absent values stay absent.
    """
    X = X_raw.copy().astype(float)

    effective_scope = scaler_scope
    if scaler_scope == "per_observable":
        if groups is None or len(groups) != len(X_raw):
            _warnings.warn(
                "scaler_scope='per_observable' requires a groups Series aligned with "
                "X_raw rows.  Falling back to 'current_selection'.",
                stacklevel=2,
            )
            effective_scope = "current_selection"
        else:
            groups = groups.reset_index(drop=True)
    elif scaler_scope != "current_selection":
        _warnings.warn(
            f"Unknown scaler_scope={scaler_scope!r}; using 'current_selection'.",
            stacklevel=2,
        )
        effective_scope = "current_selection"

    meta: Dict[str, Any] = {"scaler_scope": effective_scope, "columns": {}}

    if effective_scope == "per_observable":
        grp_labels = groups.to_numpy()
        unique_grps = [g for g in pd.unique(grp_labels) if g is not None and not (isinstance(g, float) and np.isnan(g))]
        grp_idx: Dict[Any, np.ndarray] = {
            g: np.where(grp_labels == g)[0] for g in unique_grps
        }
        for col in X.columns:
            rule = rules.get(str(col), "none")
            out = X[col].to_numpy(dtype=float)
            for idx in grp_idx.values():
                out[idx] = _apply_rule(out[idx], rule)
            X[col] = out
            meta["columns"][col] = rule if rule in ("zscore", "log1p_zscore", "sqrt_zscore") else "none"
    else:
        for col in X.columns:
            rule = rules.get(str(col), "none")
            vals = X[col].to_numpy(dtype=float)
            X[col] = _apply_rule(vals, rule)
            meta["columns"][col] = rule if rule in ("zscore", "log1p_zscore", "sqrt_zscore") else "none"

    return X, meta


# ---------------------------------------------------------------------------
# 9. Reference feature guard
# ---------------------------------------------------------------------------

def _check_reference_validity(
    df: pd.DataFrame,
    features: List[str],
) -> Tuple[pd.DataFrame, List[str], List[str]]:
    """
    Drop rows where reference comparison is known to be invalid.

    Returns (filtered_df, filtered_features, warns).

    If the guard columns (pmf_jsd_reference_valid, pmf_support_matches_reference)
    are absent — which is the current BATCH_ANA state — no rows are removed and
    a warning is emitted explaining that no compatibility check was possible.

    When guard columns ARE present and some rows fail them, those rows are
    removed from df and from X so that invalid JSD values never reach the
    embedding.  Reference features are also removed from the feature list if
    every surviving row still has NaN for those columns.
    """
    warns: List[str] = []
    ref_requested = [f for f in features if f in set(PMF_REFERENCE_FEATURES)]
    if not ref_requested:
        return df, features, warns

    valid_flag = "pmf_jsd_reference_valid"
    support_flag = "pmf_support_matches_reference"

    missing_guards = [c for c in (valid_flag, support_flag) if c not in df.columns]
    if missing_guards:
        warns.append(
            f"Reference features {ref_requested} requested but guard columns "
            f"{missing_guards} are absent (not yet produced by BATCH_ANA).  "
            "No reference-validity row filter applied."
        )
        # Drop reference features that are entirely NaN — they degrade embeddings
        # as fully dead dimensions even without row filtering.
        ref_in_df = [f for f in ref_requested if f in df.columns]
        if ref_in_df:
            all_nan = df[ref_in_df].isna().all(axis=0)
            drop_feats = [f for f in ref_in_df if all_nan[f]]
            if drop_feats:
                features = [f for f in features if f not in drop_feats]
                warns.append(
                    f"Removed all-NaN reference features {drop_feats} "
                    "(not yet produced by BATCH_ANA; no valid values in data)."
                )
        return df, features, warns

    valid_mask = df[[valid_flag, support_flag]].fillna(False).all(axis=1)
    n_invalid = int((~valid_mask).sum())
    if n_invalid:
        warns.append(
            f"Removed {n_invalid} rows where reference comparison is invalid "
            f"({valid_flag} or {support_flag} is False or NaN)."
        )
        df = df.loc[valid_mask].copy()

    # If all remaining rows have NaN for reference features, drop from feature list.
    after_ref_cols = [f for f in ref_requested if f in df.columns]
    if after_ref_cols:
        all_nan = df[after_ref_cols].isna().all(axis=0)
        drop_feats = [f for f in after_ref_cols if all_nan[f]]
        if drop_feats:
            features = [f for f in features if f not in drop_feats]
            warns.append(
                f"Removed reference features {drop_feats} — all values are NaN "
                "after validity filtering."
            )

    return df, features, warns


# ---------------------------------------------------------------------------
# 10. Row-ID preservation helper
# ---------------------------------------------------------------------------

_ROW_ID_CANDIDATES: List[str] = [
    "variant", "metric",           # always present in summary table
    "variant_id", "observable_name",
    "replicate_id", "replica_id",
    "batch_id", "pmf_id", "pmf_reference_id",
]


def _collect_row_ids(df: pd.DataFrame) -> pd.DataFrame:
    present = [c for c in _ROW_ID_CANDIDATES if c in df.columns]
    return df[present].copy() if present else pd.DataFrame(index=df.index)


# ---------------------------------------------------------------------------
# 11. NaN diagnostic
# ---------------------------------------------------------------------------

def _warn_high_nan_features(X_raw: pd.DataFrame) -> List[str]:
    """
    Return warning strings for columns whose NaN fraction exceeds the threshold.

    High-NaN columns pass through to PCA/UMAP silently unless flagged here.
    Downstream sklearn/umap implementations may drop them, impute them, or fail.

    Features in ``_EXPECTED_SPARSE_FEATURES`` use a looser threshold
    (``_NAN_WARN_THRESHOLD_SPARSE``) because secondary-basin absence is normal.
    """
    warns: List[str] = []
    n = len(X_raw)
    if n == 0:
        return warns
    for col in X_raw.columns:
        nan_frac = float(X_raw[col].isna().sum()) / n
        threshold = (
            _NAN_WARN_THRESHOLD_SPARSE
            if str(col) in _EXPECTED_SPARSE_FEATURES
            else _NAN_WARN_THRESHOLD
        )
        if nan_frac > threshold:
            warns.append(
                f"Feature '{col}' is {nan_frac:.0%} NaN ({int(nan_frac * n)}/{n} rows).  "
                "This column will have minimal effect on embedding and may cause "
                "failures in some PCA/UMAP implementations."
            )
    return warns


# ---------------------------------------------------------------------------
# 12. Main public API
# ---------------------------------------------------------------------------

def build_tab_input(
    df: pd.DataFrame,
    preset: str = "pmf_intrinsic",
    include_annotations: bool = True,
    include_metadata: bool = True,
    include_quality: bool = True,
    transformed: bool = True,
    strict_validity: bool = True,
    scaler_scope: str = "current_selection",
    allow_physical_coordinates: bool = False,
) -> Dict[str, Any]:
    """
    Build a scientifically safe tab input payload from a long-format PMF DataFrame.

    ``df`` is expected to be the ``summary`` table from the BATCH_ANA SQLite
    output (one row per variant × observable/metric).

    Returns a dict with keys:
        dataframe          filtered input DataFrame (index aligned with X_raw/X)
        features           selected analysis feature column names
        X_raw              DataFrame of raw feature values (float64, NaN preserved)
        X                  DataFrame of transformed values (or copy of X_raw if
                           transformed=False).  Never use X values as physical
                           tooltip values — they are rescaled.
        transform_metadata per-column transform info
        feature_roles      role string per selected feature
        annotations        physical annotation columns (NOT included in X)
        metadata           metadata columns (may be empty if not produced by BATCH_ANA)
        quality            quality flag columns (may be empty)
        row_ids            stable identifier columns (variant, metric, …)
        preset             the preset used
        scaler_scope       scaler scope string
        warnings           list of warning strings

    Preset-specific behaviour:
      "pmf_annotations"   Physical coordinate columns are returned as-is.
                          allow_physical_coordinates is effectively True for this
                          preset — it is designed for tooltip/detail use, not
                          embedding.  can_include_physical_coordinates() is still
                          checked and a warning is emitted if multiple observables
                          are present.
    """
    warns: List[str] = []

    if not isinstance(df, pd.DataFrame) or df.empty:
        warns.append("Input DataFrame is empty.")
        return _empty_payload(preset, scaler_scope, warns)

    # Resolve feature names for this preset against available columns.
    requested, missing = resolve_feature_preset(preset, df.columns.tolist())
    if missing:
        warns.append(f"Missing expected PMF columns for preset '{preset}': {missing}.")

    # Row validity filtering (quality flags).
    df_filtered, v_warns = apply_pmf_validity_filters(df, preset, strict=strict_validity)
    warns.extend(v_warns)

    if df_filtered.empty:
        warns.append("All rows removed by validity filters.")
        return _empty_payload(preset, scaler_scope, warns)

    # Reference feature guard — filters rows AND features list.
    df_filtered, requested, ref_warns = _check_reference_validity(df_filtered, requested)
    warns.extend(ref_warns)

    if df_filtered.empty:
        warns.append("All rows removed by reference-validity filter.")
        return _empty_payload(preset, scaler_scope, warns)

    # Restrict to columns still present after row filtering.
    available = set(df_filtered.columns)
    features = [f for f in requested if f in available]

    # Remove physical annotations / metadata from embedding features.
    features, excl_warns = remove_default_exclusions(
        features,
        df_filtered,
        allow_physical_coordinates=allow_physical_coordinates,
        preset=preset,
    )
    warns.extend(excl_warns)

    if not features:
        warns.append(
            f"No analysis features remain after exclusions for preset '{preset}'."
        )
        return _empty_payload(preset, scaler_scope, warns)

    # Stable row identifiers — carry through for PMF → curve → basin linkage.
    row_ids = _collect_row_ids(df_filtered)

    # Feature matrices — always float64 regardless of SQLite storage dtype.
    X_raw = df_filtered[features].copy().astype(float)

    # Warn on high-NaN columns before they silently degrade embedding quality.
    warns.extend(_warn_high_nan_features(X_raw))

    if transformed:
        # For per_observable scoping, pass the metric/observable column as groups.
        obs_groups: Optional[pd.Series] = None
        if scaler_scope == "per_observable":
            obs_col = "metric" if "metric" in row_ids.columns else (
                "observable_name" if "observable_name" in row_ids.columns else None
            )
            if obs_col is not None:
                obs_groups = row_ids[obs_col].reset_index(drop=True)
        X_t, tmeta = transform_features(
            X_raw, PMF_TRANSFORM_RULES, scaler_scope=scaler_scope, groups=obs_groups
        )
    else:
        X_t = X_raw.copy()
        tmeta = {"scaler_scope": scaler_scope, "columns": {c: "none" for c in features}}

    # Ancillary frames — separate from X, must not be used as X inputs.
    ann_cols = [c for c in PMF_PHYSICAL_ANNOTATIONS if c in df_filtered.columns]
    meta_cols = [c for c in PMF_METADATA_COLUMNS if c in df_filtered.columns]
    qual_cols = [c for c in PMF_QUALITY_COLUMNS if c in df_filtered.columns]

    annotations = (
        df_filtered[ann_cols].copy()
        if (include_annotations and ann_cols)
        else pd.DataFrame(index=df_filtered.index)
    )
    metadata = (
        df_filtered[meta_cols].copy()
        if (include_metadata and meta_cols)
        else pd.DataFrame(index=df_filtered.index)
    )
    quality = (
        df_filtered[qual_cols].copy()
        if (include_quality and qual_cols)
        else pd.DataFrame(index=df_filtered.index)
    )

    feature_roles = {f: FEATURE_ROLES.get(f, "unknown") for f in features}

    return {
        "dataframe": df_filtered,
        "features": features,
        "X_raw": X_raw,
        "X": X_t,
        "transform_metadata": tmeta,
        "feature_roles": feature_roles,
        "annotations": annotations,
        "metadata": metadata,
        "quality": quality,
        "row_ids": row_ids,
        "preset": preset,
        "scaler_scope": scaler_scope,
        "warnings": warns,
    }


def _empty_payload(preset: str, scaler_scope: str, warns: List[str]) -> Dict[str, Any]:
    empty = pd.DataFrame()
    return {
        "dataframe": empty,
        "features": [],
        "X_raw": empty,
        "X": empty,
        "transform_metadata": {"scaler_scope": scaler_scope, "columns": {}},
        "feature_roles": {},
        "annotations": empty,
        "metadata": empty,
        "quality": empty,
        "row_ids": empty,
        "preset": preset,
        "scaler_scope": scaler_scope,
        "warnings": warns,
    }
