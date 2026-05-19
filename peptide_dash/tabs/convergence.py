from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from plotly.subplots import make_subplots
from dash import Input, Output, State, dcc, html
from dash.exceptions import PreventUpdate

from ..data import io as data_io
from ..theming.errors import error_fig
from ..metrics import metric_display_label, torsion_sort_key
from .shared import metric_options as _metric_options_list

TAB_LABEL = "Convergence"

_metric_sort_key = torsion_sort_key
_MAX_DEFAULT_VARIANTS = 6
_MAX_DISCOVERY_FILES = 32

_MODE_DASHBOARD = "dashboard"
_MODE_SINGLE = "single"

_META_COLUMNS = {
    "variant", "metric", "checkpoint", "checkpoint_frac", "fraction", "frac",
    "file", "path", "n_frames", "frame", "time_ps", "time_ns", "step",
    "replica", "repl", "source", "block", "window", "start", "stop",
}
_X_CANDIDATES = [
    "checkpoint_frac", "frac", "fraction", "checkpoint", "n_frames",
    "frame", "time_ns", "time_ps", "step",
]
_LONG_VALUE_PRIORITY = [
    "value", "metric_value", "mean", "estimate", "score", "stat", "y",
]

_RECOMMENDED_METRIC_TIERS: dict[str, list[str]] = {
    "Tier 1: minimal but high-value": [
        "RMSE_F", "MAE_F", "MAX_ABS_F", "JS", "barrier_error", "min_position_shift",
    ],
    "Tier 2: essential for nonconverged pooled replicas": [
        "leave_one_replica_out_RMSE", "leave_one_replica_out_JS",
        "replica_to_pool_RMSE_mean", "replica_to_pool_RMSE_max",
        "between_replica_var_mean", "between_replica_var_max",
        "min_eff_replicas_per_bin", "frac_bins_dominated_by_one_replica",
    ],
    "Tier 3: temporal diagnostics": [
        "forward_backward_RMSE", "forward_backward_JS", "block_to_block_RMSE",
        "last_half_vs_first_half_RMSE", "running_slope_RMSE_F", "running_slope_barrier",
    ],
    "Tier 4: uncertainty": [
        "bootstrap_CI_width_mean", "bootstrap_CI_width_max", "barrier_CI_width",
        "jackknife_max_shift",
    ],
    "Tier 5: hidden conformer validity": [
        "starting_conformer_class_RMSE", "starting_conformer_class_JS",
        "orthogonal_state_JS", "dominant_conformer_fraction_per_bin",
    ],
}
_RECOMMENDED_METRICS: list[str] = [m for vals in _RECOMMENDED_METRIC_TIERS.values() for m in vals]
_RECOMMENDED_ORDER: dict[str, int] = {m: i for i, m in enumerate(_RECOMMENDED_METRICS)}

_DASHBOARD_PANELS: list[dict[str, Any]] = [
    {
        "title": "Total pooled PMF change over time",
        "sources": ("pooled",),
        "metrics": ("RMSE_F", "JS", "MAX_ABS_F", "barrier_error"),
    },
    {
        "title": "Replica influence",
        "sources": ("pooled", "per_replica"),
        "metrics": (
            "leave_one_replica_out_RMSE", "leave_one_replica_out_JS",
            "replica_to_pool_RMSE_mean", "replica_to_pool_RMSE_max",
            "jackknife_max_shift",
        ),
    },
    {
        "title": "Coverage / dominance",
        "sources": ("pooled", "per_replica"),
        "metrics": (
            "min_eff_replicas_per_bin", "frac_bins_dominated_by_one_replica",
            "between_replica_var_mean", "between_replica_var_max",
        ),
    },
    {
        "title": "Uncertainty over time",
        "sources": ("pooled",),
        "metrics": (
            "bootstrap_CI_width_mean", "bootstrap_CI_width_max", "barrier_CI_width",
        ),
    },
]

_METRIC_ALIASES: dict[str, tuple[str, ...]] = {
    "RMSE_F": ("RMSE_F", "RMSEF", "rmse_f", "pmf_rmse", "rmse_pmf", "tail_median_RMSE_F"),
    "MAE_F": ("MAE_F", "MAEF", "mae_f", "pmf_mae", "mae_pmf"),
    "MAX_ABS_F": ("MAX_ABS_F", "max_abs_f", "max_abs_error", "max_abs_pmf", "pmf_max_abs"),
    "JS": ("JS", "JSD", "js", "jsd", "jensen_shannon", "jensen_shannon_divergence", "tail_median_JS"),
    "barrier_error": ("barrier_error", "barrier_err", "delta_barrier", "barrier_delta", "barrier_rmse"),
    "min_position_shift": ("min_position_shift", "minimum_position_shift", "min_shift", "x_min_shift", "xmin_shift"),
    "leave_one_replica_out_RMSE": (
        "leave_one_replica_out_RMSE", "leave_one_replica_out_RMSE_F", "leave_one_out_RMSE",
        "loro_RMSE", "loo_RMSE", "jackknife_RMSE", "jackknife_RMSE_F",
    ),
    "leave_one_replica_out_JS": (
        "leave_one_replica_out_JS", "leave_one_replica_out_JSD", "leave_one_out_JS",
        "loro_JS", "loo_JS", "jackknife_JS", "jackknife_JSD",
    ),
    "replica_to_pool_RMSE_mean": (
        "replica_to_pool_RMSE_mean", "reps_to_pooled_RMSE_mean", "RMSE_reps_to_pooled_mean",
        "replica_pool_RMSE_mean",
    ),
    "replica_to_pool_RMSE_max": (
        "replica_to_pool_RMSE_max", "reps_to_pooled_RMSE_max", "RMSE_reps_to_pooled_max",
        "replica_pool_RMSE_max",
    ),
    "between_replica_var_mean": (
        "between_replica_var_mean", "replica_var_mean", "between_replicas_var_mean",
        "replica_variance_mean",
    ),
    "between_replica_var_max": (
        "between_replica_var_max", "replica_var_max", "between_replicas_var_max",
        "replica_variance_max",
    ),
    "min_eff_replicas_per_bin": (
        "min_eff_replicas_per_bin", "min_effective_replicas_per_bin", "effective_replicas_min",
        "min_n_eff_replicas", "pmf_n_replicas_used_for_ci",
    ),
    "frac_bins_dominated_by_one_replica": (
        "frac_bins_dominated_by_one_replica", "fraction_bins_dominated_by_one_replica",
        "dominant_replica_bin_fraction", "replica_dominance_fraction", "dominance_frac",
    ),
    "forward_backward_RMSE": ("forward_backward_RMSE", "fb_RMSE", "forward_backward_RMSE_F"),
    "forward_backward_JS": ("forward_backward_JS", "forward_backward_JSD", "fb_JS", "fb_JSD"),
    "block_to_block_RMSE": ("block_to_block_RMSE", "block_RMSE", "blockwise_RMSE"),
    "last_half_vs_first_half_RMSE": (
        "last_half_vs_first_half_RMSE", "second_half_vs_first_half_RMSE", "half_split_RMSE",
    ),
    "running_slope_RMSE_F": ("running_slope_RMSE_F", "RMSE_F_running_slope", "slope_RMSE_F"),
    "running_slope_barrier": ("running_slope_barrier", "barrier_running_slope", "slope_barrier"),
    "bootstrap_CI_width_mean": (
        "bootstrap_CI_width_mean", "bootstrap_ci_width_mean", "pmf_CI_width_mean",
        "pmf_F_CI_width_mean", "pmf_F_std_mean_kJmol", "bootstrap_pmf_ci_mean",
    ),
    "bootstrap_CI_width_max": (
        "bootstrap_CI_width_max", "bootstrap_ci_width_max", "pmf_CI_width_max",
        "pmf_F_CI_width_max", "pmf_F_std_max_kJmol", "bootstrap_pmf_ci_max",
    ),
    "barrier_CI_width": ("barrier_CI_width", "barrier_ci_width", "bootstrap_barrier_ci_width"),
    "jackknife_max_shift": ("jackknife_max_shift", "jackknife_shift_max", "max_jackknife_shift"),
    "starting_conformer_class_RMSE": (
        "starting_conformer_class_RMSE", "conformer_class_RMSE", "start_conformer_RMSE",
    ),
    "starting_conformer_class_JS": (
        "starting_conformer_class_JS", "starting_conformer_class_JSD", "conformer_class_JS",
        "start_conformer_JS",
    ),
    "orthogonal_state_JS": ("orthogonal_state_JS", "orthogonal_state_JSD", "state_JS", "state_JSD"),
    "dominant_conformer_fraction_per_bin": (
        "dominant_conformer_fraction_per_bin", "dominant_conformer_frac_per_bin",
        "dominant_conformer_fraction", "conformer_dominance_fraction",
    ),
}




def _ensure_df(df) -> pd.DataFrame:
    if isinstance(df, pd.DataFrame):
        return df
    return pd.DataFrame()


def _source_key(source: str | None) -> str:
    src = (source or "pooled").lower()
    if src in {"per_replica", "replica", "replicas"}:
        return "per_replica"
    return "pooled"


def _base_dir(ctx) -> Path | None:
    if ctx is None:
        return None
    try:
        base, _ = data_io._resolve_layout(getattr(ctx, "data_dir", ""))
        return Path(base).expanduser().resolve()
    except Exception:
        return None


def _patterns_for_source(source: str | None) -> list[str]:
    if _source_key(source) == "per_replica":
        return ["*_convergence_replica.parquet", "*_convergence_replica.csv.gz", "*_convergence_replica.csv", "convergence_replica.parquet", "convergence_replica.csv.gz", "convergence_replica.csv"]
    return ["*_convergence.parquet", "*_convergence.csv.gz", "*_convergence.csv", "convergence_*.parquet", "convergence_*.csv.gz", "convergence_*.csv", "convergence.parquet", "convergence.csv.gz", "convergence.csv"]


@lru_cache(maxsize=32)
def _discover_files_cached(base_dir_str: str, source: str) -> tuple[str, ...]:
    base = Path(base_dir_str)
    seen: set[Path] = set()
    out: list[Path] = []
    for recursive in (False, True):
        if out and recursive:
            break
        for pat in _patterns_for_source(source):
            iterator = base.rglob(pat) if recursive else base.glob(pat)
            for p in iterator:
                if not p.is_file():
                    continue
                if source == "pooled" and "convergence_replica" in p.name:
                    continue
                rp = p.resolve()
                if rp not in seen:
                    seen.add(rp)
                    out.append(rp)
    out.sort(key=lambda q: q.name)
    return tuple(str(q) for q in out)


def _discover_files(ctx, source: str | None) -> list[Path]:
    base = _base_dir(ctx)
    if base is None:
        return []
    return [Path(x) for x in _discover_files_cached(str(base), _source_key(source))]


def _conv_suffix(source: str | None) -> str:
    return "_convergence_replica" if _source_key(source) == "per_replica" else "_convergence"


def _variant_from_path(path: Path, source: str | None) -> str | None:
    suffix = _conv_suffix(source)
    for ext in (".parquet", ".csv.gz", ".csv"):
        full = f"{suffix}{ext}"
        if path.name.endswith(full):
            v = path.name[:-len(full)]
            return v or None
    return None


def _schema_columns(path: Path) -> list[str]:
    try:
        if path.suffix.lower() == ".parquet":
            import pyarrow.parquet as pq
            return [str(x) for x in pq.read_schema(path).names]
    except Exception:
        pass
    try:
        if path.suffix.lower() == ".parquet":
            return list(pd.read_parquet(path).columns)
        return list(pd.read_csv(path, nrows=0).columns)
    except Exception:
        return []


def _read_columns(path: Path, columns: Sequence[str] | None = None) -> pd.DataFrame:
    try:
        if path.suffix.lower() == ".parquet":
            available = set(_schema_columns(path))
            use = [c for c in (columns or []) if c in available] if columns else None
            return pd.read_parquet(path, columns=use)
        header = set(pd.read_csv(path, nrows=0).columns)
        use = [c for c in (columns or []) if c in header] if columns else None
        return pd.read_csv(path, usecols=use)
    except Exception:
        return pd.DataFrame()


def _metric_options_from_files(ctx, source: str | None) -> tuple[list[dict], Optional[str], str]:
    paths = _discover_files(ctx, source)
    if not paths:
        src = "per-replica" if _source_key(source) == "per_replica" else "pooled"
        return [], None, f"No {src} convergence files found."
    metrics: set[str] = set()
    checked = 0
    for p in paths[: min(len(paths), _MAX_DISCOVERY_FILES)]:
        if "metric" not in _schema_columns(p):
            continue
        d = _read_columns(p, ["metric"])
        checked += 1
        if "metric" in d.columns and not d.empty:
            metrics.update(d["metric"].dropna().astype(str).unique().tolist())
        if metrics and checked >= 4:
            break
    mets = sorted(metrics, key=_metric_sort_key)
    opts = _metric_options_list(mets)
    msg = f"Discovered {len(paths):,} {_source_key(source).replace('_', '-')} convergence files."
    if checked:
        msg += f" Metric list sampled from {checked} file(s)."
    return opts, (mets[0] if mets else None), msg


def _variant_options_from_files(ctx, source: str | None, metric: Optional[str]) -> tuple[list[dict], list[str], str]:
    paths = _discover_files(ctx, source)
    variants: set[str] = set()
    generic: list[Path] = []
    for p in paths:
        v = _variant_from_path(p, source)
        if v:
            variants.add(v)
        else:
            generic.append(p)
    for p in generic[: min(len(generic), _MAX_DISCOVERY_FILES)]:
        cols = ["variant"] + (["metric"] if metric else [])
        d = _read_columns(p, cols)
        if d.empty or "variant" not in d.columns:
            continue
        if metric and "metric" in d.columns:
            d = d[d["metric"].astype(str) == str(metric)]
        variants.update(d["variant"].dropna().astype(str).unique().tolist())
    vals = sorted(variants)
    default = vals[: min(_MAX_DEFAULT_VARIANTS, len(vals))]
    return [{"label": v, "value": v} for v in vals], default, f"Detected {len(vals):,} variant(s); defaulting to {len(default)}."


def _y_options_from_files(ctx, source: str | None, metric: Optional[str]) -> tuple[list[dict], Optional[str]]:
    drop = {"variant", "metric", "checkpoint", "checkpoint_frac", "fraction", "frac", "file", "path", "n_frames", "replica", "repl", "source"}
    y_cols: list[str] = []
    for p in _discover_files(ctx, source)[:_MAX_DISCOVERY_FILES]:
        cols = _schema_columns(p)
        if not cols:
            continue
        d = _read_columns(p, cols)
        if metric and "metric" in d.columns:
            d = d[d["metric"].astype(str) == str(metric)]
        if d.empty:
            continue
        y_cols = [c for c in d.columns if c not in drop and pd.api.types.is_numeric_dtype(d[c])]
        if y_cols:
            break
    import re
    pri = [c for c in y_cols if re.search(r"(jsd|diverg|mse|rmse|error|diff|bias|l1|l2|rmse|js)", c, flags=re.I)]
    vals = pri or y_cols
    return ([{"label": y, "value": y} for y in vals], (vals[0] if vals else None))


def _load_selected_convergence(ctx, source: str | None, variants_sel: Optional[Sequence[str]], ycol: Optional[str]) -> pd.DataFrame:
    selected = {str(v) for v in (variants_sel or []) if v is not None}
    if not selected:
        return pd.DataFrame()
    chosen: list[Path] = []
    generic: list[Path] = []
    for p in _discover_files(ctx, source):
        v = _variant_from_path(p, source)
        if v is None:
            generic.append(p)
        elif v in selected:
            chosen.append(p)
    load_paths = chosen + generic
    if not load_paths:
        return pd.DataFrame()
    requested = ["variant", "metric", "checkpoint_frac", "frac", "fraction", "checkpoint", "n_frames", "replica", "repl", "source"]
    if ycol:
        requested.append(str(ycol))
    frames: list[pd.DataFrame] = []
    data_io._GLOBAL_PROGRESS.start(len(load_paths), f"loading convergence {_source_key(source)}")
    try:
        for p in load_paths:
            cols = [c for c in requested if c in set(_schema_columns(p))]
            d = _read_columns(p, cols)
            data_io._GLOBAL_PROGRESS.tick(1)
            if d.empty:
                continue
            parsed = _variant_from_path(p, source)
            if "variant" not in d.columns and parsed:
                d["variant"] = parsed
            if "variant" in d.columns:
                d = d[d["variant"].astype(str).isin(selected)]
            if not d.empty:
                frames.append(d)
    finally:
        data_io._GLOBAL_PROGRESS.finish()
    return pd.concat(frames, ignore_index=True, copy=False) if frames else pd.DataFrame()


def _get_conv_df(ctx, source: str | None = None) -> pd.DataFrame:
    # Backwards-compatible full-table accessor.  The tab callbacks below do not
    # use this for discovery/plotting because loading every convergence parquet
    # in a large GLOBAL_DATA directory can freeze the dashboard.
    if ctx is None:
        return pd.DataFrame()
    if _source_key(source) == "per_replica":
        return _ensure_df(getattr(ctx, "conv_replica_df", None))
    return _ensure_df(getattr(ctx, "conv_df", None))

def _available_sources(ctx) -> list[dict]:
    opts = []
    if not _get_conv_df(ctx, "pooled").empty:
        opts.append({"label": "Pooled", "value": "pooled"})
    if not _get_conv_df(ctx, "per_replica").empty:
        opts.append({"label": "Per-replica", "value": "per_replica"})
    return opts


def _default_source(ctx) -> str:
    pooled = _get_conv_df(ctx, "pooled")
    replica = _get_conv_df(ctx, "per_replica")
    if not pooled.empty:
        return "pooled"
    if not replica.empty:
        return "per_replica"
    return "pooled"


def _conv_default_columns(dfc: pd.DataFrame) -> Tuple[List[str], List[str]]:
    if dfc.empty:
        return [], []
    drop = {
        "variant", "metric", "checkpoint", "checkpoint_frac", "fraction", "frac",
        "file", "path", "n_frames", "replica", "repl", "source",
    }
    num_cols: List[str] = [
        c for c in dfc.columns if pd.api.types.is_numeric_dtype(dfc[c]) and c not in drop
    ]
    import re
    pri = [
        c for c in num_cols
        if re.search(r"(jsd|diverg|mse|rmse|error|diff|bias|l1|l2|rmse|js)", c, flags=re.I)
    ]
    y_opts = pri or num_cols
    return y_opts, (y_opts[:1] if y_opts else [])



def _norm_metric_name(value: object) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


_ALIAS_NORMS: dict[str, set[str]] = {
    canon: {_norm_metric_name(canon), *{_norm_metric_name(a) for a in aliases}}
    for canon, aliases in _METRIC_ALIASES.items()
}


def _canonical_metric_name(value: object) -> Optional[str]:
    norm = _norm_metric_name(value)
    if not norm:
        return None
    for canon, aliases in _ALIAS_NORMS.items():
        if norm in aliases:
            return canon
    for canon in _RECOMMENDED_METRICS:
        c_norm = _norm_metric_name(canon)
        if norm == c_norm:
            return canon
        # Accept decorated pipeline names such as "foo__RMSE_F" without
        # allowing very short aliases such as "JS" to match arbitrary strings.
        if len(c_norm) >= 5 and (norm.endswith(c_norm) or c_norm in norm):
            return canon
    return None


def _matches_recommended_metric(value: object, desired: Sequence[str]) -> bool:
    canon = _canonical_metric_name(value)
    return bool(canon and canon in set(desired))


def _metric_dashboard_sort_key(metric: object) -> tuple[int, str]:
    canon = _canonical_metric_name(metric) or str(metric)
    return (_RECOMMENDED_ORDER.get(canon, 10_000), str(metric).lower())


def _is_meta_or_x(col: object) -> bool:
    c = str(col)
    return c in _META_COLUMNS or c in _X_CANDIDATES


def _discover_metrics_from_files(ctx, source: str | None) -> set[str]:
    out: set[str] = set()
    for p in _discover_files(ctx, source)[:_MAX_DISCOVERY_FILES]:
        cols = _schema_columns(p)
        if not cols:
            continue
        if "metric" in cols:
            d = _read_columns(p, ["metric"])
            if "metric" in d.columns and not d.empty:
                out.update(d["metric"].dropna().astype(str).unique().tolist())
        # Wide convergence files store the metrics directly as numeric columns.
        for c in cols:
            if not _is_meta_or_x(c):
                canon = _canonical_metric_name(c)
                if canon:
                    out.add(str(c))
    return out


def _available_recommended_metrics(ctx) -> dict[str, set[str]]:
    return {
        "pooled": _discover_metrics_from_files(ctx, "pooled"),
        "per_replica": _discover_metrics_from_files(ctx, "per_replica"),
    }


def _variant_options_dashboard(ctx) -> tuple[list[dict], list[str], str]:
    variants: set[str] = set()
    msgs: list[str] = []
    for source in ("pooled", "per_replica"):
        opts, defaults, msg = _variant_options_from_files(ctx, source, None)
        variants.update(str(o.get("value")) for o in opts if o.get("value") is not None)
        if msg:
            src = "per-replica" if source == "per_replica" else "pooled"
            msgs.append(f"{src}: {msg}")
    vals = sorted(variants)
    default = vals[: min(_MAX_DEFAULT_VARIANTS, len(vals))]
    msg = " ".join(msgs) if msgs else "No convergence variants detected."
    return [{"label": v, "value": v} for v in vals], default, msg


def _requested_dashboard_metrics() -> list[str]:
    wanted: list[str] = []
    for panel in _DASHBOARD_PANELS:
        wanted.extend([str(m) for m in panel.get("metrics", ())])
    # Preserve order and avoid duplicate I/O requests.
    return list(dict.fromkeys(wanted))


def _dashboard_columns_for_schema(cols: Sequence[str], desired_metrics: Sequence[str]) -> list[str]:
    keep: list[str] = []
    for c in cols:
        if c in _META_COLUMNS or c in _X_CANDIDATES or c == "metric":
            keep.append(c)
    has_metric_col = "metric" in cols
    if has_metric_col:
        # Long format: keep likely value columns. If the pipeline used a custom
        # value column name, the later fallback can still find it among non-meta
        # columns if it was explicitly requested as a recommended wide metric.
        for c in cols:
            if c in keep:
                continue
            if str(c) in _LONG_VALUE_PRIORITY:
                keep.append(c)
    for c in cols:
        if c in keep or _is_meta_or_x(c) or c == "metric":
            continue
        if _matches_recommended_metric(c, desired_metrics):
            keep.append(c)
    # If long format has no obvious value column, include the first few non-meta
    # columns so we can still auto-detect a numeric value without loading the
    # whole file schema blindly.
    if has_metric_col and not any(c in keep for c in _LONG_VALUE_PRIORITY):
        for c in cols:
            if c in keep or _is_meta_or_x(c) or c == "metric":
                continue
            keep.append(c)
            if len([x for x in keep if not _is_meta_or_x(x) and x != "metric"]) >= 8:
                break
    return list(dict.fromkeys(keep))


def _best_long_value_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [c for c in _LONG_VALUE_PRIORITY if c in df.columns]
    candidates.extend(
        c for c in df.columns
        if c not in candidates and not _is_meta_or_x(c) and c != "metric"
    )
    for c in candidates:
        vals = pd.to_numeric(df[c], errors="coerce")
        if vals.notna().any():
            return str(c)
    return None


def _x_column_for_dashboard(df: pd.DataFrame) -> Optional[str]:
    for c in _X_CANDIDATES:
        if c in df.columns:
            return c
    return None


def _load_dashboard_source(ctx, source: str, variants_sel: Sequence[str], desired_metrics: Sequence[str]) -> pd.DataFrame:
    selected = {str(v) for v in (variants_sel or []) if v is not None}
    if not selected:
        return pd.DataFrame()
    chosen: list[Path] = []
    generic: list[Path] = []
    for p in _discover_files(ctx, source):
        v = _variant_from_path(p, source)
        if v is None:
            generic.append(p)
        elif v in selected:
            chosen.append(p)
    load_paths = chosen + generic
    if not load_paths:
        return pd.DataFrame()
    frames: list[pd.DataFrame] = []
    data_io._GLOBAL_PROGRESS.start(len(load_paths), f"loading convergence dashboard {source}")
    try:
        for p in load_paths:
            cols = _schema_columns(p)
            requested = _dashboard_columns_for_schema(cols, desired_metrics)
            d = _read_columns(p, requested)
            data_io._GLOBAL_PROGRESS.tick(1)
            if d.empty:
                continue
            parsed = _variant_from_path(p, source)
            if "variant" not in d.columns and parsed:
                d["variant"] = parsed
            if "variant" in d.columns:
                d = d[d["variant"].astype(str).isin(selected)]
            if d.empty:
                continue
            d["_dashboard_source"] = source
            frames.append(d)
    finally:
        data_io._GLOBAL_PROGRESS.finish()
    return pd.concat(frames, ignore_index=True, copy=False) if frames else pd.DataFrame()


def _dashboard_long_frame(df: pd.DataFrame, desired_metrics: Sequence[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    xcol = _x_column_for_dashboard(df)
    if not xcol:
        return pd.DataFrame()
    base_cols = [c for c in ["variant", xcol, "_dashboard_source", "replica", "repl"] if c in df.columns]
    out_frames: list[pd.DataFrame] = []

    if "metric" in df.columns:
        vcol = _best_long_value_column(df)
        if vcol:
            long = df[base_cols + ["metric", vcol]].copy()
            long["metric_canonical"] = long["metric"].map(_canonical_metric_name)
            long = long[long["metric_canonical"].isin(list(desired_metrics))]
            if not long.empty:
                long = long.rename(columns={xcol: "x", vcol: "value", "_dashboard_source": "source"})
                long["value"] = pd.to_numeric(long["value"], errors="coerce")
                long["x"] = pd.to_numeric(long["x"], errors="coerce")
                out_frames.append(long[[c for c in ["variant", "source", "replica", "repl", "x", "metric", "metric_canonical", "value"] if c in long.columns]])

    wide_cols = [
        c for c in df.columns
        if c not in base_cols and c != "metric" and not _is_meta_or_x(c)
        and _matches_recommended_metric(c, desired_metrics)
    ]
    for c in wide_cols:
        tmp = df[base_cols + [c]].copy()
        tmp = tmp.rename(columns={xcol: "x", c: "value", "_dashboard_source": "source"})
        tmp["metric"] = str(c)
        tmp["metric_canonical"] = _canonical_metric_name(c) or str(c)
        tmp["value"] = pd.to_numeric(tmp["value"], errors="coerce")
        tmp["x"] = pd.to_numeric(tmp["x"], errors="coerce")
        out_frames.append(tmp[[col for col in ["variant", "source", "replica", "repl", "x", "metric", "metric_canonical", "value"] if col in tmp.columns]])

    if not out_frames:
        return pd.DataFrame()
    out = pd.concat(out_frames, ignore_index=True, copy=False)
    out = out[np.isfinite(out["x"]) & np.isfinite(out["value"])]
    return out


def _load_recommended_dashboard_data(ctx, variants_sel: Sequence[str]) -> pd.DataFrame:
    desired = _requested_dashboard_metrics()
    frames = []
    for source in ("pooled", "per_replica"):
        raw = _load_dashboard_source(ctx, source, variants_sel, desired)
        long = _dashboard_long_frame(raw, desired)
        if not long.empty:
            frames.append(long)
    return pd.concat(frames, ignore_index=True, copy=False) if frames else pd.DataFrame()


def _metric_chip(metric: str, available: dict[str, set[str]]) -> html.Span:
    present = False
    for names in available.values():
        present = present or any((_canonical_metric_name(n) == metric) for n in names)
    color = "#86efac" if present else "#fca5a5"
    bg = "rgba(34,197,94,0.12)" if present else "rgba(248,113,113,0.12)"
    return html.Span(
        metric,
        style={
            "display": "inline-block", "padding": "2px 6px", "margin": "2px",
            "border": f"1px solid {color}", "borderRadius": "999px",
            "backgroundColor": bg, "fontSize": "0.75em",
        },
    )


def _recommended_metric_summary(ctx) -> html.Div:
    available = _available_recommended_metrics(ctx)
    return html.Div([
        html.Details([
            html.Summary("Recommended metric set availability", style={"cursor": "pointer", "fontWeight": 600}),
            *[
                html.Div([
                    html.Div(tier, style={"fontWeight": 600, "marginTop": "6px"}),
                    html.Div([_metric_chip(m, available) for m in metrics]),
                ])
                for tier, metrics in _RECOMMENDED_METRIC_TIERS.items()
            ],
            html.Div("Green = found in discovered convergence schemas/metric rows; red = not found yet.", style={"fontSize": "0.75em", "opacity": 0.75, "marginTop": "6px"}),
        ], open=False)
    ], style={"marginTop": "8px"})


def _dashboard_summary_table(long: pd.DataFrame) -> html.Table:
    if long is None or long.empty:
        return html.Table()
    rows = []
    for (variant, source, metric), sub in long.groupby(["variant", "source", "metric_canonical"], sort=False):
        sub = sub.dropna(subset=["x", "value"]).sort_values("x")
        if sub.empty:
            continue
        rows.append({
            "variant": str(variant),
            "source": str(source).replace("_", "-"),
            "metric": str(metric),
            "final": float(sub["value"].iloc[-1]),
            "delta_total": float(sub["value"].iloc[-1] - sub["value"].iloc[0]) if len(sub) > 1 else np.nan,
        })
    if not rows:
        return html.Table()
    sdf = pd.DataFrame(rows).sort_values(["variant", "source", "metric"], key=lambda col: col.map(str))
    for c in ["final", "delta_total"]:
        sdf[c] = sdf[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
    cols = ["variant", "source", "metric", "final", "delta_total"]
    return html.Table([
        html.Thead(html.Tr([html.Th(c) for c in cols])),
        html.Tbody([html.Tr([html.Td(sdf.iloc[i][c]) for c in cols]) for i in range(min(len(sdf), 32))]),
    ], style={"width": "100%", "borderCollapse": "collapse"})


def _recommended_dashboard_figure(long: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[str(p["title"]) for p in _DASHBOARD_PANELS],
        horizontal_spacing=0.08,
        vertical_spacing=0.14,
    )
    if long is None or long.empty:
        fig.update_layout(template="plotly_dark", title="Recommended PMF convergence dashboard")
        return fig

    for idx, panel in enumerate(_DASHBOARD_PANELS):
        row = idx // 2 + 1
        col = idx % 2 + 1
        metrics = set(panel.get("metrics", ()))
        sources = set(panel.get("sources", ()))
        sub = long[long["metric_canonical"].isin(metrics)]
        if sources:
            sub = sub[sub["source"].isin(sources)]
        if sub.empty:
            fig.add_annotation(
                text="No matching metrics found",
                row=row, col=col, x=0.5, y=0.5, xref=f"x{idx + 1} domain", yref=f"y{idx + 1} domain",
                showarrow=False, font=dict(size=11, color="#aaa"),
            )
            continue
        group_cols = ["variant", "source", "metric_canonical"]
        for (variant, source, metric), g in sub.groupby(group_cols, sort=False):
            g = g.groupby("x", observed=True)["value"].mean().reset_index().sort_values("x")
            if g.empty:
                continue
            src_suffix = "" if source == "pooled" else " · replica"
            fig.add_trace(
                go.Scatter(
                    x=g["x"], y=g["value"], mode="lines+markers",
                    name=f"{variant} · {metric}{src_suffix}",
                    legendgroup=f"{variant}-{metric}-{source}",
                    hovertemplate=(
                        f"variant={variant}<br>source={source}<br>metric={metric}<br>"
                        "checkpoint=%{x:.4g}<br>value=%{y:.4g}<extra></extra>"
                    ),
                ),
                row=row, col=col,
            )
        fig.update_xaxes(title_text="checkpoint / fraction / frames", row=row, col=col)
        fig.update_yaxes(title_text="metric value", row=row, col=col)

    fig.update_layout(
        template="plotly_dark",
        title="Recommended PMF convergence dashboard",
        height=660,
        legend_title="variant · metric",
        margin=dict(l=45, r=25, t=55, b=45),
    )
    return fig


def _recommended_dashboard(ctx, variants_sel: Optional[Sequence[str]]) -> tuple[go.Figure, html.Div]:
    if isinstance(variants_sel, str):
        variants_sel = [variants_sel]
    variants = [str(v) for v in (variants_sel or []) if v is not None]
    if not variants:
        return error_fig("Select at least one variant to plot."), html.Div("No variants selected. Nothing was loaded.")
    long = _load_recommended_dashboard_data(ctx, variants)
    if long.empty:
        summary = html.Div([
            html.Div("No recommended convergence metrics found for the selected variant(s).", style={"fontWeight": 600}),
            html.Div("The tab looked for long-format metric rows and wide-format columns matching the recommended PMF convergence names."),
            _recommended_metric_summary(ctx),
        ])
        return error_fig("No recommended PMF convergence metrics found."), summary
    fig = _recommended_dashboard_figure(long)
    plotted = sorted(long["metric_canonical"].dropna().astype(str).unique().tolist(), key=_metric_dashboard_sort_key)
    missing = [m for m in _requested_dashboard_metrics() if m not in set(plotted)]
    summary = html.Div([
        html.Div(
            "Interpretation warning: the pooled PMF estimate may only be stabilized with respect to additional replicas/frames. "
            "If starting conformers do not interconvert, treat it as a PMF over the sampled conformer ensemble under this initialization protocol, not automatically an equilibrium PMF.",
            style={"padding": "8px", "border": "1px solid #a16207", "borderRadius": "8px", "backgroundColor": "rgba(161,98,7,0.16)", "marginBottom": "8px"},
        ),
        html.Div(f"Loaded recommended convergence rows: {len(long):,} across {len(set(long['variant'])):,} variant(s).", style={"fontWeight": 600, "marginBottom": "4px"}),
        html.Div("Plotted metrics: " + ", ".join(plotted), style={"fontSize": "0.82em", "marginBottom": "4px"}),
        html.Div("Missing recommended metrics: " + (", ".join(missing) if missing else "none"), style={"fontSize": "0.82em", "opacity": 0.8, "marginBottom": "8px"}),
        _dashboard_summary_table(long),
        _recommended_metric_summary(ctx),
    ])
    return fig, summary


def _metric_options(df: pd.DataFrame) -> tuple[list[dict], Optional[str]]:
    if df.empty or "metric" not in df.columns:
        return [], None
    mets = sorted(df["metric"].dropna().astype(str).unique().tolist(), key=_metric_sort_key)
    opts = _metric_options_list(mets)
    return opts, (mets[0] if mets else None)


def _variant_options(df: pd.DataFrame, metric: Optional[str]) -> tuple[list[dict], list[str]]:
    d = df
    if metric and "metric" in d.columns:
        d = d[d["metric"].astype(str) == str(metric)]
    if d.empty or "variant" not in d.columns:
        return [], []
    vals = sorted(d["variant"].dropna().astype(str).unique().tolist())
    return [{"label": v, "value": v} for v in vals], vals[: min(6, len(vals))]


def _y_options(df: pd.DataFrame, metric: Optional[str]) -> tuple[list[dict], Optional[str]]:
    d = df
    if metric and "metric" in d.columns:
        d = d[d["metric"].astype(str) == str(metric)]
    y_cols, y_default = _conv_default_columns(d)
    return ([{"label": y, "value": y} for y in y_cols], (y_default[0] if y_default else None))


def _compute_conv_stats(
    df_curve: pd.DataFrame,
    xcol: str,
    ycol: str,
    eps: Optional[float],
    eps_rel: Optional[float],
    window_frac: Optional[float],
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "final": np.nan,
        "delta_total": np.nan,
        "t_conv": np.nan,
        "converged": False,
    }
    if df_curve is None or df_curve.empty:
        return out
    yc = pd.to_numeric(df_curve[ycol], errors="coerce").to_numpy()
    xc = pd.to_numeric(df_curve[xcol], errors="coerce").to_numpy()
    mask = np.isfinite(xc) & np.isfinite(yc)
    if not np.any(mask):
        return out
    xc = xc[mask]
    yc = yc[mask]
    if xc.size == 0:
        return out
    final_val = float(yc[-1])
    first_val = float(yc[0])
    out["final"] = final_val
    out["delta_total"] = final_val - first_val
    abs_eps = max(float(eps) if eps is not None else 0.0, 0.0)
    rel_eps = max(float(eps_rel) if eps_rel is not None else 0.0, 0.0)
    if (abs_eps <= 0.0 and rel_eps <= 0.0) or yc.size < 2:
        return out
    diffs = np.abs(yc - final_val)
    scale = float(max(np.nanmax(yc) - np.nanmin(yc), abs(final_val), 1e-12))
    eps_eff = float(max(abs_eps, rel_eps * scale))
    wf = float(window_frac) if window_frac is not None else 0.2
    wf = float(np.clip(wf, 0.01, 0.95))
    win = int(max(3, round(wf * yc.size)))
    win = min(win, yc.size)
    ok = np.zeros(max(yc.size - win + 1, 1), dtype=bool)
    for j in range(ok.size):
        ok[j] = bool(np.nanmax(diffs[j:(j + win)]) <= eps_eff)
    for j in range(ok.size):
        if np.all(ok[j:]):
            out["t_conv"] = float(xc[j])
            out["converged"] = True
            break
    return out


def layout(ctx):
    # Do NOT access ctx.conv_df / ctx.conv_replica_df here — those trigger a
    # full lazy load of all convergence files (potentially 50M+ rows) and would
    # block the splash-screen render.  Dropdowns are populated by callbacks on
    # first interaction instead.
    source = "pooled"
    source_options = [
        {"label": "Pooled", "value": "pooled"},
        {"label": "Per-replica", "value": "per_replica"},
    ]
    metric_options, metric_value = [], None
    variants_options, variants_value = [], []
    y_options, y_value = [], None

    controls = html.Div(
        [
            html.Div("Convergence", style={"fontWeight": 700, "marginBottom": "6px", "fontSize": "1.05em"}),
            html.Div(
                [
                    html.Div(
                        [
                            html.Label("Mode", style={"fontSize": "0.8em"}),
                            dcc.Dropdown(
                                id="conv-mode",
                                options=[
                                    {"label": "Recommended PMF dashboard", "value": _MODE_DASHBOARD},
                                    {"label": "Single-metric explorer", "value": _MODE_SINGLE},
                                ],
                                value=_MODE_DASHBOARD,
                                clearable=False,
                                style={"fontSize": "0.85em"},
                            ),
                        ],
                        style={"flex": "0 0 22%", "paddingRight": "6px"},
                    ),
                    html.Div(
                        [
                            html.Label("Mode", style={"fontSize": "0.8em"}),
                            dcc.Dropdown(
                                id="conv-mode",
                                options=[
                                    {"label": "Recommended PMF dashboard", "value": _MODE_DASHBOARD},
                                    {"label": "Single-metric explorer", "value": _MODE_SINGLE},
                                ],
                                value=_MODE_DASHBOARD,
                                clearable=False,
                                style={"fontSize": "0.85em"},
                            ),
                        ],
                        style={"flex": "0 0 22%", "paddingRight": "6px"},
                    ),
                    html.Div(
                        [
                            html.Label("Source", style={"fontSize": "0.8em"}),
                            dcc.Dropdown(
                                id="conv-source",
                                options=source_options,
                                value=source,
                                clearable=False,
                                style={"fontSize": "0.85em"},
                            ),
                        ],
                        style={"flex": "0 0 16%", "paddingRight": "6px"},
                    ),
                    html.Div(
                        [
                            html.Label("Metric", style={"fontSize": "0.8em"}),
                            dcc.Dropdown(id="conv-metric", options=metric_options, value=metric_value, placeholder="auto-detect", clearable=True, style={"fontSize": "0.85em"}),
                        ],
                        style={"flex": "0 0 20%", "paddingRight": "6px"},
                    ),
                    html.Div(
                        [
                            html.Label("Variant(s)", style={"fontSize": "0.8em"}),
                            dcc.Dropdown(id="conv-variants", options=variants_options, value=variants_value, multi=True, style={"fontSize": "0.85em"}),
                        ],
                        style={"flex": "0 0 30%", "paddingRight": "6px"},
                    ),
                    html.Div(
                        [
                            html.Label("Y to show", style={"fontSize": "0.8em"}),
                            dcc.Dropdown(id="conv-y", options=y_options, value=y_value, placeholder="auto: pick divergence columns", clearable=True, style={"fontSize": "0.85em"}),
                        ],
                        style={"flex": "0 0 18%", "paddingRight": "6px"},
                    ),
                    html.Div(
                        [
                            html.Label("Tolerance ε", style={"fontSize": "0.8em"}),
                            dcc.Input(id="conv-eps", type="number", value=0.01, min=0, step=0.001, style={"width": "100%", "fontSize": "0.85em"}),
                        ],
                        style={"flex": "0 0 8%"},
                    ),
                    html.Div(
                        [
                            html.Label("Relative ε", style={"fontSize": "0.8em"}),
                            dcc.Input(id="conv-eps-rel", type="number", value=0.0, min=0, step=0.001, style={"width": "100%", "fontSize": "0.85em"}),
                        ],
                        style={"flex": "0 0 8%"},
                    ),
                    html.Div(
                        [
                            html.Label("Stable window (fraction)", style={"fontSize": "0.8em"}),
                            dcc.Slider(id="conv-window-frac", min=0.05, max=0.5, step=0.05, value=0.2, tooltip={"placement": "bottom"}),
                        ],
                        style={"flex": "1 1 20%"},
                    ),
                ],
                style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-end", "marginBottom": "6px", "gap": "4px"},
            ),
            html.Div(
                [
                    html.Div(
                        html.Div(id="conv-progress-fill", style={"height": "100%", "width": "0%", "background": "linear-gradient(90deg, #6366f1, #a78bfa)", "borderRadius": "5px"}),
                        style={"height": "8px", "backgroundColor": "rgba(255,255,255,0.12)", "borderRadius": "5px", "overflow": "hidden", "marginTop": "6px"},
                    ),
                    html.Div(id="conv-progress-label", children="Convergence loader idle.", style={"fontSize": "0.72em", "opacity": 0.75, "marginTop": "4px"}),
                    dcc.Interval(id="conv-progress-poll", interval=500, n_intervals=0),
                ]
            ),
            html.Div(
                id="conv-status",
                children="Open this tab to discover convergence files without loading the full convergence dataset.",
                style={"fontSize": "0.7em", "opacity": 0.7},
            ),
        ],
        style={"padding": "10px", "border": "1px solid #444", "borderRadius": "10px", "marginBottom": "8px", "backgroundColor": "rgba(255,255,255,0.02)"},
    )

    graph = dcc.Loading(dcc.Graph(id="convergence-graph", style={"height": "78vh"}, config={"displaylogo": False}), type="circle")
    hover_info = html.Div(id="convergence-hover-info", className="text-muted", style={"fontSize": "0.8em", "minHeight": "1.2em", "marginTop": "2px"})
    summary = html.Div(id="conv-summary", style={"marginTop": "10px", "fontSize": "0.8em"})
    return html.Div([controls, graph, hover_info, summary], style={"padding": "8px", "maxWidth": "1200px", "margin": "0 auto"})


def _option_values(options: Iterable[dict]) -> set[str]:
    return {str(o.get("value")) for o in options if o.get("value") is not None}


def register_callbacks(app, ctx):
    from .shared import apply_theme
    @app.callback(
        Output("conv-metric", "options"),
        Output("conv-metric", "value"),
        Input("tabs", "value"),
        Input("conv-mode", "value"),
        Input("conv-source", "value"),
        prevent_initial_call=False,
    )
    def _update_metric(active_tab: Optional[str], mode: Optional[str], source: Optional[str]):
        if active_tab != "convergence":
            raise PreventUpdate
        opts, default, _msg = _metric_options_from_files(ctx, source)
        return opts, default

    @app.callback(
        Output("conv-variants", "options"),
        Output("conv-variants", "value"),
        Output("conv-y", "options"),
        Output("conv-y", "value"),
        Output("conv-status", "children"),
        Input("tabs", "value"),
        Input("conv-mode", "value"),
        Input("conv-source", "value"),
        Input("conv-metric", "value"),
        State("conv-variants", "value"),
        State("conv-y", "value"),
        prevent_initial_call=False,
    )
    def _update_variant_y(active_tab, mode, source, metric, current_variants, current_y):
        if active_tab != "convergence":
            raise PreventUpdate
        if mode == _MODE_DASHBOARD:
            v_opts, v_default, dash_msg = _variant_options_dashboard(ctx)
            v_values = _option_values(v_opts)
            cur = [str(v) for v in (current_variants or []) if str(v) in v_values]
            variants = cur if cur else v_default
            y_opts, y_default = _y_options_from_files(ctx, source, metric)
            y_values = _option_values(y_opts)
            y = str(current_y) if current_y is not None and str(current_y) in y_values else y_default
            status = html.Div([
                html.Div(dash_msg),
                html.Div("Recommended dashboard uses pooled files for PMF change/uncertainty and pooled or per-replica files for replica influence/coverage."),
                html.Div("Single-metric Source/Metric/Y controls are ignored in recommended mode."),
            ])
            return v_opts, variants, y_opts, y, status
        metric_opts, _metric_default, metric_msg = _metric_options_from_files(ctx, source)
        metric_values = _option_values(metric_opts)
        metric_value = str(metric) if metric is not None and str(metric) in metric_values else None
        v_opts, v_default, v_msg = _variant_options_from_files(ctx, source, metric_value)
        v_values = _option_values(v_opts)
        cur = [str(v) for v in (current_variants or []) if str(v) in v_values]
        variants = cur if cur else v_default
        y_opts, y_default = _y_options_from_files(ctx, source, metric_value)
        y_values = _option_values(y_opts)
        y = str(current_y) if current_y is not None and str(current_y) in y_values else y_default
        status = html.Div([
            html.Div(metric_msg),
            html.Div(v_msg),
            html.Div("Plotting loads only selected variant files; clearing the variant dropdown intentionally loads nothing."),
        ])
        return v_opts, variants, y_opts, y, status

    @app.callback(
        Output("conv-progress-fill", "style"),
        Output("conv-progress-label", "children"),
        Input("conv-progress-poll", "n_intervals"),
        Input("tabs", "value"),
        prevent_initial_call=False,
    )
    def _update_progress(_n, active_tab):
        if active_tab != "convergence":
            raise PreventUpdate
        snap = data_io.get_progress()
        phase = str(snap.get("phase") or "")
        pct = float(snap.get("pct") or 0.0)
        total = int(snap.get("total") or 0)
        done = int(snap.get("done") or 0)
        elapsed = float(snap.get("elapsed_s") or 0.0)
        eta = float(snap.get("eta_s") or 0.0)
        width = max(0.0, min(100.0, pct * 100.0))
        style = {"height": "100%", "width": f"{width:.1f}%", "background": "linear-gradient(90deg, #6366f1, #a78bfa)", "borderRadius": "5px"}
        if total <= 0 or "convergence" not in phase:
            return style, "Convergence loader idle."
        label = f"{phase}: {done}/{total} files ({width:.1f}%), elapsed {elapsed:.1f}s"
        if eta > 1 and done < total:
            label += f", eta {eta:.1f}s"
        return style, label

    @app.callback(
        Output("convergence-graph", "figure"),
        Output("conv-summary", "children"),
        Input("tabs", "value"),
        Input("conv-mode", "value"),
        Input("conv-source", "value"),
        Input("conv-metric", "value"),
        Input("conv-variants", "value"),
        Input("conv-y", "value"),
        Input("conv-eps", "value"),
        Input("conv-eps-rel", "value"),
        Input("conv-window-frac", "value"),
        Input("theme-store", "data"),
        prevent_initial_call=False,
    )
    def _conv_plot(
        active_tab: Optional[str],
        mode: Optional[str],
        source: Optional[str],
        metric: Optional[str],
        variants_sel: Optional[Sequence[str]],
        ycol: Optional[str],
        eps: Optional[float],
        eps_rel: Optional[float],
        window_frac: Optional[float],
        theme,
    ):
        if active_tab != "convergence":
            raise PreventUpdate
        if mode == _MODE_DASHBOARD:
            fig_rec, summary_rec = _recommended_dashboard(ctx, variants_sel)
            apply_theme(fig_rec, theme)
            return fig_rec, summary_rec
        src_label = "per-replica" if _source_key(source) == "per_replica" else "pooled"
        if isinstance(variants_sel, str):
            variants_sel = [variants_sel]
        variants_sel = [str(v) for v in (variants_sel or []) if v is not None]
        if not variants_sel:
            return apply_theme(error_fig("Select at least one variant to plot."), theme), html.Div("No variants selected. Nothing was loaded.")

        d = _load_selected_convergence(ctx, source, variants_sel, ycol)
        if d is None or d.empty:
            return apply_theme(error_fig(f"No {src_label} convergence rows for selected variant(s)."), theme), html.Div(
                f"No {src_label} convergence rows loaded for: {', '.join(variants_sel[:12])}"
            )
        d = d.copy()
        if metric and "metric" in d.columns:
            d = d[d["metric"].astype(str) == str(metric)]
        if d.empty:
            return apply_theme(error_fig("No rows for current selection."), theme), html.Div("No rows for current selection.")
        x_candidates = [c for c in ["checkpoint_frac", "frac", "fraction", "checkpoint", "n_frames"] if c in d.columns]
        if not x_candidates:
            return apply_theme(error_fig("Convergence file missing checkpoint/frac columns."), theme), html.Div("Convergence file missing checkpoint/frac columns.")
        xcol = x_candidates[0]
        d[xcol] = pd.to_numeric(d[xcol], errors="coerce")
        if not ycol or ycol not in d.columns:
            y_opts, y_default = _conv_default_columns(d)
            if not y_default:
                return apply_theme(error_fig("No numeric convergence columns to plot."), theme), html.Div("No numeric convergence columns to plot.")
            ycol = y_default[0]
        d[ycol] = pd.to_numeric(d[ycol], errors="coerce")

        fig = go.Figure()
        summary_rows: List[Dict[str, Any]] = []
        grouped_variants = d.groupby("variant", sort=False) if "variant" in d.columns else [("all", d)]
        use_replica_agg = _source_key(source) == "per_replica" and ("replica" in d.columns or "repl" in d.columns)
        rep_col = "replica" if "replica" in d.columns else ("repl" if "repl" in d.columns else None)

        for v, sub in grouped_variants:
            sub = sub.copy()
            curve_df: pd.DataFrame
            if use_replica_agg and rep_col is not None:
                agg = sub.groupby(xcol, observed=True)[ycol].agg(["mean", "std"]).reset_index().sort_values(xcol)
                if agg.empty:
                    continue
                std_vals = agg["std"].fillna(0.0).to_numpy()
                fig.add_trace(go.Scatter(
                    x=agg[xcol], y=agg["mean"], mode="lines+markers", name=str(v),
                    error_y=dict(type="data", array=std_vals, visible=True),
                    hovertemplate=("variant=%{text}<br>" + f"{xcol}=%{{x:.3f}}<br>" + f"{ycol}=%{{y:.4g}}" + "<extra></extra>"),
                    text=[v] * len(agg),
                ))
                curve_df = agg.rename(columns={"mean": ycol})
            else:
                keep = [xcol, ycol] + (["metric"] if "metric" in sub.columns else [])
                sub = sub[keep].dropna().sort_values(xcol)
                if sub.empty:
                    continue
                fig.add_trace(go.Scatter(
                    x=sub[xcol], y=sub[ycol], mode="lines+markers", name=str(v),
                    hovertemplate=("variant=%{text}<br>" + f"{xcol}=%{{x:.3f}}<br>" + f"{ycol}=%{{y:.4g}}" + "<extra></extra>"),
                    text=[v] * len(sub),
                ))
                curve_df = sub
            met = metric or (str(sub["metric"].iloc[0]) if "metric" in sub.columns and len(sub) else "")
            stats = _compute_conv_stats(curve_df, xcol=xcol, ycol=ycol, eps=eps, eps_rel=eps_rel, window_frac=window_frac)
            summary_rows.append({
                "variant": str(v), "metric": met, "ycol": ycol,
                "final": stats["final"], "delta_total": stats["delta_total"],
                "t_conv": stats["t_conv"], "converged": stats["converged"],
            })

        metric_lbl = metric_display_label(metric) if metric else "all metrics"
        fig.update_layout(
            template="plotly_dark",
            title=f"Convergence — {src_label} — {metric_lbl} — y={ycol}",
            xaxis_title=xcol,
            yaxis_title=ycol,
            legend_title="variant",
            margin=dict(l=40, r=20, t=60, b=40),
        )

        if not summary_rows:
            summary_children = html.Div("No convergence stats (nothing to plot).")
        else:
            sdf = pd.DataFrame(summary_rows)
            cols = ["variant", "metric", "ycol", "final", "delta_total", "t_conv", "converged"]
            for c in ["final", "delta_total", "t_conv"]:
                if c in sdf.columns:
                    sdf[c] = sdf[c].map(lambda x: "" if pd.isna(x) else f"{float(x):.4g}")
            summary_children = html.Div([
                html.Div(f"Loaded {src_label} convergence rows for {len(variants_sel):,} selected variant(s): {len(d):,}", style={"marginBottom": "6px", "fontWeight": 600}),
                html.Table([
                    html.Thead(html.Tr([html.Th(c) for c in cols])),
                    html.Tbody([
                        html.Tr([html.Td(sdf.iloc[i][c]) for c in cols])
                        for i in range(min(len(sdf), 24))
                    ]),
                ], style={"width": "100%", "borderCollapse": "collapse"}),
            ])
        apply_theme(fig, theme)
        return fig, summary_children

    @app.callback(
        Output("convergence-hover-info", "children"),
        Input("convergence-graph", "hoverData"),
        State("convergence-graph", "figure"),
        prevent_initial_call=True,
    )
    def _convergence_hover(hover_data, current_fig):
        if not hover_data:
            raise PreventUpdate
        pts = hover_data.get("points", [])
        if not pts:
            raise PreventUpdate
        pt = pts[0]
        curve_num = pt.get("curveNumber")
        name = "?"
        if curve_num is not None and current_fig:
            try:
                name = (current_fig.get("data") or [])[curve_num].get("name", "?") or "?"
            except (IndexError, AttributeError, TypeError):
                pass
        x_val = pt.get("x")
        y_val = pt.get("y")
        x_str = f"{x_val:.3g}" if isinstance(x_val, (int, float)) else str(x_val or "")
        y_str = f"{y_val:.4g}" if isinstance(y_val, (int, float)) else str(y_val or "")
        return f"{name}  x={x_str}  y={y_str}"
