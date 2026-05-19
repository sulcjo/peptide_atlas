from __future__ import annotations

"""
Fast, shared per-variant PMF retrieval for interactive viewers.

The PCA and UMAP curve viewers need small slices of the PMF data: a handful of
variants and usually one metric. Loading ctx.pmf_df for that interaction can
force the entire PMF table into memory. This module instead prefers projected,
predicate-filtered reads from per-variant parquet files and caches those small
variant/metric slices.
"""

from functools import lru_cache
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Any, Iterable, Optional, Sequence
import os

import pandas as pd
import numpy as np

from .io import _resolve_layout

try:  # optional, but used by pandas/pyarrow parquet filters when available
    import pyarrow.parquet as _pq  # type: ignore
    _HAS_PYARROW = True
except Exception:  # pragma: no cover - optional dependency
    _pq = None
    _HAS_PYARROW = False


PMF_F_ALIASES: tuple[str, ...] = (
    "F_kJ_mol",
    "pmf_F_kJmol",
    "F",
    "y",
    "free_energy",
    "F_kJmol",
)
PMF_P_ALIASES: tuple[str, ...] = (
    "P",
    "p",
    "probability",
    "prob",
    "density",
    "P_mass",
)
PMF_CI_ALIASES: tuple[str, ...] = (
    "F_ci_lo_kJ_mol",
    "F_ci_hi_kJ_mol",
    "F_ci_lo_kJmol",
    "F_ci_hi_kJmol",
)
PMF_BASE_COLS: tuple[str, ...] = (
    "variant",
    "metric",
    "x",
    *PMF_P_ALIASES,
    *PMF_F_ALIASES,
    *PMF_CI_ALIASES,
)

# Minimal column set for interactive PMF curve plotting. Keeping this separate
# from PMF_BASE_COLS avoids reading probability/CI aliases when a viewer only
# needs an x/F line.
PMF_PLOT_COLS: tuple[str, ...] = (
    "variant",
    "metric",
    "x",
    *PMF_F_ALIASES,
)


def _as_list_str(values: Optional[Iterable[Any]]) -> list[str]:
    if values is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for v in values:
        if v is None:
            continue
        s = str(v).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _metric_sort_key(v: str) -> tuple[str, str]:
    return (str(v).lower(), str(v))


def _ctx_data_dir(ctx_or_data_dir: Any) -> Any:
    return getattr(ctx_or_data_dir, "data_dir", ctx_or_data_dir)


def _base_dir(ctx_or_data_dir: Any) -> Path:
    base, _ = _resolve_layout(_ctx_data_dir(ctx_or_data_dir))
    return Path(base)


@lru_cache(maxsize=4096)
def _schema_columns(path_s: str, mtime_ns: int, size: int) -> tuple[str, ...]:
    path = Path(path_s)
    if _HAS_PYARROW:
        try:
            return tuple(_pq.ParquetFile(str(path)).schema.names)  # type: ignore[union-attr]
        except Exception:
            pass
    try:
        return tuple(pd.read_parquet(str(path), engine="pyarrow").columns)
    except Exception:
        try:
            return tuple(pd.read_parquet(str(path)).columns)
        except Exception:
            return tuple()


def _file_token(path: Path) -> tuple[str, int, int]:
    try:
        st = path.stat()
        return str(path), int(st.st_mtime_ns), int(st.st_size)
    except OSError:
        return str(path), 0, 0


def _columns_for_file(path: Path, requested: Sequence[str]) -> list[str] | None:
    path_s, mt, sz = _file_token(path)
    schema = set(_schema_columns(path_s, mt, sz))
    if not schema:
        return None
    cols = [c for c in requested if c in schema]
    # Always keep filter/identity columns when possible.
    for c in ("variant", "metric", "x"):
        if c in schema and c not in cols:
            cols.insert(0, c)
    return cols or None


@lru_cache(maxsize=8192)
def _metrics_for_parquet_file_cached(path_s: str, mtime_ns: int, size: int) -> tuple[str, ...]:
    """Return metric names in a PMF parquet file, cached by file token."""
    path = Path(path_s)
    schema = set(_schema_columns(path_s, mtime_ns, size))
    if "metric" not in schema:
        return tuple()
    vals: set[str] = set()
    try:
        if _HAS_PYARROW:
            table = _pq.read_table(str(path), columns=["metric"])  # type: ignore[union-attr]
            ser = table.column("metric").to_pandas()
        else:
            ser = pd.read_parquet(str(path), columns=["metric"])["metric"]
        vals.update(str(x).strip() for x in pd.Series(ser).dropna().unique() if str(x).strip())
    except Exception:
        try:
            df = pd.read_parquet(str(path), columns=["metric"])
            vals.update(str(x).strip() for x in df["metric"].dropna().unique() if str(x).strip())
        except Exception:
            return tuple()
    return tuple(sorted(vals, key=_metric_sort_key))


def _default_pmf_viewer_backend() -> str:
    """Default interactive PMF loading backend.

    Multi-worker process loading is the default because selected variants are
    normally independent parquet files.  ``multiwalker`` is accepted as a
    forgiving alias for the intended multi-worker/process mode.
    """
    raw = os.environ.get("PEPTIDE_DASH_PMF_VIEWER_BACKEND", "multiworker")
    backend = str(raw or "multiworker").strip().lower()
    aliases = {
        "multiworker": "process",
        "multi-worker": "process",
        "multi_workers": "process",
        "multiworkers": "process",
        "multiwalker": "process",
        "multi-walker": "process",
        "multiprocess": "process",
        "multiprocessing": "process",
        "processes": "process",
        "mp": "process",
        "threads": "thread",
        "threaded": "thread",
        "single": "serial",
        "none": "serial",
        "off": "serial",
        "false": "serial",
        "0": "serial",
    }
    return aliases.get(backend, backend)


def _default_pmf_viewer_workers(n_tasks: int) -> int:
    """Return default worker count for selected-variant PMF reads."""
    n_tasks = max(1, int(n_tasks or 1))
    cpu = max(1, int(os.cpu_count() or 1))
    # Use multiple workers by default on multi-core machines, but cap to avoid
    # spawning a silly number of short-lived parquet readers for small lasso
    # selections.  Environment override remains available for batch boxes.
    default_workers = min(max(2, cpu), 8, n_tasks) if n_tasks > 1 and cpu > 1 else 1
    raw = os.environ.get("PEPTIDE_DASH_PMF_VIEWER_WORKERS")
    if raw is None or str(raw).strip() == "":
        return default_workers
    try:
        requested = int(raw)
    except Exception:
        return default_workers
    return min(max(1, requested), n_tasks)


def downsample_pmf_curves(pmf_df: pd.DataFrame, max_points_per_curve: Optional[int]) -> pd.DataFrame:
    """Downsample PMF curves for interactive plotting while preserving minima."""
    if not isinstance(pmf_df, pd.DataFrame) or pmf_df.empty:
        return pd.DataFrame()
    if max_points_per_curve is None or int(max_points_per_curve) <= 0:
        return pmf_df
    nmax = int(max_points_per_curve)
    if "x" not in pmf_df.columns or len(pmf_df) <= nmax:
        return pmf_df
    y_col = next((c for c in ("F_kJ_mol", *PMF_F_ALIASES) if c in pmf_df.columns), None)
    group_cols = [c for c in ("variant", "metric") if c in pmf_df.columns]
    if not group_cols:
        group_cols = ["variant"] if "variant" in pmf_df.columns else []
    if not group_cols:
        take = np.linspace(0, len(pmf_df) - 1, min(len(pmf_df), nmax), dtype=int)
        return pmf_df.iloc[take].copy()

    parts: list[pd.DataFrame] = []
    for _, grp in pmf_df.groupby(group_cols, sort=False, observed=True):
        if len(grp) <= nmax:
            parts.append(grp)
            continue
        g = grp.sort_values("x", kind="mergesort")
        idx = set(np.linspace(0, len(g) - 1, max(2, nmax), dtype=int).tolist())
        idx.add(0)
        idx.add(len(g) - 1)
        if y_col is not None:
            yy = pd.to_numeric(g[y_col], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(yy).any():
                idx.add(int(np.nanargmin(yy)))
        take = sorted(i for i in idx if 0 <= i < len(g))[:nmax]
        parts.append(g.iloc[take])
    if not parts:
        return pd.DataFrame(columns=pmf_df.columns)
    return pd.concat(parts, ignore_index=True, copy=False)


@lru_cache(maxsize=2048)
def _read_variant_parquet_cached(
    path_s: str,
    metric: str,
    columns_key: tuple[str, ...],
    mtime_ns: int,
    size: int,
) -> pd.DataFrame:
    path = Path(path_s)
    schema = set(_schema_columns(path_s, mtime_ns, size))
    columns = [c for c in columns_key if c in schema]
    if not columns:
        columns = None  # type: ignore[assignment]
    filters = [[("metric", "==", metric)]] if metric and "metric" in schema else None
    try:
        return pd.read_parquet(str(path), columns=columns, filters=filters)
    except Exception:
        try:
            df = pd.read_parquet(str(path), columns=columns)
        except Exception:
            return pd.DataFrame()
        if metric and "metric" in df.columns:
            df = df[df["metric"].astype(str) == str(metric)]
        return df


def _read_variant_parquet_process_worker(args: tuple[str, str, str, tuple[str, ...], int, int]) -> pd.DataFrame:
    """Read one variant/metric PMF slice in a worker process.

    This top-level function is intentionally pickleable.  It receives only file
    paths and primitive values, not the Dash context or store object.
    """
    variant, path_s, metric, columns_key, mtime_ns, size = args
    try:
        df = _read_variant_parquet_cached(path_s, metric, columns_key, mtime_ns, size).copy()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    if "variant" not in df.columns:
        df.insert(0, "variant", str(variant))
    else:
        df["variant"] = df["variant"].astype(str)
    return df


def _parallel_read_variant_tasks(
    tasks: list[tuple[str, str, str, tuple[str, ...], int, int]],
    *,
    workers: int,
    backend: str,
) -> tuple[list[pd.DataFrame], str]:
    """Read variant parquet tasks with a process or thread pool.

    backend: 'process', 'thread', or 'serial'.  Process loading is the default
    for interactive viewers because each selected variant is an independent file
    slice.  If multiprocessing fails, callers can fall back to threads/serial.
    """
    if not tasks:
        return [], "none"
    workers = min(max(1, int(workers)), max(1, len(tasks)))
    backend = _default_pmf_viewer_backend() if backend in {None, ""} else str(backend).strip().lower()
    backend = _default_pmf_viewer_backend() if backend == "default" else backend
    if backend in {"multiworker", "multi-worker", "multi_workers", "multiworkers", "multiwalker", "multi-walker"}:
        backend = "process"

    if workers <= 1 or backend in {"0", "false", "off", "none", "serial", "single"}:
        frames = [_read_variant_parquet_process_worker(t) for t in tasks]
        return [df for df in frames if isinstance(df, pd.DataFrame) and not df.empty], "serial"

    if backend in {"process", "processes", "multiprocess", "multiprocessing", "mp"}:
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                frames = list(pool.map(_read_variant_parquet_process_worker, tasks, chunksize=1))
            return [df for df in frames if isinstance(df, pd.DataFrame) and not df.empty], "process"
        except Exception:
            # Fall through to threads; Dash should keep working even on platforms
            # where request-time process spawning is restricted.
            pass

    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            frames = list(pool.map(_read_variant_parquet_process_worker, tasks))
        return [df for df in frames if isinstance(df, pd.DataFrame) and not df.empty], "thread"
    except Exception:
        frames = [_read_variant_parquet_process_worker(t) for t in tasks]
        return [df for df in frames if isinstance(df, pd.DataFrame) and not df.empty], "serial"


def normalize_pmf_curve_columns(pmf_df: pd.DataFrame) -> pd.DataFrame:
    """Normalize common PMF curve column aliases used by plotting/vectorization."""
    if not isinstance(pmf_df, pd.DataFrame) or pmf_df.empty:
        return pd.DataFrame()
    needs_p = "P" not in pmf_df.columns
    needs_f = "F_kJ_mol" not in pmf_df.columns
    if not needs_p and not needs_f:
        return pmf_df
    d = pmf_df.copy()
    if needs_p:
        p_col = next((c for c in PMF_P_ALIASES if c in d.columns), None)
        if p_col is not None:
            d["P"] = pd.to_numeric(d[p_col], errors="coerce")
    if needs_f:
        f_col = next((c for c in PMF_F_ALIASES if c in d.columns), None)
        if f_col is not None:
            d["F_kJ_mol"] = pd.to_numeric(d[f_col], errors="coerce")
    return d


class VariantPmfStore:
    """Cached PMF slice reader for one GLOBAL_DATA directory."""

    def __init__(self, ctx_or_data_dir: Any):
        self.ctx = ctx_or_data_dir if hasattr(ctx_or_data_dir, "data_dir") else None
        self.base = _base_dir(ctx_or_data_dir)
        self._variant_files: Optional[dict[str, Path]] = None
        self._metrics_cache: dict[tuple[str, ...], tuple[str, ...]] = {}

    def variant_files(self) -> dict[str, Path]:
        if self._variant_files is None:
            files: dict[str, Path] = {}
            for p in self.base.glob("*_pmf.parquet"):
                name = p.name
                if not name.endswith("_pmf.parquet"):
                    continue
                variant = name[: -len("_pmf.parquet")]
                if variant:
                    files[str(variant)] = p
            self._variant_files = files
        return self._variant_files

    def variants(self) -> list[str]:
        return sorted(self.variant_files().keys())

    def _global_pmf_paths(self) -> list[Path]:
        out: list[Path] = []
        for name in ("pmf.parquet",):
            p = self.base / name
            if p.exists():
                out.append(p)
        return out

    def available_metrics(self, variants: Optional[Sequence[Any]] = None, *, max_scan: int = 64) -> list[str]:
        requested = tuple(sorted(_as_list_str(variants)))
        if requested in self._metrics_cache:
            return list(self._metrics_cache[requested])

        metrics: set[str] = set()
        files = self.variant_files()
        if requested:
            scan_paths = [files[v] for v in requested if v in files]
        else:
            scan_paths = list(files.values())[: int(max_scan)]

        for p in scan_paths:
            try:
                metrics.update(_metrics_for_parquet_file_cached(*_file_token(p)))
            except Exception:
                continue

        if not metrics:
            for p in self._global_pmf_paths():
                try:
                    metrics.update(_metrics_for_parquet_file_cached(*_file_token(p)))
                    if metrics:
                        break
                except Exception:
                    continue

        # Last-resort fallback: use an already available context table. This may
        # be expensive if it triggers lazy loading, so only do it when files did
        # not provide an answer.
        if not metrics and self.ctx is not None:
            try:
                df = getattr(self.ctx, "pmf_df", pd.DataFrame())
                if isinstance(df, pd.DataFrame) and not df.empty and "metric" in df.columns:
                    if requested and "variant" in df.columns:
                        df = df[df["variant"].astype(str).isin(requested)]
                    metrics.update(str(x).strip() for x in df["metric"].dropna().unique() if str(x).strip())
            except Exception:
                pass

        ans = tuple(sorted(metrics, key=_metric_sort_key))
        self._metrics_cache[requested] = ans
        return list(ans)

    def _read_one_variant(self, variant: str, metric: Optional[str], columns: Sequence[str]) -> pd.DataFrame:
        files = self.variant_files()
        p = files.get(str(variant))
        if p is None or not p.exists():
            return pd.DataFrame()
        path_s, mt, sz = _file_token(p)
        df = _read_variant_parquet_cached(path_s, str(metric or ""), tuple(columns), mt, sz).copy()
        if df.empty:
            return df
        if "variant" not in df.columns:
            df.insert(0, "variant", str(variant))
        else:
            df["variant"] = df["variant"].astype(str)
        return df

    def _read_global(self, variants: Sequence[str], metric: Optional[str], columns: Sequence[str]) -> pd.DataFrame:
        paths = self._global_pmf_paths()
        if not paths:
            return pd.DataFrame()
        p = paths[0]
        schema = set(_schema_columns(*_file_token(p)))
        use_cols = [c for c in columns if c in schema] or None
        filters = []
        if variants and "variant" in schema:
            filters.append(("variant", "in", list(map(str, variants))))
        if metric and "metric" in schema:
            filters.append(("metric", "==", str(metric)))
        try:
            df = pd.read_parquet(str(p), columns=use_cols, filters=filters or None)
        except Exception:
            try:
                df = pd.read_parquet(str(p), columns=use_cols)
            except Exception:
                return pd.DataFrame()
            if variants and "variant" in df.columns:
                df = df[df["variant"].astype(str).isin(list(map(str, variants)))]
            if metric and "metric" in df.columns:
                df = df[df["metric"].astype(str) == str(metric)]
        return df

    def load(
        self,
        variants: Sequence[Any],
        metric: Optional[str] = None,
        *,
        columns: Optional[Sequence[str]] = None,
        max_points_per_curve: Optional[int] = None,
    ) -> pd.DataFrame:
        vars_use = _as_list_str(variants)
        if not vars_use:
            return pd.DataFrame()
        cols = tuple(dict.fromkeys([*(columns or PMF_BASE_COLS), "variant", "metric", "x"]))

        workers = _default_pmf_viewer_workers(len(vars_use))
        backend_requested = _default_pmf_viewer_backend()
        backend_used = "none"
        frames: list[pd.DataFrame] = []
        files = self.variant_files()
        if files:
            tasks: list[tuple[str, str, str, tuple[str, ...], int, int]] = []
            for v in vars_use:
                p = files.get(str(v))
                if p is None or not p.exists():
                    continue
                path_s, mt, sz = _file_token(p)
                tasks.append((str(v), path_s, str(metric or ""), cols, mt, sz))
            frames, backend_used = _parallel_read_variant_tasks(tasks, workers=workers, backend=backend_requested)

        if not frames:
            df = self._read_global(vars_use, metric, cols)
            if isinstance(df, pd.DataFrame) and not df.empty:
                frames.append(df)
                backend_used = "global-parquet"

        # Last-resort fallback: already-loaded context table.
        if not frames and self.ctx is not None:
            try:
                df = getattr(self.ctx, "pmf_df", pd.DataFrame())
                if isinstance(df, pd.DataFrame) and not df.empty and "variant" in df.columns:
                    df = df[df["variant"].astype(str).isin(vars_use)].copy()
                    if metric and "metric" in df.columns:
                        df = df[df["metric"].astype(str) == str(metric)]
                    keep = [c for c in cols if c in df.columns]
                    if keep:
                        df = df[keep]
                    if not df.empty:
                        frames.append(df)
            except Exception:
                pass

        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames, ignore_index=True, copy=False)
        out = normalize_pmf_curve_columns(out)
        out = downsample_pmf_curves(out, max_points_per_curve)
        try:
            out.attrs["pmf_loader_backend"] = backend_used
            out.attrs["pmf_loader_workers"] = workers
            out.attrs["pmf_loader_variants_requested"] = len(vars_use)
            out.attrs["pmf_loader_metric"] = str(metric or "")
        except Exception:
            pass
        return out


_STORES: dict[str, VariantPmfStore] = {}


def get_variant_pmf_store(ctx_or_data_dir: Any) -> VariantPmfStore:
    base = str(_base_dir(ctx_or_data_dir))
    store = _STORES.get(base)
    if store is None:
        store = VariantPmfStore(ctx_or_data_dir)
        _STORES[base] = store
    return store


def available_pmf_metrics(ctx_or_data_dir: Any, variants: Optional[Sequence[Any]] = None) -> list[str]:
    return get_variant_pmf_store(ctx_or_data_dir).available_metrics(variants)


def available_pmf_variants(ctx_or_data_dir: Any) -> list[str]:
    return get_variant_pmf_store(ctx_or_data_dir).variants()


def load_variant_pmfs(
    ctx_or_data_dir: Any,
    variants: Sequence[Any],
    metric: Optional[str] = None,
    *,
    columns: Optional[Sequence[str]] = None,
    max_points_per_curve: Optional[int] = None,
) -> pd.DataFrame:
    return get_variant_pmf_store(ctx_or_data_dir).load(
        variants,
        metric=metric,
        columns=columns,
        max_points_per_curve=max_points_per_curve,
    )
