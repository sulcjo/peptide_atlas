from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import os
import sys
import time
import hashlib
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd

# ---------- config ----------

# I/O is the bottleneck, but at 64 workers we were stacking 64 full
# in-memory DataFrames concurrently — a 64 GB memory blow-up on
# datasets whose raw size is only a few GB.  4 workers pipelines
# reads without making peak memory scale with CPU count.  Override
# via PEPTIDE_DASH_IO_WORKERS if the host has plenty of RAM.
_MAX_WORKERS = int(os.environ.get("PEPTIDE_DASH_IO_WORKERS", "4"))
_ENABLE_CONSOLE_PROGRESS = True
_PROGRESS_BAR_WIDTH = 40

# Cache pyarrow imports at module level — importing inside functions on every
# call hits Python's import lock even when the module is already cached.
try:
    import pyarrow as _pa
    import pyarrow.parquet as _pq
    import pyarrow.dataset as _pads
    import pyarrow.csv as _pacsv
    _HAS_PYARROW = True
except ImportError:
    _HAS_PYARROW = False


# ---------- console progress ----------

class _Progress:
    def __init__(self) -> None:
        self.lock = Lock()
        self.total: int = 0
        self.done: int = 0
        self.phase: str = ""
        self.start_ts: Optional[float] = None
        self.finished: bool = False

    def start(self, total: int, phase: str) -> None:
        with self.lock:
            self.total = int(total)
            self.done = 0
            self.phase = phase
            self.start_ts = time.time()
            self.finished = False
        _print_progress(self.snapshot(), True)

    def tick(self, n: int = 1) -> None:
        with self.lock:
            self.done += int(n)
            if self.total > 0 and self.done >= self.total:
                self.finished = True
        _print_progress(self.snapshot())

    def finish(self) -> None:
        with self.lock:
            self.done = self.total
            self.finished = True
        _print_progress(self.snapshot(), True)
        _write_progress_line("")

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            total = self.total
            done = min(self.done, total)
            phase = self.phase
            st = self.start_ts
            fin = self.finished
        elapsed = time.time() - (st or time.time())
        pct = (done / total) if total > 0 else 1.0
        eta = (elapsed / pct - elapsed) if (0 < pct < 1) else 0.0
        return {
            "phase": phase,
            "total": total,
            "done": done,
            "pct": pct,
            "elapsed_s": elapsed,
            "eta_s": eta,
            "finished": fin,
        }


_GLOBAL_PROGRESS = _Progress()


def get_progress() -> Dict[str, Any]:
    """Return a snapshot of the global loading progress (for callbacks)."""
    return _GLOBAL_PROGRESS.snapshot()


def _write_progress_line(s: str) -> None:
    if _ENABLE_CONSOLE_PROGRESS:
        try:
            sys.stdout.write(s)
            sys.stdout.flush()
        except Exception:
            pass


def _print_progress(snap: Dict[str, Any], force: bool = False) -> None:
    if not _ENABLE_CONSOLE_PROGRESS or snap["total"] <= 0:
        return
    width = _PROGRESS_BAR_WIDTH
    filled = int(round(width * snap["pct"]))
    bar = "█" * filled + "░" * (width - filled)
    msg = (
        f"\r[{bar}] {snap['done']}/{snap['total']} "
        f"({snap['pct'] * 100:5.1f}%)  {snap['phase']}  "
        f"elapsed {snap['elapsed_s']:5.1f}s"
    )
    if not snap["finished"] and snap["eta_s"] > 0:
        msg += f"  eta {snap['eta_s']:5.1f}s"
    _write_progress_line(msg)
    if snap["finished"] or force:
        sys.stdout.flush()



def _log_df_size(label: str, df: pd.DataFrame) -> None:
    """Log rows × cols × memory if PEPTIDE_DASH_IO_DEBUG=1 is set."""
    if os.environ.get("PEPTIDE_DASH_IO_DEBUG", "").strip() not in {"1", "true", "yes", "on"}:
        return
    try:
        mem_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
        print(
            f"[IO-DBG] {label}: {len(df):,} rows × {df.shape[1]} cols, "
            f"{mem_mb:.1f} MB",
            file=sys.stderr,
        )
    except Exception:
        pass


# ---------- path resolution ----------

def _resolve_layout(data_dir: str | os.PathLike) -> tuple[Path, Path]:
    """
    Returns (global_data_path, project_root).

    Accepts either:
      • path to GLOBAL_DATA
      • path whose child GLOBAL_DATA exists
      • naked directory with data files (then base_dir=data_dir itself)
    """
    p = Path(data_dir).expanduser().resolve()
    if (p / "GLOBAL_DATA").is_dir():
        return (p / "GLOBAL_DATA", p)
    if p.name == "GLOBAL_DATA" and p.is_dir():
        return (p, p.parent)
    return (p, p.parent if p.name == "GLOBAL_DATA" else p)


@contextmanager
def _with_legacy_cwd(data_dir: str | os.PathLike | None):
    old = os.getcwd()
    try:
        if data_dir is not None:
            _, chdir_target = _resolve_layout(data_dir)
            os.chdir(str(chdir_target))
        yield
    finally:
        os.chdir(old)


# ---------- discovery ----------

_FEATURE_PATTERNS = [
    "*_features.parquet",
    "*_features.csv.gz",
    "features_per_peptide_all.parquet",
    "features_per_peptide_all.csv.gz",
    "features_per_peptide_all.csv",
    "features_per_peptide.parquet",
    "features_per_peptide.csv.gz",
    "features_per_peptide.csv",
]


def _list_feature_paths(base_dir: Path, max_files: int | None = None) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pat in _FEATURE_PATTERNS:
        for p in base_dir.glob(pat):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(rp)

    # If BATCH_ANA emitted both the safe PMF-core feature file
    # (features_per_peptide.*) and the full all-column table
    # (features_per_peptide_all.*), prefer the full table for file-based
    # loading. Otherwise sequence/letter-space descriptors live only in the
    # all-column file and the UMAP sequence mode appears empty. Avoid loading
    # both monoliths, because that would duplicate variants row-wise.
    all_mono = {
        (base_dir / "features_per_peptide_all.parquet").resolve(),
        (base_dir / "features_per_peptide_all.csv.gz").resolve(),
        (base_dir / "features_per_peptide_all.csv").resolve(),
    }
    core_mono = {
        (base_dir / "features_per_peptide.parquet").resolve(),
        (base_dir / "features_per_peptide.csv.gz").resolve(),
        (base_dir / "features_per_peptide.csv").resolve(),
    }
    if any(p in all_mono for p in out):
        out = [p for p in out if p not in core_mono]

    if isinstance(max_files, int) and max_files > 0:
        return out[:max_files]
    return out


def _find_files(base_dir: Path, patterns: list[str]) -> list[Path]:
    seen: set[Path] = set()
    out: list[Path] = []
    for pat in patterns:
        for p in base_dir.glob(pat):
            rp = p.resolve()
            if rp not in seen:
                seen.add(rp)
                out.append(rp)
    return out


def _filter_curve_paths(key: str, paths: list[Path]) -> list[Path]:
    """Remove accidental cross-matches from broad legacy curve globs.

    The RMSF/replica pattern ``*_replica.parquet`` also matches PMF,
    cumulative, and convergence replica files.  On large GLOBAL_DATA folders
    this can make unrelated tabs load the wrong universe of files.
    """
    if key != "rmsf":
        return paths
    bad_suffixes = (
        "_pmf_replica.parquet", "_pmf_replica.csv.gz", "_pmf_replica.csv",
        "_cumulative_replica.parquet", "_cumulative_replica.csv.gz", "_cumulative_replica.csv",
        "_cum_replica.parquet", "_cum_replica.csv.gz", "_cum_replica.csv",
        "_convergence_replica.parquet", "_convergence_replica.csv.gz", "_convergence_replica.csv",
    )
    return [p for p in paths if not p.name.endswith(bad_suffixes)]


# ---------- low-level readers ----------

def _read_parquet_cols(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """Read a parquet file, optionally projecting to a column subset.

    Every frame is dtype-optimized before return so the loader pipeline never
    holds an un-squeezed copy in memory.
    """
    if _HAS_PYARROW:
        try:
            ds_ = _pads.dataset([str(path)])
            tbl = ds_.to_table(columns=columns) if columns else ds_.to_table()
            df = tbl.to_pandas(use_threads=False, split_blocks=True, self_destruct=True)
        except Exception:
            tbl = _pq.read_table(path, columns=columns)
            df = tbl.to_pandas(use_threads=False, split_blocks=True, self_destruct=True)
        return _optimize_curves_df(df)
    return _optimize_curves_df(pd.read_parquet(path, columns=columns))


def _read_csv_cols(path: Path, columns: list[str] | None = None) -> pd.DataFrame:
    """CSV reader with dtype downcast applied before returning."""
    suff = "".join(path.suffixes).lower()
    if _HAS_PYARROW:
        try:
            # 16 MB blocks — cuts peak memory vs 64 MB; throughput loss is negligible.
            read_opts = _pacsv.ReadOptions(use_threads=False, block_size=1 << 24)
            conv = _pacsv.ConvertOptions(include_columns=columns) if columns else _pacsv.ConvertOptions()
            tbl = _pacsv.read_csv(str(path), read_options=read_opts, convert_options=conv)
            return _optimize_curves_df(
                tbl.to_pandas(use_threads=False, split_blocks=True, self_destruct=True)
            )
        except Exception:
            pass
    kw: dict[str, Any] = {}
    if suff.endswith(".gz"):
        kw["compression"] = "gzip"
    return _optimize_curves_df(pd.read_csv(path, usecols=columns, **kw))


def _is_empty_df(df: pd.DataFrame) -> bool:
    """Fast emptiness check — avoids expensive dropna(how='all') on wide frames."""
    if df is None or not isinstance(df, pd.DataFrame):
        return True
    if len(df) == 0 or len(df.columns) == 0:
        return True
    return not bool(df.notna().any(axis=None))


def _drop_all_na_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Drop columns that are entirely NA — fast path using a boolean mask."""
    if df.empty:
        return df
    mask = df.notna().any(axis=0)
    if mask.all():
        return df
    return df.loc[:, mask]


def _read_any_df(path: Path) -> pd.DataFrame | None:
    suff = "".join(path.suffixes).lower()
    try:
        if suff.endswith(".parquet"):
            df = _read_parquet_cols(path, None)
        else:
            df = _read_csv_cols(path, None)
        df = _drop_all_na_cols(df)
        if _is_empty_df(df):
            return None
        return df
    except Exception:
        return None


def _concat_nonempty(frames: list[pd.DataFrame]) -> pd.DataFrame:
    frames = [f for f in frames if isinstance(f, pd.DataFrame) and len(f)]
    if not frames:
        return pd.DataFrame()
    # copy=False avoids an extra memory allocation when frames have compatible dtypes
    out = pd.concat(frames, ignore_index=True, copy=False)
    if hasattr(out.columns, "duplicated") and out.columns.duplicated().any():
        out = out.loc[:, ~out.columns.duplicated()]
    return out


# ---------- sampling (private; public alias at end) ----------

def _sample_variants_from_features(
    data_dir: str,
    frac: float,
    seed: int,
    variant_col: str = "variant",
) -> list[str] | None:
    base_dir, _ = _resolve_layout(data_dir)
    paths = _list_feature_paths(base_dir)
    if not paths:
        return None

    _GLOBAL_PROGRESS.start(len(paths), "scanning variants")

    def _one(p: Path):
        suff = "".join(p.suffixes).lower()
        try:
            if suff.endswith(".parquet") and _HAS_PYARROW:
                tbl = _pq.read_table(p, columns=[variant_col])
                return pd.Series(tbl[variant_col].to_pylist()).dropna().unique()
            else:
                if _HAS_PYARROW:
                    try:
                        conv = _pacsv.ConvertOptions(include_columns=[variant_col])
                        tbl = _pacsv.read_csv(str(p), convert_options=conv)
                        return pd.Series(tbl[variant_col].to_pylist()).dropna().unique()
                    except Exception:
                        pass
                acc: list[np.ndarray] = []
                for ch in pd.read_csv(p, usecols=[variant_col], chunksize=250_000):
                    acc.append(ch[variant_col].dropna().unique())
                return np.concatenate(acc, axis=0) if acc else None
        finally:
            _GLOBAL_PROGRESS.tick(1)

    uniq_chunks: list[np.ndarray] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futs = [ex.submit(_one, p) for p in paths]
        for arr in (f.result() for f in as_completed(futs)):
            if arr is not None and len(arr) > 0:
                uniq_chunks.append(arr)
    _GLOBAL_PROGRESS.finish()

    if not uniq_chunks:
        return None
    uniq = pd.unique(pd.Series(np.concatenate(uniq_chunks, axis=0)))
    if len(uniq) == 0:
        return None
    k = max(1, int(round(len(uniq) * float(frac))))
    return (
        pd.Series(uniq)
        .sample(n=min(k, len(uniq)), random_state=int(seed), replace=False)
        .tolist()
    )


# ---------- cache helpers ----------

def _cache_dir(default_root: str | None) -> Path:
    if default_root:
        return Path(default_root).expanduser().resolve()
    xdg = os.environ.get("XDG_CACHE_HOME")
    root = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return root / "peptide_dash"


def _cache_key(
    data_dir: str,
    variants: list[str] | None,
    usecols: list[str] | None,
    max_files: int | None,
) -> str:
    payload = {
        "dir": str(Path(data_dir).resolve()),
        "variants": sorted(list(map(str, variants or []))),
        "usecols": sorted(usecols or []) if usecols else [],
        "max_files": int(max_files or 0),
    }
    j = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(j).hexdigest()[:16]


def maybe_load_cache(
    cache_root: str | None,
    data_dir: str,
    variants: list[str] | None,
    usecols: list[str] | None,
    max_files: int | None,
) -> pd.DataFrame | None:
    cdir = _cache_dir(cache_root)
    cdir.mkdir(parents=True, exist_ok=True)
    key = _cache_key(data_dir, variants, usecols, max_files)
    p = cdir / f"{key}.parquet"
    if p.exists():
        try:
            return _read_parquet_cols(p, None)
        except Exception:
            return None
    return None


def write_cache(
    cache_root: str | None,
    data_dir: str,
    variants: list[str] | None,
    usecols: list[str] | None,
    max_files: int | None,
    df: pd.DataFrame,
) -> None:
    try:
        cdir = _cache_dir(cache_root)
        cdir.mkdir(parents=True, exist_ok=True)
        key = _cache_key(data_dir, variants, usecols, max_files)
        p = cdir / f"{key}.parquet"
        if _HAS_PYARROW:
            import pyarrow.parquet as pq
            tbl = _pa.Table.from_pandas(df, preserve_index=False)
            pq.write_table(
                tbl,
                p,
                compression="zstd",
                compression_level=3,
                use_dictionary=True,
                write_statistics=True,
            )
        else:
            df.to_parquet(p, index=False)
        print(f"[FASTBOOT] subset cache written: {p}")
    except Exception as e:
        print(f"[FASTBOOT] cache write failed: {e}")


# ---------- numeric sniffer ----------

def _is_numeric_dtype_pd(dtype) -> bool:
    try:
        return pd.api.types.is_numeric_dtype(dtype)
    except Exception:
        return False


def sniff_numeric_columns(
    data_dir: str,
    max_files: int = 3,
    max_numerics: int = 200,
    variant_col: str = "variant",
) -> list[str]:
    """
    Sniff numeric column names.  Tries SQLite summary pivot first (instant),
    falls back to file schema sniffing.
    """
    base_dir, _ = _resolve_layout(data_dir)

    # Fast path: derive column names from SQLite summary pivot
    db_path = base_dir / "global_peptide_analysis.sqlite"
    if db_path.is_file():
        try:
            with sqlite3.connect(str(db_path)) as con:
                existing = {r[0] for r in con.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()}
                if "summary" in existing:
                    # Column names come from metric values × stat columns
                    metrics = [r[0] for r in con.execute(
                        "SELECT DISTINCT metric FROM summary WHERE metric IS NOT NULL"
                    ).fetchall()]
                    stat_cols = [r[1] for r in con.execute("PRAGMA table_info(summary)").fetchall()
                                 if r[1] not in ("variant", "metric")]
                    cols = [variant_col] + [f"{m}__{s}" for m in metrics for s in stat_cols]
                    return list(dict.fromkeys(cols))[:max_numerics + 1]
        except Exception:
            pass

    # Fallback: file schema sniff
    paths = _list_feature_paths(base_dir, max_files=max_files)
    numerics: list[str] = []
    seen = {variant_col}

    _NUMERIC_PA_IDS = frozenset({
        _pa.bool_().id,
        _pa.int8().id, _pa.int16().id, _pa.int32().id, _pa.int64().id,
        _pa.uint8().id, _pa.uint16().id, _pa.uint32().id, _pa.uint64().id,
        _pa.float16().id, _pa.float32().id, _pa.float64().id,
    }) if _HAS_PYARROW else frozenset()

    for p in paths:
        suff = "".join(p.suffixes).lower()
        try:
            if suff.endswith(".parquet") and _HAS_PYARROW:
                sch = _pq.read_schema(p)
                for f in sch:
                    name = f.name
                    if name in seen:
                        continue
                    if f.type.id in _NUMERIC_PA_IDS:
                        numerics.append(name)
                        seen.add(name)
            else:
                df_head = pd.read_csv(p, nrows=0)
                for c in df_head.columns:
                    if c in seen:
                        continue
                    try:
                        chunk = pd.read_csv(p, usecols=[c], nrows=100)
                        if _is_numeric_dtype_pd(chunk[c].dtype):
                            numerics.append(c)
                            seen.add(c)
                    except Exception:
                        pass
        except Exception:
            continue
        if len(numerics) >= max_numerics:
            break

    out = [variant_col] + numerics[:max_numerics]
    return list(dict.fromkeys(out))


# ---------- variant-aware file picking ----------

def _file_variant_presence(path: Path, variant_col: str = "variant") -> set[str]:
    suff = "".join(path.suffixes).lower()
    try:
        if suff.endswith(".parquet") and _HAS_PYARROW:
            tbl = _pq.read_table(path, columns=[variant_col])
            return set(map(str, tbl[variant_col].to_pylist()))
        if _HAS_PYARROW:
            try:
                conv = _pacsv.ConvertOptions(include_columns=[variant_col])
                tbl = _pacsv.read_csv(str(path), convert_options=conv)
                return set(map(str, tbl[variant_col].to_pylist()))
            except Exception:
                pass
        seen: set[str] = set()
        for ch in pd.read_csv(path, usecols=[variant_col], chunksize=200_000):
            seen.update(map(str, ch[variant_col].dropna().unique()))
        return seen
    except Exception:
        return set()


def choose_files_covering_variants(
    data_dir: str,
    sampled_variants: list[str],
    variant_col: str = "variant",
    max_files: int = 2000000,
) -> list[Path]:
    base_dir, _ = _resolve_layout(data_dir)
    all_paths = _list_feature_paths(base_dir)
    if not all_paths or not sampled_variants or max_files <= 0:
        return all_paths[: max(0, max_files)]

    target = set(map(str, sampled_variants))
    file_cov: list[tuple[Path, set[str]]] = []
    for p in all_paths:
        cov = _file_variant_presence(p, variant_col=variant_col)
        if cov:
            file_cov.append((p, cov))

    chosen: list[Path] = []
    covered: set[str] = set()
    remaining = set(target)

    while remaining and len(chosen) < max_files and file_cov:
        file_cov.sort(key=lambda t: len(t[1] & remaining), reverse=True)
        best, best_set = file_cov.pop(0)
        if len(best_set & remaining) == 0:
            break
        chosen.append(best)
        covered |= best_set
        remaining = target - covered
        file_cov = [(p, s) for (p, s) in file_cov if len(s & remaining) > 0]

    if not chosen:
        chosen = all_paths[:max_files]
    return chosen


# ---------- SQLite summary → wide features pivot ----------

# Stat columns to EXCLUDE from the pivot (diagnostic, not plottable features).
# These are excluded here so we don't have to re-run context._EXCLUDE_RE on
# every pivoted column name.
_SUMMARY_EXCLUDE_STATS: frozenset[str] = frozenset({
    "js_reps_to_pooled_mean", "js_reps_to_pooled_max",
    "L1_reps_to_pooled_mean", "L1_reps_to_pooled_max",
    "L2_reps_to_pooled_mean", "L2_reps_to_pooled_max",
    "n_frames_JS_lt_0.01", "frac_JS_lt_0.01",
    "n_frames_RMSEF_lt_0.1", "frac_RMSEF_lt_0.1",
    "converged_bool",
    "roughness_rms_dFdx", "entropy_bits_P",
    "tau_int_frames_mean", "tau_int_frames_max",
})

# Stat columns that are genuinely useful as features
_SUMMARY_KEEP_STATS: list[str] = [
    "mean", "std", "median", "mad", "iqr", "skew", "kurt_fisher",
    "min", "max", "n",
    "num_wells_pmf", "num_wells",
    "min1_x", "min2_x", "barrier_12_kJmol", "well1_width_at_kT",
    "curvature_abs_mean", "x_eqpop",
    "DeltaF_lr_at_median",
    "P_state1", "P_state2", "DeltaG_state2_vs_state1_kJmol",
    "auc_Fcum_left", "auc_Fcum_right",
    "slope_Fcum_left_at_eqpop", "slope_Fcum_right_at_eqpop",
    "circular_mean_deg", "circular_R", "circular_std_deg",
    "circular_median_deg", "circular_q25_deg", "circular_q75_deg",
    "circular_mode_deg",
]


def load_pmf_summary_long(
    db_path: Path,
    variants: list[str] | None = None,
    variant_col: str = "variant",
) -> pd.DataFrame:
    """
    Load the PMF summary table in its native long format (one row per variant × metric).

    This is the canonical input for :func:`~data.pmf_input.build_tab_input`.
    Unlike :func:`load_features_from_sqlite`, no pivot is performed — the full
    column set (including new PMF landscape metrics) is returned as-is.

    Reads in chunks of 250 000 rows so the full table never exists in memory at
    float64 before dtype-squeezing.  Each chunk is optimised before concatenation.

    Parameters
    ----------
    db_path     : path to global_peptide_analysis.sqlite
    variants    : optional variant allow-list
    variant_col : name of the variant identifier column
    """
    _GLOBAL_PROGRESS.start(1, "loading PMF summary (long format)")
    try:
        with sqlite3.connect(str(db_path), check_same_thread=False) as con:
            existing = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            if "summary" not in existing:
                return pd.DataFrame()

            if variants:
                placeholders = ",".join("?" * len(variants))
                query = (
                    f"SELECT * FROM summary WHERE metric IS NOT NULL"
                    f" AND {variant_col} IN ({placeholders})"
                )
                chunks = pd.read_sql_query(
                    query, con, params=variants, chunksize=250_000
                )
            else:
                chunks = pd.read_sql_query(
                    "SELECT * FROM summary WHERE metric IS NOT NULL",
                    con,
                    chunksize=250_000,
                )

            parts: list[pd.DataFrame] = []
            for chunk in chunks:
                if chunk is not None and not chunk.empty:
                    parts.append(_optimize_curves_df(chunk))
    finally:
        _GLOBAL_PROGRESS.tick(1)
        _GLOBAL_PROGRESS.finish()

    if not parts:
        return pd.DataFrame()

    df = parts[0] if len(parts) == 1 else _concat_nonempty(parts)
    _log_df_size("load_pmf_summary_long", df)
    return df


def _load_wide_features_from_table(
    db_path: Path,
    table_name: str,
    variants: list[str] | None = None,
    variant_col: str = "variant",
) -> pd.DataFrame:
    """
    Load a pre-built wide feature table from SQLite in chunks.

    BATCH_ANA writes ``features_pmf_landscape_core`` and ``features`` as
    pre-pivoted wide tables.  These contain the new PMF landscape metrics in
    column names like ``phi__effective_support_frac``.  Chunked reading keeps
    peak memory consistent with the rest of the I/O layer.
    """
    _CHUNK = 250_000
    with sqlite3.connect(str(db_path), check_same_thread=False) as con:
        existing = {r[0] for r in con.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()}
        if table_name not in existing:
            return pd.DataFrame()

        if variants:
            placeholders = ",".join("?" * len(variants))
            query = (
                f"SELECT * FROM {table_name} "
                f"WHERE {variant_col} IN ({placeholders})"
            )
            chunks = pd.read_sql_query(query, con, params=variants, chunksize=_CHUNK)
        else:
            chunks = pd.read_sql_query(
                f"SELECT * FROM {table_name}", con, chunksize=_CHUNK
            )

        parts: list[pd.DataFrame] = []
        for chunk in chunks:
            if chunk is not None and not chunk.empty:
                parts.append(_optimize_curves_df(chunk))

    if not parts:
        return pd.DataFrame()
    return parts[0] if len(parts) == 1 else _concat_nonempty(parts)


def _sqlite_table_columns(con: sqlite3.Connection, table_name: str) -> list[str]:
    try:
        return [r[1] for r in con.execute(f"PRAGMA table_info({_q_sql_ident(table_name)})").fetchall()]
    except Exception:
        return []


def _q_sql_ident(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _sequence_columns_from_names(cols: list[str], variant_col: str = "variant") -> list[str]:
    """Return auditable sequence/letter-space columns from a wide feature schema."""
    out: list[str] = []
    legacy = {"sequence", "L", "KD_mean", "KD_std", "KD_min", "KD_max", "KD_Nterm", "KD_Cterm"}
    meta = {"seq_valid", "seq_parse_method", "seq_parse_warning"}
    for c in cols:
        s = str(c)
        if s == variant_col:
            continue
        if s.startswith("seq_") or s in legacy or s in meta:
            out.append(s)
    return out


def _load_sequence_features_from_sqlite(
    db_path: Path,
    variants: list[str] | None = None,
    variant_col: str = "variant",
) -> pd.DataFrame:
    """Load only sequence/letter-space descriptors from SQLite if present.

    BATCH_ANA writes a dedicated ``features_sequence_language`` table. Older
    outputs may only have sequence descriptors in ``features_all``; both layouts
    are supported so the safe PMF-core table can stay compact.
    """
    try:
        with sqlite3.connect(str(db_path), check_same_thread=False) as con:
            existing = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            for table_name in ("features_sequence_language", "features_all"):
                if table_name not in existing:
                    continue
                cols = _sqlite_table_columns(con, table_name)
                seq_cols = _sequence_columns_from_names(cols, variant_col=variant_col)
                if not seq_cols or variant_col not in cols:
                    continue
                select_cols = [variant_col] + seq_cols
                quoted = ", ".join(_q_sql_ident(c) for c in select_cols)
                if variants:
                    placeholders = ",".join("?" * len(variants))
                    query = f"SELECT {quoted} FROM {_q_sql_ident(table_name)} WHERE {_q_sql_ident(variant_col)} IN ({placeholders})"
                    df = pd.read_sql_query(query, con, params=list(variants))
                else:
                    query = f"SELECT {quoted} FROM {_q_sql_ident(table_name)}"
                    df = pd.read_sql_query(query, con)
                break
            else:
                return pd.DataFrame()
    except Exception:
        return pd.DataFrame()
    if df.empty:
        return df
    return _optimize_curves_df(df)


def _merge_sequence_features(base: pd.DataFrame, seq: pd.DataFrame, variant_col: str = "variant") -> pd.DataFrame:
    if base is None or base.empty or seq is None or seq.empty:
        return base
    if variant_col not in base.columns or variant_col not in seq.columns:
        return base
    add_cols = [c for c in seq.columns if c != variant_col and c not in base.columns]
    if not add_cols:
        return base
    return base.merge(seq[[variant_col] + add_cols], on=variant_col, how="left")


def load_features_from_sqlite(
    db_path: Path,
    variants: list[str] | None = None,
    stat_cols: list[str] | None = None,
    variant_col: str = "variant",
) -> pd.DataFrame:
    """
    Load wide feature table from SQLite.

    Priority:
      1. ``features_pmf_landscape_core`` — pre-built by BATCH_ANA, contains new
         PMF landscape metrics (effective_support_frac, global_basin_*, etc.)
      2. ``features`` — same content, written as the default table name
      3. Summary pivot (legacy fallback, limited to ``_SUMMARY_KEEP_STATS``)

    Result columns: ``metric__stat`` e.g. ``phi__effective_support_frac``.
    """
    if stat_cols is None:
        stat_cols = _SUMMARY_KEEP_STATS

    _GLOBAL_PROGRESS.start(1, "loading features from SQLite")
    try:
        # Prefer pre-built PMF-core wide tables from BATCH_ANA, then merge the
        # sequence/letter-space block from features_all when available. This keeps
        # the default table compact while making UMAP's sequence mode functional.
        for tbl in ("features_pmf_landscape_core", "features"):
            try:
                df = _load_wide_features_from_table(
                    db_path, tbl, variants=variants, variant_col=variant_col
                )
                if not df.empty:
                    seq_df = _load_sequence_features_from_sqlite(
                        db_path, variants=variants, variant_col=variant_col
                    )
                    df = _merge_sequence_features(df, seq_df, variant_col=variant_col)
                    print(
                        f"[IO] features loaded from SQLite table '{tbl}' "
                        f"({len(df)} variants; seq_cols={max(0, df.filter(regex=r'^seq_|^sequence$|^KD_|^L$').shape[1])})",
                        file=sys.stderr,
                    )
                    _log_df_size(f"load_features_from_sqlite ({tbl}+sequence)", df)
                    return df
            except Exception as e:
                print(f"[IO] {tbl} load failed ({e}), trying next", file=sys.stderr)

        # If only the all-column table exists, load it as a final pre-built
        # fallback. This is wider but preserves sequence descriptors.
        try:
            df = _load_wide_features_from_table(
                db_path, "features_all", variants=variants, variant_col=variant_col
            )
            if not df.empty:
                print(
                    f"[IO] features loaded from SQLite table 'features_all' ({len(df)} variants)",
                    file=sys.stderr,
                )
                _log_df_size("load_features_from_sqlite (features_all)", df)
                return df
        except Exception as e:
            print(f"[IO] features_all load failed ({e}), trying legacy summary pivot", file=sys.stderr)

        # Legacy fallback: summary pivot
        with sqlite3.connect(str(db_path), check_same_thread=False) as con:
            existing = {r[0] for r in con.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
            if "summary" not in existing:
                return pd.DataFrame()

            pragma = con.execute("PRAGMA table_info(summary)").fetchall()
            db_cols = {r[1] for r in pragma}
            use_stats = [c for c in stat_cols if c in db_cols]
            if not use_stats:
                return pd.DataFrame()

            if variants:
                placeholders = ",".join("?" * len(variants))
                query = (
                    f"SELECT {variant_col}, metric, "
                    + ", ".join(use_stats)
                    + f" FROM summary WHERE metric IS NOT NULL"
                    + f" AND {variant_col} IN ({placeholders})"
                )
                df_long = pd.read_sql_query(query, con, params=variants)
            else:
                query = (
                    f"SELECT {variant_col}, metric, "
                    + ", ".join(use_stats)
                    + " FROM summary WHERE metric IS NOT NULL"
                )
                df_long = pd.read_sql_query(query, con)
    finally:
        _GLOBAL_PROGRESS.tick(1)
        _GLOBAL_PROGRESS.finish()

    if df_long.empty:
        return pd.DataFrame()

    df_long = _optimize_curves_df(df_long)

    try:
        n_variants = int(df_long[variant_col].nunique(dropna=True))
        n_metrics  = int(df_long["metric"].nunique(dropna=True))
        est_cells  = n_variants * n_metrics * max(1, len(use_stats))
        if est_cells > 5_000_000:
            print(
                f"[IO] refusing SQL pivot: estimated {n_variants:,} × "
                f"{n_metrics:,} × {len(use_stats)} = {est_cells:,} cells.  "
                "Falling back to file loader.",
                file=sys.stderr,
            )
            return pd.DataFrame()
    except Exception:
        pass

    try:
        wide = df_long.pivot_table(
            index=variant_col,
            columns="metric",
            values=use_stats,
            aggfunc="first",
        )
        wide.columns = [f"{metric}__{stat}" for stat, metric in wide.columns]
        wide = wide.reset_index()
        wide.columns.name = None
    except Exception as e:
        print(f"[IO] summary pivot failed ({e}), returning empty", file=sys.stderr)
        return pd.DataFrame()

    wide = _optimize_curves_df(wide)
    _log_df_size("load_features_from_sqlite (summary pivot)", wide)
    return wide


# ---------- aggregate subset loader (file-based fallback) ----------

def _load_one_features_file(path: Path, usecols: list[str] | None) -> pd.DataFrame:
    try:
        suff = "".join(path.suffixes).lower()
        return _read_parquet_cols(path, usecols) if suff.endswith(".parquet") else _read_csv_cols(path, usecols)
    finally:
        _GLOBAL_PROGRESS.tick(1)


def _load_one_features_file_filtered(
    path: Path,
    variants: list[str],
    variant_col: str,
    usecols: list[str] | None,
) -> pd.DataFrame:
    try:
        suff = "".join(path.suffixes).lower()
        if suff.endswith(".parquet") and _HAS_PYARROW:
            try:
                filt = _pads.field(variant_col).isin(variants)
                ds_ = _pads.dataset([str(path)])
                tbl = (
                    ds_.to_table(columns=usecols, filter=filt)
                    if usecols
                    else ds_.to_table(filter=filt)
                )
                return tbl.to_pandas(use_threads=True, split_blocks=True, self_destruct=True)
            except Exception:
                df = _read_parquet_cols(path, usecols)
                return df[df[variant_col].isin(variants)] if variant_col in df.columns else df
        frames: list[pd.DataFrame] = []
        for ch in pd.read_csv(path, usecols=usecols, chunksize=200_000):
            frames.append(ch[ch[variant_col].isin(variants)] if variant_col in ch.columns else ch)
        return _concat_nonempty(frames)
    finally:
        _GLOBAL_PROGRESS.tick(1)


def load_features_subset(
    data_dir: str,
    variants: list[str] | None,
    variant_col: str,
    usecols: list[str] | None,
    max_files: int | None,
    paths: list[Path] | None = None,
) -> pd.DataFrame:
    """
    Load features.  SQLite summary pivot is tried first (instant for most
    datasets); falls back to parallel file loading if not available.
    """
    base_dir, _ = _resolve_layout(data_dir)
    db_path = base_dir / "global_peptide_analysis.sqlite"

    if db_path.is_file():
        try:
            df = load_features_from_sqlite(db_path, variants=variants, variant_col=variant_col)
            if not df.empty:
                print(f"[IO] features loaded from SQLite summary ({len(df)} variants)", file=sys.stderr)
                return df
        except Exception as e:
            print(f"[IO] SQLite features load failed ({e}), falling back to files", file=sys.stderr)

    # File fallback
    paths = paths or _list_feature_paths(base_dir, max_files=max_files)
    if not paths:
        return pd.DataFrame()

    phase = "loading features (filtered)" if variants else "loading features"
    _GLOBAL_PROGRESS.start(len(paths), phase)
    frames: list[pd.DataFrame] = []
    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        if variants:
            futs = [
                ex.submit(_load_one_features_file_filtered, p, variants, variant_col, usecols)
                for p in paths
            ]
        else:
            futs = [ex.submit(_load_one_features_file, p, usecols) for p in paths]
        for fut in as_completed(futs):
            try:
                df = fut.result()
                if isinstance(df, pd.DataFrame) and len(df):
                    frames.append(df)
            except Exception as e:
                print(f"[IO] file load error: {e}", file=sys.stderr)
    _GLOBAL_PROGRESS.finish()
    return _concat_nonempty(frames)


# ---------- full features aggregate (monolith clone) ----------

def load_features_aggregate(data_dir: str) -> pd.DataFrame:
    """
    Full feature loader.  SQLite summary pivot first, file scan as fallback.
    """
    base_dir, _ = _resolve_layout(data_dir)
    db_path = base_dir / "global_peptide_analysis.sqlite"

    if db_path.is_file():
        try:
            df = load_features_from_sqlite(db_path, variants=None)
            if not df.empty:
                print(f"[IO] features loaded from SQLite summary ({len(df)} variants)", file=sys.stderr)
                return df
        except Exception as e:
            print(f"[IO] SQLite features load failed ({e}), falling back to files", file=sys.stderr)

    files = _list_feature_paths(base_dir)
    if not files:
        raise SystemExit(
            f"[ERR] No features files found in {base_dir} and no SQLite summary available."
        )

    _GLOBAL_PROGRESS.start(len(files), "loading features (aggregate)")
    # Rolling-concat strategy: keep only `_BATCH` finished frames in a small
    # staging list, merge to a `rollup` frame, then discard the staging list.
    # Peak memory is bounded by (rollup size) + (_BATCH small frames), not by
    # (all frames × count).  Critical for hosts with many feature files.
    _BATCH = 16
    staging: list[pd.DataFrame] = []
    rollup: pd.DataFrame = pd.DataFrame()

    def _flush(staging_list: list[pd.DataFrame], rollup_df: pd.DataFrame) -> pd.DataFrame:
        if not staging_list:
            return rollup_df
        merged = _concat_nonempty(staging_list)
        staging_list.clear()
        if rollup_df.empty:
            return _optimize_curves_df(merged)
        return _optimize_curves_df(_concat_nonempty([rollup_df, merged]))

    with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
        futs = [ex.submit(_load_one_features_file, f, None) for f in files]
        for fut in as_completed(futs):
            try:
                d = fut.result()
                if isinstance(d, pd.DataFrame) and len(d):
                    staging.append(_drop_all_na_cols(d))
                    if len(staging) >= _BATCH:
                        rollup = _flush(staging, rollup)
            except Exception as e:
                print(f"[IO] aggregate load error: {e}", file=sys.stderr)
    rollup = _flush(staging, rollup)
    _GLOBAL_PROGRESS.finish()

    if not rollup.empty and rollup.columns.duplicated().any():
        rollup = rollup.loc[:, ~rollup.columns.duplicated()]
    return rollup


# Columns whose values are repeated across many rows.  Converting these to
# pandas categorical cuts memory by 10-50x on typical datasets.
_CATEGORICAL_CANDIDATES: frozenset[str] = frozenset({
    "variant", "metric", "source", "residue", "replica", "rep", "rep_id",
    "replicate", "replica_id", "replicate_id", "replica_idx", "replica_num",
    "replica_number", "state", "region",
})


def _optimize_curves_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downcast every frame we load so peak memory stays close to raw file size.

    The previous version only touched curve tables; the feature aggregate and
    long-format SQL pivots escaped it and kept their int64 / float64 / object
    dtypes, which is where most of the 64 GB explosion was coming from.  Now
    every loader routes through here.

    Rules:
      - Known id-like string columns → pandas categorical (huge win).
      - Other object columns with <= 50 % unique values → categorical.
      - Floats → smallest float dtype that still fits (float64 → float32).
      - Integers → smallest int dtype that fits.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return df

    for c in df.columns:
        try:
            s = df[c]
            dt = s.dtype
            lc = str(c).lower()
            # Object / string columns → categorical if cheap
            if dt == object or pd.api.types.is_string_dtype(dt):
                if lc in _CATEGORICAL_CANDIDATES:
                    df[c] = s.astype("category")
                else:
                    # Only convert if cardinality is low (< half the rows).
                    try:
                        n_uniq = s.nunique(dropna=True)
                        if n_uniq > 0 and n_uniq * 2 < len(s):
                            df[c] = s.astype("category")
                    except Exception:
                        pass
                continue
            # Floats: drop to float32 where safe
            if pd.api.types.is_float_dtype(dt):
                df[c] = pd.to_numeric(s, errors="coerce", downcast="float")
                continue
            # Integers: drop to smallest signed/unsigned int that fits
            if pd.api.types.is_integer_dtype(dt):
                df[c] = pd.to_numeric(s, errors="coerce", downcast="integer")
                continue
        except Exception:
            # Never let dtype squeezing break a load
            pass

    return df


# Backwards-compat alias — every loader now goes through _optimize_curves_df,
# not just curves; the name stays for the callers that already reference it.
_optimize_df = _optimize_curves_df


# ---------- lazy curves loader ----------

class LazyCurvesLoader:
    """
    Lazy loader for PMF / cumulative / replica / Rama2D / convergence tables.

    Construction is instant — only a cheap ``sqlite_master`` scan is done at
    init time.  Each table attribute is loaded from SQLite (or files) on first
    access and cached in memory.  This eliminates the multi-second (or never-
    finishing) startup cost that occurred when every table was loaded eagerly.
    """

    _DB_NAMES: dict[str, list[str]] = {
        "pmf":           ["pmf"],
        "cum":           ["cumulative"],
        "rmsf":          ["replica"],
        "rama2d_pooled": ["rama2d_pooled", "rama2d"],
        "rama2d_perres": ["rama2d_perres", "rama2d_byres"],
        "conv":          ["convergence"],
        "pmf_replica":   ["pmf_replica"],
        "cum_replica":   ["cumulative_replica", "cum_replica"],
        "conv_replica":  ["convergence_replica"],
    }

    _FILE_PATTERNS: dict[str, list[str]] = {
        "pmf":           ["*_pmf.parquet", "*_pmf.csv.gz", "pmf.parquet", "pmf.csv.gz"],
        "cum":           ["*_cumulative.parquet", "*_cumulative.csv.gz",
                          "cumulative.parquet", "cumulative.csv.gz"],
        "rmsf":          ["*_replica.parquet", "*_replica.csv.gz",
                          "replica.parquet", "replica.csv.gz"],
        "rama2d_pooled": ["*_*cgrama2d.parquet", "*_*cgrama2d.csv.gz",
                          "*cgrama2d.parquet", "*cgrama2d.csv.gz",
                          "rama2d_pooled.parquet", "rama2d_pooled.csv.gz"],
        "rama2d_perres": [
            "*_*cgrama2d_perres.parquet", "*_*cgrama2d_perres.csv.gz",
            "*_*cgrama2d_byres.parquet",  "*_*cgrama2d_byres.csv.gz",
            "rama2d_perres.parquet", "rama2d_perres.csv.gz",
        ],
        "conv":          ["*_convergence.parquet", "*_convergence.csv.gz",
                          "convergence_*.parquet",  "convergence_*.csv.gz",
                          "convergence.parquet",    "convergence.csv.gz"],
        "pmf_replica":   ["*_pmf_replica.parquet", "*_pmf_replica.csv.gz",
                          "pmf_replica.parquet", "pmf_replica.csv.gz"],
        "cum_replica":   ["*_cumulative_replica.parquet", "*_cumulative_replica.csv.gz",
                          "*_cum_replica.parquet",         "*_cum_replica.csv.gz",
                          "cumulative_replica.parquet",    "cumulative_replica.csv.gz",
                          "cum_replica.parquet",           "cum_replica.csv.gz"],
        "conv_replica":  ["*_convergence_replica.parquet", "*_convergence_replica.csv.gz",
                          "convergence_replica.parquet",   "convergence_replica.csv.gz"],
    }

    def __init__(self, data_dir: str) -> None:
        self._base_dir, _ = _resolve_layout(data_dir)
        self._base_dir = Path(self._base_dir).expanduser().resolve()
        self._cache: dict[str, pd.DataFrame] = {}
        # Cheap: only reads sqlite_master — no table data loaded here.
        db_path = self._base_dir / "global_peptide_analysis.sqlite"
        if db_path.is_file():
            self._db_path: Path | None = db_path
            try:
                with sqlite3.connect(str(db_path)) as con:
                    self._db_tables: frozenset[str] = frozenset(
                        r[0] for r in con.execute(
                            "SELECT name FROM sqlite_master WHERE type='table'"
                        ).fetchall()
                    )
            except Exception:
                self._db_path = None
                self._db_tables = frozenset()
        else:
            self._db_path = None
            self._db_tables = frozenset()

    # ------------------------------------------------------------------ helpers

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        df = _drop_all_na_cols(df)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()].copy()
        return _optimize_curves_df(df)

    # Stream SQL reads in chunks and downcast each chunk immediately.  Without
    # this, SELECT * FROM pmf_replica on a large table produces one monolithic
    # float64/object DataFrame — the single biggest source of the 64 GB OOMs.
    _SQL_CHUNK = 250_000

    def _load_from_db(self, key: str) -> pd.DataFrame:
        if self._db_path is None:
            return pd.DataFrame()
        for name in self._DB_NAMES.get(key, []):
            if name not in self._db_tables:
                continue
            try:
                parts: list[pd.DataFrame] = []
                with sqlite3.connect(str(self._db_path)) as con:
                    # chunksize turns read_sql_query into an iterator; each
                    # chunk gets dtype-squeezed before it joins the rollup.
                    for chunk in pd.read_sql_query(
                        f"SELECT * FROM {name}", con, chunksize=self._SQL_CHUNK
                    ):
                        if chunk is None or chunk.empty:
                            continue
                        parts.append(_optimize_curves_df(chunk))
                if not parts:
                    continue
                df = parts[0] if len(parts) == 1 else _concat_nonempty(parts)
                if not df.empty:
                    return df
            except Exception as e:
                print(f"[IO] SQL chunked read failed for {name}: {e}", file=sys.stderr)
        return pd.DataFrame()

    def _load_from_files(self, key: str) -> pd.DataFrame:
        patterns = self._FILE_PATTERNS.get(key, [])
        paths = _filter_curve_paths(key, _find_files(self._base_dir, patterns))
        if not paths:
            return pd.DataFrame()
        _GLOBAL_PROGRESS.start(len(paths), f"loading {key} (files)")
        frames: list[pd.DataFrame] = []
        for p in paths:
            d = _read_any_df(p)
            _GLOBAL_PROGRESS.tick(1)
            if d is not None and not d.empty:
                frames.append(d)
        _GLOBAL_PROGRESS.finish()
        return _concat_nonempty(frames)

    def _postprocess(self, key: str, df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        if key == "rama2d_pooled":
            if {"x", "y", "density"}.issubset(df.columns) and "residue" in df.columns:
                df = df[df["residue"].isna()].copy()
        elif key == "rama2d_perres":
            if {"x", "y", "density"}.issubset(df.columns):
                if "residue" not in df.columns and "dataset_index" in df.columns:
                    df = df.rename(columns={"dataset_index": "residue"})
                if "residue" in df.columns:
                    df["residue"] = pd.to_numeric(df["residue"], errors="coerce")
        return df

    def _get(self, key: str) -> pd.DataFrame:
        if key not in self._cache:
            df = self._load_from_db(key)
            if df.empty:
                df = self._load_from_files(key)
            df = self._clean(df)
            df = self._postprocess(key, df)
            self._cache[key] = df
        return self._cache[key]

    # ------------------------------------------------------------------ public properties

    @property
    def pmf_df(self) -> pd.DataFrame:          return self._get("pmf")
    @property
    def cum_df(self) -> pd.DataFrame:          return self._get("cum")
    @property
    def rmsf_df(self) -> pd.DataFrame:         return self._get("rmsf")
    @property
    def rama2d_pooled_df(self) -> pd.DataFrame: return self._get("rama2d_pooled")
    @property
    def rama2d_perres_df(self) -> pd.DataFrame: return self._get("rama2d_perres")
    @property
    def conv_df(self) -> pd.DataFrame:         return self._get("conv")
    @property
    def pmf_replica_df(self) -> pd.DataFrame:  return self._get("pmf_replica")
    @property
    def cum_replica_df(self) -> pd.DataFrame:  return self._get("cum_replica")
    @property
    def conv_replica_df(self) -> pd.DataFrame: return self._get("conv_replica")

    # ------------------------------------------------------------------ PMF input helpers

    @property
    def pmf_summary_long_df(self) -> pd.DataFrame:
        """
        PMF summary in long format (one row per variant × observable/metric).

        This is the canonical input for ``data.pmf_input.build_tab_input``.
        Column ``metric`` corresponds to observable_name in the spec.
        """
        if "pmf_summary_long" not in self._cache:
            df = pd.DataFrame()
            if self._db_path is not None and "summary" in self._db_tables:
                try:
                    df = load_pmf_summary_long(self._db_path)
                except Exception as e:
                    print(f"[IO] pmf_summary_long load failed: {e}", file=sys.stderr)
            self._cache["pmf_summary_long"] = self._clean(df)
        return self._cache["pmf_summary_long"]

    @property
    def pmf_annotations_df(self) -> pd.DataFrame:
        """
        Wide-format physical annotation features per variant
        (``features_pmf_annotations`` table built by BATCH_ANA).
        """
        if "pmf_annotations" not in self._cache:
            df = pd.DataFrame()
            if self._db_path is not None:
                for tbl in ("features_pmf_annotations",):
                    if tbl in self._db_tables:
                        try:
                            df = _load_wide_features_from_table(self._db_path, tbl)
                            if not df.empty:
                                break
                        except Exception as e:
                            print(f"[IO] {tbl} load failed: {e}", file=sys.stderr)
                if df.empty:
                    paths = _find_files(
                        self._base_dir, ["features_pmf_annotations.parquet"]
                    )
                    if paths:
                        df = _read_any_df(paths[0]) or pd.DataFrame()
            self._cache["pmf_annotations"] = self._clean(df)
        return self._cache["pmf_annotations"]

    # Keys handled by dedicated properties rather than _get(); included in prefetch.
    _EXTRA_PREFETCH_KEYS: frozenset[str] = frozenset({"pmf_summary_long", "pmf_annotations"})

    def prefetch(self, *keys: str) -> None:
        """
        Eagerly load the listed keys in parallel background threads.

        Call with no arguments to prefetch every table (including
        ``pmf_summary_long`` and ``pmf_annotations``).  Useful when startup I/O
        is acceptable (e.g. CLI batch mode) and all caches should be warm before
        the first user interaction.
        """
        if keys:
            db_keys = [k for k in keys if k in self._DB_NAMES]
            extra_keys = [k for k in keys if k in self._EXTRA_PREFETCH_KEYS]
        else:
            db_keys = list(self._DB_NAMES)
            extra_keys = list(self._EXTRA_PREFETCH_KEYS)

        def _load_extra(name: str) -> None:
            if name == "pmf_summary_long":
                _ = self.pmf_summary_long_df
            elif name == "pmf_annotations":
                _ = self.pmf_annotations_df

        all_keys = db_keys + extra_keys
        if not all_keys:
            return
        with ThreadPoolExecutor(max_workers=min(len(all_keys), _MAX_WORKERS)) as ex:
            futs: dict = {ex.submit(self._get, k): k for k in db_keys}
            futs.update({ex.submit(_load_extra, k): k for k in extra_keys})
            for fut in as_completed(futs):
                try:
                    fut.result()
                except Exception as e:
                    print(f"[IO] prefetch error for '{futs[fut]}': {e}", file=sys.stderr)


def load_curves_tables_lazy(data_dir: str) -> LazyCurvesLoader:
    """
    Return a :class:`LazyCurvesLoader` for *data_dir*.

    No disk I/O beyond a cheap ``sqlite_master`` scan happens here; each curve
    table is read only when the corresponding attribute is first accessed.
    """
    return LazyCurvesLoader(data_dir)


# ---------- curves / PMF loaders (SQLITE-FIRST, file fallback) ----------

def load_curves_tables(data_dir: str):
    """
    Load PMF / cumulative / replica / Rama2D / convergence tables.

    Priority:
      1) global_peptide_analysis.sqlite (single read, fastest)
      2) Parquet / CSV files in the same directory
    """
    base_dir, _ = _resolve_layout(data_dir)
    base_dir = Path(base_dir).expanduser().resolve()

    table_map = {
        "pmf":           ["pmf"],
        "cum":           ["cumulative"],
        "rmsf":          ["replica"],
        "rama2d_pooled": ["rama2d_pooled", "rama2d"],
        "rama2d_perres": ["rama2d_perres", "rama2d_byres"],
        "conv":          ["convergence"],
        "pmf_replica":   ["pmf_replica"],
        "cum_replica":   ["cumulative_replica", "cum_replica"],
        "conv_replica":  ["convergence_replica"],
    }

    def _clean_df(df: pd.DataFrame) -> pd.DataFrame:
        if not isinstance(df, pd.DataFrame) or df.empty:
            return pd.DataFrame()
        df = _drop_all_na_cols(df)
        if df.columns.duplicated().any():
            df = df.loc[:, ~df.columns.duplicated()].copy()
        return _optimize_curves_df(df)

    loaded = {k: pd.DataFrame() for k in table_map}

    # --- SQLite path (single connection, all tables in one pass) ---
    db_path = base_dir / "global_peptide_analysis.sqlite"
    if db_path.is_file():
        try:
            with sqlite3.connect(str(db_path)) as con:
                # Get available tables in one query
                existing = {
                    r[0]
                    for r in con.execute(
                        "SELECT name FROM sqlite_master WHERE type='table'"
                    ).fetchall()
                }
                _SQL_CHUNK = 250_000
                for key, names in table_map.items():
                    for name in names:
                        if name not in existing:
                            continue
                        try:
                            parts: list[pd.DataFrame] = []
                            for chunk in pd.read_sql_query(
                                f"SELECT * FROM {name}", con, chunksize=_SQL_CHUNK
                            ):
                                if chunk is not None and not chunk.empty:
                                    parts.append(_optimize_curves_df(chunk))
                            df = (
                                parts[0] if len(parts) == 1
                                else _concat_nonempty(parts)
                                if parts else pd.DataFrame()
                            )
                        except Exception:
                            df = pd.DataFrame()
                        if not df.empty:
                            loaded[key] = _clean_df(df)
                            break
        except Exception as e:
            print(f"[IO] Failed to read {db_path} ({e}); falling back to files.", file=sys.stderr)

    # --- File fallback for any tables still missing ---
    spec: dict[str, list[str]] = {
        "pmf":           ["*_pmf.parquet", "*_pmf.csv.gz", "pmf.parquet", "pmf.csv.gz"],
        "cum":           ["*_cumulative.parquet", "*_cumulative.csv.gz", "cumulative.parquet", "cumulative.csv.gz"],
        "rmsf":          ["*_replica.parquet", "*_replica.csv.gz", "replica.parquet", "replica.csv.gz"],
        "rama2d_pooled": ["*_*cgrama2d.parquet", "*_*cgrama2d.csv.gz", "*cgrama2d.parquet", "*cgrama2d.csv.gz",
                          "rama2d_pooled.parquet", "rama2d_pooled.csv.gz"],
        "rama2d_perres": [
            "*_*cgrama2d_perres.parquet", "*_*cgrama2d_perres.csv.gz",
            "*_*cgrama2d_byres.parquet", "*_*cgrama2d_byres.csv.gz",
            "rama2d_perres.parquet", "rama2d_perres.csv.gz",
        ],
        "conv":          ["*_convergence.parquet", "*_convergence.csv.gz", "convergence_*.parquet", "convergence_*.csv.gz", "convergence.parquet", "convergence.csv.gz"],
        "pmf_replica":   ["*_pmf_replica.parquet", "*_pmf_replica.csv.gz", "pmf_replica.parquet", "pmf_replica.csv.gz"],
        "cum_replica":   ["*_cumulative_replica.parquet", "*_cumulative_replica.csv.gz", "*_cum_replica.parquet", "*_cum_replica.csv.gz", "cumulative_replica.parquet", "cumulative_replica.csv.gz", "cum_replica.parquet", "cum_replica.csv.gz"],
        "conv_replica":  ["*_convergence_replica.parquet", "*_convergence_replica.csv.gz", "convergence_replica.parquet", "convergence_replica.csv.gz"],
    }

    # Collect all missing keys and load their files in parallel
    missing_items = [
        (key, _filter_curve_paths(key, _find_files(base_dir, spec[key])))
        for key in spec
        if loaded[key].empty
    ]
    all_paths_needed = [(key, p) for key, paths in missing_items for p in paths]

    if all_paths_needed:
        _GLOBAL_PROGRESS.start(len(all_paths_needed), "loading curves (files)")

        def _load_curve_file(key: str, path: Path):
            try:
                df = _read_any_df(path)
                return key, df
            finally:
                _GLOBAL_PROGRESS.tick(1)

        key_frames: dict[str, list[pd.DataFrame]] = {key: [] for key, _ in missing_items}
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            futs = [ex.submit(_load_curve_file, key, p) for key, p in all_paths_needed]
            for fut in as_completed(futs):
                try:
                    key, df = fut.result()
                    if df is not None and not df.empty:
                        key_frames[key].append(df)
                except Exception as e:
                    print(f"[IO] curve load error: {e}", file=sys.stderr)

        _GLOBAL_PROGRESS.finish()

        for key, frames in key_frames.items():
            if frames and loaded[key].empty:
                merged = _concat_nonempty(frames)
                loaded[key] = _clean_df(merged)

    # --- Post-process Rama2D ---
    rp = loaded["rama2d_pooled"]
    if not rp.empty and {"x", "y", "density"}.issubset(rp.columns) and "residue" in rp.columns:
        loaded["rama2d_pooled"] = rp[rp["residue"].isna()].copy()

    rr = loaded["rama2d_perres"]
    if not rr.empty and {"x", "y", "density"}.issubset(rr.columns):
        if "residue" not in rr.columns and "dataset_index" in rr.columns:
            rr = rr.rename(columns={"dataset_index": "residue"})
        if "residue" in rr.columns:
            rr["residue"] = pd.to_numeric(rr["residue"], errors="coerce")
        loaded["rama2d_perres"] = rr

    return (
        loaded["pmf"],
        loaded["cum"],
        loaded["rmsf"],
        loaded["rama2d_pooled"],
        loaded["rama2d_perres"],
        loaded["conv"],
        loaded["pmf_replica"],
        loaded["cum_replica"],
        loaded["conv_replica"],
    )


# ---------- public aliases ----------

sample_variants_from_features = _sample_variants_from_features
