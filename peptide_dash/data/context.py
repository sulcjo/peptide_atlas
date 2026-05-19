from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from . import io
from .io import LazyCurvesLoader


# Technical / diagnostic columns to *exclude* from feature selection in the
# dashboard. These are anchored regex patterns that match a full column name
# or a full underscore-separated token, NOT arbitrary substrings.
_EXCLUDE_PATTERNS: tuple[str, ...] = (
    r"(^|__)mbar_",
    r"(^|__)ESS_",
    r"(^|__)jsd?2?_reps_to_pooled(_|$)",
    r"(^|__)L[12]_reps_to_pooled(_|$)",
    r"(^|__)cv_time(_|$)",
    r"(^|__)hops(_|$)",
    r"(^|__)psd_",
    r"(^|__)rms_dFdx$",
    r"(_|^)bootstrap_",
    r"(^|__)pmf_F_std_(mean|max)_kJmol$",
    r"(^|__)pmf_n_boot_effective$",
    r"(^|__)pmf_n_replicas_used_for_ci$",
    r"(^|__)tau_int(_|$)",
    r"(^|__)n_frames_(JS|RMSEF)_lt_",
    r"(^|__)frac_(JS|RMSEF)_lt_",
    r"(^|__)tail_median_(JS|RMSE_F)$",
    r"(^|__)converged_bool$",
)

_EXCLUDE_RE = re.compile("|".join(f"(?:{p})" for p in _EXCLUDE_PATTERNS))


def _is_excluded(colname: str) -> bool:
    return bool(_EXCLUDE_RE.search(str(colname)))


def _filter_numeric_columns(cols: Iterable[str]) -> list[str]:
    out: list[str] = []
    for c in cols:
        if c == "variant":
            continue
        if _is_excluded(str(c)):
            continue
        out.append(str(c))
    return out


# Public aliases
EXCLUDE_PATTERNS: tuple[str, ...] = _EXCLUDE_PATTERNS
EXCLUDE_SUBSTRINGS: tuple[str, ...] = _EXCLUDE_PATTERNS


def is_excluded_feature_column(colname: str) -> bool:
    return _is_excluded(colname)


def filter_numeric_columns(cols: Iterable[str]) -> list[str]:
    return _filter_numeric_columns(cols)


def _ensure_variant_in_usecols(
    usecols: Optional[list[str]], variant_col: str = "variant"
) -> Optional[list[str]]:
    if usecols is None:
        return None
    cols = list(usecols)
    if variant_col not in cols:
        cols.insert(0, variant_col)
    return cols


def _validate_sample_frac(frac: Optional[float]) -> Optional[float]:
    if frac is None:
        return None
    f = float(frac)
    if not (0.0 < f <= 1.0):
        raise ValueError("sample_frac must be in the range (0, 1].")
    return f


class DataContext:
    """
    Modular data context shared by all tabs.

    Curve tables (pmf_df, cum_df, rmsf_df, …) are loaded **lazily** on first
    access via a :class:`~data.io.LazyCurvesLoader`.  Startup is instant even
    when the underlying files are large or numerous — a tab only pays the I/O
    cost the first time it renders a curve.
    """

    def __init__(
        self,
        data_dir: str,
        timeseries_dir: Optional[str] = None,
        features: Optional[pd.DataFrame] = None,
        numeric_columns: Optional[list[str]] = None,
        *,
        lazy_curves: Optional[LazyCurvesLoader] = None,
    ) -> None:
        self.data_dir = str(Path(data_dir).resolve())
        self.timeseries_dir = (
            str(Path(timeseries_dir).resolve()) if timeseries_dir is not None else None
        )
        self.features: pd.DataFrame = (
            features if isinstance(features, pd.DataFrame) else pd.DataFrame()
        )
        self._lazy_curves: Optional[LazyCurvesLoader] = lazy_curves
        self.numeric_columns: list[str] = _filter_numeric_columns(numeric_columns or [])
        self._annotation_cols: set = set()

    # ------------------------------------------------------------------
    # Lazy curve properties — load on first access, cached thereafter
    # ------------------------------------------------------------------

    def _curves(self) -> LazyCurvesLoader:
        """Return the loader, creating it on-demand if not provided at init."""
        if self._lazy_curves is None:
            self._lazy_curves = io.load_curves_tables_lazy(self.data_dir)
        return self._lazy_curves

    @property
    def pmf_df(self) -> pd.DataFrame:
        return self._curves().pmf_df

    @property
    def cum_df(self) -> pd.DataFrame:
        return self._curves().cum_df

    @property
    def rmsf_df(self) -> pd.DataFrame:
        return self._curves().rmsf_df

    @property
    def rama2d_pooled_df(self) -> pd.DataFrame:
        return self._curves().rama2d_pooled_df

    @property
    def rama2d_perres_df(self) -> pd.DataFrame:
        return self._curves().rama2d_perres_df

    @property
    def conv_df(self) -> pd.DataFrame:
        return self._curves().conv_df

    @property
    def pmf_replica_df(self) -> pd.DataFrame:
        return self._curves().pmf_replica_df

    @property
    def cum_replica_df(self) -> pd.DataFrame:
        return self._curves().cum_replica_df

    @property
    def conv_replica_df(self) -> pd.DataFrame:
        return self._curves().conv_replica_df

    @property
    def pmf_summary_long_df(self) -> pd.DataFrame:
        """
        PMF summary in long format (one row per variant × observable/metric).

        Canonical input for :func:`~data.pmf_input.build_tab_input`.
        The ``metric`` column is the observable name.
        """
        return self._curves().pmf_summary_long_df

    @property
    def pmf_annotations_df(self) -> pd.DataFrame:
        """
        Wide-format physical annotation features per variant.

        Contains coordinate-space basin positions and widths from BATCH_ANA.
        For embedding use, filter via :func:`~data.pmf_input.build_tab_input`
        with ``preset="pmf_annotations"`` rather than including these directly.
        """
        return self._curves().pmf_annotations_df

    # ------------------------------------------------------------------
    # Convenience aliases used by tab implementations
    # ------------------------------------------------------------------

    @property
    def numeric_cols(self) -> list[str]:
        """
        Backwards-compatible alias for numeric_columns.

        If numeric_columns is empty but a non-empty feature table is available,
        lazily infer numeric columns from the dataframe and apply exclusion rules.
        """
        if not self.numeric_columns and not self.features.empty:
            numeric = self.features.select_dtypes(include="number").columns
            self.numeric_columns = _filter_numeric_columns(numeric)
        return self.numeric_columns

    @numeric_cols.setter
    def numeric_cols(self, value: list[str]) -> None:
        self.numeric_columns = _filter_numeric_columns(list(value or []))

    @property
    def df(self) -> pd.DataFrame:
        """Alias used by tabs/features.py for the main feature table."""
        return self.features

    # ------------------------------------------------------------------
    # PMF annotation merge
    # ------------------------------------------------------------------

    def _merge_pmf_annotations(self) -> None:
        """
        Left-join PMF annotation columns (basin geometry, landscape metrics) from
        ``pmf_annotations_df`` into ``self.features``.

        Called at construction time so that all tabs see annotation columns via
        the normal ``ctx.df`` / ``ctx.numeric_cols`` paths.  Silently no-ops when
        the annotations table is absent or empty.
        """
        if self.features.empty or "variant" not in self.features.columns:
            return
        try:
            ann = self.pmf_annotations_df
        except Exception:
            return
        if ann.empty or "variant" not in ann.columns:
            return

        existing = set(self.features.columns)
        ann_cols = [c for c in ann.columns if c != "variant" and c not in existing]
        if not ann_cols:
            return

        self.features = self.features.merge(
            ann[["variant"] + ann_cols], on="variant", how="left"
        )
        self._annotation_cols = getattr(self, "_annotation_cols", set()) | set(ann_cols)

        # Extend explicit numeric_columns list when it was pre-populated
        if self.numeric_columns:
            new_num = _filter_numeric_columns(
                [c for c in ann_cols if pd.api.types.is_numeric_dtype(self.features[c])]
            )
            seen = set(self.numeric_columns)
            self.numeric_columns = self.numeric_columns + [c for c in new_num if c not in seen]

    # ------------------------------------------------------------------
    # Empty / placeholder constructor (used by async loader)
    # ------------------------------------------------------------------

    @classmethod
    def empty(
        cls,
        data_dir: str,
        *,
        timeseries_dir: Optional[str] = None,
    ) -> "DataContext":
        """
        Return an empty DataContext.  The Dash app can use this immediately
        while a background thread loads the real data.
        """
        return cls(data_dir=data_dir, timeseries_dir=timeseries_dir)

    # ------------------------------------------------------------------
    # Simple constructor – "load everything from this dir"
    # ------------------------------------------------------------------

    @classmethod
    def from_dir(
        cls, data_dir: str, *, timeseries_dir: Optional[str] = None
    ) -> "DataContext":
        """
        Load features from a GLOBAL_DATA directory; curve tables are lazy.
        """
        base_dir, _ = io._resolve_layout(data_dir or str(Path.cwd()))
        data_dir = str(base_dir)
        feats = io.load_features_aggregate(data_dir)
        ctx = cls(
            data_dir=data_dir,
            timeseries_dir=timeseries_dir,
            features=feats,
            numeric_columns=[],
            lazy_curves=io.load_curves_tables_lazy(data_dir),
        )
        ctx._merge_pmf_annotations()
        return ctx

    # ------------------------------------------------------------------
    # CLI-oriented constructor – matches peptide_dash/cli.py arguments
    # ------------------------------------------------------------------

    @classmethod
    def from_cli(
        cls,
        data_dir: Optional[str],
        timeseries_dir: Optional[str],
        sample5: bool = False,
        sample_frac: Optional[float] = None,
        sample_seed: int = 0,
        sample_key: Optional[str] = None,  # kept for API parity
        dev_quick: bool = False,
        dev_cache: bool = False,
        dev_mock: bool = False,
        dev_max_files: Optional[int] = None,
        dev_cols: Optional[list[str]] = None,
        dev_cache_dir: Optional[str] = None,
    ) -> "DataContext":
        """
        CLI-oriented constructor.  Curve tables are always lazy.
        """
        base_dir, _ = io._resolve_layout(data_dir or str(Path.cwd()))
        data_dir = str(base_dir)

        if dev_mock:
            return cls(data_dir=data_dir, timeseries_dir=timeseries_dir)

        # ---------------- sampling of variants ----------------
        frac: Optional[float] = 0.05 if sample5 else sample_frac
        frac = _validate_sample_frac(frac) if frac is not None else None

        variants = None
        if frac is not None:
            variants = io.sample_variants_from_features(
                data_dir=data_dir,
                frac=frac,
                seed=int(sample_seed),
                variant_col="variant",
            )

        # ---------------- columns for quick mode ----------------
        usecols: Optional[list[str]] = dev_cols
        if usecols is None and dev_quick:
            sniffed = io.sniff_numeric_columns(
                data_dir=data_dir, max_files=2, max_numerics=60, variant_col="variant"
            )
            usecols = _filter_numeric_columns(sniffed)

        usecols = _ensure_variant_in_usecols(usecols, variant_col="variant")

        # ---------------- pick feature files (if variants known) ----------------
        paths = None
        if variants is not None and dev_max_files is not None:
            paths = io.choose_files_covering_variants(
                data_dir=data_dir,
                sampled_variants=variants,
                variant_col="variant",
                max_files=int(dev_max_files),
            )

        # ---------------- load features (with optional cache) ----------------
        cache_root = dev_cache_dir
        if dev_cache and cache_root is None:
            cache_root = str(Path(data_dir) / ".peptide_dash_cache")

        if dev_cache and cache_root is not None:
            cached = io.maybe_load_cache(
                cache_root=cache_root,
                data_dir=data_dir,
                variants=variants,
                usecols=usecols,
                max_files=dev_max_files,
            )
            if cached is not None:
                features = cached
            else:
                features = io.load_features_subset(
                    data_dir=data_dir,
                    variants=variants,
                    variant_col="variant",
                    usecols=usecols,
                    max_files=dev_max_files,
                    paths=paths,
                )
                io.write_cache(
                    cache_root=cache_root,
                    data_dir=data_dir,
                    variants=variants,
                    usecols=usecols,
                    max_files=dev_max_files,
                    df=features,
                )
        else:
            features = io.load_features_subset(
                data_dir=data_dir,
                variants=variants,
                variant_col="variant",
                usecols=usecols,
                max_files=dev_max_files,
                paths=paths,
            )

        numeric_only = _filter_numeric_columns(usecols) if usecols is not None else []

        ctx = cls(
            data_dir=data_dir,
            timeseries_dir=timeseries_dir,
            features=features,
            numeric_columns=numeric_only,
            lazy_curves=io.load_curves_tables_lazy(data_dir),
        )
        ctx._merge_pmf_annotations()
        return ctx
