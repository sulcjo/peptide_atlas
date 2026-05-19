# peptide_dash/data/loader.py
"""
Asynchronous background data loader.

Usage
-----
    from .data.loader import BackgroundLoader

    loader = BackgroundLoader(cli_args)
    loader.start()          # returns immediately; loading runs in a thread
    ctx = loader.wait()     # blocks until done (optional – only needed in tests)

The loader populates a *shared* DataContext instance in-place so that Dash
callbacks that hold a reference to `ctx` will automatically see the data once
loading finishes.  Callbacks should guard with `loader.is_ready()` and return
`dash.no_update` (or a placeholder figure) while data is still loading.
"""
from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass
from typing import Optional

from .context import DataContext
from . import io


# ---------------------------------------------------------------------------
# Phase constants
# ---------------------------------------------------------------------------

PHASE_IDLE = "idle"
PHASE_FEATURES = "loading features"
PHASE_CURVES = "loading curves"
PHASE_FINALISING = "finalising"
PHASE_DONE = "done"
PHASE_ERROR = "error"


# ---------------------------------------------------------------------------
# Public state snapshot (safe to read from any thread)
# ---------------------------------------------------------------------------

@dataclass
class LoaderState:
    phase: str = PHASE_IDLE
    message: str = ""
    # sub-progress from io._GLOBAL_PROGRESS (0–1)
    sub_frac: float = 0.0
    sub_done: int = 0
    sub_total: int = 0
    elapsed_s: float = 0.0
    eta_s: float = 0.0
    error: Optional[str] = None

    @property
    def overall_frac(self) -> float:
        """Rough 0-1 progress across the startup phases."""
        _weights = {
            PHASE_IDLE: 0.0,
            PHASE_FEATURES: 0.25,
            PHASE_CURVES: 0.85,
            PHASE_FINALISING: 0.92,
            PHASE_DONE: 1.0,
            PHASE_ERROR: 1.0,
        }
        base = _weights.get(self.phase, 0.0)
        if self.phase == PHASE_FEATURES:
            return base + 0.60 * self.sub_frac
        if self.phase == PHASE_CURVES:
            return base + 0.10 * self.sub_frac
        return base

    def as_dict(self) -> dict:
        return {
            "phase": self.phase,
            "message": self.message,
            "sub_frac": self.sub_frac,
            "sub_done": self.sub_done,
            "sub_total": self.sub_total,
            "elapsed_s": self.elapsed_s,
            "eta_s": self.eta_s,
            "overall_frac": self.overall_frac,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# BackgroundLoader
# ---------------------------------------------------------------------------

class BackgroundLoader:
    """
    Loads data in a background thread and populates `ctx` in-place.

    Parameters
    ----------
    ctx : DataContext
        A pre-constructed *empty* DataContext (from DataContext.empty()).
    cli_kwargs : dict
        The same keyword arguments that would have been passed to
        DataContext.from_cli() – minus `data_dir` which is already on ctx.
    """

    def __init__(self, ctx: DataContext, cli_kwargs: dict) -> None:
        self._ctx = ctx
        self._cli_kwargs = cli_kwargs
        self._state = LoaderState()
        self._lock = threading.Lock()
        self._ready_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._start_ts: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Kick off the background loading thread (non-blocking)."""
        self._start_ts = time.time()
        self._thread = threading.Thread(
            target=self._run,
            name="peptide-dash-loader",
            daemon=True,
        )
        self._thread.start()

    def is_ready(self) -> bool:
        return self._ready_event.is_set()

    def wait(self, timeout: Optional[float] = None) -> DataContext:
        self._ready_event.wait(timeout=timeout)
        return self._ctx

    def state(self) -> LoaderState:
        with self._lock:
            return LoaderState(**self._state.__dict__)

    def state_dict(self) -> dict:
        return self.state().as_dict()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _set_phase(self, phase: str, message: str = "") -> None:
        with self._lock:
            self._state.phase = phase
            self._state.message = message
            self._state.elapsed_s = time.time() - self._start_ts

    def _sync_sub_progress(self) -> None:
        """Copy the latest sub-progress from io._GLOBAL_PROGRESS."""
        snap = io.get_progress()
        with self._lock:
            total = int(snap.get("total", 0))
            done = int(snap.get("done", 0))
            self._state.sub_total = total
            self._state.sub_done = done
            self._state.sub_frac = snap.get("pct", 0.0)
            self._state.eta_s = snap.get("eta_s", 0.0)
            self._state.elapsed_s = time.time() - self._start_ts

    def _run(self) -> None:
        ctx = self._ctx
        kw = self._cli_kwargs

        try:
            data_dir = ctx.data_dir
            timeseries_dir = ctx.timeseries_dir

            # ---------- resolve layout (fast) ----------
            base_dir, _ = io._resolve_layout(data_dir)
            data_dir_resolved = str(base_dir)

            # ---------- sampling ----------
            self._set_phase(PHASE_FEATURES, "scanning variants …")

            frac = kw.get("sample_frac")
            seed = int(kw.get("sample_seed", 0))
            variants = None
            if frac is not None:
                variants = io.sample_variants_from_features(
                    data_dir=data_dir_resolved,
                    frac=float(frac),
                    seed=seed,
                    variant_col="variant",
                )

            # ---------- column selection ----------
            usecols = kw.get("dev_cols")
            if usecols is None and kw.get("dev_quick"):
                from .context import _filter_numeric_columns

                sniffed = io.sniff_numeric_columns(
                    data_dir=data_dir_resolved,
                    max_files=2,
                    max_numerics=60,
                    variant_col="variant",
                )
                usecols = _filter_numeric_columns(sniffed)

            from .context import _ensure_variant_in_usecols

            usecols = _ensure_variant_in_usecols(usecols, variant_col="variant")

            # ---------- file selection ----------
            paths = None
            dev_max = kw.get("dev_max_files")
            if variants is not None and dev_max is not None:
                paths = io.choose_files_covering_variants(
                    data_dir=data_dir_resolved,
                    sampled_variants=variants,
                    variant_col="variant",
                    max_files=int(dev_max),
                )

            # ---------- cache ----------
            cache_root = kw.get("dev_cache_dir")
            if kw.get("dev_cache") and cache_root is None:
                from pathlib import Path

                cache_root = str(Path(data_dir_resolved) / ".peptide_dash_cache")

            self._set_phase(PHASE_FEATURES, "loading feature files …")

            # Progress polling while features load.
            _poll_stop = threading.Event()

            def _poll() -> None:
                while not _poll_stop.is_set():
                    self._sync_sub_progress()
                    time.sleep(0.25)

            _poll_t = threading.Thread(target=_poll, daemon=True)
            _poll_t.start()

            try:
                if kw.get("dev_cache") and cache_root is not None:
                    cached = io.maybe_load_cache(
                        cache_root=cache_root,
                        data_dir=data_dir_resolved,
                        variants=variants,
                        usecols=usecols,
                        max_files=dev_max,
                    )
                    if cached is not None:
                        features = cached
                    else:
                        features = io.load_features_subset(
                            data_dir=data_dir_resolved,
                            variants=variants,
                            variant_col="variant",
                            usecols=usecols,
                            max_files=dev_max,
                            paths=paths,
                        )
                        io.write_cache(
                            cache_root=cache_root,
                            data_dir=data_dir_resolved,
                            variants=variants,
                            usecols=usecols,
                            max_files=dev_max,
                            df=features,
                        )
                else:
                    features = io.load_features_subset(
                        data_dir=data_dir_resolved,
                        variants=variants,
                        variant_col="variant",
                        usecols=usecols,
                        max_files=dev_max,
                        paths=paths,
                    )
            finally:
                _poll_stop.set()
                _poll_t.join(timeout=1.0)
                self._sync_sub_progress()

            # ---------- numeric columns ----------
            self._set_phase(PHASE_FINALISING, "preparing dashboard …")

            from .context import _filter_numeric_columns

            if usecols is not None:
                numeric_only = _filter_numeric_columns(usecols)
            else:
                numeric_only = []

            # ---------- populate ctx IN-PLACE ----------
            ctx.features = features
            ctx.numeric_columns = numeric_only
            if timeseries_dir is not None:
                ctx.timeseries_dir = timeseries_dir

            # Prime the lazy curves loader without actually reading curves/PMF
            # tables. Those should load on first access from the relevant tab.
            ctx._lazy_curves = io.load_curves_tables_lazy(data_dir_resolved)

            # Merge PMF annotation columns (basin geometry) into ctx.features
            # so that ctx._annotation_cols is populated for the PCA thermo preset.
            ctx._merge_pmf_annotations()

            self._set_phase(PHASE_DONE, "ready")
            self._ready_event.set()

        except Exception as exc:  # noqa: BLE001
            tb = traceback.format_exc()
            with self._lock:
                self._state.phase = PHASE_ERROR
                self._state.message = str(exc)
                self._state.error = tb
            self._ready_event.set()  # unblock any waiter
            raise
