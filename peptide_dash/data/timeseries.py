from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

__TORSION_PERRES_RE = re.compile(
    r"^(?P<prefix>.+?)(?:__|_)(?P<torsion>phi|psi)_repl_?(?P<repl>[^_]+)_(?P<resid>\d+)$"
)


def _parse_variant_metric_replica(filename_noext: str, variant_dir: Optional[str]) -> Tuple[str, str, Optional[int]]:
    """Parse (variant, metric, replica) from a timeseries filename (no extension).

    Best-effort rules (in order):
      1) Per-residue torsions: <prefix>__phi_repl61_5 (or ..._psi_repl_61_5)
      2) Common replica suffixes: *_rep2, *_rep_2, *_repl2, *_replica_2, *_r1
      3) Trailing underscore digits: rg_3 -> metric=rg, replica=3
      4) Fallback: replica=None

    Variant is taken from the parent directory name (`variant_dir`) when available.
    """
    base = str(filename_noext)
    variant = variant_dir or "unknown"

    # Per-residue torsions (batch naming)
    m = __TORSION_PERRES_RE.match(base)
    if m:
        tors = str(m.group("torsion"))
        repl_s = str(m.group("repl"))
        resid_s = str(m.group("resid"))
        metric = f"{tors}_res{int(resid_s)}"
        replica = int(repl_s) if repl_s.isdigit() else None
        if not variant_dir:
            prefix = m.groupdict().get("prefix") or "unknown"
            variant = str(prefix)
        return variant, metric, replica

    # Common replica suffixes
    m = re.match(
        r"^(?P<metric>.+?)(?:[_-](?:replica|repl|rep|r)[_-]?(?P<rep>\d+))$",
        base,
        flags=re.IGNORECASE,
    )
    if m:
        metric = str(m.group("metric")).strip("_-") or base
        replica = int(m.group("rep"))
        return variant, metric, replica

    # Trailing underscore digits
    parts = base.split("_")
    if parts and parts[-1].isdigit():
        replica = int(parts[-1])
        metric = "_".join(parts[:-1]) or base
        return variant, metric, replica

    return variant, base, None


def discover_timeseries_files(data_dir: str | None, timeseries_dir: str | None = None) -> pd.DataFrame:
    """Discover timeseries *.xvg files.

    Priority:
      1) explicit timeseries_dir (if exists)
      2) data_dir/TIMESERIES (if exists)

    Returns a DataFrame with columns: variant, metric, path, replica
    """
    ts_dir: Optional[str] = None
    if timeseries_dir and os.path.isdir(timeseries_dir):
        ts_dir = timeseries_dir
    elif data_dir:
        cand = os.path.join(str(data_dir), "TIMESERIES")
        if os.path.isdir(cand):
            ts_dir = cand

    rows: List[Dict[str, Any]] = []
    if not ts_dir:
        return pd.DataFrame(columns=["variant", "metric", "path", "replica"])

    for cur_root, _, files in os.walk(ts_dir):
        xvg_files = [f for f in files if f.endswith(".xvg")]
        if not xvg_files:
            continue

        rel = os.path.relpath(cur_root, ts_dir)
        parts = rel.split(os.sep)
        variant_dir = parts[0] if parts and parts[0] not in (".", "") else None

        for fn in xvg_files:
            path = os.path.join(cur_root, fn)
            base = os.path.splitext(os.path.basename(fn))[0]
            variant, metric, replica = _parse_variant_metric_replica(base, variant_dir)
            rows.append(dict(variant=variant, metric=metric, path=path, replica=replica))

    if not rows:
        return pd.DataFrame(columns=["variant", "metric", "path", "replica"])

    return pd.DataFrame(rows)
