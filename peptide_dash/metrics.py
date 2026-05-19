from __future__ import annotations

"""peptide_dash.metrics

Small utilities for parsing + formatting metric names, with special support for
per-residue torsion metrics (phi/psi) that may optionally include residue names.

Why this exists:
- multiple pipelines encode torsion metrics differently (phi_res5, phi_Y5, psi_TYR12, ...)
- tabs should sort + display those consistently

Accepted torsion metric examples (case-insensitive):
- phi_res5
- psi_res12
- phi_5
- psi-12
- phi_Y5
- psi_TYR12
- phi_res5_Y
- psi_res12_TYR
- phi12
"""

from dataclasses import dataclass
import numpy as np
import re
from typing import Optional, Tuple


@dataclass(frozen=True, slots=True)
class TorsionMetric:
    """Parsed representation of a per-residue torsion metric."""

    kind: str  # "phi" or "psi"
    resid: int
    resname: Optional[str] = None  # e.g. Y / TYR (already encoded in metric), optional


_TORSION_RE: tuple[re.Pattern[str], ...] = (
    # phi_res12, phi-res12, phi_res12_TYR
    re.compile(
        r"^(?P<kind>phi|psi)[_-]res(?P<resid>\d+)(?:[_-](?P<resname>[A-Za-z]{1,6}))?$", re.I
    ),
    # phi_12, phi-12, phi_12_TYR
    re.compile(
        r"^(?P<kind>phi|psi)[_-](?P<resid>\d+)(?:[_-](?P<resname>[A-Za-z]{1,6}))?$", re.I
    ),
    # phiY12, psiTYR5, phi_Y12
    re.compile(r"^(?P<kind>phi|psi)[_-]?(?P<resname>[A-Za-z]{1,6})(?P<resid>\d+)$", re.I),
    # phi12, psi5
    re.compile(r"^(?P<kind>phi|psi)(?P<resid>\d+)$", re.I),
)


def parse_torsion_metric(metric: object) -> Optional[TorsionMetric]:
    """Parse a torsion metric name. Returns None if not a phi/psi metric."""

    s = str(metric).strip()
    if not s:
        return None

    for rx in _TORSION_RE:
        m = rx.match(s)
        if not m:
            continue

        kind = str(m.group("kind")).lower()
        resid = int(m.group("resid"))
        resname = m.groupdict().get("resname")
        if resname is not None:
            resname = str(resname).strip().upper() or None

        return TorsionMetric(kind=kind, resid=resid, resname=resname)

    return None


def is_torsion_metric(metric: object) -> bool:
    """True if metric looks like a per-residue phi/psi metric."""

    return parse_torsion_metric(metric) is not None


def torsion_sort_key(metric: object) -> tuple:
    """Consistent ordering across tabs.

    - torsions first: phi before psi, then residue index
    - everything else: case-insensitive lexical
    """

    t = parse_torsion_metric(metric)
    if t is None:
        return (1, str(metric).lower())

    kind_rank = 0 if t.kind == "phi" else 1
    return (0, kind_rank, int(t.resid), str(metric).lower())


def _variant_residue_index0(variant: str, resid: int) -> Optional[Tuple[int, int]]:
    """Best-effort mapping from resid to 0-based index in a variant string.

    Heuristic:
      - if 1..L: assume 1-based, index0=resid-1, display_resid=resid
      - else if 0..L-1: assume 0-based, index0=resid, display_resid=resid+1
    """

    if not variant:
        return None
    L = len(str(variant))
    if L <= 0:
        return None

    if 1 <= resid <= L:
        return resid - 1, resid
    if 0 <= resid < L:
        return resid, resid + 1
    return None


def metric_display_label(metric: object, *, variant: str | None = None) -> str:
    """Human label for dropdowns, titles, legend entries.

    If metric is torsion:
      - use φ/ψ
      - prefer resname already encoded in metric
      - else, if `variant` provided, infer residue letter from variant sequence
    """

    t = parse_torsion_metric(metric)
    if t is None:
        return str(metric)

    sym = "φ" if t.kind == "phi" else "ψ"

    resname = t.resname
    display_resid = t.resid

    if resname is None and variant:
        idx = _variant_residue_index0(str(variant), int(t.resid))
        if idx is not None:
            i0, display_resid = idx
            try:
                resname = str(variant)[i0].upper()
            except Exception:
                resname = None

    if resname:
        return f"{sym} {resname}{int(display_resid)}"
    return f"{sym} res{int(display_resid)}"


def all_torsion(metrics: object) -> bool:
    """True if `metrics` is a non-empty iterable and all entries are torsions."""

    try:
        items = list(metrics or [])
    except TypeError:
        return False
    return bool(items) and all(is_torsion_metric(m) for m in items)


def residue_display(variant: str, resid: int) -> str:
    """Label helper for residue selectors. Returns e.g. "Y5" or "res5"."""

    idx = _variant_residue_index0(str(variant or ""), int(resid))
    if idx is None:
        return f"res{int(resid)}"
    i0, display_resid = idx
    aa = str(variant)[i0].upper() if variant else ""
    return f"{aa}{int(display_resid)}" if aa else f"res{int(display_resid)}"


def torsion_prefix_from_column(col: object) -> Optional[TorsionMetric]:
    """Return parsed torsion metric from a feature column prefix.

    Examples
    --------
    - "phi_res5__circular_mean_deg" -> TorsionMetric(kind="phi", resid=5)
    - "psi_TYR12__std" -> TorsionMetric(kind="psi", resid=12, resname="TYR")
    """

    c = str(col)

    # Circular encoding appends _sin/_cos to the full feature name, e.g.
    # phi_res5__circular_mean_deg_sin. Strip that suffix and parse the
    # original prefix.
    if c.endswith("_sin") or c.endswith("_cos"):
        c = c.rsplit("_", 1)[0]
    prefix = c.split("__", 1)[0]
    return parse_torsion_metric(prefix)


def is_torsion_feature_column(col: object) -> bool:
    """True if a feature column belongs to a torsion metric prefix."""
    return torsion_prefix_from_column(col) is not None



# ---------------------- Circular encoding for torsion angles ----------------------

_TORSION_ANGLE_REST: set[str] = {
    # circular stats (degrees)
    "circular_mean_deg",
    "circular_median_deg",
    "circular_q25_deg",
    "circular_q75_deg",
    "circular_mode_deg",
    # PMF-derived location features (angles)
    "min1_x",
    "min2_x",
    "x_eqpop",
    # robust linear stats that represent an angle (not a spread)
    "mean",
    "median",
    "min",
    "max",
}

def is_torsion_angle_feature_column(col: object) -> bool:
    """True if `col` is a torsion-derived feature that represents an angle."""
    c = str(col)
    if "__" not in c:
        return False
    prefix, rest = c.split("__", 1)
    if parse_torsion_metric(prefix) is None:
        return False
    # handle encoded derivatives (e.g. ..._sin/_cos)
    if rest.endswith("_sin") or rest.endswith("_cos"):
        rest = rest.rsplit("_", 1)[0]
    if rest in _TORSION_ANGLE_REST:
        return True
    # permissive: anything explicitly tagged as degrees
    if rest.lower().endswith("_deg"):
        return True
    return False


def circular_encode_torsion_angles(
    X: np.ndarray,
    cols: list[str],
    *,
    enabled: bool = True,
    drop_original: bool = True,
) -> tuple[np.ndarray, list[str], str]:
    """Replace torsion-angle columns with sin/cos encoding.

    Why: angles wrap at ±180°; representing them as sin/cos avoids artificial
    discontinuities that can distort PCA/UMAP.

    Parameters
    ----------
    X : (n, p) matrix aligned with `cols`
    cols : list of column names
    enabled : if False, no-op
    drop_original : if True, remove original angle columns; otherwise keep them too

    Returns
    -------
    X2, cols2, note
    """
    if not enabled or X.size == 0 or not cols:
        return X, cols, ""

    idxs = [i for i, c in enumerate(cols) if is_torsion_angle_feature_column(c)]
    if not idxs:
        return X, cols, ""

    # Build new columns in-place order: replace each angle col with [sin, cos]
    keep_cols: list[str] = []
    blocks: list[np.ndarray] = []
    for i, c in enumerate(cols):
        if i not in set(idxs):
            keep_cols.append(c)
            blocks.append(X[:, [i]])
            continue

        ang = X[:, i]
        rad = np.deg2rad(ang.astype(float))
        s = np.sin(rad)
        co = np.cos(rad)

        if not drop_original:
            keep_cols.append(c)
            blocks.append(ang.reshape(-1, 1))

        keep_cols.append(f"{c}_sin")
        keep_cols.append(f"{c}_cos")
        blocks.append(s.reshape(-1, 1))
        blocks.append(co.reshape(-1, 1))

    X2 = np.concatenate(blocks, axis=1) if blocks else np.zeros((X.shape[0], 0), float)
    note = f"circular-encoded torsion angles: {len(idxs)} → {len(idxs) * (2 if drop_original else 3)} cols"
    return X2, keep_cols, note



def prettify_column_label(col: object, *, variant: str | None = None) -> str:
    """Prettify a feature/metric column label.

    - If the column is a plain torsion metric (phi_res5 / psi_Y12), returns a compact φ/ψ label.
    - If the column is a feature derived from a torsion metric (phi_res5__mean),
      returns "φ res5 — mean" (or with residue letter if inferrable).
    """

    c = str(col)
    if is_torsion_metric(c):
        return metric_display_label(c, variant=variant)

    if "__" in c:
        prefix, rest = c.split("__", 1)
        t = parse_torsion_metric(prefix)
        if t is not None:
            return f"{metric_display_label(prefix, variant=variant)} — {rest}"

    return c
