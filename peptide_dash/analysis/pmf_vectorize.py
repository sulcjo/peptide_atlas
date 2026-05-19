from __future__ import annotations

"""
peptide_dash.analysis.pmf_vectorize

Shared PMF preprocessing utilities used across tabs (UMAP, dendrogram, region PMF).

Scientific goals:
- Robust bin alignment (collapse float jitter, regularize grids).
- Treat PMFs as probability *mass* on a grid (normalize by ∫ p(x) dx).
- Provide consistent missingness policies (intersection, union+zeros, union+mean-impute).
- Produce consistent feature names and "family" parsing (metric blocks).

PMF table is expected to have columns:
- variant, metric, x
- either P (probability/density) or F_kJ_mol (free energy in kJ/mol)

Notes:
- If P is a density, we convert to mass via dx and normalize.
- If F is used, we convert to density via exp(-F/kT) (with per-variant min(F)=0 shift).
"""

from dataclasses import dataclass
import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

R_GAS = 0.008314462618  # kJ/mol/K


@dataclass(frozen=True)
class DefaultPmfCols:
    variant: str = "variant"
    metric: str = "metric"
    x: str = "x"
    p: str = "P"
    f: str = "F_kJ_mol"


def parse_family(colname: str) -> str:
    """
    Extract metric family from a column name.
    Convention: "<metric>|x=<bin>" => "<metric>"
    """
    if "|" in colname:
        return colname.split("|", 1)[0]
    return colname


def _finite_1d(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    return a[np.isfinite(a)]


_TORSION_METRIC_RE = re.compile(
    r"^(?:phi|psi)(?:$|[_-]res\d+(?:[_-][A-Za-z]{1,6})?$|[_-]?\d+(?:[_-][A-Za-z]{1,6})?$|[_-]?[A-Za-z]{1,6}\d+$)",
    re.I,
)


def is_periodic_pmf_metric(metric: object) -> bool:
    """True for phi/psi torsion PMFs whose support is periodic in degrees."""

    return bool(_TORSION_METRIC_RE.match(str(metric).strip()))


def _dx_from_unique(u: np.ndarray) -> float:
    if u.size < 2:
        return float("nan")
    d = np.diff(u)
    d = d[np.isfinite(d) & (d > 0)]
    if d.size == 0:
        return float("nan")
    return float(np.median(d))


def _decimals_from_dx(dx: float) -> int:
    """
    Conservative rounding based on dx.
    Example:
      dx ~ 0.1 -> decimals ~ 3
      dx ~ 0.01 -> decimals ~ 4
    """
    if not np.isfinite(dx) or dx <= 0:
        return 6
    if dx >= 1:
        return 0
    # keep 2 extra digits beyond dx scale
    # e.g., dx=0.12 -> -log10=0.92 -> floor=0 -> +2 => 2 (but we want 3-ish)
    # use ceil to be slightly safer
    order = int(np.ceil(-np.log10(dx)))
    return int(max(0, order + 2))


def stable_grid_from_xs(
    xs: np.ndarray,
    *,
    max_bins: int = 256,
    round_decimals: Optional[int] = None,
) -> np.ndarray:
    """
    Build a stable per-metric grid from raw x samples.

    Steps:
    1) drop non-finite
    2) collapse float jitter via rounding (decimals derived from dx unless provided)
    3) use unique rounded x; if too many bins, regularize to max_bins points
    """
    xs = _finite_1d(xs)
    if xs.size == 0:
        return np.array([], dtype=float)

    # collapse float jitter before dx estimation (helps avoid ultra-small diffs)
    u0 = np.unique(xs)
    dx0 = _dx_from_unique(u0)
    dec = int(round_decimals) if round_decimals is not None else _decimals_from_dx(dx0)

    u = np.unique(np.round(xs, dec)).astype(float)
    if u.size == 0:
        return np.array([], dtype=float)
    if u.size <= int(max_bins):
        return u

    dx = _dx_from_unique(u)
    x0 = float(np.min(u))
    x1 = float(np.max(u))

    if not np.isfinite(dx) or dx <= 0:
        return np.round(np.linspace(x0, x1, int(max_bins)), dec)

    n = int(round((x1 - x0) / dx)) + 1
    n = max(2, n)

    if n > int(max_bins):
        return np.round(np.linspace(x0, x1, int(max_bins)), dec)

    grid = x0 + dx * np.arange(n, dtype=float)
    return np.round(grid, dec)


def _kT(energy_units: str, T_K: float, kT_override: Optional[float]) -> float:
    if kT_override is not None and np.isfinite(kT_override) and float(kT_override) > 0:
        return float(kT_override)
    if str(energy_units) == "kT":
        return 1.0
    return float(R_GAS * max(1.0, float(T_K)))


def _temperature_for_curve(g: pd.DataFrame, fallback_T_K: float) -> float:
    """Median positive T_K from one curve, falling back to the caller value."""

    if isinstance(g, pd.DataFrame) and "T_K" in g.columns:
        vals = pd.to_numeric(g["T_K"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        if vals.size:
            return float(np.nanmedian(vals))
    return float(fallback_T_K)


def _variant_curve_density(
    sub_metric: pd.DataFrame,
    cols: DefaultPmfCols,
    variant: str,
    *,
    use_repr: str,
    energy_units: str,
    T_K: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return (x, density) for one metric+variant.
    Density is non-negative, but not necessarily normalized.
    """
    g = sub_metric[sub_metric[cols.variant].astype(str) == str(variant)]
    if g.empty:
        return np.array([], dtype=float), np.array([], dtype=float)

    x = pd.to_numeric(g[cols.x], errors="coerce").to_numpy(dtype=float)
    m = np.isfinite(x)
    x = x[m]
    if x.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    use_P = (str(use_repr) == "P") and (cols.p in g.columns)
    if use_P:
        p = pd.to_numeric(g[cols.p], errors="coerce").to_numpy(dtype=float)
        p = p[m]
        p = np.clip(p, 0.0, None)
    else:
        if cols.f in g.columns:
            F = pd.to_numeric(g[cols.f], errors="coerce").to_numpy(dtype=float)
            F = F[m]
            if F.size == 0 or not np.isfinite(F).any():
                return np.array([], dtype=float), np.array([], dtype=float)
            F = F - np.nanmin(F)
            kt = _kT(energy_units, _temperature_for_curve(g, T_K), None)
            p = np.exp(-F / max(kt, 1e-12))
        elif cols.p in g.columns:
            p = pd.to_numeric(g[cols.p], errors="coerce").to_numpy(dtype=float)
            p = p[m]
            p = np.clip(p, 0.0, None)
        else:
            return np.array([], dtype=float), np.array([], dtype=float)

    m2 = np.isfinite(p)
    x = x[m2]
    p = p[m2]
    if x.size == 0:
        return np.array([], dtype=float), np.array([], dtype=float)

    order = np.argsort(x)
    return x[order], p[order]


def _collapse_duplicate_x(x: np.ndarray, dens: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Average duplicate x coordinates before interpolation."""

    x = np.asarray(x, dtype=float)
    dens = np.asarray(dens, dtype=float)
    m = np.isfinite(x) & np.isfinite(dens)
    x = x[m]
    dens = dens[m]
    if x.size == 0:
        return x, dens
    order = np.argsort(x)
    x = x[order]
    dens = dens[order]
    xu, inv = np.unique(x, return_inverse=True)
    if xu.size == x.size:
        return x, dens
    sums = np.bincount(inv, weights=dens, minlength=xu.size)
    counts = np.bincount(inv, minlength=xu.size)
    return xu, sums / np.clip(counts, 1, None)


def density_to_mass_on_grid(
    x_grid: np.ndarray,
    x: np.ndarray,
    dens: np.ndarray,
    *,
    dirichlet_alpha: float = 0.0,
    periodic: bool = False,
    period: float = 360.0,
) -> np.ndarray:
    """
    Interpolate density to x_grid, convert to probability mass (dens*dx), normalize to sum=1.
    """
    x_grid = np.asarray(x_grid, dtype=float)
    if x_grid.size == 0:
        return np.array([], dtype=float)

    x = np.asarray(x, dtype=float)
    dens = np.asarray(dens, dtype=float)
    if x.size == 0 or dens.size == 0:
        # neutral fallback: uniform
        return np.full_like(x_grid, 1.0 / max(1, x_grid.size), dtype=float)

    x, dens = _collapse_duplicate_x(x, dens)
    if x.size == 0:
        return np.full_like(x_grid, 1.0 / max(1, x_grid.size), dtype=float)

    if periodic:
        dens_i = np.interp(x_grid, x, dens, period=float(period))
    else:
        dens_i = np.interp(x_grid, x, dens, left=0.0, right=0.0)
    dx = float(np.median(np.diff(x_grid))) if x_grid.size >= 2 else 1.0
    mass = np.clip(dens_i, 0.0, None) * dx

    if dirichlet_alpha and float(dirichlet_alpha) > 0:
        mass = mass + float(dirichlet_alpha)

    s = float(np.sum(mass))
    if not np.isfinite(s) or s <= 0:
        return np.full_like(x_grid, 1.0 / max(1, x_grid.size), dtype=float)
    return mass / s


def build_pmf_design_matrix(
    pmf_df: pd.DataFrame,
    metrics: Sequence[str],
    *,
    cols: DefaultPmfCols = DefaultPmfCols(),
    use_repr: str = "P",
    energy_units: str = "kJ/mol",
    T_K: float = 300.0,
    max_bins_per_metric: int = 256,
    round_decimals: Optional[int] = None,
    variant_policy: str = "intersection",
    missing_impute: str = "zeros",  # for union policies: "zeros"|"mean"|"uniform"
    dirichlet_alpha: float = 0.0,
) -> Tuple[np.ndarray, List[str], pd.DataFrame, Dict[str, np.ndarray]]:
    """
    Build a concatenated PMF design matrix (probability mass per bin) for embedding or PCA.

    Returns
    -------
    X        : (n_variants, n_features) probability mass values (each metric block sums to 1 per row)
    colnames : "<metric>|x=<bin>"
    meta     : DataFrame({"variant": variants})
    grids    : {metric: x_grid}

    Variant policies:
    - "intersection": keep only variants present in ALL selected metrics
    - "union": include union of variants, fill missing with `missing_impute`
    - "union_mean": union + mean-impute per metric
    """
    if not isinstance(pmf_df, pd.DataFrame) or pmf_df.empty or not metrics:
        return np.zeros((0, 0)), [], pd.DataFrame(), {}

    # Determine per-metric variant sets
    variant_sets: List[set[str]] = []
    variant_union: set[str] = set()
    sub_by_metric: Dict[str, pd.DataFrame] = {}

    for m in metrics:
        sub = pmf_df[pmf_df[cols.metric].astype(str) == str(m)].copy()
        if sub.empty:
            continue
        sub[cols.variant] = sub[cols.variant].astype(str)
        sub_by_metric[str(m)] = sub
        vs = set(sub[cols.variant].dropna().astype(str).tolist())
        if vs:
            variant_sets.append(vs)
            variant_union |= vs

    if not sub_by_metric:
        return np.zeros((0, 0)), [], pd.DataFrame(), {}

    policy = str(variant_policy or "intersection").lower().strip()
    if policy.startswith("inter") and variant_sets:
        variants = sorted(set.intersection(*variant_sets))
    else:
        variants = sorted(variant_union)

    if not variants:
        return np.zeros((0, 0)), [], pd.DataFrame(), {}

    grids: Dict[str, np.ndarray] = {}
    blocks: List[np.ndarray] = []
    colnames: List[str] = []

    for m, sub in sub_by_metric.items():
        periodic = is_periodic_pmf_metric(m)
        xs = pd.to_numeric(sub[cols.x], errors="coerce").dropna().to_numpy(dtype=float)
        x_grid = stable_grid_from_xs(xs, max_bins=int(max_bins_per_metric), round_decimals=round_decimals)
        if x_grid.size == 0:
            continue
        grids[m] = x_grid

        P = np.zeros((len(variants), x_grid.size), dtype=float)
        present_mask = np.zeros(len(variants), dtype=bool)

        for i, v in enumerate(variants):
            x, dens = _variant_curve_density(sub, cols, v, use_repr=use_repr, energy_units=energy_units, T_K=T_K)
            if x.size == 0:
                continue
            P[i, :] = density_to_mass_on_grid(
                x_grid,
                x,
                dens,
                dirichlet_alpha=float(dirichlet_alpha),
                periodic=periodic,
                period=360.0,
            )
            present_mask[i] = True

        if not policy.startswith("inter"):
            impute_mode = str(missing_impute or "zeros").lower().strip()
            missing_idx = np.where(~present_mask)[0]
            if missing_idx.size:
                if policy in {"union_mean", "union-mean", "union_mean_impute"} or impute_mode == "mean":
                    mean_vec = P[present_mask].mean(axis=0) if present_mask.any() else np.full(x_grid.size, 1.0 / x_grid.size)
                    P[missing_idx, :] = mean_vec
                elif impute_mode in {"uniform", "uni"}:
                    P[missing_idx, :] = 1.0 / max(1, x_grid.size)
                else:
                    # zeros: neutral for cosine-like metrics, but can cluster missingness; kept as option.
                    P[missing_idx, :] = 0.0

        blocks.append(P)
        colnames.extend([f"{m}|x={x:g}" for x in x_grid])

    if not blocks:
        return np.zeros((0, 0)), [], pd.DataFrame(), {}

    X = np.concatenate(blocks, axis=1)
    meta = pd.DataFrame({"variant": variants})
    return X, colnames, meta, grids
