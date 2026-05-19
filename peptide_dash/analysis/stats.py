from __future__ import annotations

from typing import Dict

import numpy as np


def moments_from_hist(counts, edges) -> Dict[str, float]:
    """
    Compute mean/var/skew/kurt from a histogram.
    `edges` are bin edges, `counts` are per-bin counts.
    """
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    if counts.size == 0 or edges.size != counts.size + 1:
        return {"mean": np.nan, "var": np.nan, "skew": np.nan, "kurt": np.nan}

    centers = 0.5 * (edges[:-1] + edges[1:])
    w = counts / max(np.sum(counts), 1e-12)
    mean = float(np.sum(w * centers))
    var = float(np.sum(w * (centers - mean) ** 2))
    sd = max(np.sqrt(var), 1e-12)
    skew = float(np.sum(w * ((centers - mean) / sd) ** 3))
    kurt = float(np.sum(w * ((centers - mean) / sd) ** 4) - 3.0)
    return {"mean": mean, "var": var, "skew": skew, "kurt": kurt}


def circ_stats_from_hist_deg(counts, edges_deg) -> Dict[str, float]:
    """
    Circular stats from a histogram on degrees.

    Returns mean_angle_deg and circ_var (1 - R).
    """
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges_deg, dtype=float)
    if counts.size == 0 or edges.size != counts.size + 1:
        return {"mean_angle_deg": np.nan, "circ_var": np.nan}

    centers_deg = 0.5 * (edges[:-1] + edges[1:])
    ang = np.deg2rad(centers_deg)
    w = counts / max(np.sum(counts), 1e-12)

    C = float(np.sum(w * np.cos(ang)))
    S = float(np.sum(w * np.sin(ang)))
    R = float(np.sqrt(C * C + S * S))

    mean_ang = float(np.arctan2(S, C))
    mean_deg = float((np.rad2deg(mean_ang) + 360.0) % 360.0)
    circ_var = float(1.0 - R)
    return {"mean_angle_deg": mean_deg, "circ_var": circ_var}


# -------------------------------------------------------------------
# Time-series aware statistics
# -------------------------------------------------------------------

def integrated_autocorr_time(acf: np.ndarray, *, cutoff: str = "positive") -> float:
    """Estimate integrated autocorrelation time τ_int from an ACF.

    Parameters
    ----------
    acf:
        Autocorrelation values starting at lag 0 where acf[0] == 1.
    cutoff:
        - "positive": stop at first non-positive ACF value.
        - "none": sum all finite values.

    Notes
    -----
    Uses the common definition:
        τ_int = 0.5 + Σ_{k>=1} ρ_k
    so that ESS ≈ N / (2 τ_int).

    Returns
    -------
    float
        τ_int >= 0.5 (falls back to 0.5 if insufficient data).
    """
    if acf is None:
        return 0.5
    acf = np.asarray(acf, dtype=float)
    if acf.size < 2 or not np.isfinite(acf[0]) or acf[0] == 0:
        return 0.5

    rho = acf / float(acf[0])
    rho = rho[np.isfinite(rho)]
    if rho.size < 2:
        return 0.5

    s = 0.0
    for r in rho[1:]:
        if cutoff == "positive" and r <= 0.0:
            break
        s += float(r)

    tau = 0.5 + s
    return float(max(tau, 0.5))


def effective_sample_size(n: int, tau_int: float) -> float:
    """Approximate ESS for an autocorrelated series."""
    try:
        n_i = int(n)
    except Exception:
        return float("nan")
    if n_i <= 1:
        return float(n_i)
    tau = float(tau_int) if np.isfinite(tau_int) else 0.5
    tau = max(tau, 0.5)
    return float(n_i / (2.0 * tau))


def js_divergence(p: np.ndarray, q: np.ndarray, *, eps: float = 1e-12) -> float:
    """Jensen–Shannon divergence (natural log)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    p = np.clip(p, 0.0, np.inf)
    q = np.clip(q, 0.0, np.inf)
    ps = float(np.sum(p))
    qs = float(np.sum(q))
    if ps <= 0 or qs <= 0:
        return float("nan")
    p = p / ps
    q = q / qs
    m = 0.5 * (p + q)

    def _kl(a, b):
        a = np.clip(a, eps, 1.0)
        b = np.clip(b, eps, 1.0)
        return float(np.sum(a * np.log(a / b)))

    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def normal_ci_for_prob(p: np.ndarray, n_eff: float, *, z: float = 1.959964) -> tuple[np.ndarray, np.ndarray]:
    """Normal-approx CI for a probability vector p with effective sample size n_eff.

    Clips bounds to [0, 1] and does not enforce sum-to-1.
    """
    p = np.asarray(p, dtype=float)
    n = float(n_eff) if np.isfinite(n_eff) else float(np.sum(p))
    n = max(n, 1.0)
    se = np.sqrt(np.clip(p * (1.0 - p) / n, 0.0, np.inf))
    lo = np.clip(p - z * se, 0.0, 1.0)
    hi = np.clip(p + z * se, 0.0, 1.0)
    return lo, hi


def prob_mass_from_counts(counts: np.ndarray, *, alpha: float = 0.0) -> np.ndarray:
    """Dirichlet-smoothed probability mass from histogram counts."""
    c = np.asarray(counts, dtype=float)
    c = np.clip(c, 0.0, np.inf)
    a = float(alpha) if alpha is not None else 0.0
    a = max(a, 0.0)
    denom = float(np.sum(c) + a * c.size)
    if denom <= 0:
        return np.full_like(c, np.nan, dtype=float)
    return (c + a) / denom
