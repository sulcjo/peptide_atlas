from __future__ import annotations

import numpy as np


def bin_width(edges) -> float:
    edges = np.asarray(edges, dtype=float)
    if edges.size < 2:
        return float("nan")
    return float(np.median(np.diff(edges)))


def density_to_pmf(hist, bin_width_val: float) -> np.ndarray:
    """
    Convert a probability density (or counts) into a PMF-like free energy.

    F = -ln(p) with p normalized.
    """
    h = np.asarray(hist, dtype=float)
    h = np.clip(h, 0.0, None)
    bw = max(float(bin_width_val), 1e-12)
    p = h / max(np.sum(h) * bw, 1e-12)
    p = np.clip(p, 1e-300, None)
    return -np.log(p)
