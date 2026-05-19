from .shared import *


def _pmf_temperature_K() -> float:
    try:
        T = float(os.environ.get("PEPTIDE_DASH_PMF_KT_TEMPERATURE", "300"))
        return T if np.isfinite(T) and T > 0 else 300.0
    except Exception:
        return 300.0


def _pmf_temperature_from_frame(df: pd.DataFrame, default: Optional[float] = None) -> float:
    fallback = _pmf_temperature_K() if default is None else float(default)
    if isinstance(df, pd.DataFrame) and "T_K" in df.columns:
        vals = pd.to_numeric(df["T_K"], errors="coerce").to_numpy(dtype=float)
        vals = vals[np.isfinite(vals) & (vals > 0.0)]
        if vals.size:
            return float(np.nanmedian(vals))
    return fallback


def _pmf_temperature_from_curve_map(curve_map: dict[str, pd.DataFrame], default: Optional[float] = None) -> float:
    frames = [g for g in curve_map.values() if isinstance(g, pd.DataFrame) and "T_K" in g.columns]
    if not frames:
        return _pmf_temperature_K() if default is None else float(default)
    return _pmf_temperature_from_frame(pd.concat(frames, axis=0, ignore_index=True), default=default)


def _pmf_unit_factor_label(unit: Optional[str], T_K: Optional[float] = None) -> tuple[float, str]:
    """Factor multiplying kJ/mol values to display unit, plus axis label."""
    u = str(unit or "kJ/mol").strip().lower()
    if u in {"kcal", "kcal/mol", "kcal mol-1"}:
        return 1.0 / 4.184, "kcal/mol"
    if u in {"kt", "kbt", "kbT".lower()}:
        T_eff = float(T_K) if T_K is not None and np.isfinite(T_K) and float(T_K) > 0 else _pmf_temperature_K()
        kT = R_GAS * T_eff
        return 1.0 / max(kT, 1e-12), "kT"
    return 1.0, "kJ/mol"


def _pmf_y_column(df: pd.DataFrame) -> Optional[str]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return None
    return next((c for c in ("F_kJ_mol", "pmf_F_kJmol", "F", "y", "free_energy", "F_kJmol") if c in df.columns), None)


def _pmf_to_probability(F_kJ: np.ndarray, T_K: Optional[float] = None) -> np.ndarray:
    F = np.asarray(F_kJ, dtype=float)
    if not np.isfinite(F).any():
        return np.zeros_like(F, dtype=float)
    F = F - np.nanmin(F)
    T_eff = float(T_K) if T_K is not None and np.isfinite(T_K) and float(T_K) > 0 else _pmf_temperature_K()
    kT = max(R_GAS * T_eff, 1e-12)
    P = np.exp(-np.clip(F / kT, 0.0, 700.0))
    P = np.where(np.isfinite(P), P, 0.0)
    s = float(np.nansum(P))
    if s > 0:
        P = P / s
    return P


def _probability_to_pmf_kJ(P: np.ndarray, T_K: Optional[float] = None) -> np.ndarray:
    P = np.asarray(P, dtype=float)
    P = np.where(np.isfinite(P), P, 0.0)
    s = float(np.nansum(P))
    if s > 0:
        P = P / s
    T_eff = float(T_K) if T_K is not None and np.isfinite(T_K) and float(T_K) > 0 else _pmf_temperature_K()
    kT = max(R_GAS * T_eff, 1e-12)
    F = -kT * np.log(np.clip(P, 1e-300, None))
    if np.isfinite(F).any():
        F = F - np.nanmin(F)
    return F

def _variant_curve_map_for_metric(curves: pd.DataFrame, metric: str) -> tuple[dict[str, pd.DataFrame], Optional[str]]:
    if not isinstance(curves, pd.DataFrame) or curves.empty:
        return {}, None
    d = curves.copy()
    if "metric" in d.columns:
        d = d[d["metric"].astype(str) == str(metric)].copy()
    y_col = _pmf_y_column(d)
    if y_col is None or "x" not in d.columns or "variant" not in d.columns:
        return {}, None
    d["variant"] = d["variant"].astype(str)
    d["x"] = pd.to_numeric(d["x"], errors="coerce")
    d[y_col] = pd.to_numeric(d[y_col], errors="coerce")
    if "T_K" in d.columns:
        d["T_K"] = pd.to_numeric(d["T_K"], errors="coerce")
    d = d.dropna(subset=["variant", "x", y_col]).sort_values(["variant", "x"])
    out: dict[str, pd.DataFrame] = {}
    keep_cols = ["x", y_col] + (["T_K"] if "T_K" in d.columns else [])
    for v, g in d.groupby("variant", observed=True, sort=False):
        gg = g[keep_cols].dropna(subset=["x", y_col]).copy().sort_values("x")
        if len(gg) >= 2:
            out[str(v)] = gg
    return out, y_col


def _representative_pmf_for_variants(curve_map: dict[str, pd.DataFrame], variants: Sequence[str], y_col: str) -> Optional[dict]:
    variants = [str(v) for v in variants if str(v) in curve_map]
    if not variants:
        return None
    xs_all = []
    for v in variants:
        x = pd.to_numeric(curve_map[v]["x"], errors="coerce").dropna().to_numpy(dtype=float)
        if x.size >= 2:
            xs_all.append(x)
    if not xs_all:
        return None
    xmin = min(float(np.nanmin(x)) for x in xs_all)
    xmax = max(float(np.nanmax(x)) for x in xs_all)
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmax <= xmin:
        return None
    n_grid = int(np.clip(np.nanmedian([len(x) for x in xs_all]), 64, 320))
    grid = np.linspace(xmin, xmax, n_grid)
    P_rows = []
    F_rows = []
    T_rows = []
    kept = []
    for v in variants:
        g = curve_map[v]
        x = pd.to_numeric(g["x"], errors="coerce").to_numpy(dtype=float)
        F = pd.to_numeric(g[y_col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(x) & np.isfinite(F)
        if np.sum(m) < 2:
            continue
        x = x[m]
        F = F[m]
        order = np.argsort(x)
        x = x[order]
        F = F[order]
        if np.unique(x).size < 2:
            continue
        Fi = np.interp(grid, x, F, left=np.nan, right=np.nan)
        if not np.isfinite(Fi).any():
            continue
        if np.isfinite(Fi).any():
            Fi = Fi - np.nanmin(Fi)
        Pi = np.full_like(Fi, np.nan, dtype=float)
        finite = np.isfinite(Fi)
        T_v = _pmf_temperature_from_frame(g)
        Pi[finite] = _pmf_to_probability(Fi[finite], T_K=T_v)
        P_rows.append(Pi)
        F_rows.append(Fi)
        T_rows.append(T_v)
        kept.append(v)
    if not kept:
        return None
    P_arr = np.vstack(P_rows)
    F_arr = np.vstack(F_rows)
    with np.errstate(invalid="ignore"):
        meanP = np.nanmean(P_arr, axis=0)
        q25 = np.nanpercentile(F_arr, 25, axis=0)
        q75 = np.nanpercentile(F_arr, 75, axis=0)
    T_eff = float(np.nanmedian(np.asarray(T_rows, dtype=float))) if T_rows else _pmf_temperature_K()
    meanF = _probability_to_pmf_kJ(np.where(np.isfinite(meanP), meanP, 0.0), T_K=T_eff)
    # medoid: nearest sqrt(P) row to mean sqrt(P)
    medoid = kept[0]
    try:
        S = np.sqrt(np.nan_to_num(P_arr, nan=0.0, posinf=0.0, neginf=0.0))
        center = np.sqrt(np.nan_to_num(meanP, nan=0.0, posinf=0.0, neginf=0.0))
        dist = np.linalg.norm(S - center[None, :], axis=1)
        medoid = kept[int(np.nanargmin(dist))]
    except Exception:
        pass
    med_x = med_y = np.array([], dtype=float)
    try:
        gm = curve_map[medoid]
        med_x = pd.to_numeric(gm["x"], errors="coerce").to_numpy(dtype=float)
        med_y = pd.to_numeric(gm[y_col], errors="coerce").to_numpy(dtype=float)
        mm = np.isfinite(med_x) & np.isfinite(med_y)
        med_x = med_x[mm]
        med_y = med_y[mm]
        if med_y.size:
            med_y = med_y - np.nanmin(med_y)
    except Exception:
        pass
    return {"x": grid, "meanF": meanF, "q25": q25, "q75": q75, "medoid": medoid, "med_x": med_x, "med_y": med_y, "n": len(kept), "variants": kept, "T_K": T_eff}
