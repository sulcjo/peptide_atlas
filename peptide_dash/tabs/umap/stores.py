from .shared import *
from .figures import _embedding_method_label

UMAP_CACHE_VERSION = 1

def _umap_cache_dir(ctx=None) -> Path:
    """Persistent on-disk cache folder for embedding coordinates.

    Default location is inside the active data directory when discoverable,
    otherwise ./.peptide_dash_cache/umap_embeddings. Override with:
      PEPTIDE_DASH_UMAP_CACHE_DIR=/path/to/cache
    """
    env = os.environ.get("PEPTIDE_DASH_UMAP_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    for attr in ("data_dir", "root_dir", "base_dir", "root", "path"):
        try:
            val = getattr(ctx, attr, None)
        except Exception:
            val = None
        if val:
            try:
                pp = Path(val).expanduser().resolve()
                if pp.exists():
                    return pp / ".peptide_dash_cache" / "umap_embeddings"
            except Exception:
                pass
    return (Path.cwd() / ".peptide_dash_cache" / "umap_embeddings").resolve()


def _json_safe_records(df: pd.DataFrame) -> list[dict]:
    if not isinstance(df, pd.DataFrame) or df.empty:
        return []
    clean = df.copy()
    clean = clean.replace([np.inf, -np.inf], np.nan)
    return json.loads(clean.to_json(orient="records", date_format="iso"))


def _df_light_fingerprint(df: pd.DataFrame, *, include_numeric_summary: bool = False) -> dict:
    """Cheap, stable-ish fingerprint for cache invalidation.

    Avoid hashing complete PMF tables because they can be huge. We fingerprint
    shapes, columns, variants/metrics, and optionally a coarse numeric summary
    for feature matrices. This is intended to prevent obvious stale-cache reuse;
    users can delete the cache folder if the underlying data changed radically.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        return {"empty": True}
    out = {
        "shape": [int(df.shape[0]), int(df.shape[1])],
        "columns_hash": hashlib.sha1("\n".join(map(str, df.columns)).encode("utf-8", "ignore")).hexdigest()[:16],
    }
    for col in ("variant", "metric"):
        if col in df.columns:
            vals = df[col].dropna().astype(str).drop_duplicates().sort_values().tolist()
            out[f"n_{col}"] = len(vals)
            out[f"{col}_hash"] = hashlib.sha1("\n".join(vals).encode("utf-8", "ignore")).hexdigest()[:16]
    if include_numeric_summary:
        try:
            num = df.select_dtypes(include=[np.number])
            if not num.empty:
                means = num.mean(axis=0, skipna=True).replace([np.inf, -np.inf], np.nan).fillna(0.0)
                # Round to avoid noisy JSON float tails while still detecting common changes.
                txt = "\n".join(f"{k}:{float(v):.8g}" for k, v in means.items())
                out["numeric_mean_hash"] = hashlib.sha1(txt.encode("utf-8", "ignore")).hexdigest()[:16]
        except Exception:
            pass
    return out


def _embedding_cache_params(**kwargs) -> dict:
    """Return a JSON-stable subset of user parameters that define an embedding."""
    def norm(v):
        if isinstance(v, (list, tuple, set)):
            return sorted(str(x) for x in v)
        if isinstance(v, (np.integer, int)):
            return int(v)
        if isinstance(v, (np.floating, float)):
            return float(v)
        if isinstance(v, bool) or v is None:
            return v
        return str(v)
    return {k: norm(v) for k, v in sorted(kwargs.items())}


def _embedding_cache_key(params: dict, feats_df: pd.DataFrame, pmf_df: pd.DataFrame) -> str:
    payload = {
        "version": UMAP_CACHE_VERSION,
        "params": params,
        "features": _df_light_fingerprint(feats_df, include_numeric_summary=True),
        "pmf": _df_light_fingerprint(pmf_df, include_numeric_summary=True),
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode("utf-8", "ignore")).hexdigest()


def _embedding_cache_path(ctx, key: str) -> Path:
    return _umap_cache_dir(ctx) / f"embedding_{str(key)[:16]}.json"


def _save_embedding_cache(ctx, key: str, plot_df: pd.DataFrame, info: str, params: dict) -> Optional[Path]:
    try:
        cache_dir = _umap_cache_dir(ctx)
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = _embedding_cache_path(ctx, key)
        payload = {
            "version": UMAP_CACHE_VERSION,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "cache_key": key,
            "params": params,
            "info": str(info or ""),
            "plot_df_records": _json_safe_records(plot_df),
        }
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)
        last = cache_dir / "last_embedding.json"
        last_payload = {"path": str(path), "cache_key": key, "created_at": payload["created_at"]}
        last.write_text(json.dumps(last_payload, indent=2, sort_keys=True), encoding="utf-8")
        return path
    except Exception:
        return None


def _load_embedding_cache_payload(ctx, *, key: Optional[str] = None, last: bool = False) -> tuple[Optional[dict], str]:
    try:
        cache_dir = _umap_cache_dir(ctx)
        if last:
            last_path = cache_dir / "last_embedding.json"
            if not last_path.exists():
                return None, f"No cached embedding pointer found in {cache_dir}."
            ptr = json.loads(last_path.read_text(encoding="utf-8"))
            path = Path(ptr.get("path", ""))
        elif key:
            path = _embedding_cache_path(ctx, key)
        else:
            return None, "No cache key supplied."
        if not path.exists():
            return None, f"Cached embedding file not found: {path}"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if int(payload.get("version", -1)) != UMAP_CACHE_VERSION:
            return None, "Cached embedding version is incompatible."
        recs = payload.get("plot_df_records", [])
        if not recs:
            return None, "Cached embedding contained no plotted records."
        return payload, f"Loaded cached embedding from {path}."
    except Exception as e:
        return None, f"Could not load cached embedding: {e}"


def _plot_df_from_embedding_cache(payload: dict) -> pd.DataFrame:
    recs = payload.get("plot_df_records", []) if isinstance(payload, dict) else []
    df = pd.DataFrame(recs)
    for c in ("UMAP1", "UMAP2", "UMAP3"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "variant" in df.columns:
        df["variant"] = df["variant"].astype(str)
    return df


def _embedding_array_from_plot_df(plot_df: pd.DataFrame, dims: int) -> np.ndarray:
    cols = ["UMAP1", "UMAP2"] + (["UMAP3"] if int(dims or 2) >= 3 and "UMAP3" in plot_df.columns else [])
    cols = [c for c in cols if c in plot_df.columns]
    if not cols:
        return np.zeros((0, 0), dtype=float)
    return plot_df[cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)


def _cache_summary_text(path: Optional[Path], key: str) -> str:
    cache_dir = _umap_cache_dir(None)
    if path is not None:
        return f"embedding cache saved: {path.name}; key={str(key)[:12]}…"
    return f"embedding cache save failed; cache_dir={cache_dir}"

def _plot_df_store_payload(plot_df: pd.DataFrame, method: Optional[str], dims: int, ui_revision: Optional[str] = None) -> dict:
    """Serialize the plotted embedding table for downstream trajectory/residue modules."""
    if not isinstance(plot_df, pd.DataFrame) or plot_df.empty:
        return {"records": [], "method": _embedding_method_label(method), "dims": int(dims or 2)}
    keep = [c for c in ["variant", "UMAP1", "UMAP2", "UMAP3", "role", "dbscan_cluster"] if c in plot_df.columns]
    # Keep scalar feature columns too, so recoloring and feature-driven
    # trajectories work after quickload/app restart. Cap gently to avoid
    # turning dcc.Store into a small data warehouse with delusions of grandeur.
    preferred: list = []
    rest: list = []
    for c in plot_df.columns:
        if c in keep or c == "variant":
            continue
        cs = str(c)
        if "global_basin_min_x" in cs or "secondary_basin_min_x" in cs or "basin" in cs:
            preferred.append(c)
        elif cs.startswith("DC") and cs[2:].isdigit():
            preferred.append(c)
        else:
            rest.append(c)
    for c in preferred + rest:
        if len(keep) >= 256:
            break
        keep.append(c)
    return {
        "records": _json_safe_records(plot_df[keep]),
        "method": _embedding_method_label(method),
        "dims": int(dims or 2),
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "ui_revision": str(ui_revision or time.strftime("%Y%m%d%H%M%S")),
    }


def _plot_df_from_store(data: Any) -> pd.DataFrame:
    if isinstance(data, dict):
        recs = data.get("records", []) or []
    elif isinstance(data, list):
        recs = data
    else:
        recs = []
    df = pd.DataFrame(recs)
    if df.empty:
        return df
    if "variant" in df.columns:
        df["variant"] = df["variant"].astype(str)
    for c in ("UMAP1", "UMAP2", "UMAP3"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df
