from .shared import *


def _ctx_features_df(ctx: Any, fallback: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Return the freshest feature dataframe available on the shared context."""
    try:
        df = getattr(ctx, "df", getattr(ctx, "features_df", getattr(ctx, "features", None)))
        if isinstance(df, pd.DataFrame):
            return df
    except Exception:
        pass
    return fallback if isinstance(fallback, pd.DataFrame) else pd.DataFrame()


def _ctx_pmf_df(ctx: Any) -> pd.DataFrame:
    """Load PMF data lazily from the shared context; never raise."""
    try:
        df = getattr(ctx, "pmf_df", pd.DataFrame())
        if isinstance(df, pd.DataFrame):
            return df
    except Exception:
        pass
    return pd.DataFrame()


def _ctx_pmf_metrics(ctx: Any) -> List[str]:
    """Fast PMF metric discovery without forcing full ctx.pmf_df load."""
    try:
        return available_pmf_metrics(ctx)
    except Exception:
        return []


def _ctx_pmf_variants(ctx: Any) -> List[str]:
    """Fast PMF variant discovery from per-variant PMF files."""
    try:
        return available_pmf_variants(ctx)
    except Exception:
        return []


def _pmf_metric_values(pmf_df: pd.DataFrame) -> List[str]:
    if not isinstance(pmf_df, pd.DataFrame) or pmf_df.empty:
        return []
    for col in ("metric", "family", "cv", "coord", "cv_name"):
        if col in pmf_df.columns:
            vals = []
            for v in pmf_df[col].dropna().astype(str).tolist():
                s = str(v).strip()
                if s:
                    vals.append(s)
            return sorted(set(vals))
    return []


def _variant_values(feats_df: pd.DataFrame, pmf_df: pd.DataFrame) -> List[str]:
    vals: List[str] = []
    if isinstance(feats_df, pd.DataFrame) and (not feats_df.empty) and ("variant" in feats_df.columns):
        vals.extend(feats_df["variant"].dropna().astype(str).tolist())
    if isinstance(pmf_df, pd.DataFrame) and (not pmf_df.empty) and ("variant" in pmf_df.columns):
        vals.extend(pmf_df["variant"].dropna().astype(str).tolist())
    return sorted(set(v for v in vals if str(v).strip()))


def _variant_dropdown_options(variants: List[str]) -> List[dict]:
    return [{"label": v, "value": v} for v in (variants or [])]


def _umap_data_status_children(feats_df: pd.DataFrame, pmf_df: pd.DataFrame, metrics: List[str]) -> html.Small:
    n_feat_rows = int(len(feats_df)) if isinstance(feats_df, pd.DataFrame) else 0
    n_feat_vars = int(feats_df["variant"].astype(str).nunique()) if isinstance(feats_df, pd.DataFrame) and (not feats_df.empty) and ("variant" in feats_df.columns) else 0
    n_pmf_rows = int(len(pmf_df)) if isinstance(pmf_df, pd.DataFrame) else 0
    n_pmf_vars = int(pmf_df["variant"].astype(str).nunique()) if isinstance(pmf_df, pd.DataFrame) and (not pmf_df.empty) and ("variant" in pmf_df.columns) else 0
    return html.Small(
        f"features: {n_feat_rows} rows / {n_feat_vars} variants; "
        f"pmf: {n_pmf_rows} rows / {n_pmf_vars} variants; "
        f"pmf metrics: {len(metrics)}",
        className="text-muted",
    )
