from .shared import *
from .figures import _error_fig, _template
from .data_access import _ctx_pmf_variants


AA20 = list("ACDEFGHIKLMNPQRSTVWY")
AA_NAMES = {"A":"Ala", "C":"Cys", "D":"Asp", "E":"Glu", "F":"Phe", "G":"Gly", "H":"His", "I":"Ile", "K":"Lys", "L":"Leu", "M":"Met", "N":"Asn", "P":"Pro", "Q":"Gln", "R":"Arg", "S":"Ser", "T":"Thr", "V":"Val", "W":"Trp", "Y":"Tyr"}
AA_CLASS_MAP = {
    "A": "hydrophobic", "V": "hydrophobic", "I": "hydrophobic", "L": "hydrophobic", "M": "hydrophobic",
    "F": "aromatic", "W": "aromatic", "Y": "aromatic",
    "S": "polar", "T": "polar", "N": "polar", "Q": "polar", "C": "polar", "H": "polar",
    "K": "positive", "R": "positive",
    "D": "negative", "E": "negative",
    "G": "special", "P": "special",
}
AA_CLASS_ORDER = ["hydrophobic", "aromatic", "polar", "positive", "negative", "special"]
AA_CLASS_ABBR = {"hydrophobic": "Hyd", "aromatic": "Aro", "polar": "Pol", "positive": "+", "negative": "-", "special": "GP"}


def _variant_from_plot_point(pt: dict) -> Optional[str]:
    cd = pt.get("customdata")
    if cd is not None:
        if isinstance(cd, (list, tuple, np.ndarray)) and len(cd) > 0 and cd[0] is not None:
            # trajectory viewer uses [bin, variant], UMAP viewer usually stores only variant
            cand = str(cd[-1]).strip() if len(cd) > 1 else str(cd[0]).strip()
            if cand:
                return cand
        elif not isinstance(cd, (list, tuple, np.ndarray)):
            cand = str(cd).strip()
            if cand:
                return cand
    for key in ("hovertext", "text"):
        raw = pt.get(key)
        if raw and str(raw).strip():
            return str(raw).strip()
    return None


def _variants_from_plot_points(pts: Sequence[dict]) -> List[str]:
    seen: dict[str, None] = {}
    for pt in (pts or []):
        v = _variant_from_plot_point(pt)
        if v is not None:
            seen[str(v)] = None
    return list(seen)


def _sequence_from_variant_name(name: str) -> str:
    s = str(name or "").upper()
    tokens = re.findall(r"[ACDEFGHIKLMNPQRSTVWY]{2,}", s)
    if not tokens:
        return ""
    return max(tokens, key=len)


def _residue_composition_for_variants(variants: Sequence[str]) -> tuple[pd.DataFrame, str]:
    counts = {aa: 0 for aa in AA20}
    parsed = 0
    total = 0
    examples = []
    for v in variants or []:
        seq = _sequence_from_variant_name(str(v))
        if not seq:
            continue
        parsed += 1
        if len(examples) < 4:
            examples.append(f"{v}→{seq}")
        for aa in seq:
            if aa in counts:
                counts[aa] += 1
                total += 1
    rows = []
    for aa in AA20:
        pct = 100.0 * counts[aa] / total if total else 0.0
        rows.append({"AA": aa, "Residue": f"{aa} ({AA_NAMES.get(aa, aa)})", "percent": pct, "count": counts[aa]})
    msg = f"variants={len(variants or [])}; parsed={parsed}; residues={total}"
    if examples:
        msg += "; examples: " + "; ".join(examples)
    return pd.DataFrame(rows), msg


def _sequence_lookup_for_variants(variants: Sequence[str], feats_now: Optional[pd.DataFrame] = None) -> tuple[dict[str, str], str]:
    wanted = [str(v) for v in (variants or []) if str(v).strip()]
    seq_map: dict[str, str] = {}
    source = "variant name"
    if isinstance(feats_now, pd.DataFrame) and not feats_now.empty and "variant" in feats_now.columns:
        seq_col = next((c for c in ("sequence", "seq", "Sequence") if c in feats_now.columns), None)
        if seq_col is not None:
            tmp = feats_now[["variant", seq_col]].dropna().copy()
            tmp["variant"] = tmp["variant"].astype(str)
            tmp[seq_col] = tmp[seq_col].astype(str).str.upper().str.replace(r"[^ACDEFGHIKLMNPQRSTVWY]", "", regex=True)
            seq_map = {str(v): str(s) for v, s in zip(tmp["variant"], tmp[seq_col]) if str(s)}
            source = seq_col
    out: dict[str, str] = {}
    parsed = 0
    for v in wanted:
        seq = str(seq_map.get(v, "") or "")
        if not seq:
            seq = _sequence_from_variant_name(v)
        if seq:
            out[v] = seq
            parsed += 1
    return out, f"variants={len(wanted)}; parsed={parsed}; source={source if source else 'variant name'}"


def _dataset_background_variants(ctx: Any, feats_now: Optional[pd.DataFrame], fallback: Optional[Sequence[str]] = None) -> List[str]:
    vals: list[str] = []
    if isinstance(feats_now, pd.DataFrame) and not feats_now.empty and "variant" in feats_now.columns:
        vals.extend(feats_now["variant"].dropna().astype(str).tolist())
    try:
        vals.extend([str(v) for v in _ctx_pmf_variants(ctx)])
    except Exception:
        pass
    vals.extend([str(v) for v in (fallback or [])])
    out = []
    seen = set()
    for v in vals:
        if v and v not in seen:
            seen.add(v)
            out.append(v)
    return out


def _sequence_frequency_matrices(selected_variants: Sequence[str], background_variants: Sequence[str], feats_now: Optional[pd.DataFrame] = None) -> tuple[dict, str]:
    sel_map, sel_msg = _sequence_lookup_for_variants(selected_variants, feats_now)
    bg_map, bg_msg = _sequence_lookup_for_variants(background_variants, feats_now)
    sel_seqs = list(sel_map.values())
    bg_seqs = list(bg_map.values())
    if not sel_seqs or not bg_seqs:
        return {}, f"Selected: {sel_msg}. Background: {bg_msg}."
    max_len = max(max((len(s) for s in sel_seqs), default=0), max((len(s) for s in bg_seqs), default=0))
    if max_len <= 0:
        return {}, f"Selected: {sel_msg}. Background: {bg_msg}."
    aa_idx = {aa: i for i, aa in enumerate(AA20)}
    class_idx = {cl: i for i, cl in enumerate(AA_CLASS_ORDER)}
    sel_counts = np.zeros((len(AA20), max_len), dtype=float)
    bg_counts = np.zeros((len(AA20), max_len), dtype=float)
    sel_tot = np.zeros(max_len, dtype=float)
    bg_tot = np.zeros(max_len, dtype=float)
    sel_class_counts = np.zeros((len(AA_CLASS_ORDER), max_len), dtype=float)
    bg_class_counts = np.zeros((len(AA_CLASS_ORDER), max_len), dtype=float)
    sel_class_total = np.zeros(max_len, dtype=float)
    bg_class_total = np.zeros(max_len, dtype=float)
    for seq in sel_seqs:
        for pos, aa in enumerate(seq[:max_len]):
            if aa in aa_idx:
                sel_counts[aa_idx[aa], pos] += 1.0
                sel_tot[pos] += 1.0
            cl = AA_CLASS_MAP.get(aa)
            if cl in class_idx:
                sel_class_counts[class_idx[cl], pos] += 1.0
                sel_class_total[pos] += 1.0
    for seq in bg_seqs:
        for pos, aa in enumerate(seq[:max_len]):
            if aa in aa_idx:
                bg_counts[aa_idx[aa], pos] += 1.0
                bg_tot[pos] += 1.0
            cl = AA_CLASS_MAP.get(aa)
            if cl in class_idx:
                bg_class_counts[class_idx[cl], pos] += 1.0
                bg_class_total[pos] += 1.0
    with np.errstate(invalid="ignore", divide="ignore"):
        sel_freq = np.divide(sel_counts, sel_tot[None, :], where=sel_tot[None, :] > 0)
        bg_freq = np.divide(bg_counts, bg_tot[None, :], where=bg_tot[None, :] > 0)
        sel_class_freq = np.divide(sel_class_counts, sel_class_total[None, :], where=sel_class_total[None, :] > 0)
        bg_class_freq = np.divide(bg_class_counts, bg_class_total[None, :], where=bg_class_total[None, :] > 0)
    # overall residue-class composition across all parsed residues
    sel_class_overall = {cl: 0.0 for cl in AA_CLASS_ORDER}
    bg_class_overall = {cl: 0.0 for cl in AA_CLASS_ORDER}
    sel_total_res = 0
    bg_total_res = 0
    for seq in sel_seqs:
        for aa in seq:
            cl = AA_CLASS_MAP.get(aa)
            if cl:
                sel_class_overall[cl] += 1.0
                sel_total_res += 1
    for seq in bg_seqs:
        for aa in seq:
            cl = AA_CLASS_MAP.get(aa)
            if cl:
                bg_class_overall[cl] += 1.0
                bg_total_res += 1
    if sel_total_res > 0:
        for cl in sel_class_overall:
            sel_class_overall[cl] = 100.0 * sel_class_overall[cl] / sel_total_res
    if bg_total_res > 0:
        for cl in bg_class_overall:
            bg_class_overall[cl] = 100.0 * bg_class_overall[cl] / bg_total_res
    return {
        "positions": list(range(1, max_len + 1)),
        "sel_freq": sel_freq,
        "bg_freq": bg_freq,
        "sel_class_freq": sel_class_freq,
        "bg_class_freq": bg_class_freq,
        "sel_class_overall": sel_class_overall,
        "bg_class_overall": bg_class_overall,
        "n_selected": len(sel_seqs),
        "n_background": len(bg_seqs),
        "sel_msg": sel_msg,
        "bg_msg": bg_msg,
    }, f"Selected: {sel_msg}. Background: {bg_msg}."


def _build_region_sequence_grammar_figure(selected_variants: Sequence[str], background_variants: Sequence[str], feats_now: Optional[pd.DataFrame] = None) -> tuple[go.Figure, str]:
    data, status = _sequence_frequency_matrices(selected_variants, background_variants, feats_now)
    if not data:
        return _error_fig("No parseable sequences found for selected region."), status
    positions = data["positions"]
    sel = np.asarray(data["sel_freq"], dtype=float) * 100.0
    bg = np.asarray(data["bg_freq"], dtype=float) * 100.0
    delta = sel - bg
    max_abs_delta = max(2.0, float(np.nanmax(np.abs(delta))) if np.isfinite(delta).any() else 2.0)
    fig = make_subplots(
        rows=1, cols=3,
        column_widths=[0.36, 0.36, 0.28],
        horizontal_spacing=0.05,
        subplot_titles=("Selected region residue frequency (%)", "Selected − background enrichment (pp)", "Residue-class composition"),
        specs=[[{"type": "heatmap"}, {"type": "heatmap"}, {"type": "xy"}]],
    )
    y_labels = list(reversed(AA20))
    fig.add_trace(go.Heatmap(x=positions, y=y_labels, z=np.flipud(sel), colorscale="Viridis", colorbar=dict(title="%", len=0.72, x=0.29), hovertemplate="pos=%{x}<br>AA=%{y}<br>selected=%{z:.1f}%<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Heatmap(x=positions, y=y_labels, z=np.flipud(delta), colorscale="RdBu", zmid=0.0, zmin=-max_abs_delta, zmax=max_abs_delta, colorbar=dict(title="Δ pp", len=0.72, x=0.69), hovertemplate="pos=%{x}<br>AA=%{y}<br>Δ=%{z:+.1f} pp<extra></extra>"), row=1, col=2)
    class_df = pd.DataFrame({
        "class": AA_CLASS_ORDER,
        "selected": [data["sel_class_overall"].get(c, 0.0) for c in AA_CLASS_ORDER],
        "background": [data["bg_class_overall"].get(c, 0.0) for c in AA_CLASS_ORDER],
    })
    fig.add_trace(go.Bar(x=class_df["class"], y=class_df["selected"], name="selected", hovertemplate="%{x}<br>selected=%{y:.1f}%<extra></extra>"), row=1, col=3)
    fig.add_trace(go.Bar(x=class_df["class"], y=class_df["background"], name="background", hovertemplate="%{x}<br>background=%{y:.1f}%<extra></extra>"), row=1, col=3)
    fig.update_xaxes(title_text="sequence position", row=1, col=1)
    fig.update_xaxes(title_text="sequence position", row=1, col=2)
    fig.update_xaxes(title_text="residue class", tickangle=-25, row=1, col=3)
    fig.update_yaxes(title_text="residue", row=1, col=1)
    fig.update_yaxes(title_text="residue", row=1, col=2)
    fig.update_yaxes(title_text="residue %", row=1, col=3)
    fig.update_layout(template=_template(), height=560, barmode="group", margin=dict(l=60, r=24, t=65, b=62), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1.0))
    status += f" Selected region grammar shown as per-position residue frequencies and enrichment vs dataset background; parsed selected sequences={data['n_selected']}, background={data['n_background']}."
    return fig, status


def _build_trajectory_sequence_grammar_figure(bins_data: dict, background_variants: Sequence[str], feats_now: Optional[pd.DataFrame] = None) -> tuple[go.Figure, str]:
    bins = (bins_data or {}).get("bins", []) if isinstance(bins_data, dict) else []
    if not bins:
        return _error_fig("No trajectory bins available."), "Trajectory sequence grammar idle: compute trajectory first."
    bg_data, bg_status = _sequence_frequency_matrices(background_variants, background_variants, feats_now)
    if not bg_data:
        return _error_fig("No parseable background sequences available."), bg_status
    positions = bg_data["positions"]
    max_len = len(positions)
    n_bins = len(bins)
    res_delta = np.full((n_bins, max_len), np.nan, dtype=float)
    res_text = np.full((n_bins, max_len), "", dtype=object)
    class_delta = np.full((n_bins, max_len), np.nan, dtype=float)
    class_text = np.full((n_bins, max_len), "", dtype=object)
    pos_div = np.full((n_bins, max_len), np.nan, dtype=float)
    y_bins = []
    for i, b in enumerate(bins):
        y_bins.append(int(b.get("bin", i + 1)))
        sel_vars = [str(v) for v in b.get("variants", [])]
        data, _ = _sequence_frequency_matrices(sel_vars, background_variants, feats_now)
        if not data:
            continue
        sel = np.asarray(data["sel_freq"], dtype=float)
        bg = np.asarray(data["bg_freq"], dtype=float)
        # Pad in case this bin has shorter parsed sequences.
        if sel.shape[1] < max_len:
            sel = np.pad(sel, ((0,0),(0,max_len-sel.shape[1])), constant_values=np.nan)
            bg = np.pad(bg, ((0,0),(0,max_len-bg.shape[1])), constant_values=np.nan)
        delta = (sel - bg) * 100.0
        class_sel = np.asarray(data["sel_class_freq"], dtype=float)
        class_bg = np.asarray(data["bg_class_freq"], dtype=float)
        if class_sel.shape[1] < max_len:
            class_sel = np.pad(class_sel, ((0,0),(0,max_len-class_sel.shape[1])), constant_values=np.nan)
            class_bg = np.pad(class_bg, ((0,0),(0,max_len-class_bg.shape[1])), constant_values=np.nan)
        class_d = (class_sel - class_bg) * 100.0
        for pos in range(max_len):
            col = delta[:, pos]
            if np.isfinite(col).any():
                idx = int(np.nanargmax(np.abs(col)))
                res_delta[i, pos] = float(col[idx])
                res_text[i, pos] = AA20[idx]
                pos_div[i, pos] = float(0.5 * np.nansum(np.abs(col)))
            colc = class_d[:, pos]
            if np.isfinite(colc).any():
                idxc = int(np.nanargmax(np.abs(colc)))
                class_delta[i, pos] = float(colc[idxc])
                class_text[i, pos] = AA_CLASS_ABBR.get(AA_CLASS_ORDER[idxc], AA_CLASS_ORDER[idxc][:3])
    max_abs_res = max(2.0, float(np.nanmax(np.abs(res_delta))) if np.isfinite(res_delta).any() else 2.0)
    max_abs_class = max(2.0, float(np.nanmax(np.abs(class_delta))) if np.isfinite(class_delta).any() else 2.0)
    max_div = max(2.0, float(np.nanmax(pos_div)) if np.isfinite(pos_div).any() else 2.0)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.08,
        row_heights=[0.34, 0.33, 0.33],
        subplot_titles=(
            "Per-bin top enriched residue by position (text = residue; color = Δ pp)",
            "Per-bin top enriched residue class by position (text = class; color = Δ pp)",
            "Position divergence from background (total variation, pp)",
        ),
    )
    fig.add_trace(go.Heatmap(x=positions, y=y_bins, z=res_delta, text=res_text, texttemplate="%{text}", colorscale="RdBu", zmid=0.0, zmin=-max_abs_res, zmax=max_abs_res, colorbar=dict(title="Δ pp", len=0.22, x=1.02), hovertemplate="bin=%{y}<br>pos=%{x}<br>top residue=%{text}<br>Δ=%{z:+.1f} pp<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Heatmap(x=positions, y=y_bins, z=class_delta, text=class_text, texttemplate="%{text}", colorscale="RdBu", zmid=0.0, zmin=-max_abs_class, zmax=max_abs_class, colorbar=dict(title="Δ pp", len=0.22, x=1.02, y=0.50), hovertemplate="bin=%{y}<br>pos=%{x}<br>top class=%{text}<br>Δ=%{z:+.1f} pp<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Heatmap(x=positions, y=y_bins, z=pos_div, colorscale="Magma", zmin=0.0, zmax=max_div, colorbar=dict(title="TV (pp)", len=0.22, x=1.02, y=0.14), hovertemplate="bin=%{y}<br>pos=%{x}<br>divergence=%{z:.1f} pp<extra></extra>"), row=3, col=1)
    fig.update_xaxes(title_text="sequence position", row=3, col=1)
    fig.update_yaxes(title_text="trajectory bin", autorange="reversed", row=1, col=1)
    fig.update_yaxes(title_text="trajectory bin", autorange="reversed", row=2, col=1)
    fig.update_yaxes(title_text="trajectory bin", autorange="reversed", row=3, col=1)
    fig.update_layout(template=_template(), height=760, margin=dict(l=70, r=68, t=84, b=62))
    return fig, f"Trajectory sequence grammar: top enriched residue and residue-class per position are shown for each bin relative to dataset background; bottom panel shows overall positional divergence (total variation distance). Background: {bg_data['bg_msg']}"


# ----------------------------------------------------------------------
# Automatic sequence-language clustering
# ----------------------------------------------------------------------

def _safe_zscore_matrix(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim != 2 or X.size == 0:
        return np.zeros((0, 0), dtype=float)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    mu = np.nanmean(X, axis=0)
    sd = np.nanstd(X, axis=0, ddof=1) if X.shape[0] > 1 else np.ones(X.shape[1], dtype=float)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1.0)
    Z = (X - mu) / sd
    finite_var = np.nanvar(Z, axis=0)
    keep = np.isfinite(finite_var) & (finite_var > 1e-12)
    if not np.any(keep):
        return np.zeros((X.shape[0], 0), dtype=float)
    return Z[:, keep]


def _sequence_language_features(
    variants: Sequence[str],
    feats_now: Optional[pd.DataFrame] = None,
    mode: str = "all",
    *,
    kmer_k: int = 2,
    max_len_cap: int = 64,
    max_kmers: int = 256,
) -> tuple[pd.DataFrame, np.ndarray, List[str], str]:
    """Build lightweight sequence-language features for motif/composition clustering.

    Modes:
      composition: length + AA fractions + residue-class fractions + termini classes
      position:    aligned position AA/class one-hot features
      motif:       frequent k-mer frequencies + residue-class k-mer frequencies
      all:         composition + position + motif
    """
    seq_map, lookup_msg = _sequence_lookup_for_variants(variants, feats_now)
    rows_meta = []
    seqs = []
    for v in variants or []:
        vv = str(v)
        seq = seq_map.get(vv, "")
        if seq:
            rows_meta.append({"variant": vv, "sequence": seq, "length": len(seq)})
            seqs.append(seq)
    meta = pd.DataFrame(rows_meta)
    if meta.empty:
        return meta, np.zeros((0, 0), dtype=float), [], lookup_msg

    mode = str(mode or "all").strip().lower()
    use_comp = mode in {"all", "auto", "composition", "comp"}
    use_pos = mode in {"all", "auto", "position", "pos", "aligned"}
    use_motif = mode in {"all", "auto", "motif", "motifs", "kmer", "kmers"}

    feats: list[np.ndarray] = []
    names: list[str] = []
    lengths = np.asarray([len(s) for s in seqs], dtype=float)
    max_len = int(min(max(lengths.max(initial=0), 1), int(max_len_cap or 64)))

    if use_comp:
        comp_rows = []
        for seq in seqs:
            L = max(1, len(seq))
            row = [float(len(seq))]
            row.extend([seq.count(aa) / float(L) for aa in AA20])
            row.extend([sum(1 for aa in seq if AA_CLASS_MAP.get(aa) == cl) / float(L) for cl in AA_CLASS_ORDER])
            ncl = AA_CLASS_MAP.get(seq[0], "") if seq else ""
            ccl = AA_CLASS_MAP.get(seq[-1], "") if seq else ""
            row.extend([1.0 if ncl == cl else 0.0 for cl in AA_CLASS_ORDER])
            row.extend([1.0 if ccl == cl else 0.0 for cl in AA_CLASS_ORDER])
            # crude grammar-ish pattern features
            charges = np.asarray([(1 if aa in "KR" else (-1 if aa in "DE" else 0)) for aa in seq], dtype=float)
            hyd = np.asarray([(1 if AA_CLASS_MAP.get(aa) in {"hydrophobic", "aromatic"} else 0) for aa in seq], dtype=float)
            row.append(float(np.sum(charges)) / float(L))
            row.append(float(np.sum(np.abs(charges))) / float(L))
            row.append(float(np.mean(hyd)) if hyd.size else 0.0)
            row.append(float(_max_run_len(seq, lambda aa: AA_CLASS_MAP.get(aa) in {"hydrophobic", "aromatic"}) / float(L)))
            row.append(float(_max_run_len(seq, lambda aa: aa in "GP") / float(L)))
            comp_rows.append(row)
        comp_names = (["length"] + [f"aa_frac_{aa}" for aa in AA20] + [f"class_frac_{cl}" for cl in AA_CLASS_ORDER]
                      + [f"nterm_{cl}" for cl in AA_CLASS_ORDER] + [f"cterm_{cl}" for cl in AA_CLASS_ORDER]
                      + ["net_charge_density", "abs_charge_density", "hydrophobic_aromatic_frac", "hydrophobic_aromatic_max_run_frac", "GP_max_run_frac"])
        feats.append(np.asarray(comp_rows, dtype=float))
        names.extend(comp_names)

    if use_pos:
        pos_rows = []
        for seq in seqs:
            row = []
            for pos in range(max_len):
                aa = seq[pos] if pos < len(seq) else ""
                row.extend([1.0 if aa == a else 0.0 for a in AA20])
                cl = AA_CLASS_MAP.get(aa, "")
                row.extend([1.0 if cl == c else 0.0 for c in AA_CLASS_ORDER])
            pos_rows.append(row)
        pos_names = []
        for pos in range(max_len):
            pos_names.extend([f"pos{pos+1}_{aa}" for aa in AA20])
            pos_names.extend([f"pos{pos+1}_class_{cl}" for cl in AA_CLASS_ORDER])
        feats.append(np.asarray(pos_rows, dtype=float))
        names.extend(pos_names)

    if use_motif:
        k = max(2, min(int(kmer_k or 2), 4))
        motif_counts: dict[str, int] = {}
        class_counts: dict[str, int] = {}
        for seq in seqs:
            for i in range(0, max(0, len(seq) - k + 1)):
                km = seq[i:i+k]
                if len(km) == k and all(aa in AA20 for aa in km):
                    motif_counts[km] = motif_counts.get(km, 0) + 1
                    ckm = "-".join(AA_CLASS_ABBR.get(AA_CLASS_MAP.get(aa, ""), "?") for aa in km)
                    class_counts[ckm] = class_counts.get(ckm, 0) + 1
        motifs = [m for m, _ in sorted(motif_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:int(max_kmers or 256)]]
        class_motifs = [m for m, _ in sorted(class_counts.items(), key=lambda kv: (-kv[1], kv[0]))[:min(128, int(max_kmers or 256))]]
        motif_rows = []
        for seq in seqs:
            denom = max(1, len(seq) - k + 1)
            row = []
            for m in motifs:
                row.append(sum(1 for i in range(0, max(0, len(seq) - k + 1)) if seq[i:i+k] == m) / float(denom))
            for m in class_motifs:
                cnt = 0
                for i in range(0, max(0, len(seq) - k + 1)):
                    km = seq[i:i+k]
                    ckm = "-".join(AA_CLASS_ABBR.get(AA_CLASS_MAP.get(aa, ""), "?") for aa in km)
                    if ckm == m:
                        cnt += 1
                row.append(cnt / float(denom))
            motif_rows.append(row)
        if motif_rows and (motifs or class_motifs):
            feats.append(np.asarray(motif_rows, dtype=float))
            names.extend([f"kmer_{m}" for m in motifs] + [f"class_kmer_{m}" for m in class_motifs])

    if not feats:
        return meta, np.zeros((len(meta), 0), dtype=float), [], lookup_msg
    X = np.hstack(feats) if len(feats) > 1 else feats[0]
    Z = _safe_zscore_matrix(X)
    return meta, Z, names, lookup_msg


def _max_run_len(seq: str, predicate) -> int:
    best = 0
    cur = 0
    for aa in str(seq or ""):
        if predicate(aa):
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def _fallback_kmeans(X: np.ndarray, k: int, random_state: int = 42, n_iter: int = 80) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    rng = np.random.default_rng(int(random_state))
    if n == 0 or k <= 1:
        return np.zeros(n, dtype=int)
    k = max(1, min(int(k), n))
    # k-means++-ish initialization
    centers = [X[int(rng.integers(0, n))]]
    for _ in range(1, k):
        d2 = np.min(np.sum((X[:, None, :] - np.asarray(centers)[None, :, :]) ** 2, axis=2), axis=1)
        p = d2 / float(np.sum(d2)) if np.sum(d2) > 0 else np.full(n, 1.0 / n)
        centers.append(X[int(rng.choice(n, p=p))])
    C = np.asarray(centers, dtype=float)
    labels = np.zeros(n, dtype=int)
    for _ in range(int(n_iter)):
        D = np.sum((X[:, None, :] - C[None, :, :]) ** 2, axis=2)
        new_labels = np.argmin(D, axis=1)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels
        for j in range(k):
            mask = labels == j
            if np.any(mask):
                C[j] = np.mean(X[mask], axis=0)
    return labels.astype(int)


def _auto_sequence_clusters(X: np.ndarray, max_clusters: int = 8, random_state: int = 42) -> tuple[np.ndarray, dict]:
    X = np.asarray(X, dtype=float)
    n = X.shape[0]
    if n < 3 or X.shape[1] < 1:
        return np.zeros(n, dtype=int), {"method": "none", "k": 1, "score": np.nan, "note": "not enough parseable sequences/features"}
    max_k = max(2, min(int(max_clusters or 8), n - 1, 12))
    best_labels = None
    best_score = -np.inf
    best_k = 1
    best_method = "fallback-kmeans"
    for k in range(2, max_k + 1):
        try:
            from sklearn.cluster import KMeans  # type: ignore
            model = KMeans(n_clusters=k, random_state=int(random_state), n_init=20)
            labels = model.fit_predict(X)
            method = "KMeans"
        except Exception:
            labels = _fallback_kmeans(X, k, random_state=random_state)
            method = "fallback-kmeans"
        if len(set(labels.tolist())) < 2:
            continue
        try:
            if silhouette_score is not None and n > k:
                score = float(silhouette_score(X, labels))
            else:
                # Simple fallback: separation / compactness proxy
                overall = np.mean(X, axis=0)
                between = 0.0
                within = 0.0
                for lab in sorted(set(labels.tolist())):
                    mask = labels == lab
                    if not np.any(mask):
                        continue
                    cen = np.mean(X[mask], axis=0)
                    between += float(mask.sum()) * float(np.sum((cen - overall) ** 2))
                    within += float(np.sum((X[mask] - cen) ** 2))
                score = between / max(within, 1e-12)
        except Exception:
            score = -np.inf
        # avoid pathological singletons unless they are clearly best
        counts = np.bincount(labels.astype(int))
        singleton_penalty = 0.04 * int(np.sum(counts < 2))
        adj_score = score - singleton_penalty
        if adj_score > best_score:
            best_score = adj_score
            best_labels = labels.astype(int)
            best_k = k
            best_method = method
    if best_labels is None:
        return np.zeros(n, dtype=int), {"method": "none", "k": 1, "score": np.nan, "note": "could not split sequence-language features"}
    # Re-number by cluster size, largest first, for stable UI labels.
    counts = pd.Series(best_labels).value_counts().sort_values(ascending=False)
    remap = {int(old): i for i, old in enumerate(counts.index.tolist())}
    labels = np.asarray([remap[int(x)] for x in best_labels], dtype=int)
    return labels, {"method": best_method, "k": int(len(set(labels.tolist()))), "score": float(best_score), "note": "auto-selected k by silhouette/proxy"}


def _top_sequence_motifs_for_cluster(cluster_seqs: Sequence[str], background_seqs: Sequence[str], *, k: int = 2, top_n: int = 5) -> str:
    def counts(seqs):
        c: dict[str, int] = {}
        total = 0
        for seq in seqs:
            for i in range(0, max(0, len(seq) - k + 1)):
                m = seq[i:i+k]
                if len(m) == k and all(aa in AA20 for aa in m):
                    c[m] = c.get(m, 0) + 1
                    total += 1
        return c, total
    cc, ct = counts(cluster_seqs)
    bc, bt = counts(background_seqs)
    rows = []
    pseudo = 0.5
    for m, n in cc.items():
        pf = (n + pseudo) / max(ct + pseudo * max(1, len(cc)), 1e-12)
        bf = (bc.get(m, 0) + pseudo) / max(bt + pseudo * max(1, len(bc)), 1e-12)
        rows.append((m, pf / max(bf, 1e-12), 100.0 * (pf - bf)))
    rows.sort(key=lambda x: (x[1], abs(x[2])), reverse=True)
    return ", ".join(f"{m} ({fold:.1f}x)" for m, fold, _ in rows[:top_n]) if rows else ""


def _top_sequence_classes_for_cluster(cluster_seqs: Sequence[str], background_seqs: Sequence[str], top_n: int = 4) -> str:
    def freqs(seqs):
        c = {cl: 0 for cl in AA_CLASS_ORDER}
        total = 0
        for seq in seqs:
            for aa in seq:
                cl = AA_CLASS_MAP.get(aa)
                if cl:
                    c[cl] += 1
                    total += 1
        return {cl: (100.0 * c[cl] / total if total else 0.0) for cl in AA_CLASS_ORDER}
    cf = freqs(cluster_seqs)
    bf = freqs(background_seqs)
    rows = [(cl, cf[cl] - bf[cl]) for cl in AA_CLASS_ORDER]
    rows.sort(key=lambda x: abs(x[1]), reverse=True)
    return ", ".join(f"{AA_CLASS_ABBR.get(cl, cl)} {delta:+.1f}pp" for cl, delta in rows[:top_n])


def _summarize_sequence_language_clusters(meta: pd.DataFrame, labels: np.ndarray, background_variants: Sequence[str], feats_now: Optional[pd.DataFrame]) -> pd.DataFrame:
    bg_map, _ = _sequence_lookup_for_variants(background_variants, feats_now)
    bg_seqs = list(bg_map.values())
    rows = []
    if meta.empty or labels.size != len(meta):
        return pd.DataFrame(columns=["cluster", "n", "frac", "mean_len", "top_motifs", "class_shift", "examples"])
    d = meta.copy()
    d["cluster_id"] = labels.astype(int)
    total = max(1, len(d))
    for cid, g in d.groupby("cluster_id", sort=True):
        seqs = g["sequence"].astype(str).tolist()
        vars_here = g["variant"].astype(str).tolist()
        rows.append({
            "cluster": f"S{int(cid)}",
            "n": int(len(g)),
            "frac": float(len(g) / total),
            "mean_len": float(np.mean([len(s) for s in seqs])) if seqs else np.nan,
            "top_motifs": _top_sequence_motifs_for_cluster(seqs, bg_seqs, k=2, top_n=5),
            "class_shift": _top_sequence_classes_for_cluster(seqs, bg_seqs, top_n=4),
            "examples": ", ".join(vars_here[:5]) + (f" ... +{len(vars_here)-5}" if len(vars_here) > 5 else ""),
        })
    return pd.DataFrame(rows)


def _build_sequence_language_cluster_outputs(
    emb_df: pd.DataFrame,
    feats_now: Optional[pd.DataFrame],
    ctx: Any,
    mode: str = "all",
    max_clusters: int = 8,
) -> tuple[go.Figure, html.Div, str]:
    """Detect sequence-language clusters and render a UMAP overlay + summary table."""
    if not isinstance(emb_df, pd.DataFrame) or emb_df.empty or "variant" not in emb_df.columns:
        return _error_fig("Compute or quickload an embedding first."), html.Div(), "Sequence-language clustering idle: no embedding available."
    variants = emb_df["variant"].dropna().astype(str).tolist()
    meta, X, feature_names, lookup_msg = _sequence_language_features(variants, feats_now, mode=mode)
    if meta.empty or X.shape[0] < 3 or X.shape[1] < 1:
        return _error_fig("Not enough parseable sequence-language signal."), html.Div(), f"Auto sequence clusters failed: {lookup_msg}; features={X.shape}."
    labels, info = _auto_sequence_clusters(X, max_clusters=max_clusters)
    meta = meta.copy()
    meta["seq_cluster"] = [f"S{int(x)}" for x in labels]

    # The embedding table may already carry scalar columns copied from the
    # feature table, including sequence/length from earlier merges or stores.
    # Drop these before joining the auto-cluster metadata so pandas does not
    # silently create sequence_x/sequence_y and then make Plotly fail when
    # hover_data asks for plain "sequence". Tiny refactor goblin, very proud
    # of itself.
    # Drop prior cluster metadata and pandas merge leftovers defensively.
    # Feature tables may already contain sequence/length, and repeated callback
    # cycles or older patched versions may leave sequence_x/sequence_y,
    # length_x/length_y, or seq_cluster_x/seq_cluster_y in the embedding store.
    # Keep real seq_* descriptors such as seq_dipep_*; only remove the small
    # metadata family used for this overlay.
    def _is_seq_cluster_overlay_meta(col: object) -> bool:
        s = str(col)
        return bool(re.match(r"^(sequence|length|seq_cluster)(?:_[xy])?$", s))

    drop_before_join = [c for c in emb_df.columns if c != "variant" and _is_seq_cluster_overlay_meta(c)]
    emb_base = emb_df.drop(columns=drop_before_join, errors="ignore") if drop_before_join else emb_df.copy()
    plot_df = emb_base.merge(
        meta[["variant", "seq_cluster", "sequence", "length"]],
        on="variant",
        how="inner",
        validate="many_to_one",
    )
    if plot_df.empty:
        return _error_fig("Sequence clusters could not be joined to the embedding."), html.Div(), "Auto sequence clusters failed after joining sequence records to embedding."

    hover_cols = [c for c in ("sequence", "length") if c in plot_df.columns]
    cluster_order = sorted(plot_df["seq_cluster"].dropna().astype(str).unique().tolist())
    if "UMAP3" in plot_df.columns and pd.to_numeric(plot_df["UMAP3"], errors="coerce").notna().any():
        fig = px.scatter_3d(plot_df, x="UMAP1", y="UMAP2", z="UMAP3", color="seq_cluster", hover_name="variant", hover_data=hover_cols or None, category_orders={"seq_cluster": cluster_order})
        fig.update_layout(scene=dict(xaxis_title="UMAP1", yaxis_title="UMAP2", zaxis_title="UMAP3", aspectmode="cube"))
    else:
        fig = px.scatter(plot_df, x="UMAP1", y="UMAP2", color="seq_cluster", hover_name="variant", hover_data=hover_cols or None, category_orders={"seq_cluster": cluster_order}, render_mode="webgl")
        fig.update_xaxes(title_text="UMAP1")
        fig.update_yaxes(title_text="UMAP2")
    fig.update_layout(template=_template(), height=560, title="Auto-detected sequence-language clusters on UMAP", legend_title_text="Seq cluster", margin=dict(l=55, r=20, t=65, b=55))
    background = _dataset_background_variants(ctx, feats_now, fallback=variants)
    summary = _summarize_sequence_language_clusters(meta, labels, background, feats_now)
    table = html.Div()
    if not summary.empty:
        show = summary.copy()
        if "frac" in show.columns:
            show["frac"] = (100.0 * show["frac"]).round(1).astype(str) + "%"
        if "mean_len" in show.columns:
            show["mean_len"] = pd.to_numeric(show["mean_len"], errors="coerce").round(2)
        table = html.Div([
            html.H6("Auto sequence-language cluster summary"),
            dbc.Table.from_dataframe(show, striped=True, bordered=True, hover=True, size="sm"),
            html.Small("Clusters are inferred from sequence-language features; motif/class labels are enrichment summaries, not causal proof.", className="text-muted"),
        ])
    status = (
        f"Auto sequence-language clusters: mode={mode}; parsed={len(meta)}/{len(variants)} variants; "
        f"features={X.shape[1]}; method={info.get('method')}; k={info.get('k')}; "
        f"score={info.get('score'):.3g} if finite; {info.get('note')}; {lookup_msg}"
    )
    return fig, table, status
