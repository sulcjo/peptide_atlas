from .shared import *
from .figures import _error_fig, _template
from .geometry import _trajectory_path_coordinate, _axis_bins_for_values
from .pmf_landscape import (_pmf_temperature_from_curve_map, _pmf_unit_factor_label, _variant_curve_map_for_metric, _representative_pmf_for_variants)
from .sequence_grammar import AA20, AA_NAMES, _residue_composition_for_variants


def _build_trajectory_figure(emb_df: pd.DataFrame, curves: pd.DataFrame, metric: str, mode: str, n_bins: int, unit: str, method_label: str = "UMAP", anchors: Optional[Sequence[str]] = None, feature_col: Optional[str] = None) -> tuple[go.Figure, str, dict]:
    if not isinstance(emb_df, pd.DataFrame) or emb_df.empty:
        return _error_fig("Compute embedding first."), "No embedding available for trajectory.", {"bins": []}
    if "variant" not in emb_df.columns:
        return _error_fig("Embedding has no variant column."), "Trajectory needs variant IDs.", {"bins": []}
    emb_df = emb_df.copy()
    emb_df["variant"] = emb_df["variant"].astype(str)
    coord, coord_label = _trajectory_path_coordinate(emb_df, mode, anchors, feature_col=feature_col)
    emb_df["_path_coord"] = pd.to_numeric(coord, errors="coerce")
    bins_idx = _axis_bins_for_values(emb_df["_path_coord"], int(n_bins or 8))
    if not bins_idx:
        return _error_fig("Could not bin variants along the selected path."), "No finite path coordinate.", {"bins": []}
    curve_map, y_col = _variant_curve_map_for_metric(curves, metric)
    if not curve_map or y_col is None:
        return _error_fig("No PMF curves available for trajectory."), "No PMF curves found for selected metric.", {"bins": []}
    T_display = _pmf_temperature_from_curve_map(curve_map)
    factor, unit_label = _pmf_unit_factor_label(unit, T_K=T_display)
    reps = []
    bin_records = []
    for i, idx in enumerate(bins_idx, start=1):
        b = emb_df.loc[list(idx)].copy()
        vars_bin = b["variant"].astype(str).tolist()
        rep = _representative_pmf_for_variants(curve_map, vars_bin, y_col)
        if rep is None:
            continue
        rep["bin"] = i
        rep["path_min"] = float(np.nanmin(pd.to_numeric(b["_path_coord"], errors="coerce")))
        rep["path_max"] = float(np.nanmax(pd.to_numeric(b["_path_coord"], errors="coerce")))
        reps.append(rep)
        bin_records.append({"bin": i, "variants": vars_bin, "path_min": rep["path_min"], "path_max": rep["path_max"], "n_variants": len(vars_bin)})
    if not reps:
        return _error_fig("No usable PMF representative could be built."), "Trajectory bins had no usable PMFs.", {"bins": []}
    # display ranges
    allx = []
    ally = []
    for r in reps:
        allx.extend(np.asarray(r["x"], dtype=float).tolist())
        allx.extend(np.asarray(r.get("med_x", []), dtype=float).tolist())
        for y in (r["meanF"], r["q25"], r["q75"], r.get("med_y", [])):
            yy = np.asarray(y, dtype=float) * factor
            ally.extend(yy[np.isfinite(yy)].tolist())
    xmin, xmax = (float(np.nanmin(allx)), float(np.nanmax(allx))) if allx else (0.0, 1.0)
    xpad = 0.03 * max(1e-9, xmax - xmin)
    ymin, ymax = (float(np.nanmin(ally)), float(np.nanmax(ally))) if ally else (0.0, 1.0)
    ypad = 0.06 * max(1e-9, ymax - ymin)
    y_range = [max(0.0, ymin - ypad), ymax + ypad]
    # deltas on the common representative grid per adjacent pair
    delta_profiles = [np.zeros_like(reps[0]["meanF"], dtype=float)]
    delta_x = [reps[0]["x"]]
    rms = [0.0]
    mx = [0.0]
    for i in range(1, len(reps)):
        x0, f0 = np.asarray(reps[i-1]["x"], dtype=float), np.asarray(reps[i-1]["meanF"], dtype=float)
        x1, f1 = np.asarray(reps[i]["x"], dtype=float), np.asarray(reps[i]["meanF"], dtype=float)
        lo, hi = max(np.nanmin(x0), np.nanmin(x1)), min(np.nanmax(x0), np.nanmax(x1))
        if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
            gx = np.linspace(lo, hi, min(len(x0), len(x1), 256))
            dF = (np.interp(gx, x1, f1) - np.interp(gx, x0, f0)) * factor
        else:
            gx = x1
            dF = np.zeros_like(x1)
        delta_x.append(gx)
        delta_profiles.append(dF)
        finite = dF[np.isfinite(dF)]
        rms.append(float(np.sqrt(np.mean(finite**2))) if finite.size else 0.0)
        mx.append(float(np.max(np.abs(finite))) if finite.size else 0.0)
    delta_y_all = np.concatenate([np.asarray(d, dtype=float)[np.isfinite(d)] for d in delta_profiles if len(d)]) if delta_profiles else np.array([0.0])
    dmax = float(np.nanmax(np.abs(delta_y_all))) if delta_y_all.size else 1.0
    dmax = max(dmax, 1e-9)

    # Static pseudo-trajectory surfaces below the animated trajectory.  These
    # show the whole PMF landscape as a function of trajectory bin, plus the
    # adjacent-bin change surface.  Values are clipped only for display so one
    # extreme bin cannot flatten the whole mountain range.
    bins_nums = [r["bin"] for r in reps]
    n_surf_x = int(np.clip(max((len(np.asarray(r["x"], dtype=float)) for r in reps), default=0), 80, 240))
    surf_x = np.linspace(xmin, xmax, n_surf_x, dtype=float) if np.isfinite(xmin) and np.isfinite(xmax) and xmax > xmin else np.linspace(0.0, 1.0, 120, dtype=float)
    f_surface = np.full((len(reps), len(surf_x)), np.nan, dtype=float)
    for ii, rr in enumerate(reps):
        xr = np.asarray(rr["x"], dtype=float)
        fr = np.asarray(rr["meanF"], dtype=float) * factor
        ok = np.isfinite(xr) & np.isfinite(fr)
        if ok.sum() >= 2:
            order = np.argsort(xr[ok])
            xs = xr[ok][order]
            ys = fr[ok][order]
            f_surface[ii, :] = np.interp(surf_x, xs, ys, left=np.nan, right=np.nan)
    dF_surface = np.full_like(f_surface, np.nan)
    if len(reps):
        dF_surface[0, :] = 0.0
    for ii in range(1, len(reps)):
        prev = f_surface[ii - 1, :]
        cur = f_surface[ii, :]
        ok = np.isfinite(prev) & np.isfinite(cur)
        if ok.any():
            dF_surface[ii, ok] = cur[ok] - prev[ok]

    def _finite_percentile(vals: np.ndarray, q: float, fallback: float) -> float:
        vals = np.asarray(vals, dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            return float(fallback)
        return float(np.nanpercentile(vals, q))

    f_cap = {"kJ/mol": 50.0, "kcal/mol": 12.0, "kT": 20.0}.get(str(unit_label), 50.0)
    df_cap = {"kJ/mol": 30.0, "kcal/mol": 7.5, "kT": 12.0}.get(str(unit_label), 30.0)
    f_vals = f_surface[np.isfinite(f_surface)]
    f_lo = max(0.0, _finite_percentile(f_vals, 1.0, 0.0))
    f_hi = min(f_cap, max(_finite_percentile(f_vals, 98.0, f_cap), f_lo + 1e-6))
    d_vals = dF_surface[np.isfinite(dF_surface)]
    d_clip = min(df_cap, max(_finite_percentile(np.abs(d_vals), 98.0, dmax), 1e-6))
    f_surface_plot = np.clip(f_surface, f_lo, f_hi)
    dF_surface_plot = np.clip(dF_surface, -d_clip, d_clip)
    if len(bins_nums) == 1:
        surf_y = np.asarray([float(bins_nums[0]), float(bins_nums[0]) + 1e-6], dtype=float)
        f_surface_plot_show = np.vstack([f_surface_plot[0, :], f_surface_plot[0, :]])
        dF_surface_plot_show = np.vstack([dF_surface_plot[0, :], dF_surface_plot[0, :]])
    else:
        surf_y = np.asarray(bins_nums, dtype=float)
        f_surface_plot_show = f_surface_plot
        dF_surface_plot_show = dF_surface_plot

    is3d = "UMAP3" in emb_df.columns and pd.to_numeric(emb_df["UMAP3"], errors="coerce").notna().any()
    specs = [
        [{"type": "scene" if is3d else "xy"}, {"type": "xy"}],
        [{"type": "xy"}, {"type": "xy"}],
        [{"type": "scene"}, {"type": "scene"}],
    ]
    fig = make_subplots(
        rows=3,
        cols=2,
        specs=specs,
        column_widths=[0.42, 0.58],
        row_heights=[0.48, 0.22, 0.30],
        horizontal_spacing=0.09,
        vertical_spacing=0.10,
        subplot_titles=(
            "Embedding/path bin",
            "Representative PMF",
            "Change magnitude per bin",
            "ΔPMF vs previous bin",
            "F(path): bin × PMF coordinate",
            "ΔF(path): change vs previous bin",
        ),
    )
    first = reps[0]
    b0vars = set(first["variants"])
    hi = emb_df[emb_df["variant"].isin(b0vars)]
    if is3d:
        fig.add_trace(go.Scatter3d(x=emb_df["UMAP1"], y=emb_df["UMAP2"], z=emb_df["UMAP3"], mode="markers", name="all variants", marker=dict(size=3, opacity=0.28), customdata=emb_df["variant"], hovertemplate="%{customdata}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter3d(x=hi["UMAP1"], y=hi["UMAP2"], z=hi["UMAP3"], mode="markers", name="current bin", marker=dict(size=5, opacity=0.95), customdata=np.c_[np.full(len(hi), first["bin"]), hi["variant"].astype(str)], hovertemplate="bin=%{customdata[0]}<br>%{customdata[1]}<extra></extra>"), row=1, col=1)
    else:
        fig.add_trace(go.Scatter(x=emb_df["UMAP1"], y=emb_df["UMAP2"], mode="markers", name="all variants", marker=dict(size=6, opacity=0.28), customdata=emb_df["variant"], hovertemplate="%{customdata}<extra></extra>"), row=1, col=1)
        fig.add_trace(go.Scatter(x=hi["UMAP1"], y=hi["UMAP2"], mode="markers", name="current bin", marker=dict(size=8, opacity=0.95), customdata=np.c_[np.full(len(hi), first["bin"]), hi["variant"].astype(str)], hovertemplate="bin=%{customdata[0]}<br>%{customdata[1]}<extra></extra>"), row=1, col=1)
    fig.add_trace(go.Scatter(x=first["x"], y=first["q75"]*factor, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip", name="IQR upper"), row=1, col=2)
    fig.add_trace(go.Scatter(x=first["x"], y=first["q25"]*factor, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(31,119,180,0.18)", name="IQR band", hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter(x=first["x"], y=first["meanF"]*factor, mode="lines", name="mean P→PMF", line=dict(width=2, dash="dash"), customdata=np.full(len(first["x"]), first["bin"]), hovertemplate="bin=%{customdata}<br>x=%{x:.4g}<br>F=%{y:.3g} " + unit_label + "<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=first.get("med_x", []), y=np.asarray(first.get("med_y", []))*factor, mode="lines", name="medoid PMF", line=dict(width=3), customdata=np.full(len(first.get("med_x", [])), first["bin"]), hovertemplate=f"medoid={first.get('medoid','')}<br>x=%{{x:.4g}}<br>F=%{{y:.3g}} {unit_label}<extra></extra>"), row=1, col=2)
    fig.add_trace(go.Scatter(x=bins_nums, y=rms, mode="lines+markers", name="RMS ΔPMF", customdata=np.asarray(bins_nums), hovertemplate="bin=%{x}<br>RMS=%{y:.3g}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=bins_nums, y=mx, mode="lines+markers", name="max |ΔPMF|", customdata=np.asarray(bins_nums), hovertemplate="bin=%{x}<br>max=%{y:.3g}<extra></extra>"), row=2, col=1)
    fig.add_trace(go.Scatter(x=[first["bin"]], y=[rms[0]], mode="markers", name="active bin", marker=dict(size=12), showlegend=False, customdata=[first["bin"]]), row=2, col=1)
    fig.add_trace(go.Scatter(x=delta_x[0], y=delta_profiles[0], mode="lines", name="ΔPMF", customdata=np.full(len(delta_x[0]), first["bin"]), hovertemplate="bin=%{customdata}<br>x=%{x:.4g}<br>ΔF=%{y:.3g} " + unit_label + "<extra></extra>"), row=2, col=2)
    fig.add_trace(
        go.Surface(
            x=surf_x,
            y=surf_y,
            z=f_surface_plot_show,
            name="F(path) surface",
            colorbar=dict(title=f"F ({unit_label})", len=0.26, x=0.45, y=0.15),
            contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
            hovertemplate="bin=%{y:.0f}<br>x=%{x:.4g}<br>F(display)=%{z:.3g} " + unit_label + "<extra></extra>",
            showscale=True,
        ),
        row=3, col=1,
    )
    fig.add_trace(
        go.Surface(
            x=surf_x,
            y=surf_y,
            z=dF_surface_plot_show,
            name="ΔF(path) surface",
            colorscale="RdBu",
            cmin=-d_clip,
            cmax=d_clip,
            colorbar=dict(title=f"ΔF ({unit_label})", len=0.26, x=1.02, y=0.15),
            contours={"z": {"show": True, "usecolormap": True, "project_z": True}},
            hovertemplate="bin=%{y:.0f}<br>x=%{x:.4g}<br>ΔF(display)=%{z:.3g} " + unit_label + "<extra></extra>",
            showscale=True,
        ),
        row=3, col=2,
    )
    frames = []
    for i, r in enumerate(reps):
        bvars = set(r["variants"])
        h = emb_df[emb_df["variant"].isin(bvars)]
        if is3d:
            hi_trace = go.Scatter3d(x=h["UMAP1"], y=h["UMAP2"], z=h["UMAP3"], mode="markers", marker=dict(size=5, opacity=0.95), customdata=np.c_[np.full(len(h), r["bin"]), h["variant"].astype(str)], hovertemplate="bin=%{customdata[0]}<br>%{customdata[1]}<extra></extra>")
        else:
            hi_trace = go.Scatter(x=h["UMAP1"], y=h["UMAP2"], mode="markers", marker=dict(size=8, opacity=0.95), customdata=np.c_[np.full(len(h), r["bin"]), h["variant"].astype(str)], hovertemplate="bin=%{customdata[0]}<br>%{customdata[1]}<extra></extra>")
        frames.append(go.Frame(data=[hi_trace, go.Scatter(x=r["x"], y=r["q75"]*factor, mode="lines", line=dict(width=0), hoverinfo="skip"), go.Scatter(x=r["x"], y=r["q25"]*factor, mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(31,119,180,0.18)", hoverinfo="skip"), go.Scatter(x=r["x"], y=r["meanF"]*factor, mode="lines", line=dict(width=2, dash="dash"), customdata=np.full(len(r["x"]), r["bin"])), go.Scatter(x=r.get("med_x", []), y=np.asarray(r.get("med_y", []))*factor, mode="lines", line=dict(width=3), customdata=np.full(len(r.get("med_x", [])), r["bin"])), go.Scatter(x=[r["bin"]], y=[rms[i]], mode="markers", marker=dict(size=12), showlegend=False, customdata=[r["bin"]]), go.Scatter(x=delta_x[i], y=delta_profiles[i], mode="lines", customdata=np.full(len(delta_x[i]), r["bin"]))], traces=[1,2,3,4,5,8,9], name=str(r["bin"]), layout=go.Layout(title_text=f"Trajectory {coord_label}: bin {r['bin']}/{len(reps)} | n={r['n']} | medoid={r.get('medoid','')}")))
    fig.frames = frames
    steps = [dict(method="animate", args=[[str(r["bin"])], {"mode":"immediate", "frame":{"duration":0, "redraw":True}, "transition":{"duration":0}}], label=str(r["bin"])) for r in reps]
    fig.update_layout(template=_template(), height=1080, title=f"Trajectory {coord_label}: bin 1/{len(reps)} | n={first['n']} | medoid={first.get('medoid','')}", margin=dict(l=60, r=20, t=80, b=70), legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0), updatemenus=[dict(type="buttons", direction="left", x=0, y=1.12, buttons=[dict(label="Play", method="animate", args=[None, {"frame":{"duration":650, "redraw":True}, "fromcurrent":True, "transition":{"duration":120}}]), dict(label="Pause", method="animate", args=[[None], {"frame":{"duration":0, "redraw":False}, "mode":"immediate"}])])], sliders=[dict(active=0, currentvalue={"prefix":"bin "}, pad={"t":38}, steps=steps)])
    fig.update_xaxes(title_text="UMAP1", row=1, col=1)
    fig.update_yaxes(title_text="UMAP2", row=1, col=1)
    if is3d:
        fig.update_layout(scene=dict(xaxis_title="UMAP1", yaxis_title="UMAP2", zaxis_title="UMAP3", aspectmode="cube"))
    fig.update_xaxes(title_text=str(metric), range=[xmin-xpad, xmax+xpad], row=1, col=2)
    fig.update_yaxes(title_text=f"F ({unit_label})", range=y_range, row=1, col=2)
    fig.update_xaxes(title_text="bin", row=2, col=1)
    fig.update_yaxes(title_text=f"Δ magnitude ({unit_label})", row=2, col=1)
    fig.update_xaxes(title_text=str(metric), range=[xmin-xpad, xmax+xpad], row=2, col=2)
    fig.update_yaxes(title_text=f"ΔF ({unit_label})", range=[-1.08*dmax, 1.08*dmax], row=2, col=2)

    f_scene_name = "scene2" if is3d else "scene"
    d_scene_name = "scene3" if is3d else "scene2"
    surface_camera = dict(eye=dict(x=1.35, y=-1.55, z=1.45))
    fig.update_layout(**{
        f_scene_name: dict(
            xaxis_title=str(metric),
            yaxis_title="bin",
            zaxis_title=f"F ({unit_label})",
            aspectmode="manual",
            aspectratio=dict(x=1.7, y=1.0, z=0.70),
            camera=surface_camera,
            xaxis=dict(range=[xmin-xpad, xmax+xpad]),
            yaxis=dict(range=[float(np.nanmin(surf_y)) - 0.3, float(np.nanmax(surf_y)) + 0.3]),
            zaxis=dict(range=[f_lo, f_hi]),
        ),
        d_scene_name: dict(
            xaxis_title=str(metric),
            yaxis_title="bin",
            zaxis_title=f"ΔF ({unit_label})",
            aspectmode="manual",
            aspectratio=dict(x=1.7, y=1.0, z=0.70),
            camera=surface_camera,
            xaxis=dict(range=[xmin-xpad, xmax+xpad]),
            yaxis=dict(range=[float(np.nanmin(surf_y)) - 0.3, float(np.nanmax(surf_y)) + 0.3]),
            zaxis=dict(range=[-d_clip, d_clip]),
        ),
    })
    status = f"Trajectory built: {len(reps)} bin(s), path={coord_label}, metric={metric}, unit={unit_label}; representative=mean P→PMF + medoid + IQR; surfaces clipped to F=[{f_lo:.3g},{f_hi:.3g}] and ΔF=±{d_clip:.3g} for display."
    return fig, status, {"bins": bin_records, "path_mode": mode, "metric": metric, "unit": unit_label, "coord_label": coord_label}

def _build_residue_composition_figure(selected_variants: Sequence[str], background_variants: Sequence[str]) -> tuple[go.Figure, str]:
    """Composition + enrichment of selected trajectory bin vs whole-dataset background."""
    sel_df, sel_msg = _residue_composition_for_variants(selected_variants)
    bg_df, bg_msg = _residue_composition_for_variants(background_variants)
    merged = sel_df.merge(bg_df[["AA", "percent", "count"]], on="AA", suffixes=("_selected", "_dataset"))
    merged["Residue"] = [f"{aa} ({AA_NAMES.get(aa, aa)})" for aa in merged["AA"]]
    merged["delta_pp"] = merged["percent_selected"] - merged["percent_dataset"]
    pseudo = 0.05
    merged["fold"] = (merged["percent_selected"] + pseudo) / (merged["percent_dataset"] + pseudo)
    merged["log2_fold"] = np.log2(np.clip(merged["fold"], 1e-12, None))
    fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        horizontal_spacing=0.045,
        column_widths=[0.32, 0.32, 0.36],
        subplot_titles=("Selected bin %", "Whole dataset %", "Enrichment Δ percentage points"),
    )
    fig.add_trace(go.Bar(x=merged["percent_selected"], y=merged["Residue"], orientation="h", customdata=merged[["AA", "count_selected", "fold", "log2_fold"]], hovertemplate="%{customdata[0]}<br>selected=%{x:.2f}%<br>count=%{customdata[1]}<br>fold=%{customdata[2]:.2f}<br>log2=%{customdata[3]:.2f}<extra></extra>", name="selected"), row=1, col=1)
    fig.add_trace(go.Bar(x=merged["percent_dataset"], y=merged["Residue"], orientation="h", customdata=merged[["AA", "count_dataset"]], hovertemplate="%{customdata[0]}<br>dataset=%{x:.2f}%<br>count=%{customdata[1]}<extra></extra>", name="dataset"), row=1, col=2)
    fig.add_trace(go.Bar(x=merged["delta_pp"], y=merged["Residue"], orientation="h", customdata=merged[["AA", "percent_selected", "percent_dataset", "fold", "log2_fold"]], hovertemplate="%{customdata[0]}<br>Δ=%{x:+.2f} pp<br>selected=%{customdata[1]:.2f}%<br>dataset=%{customdata[2]:.2f}%<br>fold=%{customdata[3]:.2f}<br>log2=%{customdata[4]:.2f}<extra></extra>", name="enrichment"), row=1, col=3)
    max_pct = max(1.0, float(max(merged["percent_selected"].max(), merged["percent_dataset"].max())) * 1.15)
    max_delta = max(1.0, float(np.nanmax(np.abs(merged["delta_pp"]))) * 1.20)
    aa_order = [f"{aa} ({AA_NAMES.get(aa, aa)})" for aa in AA20]
    fig.update_yaxes(categoryorder="array", categoryarray=list(reversed(aa_order)))
    fig.update_xaxes(title_text="residue %", range=[0, max_pct], row=1, col=1)
    fig.update_xaxes(title_text="residue %", range=[0, max_pct], row=1, col=2)
    fig.update_xaxes(title_text="selected − dataset (pp)", range=[-max_delta, max_delta], zeroline=True, row=1, col=3)
    fig.update_layout(template=_template(), height=450, margin=dict(l=105, r=20, t=60, b=48), showlegend=False)
    return fig, f"Selected: {sel_msg}. Whole-dataset background: {bg_msg}. Enrichment shown as selected − dataset percentage points; hover shows fold/log2 enrichment."
