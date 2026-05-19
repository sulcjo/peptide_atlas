# Peptide Atlas

Interactive browser for peptide free-energy simulation data.  
Built with [Plotly Dash](https://dash.plotly.com/). Designed to sit on top of
output produced by [GaREUS](https://github.com/sulcjo/peptide_sampler) or
any analysis pipeline that writes per-variant scalar feature tables and PMF
files in the expected layout.

---

## Features

| Group | Tabs |
|---|---|
| **Features** | Raw scalar features table, per-feature statistics |
| **Dim. Reduction** | PCA, UMAP / DensMAP / Isomap / Diffusion Map / PHATE, UMAP→PMF overlay, Basin Landscape |
| **Thermodynamics** | PMF Dendrogram, Curve overlays, Timeseries, Ramachandran 2D |
| **Convergence** | Convergence diagnostics, Diagnosis, Cross-variant Correlation |
| **Info** | README viewer |

### UMAP / Manifold tab highlights

- **Five embedding methods** in one tab: UMAP, DensMAP, Isomap, Diffusion Map, PHATE.
- **Diffusion Map** extra controls: density-correction alpha, configurable number of
  eigenvector components (up to 20), and DC axis selectors (X/Y/Z) that re-render
  without recomputing.
- **PMF curve viewer**: click or lasso-select variants in the scatter to overlay their
  free-energy profiles.
- **Trajectory viewer**: follow a feature gradient or a manually anchored polyline
  through the manifold and inspect per-bin residue composition.
- **Sequence-language clustering**: auto-clusters variants by composition, aligned
  position, and k-mer motifs with silhouette-score selection.
- **Feature gradient arrows** (biplot): top-5 features by Spearman correlation to
  the displayed axes.
- **Embedding cache**: exact-parameter hash cache so re-opening the app or switching
  tabs never recomputes a previously seen embedding.

---

## Quick start

```bash
# Install dependencies
pip install dash dash-bootstrap-components plotly pandas numpy
pip install umap-learn          # optional – required for UMAP/DensMAP
pip install scikit-learn        # optional – Isomap, DBSCAN, kNN stability

# Run (data loads in the background; open the URL immediately)
python -m peptide_dash --data-dir /path/to/GLOBAL_DATA --port 8050

# Synchronous load (waits for data before starting)
python -m peptide_dash --data-dir /path/to/GLOBAL_DATA --sync

# Mock mode (no data – test the layout)
python -m peptide_dash --dev-mock
```

### Key CLI flags

| Flag | Description |
|---|---|
| `--data-dir` | Root of the GLOBAL_DATA directory (or its parent) |
| `--timeseries-dir` | Directory with `$VARIANT/METRIC_REP#.xvg` timeseries files |
| `--sample5` | Load ~5 % of variants for quick testing |
| `--sample-frac F` | Load fraction F of variants (0–1] |
| `--sync` | Block until data is fully loaded before starting |
| `--dev-mock` | Empty context — test layout only |
| `--dev-quick` | Sniff a small numeric column subset on startup |
| `--debug` | Enable Dash debug/hot-reload mode |

---

## Expected data layout

```
GLOBAL_DATA/
  VARIANT_A/
    features.csv          # per-variant scalar features (one row per variant)
    pmf_*.csv             # free-energy profiles
    timeseries/           # optional XVG timeseries
  VARIANT_B/
    ...
```

The loader (`peptide_dash/data/`) auto-discovers variants and metrics from this
tree. Columns named `variant` and `metric` are reserved.

---

## Package structure

```
peptide_dash/
  app.py              – Dash app factory and grouped nav shell
  cli.py              – CLI entry point (python -m peptide_dash)
  metrics.py          – Metric label helpers
  data/               – DataContext, background loader, IO helpers
  analysis/           – Pure-Python analysis (PMF, PCA, stats, UMAP feedback)
  theming/            – CSS helpers and error-figure builders
  tabs/
    features.py       – Feature table tab
    pca_tab.py        – Variance & PCA tab
    umap/             – UMAP subpackage (see tabs/umap/README.md)
    umap_tab.py       – Compatibility wrapper for umap/
    convergence.py    – Convergence tab
    ...
```

---

## Dependencies

| Package | Required | Purpose |
|---|---|---|
| `dash` | yes | UI framework |
| `dash-bootstrap-components` | yes | Layout components |
| `plotly` | yes | Figures |
| `pandas` / `numpy` | yes | Data handling |
| `umap-learn` | for UMAP/DensMAP | Manifold embedding |
| `scikit-learn` | recommended | Isomap, DBSCAN, kNN |

Diffusion Map and PHATE run without any optional dependencies (pure NumPy).

---

## License

MIT
