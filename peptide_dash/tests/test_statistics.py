"""
Unit tests for the statistical core of the peptide analysis pipeline.

What these catch:

1. tau_int convention mismatches between modular and batch code paths
   (previously: analysis/stats.py used Sokal convention, 3_BATCH_ANA
   used emcee convention, producing tau values that disagreed by 2x).

2. Regressions in the decorrelation stride (should be ~2 * tau_Sokal,
   not 1 * tau_emcee - these are numerically equivalent, but under the
   Sokal convention the factor of 2 is explicit).

3. Bootstrap PMF recovering a known analytic free energy surface from
   Gaussian samples (sanity check that the estimator is unbiased and
   that the CI contains the truth at the nominal rate).

4. Circular statistics on uniform vs. concentrated torsion angles
   (guards against the common mistake of computing linear mean on
   angular data).

5. Exclusion-pattern anchoring - regression test that "rms" no longer
   eats "rmsd" / "rmsf", and "js" no longer eats arbitrary columns.

Run with:  python -m pytest tests/ -v
or:        python tests/test_statistics.py
"""
from __future__ import annotations

import importlib.util
import os
import sys
import unittest

import numpy as np


HERE = os.path.dirname(os.path.abspath(__file__))
PKG_ROOT = os.path.dirname(HERE)
REPO_ROOT = os.path.dirname(PKG_ROOT)


def _load_batch_module():
    """Load 3_BATCH_ANA_replica_curves.py as a module (the leading digit
    makes it not a regular import target)."""
    candidates = [
        os.path.join(PKG_ROOT, "3_BATCH_ANA_replica_curves.py"),
        os.path.join(REPO_ROOT, "3_BATCH_ANA_replica_curves.py"),
    ]
    path = next((p for p in candidates if os.path.exists(p)), candidates[0])
    spec = importlib.util.spec_from_file_location("batch_ana", path)
    m = importlib.util.module_from_spec(spec)
    sys.modules["batch_ana"] = m
    spec.loader.exec_module(m)
    return m


BATCH = _load_batch_module()

# Modular analysis side (import via the package layout). We shim a parent
# package so relative imports in data/context.py do not fire.
sys.path.insert(0, PKG_ROOT)
from analysis import stats as mod_stats  # noqa: E402


# ---------------------------------------------------------------------------
# AR(1) generator: known autocorrelation time for validation
# ---------------------------------------------------------------------------

def ar1(n: int, phi: float, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    """
    First-order autoregressive process.  For |phi| < 1 the analytic
    integrated autocorrelation time (Sokal convention) is

        tau_int = 1/2 + phi / (1 - phi)       (sum of rho_k = phi^k, k>=1)

    which simplifies to (1 + phi) / (2 (1 - phi)).

    So phi = 0.8 ->  tau_int = 1.8 / 0.4 = 4.5 frames.
    """
    rng = np.random.default_rng(seed)
    eps = rng.normal(0.0, sigma, size=int(n))
    x = np.empty(int(n), float)
    x[0] = eps[0] / np.sqrt(1.0 - phi * phi)
    for i in range(1, int(n)):
        x[i] = phi * x[i - 1] + eps[i]
    return x


def analytic_tau_sokal(phi: float) -> float:
    return (1.0 + phi) / (2.0 * (1.0 - phi))


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTauIntConvention(unittest.TestCase):
    """Confirm both code paths agree on Sokal convention, tau >= 0.5."""

    def test_batch_ana_tau_recovers_analytic_ar1(self):
        phi = 0.8
        tau_true = analytic_tau_sokal(phi)  # 4.5
        # Long series so FFT ACF is clean; average over a few chains.
        taus = [BATCH._autocorr_time_int(ar1(200_000, phi, seed=s)) for s in range(5)]
        tau_hat = float(np.mean(taus))
        # Allow 10% error - FFT ACF + initial-positive truncation is biased low
        self.assertGreater(tau_hat, 0.9 * tau_true,
                           f"tau_hat={tau_hat:.3f} too low vs true {tau_true}")
        self.assertLess(tau_hat, 1.2 * tau_true,
                        f"tau_hat={tau_hat:.3f} too high vs true {tau_true}")

    def test_tau_is_half_for_iid(self):
        """Sokal convention: tau_int = 0.5 for iid data."""
        rng = np.random.default_rng(42)
        x = rng.normal(size=50_000)
        tau = BATCH._autocorr_time_int(x)
        self.assertGreaterEqual(tau, 0.5 - 1e-9)
        # White noise: initial positive sum is near zero -> tau ~ 0.5
        self.assertLess(tau, 0.9, f"iid tau should be near 0.5, got {tau}")

    def test_modular_and_batch_agree_on_convention(self):
        """
        Both implementations now use tau = 0.5 + sum rho_k. Feed the same ACF
        (computed in the batch script's style) to the modular helper and
        confirm the result matches the batch _autocorr_time_int value.
        """
        phi = 0.6
        x = ar1(100_000, phi, seed=7)

        # Use the batch ACF pipeline so the inputs are identical
        xc = x - x.mean()
        n = x.size
        nfft = 1 << (2 * n - 1).bit_length()
        fx = np.fft.rfft(xc, n=nfft)
        acf = np.fft.irfft(fx * np.conjugate(fx), n=nfft)[:n]
        acf = acf / acf[0]

        tau_mod = mod_stats.integrated_autocorr_time(acf)
        tau_batch = BATCH._autocorr_time_int(x)

        # Relative difference should be tiny (same formula on same ACF).
        self.assertAlmostEqual(tau_mod, tau_batch, delta=0.05,
                               msg=f"mod={tau_mod} vs batch={tau_batch}")

    def test_ess_consistency_sokal(self):
        """Under Sokal convention ESS = N / (2 tau)."""
        n = 50_000
        phi = 0.8
        x = ar1(n, phi, seed=1)
        tau = BATCH._autocorr_time_int(x)
        ess_expected = n / (2.0 * tau)
        ess_got = mod_stats.effective_sample_size(n, tau)
        self.assertAlmostEqual(ess_expected, ess_got, delta=1.0)


class TestDecorrelationStride(unittest.TestCase):

    def test_iid_stride_is_one(self):
        self.assertEqual(BATCH._decorrelation_stride(0.5), 1)
        self.assertEqual(BATCH._decorrelation_stride(0.3), 1)

    def test_ar1_stride_is_about_2tau(self):
        # For tau_Sokal = 4.5, expected stride = round(9.0) = 9.
        self.assertEqual(BATCH._decorrelation_stride(4.5), 9)
        self.assertEqual(BATCH._decorrelation_stride(10.0), 20)

    def test_nan_stride_is_one(self):
        self.assertEqual(BATCH._decorrelation_stride(float("nan")), 1)


class TestBootstrapPMF(unittest.TestCase):
    """
    Generate independent replica series from a known analytic potential
    U(x) = 0.5 * k * x^2 (harmonic), for which the true PMF relative to
    its minimum is F(x) = 0.5 * k * x^2. Check that the bootstrap CI
    contains the truth at roughly the nominal rate across bins.
    """

    def setUp(self):
        self.T = 300.0
        self.kT = BATCH.R_GAS * self.T
        self.k = 2.0  # kJ/mol/A^2
        sigma = np.sqrt(self.kT / self.k)  # Boltzmann width
        self.sigma = sigma
        rng = np.random.default_rng(123)
        # 6 independent replicas of 20k frames each
        self.reps = [rng.normal(0.0, sigma, 20_000) for _ in range(6)]
        self.edges = np.linspace(-3.0 * sigma, 3.0 * sigma, 61)
        self.centers = 0.5 * (self.edges[:-1] + self.edges[1:])
        self.F_true = 0.5 * self.k * self.centers ** 2
        # Zero the truth at its minimum, same as bootstrap does
        self.F_true = self.F_true - float(self.F_true.min())

    def test_point_estimate_matches_truth_in_well(self):
        pooled = np.concatenate(self.reps)
        counts, _ = np.histogram(pooled, bins=self.edges)
        _, F, _ = BATCH._pmf_from_counts(counts, self.edges, T=self.T)
        # Look only at the well (|x| <= 1.5 sigma) where statistics are good.
        mask = np.abs(self.centers) <= 1.5 * self.sigma
        resid = F[mask] - self.F_true[mask]
        # Central bins: well within 0.3 kJ/mol of truth
        self.assertLess(float(np.max(np.abs(resid))), 0.3,
                        f"max |F - F_true| = {np.max(np.abs(resid)):.3f}")

    def test_bootstrap_ci_covers_truth(self):
        """
        Coverage test for the replica-block percentile bootstrap.

        PERCENTILE BOOTSTRAP UNDERCOVERAGE: for histogram PMFs this
        estimator is known to be narrower than nominal, especially in
        low-count (tail) bins, because:
          - log(p_hat) is a biased estimate of log(p) by Jensen's inequality
            when counts are small, shifting the point estimate below truth;
          - the percentile method doesn't correct for bias or skewness.
        Hub, de Groot & van der Spoel (JCTC 2010, g_wham paper) document
        this for their method (ii). BCa would improve coverage at
        additional cost; not implemented here.

        We therefore require mean coverage across 8 independent replica
        realizations to be >= 0.70 for a nominal 0.95 CI. A regression
        that pushes it below 0.70 (narrower bars) or above 0.99
        (overly wide bars) indicates a real bug.
        """
        T = self.T
        sigma = self.sigma
        edges = self.edges
        centers = self.centers
        F_true = self.F_true
        well = np.abs(centers) <= 1.5 * sigma

        coverages = []
        for trial in range(8):
            rng = np.random.default_rng(10_000 + trial)
            reps = [rng.normal(0.0, sigma, 20_000) for _ in range(6)]
            out = BATCH.bootstrap_pmf_over_replicas(
                reps, edges=edges, T=T, n_boot=200, rng_seed=trial,
            )
            F_lo = out["F_lo"]
            F_hi = out["F_hi"]
            covered = (F_true[well] >= F_lo[well]) & (F_true[well] <= F_hi[well])
            coverages.append(float(covered.mean()))

        mean_cov = float(np.mean(coverages))
        self.assertGreater(
            mean_cov, 0.70,
            f"mean coverage across trials = {mean_cov:.2f} "
            f"(per-trial: {[round(c, 2) for c in coverages]}) - "
            "bootstrap bars are unrealistically narrow, investigate bias."
        )
        self.assertLess(
            mean_cov, 0.99,
            f"mean coverage {mean_cov:.2f} is > 99% for a 95% CI - "
            "bootstrap bars are unrealistically wide, investigate."
        )

        # Sanity: the point-estimate CI has finite width in the well, and
        # F_std is strictly positive there.
        out_single = BATCH.bootstrap_pmf_over_replicas(
            self.reps, edges=edges, T=T, n_boot=200, rng_seed=0,
        )
        self.assertTrue(np.all(np.isfinite(out_single["F_std"][well])))
        self.assertTrue(np.all(out_single["F_std"][well] > 0.0))
        self.assertTrue(np.all(out_single["F_hi"][well] >= out_single["F_lo"][well]))

    def test_bootstrap_nan_with_single_replica(self):
        """With only 1 replica, block bootstrap has no signal -> NaN CI."""
        out = BATCH.bootstrap_pmf_over_replicas(
            [self.reps[0]], edges=self.edges, T=self.T, n_boot=50, rng_seed=0,
        )
        self.assertTrue(np.all(np.isnan(out["F_std"])))
        self.assertEqual(out["n_boot_effective"], 0)


class TestJSDivergence(unittest.TestCase):

    def test_identical_is_zero(self):
        rng = np.random.default_rng(0)
        p = rng.dirichlet(np.ones(20))
        self.assertAlmostEqual(BATCH.js_divergence(p, p), 0.0, places=10)
        self.assertAlmostEqual(mod_stats.js_divergence(p, p), 0.0, places=10)

    def test_disjoint_is_log2(self):
        # Two disjoint supports -> JS = ln(2) in nats (~0.693).
        p = np.array([1.0, 0.0, 0.0])
        q = np.array([0.0, 0.0, 1.0])
        self.assertAlmostEqual(BATCH.js_divergence(p, q), np.log(2), places=6)

    def test_symmetry(self):
        rng = np.random.default_rng(1)
        p = rng.dirichlet(np.ones(10))
        q = rng.dirichlet(np.ones(10))
        self.assertAlmostEqual(
            BATCH.js_divergence(p, q),
            BATCH.js_divergence(q, p),
            places=12,
        )


class TestCircularStats(unittest.TestCase):

    def test_concentrated_mean(self):
        # Angles clustered tightly around 60 deg
        rng = np.random.default_rng(0)
        a = 60.0 + rng.normal(0.0, 5.0, 10_000)
        cs = BATCH.circular_stats_deg(a)
        self.assertAlmostEqual(cs["circular_mean_deg"], 60.0, delta=0.5)
        self.assertGreater(cs["circular_R"], 0.99)

    def test_wraparound_mean(self):
        """Mean of angles split between +170 and -170 must be near +/-180, not 0."""
        a = np.concatenate([
            170.0 + np.random.default_rng(0).normal(0.0, 2.0, 5000),
            -170.0 + np.random.default_rng(1).normal(0.0, 2.0, 5000),
        ])
        cs = BATCH.circular_stats_deg(a)
        # Linear mean would be ~0; circular mean must be ~180 (or -180).
        mean = cs["circular_mean_deg"]
        self.assertTrue(abs(abs(mean) - 180.0) < 2.0,
                        f"wraparound failed: circular mean = {mean}")

    def test_uniform_has_small_R(self):
        rng = np.random.default_rng(2)
        a = rng.uniform(-180.0, 180.0, 20_000)
        cs = BATCH.circular_stats_deg(a)
        self.assertLess(cs["circular_R"], 0.05)


class TestExclusionPatterns(unittest.TestCase):
    """Regression: the old substring-based exclusion list ate legitimate
    features. The new regex-anchored list must not."""

    def setUp(self):
        # Import data.context by path (package layout assumed).
        import importlib.util as u
        path = os.path.join(PKG_ROOT, "data", "context.py")
        # context.py does `from . import io`, so we need to load the sibling too.
        sys_path_added = False
        if PKG_ROOT not in sys.path:
            sys.path.insert(0, PKG_ROOT)
            sys_path_added = True
        # Give it a proper package name so relative imports resolve.
        spec_pkg = u.spec_from_file_location(
            "ptd_data",
            os.path.join(PKG_ROOT, "data", "__init__.py"),
            submodule_search_locations=[os.path.join(PKG_ROOT, "data")],
        )
        pkg = u.module_from_spec(spec_pkg)
        sys.modules["ptd_data"] = pkg
        spec_pkg.loader.exec_module(pkg)

        spec = u.spec_from_file_location("ptd_data.context", path)
        self.ctx_mod = u.module_from_spec(spec)
        sys.modules["ptd_data.context"] = self.ctx_mod
        # io is needed as relative import target
        io_spec = u.spec_from_file_location(
            "ptd_data.io", os.path.join(PKG_ROOT, "data", "io.py")
        )
        io_mod = u.module_from_spec(io_spec)
        sys.modules["ptd_data.io"] = io_mod
        io_spec.loader.exec_module(io_mod)
        spec.loader.exec_module(self.ctx_mod)
        self._sys_path_added = sys_path_added

    def tearDown(self):
        if self._sys_path_added:
            try:
                sys.path.remove(PKG_ROOT)
            except ValueError:
                pass

    def test_rmsd_not_excluded(self):
        self.assertFalse(self.ctx_mod._is_excluded("rmsd_backbone"))
        self.assertFalse(self.ctx_mod._is_excluded("rmsf_per_residue"))
        self.assertFalse(self.ctx_mod._is_excluded("rmsd"))

    def test_js_substring_not_excluded(self):
        # Columns like "adjust_level" or "n_justified" must survive.
        self.assertFalse(self.ctx_mod._is_excluded("adjust_level"))
        # But the intended excluded name still goes.
        self.assertTrue(self.ctx_mod._is_excluded("js_reps_to_pooled_mean"))
        self.assertTrue(self.ctx_mod._is_excluded("dist_term__js_reps_to_pooled_mean"))
        self.assertTrue(self.ctx_mod._is_excluded("dist_term__L1_reps_to_pooled_max"))

    def test_underscore_n_substring_not_excluded(self):
        # "_n" used to eat "n_contacts" etc.
        self.assertFalse(self.ctx_mod._is_excluded("n_contacts"))
        self.assertFalse(self.ctx_mod._is_excluded("sasa_polar_norm"))

    def test_mbar_prefix_excluded(self):
        self.assertTrue(self.ctx_mod._is_excluded("mbar_dF"))
        # But a metric that merely contains "mbar" mid-name is kept.
        self.assertFalse(self.ctx_mod._is_excluded("some_mbar_thing"))

    def test_tau_int_excluded(self):
        self.assertTrue(self.ctx_mod._is_excluded("tau_int_frames_mean"))
        self.assertTrue(self.ctx_mod._is_excluded("tau_int"))
        # "tau_integrated_whatever" should still be kept (not a real column,
        # but the point is that the pattern is anchored).
        self.assertFalse(self.ctx_mod._is_excluded("tau_integrated_whatever"))


# ---------------------------------------------------------------------------

class TestPmfPlotCi(unittest.TestCase):
    """Smoke tests for the shared CI plotting helpers in tabs/pmf_plot_ci.py.

    These don't render figures - they just verify the module imports, the
    helper functions return sensibly-shaped outputs on realistic DataFrames,
    and the capability probe works. This catches the kind of silent breakage
    that would reach production only when someone opens the dashboard tab.
    """

    @classmethod
    def setUpClass(cls):
        # Load pmf_plot_ci directly by path, NOT via the tabs package, so we
        # don't import dash (the package __init__ does `from dash import ...`
        # which is not required to test the plotting primitives).
        import importlib.util
        path = os.path.join(PKG_ROOT, "tabs", "pmf_plot_ci.py")
        spec = importlib.util.spec_from_file_location("pmf_plot_ci_test", path)
        mod = importlib.util.module_from_spec(spec)
        sys.modules["pmf_plot_ci_test"] = mod
        spec.loader.exec_module(mod)
        cls.plot_ci = mod

    def _make_pmf_df(self, with_ci: bool = True) -> "pd.DataFrame":
        import pandas as pd  # local to avoid top-of-file hard dep ordering
        x = np.linspace(-3.0, 3.0, 20)
        F = 0.5 * 2.0 * x ** 2
        F = F - F.min()
        rows = []
        for v in ("pep_A", "pep_B", "pep_C"):
            for xi, fi in zip(x, F):
                row = {"variant": v, "metric": "rg_prot",
                       "x": float(xi), "F_kJ_mol": float(fi),
                       "P": float(np.exp(-fi / 2.5)), "T_K": 300.0,
                       "method": "decorrelated"}
                if with_ci:
                    row["F_ci_lo_kJ_mol"] = float(fi - 0.2)
                    row["F_ci_hi_kJ_mol"] = float(fi + 0.2)
                rows.append(row)
        return pd.DataFrame(rows)

    def test_has_ci_columns(self):
        df_with = self._make_pmf_df(with_ci=True)
        df_without = self._make_pmf_df(with_ci=False)
        self.assertTrue(self.plot_ci.pmf_has_ci_columns(df_with))
        self.assertFalse(self.plot_ci.pmf_has_ci_columns(df_without))

    def test_per_variant_returns_shaped_data(self):
        df = self._make_pmf_df(with_ci=True)
        curves, bands = self.plot_ci.per_variant_raw_pmf_with_ci(
            df, variants=["pep_A", "pep_B"], metric="rg_prot",
        )
        self.assertEqual(set(curves.keys()), {"pep_A", "pep_B"})
        self.assertEqual(set(bands.keys()), {"pep_A", "pep_B"})
        x, F = curves["pep_A"]
        lo, hi = bands["pep_A"]
        self.assertEqual(x.shape, F.shape)
        self.assertEqual(lo.shape, F.shape)
        self.assertTrue(np.all(hi >= lo))

    def test_per_variant_caps_at_max_variants(self):
        df = self._make_pmf_df(with_ci=True)
        curves, _ = self.plot_ci.per_variant_raw_pmf_with_ci(
            df, variants=["pep_A", "pep_B", "pep_C"],
            metric="rg_prot", max_variants=2,
        )
        self.assertEqual(len(curves), 2)

    def test_per_variant_empty_bands_when_no_ci(self):
        df = self._make_pmf_df(with_ci=False)
        curves, bands = self.plot_ci.per_variant_raw_pmf_with_ci(
            df, variants=["pep_A"], metric="rg_prot",
        )
        self.assertEqual(len(curves), 1)
        self.assertEqual(bands, {})

    def test_overlay_fig_handles_empty(self):
        fig = self.plot_ci.pmf_overlay_fig({}, title="empty test",
                                           error_text="no data")
        # Should return a Figure with a single annotation, no data traces
        self.assertEqual(len(fig.data), 0)

    def test_overlay_fig_with_bands(self):
        df = self._make_pmf_df(with_ci=True)
        curves, bands = self.plot_ci.per_variant_raw_pmf_with_ci(
            df, variants=["pep_A", "pep_B"], metric="rg_prot",
        )
        fig = self.plot_ci.pmf_overlay_fig(
            curves, title="t", y_title="F (kJ/mol)", ci_bands=bands,
        )
        # 2 variants -> 2 line traces + 2 band traces = 4
        self.assertEqual(len(fig.data), 4)
        # Band should precede line so line sits on top
        modes_and_fills = [(t.mode, getattr(t, "fill", None)) for t in fig.data]
        self.assertEqual(modes_and_fills[0][1], "toself")  # band first
        self.assertIsNone(modes_and_fills[1][1])           # line next


class TestPmfStorageConsistency(unittest.TestCase):
    """PMF smoothing must keep stored F and P mutually consistent."""

    def test_probability_from_smoothed_pmf_roundtrips_to_same_F(self):
        T = 300.0
        counts = np.array([1, 3, 12, 30, 12, 3, 1], dtype=int)
        edges = np.linspace(-3.5, 3.5, counts.size + 1)
        _x, F_raw, P_raw = BATCH._pmf_from_counts(counts, edges, T=T)
        F = BATCH._smooth_pmf_savgol(F_raw, counts, window=5, min_count=1)
        P = BATCH._probability_from_pmf(F, T=T, fallback=P_raw)
        kT = BATCH.R_GAS * T
        with np.errstate(divide="ignore", invalid="ignore"):
            F_back = -kT * np.log(np.clip(P, 1e-300, 1.0))
        F_back = F_back - float(np.nanmin(F_back[np.isfinite(F_back)]))
        mask = np.isfinite(F) & np.isfinite(F_back) & (P > 0.0)
        np.testing.assert_allclose(F_back[mask], F[mask], atol=1e-10)

    def test_bootstrap_smoothing_keeps_ci_shape(self):
        rng = np.random.default_rng(123)
        reps = [rng.normal(0.0, 1.0, 2000) for _ in range(4)]
        edges = np.linspace(-4.0, 4.0, 41)
        out = BATCH.bootstrap_pmf_over_replicas(
            reps,
            edges=edges,
            T=300.0,
            n_boot=25,
            rng_seed=5,
            smooth_window=5,
            smooth_mincount=1,
        )
        self.assertEqual(out["F_lo"].shape[0], edges.size - 1)
        self.assertGreater(out["n_boot_effective"], 0)


class TestPmfVectorizerPhysicalSemantics(unittest.TestCase):
    """Shared PMF vectorization should respect temperature and torsion periodicity."""

    def test_F_to_P_uses_curve_T_K_column(self):
        import pandas as pd
        from analysis import pmf_vectorize as pv

        T_curve = 600.0
        kT_curve = pv.R_GAS * T_curve
        df = pd.DataFrame(
            {
                "variant": ["A", "A"],
                "metric": ["rg", "rg"],
                "x": [0.0, 1.0],
                "F_kJ_mol": [0.0, kT_curve],
                "T_K": [T_curve, T_curve],
            }
        )
        X, _cols, _meta, _grids = pv.build_pmf_design_matrix(
            df, ["rg"], use_repr="F", T_K=300.0, variant_policy="intersection"
        )
        expected = np.array([1.0, np.exp(-1.0)])
        expected = expected / expected.sum()
        np.testing.assert_allclose(X[0], expected, rtol=1e-6, atol=1e-8)

    def test_periodic_interpolation_wraps_torsion_edges(self):
        from analysis import pmf_vectorize as pv

        grid = np.array([-179.0, 0.0, 179.0])
        x = np.array([170.0, 179.0])
        dens = np.array([1.0, 1.0])
        linear = pv.density_to_mass_on_grid(grid, x, dens, periodic=False)
        periodic = pv.density_to_mass_on_grid(grid, x, dens, periodic=True)
        self.assertEqual(float(linear[0]), 0.0)
        self.assertGreater(float(periodic[0]), 0.0)
        self.assertTrue(pv.is_periodic_pmf_metric("phi_res1"))


class TestUmapPhysicalDefaults(unittest.TestCase):
    """UMAP defaults should match the physical meaning of the selected input."""

    def test_pmf_source_presets_default_to_hellinger(self):
        from peptide_dash.tabs import umap_tab as u

        self.assertEqual(u._umap_defaults("global", 2, u.UMAP_SOURCE_PMF)[2], "hellinger")
        self.assertEqual(u._umap_defaults("local", 3, u.UMAP_SOURCE_PMF)[2], "hellinger")
        self.assertEqual(u._umap_defaults("global", 2, u.UMAP_SOURCE_BASIC)[2], "cosine")

    def test_basic_source_excludes_sequence_and_physical_annotations(self):
        import pandas as pd
        from peptide_dash.tabs import umap_tab as u

        df = pd.DataFrame(
            {
                "variant": ["A", "B", "C", "D"],
                "native_a": [0.0, 1.0, 2.0, 3.0],
                "native_b": [3.0, 2.0, 1.0, 0.0],
                "seq_length": [8, 8, 8, 8],
                "phi__global_basin_min_x": [-170.0, 80.0, 20.0, -60.0],
            }
        )
        X, cols, meta, note = u._prepare_feature_design_matrix(df, u.UMAP_SOURCE_BASIC)
        self.assertEqual(X.shape, (4, 2))
        self.assertEqual(cols, ["native_a", "native_b"])
        self.assertEqual(meta["variant"].tolist(), ["A", "B", "C", "D"])
        self.assertIn("mixed-unit PMF annotations excluded", note)

    def test_colored_points_are_not_hidden_by_fit_overlay(self):
        import pandas as pd
        from peptide_dash.tabs import umap_tab as u

        df = pd.DataFrame(
            {
                "variant": ["A", "B", "C", "D"],
                "UMAP1": [0.0, 1.0, 2.0, 3.0],
                "UMAP2": [0.0, 1.0, 0.0, 1.0],
                "role": ["fit+plot", "fit+plot", "fit+plot", "fit+plot"],
                "score": [0.1, 0.4, 0.7, 1.0],
            }
        )

        fig = u._build_umap_figure(df, dims=2, colorby="score", label_col=None)

        self.assertEqual(len(fig.data), 1)
        colors = list(fig.data[0].marker.color)
        self.assertEqual(colors, [0.1, 0.4, 0.7, 1.0])


class TestPcaSequencePreset(unittest.TestCase):
    """PCA should expose sequence descriptors as a dedicated input space."""

    def test_sequence_aliases_and_columns(self):
        import pandas as pd
        from peptide_dash.tabs import pca_tab as p

        self.assertEqual(p._pca_preset_mode("seq"), p.PCA_PRESET_SEQUENCE)
        self.assertTrue(p._is_sequence_preset("sequence"))

        df = pd.DataFrame(
            {
                "variant": ["A", "B", "C", "D"],
                "native_a": [0.0, 1.0, 2.0, 3.0],
                "seq_length": [7, 8, 9, 10],
                "seq_aa_A_frac": [0.1, 0.2, 0.3, 0.4],
                "seq_aa_A_count": [1, 2, 3, 4],
                "seq_dipep_AG_frac": [0.0, 0.1, 0.2, 0.3],
                "phi__global_basin_min_x": [-170.0, 80.0, 20.0, -60.0],
            }
        )

        raw = [c for c in df.columns if c != "variant" and pd.api.types.is_numeric_dtype(df[c])]
        cols = p._sequence_pca_feature_cols(df, raw)

        self.assertIn("seq_length", cols)
        self.assertIn("seq_aa_A_frac", cols)
        self.assertIn("seq_dipep_AG_frac", cols)
        self.assertNotIn("seq_aa_A_count", cols)
        self.assertNotIn("native_a", cols)
        self.assertNotIn("phi__global_basin_min_x", cols)


class TestFeaturesSequenceFamily(unittest.TestCase):
    """Sequence feature columns should be grouped visibly in the Features tab."""

    def test_seq_columns_group_as_sequence_family(self):
        from peptide_dash.tabs import features as f

        self.assertEqual(f._feature_family_name("seq_length"), "seq")
        self.assertEqual(f._feature_family_name("seq_aa_A_frac"), "seq")
        self.assertEqual(f._feature_family_name("KD_mean"), "seq")
        self.assertEqual(f._feature_family_name("rg__mean"), "rg")


class TestHellingerPreservation(unittest.TestCase):
    """
    Directly verify that the Hellinger-mode preprocessing pipeline preserves
    Hellinger distances on PMFs, up to the PCA truncation error.

    Hellinger distance between two probability vectors p, q:
        H(p, q) = (1/sqrt(2)) * ||sqrt(p) - sqrt(q)||_2
    so Euclidean distance on sqrt(P) = sqrt(2) * H.

    The pipeline in Hellinger mode applies:
        (1) sqrt(P)
        (2) drop low-support bins (introduces bounded error)
        (3) center (rigid translation, preserves Euclidean distance exactly)
        (4) family balancing (per-family scalar, reweights H^2 per family)
        (5) PCA truncation (loses <=1% variance by EVR=0.99 criterion)

    Steps (3) and (4) are exact for the reweighted H. Step (2) introduces a
    small error bounded by the dropped tail mass. Step (5) is controlled by
    EVR. So the final pairwise Euclidean distances in PC space should equal
    the family-weighted Hellinger distances to <5% relative error on
    reasonably well-populated synthetic PMFs.

    We verify this by replicating the preprocessing in the test and comparing
    full pairwise distance matrices.
    """

    def _make_pmf_matrix(self, n_variants: int = 12, n_bins: int = 40,
                         n_families: int = 2, seed: int = 0):
        """Build a synthetic PMF design matrix: (n_variants, n_families*n_bins)
        with n_families blocks, each row normalized per-family to sum 1."""
        rng = np.random.default_rng(seed)
        x = np.linspace(-3.0, 3.0, n_bins)
        blocks = []
        colnames = []
        for f in range(n_families):
            block = np.zeros((n_variants, n_bins))
            for i in range(n_variants):
                # Gaussian-ish PMFs with random shifts/widths per variant
                mu_i = rng.normal(0.0, 0.8)
                sigma_i = 0.5 + 0.3 * rng.random()
                p = np.exp(-0.5 * ((x - mu_i) / sigma_i) ** 2)
                p = p / p.sum()
                block[i, :] = p
            blocks.append(block)
            colnames.extend([f"metric{f}|x={xi:g}" for xi in x])
        X = np.concatenate(blocks, axis=1)
        return X, colnames

    def _hellinger_distance_matrix(self, P):
        """Classical Hellinger distance on each family block separately,
        then sum of squared distances across families (matches what
        family-weighted Hellinger computes)."""
        sq = np.sqrt(np.clip(P, 0.0, None))
        n = sq.shape[0]
        D2 = np.zeros((n, n))
        for i in range(n):
            for j in range(i + 1, n):
                d2 = 0.5 * float(np.sum((sq[i] - sq[j]) ** 2))
                D2[i, j] = D2[j, i] = d2
        return D2  # squared Hellinger, unweighted

    def _apply_pipeline_hellinger(self, X_raw, colnames):
        """Mirror the preprocessing in umap_tab._compute_embedding for
        Hellinger mode, but stop before UMAP and return the PC-space matrix."""
        # (1) sqrt
        X = np.sqrt(np.clip(X_raw, 0.0, None))
        # (2) drop low-support bins
        min_support = max(3, int(0.10 * X.shape[0]))
        support = np.sum(X > 1e-6, axis=0)
        keep = support >= min_support
        X = X[:, keep]
        colnames = [c for c, k in zip(colnames, keep.tolist()) if k]
        # (3) center only (sd=1)
        mu = X.mean(axis=0)
        Xz = X - mu
        # (4) family balancing
        fams = np.asarray([c.split("|", 1)[0] for c in colnames])
        uniq, counts = np.unique(fams, return_counts=True)
        fam2cnt = dict(zip(uniq.tolist(), counts.tolist()))
        scales = np.asarray([1.0 / np.sqrt(float(fam2cnt[f])) for f in fams])
        Xz = Xz * scales
        # (5) PCA to 99% EVR
        _, s, Vt = np.linalg.svd(Xz, full_matrices=False)
        evr = (s ** 2) / np.sum(s ** 2)
        k = int(np.searchsorted(np.cumsum(evr), 0.99) + 1)
        k = min(k, 64, Xz.shape[1])
        X_pca = Xz @ Vt[:k, :].T
        return X_pca, scales, fams

    def _pairwise_euclidean(self, X):
        diff = X[:, None, :] - X[None, :, :]
        return np.sqrt(np.sum(diff ** 2, axis=-1))

    def test_centering_preserves_pairwise_distances(self):
        """Sanity: centering must not change pairwise Euclidean distances."""
        X_raw, _ = self._make_pmf_matrix(n_variants=8, seed=1)
        sq = np.sqrt(X_raw)
        D_before = self._pairwise_euclidean(sq)
        D_after = self._pairwise_euclidean(sq - sq.mean(axis=0))
        np.testing.assert_allclose(D_before, D_after, atol=1e-10)

    def test_std_scaling_does_change_distances(self):
        """Guard: per-column std scaling MUST change distances (otherwise
        the whole premise of the Hellinger-mode patch is wrong)."""
        X_raw, _ = self._make_pmf_matrix(n_variants=8, seed=1)
        sq = np.sqrt(X_raw)
        D_before = self._pairwise_euclidean(sq - sq.mean(axis=0))
        sd = sq.std(axis=0, ddof=1)
        sd = np.where(sd == 0, 1.0, sd)
        Xz = (sq - sq.mean(axis=0)) / sd
        D_after = self._pairwise_euclidean(Xz)
        # Should differ; require Frobenius ratio not within 5%.
        rel = np.linalg.norm(D_after - D_before) / max(np.linalg.norm(D_before), 1e-12)
        self.assertGreater(rel, 0.05,
                           "z-scoring did not change distances? Test is broken or data too trivial.")

    def test_pipeline_preserves_family_weighted_hellinger(self):
        """
        Main claim: after the full Hellinger-mode pipeline, pairwise Euclidean
        distances in PC space match the family-weighted Hellinger distances
        on the original PMFs, to better than 5% relative Frobenius error.

        Family-weighted Hellinger^2 = sum_f (1/n_bins_f) * H_f^2
        where H_f is the standard Hellinger on family f.
        """
        X_raw, colnames = self._make_pmf_matrix(
            n_variants=14, n_bins=40, n_families=2, seed=2,
        )

        # Expected: family-weighted Hellinger distance on original P
        fams = np.asarray([c.split("|", 1)[0] for c in colnames])
        uniq_fams = np.unique(fams)
        n = X_raw.shape[0]
        D2_expected = np.zeros((n, n))
        for f in uniq_fams:
            cols = np.where(fams == f)[0]
            Pf = X_raw[:, cols]
            # Rows should already be normalized per family in the synthetic
            # data; double-check.
            row_sums = Pf.sum(axis=1, keepdims=True)
            Pf = Pf / np.clip(row_sums, 1e-12, None)
            sq = np.sqrt(np.clip(Pf, 0.0, None))
            # Sum of squared sqrt(P) differences, with 1/n_bins family weight.
            # Note: the pipeline's family balancing scales sqrt(P) by
            # 1/sqrt(n_bins), so in the squared-distance sum each family's
            # contribution is multiplied by 1/n_bins. That matches the
            # "family-weighted Hellinger^2" we target here.
            n_bins_f = cols.size
            for i in range(n):
                for j in range(i + 1, n):
                    d2_ij = float(np.sum((sq[i] - sq[j]) ** 2)) / float(n_bins_f)
                    D2_expected[i, j] += d2_ij
                    D2_expected[j, i] = D2_expected[i, j]
        D_expected = np.sqrt(D2_expected)

        # Actual: apply the pipeline, compute pairwise euclidean in PC space.
        X_pca, _scales, _fams = self._apply_pipeline_hellinger(X_raw, colnames)
        D_actual = self._pairwise_euclidean(X_pca)

        # Relative Frobenius error.
        rel = (np.linalg.norm(D_actual - D_expected)
               / max(np.linalg.norm(D_expected), 1e-12))
        self.assertLess(
            rel, 0.05,
            f"Pipeline does not preserve family-weighted Hellinger: "
            f"relative Frobenius error = {rel:.4f} > 0.05"
        )

    def test_low_support_filter_preserves_structure(self):
        """
        Dropping bins that are populated by <3 variants or <10% of the
        fit-set (whichever is larger) should barely change pairwise
        distances on reasonably well-populated synthetic data.

        If this starts failing, it means the filter is dropping real
        signal - check the min_support threshold.
        """
        X_raw, _colnames = self._make_pmf_matrix(n_variants=20, n_bins=60, seed=3)
        sq = np.sqrt(X_raw)
        D_before = self._pairwise_euclidean(sq - sq.mean(axis=0))

        min_support = max(3, int(0.10 * sq.shape[0]))
        keep = np.sum(sq > 1e-6, axis=0) >= min_support
        sq_filt = sq[:, keep]
        D_after = self._pairwise_euclidean(sq_filt - sq_filt.mean(axis=0))

        rel = (np.linalg.norm(D_after - D_before)
               / max(np.linalg.norm(D_before), 1e-12))
        # Gaussian-ish PMFs on a grid from -3 to 3 with mu ~ N(0, 0.8) will
        # have most bins well-supported across 20 variants, so the filter
        # should drop only deep tails.
        self.assertLess(
            rel, 0.05,
            f"Low-support filter dropped too much signal: rel error {rel:.4f}"
        )


# ---------------------------------------------------------------------------

class TestPmfInput(unittest.TestCase):
    """
    Tests for data.pmf_input against a synthetic long-format DataFrame
    matching the BATCH_ANA summary table schema.
    """

    @classmethod
    def setUpClass(cls):
        import importlib.util as u
        pkg_spec = u.spec_from_file_location(
            "ptd_data2",
            os.path.join(PKG_ROOT, "data", "__init__.py"),
            submodule_search_locations=[os.path.join(PKG_ROOT, "data")],
        )
        pkg = u.module_from_spec(pkg_spec)
        sys.modules["ptd_data2"] = pkg

        io_spec = u.spec_from_file_location(
            "ptd_data2.io", os.path.join(PKG_ROOT, "data", "io.py")
        )
        io_mod = u.module_from_spec(io_spec)
        sys.modules["ptd_data2.io"] = io_mod
        io_spec.loader.exec_module(io_mod)

        pi_spec = u.spec_from_file_location(
            "ptd_data2.pmf_input", os.path.join(PKG_ROOT, "data", "pmf_input.py")
        )
        cls.pi = u.module_from_spec(pi_spec)
        sys.modules["ptd_data2.pmf_input"] = cls.pi
        pi_spec.loader.exec_module(cls.pi)

        pkg_spec.loader.exec_module(pkg)

    def _make_summary_df(self, n_variants=10, observables=("rg", "phi"),
                         nan_frac_secondary=0.0):
        import pandas as pd
        rng = np.random.default_rng(0)
        rows = []
        n_total = n_variants * len(observables)
        nan_every = max(1, int(1.0 / nan_frac_secondary)) if nan_frac_secondary > 0 else None
        i_row = 0
        for v in [f"var_{i:03d}" for i in range(n_variants)]:
            for obs in observables:
                # Simulate absent secondary basin as NaN for some rows
                use_nan = nan_every is not None and (i_row % nan_every == 0)
                rows.append({
                    "variant": v,
                    "metric": obs,
                    "effective_support_frac": float(rng.uniform(0.3, 0.9)),
                    "n_basins_persist_2kT": float(rng.integers(1, 4)),
                    "global_basin_population": float(rng.uniform(0.4, 1.0)),
                    "global_basin_escape_barrier_kT": float(rng.uniform(0.5, 10.0)),
                    "max_secondary_persistence_kT": np.nan if use_nan else float(rng.uniform(0.0, 5.0)),
                    "basin_pop_entropy_norm": 0.0 if use_nan else float(rng.uniform(0.0, 1.0)),
                    "local_ruggedness_kT": float(rng.uniform(0.1, 3.0)),
                    "global_basin_width_1kT": float(rng.uniform(0.1, 2.0)),
                    "global_basin_min_x": float(rng.uniform(-3.0, 3.0)),
                    "global_basin_left_boundary_x": float(rng.uniform(-5.0, 0.0)),
                    "global_basin_right_boundary_x": float(rng.uniform(0.0, 5.0)),
                    "secondary_basin_min_x": np.nan if use_nan else float(rng.uniform(-3.0, 3.0)),
                    "x_eqpop": float(rng.uniform(-1.0, 1.0)),
                    "mean": float(rng.normal()),
                    "std": float(rng.uniform(0.1, 1.0)),
                })
                i_row += 1
        return pd.DataFrame(rows)

    # --- Core correctness ---

    def test_intrinsic_preset_excludes_physical_annotations(self):
        pi = self.pi
        df = self._make_summary_df()
        payload = pi.build_tab_input(df, preset="pmf_intrinsic")
        phys_cols = set(pi.PMF_PHYSICAL_ANNOTATIONS)
        for feat in payload["features"]:
            self.assertNotIn(feat, phys_cols,
                             f"Physical annotation '{feat}' leaked into analysis features.")

    def test_global_basin_width_1kT_not_in_core_features(self):
        """global_basin_width_1kT is coordinate-bearing (nm/deg) — must not be in PMF_CORE_FEATURES."""
        pi = self.pi
        self.assertNotIn("global_basin_width_1kT", pi.PMF_CORE_FEATURES)
        self.assertIn("global_basin_width_1kT", pi.PMF_PHYSICAL_ANNOTATIONS)
        self.assertEqual(pi.FEATURE_ROLES.get("global_basin_width_1kT"), "physical_annotation")

    def test_pmf_annotations_preset_returns_data_by_default(self):
        """pmf_annotations preset should return features even without allow_physical_coordinates."""
        pi = self.pi
        df = self._make_summary_df(observables=("rg",))
        payload = pi.build_tab_input(df, preset="pmf_annotations")
        self.assertGreater(len(payload["features"]), 0,
                           "pmf_annotations preset returned empty — exclusion bypass broken.")

    def test_pmf_annotations_multi_observable_warns(self):
        """pmf_annotations across multiple observables should warn about coordinate mixing."""
        pi = self.pi
        df = self._make_summary_df(observables=("rg", "phi"))
        payload = pi.build_tab_input(df, preset="pmf_annotations")
        # Warning text uses "multiple observables or coordinate units"
        coord_warns = [w for w in payload["warnings"]
                       if "multiple observables" in w or "native units" in w]
        self.assertTrue(len(coord_warns) > 0,
                        "Expected coordinate-mixing warning for multi-observable annotations.")

    def test_x_raw_and_x_same_shape_and_nonzero(self):
        pi = self.pi
        df = self._make_summary_df()
        payload = pi.build_tab_input(df, preset="pmf_intrinsic", transformed=True)
        self.assertEqual(payload["X_raw"].shape, payload["X"].shape)
        self.assertGreater(payload["X"].shape[1], 0)

    def test_annotations_not_in_features(self):
        pi = self.pi
        df = self._make_summary_df()
        payload = pi.build_tab_input(df, preset="pmf_intrinsic", include_annotations=True)
        overlap = set(payload["annotations"].columns) & set(payload["features"])
        self.assertEqual(overlap, set(), f"Annotation/feature overlap: {overlap}")

    def test_row_ids_preserved(self):
        pi = self.pi
        df = self._make_summary_df(n_variants=5)
        payload = pi.build_tab_input(df, preset="pmf_intrinsic")
        self.assertIn("variant", payload["row_ids"].columns)
        self.assertIn("metric", payload["row_ids"].columns)

    def test_row_ids_length_matches_x_raw(self):
        pi = self.pi
        df = self._make_summary_df()
        payload = pi.build_tab_input(df, preset="pmf_intrinsic")
        self.assertEqual(len(payload["row_ids"]), len(payload["X_raw"]))

    def test_missing_columns_produce_warnings(self):
        pi = self.pi
        df = self._make_summary_df()
        payload = pi.build_tab_input(df, preset="pmf_reference", strict_validity=False)
        self.assertTrue(len(payload["warnings"]) > 0)

    def test_transform_does_not_mutate_x_raw(self):
        pi = self.pi
        df = self._make_summary_df()
        payload = pi.build_tab_input(df, preset="pmf_intrinsic", transformed=True)
        not_identical = not payload["X_raw"].equals(payload["X"])
        self.assertTrue(not_identical, "X must differ from X_raw after z-score.")

    def test_empty_df_returns_empty_payload(self):
        pi = self.pi
        import pandas as pd
        payload = pi.build_tab_input(pd.DataFrame(), preset="pmf_intrinsic")
        self.assertEqual(payload["features"], [])
        self.assertTrue(payload["X_raw"].empty)
        self.assertTrue(len(payload["warnings"]) > 0)

    def test_feature_roles_keys_match_features(self):
        pi = self.pi
        df = self._make_summary_df()
        payload = pi.build_tab_input(df, preset="pmf_intrinsic")
        self.assertEqual(set(payload["feature_roles"].keys()), set(payload["features"]))
        valid_roles = {"analysis", "reference_analysis", "physical_annotation", "unknown"}
        for role in payload["feature_roles"].values():
            self.assertIn(role, valid_roles)

    # --- NaN handling ---

    def test_nan_preserved_in_x_raw(self):
        """NaN rows (absent secondary basin) must stay NaN in X_raw, not become 0."""
        pi = self.pi
        import pandas as pd
        df = self._make_summary_df(n_variants=10, observables=("rg",), nan_frac_secondary=0.5)
        payload = pi.build_tab_input(df, preset="pmf_intrinsic")
        col = "max_secondary_persistence_kT"
        if col in payload["X_raw"].columns:
            raw_nan = payload["X_raw"][col].isna().sum()
            self.assertGreater(raw_nan, 0, "NaN rows disappeared from X_raw.")

    def test_nan_preserved_in_x_transformed(self):
        """NaN must stay NaN after log1p_zscore — absent barriers must not become 0."""
        pi = self.pi
        import pandas as pd
        df = self._make_summary_df(n_variants=10, observables=("rg",), nan_frac_secondary=0.5)
        payload = pi.build_tab_input(df, preset="pmf_intrinsic", transformed=True)
        col = "max_secondary_persistence_kT"
        if col in payload["X"].columns:
            raw_nan_idx = payload["X_raw"][col].isna()
            transformed_nan_idx = payload["X"][col].isna()
            self.assertTrue(
                (raw_nan_idx == transformed_nan_idx).all(),
                "NaN positions changed after transform — absent barriers converted to 0."
            )

    def test_zscore_safe_preserves_nan_in_zero_sd_case(self):
        """When all non-NaN values are equal (sd=0), NaN must not become 0."""
        pi = self.pi
        s = np.array([1.0, np.nan, 1.0, 1.0, np.nan])
        result = pi._zscore_safe(s)
        self.assertTrue(np.isnan(result[1]), "NaN at index 1 became non-NaN after zero-sd zscore.")
        self.assertTrue(np.isnan(result[4]), "NaN at index 4 became non-NaN after zero-sd zscore.")
        self.assertEqual(result[0], 0.0)

    def test_high_nan_feature_generates_warning(self):
        """A feature that is >50% NaN should produce a warning."""
        pi = self.pi
        import pandas as pd
        df = self._make_summary_df(n_variants=10, observables=("rg",), nan_frac_secondary=1.0)
        payload = pi.build_tab_input(df, preset="pmf_intrinsic")
        nan_warns = [w for w in payload["warnings"] if "NaN" in w or "nan" in w.lower()]
        self.assertTrue(len(nan_warns) > 0,
                        "Expected high-NaN warning; got: " + str(payload["warnings"]))

    # --- Single-observable physical coordinate logic ---

    def test_allow_physical_coords_single_observable_no_warn(self):
        pi = self.pi
        df = self._make_summary_df(observables=("rg",))
        payload = pi.build_tab_input(df, preset="pmf_intrinsic",
                                     allow_physical_coordinates=True)
        self.assertTrue(pi.can_include_physical_coordinates(df))
        coord_warns = [w for w in payload["warnings"] if "multiple observables" in w]
        self.assertEqual(coord_warns, [])

    # --- Reference guard actually filters rows ---

    def test_reference_guard_drops_invalid_rows(self):
        """Rows with pmf_jsd_reference_valid=False must be absent from X_raw."""
        pi = self.pi
        import pandas as pd
        df = self._make_summary_df(n_variants=6, observables=("rg",))
        df["jsd_to_reference_norm"] = 0.1
        df["pmf_jsd_reference_valid"] = True
        df["pmf_support_matches_reference"] = True
        # Mark 2 rows as invalid
        df.loc[df.index[[0, 3]], "pmf_jsd_reference_valid"] = False
        payload = pi.build_tab_input(df, preset="pmf_reference", strict_validity=False)
        drop_warns = [w for w in payload["warnings"] if "Removed" in w and "reference" in w.lower()]
        self.assertTrue(len(drop_warns) > 0, "Reference-invalid rows not flagged as removed.")
        self.assertEqual(len(payload["X_raw"]), 4,
                         "Expected 4 rows after dropping 2 invalid reference rows.")

    # --- FEATURE_PRESETS mutation safety ---

    def test_feature_presets_are_copies(self):
        """Mutating a preset list must not affect the source constant."""
        pi = self.pi
        original_len = len(pi.PMF_CORE_FEATURES)
        pi.FEATURE_PRESETS["pmf_intrinsic"].append("__test_sentinel__")
        self.assertEqual(len(pi.PMF_CORE_FEATURES), original_len,
                         "Mutating FEATURE_PRESETS[pmf_intrinsic] mutated PMF_CORE_FEATURES.")
        # Clean up
        pi.FEATURE_PRESETS["pmf_intrinsic"] = list(pi.PMF_CORE_FEATURES)

    # --- Fix 3: all-NaN reference features auto-dropped when guard columns absent ---

    def test_reference_features_dropped_when_all_nan_and_no_guard_columns(self):
        """Reference features that are all-NaN and have no guard columns must be
        removed from the feature list so they don't appear as dead PCA dimensions."""
        pi = self.pi
        import pandas as pd
        df = self._make_summary_df(n_variants=8, observables=("rg",))
        # Reference features all-NaN; guard columns absent (simulates current BATCH_ANA)
        df["jsd_to_reference_norm"] = np.nan
        df["harmonic_hellinger_to_reference"] = np.nan
        payload = pi.build_tab_input(df, preset="pmf_reference")
        self.assertNotIn("jsd_to_reference_norm", payload["features"],
                         "All-NaN reference feature must be auto-dropped from features list.")
        self.assertNotIn("harmonic_hellinger_to_reference", payload["features"])
        # A warning about removal must be present
        drop_warns = [w for w in payload["warnings"] if "all-NaN reference features" in w]
        self.assertTrue(len(drop_warns) > 0, "Expected warning about all-NaN reference feature removal.")

    def test_reference_features_kept_when_guard_absent_but_values_present(self):
        """If guard columns are absent but reference values are present,
        features stay (guard-absent case only removes if all-NaN)."""
        pi = self.pi
        import pandas as pd
        df = self._make_summary_df(n_variants=8, observables=("rg",))
        # Reference features with real values; guard columns absent
        df["jsd_to_reference_norm"] = 0.05
        payload = pi.build_tab_input(df, preset="pmf_reference")
        self.assertIn("jsd_to_reference_norm", payload["features"],
                      "Reference feature with real values must be retained even without guard columns.")

    # --- Fix 6: scaler_scope per_observable ---

    def test_per_observable_scope_produces_different_result_from_global(self):
        """Z-scoring per observable should differ from global when observables have
        different marginal distributions."""
        pi = self.pi
        import pandas as pd
        rng = np.random.default_rng(42)
        rows = []
        # Two observables with very different means: rg~1-2, psi~100-200
        for v in [f"v{i}" for i in range(20)]:
            rows.append({"variant": v, "metric": "rg",
                         "effective_support_frac": float(rng.uniform(0.3, 0.9)),
                         "n_basins_persist_2kT": 1.0,
                         "global_basin_population": float(rng.uniform(0.4, 1.0)),
                         "global_basin_escape_barrier_kT": float(rng.uniform(1.0, 2.0)),
                         "max_secondary_persistence_kT": float(rng.uniform(0.5, 1.5)),
                         "basin_pop_entropy_norm": float(rng.uniform(0.0, 0.5)),
                         "local_ruggedness_kT": float(rng.uniform(0.1, 1.0))})
            rows.append({"variant": v, "metric": "psi",
                         "effective_support_frac": float(rng.uniform(0.3, 0.9)),
                         "n_basins_persist_2kT": 1.0,
                         "global_basin_population": float(rng.uniform(0.4, 1.0)),
                         "global_basin_escape_barrier_kT": float(rng.uniform(100.0, 200.0)),
                         "max_secondary_persistence_kT": float(rng.uniform(50.0, 150.0)),
                         "basin_pop_entropy_norm": float(rng.uniform(0.0, 0.5)),
                         "local_ruggedness_kT": float(rng.uniform(10.0, 100.0))})
        df = pd.DataFrame(rows)
        p_global = pi.build_tab_input(df, preset="pmf_intrinsic", scaler_scope="current_selection")
        p_per    = pi.build_tab_input(df, preset="pmf_intrinsic", scaler_scope="per_observable")
        # Results must differ on the kT columns (scale differs 100× between observables)
        self.assertFalse(
            np.allclose(
                p_global["X"]["global_basin_escape_barrier_kT"].to_numpy(),
                p_per["X"]["global_basin_escape_barrier_kT"].to_numpy(),
                equal_nan=True,
            ),
            "per_observable and global scaler_scope should differ when observables have "
            "very different scale.",
        )

    def test_per_observable_scope_zero_mean_within_group(self):
        """Per-observable z-scored features should have near-zero mean within each group."""
        pi = self.pi
        import pandas as pd
        df = self._make_summary_df(n_variants=20, observables=("rg", "psi"))
        payload = pi.build_tab_input(df, preset="pmf_intrinsic", scaler_scope="per_observable")
        X = payload["X"]
        meta_col = payload["row_ids"]["metric"] if "metric" in payload["row_ids"].columns else None
        if meta_col is None:
            self.skipTest("metric column not in row_ids")
        for grp in meta_col.unique():
            mask = (meta_col == grp).to_numpy()
            for col in ["effective_support_frac", "global_basin_population"]:
                vals = X.loc[mask, col].dropna().to_numpy()
                if len(vals) > 1:
                    self.assertAlmostEqual(float(np.mean(vals)), 0.0, places=10,
                                           msg=f"Per-observable z-score mean not zero for {col}/{grp}")

    def test_unknown_scaler_scope_warns_and_falls_back(self):
        """An unrecognised scaler_scope should warn and fall back to current_selection."""
        import warnings
        pi = self.pi
        df = self._make_summary_df()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            payload = pi.build_tab_input(df, preset="pmf_intrinsic", scaler_scope="bogus_scope")
        scope_warns = [str(w.message) for w in caught if "scaler_scope" in str(w.message).lower()
                       or "bogus_scope" in str(w.message)]
        self.assertTrue(len(scope_warns) > 0, "Expected warning for unknown scaler_scope.")
        self.assertEqual(payload["transform_metadata"]["scaler_scope"], "current_selection")

    # --- Fix 7: sparse secondary-basin features don't trigger the NaN warning ---

    def test_sparse_secondary_feature_no_warn_below_90pct_nan(self):
        """max_secondary_persistence_kT at 60% NaN must not warn (expected sparse)."""
        pi = self.pi
        import pandas as pd
        # nan_frac_secondary=1.0 means every other row is NaN → ~50% NaN.
        # We need ~60%; use a larger dataset where secondary is NaN in 6/10 rows.
        rng = np.random.default_rng(7)
        rows = []
        for i in range(10):
            nan_sec = i < 6  # 6/10 = 60% NaN
            rows.append({
                "variant": f"v{i}", "metric": "rg",
                "effective_support_frac": 0.5,
                "n_basins_persist_2kT": 1.0,
                "global_basin_population": 0.8,
                "global_basin_escape_barrier_kT": 2.0,
                "max_secondary_persistence_kT": np.nan if nan_sec else 1.0,
                "basin_pop_entropy_norm": 0.3,
                "local_ruggedness_kT": 0.5,
            })
        df = pd.DataFrame(rows)
        payload = pi.build_tab_input(df, preset="pmf_intrinsic")
        secondary_warns = [
            w for w in payload["warnings"]
            if "max_secondary_persistence_kT" in w and "NaN" in w
        ]
        self.assertEqual(secondary_warns, [],
                         "max_secondary_persistence_kT at 60% NaN must not warn (expected sparse).")

    def test_non_sparse_feature_still_warns_at_60pct_nan(self):
        """A non-sparse feature (effective_support_frac) at 60% NaN must still warn."""
        pi = self.pi
        import pandas as pd
        rows = []
        for i in range(10):
            rows.append({
                "variant": f"v{i}", "metric": "rg",
                "effective_support_frac": np.nan if i < 6 else 0.5,
                "n_basins_persist_2kT": 1.0,
                "global_basin_population": 0.8,
                "global_basin_escape_barrier_kT": 2.0,
                "max_secondary_persistence_kT": 1.0,
                "basin_pop_entropy_norm": 0.3,
                "local_ruggedness_kT": 0.5,
            })
        df = pd.DataFrame(rows)
        payload = pi.build_tab_input(df, preset="pmf_intrinsic")
        support_warns = [
            w for w in payload["warnings"]
            if "effective_support_frac" in w and "NaN" in w
        ]
        self.assertTrue(len(support_warns) > 0,
                        "effective_support_frac at 60% NaN should still warn.")

    # --- Fix 4: _is_empty_df full-frame check ---

    def test_is_empty_df_not_fooled_by_nan_in_first_5_rows(self):
        """_is_empty_df must return False when real data exists beyond row 5."""
        import importlib.util as u
        import pandas as pd
        io_spec = u.spec_from_file_location(
            "_ptd_io_fix4", os.path.join(PKG_ROOT, "data", "io.py")
        )
        io_mod = u.module_from_spec(io_spec)
        io_spec.loader.exec_module(io_mod)
        # First 5 rows all-NaN, row 6 has real data
        df = pd.DataFrame({"a": [np.nan] * 5 + [1.0], "b": [np.nan] * 5 + [2.0]})
        self.assertFalse(io_mod._is_empty_df(df),
                         "_is_empty_df incorrectly classified a DataFrame with data at row 6 as empty.")

    def test_is_empty_df_true_for_all_nan(self):
        """_is_empty_df must return True when every value is NaN."""
        import importlib.util as u
        import pandas as pd
        io_spec = u.spec_from_file_location(
            "_ptd_io_fix4b", os.path.join(PKG_ROOT, "data", "io.py")
        )
        io_mod = u.module_from_spec(io_spec)
        io_spec.loader.exec_module(io_mod)
        df = pd.DataFrame({"a": [np.nan] * 8})
        self.assertTrue(io_mod._is_empty_df(df))


class TestMergePmfAnnotations(unittest.TestCase):
    """DataContext._merge_pmf_annotations integrates annotation columns into ctx.features."""

    def _make_ctx(self, features_df, ann_df):
        """Build a minimal DataContext and inject a fake lazy loader for pmf_annotations."""
        import pandas as pd
        from unittest.mock import MagicMock
        import sys

        # Use proper package import (avoids relative-import issues with spec_from_file_location)
        project_root = os.path.dirname(PKG_ROOT)
        if project_root not in sys.path:
            sys.path.insert(0, project_root)
        from peptide_dash.data.context import DataContext

        ctx = DataContext.__new__(DataContext)
        ctx.data_dir = "/tmp"
        ctx.timeseries_dir = None
        ctx.features = features_df.copy()
        ctx.numeric_columns = []
        fake_loader = MagicMock()
        fake_loader.pmf_annotations_df = ann_df
        ctx._lazy_curves = fake_loader
        return ctx

    def test_annotation_columns_merged_into_features(self):
        import pandas as pd
        feats = pd.DataFrame({"variant": ["A", "B", "C"], "score": [1.0, 2.0, 3.0]})
        ann = pd.DataFrame({
            "variant": ["A", "B", "C"],
            "phi__n_basins": [1, 2, 1],
            "phi__effective_support_frac": [0.8, 0.9, 0.7],
        })
        ctx = self._make_ctx(feats, ann)
        ctx._merge_pmf_annotations()
        self.assertIn("phi__n_basins", ctx.features.columns)
        self.assertIn("phi__effective_support_frac", ctx.features.columns)
        self.assertEqual(len(ctx.features), 3)

    def test_existing_columns_not_overwritten(self):
        import pandas as pd
        feats = pd.DataFrame({"variant": ["A"], "phi__n_basins": [99.0]})
        ann = pd.DataFrame({"variant": ["A"], "phi__n_basins": [1.0]})
        ctx = self._make_ctx(feats, ann)
        ctx._merge_pmf_annotations()
        self.assertEqual(ctx.features["phi__n_basins"].iloc[0], 99.0)

    def test_left_join_keeps_all_variants(self):
        """Variants absent from annotations get NaN, not dropped."""
        import pandas as pd
        feats = pd.DataFrame({"variant": ["A", "B", "C"], "score": [1.0, 2.0, 3.0]})
        ann = pd.DataFrame({"variant": ["A"], "phi__eff": [0.8]})
        ctx = self._make_ctx(feats, ann)
        ctx._merge_pmf_annotations()
        self.assertEqual(len(ctx.features), 3)
        self.assertTrue(pd.isna(ctx.features.loc[ctx.features["variant"] == "B", "phi__eff"].iloc[0]))

    def test_empty_annotations_noop(self):
        import pandas as pd
        feats = pd.DataFrame({"variant": ["A"], "score": [1.0]})
        ann = pd.DataFrame()
        ctx = self._make_ctx(feats, ann)
        ctx._merge_pmf_annotations()
        self.assertEqual(list(ctx.features.columns), ["variant", "score"])

    def test_numeric_columns_extended_when_prepopulated(self):
        import pandas as pd
        feats = pd.DataFrame({"variant": ["A", "B"], "score": [1.0, 2.0]})
        ann = pd.DataFrame({"variant": ["A", "B"], "phi__eff": [0.8, 0.9]})
        ctx = self._make_ctx(feats, ann)
        ctx.numeric_columns = ["score"]
        ctx._merge_pmf_annotations()
        self.assertIn("phi__eff", ctx.numeric_columns)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    unittest.main(verbosity=2)
