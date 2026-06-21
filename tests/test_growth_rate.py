"""Tests for the fσ₈ growth-rate experiment (P3.1–P3.4).

These pin the scientific claims of the S₈-tension analysis so any regression
in the loader, GR baseline, or discovery library is caught immediately.
"""

import pytest

from adcd.experiments.growth_rate_data import (
    GrowthRateResult,
    growth_factor,
    gr_fs8_prediction,
    load_growth_rate,
    PLANCK_OM_M0,
    PLANCK_OM_DE0,
)
from adcd.experiments.growth_rate_analysis import (
    CANDIDATE_LIBRARY,
    DECISIVE_DELTA_BIC,
    SUBSET_FACTORIES,
    run_growth_rate_discovery,
    run_subset_crosscheck,
)


# ---------------------------------------------------------------------------
# P3.1 — data loader
# ---------------------------------------------------------------------------

def test_loader_returns_real_data():
    res = load_growth_rate()
    assert res.data_source == "REAL"
    assert res.n_points == 63               # Alestas+2022 Table II
    assert res.z.min() < 0.01               # includes 6dFGRS very-low-z point
    assert res.z.max() > 1.9                # extends to FastSound z~1.94


def test_loader_residual_is_obs_minus_gr():
    res = load_growth_rate()
    # By construction residual = fs8_obs - fs8_gr
    import numpy as np
    assert np.allclose(res.residual, res.fs8_obs - res.fs8_gr)


# ---------------------------------------------------------------------------
# P3.2 — GR baseline physics
# ---------------------------------------------------------------------------

def test_growth_factor_normalised_to_one_at_z0():
    D0 = growth_factor(0.0)
    assert abs(D0 - 1.0) < 1e-6


def test_growth_factor_monotonic_decrease():
    import numpy as np
    zs = np.array([0.0, 0.3, 0.8, 1.5, 2.0])
    Ds = growth_factor(zs)
    assert all(Ds[i] > Ds[i + 1] for i in range(len(zs) - 1))


def test_gr_fs8_baseline_recovers_sigma8_at_z0():
    """fσ₈(z=0) = σ₈₀ · D(0) · f(0) = σ₈₀ · 1 · Ωₘ₀^γ."""
    import numpy as np
    s80 = 0.811
    pred = gr_fs8_prediction(np.array([0.0]),
                             np.array([PLANCK_OM_M0]),
                             np.array([PLANCK_OM_DE0]),
                             sigma8_0=s80)
    # f(0) = Ωₘ₀^0.55
    expected = s80 * (PLANCK_OM_M0 ** 0.55)
    assert abs(pred[0] - expected) < 1e-3


def test_gr_baseline_sigma8_recovers_s8_tension():
    """Headline: RSD data prefers σ₈₀ ≈ 0.78, ~3-4% below Planck's 0.811."""
    res = load_growth_rate()
    assert res.sigma8_0_fit < 0.80          # S₈-tension direction
    assert res.sigma8_0_fit > 0.75
    # GR baseline fits each point well once σ₈₀ is freed
    chi2_red = sum((res.residual / res.sigma_fs8) ** 2) / (res.n_points - 1)
    assert chi2_red < 0.6


# ---------------------------------------------------------------------------
# P3.3 — ADCD discovery on Δ(z)
# ---------------------------------------------------------------------------

def test_candidate_library_includes_constant_null():
    names = [c[0] for c in CANDIDATE_LIBRARY]
    assert "Constant offset" in names         # the all-important null model


def test_candidate_library_spans_all_structural_families():
    """Library must cover polynomial, power_law, exponential, rational, log."""
    formulas = " ".join(c[1] for c in CANDIDATE_LIBRARY)
    assert "z**" in formulas                   # power-law family
    assert "exp" in formulas                   # exponential family
    assert "/" in formulas and "**2" in formulas  # rational family
    assert "log" in formulas                   # log family


def test_discovery_verdict_is_constant_wins():
    """P3.3 headline: no z-dependent correction beats a constant offset."""
    summary = run_growth_rate_discovery()
    assert summary.verdict == "CONSTANT_WINS"
    assert summary.best_candidate == "Constant offset"
    # ΔBIC vs null must be ~0 (constant IS the null)
    assert abs(summary.delta_bic_vs_null) < 1e-6


def test_bin_means_show_no_systematic_redshift_trend():
    """Δ(z) bin pulls must all be |pull| < 1σ — no functional signal."""
    summary = run_growth_rate_discovery()
    for b in summary.bin_means:
        assert abs(b["pull"]) < 1.0


# ---------------------------------------------------------------------------
# P3.4 — homogeneous subset cross-check
# ---------------------------------------------------------------------------

def test_three_subsets_defined():
    assert set(SUBSET_FACTORIES.keys()) == {
        "BOSS DR12 (Alam+2017)",
        "Precision (top-20)",
        "Mid-z window [0.35,0.75]",
    }


def test_subsets_all_return_constant_wins():
    """P3.4 headline: CONSTANT_WINS verdict is robust on all subsets."""
    res = load_growth_rate()
    summaries = run_subset_crosscheck(res=res)
    assert len(summaries) == 3
    for s in summaries:
        assert s.verdict == "CONSTANT_WINS", (
            f"Subset '{s.name}' surprisingly shows {s.verdict} "
            f"(ΔBIC vs null = {s.delta_bic_vs_null:+.2f})"
        )
        assert s.delta_bic_vs_null > -DECISIVE_DELTA_BIC


def test_boss_dr12_subset_has_six_points():
    """Verify the BOSS DR12 (Alam+2017) signature: 6 points at z∈{0.38,0.51,0.61}."""
    import numpy as np
    res = load_growth_rate()
    factory = SUBSET_FACTORIES["BOSS DR12 (Alam+2017)"]
    mask = factory(res.z, res.fs8_obs, res.sigma_fs8, res.Om_fid)
    assert mask.sum() == 6
    zs_in = set(np.round(res.z[mask], 2))
    assert zs_in == {0.38, 0.51, 0.61}


# ---------------------------------------------------------------------------
# P3.5 — Cosmic Chronometers H(z) secondary validation
# ---------------------------------------------------------------------------

def test_cosmic_chronometers_verdict_is_constant_wins():
    """P3.5 headline: independent observable also shows CONSTANT_WINS."""
    from adcd.experiments.growth_rate_analysis import run_cosmic_chronometers
    summary = run_cosmic_chronometers()
    assert summary.verdict == "CONSTANT_WINS"
    assert summary.n_points >= 30                    # canonical 31–34 point compilation
    # ΛCDM baseline parameters must land in physically reasonable ranges
    assert 60.0 < summary.H0_fit < 80.0
    assert 0.20 < summary.Om_m0_fit < 0.40
    assert summary.chi2_reduced < 1.0                 # baseline fits the data


def test_cosmic_chronometers_baseline_is_physical():
    """ΛCDM H(z) baseline must be monotonic increasing — sanity check."""
    import numpy as np
    from adcd.experiments.growth_rate_analysis import _H_gr
    zs = np.array([0.0, 0.5, 1.0, 2.0])
    Hs = _H_gr(zs, H0=70.0, Om_m0=0.3)
    assert all(Hs[i] < Hs[i + 1] for i in range(len(zs) - 1))
    # H(0) = H₀ exactly
    assert abs(Hs[0] - 70.0) < 1e-9
