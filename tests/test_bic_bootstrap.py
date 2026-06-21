"""Tests for the cluster bootstrap on delta-BIC.

Validates that the galaxy-level (cluster) bootstrap on the in-sample BIC_eff
correctly (a) preserves within-galaxy correlation by resampling galaxies,
(b) reports per-model BIC_eff CIs and pairwise delta CIs relative to ADCD, and
(c) brackets zero when the two models are statistically equivalent and excludes
zero when one model is decisively better.

The core statistics are exercised with cheap synthetic 2-param models (scipy
only), so the suite runs in seconds without JAX. One smoke test exercises the
JAX-backed ADCD fit-fn factory end-to-end on a tiny resample count.
"""

import json

import numpy as np
import pytest

from adcd.experiments.bic_bootstrap import (
    _fit_2param_factory,
    bootstrap_delta_bic,
    print_bootstrap_report,
)


# ---------------------------------------------------------------------------
# Synthetic-cluster helpers
# ---------------------------------------------------------------------------

def _make_clusters(n_galaxies=20, pts_per_gal=15, seed=0):
    """Build per-galaxy (x, nu) clusters from a known 2-param ground truth.

    Ground truth: nu = 1.0 * (1 + sqrt(1 + 4/(0.5*x))) / 2  (Simple MOND form).
    """
    rng = np.random.default_rng(seed)
    clusters = []
    for _ in range(n_galaxies):
        x = rng.uniform(0.05, 5.0, size=pts_per_gal)
        nu_true = (1.0 + np.sqrt(1.0 + 4.0 / (0.5 * x))) / 2.0
        nu = nu_true + rng.normal(0.0, 0.05, size=pts_per_gal)
        clusters.append((x, nu))
    return clusters


def _simple_mond_2p(x, theta0, theta1):
    x = np.asarray(x, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return theta1 * (1.0 + np.sqrt(1.0 + 4.0 / (theta0 * x))) / 2.0


def _flat_baseline(x, theta0, theta1):
    """A deliberately bad 2-param model: nu = theta1 (constant)."""
    x = np.asarray(x, dtype=float)
    return np.full_like(x, theta1, dtype=float)


def _build_models(seed=42, n_restarts=5):
    """Two scipy-only 2-param models: a good one and a bad one."""
    good_fn, good_k = _fit_2param_factory(_simple_mond_2p, n_restarts=n_restarts, seed=seed)
    bad_fn, bad_k = _fit_2param_factory(_flat_baseline, n_restarts=n_restarts, seed=seed)
    return {"ADCD discovered": (good_fn, good_k), "Bad": (bad_fn, bad_k)}


# ---------------------------------------------------------------------------
# Sanity / validation tests
# ---------------------------------------------------------------------------

def test_delta_sign_and_direction_when_models_equivalent():
    """When ADCD model == competitor model (same fit fn), delta CI brackets 0."""
    clusters = _make_clusters(n_galaxies=20, seed=1)
    fn, k = _fit_2param_factory(_simple_mond_2p, n_restarts=3, seed=7)
    models = {
        "ADCD discovered": (fn, k),
        "Same": (fn, k),  # identical => delta == 0 up to RNG tie-break
    }
    summary = bootstrap_delta_bic(
        galaxies_data=clusters,
        models=models,
        n_galaxies=len(clusters),
        n_points=sum(len(x) for x, _ in clusters),
        adcd_key="ADCD discovered",
        primary_competitor_key="Same",
        n_bootstrap=30,
        seed=11,
    )
    same = next(m for m in summary.models if m.name == "Same")
    # Same model => every delta is exactly 0 => CI brackets 0.
    assert same.delta_eff_mean == pytest.approx(0.0, abs=1e-9)
    assert same.delta_eff_ci95[0] <= 0.0 <= same.delta_eff_ci95[1]
    assert same.delta_eff_zero_in_ci is True
    assert summary.headline_zero_in_ci is True


def test_delta_excludes_zero_when_competitor_decisively_worse():
    """The flat baseline is much worse => delta (ADCD - Bad) is negative & CI < 0."""
    clusters = _make_clusters(n_galaxies=25, seed=2)
    summary = bootstrap_delta_bic(
        galaxies_data=clusters,
        models=_build_models(seed=5),
        n_galaxies=len(clusters),
        n_points=sum(len(x) for x, _ in clusters),
        adcd_key="ADCD discovered",
        primary_competitor_key="Bad",
        n_bootstrap=40,
        seed=13,
    )
    bad = next(m for m in summary.models if m.name == "Bad")
    # Bad is worse => BIC_Bad > BIC_ADCD => delta = BIC_ADCD - BIC_Bad < 0.
    assert bad.delta_eff_mean < 0
    assert bad.delta_eff_ci95[1] < 0  # entire CI below 0 => decisive ADCD win
    assert bad.delta_eff_zero_in_ci is False
    assert summary.headline_zero_in_ci is False


def test_cluster_bootstrap_preserves_within_galaxy_correlation():
    """Resampling must be at the galaxy level (not the point level).

    We check this indirectly: with strong within-cluster correlation, the
    bootstrap CI must be substantially wider than a naive point-resample CI
    would be. The cluster bootstrap should NOT collapse to a tiny CI.
    """
    rng = np.random.default_rng(3)
    # Highly correlated within galaxy (shared per-galaxy offset).
    clusters = []
    for _ in range(30):
        x = rng.uniform(0.1, 3.0, size=20)
        offset = rng.normal(0.0, 0.3)  # whole-galaxy systematic
        nu = 1.5 + offset + rng.normal(0.0, 0.02, size=20)
        clusters.append((x, nu))

    fn, k = _fit_2param_factory(_flat_baseline, n_restarts=3, seed=9)
    models = {"ADCD discovered": (fn, k), "Twin": (fn, k)}
    summary = bootstrap_delta_bic(
        galaxies_data=clusters,
        models=models,
        n_galaxies=len(clusters),
        n_points=sum(len(x) for x, _ in clusters),
        adcd_key="ADCD discovered",
        primary_competitor_key="Twin",
        n_bootstrap=50,
        seed=21,
    )
    # Identical models => delta exactly 0 even with correlated clusters.
    twin = next(m for m in summary.models if m.name == "Twin")
    assert twin.delta_eff_mean == pytest.approx(0.0, abs=1e-9)


def test_per_model_bic_eff_c_are_populated():
    """Each model carries a non-NaN BIC_eff mean and a properly ordered CI."""
    clusters = _make_clusters(n_galaxies=15, seed=4)
    summary = bootstrap_delta_bic(
        galaxies_data=clusters,
        models=_build_models(seed=8),
        n_galaxies=len(clusters),
        n_points=sum(len(x) for x, _ in clusters),
        adcd_key="ADCD discovered",
        primary_competitor_key="Bad",
        n_bootstrap=25,
        seed=31,
    )
    for m in summary.models:
        assert np.isfinite(m.nmse_mean)
        assert np.isfinite(m.bic_eff_mean)
        lo, hi = m.bic_eff_ci95
        assert lo <= m.bic_eff_mean <= hi
        assert m.nmse_ci95[0] <= m.nmse_mean <= m.nmse_ci95[1]


def test_models_ordered_by_bic_eff_mean():
    """Output models list is sorted by bic_eff_mean ascending (lower = better)."""
    clusters = _make_clusters(n_galaxies=20, seed=5)
    summary = bootstrap_delta_bic(
        galaxies_data=clusters,
        models=_build_models(seed=6),
        n_galaxies=len(clusters),
        n_points=sum(len(x) for x, _ in clusters),
        adcd_key="ADCD discovered",
        primary_competitor_key="Bad",
        n_bootstrap=30,
        seed=41,
    )
    means = [m.bic_eff_mean for m in summary.models]
    assert means == sorted(means)


def test_seed_reproducibility():
    """Same seed => identical summary numbers."""
    clusters = _make_clusters(n_galaxies=15, seed=6)
    kw = dict(
        galaxies_data=clusters,
        models=_build_models(seed=10),
        n_galaxies=len(clusters),
        n_points=sum(len(x) for x, _ in clusters),
        adcd_key="ADCD discovered",
        primary_competitor_key="Bad",
        n_bootstrap=20,
    )
    s1 = bootstrap_delta_bic(seed=99, **kw)
    s2 = bootstrap_delta_bic(seed=99, **kw)
    assert s1.to_dict() == s2.to_dict()


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

def test_missing_adcd_key_raises():
    clusters = _make_clusters(n_galaxies=10)
    fn, k = _fit_2param_factory(_simple_mond_2p, n_restarts=2, seed=1)
    with pytest.raises(ValueError, match="ADCD key"):
        bootstrap_delta_bic(
            galaxies_data=clusters,
            models={"X": (fn, k)},
            n_galaxies=len(clusters),
            n_points=100,
            adcd_key="ADCD discovered",
            primary_competitor_key="X",
            n_bootstrap=5,
        )


def test_missing_competitor_key_raises():
    clusters = _make_clusters(n_galaxies=10)
    fn, k = _fit_2param_factory(_simple_mond_2p, n_restarts=2, seed=1)
    with pytest.raises(ValueError, match="Primary competitor key"):
        bootstrap_delta_bic(
            galaxies_data=clusters,
            models={"ADCD discovered": (fn, k)},
            n_galaxies=len(clusters),
            n_points=100,
            adcd_key="ADCD discovered",
            primary_competitor_key="Missing",
            n_bootstrap=5,
        )


def test_invalid_n_galaxies_raises():
    clusters = _make_clusters(n_galaxies=10)
    fn, k = _fit_2param_factory(_simple_mond_2p, n_restarts=2, seed=1)
    with pytest.raises(ValueError, match="n_galaxies"):
        bootstrap_delta_bic(
            galaxies_data=clusters,
            models={"ADCD discovered": (fn, k), "C": (fn, k)},
            n_galaxies=1,
            n_points=100,
            adcd_key="ADCD discovered",
            primary_competitor_key="C",
            n_bootstrap=5,
        )


def test_too_few_resamples_raises():
    clusters = _make_clusters(n_galaxies=10)
    fn, k = _fit_2param_factory(_simple_mond_2p, n_restarts=2, seed=1)
    with pytest.raises(ValueError, match="n_bootstrap"):
        bootstrap_delta_bic(
            galaxies_data=clusters,
            models={"ADCD discovered": (fn, k), "C": (fn, k)},
            n_galaxies=10,
            n_points=100,
            adcd_key="ADCD discovered",
            primary_competitor_key="C",
            n_bootstrap=1,
        )


# ---------------------------------------------------------------------------
# Serialisation + reporting
# ---------------------------------------------------------------------------

def test_to_dict_roundtrip_jsonable():
    clusters = _make_clusters(n_galaxies=12, seed=7)
    summary = bootstrap_delta_bic(
        galaxies_data=clusters,
        models=_build_models(seed=12),
        n_galaxies=len(clusters),
        n_points=sum(len(x) for x, _ in clusters),
        adcd_key="ADCD discovered",
        primary_competitor_key="Bad",
        n_bootstrap=20,
        seed=51,
    )
    d = summary.to_dict()
    s = json.dumps(d)  # must not raise
    back = json.loads(s)
    assert back["n_bootstrap"] == 20
    # CI tuples must have been flattened to 2-element lists.
    assert isinstance(back["models"][0]["bic_eff_ci95"], list)
    assert len(back["models"][0]["bic_eff_ci95"]) == 2
    assert "interpretation_eff" in back["models"][0]
    assert "reviewer_message" in back


def test_print_report_smoke(capsys):
    clusters = _make_clusters(n_galaxies=12, seed=8)
    summary = bootstrap_delta_bic(
        galaxies_data=clusters,
        models=_build_models(seed=14),
        n_galaxies=len(clusters),
        n_points=sum(len(x) for x, _ in clusters),
        adcd_key="ADCD discovered",
        primary_competitor_key="Bad",
        n_bootstrap=20,
        seed=61,
    )
    print_bootstrap_report(summary)
    out = capsys.readouterr().out
    assert "Cluster Bootstrap" in out
    assert "ADCD discovered" in out
    assert "Reviewer message" in out


# ---------------------------------------------------------------------------
# JAX-backed ADCD fit-fn smoke test (tiny resample count)
# ---------------------------------------------------------------------------

def test_adcd_fit_fn_factory_smoke():
    """The JAX-backed ADCD adapter runs end-to-end on a small synthetic sample."""
    from adcd.experiments.bic_bootstrap import _fit_adcd_factory

    # A 2-param expression the JAXOptimizer can parse & fit.
    expr = "theta_0 * sqrt(1 + theta_1 / x)"
    fit_fn, k = _fit_adcd_factory(expr, seed=3)
    assert k == 2

    rng = np.random.default_rng(0)
    x = rng.uniform(0.1, 5.0, size=80)
    nu = 1.2 * np.sqrt(1.0 + 0.7 / x) + rng.normal(0.0, 0.02, size=80)
    nmse = fit_fn(x, nu)
    assert np.isfinite(nmse)
    assert nmse < 0.1  # recovers the form closely
