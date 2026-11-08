"""
tests/test_calibration.py
=========================
Unit tests for Phase 5: Calibration metrics, Clopper-Pearson intervals,
prediction flip logic, and credibility gate evaluations.
"""

import numpy as np

from calibration.run import clopper_pearson, run_single_audit_seed
from adcd.claim_card import ClaimCard, NoiseSpec
from adcd.asimov_designer import design, get_sphere_drag_models


def test_clopper_pearson_exact():
    """Verify exact behavior of Clopper-Pearson binomial confidence intervals."""
    # When k=0, lower bound must be exactly 0.0
    low_0, high_0 = clopper_pearson(0, 460, alpha=0.05)
    assert low_0 == 0.0
    assert 0.007 < high_0 < 0.009  # ~0.798%

    # When k=n, upper bound must be exactly 1.0
    low_n, high_n = clopper_pearson(460, 460, alpha=0.05)
    assert high_n == 1.0
    assert 0.991 < low_n < 0.993

    # Symmetric interval when k=n/2
    low_half, high_half = clopper_pearson(230, 460, alpha=0.05)
    assert np.isclose(0.5 - low_half, high_half - 0.5, atol=1e-5)
    assert 0.45 < low_half < 0.50
    assert 0.50 < high_half < 0.55


def test_sphere_drag_asimov_prediction_flip():
    """
    Mathematical verification of the prediction flip for Sphere Drag (Re <= 0.5, N=200):
    - At sigma = 0.3% (> sigma* = 0.137%), expected Delta-BIC < 10 (EQUIVALENCE CLASS).
    - At sigma = 0.1% (< sigma* = 0.137%), expected Delta-BIC > 10 (UNIQUELY IDENTIFIABLE).
    """
    m1, m2 = get_sphere_drag_models(use_verified_series=False, rival_type="linear")
    res_03 = design(m1, m2, window=(0.01, 0.5), N=200, sigma_model=0.003)
    res_01 = design(m1, m2, window=(0.01, 0.5), N=200, sigma_model=0.001)

    assert res_03.expected_delta_bic < 10.0, "At sigma=0.3%, expected Delta-BIC must be < 10.0"
    assert res_01.expected_delta_bic > 10.0, "At sigma=0.1%, expected Delta-BIC must be > 10.0"
    assert 0.0013 < res_03.sigma_star < 0.0015, "sigma* must be approximately 0.14%"


def test_null_baseline_produces_withheld():
    """On pure Gaussian noise without physical anomaly, auditor issues WITHHELD."""
    cands = [
        "theta_0 * (exp(-r) - 1.0)",
        "theta_0 * log(1.0 + r)",
    ]
    card = ClaimCard(
        name="Null Test",
        variables={"r": "m"},
        baseline_expr="1.0 / r",
        limit_variable="r",
        limit_direction="0",
        candidate_expressions=cands,
        candidate_families={
            cands[0]: "exponential",
            cands[1]: "logarithmic",
        },
        noise_spec=NoiseSpec(type="relative", value=0.01),
        seed=42,
    )

    rng = np.random.RandomState(42)
    r = np.linspace(0.2, 4.0, 100)
    y_cl = 1.0 / r
    y_obs = y_cl * (1.0 + rng.normal(0, 0.01, size=len(r)))

    res = run_single_audit_seed(card, {"r": r, "y_obs": y_obs, "y_cl": y_cl})
    assert res["verdict"] == "WITHHELD"


def test_screened_coulomb_identifiable_low_noise():
    """At low noise (0.1%), Screened Coulomb is robustly UNIQUELY IDENTIFIABLE."""
    cands = [
        "theta_0 * (exp(-r/theta_1) - 1.0)",
        "theta_0 * (r/theta_1 / (1.0 - r/theta_1))",
    ]
    card = ClaimCard(
        name="Screened Coulomb Test",
        variables={"r": "m"},
        baseline_expr="1.0 / r",
        limit_variable="r",
        limit_direction="0",
        candidate_expressions=cands,
        candidate_families={
            cands[0]: "exponential",
            cands[1]: "rational",
        },
        noise_spec=NoiseSpec(type="relative", value=0.001),
        seed=42,
    )

    rng = np.random.RandomState(42)
    r = np.linspace(0.2, 4.0, 100)
    y_cl = 1.0 / r
    delta_true = np.exp(-r / 1.5) - 1.0
    y_obs = y_cl * (1.0 + delta_true) * (1.0 + rng.normal(0, 0.001, size=len(r)))

    res = run_single_audit_seed(card, {"r": r, "y_obs": y_obs, "y_cl": y_cl})
    assert res["verdict"] == "UNIQUELY IDENTIFIABLE"
    assert res["best_family"] == "exponential"
