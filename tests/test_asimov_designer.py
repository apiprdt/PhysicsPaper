"""
test_asimov_designer.py
=======================
Unit and validation tests for the Phase 4 Asimov Design Calculator.

Verifies:
1. Analytic sigma* and chi^2 ~ 1/sigma^2 scaling.
2. N* root-finding and window extension solvers.
3. Sphere drag known-answer test: Stokes limit, sign of log term (+9/160),
   and sigma* predictions for stand-in vs verified Chester & Breach series.
4. Brute-force Monte Carlo Delta BIC agreement on two distinct physical scenarios:
   - Sphere Drag (Proudman-Pearson log vs Oseen linear rival)
   - Screened Coulomb (Yukawa exponential vs Rational rival)
5. Expectation correction (k2 - k1)(ln N - 1) calculation.
"""

import numpy as np

from adcd.asimov_designer import (
    CandidateModel,
    design,
    get_sphere_drag_models,
    monte_carlo_delta_bic,
)


def test_analytic_sigma_scaling_and_solver():
    """Verify that chi^2_min scales strictly as 1/sigma^2 and solve_sigma_star is exact."""
    # Simple linear vs quadratic model
    m1 = CandidateModel(
        name="Linear Truth",
        fn=lambda x, th: th[0] * x,
        k=1,
        theta_init=np.array([2.0]),
        theta_true=np.array([2.0]),
    )
    m2 = CandidateModel(
        name="Constant Rival",
        fn=lambda x, th: np.full_like(x, th[0]),
        k=1,
        theta_init=np.array([1.0]),
    )

    window = (1.0, 5.0)
    N = 100

    # Run at sigma = 0.05
    res_05 = design(m1, m2, window, N, sigma_model=0.05, threshold=10.0)
    # Run at sigma = 0.02
    res_02 = design(m1, m2, window, N, sigma_model=0.02, threshold=10.0)

    # s0 must be identical because s0 is independent of sigma
    assert np.isclose(res_05.s0, res_02.s0, rtol=1e-5)

    # chi^2 must scale exactly as (0.05 / 0.02)^2
    expected_ratio = (0.05 / 0.02) ** 2
    actual_ratio = res_02.chi2_min / res_05.chi2_min
    assert np.isclose(actual_ratio, expected_ratio, rtol=1e-4)

    # Verify that at sigma = sigma_star, Delta BIC reaches threshold
    sigma_star = res_05.sigma_star
    assert sigma_star is not None
    res_star = design(m1, m2, window, N, sigma_model=sigma_star, threshold=10.0)
    assert np.isclose(res_star.expected_delta_bic, 10.0, atol=1e-3)


def test_sphere_drag_stokes_limit_and_log_sign():
    """
    Verify physics constraints on sphere drag series:
    1. Stokes limit: C_D -> 24/Re as Re -> 0.
    2. Sign of log term: +9/160 * Re^2 * ln(Re) is negative for Re < 1.0.
    """
    m_true_vf, _ = get_sphere_drag_models(use_verified_series=True)
    m_true_si, _ = get_sphere_drag_models(use_verified_series=False)

    re_small = np.array([1e-5, 1e-4, 1e-3], dtype=np.float64)
    stokes = 24.0 / re_small

    # Stokes limit
    pred_vf = m_true_vf.predict(re_small)
    pred_si = m_true_si.predict(re_small)
    assert np.allclose(pred_vf / stokes, 1.0, atol=1e-3)
    assert np.allclose(pred_si / stokes, 1.0, atol=1e-3)

    # At Re = 0.5, log term is (9/160) * Re^2 * ln(Re) < 0
    re_mid = np.array([0.5])
    cd_mid_si = m_true_si.predict(re_mid)[0]
    cd_stokes_mid = 24.0 / 0.5
    oseen_mid = cd_stokes_mid * (1.0 + (3.0 / 16.0) * 0.5)

    # Because ln(0.5) < 0, the log correction reduces drag relative to Oseen
    assert cd_mid_si < oseen_mid


def test_sphere_drag_predictions_stand_in_and_verified():
    """
    Verify the Phase 4 predictions for Sphere Drag:
    Brief prediction:
      For stand-in truth (Delta = 3/16*Re + 9/160*Re^2*ln Re, N=200, relative noise on C_D):
      the log term needs sigma ~ 0.14% for Re <= 0.5.
    Also verify predictions for verified Chester & Breach series with ln(Re/2).
    """
    # 1. Stand-in truth at Re <= 0.5, N=200
    m_true_si, m_rival_lin = get_sphere_drag_models(use_verified_series=False, rival_type="linear")
    res_si_05 = design(m_true_si, m_rival_lin, window=(0.01, 0.5), N=200, sigma_model=0.01, threshold=10.0)

    # Expected sigma* ~ 0.14% (0.00137)
    assert res_si_05.sigma_star is not None
    assert 0.0012 <= res_si_05.sigma_star <= 0.0016
    assert np.isclose(res_si_05.sigma_star * 100.0, 0.137, atol=0.015)

    # 2. Stand-in truth at Re <= 1.0, N=200 vs quadratic rival (log grid)
    _, m_rival_quad = get_sphere_drag_models(use_verified_series=False, rival_type="quadratic")
    res_si_10_log = design(
        m_true_si, m_rival_quad, window=(0.01, 1.0), N=200, sigma_model=0.01,
        threshold=10.0, grid_type="log"
    )
    # Brief prediction ~ 0.47% (0.00463)
    assert res_si_10_log.sigma_star is not None
    assert 0.0040 <= res_si_10_log.sigma_star <= 0.0055

    # 3. Verified Chester & Breach series (ln(Re/2))
    m_true_vf, _ = get_sphere_drag_models(use_verified_series=True, rival_type="linear")
    res_vf_05 = design(m_true_vf, m_rival_lin, window=(0.01, 0.5), N=200, sigma_model=0.01, threshold=10.0)
    assert res_vf_05.sigma_star is not None
    assert 0.0040 <= res_vf_05.sigma_star <= 0.0055  # ~ 0.477%


def test_solve_n_star_and_window_star():
    """Verify solver for N* and window extension."""
    m_true, m_rival = get_sphere_drag_models(use_verified_series=False, rival_type="linear")

    # At sigma = 0.005 (0.5%) and Re <= 0.5, N=200, models are indistinguishable
    res_init = design(m_true, m_rival, window=(0.01, 0.5), N=200, sigma_model=0.005, threshold=10.0)
    assert not res_init.separable_current
    assert res_init.n_star is not None
    assert res_init.n_star > 200

    # Verify that running design with N = n_star achieves separability
    res_separated_n = design(m_true, m_rival, window=(0.01, 0.5), N=res_init.n_star, sigma_model=0.005, threshold=10.0)
    assert res_separated_n.expected_delta_bic >= 9.9  # within numerical tolerance of 10.0


def test_monte_carlo_agreement_sphere_drag():
    """
    Scenario 1: Brute-force Monte Carlo reproduces Asimov Delta BIC within tolerance
    on Sphere Drag (Proudman-Pearson vs Oseen linear rival).
    """
    m_true, m_rival = get_sphere_drag_models(use_verified_series=False, rival_type="linear")
    # Using sigma = 0.002 (0.2%), N=200, Re <= 0.5
    mc_res = monte_carlo_delta_bic(
        m_true, m_rival, window=(0.01, 0.5), N=200, sigma_model=0.002,
        n_seeds=100, seed=42, tolerance_sigma=3.0
    )
    assert mc_res.within_tolerance is True
    # Difference between Asimov prediction and empirical mean should be small
    assert mc_res.abs_difference <= 1.5


def test_monte_carlo_agreement_screened_coulomb():
    """
    Scenario 2: Brute-force Monte Carlo reproduces Asimov Delta BIC within tolerance
    on Screened Coulomb (Yukawa exponential vs Rational candidate).
    """
    # M1: Yukawa exponential screening V = (1/r) * exp(-r / lambda)
    m1 = CandidateModel(
        name="Yukawa Exponential",
        fn=lambda r, th: (1.0 / r) * np.exp(-r / th[0]),
        k=1,
        theta_init=np.array([1.0]),
        theta_true=np.array([1.0]),
        bounds=[(0.1, 10.0)],
    )

    # M2: Rational candidate V = (1/r) / (1 + r / lambda_2)
    m2 = CandidateModel(
        name="Rational Correction",
        fn=lambda r, th: (1.0 / r) / (1.0 + r / th[0]),
        k=1,
        theta_init=np.array([1.0]),
        bounds=[(0.1, 10.0)],
    )

    # Test on r in [0.2, 2.0], N=100, relative noise 5% (sigma = 0.05)
    mc_res = monte_carlo_delta_bic(
        m1, m2, window=(0.2, 2.0), N=100, sigma_model=0.05,
        n_seeds=80, seed=99, tolerance_sigma=3.0
    )
    assert mc_res.within_tolerance is True


def test_expectation_correction_calculation():
    """Verify that expectation correction properly applies (k2 - k1)(ln N - 1)."""
    m1 = CandidateModel(name="M1", fn=lambda x, th: th[0]*x, k=1, theta_init=np.array([1.0]))
    m2 = CandidateModel(name="M2", fn=lambda x, th: th[0]*x + th[1]*x**2, k=2, theta_init=np.array([1.0, 0.0]))

    res = design(m1, m2, window=(1.0, 5.0), N=200, sigma_model=0.01)

    # Expected difference between standard and corrected is (k2 - k1) * 1.0 = 1.0
    diff = res.expected_delta_bic - res.expected_delta_bic_corrected
    assert np.isclose(diff, 1.0, atol=1e-5)
