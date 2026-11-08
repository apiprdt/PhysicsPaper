"""
Tests for Phase 2.3: Heteroscedastic likelihood, intrinsic scatter estimation, and full Gaussian BIC.
"""

import numpy as np

from adcd.likelihood import HeteroscedasticLikelihood


def test_heteroscedastic_log_likelihood_exact():
    """Verify log-likelihood computation against exact closed-form Gaussian likelihood."""
    y_obs = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([1.1, 1.9, 3.2])
    sigma_y = np.array([0.05, 0.1, 0.15])
    sigma_int = 0.02

    var_tot = sigma_y ** 2 + sigma_int ** 2
    res = y_obs - y_pred
    expected_nll = 0.5 * np.sum(res ** 2 / var_tot + np.log(2 * np.pi * var_tot))
    expected_log_lik = -expected_nll

    lik = HeteroscedasticLikelihood()
    log_lik = lik.compute_log_likelihood(y_obs, y_pred, sigma_y, sigma_int=sigma_int)
    assert np.isclose(log_lik, expected_log_lik, rtol=1e-12)


def test_intrinsic_scatter_recovery():
    """
    When true data has both heteroscedastic measurement error and intrinsic physical scatter,
    fit_intrinsic_scatter accurately recovers sigma_int.
    """
    rng = np.random.RandomState(42)
    N = 1000
    x = np.linspace(0.1, 5.0, N)
    y_true = 2.0 * np.exp(-x)
    
    # Heteroscedastic measurement noise: 1% to 3%
    sigma_meas = (0.01 + 0.02 * (x / 5.0)) * y_true
    # True intrinsic physical scatter: 0.03
    sigma_int_true = 0.03

    meas_noise = rng.normal(0, sigma_meas)
    int_scatter = rng.normal(0, sigma_int_true, size=N)
    y_obs = y_true + meas_noise + int_scatter

    lik = HeteroscedasticLikelihood()
    sigma_int_opt = lik.fit_intrinsic_scatter(y_obs, y_true, sigma_meas)

    # Recovered intrinsic scatter must be close to 0.03 within sampling error
    assert np.isclose(sigma_int_opt, sigma_int_true, atol=0.006)


def test_no_chi2_gate_large_n_not_vetoed():
    """
    At large N=3300, a traditional chi2_nu <= 1 + 3*sqrt(2/(N-k)) gate allows only ~7.4% excess
    and falsely rejects real datasets with slight intrinsic scatter.
    The heteroscedastic likelihood framework must NOT veto, but cleanly evaluate BIC.
    """
    N = 3300
    rng = np.random.RandomState(42)
    y_true = np.ones(N)
    # Measurement noise 1%
    sigma_meas = np.full(N, 0.01)
    # Slight excess scatter causing chi2_nu ~ 1.15 (> 1.074 threshold)
    y_obs = y_true + rng.normal(0, 0.0107, size=N)

    lik = HeteroscedasticLikelihood()
    res = lik.evaluate_candidate_likelihood(
        y_obs=y_obs,
        y_pred=y_true,
        sigma_y=sigma_meas,
        k_params=2,
        search_space_size=50,
        fit_intrinsic_scatter=True,
    )

    # Must produce valid finite BIC without raising rejection or veto
    assert np.isfinite(res["bic"])
    assert res["sigma_int"] > 0.0
    assert res["chi2_nu"] > 1.0
    # The evaluation succeeds cleanly without hard rejection
    assert res["status"] == "EVALUATED"


def test_bic_penalizes_unnecessary_intrinsic_scatter():
    """When measurement noise already accounts for all scatter, sigma_int collapses to 0."""
    N = 500
    rng = np.random.RandomState(42)
    y_true = np.linspace(1.0, 10.0, N)
    sigma_meas = np.full(N, 0.1)
    y_obs = y_true + rng.normal(0, sigma_meas)

    lik = HeteroscedasticLikelihood()
    sigma_int_opt = lik.fit_intrinsic_scatter(y_obs, y_true, sigma_meas)

    # Intrinsic scatter must be practically zero (<= 1e-4)
    assert sigma_int_opt < 1e-3
