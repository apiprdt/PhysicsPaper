"""
Tests for Phase 2.1: Joint baseline-correction fit, Hessian condition number, and phi-theta degeneracy detection.
"""

import numpy as np

from adcd.joint_fit import fit_joint_baseline_correction


def test_joint_fit_well_conditioned():
    """A non-degenerate model with distinct parameters yields a well-conditioned Hessian."""
    # Data: y = (phi_0 / r) * (1 + theta_0 * exp(-r / theta_1))
    rng = np.random.RandomState(42)
    r = np.linspace(0.5, 5.0, 100)
    phi_true = {"k_e": 2.0}
    theta_true = {"theta_0": 0.5, "theta_1": 1.5}

    y_cl = phi_true["k_e"] / r
    delta = theta_true["theta_0"] * (np.exp(-r / theta_true["theta_1"]) - 1.0)
    y_true = y_cl * (1.0 + delta)
    sigma_y = 0.01 * y_true
    y_obs = y_true + rng.normal(0, sigma_y)

    # Baseline: k_e / r
    def baseline_fn(phi, X):
        return phi["k_e"] / X["r"]

    # Correction: theta_0 * (exp(-r / theta_1) - 1.0)
    def corr_fn(theta, X):
        return theta["theta_0"] * (np.exp(-X["r"] / theta["theta_1"]) - 1.0)

    # Prior on k_e: mean=2.0, std=0.2
    phi_priors = {"k_e": (2.0, 0.2)}

    result = fit_joint_baseline_correction(
        baseline_fn=baseline_fn,
        corr_fn=corr_fn,
        X={"r": r},
        y_obs=y_obs,
        sigma_y=sigma_y,
        phi_priors=phi_priors,
        init_phi={"k_e": 2.1},
        init_theta={"theta_0": 0.4, "theta_1": 1.2},
        mode="multiplicative",
    )

    assert result.converged is True
    assert result.condition_number < 1e5
    assert result.is_degenerate is False
    assert np.isclose(result.phi_opt["k_e"], 2.0, atol=0.1)
    assert np.isclose(result.theta_opt["theta_0"], 0.5, atol=0.1)


def test_joint_fit_detects_phi_theta_degeneracy():
    """
    When baseline parameter phi_0 and correction parameter theta_0 are mathematically
    degenerate (e.g. both scale as 1/r at leading order), the Hessian condition number
    explodes and is_degenerate must be True.
    """
    rng = np.random.RandomState(42)
    r = np.linspace(0.5, 5.0, 100)

    # y = (phi_0 / r) * (1 + theta_0) = (phi_0 + phi_0 * theta_0) / r
    # Flat prior on phi_0 => exact collinearity / singular Hessian
    def baseline_fn(phi, X):
        return phi["A"] / X["r"]

    def corr_fn(theta, X):
        # A constant offset in multiplicative mode directly multiplies phi["A"]
        return theta["theta_0"] * np.ones_like(X["r"])

    y_obs = 3.0 / r + rng.normal(0, 0.01, size=len(r))
    sigma_y = np.full_like(y_obs, 0.01)

    # Broad prior allowing phi to drift freely
    phi_priors = {"A": (3.0, 100.0)}

    result = fit_joint_baseline_correction(
        baseline_fn=baseline_fn,
        corr_fn=corr_fn,
        X={"r": r},
        y_obs=y_obs,
        sigma_y=sigma_y,
        phi_priors=phi_priors,
        init_phi={"A": 2.5},
        init_theta={"theta_0": 0.2},
        mode="multiplicative",
    )

    assert result.condition_number > 1e6 or not np.isfinite(result.condition_number)
    assert result.is_degenerate is True
    assert "Degeneracy flagged" in result.diagnostic_message


def test_joint_fit_biased_phi_prior_degrades_verdict_not_silent_absorption():
    """
    When a known bias is injected into the baseline parameter prior phi_0,
    the joint optimizer must NOT silently absorb the correction if the parameters are degenerate,
    and the penalty/loss must significantly increase.
    """
    rng = np.random.RandomState(42)
    r = np.linspace(0.5, 5.0, 100)

    # True model: A = 1.0, correction theta_0 * (exp(-r) - 1.0) with theta_0 = 0.5
    y_true = (1.0 / r) * (1.0 + 0.5 * (np.exp(-r) - 1.0))
    sigma_y = 0.01 * y_true
    y_obs = y_true + rng.normal(0, sigma_y)

    def baseline_fn(phi, X):
        return phi["A"] / X["r"]

    def corr_fn(theta, X):
        return theta["theta_0"] * (np.exp(-X["r"]) - 1.0)

    # Case 1: Unbiased tight prior
    res_unbiased = fit_joint_baseline_correction(
        baseline_fn=baseline_fn,
        corr_fn=corr_fn,
        X={"r": r},
        y_obs=y_obs,
        sigma_y=sigma_y,
        phi_priors={"A": (1.0, 0.05)},
        init_phi={"A": 1.0},
        init_theta={"theta_0": 0.5},
    )

    # Case 2: Injected biased prior (A forced toward 2.0 with tight prior std=0.01)
    res_biased = fit_joint_baseline_correction(
        baseline_fn=baseline_fn,
        corr_fn=corr_fn,
        X={"r": r},
        y_obs=y_obs,
        sigma_y=sigma_y,
        phi_priors={"A": (2.0, 0.01)},
        init_phi={"A": 2.0},
        init_theta={"theta_0": 0.5},
    )

    # The loss under the biased prior must degrade significantly
    assert res_biased.total_loss > res_unbiased.total_loss + 50.0
    # Prior penalty must reflect the bias
    assert res_biased.prior_penalty > 50.0


def test_joint_fit_unit_invariance_of_normalized_hessian():
    """
    Rescaling a physical parameter by 10^6 changes the unnormalized Hessian condition number
    by ~10^12, but the scale-normalized Hessian condition number remains invariant and well-conditioned.
    """
    rng = np.random.RandomState(42)
    r = np.linspace(0.5, 5.0, 100)
    y_true = (2.0 / r) * (1.0 + 0.5 * (np.exp(-r) - 1.0))
    sigma_y = 0.01 * y_true
    y_obs = y_true + rng.normal(0, sigma_y)

    # Unit system 1: A ~ O(1)
    res_unit1 = fit_joint_baseline_correction(
        baseline_fn=lambda phi, X: phi["A"] / X["r"],
        corr_fn=lambda theta, X: theta["theta_0"] * (np.exp(-X["r"]) - 1.0),
        X={"r": r},
        y_obs=y_obs,
        sigma_y=sigma_y,
        phi_priors={"A": (2.0, 0.2)},
        init_phi={"A": 2.0},
        init_theta={"theta_0": 0.5},
    )

    # Unit system 2: A_micro = 1e6 * A ~ O(1e6)
    res_unit2 = fit_joint_baseline_correction(
        baseline_fn=lambda phi, X: (1e-6 * phi["A_micro"]) / X["r"],
        corr_fn=lambda theta, X: theta["theta_0"] * (np.exp(-X["r"]) - 1.0),
        X={"r": r},
        y_obs=y_obs,
        sigma_y=sigma_y,
        phi_priors={"A_micro": (2.0e6, 0.2e6)},
        init_phi={"A_micro": 2.0e6},
        init_theta={"theta_0": 0.5},
    )

    assert res_unit1.is_degenerate is False
    assert res_unit2.is_degenerate is False
    assert res_unit2.raw_condition_number > 1e10
    assert np.isclose(res_unit1.condition_number, res_unit2.condition_number, rtol=1e-2)
