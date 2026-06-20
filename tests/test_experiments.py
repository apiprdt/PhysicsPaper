"""Tests for ADCD physics experiment scripts."""

import numpy as np
import pytest
import pandas as pd

from adcd.experiments.muon_g2_validation import (
    generate_muon_g2_data,
    validate_single_order,
    validate_residual_order,
    QED_COEFFICIENTS,
    _ols_monomial_coeff,
)
from adcd.experiments.proposers import (
    extract_linear_coefficient,
    perturbative_order_proposer,
)
from adcd.experiments.sparc_data import _simulate_sparc_stack, stack_sparc_galaxies
from adcd.llm_proposer import ProposalContext


def test_perturbative_proposer_emits_linear_template():
    prop = perturbative_order_proposer(order=1, variable="x", seed=0)
    ctx = ProposalContext(variable_names=["x"], target_name="y", data_statistics={})
    cands = prop.propose(ctx)
    assert any("theta_0 * x" in c for c in cands)


def test_extract_linear_coefficient():
    coeff = extract_linear_coefficient("theta_r1_0 * x", {"theta_r1_0": 0.5}, "x")
    assert coeff == pytest.approx(0.5)


def test_muon_g2_synthetic_data_shape():
    X, y_obs, y_classical = generate_muon_g2_data(n_points=50, seed=1)
    assert "x" in X
    assert len(X["x"]) == 50
    assert np.allclose(y_classical, 0.0)


def test_ols_monomial_exact():
    x = np.linspace(0.01, 0.1, 50)
    coeff = _ols_monomial_coeff(x, QED_COEFFICIENTS[2] * x ** 2, 2)
    assert coeff == pytest.approx(QED_COEFFICIENTS[2], rel=1e-6)


def test_muon_tier_a_recovers_schwinger():
    r = validate_single_order(1, noise_level=0.0, seed=42, n_points=120)
    assert r.tier == "A_isolated"
    assert r.passed, f"C1 error {r.relative_error:.1%}"


def test_muon_tier_b_recovers_order2_on_exact_baseline():
    r = validate_residual_order(2, noise_level=0.0, seed=42, n_points=120)
    assert r.tier == "B_residual"
    assert r.passed, f"C2 error {r.relative_error:.1%}"


def test_sparc_stack_from_minimal_dataframe():
    rows = []
    for r, vo, vg, vd in [(1.0, 50, 10, 40), (2.0, 60, 12, 45), (3.0, 65, 15, 48),
                          (4.0, 68, 16, 50), (5.0, 70, 17, 52)]:
        rows.append({"galaxy": "G1", "radius_kpc": r, "v_obs": float(vo), "e_v_obs": 2.0,
                     "v_gas": float(vg), "v_disk": float(vd), "v_bulge": 0.0})
    df = pd.DataFrame(rows)
    x, nu, sigma, nu_c, n_gal = stack_sparc_galaxies(df, min_v_bar=5.0)
    assert n_gal == 1
    assert len(x) == 5
    assert np.all(nu > 0)


def test_muon_tier_d_simultaneous():
    from adcd.experiments.muon_g2_validation import validate_simultaneous_reference
    rows = validate_simultaneous_reference(max_order=2, seed=42)
    assert all(r.passed for r in rows)


def test_muon_tier_c_ols_with_exact_c1_prior():
    """Tier C+: OLS projection round 2 with exact Schwinger term pre-subtracted."""
    from adcd.experiments.muon_g2_validation import generate_muon_g2_data, QED_COEFFICIENTS, VARIABLE
    from adcd.experiments.proposers import perturbative_order_proposer
    from adcd.iadcd_orchestrator import iADCDOrchestrator

    X, y_obs, y_classical = generate_muon_g2_data(150, 0.0, 42, max_order=2)
    orch = iADCDOrchestrator(
        max_rounds=1, convergence_nmse=1e-6, min_snr=0.01, verbose=False,
        subtraction_mode="ols_projection", projection_variable=VARIABLE,
        prior_subtractions={1: QED_COEFFICIENTS[1]},
    )
    res = orch.run_iterative_discovery(
        X=X, y_obs=y_obs, y_classical=y_classical,
        limit_variable=VARIABLE, limit_direction="0", classical_expr="0.0",
        variables_with_units={VARIABLE: "dimensionless"},
        round_proposers=[perturbative_order_proposer(2, VARIABLE, 43)],
        correction_mode="additive",
        seed=43,
    )
    coeff = res.rounds[0].discovered_theta.get("theta_r1_0")
    err = abs(coeff - QED_COEFFICIENTS[2]) / QED_COEFFICIENTS[2]
    assert err < 0.20, f"C2 error {err:.1%}"


def test_simulated_sparc_stack_is_labeled():
    stack = _simulate_sparc_stack(n_samples=100)
    assert stack.data_source == "SIMULATED"
    assert stack.n_points == 100


# ---------------------------------------------------------------------------
# T2.1 — SPARC fair-comparison (2-param fitted baselines)
# ---------------------------------------------------------------------------

def test_2param_evaluators_match_canonical_at_theta_identity():
    """At θ₀=1, θ₁=1 the 2-param variants reduce to the canonical 0-param forms."""
    from adcd.experiments.mond_comparison import (
        nu_simple_mond, nu_standard_mond, nu_rar,
        nu_simple_mond_2param, nu_standard_mond_2param, nu_rar_2param,
    )
    x = np.linspace(0.1, 10, 50)
    assert np.allclose(nu_simple_mond_2param(x, 1.0, 1.0), nu_simple_mond(x))
    assert np.allclose(nu_standard_mond_2param(x, 1.0, 1.0), nu_standard_mond(x))
    assert np.allclose(nu_rar_2param(x, 1.0, 1.0), nu_rar(x))


def test_2param_evaluators_are_finite_under_extreme_params():
    """Optimiser explores extreme θ values — evaluators must not blow up."""
    from adcd.experiments.mond_comparison import (
        nu_simple_mond_2param, nu_standard_mond_2param, nu_rar_2param,
    )
    x = np.array([1e-3, 1.0, 1e3])
    for theta0 in [1e-6, 1e-2, 1.0, 1e2, 1e6]:
        for theta1 in [1e-4, 1.0, 1e4]:
            for fn in (nu_simple_mond_2param, nu_standard_mond_2param, nu_rar_2param):
                out = fn(x, theta0, theta1)
                # Allow inf/nan at pathological corners but not exceptions
                # (the objective wraps these to a large finite value).
                assert out.shape == x.shape


def test_score_fitted_baselines_returns_4_models_with_2_params():
    from adcd.experiments.mond_comparison import score_fitted_baselines
    x = np.linspace(0.1, 5, 60)
    nu = 1.0 / (1.0 - np.exp(-np.sqrt(x))) + 0.01 * np.random.default_rng(0).normal(size=60)
    scores = score_fitted_baselines(x, nu, adcd_nmse=0.5, n_restarts=3, seed=0)
    assert len(scores) == 4  # 3 fitted baselines + ADCD
    # Every baseline entry should report exactly 2 free parameters
    for s in scores:
        assert s.n_params == 2
    # The fitted_theta dict must be populated for the 3 fitted baselines (not ADCD)
    fitted = [s for s in scores if "ADCD" not in s.name]
    assert len(fitted) == 3
    for s in fitted:
        assert s.fitted_theta is not None
        assert set(s.fitted_theta.keys()) == {"theta_0", "theta_1"}
    # ADCD entry has no fitted_theta (discovered upstream)
    adcd = [s for s in scores if "ADCD" in s.name][0]
    assert adcd.fitted_theta is None


def test_2param_fitting_recovers_known_synthetic_truth():
    """Fit ν = θ₁/(1 − exp(−√(θ₀·x))) on noise-free data → recover θ₀,θ₁."""
    from adcd.experiments.mond_comparison import nu_rar_2param, _fit_2param
    x = np.linspace(0.1, 5, 100)
    theta0_true, theta1_true = 0.7, 1.3
    nu = nu_rar_2param(x, theta0_true, theta1_true)
    res = _fit_2param(nu_rar_2param, x, nu, "RAR", "test", n_restarts=5, seed=0)
    assert res.theta0 == pytest.approx(theta0_true, rel=0.05)
    assert res.theta1 == pytest.approx(theta1_true, rel=0.05)
    assert res.nmse < 1e-6
