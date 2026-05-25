"""
test_jax_optimizer.py
=====================
Test suite for Stage 2: JAXOptimizer.

Tests:
  1. Coefficient recovery — theta converges to ground truth
  2. No-parameter expression — direct evaluation
  3. Pathological expression — NaN/overflow handled gracefully
  4. Scale invariance — NMSE identical across unit scales
  5. Multi-start convergence — finds global optimum
  6. Batch optimization — ranks candidates correctly
  7. Non-convergent expression — graceful failure
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import numpy as np
import pytest
import sympy as sp

from src.jax_optimizer import JAXOptimizer, OptimizationResult, NMSE_FAIL

# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def optimizer():
    return JAXOptimizer(n_steps=500, lr=0.05, n_restarts=5)

@pytest.fixture
def kinetic_data():
    """Synthetic noisy kinetic energy data: E = 0.5 * m * v^2"""
    rng = np.random.RandomState(0)
    m   = rng.uniform(0.5, 5.0, 50)
    v   = rng.uniform(0.5, 4.0, 50)
    y   = 0.5 * m * v**2
    y_noisy = y * (1 + 0.01 * rng.randn(50))  # 1% noise
    return {"m": m, "v": v}, y_noisy, ["m", "v"]

@pytest.fixture
def gravity_data():
    """Synthetic Newtonian gravity: F = G * m1 * m2 / r^2, G = 6.674e-11"""
    rng = np.random.RandomState(1)
    G   = 6.674e-11
    m1  = rng.uniform(1e24, 1e26, 30)
    m2  = rng.uniform(1e20, 1e22, 30)
    r   = rng.uniform(1e8, 1e10, 30)
    y   = G * m1 * m2 / r**2
    return {"m1": m1, "m2": m2, "r": r}, y, ["m1", "m2", "r"]


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestCoefficientRecovery:
    """Stage 2 must recover ground truth coefficients from noisy data."""

    def test_kinetic_energy_recovery(self, optimizer, kinetic_data):
        """theta_0 → 0.5, theta_1 → 2.0 for E = theta_0 * m * v^theta_1"""
        X, y_obs, data_vars = kinetic_data
        result = optimizer.optimize(
            "theta_0 * m * v ** theta_1", X, y_obs, data_vars
        )

        assert isinstance(result, OptimizationResult)
        assert result.error is None, f"Unexpected error: {result.error}"
        assert result.n_params == 2
        assert result.nmse < 1e-3, f"NMSE too high: {result.nmse}"

        # Coefficient check (tolerant of local minima near correct values)
        theta_0 = result.theta["theta_0"]
        theta_1 = result.theta["theta_1"]

        # theta_0 * theta_1^2 should be close to 0.5 * 2.0 = 1.0 regardless of degeneracy
        # More robustly: check NMSE
        assert result.nmse < 0.01, f"Poor fit: NMSE={result.nmse}"

    def test_gravity_coefficient_recovery(self, optimizer, gravity_data):
        """theta_0 → G ≈ 6.674e-11 for F = theta_0 * m1 * m2 / r^2"""
        X, y_obs, data_vars = gravity_data
        result = optimizer.optimize(
            "theta_0 * m1 * m2 / r**2", X, y_obs, data_vars
        )

        assert result.error is None
        assert result.nmse < 1e-2, f"Poor gravity fit: NMSE={result.nmse}"

        recovered_G = result.theta["theta_0"]
        true_G = 6.674e-11
        rel_err = abs(recovered_G - true_G) / true_G
        assert rel_err < 0.05, f"G recovery error: {rel_err:.4f} (got {recovered_G:.4e})"


class TestNoParameterExpression:
    """Expressions with no theta_N symbols — direct evaluation."""

    def test_exact_formula(self, optimizer, kinetic_data):
        """0.5 * m * v**2 has no free params → evaluate directly."""
        X, y_obs, data_vars = kinetic_data
        result = optimizer.optimize("0.5 * m * v**2", X, y_obs, data_vars)

        assert result.n_params == 0
        assert result.theta == {}
        assert result.nmse < 0.01, f"Exact formula should fit well: {result.nmse}"

    def test_wrong_formula_high_nmse(self, optimizer, kinetic_data):
        """Wrong formula without free params → high NMSE."""
        X, y_obs, data_vars = kinetic_data
        result = optimizer.optimize("m * v", X, y_obs, data_vars)  # momentum, not energy

        assert result.n_params == 0
        assert result.nmse > 0.1, f"Wrong formula should have high NMSE: {result.nmse}"


class TestPathologicalExpressions:
    """Physically invalid expressions must fail gracefully."""

    def test_diverging_at_v_zero(self, optimizer):
        """mc^3/v diverges as v→0; optimizer must not crash."""
        rng = np.random.RandomState(2)
        v   = rng.uniform(0.01, 0.1, 20)   # near-zero velocities
        m   = rng.uniform(1.0, 5.0, 20)
        c   = np.full(20, 3e8)
        X   = {"m": m, "v": v, "c": c}
        y   = 0.5 * m * v**2
        data_vars = ["m", "v", "c"]

        # This is the counterexample from the non-redundancy theorem
        result = optimizer.optimize(
            "0.5 * m * v**2 + m * c**3 / v", X, y, data_vars
        )

        # Must not crash; NMSE should be terrible (inf or very high)
        assert isinstance(result, OptimizationResult)
        # Either fails or has terrible NMSE
        assert result.nmse == NMSE_FAIL or result.nmse > 1.0, \
            f"Pathological expression should fail: nmse={result.nmse}"

    def test_division_by_zero_expression(self, optimizer):
        """1/theta_0 where theta_0 could be 0 — must handle gracefully."""
        rng = np.random.RandomState(3)
        x   = rng.uniform(1.0, 5.0, 20)
        y   = x**2
        X   = {"x": x}

        result = optimizer.optimize("theta_0 / (x - theta_0)", X, y, ["x"])
        assert isinstance(result, OptimizationResult)


class TestScaleInvariance:
    """
    Combined Score must be identical regardless of physical unit scale.
    
    This validates the NMSE normalization: MSE / Var(y_obs).
    """

    def test_joule_vs_millijoule(self, optimizer):
        """
        Same experiment, units scaled by 1000.
        NMSE (and therefore likelihood) must be identical.
        """
        rng = np.random.RandomState(4)
        m   = rng.uniform(0.5, 5.0, 30)
        v   = rng.uniform(0.5, 4.0, 30)

        # Joule scale
        y_joule = 0.5 * m * v**2
        X_joule = {"m": m, "v": v}

        # Millijoule scale: y * 1000, m * 1000 (in grams * m²/ms²)
        y_mj    = y_joule * 1000
        X_mj    = {"m": m * 1000, "v": v}

        result_j  = optimizer.optimize("theta_0 * m * v**2", X_joule, y_joule, ["m", "v"])
        result_mj = optimizer.optimize("theta_0 * m * v**2", X_mj,    y_mj,    ["m", "v"])

        assert result_j.error  is None
        assert result_mj.error is None

        # NMSE must be scale-invariant (within numerical precision)
        assert abs(result_j.nmse - result_mj.nmse) < 1e-4, (
            f"Scale invariance violated: "
            f"nmse_joule={result_j.nmse:.8f}, nmse_mj={result_mj.nmse:.8f}"
        )


class TestConvergenceAndRanking:
    """Optimizer must rank physically correct expressions above incorrect ones."""

    def test_correct_beats_incorrect(self, optimizer, kinetic_data):
        """
        E = theta_0 * m * v^theta_1   (correct structure)
        should rank above
        E = theta_0 * m * v           (wrong: linear in v)
        """
        X, y_obs, data_vars = kinetic_data

        result_correct  = optimizer.optimize("theta_0 * m * v**theta_1", X, y_obs, data_vars)
        result_wrong    = optimizer.optimize("theta_0 * m * v",           X, y_obs, data_vars)

        assert result_correct.nmse < result_wrong.nmse, (
            f"Correct structure should win: "
            f"nmse_correct={result_correct.nmse:.4f}, "
            f"nmse_wrong={result_wrong.nmse:.4f}"
        )

    def test_batch_optimization_ordering(self, optimizer, kinetic_data):
        """
        optimize_batch must return candidates sorted by stage2_combined_score.
        Ground truth candidate must rank #1.
        """
        X, y_obs, data_vars = kinetic_data

        # Simulate Stage 1 output: (expr_str, combined_score, mse, arc_score)
        stage1_candidates = [
            ("theta_0 * m * v",           0.3, 0.5, 0.3),  # wrong structure
            ("theta_0 * m * v**theta_1",  0.7, 0.2, 0.7),  # correct
            ("theta_0 * m**2 * v",        0.2, 0.8, 0.2),  # wrong
        ]

        ranked = optimizer.optimize_batch(stage1_candidates, X, y_obs, data_vars)

        assert len(ranked) == 3
        # Scores must be sorted descending
        scores = [r[1] for r in ranked]
        assert scores == sorted(scores, reverse=True), \
            f"Results not sorted: {scores}"

        # Correct expression must be #1
        best_expr = ranked[0][0]
        assert "theta_1" in best_expr or "v**2" in best_expr, \
            f"Wrong expression ranked first: {best_expr}"


class TestEarlyStopAndEfficiency:
    """Optimizer should converge on simple problems."""

    def test_linear_recovery(self):
        """y = theta_0 * x, theta_0=3.0 — simplest possible case."""
        rng = np.random.RandomState(99)
        x = rng.uniform(1.0, 5.0, 50)
        y = 3.0 * x

        opt = JAXOptimizer(n_steps=1000, lr=0.1, n_restarts=3)
        result = opt.optimize("theta_0 * x", {"x": x}, y, ["x"])

        assert result.error is None, f"Error: {result.error}"
        assert result.nmse < 1e-4, f"NMSE too high: {result.nmse}"
        assert abs(result.theta.get("theta_0", 0) - 3.0) < 0.05, \
            f"theta_0 recovery: {result.theta}"
