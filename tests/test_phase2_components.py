"""
Phase 2 component tests. Written BEFORE implementation (TDD).
Component tests become PASS as implementation progresses.
End-to-end test skipped until MultivariableOrchestrator (MV-5).
"""

import numpy as np
import sympy as sp

from adcd.buckingham_pi import BuckinghamPiEngine
from adcd.residual_factorizer_v2 import ResidualFactorizerV2
from adcd.sequential_arc import SequentialARCChecker


class TestBuckinghamPiEngine:
    def test_mass_ratio_scenario(self):
        """m/M and r/r_0 are Pi groups for Yukawa Mass-Ratio."""
        engine = BuckinghamPiEngine()
        engine.register("m", [1, 0, 0])
        engine.register("M", [1, 0, 0])
        engine.register("r", [0, 1, 0])
        engine.register("r_0", [0, 1, 0])
        pi_groups = engine.compute_pi_groups()
        assert len(pi_groups) == 2
        joined = " ".join(str(sp.simplify(g)) for g in pi_groups)
        assert "m" in joined and "M" in joined
        assert "r" in joined

    def test_turbulent_drag_scenario(self):
        """v/v_ref and rho/rho_ref for turbulent drag."""
        engine = BuckinghamPiEngine()
        engine.register("v", [0, 1, -1])
        engine.register("v_ref", [0, 1, -1])
        engine.register("rho", [1, -3, 0])
        engine.register("rho_ref", [1, -3, 0])
        pi_groups = engine.compute_pi_groups()
        assert len(pi_groups) == 2


class TestSequentialARCChecker:
    def test_independent_limit_pass(self):
        """(m/M)*exp(-r/r_0) vanishes at m→0 independently."""
        checker = SequentialARCChecker(seed=42)
        expr = sp.sympify("theta_0 * (m/M) * exp(-r/r_0)")
        result = checker.check(
            expr,
            limit_vars=["r", "m"],
            limit_dirs=["oo", "0"],
            constants={"r_0": 2.5},
        )
        assert result.passes
        assert "m→0" in result.vanishing_at
        assert "r→∞" in result.vanishing_at

    def test_simultaneous_only_fails(self):
        """Expression that fails independent limit check should not pass."""
        checker = SequentialARCChecker(seed=42)
        expr_fail = sp.sympify("theta_0 * (x + y)")
        result = checker.check(
            expr_fail,
            limit_vars=["x", "y"],
            limit_dirs=["0", "0"],
            constants={},
        )
        assert not result.passes

    def test_product_of_arc_safe_forms(self):
        """Product of two ARC-safe 1D forms is ARC-safe 2D."""
        checker = SequentialARCChecker(seed=42)
        expr = sp.sympify("theta_0 * (v/v_ref)**2 * (rho/rho_ref)")
        result = checker.check(
            expr,
            limit_vars=["v", "rho"],
            limit_dirs=["0", "0"],
            constants={"v_ref": 10.0, "rho_ref": 1.0},
        )
        assert result.passes


class TestResidualFactorizerV2:
    def test_multiplicative_separable(self):
        """f(x)·g(y) detected as multiplicative separable."""
        rng = np.random.default_rng(42)
        x = rng.uniform(1, 5, 100)
        y = rng.uniform(0.1, 2, 100)
        X = {"v": x, "rho": y}
        delta = 0.5 * x**2 * y
        factorizer = ResidualFactorizerV2()
        result = factorizer.test_separability(X, delta)
        assert result.factorization_type == "multiplicative"
        assert result.explained_variance > 0.95

    def test_additive_separable(self):
        rng = np.random.default_rng(42)
        x = rng.uniform(1, 5, 100)
        y = rng.uniform(0.1, 2, 100)
        X = {"v": x, "rho": y}
        delta = 0.3 * x**2 + 0.7 * np.exp(-y)
        result = ResidualFactorizerV2().test_separability(X, delta)
        assert result.factorization_type in ("additive", "multiplicative")

    def test_non_separable_returns_none(self):
        """Coupled non-separable correction."""
        rng = np.random.default_rng(42)
        x = rng.uniform(1, 5, 100)
        y = rng.uniform(0.1, 2, 100)
        X = {"v": x, "rho": y}
        delta = np.sin(x * y)
        result = ResidualFactorizerV2().test_separability(X, delta)
        assert result.factorization_type == "none" or result.explained_variance < 0.8


class TestMultivariableEndToEnd:
    def test_yukawa_mass_ratio_discovery(self):
        """ADCD discovers θ₀·(m/M)·exp(-r/r_0) at 0% noise."""
        from adcd.multivar_orchestrator import get_mv_scenario, run_adcd_mv

        scenario = get_mv_scenario("MV-1: Yukawa Mass-Ratio")
        result = run_adcd_mv(scenario, noise=0.0, seed=42)
        assert result.evaluation.class_match
        assert result.best_nmse_residual < 0.01
        expr = sp.sympify(result.best_expr)
        vars_found = {str(s) for s in expr.free_symbols if s.name in ["m", "M", "r"]}
        assert len(vars_found) >= 2
