"""Unit tests for Buckingham-Pi dimensionless group engine."""

from adcd.buckingham_pi import BuckinghamPiEngine


def test_buckingham_rational_nullspace():
    """Verify that rational nullspace vectors don't lose terms when cast to int."""
    engine = BuckinghamPiEngine()
    # Kinetic energy density style: E (M L^2 T^-2), rho (M L^-3), v (L T^-1)
    engine.register("E", [1, 2, -2])
    engine.register("rho", [1, -3, 0])
    engine.register("v", [0, 1, -1])
    engine.register("r", [0, 1, 0])
    pi_groups = engine.compute_pi_groups()
    assert len(pi_groups) > 0
    # E / (rho * v^2) is dimensionless
    expr_str = " ".join(str(g) for g in pi_groups)
    assert "E" in expr_str
    assert "rho" in expr_str
    assert "v" in expr_str


def test_extreme_scale_constants():
    """Verify that dimensionless scale constants don't pollute matrix SVD."""
    engine = BuckinghamPiEngine()
    engine.register("r", [0, 1, 0])
    engine.register("r_0", [0, 1, 0])
    engine.register("theta_0", [0, 0, 0])  # dimensionless parameter
    pi_groups = engine.compute_pi_groups()
    assert len(pi_groups) > 0
    expr_str = " ".join(str(g) for g in pi_groups)
    assert "r" in expr_str
