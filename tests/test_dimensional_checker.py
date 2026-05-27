import pytest
from src.dimensional_checker import DimensionalChecker, ASTValidator, validate_transcendental_args

def test_valid_and_invalid_dimensions():
    checker = DimensionalChecker()
    
    # Valid Energy dimension: 0.5 * m * v^2 -> [1, 2, -2]
    assert checker.verify("0.5 * m * v**2", "E") is True
    
    # Invalid Energy dimension: m * v (This is momentum)
    assert checker.verify("m * v", "E") is False
    
    # Dimensional breakdown: Cannot add mass and length (m + r)
    assert checker.verify("m + r", "E") is False

def test_nonlinear_and_transcendental_dimensions():
    checker = DimensionalChecker()
    
    # 1. Relativistic factor (sqrt(1 - v^2/c^2)) which is dimensionless
    assert checker.verify("sqrt(1 - v**2/c**2)", "E") is False  # Target is energy, candidate is dimensionless
    
    # Let's verify it is dimensionless [0, 0, 0] under the hood
    import sympy as sp
    expr = sp.sympify("sqrt(1 - v**2/c**2)", locals=checker.locals)
    assert checker._get_dim_vector(expr) == [0, 0, 0]
    
    # 2. Transcendental function with dimensionless argument (sin(v/c))
    assert checker._get_dim_vector(sp.sympify("sin(v/c)", locals=checker.locals)) == [0, 0, 0]
    
    # 3. Transcendental function with dimensioned argument (sin(v)) should raise TypeError
    with pytest.raises(TypeError):
        checker._get_dim_vector(sp.sympify("sin(v)", locals=checker.locals))

def test_ast_bloat_control():
    validator = ASTValidator(max_depth=4, max_tokens=10)
    
    # Compact clean formulas pass cleanly
    assert validator.verify("0.5 * m * v**2") is True
    
    # Complex, over-fitted formulas are instantly rejected
    bloated_formula = "0.5 * m * v**2 + m * v**3 / c - m * v**4 / c**2 + r / t"
    assert validator.verify(bloated_formula) is False

def test_dynamic_thresholding():
    validator = ASTValidator(max_depth=1, max_tokens=1)  # Strict limits
    
    # 0.5 * m * v**2 fails under strict limits
    assert validator.verify("0.5 * m * v**2") is False
    
    # Dynamically set threshold based on Kinetic Energy target + 5 tokens, + 2 depth
    # N_target = 7 tokens, D_target = 3 depth -> Max Tokens = 12, Max Depth = 5
    validator.set_threshold_relative_to("0.5 * m * v**2", delta_tokens=5, delta_depth=2)
    
    # Now it passes!
    assert validator.verify("0.5 * m * v**2") is True
    
    # A slightly larger but acceptable candidate passes
    assert validator.verify("0.5 * m * v**2 + m") is True
    
    # A bloated formula still fails
    bloated_formula = "0.5 * m * v**2 + m * v**3 / c - m * v**4 / c**2 + r / t"
    assert validator.verify(bloated_formula) is False


def test_validate_transcendental_args():
    checker = DimensionalChecker()
    import sympy as sp
    
    # sin(v/c) has dimensionless argument -> True
    expr_ok = sp.sympify("sin(v/c)", locals=checker.locals)
    assert validate_transcendental_args(expr_ok, checker) is True
    
    # sin(v) has dimensioned argument, but only 1 physical symbol -> True (due to adaptive relaxation)
    expr_bad = sp.sympify("sin(v)", locals=checker.locals)
    assert validate_transcendental_args(expr_bad, checker) is True
    
    # sin(v * t) has 2 physical symbols and is dimensioned -> False
    expr_two_phys = sp.sympify("sin(v * t)", locals=checker.locals)
    assert validate_transcendental_args(expr_two_phys, checker) is False
    
    # Compound expression with valid sin inside
    expr_compound_ok = sp.sympify("0.5 * m * v**2 * sin(v/c)", locals=checker.locals)
    assert validate_transcendental_args(expr_compound_ok, checker) is True
    
    # Compound expression with invalid sin inside (sin(v) is True due to relaxation, but let's test sin(v * t))
    expr_compound_bad = sp.sympify("0.5 * m * v**2 * sin(v * t)", locals=checker.locals)
    assert validate_transcendental_args(expr_compound_bad, checker) is False
    
    # Expression with theta_ parameter inside transcendental function
    # theta_1 * t has 1 physical symbol (t), so it passes (theta_1 can scale it)
    expr_theta_time = sp.sympify("sin(theta_1 * t)", locals=checker.locals)
    assert validate_transcendental_args(expr_theta_time, checker) is True
    
    # But if there are multiple physical symbols (e.g. v and t), and it is not dimensionless, it fails
    expr_theta_multi = sp.sympify("sin(theta_1 * v * t)", locals=checker.locals)
    assert validate_transcendental_args(expr_theta_multi, checker) is False
    
    # but if the argument is theta_1 (dimensionless) or theta_1 * v/c, it should pass
    expr_theta_ok = sp.sympify("sin(theta_1)", locals=checker.locals)
    assert validate_transcendental_args(expr_theta_ok, checker) is True
    
    expr_theta_vc = sp.sympify("sin(theta_1 * v/c)", locals=checker.locals)
    assert validate_transcendental_args(expr_theta_vc, checker) is True
