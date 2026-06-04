import pytest
import numpy as np
import sympy as sp
from adcd.coarse_evaluator import CoarseEvaluator

def test_coarse_evaluator_basic():
    # Setup simple synthetic data
    X = {"x": np.array([1.0, 2.0, 3.0]), "y": np.array([4.0, 5.0, 6.0])}
    y_obs = np.array([5.0, 7.0, 9.0])  # ground-truth is x + y
    
    evaluator = CoarseEvaluator(X, y_obs)
    
    # 1. Perfect model: x + y
    expr_perfect = sp.sympify("x + y")
    mse, nmse = evaluator.evaluate(expr_perfect)
    assert mse == pytest.approx(0.0, abs=1e-10)
    assert nmse == pytest.approx(0.0, abs=1e-10)
    
    # 2. Bad model: x * y
    expr_bad = sp.sympify("x * y")
    mse_bad, nmse_bad = evaluator.evaluate(expr_bad)
    # y_pred = [4.0, 10.0, 18.0], y_obs = [5.0, 7.0, 9.0]
    # diff = [-1.0, 3.0, 9.0] -> squared = [1.0, 9.0, 81.0] -> mean = 91/3 = 30.3333
    assert mse_bad == pytest.approx(30.333333, abs=1e-4)

def test_numerical_protection_and_nans():
    # Domain error or infinity limits
    X = {"x": np.array([0.0, 1.0, 2.0])}
    y_obs = np.array([1.0, 2.0, 3.0])
    
    evaluator = CoarseEvaluator(X, y_obs)
    
    # 1/x triggers division by zero for x=0.0
    expr_div_zero = sp.sympify("1 / x")
    mse, nmse = evaluator.evaluate(expr_div_zero)
    assert mse == float('inf')
    assert nmse == float('inf')
    
    # sqrt(-x) triggers invalid values for x=1.0, 2.0
    expr_invalid_domain = sp.sympify("sqrt(-x)")
    mse_domain, nmse_domain = evaluator.evaluate(expr_invalid_domain)
    assert mse_domain == float('inf')
    assert nmse_domain == float('inf')
