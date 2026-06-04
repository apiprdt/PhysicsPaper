import os
import sys
import numpy as np
import sympy as sp
from adcd.feynman_dataset import get_problem
from adcd.coarse_evaluator import CoarseEvaluator

problem = get_problem("Ideal Gas Law")
X, y_obs = problem.generate_data(n_points=100, seed=42)

# Check Stage 1 evaluation
evaluator = CoarseEvaluator(X, y_obs, constants=problem.constants)
N = sp.Symbol("N")
t = sp.Symbol("t")
expr = N * t
mse, nmse = evaluator.evaluate(expr, has_params=True)
print("For N * t with has_params=True:")
print("MSE:", mse)
print("NMSE:", nmse)

mse_no, nmse_no = evaluator.evaluate(expr, has_params=False)
print("For N * t with has_params=False:")
print("MSE:", mse_no)
print("NMSE:", nmse_no)
