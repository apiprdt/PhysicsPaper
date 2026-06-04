import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import numpy as np
from adcd.feynman_dataset import get_problem
from adcd.jax_optimizer import JAXOptimizer

problem = get_problem("Ideal Gas Law")
X, y_obs = problem.generate_data(n_points=100, seed=42)

optimizer = JAXOptimizer(n_restarts=5)
# Print variables and constants
print("Variables in X:", list(X.keys()))
print("y_obs shape:", y_obs.shape)
print("y_obs mean:", np.mean(y_obs))

res = optimizer.optimize("theta_0 * N * t", X, y_obs, list(X.keys()))
print("Optimisation Result:")
print("Theta:", res.theta)
print("NMSE:", res.nmse)
print("Error:", res.error)
