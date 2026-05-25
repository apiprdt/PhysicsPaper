import time
import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import grad, jit, value_and_grad
import numpy as np
import sympy as sp
import jaxopt
from scipy.optimize import minimize

def run_benchmark():
    # Setup data
    rng = np.random.RandomState(42)
    m = rng.uniform(1e24, 1e26, 50)
    v = rng.uniform(1e3, 1e4, 50)
    y_obs = 0.5 * m * v**2
    
    # SymPy to JAX
    expr = sp.sympify("theta_0 * m * v**theta_1")
    theta_symbols = [sp.Symbol("theta_0"), sp.Symbol("theta_1")]
    data_vars = ["m", "v"]
    
    raw_fn = sp.lambdify(data_vars + theta_symbols, expr, jnp)
    
    X_jax = {"m": jnp.array(m), "v": jnp.array(v)}
    y_jax = jnp.array(y_obs)
    var_y = jnp.var(y_jax) + 1e-10
    
    def loss_fn(theta):
        y_pred = raw_fn(X_jax["m"], X_jax["v"], theta[0], theta[1])
        mse = jnp.mean((y_pred - y_jax) ** 2)
        return mse / var_y

    init_theta = jnp.array([1.0, 1.0])
    
    # 1. JIT Compilation Time
    t0 = time.time()
    loss_jit = jit(loss_fn)
    grad_jit = jit(grad(loss_fn))
    loss_jit(init_theta)
    grad_jit(init_theta)
    t1 = time.time()
    print(f"JIT Compilation Time: {(t1 - t0)*1000:.2f} ms")
    
    # 2. Eager Adam Time (500 steps)
    t0 = time.time()
    theta = init_theta
    grad_fn_eager = grad(loss_fn)
    for _ in range(500):
        g = grad_fn_eager(theta)
        theta = theta - 0.05 * g
    t1 = time.time()
    print(f"Eager Adam (500 steps): {(t1 - t0)*1000:.2f} ms")
    
    # 3. JIT Adam Time (500 steps)
    t0 = time.time()
    theta = init_theta
    for _ in range(500):
        g = grad_jit(theta)
        theta = theta - 0.05 * g
    t1 = time.time()
    print(f"JIT Adam (500 steps): {(t1 - t0)*1000:.2f} ms")
    
    # 4. SciPy L-BFGS-B with JAX value_and_grad (Eager)
    val_grad = value_and_grad(loss_fn)
    def scipy_obj(theta_np):
        v, g = val_grad(jnp.array(theta_np))
        return np.array(v), np.array(g)
    
    t0 = time.time()
    res = minimize(scipy_obj, np.array([1.0, 1.0]), method='L-BFGS-B', jac=True)
    t1 = time.time()
    print(f"SciPy L-BFGS-B (Eager) - steps={res.nit}: {(t1 - t0)*1000:.2f} ms, Loss: {res.fun:.2e}")
    
    # 5. jaxopt LBFGS
    lbfgs = jaxopt.LBFGS(fun=loss_fn, maxiter=100)
    t0 = time.time()
    res_jaxopt = lbfgs.run(init_theta)
    t1 = time.time()
    print(f"jaxopt LBFGS - steps={res_jaxopt.state.iter_num}: {(t1 - t0)*1000:.2f} ms, Loss: {res_jaxopt.state.value:.2e}")

if __name__ == "__main__":
    run_benchmark()
