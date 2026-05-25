"""
jax_optimizer.py
================
Stage 2: Continuous Parameter Optimization via JAX Autodiff and SciPy L-BFGS-B.

Design decisions:
  - CPU-only (JAX_PLATFORM_NAME=cpu) — optimized for local machines without GPU
  - Free parameters named explicitly as theta_0, theta_1, ..., theta_N
  - SciPy L-BFGS-B for high-efficiency Quasi-Newton optimization (~20-50 steps)
  - JIT compile once per expression structure using jax.value_and_grad
  - Log-Uniform Multi-Start Initialization to handle extreme astrophysical scales
  - Gradient NaN protection to prevent L-BFGS matrix corruption
  - NMSE (Normalized MSE) as scale-invariant loss

Usage:
    optimizer = JAXOptimizer()
    result = optimizer.optimize(
        expr_str  = "theta_0 * m * v ** theta_1",
        X         = {"m": np.array([1,2,3]), "v": np.array([1,2,3])},
        y_obs     = np.array([0.5, 4.0, 13.5]),
        data_vars = ["m", "v"]
    )
"""

import os
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
jax.config.update("jax_enable_x64", True)

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import jax.numpy as jnp
from jax import jit, value_and_grad
import numpy as np
import sympy as sp
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

OVERFLOW_THRESHOLD = 1e30
NMSE_FAIL          = float("inf")
CONVERGENCE_TOL    = 1e-5
DEFAULT_N_RESTARTS = 5

@dataclass
class OptimizationResult:
    theta       : Dict[str, float]
    nmse        : float
    likelihood  : float
    history     : List[float]
    converged   : bool
    n_params    : int
    expr_str    : str
    error       : Optional[str] = None


class JAXOptimizer:
    def __init__(
        self,
        n_restarts : int   = DEFAULT_N_RESTARTS,
        beta       : float = 1.0,
        n_steps    : int   = 500,  # Kept for API compatibility, not used by L-BFGS directly
        lr         : float = 0.05, # Kept for API compatibility
    ):
        self.n_restarts = n_restarts
        self.beta       = beta

    def optimize(
        self,
        expr_str  : str,
        X         : Dict[str, np.ndarray],
        y_obs     : np.ndarray,
        data_vars : List[str],
        seed      : int = 42,
    ) -> OptimizationResult:
        try:
            expr, theta_symbols = self._parse_expression(expr_str, data_vars)

            if not theta_symbols:
                return self._evaluate_no_params(expr_str, expr, X, y_obs, data_vars)

            jax_fn = self._build_jax_fn(expr, theta_symbols, data_vars)
            X_jax  = {k: jnp.array(v, dtype=jnp.float64) for k, v in X.items()}
            y_jax  = jnp.array(y_obs, dtype=jnp.float64)

            # Pre-flight check
            test_theta = jnp.ones(len(theta_symbols), dtype=jnp.float64)
            if not self._is_finite(jax_fn, test_theta, X_jax, y_jax):
                return self._fail_result(expr_str, len(theta_symbols), "Non-finite output at test theta")

            scale = 1.0
            try:
                y_pred_1 = np.array(jax_fn(test_theta, X_jax))
                pred_mean = np.mean(np.abs(y_pred_1))
                obs_mean = np.mean(np.abs(y_obs))
                if pred_mean > 1e-30 and np.isfinite(pred_mean):
                    scale = float(obs_mean / pred_mean)
                if not np.isfinite(scale) or scale < 1e-35 or scale > 1e35:
                    scale = 1.0
            except Exception:
                pass

            # Loss function & JIT compilation of value_and_grad
            loss_fn = self._make_loss_fn(jax_fn, X_jax, y_jax)
            val_and_grad_jit = jit(value_and_grad(loss_fn))
            
            # Trigger JIT compilation once
            _ = val_and_grad_jit(test_theta)

            # Multi-start L-BFGS-B
            best_theta, best_nmse = self._multi_start_lbfgs(
                val_and_grad_jit, len(theta_symbols), seed, scale
            )

            theta_dict = {str(s): float(best_theta[i]) for i, s in enumerate(theta_symbols)}
            likelihood = float(np.exp(-self.beta * best_nmse))

            return OptimizationResult(
                theta      = theta_dict,
                nmse       = best_nmse,
                likelihood = likelihood,
                history    = [best_nmse], # L-BFGS doesn't easily yield per-step history without overhead
                converged  = best_nmse < CONVERGENCE_TOL,
                n_params   = len(theta_symbols),
                expr_str   = expr_str,
            )

        except Exception as e:
            logger.warning(f"Optimization failed for '{expr_str}': {e}")
            return self._fail_result(expr_str, 0, str(e))

    def optimize_batch(
        self,
        candidates  : List[Tuple[str, float, float, float]],
        X           : Dict[str, np.ndarray],
        y_obs       : np.ndarray,
        data_vars   : List[str],
    ) -> List[Tuple[str, float, float, float, OptimizationResult]]:
        results = []
        for expr_str, _, _, arc_score in candidates:
            opt_result = self.optimize(expr_str, X, y_obs, data_vars)
            stage2_combined = arc_score * opt_result.likelihood
            results.append((expr_str, stage2_combined, opt_result.nmse, arc_score, opt_result))

        return sorted(results, key=lambda x: -x[1])

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _parse_expression(self, expr_str: str, data_vars: List[str] = None) -> Tuple[sp.Expr, List[sp.Symbol]]:
        sym_locals = {}
        if data_vars:
            for v in data_vars:
                sym_locals[v] = sp.Symbol(v)
        for i in range(100):
            sym_locals[f"theta_{i}"] = sp.Symbol(f"theta_{i}")

        expr = sp.sympify(expr_str, locals=sym_locals)
        theta_symbols = sorted(
            [s for s in expr.free_symbols if str(s).startswith("theta_")],
            key=lambda s: int(str(s).split("_")[1]),
        )
        return expr, theta_symbols

    def _build_jax_fn(self, expr: sp.Expr, theta_symbols: List[sp.Symbol], data_vars: List[str]):
        clean_vars = [v for v in data_vars if not v.startswith("theta_")]
        data_syms  = [sp.Symbol(v) for v in clean_vars]
        all_syms   = data_syms + theta_symbols
        raw_fn = sp.lambdify(all_syms, expr, jnp)

        def jax_fn(theta: jnp.ndarray, X: Dict[str, jnp.ndarray]) -> jnp.ndarray:
            data_vals  = [X[v] for v in clean_vars]
            theta_vals = [theta[i] for i in range(len(theta_symbols))]
            return raw_fn(*data_vals, *theta_vals)

        return jax_fn

    def _make_loss_fn(self, jax_fn, X_jax, y_jax):
        var_y = jnp.var(y_jax) + 1e-10

        def loss(theta: jnp.ndarray) -> jnp.ndarray:
            y_pred = jax_fn(theta, X_jax)
            mse    = jnp.mean((y_pred - y_jax) ** 2)
            return mse / var_y

        return loss

    def _is_finite(self, jax_fn, theta, X_jax, y_jax) -> bool:
        try:
            y_pred = jax_fn(theta, X_jax)
            return bool(
                jnp.all(jnp.isfinite(y_pred)) and
                jnp.all(jnp.abs(y_pred) < OVERFLOW_THRESHOLD)
            )
        except Exception:
            return False

    def _multi_start_lbfgs(
        self, val_and_grad_jit, n_params: int, seed: int, scale: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        rng = np.random.RandomState(seed)
        best_theta = np.ones(n_params, dtype=np.float64)
        if n_params > 0:
            best_theta[0] *= scale
        best_nmse  = NMSE_FAIL

        # Log-Uniform Initialization to handle extreme scales (e.g., G = 1e-11)
        # We sample exponents uniformly between -12 and 12, then randomize sign
        init1 = np.ones(n_params, dtype=np.float64)
        init1[0] *= scale
        init2 = np.ones(n_params, dtype=np.float64) * 0.5
        init2[0] *= scale
        inits = [init1, init2]
        for _ in range(max(0, self.n_restarts - 2)):
            exponents = rng.uniform(-12, 12, size=n_params)
            signs = rng.choice([-1, 1], size=n_params)
            init = signs * (10 ** exponents)
            init = init.astype(np.float64)
            init[0] *= scale
            inits.append(init)

        # SciPy objective wrapper with NaN gradient protection
        def scipy_obj(theta_np):
            v, g = val_and_grad_jit(jnp.array(theta_np))
            v_np, g_np = np.array(v), np.array(g)
            
            # The NaN Trap Mitigation
            if not np.isfinite(v_np) or not np.all(np.isfinite(g_np)):
                # Return huge loss and zero gradient to trigger line search failure securely
                return 1e10, np.zeros_like(g_np)
            
            return v_np, g_np

        for init in inits:
            res = minimize(
                scipy_obj,
                init,
                method="L-BFGS-B",
                jac=True,
                options={"maxiter": 150, "ftol": 1e-7}
            )
            
            if res.fun < best_nmse and np.isfinite(res.fun):
                best_nmse  = float(res.fun)
                best_theta = res.x
                if best_nmse < CONVERGENCE_TOL:
                    break  # Early convergence

        return best_theta, best_nmse

    def _evaluate_no_params(
        self, expr_str: str, expr: sp.Expr, X: Dict[str, np.ndarray], y_obs: np.ndarray, data_vars: List[str]
    ) -> OptimizationResult:
        try:
            clean_vars = [v for v in data_vars if not v.startswith("theta_")]
            data_syms  = [sp.Symbol(v) for v in clean_vars]
            f = sp.lambdify(data_syms, expr, np)
            y_pred = f(*[X[v] for v in clean_vars])

            if not np.all(np.isfinite(y_pred)):
                return self._fail_result(expr_str, 0, "Non-finite output")

            nmse = float(np.mean((y_pred - y_obs) ** 2) / (np.var(y_obs) + 1e-10))
            likelihood = float(np.exp(-self.beta * nmse))

            return OptimizationResult(
                theta={}, nmse=nmse, likelihood=likelihood, history=[nmse],
                converged=nmse < CONVERGENCE_TOL, n_params=0, expr_str=expr_str
            )
        except Exception as e:
            return self._fail_result(expr_str, 0, str(e))

    def _fail_result(self, expr_str: str, n_params: int, error: str) -> OptimizationResult:
        return OptimizationResult(
            theta={}, nmse=NMSE_FAIL, likelihood=0.0, history=[],
            converged=False, n_params=n_params, expr_str=expr_str, error=error
        )
