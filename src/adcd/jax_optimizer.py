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
  - loss_mode='auto': uses full reconstruction loss when dynamic_range(y_classical) > 1e4,
    preserving backward compatibility for all 9 standard scenarios (max DR ~8.74e3 < 1e4)

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
DEFAULT_N_RESTARTS = 15

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
        log_param  : bool  = False, # Log-parameterization to handle extreme scales safely
        maxiter    : int   = 150,  # L-BFGS-B maximum iterations per restart
    ):
        self.n_restarts = n_restarts
        self.beta       = beta
        self.log_param  = log_param
        self.maxiter    = maxiter

    def optimize(
        self,
        expr_str      : str,
        X             : Dict[str, np.ndarray],
        y_obs         : np.ndarray,
        data_vars     : List[str],
        seed          : int = 42,
        loss_mode     : str = 'auto',
        y_classical   : Optional[np.ndarray] = None,
        correction_type: str = 'additive',
    ) -> OptimizationResult:
        """
        Optimize the free parameters of expr_str to fit y_obs.

        Parameters
        ----------
        loss_mode : str
            'residual' — minimize NMSE of expression vs y_obs (the residual/correction term).
                         This is the original behaviour and is used for standard scenarios.
            'full'     — minimize NMSE of the full reconstruction
                         (y_classical + delta or y_classical*(1+delta)) vs y_obs_full.
                         Requires y_classical and correction_type.
            'auto'     — use 'full' if dynamic_range(y_classical) > 1e3, else 'residual'.
                         This is the default: protects existing benchmarks while fixing
                         Blackbody-type scenarios with extreme dynamic range.
        y_classical : np.ndarray or None
            The classical model prediction array.  Required when loss_mode != 'residual'.
            When provided under 'auto', the dynamic range is computed from this array.
        correction_type : str
            'additive' or 'multiplicative'.  Controls how delta is combined with y_classical
            to form the full reconstruction when loss_mode='full'.
        """
        try:
            expr, theta_symbols = self._parse_expression(expr_str, data_vars)

            if not theta_symbols:
                return self._evaluate_no_params(expr_str, expr, X, y_obs, data_vars)

            jax_fn = self._build_jax_fn(expr, theta_symbols, data_vars)
            X_jax  = {k: jnp.array(v, dtype=jnp.float64) for k, v in X.items()}
            y_jax  = jnp.array(y_obs, dtype=jnp.float64)

            # Resolve effective loss mode based on dynamic range auto-detection
            effective_mode = self._resolve_loss_mode(loss_mode, y_classical)

            # Build JAX arrays for full-reconstruction loss (if needed)
            y_classical_jax: Optional[jnp.ndarray] = None
            y_full_jax: Optional[jnp.ndarray] = None
            if effective_mode == 'full' and y_classical is not None:
                y_classical_jax = jnp.array(y_classical, dtype=jnp.float64)
                # y_obs in this context is the *residual*.  Reconstruct true y_obs_full:
                #   additive:       y_obs_full = y_classical + residual
                #   multiplicative: y_obs_full = y_classical * (1 + residual)
                if correction_type == 'multiplicative':
                    y_full_jax = y_classical_jax * (1.0 + y_jax)
                else:
                    y_full_jax = y_classical_jax + y_jax
                logger.debug(
                    f"loss_mode='full' active (dynamic_range={self._dynamic_range(y_classical):.1e}), "
                    f"correction_type='{correction_type}'"
                )

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
            if effective_mode == 'full' and y_classical_jax is not None and y_full_jax is not None:
                loss_fn = self._make_full_loss_fn(
                    jax_fn, X_jax, y_classical_jax, y_full_jax, correction_type
                )
            else:
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
        candidates     : List[Tuple[str, float, float, float]],
        X              : Dict[str, np.ndarray],
        y_obs          : np.ndarray,
        data_vars      : List[str],
        loss_mode      : str = 'auto',
        y_classical    : Optional[np.ndarray] = None,
        correction_type: str = 'additive',
    ) -> List[Tuple[str, float, float, float, OptimizationResult]]:
        results = []
        for expr_str, _, _, arc_score in candidates:
            opt_result = self.optimize(
                expr_str, X, y_obs, data_vars,
                loss_mode=loss_mode,
                y_classical=y_classical,
                correction_type=correction_type,
            )
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

        def _sort_key(s: sp.Symbol) -> int:
            parts = str(s).split("_")
            try:
                return int(parts[-1])
            except ValueError:
                return 999

        expr = sp.sympify(expr_str, locals=sym_locals)
        theta_symbols = sorted(
            [s for s in expr.free_symbols if str(s).startswith("theta_")],
            key=_sort_key,
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

    @staticmethod
    def _nmse_denominator(y: jnp.ndarray) -> jnp.ndarray:
        """Scale-adaptive variance floor for NMSE normalization.

        A fixed epsilon (e.g. 1e-10) collapses NMSE toward zero for sub-nano
        residuals (binary pulsar |dP/dt| ~ 1e-15), letting degenerate
        theta→0 solutions win BIC despite nonzero data.
        """
        var_y = jnp.var(y)
        eps = jnp.maximum(var_y * 1e-6, 1e-30)
        return var_y + eps

    def _make_loss_fn(self, jax_fn, X_jax, y_jax):
        """Residual loss: NMSE of expression vs residual y_obs (original behaviour)."""
        denom = self._nmse_denominator(y_jax)

        def loss(theta: jnp.ndarray) -> jnp.ndarray:
            y_pred = jax_fn(theta, X_jax)
            mse    = jnp.mean((y_pred - y_jax) ** 2)
            return mse / denom

        return loss

    def _make_full_loss_fn(self, jax_fn, X_jax, y_classical_jax, y_full_jax, correction_type: str):
        """Full reconstruction loss: NMSE of reconstructed full signal vs true y_obs_full.

        For multiplicative corrections: y_recon = y_classical * (1 + delta)
        For additive corrections:       y_recon = y_classical + delta

        This correctly weights errors by the magnitude of y_obs_full, which prevents
        low-frequency (high-amplitude) Planck spectrum points from dominating and
        swamping the high-frequency correction signal.
        """
        denom_full = self._nmse_denominator(y_full_jax)

        if correction_type == 'multiplicative':
            def loss(theta: jnp.ndarray) -> jnp.ndarray:
                delta   = jax_fn(theta, X_jax)
                y_recon = y_classical_jax * (1.0 + delta)
                mse     = jnp.mean((y_recon - y_full_jax) ** 2)
                return mse / denom_full
        else:  # additive
            def loss(theta: jnp.ndarray) -> jnp.ndarray:
                delta   = jax_fn(theta, X_jax)
                y_recon = y_classical_jax + delta
                mse     = jnp.mean((y_recon - y_full_jax) ** 2)
                return mse / denom_full

        return loss

    @staticmethod
    def _dynamic_range(y: np.ndarray) -> float:
        """Compute dynamic range: max(|y|) / (min(|y|) + ε)."""
        abs_y = np.abs(y)
        return float(np.max(abs_y) / (np.min(abs_y) + 1e-10))

    @classmethod
    def _resolve_loss_mode(cls, loss_mode: str, y_classical: Optional[np.ndarray]) -> str:
        """Resolve 'auto' loss_mode to 'residual' or 'full' based on dynamic range.

        Auto-detection threshold: if dynamic_range(y_classical) > 1e4, use 'full'.
        Rationale: scenarios with > 10,000x amplitude variation across the domain
        (e.g. Blackbody: DR ~7.1e4) need the full reconstruction loss to weight
        fitting correctly. All 9 standard scenarios have dynamic_range < 8.74e3,
        so they remain unaffected and continue using 'residual' loss.
        """
        DYNAMIC_RANGE_THRESHOLD = 1e4
        if loss_mode == 'residual':
            return 'residual'
        if loss_mode == 'full':
            return 'full'
        # loss_mode == 'auto'
        if y_classical is not None and len(y_classical) > 0:
            dr = cls._dynamic_range(y_classical)
            if dr > DYNAMIC_RANGE_THRESHOLD:
                logger.info(
                    f"loss_mode='auto': dynamic_range={dr:.2e} > {DYNAMIC_RANGE_THRESHOLD:.0e} — "
                    f"switching to 'full' reconstruction loss."
                )
                return 'full'
        return 'residual'

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
        # We sample exponents uniformly using a 50/50 mixture of:
        #   - narrow range [-6, 6] (maximizes efficiency for standard O(1) scales)
        #   - wide range [-20, 20] (handles extreme astrophysical/quantum scales)
        init1 = np.ones(n_params, dtype=np.float64)
        init1[0] *= scale
        init2 = np.ones(n_params, dtype=np.float64) * 0.5
        init2[0] *= scale
        inits = [init1, init2]
        for _ in range(max(0, self.n_restarts - 2)):
            if rng.choice([True, False]):
                exponents = rng.uniform(-6, 6, size=n_params)
            else:
                exponents = rng.uniform(-20, 20, size=n_params)
            signs = rng.choice([-1, 1], size=n_params)
            init = signs * (10 ** exponents)
            init = init.astype(np.float64)
            init[0] *= scale
            inits.append(init)

        for init in inits:
            if self.log_param:
                # Log-parameterization: theta_i = sign_i * exp(u_i).
                # Two separate masks are maintained:
                #   signs[i]: the sign of init[i], frozen throughout optimization so that
                #             exp(u_i) > 0 always, and theta_i = sign * exp(u) recovers
                #             the correct sign at decode time.
                #   is_log[i]: True iff |init[i]| > 1e-30, i.e., the parameter has
                #              non-trivial magnitude worth log-compressing. For exact zeros,
                #              u_i = 0 and theta_i = u_i (linear passthrough) to avoid
                #              log(0) domain errors. This is safe because theta_i = 0
                #              has no magnitude to compress.
                signs = np.sign(init)
                signs = np.where(signs == 0, 1.0, signs)  # prevent sign=0 for zero inits
                is_log = np.abs(init) > 1e-30
                u_init = np.where(is_log, np.log(np.maximum(np.abs(init), 1e-30)), init)

                def scipy_obj_scaled(u_np):
                    theta_np = np.where(is_log, signs * np.exp(u_np), u_np)
                    v, g = val_and_grad_jit(jnp.array(theta_np))
                    v_np, g_np = np.array(v), np.array(g)
                    
                    if not np.isfinite(v_np) or not np.all(np.isfinite(g_np)):
                        return 1e10, np.zeros_like(u_np)
                    
                    # Gradient w.r.t u:
                    # For log-params: dL/du = dL/dtheta * dtheta/du = g * sign * exp(u) = g * theta
                    g_u_np = np.where(is_log, g_np * theta_np, g_np)
                    return v_np, g_u_np

                res = minimize(
                    scipy_obj_scaled,
                    u_init,
                    method="L-BFGS-B",
                    jac=True,
                    options={"maxiter": self.maxiter, "ftol": 1e-7}
                )
                opt_theta = np.where(is_log, signs * np.exp(res.x), res.x)
            else:
                # Scale L-BFGS-B parameters by their initial value (init_scale)
                # so the optimizer always operates on variables of order 1.0.
                # This completely avoids gradient underflow/early termination.
                init_scale = np.where(np.abs(init) > 1e-30, init, 1.0)

                def scipy_obj_scaled(u_np):
                    theta_np = u_np * init_scale
                    v, g = val_and_grad_jit(jnp.array(theta_np))
                    v_np, g_np = np.array(v), np.array(g)
                    
                    if not np.isfinite(v_np) or not np.all(np.isfinite(g_np)):
                        return 1e10, np.zeros_like(u_np)
                    
                    # Gradient w.r.t u: g_u = g_theta * init_scale
                    g_u_np = g_np * init_scale
                    return v_np, g_u_np

                u_init = np.ones_like(init)
                res = minimize(
                    scipy_obj_scaled,
                    u_init,
                    method="L-BFGS-B",
                    jac=True,
                    options={"maxiter": self.maxiter, "ftol": 1e-7}
                )
                
                opt_theta = res.x * init_scale

            if res.fun < best_nmse and np.isfinite(res.fun):
                best_nmse  = float(res.fun)
                best_theta = opt_theta
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

            denom = float(JAXOptimizer._nmse_denominator(jnp.array(y_obs)))
            nmse = float(np.mean((y_pred - y_obs) ** 2) / denom)
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
