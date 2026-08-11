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

OVERFLOW_THRESHOLD = 1e30
NMSE_FAIL          = float("inf")
CONVERGENCE_TOL    = 1e-5
DEFAULT_N_RESTARTS = 15
_DOMAIN_EPS        = 1e-9
_EXP_CLIP          = 700.0


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


def _domain_harden_expr(expr: sp.Expr) -> sp.Expr:
    """
    Symbolically rewrite an expression so sqrt/pow and log never see
    out-of-domain arguments, and exp never overflows float64.
    For all physically valid inputs (positive quantities), the hardened
    expression is identical to the original.
    """
    if expr.is_Atom:
        return expr

    new_args = [_domain_harden_expr(a) for a in expr.args]
    expr2 = expr.func(*new_args)

    if isinstance(expr2, sp.Pow):
        base, exponent = expr2.args
        if exponent.is_Number and not exponent.is_Integer:
            return sp.Pow(sp.Max(base, _DOMAIN_EPS), exponent)
        return expr2

    if isinstance(expr2, sp.log):
        (arg,) = expr2.args
        return sp.log(sp.Max(arg, _DOMAIN_EPS))

    if isinstance(expr2, sp.exp):
        (arg,) = expr2.args
        return sp.exp(sp.Min(sp.Max(arg, -_EXP_CLIP), _EXP_CLIP))

    return expr2


class JAXOptimizer:
    def __init__(
        self,
        n_restarts : int   = DEFAULT_N_RESTARTS,
        beta       : float = 1.0,
        n_steps    : int   = 500,   # kept for API compatibility, unused by L-BFGS directly
        lr         : float = 0.05,  # kept for API compatibility
        log_param  : bool  = True,  # log-reparameterization for order-of-magnitude traversal
        maxiter    : int   = 150,
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
        try:
            expr, theta_symbols = self._parse_expression(expr_str, data_vars)

            if not theta_symbols:
                return self._evaluate_no_params(expr_str, expr, X, y_obs, data_vars)

            jax_fn = self._build_jax_fn(expr, theta_symbols, data_vars)
            X_jax  = {k: jnp.array(v, dtype=jnp.float64) for k, v in X.items()}
            y_jax  = jnp.array(y_obs, dtype=jnp.float64)

            effective_mode = self._resolve_loss_mode(loss_mode, y_classical)

            y_classical_jax: Optional[jnp.ndarray] = None
            y_full_jax: Optional[jnp.ndarray] = None
            if effective_mode == 'full' and y_classical is not None:
                y_classical_jax = jnp.array(y_classical, dtype=jnp.float64)
                if correction_type == 'multiplicative':
                    y_full_jax = y_classical_jax * (1.0 + y_jax)
                else:
                    y_full_jax = y_classical_jax + y_jax
                logger.debug(
                    f"loss_mode='full' active (dynamic_range={self._dynamic_range(y_classical):.1e}), "
                    f"correction_type='{correction_type}'"
                )

            test_theta = jnp.ones(len(theta_symbols), dtype=jnp.float64)
            if not self._is_finite(jax_fn, test_theta, X_jax, y_jax):
                y_pred_debug = jax_fn(test_theta, X_jax)
                logger.warning(
                    f"[JAXOptimizer] Pre-flight _is_finite FAILED for '{expr_str}' "
                    f"(loss_mode={effective_mode}): y_pred min={float(jnp.min(y_pred_debug))}, "
                    f"max={float(jnp.max(y_pred_debug))}, "
                    f"any_nan={bool(jnp.any(jnp.isnan(y_pred_debug)))}, "
                    f"any_inf={bool(jnp.any(jnp.isinf(y_pred_debug)))}"
                )
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

            if effective_mode == 'full' and y_classical_jax is not None and y_full_jax is not None:
                loss_fn = self._make_full_loss_fn(jax_fn, X_jax, y_classical_jax, y_full_jax, correction_type)
            else:
                loss_fn = self._make_loss_fn(jax_fn, X_jax, y_jax)

            val_and_grad_jit = jit(value_and_grad(loss_fn))
            _ = val_and_grad_jit(test_theta)

            best_theta, best_nmse = self._multi_start_lbfgs(
                val_and_grad_jit, len(theta_symbols), seed, scale
            )

            theta_dict = {str(s): float(best_theta[i]) for i, s in enumerate(theta_symbols)}
            likelihood = float(np.exp(-self.beta * best_nmse)) if np.isfinite(best_nmse) else 0.0

            return OptimizationResult(
                theta      = theta_dict,
                nmse       = best_nmse,
                likelihood = likelihood,
                history    = [best_nmse],
                converged  = best_nmse < CONVERGENCE_TOL,
                n_params   = len(theta_symbols),
                expr_str   = expr_str,
            )

        except Exception as e:
            logger.warning(f"Optimization failed with exception for '{expr_str}': {e}")
            return self._fail_result(expr_str, len(theta_symbols) if 'theta_symbols' in dir() else 0, str(e))

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
                loss_mode=loss_mode, y_classical=y_classical, correction_type=correction_type,
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
        # sympy's Max/Min must be mapped to jnp.maximum/jnp.minimum explicitly;
        # lambdify does not handle jax.numpy as a named translation target.
        hardened_expr = _domain_harden_expr(expr)
        data_syms = sorted([s for s in hardened_expr.free_symbols if not str(s).startswith("theta_")], key=lambda s: str(s))
        all_syms = data_syms + theta_symbols
        _jax_function_mapping = {"Max": jnp.maximum, "Min": jnp.minimum}
        raw_fn = sp.lambdify(all_syms, hardened_expr, modules=[_jax_function_mapping, jnp])

        def jax_fn(theta: jnp.ndarray, X: Dict[str, jnp.ndarray]) -> jnp.ndarray:
            data_vals  = [X[str(sym)] for sym in data_syms]
            theta_vals = [theta[i] for i in range(len(theta_symbols))]
            return raw_fn(*data_vals, *theta_vals)

        return jax_fn

    @staticmethod
    def _dynamic_range(y: np.ndarray) -> float:
        abs_y = np.abs(y)
        return float(np.max(abs_y) / (np.min(abs_y) + 1e-10))

    @classmethod
    def _resolve_loss_mode(cls, loss_mode: str, y_classical: Optional[np.ndarray]) -> str:
        DYNAMIC_RANGE_THRESHOLD = 1e4
        if loss_mode == 'residual':
            return 'residual'
        if loss_mode == 'full':
            return 'full'
        if y_classical is not None and len(y_classical) > 0:
            dr = cls._dynamic_range(y_classical)
            if dr > DYNAMIC_RANGE_THRESHOLD:
                logger.info(
                    f"loss_mode='auto': dynamic_range={dr:.2e} > {DYNAMIC_RANGE_THRESHOLD:.0e} -- "
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

    @staticmethod
    def _nmse_denominator(y: jnp.ndarray) -> jnp.ndarray:
        var_y = jnp.var(y)
        eps = jnp.maximum(var_y * 1e-6, 1e-30)
        return var_y + eps

    def _make_loss_fn(self, jax_fn, X_jax, y_jax):
        denom = self._nmse_denominator(y_jax)

        def loss(theta: jnp.ndarray) -> jnp.ndarray:
            y_pred = jax_fn(theta, X_jax)
            mse    = jnp.mean((y_pred - y_jax) ** 2)
            nmse   = mse / denom
            return jnp.where(jnp.isfinite(nmse), nmse, 1e9)

        return loss

    def _make_full_loss_fn(self, jax_fn, X_jax, y_classical_jax, y_full_jax, correction_type: str):
        denom_full = self._nmse_denominator(y_full_jax)

        if correction_type == 'multiplicative':
            def loss(theta: jnp.ndarray) -> jnp.ndarray:
                delta   = jax_fn(theta, X_jax)
                y_recon = y_classical_jax * (1.0 + delta)
                mse     = jnp.mean((y_recon - y_full_jax) ** 2)
                nmse    = mse / denom_full
                return jnp.where(jnp.isfinite(nmse), nmse, 1e9)
        else:
            def loss(theta: jnp.ndarray) -> jnp.ndarray:
                delta   = jax_fn(theta, X_jax)
                y_recon = y_classical_jax + delta
                mse     = jnp.mean((y_recon - y_full_jax) ** 2)
                nmse    = mse / denom_full
                return jnp.where(jnp.isfinite(nmse), nmse, 1e9)

        return loss

    def _multi_start_lbfgs(
        self, val_and_grad_jit, n_params: int, seed: int, scale: float = 1.0
    ) -> Tuple[np.ndarray, float]:
        """
        RESTORED to be logically identical to the original file you uploaded
        (same 50/50 narrow/wide log-uniform mixture, same log_param branch),
        with ONLY the domain-hardening applied upstream in _build_jax_fn as
        the addition. Nothing about the restart/parameterization logic
        itself was changed from the version that is known to pass Time
        Dilation and Entropy Expansion.
        """
        rng = np.random.RandomState(seed)
        best_theta = np.ones(n_params, dtype=np.float64)
        if n_params > 0:
            best_theta[0] *= scale
        best_nmse = NMSE_FAIL

        init1 = np.ones(n_params, dtype=np.float64)
        if n_params > 0:
            init1[0] *= scale
        init2 = np.ones(n_params, dtype=np.float64) * 0.5
        if n_params > 0:
            init2[0] *= scale
        inits = [init1, init2]

        for _ in range(max(0, self.n_restarts - 2)):
            if rng.choice([True, False]):
                exponents = rng.uniform(-6, 6, size=n_params)
            else:
                exponents = rng.uniform(-20, 20, size=n_params)
            signs = rng.choice([-1, 1], size=n_params)
            init = (signs * (10 ** exponents)).astype(np.float64)
            if n_params > 0:
                init[0] *= scale
            inits.append(init)

        for init in inits:
            if self.log_param:
                signs = np.sign(init)
                signs = np.where(signs == 0, 1.0, signs)
                is_log = np.abs(init) > 1e-30
                u_init = np.where(is_log, np.log(np.maximum(np.abs(init), 1e-30)), init)

                def scipy_obj_scaled(u_np, _signs=signs, _is_log=is_log):
                    theta_np = np.where(_is_log, _signs * np.exp(u_np), u_np)
                    v, g = val_and_grad_jit(jnp.array(theta_np))
                    v_np, g_np = np.array(v), np.array(g)
                    if not np.isfinite(v_np) or not np.all(np.isfinite(g_np)):
                        return 1e10, np.zeros_like(u_np)
                    g_u_np = np.where(_is_log, g_np * theta_np, g_np)
                    return v_np, g_u_np

                res = minimize(
                    scipy_obj_scaled, u_init, method="L-BFGS-B", jac=True,
                    options={"maxiter": self.maxiter, "ftol": 1e-7}
                )
                opt_theta = np.where(is_log, signs * np.exp(res.x), res.x)
            else:
                init_scale = np.where(np.abs(init) > 1e-30, init, 1.0)

                def scipy_obj_scaled(u_np, _init_scale=init_scale):
                    theta_np = u_np * _init_scale
                    v, g = val_and_grad_jit(jnp.array(theta_np))
                    v_np, g_np = np.array(v), np.array(g)
                    if not np.isfinite(v_np) or not np.all(np.isfinite(g_np)):
                        return 1e10, np.zeros_like(u_np)
                    g_u_np = g_np * _init_scale
                    return v_np, g_u_np

                u_init = np.ones_like(init)
                res = minimize(
                    scipy_obj_scaled, u_init, method="L-BFGS-B", jac=True,
                    options={"maxiter": self.maxiter, "ftol": 1e-7}
                )
                opt_theta = res.x * init_scale

            if res.fun < best_nmse and np.isfinite(res.fun):
                best_nmse = float(res.fun)
                best_theta = opt_theta
                if best_nmse < CONVERGENCE_TOL:
                    break

        return best_theta, best_nmse

    def _evaluate_no_params(
        self, expr_str: str, expr: sp.Expr, X: Dict[str, np.ndarray], y_obs: np.ndarray, data_vars: List[str]
    ) -> OptimizationResult:
        try:
            hardened_expr = _domain_harden_expr(expr)
            data_syms = sorted([s for s in hardened_expr.free_symbols if not str(s).startswith("theta_")], key=lambda s: str(s))
            _np_function_mapping = {"Max": np.maximum, "Min": np.minimum}
            f = sp.lambdify(data_syms, hardened_expr, modules=[_np_function_mapping, np])
            y_pred = f(*[X[str(sym)] for sym in data_syms])

            if not isinstance(y_pred, np.ndarray):
                y_pred = np.full(y_obs.shape, float(y_pred))

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
