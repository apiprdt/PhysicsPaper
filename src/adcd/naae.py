"""
naae.py
=======
Noise-Aware Asymptotic Estimator (NAAE) for ADCD.

Mitigates logarithmic bias in standard Residual Asymptotic Scaling (RAS)
by performing a joint non-linear fit in the linear physical space rather than log-space.
"""

import numpy as np
import logging
from typing import Dict, Any, Optional
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

def estimate_exponent_naae(
    x_vals: np.ndarray,
    y_obs: np.ndarray,
    y_classical: np.ndarray,
    correction_type: str = "additive",
    limit_direction: str = "0",
) -> Dict[str, Any]:
    """
    Perform a joint non-linear fit to estimate the leading scaling exponent gamma
    and prefactor A directly in the physical space to avoid log-transform noise bias.
    
    Model:
      additive:       y_obs = y_classical + A * x**gamma
      multiplicative: y_obs = y_classical * (1.0 + A * x**gamma)
      
    Returns:
      dict containing:
        - leading_exponent (float or None): estimated gamma
        - prefactor (float or None): estimated A
        - fit_quality (float): R^2 of the fit
        - degenerate (bool): True if gamma matches baseline scaling (leading to parameter degeneracy)
    """
    x = np.asarray(x_vals, dtype=float)
    y = np.asarray(y_obs, dtype=float)
    y_cls = np.asarray(y_classical, dtype=float)
    
    n_points = len(x)
    if n_points < 4:
        return {"leading_exponent": None, "prefactor": None, "fit_quality": 0.0, "degenerate": False}
        
    # Select data near the classical limit boundary using a percentile threshold.
    # This is robust to non-uniform (e.g. log-spaced or clustered) x distributions,
    # unlike the previously hard-coded 60% index slice.
    abs_x = np.abs(x)
    if limit_direction == "0":
        # Select points where |x| <= 60th percentile of |x| (closest to the zero limit)
        threshold = np.percentile(abs_x, 60)
        idx = np.where(abs_x <= threshold)[0]
    else:  # "oo"
        # Select points where |x| >= 40th percentile of |x| (closest to the infinity limit)
        threshold = np.percentile(abs_x, 40)
        idx = np.where(abs_x >= threshold)[0]
        
    x_fit = x[idx]
    y_fit = y[idx]
    y_cls_fit = y_cls[idx]
    
    # Avoid fitting on points where x_fit is zero to prevent division by zero or log domain errors.
    valid = np.abs(x_fit) > 1e-15
    x_fit = x_fit[valid]
    y_fit = y_fit[valid]
    y_cls_fit = y_cls_fit[valid]
    
    if len(x_fit) < 3:
        return {"leading_exponent": None, "prefactor": None, "fit_quality": 0.0, "degenerate": False}

    # Extract target delta to fit
    if correction_type == "multiplicative":
        # delta = y_obs / y_classical - 1
        denom = np.where(np.abs(y_cls_fit) > 1e-15, y_cls_fit, 1.0)
        delta_fit = (y_fit / denom) - 1.0
    else:
        # delta = y_obs - y_classical
        delta_fit = y_fit - y_cls_fit

    # Define loss function: MSE of (A * x**gamma) vs delta_fit
    def loss_fn(params):
        A, gamma = params
        pred = A * (x_fit ** gamma)
        if not np.all(np.isfinite(pred)):
            return 1e10
        return np.mean((pred - delta_fit) ** 2)

    # Multi-start initial guesses to avoid local minima
    best_loss = float("inf")
    best_params = (0.0, 0.0)
    
    # Estimate scaling of A from delta_fit mean
    mean_delta = np.mean(np.abs(delta_fit))
    a_scales = [mean_delta, -mean_delta, 1.0, -1.0]
    # Extended gamma guesses covering physically motivated scalings:
    # -4 (inverse quartic), -7/3 (GW Peters), -2, -1, 0.5, 1, 2, 3, 4 (Stefan-Boltzmann), 5
    gamma_guesses = [-4.0, -7.0/3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0]

    initial_guesses = []
    for a_s in a_scales:
        for g_g in gamma_guesses:
            initial_guesses.append((a_s, g_g))

    # Extended bounds to cover T^4 (Stefan-Boltzmann), 5PN relativistic scaling,
    # and cosmological power laws up to gamma = ±8.
    # Prefactor bound is data-driven: ±1000 * max(|delta|) to avoid constraint saturation.
    max_delta = max(np.max(np.abs(delta_fit)), 1.0)
    bounds = [(-1000.0 * max_delta, 1000.0 * max_delta), (-8.0, 8.0)]
    
    for guess in initial_guesses:
        res = minimize(loss_fn, np.array(guess), method="L-BFGS-B", bounds=bounds, options={"maxiter": 100})
        if res.fun < best_loss:
            best_loss = res.fun
            best_params = tuple(res.x)

    A_est, gamma_est = best_params
    
    # Calculate fit quality (R^2 of the delta_fit representation)
    pred_final = A_est * (x_fit ** gamma_est)
    ss_res = np.sum((delta_fit - pred_final) ** 2)
    ss_tot = np.sum((delta_fit - np.mean(delta_fit)) ** 2)
    
    fit_quality = 1.0 - (ss_res / (ss_tot + 1e-15))
    fit_quality = float(np.clip(fit_quality, 0.0, 1.0))
    
    # Degeneracy check: is the estimated exponent close to the leading exponent of y_classical?
    # This is a warning system for Nonlinear Drag where baseline and anomaly collapse.
    degenerate = False
    try:
        # Estimate power of classical baseline near the limit
        # y_classical ~ C * x**gamma_cls
        log_x = np.log(np.abs(x_fit))
        log_y_cls = np.log(np.abs(y_cls_fit) + 1e-30)
        # Linear fit to log-log data to estimate baseline power
        slope, _ = np.polyfit(log_x, log_y_cls, 1)
        if np.abs(gamma_est - slope) < 0.25:
            degenerate = True
    except Exception:
        pass

    return {
        "leading_exponent": float(gamma_est) if np.isfinite(gamma_est) else None,
        "prefactor": float(A_est) if np.isfinite(A_est) else None,
        "fit_quality": fit_quality,
        "degenerate": degenerate,
    }
