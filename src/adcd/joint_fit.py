"""
joint_fit.py
============
Joint fitting of baseline parameters (phi) and asymptotic correction parameters (theta).
Computes the joint Hessian matrix, calculates the condition number, and flags phi-theta degeneracies.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.optimize

logger = logging.getLogger(__name__)

DEGENERACY_CONDITION_THRESHOLD = 1e6


@dataclass
class JointFitResult:
    """Result of joint baseline-correction parameter optimization."""
    phi_opt: Dict[str, float]
    theta_opt: Dict[str, float]
    chi2: float
    prior_penalty: float
    total_loss: float
    hessian: np.ndarray
    condition_number: float
    is_degenerate: bool
    converged: bool
    diagnostic_message: str = ""
    param_names: List[str] = field(default_factory=list)


def _compute_numerical_hessian(
    loss_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    eps: float = 1e-5,
) -> np.ndarray:
    """
    Compute 2nd-order central finite-difference Hessian for arbitrary loss function.
    """
    n = len(params)
    H = np.zeros((n, n), dtype=np.float64)
    f0 = loss_fn(params)

    # Step sizes scaled to parameter magnitudes
    h = np.maximum(np.abs(params) * eps, eps)

    for i in range(n):
        p_plus_i = params.copy()
        p_plus_i[i] += h[i]
        f_plus_i = loss_fn(p_plus_i)

        p_minus_i = params.copy()
        p_minus_i[i] -= h[i]
        f_minus_i = loss_fn(p_minus_i)

        # Diagonal entries
        H[i, i] = (f_plus_i - 2.0 * f0 + f_minus_i) / (h[i] ** 2)

        # Off-diagonal entries
        for j in range(i + 1, n):
            p_pp = params.copy()
            p_pp[i] += h[i]
            p_pp[j] += h[j]

            p_pm = params.copy()
            p_pm[i] += h[i]
            p_pm[j] -= h[j]

            p_mp = params.copy()
            p_mp[i] -= h[i]
            p_mp[j] += h[j]

            p_mm = params.copy()
            p_mm[i] -= h[i]
            p_mm[j] -= h[j]

            f_pp = loss_fn(p_pp)
            f_pm = loss_fn(p_pm)
            f_mp = loss_fn(p_mp)
            f_mm = loss_fn(p_mm)

            val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h[i] * h[j])
            H[i, j] = val
            H[j, i] = val

    return H


def fit_joint_baseline_correction(
    baseline_fn: Callable[[Dict[str, float], Dict[str, np.ndarray]], np.ndarray],
    corr_fn: Callable[[Dict[str, float], Dict[str, np.ndarray]], np.ndarray],
    X: Dict[str, np.ndarray],
    y_obs: np.ndarray,
    sigma_y: np.ndarray,
    phi_priors: Optional[Dict[str, Tuple[float, float]]] = None,
    init_phi: Optional[Dict[str, float]] = None,
    init_theta: Optional[Dict[str, float]] = None,
    mode: str = "multiplicative",
) -> JointFitResult:
    """
    Jointly fit baseline constants (phi) and correction parameters (theta) under Gaussian priors.

    Parameters
    ----------
    baseline_fn : Callable mapping (phi_dict, X) -> y_classical
    corr_fn : Callable mapping (theta_dict, X) -> Delta
    X : Dictionary of input variable arrays
    y_obs : Observed target array
    sigma_y : Measurement uncertainty array
    phi_priors : Dictionary of {param_name: (mean, std)} for baseline constants
    init_phi : Initial guesses for phi
    init_theta : Initial guesses for theta
    mode : 'multiplicative' (y = y_cl * (1 + Delta)) or 'additive' (y = y_cl + Delta)
    """
    phi_keys = sorted(list(init_phi.keys())) if init_phi else []
    theta_keys = sorted(list(init_theta.keys())) if init_theta else []
    param_names = [f"phi_{k}" for k in phi_keys] + [f"theta_{k}" for k in theta_keys]

    n_phi = len(phi_keys)

    priors = phi_priors or {}
    sigma_inv_sq = 1.0 / np.maximum(1e-14, np.asarray(sigma_y, dtype=np.float64) ** 2)

    def unpack_params(p: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        phi_dict = {k: float(p[i]) for i, k in enumerate(phi_keys)}
        theta_dict = {k: float(p[n_phi + j]) for j, k in enumerate(theta_keys)}
        return phi_dict, theta_dict

    def joint_loss(p: np.ndarray) -> float:
        phi_dict, theta_dict = unpack_params(p)
        try:
            y_cl = baseline_fn(phi_dict, X)
            delta = corr_fn(theta_dict, X)
            if mode == "multiplicative":
                y_pred = y_cl * (1.0 + delta)
            else:
                y_pred = y_cl + delta

            res = y_obs - y_pred
            chi2_val = float(np.sum(res ** 2 * sigma_inv_sq))

            # Prior penalty on phi: sum ((phi_j - mu_j) / sigma_phi_j)^2
            prior_penalty_val = 0.0
            for k, val in phi_dict.items():
                if k in priors:
                    mu, std = priors[k]
                    prior_penalty_val += float(((val - mu) / std) ** 2)

            return chi2_val + prior_penalty_val
        except Exception:
            return 1e12

    init_p = np.array(
        [init_phi[k] for k in phi_keys] + [init_theta[k] for k in theta_keys],
        dtype=np.float64,
    )

    opt_res = scipy.optimize.minimize(
        joint_loss,
        init_p,
        method="L-BFGS-B",
        options={"ftol": 1e-12, "gtol": 1e-8, "maxiter": 1000},
    )

    p_opt = opt_res.x
    phi_opt, theta_opt = unpack_params(p_opt)

    # Compute components at minimum
    try:
        y_cl_opt = baseline_fn(phi_opt, X)
        delta_opt = corr_fn(theta_opt, X)
        y_pred_opt = y_cl_opt * (1.0 + delta_opt) if mode == "multiplicative" else y_cl_opt + delta_opt
        chi2_final = float(np.sum((y_obs - y_pred_opt) ** 2 * sigma_inv_sq))
    except Exception:
        chi2_final = float(opt_res.fun)

    prior_penalty_final = 0.0
    for k, val in phi_opt.items():
        if k in priors:
            mu, std = priors[k]
            prior_penalty_final += float(((val - mu) / std) ** 2)

    total_loss_final = float(opt_res.fun)

    # Compute Hessian matrix at minimum
    H = _compute_numerical_hessian(joint_loss, p_opt)

    # Condition number analysis via SVD
    try:
        s = np.linalg.svd(H, compute_uv=False)
        s_max = float(np.max(s))
        s_min = float(np.min(s))
        if s_min > 1e-12 and np.isfinite(s_max):
            cond_num = float(s_max / s_min)
        else:
            cond_num = float("inf")
    except Exception:
        cond_num = float("inf")

    is_degenerate = bool(cond_num > DEGENERACY_CONDITION_THRESHOLD or not np.isfinite(cond_num))

    diag_msg = (
        f"Joint fit completed (converged={opt_res.success}). "
        f"Hessian cond={cond_num:.2e}. "
    )
    if is_degenerate:
        diag_msg += (
            f"WARNING: Degeneracy flagged between baseline parameters (phi) and correction parameters (theta). "
            f"Condition number {cond_num:.2e} exceeds threshold {DEGENERACY_CONDITION_THRESHOLD:.1e}."
        )
    else:
        diag_msg += "Parameters are identifiable and well-conditioned."

    return JointFitResult(
        phi_opt=phi_opt,
        theta_opt=theta_opt,
        chi2=chi2_final,
        prior_penalty=prior_penalty_final,
        total_loss=total_loss_final,
        hessian=H,
        condition_number=cond_num,
        is_degenerate=is_degenerate,
        converged=bool(opt_res.success),
        diagnostic_message=diag_msg,
        param_names=param_names,
    )
