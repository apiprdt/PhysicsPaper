"""
joint_fit.py
============
Joint fitting of baseline parameters (phi) and asymptotic correction parameters (theta).
Computes both the raw Hessian and the unit-invariant Scale-Normalized Hessian (curvature
correlation matrix) to accurately flag genuine phi-theta degeneracies without false
positives from differing physical units.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.optimize

logger = logging.getLogger(__name__)

# Threshold on the scale-normalized Hessian condition number (unit-invariant collinearity)
DEGENERACY_CONDITION_THRESHOLD = 1e4
MAX_CROSS_CORRELATION_THRESHOLD = 0.995
PENALTY_WALL_VALUE = 1e18


@dataclass
class JointFitResult:
    """Result of joint baseline-correction parameter optimization."""
    phi_opt: Dict[str, float]
    theta_opt: Dict[str, float]
    chi2: float
    prior_penalty: float
    total_loss: float
    hessian: np.ndarray
    normalized_hessian: np.ndarray
    condition_number: float
    raw_condition_number: float
    max_phi_theta_correlation: float
    is_degenerate: bool
    converged: bool
    diagnostic_message: str = ""
    param_names: List[str] = field(default_factory=list)

    @property
    def phi_hat(self) -> Dict[str, float]:
        return self.phi_opt

    @property
    def theta_hat(self) -> Dict[str, float]:
        return self.theta_opt


def _compute_numerical_hessian(
    loss_fn: Callable[[np.ndarray], float],
    params: np.ndarray,
    eps: float = 1e-4,
) -> np.ndarray:
    """
    Compute 2nd-order central finite-difference Hessian for arbitrary loss function.
    Uses eps ~ (machine_eps)^(1/4) ~ 1e-4 for optimal float64 2nd-derivative accuracy,
    with adaptive step-shrinking if a perturbation hits an out-of-domain penalty wall.
    """
    n = len(params)
    if n == 0:
        return np.zeros((0, 0), dtype=np.float64)

    H = np.zeros((n, n), dtype=np.float64)
    f0 = loss_fn(params)

    # Step sizes scaled to parameter magnitudes, shrinking if hitting domain boundary wall
    h = np.maximum(np.abs(params) * eps, eps)
    for i in range(n):
        for _ in range(5):
            p_plus = params.copy()
            p_plus[i] += h[i]
            p_minus = params.copy()
            p_minus[i] -= h[i]
            if loss_fn(p_plus) < 0.1 * PENALTY_WALL_VALUE and loss_fn(p_minus) < 0.1 * PENALTY_WALL_VALUE:
                break
            h[i] *= 0.2

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

            val = (loss_fn(p_pp) - loss_fn(p_pm) - loss_fn(p_mp) + loss_fn(p_mm)) / (
                4.0 * h[i] * h[j]
            )
            H[i, j] = val
            H[j, i] = val

    return H


def _analyze_hessian_degeneracy(
    H: np.ndarray,
    n_phi: int,
) -> Tuple[np.ndarray, float, float, float, bool]:
    """
    Analyze Hessian for structural parameter collinearity in a unit-invariant manner.
    Returns (H_norm, norm_cond_num, raw_cond_num, max_cross_corr, is_degenerate).
    """
    n = H.shape[0]
    if n == 0:
        return H, 1.0, 1.0, 0.0, False

    # 1. Raw condition number
    try:
        s_raw = np.linalg.svd(H, compute_uv=False)
        raw_cond = float(np.max(s_raw) / np.min(s_raw)) if np.min(s_raw) > 1e-15 else float("inf")
    except Exception:
        raw_cond = float("inf")

    # 2. Check if any diagonal curvature is non-positive, vanishing (flat direction), or hit penalty wall
    diag = np.diag(H)
    if np.any(diag <= 1e-12) or np.any(diag >= 1e20) or not np.all(np.isfinite(H)):
        return np.eye(n), float("inf"), raw_cond, 1.0, True

    # 3. Scale-normalized Hessian: H_norm[i, j] = H[i, j] / sqrt(H[i, i] * H[j, j])
    scales = np.sqrt(np.abs(diag))
    H_norm = H / np.outer(scales, scales)

    try:
        s_norm = np.linalg.svd(H_norm, compute_uv=False)
        s_min = float(np.min(s_norm))
        s_max = float(np.max(s_norm))
        norm_cond = float(s_max / s_min) if s_min > 1e-14 else float("inf")
    except Exception:
        norm_cond = float("inf")

    # 4. Compute maximum cross-correlation between phi block and theta block
    max_cross_corr = 0.0
    if 0 < n_phi < n:
        # Direct cosine of gradient overlap from normalized Hessian off-diagonal block
        direct_cos_corr = float(np.max(np.abs(H_norm[:n_phi, n_phi:])))
        max_cross_corr = direct_cos_corr
        try:
            cov_norm = np.linalg.pinv(H_norm, rcond=1e-12)
            cov_diag = np.sqrt(np.maximum(1e-15, np.abs(np.diag(cov_norm))))
            corr_matrix = cov_norm / np.outer(cov_diag, cov_diag)
            cross_block = np.abs(corr_matrix[:n_phi, n_phi:])
            max_cross_corr = max(max_cross_corr, float(np.max(cross_block)))
        except Exception:
            max_cross_corr = 1.0

    is_degen = bool(
        norm_cond > DEGENERACY_CONDITION_THRESHOLD
        or max_cross_corr > MAX_CROSS_CORRELATION_THRESHOLD
        or not np.isfinite(norm_cond)
    )
    return H_norm, norm_cond, raw_cond, max_cross_corr, is_degen


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
    bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
) -> JointFitResult:
    """
    Jointly fit baseline constants (phi) and correction parameters (theta) under Gaussian priors.
    """
    phi_keys = sorted(list(init_phi.keys())) if init_phi else []
    theta_keys = sorted(list(init_theta.keys())) if init_theta else []
    param_names = [f"phi_{k}" for k in phi_keys] + [f"theta_{k}" for k in theta_keys]

    n_phi = len(phi_keys)
    priors = phi_priors or {}
    y_obs_arr = np.asarray(y_obs, dtype=np.float64)
    sigma_inv_sq = 1.0 / np.maximum(1e-16, np.asarray(sigma_y, dtype=np.float64) ** 2)

    def unpack_params(p: np.ndarray) -> Tuple[Dict[str, float], Dict[str, float]]:
        phi_dict = {k: float(p[i]) for i, k in enumerate(phi_keys)}
        theta_dict = {k: float(p[n_phi + j]) for j, k in enumerate(theta_keys)}
        return phi_dict, theta_dict

    def joint_loss(p: np.ndarray) -> float:
        phi_dict, theta_dict = unpack_params(p)
        try:
            y_cl = baseline_fn(phi_dict, X)
            delta = corr_fn(theta_dict, X)
            y_pred = y_cl * (1.0 + delta) if mode == "multiplicative" else y_cl + delta

            if np.any(np.isnan(y_pred)) or np.any(np.isinf(y_pred)):
                return PENALTY_WALL_VALUE

            res = y_obs_arr - y_pred
            chi2_val = float(np.sum((res ** 2) * sigma_inv_sq))

            prior_penalty_val = 0.0
            for k, val in phi_dict.items():
                if k in priors:
                    mu, std = priors[k]
                    if std > 0:
                        prior_penalty_val += float(((val - mu) / std) ** 2)

            return chi2_val + prior_penalty_val
        except Exception:
            return PENALTY_WALL_VALUE

    init_p = np.array(
        [init_phi[k] for k in phi_keys] + [init_theta[k] for k in theta_keys],
        dtype=np.float64,
    )

    if len(init_p) > 0:
        opt_res = scipy.optimize.minimize(
            joint_loss,
            init_p,
            method="L-BFGS-B",
            bounds=bounds,
            options={"ftol": 1e-13, "gtol": 1e-9, "maxiter": 1500},
        )
        p_opt = np.asarray(opt_res.x, dtype=np.float64)
        converged = bool(opt_res.success)
        total_loss_final = float(opt_res.fun)
    else:
        p_opt = init_p
        converged = True
        total_loss_final = float(joint_loss(init_p))

    phi_opt, theta_opt = unpack_params(p_opt)

    try:
        y_cl_opt = baseline_fn(phi_opt, X)
        delta_opt = corr_fn(theta_opt, X)
        y_pred_opt = y_cl_opt * (1.0 + delta_opt) if mode == "multiplicative" else y_cl_opt + delta_opt
        chi2_final = float(np.sum(((y_obs_arr - y_pred_opt) ** 2) * sigma_inv_sq))
    except Exception:
        chi2_final = total_loss_final

    prior_penalty_final = 0.0
    for k, val in phi_opt.items():
        if k in priors:
            mu, std = priors[k]
            if std > 0:
                prior_penalty_final += float(((val - mu) / std) ** 2)

    # Compute Hessian and unit-invariant normalized Hessian
    H = _compute_numerical_hessian(joint_loss, p_opt)
    H_norm, norm_cond, raw_cond, max_corr, is_degenerate = _analyze_hessian_degeneracy(H, n_phi)

    diag_msg = (
        f"Joint fit completed (converged={converged}). "
        f"Normalized Hessian cond={norm_cond:.2e} (raw={raw_cond:.2e}), "
        f"max |corr(phi, theta)|={max_corr:.4f}. "
    )
    if is_degenerate:
        diag_msg += (
            f"WARNING: Degeneracy flagged between baseline parameters (phi) "
            f"and correction parameters (theta) (cond={norm_cond:.2e} > {DEGENERACY_CONDITION_THRESHOLD:.1e} "
            f"or |corr|={max_corr:.4f} > {MAX_CROSS_CORRELATION_THRESHOLD})."
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
        normalized_hessian=H_norm,
        condition_number=norm_cond,
        raw_condition_number=raw_cond,
        max_phi_theta_correlation=max_corr,
        is_degenerate=is_degenerate,
        converged=converged,
        diagnostic_message=diag_msg,
        param_names=param_names,
    )
