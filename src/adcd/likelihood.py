"""
likelihood.py
=============
Heteroscedastic Gaussian likelihood and intrinsic scatter estimation for the ADCD Identifiability Auditor.

Replaces the brittle chi2_nu gate with full Gaussian likelihood modeling:
- Handles per-point heteroscedastic uncertainties sigma_y.
- Estimates optional physical intrinsic scatter sigma_int via penalized likelihood.
- Evaluates extended BIC: k_eff * ln(N) - 2 * ln(L) + 2 * ln(|C|).
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Tuple

import numpy as np
import scipy.optimize

logger = logging.getLogger(__name__)


class HeteroscedasticLikelihood:
    """
    Computes Gaussian likelihood with heteroscedastic observational noise
    and optional physical intrinsic scatter.
    """
    def __init__(self, min_variance: float = 1e-14):
        self.min_variance = min_variance

    def compute_log_likelihood(
        self,
        y_obs: np.ndarray,
        y_pred: np.ndarray,
        sigma_y: np.ndarray,
        sigma_int: float = 0.0,
    ) -> float:
        """
        Evaluate full Gaussian log-likelihood:
        ln(L) = -0.5 * sum [ (y_i - y_hat_i)^2 / (sigma_i^2 + sigma_int^2) + ln(2*pi*(sigma_i^2 + sigma_int^2)) ]
        """
        y_obs = np.asarray(y_obs, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        sigma_y = np.asarray(sigma_y, dtype=np.float64)

        var_tot = np.maximum(self.min_variance, sigma_y ** 2 + sigma_int ** 2)
        res = y_obs - y_pred

        chi2_term = np.sum(res ** 2 / var_tot)
        norm_term = np.sum(np.log(2.0 * np.pi * var_tot))
        log_lik = -0.5 * (chi2_term + norm_term)
        return float(log_lik)

    def fit_intrinsic_scatter(
        self,
        y_obs: np.ndarray,
        y_pred: np.ndarray,
        sigma_y: np.ndarray,
        max_scatter_factor: float = 5.0,
    ) -> float:
        """
        Estimate optimal intrinsic scatter sigma_int >= 0 by maximizing the log-likelihood
        (minimizing the negative log-likelihood).
        """
        y_obs = np.asarray(y_obs, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        sigma_y = np.asarray(sigma_y, dtype=np.float64)

        res_sq = (y_obs - y_pred) ** 2
        rms_res = float(np.sqrt(np.mean(res_sq)))
        mean_meas_sigma = float(np.mean(sigma_y))

        # If data scatter is already below measurement noise, sigma_int = 0
        if rms_res <= mean_meas_sigma:
            return 0.0

        def nll(log_s_int_arr: np.ndarray) -> float:
            s_int = float(np.exp(log_s_int_arr[0]))
            return -self.compute_log_likelihood(y_obs, y_pred, sigma_y, sigma_int=s_int)

        # Optimize log(sigma_int) to ensure positivity
        init_log_s = float(np.log(max(1e-6, np.sqrt(max(0.0, rms_res ** 2 - mean_meas_sigma ** 2)))))
        bounds = [(np.log(1e-8), np.log(max(1e-6, max_scatter_factor * rms_res)))]

        opt = scipy.optimize.minimize(nll, [init_log_s], bounds=bounds, method="L-BFGS-B")
        sigma_int_opt = float(np.exp(opt.x[0]))

        # Check if likelihood at sigma_int=0 is better or equal
        nll_zero = -self.compute_log_likelihood(y_obs, y_pred, sigma_y, sigma_int=0.0)
        nll_opt = float(opt.fun)

        if nll_zero <= nll_opt or sigma_int_opt < 1e-6:
            return 0.0

        return sigma_int_opt

    def compute_bic(
        self,
        y_obs: np.ndarray,
        y_pred: np.ndarray,
        sigma_y: np.ndarray,
        k_params: int,
        search_space_size: int = 1,
        fit_intrinsic_scatter: bool = False,
    ) -> Tuple[float, float, float]:
        """
        Compute extended BIC under full heteroscedastic Gaussian likelihood:
        BIC = k_eff * ln(N) - 2 * ln(L) + 2 * ln(|C|)

        Returns
        -------
        Tuple of (bic, log_likelihood, sigma_int_opt)
        """
        N = len(y_obs)
        if fit_intrinsic_scatter:
            sigma_int_opt = self.fit_intrinsic_scatter(y_obs, y_pred, sigma_y)
            k_eff = k_params + 1  # 1 parameter for sigma_int
        else:
            sigma_int_opt = 0.0
            k_eff = k_params

        log_lik = self.compute_log_likelihood(y_obs, y_pred, sigma_y, sigma_int=sigma_int_opt)
        complexity_penalty = float(k_eff * np.log(max(1, N)))
        search_space_penalty = float(2.0 * np.log(max(1, search_space_size)))

        bic = complexity_penalty - 2.0 * log_lik + search_space_penalty
        return float(bic), float(log_lik), sigma_int_opt

    def evaluate_candidate_likelihood(
        self,
        y_obs: np.ndarray,
        y_pred: np.ndarray,
        sigma_y: np.ndarray,
        k_params: int,
        search_space_size: int = 1,
        fit_intrinsic_scatter: bool = False,
    ) -> Dict[str, Any]:
        """
        Full statistical evaluation without hard chi2_nu gate.
        """
        N = len(y_obs)
        bic, log_lik, sigma_int = self.compute_bic(
            y_obs=y_obs,
            y_pred=y_pred,
            sigma_y=sigma_y,
            k_params=k_params,
            search_space_size=search_space_size,
            fit_intrinsic_scatter=fit_intrinsic_scatter,
        )

        res = y_obs - y_pred
        chi2_raw = float(np.sum((res / np.maximum(1e-12, sigma_y)) ** 2))
        dof = max(1, N - k_params)
        chi2_nu = float(chi2_raw / dof)

        return {
            "bic": bic,
            "log_likelihood": log_lik,
            "sigma_int": sigma_int,
            "chi2_raw": chi2_raw,
            "chi2_nu": chi2_nu,
            "k_params": k_params,
            "n_points": N,
            "status": "EVALUATED",
        }
