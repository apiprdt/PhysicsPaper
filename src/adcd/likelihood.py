"""
likelihood.py
=============
Heteroscedastic Gaussian likelihood and intrinsic scatter estimation for the ADCD Identifiability Auditor.

Replaces the brittle chi2_nu gate with full Gaussian likelihood modeling:
- Handles per-point heteroscedastic uncertainties sigma_y.
- Estimates optional physical intrinsic scatter sigma_int via exact score-checked profile likelihood.
- Evaluates extended BIC: k_eff * ln(N) - 2 * ln(L) + 2 * ln(|C|) + prior_penalty.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
import scipy.optimize

logger = logging.getLogger(__name__)


@dataclass
class LikelihoodEvaluationResult:
    """Result of evaluating Gaussian likelihood on residuals."""
    log_likelihood: float
    twice_nll: float
    chi2: float
    sigma_int: float
    n_points: int
    fitted_intrinsic_scatter: bool = False


class HeteroscedasticLikelihood:
    """
    Computes Gaussian likelihood with heteroscedastic observational noise
    and optional physical intrinsic scatter.
    """
    def __init__(self, min_variance: float = 1e-16):
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

        var_tot = np.maximum(self.min_variance, sigma_y ** 2 + float(sigma_int) ** 2)
        res = y_obs - y_pred

        chi2_term = np.sum((res ** 2) / var_tot)
        norm_term = np.sum(np.log(2.0 * np.pi * var_tot))
        log_lik = -0.5 * (chi2_term + norm_term)
        return float(log_lik)

    def fit_intrinsic_scatter(
        self,
        y_obs: np.ndarray,
        y_pred: np.ndarray,
        sigma_y: np.ndarray,
        max_scatter_factor: float = 10.0,
    ) -> float:
        """
        Estimate optimal intrinsic scatter sigma_int >= 0 by maximizing the log-likelihood.
        Uses the exact analytical score derivative at sigma_int^2 = 0 for fast exit on
        nearly homoscedastic data, and bounded profile likelihood optimization otherwise
        to prevent extreme single-point sigma_i^-4 leverage from masking global scatter.
        """
        y_obs = np.asarray(y_obs, dtype=np.float64)
        y_pred = np.asarray(y_pred, dtype=np.float64)
        sigma_y = np.asarray(sigma_y, dtype=np.float64)

        var_meas = np.maximum(self.min_variance, sigma_y ** 2)
        res_sq = (y_obs - y_pred) ** 2

        # Exact derivative d(ln L)/d(sigma_int^2) evaluated at sigma_int^2 = 0:
        # Score_0 = 0.5 * sum((res_i^2 - sigma_i^2) / sigma_i^4)
        score_at_zero = float(np.sum((res_sq - var_meas) / (var_meas ** 2)))
        var_ratio = float(np.max(var_meas) / np.min(var_meas)) if len(var_meas) > 0 else 1.0

        # When variance is nearly uniform across points, log-likelihood is globally concave
        # in sigma_int^2, so a non-positive score at 0 guarantees optimal sigma_int = 0.
        if score_at_zero <= 0.0 and (var_ratio <= 4.0 or float(np.mean(res_sq)) <= float(np.mean(var_meas))):
            return 0.0

        rms_res = float(np.sqrt(np.mean(res_sq)))
        upper_bound = max(1e-6, max_scatter_factor * rms_res)

        def nll_scalar(s_int: float) -> float:
            return -self.compute_log_likelihood(y_obs, y_pred, sigma_y, sigma_int=s_int)

        opt_res = scipy.optimize.minimize_scalar(
            nll_scalar,
            bounds=(0.0, upper_bound),
            method="bounded",
            options={"xatol": 1e-10},
        )

        sigma_int_opt = float(opt_res.x)
        nll_zero = nll_scalar(0.0)
        nll_opt = float(opt_res.fun)

        if nll_zero <= nll_opt or sigma_int_opt < 1e-8:
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
        prior_penalty: float = 0.0,
        prior_Working_penalty: Optional[float] = None,
    ) -> Tuple[float, float, float]:
        """
        Compute extended BIC under full heteroscedastic Gaussian likelihood:
        BIC = k_eff * ln(N) - 2 * ln(L) + 2 * ln(|C|) + prior_penalty

        Returns
        -------
        Tuple of (bic, log_likelihood, sigma_int_opt)
        """
        if prior_Working_penalty is not None:
            prior_penalty = float(prior_Working_penalty)

        N = len(y_obs)
        if fit_intrinsic_scatter:
            sigma_int_opt = self.fit_intrinsic_scatter(y_obs, y_pred, sigma_y)
            k_eff = k_params + 1  # Always penalize the extra degree of freedom when enabled
        else:
            sigma_int_opt = 0.0
            k_eff = k_params

        log_lik = self.compute_log_likelihood(y_obs, y_pred, sigma_y, sigma_int=sigma_int_opt)
        complexity_penalty = float(k_eff * np.log(max(1, N)))
        search_space_penalty = float(2.0 * np.log(max(1, search_space_size)))

        bic = complexity_penalty - 2.0 * log_lik + search_space_penalty + float(prior_penalty)
        return float(bic), float(log_lik), sigma_int_opt

    def evaluate_candidate_likelihood(
        self,
        y_obs: np.ndarray,
        y_pred: np.ndarray,
        sigma_y: np.ndarray,
        k_params: int,
        search_space_size: int = 1,
        fit_intrinsic_scatter: bool = False,
        prior_penalty: float = 0.0,
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
            prior_penalty=prior_penalty,
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


class GaussianLikelihoodEngine:
    """
    Stateful residual-based interface around HeteroscedasticLikelihood used by the audit pipeline.
    """
    def __init__(self, fit_intrinsic_scatter: bool = False, min_variance: float = 1e-16):
        self.fit_intrinsic_scatter = fit_intrinsic_scatter
        self._lik = HeteroscedasticLikelihood(min_variance=min_variance)

    def evaluate(self, residuals: np.ndarray, sigma: np.ndarray) -> LikelihoodEvaluationResult:
        res = np.asarray(residuals, dtype=np.float64)
        sig = np.asarray(sigma, dtype=np.float64)
        zeros = np.zeros_like(res)

        if self.fit_intrinsic_scatter:
            s_int = self._lik.fit_intrinsic_scatter(res, zeros, sig)
        else:
            s_int = 0.0

        log_lik = self._lik.compute_log_likelihood(res, zeros, sig, sigma_int=s_int)
        var_tot = np.maximum(self._lik.min_variance, sig ** 2 + s_int ** 2)
        chi2 = float(np.sum((res ** 2) / var_tot))

        return LikelihoodEvaluationResult(
            log_likelihood=log_lik,
            twice_nll=float(-2.0 * log_lik),
            chi2=chi2,
            sigma_int=s_int,
            n_points=len(res),
            fitted_intrinsic_scatter=self.fit_intrinsic_scatter,
        )

    def compute_bic(
        self,
        lik_res: LikelihoodEvaluationResult,
        k_params: int,
        search_space_size: int = 1,
        prior_penalty: float = 0.0,
    ) -> float:
        k_eff = k_params + (1 if lik_res.fitted_intrinsic_scatter else 0)
        n_pts = max(1, lik_res.n_points)
        c_size = max(1, search_space_size)
        return float(
            k_eff * np.log(n_pts)
            + lik_res.twice_nll
            + 2.0 * np.log(c_size)
            + float(prior_penalty)
        )
