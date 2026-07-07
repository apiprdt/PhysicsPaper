"""
Identifiability analysis for discovered corrections.

Diagnoses WHY a correction cannot be uniquely identified from data,
converting "system failed" narratives into precise, scientifically
meaningful statements about data information limits.

Three failure modes diagnosed:
1. model_degeneracy:        top-2 candidates statistically indistinguishable
2. low_snr:                 correction signal buried in measurement noise
3. undetectable_magnitude:  correction too small relative to classical prediction

Reference:
    Fisher (1925) Statistical Methods for Research Workers -- identifiability
    Rissanen (1978) Modeling by shortest data description -- MDL
    Physics context: screened Coulomb vs Yukawa are indistinguishable below SNR threshold
"""

from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import numpy as np


@dataclass
class IdentifiabilityReport:
    """
    Identifiability analysis result for a discovered correction.

    Attributes:
        is_identifiable: True if correction can be reliably identified
        failure_mode: "model_degeneracy" | "low_snr" | "undetectable_magnitude" | None
        snr: Signal-to-noise ratio of the correction (correction_std / noise_std)
        weight_ratio: Posterior weight ratio of top-2 candidates (inf if only 1 candidate)
        relative_magnitude: Median of |delta| / |y_classical| across data points
        summary: Human-readable summary for paper reporting
        parameter_uncertainties: Optional dict of standard errors of parameter values
        degenerate_parameter_pairs: Optional list of highly correlated parameter name pairs
    """
    is_identifiable: bool
    failure_mode: Optional[str]
    snr: float
    weight_ratio: float
    relative_magnitude: float
    summary: str
    parameter_uncertainties: Optional[Dict[str, float]] = None
    degenerate_parameter_pairs: Optional[List[Tuple[str, str]]] = None


class IdentifiabilityAnalyzer:
    """
    Analyzes whether the discovered correction is identifiable from data.

    A correction is identifiable if:
    1. It is large enough to be detected above noise (SNR > SNR_THRESHOLD)
    2. Its relative magnitude is non-negligible (> MAGNITUDE_THRESHOLD)
    3. The posterior unambiguously favors one candidate over others
       (weight_ratio > WEIGHT_RATIO_THRESHOLD)

    Inspired by:
    - Fisher information / Cramer-Rao bound (parameter identifiability)
    - Rissanen's MDL (information-theoretic identifiability)
    - Physics: indistinguishability below SNR threshold
    """

    SNR_THRESHOLD = 1.0
    WEIGHT_RATIO_THRESHOLD = 3.0
    MAGNITUDE_THRESHOLD = 1e-10

    def analyze(
        self,
        bayesian_output,          # BayesianCorrectionOutput
        residual: np.ndarray,
        y_classical: np.ndarray,
        noise_level: float = 0.0,
        X_data: Optional[Dict[str, np.ndarray]] = None,
        data_vars: Optional[List[str]] = None,
    ) -> IdentifiabilityReport:
        """
        Analyze identifiability of a discovered correction.

        Args:
            bayesian_output: BayesianCorrectionOutput from BayesianReranker.rank()
            residual: Array of residual values (observed - classical)
            y_classical: Array of classical model predictions
            noise_level: Fractional noise level (e.g. 0.05 for 5% noise)
            X_data: Optional dictionary of variables
            data_vars: List of variable names
        """
        residual = np.asarray(residual, dtype=float)
        y_classical = np.asarray(y_classical, dtype=float)

        # --- 1. Compute SNR ---
        correction_magnitude = float(np.std(residual))
        noise_magnitude = float(noise_level * np.std(y_classical)) + 1e-15
        snr = correction_magnitude / noise_magnitude

        # --- 2. Compute relative magnitude ---
        relative_magnitude = float(
            np.median(np.abs(residual) / (np.abs(y_classical) + 1e-15))
        )

        # --- 3. Compute posterior weight ratio ---
        weights = bayesian_output.posterior_weights
        if len(weights) >= 2:
            weight_ratio = float(weights[0] / (weights[1] + 1e-15))
        else:
            weight_ratio = float("inf")

        # --- 4. Parameter Uncertainty & Correlation Check ---
        parameter_uncertainties: Optional[Dict[str, float]] = None
        degenerate_parameter_pairs: List[Tuple[str, str]] = []
        
        # Analyze parameters of the top candidate
        top_cand = bayesian_output.candidates[0] if len(bayesian_output.candidates) > 0 else None
        if top_cand and X_data is not None and data_vars is not None:
            expr_str = top_cand.expr_str
            theta_opt = top_cand.theta
            if theta_opt:
                cov, corr = self._compute_covariance(expr_str, theta_opt, X_data, y_classical + residual, data_vars)
                if cov is not None:
                    p_names = sorted(list(theta_opt.keys()))
                    std_errs = np.sqrt(np.maximum(np.diagonal(cov), 1e-30))
                    parameter_uncertainties = {p_names[i]: float(std_errs[i]) for i in range(len(p_names))}
                    
                    # Detect degenerate pairs (|r| > 0.95)
                    if corr is not None:
                        for i in range(len(p_names)):
                            for j in range(i + 1, len(p_names)):
                                if abs(corr[i, j]) > 0.95:
                                    degenerate_parameter_pairs.append((p_names[i], p_names[j]))

        # --- 5. Diagnose failure mode (ordered by severity) ---
        failure_mode: Optional[str] = None

        if relative_magnitude < self.MAGNITUDE_THRESHOLD:
            failure_mode = "undetectable_magnitude"
        elif snr < self.SNR_THRESHOLD:
            failure_mode = "low_snr"
        elif weight_ratio < self.WEIGHT_RATIO_THRESHOLD or len(degenerate_parameter_pairs) > 0:
            failure_mode = "model_degeneracy"

        is_identifiable = failure_mode is None

        # --- 6. Generate human-readable summary ---
        summary = self._build_summary(
            is_identifiable, failure_mode, snr, weight_ratio, relative_magnitude
        )
        if degenerate_parameter_pairs:
            pair_strs = [f"({p1}, {p2})" for p1, p2 in degenerate_parameter_pairs]
            summary += f" | Warning: Degenerate parameter pairs detected: {', '.join(pair_strs)}"

        return IdentifiabilityReport(
            is_identifiable=is_identifiable,
            failure_mode=failure_mode,
            snr=snr,
            weight_ratio=weight_ratio,
            relative_magnitude=relative_magnitude,
            summary=summary,
            parameter_uncertainties=parameter_uncertainties,
            degenerate_parameter_pairs=degenerate_parameter_pairs if len(degenerate_parameter_pairs) > 0 else None,
        )

    def _compute_covariance(
        self,
        expr_str: str,
        theta_opt: Dict[str, float],
        X: Dict[str, np.ndarray],
        y_obs: np.ndarray,
        data_vars: List[str],
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Compute numeric parameter covariance and correlation matrices via local linearization."""
        import sympy as sp
        
        n_points = len(y_obs)
        p_names = sorted(list(theta_opt.keys()))
        n_params = len(p_names)
        
        if n_params == 0:
            return None, None
            
        sym_locals = {}
        for v in data_vars:
            sym_locals[v] = sp.Symbol(v)
        for p in p_names:
            sym_locals[p] = sp.Symbol(p)
            
        try:
            expr = sp.sympify(expr_str, locals=sym_locals)
            clean_vars = [v for v in data_vars if not v.startswith("theta_")]
            sym_order = [sp.Symbol(v) for v in clean_vars] + [sp.Symbol(p) for p in p_names]
            f_eval = sp.lambdify(sym_order, expr, modules=["numpy"])
        except Exception:
            return None, None
            
        def predict(theta_vals):
            args = [X[v] for v in clean_vars] + list(theta_vals)
            return np.asarray(f_eval(*args), dtype=float)
            
        try:
            theta_vals = np.array([theta_opt[p] for p in p_names], dtype=float)
            y_pred = predict(theta_vals)
            
            eps = 1e-6
            J = np.zeros((n_points, n_params))
            for j in range(n_params):
                theta_eps = theta_vals.copy()
                step = eps * max(abs(theta_vals[j]), 1.0)
                theta_eps[j] += step
                y_pred_eps = predict(theta_eps)
                J[:, j] = (y_pred_eps - y_pred) / step
                
            JTJ = J.T @ J
            JTJ += np.eye(n_params) * 1e-12
            JTJ_inv = np.linalg.inv(JTJ)
            
            residuals = y_obs - y_pred
            dof = max(n_points - n_params, 1)
            sigma2 = np.sum(residuals ** 2) / dof
            
            cov_matrix = sigma2 * JTJ_inv
            std_errs = np.sqrt(np.maximum(np.diagonal(cov_matrix), 1e-30))
            cor_matrix = cov_matrix / np.outer(std_errs, std_errs)
            return cov_matrix, cor_matrix
        except Exception:
            return None, None

    def _build_summary(
        self,
        is_identifiable: bool,
        failure_mode: Optional[str],
        snr: float,
        weight_ratio: float,
        relative_magnitude: float,
    ) -> str:
        if is_identifiable:
            return (
                f"Correction is identifiable: SNR={snr:.2f}, "
                f"weight_ratio={weight_ratio:.1f}, "
                f"relative_magnitude={relative_magnitude:.2e}"
            )
        elif failure_mode == "undetectable_magnitude":
            return (
                f"Correction undetectable: relative magnitude {relative_magnitude:.2e} "
                f"is below threshold {self.MAGNITUDE_THRESHOLD:.0e}. "
                f"Classical model already explains data completely."
            )
        elif failure_mode == "low_snr":
            return (
                f"Correction not identifiable: SNR={snr:.2f} < {self.SNR_THRESHOLD} "
                f"(correction magnitude {relative_magnitude:.2e} relative to classical). "
                f"More precise measurements needed."
            )
        elif failure_mode == "model_degeneracy":
            wr_str = f"{weight_ratio:.1f}" if np.isfinite(weight_ratio) else "inf"
            return (
                f"Model degeneracy: posterior weight ratio={wr_str} < "
                f"{self.WEIGHT_RATIO_THRESHOLD} (ambiguous between top candidates). "
                f"SNR={snr:.2f} is sufficient but data cannot distinguish functional forms."
            )
        else:
            return "Unknown identifiability status."
