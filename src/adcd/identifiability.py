from dataclasses import dataclass
from typing import Optional, Dict, Tuple, List
import numpy as np


@dataclass
class IdentifiabilityReport:
    is_identifiable: bool
    failure_mode: Optional[str]
    snr: float
    weight_ratio: float
    relative_magnitude: float
    summary: str
    parameter_uncertainties: Optional[Dict[str, float]] = None
    degenerate_parameter_pairs: Optional[List[Tuple[str, str]]] = None


class IdentifiabilityAnalyzer:

    SNR_THRESHOLD = 1.0
    WEIGHT_RATIO_THRESHOLD = 3.0
    MAGNITUDE_THRESHOLD = 1e-10

    def analyze(
        self,
        bayesian_output,
        residual: np.ndarray,
        y_classical: np.ndarray,
        noise_level: float = 0.0,
        X_data: Optional[Dict[str, np.ndarray]] = None,
        data_vars: Optional[List[str]] = None,
        correction_type: str = "additive",   # NEW: needed to reconstruct y_obs correctly
    ) -> IdentifiabilityReport:
        residual = np.asarray(residual, dtype=float)
        y_classical = np.asarray(y_classical, dtype=float)

        # --- FIXED: reconstruct the actual observed signal to reference the
        # noise scale against, instead of the (possibly constant/zero-variance)
        # classical baseline alone.
        if correction_type == "multiplicative":
            y_obs_reconstructed = y_classical * (1.0 + residual)
        else:
            y_obs_reconstructed = y_classical + residual

        reference_std = float(np.std(y_obs_reconstructed))
        degenerate_reference = reference_std < 1e-12

        correction_magnitude = float(np.std(residual))

        if degenerate_reference:
            # Cannot honestly estimate a noise scale from a dataset with (near)
            # zero variance in the observed signal. Do NOT fabricate an
            # infinite SNR -- report the failure explicitly.
            snr = 0.0
        else:
            noise_magnitude = float(noise_level * reference_std) + 1e-15
            snr = correction_magnitude / noise_magnitude

        relative_magnitude = float(
            np.median(np.abs(residual) / (np.abs(y_classical) + 1e-15))
        )

        weights = bayesian_output.posterior_weights
        if len(weights) >= 2:
            weight_ratio = float(weights[0] / (weights[1] + 1e-15))
        else:
            weight_ratio = float("inf")

        parameter_uncertainties: Optional[Dict[str, float]] = None
        degenerate_parameter_pairs: List[Tuple[str, str]] = []

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
                    if corr is not None:
                        for i in range(len(p_names)):
                            for j in range(i + 1, len(p_names)):
                                if abs(corr[i, j]) > 0.95:
                                    degenerate_parameter_pairs.append((p_names[i], p_names[j]))

        failure_mode: Optional[str] = None

        if degenerate_reference:
            failure_mode = "degenerate_reference"
        elif relative_magnitude < self.MAGNITUDE_THRESHOLD:
            failure_mode = "undetectable_magnitude"
        elif snr < self.SNR_THRESHOLD:
            failure_mode = "low_snr"
        elif weight_ratio < self.WEIGHT_RATIO_THRESHOLD:
            failure_mode = "posterior_ambiguity"
        elif len(degenerate_parameter_pairs) > 0:
            failure_mode = "parameter_degeneracy"

        is_identifiable = failure_mode is None

        summary = self._build_summary(is_identifiable, failure_mode, snr, weight_ratio, relative_magnitude)
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
            degenerate_parameter_pairs=degenerate_parameter_pairs if degenerate_parameter_pairs else None,
        )

    def _compute_covariance(self, expr_str, theta_opt, X, y_obs, data_vars):
        import sympy as sp

        n_points = len(y_obs)
        p_names = sorted(list(theta_opt.keys()))
        n_params = len(p_names)
        if n_params == 0:
            return None, None

        sym_locals = {v: sp.Symbol(v) for v in data_vars}
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
            if np.ndim(y_pred) == 0 or len(np.atleast_1d(y_pred)) == 1:
                y_pred = np.full(n_points, float(np.squeeze(y_pred)))

            _EPS_MACH = np.sqrt(np.finfo(float).eps)
            J = np.zeros((n_points, n_params))
            for j in range(n_params):
                theta_eps = theta_vals.copy()
                step = _EPS_MACH * max(abs(theta_vals[j]), 1.0)
                theta_eps[j] += step
                y_pred_eps = predict(theta_eps)
                if np.ndim(y_pred_eps) == 0 or len(np.atleast_1d(y_pred_eps)) == 1:
                    y_pred_eps = np.full(n_points, float(np.squeeze(y_pred_eps)))
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

    def _build_summary(self, is_identifiable, failure_mode, snr, weight_ratio, relative_magnitude) -> str:
        if is_identifiable:
            return (
                f"Correction is identifiable: SNR={snr:.2f}, "
                f"weight_ratio={weight_ratio:.1f}, "
                f"relative_magnitude={relative_magnitude:.2e}"
            )
        if failure_mode == "degenerate_reference":
            return (
                "Identifiability cannot be assessed: the reconstructed "
                "observed signal has (near) zero variance, so no meaningful "
                "noise scale can be estimated. This is NOT the same as "
                "'identifiable' -- report this scenario as inconclusive, "
                "not as a pass."
            )
        if failure_mode == "undetectable_magnitude":
            return (
                f"Correction undetectable: relative magnitude {relative_magnitude:.2e} "
                f"is below threshold {self.MAGNITUDE_THRESHOLD:.0e}. "
                f"Classical model already explains data completely."
            )
        if failure_mode == "low_snr":
            return (
                f"Correction not identifiable: SNR={snr:.2f} < {self.SNR_THRESHOLD} "
                f"(correction magnitude {relative_magnitude:.2e} relative to classical). "
                f"More precise measurements needed."
            )
        if failure_mode == "posterior_ambiguity":
            wr_str = f"{weight_ratio:.1f}" if np.isfinite(weight_ratio) else "inf"
            return (
                f"Posterior ambiguity: weight ratio={wr_str} < "
                f"{self.WEIGHT_RATIO_THRESHOLD} (data cannot distinguish competing functional forms). "
                f"SNR={snr:.2f} is sufficient; more diverse data geometry is needed."
            )
        if failure_mode == "parameter_degeneracy":
            return (
                f"Parameter degeneracy: best candidate dominates posterior (weight_ratio={weight_ratio:.1f}) "
                f"but contains linearly dependent parameters. Model is over-parameterized for available data."
            )
        return "Unknown identifiability status."
