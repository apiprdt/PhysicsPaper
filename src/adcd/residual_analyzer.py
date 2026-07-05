import logging
import numpy as np
import scipy.stats as stats
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

@dataclass
class ResidualFeatures:
    """
    Features extracted from physical residuals.
    
    Attributes:
        monotonicity: Spearman correlation of residual vs primary variable
        curvature_sign: Sign of quadratic coefficient in polynomial fit
        oscillation_score: Detrended zero-crossing density
        decay_rate: R² value of exponential decay fit
        symmetry: Relative alignment to even vs odd polynomial bases
    """
    monotonicity: float      # Spearman correlation of residual vs primary variable
    curvature_sign: float    # Sign of quadratic coefficient in polynomial fit
    oscillation_score: float # Number of zero-crossings of smoothed/detrended residual / n_points
    decay_rate: float        # Exponential fit R² on |residual| vs primary variable if decaying
    symmetry: float          # Even R² - Odd R² (positive for even, negative for odd)
    leading_exponent: Optional[float] = None
    ras_suggested_class: Optional[str] = None
    ras_fit_quality: Optional[float] = None
    shapiro_stat: Optional[float] = None
    shapiro_p: Optional[float] = None
    likely_gaussian: Optional[bool] = None

def analyze_residual(x: np.ndarray, residual: np.ndarray, classical_limit_val: Optional[float] = None) -> ResidualFeatures:
    """
    Extracts statistical features from the physical residual to identify the structural class.
    
    Args:
        x: Independent variable array.
        residual: Observed residual values (observed - classical).
        classical_limit_val: Optional limit value to compute Residual Asymptotic Signature.
        
    Returns:
        ResidualFeatures containing the analyzed properties.
        
    Example:
        >>> features = analyze_residual(x_data, y_obs - y_classical)
        >>> print(features.decay_rate)
    """
    n_points = len(x)
    if n_points < 5:
        return ResidualFeatures(0.0, 0.0, 0.0, 0.0, 0.0)

    # 1. Monotonicity (Spearman rank correlation)
    spearman_corr, _ = stats.spearmanr(x, residual)
    if np.isnan(spearman_corr):
        spearman_corr = 0.0
    monotonicity = float(spearman_corr)

    # 2. Curvature (quadratic polynomial fit: y = a*x^2 + b*x + c)
    try:
        p = np.polyfit(x, residual, 2)
        curvature_sign = float(np.sign(p[0]))
    except Exception:
        logger.debug("curvature_sign computation failed, defaulting to 0.0", exc_info=True)
        curvature_sign = 0.0

    # 3. Oscillation score (zero-crossings of detrended, smoothed residual)
    try:
        # Sort by x first
        idx = np.argsort(x)
        x_sorted = x[idx]
        res_sorted = residual[idx]

        # Detrend with a 2nd degree polynomial
        p_trend = np.polyfit(x_sorted, res_sorted, 2)
        trend = np.polyval(p_trend, x_sorted)
        detrended = res_sorted - trend

        # Smooth using a running mean (window size 5)
        window = 5
        smoothed = np.convolve(detrended, np.ones(window)/window, mode='valid')

        # Count zero crossings
        zero_crossings = np.sum(np.diff(np.sign(smoothed)) != 0)
        oscillation_score = float(zero_crossings / len(smoothed)) if len(smoothed) > 0 else 0.0
    except Exception:
        logger.debug("oscillation_score computation failed, defaulting to 0.0", exc_info=True)
        oscillation_score = 0.0

    # 4. Decay rate (Exponential fit: log(|y| + eps) = a*x + b)
    try:
        eps = 1e-9
        abs_res = np.abs(residual)
        # Avoid log(0)
        log_res = np.log(abs_res + eps)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, log_res)
        
        # We only consider it decaying if the slope is negative
        if slope < 0 and not np.isnan(r_value):
            decay_rate = float(r_value ** 2)
        else:
            decay_rate = 0.0
    except Exception:
        logger.debug("decay_rate computation failed, defaulting to 0.0", exc_info=True)
        decay_rate = 0.0

    # 5. Symmetry (Even polynomial fit vs Odd polynomial fit)
    # y = a * x^2 + b * x^4  (Even)
    # y = a * x + b * x^3    (Odd)
    try:
        # Even design matrix
        A_even = np.vstack([x**2, x**4]).T
        # Odd design matrix
        A_odd = np.vstack([x, x**3]).T

        # Least squares fits
        _, residuals_even, _, _ = np.linalg.lstsq(A_even, residual, rcond=None)
        _, residuals_odd, _, _ = np.linalg.lstsq(A_odd, residual, rcond=None)

        rss_even = residuals_even[0] if len(residuals_even) > 0 else np.sum((residual - A_even @ np.linalg.pinv(A_even) @ residual)**2)
        rss_odd = residuals_odd[0] if len(residuals_odd) > 0 else np.sum((residual - A_odd @ np.linalg.pinv(A_odd) @ residual)**2)

        tss = np.sum((residual - np.mean(residual))**2) + 1e-9
        
        r2_even = max(0.0, 1.0 - rss_even / tss)
        r2_odd = max(0.0, 1.0 - rss_odd / tss)
        
        symmetry = float(r2_even - r2_odd)
    except Exception:
        logger.debug("symmetry computation failed, defaulting to 0.0", exc_info=True)
        symmetry = 0.0

    # Task P1-6: Compute RAS features if classical_limit_val is provided
    leading_exponent = None
    ras_suggested_class = None
    ras_fit_quality = None
    if classical_limit_val is not None:
        ras = compute_ras(x, residual, classical_limit_val)
        leading_exponent = ras.get("leading_exponent")
        ras_suggested_class = ras.get("suggested_class")
        ras_fit_quality = ras.get("fit_quality")

    # 6. Shapiro-Wilk normality check
    try:
        shapiro_stat, shapiro_p = stats.shapiro(residual)
        likely_gaussian = bool(shapiro_p > 0.05)
    except Exception:
        logger.debug("Shapiro-Wilk normality test failed", exc_info=True)
        shapiro_stat, shapiro_p, likely_gaussian = None, None, None

    return ResidualFeatures(
        monotonicity=monotonicity,
        curvature_sign=curvature_sign,
        oscillation_score=oscillation_score,
        decay_rate=decay_rate,
        symmetry=symmetry,
        leading_exponent=leading_exponent,
        ras_suggested_class=ras_suggested_class,
        ras_fit_quality=ras_fit_quality,
        shapiro_stat=shapiro_stat,
        shapiro_p=shapiro_p,
        likely_gaussian=likely_gaussian
    )

def compute_ras(x_vals: np.ndarray, delta_vals: np.ndarray, 
                limit_val: float) -> dict:
    """
    Residual Asymptotic Signature: estimates leading-order behavior
    of residual as x -> classical limit.
    
    Returns:
        {
          "leading_exponent": float,  # n in delta ~ (x-x0)^n
          "leading_coeff": float,     # C
          "fit_quality": float,       # R^2 of log-log fit
          "suggested_class": str,     # "polynomial", "power_law", "exponential"
        }
    """
    # Take points nearest to classical limit (bottom 20%)
    dist_to_limit = np.abs(x_vals - limit_val)
    thresh = np.percentile(dist_to_limit, 20)
    mask = dist_to_limit <= thresh
    
    d_near = np.abs(delta_vals[mask])
    
    # Filter zeros and invalid
    valid = (d_near > 1e-15) & (dist_to_limit[mask] > 1e-10)
    if valid.sum() < 5:
        return {"leading_exponent": None, "leading_coeff": None, "fit_quality": 0.0, 
                "suggested_class": "unknown"}
    
    log_dist = np.log(dist_to_limit[mask][valid])
    log_delta = np.log(d_near[valid])
    
    try:
        # Linear regression in log-log space
        coeffs = np.polyfit(log_dist, log_delta, 1)
        n_estimate = coeffs[0]
        residuals = log_delta - np.polyval(coeffs, log_dist)
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((log_delta - log_delta.mean())**2)
        r2 = 1 - ss_res/ss_tot if ss_tot > 0 else 0.0
        
        # Classify
        if r2 < 0.5:
            suggested = "exponential"  # poor power-law fit -> likely exp
        elif abs(n_estimate - round(n_estimate)) < 0.15:
            suggested = "polynomial"   # integer exponent
        else:
            suggested = "power_law"    # non-integer exponent
            
        return {
            "leading_exponent": float(n_estimate),
            "leading_coeff": float(np.exp(coeffs[1])),
            "fit_quality": float(r2),
            "suggested_class": suggested,
        }
    except Exception:
        logger.debug("RAS log-log fit failed, returning unknown", exc_info=True)
        return {"leading_exponent": None, "leading_coeff": None, "fit_quality": 0.0,
                "suggested_class": "unknown"}

