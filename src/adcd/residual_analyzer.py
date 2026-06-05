import numpy as np
import scipy.stats as stats
from dataclasses import dataclass

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

def analyze_residual(x: np.ndarray, residual: np.ndarray) -> ResidualFeatures:
    """
    Extracts statistical features from the physical residual to identify the structural class.
    
    Args:
        x: Independent variable array.
        residual: Observed residual values (observed - classical).
        
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
        symmetry = 0.0

    return ResidualFeatures(
        monotonicity=monotonicity,
        curvature_sign=curvature_sign,
        oscillation_score=oscillation_score,
        decay_rate=decay_rate,
        symmetry=symmetry
    )
