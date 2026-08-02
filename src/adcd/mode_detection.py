import numpy as np
from typing import Tuple

def detect_correction_mode(y_obs: np.ndarray, y_classical: np.ndarray) -> Tuple[str, float]:
    """
    Statically analyzes data to detect whether the correction is additive or multiplicative.
    
    Method:
      1. Computes absolute residuals: r_add = y_obs - y_classical.
      2. Computes relative residuals: r_mult = y_obs / y_classical - 1 (for non-zero classical values).
      3. Normalizes r_add by mean(|y_classical|) to make it dimensionless.
      4. Measures Pearson correlation between |y_classical| and |r_add|. If the anomaly magnitude 
         scales with the classical model's scale, the correction is likely multiplicative.
      5. Compares dispersion/variance. If relative dispersion is lower, it favors multiplicative.
      
    Returns:
      ("additive" | "multiplicative", confidence: float between 0.0 and 1.0)
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_classical = np.asarray(y_classical, dtype=float)
    
    eps = 1e-15
    abs_classical = np.abs(y_classical)
    mean_abs_cls = np.mean(abs_classical)
    
    # If classical theory predicts zero everywhere, multiplicative scaling is undefined
    if mean_abs_cls < eps:
        return "additive", 1.0
        
    res_add = y_obs - y_classical
    
    # Filter points where classical theory is non-zero
    nonzero_mask = abs_classical > eps
    if not np.any(nonzero_mask):
        return "additive", 1.0
        
    res_mult = (y_obs[nonzero_mask] / y_classical[nonzero_mask]) - 1.0
    
    # Scale additive residuals by the mean of classical values for a dimensionless comparison
    res_add_norm = res_add / mean_abs_cls
    
    var_add = np.var(res_add_norm)
    var_mult = np.var(res_mult)
    
    # Check if absolute error scales linearly with absolute classical value
    # (Pearson correlation between |y_classical| and |y_obs - y_classical|)
    try:
        # Avoid constant array warnings in pearsonr
        if np.std(abs_classical) > eps and np.std(np.abs(res_add)) > eps:
            # Simple calculation instead of importing scipy.stats to make it faster/lightweight
            corr = np.corrcoef(abs_classical, np.abs(res_add))[0, 1]
            if np.isnan(corr):
                corr = 0.0
        else:
            corr = 0.0
    except Exception:
        corr = 0.0
        
    # Decision logic
    # If the error scales with the magnitude (positive correlation) and the relative error has 
    # lower variance, it is multiplicative.
    if corr > 0.25 and var_mult < var_add:
        # Confidence scales with the strength of the correlation
        confidence = float(min(1.0, 0.5 + corr / 2.0))
        return "multiplicative", confidence
    elif var_mult < var_add * 0.1:
        # Relative variance is vastly smaller (e.g. over 10x smaller)
        return "multiplicative", 0.9
    else:
        # Default or lower correlation/higher relative variance implies additive
        confidence = float(min(1.0, 0.5 + abs(corr) / 2.0 if corr < 0 else 0.6))
        return "additive", confidence
