from typing import Tuple
import numpy as np
from scipy.stats import rankdata


def _scale_dependence(y_classical: np.ndarray, candidate_residual: np.ndarray) -> float:
    a = np.abs(np.asarray(y_classical, dtype=float))
    b = np.abs(np.asarray(candidate_residual, dtype=float))
    
    if len(a) < 5 or np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return 0.0
        
    # FIX AUDIT: Gunakan average ranking untuk menangani data ties secara matematis valid
    ra = rankdata(a, method="average")
    rb = rankdata(b, method="average")
    
    if np.std(ra) < 1e-12 or np.std(rb) < 1e-12:
        return 0.0
        
    corr = np.corrcoef(ra, rb)[0, 1]
    return float(abs(corr)) if np.isfinite(corr) else 0.0


def detect_correction_mode(
    y_obs: np.ndarray, y_classical: np.ndarray
) -> Tuple[str, float]:
    y_obs = np.asarray(y_obs, dtype=float)
    y_classical = np.asarray(y_classical, dtype=float)

    if len(y_obs) < 5 or np.std(y_classical) < 1e-12:
        return "additive", 0.5

    residual_additive = y_obs - y_classical

    safe_classical = np.where(np.abs(y_classical) < 1e-12, np.nan, y_classical)
    with np.errstate(invalid="ignore", divide="ignore"):
        residual_multiplicative = y_obs / safe_classical - 1.0
        
    valid_mask = np.isfinite(residual_multiplicative)
    if np.sum(valid_mask) < 5:
        return "additive", 0.5

    dep_additive = _scale_dependence(y_classical, residual_additive)
    dep_multiplicative = _scale_dependence(
        y_classical[valid_mask], residual_multiplicative[valid_mask]
    )

    total = dep_additive + dep_multiplicative
    if total < 1e-9:
        return "additive", 0.5

    if dep_additive <= dep_multiplicative:
        confidence = 0.5 + 0.5 * (dep_multiplicative - dep_additive) / total
        return "additive", float(np.clip(confidence, 0.5, 1.0))
    else:
        confidence = 0.5 + 0.5 * (dep_additive - dep_multiplicative) / total
        return "multiplicative", float(np.clip(confidence, 0.5, 1.0))
