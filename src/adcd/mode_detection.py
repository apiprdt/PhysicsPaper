from typing import Tuple
import numpy as np


def _scale_dependence(y_classical: np.ndarray, candidate_residual: np.ndarray) -> float:
    """Absolute rank correlation between |y_classical| and |candidate_residual|.
    Low value -> residual magnitude does not depend on the classical scale
    (consistent with that hypothesis being correctly specified)."""
    a = np.abs(np.asarray(y_classical, dtype=float))
    b = np.abs(np.asarray(candidate_residual, dtype=float))
    if len(a) < 3 or np.std(a) < 1e-15 or np.std(b) < 1e-15:
        return 0.0
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    if np.std(ra) < 1e-15 or np.std(rb) < 1e-15:
        return 0.0
    return float(abs(np.corrcoef(ra, rb)[0, 1]))


def detect_correction_mode(
    y_obs: np.ndarray, y_classical: np.ndarray
) -> Tuple[str, float]:
    """
    Returns (mode, confidence) where mode in {"additive", "multiplicative"}.

    confidence in [0.5, 1.0]: 0.5 means the two hypotheses are statistically
    indistinguishable on this data (report this honestly rather than picking
    one arbitrarily); 1.0 means one hypothesis is clearly, strongly favored.
    """
    y_obs = np.asarray(y_obs, dtype=float)
    y_classical = np.asarray(y_classical, dtype=float)

    if len(y_obs) < 5 or np.std(y_classical) < 1e-15:
        # Cannot meaningfully test scale-dependence on a near-constant
        # baseline or too little data; default to additive (the more
        # conservative assumption -- it does not silently divide by a
        # near-zero baseline the way multiplicative would) and report
        # minimum confidence rather than a fabricated high one.
        return "additive", 0.5

    residual_additive = y_obs - y_classical

    safe_classical = np.where(np.abs(y_classical) < 1e-15, np.nan, y_classical)
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
        # Neither residual shows any scale dependence at all -- genuinely
        # ambiguous on this data. Report low confidence honestly.
        return "additive", 0.5

    if dep_additive <= dep_multiplicative:
        confidence = 0.5 + 0.5 * (dep_multiplicative - dep_additive) / total
        return "additive", float(np.clip(confidence, 0.5, 1.0))
    else:
        confidence = 0.5 + 0.5 * (dep_additive - dep_multiplicative) / total
        return "multiplicative", float(np.clip(confidence, 0.5, 1.0))
