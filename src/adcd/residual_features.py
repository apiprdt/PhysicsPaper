from dataclasses import dataclass
import numpy as np


@dataclass
class ResidualFeatures:
    """Purely data-driven descriptors of a residual signal's shape, computed
    ONLY from (leading_variable, residual) pairs -- never from scenario
    ground truth, scenario name, or correction_class."""
    decay_rate: float          # [0,1]; strength of |residual| falloff with x
    monotonicity: float        # [-1,1]; Spearman rank correlation (x, residual)
    oscillation_score: float   # [0,1]; permutation-calibrated evidence of
    # genuine curvature reversal beyond pure noise
    symmetry: float            # [-1,1]; +1 ~ even function of x, -1 ~ odd
    leading_exponent: float    # robust log-log slope d(log|r|)/d(log|x|)


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-15 or np.std(y) < 1e-15:
        return 0.0
    rx = np.argsort(np.argsort(x))
    ry = np.argsort(np.argsort(y))
    if np.std(rx) < 1e-15 or np.std(ry) < 1e-15:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def _turning_points(v: np.ndarray) -> int:
    """Count local maxima/minima (Kendall's turning-point statistic)."""
    d = np.diff(v)
    s = np.sign(d)
    s = s[s != 0]
    if len(s) < 2:
        return 0
    return int(np.sum(s[1:] != s[:-1]))


def _moving_average(v: np.ndarray, w: int) -> np.ndarray:
    kernel = np.ones(w) / w
    return np.convolve(v, kernel, mode="valid")


def _oscillation_score(
    r_sorted: np.ndarray, smooth_frac: float = 0.05, n_perm: int = 200, seed: int = 0,
) -> float:
    """
    Permutation-calibrated oscillation detector.

    Raw zero-crossing / FFT-power heuristics are unreliable here because
    (a) a one-sided decay/growth curve has broad-spectrum FFT content
    despite being perfectly monotonic, and (b) moving-average smoothing
    itself induces autocorrelation, so a closed-form i.i.d.-noise null
    (e.g. the textbook Kendall formula) under-estimates the "pure noise"
    turning-point count once smoothing is applied and falsely flags
    smoothed noise as structured.

    Fix: smooth the ordered residual, count turning points, then build the
    null distribution EMPIRICALLY by shuffling the same values and applying
    the identical smoothing + counting pipeline (this correctly captures
    the smoothing-induced autocorrelation). Only when the observed count is
    significantly below the shuffled-null distribution (z <= -1) do we
    report a nonzero score, scaled by how many genuine turning points were
    found (0 -> monotonic, more -> more oscillatory).
    """
    n = len(r_sorted)
    w = max(3, int(n * smooth_frac))
    if np.std(r_sorted) < 1e-14 or n < 5 * w:
        return 0.0

    observed_tp = _turning_points(_moving_average(r_sorted, w))

    rng = np.random.RandomState(seed)
    null_tp = np.empty(n_perm)
    shuffled = r_sorted.copy()
    for i in range(n_perm):
        rng.shuffle(shuffled)
        null_tp[i] = _turning_points(_moving_average(shuffled, w))

    mu, sigma = float(np.mean(null_tp)), float(np.std(null_tp) + 1e-9)
    z = (observed_tp - mu) / sigma

    if z >= -1.0:
        return 0.0
    return float(1.0 - np.exp(-observed_tp / 2.0))


def compute_residual_features(leading_var: np.ndarray, residual: np.ndarray) -> ResidualFeatures:
    """Compute shape features of `residual` as a function of `leading_var`
    (typically the scenario's classical_limit_variable). Uses ONLY these
    two arrays -- no scenario metadata, no ground truth."""
    x = np.asarray(leading_var, dtype=float)
    r = np.asarray(residual, dtype=float)
    n = len(r)
    if n < 4 or np.std(r) < 1e-15:
        return ResidualFeatures(0.0, 0.0, 0.0, 0.0, 0.0)

    order = np.argsort(x)
    r_sorted = r[order]

    monotonicity = _spearman(x, r)
    decay_rate = float(max(0.0, -_spearman(x, np.abs(r))))
    oscillation_score = _oscillation_score(r_sorted)

    x_centered = x - np.mean(x)
    symmetry = 0.0 if np.std(x_centered) < 1e-15 else _spearman(
        np.sign(x_centered) * np.abs(x_centered), r
    )

    abs_r = np.abs(r)
    mask = (x > 1e-12) & (abs_r > 1e-12)
    leading_exponent = 0.0
    if np.sum(mask) >= 4:
        log_x, log_r = np.log(x[mask]), np.log(abs_r[mask])
        if np.std(log_x) > 1e-10:
            leading_exponent = float(np.polyfit(log_x, log_r, 1)[0])

    return ResidualFeatures(
        decay_rate=float(np.clip(decay_rate, 0.0, 1.0)),
        monotonicity=float(np.clip(monotonicity, -1.0, 1.0)),
        oscillation_score=float(np.clip(oscillation_score, 0.0, 1.0)),
        symmetry=float(np.clip(symmetry, -1.0, 1.0)),
        leading_exponent=leading_exponent,
    )
