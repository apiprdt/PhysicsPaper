"""
spike/crossover.py
==================
Phase 6 Prototype: Two-Limit Crossover Grammar, Convex Validity Diagnostics,
Smooth-Max / Smooth-Min (CES) Grammar, and Locked-vs-Free Hypothesis Testing.

This module implements:
1. Five convex crossover families y = (1 - w(u/s)) * A(u) + w(u/s) * B(u) with
   regularized small-u and large-u behavior:
   - convex_rat   (1 param s):     w(z) = z / (1 + z)
   - convex_exp   (1 param s):     w(z) = -expm1(-z)
   - convex_tanh  (1 param s):     w(z) = tanh(z)
   - convex_rat_p (2 params s, p): w(z) = z^p / (1 + z^p)
   - convex_exp_p (2 params s, p): w(z) = -expm1(-z^p)
2. Four smooth-max / smooth-min (CES / L_n norm) crossover families:
   - smooth_max_1p (1 param n, s=1 fixed): (A^n + B^n)^(1/n)
   - smooth_max_2p (2 params s, n):        (A^n + (s*B)^n)^(1/n)
   - smooth_min_1p (1 param n, s=1 fixed): (A^-n + B^-n)^(-1/n)
   - smooth_min_2p (2 params s, n):        (A^-n + (s*B)^-n)^(-1/n)
3. Structural validity checker `check_convex_validity`:
   Detects crossing poles B(u) - A(u) = 0, empirical weight out-of-bounds
   w_req not in [0, 1], and small-u asymptotic divergence when B(u)/A(u) ~ u^-alpha.
4. Solved physical testbeds (Planck, Debye heat capacity, Michaelis-Menten, RAR)
   with exact numerical limit verifiers at u -> 0 and u -> oo.
5. Locked two-limit vs free one-limit Delta-BIC hypothesis test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import scipy.optimize
import scipy.special

from adcd.validation_v2 import (
    CandidateFit,
    ProtocolVerdict,
    ThreeTierProtocolResult,
    run_three_tier_audit,
)


# ---------------------------------------------------------------------------
# 1. Convex Crossover Weight Families (Regularized for u -> 0 and u -> oo)
# ---------------------------------------------------------------------------

def weight_convex_rat(z: np.ndarray) -> np.ndarray:
    """Rational crossover weight w(z) = z / (1 + z), regularized for large z."""
    z_arr = np.asarray(z, dtype=np.float64)
    z_pos = np.maximum(z_arr, 0.0)
    out = np.empty_like(z_pos)
    small = z_pos <= 1.0
    out[small] = z_pos[small] / (1.0 + z_pos[small])
    inv_z = 1.0 / z_pos[~small]
    out[~small] = 1.0 / (1.0 + inv_z)
    return out


def weight_convex_exp(z: np.ndarray) -> np.ndarray:
    """Exponential crossover weight w(z) = 1 - exp(-z) = -expm1(-z)."""
    z_arr = np.asarray(z, dtype=np.float64)
    z_pos = np.maximum(z_arr, 0.0)
    return -np.expm1(-z_pos)


def weight_convex_tanh(z: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent crossover weight w(z) = tanh(z)."""
    z_arr = np.asarray(z, dtype=np.float64)
    z_pos = np.maximum(z_arr, 0.0)
    return np.tanh(z_pos)


def weight_convex_rat_p(z: np.ndarray, p: float) -> np.ndarray:
    """Generalized Hill/rational weight w(z) = z^p / (1 + z^p)."""
    z_arr = np.asarray(z, dtype=np.float64)
    z_pos = np.maximum(z_arr, 0.0)
    out = np.zeros_like(z_pos)
    nz = z_pos > 0.0
    if not np.any(nz):
        return out
    log_zp = float(p) * np.log(z_pos[nz])
    # Stable logistic evaluation: 1 / (1 + exp(-log_zp))
    pos_mask = log_zp >= 0.0
    vals = np.empty_like(log_zp)
    vals[pos_mask] = 1.0 / (1.0 + np.exp(-log_zp[pos_mask]))
    exp_pos = np.exp(log_zp[~pos_mask])
    vals[~pos_mask] = exp_pos / (1.0 + exp_pos)
    out[nz] = vals
    return out


def weight_convex_exp_p(z: np.ndarray, p: float) -> np.ndarray:
    """Weibull/stretched-exponential weight w(z) = 1 - exp(-z^p) = -expm1(-z^p)."""
    z_arr = np.asarray(z, dtype=np.float64)
    z_pos = np.maximum(z_arr, 0.0)
    out = np.zeros_like(z_pos)
    nz = z_pos > 0.0
    if not np.any(nz):
        return out
    log_zp = np.clip(float(p) * np.log(z_pos[nz]), -700.0, 700.0)
    zp = np.exp(log_zp)
    out[nz] = -np.expm1(-zp)
    return out


def eval_convex_crossover(
    u: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    family: str,
    s: float,
    p: float = 1.0,
) -> np.ndarray:
    """
    Evaluate y(u) = (1 - w(u/s)) * A(u) + w(u/s) * B(u) in stabilized form:
    y(u) = A(u) + w(u/s) * (B(u) - A(u)).
    """
    u_arr = np.asarray(u, dtype=np.float64)
    A_arr = np.broadcast_to(np.asarray(A, dtype=np.float64), u_arr.shape)
    B_arr = np.broadcast_to(np.asarray(B, dtype=np.float64), u_arr.shape)
    z = u_arr / max(float(s), 1e-15)

    if family == "convex_rat":
        w = weight_convex_rat(z)
    elif family == "convex_exp":
        w = weight_convex_exp(z)
    elif family == "convex_tanh":
        w = weight_convex_tanh(z)
    elif family == "convex_rat_p":
        w = weight_convex_rat_p(z, p)
    elif family == "convex_exp_p":
        w = weight_convex_exp_p(z, p)
    else:
        raise ValueError(f"Unknown convex family: {family}")

    return A_arr + w * (B_arr - A_arr)


# ---------------------------------------------------------------------------
# 2. Smooth-Max / Smooth-Min (CES / L_n Norm) Crossover Families
# ---------------------------------------------------------------------------

def eval_smooth_max(
    A: np.ndarray,
    B: np.ndarray,
    n: float,
    s: float = 1.0,
) -> np.ndarray:
    """
    Evaluate smooth-max (upper CES norm):
        y = (A^n + (s * B)^n)^(1/n)
    using log-sum-exp stabilization for positive A, B > 0.
    """
    A_arr = np.maximum(np.asarray(A, dtype=np.float64), 1e-300)
    sB_arr = np.maximum(float(s) * np.asarray(B, dtype=np.float64), 1e-300)
    n_val = max(float(n), 1e-6)
    log_a = n_val * np.log(A_arr)
    log_b = n_val * np.log(sB_arr)
    m = np.maximum(log_a, log_b)
    lse = m + np.log(np.exp(log_a - m) + np.exp(log_b - m))
    return np.exp(lse / n_val)


def eval_smooth_min(
    A: np.ndarray,
    B: np.ndarray,
    n: float,
    s: float = 1.0,
) -> np.ndarray:
    """
    Evaluate smooth-min (harmonic/lower CES norm):
        y = (A^(-n) + (s * B)^(-n))^(-1/n)
    using log-sum-exp stabilization for positive A, B > 0.
    """
    A_arr = np.maximum(np.asarray(A, dtype=np.float64), 1e-300)
    sB_arr = np.maximum(float(s) * np.asarray(B, dtype=np.float64), 1e-300)
    n_val = max(float(n), 1e-6)
    log_a = -n_val * np.log(A_arr)
    log_b = -n_val * np.log(sB_arr)
    m = np.maximum(log_a, log_b)
    lse = m + np.log(np.exp(log_a - m) + np.exp(log_b - m))
    return np.exp(-lse / n_val)


# ---------------------------------------------------------------------------
# 3. Solved Physical Benchmark Systems & Exact Limit Verifiers
# ---------------------------------------------------------------------------

def planck_exact(x: np.ndarray) -> np.ndarray:
    """
    Exact dimensionless Planck oscillator energy y(x) = x / (exp(x) - 1),
    regularized via 6th-order Bernoulli series for x < 1e-4 to maintain
    full float64 precision down to x = 1e-16.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr)
    small = np.abs(x_arr) < 1e-4
    xs = x_arr[small]
    # x / (e^x - 1) = 1 - x/2 + x^2/12 - x^4/720 + x^6/30240
    out[small] = 1.0 - 0.5 * xs + (xs ** 2) / 12.0 - (xs ** 4) / 720.0
    xl = x_arr[~small]
    # Avoid overflow for very large x
    large_mask = xl > 700.0
    vals = np.empty_like(xl)
    vals[large_mask] = xl[large_mask] * np.exp(-xl[large_mask])
    vals[~large_mask] = xl[~large_mask] / np.expm1(xl[~large_mask])
    out[~small] = vals
    return out


def planck_true_weight(x: np.ndarray) -> np.ndarray:
    """
    Exact required convex weight w_true(x) = (A - y_Planck(x)) / (A - B(x))
    where A(x) = 1 and B(x) = x * exp(-x).
    Note: Since A(x) = 1 > x * exp(-x) = B(x) for all x > 0 and
    B(x) <= y_Planck(x) <= A(x), w_true(x) lies strictly in [0, 1] for all x >= 0.
    """
    x_arr = np.asarray(x, dtype=np.float64)
    out = np.empty_like(x_arr)
    small = x_arr < 1e-4
    xs = x_arr[small]
    # Small-x expansion: 1 - y = x/2 - x^2/12; 1 - x e^-x = 1 - x + x^2/2
    # Ratio = (x/2)*(1 + (5/6)x + ...)
    out[small] = 0.5 * xs * (1.0 + (5.0 / 6.0) * xs)
    xl = x_arr[~small]
    y_l = planck_exact(xl)
    b_l = xl * np.exp(-np.minimum(xl, 700.0))
    out[~small] = (1.0 - y_l) / (1.0 - b_l)
    return out


_GL_NODES_32, _GL_WEIGHTS_32 = scipy.special.roots_legendre(32)
# Map from [-1, 1] to [0, 1]
_GL_T01 = 0.5 * (_GL_NODES_32 + 1.0)
_GL_W01 = 0.5 * _GL_WEIGHTS_32


def debye_exact(u: np.ndarray) -> np.ndarray:
    """
    Exact normalized Debye heat capacity C_V / (3 N k_B) as a function of u = T_D / T:
        y(u) = (3 / u^3) * int_0^u (t^4 e^t / (e^t - 1)^2) dt
    Evaluated to machine precision (< 1e-14 relative error) via vectorized 32-node
    Gauss-Legendre quadrature on [0, u] with exact asymptotic branches at small/large u.
    Limits:
      - u -> 0 (high T, Dulong-Petit): y(u) -> 1 - u^2 / 20 + u^4 / 560 - u^6 / 18144
      - u -> oo (low T, phonon T^3):   y(u) -> 4 * pi^4 / (5 * u^3)
    """
    u_arr = np.asarray(u, dtype=np.float64)
    scalar_input = u_arr.ndim == 0
    u_flat = np.atleast_1d(u_arr)
    out = np.empty_like(u_flat)

    small = u_flat < 1e-3
    large = u_flat > 60.0
    mid = (~small) & (~large)

    if np.any(small):
        u2 = u_flat[small] ** 2
        out[small] = 1.0 - u2 / 20.0 + (u2 ** 2) / 560.0 - (u2 ** 3) / 18144.0

    if np.any(large):
        ul = u_flat[large]
        out[large] = (4.0 * (np.pi ** 4)) / (5.0 * (ul ** 3))

    if np.any(mid):
        um = u_flat[mid][:, None]  # shape (M, 1)
        t = um * _GL_T01[None, :]  # shape (M, 32)
        t_small = t < 1e-5
        integrand = np.empty_like(t)
        ts = t[t_small]
        integrand[t_small] = (ts ** 2) * (1.0 - (ts ** 2) / 12.0)
        tl = t[~t_small]
        em1 = np.expm1(tl)
        integrand[~t_small] = (tl ** 4) * np.exp(tl) / (em1 * em1)
        # int_0^u f(t) dt = u * sum_i w_i f(u * tau_i), so (3 / u^3) * int = (3 / u^2) * sum
        integral_norm = np.sum(integrand * _GL_W01[None, :], axis=1)
        um_1d = u_flat[mid]
        out[mid] = (3.0 / (um_1d ** 2)) * integral_norm

    return out[0] if scalar_input else out.reshape(u_arr.shape)


def michaelis_menten_exact(u: np.ndarray) -> np.ndarray:
    """
    Exact dimensionless Michaelis-Menten rate y(u) = u / (1 + u), where u = [S] / K_m.
    Limits:
      - u -> 0:  y(u) ~ u = A_asympt(u) (or y(0) = 0 = A_0)
      - u -> oo: y(u) -> 1 = B(u)
    """
    return weight_convex_rat(u)


def rar_exact(x: np.ndarray) -> np.ndarray:
    """
    McGaugh (2016) Radial Acceleration Relation boost factor y(x) = g_obs / g_bar:
        y(x) = 1 / (1 - exp(-sqrt(x))), where x = g_bar / a_0.
    Limits:
      - x -> oo (Newtonian regime): y(x) -> 1
      - x -> 0  (Deep-MOND regime): y(x) ~ x^(-1/2)
    """
    x_arr = np.asarray(x, dtype=np.float64)
    sq = np.sqrt(np.maximum(x_arr, 1e-30))
    out = np.empty_like(sq)
    small = sq < 1e-4
    s_sm = sq[small]
    out[small] = (1.0 / s_sm) * (1.0 + 0.5 * s_sm + (s_sm ** 2) / 12.0)
    s_lg = sq[~small]
    out[~small] = 1.0 / (-np.expm1(-s_lg))
    return out


# ---------------------------------------------------------------------------
# 4. Structural Convex Validity & Crossing-Pole Diagnostics
# ---------------------------------------------------------------------------

@dataclass
class ConvexValidityReport:
    """Diagnostic report checking whether a two-limit pair (A, B) admits a convex crossover."""
    is_valid: bool
    has_crossing_pole: bool
    crossing_u: Optional[float]
    w_req_min: float
    w_req_max: float
    small_u_divergence_exponent: float
    required_min_power_p: float
    reason: str


def check_convex_validity(
    u: np.ndarray,
    y_obs: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    tol: float = 0.05,
) -> ConvexValidityReport:
    """
    Check mathematically and empirically whether y(u) can be represented as a
    convex combination y(u) = (1 - w(u)) * A(u) + w(u) * B(u) with w(u) in [0, 1].

    Detects two fundamental structural failure modes:
    1. Crossing Pole: B(u) - A(u) changes sign at some u_c inside the domain while
       y(u_c) != A(u_c), causing w_req(u) = (y - A) / (B - A) to diverge to +/- oo.
       (Occurs in RAR at x = 1 and in Debye at u_c = (4*pi^4/5)^(1/3) ~ 4.2709.)
    2. Small-u Divergence: If B(u)/A(u) ~ u^(-alpha) with alpha > 0 as u -> 0,
       then w(u)*B(u)/A(u) ~ u^(p - alpha), which diverges as u -> 0 for any
       weight family with power p <= alpha!
    """
    u_arr = np.asarray(u, dtype=np.float64)
    y_arr = np.asarray(y_obs, dtype=np.float64)
    A_arr = np.broadcast_to(np.asarray(A, dtype=np.float64), u_arr.shape)
    B_arr = np.broadcast_to(np.asarray(B, dtype=np.float64), u_arr.shape)

    diff_BA = B_arr - A_arr
    signs = np.sign(diff_BA)
    has_zero = np.any(np.abs(diff_BA) < 1e-12)
    sign_changes = np.where(signs[:-1] * signs[1:] <= 0)[0]
    has_crossing = bool(has_zero or len(sign_changes) > 0)

    crossing_u: Optional[float] = None
    if len(sign_changes) > 0:
        idx = int(sign_changes[0])
        u0, u1 = u_arr[idx], u_arr[idx + 1]
        d0, d1 = diff_BA[idx], diff_BA[idx + 1]
        if abs(d1 - d0) > 1e-15:
            crossing_u = float(u0 - d0 * (u1 - u0) / (d1 - d0))
        else:
            crossing_u = float(0.5 * (u0 + u1))

    # Compute empirical required weight w_req = (y - A) / (B - A)
    safe_denom = np.where(np.abs(diff_BA) < 1e-12, np.copysign(1e-12, diff_BA + 1e-30), diff_BA)
    w_req = (y_arr - A_arr) / safe_denom
    w_min = float(np.min(w_req))
    w_max = float(np.max(w_req))

    # Estimate small-u divergence exponent alpha where B(u)/A(u) ~ u^(-alpha) as u -> 0
    order = np.argsort(u_arr)
    u_low = u_arr[order[:5]]
    ratio_low = np.abs(B_arr[order[:5]]) / np.maximum(np.abs(A_arr[order[:5]]), 1e-30)
    if np.all(ratio_low > 0) and u_low[-1] > u_low[0]:
        slope = np.polyfit(np.log(u_low), np.log(ratio_low), 1)[0]
        alpha_div = float(max(0.0, -slope))
    else:
        alpha_div = 0.0

    in_bounds = (w_min >= -tol) and (w_max <= 1.0 + tol)
    is_valid = (not has_crossing) and in_bounds

    reasons: List[str] = []
    if has_crossing:
        reasons.append(
            f"Regime baselines cross inside window (B(u) - A(u) = 0 at u ~ {crossing_u:.4f}), "
            f"creating a pole in w_req = (y - A)/(B - A) in [{w_min:.2f}, {w_max:.2f}]."
        )
    elif not in_bounds:
        reasons.append(
            f"Empirical weight w_req in [{w_min:.3f}, {w_max:.3f}] violates convex bounds [0, 1]."
        )
    if alpha_div > 0.05:
        reasons.append(
            f"High-u baseline B(u) diverges as u^(-{alpha_div:.2f}) at small u; "
            f"convex weight requires power p > {alpha_div:.2f} to preserve the u -> 0 limit."
        )
    if not reasons:
        reasons.append("Valid bounded convex crossover: y(u) lies strictly between A(u) and B(u).")

    return ConvexValidityReport(
        is_valid=is_valid,
        has_crossing_pole=has_crossing,
        crossing_u=crossing_u,
        w_req_min=w_min,
        w_req_max=w_max,
        small_u_divergence_exponent=alpha_div,
        required_min_power_p=alpha_div,
        reason=" ".join(reasons),
    )


def verify_crossover_limits(
    eval_fn: Callable[[np.ndarray], np.ndarray],
    A_fn: Callable[[np.ndarray], np.ndarray],
    B_fn: Callable[[np.ndarray], np.ndarray],
    u_small: float = 1e-7,
    u_large: float = 1e4,
    rtol_small: float = 1e-3,
    rtol_large: float = 1e-3,
) -> Tuple[bool, float, float]:
    """
    Numerically verify that a crossover function `eval_fn(u)` converges to A(u)
    as u -> 0 and to B(u) as u -> oo within relative tolerances.
    """
    u_s = np.array([u_small], dtype=np.float64)
    u_l = np.array([u_large], dtype=np.float64)

    y_s = float(eval_fn(u_s)[0])
    a_s = float(A_fn(u_s)[0])
    err_small = abs(y_s - a_s) / max(abs(a_s), 1e-12)

    y_l = float(eval_fn(u_l)[0])
    b_l = float(B_fn(u_l)[0])
    err_large = abs(y_l - b_l) / max(abs(b_l), 1e-12)

    passed = (err_small <= rtol_small) and (err_large <= rtol_large)
    return passed, err_small, err_large


# ---------------------------------------------------------------------------
# 5. Fitting Convex & Smooth-Max/Min Crossover Families
# ---------------------------------------------------------------------------

CONVEX_FAMILIES_1P = ["convex_rat", "convex_exp", "convex_tanh"]
CONVEX_FAMILIES_2P = ["convex_rat_p", "convex_exp_p"]
ALL_CONVEX_FAMILIES = CONVEX_FAMILIES_1P + CONVEX_FAMILIES_2P


@dataclass
class CrossoverFitSummary:
    """Result of fitting a single crossover candidate on data."""
    name: str
    family: str
    k_params: int
    chi2: float
    bic: float
    nmse: float
    rms_dex: float
    params: Dict[str, float] = field(default_factory=dict)
    y_pred: Optional[np.ndarray] = None
    limits_verified: bool = True


def _compute_metrics(
    y_obs: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    k_params: int,
) -> Tuple[float, float, float, float]:
    """Compute (chi2, bic, nmse, rms_dex) under Gaussian observational noise."""
    n = len(y_obs)
    resid = (y_obs - y_pred) / sigma
    chi2 = float(np.sum(resid ** 2))
    log_l = -0.5 * chi2 - 0.5 * float(np.sum(np.log(2.0 * np.pi * (sigma ** 2))))
    bic = float(k_params * np.log(n) - 2.0 * log_l)
    var_sum = float(np.sum((y_obs - np.mean(y_obs)) ** 2))
    nmse = float(np.sum((y_obs - y_pred) ** 2) / max(var_sum, 1e-30))
    # RMS in dex for positive predictions
    if np.all(y_obs > 0) and np.all(y_pred > 0):
        rms_dex = float(np.sqrt(np.mean((np.log10(y_obs) - np.log10(y_pred)) ** 2)))
    else:
        rms_dex = float("nan")
    return chi2, bic, nmse, rms_dex


def fit_convex_family(
    u: np.ndarray,
    y_obs: np.ndarray,
    sigma: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    family: str,
    min_power_p: float = 0.05,
) -> CrossoverFitSummary:
    """
    Fit a 1-parameter or 2-parameter convex crossover family using multi-start
    L-BFGS-B optimization in log-parameter space.
    """
    u_arr = np.asarray(u, dtype=np.float64)
    y_arr = np.asarray(y_obs, dtype=np.float64)
    sig_arr = np.asarray(sigma, dtype=np.float64)
    A_arr = np.broadcast_to(np.asarray(A, dtype=np.float64), u_arr.shape)
    B_arr = np.broadcast_to(np.asarray(B, dtype=np.float64), u_arr.shape)

    if family in CONVEX_FAMILIES_1P:
        k_params = 1

        def _obj_1p(log_s_arr: np.ndarray) -> float:
            s_val = float(np.exp(np.clip(log_s_arr[0], -12.0, 12.0)))
            y_hat = eval_convex_crossover(u_arr, A_arr, B_arr, family, s=s_val)
            return float(np.sum(((y_arr - y_hat) / sig_arr) ** 2))

        grid_s = np.linspace(np.log(float(np.min(u_arr)) * 0.5), np.log(float(np.max(u_arr)) * 2.0), 15)
        grid_vals = [(_obj_1p(np.array([ls0])), ls0) for ls0 in grid_s]
        grid_vals.sort(key=lambda t: t[0])

        best_chi2 = float("inf")
        best_s = 1.0
        for _, log_s0 in grid_vals[:2]:
            res = scipy.optimize.minimize(
                _obj_1p,
                x0=np.array([log_s0]),
                method="L-BFGS-B",
                bounds=[(-12.0, 12.0)],
            )
            if float(res.fun) < best_chi2:
                best_chi2 = float(res.fun)
                best_s = float(np.exp(res.x[0]))

        y_pred = eval_convex_crossover(u_arr, A_arr, B_arr, family, s=best_s)
        chi2, bic, nmse, rms_dex = _compute_metrics(y_arr, y_pred, sig_arr, k_params=k_params)
        return CrossoverFitSummary(
            name=family,
            family=family,
            k_params=k_params,
            chi2=chi2,
            bic=bic,
            nmse=nmse,
            rms_dex=rms_dex,
            params={"s": best_s},
            y_pred=y_pred,
        )

    if family in CONVEX_FAMILIES_2P:
        k_params = 2
        p_lower = max(float(min_power_p), 0.05)

        def _obj_2p(theta: np.ndarray) -> float:
            s_val = float(np.exp(np.clip(theta[0], -12.0, 12.0)))
            p_val = float(np.clip(theta[1], p_lower, 15.0))
            y_hat = eval_convex_crossover(u_arr, A_arr, B_arr, family, s=s_val, p=p_val)
            return float(np.sum(((y_arr - y_hat) / sig_arr) ** 2))

        best_chi2 = float("inf")
        best_s, best_p = 1.0, max(1.0, p_lower)
        s_grid = np.linspace(np.log(float(np.min(u_arr))), np.log(float(np.max(u_arr))), 7)
        p_grid = [max(0.7, p_lower), max(1.0, p_lower), max(1.5, p_lower), max(2.5, p_lower), max(3.5, p_lower)]
        candidates_2p = [
            (_obj_2p(np.array([ls0, p0])), ls0, p0)
            for ls0 in s_grid
            for p0 in p_grid
        ]
        candidates_2p.sort(key=lambda t: t[0])
        for _, ls0, p0 in candidates_2p[:3]:
            res = scipy.optimize.minimize(
                _obj_2p,
                x0=np.array([ls0, p0]),
                method="L-BFGS-B",
                bounds=[(-12.0, 12.0), (p_lower, 15.0)],
            )
            if float(res.fun) < best_chi2:
                best_chi2 = float(res.fun)
                best_s = float(np.exp(res.x[0]))
                best_p = float(res.x[1])

        y_pred = eval_convex_crossover(u_arr, A_arr, B_arr, family, s=best_s, p=best_p)
        chi2, bic, nmse, rms_dex = _compute_metrics(y_arr, y_pred, sig_arr, k_params=k_params)
        return CrossoverFitSummary(
            name=family,
            family=family,
            k_params=k_params,
            chi2=chi2,
            bic=bic,
            nmse=nmse,
            rms_dex=rms_dex,
            params={"s": best_s, "p": best_p},
            y_pred=y_pred,
        )

    raise ValueError(f"Unsupported convex family: {family}")


def fit_smooth_ces_family(
    u: np.ndarray,
    y_obs: np.ndarray,
    sigma: np.ndarray,
    A: np.ndarray,
    B: np.ndarray,
    mode: str = "max",
    fit_scale_s: bool = True,
    use_log_space_loss: bool = False,
) -> CrossoverFitSummary:
    """
    Fit a 1-parameter (n only, s=1) or 2-parameter (n, s) smooth-max or smooth-min
    CES crossover model:
      - mode='max': y = (A^n + (s*B)^n)^(1/n)
      - mode='min': y = (A^(-n) + (s*B)^(-n))^(-1/n)
    """
    u_arr = np.asarray(u, dtype=np.float64)
    y_arr = np.asarray(y_obs, dtype=np.float64)
    sig_arr = np.asarray(sigma, dtype=np.float64)
    A_arr = np.broadcast_to(np.asarray(A, dtype=np.float64), u_arr.shape)
    B_arr = np.broadcast_to(np.asarray(B, dtype=np.float64), u_arr.shape)

    eval_fn = eval_smooth_max if mode == "max" else eval_smooth_min
    fam_name = f"smooth_{mode}_{'2p' if fit_scale_s else '1p'}"
    k_params = 2 if fit_scale_s else 1

    def _loss(y_hat: np.ndarray) -> float:
        if use_log_space_loss:
            return float(np.sum((np.log10(y_arr) - np.log10(np.maximum(y_hat, 1e-300))) ** 2))
        return float(np.sum(((y_arr - y_hat) / sig_arr) ** 2))

    best_obj = float("inf")
    best_n, best_s = 1.0, 1.0

    if not fit_scale_s:
        n_cands = [(_loss(eval_fn(A_arr, B_arr, n=float(n0), s=1.0)), n0) for n0 in [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]]
        n_cands.sort(key=lambda t: t[0])
        for _, n0 in n_cands[:2]:
            res = scipy.optimize.minimize(
                lambda th: _loss(eval_fn(A_arr, B_arr, n=float(th[0]), s=1.0)),
                x0=np.array([n0]),
                method="L-BFGS-B",
                bounds=[(0.1, 20.0)],
            )
            if float(res.fun) < best_obj:
                best_obj = float(res.fun)
                best_n = float(res.x[0])
    else:
        ns_cands = [
            (_loss(eval_fn(A_arr, B_arr, n=float(n0), s=float(s0))), n0, s0)
            for n0 in [0.5, 0.8, 1.2, 1.7, 2.5, 4.0]
            for s0 in [0.7, 1.0, 1.3]
        ]
        ns_cands.sort(key=lambda t: t[0])
        for _, n0, s0 in ns_cands[:3]:
            res = scipy.optimize.minimize(
                lambda th: _loss(eval_fn(A_arr, B_arr, n=float(th[0]), s=float(th[1]))),
                x0=np.array([n0, s0]),
                method="L-BFGS-B",
                bounds=[(0.1, 20.0), (0.05, 20.0)],
            )
            if float(res.fun) < best_obj:
                best_obj = float(res.fun)
                best_n = float(res.x[0])
                best_s = float(res.x[1])

    y_pred = eval_fn(A_arr, B_arr, n=best_n, s=best_s)
    chi2, bic, nmse, rms_dex = _compute_metrics(y_arr, y_pred, sig_arr, k_params=k_params)
    params = {"n": best_n, "s": best_s} if fit_scale_s else {"n": best_n}
    return CrossoverFitSummary(
        name=fam_name,
        family=fam_name,
        k_params=k_params,
        chi2=chi2,
        bic=bic,
        nmse=nmse,
        rms_dex=rms_dex,
        params=params,
        y_pred=y_pred,
    )


def fit_v1_one_limit_single_primitive(
    u: np.ndarray,
    y_obs: np.ndarray,
    sigma: np.ndarray,
    A: np.ndarray,
    primitive: str = "sat",
) -> CrossoverFitSummary:
    """
    Fit a v1 one-limit single-primitive baseline+correction model:
        y(u) = A(u) * (1 + theta_0 * D(u / theta_1))
    where D(z) in {'sat': tanh(z), 'rat': z/(1+z), 'exp': 1 - exp(-z)}.
    Used to compare two-limit crossover grammar against v1 one-limit grammar (Criterion 3).
    """
    u_arr = np.asarray(u, dtype=np.float64)
    y_arr = np.asarray(y_obs, dtype=np.float64)
    sig_arr = np.asarray(sigma, dtype=np.float64)
    A_arr = np.broadcast_to(np.asarray(A, dtype=np.float64), u_arr.shape)

    def _eval_d(z: np.ndarray) -> np.ndarray:
        if primitive == "sat":
            return np.tanh(z)
        if primitive == "rat":
            return weight_convex_rat(z)
        if primitive == "exp":
            return weight_convex_exp(z)
        raise ValueError(f"Unknown v1 primitive: {primitive}")

    scale_arr = np.ones_like(A_arr) if np.all(np.abs(A_arr) < 1e-12) else A_arr

    def _fit_for_log_s(log_s: float) -> Tuple[float, float, float, np.ndarray]:
        s_val = float(np.exp(np.clip(log_s, -12.0, 12.0)))
        basis = scale_arr * _eval_d(u_arr / s_val)
        target = y_arr - A_arr
        w_Num = np.sum((basis * target) / (sig_arr ** 2))
        w_Den = max(float(np.sum((basis / sig_arr) ** 2)), 1e-30)
        theta0 = float(w_Num / w_Den)
        y_hat = A_arr + theta0 * basis
        chi2_val = float(np.sum(((y_arr - y_hat) / sig_arr) ** 2))
        return chi2_val, theta0, s_val, y_hat

    best_chi2 = float("inf")
    best_theta0, best_s = 0.0, 1.0
    best_pred = A_arr.copy()

    ls_grid = np.linspace(np.log(float(np.min(u_arr)) * 0.5), np.log(float(np.max(u_arr)) * 2.0), 25)
    grid_evals = [(_fit_for_log_s(float(ls0))[0], float(ls0)) for ls0 in ls_grid]
    grid_evals.sort(key=lambda t: t[0])

    for _, ls0 in grid_evals[:2]:
        res = scipy.optimize.minimize(
            lambda x: _fit_for_log_s(float(x[0]))[0],
            x0=np.array([ls0]),
            method="L-BFGS-B",
            bounds=[(-12.0, 12.0)],
        )
        c2, th0, sv, yh = _fit_for_log_s(float(res.x[0]))
        if c2 < best_chi2:
            best_chi2, best_theta0, best_s, best_pred = c2, th0, sv, yh

    chi2, bic, nmse, rms_dex = _compute_metrics(y_arr, best_pred, sig_arr, k_params=2)
    return CrossoverFitSummary(
        name=f"v1_one_limit_{primitive}",
        family=f"v1_{primitive}",
        k_params=2,
        chi2=chi2,
        bic=bic,
        nmse=nmse,
        rms_dex=rms_dex,
        params={"theta_0": best_theta0, "theta_1": best_s},
        y_pred=best_pred,
    )


# ---------------------------------------------------------------------------
# 6. Question 3: Locked Two-Limit vs Free One-Limit Hypothesis Test
# ---------------------------------------------------------------------------

@dataclass
class SecondLimitHypothesisResult:
    """Result of testing whether a proposed second limit B(u) is supported by data."""
    bic_locked_two_limit: float
    bic_free_one_limit: float
    delta_bic_free_minus_locked: float
    best_locked_family: str
    best_free_family: str
    second_limit_verdict: str


def evaluate_second_limit_hypothesis(
    u: np.ndarray,
    y_obs: np.ndarray,
    sigma: np.ndarray,
    A: np.ndarray,
    B_proposed: np.ndarray,
    bic_threshold: float = 10.0,
    v1_baseline: Optional[np.ndarray] = None,
) -> SecondLimitHypothesisResult:
    """
    Test whether a proposed second limit B_proposed(u) should be locked or rejected
    by comparing the best locked two-limit crossover model against the best free
    one-limit baseline+correction model via Delta-BIC = BIC(free) - BIC(locked).

    Verdict rules:
      - Delta-BIC > +10: LOCKED_SECOND_LIMIT_CONFIRMED (locking B(u) decisively improves BIC)
      - Delta-BIC < -10: SECOND_LIMIT_REJECTED (proposed B(u) is misspecified; data refutes it)
      - |Delta-BIC| <= 10: INCONCLUSIVE (both locked and free models are statistically compatible)
    """
    locked_fits = [
        fit_convex_family(u, y_obs, sigma, A, B_proposed, fam)
        for fam in ALL_CONVEX_FAMILIES
    ]
    best_locked = min(locked_fits, key=lambda f: f.bic)

    base_for_free = v1_baseline if v1_baseline is not None else A
    free_fits = [
        fit_v1_one_limit_single_primitive(u, y_obs, sigma, base_for_free, prim)
        for prim in ["sat", "rat", "exp"]
    ]
    best_free = min(free_fits, key=lambda f: f.bic)

    delta_bic = float(best_free.bic - best_locked.bic)
    if delta_bic > bic_threshold:
        verdict = "LOCKED_SECOND_LIMIT_CONFIRMED"
    elif delta_bic < -bic_threshold:
        verdict = "SECOND_LIMIT_REJECTED"
    else:
        verdict = "INCONCLUSIVE"

    return SecondLimitHypothesisResult(
        bic_locked_two_limit=best_locked.bic,
        bic_free_one_limit=best_free.bic,
        delta_bic_free_minus_locked=delta_bic,
        best_locked_family=best_locked.name,
        best_free_family=best_free.name,
        second_limit_verdict=verdict,
    )


# ---------------------------------------------------------------------------
# 7. Preliminary Results Reproducer & Single-Seed Crossover Audit
# ---------------------------------------------------------------------------

def reproduce_preliminary_results() -> Dict[str, Any]:
    """
    Reproduce all preliminary numbers documented in Phase 6 of the Brief:
    1. Planck Asimov chi^2 on x = logspace(-2, 1, 200) at 1% relative noise:
       - convex_exp_p: 96.40
       - convex_tanh:  1605.17
       - convex_exp:   8770.07
       - convex_rat:   772,493.57
       - v1 sat:       1399.25
    2. RAR on x = logspace(-2, 1, 2693):
       - Crossing pole at x = 1 (A(1) = B(1) = 1 while y_RAR(1) = 1.582)
       - 2-parameter smooth-max RMS = 0.00995 dex
    3. Debye heat capacity on u = logspace(-2, 1, 200) at 1% relative noise:
       - Crossing pole at u_c = (4*pi^4/5)^(1/3) = 4.2709 and u^-3 small-u divergence
       - 2-parameter smooth-min Asimov chi^2 = 281.95
    """
    # 1. Planck Asimov
    x_p = np.logspace(-2.0, 1.0, 200)
    y_p = planck_exact(x_p)
    sig_p = 0.01 * y_p
    A_p = np.ones_like(x_p)
    B_p = x_p * np.exp(-x_p)

    planck_fits = {
        fam: fit_convex_family(x_p, y_p, sig_p, A_p, B_p, fam)
        for fam in ALL_CONVEX_FAMILIES
    }
    planck_v1_sat = fit_v1_one_limit_single_primitive(x_p, y_p, sig_p, A_p, primitive="sat")
    planck_validity = check_convex_validity(x_p, y_p, A_p, B_p)

    # 2. RAR Asimov
    x_rar = np.logspace(-2.0, 1.0, 2693)
    y_rar = rar_exact(x_rar)
    sig_rar = 0.01 * y_rar
    A_rar = np.ones_like(x_rar)
    B_rar = x_rar ** (-0.5)
    rar_validity = check_convex_validity(x_rar, y_rar, A_rar, B_rar)
    rar_smooth_max_2p = fit_smooth_ces_family(
        x_rar, y_rar, sig_rar, A_rar, B_rar, mode="max", fit_scale_s=True, use_log_space_loss=True
    )

    # 3. Debye Asimov
    u_deb = np.logspace(-2.0, 1.0, 200)
    y_deb = debye_exact(u_deb)
    sig_deb = 0.01 * y_deb
    A_deb = np.ones_like(u_deb)
    B_deb_t3 = (4.0 * (np.pi ** 4)) / (5.0 * (u_deb ** 3))
    B_deb_zero = np.zeros_like(u_deb)

    debye_validity_t3 = check_convex_validity(u_deb, y_deb, A_deb, B_deb_t3)
    debye_smooth_min_2p = fit_smooth_ces_family(
        u_deb, y_deb, sig_deb, A_deb, B_deb_t3, mode="min", fit_scale_s=True
    )
    debye_convex_zero_rat_p = fit_convex_family(
        u_deb, y_deb, sig_deb, A_deb, B_deb_zero, "convex_rat_p"
    )
    debye_v1_sat = fit_v1_one_limit_single_primitive(u_deb, y_deb, sig_deb, A_deb, primitive="sat")

    return {
        "planck": {
            "validity": planck_validity,
            "convex_exp_p_chi2": planck_fits["convex_exp_p"].chi2,
            "convex_rat_p_chi2": planck_fits["convex_rat_p"].chi2,
            "convex_tanh_chi2": planck_fits["convex_tanh"].chi2,
            "convex_exp_chi2": planck_fits["convex_exp"].chi2,
            "convex_rat_chi2": planck_fits["convex_rat"].chi2,
            "v1_sat_chi2": planck_v1_sat.chi2,
            "fits": planck_fits,
            "v1_sat": planck_v1_sat,
        },
        "rar": {
            "validity": rar_validity,
            "smooth_max_2p_rms_dex": rar_smooth_max_2p.rms_dex,
            "smooth_max_2p_params": rar_smooth_max_2p.params,
        },
        "debye": {
            "validity_t3": debye_validity_t3,
            "smooth_min_2p_chi2": debye_smooth_min_2p.chi2,
            "convex_zero_rat_p_chi2": debye_convex_zero_rat_p.chi2,
            "v1_sat_chi2": debye_v1_sat.chi2,
        },
    }


def audit_crossover_case_single_seed(
    case_name: str,
    seed: int,
    include_exact_truth_in_pool: bool = False,
    rel_noise: float = 0.01,
    n_points: int = 200,
) -> Dict[str, Any]:
    """
    Run a single noisy seed of the three-tier blind identifiability protocol on one of
    the three solved crossover benchmarks:
      - 'michaelis_menten': u in logspace(-2, 2, N), y = u / (1 + u)
      - 'planck':           u in logspace(-2, 1, N), y = u / (exp(u) - 1)
      - 'debye':            u in logspace(-2, 1, N), y = Debye_Cv(u)

    Evaluates:
      1. Elementary crossover candidate dictionary (5 convex families + smooth-min/max where positive).
      2. Optionally includes the exact physical law (`exact_truth` family) in the candidate pool
         to test whether failures of Criterion 1 & 2 come from dictionary misspecification vs protocol.
      3. Compares best crossover BIC against best v1 one-limit BIC (Criterion 3).
      4. Runs isotonic adequacy warning check on the winning model.
    """
    rng = np.random.default_rng(seed)

    if case_name == "michaelis_menten":
        u = np.logspace(-2.0, 2.0, n_points)
        y_true = michaelis_menten_exact(u)
        A = np.zeros_like(u)
        B = np.ones_like(u)
        true_family_names = {"convex_rat", "convex_rat_p", "exact_truth"}
        min_p = 0.05
    elif case_name == "planck":
        u = np.logspace(-2.0, 1.0, n_points)
        y_true = planck_exact(u)
        A = np.ones_like(u)
        B = u * np.exp(-u)
        true_family_names = {"exact_truth"}
        min_p = 0.05
    elif case_name == "debye":
        u = np.logspace(-2.0, 1.0, n_points)
        y_true = debye_exact(u)
        A = np.ones_like(u)
        B = (4.0 * (np.pi ** 4)) / (5.0 * (u ** 3))
        true_family_names = {"exact_truth"}
        # For Debye with B(u) ~ u^-3, convex weight requires p > 3 to avoid u -> 0 divergence
        min_p = 3.05
    else:
        raise ValueError(f"Unknown case_name: {case_name}")

    sigma = rel_noise * np.maximum(np.abs(y_true), 1e-4)
    y_obs = y_true + rng.normal(0.0, sigma)

    # Null baseline = A(u) alone (no crossover to B)
    _, bic_null, _, _ = _compute_metrics(y_obs, A, sigma, k_params=0)

    candidates: List[CandidateFit] = []

    # Fit convex families
    if case_name == "debye":
        # On Debye, 1-param convex families (p=1 < 3) diverge at u -> 0; evaluate 2p with p > 3
        # plus smooth_min_1p and smooth_min_2p and zero-floor convex families
        for fam in CONVEX_FAMILIES_2P:
            fit = fit_convex_family(u, y_obs, sigma, A, B, fam, min_power_p=min_p)
            candidates.append(
                CandidateFit(
                    expr=f"{fam}(u/s)",
                    bic=fit.bic,
                    nmse=fit.nmse,
                    k_params=fit.k_params,
                    family=fam,
                    theta_fit=fit.params,
                    y_pred=fit.y_pred,
                )
            )
        for fit_s in [False, True]:
            sm_fit = fit_smooth_ces_family(u, y_obs, sigma, A, B, mode="min", fit_scale_s=fit_s)
            candidates.append(
                CandidateFit(
                    expr=f"{sm_fit.name}(u)",
                    bic=sm_fit.bic,
                    nmse=sm_fit.nmse,
                    k_params=sm_fit.k_params,
                    family=sm_fit.family,
                    theta_fit=sm_fit.params,
                    y_pred=sm_fit.y_pred,
                )
            )
    else:
        for fam in ALL_CONVEX_FAMILIES:
            fit = fit_convex_family(u, y_obs, sigma, A, B, fam, min_power_p=min_p)
            # Group 1p and 2p of the same functional form into the same structural family
            # only for Michaelis-Menten where convex_rat_p(p=1) nests convex_rat
            fam_label = "convex_rat" if (case_name == "michaelis_menten" and fam in {"convex_rat", "convex_rat_p"}) else fam
            candidates.append(
                CandidateFit(
                    expr=f"{fam}(u/s)",
                    bic=fit.bic,
                    nmse=fit.nmse,
                    k_params=fit.k_params,
                    family=fam_label,
                    theta_fit=fit.params,
                    y_pred=fit.y_pred,
                )
            )

    if include_exact_truth_in_pool and case_name in {"planck", "debye"}:
        # Fit 1-parameter scale s in y_true(u / s)
        exact_fn = planck_exact if case_name == "planck" else debye_exact

        def _exact_obj(log_s_arr: np.ndarray) -> float:
            s_val = float(np.exp(np.clip(log_s_arr[0], -5.0, 5.0)))
            y_hat = exact_fn(u / s_val)
            return float(np.sum(((y_obs - y_hat) / sigma) ** 2))

        res_ex = scipy.optimize.minimize(
            _exact_obj, x0=np.array([0.0]), method="L-BFGS-B", bounds=[(-5.0, 5.0)]
        )
        s_ex = float(np.exp(res_ex.x[0]))
        y_ex = exact_fn(u / s_ex)
        _, bic_ex, nmse_ex, _ = _compute_metrics(y_obs, y_ex, sigma, k_params=1)
        candidates.append(
            CandidateFit(
                expr=f"exact_{case_name}(u/s)",
                bic=bic_ex,
                nmse=nmse_ex,
                k_params=1,
                family="exact_truth",
                theta_fit={"s": s_ex},
                y_pred=y_ex,
            )
        )

    # Evaluate Three-Tier Blind Protocol (with adequacy check enabled to test misfit detection)
    protocol_res: ThreeTierProtocolResult = run_three_tier_audit(
        candidates=candidates,
        null_bic=bic_null,
        search_space_size=len(candidates),
        check_adequacy=True,
        u_data=u,
        y_obs=y_obs,
        sigma=sigma,
    )

    # Fit best v1 one-limit single primitive for Criterion 3 comparison
    v1_Baseline = u if case_name == "michaelis_menten" else A
    v1_fits = [
        fit_v1_one_limit_single_primitive(u, y_obs, sigma, v1_Baseline, prim)
        for prim in ["sat", "rat", "exp"]
    ]
    best_v1 = min(v1_fits, key=lambda f: f.bic)

    best_cand = protocol_res.best_candidate
    best_crossover_bic = float(best_cand.bic) if best_cand is not None else float("inf")
    beats_v1_one_limit = bool(best_crossover_bic < best_v1.bic)

    equiv_families = {c.family for c in protocol_res.equivalence_class}
    best_family = best_cand.family if best_cand is not None else "none"

    truth_in_equiv = bool(len(equiv_families.intersection(true_family_names)) > 0)
    is_unique = protocol_res.verdict == ProtocolVerdict.UNIQUELY_IDENTIFIABLE
    is_true_unique = bool(is_unique and (best_family in true_family_names))
    is_false_unique = bool(is_unique and (best_family not in true_family_names))
    adequacy_warn = protocol_res.adequacy_warning or ""

    return {
        "case": case_name,
        "seed": seed,
        "include_exact_truth": include_exact_truth_in_pool,
        "verdict": protocol_res.verdict.value,
        "best_expr": best_cand.expr if best_cand else None,
        "best_family": best_family,
        "rival_expr": protocol_res.rival_candidate.expr if protocol_res.rival_candidate else None,
        "delta_bic_vs_null": float(protocol_res.delta_bic_vs_null),
        "delta_bic_vs_rival": float(protocol_res.delta_bic_vs_rival),
        "best_crossover_bic": best_crossover_bic,
        "best_v1_bic": float(best_v1.bic),
        "beats_v1_one_limit": beats_v1_one_limit,
        "truth_in_equiv": truth_in_equiv,
        "is_true_unique": is_true_unique,
        "is_false_unique": is_false_unique,
        "has_adequacy_warning": bool(len(adequacy_warn) > 0),
    }
