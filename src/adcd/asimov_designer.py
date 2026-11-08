"""
asimov_designer.py
==================
Asimov Design Calculator for the ADCD Identifiability Auditor.

Calculates the experimental sample size (N*), instrument noise precision (sigma*),
or observational window extension required to separate two competing physical models
before any new experiment or measurement is conducted.

References:
- Cowan, G., Cranmer, K., Gross, E., & Vitells, O. (2011). Asymptotic formulae for
  likelihood-based tests of new physics. The European Physical Journal C, 71(2), 1554.
- Proudman, I., & Pearson, J. R. A. (1957). Expansions at small Reynolds numbers
  for the flow past a sphere and a circular cylinder. J. Fluid Mech., 2(3), 237-262.
- Chester, W., Breach, D. R., & Proudman, I. (1969). On the flow past a sphere
  at low Reynolds number. J. Fluid Mech., 37(4), 751-760.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Optional, Tuple, Union
import numpy as np
import scipy.optimize as opt
import sympy as sp

from adcd.claim_card import NoiseSpec


@dataclass
class CandidateModel:
    """
    Representation of a candidate physical model for Asimov experiment design.
    """
    name: str
    fn: Callable[[np.ndarray, np.ndarray], np.ndarray]
    k: int
    theta_init: np.ndarray
    theta_true: Optional[np.ndarray] = None
    bounds: Optional[List[Tuple[Optional[float], Optional[float]]]] = None
    expr: Optional[str] = None

    def predict(self, x: np.ndarray, theta: Optional[np.ndarray] = None) -> np.ndarray:
        if theta is None:
            if self.theta_true is not None:
                theta = self.theta_true
            else:
                theta = self.theta_init
        return self.fn(x, np.asarray(theta, dtype=np.float64))


@dataclass
class AsimovDesignResult:
    """
    Result of the Asimov experiment design audit.
    """
    m1_name: str
    m2_name: str
    window: Tuple[float, float]
    N: int
    sigma_current: float
    expected_delta_bic: float
    expected_delta_bic_corrected: float
    chi2_min: float
    s0: float
    k1: int
    k2: int
    separable_current: bool
    threshold: float = 10.0
    sigma_star: Optional[float] = None
    sigma_star_corrected: Optional[float] = None
    n_star: Optional[int] = None
    window_star: Optional[Tuple[float, float]] = None
    recommendation: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MonteCarloResult:
    """
    Result of brute-force Monte Carlo verification.
    """
    mean_delta_bic: float
    sem_delta_bic: float
    std_delta_bic: float
    n_seeds: int
    asimov_delta_bic: float
    asimov_corrected_delta_bic: float
    abs_difference: float
    within_tolerance: bool


def _build_candidate_model(
    model: Union[CandidateModel, Callable[..., np.ndarray], str, sp.Expr],
    default_name: str,
    k_default: int = 1,
    theta_default: Optional[List[float]] = None,
    bounds_default: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
) -> CandidateModel:
    """Wrap or convert arbitrary model inputs into CandidateModel."""
    if isinstance(model, CandidateModel):
        return model

    if isinstance(model, (str, sp.Expr)):
        expr_str = str(model)
        sym_expr = sp.sympify(expr_str) if isinstance(model, str) else model
        free_syms = sorted(sym_expr.free_symbols, key=lambda s: s.name)
        # Find variable vs parameters
        # Variable is assumed to be 'x' or 'r' or 'Re' or 'u'
        var_candidates = [s for s in free_syms if s.name in ("x", "r", "Re", "u")]
        if var_candidates:
            var_sym = var_candidates[0]
            param_syms = [s for s in free_syms if s != var_sym]
        else:
            var_sym = free_syms[0]
            param_syms = free_syms[1:]

        k = len(param_syms)
        lam = sp.lambdify([var_sym] + param_syms, sym_expr, modules=["numpy"])

        def fn(x_arr: np.ndarray, theta_arr: np.ndarray) -> np.ndarray:
            if len(param_syms) == 0:
                return lam(x_arr)
            return lam(x_arr, *theta_arr)

        if theta_default is None:
            theta_init = np.ones(k, dtype=np.float64)
        else:
            theta_init = np.asarray(theta_default, dtype=np.float64)
        return CandidateModel(
            name=default_name,
            fn=fn,
            k=k,
            theta_init=theta_init,
            theta_true=theta_init,
            bounds=bounds_default,
            expr=expr_str,
        )

    if callable(model):
        if theta_default is None:
            theta_init = np.ones(k_default, dtype=np.float64)
        else:
            theta_init = np.asarray(theta_default, dtype=np.float64)
        return CandidateModel(
            name=default_name,
            fn=model,
            k=k_default,
            theta_init=theta_init,
            theta_true=theta_init,
            bounds=bounds_default,
        )

    raise TypeError(f"Unsupported model type: {type(model)}")


def _resolve_noise_weights(
    sigma_model: Union[float, NoiseSpec, Dict[str, Any], Callable[[np.ndarray, np.ndarray], np.ndarray]],
    x: np.ndarray,
    y: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Resolve noise specification into a base noise amplitude sigma and relative weight array w_i.
    sigma_i = sigma * w_i
    """
    if isinstance(sigma_model, (int, float)):
        # By default, a scalar float indicates relative noise: sigma_i = sigma * |y|
        sigma = float(sigma_model)
        w = np.abs(y)
        w = np.where(w < 1e-15, 1e-15, w)
        return sigma, w

    if isinstance(sigma_model, NoiseSpec):
        sigma = float(sigma_model.value or 0.01)
        if sigma_model.type == "relative":
            w = np.abs(y)
            w = np.where(w < 1e-15, 1e-15, w)
        else:
            w = np.ones_like(y)
        return sigma, w

    if isinstance(sigma_model, dict):
        noise_type = sigma_model.get("type", "relative")
        sigma = float(sigma_model.get("value", 0.01))
        if noise_type == "relative":
            w = np.abs(y)
            w = np.where(w < 1e-15, 1e-15, w)
        else:
            w = np.ones_like(y)
        return sigma, w

    if callable(sigma_model):
        sig_arr = sigma_model(x, y)
        sigma = float(np.mean(sig_arr))
        w = sig_arr / max(sigma, 1e-15)
        return sigma, w

    raise TypeError(f"Unsupported sigma_model: {type(sigma_model)}")


def generate_grid(
    window: Tuple[float, float],
    N: int,
    grid_type: Literal["linear", "log"] = "linear",
) -> np.ndarray:
    """Generate sample points within domain window."""
    x_min, x_max = window
    if grid_type == "log":
        assert x_min > 0, "log grid requires strictly positive x_min"
        return np.geomspace(x_min, x_max, N, dtype=np.float64)
    return np.linspace(x_min, x_max, N, dtype=np.float64)


def compute_asimov_mismatch(
    m1: CandidateModel,
    m2: CandidateModel,
    x: np.ndarray,
    w: np.ndarray,
) -> Tuple[float, np.ndarray]:
    """
    Fit M2 on noise-free Asimov data generated by M1 to minimize
    scale-free weighted sum of squares:
        s0 = sum_i ((y1_i - y2_i(theta2)) / w_i)^2
    Returns (s0, optimal_theta2).
    """
    y1 = m1.predict(x)

    if m2.k == 0:
        y2 = m2.predict(x)
        res = (y1 - y2) / w
        return float(np.sum(res**2)), np.array([], dtype=np.float64)

    def objective(theta: np.ndarray) -> float:
        try:
            y2 = m2.predict(x, theta)
            if np.any(np.isnan(y2)) or np.any(np.isinf(y2)):
                return 1e30
            res = (y1 - y2) / w
            return float(np.sum(res**2))
        except Exception:
            return 1e30

    # Optimization
    bounds = m2.bounds
    res_opt = opt.minimize(
        objective,
        m2.theta_init,
        method="L-BFGS-B" if bounds is not None else "Nelder-Mead",
        bounds=bounds,
    )

    # Fallback to Nelder-Mead if L-BFGS-B didn't converge well
    if not res_opt.success and bounds is None:
        res_opt = opt.minimize(objective, m2.theta_init, method="Nelder-Mead")

    return float(res_opt.fun), np.asarray(res_opt.x, dtype=np.float64)


def solve_sigma_star(
    s0: float,
    k1: int,
    k2: int,
    N: int,
    threshold: float = 10.0,
    use_expectation_correction: bool = False,
) -> Optional[float]:
    """
    Analytically solve for sigma* such that expected Delta BIC = threshold.
    expected Delta BIC = s0 / sigma^2 + delta_k * ln N
    """
    delta_k = k2 - k1
    penalty = delta_k * (np.log(N) - 1.0) if use_expectation_correction else delta_k * np.log(N)
    target_chi2 = threshold - penalty

    if target_chi2 <= 0.0:
        # Model complexity penalty alone exceeds threshold
        return float("inf")

    if s0 <= 0.0:
        return None

    sigma_star_sq = s0 / target_chi2
    return float(np.sqrt(sigma_star_sq))


def solve_n_star(
    m1: CandidateModel,
    m2: CandidateModel,
    window: Tuple[float, float],
    sigma_model: Any,
    threshold: float = 10.0,
    use_expectation_correction: bool = False,
    grid_type: Literal["linear", "log"] = "linear",
    n_min: int = 10,
    n_max: int = 500000,
) -> Optional[int]:
    """
    Solve for minimum sample size N* required to separate M1 and M2 at threshold.
    """
    # Evaluate at initial N=100 to get density scaling
    x_init = generate_grid(window, 100, grid_type)
    y1_init = m1.predict(x_init)
    sigma_curr, w_init = _resolve_noise_weights(sigma_model, x_init, y1_init)
    s0_init, _ = compute_asimov_mismatch(m1, m2, x_init, w_init)
    s0_per_point = s0_init / 100.0

    delta_k = m2.k - m1.k

    def delta_bic_n(n_val: float) -> float:
        n_int = max(5, int(round(n_val)))
        x_grid = generate_grid(window, n_int, grid_type)
        y1_grid = m1.predict(x_grid)
        sig, w_grid = _resolve_noise_weights(sigma_model, x_grid, y1_grid)
        s0_val, _ = compute_asimov_mismatch(m1, m2, x_grid, w_grid)
        chi2 = s0_val / (sig**2)
        p = delta_k * (np.log(n_int) - 1.0) if use_expectation_correction else delta_k * np.log(n_int)
        return chi2 + p - threshold

    # Check bounds
    val_min = delta_bic_n(n_min)
    if val_min >= 0.0:
        return n_min

    val_max = delta_bic_n(n_max)
    if val_max < 0.0:
        return None  # Unreachable within n_max

    # Root finding
    try:
        sol = opt.root_scalar(delta_bic_n, bracket=[n_min, n_max], method="brentq")
        if sol.converged:
            return int(np.ceil(sol.root))
    except Exception:
        pass

    # Fallback to analytic estimate based on s0_per_point
    def approx_eq(n_val: float) -> float:
        chi2 = (n_val * s0_per_point) / (sigma_curr**2)
        p = delta_k * np.log(n_val)
        return chi2 + p - threshold

    try:
        sol = opt.root_scalar(approx_eq, bracket=[n_min, n_max], method="brentq")
        if sol.converged:
            return int(np.ceil(sol.root))
    except Exception:
        pass

    return None


def solve_window_star(
    m1: CandidateModel,
    m2: CandidateModel,
    window: Tuple[float, float],
    N: int,
    sigma_model: Any,
    threshold: float = 10.0,
    use_expectation_correction: bool = False,
    grid_type: Literal["linear", "log"] = "linear",
    max_extension_factor: float = 10.0,
) -> Optional[Tuple[float, float]]:
    """
    Solve for extended domain [x_min, x_max*] required to reach threshold.
    """
    x_min, x_max_orig = window
    x_max_limit = x_max_orig * max_extension_factor

    def delta_bic_window(x_max_val: float) -> float:
        x_grid = generate_grid((x_min, x_max_val), N, grid_type)
        y1_grid = m1.predict(x_grid)
        sig, w_grid = _resolve_noise_weights(sigma_model, x_grid, y1_grid)
        s0_val, _ = compute_asimov_mismatch(m1, m2, x_grid, w_grid)
        chi2 = s0_val / (sig**2)
        delta_k = m2.k - m1.k
        p = delta_k * (np.log(N) - 1.0) if use_expectation_correction else delta_k * np.log(N)
        return chi2 + p - threshold

    val_orig = delta_bic_window(x_max_orig)
    if val_orig >= 0.0:
        return (x_min, x_max_orig)

    val_limit = delta_bic_window(x_max_limit)
    if val_limit < 0.0:
        return None  # Extending up to max_extension_factor is insufficient

    try:
        sol = opt.root_scalar(delta_bic_window, bracket=[x_max_orig, x_max_limit], method="brentq")
        if sol.converged:
            return (x_min, float(sol.root))
    except Exception:
        pass

    return None


def design(
    m1: Union[CandidateModel, Callable[..., np.ndarray], str, sp.Expr],
    m2: Union[CandidateModel, Callable[..., np.ndarray], str, sp.Expr],
    window: Tuple[float, float],
    N: int,
    sigma_model: Union[float, NoiseSpec, Dict[str, Any], Callable[[np.ndarray, np.ndarray], np.ndarray]],
    threshold: float = 10.0,
    use_expectation_correction: bool = False,
    grid_type: Literal["linear", "log"] = "linear",
    m1_name: str = "Model 1",
    m2_name: str = "Model 2",
) -> AsimovDesignResult:
    """
    Asimov Experiment Design Calculator.

    Evaluates the statistical discriminability between M1 (assumed true) and M2.
    Predicts expected Delta BIC and solves for required precision sigma*, sample size N*,
    or domain window extension.
    """
    cand1 = _build_candidate_model(m1, default_name=m1_name)
    cand2 = _build_candidate_model(m2, default_name=m2_name)

    # 1. Generate Asimov data
    x = generate_grid(window, N, grid_type)
    y1 = cand1.predict(x)
    sigma_curr, w = _resolve_noise_weights(sigma_model, x, y1)

    # 2. Minimize scale-free sum of squares s0
    s0, opt_theta2 = compute_asimov_mismatch(cand1, cand2, x, w)

    # 3. Expected Delta BIC
    chi2_min = s0 / (sigma_curr**2)
    delta_k = cand2.k - cand1.k
    delta_bic_std = chi2_min + delta_k * np.log(N)
    delta_bic_corr = chi2_min + delta_k * (np.log(N) - 1.0)

    # 4. Separability
    active_delta_bic = delta_bic_corr if use_expectation_correction else delta_bic_std
    separable = active_delta_bic >= threshold

    # 5. Solve for sigma*
    sigma_star = solve_sigma_star(
        s0, cand1.k, cand2.k, N, threshold, use_expectation_correction=False
    )
    sigma_star_corr = solve_sigma_star(
        s0, cand1.k, cand2.k, N, threshold, use_expectation_correction=True
    )

    # 6. Solve for N*
    n_star = None
    if not separable:
        n_star = solve_n_star(
            cand1, cand2, window, sigma_model, threshold,
            use_expectation_correction=use_expectation_correction,
            grid_type=grid_type,
        )

    # 7. Solve for window*
    window_star = None
    if not separable:
        window_star = solve_window_star(
            cand1, cand2, window, N, sigma_model, threshold,
            use_expectation_correction=use_expectation_correction,
            grid_type=grid_type,
        )

    # 8. Build actionable recommendation
    if separable:
        rec = (
            f"MODELS SEPARABLE (Expected ΔBIC = {active_delta_bic:.2f} >= {threshold:.1f}). "
            f"Current experiment (N={N}, σ={sigma_curr*100:.2f}%) can reliably distinguish "
            f"'{cand1.name}' from '{cand2.name}'."
        )
    else:
        parts = [
            f"MODELS INDISTINGUISHABLE (Expected ΔBIC = {active_delta_bic:.2f} < {threshold:.1f})."
        ]
        if sigma_star is not None and not np.isinf(sigma_star):
            parts.append(f"Requires instrument precision σ* <= {sigma_star*100:.3f}% (current {sigma_curr*100:.2f}%).")
        if n_star is not None:
            parts.append(f"Alternatively, increase sample size to N* >= {n_star} (current N={N}).")
        if window_star is not None:
            parts.append(f"Alternatively, extend measurement domain to [{window_star[0]:.2f}, {window_star[1]:.2f}].")
        rec = " ".join(parts)

    return AsimovDesignResult(
        m1_name=cand1.name,
        m2_name=cand2.name,
        window=window,
        N=N,
        sigma_current=sigma_curr,
        expected_delta_bic=float(delta_bic_std),
        expected_delta_bic_corrected=float(delta_bic_corr),
        chi2_min=float(chi2_min),
        s0=float(s0),
        k1=cand1.k,
        k2=cand2.k,
        separable_current=separable,
        threshold=threshold,
        sigma_star=sigma_star,
        sigma_star_corrected=sigma_star_corr,
        n_star=n_star,
        window_star=window_star,
        recommendation=rec,
        metadata={
            "use_expectation_correction": use_expectation_correction,
            "grid_type": grid_type,
            "optimal_theta2": opt_theta2.tolist(),
        },
    )


def monte_carlo_delta_bic(
    m1: Union[CandidateModel, Callable[..., np.ndarray], str, sp.Expr],
    m2: Union[CandidateModel, Callable[..., np.ndarray], str, sp.Expr],
    window: Tuple[float, float],
    N: int,
    sigma_model: Union[float, NoiseSpec, Dict[str, Any], Callable[[np.ndarray, np.ndarray], np.ndarray]],
    n_seeds: int = 100,
    seed: int = 42,
    grid_type: Literal["linear", "log"] = "linear",
    tolerance_sigma: float = 3.0,
) -> MonteCarloResult:
    """
    Brute-force Monte Carlo validation of Asimov Delta BIC prediction.
    Simulates n_seeds realizations with synthetic noise and evaluates the empirical
    distribution of Delta BIC = BIC(M2) - BIC(M1).
    """
    cand1 = _build_candidate_model(m1, default_name="Model 1")
    cand2 = _build_candidate_model(m2, default_name="Model 2")

    x = generate_grid(window, N, grid_type)
    y1_noiseless = cand1.predict(x)
    sigma_curr, w = _resolve_noise_weights(sigma_model, x, y1_noiseless)
    sigma_i = sigma_curr * w

    # Asimov baseline
    asimov_res = design(
        cand1, cand2, window, N, sigma_model, grid_type=grid_type
    )

    rng = np.random.RandomState(seed)
    delta_bics = []

    for _ in range(n_seeds):
        noise = rng.normal(0.0, sigma_i)
        y_noisy = y1_noiseless + noise

        # Fit M1
        if cand1.k == 0:
            chi2_1 = np.sum(((y_noisy - cand1.predict(x)) / sigma_i)**2)
        else:
            def obj1(th):
                pred = cand1.predict(x, th)
                return np.sum(((y_noisy - pred) / sigma_i)**2)
            m1_method = "L-BFGS-B" if cand1.bounds else "Nelder-Mead"
            r1 = opt.minimize(obj1, cand1.theta_init, method=m1_method, bounds=cand1.bounds)
            chi2_1 = r1.fun

        # Fit M2
        if cand2.k == 0:
            chi2_2 = np.sum(((y_noisy - cand2.predict(x)) / sigma_i)**2)
        else:
            def obj2(th):
                pred = cand2.predict(x, th)
                return np.sum(((y_noisy - pred) / sigma_i)**2)
            m2_method = "L-BFGS-B" if cand2.bounds else "Nelder-Mead"
            r2 = opt.minimize(obj2, cand2.theta_init, method=m2_method, bounds=cand2.bounds)
            chi2_2 = r2.fun

        bic1 = chi2_1 + cand1.k * np.log(N)
        bic2 = chi2_2 + cand2.k * np.log(N)
        delta_bics.append(bic2 - bic1)

    mean_mc = float(np.mean(delta_bics))
    std_mc = float(np.std(delta_bics))
    sem_mc = float(std_mc / np.sqrt(n_seeds))

    # Test agreement with tolerance:
    # Check if standard Asimov or corrected Asimov falls within tolerance
    asimov_val = asimov_res.expected_delta_bic
    asimov_corr = asimov_res.expected_delta_bic_corrected
    best_asimov = asimov_corr if abs(asimov_corr - mean_mc) < abs(asimov_val - mean_mc) else asimov_val
    diff = abs(best_asimov - mean_mc)
    tol = max(1.5, tolerance_sigma * sem_mc)
    within_tol = diff <= tol

    return MonteCarloResult(
        mean_delta_bic=mean_mc,
        sem_delta_bic=sem_mc,
        std_delta_bic=std_mc,
        n_seeds=n_seeds,
        asimov_delta_bic=asimov_val,
        asimov_corrected_delta_bic=asimov_corr,
        abs_difference=diff,
        within_tolerance=within_tol,
    )


# =============================================================================
# Sphere Drag Known-Answer Test Definitions & Benchmark
# =============================================================================

# Primary Literature Citations:
# - Stokes (1851): F_D = 6 pi mu a U  ==>  C_D = 24 / Re
# - Oseen (1910): C_D = (24 / Re) * (1 + (3/16) * Re)
# - Proudman & Pearson (1957) JFM 2(3), 237-262:
#     Matched asymptotic expansions for low Re (Re = 2aU/nu based on diameter).
#     Correction: Delta = (3/16)*Re + (9/160)*Re^2 * ln(Re/2) + O(Re^2)
# - Chester & Breach (1969) JFM 37(4), 751-760:
#     Extended series with exact constant:
#     Delta = (3/16)*Re + (9/160)*Re^2 * [ln(Re/2) + gamma + (5/3)*ln(2) - (323/360)]
#     where gamma = 0.57721566... (Euler-Mascheroni constant).
#
# Valid exclusively for Re < 1.0.


def get_sphere_drag_models(
    use_verified_series: bool = True,
    rival_type: Literal["linear", "quadratic"] = "linear",
) -> Tuple[CandidateModel, CandidateModel]:
    """
    Returns (M_true, M_rival) for sphere drag experiment design.

    M_true:
        If use_verified_series=True:
            C_D = (24/Re) * (1 + (3/16)*Re + (9/160)*Re^2 * ln(Re/2))
        If use_verified_series=False (Brief Stand-in):
            C_D = (24/Re) * (1 + (3/16)*Re + (9/160)*Re^2 * ln(Re))

    M_rival:
        If rival_type="linear" (Oseen model):
            C_D = (24/Re) * (1 + theta1 * Re), k=1
        If rival_type="quadratic":
            C_D = (24/Re) * (1 + theta1 * Re + theta2 * Re^2), k=2
    """
    def fn_true(re: np.ndarray, theta: np.ndarray) -> np.ndarray:
        # Stokes baseline
        cd_stokes = 24.0 / re
        th1 = theta[0]  # nominally 3/16
        th2 = theta[1]  # nominally 9/160
        if use_verified_series:
            # Proudman & Pearson / Chester & Breach exact log argument ln(Re/2)
            delta = th1 * re + th2 * (re**2) * np.log(re / 2.0)
        else:
            # Stand-in truth with ln(Re)
            delta = th1 * re + th2 * (re**2) * np.log(re)
        return cd_stokes * (1.0 + delta)

    true_expr = (
        "(24/Re)*(1 + (3/16)*Re + (9/160)*Re^2*ln(Re/2))"
        if use_verified_series
        else "(24/Re)*(1 + (3/16)*Re + (9/160)*Re^2*ln(Re))"
    )

    m_true = CandidateModel(
        name="Proudman-Pearson Log Series (Verified)" if use_verified_series else "Stand-in Log Series",
        fn=fn_true,
        k=2,
        theta_init=np.array([3.0 / 16.0, 9.0 / 160.0], dtype=np.float64),
        theta_true=np.array([3.0 / 16.0, 9.0 / 160.0], dtype=np.float64),
        bounds=[(0.05, 0.5), (0.01, 0.2)],
        expr=true_expr,
    )

    if rival_type == "linear":
        def fn_rival(re: np.ndarray, theta: np.ndarray) -> np.ndarray:
            cd_stokes = 24.0 / re
            return cd_stokes * (1.0 + theta[0] * re)

        m_rival = CandidateModel(
            name="Oseen Linear Correction (No Log)",
            fn=fn_rival,
            k=1,
            theta_init=np.array([3.0 / 16.0], dtype=np.float64),
            bounds=[(0.01, 1.0)],
            expr="(24/Re)*(1 + theta1*Re)",
        )
    else:
        def fn_rival(re: np.ndarray, theta: np.ndarray) -> np.ndarray:
            cd_stokes = 24.0 / re
            return cd_stokes * (1.0 + theta[0] * re + theta[1] * (re**2))

        m_rival = CandidateModel(
            name="Quadratic Polynomial Correction (No Log)",
            fn=fn_rival,
            k=2,
            theta_init=np.array([3.0 / 16.0, 0.0], dtype=np.float64),
            bounds=[(0.01, 1.0), (-0.5, 0.5)],
            expr="(24/Re)*(1 + theta1*Re + theta2*Re^2)",
        )

    return m_true, m_rival
