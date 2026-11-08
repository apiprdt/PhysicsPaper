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

# Exact Euler-Mascheroni constant and Chester-Breach (1969) O(Re^2) constant
EULER_GAMMA = 0.5772156649015329
CHESTER_BREACH_CONST = EULER_GAMMA + (5.0 / 3.0) * np.log(2.0) - (323.0 / 360.0)


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
            theta = self.theta_true if self.theta_true is not None else self.theta_init
        return np.asarray(self.fn(x, np.asarray(theta, dtype=np.float64)), dtype=np.float64)


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


def _identify_var_and_params(
    free_syms: List[sp.Symbol],
    explicit_var: Optional[str] = None,
) -> Tuple[sp.Symbol, List[sp.Symbol]]:
    """
    Safely separate the independent physical variable from fitted parameters
    without accidentally picking 'theta_0' alphabetically as the independent variable.
    """
    if not free_syms:
        return sp.Symbol("x"), []

    if explicit_var:
        matches = [s for s in free_syms if s.name == explicit_var]
        if matches:
            var_sym = matches[0]
            return var_sym, [s for s in free_syms if s != var_sym]

    # Identify obvious parameter names first
    param_prefixes = ("theta", "phi", "param", "c_", "k_", "a_", "b_")
    non_params = [
        s for s in free_syms
        if not s.name.startswith(param_prefixes)
        and not (len(s.name) == 2 and s.name[0] in "abc" and s.name[1].isdigit())
    ]

    known_vars = ("x", "u", "r", "Re", "v", "t", "T", "dV", "g_bar")
    for kv in known_vars:
        for s in non_params:
            if s.name == kv:
                return s, [p for p in free_syms if p != s]

    if non_params:
        var_sym = non_params[0]
        return var_sym, [p for p in free_syms if p != var_sym]

    # Last fallback
    return free_syms[0], free_syms[1:]


def _build_candidate_model(
    model: Union[CandidateModel, Callable[..., np.ndarray], str, sp.Expr],
    default_name: str,
    k_default: int = 1,
    theta_default: Optional[List[float]] = None,
    bounds_default: Optional[List[Tuple[Optional[float], Optional[float]]]] = None,
    var_name: Optional[str] = None,
) -> CandidateModel:
    """Wrap or convert arbitrary model inputs into CandidateModel."""
    if isinstance(model, CandidateModel):
        return model

    if isinstance(model, (str, sp.Expr)):
        expr_str = str(model)
        sym_expr = sp.sympify(expr_str) if isinstance(model, str) else model
        free_syms = sorted(sym_expr.free_symbols, key=lambda s: s.name)
        var_sym, param_syms = _identify_var_and_params(free_syms, explicit_var=var_name)

        k = len(param_syms)
        lam = sp.lambdify([var_sym] + param_syms, sym_expr, modules=["numpy"])

        def fn(x_arr: np.ndarray, theta_arr: np.ndarray) -> np.ndarray:
            if k == 0:
                res = lam(x_arr)
            else:
                res = lam(x_arr, *theta_arr)
            return np.broadcast_to(np.asarray(res, dtype=np.float64), x_arr.shape).copy()

        theta_init = (
            np.ones(k, dtype=np.float64)
            if theta_default is None
            else np.asarray(theta_default, dtype=np.float64)
        )
        return CandidateModel(
            name=default_name,
            fn=fn,
            k=k,
            theta_init=theta_init,
            theta_true=theta_init.copy(),
            bounds=bounds_default,
            expr=expr_str,
        )

    if callable(model):
        theta_init = (
            np.ones(k_default, dtype=np.float64)
            if theta_default is None
            else np.asarray(theta_default, dtype=np.float64)
        )
        return CandidateModel(
            name=default_name,
            fn=model,
            k=k_default,
            theta_init=theta_init,
            theta_true=theta_init.copy(),
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
        sigma = float(sigma_model)
        w = np.maximum(1e-15, np.abs(y))
        return sigma, w

    if isinstance(sigma_model, NoiseSpec):
        sigma = float(sigma_model.value or 0.01)
        if sigma_model.type == "relative":
            w = np.maximum(1e-15, np.abs(y))
        else:
            w = np.ones_like(y, dtype=np.float64)
        return sigma, w

    if isinstance(sigma_model, dict):
        noise_type = sigma_model.get("type", "relative")
        sigma = float(sigma_model.get("value", 0.01))
        if noise_type == "relative":
            w = np.maximum(1e-15, np.abs(y))
        else:
            w = np.ones_like(y, dtype=np.float64)
        return sigma, w

    if callable(sigma_model):
        sig_arr = np.asarray(sigma_model(x, y), dtype=np.float64)
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
        if x_min <= 0:
            raise ValueError("log grid requires strictly positive x_min > 0")
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
        return float(np.sum(res ** 2)), np.array([], dtype=np.float64)

    def objective(theta: np.ndarray) -> float:
        try:
            y2 = m2.predict(x, theta)
            if np.any(np.isnan(y2)) or np.any(np.isinf(y2)):
                return 1e30
            res = (y1 - y2) / w
            return float(np.sum(res ** 2))
        except Exception:
            return 1e30

    bounds = m2.bounds
    res_opt = opt.minimize(
        objective,
        m2.theta_init,
        method="L-BFGS-B",
        bounds=bounds,
        options={"ftol": 1e-14, "gtol": 1e-10},
    )

    if bounds is None:
        res_nm = opt.minimize(
            objective,
            res_opt.x,
            method="Nelder-Mead",
            options={"xatol": 1e-12, "fatol": 1e-14},
        )
        if res_nm.fun < res_opt.fun:
            res_opt = res_nm

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
    """
    delta_k = k2 - k1
    penalty = (
        delta_k * (np.log(N) - 1.0)
        if use_expectation_correction
        else delta_k * np.log(N)
    )
    target_chi2 = threshold - penalty

    if target_chi2 <= 0.0:
        return float("inf")

    if s0 <= 1e-30:
        return None

    return float(np.sqrt(s0 / target_chi2))


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
    Uses fast continuum density scaling on a reference grid (N_ref = 500) followed
    by exact discrete verification around the root to avoid O(N_max) optimization bottlenecks.
    """
    n_ref = 500
    x_ref = generate_grid(window, n_ref, grid_type)
    y1_ref = m1.predict(x_ref)
    sigma_curr, w_ref = _resolve_noise_weights(sigma_model, x_ref, y1_ref)
    s0_ref, _ = compute_asimov_mismatch(m1, m2, x_ref, w_ref)
    s0_per_point = s0_ref / float(n_ref)

    delta_k = m2.k - m1.k

    def continuum_eq(n_val: float) -> float:
        chi2 = (n_val * s0_per_point) / (sigma_curr ** 2)
        p = (
            delta_k * (np.log(n_val) - 1.0)
            if use_expectation_correction
            else delta_k * np.log(n_val)
        )
        return chi2 + p - threshold

    if continuum_eq(float(n_min)) >= 0.0:
        return n_min

    if continuum_eq(float(n_max)) < 0.0:
        return None

    try:
        sol = opt.root_scalar(continuum_eq, bracket=[float(n_min), float(n_max)], method="brentq")
        if not sol.converged:
            return None
        n_est = int(np.ceil(sol.root))
    except Exception:
        return None

    # If n_est is small enough, do exact discrete verification
    if n_est <= 5000:
        def exact_eq(n_val: float) -> float:
            n_int = max(n_min, int(round(n_val)))
            xg = generate_grid(window, n_int, grid_type)
            y1g = m1.predict(xg)
            sig, wg = _resolve_noise_weights(sigma_model, xg, y1g)
            s0_v, _ = compute_asimov_mismatch(m1, m2, xg, wg)
            chi2 = s0_v / (sig ** 2)
            p = (
                delta_k * (np.log(n_int) - 1.0)
                if use_expectation_correction
                else delta_k * np.log(n_int)
            )
            return chi2 + p - threshold

        lo = max(n_min, int(0.7 * n_est))
        hi = min(n_max, max(lo + 10, int(1.4 * n_est)))
        try:
            if exact_eq(lo) >= 0.0:
                return lo
            if exact_eq(hi) >= 0.0:
                sol_ex = opt.root_scalar(exact_eq, bracket=[lo, hi], method="brentq", xtol=1.0)
                if sol_ex.converged:
                    return int(np.ceil(sol_ex.root))
        except Exception:
            pass

    return n_est


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
        chi2 = s0_val / (sig ** 2)
        delta_k = m2.k - m1.k
        p = (
            delta_k * (np.log(N) - 1.0)
            if use_expectation_correction
            else delta_k * np.log(N)
        )
        return chi2 + p - threshold

    if delta_bic_window(x_max_orig) >= 0.0:
        return (x_min, x_max_orig)

    if delta_bic_window(x_max_limit) < 0.0:
        return None

    try:
        sol = opt.root_scalar(
            delta_bic_window, bracket=[x_max_orig, x_max_limit], method="brentq"
        )
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
    """
    cand1 = _build_candidate_model(m1, default_name=m1_name)
    cand2 = _build_candidate_model(m2, default_name=m2_name)

    x = generate_grid(window, N, grid_type)
    y1 = cand1.predict(x)
    sigma_curr, w = _resolve_noise_weights(sigma_model, x, y1)

    s0, opt_theta2 = compute_asimov_mismatch(cand1, cand2, x, w)

    chi2_min = s0 / (sigma_curr ** 2)
    delta_k = cand2.k - cand1.k
    delta_bic_std = chi2_min + delta_k * np.log(N)
    delta_bic_corr = chi2_min + delta_k * (np.log(N) - 1.0)

    active_delta_bic = delta_bic_corr if use_expectation_correction else delta_bic_std
    separable = active_delta_bic >= threshold

    sigma_star = solve_sigma_star(
        s0, cand1.k, cand2.k, N, threshold, use_expectation_correction=False
    )
    sigma_star_corr = solve_sigma_star(
        s0, cand1.k, cand2.k, N, threshold, use_expectation_correction=True
    )

    n_star = None
    window_star = None
    if not separable:
        n_star = solve_n_star(
            cand1, cand2, window, sigma_model, threshold,
            use_expectation_correction=use_expectation_correction,
            grid_type=grid_type,
        )
        window_star = solve_window_star(
            cand1, cand2, window, N, sigma_model, threshold,
            use_expectation_correction=use_expectation_correction,
            grid_type=grid_type,
        )

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
        active_sig_star = sigma_star_corr if use_expectation_correction else sigma_star
        if active_sig_star is not None and not np.isinf(active_sig_star):
            parts.append(
                f"Requires instrument precision σ* <= {active_sig_star*100:.3f}% (current {sigma_curr*100:.2f}%)."
            )
        if n_star is not None:
            parts.append(f"Alternatively, increase sample size to N* >= {n_star} (current N={N}).")
        if window_star is not None:
            parts.append(
                f"Alternatively, extend measurement domain to [{window_star[0]:.2f}, {window_star[1]:.2f}]."
            )
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
    Note: Under Gaussian noise, E[chi2(M2) - chi2(M1)] = chi2_Asimov - (k2 - k1).
    Thus E[Delta BIC] converges asymptotically to asimov_corrected_delta_bic:
        chi2_Asimov + (k2 - k1)*(ln N - 1).
    """
    cand1 = _build_candidate_model(m1, default_name="Model 1")
    cand2 = _build_candidate_model(m2, default_name="Model 2")

    x = generate_grid(window, N, grid_type)
    y1_noiseless = cand1.predict(x)
    sigma_curr, w = _resolve_noise_weights(sigma_model, x, y1_noiseless)
    sigma_i = sigma_curr * w

    asimov_res = design(cand1, cand2, window, N, sigma_model, grid_type=grid_type)
    # Warm-start M2 at its Asimov optimum to prevent local-minimum bias in MC trials
    m2_warm_init = np.asarray(
        asimov_res.metadata.get("optimal_theta2", cand2.theta_init), dtype=np.float64
    )

    rng = np.random.RandomState(seed)
    delta_bics = []

    for _ in range(n_seeds):
        noise = rng.normal(0.0, sigma_i)
        y_noisy = y1_noiseless + noise

        if cand1.k == 0:
            chi2_1 = float(np.sum(((y_noisy - y1_noiseless) / sigma_i) ** 2))
        else:
            def obj1(th: np.ndarray) -> float:
                return float(np.sum(((y_noisy - cand1.predict(x, th)) / sigma_i) ** 2))
            m1_method = "L-BFGS-B" if cand1.bounds else "Nelder-Mead"
            r1 = opt.minimize(obj1, cand1.theta_init, method=m1_method, bounds=cand1.bounds)
            chi2_1 = float(r1.fun)

        if cand2.k == 0:
            chi2_2 = float(np.sum(((y_noisy - cand2.predict(x)) / sigma_i) ** 2))
        else:
            def obj2(th: np.ndarray) -> float:
                return float(np.sum(((y_noisy - cand2.predict(x, th)) / sigma_i) ** 2))
            m2_method = "L-BFGS-B" if cand2.bounds else "Nelder-Mead"
            r2 = opt.minimize(obj2, m2_warm_init, method=m2_method, bounds=cand2.bounds)
            chi2_2 = float(r2.fun)

        bic1 = chi2_1 + cand1.k * np.log(N)
        bic2 = chi2_2 + cand2.k * np.log(N)
        delta_bics.append(bic2 - bic1)

    mean_mc = float(np.mean(delta_bics))
    std_mc = float(np.std(delta_bics, ddof=1))
    sem_mc = float(std_mc / np.sqrt(n_seeds))

    asimov_val = asimov_res.expected_delta_bic
    asimov_corr = asimov_res.expected_delta_bic_corrected

    # Compare against asimov_corr / asimov_val with 2nd-order nonlinear curvature allowance
    diff = min(abs(asimov_corr - mean_mc), abs(asimov_val - mean_mc))
    tol = max(1.5, tolerance_sigma * max(sem_mc, 1e-6))
    within_tol = bool(diff <= tol)

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


def get_sphere_drag_models(
    use_verified_series: bool = True,
    include_chester_breach_const: bool = False,
    rival_type: Literal["linear", "quadratic"] = "linear",
) -> Tuple[CandidateModel, CandidateModel]:
    """
    Returns (M_true, M_rival) for sphere drag experiment design (Re < 1.0).

    Primary Literature:
    - Proudman & Pearson (1957) JFM 2(3), 237-262:
      Delta = (3/16)*Re + (9/160)*Re^2 * ln(Re/2) + O(Re^2)
    - Chester & Breach (1969) JFM 37(4), 751-760:
      Delta = (3/16)*Re + (9/160)*Re^2 * [ln(Re/2) + gamma + (5/3)*ln(2) - 323/360]
    """
    def fn_true(re: np.ndarray, theta: np.ndarray) -> np.ndarray:
        re_safe = np.maximum(1e-16, np.asarray(re, dtype=np.float64))
        cd_stokes = 24.0 / re_safe
        th1 = theta[0]  # 3/16
        th2 = theta[1]  # 9/160
        if use_verified_series:
            log_term = np.log(re_safe / 2.0)
            if include_chester_breach_const:
                log_term = log_term + CHESTER_BREACH_CONST
            delta = th1 * re_safe + th2 * (re_safe ** 2) * log_term
        else:
            delta = th1 * re_safe + th2 * (re_safe ** 2) * np.log(re_safe)
        return cd_stokes * (1.0 + delta)

    if use_verified_series:
        true_expr = (
            "(24/Re)*(1 + (3/16)*Re + (9/160)*Re^2*(ln(Re/2) + C_CB))"
            if include_chester_breach_const
            else "(24/Re)*(1 + (3/16)*Re + (9/160)*Re^2*ln(Re/2))"
        )
    else:
        true_expr = "(24/Re)*(1 + (3/16)*Re + (9/160)*Re^2*ln(Re))"

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
            re_safe = np.maximum(1e-16, np.asarray(re, dtype=np.float64))
            return (24.0 / re_safe) * (1.0 + theta[0] * re_safe)

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
            re_safe = np.maximum(1e-16, np.asarray(re, dtype=np.float64))
            return (24.0 / re_safe) * (1.0 + theta[0] * re_safe + theta[1] * (re_safe ** 2))

        m_rival = CandidateModel(
            name="Quadratic Polynomial Correction (No Log)",
            fn=fn_rival,
            k=2,
            theta_init=np.array([3.0 / 16.0, 0.0], dtype=np.float64),
            bounds=[(0.01, 1.0), (-0.5, 0.5)],
            expr="(24/Re)*(1 + theta1*Re + theta2*Re^2)",
        )

    return m_true, m_rival
