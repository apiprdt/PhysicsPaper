import itertools
import sympy as sp
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union, Optional, Tuple


@dataclass
class CorrectionEvaluation:
    nmse_residual: float
    nmse_full: float
    true_class: str
    discovered_class: str
    class_match: bool
    ast_edit_distance: int
    parameter_error: Dict[str, float]
    bic: float
    # NEW: honesty flags for the parameter-error numbers above.
    parameter_match_structural: bool = False   # True only if a symbolic-exact
    # permutation was found
    parameter_count_mismatch: bool = False     # True if #true_params != #fit_params


def classify_structure(expr: Union[str, sp.Expr], theta_fit: Optional[Dict[str, float]] = None) -> str:
    """Return the functional class of a correction expression."""
    if isinstance(expr, str):
        try:
            expr = sp.sympify(expr)
        except Exception:
            return "unknown"

    has_exp = False
    has_trig = False
    has_log = False
    has_noninteger_pow = False
    has_rational_denom = False

    constants = {"c", "G", "sigma", "k_e", "pi"}

    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, sp.Function):
            fname = sub.func.__name__
            if fname in ("exp",):
                has_exp = True
            elif fname in ("sin", "cos", "tan", "tanh", "sinh", "cosh", "sinc", "asin", "acos", "atan"):
                has_trig = True
            elif fname in ("log", "ln"):
                has_log = True
        elif isinstance(sub, sp.Pow):
            base, exponent = sub.args
            base_syms = [str(s) for s in base.free_symbols]
            has_var = any(not s.startswith("theta_") and s not in constants for s in base_syms)

            if has_var:
                is_neg = exponent.is_Number and float(exponent) < 0
                is_param = exponent.is_Symbol and str(exponent).startswith("theta_")

                is_degenerate = False
                if is_param and theta_fit is not None:
                    param_name = str(exponent)
                    if param_name in theta_fit:
                        val = theta_fit[param_name]
                        if abs(val - 1.0) < 0.05:
                            is_degenerate = True

                if (not exponent.is_Integer or is_neg or is_param) and not is_degenerate:
                    has_noninteger_pow = True

            if exponent.is_Number and float(exponent) < 0:
                if isinstance(base, sp.Add):
                    has_rational_denom = True

    if has_exp:
        return "exponential"
    if has_trig:
        return "trigonometric"
    if has_log:
        return "logarithmic"
    if has_rational_denom:
        return "rational"
    if has_noninteger_pow:
        return "power_law"
    return "polynomial"


def get_ast_tokens(expr: sp.Expr) -> List[str]:
    tokens = []
    for node in sp.preorder_traversal(expr):
        if node.is_Symbol:
            name = str(node)
            tokens.append("Symbol(theta)" if name.startswith("theta_") else f"Symbol({name})")
        elif node.is_Number:
            tokens.append("Number")
        else:
            tokens.append(node.__class__.__name__)
    return tokens


def compute_levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    size_x, size_y = len(seq1) + 1, len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=int)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y
    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x - 1] == seq2[y - 1]:
                matrix[x, y] = matrix[x - 1, y - 1]
            else:
                matrix[x, y] = min(matrix[x - 1, y] + 1, matrix[x, y - 1] + 1, matrix[x - 1, y - 1] + 1)
    return int(matrix[size_x - 1, size_y - 1])


def _nmse(mse: float, reference: np.ndarray) -> float:
    var_y = float(np.var(reference))
    eps = max(var_y * 1e-6, 1e-36)
    denom = var_y + eps
    return float(mse / denom) if denom > 0 else 1.0


def bic_score(nmse: float, n_params: int, n_points: int) -> float:
    nmse_floored = max(nmse, 1e-6)
    rss = nmse_floored * n_points
    log_likelihood = -n_points / 2 * np.log(rss / n_points + 1e-30)
    return float(-2 * log_likelihood + n_params * np.log(n_points))


def extended_bic_score(nmse: float, n_params: int, n_points: int, n_candidates: int = 1) -> float:
    base_bic = bic_score(nmse, n_params, n_points)
    m_penalty = float(2.0 * np.log(max(1, n_candidates)))
    return base_bic + m_penalty


def match_parameters(
    true_params: Dict[str, float],
    fit_params: Dict[str, float],
    discovered_expr: Optional[sp.Expr] = None,
    true_expr: Optional[sp.Expr] = None,
) -> Tuple[Dict[str, float], bool, bool]:
    """
    Establish correspondence between TRUE parameter names/values and FITTED
    parameter names/values.

    Returns:
        (parameter_error_by_true_name, structural_match, count_mismatch)

    Strategy (in order):
    1. If discovered_expr/true_expr are given and parameter counts match,
       try every permutation of the fit-parameter symbols substituted into
       discovered_expr; if sp.simplify(discovered.subs(perm) - true_expr)==0
       for some permutation, that is an EXACT structural match -- use it and
       set structural_match=True.
    2. Otherwise, fall back to the permutation that numerically MINIMIZES
       total relative error (still tries all permutations -- cheap for the
       small parameter counts used throughout this project). structural_match
       is False in this branch: the pairing is a best-effort heuristic, not a
       confirmed correspondence, and should be reported as such.
    3. If len(fit_params) != len(true_params), no 1:1 correspondence can be
       established at all; return count_mismatch=True and an empty error dict
       rather than silently truncating via zip.
    """
    true_keys = sorted(true_params.keys())
    fit_keys = sorted(fit_params.keys())

    if len(fit_keys) != len(true_keys) or len(true_keys) == 0:
        return {}, False, True

    n = len(true_keys)
    best_perm = None
    best_err = float("inf")
    structural_match = False

    # Try structural (symbolic-exact) correspondence first, if we have the
    # expressions to check it against.
    if discovered_expr is not None and true_expr is not None and n <= 6:
        for perm in itertools.permutations(fit_keys, n):
            # Also need to map true theta symbol names -> perm's fit symbol
            # names to compare structurally: substitute the TRUE expr's
            # theta_i with the same VALUE the perm assigns, and compare
            # against discovered_expr with fit values substituted directly.
            try:
                # Structural check is on the SYMBOLIC form with matched roles,
                # not on plugged-in numbers (numbers would trivially differ by
                # fit error) -- so instead verify role correspondence via
                # symbol substitution equivalence:
                role_map = {sp.Symbol(fk): sp.Symbol(tk) for fk, tk in zip(perm, true_keys)}
                relabeled_discovered = discovered_expr.subs(role_map, simultaneous=True)
                diff = sp.simplify(relabeled_discovered - true_expr)
                if diff == 0:
                    best_perm = perm
                    structural_match = True
                    break
            except Exception:
                continue

    # Fall back to (or additionally compute, for reporting) the numerically
    # best permutation if no exact structural match was found.
    if best_perm is None:
        for perm in itertools.permutations(fit_keys, n):
            total_err = 0.0
            ok = True
            for tk, fk in zip(true_keys, perm):
                tv, fv = true_params[tk], fit_params[fk]
                if not (np.isfinite(tv) and np.isfinite(fv)):
                    ok = False
                    break
                total_err += abs(fv - tv) / max(abs(tv), 1e-15)
            if ok and total_err < best_err:
                best_err = total_err
                best_perm = perm
        structural_match = False

    if best_perm is None:
        return {}, False, True

    param_errors = {}
    for tk, fk in zip(true_keys, best_perm):
        tv, fv = true_params[tk], fit_params[fk]
        err = abs(fv - tv) / abs(tv) if abs(tv) > 1e-15 else abs(fv - tv)
        param_errors[tk] = float(err)

    return param_errors, structural_match, False


def _evaluate_delta_array(
    expr_str: str,
    X: Dict[str, np.ndarray],
    theta_fit: Dict[str, float],
    classical_constants: Dict[str, float],
    n_points: int,
) -> np.ndarray:
    try:
        expr = sp.sympify(expr_str)
        subs = {sp.Symbol(k): v for k, v in {**(classical_constants or {}), **theta_fit}.items()}
        expr = expr.subs(subs)
        free_syms = sorted(expr.free_symbols, key=lambda s: str(s))
        if not free_syms:
            return np.full(n_points, float(expr))
        fn = sp.lambdify(free_syms, expr, modules=["scipy", "numpy"])
        args = []
        for sym in free_syms:
            name = str(sym)
            if name not in X:
                raise KeyError(name)
            args.append(X[name])
        return np.asarray(fn(*args), dtype=float)
    except Exception:
        return np.zeros(n_points)


# WARNING FOR ANY FUTURE EDITOR (human or AI):
# NEVER OR nmse_full into the class_match criterion. nmse_full is almost
# always small under correction-first design because the baseline dominates
# -- that makes it a false shortcut to PASS. This rule is unchanged from the
# original audit and remains correct; it is preserved here as-is.
def evaluate_correction(
    discovered_expr_str: str,
    scenario,
    X: Dict[str, np.ndarray],
    y_obs: np.ndarray,
    y_classical: np.ndarray,
    theta_fit: Dict[str, float]
) -> CorrectionEvaluation:
    try:
        discovered_expr = sp.sympify(discovered_expr_str)
        true_expr = sp.sympify(scenario.correction_expr)
    except Exception:
        return CorrectionEvaluation(
            nmse_residual=1.0, nmse_full=1.0, true_class=scenario.correction_class,
            discovered_class="unparseable", class_match=False, ast_edit_distance=999,
            parameter_error={}, bic=9999.0, parameter_match_structural=False,
            parameter_count_mismatch=True,
        )

    true_cls = scenario.correction_class
    disc_cls = classify_structure(discovered_expr, theta_fit)

    seq_disc = get_ast_tokens(discovered_expr)
    seq_true = get_ast_tokens(true_expr)
    ast_dist = compute_levenshtein_distance(seq_disc, seq_true)

    n_points = len(y_obs)
    delta_discovered = _evaluate_delta_array(discovered_expr_str, X, theta_fit, scenario.classical_constants, n_points)

    if scenario.correction_type == "multiplicative":
        y_recon = y_classical * (1.0 + delta_discovered)
        residual_obs = y_obs / y_classical - 1.0
        mse_res = np.mean((delta_discovered - residual_obs) ** 2)
        nmse_res = _nmse(mse_res, residual_obs)
    else:
        y_recon = y_classical + delta_discovered
        residual_obs = y_obs - y_classical
        mse_res = np.mean((delta_discovered - residual_obs) ** 2)
        nmse_res = _nmse(mse_res, residual_obs)

    mse_full = np.mean((y_recon - y_obs) ** 2)
    nmse_full = _nmse(mse_full, y_obs)

    from adcd.constants import NMSE_SUCCESS_THRESHOLD
    is_genuinely_good_fit = bool(nmse_res < NMSE_SUCCESS_THRESHOLD)

    class_match = bool(
        (true_cls == disc_cls) and is_genuinely_good_fit and bool(discovered_expr_str.strip())
    )

    # FIXED parameter-recovery matching (see match_parameters docstring).
    param_errors, structural_match, count_mismatch = match_parameters(
        true_params=scenario.correction_constants,
        fit_params={k: v for k, v in theta_fit.items() if k.startswith("theta_")},
        discovered_expr=discovered_expr,
        true_expr=true_expr,
    )

    n_params = len([k for k in theta_fit.keys() if k.startswith("theta_")])
    n_points = len(y_obs)
    bic_val = bic_score(nmse_res, n_params, n_points)

    return CorrectionEvaluation(
        nmse_residual=float(nmse_res),
        nmse_full=float(nmse_full),
        true_class=true_cls,
        discovered_class=disc_cls,
        class_match=class_match,
        ast_edit_distance=ast_dist,
        parameter_error=param_errors,
        bic=bic_val,
        parameter_match_structural=structural_match,
        parameter_count_mismatch=count_mismatch,
    )
