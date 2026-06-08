import sympy as sp
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Union, Optional

@dataclass
class CorrectionEvaluation:
    nmse_residual: float               # NMSE of Δ_discovered vs residual data
    nmse_full: float                   # NMSE of reconstructed y vs y_obs
    true_class: str                    # Ground truth category, e.g., "exponential"
    discovered_class: str              # Classified category of discovered term
    class_match: bool                  # True if same structural family
    ast_edit_distance: int             # Levenshtein distance on AST preorder token representation
    parameter_error: Dict[str, float]  # Percentage error for recovered free parameters |θ_fit - θ_true|/|θ_true|
    bic: float                         # Bayesian Information Criterion score

def classify_structure(expr: Union[str, sp.Expr], theta_fit: Optional[Dict[str, float]] = None) -> str:
    """
    Classifies a SymPy expression or string into one of the structural classes:
    'exponential', 'trigonometric', 'logarithmic', 'power_law', 'rational', 'polynomial'
    
    Uses AST traversal for robust classification instead of fragile string matching.
    """
    if isinstance(expr, str):
        try:
            expr = sp.sympify(expr)
        except Exception:
            return "unknown"
    
    # Walk the AST to detect function types
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
            elif fname in ("sin", "cos", "tan", "tanh", "sinh", "cosh", "sinc",
                           "asin", "acos", "atan"):
                has_trig = True
            elif fname in ("log", "ln"):
                has_log = True
        elif isinstance(sub, sp.Pow):
            base, exponent = sub.args
            
            # Identify if base has physical variables (non-parameter, non-constant symbols)
            base_syms = [str(s) for s in base.free_symbols]
            has_var = any(
                not s.startswith("theta_") and s not in constants 
                for s in base_syms
            )
            
            if has_var:
                # Check for power law:
                # 1. Non-integer exponent (e.g. x**1.5)
                # 2. Negative exponent (e.g. T**-4 or T**-1)
                # 3. Exponent is a parameter symbol (e.g. x**theta_1)
                is_neg = exponent.is_Number and float(exponent) < 0
                is_param = exponent.is_Symbol and str(exponent).startswith("theta_")
                
                # Check for degenerate linear case (exponent ~ 1.0)
                is_degenerate = False
                if is_param and theta_fit is not None:
                    param_name = str(exponent)
                    if param_name in theta_fit:
                        val = theta_fit[param_name]
                        if abs(val - 1.0) < 0.05:
                            is_degenerate = True
                            
                if (not exponent.is_Integer or is_neg or is_param) and not is_degenerate:
                    has_noninteger_pow = True
                    
            # Check for rational function: x / (x + a) style denominators
            if exponent.is_Number and float(exponent) < 0:
                if isinstance(base, sp.Add):
                    has_rational_denom = True
    
    # Priority ordering: transcendental > power_law > rational > polynomial
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
    
    # Default: polynomial (integer powers, additions, multiplications)
    return "polynomial"

def get_ast_tokens(expr: sp.Expr) -> List[str]:
    """
    Returns a sequence of token strings by preorder traversal of the SymPy AST.
    This provides a robust, platform-independent representation of AST structure.
    """
    tokens = []
    for node in sp.preorder_traversal(expr):
        if node.is_Symbol:
            # Normalize theta symbols so their specific indices don't inflate edit distance
            name = str(node)
            if name.startswith("theta_"):
                tokens.append("Symbol(theta)")
            else:
                tokens.append(f"Symbol({name})")
        elif node.is_Number:
            # We don't want slight constant differences to skew structure distance
            tokens.append("Number")
        else:
            tokens.append(node.__class__.__name__)
    return tokens

def compute_levenshtein_distance(seq1: List[str], seq2: List[str]) -> int:
    """Computes the Levenshtein edit distance between two token sequences."""
    size_x = len(seq1) + 1
    size_y = len(seq2) + 1
    matrix = np.zeros((size_x, size_y), dtype=int)
    for x in range(size_x):
        matrix[x, 0] = x
    for y in range(size_y):
        matrix[0, y] = y

    for x in range(1, size_x):
        for y in range(1, size_y):
            if seq1[x-1] == seq2[y-1]:
                matrix[x, y] = matrix[x-1, y-1]
            else:
                matrix[x, y] = min(
                    matrix[x-1, y] + 1,    # deletion
                    matrix[x, y-1] + 1,    # insertion
                    matrix[x-1, y-1] + 1   # substitution
                )
    return int(matrix[size_x - 1, size_y - 1])

def bic_score(nmse: float, n_params: int, n_points: int) -> float:
    """Lower is better. Penalizes parameter count.
    
    A noise floor of 1e-6 is applied to NMSE before computing the log-likelihood.
    This prevents BIC from rewarding extra parameters when NMSE falls into 
    floating-point precision territory (< 1e-6), where numerical noise dominates 
    and additional parameters provide no real information gain.
    """
    # Noise floor: differences below 1e-6 NMSE are numerically indistinguishable
    # Without this, a 3-param model that shaves 1e-14 off a 1e-14 NMSE beats a
    # 1-param model via BIC despite conveying zero additional physical information.
    nmse_floored = max(nmse, 1e-6)
    rss = nmse_floored * n_points
    log_likelihood = -n_points / 2 * np.log(rss / n_points + 1e-30)
    return float(-2 * log_likelihood + n_params * np.log(n_points))


def evaluate_correction(
    discovered_expr_str: str,
    scenario,
    X: Dict[str, np.ndarray],
    y_obs: np.ndarray,
    y_classical: np.ndarray,
    theta_fit: Dict[str, float]
) -> CorrectionEvaluation:
    """
    Computes all standard physical, structural, and symbolic metrics 
    comparing discovered correction term against ground truth.
    """
    # 1. Structural parse
    try:
        discovered_expr = sp.sympify(discovered_expr_str)
        true_expr = sp.sympify(scenario.correction_expr)
    except Exception:
        # Return fallback high-error evaluation if expressions cannot be parsed
        return CorrectionEvaluation(
            nmse_residual=1.0,
            nmse_full=1.0,
            true_class=scenario.correction_class,
            discovered_class="unparseable",
            class_match=False,
            ast_edit_distance=999,
            parameter_error={},
            bic=9999.0
        )

    # 2. Structural classification
    true_cls = scenario.correction_class
    disc_cls = classify_structure(discovered_expr, theta_fit)
    class_match = (true_cls == disc_cls)

    # 3. AST Levenshtein Distance
    seq_disc = get_ast_tokens(discovered_expr)
    seq_true = get_ast_tokens(true_expr)
    ast_dist = compute_levenshtein_distance(seq_disc, seq_true)

    # 4. Numerical NMSE metrics
    # Re-evaluate discovered expression with fitted thetas on data
    local_eval_dict = {**X, **scenario.classical_constants, **theta_fit}
    eval_env = {
        "np": np, 
        "sp": sp,
        "exp": np.exp,
        "sin": np.sin,
        "cos": np.cos,
        "tanh": np.tanh,
        "log": np.log,
        "sqrt": np.sqrt
    }
    
    # We substitute fitted parameters into expression string for evaluation
    subbed_str = discovered_expr_str
    for k, v in theta_fit.items():
        subbed_str = subbed_str.replace(k, f"({v})")
        
    try:
        delta_discovered = eval(subbed_str, eval_env, local_eval_dict)
    except Exception:
        delta_discovered = np.zeros_like(y_obs)

    # Reconstruct y
    if scenario.correction_type == "multiplicative":
        y_recon = y_classical * (1.0 + delta_discovered)
        # Compute residual NMSE
        residual_obs = y_obs / y_classical - 1.0
        var_res = np.var(residual_obs)
        nmse_res = np.mean((delta_discovered - residual_obs)**2) / var_res if var_res > 1e-30 else 1.0
    else: # additive
        y_recon = y_classical + delta_discovered
        residual_obs = y_obs - y_classical
        var_res = np.var(residual_obs)
        nmse_res = np.mean((delta_discovered - residual_obs)**2) / var_res if var_res > 1e-30 else 1.0

    var_y = np.var(y_obs)
    nmse_full = np.mean((y_recon - y_obs)**2) / var_y if var_y > 1e-30 else 1.0

    # 5. Parameter recovery error
    # Match fitted thetas to scenario correction_constants
    # Note: Because the naming index might differ (e.g. theta_0 fit maps to theta_0 true), 
    # we map them in ascending order of their names:
    # Sorted true thetas: e.g. ["theta_0", "theta_1"]
    # Sorted fit thetas: e.g. ["theta_0", "theta_1"]
    sorted_true_keys = sorted(scenario.correction_constants.keys())
    sorted_fit_keys = sorted([k for k in theta_fit.keys() if k.startswith("theta_")])
    
    param_errors = {}
    for i, true_k in enumerate(sorted_true_keys):
        if i < len(sorted_fit_keys):
            fit_k = sorted_fit_keys[i]
            true_v = scenario.correction_constants[true_k]
            fit_v = theta_fit[fit_k]
            
            # Compute percentage absolute difference
            if abs(true_v) > 1e-15:
                err = abs(fit_v - true_v) / abs(true_v)
            else:
                err = abs(fit_v - true_v)
            param_errors[true_k] = float(err)

    # 6. BIC calculation
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
        bic=bic_val
    )
