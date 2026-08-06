"""
Independent Evaluator for ADCD Pre-Registered Benchmark Runs.
Decoupled from adcd codebase. Evaluates candidate formulas on held-out test points
and computes tree edit distance against ground truth.
"""

import sympy as sp
import numpy as np
from typing import Dict, Any


def compute_tree_distance(expr1_str: str, expr2_str: str) -> float:
    """
    Computes normalized structural distance between two SymPy expression trees.
    """
    try:
        e1 = sp.sympify(expr1_str)
        e2 = sp.sympify(expr2_str)
    except Exception:
        return 1.0

    # Symbolic simplification check
    try:
        if sp.simplify(e1 - e2) == 0:
            return 0.0
    except Exception:
        pass

    srepr1 = sp.srepr(e1)
    srepr2 = sp.srepr(e2)

    # Levenshtein distance on srepr tokens as robust AST distance proxy
    len1, len2 = len(srepr1), len(srepr2)
    if max(len1, len2) == 0:
        return 0.0

    # Simple dynamic programming Levenshtein distance
    dp = np.zeros((len1 + 1, len2 + 1), dtype=int)
    for i in range(len1 + 1):
        dp[i][0] = i
    for j in range(len2 + 1):
        dp[0][j] = j

    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            cost = 0 if srepr1[i - 1] == srepr2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost
            )

    dist = dp[len1][len2] / max(len1, len2)
    return float(dist)


def evaluate_candidate(
    candidate_str: str,
    ground_truth_str: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    var_names: list[str]
) -> Dict[str, Any]:
    """
    Evaluates candidate expression numerically on held-out test set and structurally.
    """
    try:
        candidate_expr = sp.sympify(candidate_str)
    except Exception as e:
        return {
            "valid_syntax": False,
            "nmse": 1e6,
            "tree_distance": 1.0,
            "is_recovered": False,
            "error": f"SymPy parse error: {e}"
        }

    # Evaluate numerical NMSE
    try:
        # Create lambda for numerical evaluation
        symbols = [sp.Symbol(v) for v in var_names]
        fn = sp.lambdify(symbols, candidate_expr, modules=["numpy"])

        if X_test.ndim == 1:
            y_pred = fn(X_test)
        else:
            y_pred = fn(*[X_test[:, i] for i in range(X_test.shape[1])])

        y_pred = np.asarray(y_pred, dtype=float)
        if y_pred.shape != y_test.shape:
            y_pred = np.full_like(y_test, y_pred)

        mse = np.mean((y_test - y_pred) ** 2)
        var_y = np.var(y_test)
        nmse = mse / (var_y + 1e-9)
        nmse = float(np.nan_to_num(nmse, nan=1e6, posinf=1e6, neginf=1e6))
    except Exception as e:
        nmse = 1e6

    tree_dist = compute_tree_distance(candidate_str, ground_truth_str)
    is_recovered = (nmse < 0.05) and (tree_dist < 0.4)

    return {
        "valid_syntax": True,
        "nmse": float(nmse),
        "tree_distance": float(tree_dist),
        "is_recovered": bool(is_recovered)
    }
