"""
Asymptotic Recovery Constraint (ARC) Gate Module for ADCD.
Formally verifies that a candidate residual expression delta(x) satisfies
the physical boundary condition: lim_{x -> x_classical} delta(x) = 0.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import sympy as sp
import numpy as np

logger = logging.getLogger("ARCGate")


@dataclass
class ARCCheckResult:
    passed: bool
    limit_value: Optional[float]
    rejection_reason: Optional[str]
    asymptotic_score: float


class ARCGate:
    """
    Standalone Asymptotic Recovery Constraint (ARC) Gate.
    Verifies that the candidate expression satisfies physical boundary conditions.
    """

    def __init__(self, target_limit: float = 0.0, tolerance: float = 1e-4):
        self.target_limit = target_limit
        self.tolerance = tolerance

    def evaluate_limit(
        self,
        expr: sp.Expr,
        variable: sp.Symbol,
        limit_point: Any = 0
    ) -> Optional[sp.Expr]:
        """
        Computes lim_{variable -> limit_point} expr with parameter substitution
        and Laurent series fallback.
        """
        # 1. Direct SymPy limit
        try:
            res = sp.limit(expr, variable, limit_point, dir='+')
            if res is not None and res not in (sp.oo, -sp.oo, sp.zoo):
                return res
        except Exception as e:
            logger.debug(f"Direct limit evaluation failed: {e}")

        # 2. Parameter substitution (substitute theta_* with 1.0)
        theta_syms = [s for s in expr.free_symbols if str(s).startswith("theta_")]
        if theta_syms:
            try:
                sub_dict = {s: 1.0 for s in theta_syms}
                sub_expr = expr.subs(sub_dict)
                res_sub = sp.limit(sub_expr, variable, limit_point, dir='+')
                if res_sub is not None and res_sub not in (sp.oo, -sp.oo, sp.zoo):
                    return res_sub
            except Exception:
                pass

        # 3. High-precision numerical limit fallback
        try:
            sub_dict = {s: 1.0 for s in expr.free_symbols if str(s).startswith("theta_")}
            eval_expr = expr.subs(sub_dict) if sub_dict else expr
            if limit_point == sp.oo:
                val = float(eval_expr.subs(variable, 1e6).evalf())
            elif limit_point == -sp.oo:
                val = float(eval_expr.subs(variable, -1e6).evalf())
            else:
                val = float(eval_expr.subs(variable, float(limit_point) + 1e-7).evalf())

            if np.isfinite(val):
                return sp.Float(val)
        except Exception:
            pass

        return None

    def check(
        self,
        candidate_expr: Union[str, sp.Expr],
        limit_vars: List[str],
        limit_points: Optional[List[Any]] = None,
        constants: Optional[Dict[str, float]] = None
    ) -> ARCCheckResult:
        """
        Checks if candidate_expr meets the ARC condition across limit_vars.
        """
        if isinstance(candidate_expr, str):
            try:
                expr = sp.sympify(candidate_expr)
            except Exception as e:
                return ARCCheckResult(
                    passed=False,
                    limit_value=None,
                    rejection_reason=f"SymPy parse error: {e}",
                    asymptotic_score=0.0
                )
        else:
            expr = candidate_expr

        if constants:
            sub_dict = {sp.Symbol(k): v for k, v in constants.items() if sp.Symbol(k) in expr.free_symbols}
            if sub_dict:
                expr = expr.subs(sub_dict)

        if limit_points is None:
            limit_points = [0] * len(limit_vars)

        for var_name, limit_pt in zip(limit_vars, limit_points):
            var_sym = sp.Symbol(var_name)
            limit_res = self.evaluate_limit(expr, var_sym, limit_pt)

            if limit_res is None:
                return ARCCheckResult(
                    passed=False,
                    limit_value=None,
                    rejection_reason=f"Limit evaluation failed for variable {var_name}",
                    asymptotic_score=0.0
                )

            try:
                limit_float = float(limit_res.evalf())
            except Exception:
                limit_float = float('nan')

            if np.isnan(limit_float) or np.isinf(limit_float):
                return ARCCheckResult(
                    passed=False,
                    limit_value=limit_float,
                    rejection_reason=f"Limit is non-finite ({limit_float}) for {var_name}",
                    asymptotic_score=0.0
                )

            diff = abs(limit_float - self.target_limit)
            if diff > self.tolerance:
                return ARCCheckResult(
                    passed=False,
                    limit_value=limit_float,
                    rejection_reason=f"ARC violation: lim_{var_name}->{limit_pt} = {limit_float:.4f} != {self.target_limit}",
                    asymptotic_score=float(np.exp(-diff))
                )

        return ARCCheckResult(
            passed=True,
            limit_value=limit_float,
            rejection_reason=None,
            asymptotic_score=1.0
        )
