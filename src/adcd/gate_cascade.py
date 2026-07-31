"""
Physics Gate Cascade Module for ADCD.
Applies sequential filtering on candidate expressions:
Stage 0: Parsing check
Stage 1: Complexity bound (AST depth <= max_depth, tokens <= max_tokens)
Stage 2: Dimensional homogeneity check
Stage 3: Asymptotic Recovery Constraint (ARC gate)
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union
import sympy as sp

from adcd.arc_gate import ARCGate, ARCCheckResult
from adcd.dimensional_checker import DimensionalChecker

logger = logging.getLogger("PhysicsGateCascade")


@dataclass
class GateResult:
    passed: bool
    rejection_reason: Optional[str]
    rejected_stage: Optional[int]
    ast_depth: int
    token_count: int
    dim_valid: bool
    arc_result: Optional[ARCCheckResult]


class PhysicsGateCascade:
    """
    Cascade of physics-informed filters for symbolic regression candidate expressions.
    """

    def __init__(
        self,
        max_depth: int = 7,
        max_tokens: int = 20,
        arc_target_limit: float = 0.0,
        arc_tolerance: float = 1e-4,
        enable_dimensional_check: bool = True,
        enable_arc_check: bool = True
    ):
        self.max_depth = max_depth
        self.max_tokens = max_tokens
        self.enable_dimensional_check = enable_dimensional_check
        self.enable_arc_check = enable_arc_check
        self.arc_gate = ARCGate(target_limit=arc_target_limit, tolerance=arc_tolerance)
        self.dim_checker = DimensionalChecker() if enable_dimensional_check else None

    def _compute_ast_depth(self, expr: sp.Expr) -> int:
        if not expr.args:
            return 1
        return 1 + max(self._compute_ast_depth(arg) for arg in expr.args)

    def _count_tokens(self, expr: sp.Expr) -> int:
        return len(sp.srepr(expr).replace(" ", "").split(","))

    def check(
        self,
        candidate_expr: Union[str, sp.Expr],
        limit_vars: Optional[List[str]] = None,
        limit_points: Optional[List[Any]] = None,
        variable_units: Optional[Dict[str, str]] = None,
        target_units: Optional[str] = None,
        constants: Optional[Dict[str, float]] = None
    ) -> GateResult:
        """
        Runs candidate through all 4 stages of the physics gate cascade.
        """
        # Stage 0: SymPy Parse Check
        if isinstance(candidate_expr, str):
            try:
                expr = sp.sympify(candidate_expr)
            except Exception as e:
                return GateResult(
                    passed=False,
                    rejection_reason=f"Stage 0 (Parse Error): {e}",
                    rejected_stage=0,
                    ast_depth=0,
                    token_count=0,
                    dim_valid=False,
                    arc_result=None
                )
        else:
            expr = candidate_expr

        # Stage 1: Complexity Check
        try:
            depth = self._compute_ast_depth(expr)
            tokens = self._count_tokens(expr)
        except Exception as e:
            return GateResult(
                passed=False,
                rejection_reason=f"Stage 1 (Complexity Computation Failed): {e}",
                rejected_stage=1,
                ast_depth=0,
                token_count=0,
                dim_valid=False,
                arc_result=None
            )

        if depth > self.max_depth:
            return GateResult(
                passed=False,
                rejection_reason=f"Stage 1 (Depth Violation): depth {depth} > max {self.max_depth}",
                rejected_stage=1,
                ast_depth=depth,
                token_count=tokens,
                dim_valid=False,
                arc_result=None
            )

        if tokens > self.max_tokens:
            return GateResult(
                passed=False,
                rejection_reason=f"Stage 1 (Token Violation): tokens {tokens} > max {self.max_tokens}",
                rejected_stage=1,
                ast_depth=depth,
                token_count=tokens,
                dim_valid=False,
                arc_result=None
            )

        # Stage 2: Dimensional Homogeneity Check
        dim_valid = True
        if self.enable_dimensional_check and variable_units and target_units:
            try:
                dim_result = self.dim_checker.check_homogeneity(
                    expr=expr,
                    var_units=variable_units,
                    target_unit=target_units
                )
                dim_valid = dim_result.is_valid
                if not dim_valid:
                    return GateResult(
                        passed=False,
                        rejection_reason=f"Stage 2 (Dimensional Violation): {dim_result.reason}",
                        rejected_stage=2,
                        ast_depth=depth,
                        token_count=tokens,
                        dim_valid=False,
                        arc_result=None
                    )
            except Exception as e:
                logger.debug(f"Dimensional check exception: {e}")

        # Stage 3: Asymptotic Recovery Constraint (ARC Gate)
        arc_res = None
        if self.enable_arc_check and limit_vars:
            arc_res = self.arc_gate.check(
                candidate_expr=expr,
                limit_vars=limit_vars,
                limit_points=limit_points,
                constants=constants
            )
            if not arc_res.passed:
                return GateResult(
                    passed=False,
                    rejection_reason=f"Stage 3 (ARC Violation): {arc_res.rejection_reason}",
                    rejected_stage=3,
                    ast_depth=depth,
                    token_count=tokens,
                    dim_valid=dim_valid,
                    arc_result=arc_res
                )

        return GateResult(
            passed=True,
            rejection_reason=None,
            rejected_stage=None,
            ast_depth=depth,
            token_count=tokens,
            dim_valid=dim_valid,
            arc_result=arc_res
        )
