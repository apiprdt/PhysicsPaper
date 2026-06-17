"""Buckingham-Pi dimensionless group engine for multivariable ADCD Phase 2."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import sympy as sp


class BuckinghamPiEngine:
    """
    Computes dimensionless Buckingham-Pi groups from registered variables.

    Uses SVD of the dimensional matrix to find the null space, which
    corresponds to dimensionless combinations (Buckingham, 1914).
    """

    def __init__(self) -> None:
        self.registry: Dict[str, np.ndarray] = {}

    def register(self, name: str, dim_vector: List[int]) -> None:
        """Register a variable with its dimension vector [M, L, T, ...]."""
        self.registry[name] = np.array(dim_vector, dtype=float)

    def register_from_scenario(self, scenario) -> None:
        """Register classical variables and known scale constants from a scenario."""
        from adcd.dimensional_checker import DimensionalChecker

        checker = DimensionalChecker()
        for var in scenario.classical_variables:
            if var in checker.registry:
                self.register(var, checker.registry[var])
        for const_name in scenario.classical_constants:
            if const_name in checker.registry:
                self.register(const_name, checker.registry[const_name])
            else:
                base = const_name.replace("_ref", "").replace("_0", "")
                if base in checker.registry:
                    self.register(const_name, checker.registry[base])

    def compute_pi_groups(self) -> List[sp.Expr]:
        """
        Compute independent dimensionless Pi groups.

        Returns SymPy expressions, each dimensionless by construction.
        """
        if len(self.registry) < 2:
            return []

        names = list(self.registry.keys())
        dim_matrix = np.array([self.registry[n] for n in names]).T

        _, singular_values, vt = np.linalg.svd(dim_matrix, full_matrices=True)
        rank = int(np.sum(singular_values > 1e-10))
        null_basis = vt[rank:]

        syms = {n: sp.Symbol(n) for n in names}
        pi_groups: List[sp.Expr] = []

        for row in null_basis:
            factors = []
            for name, exp in zip(names, row):
                if abs(exp) < 1e-6:
                    continue
                rational_exp = sp.Rational(int(round(exp * 12)), 12)
                factors.append(syms[name] ** rational_exp)
            if not factors:
                continue
            pi_expr = sp.Mul(*factors)
            free_vars = {str(s) for s in pi_expr.free_symbols}
            if len(free_vars) >= 2:
                pi_groups.append(sp.simplify(pi_expr))

        return pi_groups

    def get_parameterized_ratios(self) -> List[sp.Expr]:
        """Parameterized Pi forms Π/θ and Π·θ for grammar ratio candidates."""
        pi_groups = self.compute_pi_groups()
        ratios: List[sp.Expr] = []
        for i, pi in enumerate(pi_groups):
            theta = sp.Symbol(f"theta_pi_{i}")
            ratios.append(pi / theta)
            ratios.append(pi * theta)
        return ratios
