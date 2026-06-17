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
                else:
                    self.register(const_name, [0, 0, 0])

    def compute_pi_groups(self) -> List[sp.Expr]:
        """
        Compute independent dimensionless Pi groups.

        Uses exact rational nullspace (SymPy) for clean ratio forms like m/M.
        """
        if len(self.registry) < 2:
            return []

        names = list(self.registry.keys())
        dim_matrix = np.array([self.registry[n] for n in names]).T
        k, n = dim_matrix.shape

        if k >= n:
            return self._simple_same_dimension_ratios(names)

        from sympy import Matrix

        null_vectors = Matrix(dim_matrix.tolist()).nullspace()
        syms = {name: sp.Symbol(name) for name in names}
        pi_groups: List[sp.Expr] = []

        for vec in null_vectors:
            factors = []
            for name, exp in zip(names, vec):
                if exp == 0:
                    continue
                factors.append(syms[name] ** int(exp))
            if not factors:
                continue
            pi_expr = sp.simplify(sp.Mul(*factors))
            free_vars = {str(s) for s in pi_expr.free_symbols}
            if len(free_vars) >= 2:
                pi_groups.append(pi_expr)

        if not pi_groups:
            pi_groups = self._simple_same_dimension_ratios(names)

        return pi_groups

    def _simple_same_dimension_ratios(self, names: List[str]) -> List[sp.Expr]:
        """Fallback: pairwise ratios among equal-dimension variables."""
        groups: List[sp.Expr] = []
        seen: set[str] = set()
        by_dim: Dict[tuple, List[str]] = {}
        for name in names:
            key = tuple(int(x) for x in self.registry[name])
            by_dim.setdefault(key, []).append(name)

        for dim_vars in by_dim.values():
            if len(dim_vars) < 2:
                continue
            for i in range(len(dim_vars)):
                for j in range(i + 1, len(dim_vars)):
                    a, b = dim_vars[i], dim_vars[j]
                    for ratio in (f"{a}/{b}", f"{b}/{a}"):
                        if ratio not in seen:
                            seen.add(ratio)
                            groups.append(sp.sympify(ratio))
        return groups

    def get_parameterized_ratios(self) -> List[sp.Expr]:
        """Parameterized Pi forms Π/θ and Π·θ for grammar ratio candidates."""
        pi_groups = self.compute_pi_groups()
        ratios: List[sp.Expr] = []
        for i, pi in enumerate(pi_groups):
            theta = sp.Symbol(f"theta_pi_{i}")
            ratios.append(pi / theta)
            ratios.append(pi * theta)
        return ratios
