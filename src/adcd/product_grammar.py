"""Product grammar for ARC-safe multivariable correction templates."""

from __future__ import annotations

from itertools import combinations, product
from typing import List, Sequence, Tuple

import sympy as sp

from adcd.buckingham_pi import BuckinghamPiEngine
from adcd.sequential_arc import SequentialARCChecker


class ProductGrammar:
    """
    Generate multivariable correction templates as products of ARC-safe 1D forms
    applied to Buckingham-Pi groups.
    """

    ARC_SAFE_UNARIES = [
        "theta_0 * R",
        "theta_0 * R**2",
        "theta_0 * R**theta_1",
        "theta_0 * (exp(-R/theta_1) - 1)",
        "theta_0 * R * exp(-R/theta_1)",
        "theta_0 * log(1 + R/theta_1)",
        "theta_0 * R / (1 + R/theta_1)",
        "theta_0 * tanh(R/theta_1)**2",
        "theta_0 * sin(R/theta_1)",
    ]

    ARC_SAFE_UNARIES_INF = [
        "theta_0 * exp(-R/theta_1)",
        "theta_0 * (theta_1/R)",
        "theta_0 * (theta_1/R)**2",
        "theta_0 * (theta_1/R)**theta_2",
    ]

    def _get_primary_var(self, pi_group: sp.Expr, limit_specs: Sequence[Tuple[str, str]]) -> str:
        pi_vars = {str(s) for s in pi_group.free_symbols}
        for var, _ in limit_specs:
            if var in pi_vars:
                return var
        return limit_specs[0][0]

    def _get_limit_dir(self, var: str, limit_specs: Sequence[Tuple[str, str]]) -> str:
        for spec_var, spec_dir in limit_specs:
            if spec_var == var:
                return spec_dir
        return "0"

    def _instantiate_template(self, template: str, ratio: sp.Expr) -> sp.Expr:
        ratio_str = str(sp.simplify(ratio))
        instantiated = template.replace("R", f"({ratio_str})")
        return sp.sympify(instantiated)

    def generate(
        self,
        pi_groups: List[sp.Expr],
        limit_specs: Sequence[Tuple[str, str]],
        n_candidates: int = 100,
        constants: dict | None = None,
        verify_arc: bool = True,
    ) -> List[str]:
        """Generate product templates Δ = f(Π₁) · g(Π₂)."""
        if len(pi_groups) < 2:
            return []

        per_group: List[Tuple[sp.Expr, List[str]]] = []
        for pi in pi_groups:
            primary_var = self._get_primary_var(pi, limit_specs)
            limit_dir = self._get_limit_dir(primary_var, limit_specs)
            templates = self.ARC_SAFE_UNARIES if limit_dir == "0" else self.ARC_SAFE_UNARIES_INF
            per_group.append((pi, templates))

        products: List[str] = []
        checker = SequentialARCChecker() if verify_arc else None
        limit_vars = [v for v, _ in limit_specs]
        limit_dirs = [d for _, d in limit_specs]

        for (pi1, templates1), (pi2, templates2) in combinations(per_group, 2):
            for t1, t2 in product(templates1[:5], templates2[:5]):
                f_expr = self._instantiate_template(t1, pi1)
                g_expr = self._instantiate_template(t2, pi2)
                product_expr = sp.simplify(f_expr * g_expr)
                expr_str = str(product_expr)
                if expr_str in products:
                    continue
                if verify_arc and checker is not None:
                    result = checker.check(
                        product_expr,
                        limit_vars=limit_vars,
                        limit_dirs=limit_dirs,
                        constants=constants or {},
                    )
                    if not result.passes:
                        continue
                products.append(expr_str)

        return products[:n_candidates]


def pi_groups_for_scenario(scenario) -> List[sp.Expr]:
    """Build Buckingham-Pi groups for a multivariable scenario."""
    engine = BuckinghamPiEngine()
    engine.register_from_scenario(scenario)
    return engine.compute_pi_groups()


def limit_specs_from_scenario(scenario) -> List[Tuple[str, str]]:
    vars_list = [v.strip() for v in scenario.classical_limit_variable.split(",")]
    dirs_list = [d.strip() for d in scenario.classical_limit_direction.split(",")]
    if len(dirs_list) < len(vars_list):
        dirs_list.extend([dirs_list[-1]] * (len(vars_list) - len(dirs_list)))
    return list(zip(vars_list, dirs_list))
