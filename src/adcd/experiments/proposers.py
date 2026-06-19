"""
Experiment-specific proposers — narrow template banks for honest validation.

These proposers deliberately restrict the search space to physically motivated
families so iterative experiments test recovery order, not template lottery luck.
"""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
import sympy as sp

from adcd.llm_proposer import BaseProposer, ProposalContext


class TemplateListProposer(BaseProposer):
    """Propose only from an explicit template list (variables substituted)."""

    def __init__(self, templates: List[str], seed: int = 42):
        self.templates = templates
        self.seed = seed

    def propose(self, context: ProposalContext) -> List[str]:
        rng = np.random.RandomState(self.seed + context.iteration)
        vars_available = context.variable_names or ["x"]
        v1 = vars_available[0]
        sym_locals = {v: sp.Symbol(v) for v in vars_available}
        sym_locals.update({"theta_0": sp.Symbol("theta_0"), "theta_1": sp.Symbol("theta_1"),
                           "theta_2": sp.Symbol("theta_2"), "pi": sp.pi})

        candidates: List[str] = []
        for tmpl in self.templates:
            cand = tmpl.replace("{v1}", v1)
            try:
                sp.sympify(cand, locals=sym_locals)
                candidates.append(cand)
            except Exception:
                continue

        rng.shuffle(candidates)
        n = min(context.n_candidates, len(candidates))
        if n < context.n_candidates:
            for k in range(context.n_candidates - n):
                candidates.append(f"theta_{k} * {v1}")
        return candidates[: context.n_candidates]


def perturbative_order_proposer(order: int, variable: str = "x", seed: int = 42) -> TemplateListProposer:
    """
    Single perturbative order k: Δ_k = θ₀ · x^k  (+ optional lower orders already subtracted).
    """
    if order == 1:
        templates = [f"theta_0 * {variable}", f"theta_0 * {variable} + theta_1"]
    else:
        templates = [f"theta_0 * {variable}**{order}"]
    return TemplateListProposer(templates=templates, seed=seed)


def mond_correction_proposer(variable: str = "x", seed: int = 42) -> TemplateListProposer:
    """
    Templates for ν(x) − 1 with Newtonian limit ν → 1 as x → ∞.
    Includes Simple-MOND-like square-root forms and low-order rationals.
    """
    v = variable
    templates = [
        f"theta_0 * sqrt(1.0 + theta_1 / {v})",
        f"theta_0 / sqrt({v})",
        f"theta_0 * {v}**(-0.5)",
        f"theta_0 / (1.0 + theta_1 * {v})",
        f"theta_0 * {v} / ({v} + theta_1)",
        f"theta_0 * sqrt(theta_1 / {v})",
        f"theta_0 * {v}**(-theta_1)",
        f"theta_0 * {v}**2 / ({v}**2 + theta_1)",
    ]
    return TemplateListProposer(templates=templates, seed=seed)


def extract_linear_coefficient(expr: str, theta: dict, variable: str = "x") -> Optional[float]:
    """Extract θ₀ from discovered form θ₀·x (first-order perturbative term)."""
    pattern = rf"theta_r?\d+_0\s*\*\s*{re.escape(variable)}\b"
    if re.search(pattern, expr.replace(" ", "")) or f"theta_0 * {variable}" in expr:
        for key, val in theta.items():
            if key.endswith("_0") or key == "theta_0":
                return float(val)
    return None


def extract_power_coefficient(expr: str, theta: dict, variable: str = "x", power: int = 2) -> Optional[float]:
    """Extract θ₀ from discovered form θ₀·x^power."""
    if f"{variable}**{power}" in expr.replace(" ", ""):
        for key, val in theta.items():
            if key.endswith("_0") or key == "theta_0":
                return float(val)
    return None
