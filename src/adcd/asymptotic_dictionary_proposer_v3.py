from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Dict, List, Optional, Callable

import numpy as np
import sympy as sp
from adcd.context import BaseProposer, ProposalContext

_u = sp.Symbol("u", positive=True)

# Disinkronkan 100% dengan Julia PrimitiveRegistry
_RAW_FORMS: Dict[str, sp.Expr] = {
    "D_lor": 1 / sp.sqrt(1 - _u),
    "D_rat": _u / (1 + _u**2),               # Disesuaikan: rasional halus
    "D_exp": 1 - sp.exp(-_u),                 # Disesuaikan: 1 - exp(-u)
    "D_log": sp.log(1 + sp.Abs(_u)),
    "D_sqrt_inv": sp.sqrt(sp.Abs(_u)) / (1 + sp.sqrt(sp.Abs(_u))),
    "D_pow": sp.sqrt(sp.Abs(_u)) * (1 - sp.exp(-sp.Abs(_u))),
    "D_osc": 1 - sp.cos(_u),
    "D_sat": sp.tanh(_u),
    "D_nested_mond": sp.exp(-sp.sqrt(sp.Abs(_u))) * (1 - sp.exp(-sp.sqrt(sp.Abs(_u)))),
    "D_tanh_sq": sp.tanh(_u**2),
}

REGULARIZED_FORMS: Dict[str, sp.Expr] = {
    name: sp.simplify(expr - expr.subs(_u, 0)) for name, expr in _RAW_FORMS.items()
}

def _verify_regularization() -> None:
    for name, expr in REGULARIZED_FORMS.items():
        limit_at_zero = sp.limit(expr, _u, 0)
        assert limit_at_zero == 0, f"REGULARIZATION FAILURE: primitive '{name}' does not vanish at u=0"

_verify_regularization()

@dataclass(frozen=True)
class Primitive:
    name: str
    token_cost: int
    numpy_form: Callable[[np.ndarray], np.ndarray]
    string_template: str
    domain_note: str

PRIMITIVE_REGISTRY: Dict[str, Primitive] = {
    "D_lor": Primitive(
        name="D_lor",
        token_cost=7,
        numpy_form=lambda u: u / (np.sqrt(np.clip(1.0 - u, 1e-9, None)) * (1.0 + np.sqrt(np.clip(1.0 - u, 1e-9, None)))),
        string_template="({u} / (sqrt(1.0 - {u}) * (1.0 + sqrt(1.0 - {u}))))",
        domain_note="u in [0, 1)",
    ),
    "D_rat": Primitive(
        name="D_rat",
        token_cost=4,
        numpy_form=lambda u: u / (1.0 + u**2),
        string_template="(({u}) / (1.0 + ({u})**2))",
        domain_note="Smooth rational",
    ),
    "D_exp": Primitive(
        name="D_exp",
        token_cost=4,
        numpy_form=lambda u: 1.0 - np.exp(-np.abs(u)),
        string_template="(1.0 - exp(-Abs({u})))",
        domain_note="Exponential screening (positive)",
    ),
    "D_log": Primitive(
        name="D_log",
        token_cost=5,
        numpy_form=lambda u: np.log1p(np.abs(u)),
        string_template="log(1.0 + Abs({u}))",
        domain_note="Logarithmic",
    ),
    "D_sqrt_inv": Primitive(
        name="D_sqrt_inv",
        token_cost=5,
        numpy_form=lambda u: np.sqrt(np.abs(u)) / (1.0 + np.sqrt(np.abs(u))),
        string_template="(sqrt(Abs({u})) / (1.0 + sqrt(Abs({u}))))",
        domain_note="MOND interpolation",
    ),
    "D_pow": Primitive(
        name="D_pow",
        token_cost=6,
        numpy_form=lambda u: np.sqrt(np.abs(u)) * (1.0 - np.exp(-np.abs(u))),
        string_template="(sqrt(Abs({u})) * (1.0 - exp(-Abs({u}))))",
        domain_note="Power-law anomalous",
    ),
    "D_osc": Primitive(
        name="D_osc",
        token_cost=5,
        numpy_form=lambda u: 1.0 - np.cos(u),
        string_template="(1.0 - cos({u}))",
        domain_note="Oscillatory",
    ),
    "D_sat": Primitive(
        name="D_sat",
        token_cost=5,
        numpy_form=lambda u: np.tanh(u),
        string_template="tanh({u})",
        domain_note="Saturation",
    ),
    "D_nested_mond": Primitive(
        name="D_nested_mond",
        token_cost=7,
        numpy_form=lambda u: np.exp(-np.sqrt(np.abs(u))) * (1.0 - np.exp(-np.sqrt(np.abs(u)))),
        string_template="(exp(-sqrt(Abs({u}))) * (1.0 - exp(-sqrt(Abs({u})))))",
        domain_note="Bell-shaped anomaly",
    ),
    "D_rar": Primitive(
        name="D_rar",
        token_cost=7,
        numpy_form=lambda u: np.exp(-np.sqrt(np.abs(u) + 1e-15)) / np.maximum(1.0 - np.exp(-np.sqrt(np.abs(u) + 1e-15)), 1e-12),
        string_template="(exp(-sqrt(Abs({u}) + 1e-15)) / Max(1.0 - exp(-sqrt(Abs({u}) + 1e-15)), 1e-12))",
        domain_note="Exact McGaugh RAR form",
    ),
}

@dataclass
class GrammarBudget:
    max_depth: int = 3
    max_tokens: int = 35
    max_primitives_used: int = 2
    max_ratio_candidates: int = 12

def _token_count(expr_str: str) -> int:
    for ch in "+-*/()":
        expr_str = expr_str.replace(ch, " ")
    return len([t for t in expr_str.split() if t])

def enumerate_candidates(
    ratio_symbol: str,
    budget: GrammarBudget,
    active_primitives: Optional[Dict[str, Primitive]] = None,
) -> List[str]:
    prims = active_primitives or PRIMITIVE_REGISTRY
    prim_names = list(prims.keys())
    candidates: List[str] = []

    def _assign_theta(s: str) -> str:
        t = itertools.count(0)
        while "_NEXT_THETA_" in s:
            s = s.replace("_NEXT_THETA_", f"theta_{next(t)}", 1)
        return s

    def prim_expr(p: str, u_sym: str) -> str:
        return prims[p].string_template.format(u=u_sym)

    # Pola 1: Singleton dengan parameter skala inner u -> theta_1 * D(theta_0 * u)
    for p in prim_names:
        u_scaled = f"(_NEXT_THETA_ * ({ratio_symbol}))"
        cand_raw = f"_NEXT_THETA_ * {prim_expr(p, u_scaled)}"
        cand = _assign_theta(cand_raw)
        if _token_count(cand) <= budget.max_tokens:
            candidates.append(cand)

    # Pola 2: Additive -> theta_1 * D_a(theta_0 * u) + theta_2 * D_b(theta_0 * u)
    if budget.max_primitives_used >= 2:
        for pa, pb in itertools.combinations(prim_names, 2):
            u_scaled = f"(_NEXT_THETA_ * ({ratio_symbol}))"
            cand_raw = f"_NEXT_THETA_ * {prim_expr(pa, u_scaled)} + _NEXT_THETA_ * {prim_expr(pb, u_scaled)}"
            cand = _assign_theta(cand_raw)
            if _token_count(cand) <= budget.max_tokens:
                candidates.append(cand)

    # Pola 3: Multiplicative -> theta_1 * D_a(theta_0 * u) * (1 + theta_2 * D_b(theta_0 * u))
    if budget.max_primitives_used >= 2 and budget.max_depth >= 3:
        for pa, pb in itertools.permutations(prim_names, 2):
            u_scaled = f"(_NEXT_THETA_ * ({ratio_symbol}))"
            cand_raw = f"_NEXT_THETA_ * {prim_expr(pa, u_scaled)} * (1.0 + _NEXT_THETA_ * {prim_expr(pb, u_scaled)})"
            cand = _assign_theta(cand_raw)
            if _token_count(cand) <= budget.max_tokens:
                candidates.append(cand)

    # Pola 4: Nested (Sinkronisasi dengan Julia) -> theta_1 * D_outer(D_inner(theta_0 * u))
    if budget.max_primitives_used >= 2 and budget.max_depth >= 3:
        for pa, pb in itertools.permutations(prim_names, 2):
            u_scaled = f"(_NEXT_THETA_ * ({ratio_symbol}))"
            inner = prim_expr(pb, u_scaled)
            cand_raw = f"_NEXT_THETA_ * {prim_expr(pa, inner)}"
            cand = _assign_theta(cand_raw)
            if _token_count(cand) <= budget.max_tokens:
                candidates.append(cand)

    return candidates

class AsymptoticDictionaryProposerV3(BaseProposer):
    def __init__(
        self,
        ratio_symbol: str = "u",
        budget: Optional[GrammarBudget] = None,
        exclude_primitives: Optional[List[str]] = None,
    ):
        self.ratio_symbol = ratio_symbol
        self.budget = budget or GrammarBudget()
        self.exclude_primitives = set(exclude_primitives or [])
        self._active_primitives = {k: v for k, v in PRIMITIVE_REGISTRY.items() if k not in self.exclude_primitives}

    def propose(self, context: ProposalContext) -> List[str]:
        candidates = enumerate_candidates(
            ratio_symbol=self.ratio_symbol,
            budget=self.budget,
            active_primitives=self._active_primitives,
        )
        limit = context.n_candidates if context.n_candidates and context.n_candidates > 0 else len(candidates)
        return candidates[:limit]

    def search_space_size(self) -> int:
        return len(enumerate_candidates(self.ratio_symbol, self.budget, self._active_primitives))
