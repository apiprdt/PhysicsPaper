"""
dimensional_checker.py (AUDIT-HARDENED)
========================================
FIX LOG (read before touching this file again):

BUG FOUND: `verify()` and `validate_transcendental_args()` both contained a
blanket "adaptive relaxation": if a dimensionless target (or a transcendental
function argument) contains AT MOST ONE physical symbol, the check returned
True unconditionally, reasoning "a free theta parameter could always rescale
it to be dimensionless."

WHY THIS WAS A FATAL RIGOR BUG: almost every realistic correction candidate
in this project has exactly ONE physical variable per ratio (v/c, r/r_0,
x/x_0, f/T, ...). The blanket relaxation therefore did not "relax" dimensional
checking for an edge case -- it disabled it for the overwhelming majority of
real candidates, including physically nonsensical ones like `exp(-r)` (a bare
length inside a transcendental, with NO parameter anywhere near it to carry
the compensating dimension). The comment justifying the relaxation ("a free
parameter can always rescale it") was never actually verified: nothing
checked that a theta symbol was present at all, let alone structurally
positioned to carry a compensating dimension.

FIX: replace the blanket relaxation with a STRUCTURAL check. A single bare
physical symbol is only given the benefit of the doubt if it is combined
with a free theta parameter via multiplication or division in that exact
sub-expression (i.e. the candidate itself proposes something that *could*
be dimensionless once theta is fitted). A bare, unscaled physical symbol
(e.g. `exp(-r)`, `sin(v)`) is REJECTED outright -- there is no way to make
that dimensionally consistent with a free scalar.

Every relaxed pass is now also recorded (`checker.last_relaxed`) so gate
telemetry can report exactly how many candidates were waved through on
"pending Stage-2 verification" rather than fully verified. This makes the
remaining, unavoidable limitation of this architecture (theta is treated as
a dimensionless bookkeeping number, never assigned a physical unit of its
own) visible and auditable instead of silently laundered.
"""

import sympy as sp
from typing import Dict, Union, List, Optional

# Core physical constants and variables SI base vectors: [M, L, T]
DIMENSIONS = {
    'm': [1, 0, 0],
    'M': [1, 0, 0],
    'v': [0, 1, -1],
    'r': [0, 1, 0],
    't': [0, 0, 1],
    'G': [-1, 3, -2],
    'c': [0, 1, -1],
    'E': [1, 2, -2],
    'F': [1, 1, -2],
    'rho': [1, -3, 0],
    'n': [0, -3, 0],
    'T': [0, 0, 0],
    'V': [0, 3, 0],
    'k_B': [1, 2, -2],
    'b': [1, 0, -1],
    'A': [0, 2, 0],
    'sigma': [1, 0, -3],
    'a': [0, 1, -2],   # NEW: acceleration [L T^-2] -- needed for real-data
                       # scenarios (e.g. SPARC/RAR: gbar, gobs, a0 all have
                       # this dimension). Added during the audit; was
                       # absent from the original registry.
}


def _symbol_is_theta_scaled(expr: sp.Expr, symbol: sp.Symbol) -> bool:
    """
    Structural check: is `symbol` combined with at least one free theta_N
    parameter via Mul/Pow in `expr`? This is the ONLY configuration in which
    a lone physical variable can plausibly become dimensionless once theta
    is fitted (theta absorbs the compensating dimension). A bare symbol with
    only numeric coefficients can NEVER become dimensionless -- numbers carry
    no compensating unit.
    """
    thetas_in_expr = {s for s in expr.free_symbols if str(s).startswith("theta_")}
    if not thetas_in_expr:
        return False

    # Expression must contain BOTH the physical symbol and at least one theta
    # in a multiplicative/power relationship somewhere in its tree (covers
    # v/theta_1, theta_0*v, (v/theta_1)**theta_2, etc.)
    found_symbol = False
    found_theta_nearby = False
    for node in sp.preorder_traversal(expr):
        if isinstance(node, (sp.Mul, sp.Pow)):
            node_syms = node.free_symbols
            if symbol in node_syms:
                found_symbol = True
                if node_syms & thetas_in_expr:
                    found_theta_nearby = True
    return found_symbol and found_theta_nearby


class DimensionalChecker:
    """
    Verifies physical dimensional consistency of a candidate expression using
    linear algebra over SI base unit exponent vectors [M, L, T].
    """

    def __init__(self, unit_registry: Dict[str, List[int]] = None):
        self.registry = dict(unit_registry) if unit_registry is not None else dict(DIMENSIONS)
        self.locals = {s: sp.Symbol(s) for s in self.registry}
        # Telemetry: how many verify() calls were passed only via the
        # structural theta-scaling relaxation, never fully dimension-verified.
        self.last_relaxed: bool = False

    def _get_dim_vector(self, expr: sp.Expr) -> List[int]:
        if expr.is_Number or expr.is_NumberSymbol or expr == sp.I:
            return [0, 0, 0]

        if expr.is_Symbol:
            sym_str = str(expr)
            if sym_str.startswith("theta_"):
                return [0, 0, 0]
            if sym_str in self.registry:
                return self.registry[sym_str]
            raise ValueError(f"Unknown physical symbol in registry: {sym_str}")

        if isinstance(expr, sp.Add):
            args = expr.args
            first_dim = self._get_dim_vector(args[0])
            for arg in args[1:]:
                if self._get_dim_vector(arg) != first_dim:
                    raise TypeError("Dimensional Mismatch: Homogeneity rule violated in addition.")
            return first_dim

        if isinstance(expr, sp.Mul):
            base_dim = [0, 0, 0]
            for arg in expr.args:
                arg_dim = self._get_dim_vector(arg)
                base_dim = [a + b for a, b in zip(base_dim, arg_dim)]
            return base_dim

        if isinstance(expr, sp.Pow):
            base, exponent = expr.args
            if not exponent.is_Number:
                return [0, 0, 0]
            base_dim = self._get_dim_vector(base)
            exp_val = float(exponent)
            return [int(d * exp_val) if (d * exp_val).is_integer() else d * exp_val for d in base_dim]

        if isinstance(expr, sp.Function):
            arg = expr.args[0]
            arg_dim = self._get_dim_vector(arg)
            if arg_dim != [0, 0, 0]:
                raise TypeError(
                    f"Dimensional Mismatch: Argument of transcendental function {expr.func.__name__} "
                    f"must be dimensionless, but got {arg_dim}."
                )
            return [0, 0, 0]

        raise NotImplementedError(f"Operator {type(expr)} not yet supported in dimensional analysis.")

    def verify(self, candidate_expr: Union[str, sp.Expr], target_dimension_key: Optional[str]) -> bool:
        """
        Returns True if the expression's units match the physical target dimension.

        FIXED: the old "len(physical_symbols) == 1 -> always True" shortcut is
        gone. A lone physical symbol only passes if it is structurally scaled
        by a free theta parameter (see `_symbol_is_theta_scaled`); otherwise
        its raw registry dimension is used, exactly like any other symbol.
        """
        self.last_relaxed = False
        if target_dimension_key is None:
            return True
        try:
            expr = sp.sympify(candidate_expr, locals=self.locals) if isinstance(candidate_expr, str) else candidate_expr

            physical_symbols = [s for s in expr.free_symbols if str(s) in self.registry]
            if target_dimension_key == "dimensionless" and len(physical_symbols) == 1:
                sym = physical_symbols[0]
                if _symbol_is_theta_scaled(expr, sym):
                    self.last_relaxed = True
                    return True
                # else: fall through to full dimensional evaluation below --
                # a bare unscaled physical symbol must genuinely be dimensionless
                # (it never will be, but we let the real computation say so
                # rather than assuming).

            candidate_dim = self._get_dim_vector(expr)

            if target_dimension_key == "dimensionless":
                target_dim = [0, 0, 0]
            elif target_dimension_key in self.registry:
                target_dim = self.registry[target_dimension_key]
            else:
                target_expr = sp.sympify(target_dimension_key, locals=self.locals)
                target_dim = self._get_dim_vector(target_expr)

            return candidate_dim == target_dim
        except (TypeError, ValueError, KeyError, NotImplementedError):
            return False

    def enumerate_dimensionless_ratios(self, symbols: List[str], max_degree: int = 2) -> List[sp.Expr]:
        """Buckingham-Pi style nullspace enumeration of dimensionless monomials.
        (unchanged from the original -- this part of the file was already correct
        and, per the audit, is the piece that should be doing the heavy lifting
        for generic 'u' ratio construction instead of a human hand-picking it.)
        """
        import math
        import itertools

        valid_symbols = [s for s in symbols if s in self.registry]
        if not valid_symbols:
            return []

        col_vectors = [self.registry[s] for s in valid_symbols]
        A = sp.Matrix(col_vectors).T
        null_space = A.nullspace()
        if not null_space:
            return []

        int_basis = []
        for v in null_space:
            denoms = [sp.Rational(x).q for x in v]
            lcm_val = sp.lcm(denoms)
            v_int = [int(x * lcm_val) for x in v]
            int_basis.append(v_int)

        k = len(int_basis)
        n_syms = len(valid_symbols)
        coef_range = range(-max_degree, max_degree + 1)

        unique_exponent_sets = set()
        for coefs in itertools.product(coef_range, repeat=k):
            if all(c == 0 for c in coefs):
                continue
            e = [0] * n_syms
            for j in range(n_syms):
                e[j] = sum(coefs[i] * int_basis[i][j] for i in range(k))
            if not any(x != 0 for x in e) or any(abs(x) > max_degree for x in e):
                continue
            g = math.gcd(*e)
            if g != 0:
                e = [x // g for x in e]
            for x in e:
                if x != 0:
                    if x < 0:
                        e = [-val for val in e]
                    break
            unique_exponent_sets.add(tuple(e))

        ratios = []
        for e in sorted(unique_exponent_sets):
            expr = sp.Integer(1)
            for s, exp in zip(valid_symbols, e):
                if exp != 0:
                    expr *= sp.Symbol(s) ** exp
            ratios.append(expr)
        return ratios


def validate_transcendental_args(expr: sp.Expr, checker: DimensionalChecker) -> bool:
    """
    Returns True if all transcendental function arguments are dimensionless.

    FIXED: same structural theta-scaling requirement as `DimensionalChecker.verify`.
    A bare `exp(-r)` / `sin(v)` with no free parameter anywhere in the argument
    is now correctly rejected instead of being waved through.
    """
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, sp.Function):
            if sub.func.__name__ in ("sin", "cos", "tan", "exp", "log", "asin", "acos", "atan", "sinh", "cosh", "tanh"):
                try:
                    if len(sub.args) > 0:
                        arg = sub.args[0]
                        physical_symbols = [s for s in arg.free_symbols if str(s) in checker.registry]

                        if len(physical_symbols) <= 1:
                            if len(physical_symbols) == 1 and _symbol_is_theta_scaled(arg, physical_symbols[0]):
                                checker.last_relaxed = True
                                continue
                            if len(physical_symbols) == 0:
                                continue
                            # exactly one physical symbol, NOT theta-scaled -> genuinely check it
                            arg_dim = checker._get_dim_vector(arg)
                            if arg_dim != [0, 0, 0]:
                                return False
                            continue

                        arg_dim = checker._get_dim_vector(arg)
                        if arg_dim != [0, 0, 0]:
                            return False
                except Exception:
                    return False
    return True


class ASTValidator:
    """Prunes bloated expressions to prevent dynamic algebraic over-fitting/bloating.
    (unchanged from original -- no bug found here; the `set_threshold_relative_to()`
    removal note in the original was itself correct engineering discipline.)
    """

    def __init__(self, max_depth: int = 7, max_tokens: int = 25):
        self.max_depth = max_depth
        self.max_tokens = max_tokens
        self.locals = {s: sp.Symbol(s) for s in DIMENSIONS}

    def _get_depth(self, expr: sp.Expr) -> int:
        if not expr.args:
            return 1
        return 1 + max(self._get_depth(arg) for arg in expr.args)

    def verify(self, candidate_expr: Union[str, sp.Expr]) -> bool:
        try:
            expr = sp.sympify(candidate_expr, locals=self.locals) if isinstance(candidate_expr, str) else candidate_expr
            depth = self._get_depth(expr)
            tokens = len(list(sp.preorder_traversal(expr)))
            return depth <= self.max_depth and tokens <= self.max_tokens
        except Exception:
            return False
