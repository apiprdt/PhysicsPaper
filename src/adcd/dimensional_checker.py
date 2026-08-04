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
    def _is_symbol_or_power_of(factor, sym) -> bool:
        if factor == sym:
            return True
        if isinstance(factor, sp.Pow) and factor.args[0] == sym:
            return True
        return False

    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Mul):
            factors = node.args
            has_symbol = any(_is_symbol_or_power_of(f, symbol) for f in factors)
            if not has_symbol:
                continue
            local_thetas = {s for s in node.free_symbols if str(s).startswith("theta_")}
            has_theta = any(
                _is_symbol_or_power_of(f, t) for f in factors for t in local_thetas
            )
            if has_theta:
                return True
    return False


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
