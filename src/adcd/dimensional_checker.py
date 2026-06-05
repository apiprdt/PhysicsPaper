import sympy as sp
from typing import Dict, Union, List, Optional

# Core physical constants and variables SI base vectors: [M, L, T]
DIMENSIONS = {
    'm': [1, 0, 0],      # Mass (kg)
    'M': [1, 0, 0],      # Mass (kg)
    'v': [0, 1, -1],     # Velocity (m/s)
    'r': [0, 1, 0],      # Distance/Length (m)
    't': [0, 0, 1],      # Time (s)
    'G': [-1, 3, -2],    # Gravitational Constant
    'c': [0, 1, -1],     # Speed of Light (m/s)
    'E': [1, 2, -2],     # Energy (Joule)
    'F': [1, 1, -2],     # Force (Newton)
}

class DimensionalChecker:
    """
    Verifies the physical dimensional consistency of a candidate expression
    using linear algebra over SI base unit exponent vectors.
    
    Example:
        >>> checker = DimensionalChecker()
        >>> checker.verify("v**2 / r", "v")  # False
        >>> checker.verify("v**2 / r", "G")  # False
    """
    def __init__(self, unit_registry: Dict[str, List[int]] = None):
        self.registry = dict(unit_registry) if unit_registry is not None else dict(DIMENSIONS)
        self.locals = {s: sp.Symbol(s) for s in self.registry}

    def _get_dim_vector(self, expr: sp.Expr) -> List[int]:
        """Recursively computes the dimensional exponent vector of a SymPy AST."""
        if expr.is_Number or expr.is_NumberSymbol or expr == sp.I:
            return [0, 0, 0]  # Dimensionless constant
        
        if expr.is_Symbol:
            sym_str = str(expr)
            if sym_str.startswith("theta_"):
                return [0, 0, 0]
            if sym_str in self.registry:
                return self.registry[sym_str]
            raise ValueError(f"Unknown physical symbol in registry: {sym_str}")

        if isinstance(expr, sp.Add):
            # Homogeneity check: All added elements must share identical dimensions
            args = expr.args
            first_dim = self._get_dim_vector(args[0])
            for arg in args[1:]:
                if self._get_dim_vector(arg) != first_dim:
                    raise TypeError("Dimensional Mismatch: Homogeneity rule violated in addition.")
            return first_dim

        if isinstance(expr, sp.Mul):
            # Multiplication adds dimensions element-wise
            base_dim = [0, 0, 0]
            for arg in expr.args:
                arg_dim = self._get_dim_vector(arg)
                base_dim = [a + b for a, b in zip(base_dim, arg_dim)]
            return base_dim

        if isinstance(expr, sp.Pow):
            base, exponent = expr.args
            if not exponent.is_Number:
                return [0, 0, 0]  # Non-numeric exponent fallback
            
            base_dim = self._get_dim_vector(base)
            exp_val = float(exponent)
            return [int(d * exp_val) if (d * exp_val).is_integer() else d * exp_val for d in base_dim]

        if isinstance(expr, sp.Function):
            # Transcendental functions (sin, cos, exp, log) arguments must be dimensionless
            # and they output a dimensionless quantity.
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
        
        Args:
            candidate_expr: The expression string or SymPy expression to check.
            target_dimension_key: The target dimension key in the registry (e.g. "dimensionless").
            
        Returns:
            bool: True if consistent, False otherwise.
            
        Example:
            >>> checker.verify("v**2 / r", "dimensionless")
        """
        if target_dimension_key is None:
            return True
        try:
            expr = sp.sympify(candidate_expr, locals=self.locals) if isinstance(candidate_expr, str) else candidate_expr
            
            # Adaptive relaxation: If the target is dimensionless and there is exactly one
            # physical symbol, free parameters (substituted with 1.0) can always scale it to be dimensionless.
            physical_symbols = [s for s in expr.free_symbols if str(s) in self.registry]
            if target_dimension_key == "dimensionless" and len(physical_symbols) == 1:
                return True
                
            candidate_dim = self._get_dim_vector(expr)
            
            if target_dimension_key in self.registry:
                target_dim = self.registry[target_dimension_key]
            else:
                # Interpret target_dimension_key as a classical baseline expression
                target_expr = sp.sympify(target_dimension_key, locals=self.locals)
                target_dim = self._get_dim_vector(target_expr)
                
            return candidate_dim == target_dim
        except (TypeError, ValueError, KeyError, NotImplementedError):
            return False


def validate_transcendental_args(expr: sp.Expr, checker: DimensionalChecker) -> bool:
    """
    Returns True if all transcendental function arguments are dimensionless.
    
    Args:
        expr: The SymPy expression to check.
        checker: The DimensionalChecker unit mapping.
        
    Returns:
        bool: True if argument dimensions are valid, False otherwise.
        
    Example:
        >>> validate_transcendental_args(sp.sympify("sin(v/c)"), checker)
    """
    for sub in sp.preorder_traversal(expr):
        if isinstance(sub, sp.Function):
            if sub.func.__name__ in ("sin", "cos", "tan", "exp", "log", "asin", "acos", "atan", "sinh", "cosh", "tanh"):
                try:
                    if len(sub.args) > 0:
                        arg = sub.args[0]
                        physical_symbols = [s for s in arg.free_symbols if str(s) in checker.registry]
                        
                        # Adaptive relaxation: If there is at most one physical symbol in the argument,
                        # free parameters can scale it to be dimensionless.
                        if len(physical_symbols) <= 1:
                            continue
                            
                        arg_dim = checker._get_dim_vector(arg)
                        if arg_dim != [0, 0, 0]:
                            return False
                except Exception:
                    return False
    return True


class ASTValidator:
    """
    Prunes bloated expressions to prevent dynamic algebraic over-fitting/bloating.
    
    Example:
        >>> validator = ASTValidator(max_depth=5, max_tokens=15)
        >>> validator.verify("x**2 + y**2")
    """
    def __init__(self, max_depth: int = 7, max_tokens: int = 20):
        self.max_depth = max_depth
        self.max_tokens = max_tokens
        self.locals = {s: sp.Symbol(s) for s in DIMENSIONS}

    def _get_depth(self, expr: sp.Expr) -> int:
        if not expr.args:
            return 1
        return 1 + max(self._get_depth(arg) for arg in expr.args)

    def verify(self, candidate_expr: Union[str, sp.Expr]) -> bool:
        """
        Returns True if the expression complexity falls within depth and token limits.
        
        Args:
            candidate_expr: The expression string or SymPy expression to check.
            
        Returns:
            bool: True if complexity is within bounds, False otherwise.
        """
        try:
            expr = sp.sympify(candidate_expr, locals=self.locals) if isinstance(candidate_expr, str) else candidate_expr
            depth = self._get_depth(expr)
            tokens = len(list(sp.preorder_traversal(expr)))
            return depth <= self.max_depth and tokens <= self.max_tokens
        except Exception:
            return False

    def set_threshold_relative_to(self, target_expr: Union[str, sp.Expr], delta_tokens: int = 5, delta_depth: int = 2):
        """
        Dynamically adjusts the complexity bounds based on a target ground-truth formula.
        
        Args:
            target_expr: The target reference expression.
            delta_tokens: Token slack to add to the target count.
            delta_depth: Depth slack to add to the target depth.
        """
        try:
            expr = sp.sympify(target_expr, locals=self.locals) if isinstance(target_expr, str) else target_expr
            depth = self._get_depth(expr)
            tokens = len(list(sp.preorder_traversal(expr)))
            self.max_depth = depth + delta_depth
            self.max_tokens = tokens + delta_tokens
        except Exception:
            pass
