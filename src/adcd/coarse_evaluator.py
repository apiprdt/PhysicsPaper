import sympy as sp
import numpy as np
from typing import Dict, Union, Tuple

DEFAULT_CONSTANTS = {
    'c': 3.0e8,
    'G': 6.6743e-11,
    'M': 1.989e30,
}

class CoarseEvaluator:
    """
    Evaluates the empirical accuracy (MSE and Normalized MSE) of candidate
    equations on observed physical datasets using high-speed lambdified numpy arrays.
    
    Example:
        >>> evaluator = CoarseEvaluator(X={"x": np.array([1, 2, 3])}, y_obs=np.array([2, 4, 6]))
        >>> mse, nmse = evaluator.evaluate(sp.sympify("2 * x"))
    """
    def __init__(self, X: Dict[str, np.ndarray], y_obs: np.ndarray, constants: Dict[str, float] = None):
        if not X:
            raise ValueError("Dataset X tidak boleh kosong.")
        self.X = X
        self.y_obs = y_obs
        self.constants = constants if constants is not None else DEFAULT_CONSTANTS
        
        # Calculate variance of y_obs with safeguard for trivial datasets
        self.y_var = float(np.var(y_obs))
        if self.y_var < 1e-10:
            self.y_var = 1e-10

        # Determine shape of dataset from first array
        self.data_shape = next(iter(X.values())).shape

    def evaluate(self, expr: sp.Expr, has_params: bool = False) -> Tuple[float, float]:
        """
        Evaluates the candidate SymPy expression on the dataset.
        
        Args:
            expr: The SymPy expression to evaluate.
            has_params: If True, scales the prediction to fit the observation (1D OLS).
            
        Returns:
            Tuple of (MSE, NMSE). Returns (inf, inf) if any numerical overflow/error occurs.
            
        Example:
            >>> mse, nmse = evaluator.evaluate(sp.sympify("theta_0 * x"), has_params=True)
        """
        free_syms = list(expr.free_symbols)
        sym_names = [str(sym) for sym in free_syms]

        # Map each free symbol in the expression to its array or constant value
        args = []
        for name in sym_names:
            if name in self.X:
                args.append(self.X[name])
            elif name in self.constants:
                # Broadcast constant value to match the data shape
                args.append(np.full(self.data_shape, self.constants[name]))
            else:
                # Unknown variable/constant in expression -> hard failure
                return float('inf'), float('inf')

        try:
            # Vectorized lambda compilation
            f = sp.lambdify(free_syms, expr, modules=["numpy"])
            
            # Execute model prediction
            y_pred = f(*args)
            
            # Protect against non-numpy array returns (e.g. constant expression like "5.0")
            if not isinstance(y_pred, np.ndarray):
                y_pred = np.full(self.data_shape, float(y_pred))

            # Clean check for invalid numerical outputs (inf, NaN, complex numbers)
            if np.any(np.isinf(y_pred)) or np.any(np.isnan(y_pred)) or np.iscomplexobj(y_pred):
                return float('inf'), float('inf')

            # Scale prediction to match observed target scale (1D OLS)
            if has_params:
                try:
                    denom = float(np.dot(y_pred, y_pred))
                    if denom > 1e-30:
                        optimal_scale = float(np.dot(y_pred, self.y_obs)) / denom
                        y_pred = optimal_scale * y_pred
                except Exception:
                    pass

            # Calculate MSE and scale-invariant NMSE
            mse = float(np.mean((y_pred - self.y_obs) ** 2))
            nmse = mse / self.y_var
            
            return mse, nmse

        except Exception:
            # Catch division by zero, domain errors, overflow, etc.
            return float('inf'), float('inf')
