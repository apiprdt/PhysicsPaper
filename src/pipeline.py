import sympy as sp
import numpy as np
from typing import List, Tuple, Union, Dict
from src.dimensional_checker import ASTValidator, DimensionalChecker, validate_transcendental_args
from src.arc_scorer import ARCScorer
from src.coarse_evaluator import CoarseEvaluator

class Stage1Pipeline:
    """
    Orchestrates the cascading coarse screening workflow to protect low-resource CPU
    hardware from processing mathematically flawed candidates.
    """
    def __init__(self, validator: ASTValidator, checker: DimensionalChecker, scorer: ARCScorer):
        self.validator = validator
        self.checker = checker
        self.scorer = scorer
        self.locals = {s: sp.Symbol(s) for s in checker.registry}

    def execute(self, 
                candidates: Union[List[str], List[Tuple[str, bool]]], 
                target_dimension_key: str, 
                X: Dict[str, np.ndarray] = None, 
                y_obs: np.ndarray = None, 
                beta: float = 1.0,
                constants: Dict[str, float] = None) -> List[Tuple[str, float, float, float]]:
        """
        Runs candidate strings through the filter cascade. Returns a sorted list 
        of valid (candidate, combined_score, mse, arc_score) tuples in descending 
        order of combined_score.
        """
        screened_candidates = []
        
        # Instantiate the data-driven evaluator if data is supplied
        evaluator = None
        if X is not None and y_obs is not None:
            evaluator = CoarseEvaluator(X, y_obs, constants=constants)

        for item in candidates:
            if isinstance(item, tuple):
                raw_cand, has_params = item
            else:
                raw_cand = item
                has_params = False

            try:
                # Single-Pass Parsing to save CPU overhead cycles
                expr = sp.sympify(raw_cand, locals=self.locals)
            except Exception:
                continue  # Drop malformed string expressions immediately
            
            # Gate 1: Complexity Check (~μs)
            if not self.validator.verify(expr):
                continue
                
            # Gate 2: Dimensional Suitability Check (~ms)
            is_dim_ok = True
            if target_dimension_key is not None and target_dimension_key != "dimensionless":
                if has_params:
                    # Parameterized additive expression: the coefficient can adapt to match target units,
                    # but we must still enforce dimensional homogeneity (no unit clashes like m + v)
                    try:
                        self.checker._get_dim_vector(expr)
                        is_dim_ok = True
                    except TypeError:
                        is_dim_ok = False
                else:
                    is_dim_ok = self.checker.verify(expr, target_dimension_key)
            else:
                is_dim_ok = self.checker.verify(expr, target_dimension_key)
                
            if not is_dim_ok:
                continue
                
            # Gate 2.5: Transcendental Argument Guardrail (always active, even when target_dim=None)
            if not validate_transcendental_args(expr, self.checker):
                continue
                
            # Gate 3: Asymptotic Consistency Verification (~10-100ms)
            try:
                arc_score = float(self.scorer.score(expr, constants=constants))
            except Exception:
                continue

            # Gate 4: Empirical Data Assessment (Coarse Numerical Evaluation)
            mse = 0.0
            nmse = 0.0
            if evaluator is not None:
                mse, nmse = evaluator.evaluate(expr, has_params=has_params)
                if np.isinf(mse):
                    continue  # Filter out formulas with numeric overflows/crashes
            
            # Calculate Combined Bayesian Prior-Likelihood Score
            combined_score = arc_score * float(np.exp(-beta * nmse))
            screened_candidates.append((raw_cand, combined_score, mse, arc_score))
                
        # Return sorted by combined_score descending
        return sorted(screened_candidates, key=lambda x: x[1], reverse=True)
