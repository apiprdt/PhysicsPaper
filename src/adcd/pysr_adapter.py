"""
PySR Adapter with Physics-Gate Post-Filtering.
Interfaces with PySRRegressor without modifying PySR source code.
Applies PhysicsGateCascade post-filtering on PySR candidate equations.
"""

import logging
from typing import Any, Dict, List, Optional
import sympy as sp

from adcd.gate_cascade import PhysicsGateCascade, GateResult

logger = logging.getLogger("PySRWithGate")


class PySRWithGate:
    """
    Adapter combining PySRRegressor candidate discovery with PhysicsGateCascade post-filtering.
    PySR is treated strictly as a black-box candidate generator.
    """

    def __init__(
        self,
        gate_cascade: Optional[PhysicsGateCascade] = None,
        niterations: int = 25,
        binary_operators: Optional[List[str]] = None,
        unary_operators: Optional[List[str]] = None,
        population_size: int = 20,
    ):
        self.gate_cascade = gate_cascade or PhysicsGateCascade()
        self.niterations = niterations
        self.binary_operators = binary_operators or ["+", "*", "/", "-"]
        self.unary_operators = unary_operators or ["exp", "sqrt", "abs", "log"]
        self.population_size = population_size
        self.pysr_model = None

    def _init_pysr(self):
        try:
            from pysr import PySRRegressor
            self.pysr_model = PySRRegressor(
                niterations=self.niterations,
                binary_operators=self.binary_operators,
                unary_operators=self.unary_operators,
                populations=self.population_size,
                verbosity=0,
                progress=False,
            )
        except ImportError:
            logger.warning("PySR is not installed in the environment. PySR candidate discovery disabled.")
            self.pysr_model = None

    def fit_and_filter(
        self,
        X: Any,
        y: Any,
        variable_names: Optional[List[str]] = None,
        limit_vars: Optional[List[str]] = None,
        limit_points: Optional[List[Any]] = None,
        variable_units: Optional[Dict[str, str]] = None,
        target_units: Optional[str] = None,
        constants: Optional[Dict[str, float]] = None
    ) -> List[Dict[str, Any]]:
        """
        Fits PySR model to (X, y) data, retrieves Pareto front candidate expressions,
        and applies PhysicsGateCascade post-filtering.
        """
        if self.pysr_model is None:
            self._init_pysr()

        if self.pysr_model is None:
            raise RuntimeError("PySR library is unavailable. Cannot execute PySR fitting.")

        self.pysr_model.fit(X, y, variable_names=variable_names)

        # Retrieve candidates from PySR equations dataframe
        equations_df = self.pysr_model.equations_
        results = []

        for _, row in equations_df.iterrows():
            expr_str = str(row.get('sympy_format', row.get('equation', '')))
            loss = float(row.get('loss', 1e6))
            complexity = int(row.get('complexity', 0))

            gate_res: GateResult = self.gate_cascade.check(
                candidate_expr=expr_str,
                limit_vars=limit_vars,
                limit_points=limit_points,
                variable_units=variable_units,
                target_units=target_units,
                constants=constants
            )

            results.append({
                'equation': expr_str,
                'loss': loss,
                'complexity': complexity,
                'gate_passed': gate_res.passed,
                'gate_reason': gate_res.rejection_reason,
                'rejected_stage': gate_res.rejected_stage,
                'gate_result': gate_res
            })

        return results
