"""Internal ADCD fit helper for experiment scripts with custom proposers."""

from __future__ import annotations

from typing import Dict, Optional

import numpy as np

from adcd.api import CustomAnomalyScenario
from adcd.arc_scorer import ARCScorer, build_arc_regimes
from adcd.correction_orchestrator import CorrectionOrchestrator
from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.jax_optimizer import JAXOptimizer
from adcd.llm_proposer import BaseProposer
from adcd.mode_detection import detect_correction_mode
from adcd.pipeline import Stage1Pipeline
from adcd.result import ADCDResult


def fit_with_proposer(
    X: Dict[str, np.ndarray],
    y_obs: np.ndarray,
    y_classical: np.ndarray,
    proposer_obj: BaseProposer,
    limit_variable: Optional[str] = None,
    limit_direction: str = "0",
    classical_expr: str = "0",
    variables_with_units: Optional[Dict[str, str]] = None,
    correction_mode: str = "additive",
    max_iterations: int = 5,
    verbose: bool = True,
    seed: int = 42,
    scenario_name: str = "Experiment Run",
) -> ADCDResult:
    """Run correction discovery with an explicit proposer instance."""
    for k, v in X.items():
        X[k] = np.asarray(v, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    y_classical = np.asarray(y_classical, dtype=float)

    if correction_mode == "auto":
        mode, confidence = detect_correction_mode(y_obs, y_classical)
        if verbose:
            print(f"[ADCD Auto-Mode] Detected {mode} correction with confidence {confidence:.2f}")
    else:
        mode = correction_mode

    if limit_variable is None:
        limit_variable = list(X.keys())[0]

    scenario = CustomAnomalyScenario(
        X=X,
        y_obs=y_obs,
        y_classical=y_classical,
        classical_expr=classical_expr,
        correction_type=mode,
        limit_variable=limit_variable,
        limit_direction=limit_direction,
        variables_with_units=variables_with_units,
        name=scenario_name,
    )

    validator = ASTValidator()
    checker = DimensionalChecker()
    regimes = build_arc_regimes(limit_variable, limit_direction)
    scorer = ARCScorer(regimes=regimes)
    pipeline = Stage1Pipeline(validator, checker, scorer)
    optimizer = JAXOptimizer()

    orchestrator = CorrectionOrchestrator(
        proposer=proposer_obj,
        pipeline=pipeline,
        optimizer=optimizer,
        max_iterations=max_iterations,
        verbose=verbose,
    )
    search_result = orchestrator.search_correction(scenario, seed=seed)
    return ADCDResult(
        search_result=search_result,
        scenario=scenario,
        X=X,
        y_obs=y_obs,
        y_classical=y_classical,
    )
