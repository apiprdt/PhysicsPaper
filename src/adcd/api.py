import os
import warnings
import logging
import numpy as np
from typing import Dict, Tuple, Optional

from adcd.anomaly_scenarios import AnomalyScenario
from adcd.llm_proposer import (
    CorrectionMockProposer,
    CorrectionGeminiProposer,
    HybridCorrectionProposer,
)
from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.arc_scorer import ARCScorer, build_arc_regimes
from adcd.pipeline import Stage1Pipeline
from adcd.jax_optimizer import JAXOptimizer
from adcd.correction_orchestrator import CorrectionOrchestrator
from adcd.mode_detection import detect_correction_mode
from adcd.result import ADCDResult


class CustomAnomalyScenario:
    """
    Duck-typed wrapper that exposes custom numpy arrays (X, y_obs, y_classical)
    as an AnomalyScenario object to the ADCD orchestrator.
    """
    def __init__(
        self,
        X: Dict[str, np.ndarray],
        y_obs: np.ndarray,
        y_classical: np.ndarray,
        classical_expr: str,
        correction_type: str,
        limit_variable: str,
        limit_direction: str,
        variables_with_units: Optional[Dict[str, str]] = None,
        name: str = "Custom Dataset Run",
    ):
        self.name = name
        self.tier = "custom"
        self.domain = "custom"
        self.classical_expr = classical_expr
        self.classical_variables = list(X.keys())
        self.classical_constants = {}
        self.correction_type = correction_type
        self.correction_expr = "Unknown"
        self.correction_constants = {}
        self.anomaly_regime = "custom"
        self.variables_with_units = variables_with_units or {k: "dimensionless" for k in X.keys()}
        self.classical_limit_variable = limit_variable
        self.classical_limit_direction = limit_direction
        self.correction_class = "unknown"
        self._X = X
        self._y_obs = y_obs
        self._y_classical = y_classical

    def generate_data(
        self,
        n_points: int = 200,
        noise_level: float = 0.0,
        seed: int = 42,
    ) -> Tuple[Dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
        if self.correction_type == "multiplicative":
            safe_classical = np.where(self._y_classical == 0, 1e-15, self._y_classical)
            residual = self._y_obs / safe_classical - 1.0
        else:
            residual = self._y_obs - self._y_classical
        return self._X, self._y_obs, self._y_classical, residual


def fit(
    X: Dict[str, np.ndarray],
    y_obs: np.ndarray,
    y_classical: np.ndarray,
    limit_variable: Optional[str] = None,
    limit_direction: str = "0",
    classical_expr: str = "0",
    variables_with_units: Optional[Dict[str, str]] = None,
    correction_mode: str = "auto",
    max_iterations: int = 5,
    proposer: str = "mock",
    api_key: Optional[str] = None,
    verbose: bool = True,
    seed: int = 42,
    scenario_name: str = "Custom Dataset Run",
    log_param: bool = False,
) -> ADCDResult:
    """
    Fit a physical correction term to an observed anomaly dataset.

    Args:
        X: Dictionary of independent variable arrays, e.g. {"v": array, "m": array}
        y_obs: Observed outputs (containing anomaly)
        y_classical: Classical theory predictions
        limit_variable: The variable governing the asymptotic classical limit.
            Comma-separated for multivariable scenarios.
        limit_direction: The direction of the limit ("0" or "oo").
            Comma-separated for multivariable scenarios.
        classical_expr: Formula of the classical law (for LLM context)
        variables_with_units: Dictionary of variables and units (e.g. {"v": "m/s"})
        correction_mode: "additive", "multiplicative", or "auto" (automatically detected)
        max_iterations: Max number of discovery iterations
        proposer: The proposer backend ("mock", "gemini", "hybrid")
        api_key: LLM API key (falls back to GEMINI_API_KEY env variable)
        verbose: Print progress logs during optimization
        seed: Random seed for repeatability
        scenario_name: Label for this run (used in logging and result metadata)
        log_param: Log-parameterization to handle extreme physical scales safely (default: False)

    Returns:
        ADCDResult wrapping the discovery outcomes and visualization helpers.
    """
    # 1. Clean input shapes
    for k, v in X.items():
        X[k] = np.asarray(v, dtype=float)
    y_obs = np.asarray(y_obs, dtype=float)
    y_classical = np.asarray(y_classical, dtype=float)
    
    # 2. Handle auto-mode detection
    if correction_mode == "auto":
        mode, confidence = detect_correction_mode(y_obs, y_classical)
        if verbose:
            print(f"[ADCD Auto-Mode] Detected {mode} correction with confidence {confidence:.2f}")
    else:
        mode = correction_mode
        
    # 3. Handle limit variable fallback and parsing
    if limit_variable is None:
        limit_vars = [list(X.keys())[0]]
        if verbose:
            print(f"[ADCD Warning] limit_variable not specified. Defaulting to first key: '{limit_vars[0]}'")
    elif isinstance(limit_variable, str):
        limit_vars = [v.strip() for v in limit_variable.split(",")]
    elif isinstance(limit_variable, (list, tuple)):
        limit_vars = [str(v).strip() for v in limit_variable]
    else:
        limit_vars = [list(X.keys())[0]]

    if isinstance(limit_direction, str):
        limit_dirs = [d.strip() for d in limit_direction.split(",")]
    elif isinstance(limit_direction, (list, tuple)):
        limit_dirs = [str(d).strip() for d in limit_direction]
    else:
        limit_dirs = ["0"]

    # Match lengths
    if len(limit_dirs) < len(limit_vars):
        limit_dirs.extend([limit_dirs[-1]] * (len(limit_vars) - len(limit_dirs)))
    elif len(limit_dirs) > len(limit_vars):
        limit_dirs = limit_dirs[:len(limit_vars)]

    limit_var_str = ",".join(limit_vars)
    limit_dir_str = ",".join(limit_dirs)

    # 4. Construct virtual scenario
    scenario = CustomAnomalyScenario(
        X=X,
        y_obs=y_obs,
        y_classical=y_classical,
        classical_expr=classical_expr,
        correction_type=mode,
        limit_variable=limit_var_str,
        limit_direction=limit_dir_str,
        variables_with_units=variables_with_units,
        name=scenario_name,
    )
    
    # 5. Build proposer
    if proposer == "mock":
        proposer_obj = CorrectionMockProposer(seed=seed)
    elif proposer == "gemini":
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("API key must be provided via `api_key` or GEMINI_API_KEY env var.")
        proposer_obj = CorrectionGeminiProposer(api_key=key)
    elif proposer == "hybrid":
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("API key must be provided via `api_key` or GEMINI_API_KEY env var.")
        proposer_obj = HybridCorrectionProposer(api_key=key)
    else:
        raise ValueError(f"Unknown proposer type: '{proposer}'")
        
    # 6. Configure pipeline
    validator = ASTValidator()
    checker = DimensionalChecker()
    
    regimes = build_arc_regimes(limit_var_str, limit_dir_str)
    scorer = ARCScorer(regimes=regimes)
    pipeline = Stage1Pipeline(validator, checker, scorer)
    optimizer = JAXOptimizer(log_param=log_param)
    
    orchestrator = CorrectionOrchestrator(
        proposer=proposer_obj,
        pipeline=pipeline,
        optimizer=optimizer,
        max_iterations=max_iterations,
        verbose=verbose
    )
    
    # Suppress JAX backend informational noise (TPU unavailable on Windows, etc.)
    # and NumPy RuntimeWarnings from divergent candidate evaluation — these are
    # expected during screening and should never surface to the researcher's terminal.
    jax_logger = logging.getLogger("jax")
    prev_jax_level = jax_logger.level
    jax_logger.setLevel(logging.ERROR)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        search_result = orchestrator.search_correction(scenario, seed=seed)

    # Restore JAX log level after search
    jax_logger.setLevel(prev_jax_level)

    return ADCDResult(
        search_result=search_result,
        scenario=scenario,
        X=X,
        y_obs=y_obs,
        y_classical=y_classical,
    )


def discover_correction(
    scenario: AnomalyScenario,
    noise_level: float = 0.0,
    max_iterations: int = 5,
    proposer: str = "mock",
    correction_mode: str = "auto",
    api_key: Optional[str] = None,
    verbose: bool = True,
    seed: int = 42,
) -> ADCDResult:
    """
    Run ADCD correction discovery on a pre-defined AnomalyScenario.

    Args:
        scenario: The AnomalyScenario to run
        noise_level: Gaussian noise σ as fraction of classical amplitude (0.0 = clean)
        max_iterations: Max iterations for discovery search
        proposer: Proposer type ("mock", "gemini", "hybrid")
        correction_mode: "additive", "multiplicative", or "auto" (default: auto)
        api_key: LLM API key
        verbose: Print progress logs
        seed: Random seed

    Returns:
        ADCDResult wrapping discovery outcome.
    """
    # Generate scenario data
    X, y_obs, y_classical, _ = scenario.generate_data(noise_level=noise_level, seed=seed)
    
    # Route directly to fit()
    return fit(
        X=X,
        y_obs=y_obs,
        y_classical=y_classical,
        limit_variable=scenario.classical_limit_variable,
        limit_direction=scenario.classical_limit_direction,
        classical_expr=scenario.classical_expr,
        variables_with_units=scenario.variables_with_units,
        correction_mode=correction_mode,
        max_iterations=max_iterations,
        proposer=proposer,
        api_key=api_key,
        verbose=verbose,
        seed=seed,
        scenario_name=scenario.name,
    )
