"""
ADCD — Anomaly-Driven Correction Discovery
===========================================
A physics-constrained symbolic regression framework that discovers
physical correction terms rather than learning equations from scratch.

DOI: 10.5281/zenodo.20534940

Quick Start
-----------
>>> import adcd
>>> scenarios = adcd.get_all_scenarios()
>>> result = adcd.discover_correction(scenarios[0])
>>> print(result.best_expr)
"""

__version__ = "1.1.0"
__author__ = "Muhammad Afif Erdita"
__email__ = "maeapip10@gmail.com"
__license__ = "MIT"

# High-level API entries
from adcd.api import fit, discover_correction
from adcd.result import ADCDResult

# Core discovery API
from adcd.correction_orchestrator import (
    CorrectionOrchestrator,
    CorrectionIterationResult,
    CorrectionSearchResult,
)

# Scenario definitions
from adcd.anomaly_scenarios import (
    AnomalyScenario,
    get_all_scenarios,
)

# Metrics
from adcd.metrics import (
    evaluate_correction,
    classify_structure,
    bic_score,
    CorrectionEvaluation,
)

# Pipeline
from adcd.pipeline import Stage1Pipeline

# Optimiser
from adcd.jax_optimizer import JAXOptimizer, OptimizationResult

# Gates
from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.arc_scorer import ARCScorer, AsymptoticRegime

__all__ = [
    # High-level API
    "fit",
    "discover_correction",
    "ADCDResult",
    # Discovery
    "CorrectionOrchestrator",
    "CorrectionIterationResult",
    "CorrectionSearchResult",
    # Scenarios
    "AnomalyScenario",
    "get_all_scenarios",
    # Metrics
    "evaluate_correction",
    "classify_structure",
    "bic_score",
    "CorrectionEvaluation",
    # Pipeline / Gates
    "Stage1Pipeline",
    "JAXOptimizer",
    "OptimizationResult",
    "ASTValidator",
    "DimensionalChecker",
    "ARCScorer",
    "AsymptoticRegime",
    # Metadata
    "__version__",
]

