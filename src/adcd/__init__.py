"""
ADCD: Asymptotic Dictionary Correction Discovery for Identifiable Symbolic Regression

A deterministic, identifiability-aware framework for recovering algebraic
corrections to known physical laws from noisy observational data.
"""

__version__ = "1.0.0"
__author__ = "Muhammad Afif Erdita"

from adcd.anomaly_scenarios import AnomalyScenario, get_all_scenarios
from adcd.arc_scorer import ARCScorer, build_arc_regimes
from adcd.asymptotic_dictionary_proposer_v3 import GrammarBudget, PRIMITIVE_REGISTRY
from adcd.context import ProposalContext
from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.grammar_proposer_v3 import GrammarProposerV3
from adcd.jax_optimizer import JAXOptimizer
from adcd.metrics import classify_structure, extended_bic_score
from adcd.mode_detection import detect_correction_mode
from adcd.pipeline import GateStats, Stage1Pipeline
from adcd.quickfit import DOMAIN_TAXONOMY

try:
    from adcd.auto_scenario import AutoCSVScenario, build_scenario_from_csv, load_scenario_from_csv
except ImportError:
    AutoCSVScenario = None  # type: ignore[assignment,misc]
    build_scenario_from_csv = None  # type: ignore[assignment]
    load_scenario_from_csv = None  # type: ignore[assignment]

__all__ = [
    "__version__",
    "__author__",
    "AnomalyScenario",
    "get_all_scenarios",
    "AutoCSVScenario",
    "build_scenario_from_csv",
    "load_scenario_from_csv",
    "DOMAIN_TAXONOMY",
    "PRIMITIVE_REGISTRY",
    "GrammarBudget",
    "GrammarProposerV3",
    "ProposalContext",
    "ASTValidator",
    "DimensionalChecker",
    "ARCScorer",
    "build_arc_regimes",
    "Stage1Pipeline",
    "GateStats",
    "JAXOptimizer",
    "detect_correction_mode",
    "classify_structure",
    "extended_bic_score",
]
