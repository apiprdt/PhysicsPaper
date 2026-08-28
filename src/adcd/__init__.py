# ADCD V3 - Asymptotic Dictionary Correction Discovery
"""
ADCD: A Deterministic, Identifiability-Aware Framework for Recovering
Physical Corrections from Observational Data.
"""

from adcd.api import (
    register_primitive,
    list_primitives,
    discover,
    CustomUserScenario,
    ADCDResult,
)
from adcd.anomaly_scenarios import get_all_scenarios, AnomalyScenario
from adcd.asymptotic_dictionary_proposer_v3 import PRIMITIVE_REGISTRY, Primitive

__all__ = [
    "register_primitive",
    "list_primitives",
    "discover",
    "CustomUserScenario",
    "ADCDResult",
    "get_all_scenarios",
    "AnomalyScenario",
    "PRIMITIVE_REGISTRY",
    "Primitive",
]
