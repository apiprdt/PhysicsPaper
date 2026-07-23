"""
Extended Grammar Proposer for ADCD.

Provides physically justified mathematical templates beyond elementary polynomials,
labeled explicitly as 'Extended Grammar' (never 'Complete Basis').
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class FunctionalPattern:
    """Metadata for an extended grammar pattern."""

    name: str
    template_str: str
    physical_justification: str
    domain: str  # e.g., "gaussian_velocity", "cylindrical_wave", "phase_saturation"


class ExtendedGrammarProposer:
    """Proposes candidates using an extended physical grammar.

    Each template pattern is paired with an explicit physical justification.
    """

    FIRST_WAVE_PATTERNS: List[FunctionalPattern] = [
        FunctionalPattern(
            name="erf",
            template_str="theta_0 * erf(theta_1 * {var})",
            physical_justification="Gaussian velocity distribution integral / thermal error function boundary layer",
            domain="gaussian_velocity",
        ),
        FunctionalPattern(
            name="J0",
            template_str="theta_0 * besselj(0, theta_1 * {var})",
            physical_justification="Cylindrical symmetry / Bessel wave scattering / disk oscillation mode",
            domain="cylindrical_wave",
        ),
        FunctionalPattern(
            name="arctan",
            template_str="theta_0 * atan(theta_1 * {var})",
            physical_justification="Bounded phase-space saturation / smooth velocity dispersion threshold",
            domain="phase_saturation",
        ),
        FunctionalPattern(
            name="tanh",
            template_str="theta_0 * tanh(theta_1 * {var})",
            physical_justification="Smooth non-linear phase transition / Fermi-Dirac style saturation",
            domain="phase_transition",
        ),
        FunctionalPattern(
            name="yukawa_decay",
            template_str="theta_0 * exp(-theta_1 * {var})",
            physical_justification="Yukawa exponential screening / short-range force suppression",
            domain="force_screening",
        ),
        FunctionalPattern(
            name="log_profile",
            template_str="theta_0 * log(1 + theta_1 * {var})",
            physical_justification="Logarithmic potential / NFW dark matter halo density profile",
            domain="halo_density",
        ),
    ]

    def __init__(self, patterns: Optional[List[FunctionalPattern]] = None) -> None:
        self.patterns = patterns or self.FIRST_WAVE_PATTERNS

    def propose_candidates(
        self,
        variables: List[str],
        ratios: Optional[List[str]] = None,
    ) -> List[Dict[str, str]]:
        """Generates candidate expressions and metadata for the registered variables/ratios.

        Returns a list of dicts with keys 'expr' and 'justification'.
        """
        candidates: List[Dict[str, str]] = []
        target_vars = ratios if ratios else variables

        for var in target_vars:
            for pat in self.patterns:
                expr_str = pat.template_str.format(var=var)
                candidates.append(
                    {
                        "expr": expr_str,
                        "pattern_name": pat.name,
                        "justification": pat.physical_justification,
                        "domain": pat.domain,
                    }
                )

        return candidates

    def get_pattern_count(self) -> int:
        return len(self.patterns)
