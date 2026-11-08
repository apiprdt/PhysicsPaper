"""
claim_card.py
=============
Defines the ClaimCard schema: the formal input contract for the ADCD Identifiability Auditor.

A ClaimCard encapsulates all physical, mathematical, and data assumptions required
to audit whether competing hypotheses are identifiable on a given dataset.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Dict, List, Literal, Optional, Tuple
import numpy as np
from pydantic import BaseModel, Field, model_validator


class CandidateProvenance(BaseModel):
    """
    Provenance record preventing unverified novelty claims by AI agents.
    Enforces tracking of the historical and mathematical origin of every candidate formula.
    """
    expression: str
    family_name: str
    origin_type: Literal[
        "literature",
        "dictionary_primitive",
        "symbolic_regression",
        "postulate_derivation",
    ]
    citation_or_source: str = Field(
        ...,
        description="Explicit citation (e.g. 'Churchill & Usagi 1972') or formal derivation reference",
    )
    is_novel_claim: bool = Field(
        default=False,
        description="Must be False unless backed by a human-verified postulate derivation",
    )

    @model_validator(mode="after")
    def validate_novelty_integrity(self) -> CandidateProvenance:
        if self.origin_type == "literature" and self.is_novel_claim:
            raise ValueError(
                f"Integrity error: Candidate '{self.expression}' has origin_type='literature' "
                f"with citation '{self.citation_or_source}', so is_novel_claim cannot be True."
            )
        if not self.citation_or_source.strip():
            raise ValueError(
                f"Integrity error: Candidate '{self.expression}' must have a non-empty citation_or_source."
            )
        return self


class NoiseSpec(BaseModel):
    """Specification of the observational measurement noise."""
    type: Literal["relative", "absolute", "column"] = "relative"
    value: Optional[float] = 0.01
    sigma_col: Optional[str] = None
    fit_intrinsic_scatter: bool = False


class ClaimCard(BaseModel):
    """
    Formal claim card contract for identifiability auditing.

    Outsiders and researchers specify their hypothesis, variables, baseline,
    limit statements, observation window, and candidate formulas in this card.
    """
    name: str = "Anonymous Claim"
    variables: Dict[str, str] = Field(
        default_factory=dict,
        description="Variable names mapped to physical units or dimensions (e.g. {'r': 'm', 'q': 'C'})",
    )
    baseline_expr: str = Field(
        ...,
        description="Symbolic expression for the baseline/classical law (e.g. 'k_e * q1 * q2 / r')",
    )
    baseline_constants: Dict[str, float] = Field(
        default_factory=dict,
        description="Fixed physical constants or initial baseline parameter values",
    )
    parameter_priors: Optional[Dict[str, Tuple[float, float]]] = Field(
        default=None,
        description="Gaussian priors for baseline nuisance parameters phi: {param_name: (mean, std)}",
    )
    limit_variable: str = Field(
        ...,
        description="The variable or dimensionless argument along which the classical limit is taken",
    )
    limit_direction: str = Field(
        default="0",
        description="Direction of the asymptotic limit ('0', '+0', '-0', 'oo', 'inf', '+oo')",
    )
    window: Optional[Dict[str, Tuple[float, float]]] = Field(
        default=None,
        description="Observation domain window per variable: {var: (min_val, max_val)}",
    )
    mode: Literal["multiplicative", "additive"] = Field(
        default="multiplicative",
        description="Mode of correction: multiplicative y = y_cl * (1 + Delta) or additive y = y_cl + Delta",
    )
    candidate_expressions: Optional[List[str]] = Field(
        default=None,
        description="Optional list of custom imported candidate formulas (from PySR, literature, or AI)",
    )
    candidate_families: Optional[Dict[str, str]] = Field(
        default=None,
        description="Optional mapping {candidate_expr: family_name} for self-ablation grouping",
    )
    provenance: Optional[List[CandidateProvenance]] = Field(
        default=None,
        description="Optional provenance records tracking the literature source of each candidate",
    )
    custom_argument: Optional[str] = Field(
        default=None,
        description="Optional explicit composite dimensionless argument u, e.g. 'hbar * omega / (k_B * T)'",
    )
    data_reference: Optional[str] = Field(
        default=None,
        description="Optional URI, path, or description of the dataset being audited",
    )
    noise_spec: NoiseSpec = Field(
        default_factory=NoiseSpec,
        description="Observational noise model",
    )
    check_adequacy: bool = Field(
        default=False,
        description="Whether to run non-parametric monotone adequacy check (diagnostic warning only)",
    )
    seed: int = Field(
        default=42,
        description="Random seed for deterministic tie-breaking and optimization restarts",
    )

    def canonical_dict(self) -> Dict[str, Any]:
        """Produce dictionary with sorted keys for canonical serialization."""
        return self.model_dump()

    def canonical_json(self) -> str:
        """Produce deterministic, canonical JSON with sorted keys and normalized finite floats."""
        def _normalize_obj(obj: Any) -> Any:
            if isinstance(obj, dict):
                return {str(k): _normalize_obj(v) for k, v in sorted(obj.items())}
            elif isinstance(obj, (list, tuple)):
                return [_normalize_obj(x) for x in obj]
            elif isinstance(obj, (float, np.floating)):
                fval = float(obj)
                if not math.isfinite(fval):
                    raise ValueError(f"Non-finite float ({fval}) is not permitted in canonical ClaimCard JSON.")
                return 0.0 if fval == 0.0 else float(f"{fval:.15g}")
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj

        sorted_dict = _normalize_obj(self.canonical_dict())
        return json.dumps(sorted_dict, sort_keys=True, separators=(",", ":"), allow_nan=False)

    def canonical_hash(self) -> str:
        """SHA-256 hash of the canonical JSON representation."""
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serialize ClaimCard to formatted JSON string."""
        return self.model_dump_json(indent=indent)

    @classmethod
    def from_json(cls, json_str: str) -> ClaimCard:
        """Load ClaimCard from JSON string."""
        return cls.model_validate_json(json_str)
