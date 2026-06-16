"""
Bayesian posterior estimation over discovered correction candidates.

Uses BIC weight approximation: posterior_i proportional to exp(-delta_BIC_i / 2)
This is the well-established Schwarz approximation to Bayes factors
(Kass & Raftery 1995, JASA; Burnham & Anderson 2002).

Does NOT require MCMC or new infrastructure -- uses BIC scores
already computed by the pipeline.
"""

from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np


@dataclass
class BayesianCorrectionOutput:
    """
    Bayesian posterior distribution over candidate corrections.

    Attributes:
        candidates: list of (expression_str, bic_score) tuples, sorted by BIC ascending
        posterior_weights: normalized posterior probability for each candidate (sums to 1.0)
        correction_class_probs: aggregated probability per functional family
        is_ambiguous: True if top candidate weight < 3x second candidate weight
        evidence_label: "decisive" | "very strong" | "strong" | "substantial" | "weak" | "ambiguous"
        posterior_entropy: Shannon entropy of posterior (bits)
        best_expr: expression string of top candidate
        best_weight: posterior weight of top candidate
    """
    candidates: List[Tuple[str, float]]
    posterior_weights: List[float]
    correction_class_probs: Dict[str, float]
    is_ambiguous: bool
    evidence_label: str
    posterior_entropy: float
    best_expr: str
    best_weight: float


class BayesianReranker:
    """
    Converts BIC-ranked candidates to Bayesian posterior distribution.

    The BIC weight approximation is:
        w_i = exp(-delta_BIC_i / 2) / sum(exp(-delta_BIC_j / 2))
    where delta_BIC_i = BIC_i - BIC_min.

    This is equivalent to the Bayesian Information Criterion model averaging
    used in statistical model selection (Burnham & Anderson 2002).

    Evidence scale follows Kass & Raftery (1995):
        DELTA_BIC > 10  -> decisive evidence
        DELTA_BIC > 6   -> very strong evidence
        DELTA_BIC > 4   -> strong evidence
        DELTA_BIC > 2   -> substantial evidence
        otherwise       -> weak / ambiguous
    """

    # Weight ratio thresholds corresponding to evidence labels
    EVIDENCE_THRESHOLDS = [
        ("decisive",    150.0),   # delta_BIC > 10  -> weight ratio > ~150
        ("very strong",  20.0),   # delta_BIC > 6
        ("strong",        7.4),   # delta_BIC > 4
        ("substantial",   3.0),   # delta_BIC > 2.2
        ("weak",          1.0),   # weight ratio > 1 (best beats second)
    ]

    def __init__(self, threshold_ratio: float = 0.05):
        """
        Args:
            threshold_ratio: minimum posterior weight (relative to top candidate)
                             to include a candidate in output. Candidates below
                             this fraction are pruned.
        """
        self.threshold_ratio = threshold_ratio

    def rank(
        self,
        candidates_with_bic: List[Tuple[str, float]],
    ) -> BayesianCorrectionOutput:
        """
        Convert BIC scores to posterior weights.

        Args:
            candidates_with_bic: List of (expr_str, bic_score) tuples.
                                  Lower BIC = better fit.

        Returns:
            BayesianCorrectionOutput with full posterior distribution.

        Raises:
            ValueError: if candidates_with_bic is empty.
        """
        if not candidates_with_bic:
            raise ValueError("No candidates provided to BayesianReranker")

        # Sort by BIC ascending (lower BIC = better)
        sorted_cands = sorted(candidates_with_bic, key=lambda x: x[1])
        exprs = [c[0] for c in sorted_cands]
        bics = np.array([c[1] for c in sorted_cands], dtype=float)

        # Compute BIC weights: w_i proportional to exp(-delta_BIC_i / 2)
        delta_bic = bics - bics.min()
        log_weights = -0.5 * delta_bic
        # Numerical stability: subtract max before exp (already 0 for best)
        log_weights -= log_weights.max()
        raw_weights = np.exp(log_weights)

        # Prune low-weight candidates (relative to top)
        threshold = raw_weights.max() * self.threshold_ratio
        mask = raw_weights >= threshold
        exprs_pruned = [e for e, m in zip(exprs, mask) if m]
        bics_pruned = bics[mask]
        weights_pruned = raw_weights[mask]

        # Normalize to sum = 1.0
        weights_norm = weights_pruned / weights_pruned.sum()

        # Compute posterior entropy (bits)
        entropy = float(-np.sum(weights_norm * np.log2(weights_norm + 1e-15)))

        # Aggregate posterior by functional family using classify_structure
        try:
            import sympy as sp
            from adcd.metrics import classify_structure
            class_probs: Dict[str, float] = {}
            for expr_str, w in zip(exprs_pruned, weights_norm):
                try:
                    fam = classify_structure(sp.sympify(expr_str))
                except Exception:
                    fam = "unknown"
                class_probs[fam] = class_probs.get(fam, 0.0) + float(w)
        except ImportError:
            class_probs = {}

        # Determine evidence label from weight ratio of top-2
        if len(weights_norm) >= 2:
            weight_ratio = float(weights_norm[0] / (weights_norm[1] + 1e-15))
        else:
            weight_ratio = float("inf")

        evidence_label = "ambiguous"
        for label, threshold_val in self.EVIDENCE_THRESHOLDS:
            if weight_ratio >= threshold_val:
                evidence_label = label
                break

        is_ambiguous = weight_ratio < 3.0

        return BayesianCorrectionOutput(
            candidates=list(zip(exprs_pruned, bics_pruned.tolist())),
            posterior_weights=weights_norm.tolist(),
            correction_class_probs=class_probs,
            is_ambiguous=is_ambiguous,
            evidence_label=evidence_label,
            posterior_entropy=entropy,
            best_expr=exprs_pruned[0],
            best_weight=float(weights_norm[0]),
        )
