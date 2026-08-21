from dataclasses import dataclass
from typing import List, Tuple, Dict, Optional
import numpy as np


@dataclass
class BayesianCorrectionOutput:
    candidates: List[Tuple[str, float]]
    posterior_weights: List[float]
    correction_class_probs: Dict[str, float]
    is_ambiguous: bool
    evidence_label: str
    posterior_entropy: float
    best_expr: str
    best_weight: float


class BayesianReranker:
    EVIDENCE_THRESHOLDS = [
        ("decisive",    150.0),   # delta_BIC > 10
        ("very strong",  20.0),   # delta_BIC > 6
        ("strong",        7.4),   # delta_BIC > 4
        ("substantial",   3.0),   # delta_BIC > 2.2
        ("weak",          1.0),
    ]

    def __init__(self, threshold_ratio: float = 0.01):
        self.threshold_ratio = threshold_ratio

    def rank(
        self,
        candidates_with_bic: List[Tuple[str, float]],
        n_candidates: Optional[int] = None,
    ) -> BayesianCorrectionOutput:
        if not candidates_with_bic:
            raise ValueError("No candidates provided to BayesianReranker")

        sorted_cands = sorted(candidates_with_bic, key=lambda x: x[1])
        exprs = [c[0] for c in sorted_cands]
        bics = np.array([c[1] for c in sorted_cands], dtype=float)

        delta_bic = bics - bics.min()
        log_weights = -0.5 * delta_bic
        raw_weights = np.exp(log_weights - log_weights.max())
        full_normalized_weights = raw_weights / raw_weights.sum()

        # FIX AUDIT: Hitung rasio bukti SEBELUM pemangkasan kandidat minoritas
        if len(full_normalized_weights) >= 2:
            weight_ratio = float(full_normalized_weights[0] / (full_normalized_weights[1] + 1e-15))
        else:
            weight_ratio = float("inf")

        evidence_label = "ambiguous"
        for label, threshold_val in self.EVIDENCE_THRESHOLDS:
            if weight_ratio >= threshold_val:
                evidence_label = label
                break

        # Pemangkasan untuk pelaporan parsimonis
        threshold = full_normalized_weights.max() * self.threshold_ratio
        mask = full_normalized_weights >= threshold
        exprs_pruned = [e for e, m in zip(exprs, mask) if m]
        bics_pruned = bics[mask]
        weights_pruned = full_normalized_weights[mask]
        weights_norm = weights_pruned / weights_pruned.sum()

        entropy = float(-np.sum(weights_norm * np.log2(weights_norm + 1e-15)))

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

        return BayesianCorrectionOutput(
            candidates=list(zip(exprs_pruned, bics_pruned.tolist())),
            posterior_weights=weights_norm.tolist(),
            correction_class_probs=class_probs,
            is_ambiguous=weight_ratio < 3.0,
            evidence_label=evidence_label,
            posterior_entropy=entropy,
            best_expr=exprs_pruned[0],
            best_weight=float(weights_norm[0]),
        )
