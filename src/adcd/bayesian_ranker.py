"""
BayesianReranker: Continuous evidence reporting layer for ADCD.
Calculates Bayesian Model Averaging (BMA) and evidence discrimination.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from adcd.metrics import classify_structure

# Kass-Raftery (1995, Table 1) formal scale for delta_BIC
_KR_SCALE: List[Tuple[str, float]] = [
    ("decisive",    10.0),   # B10 > 150:1
    ("very_strong",  6.0),   # B10 > 20:1
    ("strong",       4.0),   # B10 > 7.4:1
    ("substantial",  2.0),   # B10 > 3:1
    ("weak",         0.0),   # B10 > 1:1
    ("negative",   -999.0),
]


def _kr_label(delta_bic: Optional[float]) -> str:
    """Returns Kass-Raftery classification label for a given delta_bic."""
    if delta_bic is None or not np.isfinite(delta_bic):
        return "unknown"
    for label, threshold in _KR_SCALE:
        if delta_bic >= threshold:
            return label
    return "negative"


def _safe_exp_bf(delta_bic: Optional[float]) -> Optional[float]:
    """Computes Bayes Factor = exp(delta_bic / 2) with overflow protection."""
    if delta_bic is None or not np.isfinite(delta_bic):
        return None
    half_d = delta_bic / 2.0
    if half_d > 700.0:
        return float("inf")
    if half_d < -700.0:
        return 0.0
    return float(np.exp(half_d))


@dataclass
class EvidenceVsNull:
    """Evidence for the existence of a correction compared to the null model."""
    delta_bic: Optional[float]
    bayes_factor: Optional[float]
    label: str


@dataclass
class EvidenceTop2:
    """Evidence discriminating the best candidate from the runner-up."""
    delta_bic: Optional[float]
    bayes_factor: Optional[float]
    label: str


@dataclass
class BayesianModelOutput:
    best_expr: str
    best_posterior_weight: float

    evidence_vs_null: EvidenceVsNull
    evidence_top2: EvidenceTop2

    candidates: List[Tuple[str, float, float]]
    posterior_entropy: float
    correction_class_probs: Dict[str, float]

    is_detection_ambiguous: bool
    is_discrimination_ambiguous: bool

    n_candidates_input: int
    n_candidates_retained: int
    search_space_size: Optional[int]


class BayesianReranker:
    """
    Ranks candidates using Bayesian Model Averaging and computes
    evidence metrics against the null model and runner-up candidates.
    """

    def __init__(self, pruning_ratio: float = 0.01):
        self.pruning_ratio = pruning_ratio

    def rank(
        self,
        ranked_candidates: List[Any],
        bic_null: Optional[float] = None,
        search_space_size: Optional[int] = None,
    ) -> BayesianModelOutput:
        """
        Calculates posterior weights and evidence metrics for a list of candidates.

        Args:
            ranked_candidates: List of candidates as 2-tuple (expr, bic),
                               3-tuple (expr, nmse, bic), or 4-tuple (expr, nmse, bic, theta).
            bic_null: BIC of the null model (no correction) for detection evidence.
                      If None, evidence_vs_null.label will be "unknown".
            search_space_size: Search space size M used in 2*ln(M) penalty, for documentation only.
        """
        if not ranked_candidates:
            raise ValueError("ranked_candidates is empty.")

        parsed: List[Tuple[str, float, Optional[Dict]]] = []
        for item in ranked_candidates:
            if not isinstance(item, (tuple, list)):
                raise ValueError(f"Unsupported candidate type: {type(item)}")
            if len(item) >= 4:
                expr_str, _nmse, bic, theta_fit = item[0], item[1], item[2], item[3]
            elif len(item) == 3:
                expr_str, _nmse, bic = item[0], item[1], item[2]
                theta_fit = None
            elif len(item) == 2:
                expr_str, bic = item[0], item[1]
                theta_fit = None
            else:
                raise ValueError(f"Unknown candidate format: len={len(item)}")
            parsed.append((str(expr_str), float(bic), theta_fit))

        parsed.sort(key=lambda x: x[1])
        n_input = len(parsed)

        exprs  = [p[0] for p in parsed]
        bics   = np.array([p[1] for p in parsed], dtype=np.float64)
        thetas = [p[2] for p in parsed]

        log_unnorm = -0.5 * bics
        log_unnorm -= log_unnorm.max()
        raw_w = np.exp(log_unnorm)
        all_weights = raw_w / raw_w.sum()

        if bic_null is not None and np.isfinite(bic_null):
            d_null = float(bic_null - bics[0])
            evn = EvidenceVsNull(
                delta_bic=d_null,
                bayes_factor=_safe_exp_bf(d_null),
                label=_kr_label(d_null),
            )
        else:
            evn = EvidenceVsNull(delta_bic=None, bayes_factor=None, label="unknown")

        if len(bics) >= 2:
            d_top2 = float(bics[1] - bics[0])
            et2 = EvidenceTop2(
                delta_bic=d_top2,
                bayes_factor=_safe_exp_bf(d_top2),
                label=_kr_label(d_top2),
            )
        else:
            et2 = EvidenceTop2(delta_bic=None, bayes_factor=None, label="unknown")

        prune_thresh = all_weights.max() * self.pruning_ratio
        mask = all_weights >= prune_thresh
        exprs_ret   = [e for e, m in zip(exprs, mask) if m]
        bics_ret    = bics[mask]
        thetas_ret  = [t for t, m in zip(thetas, mask) if m]
        weights_ret = all_weights[mask]
        weights_ret = weights_ret / weights_ret.sum()
        n_retained  = int(mask.sum())

        entropy = float(-np.sum(weights_ret * np.log2(weights_ret + 1e-300)))

        class_probs: Dict[str, float] = {}
        for expr_str, w, th in zip(exprs_ret, weights_ret, thetas_ret):
            try:
                fam = classify_structure(expr_str, th)
            except Exception:
                fam = "unknown"
            class_probs[fam] = class_probs.get(fam, 0.0) + float(w)

        _ambiguous_labels = {"weak", "negative", "unknown"}

        return BayesianModelOutput(
            best_expr=exprs_ret[0],
            best_posterior_weight=float(weights_ret[0]),
            evidence_vs_null=evn,
            evidence_top2=et2,
            candidates=[(e, float(b), float(w)) for e, b, w in zip(exprs_ret, bics_ret, weights_ret)],
            posterior_entropy=entropy,
            correction_class_probs=class_probs,
            is_detection_ambiguous=evn.label in _ambiguous_labels,
            is_discrimination_ambiguous=et2.label in _ambiguous_labels,
            n_candidates_input=n_input,
            n_candidates_retained=n_retained,
            search_space_size=search_space_size,
        )

    def format_summary(self, output: BayesianModelOutput) -> str:
        """Generates a formatted text summary of the Bayesian ranking output."""
        evn = output.evidence_vs_null
        et2 = output.evidence_top2

        def _fmt_bf(bf: Optional[float]) -> str:
            if bf is None:
                return "N/A"
            return f"{bf:.1e}" if bf > 1e4 or bf == float("inf") else f"{bf:.1f}"

        lines = [
            "─" * 65,
            f"  Best candidate : {output.best_expr}",
            f"  Posterior weight: {output.best_posterior_weight:.3f}  "
            f"(entropy={output.posterior_entropy:.2f} bits, "
            f"n_retained={output.n_candidates_retained}/{output.n_candidates_input})",
            "",
            "  [1] Evidence FOR correction (vs null model):",
            f"      delta_BIC = {evn.delta_bic:.2f}  BF = {_fmt_bf(evn.bayes_factor)}  [{evn.label.upper()}]"
            if evn.delta_bic is not None
            else "      delta_BIC = N/A  (bic_null not provided)",
            "",
            "  [2] Evidence WHICH correction (top-1 vs top-2):",
            f"      delta_BIC = {et2.delta_bic:.2f}  BF = {_fmt_bf(et2.bayes_factor)}  [{et2.label.upper()}]"
            if et2.delta_bic is not None
            else "      delta_BIC = N/A  (only 1 candidate)",
            "",
            "  Structural class probabilities (BMA):",
        ]
        for cls, prob in sorted(output.correction_class_probs.items(), key=lambda x: -x[1]):
            lines.append(f"      {cls:<20}: {prob:.3f}")

        if output.search_space_size is not None:
            lines.append(
                f"\n  Search space: M = {output.search_space_size}  "
                f"(2*ln(M) = {2 * np.log(output.search_space_size):.2f} BIC penalty)"
            )
        lines.append("─" * 65)
        return "\n".join(lines)
