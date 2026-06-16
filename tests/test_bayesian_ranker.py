"""Unit tests for BayesianReranker (Task P3-1)."""

import pytest
import numpy as np
from adcd.bayesian_ranker import BayesianReranker, BayesianCorrectionOutput


def test_decisive_evidence():
    """Single dominant candidate -> decisive evidence."""
    candidates = [("theta_0 * v**2", -1000.0), ("theta_0 * exp(-v)", -990.0)]
    reranker = BayesianReranker()
    out = reranker.rank(candidates)
    assert out.evidence_label in ("decisive", "very strong", "strong")
    assert not out.is_ambiguous
    assert out.best_expr == "theta_0 * v**2"


def test_ambiguous_evidence():
    """Nearly equal BIC -> ambiguous."""
    candidates = [("theta_0 * v**2", -1000.0), ("theta_0 * v", -999.5)]
    reranker = BayesianReranker()
    out = reranker.rank(candidates)
    assert out.is_ambiguous


def test_posterior_sums_to_one():
    """Posterior weights must sum to 1.0."""
    candidates = [("a", -100.0), ("b", -95.0), ("c", -80.0)]
    reranker = BayesianReranker()
    out = reranker.rank(candidates)
    assert abs(sum(out.posterior_weights) - 1.0) < 1e-10


def test_entropy_nonnegative():
    """Entropy should always be >= 0."""
    candidates = [("a", -100.0), ("b", -95.0)]
    reranker = BayesianReranker()
    out = reranker.rank(candidates)
    assert out.posterior_entropy >= 0.0


def test_single_candidate():
    """Single candidate -> weight=1.0, not ambiguous, entropy~0."""
    candidates = [("theta_0 * v**2", -500.0)]
    reranker = BayesianReranker()
    out = reranker.rank(candidates)
    assert abs(out.best_weight - 1.0) < 1e-10
    assert not out.is_ambiguous
    assert out.posterior_entropy < 0.01


def test_empty_candidates_raises():
    """Empty list should raise ValueError."""
    reranker = BayesianReranker()
    with pytest.raises(ValueError):
        reranker.rank([])


def test_best_expr_is_lowest_bic():
    """best_expr must correspond to the candidate with the lowest BIC."""
    candidates = [("expr_bad", -50.0), ("expr_best", -200.0), ("expr_mid", -100.0)]
    reranker = BayesianReranker()
    out = reranker.rank(candidates)
    assert out.best_expr == "expr_best"


def test_pruning_removes_low_weight():
    """Candidates with weight < threshold_ratio * max_weight are pruned."""
    # delta_BIC = 300 -> exp(-150) << threshold, will be pruned
    candidates = [("dominant", -1000.0), ("irrelevant", -700.0)]
    reranker = BayesianReranker(threshold_ratio=0.05)
    out = reranker.rank(candidates)
    # Only dominant should survive after pruning
    assert len(out.candidates) == 1
    assert out.candidates[0][0] == "dominant"


def test_output_is_dataclass():
    """Output is BayesianCorrectionOutput with correct fields."""
    candidates = [("theta_0 * v**2", -100.0), ("theta_0 * v", -90.0)]
    reranker = BayesianReranker()
    out = reranker.rank(candidates)
    assert isinstance(out, BayesianCorrectionOutput)
    assert isinstance(out.posterior_weights, list)
    assert isinstance(out.correction_class_probs, dict)
    assert isinstance(out.posterior_entropy, float)
    assert isinstance(out.is_ambiguous, bool)
    assert isinstance(out.evidence_label, str)
