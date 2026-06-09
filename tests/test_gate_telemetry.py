import numpy as np
import pytest
import sympy as sp

from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.arc_scorer import ARCScorer, AsymptoticRegime
from adcd.pipeline import Stage1Pipeline, GateStats


def test_gate_stats_tracks_rejections():
    v = sp.Symbol("v")
    regimes = [AsymptoticRegime(variable=v, limit_target=0, ground_truth_expr="0", weight=1.0)]
    pipeline = Stage1Pipeline(ASTValidator(max_depth=5, max_tokens=15), DimensionalChecker(), ARCScorer(regimes=regimes))

    stats = GateStats()
    candidates = [
        "not valid syntax ++",
        "m * v",
        "0.5 * m * v**2",
    ]
    results = pipeline.execute(candidates, target_dimension_key="E", stats=stats)

    assert stats.input_count == 3
    assert stats.parse_fail >= 1
    assert stats.output_count == len(results)
    assert stats.after_parse == stats.input_count - stats.parse_fail
    assert "overall" in stats.survival_rates()


def test_gate_stats_merge():
    a = GateStats(input_count=10, ast_reject=2, output_count=3)
    b = GateStats(input_count=5, ast_reject=1, output_count=1)
    a.merge(b)
    assert a.input_count == 15
    assert a.ast_reject == 3
    assert a.output_count == 4


def test_orchestrator_returns_gate_stats():
    from adcd.anomaly_scenarios import get_all_scenarios
    from adcd.llm_proposer import CorrectionMockProposer
    from adcd.correction_orchestrator import CorrectionOrchestrator
    from adcd.jax_optimizer import JAXOptimizer

    scenario = get_all_scenarios()[0]
    v = sp.Symbol(scenario.classical_limit_variable)
    limit_target = sp.oo if scenario.classical_limit_direction == "oo" else 0
    regimes = [AsymptoticRegime(variable=v, limit_target=limit_target, ground_truth_expr="0", weight=1.0)]
    checker = DimensionalChecker()
    checker.registry["dimensionless"] = [0, 0, 0]
    pipeline = Stage1Pipeline(ASTValidator(), checker, ARCScorer(regimes=regimes))

    orch = CorrectionOrchestrator(
        proposer=CorrectionMockProposer(seed=42),
        pipeline=pipeline,
        optimizer=JAXOptimizer(n_restarts=2),
        max_iterations=1,
        top_k=3,
        verbose=False,
    )
    result = orch.search_correction(scenario, seed=42)
    assert result.gate_stats is not None
    assert result.gate_stats.input_count > 0
    assert result.total_candidates_optimized >= 0
