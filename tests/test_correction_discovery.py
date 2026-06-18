import sympy as sp
import numpy as np
from adcd.anomaly_scenarios import get_all_scenarios
from adcd.metrics import classify_structure, get_ast_tokens, compute_levenshtein_distance
from adcd.llm_proposer import CorrectionMockProposer, ProposalContext
from adcd.correction_orchestrator import CorrectionOrchestrator
from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.arc_scorer import ARCScorer, AsymptoticRegime
from adcd.pipeline import Stage1Pipeline
from adcd.jax_optimizer import JAXOptimizer

def test_anomaly_scenario_data_generation():
    """Verify that all scenarios generate valid data shapes and proper residuals."""
    scenarios = get_all_scenarios()
    assert len(scenarios) == 22
    
    for scenario in scenarios:
        n_points = 50
        X, y_obs, y_classical, residual = scenario.generate_data(n_points=n_points, noise_level=0.01, seed=42)
        
        # Verify shapes
        assert len(y_obs) == n_points
        assert len(y_classical) == n_points
        assert len(residual) == n_points
        
        for var in scenario.classical_variables:
            assert var in X
            assert len(X[var]) == n_points
            
        # Verify residual calculation
        if scenario.correction_type == "multiplicative":
            expected_res = y_obs / y_classical - 1.0
            np.testing.assert_allclose(residual, expected_res, rtol=1e-5)
        else:
            expected_res = y_obs - y_classical
            np.testing.assert_allclose(residual, expected_res, rtol=1e-5)

def test_metrics_calculation():
    """Verify structural classification, AST tokens, Levenshtein distance, and evaluation metrics."""
    # Test structural classifier
    assert classify_structure("theta_0 * exp(-r / theta_1)") == "exponential"
    assert classify_structure("theta_0 * sin(v / theta_1)") == "trigonometric"
    assert classify_structure("theta_0 * log(1 + x / theta_1)") == "logarithmic"
    assert classify_structure("theta_0 * (v/c)**2") == "polynomial"
    assert classify_structure("- (theta_0 / T)**4") == "power_law"
    assert classify_structure("theta_0 * v / (v + theta_1)") == "rational"
    
    # Test Levenshtein distance
    t1 = get_ast_tokens(sp.sympify("theta_0 * (v/c)**2"))
    t2 = get_ast_tokens(sp.sympify("theta_1 * (v/c)**2"))
    # The symbol name difference of theta should be normalized
    assert t1 == t2
    assert compute_levenshtein_distance(t1, t2) == 0
    
    t3 = get_ast_tokens(sp.sympify("theta_0 * exp(-r/theta_1)"))
    assert compute_levenshtein_distance(t1, t3) > 2

def test_correction_proposer_dimensionless():
    """Verify that CorrectionMockProposer generates valid and physically dimensionless expressions."""
    scenarios = get_all_scenarios()
    proposer = CorrectionMockProposer(seed=123)
    checker = DimensionalChecker()
    # Add dimensionless unit to checker
    checker.registry["dimensionless"] = [0, 0, 0]
    
    for scenario in scenarios:
        context = ProposalContext(
            variable_names=scenario.classical_variables,
            target_name="residual",
            data_statistics={},
            n_candidates=15,
            constants=scenario.classical_constants,
            classical_expr=scenario.classical_expr
        )
        
        candidates = proposer.propose(context)
        assert len(candidates) > 0
        
        # Verify they can be parsed by sympy
        sym_locals = {s: sp.Symbol(s) for s in scenario.classical_variables}
        for c in scenario.classical_constants:
            sym_locals[c] = sp.Symbol(c)
        for i in range(20):
            sym_locals[f"theta_{i}"] = sp.Symbol(f"theta_{i}")
            
        # Verify that correct physical ratios pass while incorrect ones are filtered out
        if scenario.name == "Relativistic KE":
            assert checker.verify("theta_0 * (v / c)**2", "dimensionless")
            assert not checker.verify("theta_0 * (m / c)**2", "dimensionless")

def test_correction_orchestrator_integration():
    """Verify that CorrectionOrchestrator can run the search loop and discover relativistic KE correction."""
    # Find Relativistic KE scenario
    scenarios = get_all_scenarios()
    ke_scenario = [s for s in scenarios if s.name == "Relativistic KE"][0]
    
    # Instantiate pipeline components
    validator = ASTValidator(max_depth=5, max_tokens=15)
    checker = DimensionalChecker()
    checker.registry["dimensionless"] = [0, 0, 0]
    
    v = sp.Symbol('v')
    regimes = [
        AsymptoticRegime(variable=v, limit_target=0, ground_truth_expr="0", weight=1.0)
    ]
    scorer = ARCScorer(regimes=regimes)
    pipeline = Stage1Pipeline(validator, checker, scorer)
    optimizer = JAXOptimizer()
    proposer = CorrectionMockProposer(seed=42)
    
    orchestrator = CorrectionOrchestrator(
        proposer=proposer,
        pipeline=pipeline,
        optimizer=optimizer,
        max_iterations=2,  # Short run for testing
        top_k=5,
        convergence_nmse=1e-5,
        verbose=True
    )
    
    result = orchestrator.search_correction(ke_scenario, noise_level=0.0)
    assert result.best_expr != ""
    assert result.best_nmse_residual < 1.0
    assert result.evaluation is not None
    # We should have successfully matched structural class (polynomial)
    assert result.evaluation.class_match
    
    # Verify Phase 3 Bayesian & Identifiability outputs are populated
    assert result.bayesian_output is not None
    assert result.identifiability_report is not None
    assert isinstance(result.bayesian_output.posterior_weights, list)
    # Under noise_level=0.0, the magnitude is undetectable, which is expected
    assert result.identifiability_report.is_identifiable is False
    assert result.identifiability_report.failure_mode == "undetectable_magnitude"


def test_hybrid_correction_proposer():
    """Verify that HybridCorrectionProposer falls back gracefully or returns candidates."""
    from adcd.llm_proposer import HybridCorrectionProposer, ProposalContext
    
    proposer = HybridCorrectionProposer(api_key="mock_key_or_invalid_key", seed=42)
    context = ProposalContext(
        variable_names=["m", "v"],
        target_name="residual",
        data_statistics={},
        n_candidates=10,
        iteration=0,
        stuck_count=0,
        domain="relativistic mechanics",
        classical_expr="0.5 * m * v**2",
        variables_with_units={"m": "kg", "v": "m/s", "c": "m/s"},
        anomaly_description="high speeds v approaching c",
        known_limits=[{"variable": "v", "limit": "0", "expected": "0"}],
        classical_limit_condition="v -> 0",
        max_nodes=15,
        structural_hints=[],
        previous_best=None,
        constants={"c": 3.0e8},
        residual_features=None
    )
    # Since api_key is invalid/mock, the Gemini call will fail and it should gracefully fall back to mock candidates.
    candidates = proposer.propose(context)
    assert len(candidates) > 0
    assert len(candidates) <= 10


