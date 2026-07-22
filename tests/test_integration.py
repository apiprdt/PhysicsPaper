import pytest
import numpy as np
import sympy as sp

from adcd.feynman_dataset import get_all_problems, get_problem, FeynmanProblem
from adcd.llm_proposer import MockProposer, AnthropicProposer, ProposalContext
from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.arc_scorer import ARCScorer, AsymptoticRegime
from adcd.pipeline import Stage1Pipeline
from adcd.jax_optimizer import JAXOptimizer
from adcd.orchestrator import SROrchestrator, SearchResult

def test_feynman_problems_and_data_generation():
    """Verify that all 20 benchmark equations can be retrieved and generate valid data."""
    problems = get_all_problems()
    assert len(problems) == 20

    for prob in problems:
        assert isinstance(prob, FeynmanProblem)
        assert len(prob.variables) > 0
        assert isinstance(prob.equation, str)
        
        # Test synthetic data generation
        n_points = 50
        X, y = prob.generate_data(n_points=n_points, seed=123)
        
        # Verify shape consistency
        assert y.shape == (n_points,)
        for var in prob.variables:
            assert X[var].shape == (n_points,)
        
        for const_name in prob.constants:
            assert X[const_name].shape == (n_points,)
            assert np.all(X[const_name] == prob.constants[const_name])

    # Test single retrieval
    ke_prob = get_problem("Kinetic Energy")
    assert ke_prob.name == "Kinetic Energy"
    assert ke_prob.target_dimension == "E"

def test_mock_proposer_and_anthropic_stub():
    """Verify that the MockProposer generates unique SymPy expressions and AnthropicProposer raises NotImplementedError."""
    proposer = MockProposer(seed=100)
    
    # Setup a context
    context = ProposalContext(
        variable_names=["m", "v"],
        target_name="E",
        data_statistics={
            "m": {"mean": 2.0, "std": 1.0, "min": 0.5, "max": 4.0},
            "v": {"mean": 3.0, "std": 1.5, "min": 0.5, "max": 6.0}
        },
        n_candidates=30,
        iteration=0
    )
    
    candidates = proposer.propose(context)
    
    assert len(candidates) == 30
    assert len(set(candidates)) == 30  # Should be unique
    
    # All candidates must be SymPy parseable
    for cand in candidates:
        expr = sp.sympify(cand)
        assert isinstance(expr, sp.Expr)
        
    # AnthropicProposer stub must raise NotImplementedError
    stub = AnthropicProposer()
    with pytest.raises(NotImplementedError):
        stub.propose(context)

def test_orchestrator_end_to_end_kinetic_energy():
    """Verify the entire closed-loop pipeline can successfully run and recover the Kinetic Energy law."""
    # 1. Load the Kinetic Energy benchmark problem
    problem = get_problem("Kinetic Energy")
    X, y_obs = problem.generate_data(n_points=100, seed=42)
    
    # 2. Instantiate pipeline components
    # We build the Asymptotic regimes
    regimes = []
    for r_dict in problem.regimes:
        limit_val = sp.oo if r_dict["limit"] == "oo" else r_dict["limit"]
        regimes.append(
            AsymptoticRegime(
                variable=r_dict["variable"],
                limit_target=limit_val,
                ground_truth_expr=r_dict["expected"],
                weight=1.0
            )
        )
        
    validator = ASTValidator(max_depth=6, max_tokens=20)
    
    # Configure Dimensional Checker registry (inject theta variables dynamically as dimensionless)
    checker = DimensionalChecker()
    for i in range(100):
        checker.registry[f"theta_{i}"] = [0, 0, 0]
    checker.locals = {s: sp.Symbol(s) for s in checker.registry}
    
    scorer = ARCScorer(regimes=regimes)
    pipeline = Stage1Pipeline(validator, checker, scorer)
    
    # 3. Instantiate JAX optimizer
    optimizer = JAXOptimizer(n_restarts=3)
    
    # 4. Instantiate proposer
    proposer = MockProposer(seed=42)
    
    # 5. Connect all stages in Orchestrator
    orchestrator = SROrchestrator(
        proposer=proposer,
        pipeline=pipeline,
        optimizer=optimizer,
        max_iterations=3,
        top_k=10,
        convergence_nmse=1e-5,
        verbose=True
    )
    
    # Run the closed loop search
    result = orchestrator.search(
        variable_names=problem.variables,
        target_name="E",
        target_dimension=problem.target_dimension,
        X=X,
        y_obs=y_obs,
        data_vars=problem.variables,
        seed=42
    )
    
    # Verify SearchResult outcomes
    assert isinstance(result, SearchResult)
    assert len(result.history) > 0
    assert result.total_candidates_proposed > 0
    assert result.total_candidates_survived_stage1 > 0
    
    # The MockProposer template bank contains "theta_0 * {v1} * {v2}**theta_1"
    # When substituted with m and v, this becomes "theta_0 * m * v**theta_1"
    # The JAX optimizer should converge this to theta_0 ≈ 0.5, theta_1 ≈ 2.0
    # Thus, the best equation should have a very low NMSE (< 1e-4) and be converged!
    assert result.best_nmse < 1e-4
    assert any(k.startswith("theta_") for k in result.best_theta)
    assert result.converged is True
