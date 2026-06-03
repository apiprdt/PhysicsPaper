import os
import numpy as np
import sympy as sp
from src.anomaly_scenarios import get_all_scenarios
from src.metrics import evaluate_correction, bic_score
from src.llm_proposer import CorrectionMockProposer, ProposalContext
from src.correction_orchestrator import CorrectionOrchestrator
from src.dimensional_checker import ASTValidator, DimensionalChecker
from src.arc_scorer import ARCScorer, AsymptoticRegime
from src.pipeline import Stage1Pipeline
from src.jax_optimizer import JAXOptimizer

# Get scenario
scenarios = get_all_scenarios()
scenario = [s for s in scenarios if s.name == "Anharmonic Spring"][0]

# Setup
validator = ASTValidator()
checker = DimensionalChecker()
checker.registry["dimensionless"] = [0, 0, 0]
limit_var = sp.Symbol(scenario.classical_limit_variable)
limit_target = sp.oo if scenario.classical_limit_direction == "oo" else 0
regimes = [
    AsymptoticRegime(
        variable=limit_var,
        limit_target=limit_target,
        ground_truth_expr="0",
        weight=1.0
    )
]
scorer = ARCScorer(regimes=regimes)
pipeline = Stage1Pipeline(validator, checker, scorer)
optimizer = JAXOptimizer()
proposer = CorrectionMockProposer(seed=42)

# Set symbols
for var in scenario.classical_variables:
    if var not in pipeline.checker.registry:
        pipeline.checker.registry[var] = [0, 0, 0]
    if var not in pipeline.locals:
        pipeline.locals[var] = sp.Symbol(var)

X, y_obs, y_classical, residual = scenario.generate_data(noise_level=0.0, seed=42)

# Generate candidates
context = ProposalContext(
    variable_names=scenario.classical_variables,
    target_name="residual",
    data_statistics={},
    n_candidates=50,
    iteration=0,
    domain=scenario.domain,
    classical_expr=scenario.classical_expr,
    variables_with_units=scenario.variables_with_units,
    known_limits=[{"variable": scenario.classical_limit_variable, "limit": scenario.classical_limit_direction, "expected": "0"}],
    classical_limit_condition=f"{scenario.classical_limit_variable} -> {scenario.classical_limit_direction}",
    constants=scenario.classical_constants
)

proposed_candidates = proposer.propose(context)
print(f"Proposed {len(proposed_candidates)} unique candidates.")

# Screen
subbed_candidates = []
orig_by_subbed = {}
for cand in proposed_candidates:
    # Substitute thetas
    try:
        expr = sp.sympify(cand, locals=pipeline.locals)
        subs_dict = {s: 1.0 for s in expr.free_symbols if str(s).startswith("theta_")}
        sub_expr = str(expr.subs(subs_dict)) if subs_dict else cand
    except Exception:
        sub_expr = cand
    has_params = (sub_expr != cand)
    subbed_candidates.append((sub_expr, has_params))
    if sub_expr not in orig_by_subbed:
        orig_by_subbed[sub_expr] = []
    orig_by_subbed[sub_expr].append(cand)

stage1_results = pipeline.execute(
    subbed_candidates,
    None,
    X,
    residual,
    constants=scenario.classical_constants
)

print(f"{len(stage1_results)} candidates survived Stage 1.")

# Optimize
subbed_top = []
for sub_expr, combined_score, mse, arc_score in stage1_results[:15]:
    cand = orig_by_subbed[sub_expr][0]
    subbed_top.append((cand, combined_score, mse, arc_score))

stage2_results = optimizer.optimize_batch(
    subbed_top,
    X,
    residual,
    scenario.classical_variables
)

for expr_str, stage2_combined, opt_nmse, arc_score, opt_result in stage2_results:
    n_params = len([k for k in opt_result.theta.keys() if k.startswith("theta_")])
    n_points = len(residual)
    b_score = bic_score(opt_nmse, n_params, n_points)
    print(f"Cand: {expr_str} | Params: {n_params} | NMSE: {opt_nmse:.2e} | BIC: {b_score:.2f} | Theta: {opt_result.theta}")
