import sympy as sp
from adcd.anomaly_scenarios import get_all_scenarios
from adcd.arc_scorer import ARCScorer, AsymptoticRegime
from adcd.pipeline import Stage1Pipeline
from adcd.dimensional_checker import ASTValidator, DimensionalChecker

scenarios = get_all_scenarios()
yukawa = [s for s in scenarios if s.name == "Yukawa Gravity"][0]

limit_var = sp.Symbol(yukawa.classical_limit_variable)
limit_target = sp.oo if yukawa.classical_limit_direction == "oo" else 0
regimes = [
    AsymptoticRegime(
        variable=limit_var,
        limit_target=limit_target,
        ground_truth_expr="0",
        weight=1.0
    )
]
scorer = ARCScorer(regimes=regimes)

# Test candidate
cand1 = "theta_0 * exp(-r / theta_1)"
cand2 = "theta_0 * (r / theta_1)**theta_2"

subbed_cand1 = cand1.replace("theta_0", "1.0").replace("theta_1", "1.0").replace("theta_2", "1.0")
subbed_cand2 = cand2.replace("theta_0", "1.0").replace("theta_1", "1.0").replace("theta_2", "1.0")

score1 = scorer.score(subbed_cand1, constants=yukawa.classical_constants)
score2 = scorer.score(subbed_cand2, constants=yukawa.classical_constants)

print(f"Candidate 1: {cand1} -> subbed: {subbed_cand1} -> ARC score: {score1}")
print(f"Candidate 2: {cand2} -> subbed: {subbed_cand2} -> ARC score: {score2}")
