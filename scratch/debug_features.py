import numpy as np
from src.anomaly_scenarios import get_all_scenarios
from src.residual_analyzer import analyze_residual

scenarios = get_all_scenarios()
for scenario in scenarios:
    X, y_obs, y_classical, residual = scenario.generate_data(noise_level=0.0, seed=42)
    primary_var_name = scenario.classical_limit_variable
    res_feat = analyze_residual(X[primary_var_name], residual)
    print(f"Scenario: {scenario.name}")
    print(f"  Monotonicity: {res_feat.monotonicity:.3f}")
    print(f"  Curvature Sign: {res_feat.curvature_sign:.1f}")
    print(f"  Oscillation Score: {res_feat.oscillation_score:.3f}")
    print(f"  Decay Rate: {res_feat.decay_rate:.3f}")
    print(f"  Symmetry: {res_feat.symmetry:.3f}")
    
    # Calculate weights
    weights = {
        "power_law": 1.0,
        "polynomial": 1.0,
        "exponential": 1.0,
        "rational": 1.0,
        "trigonometric": 1.0,
        "logarithmic": 1.0
    }
    rf = res_feat
    if rf.decay_rate > 0.4:
        weights["exponential"] += 8.0
        weights["rational"] += 4.0
    if abs(rf.monotonicity) > 0.7:
        if rf.decay_rate < 0.2:
            weights["logarithmic"] += 6.0
            weights["power_law"] += 4.0
            weights["rational"] += 2.0
    if rf.oscillation_score > 0.15:
        weights["trigonometric"] += 10.0
    if rf.oscillation_score < 0.05:
        if abs(rf.symmetry) > 0.1:
            if rf.symmetry > 0:
                weights["polynomial"] += 8.0
            else:
                weights["polynomial"] += 4.0
        else:
            weights["polynomial"] += 4.0
            
    print(f"  Weights: {weights}\n")
