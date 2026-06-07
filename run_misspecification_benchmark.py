import json
import logging
from typing import List, Dict, Any

from adcd.anomaly_scenarios import AnomalyScenario
from run_correction_discovery import run_scenario_benchmark

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("MisspecificationBenchmark")

def get_extreme_scenarios() -> List[AnomalyScenario]:
    """Returns scenarios for extreme misspecification testing matching anomaly_scenarios.py exact names."""
    return [
        # Correct baseline
        AnomalyScenario(
            name="Relativistic KE",
            tier="cross_domain",
            domain="mechanics",
            classical_expr="0.5 * m * v**2",
            classical_variables=["m", "v"],
            classical_constants={"c": 3.0e8},
            correction_type="multiplicative",
            correction_expr="theta_0 * (v / c)**2",
            correction_constants={"theta_0": 0.75},
            anomaly_regime="high speed v approaching c",
            variables_with_units={"m": "kg", "v": "m/s", "c": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="polynomial"
        ),
        # Case 1: Wrong Functional Form (Dimensional Mismatch in Baseline)
        AnomalyScenario(
            name="Misspecification 1: Wrong Baseline Form",
            tier="synthetic",
            domain="mechanics",
            classical_expr="m * v",  # Wrong! Should be 0.5 * m * v**2
            classical_variables=["m", "v"],
            classical_constants={},
            correction_type="additive",
            correction_expr="0.5 * m * v**2 + theta_0 * v**4",
            correction_constants={"theta_0": 0.1},
            anomaly_regime="high speed v",
            variables_with_units={"m": "kg", "v": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="polynomial"
        ),
        # Case 2: Missing Vital Variable (Omitted Variable Bias)
        AnomalyScenario(
            name="Misspecification 2: Missing Variable",
            tier="synthetic",
            domain="fluid dynamics",
            classical_expr="m * g",
            classical_variables=["m", "g"], # User forgets to pass 'v' (velocity)
            classical_constants={},
            correction_type="additive",
            correction_expr="theta_0 * v**2",
            correction_constants={"theta_0": 0.25},
            anomaly_regime="high speed flow",
            variables_with_units={"m": "kg", "g": "m/s^2", "v": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="polynomial"
        ),
        # Case 3: Spurious Irrelevant Variable (Over-specification)
        AnomalyScenario(
            name="Misspecification 3: Spurious Variable",
            tier="synthetic",
            domain="mechanics",
            classical_expr="0.5 * k * x**2",
            classical_variables=["k", "x", "T"], # T is completely irrelevant
            classical_constants={},
            correction_type="additive",
            correction_expr="theta_0 * x**4",
            correction_constants={"theta_0": 0.15},
            anomaly_regime="large displacement",
            variables_with_units={"k": "N/m", "x": "m", "T": "K"},
            classical_limit_variable="x",
            classical_limit_direction="0",
            correction_class="polynomial"
        ),
    ]

def get_subtle_scenarios() -> List[AnomalyScenario]:
    """Returns scenarios for subtle misspecification testing (Nonlinear Drag baseline)."""
    return [
        # Correct baseline
        AnomalyScenario(
            name="Nonlinear Drag (Correct Baseline)",
            tier="cross_domain",
            domain="mechanics",
            classical_expr="b * v",
            classical_variables=["b", "v"],
            classical_constants={},
            correction_type="additive",
            correction_expr="theta_0 * v**2",
            correction_constants={"theta_0": 0.25},
            anomaly_regime="high speed v",
            variables_with_units={"b": "N*s/m", "v": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="polynomial"
        ),
        # Case D: Constant 5% off
        AnomalyScenario(
            name="Nonlinear Drag Misspecification D: Constant 5% Off",
            tier="synthetic",
            domain="mechanics",
            classical_expr="1.05 * b * v",
            classical_variables=["b", "v"],
            classical_constants={},
            correction_type="additive",
            correction_expr="-0.05 * b * v + theta_0 * v**2",
            correction_constants={"theta_0": 0.25},
            anomaly_regime="high speed v",
            variables_with_units={"b": "N*s/m", "v": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="polynomial"
        ),
        # Case E: Constant Offset
        AnomalyScenario(
            name="Nonlinear Drag Misspecification E: Constant Offset",
            tier="synthetic",
            domain="mechanics",
            classical_expr="b * v + 0.1",
            classical_variables=["b", "v"],
            classical_constants={},
            correction_type="additive",
            correction_expr="-0.1 + theta_0 * v**2",
            correction_constants={"theta_0": 0.25},
            anomaly_regime="high speed v",
            variables_with_units={"b": "N*s/m", "v": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="polynomial"
        ),
        # Case F: Wrong Structural Coefficient (20% off)
        AnomalyScenario(
            name="Nonlinear Drag Misspecification F: Wrong Structural Coefficient",
            tier="synthetic",
            domain="mechanics",
            classical_expr="0.8 * b * v",
            classical_variables=["b", "v"],
            classical_constants={},
            correction_type="additive",
            correction_expr="0.2 * b * v + theta_0 * v**2",
            correction_constants={"theta_0": 0.25},
            anomaly_regime="high speed v",
            variables_with_units={"b": "N*s/m", "v": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="polynomial"
        ),
    ]

def run_benchmark():
    print("=" * 80)
    print("   RUNNING ADCD SYSTEM MISSPECIFICATION BENCHMARK STUDY")
    print("=" * 80)
    
    all_results = []
    
    # Tier 1: Extreme Misspecifications (Relativistic KE baseline, 1% noise)
    print("\n--- TIER 1: EXTREME MISSPECIFICATION TESTS (Noise = 1.0%) ---")
    extreme_scens = get_extreme_scenarios()
    extreme_results = []
    for scenario in extreme_scens:
        # Relativistic KE requires max_iter=6 for full convergence
        res = run_scenario_benchmark(scenario, noise_level=0.01, max_iter=6, proposer_type="mock", seed=42)
        extreme_results.append(res)
        all_results.append(res)
        print(f"Scenario: {res['scenario']:<45} | ClassMatch: {str(res['class_match']):<5} | Full NMSE: {res['nmse_full']:.2e} | BIC: {res['bic']:.2f}")

    # Tier 2: Subtle Misspecifications (Nonlinear Drag baseline, 5% noise)
    print("\n--- TIER 2: SUBTLE MISSPECIFICATION TESTS (Noise = 5.0%) ---")
    subtle_scens = get_subtle_scenarios()
    subtle_results = []
    for scenario in subtle_scens:
        res = run_scenario_benchmark(scenario, noise_level=0.05, max_iter=6, proposer_type="mock", seed=42)
        subtle_results.append(res)
        all_results.append(res)
        print(f"Scenario: {res['scenario']:<45} | ClassMatch: {str(res['class_match']):<5} | Full NMSE: {res['nmse_full']:.2e} | BIC: {res['bic']:.2f}")

    # Save all results
    with open("misspecification_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved results to: misspecification_results.json ({len(all_results)} entries)")

    # Print consolidated Markdown comparison tables for latex transfer
    print("\n" + "=" * 80)
    print("CONSOLIDATED COMPARISON TABLES FOR LATEX DRAFT")
    print("=" * 80)
    
    print("\n### Extreme Misspecification Table (Noise = 1.0%)")
    print(f"| {'Scenario Name':<45} | {'Converged':<10} | {'Full NMSE':<12} | {'BIC':<10} | {'Class Match':<12} |")
    print("|" + "-"*47 + "|" + "-"*12 + "|" + "-"*14 + "|" + "-"*12 + "|" + "-"*14 + "|")
    for r in extreme_results:
        print(f"| {r['scenario']:<45} | {str(r['converged']):<10} | {r['nmse_full']:.4e} | {r['bic']:<10.2f} | {str(r['class_match']):<12} |")
        
    print("\n### Subtle Misspecification Table (Noise = 5.0%)")
    print(f"| {'Scenario Name':<45} | {'Converged':<10} | {'Full NMSE':<12} | {'BIC':<10} | {'Class Match':<12} |")
    print("|" + "-"*47 + "|" + "-"*12 + "|" + "-"*14 + "|" + "-"*12 + "|" + "-"*14 + "|")
    for r in subtle_results:
        print(f"| {r['scenario']:<45} | {str(r['converged']):<10} | {r['nmse_full']:.4e} | {r['bic']:<10.2f} | {str(r['class_match']):<12} |")
    print("=" * 80)

if __name__ == "__main__":
    run_benchmark()
