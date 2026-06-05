import json
import logging
from typing import List

from src.adcd.anomaly_scenarios import AnomalyScenario
from run_correction_discovery import run_scenario_benchmark

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_misspecification_scenarios() -> List[AnomalyScenario]:
    from src.adcd.anomaly_scenarios import get_all_scenarios
    all_scens = get_all_scenarios()
    baseline_ke = next(s for s in all_scens if s.name == "Relativistic KE")
    baseline_ke.name = "Baseline Truth: Relativistic KE"
    
    return [
        baseline_ke,
        # Case 1: Wrong Functional Form (Dimensional Mismatch in Baseline)
        # Ground truth: y_obs is Kinetic Energy in Joules.
        # User provides Momentum (m*v) as the baseline.
        AnomalyScenario(
            name="Misspecification 1: Wrong Baseline Form",
            tier="synthetic",
            domain="mechanics",
            classical_expr="m * v",  # Wrong! Should be 0.5 * m * v**2
            classical_variables=["m", "v"],
            classical_constants={},
            correction_type="additive",
            correction_expr="0.5 * m * v**2 + theta_0 * v**4", # The "true" data generator will use this
            correction_constants={"theta_0": 0.1},
            anomaly_regime="high speed v",
            variables_with_units={"m": "kg", "v": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="polynomial"
        ),
        
        # Case 2: Missing Vital Variable (Omitted Variable Bias)
        # Ground truth involves velocity 'v', but user doesn't provide 'v' to the pipeline.
        AnomalyScenario(
            name="Misspecification 2: Missing Variable",
            tier="synthetic",
            domain="fluid dynamics",
            classical_expr="m * g",  # Gravity
            classical_variables=["m", "g"], # User forgets to pass 'v' (velocity)
            classical_constants={},
            correction_type="additive",
            correction_expr="theta_0 * v**2",  # Drag force depends on v
            # Since 'v' is not in classical_variables, we need to hack the generator to provide it 
            # internally for y_true, but hide it from the orchestrator.
            # We'll just define it with v in variables but hide it during benchmark run.
            correction_constants={"theta_0": 0.25},
            anomaly_regime="high speed flow",
            variables_with_units={"m": "kg", "g": "m/s^2", "v": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="polynomial"
        ),
        
        # Case 3: Spurious Irrelevant Variable (Over-specification)
        # User provides an extra random variable 'T' that doesn't affect the physics.
        AnomalyScenario(
            name="Misspecification 3: Spurious Variable",
            tier="synthetic",
            domain="mechanics",
            classical_expr="0.5 * k * x**2",
            classical_variables=["k", "x", "T"], # T is completely irrelevant
            classical_constants={},
            correction_type="additive",
            correction_expr="theta_0 * x**4", # Anharmonic spring
            correction_constants={"theta_0": 0.15},
            anomaly_regime="large displacement",
            variables_with_units={"k": "N/m", "x": "m", "T": "K"},
            classical_limit_variable="x",
            classical_limit_direction="0",
            correction_class="polynomial"
        ),
        
        # Case 4 (Subtle D): Constant off by 5%
        AnomalyScenario(
            name="Subtle Misspecification D: Constant 5% Off",
            tier="synthetic",
            domain="mechanics",
            classical_expr="0.525 * m * v**2",
            classical_variables=["m", "v"],
            classical_constants={"c": 3.0e8},
            correction_type="additive",
            correction_expr="-0.025 * m * v**2 + theta_0 * (v/c)**2",
            correction_constants={"theta_0": 0.75},
            anomaly_regime="high speeds v approaching c",
            variables_with_units={"m": "kg", "v": "m/s", "c": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="polynomial"
        ),
        
        # Case 5 (Subtle E): Constant offset
        AnomalyScenario(
            name="Subtle Misspecification E: Constant Offset",
            tier="synthetic",
            domain="mechanics",
            classical_expr="0.5 * m * v**2 + 0.1",
            classical_variables=["m", "v"],
            classical_constants={"c": 3.0e8},
            correction_type="additive",
            correction_expr="-0.1 + theta_0 * (v/c)**2",
            correction_constants={"theta_0": 0.75},
            anomaly_regime="high speeds v approaching c",
            variables_with_units={"m": "kg", "v": "m/s", "c": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="polynomial"
        ),
        
        # Case 6 (Subtle F): Wrong structural coefficient
        AnomalyScenario(
            name="Subtle Misspecification F: Wrong Structural Coefficient",
            tier="synthetic",
            domain="mechanics",
            classical_expr="0.4 * m * v**2",
            classical_variables=["m", "v"],
            classical_constants={"c": 3.0e8},
            correction_type="additive",
            correction_expr="0.1 * m * v**2 + theta_0 * (v/c)**2",
            correction_constants={"theta_0": 0.75},
            anomaly_regime="high speeds v approaching c",
            variables_with_units={"m": "kg", "v": "m/s", "c": "m/s"},
            classical_limit_variable="v",
            classical_limit_direction="0",
            correction_class="polynomial"
        )
    ]

def run_misspecifications():
    scenarios = get_misspecification_scenarios()
    results = []
    
    for scenario in scenarios:
        logger.info(f"=== Running Misspecification Test: {scenario.name} ===")
        # We use MockProposer for deterministic, fast testing of the pipeline's rejection capabilities
        result = run_scenario_benchmark(scenario, noise_level=0.01, max_iter=3, proposer_type="mock")
        results.append(result)
        
    with open("misspecification_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
if __name__ == "__main__":
    run_misspecifications()
