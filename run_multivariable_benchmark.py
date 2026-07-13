"""
ADCD Multivariable Generalization Benchmark — ProductGrammar vs Mock
Tests: Mock (extended) | ProductGrammar
on all 4 multivariable scenarios across 4 noise levels.
"""

import os
import sys
import json
import time
from pathlib import Path
import numpy as np
import sympy as sp

# Add src folder to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from adcd.anomaly_scenarios import get_mv_scenarios
from adcd.llm_proposer import CorrectionMockProposer
from adcd.multivar_orchestrator import run_adcd_mv
from adcd.metrics import evaluate_correction

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]
SEED = 42

def main():
    scenarios = get_mv_scenarios()
    
    print("=" * 90)
    print("     ADCD MULTIVARIABLE GENERALIZATION BENCHMARK: Grammar vs Mock")
    print("=" * 90)
    print(f"Testing {len(scenarios)} multivariable scenarios across 4 noise levels.")
    for idx, sc in enumerate(scenarios, 1):
        print(f"  {idx}. {sc.name} (limit: {sc.classical_limit_variable} -> {sc.classical_limit_direction})")
    print("\nStarting execution...")
    
    results = []
    
    for scenario in scenarios:
        for proposer_name in ["Mock (extended)", "ProductGrammar"]:
            for noise in NOISE_LEVELS:
                print(f"Running {scenario.name} | {proposer_name} | Noise {noise*100:.0f}%...", end="", flush=True)
                t0 = time.time()
                
                if proposer_name == "ProductGrammar":
                    res = run_adcd_mv(scenario, noise=noise, seed=SEED, max_iterations=5)
                else:
                    from adcd.correction_orchestrator import CorrectionOrchestrator
                    from adcd.dimensional_checker import ASTValidator, DimensionalChecker
                    from adcd.arc_scorer import ARCScorer
                    from adcd.pipeline import Stage1Pipeline
                    from adcd.jax_optimizer import JAXOptimizer
                    
                    validator = ASTValidator()
                    checker = DimensionalChecker()
                    for var in scenario.classical_variables:
                        checker.registry[var] = [0, 0, 0]
                    for const in scenario.classical_constants:
                        checker.registry[const] = [0, 0, 0]
                    
                    from adcd.arc_scorer import build_arc_regimes
                    limit_var = scenario.classical_limit_variable.split(",")[0].strip()
                    limit_dir = scenario.classical_limit_direction.split(",")[0].strip()
                    regimes = build_arc_regimes(limit_var, limit_dir)
                    scorer = ARCScorer(regimes=regimes)
                    
                    pipeline = Stage1Pipeline(validator, checker, scorer)
                    optimizer = JAXOptimizer()  # Reverted to default linear scale optimization
                    proposer = CorrectionMockProposer(seed=SEED, extended=True)
                    
                    orchestrator = CorrectionOrchestrator(
                        proposer=proposer,
                        pipeline=pipeline,
                        optimizer=optimizer,
                        max_iterations=5,
                        verbose=False
                    )
                    orchestrator._register_scenario_symbols(scenario)
                    res = orchestrator.search_correction(scenario, noise_level=noise, seed=SEED)
                
                elapsed = time.time() - t0
                
                X, y_obs, y_classical, residual = scenario.generate_data(noise_level=noise, seed=SEED)
                eval_res = evaluate_correction(
                    res.best_expr,
                    scenario,
                    X,
                    y_obs,
                    y_classical,
                    res.best_theta
                )
                
                success = eval_res.class_match and eval_res.nmse_residual < 0.1
                print(f" Success: {success} (Class Match: {eval_res.class_match}, NMSE: {eval_res.nmse_residual:.4f}) in {elapsed:.1f}s")
                
                results.append({
                    "scenario": scenario.name,
                    "proposer": proposer_name,
                    "noise": noise,
                    "discovered_expr": res.best_expr,
                    "nmse_residual": eval_res.nmse_residual,
                    "class_match": eval_res.class_match,
                    "success": success,
                    "wall_seconds": elapsed,
                })
                
    with open("multivariable_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
        
    print("\n" + "=" * 90)
    print("                      BENCHMARK SUMMARY TABLE")
    print("=" * 90)
    print(f"{'Scenario':<25} | {'Proposer':<18} | {'0%':<8} | {'1%':<8} | {'5%':<8} | {'10%':<8}")
    print("-" * 90)
    
    for scenario in scenarios:
        for proposer in ["Mock (extended)", "ProductGrammar"]:
            row = [f"{scenario.name:<25}", f"{proposer:<18}"]
            for noise in NOISE_LEVELS:
                match = [r for r in results if r["scenario"] == scenario.name and r["proposer"] == proposer and r["noise"] == noise]
                if match and match[0]["success"]:
                    row.append("PASS    ")
                else:
                    row.append("FAIL    ")
            print(" | ".join(row))
    print("=" * 90)

if __name__ == "__main__":
    main()
