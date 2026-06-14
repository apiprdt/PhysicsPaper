import os
import sys
import time
import numpy as np
import sympy as sp
from typing import List

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.llm_proposer import CorrectionMockProposer, HybridCorrectionProposer
from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.arc_scorer import ARCScorer, build_arc_regimes
from adcd.pipeline import Stage1Pipeline
from adcd.jax_optimizer import JAXOptimizer
from adcd.correction_orchestrator import CorrectionOrchestrator
from adcd.metrics import evaluate_correction

def run_benchmark():
    print("="*60)
    print("ADCD PHASE 1 BENCHMARK: BLIND SCENARIOS EVALUATION")
    print("="*60)
    
    # 1. Retrieve all blind scenarios
    all_scenarios = get_all_scenarios()
    blind_scenarios = [s for s in all_scenarios if s.tier == "blind"]
    print(f"Loaded {len(blind_scenarios)} blind scenarios.")
    for idx, sc in enumerate(blind_scenarios, 1):
        print(f"  {idx}. {sc.name} (domain: {sc.domain}, target class: {sc.correction_class})")
    
    print("\nStarting runs...")
    
    results = []
    
    # 2. Iterate and evaluate each scenario under baseline (mock) and grammar (hybrid)
    for idx, scenario in enumerate(blind_scenarios, 1):
        print("\n" + "-"*50)
        print(f"Scenario {idx}/{len(blind_scenarios)}: {scenario.name}")
        print("-"*50)
        
        # Generate data (5% noise as in ablation study)
        X, y_obs, y_classical, residual = scenario.generate_data(n_points=200, noise_level=0.05, seed=42)
        
        # We run both proposers
        proposers = {
            "Baseline (Mock)": CorrectionMockProposer(seed=42),
            "New (Grammar)": HybridCorrectionProposer(api_key="", seed=42)  # api_key="" forces pure grammar/mock fallback
        }
        
        scenario_results = {"name": scenario.name, "target_class": scenario.correction_class}
        
        for name, proposer in proposers.items():
            print(f"Running {name}...")
            
            # Setup fresh pipeline components for each run
            validator = ASTValidator()
            checker = DimensionalChecker()
            regimes = build_arc_regimes(scenario.classical_limit_variable, scenario.classical_limit_direction)
            scorer = ARCScorer(regimes=regimes)
            pipeline = Stage1Pipeline(validator, checker, scorer)
            optimizer = JAXOptimizer()
            
            orchestrator = CorrectionOrchestrator(
                proposer=proposer,
                pipeline=pipeline,
                optimizer=optimizer,
                max_iterations=5,
                verbose=False
            )
            
            start_time = time.time()
            res = orchestrator.search_correction(scenario, noise_level=0.05)
            elapsed = time.time() - start_time
            
            # Evaluate discovered correction
            eval_res = evaluate_correction(
                res.best_expr,
                scenario,
                X,
                y_obs,
                y_classical,
                res.best_theta
            )
            
            # Determine success: class match and low residual NMSE
            success = eval_res.class_match and eval_res.nmse_residual < 0.1
            
            print(f"  Result: {res.best_expr}")
            print(f"  Class Match: {eval_res.class_match} (got: {eval_res.discovered_class})")
            print(f"  NMSE Residual: {eval_res.nmse_residual:.6e}")
            print(f"  Success: {success} | Time: {elapsed:.2f}s")
            
            scenario_results[name] = {
                "expr": res.best_expr,
                "nmse": eval_res.nmse_residual,
                "class_match": eval_res.class_match,
                "success": success,
                "time": elapsed
            }
            
        results.append(scenario_results)

    # 3. Print final summary table
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY TABLE")
    print("="*80)
    print(f"{'Scenario Name':<30} | {'Baseline Success':<18} | {'Grammar Success':<18}")
    print("-"*80)
    
    baseline_successes = 0
    grammar_successes = 0
    
    for r in results:
        b_succ = r["Baseline (Mock)"]["success"]
        g_succ = r["New (Grammar)"]["success"]
        
        if b_succ: baseline_successes += 1
        if g_succ: grammar_successes += 1
        
        b_status = "SUCCESS" if b_succ else "FAILED"
        g_status = "SUCCESS" if g_succ else "FAILED"
        
        print(f"{r['name']:<30} | {b_status:<18} | {g_status:<18}")
        
    print("="*80)
    baseline_rate = (baseline_successes / len(blind_scenarios)) * 100
    grammar_rate = (grammar_successes / len(blind_scenarios)) * 100
    print(f"Baseline (Mock) Success Rate: {baseline_successes}/{len(blind_scenarios)} ({baseline_rate:.1f}%)")
    print(f"Grammar-Guided Success Rate:  {grammar_successes}/{len(blind_scenarios)} ({grammar_rate:.1f}%)")
    print(f"Absolute Improvement:         +{grammar_rate - baseline_rate:.1f}%")
    print("="*80)

if __name__ == "__main__":
    run_benchmark()
