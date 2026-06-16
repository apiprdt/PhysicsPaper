"""
ADCD Blind Generalization Benchmark — 3-Column Comparison
Tests: Mock (base) | Mock (extended) | Grammar (standalone)
on all blind scenarios across 4 noise levels.
"""

import os
import sys
import json
import time
import numpy as np
import sympy as sp
from typing import List

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.llm_proposer import CorrectionMockProposer
from adcd.grammar_proposer import GrammarProposer
from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.arc_scorer import ARCScorer, build_arc_regimes
from adcd.pipeline import Stage1Pipeline
from adcd.jax_optimizer import JAXOptimizer
from adcd.correction_orchestrator import CorrectionOrchestrator
from adcd.metrics import evaluate_correction

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]
SEED = 42

def run_single(scenario, proposer, noise, proposer_name):
    """Run discovery for one scenario+proposer+noise combination."""
    # Generate data
    X, y_obs, y_classical, residual = scenario.generate_data(n_points=200, noise_level=noise, seed=SEED)
    
    # Setup components
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
    
    t0 = time.time()
    result = orchestrator.search_correction(scenario, noise_level=noise, seed=SEED)
    elapsed = time.time() - t0
    
    eval_res = evaluate_correction(
        result.best_expr,
        scenario,
        X,
        y_obs,
        y_classical,
        result.best_theta
    )
    
    success = eval_res.class_match and eval_res.nmse_residual < 0.1
    
    return {
        "scenario": scenario.name,
        "proposer": proposer_name,
        "noise": noise,
        "discovered_expr": result.best_expr,
        "nmse_residual": eval_res.nmse_residual,
        "class_match": eval_res.class_match,
        "success": success,
        "n_proposed": sum(h.n_proposed for h in result.history),
        "n_survived_gates": sum(h.n_survived_stage1 for h in result.history),
        "wall_seconds": elapsed,
    }

def main():
    all_scenarios = get_all_scenarios()
    # Blind scenarios 1 to 9 (single variable)
    blind_scenarios = [s for s in all_scenarios if s.tier == "blind" and not isinstance(s.classical_limit_variable, list)]
    
    print("=" * 90)
    print("     ADCD BLIND GENERALIZATION BENCHMARK: 3-COLUMN COMPARISON")
    print("=" * 90)
    print(f"Testing {len(blind_scenarios)} single-variable blind scenarios across 4 noise levels.")
    for idx, sc in enumerate(blind_scenarios, 1):
        print(f"  {idx}. {sc.name} (domain: {sc.domain}, limit: {sc.classical_limit_variable} -> {sc.classical_limit_direction})")
    print("\nStarting execution...")

    proposers_factory = {
        "Mock (base)":     lambda: CorrectionMockProposer(seed=SEED),
        "Mock (extended)": lambda: CorrectionMockProposer(seed=SEED, extended=True),
        "Grammar":         lambda: GrammarProposer(checker=DimensionalChecker(), seed=SEED),
    }
    
    results = []
    total_runs = len(blind_scenarios) * len(proposers_factory) * len(NOISE_LEVELS)
    run_idx = 0
    
    for scenario in blind_scenarios:
        for p_name, p_factory in proposers_factory.items():
            for noise in NOISE_LEVELS:
                run_idx += 1
                proposer = p_factory()
                
                res = run_single(scenario, proposer, noise, p_name)
                results.append(res)
                
                status = "SUCCESS" if res["success"] else "FAILED"
                print(f"[{run_idx}/{total_runs}] {status:7} | {p_name:15} | {scenario.name:32} | "
                      f"noise={noise:.0%} | NMSE={res['nmse_residual']:.3e} | "
                      f"expr={res['discovered_expr'][:40]}")
    
    # Save results to JSON
    with open("blind_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to blind_benchmark_results.json")
    
    # Output the multi-column report
    print("\n" + "=" * 90)
    print("SUMMARY: Success Rate per Proposer across Noise Levels")
    print("=" * 90)
    print(f"{'Proposer':<18} | {'0% noise':<10} | {'1% noise':<10} | {'5% noise':<10} | {'10% noise':<10}")
    print("-" * 90)
    
    for p_name in proposers_factory:
        row_str = f"{p_name:<18}"
        for noise in NOISE_LEVELS:
            subset = [r for r in results if r["proposer"] == p_name and r["noise"] == noise]
            successes = sum(1 for r in subset if r["success"])
            total = len(subset)
            rate = (successes / total * 100) if total > 0 else 0.0
            row_str += f" | {successes}/{total} ({rate:.0f}%)"
        print(row_str)
    print("=" * 90)

if __name__ == "__main__":
    main()
