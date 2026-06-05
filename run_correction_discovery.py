import os
import argparse
import time
import logging
import json
import numpy as np
import sympy as sp
from typing import Dict, List, Any

from adcd.anomaly_scenarios import get_all_scenarios, AnomalyScenario
from adcd.metrics import CorrectionEvaluation, classify_structure
from adcd.llm_proposer import CorrectionMockProposer
from adcd.correction_orchestrator import CorrectionOrchestrator
from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.arc_scorer import ARCScorer, AsymptoticRegime
from adcd.pipeline import Stage1Pipeline
from adcd.jax_optimizer import JAXOptimizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CorrectionDiscoveryBenchmark")

def run_scenario_benchmark(scenario: AnomalyScenario, noise_level: float, max_iter: int = 4, proposer_type: str = "mock", seed: int = 42) -> Dict[str, Any]:
    """Runs a single scenario under the specified noise level and returns metrics."""
    logger.info(f"Starting discovery: Scenario='{scenario.name}', Noise={noise_level * 100:.1f}%, Proposer={proposer_type}, Seed={seed}")
    
    # 1. Setup dynamic physical limit regime (ARC Scorer)
    validator = ASTValidator()  # defaults: max_depth=7, max_tokens=20
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
    
    if proposer_type == "mock":
        proposer = CorrectionMockProposer(seed=seed)
    elif proposer_type == "gemini":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for Gemini proposer.")
        from adcd.llm_proposer import CorrectionGeminiProposer
        proposer = CorrectionGeminiProposer(api_key=api_key)
    elif proposer_type == "hybrid":
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is required for Hybrid proposer.")
        from adcd.llm_proposer import HybridCorrectionProposer
        proposer = HybridCorrectionProposer(api_key=api_key, seed=42)
    else:
        raise ValueError(f"Unknown proposer type: {proposer_type}")
    
    orchestrator = CorrectionOrchestrator(
        proposer=proposer,
        pipeline=pipeline,
        optimizer=optimizer,
        max_iterations=max_iter,
        top_k=8,
        convergence_nmse=1e-5,
        verbose=False
    )
    
    start_time = time.time()
    search_res = orchestrator.search_correction(scenario, noise_level=noise_level, seed=seed)
    elapsed = time.time() - start_time
    
    eval_res = search_res.evaluation
    if eval_res is None:
        eval_res = CorrectionEvaluation(
            nmse_residual=1.0,
            nmse_full=1.0,
            true_class=scenario.correction_class,
            discovered_class="failed",
            class_match=False,
            ast_edit_distance=999,
            parameter_error={},
            bic=9999.0
        )

        
    return {
        "scenario": scenario.name,
        "tier": scenario.tier,
        "noise": noise_level,
        "discovered_expr": search_res.best_expr,
        "nmse_residual": eval_res.nmse_residual,
        "nmse_full": eval_res.nmse_full,
        "discovered_class": eval_res.discovered_class,
        "class_match": eval_res.class_match,
        "ast_edit_distance": eval_res.ast_edit_distance,
        "parameter_error": eval_res.parameter_error,
        "bic": eval_res.bic,
        "time_seconds": elapsed,
        "converged": search_res.converged
    }

def main():
    parser = argparse.ArgumentParser(description="Run ADCD Anomaly-Driven Correction Discovery benchmark.")
    parser.add_argument("--proposer", type=str, default="mock", choices=["mock", "gemini", "hybrid"],
                        help="The type of proposer to use: mock, gemini, or hybrid.")
    args = parser.parse_args()
    
    print("======================================================================")
    print(f"      STARTING ADCD BENCHMARK: {args.proposer.upper()} PROPOSER      ")
    print("======================================================================")
    
    scenarios = get_all_scenarios()
    noise_levels = [0.0, 0.01, 0.05, 0.10]
    
    results = []
    
    for scenario in scenarios:
        for noise in noise_levels:
            # We run textbook and simple ones with 4 iterations, synthetics with 4 iterations for speed
            run_res = run_scenario_benchmark(scenario, noise, max_iter=4, proposer_type=args.proposer)
            results.append(run_res)
            
            # Print brief progress
            status_str = "SUCCESS" if run_res["class_match"] else "PARTIAL" if run_res["nmse_full"] < 1e-2 else "FAILED"
            print(f"[{status_str}] {scenario.name} (Noise={noise*100}%): "
                  f"Discovered: {run_res['discovered_expr']} | Full NMSE={run_res['nmse_full']:.2e} | "
                  f"ClassMatch={run_res['class_match']}")
            
    # Save raw results
    output_path = f"scratch_correction_results_{args.proposer}.json"
    if args.proposer == "mock":
        output_path = "scratch_correction_results.json"
        
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved raw benchmark results to {output_path}")
    
    # ── GENERATE PREMIUM MARKDOWN REPORT ──
    print("\n\n======================================================================")
    print("                     BENCHMARK RESULTS SUMMARY                        ")
    print("======================================================================")
    
    # Group by Tier
    for tier in ["textbook", "cross_domain", "synthetic"]:
        print(f"\n### Tier: {tier.upper()} SCENARIOS\n")
        print("| Scenario | Noise Level | Discovered Correction (Delta) | NMSE (Full) | Class Match | AST Dist | Parameter Error |")
        print("|---|---|---|---|---|---|---|")
        
        tier_results = [r for r in results if r["tier"] == tier]
        for r in tier_results:
            param_err_str = "N/A"
            if r["parameter_error"]:
                errs = [f"{k}: {v*100:.1f}%" for k, v in r["parameter_error"].items()]
                param_err_str = ", ".join(errs)
                
            class_match_str = "YES" if r["class_match"] else "NO"
            
            print(f"| {r['scenario']} | {r['noise']*100:.0f}% | `{r['discovered_expr']}` | {r['nmse_full']:.2e} | {class_match_str} | {r['ast_edit_distance']} | {param_err_str} |")

    # Compute overall performance targets
    print("\n### METRICS TARGET EVALUATION")
    
    # Class match rates by tier
    for tier in ["textbook", "cross_domain", "synthetic"]:
        tier_r_0 = [r for r in results if r["tier"] == tier and r["noise"] == 0.0]
        match_0 = sum(1 for r in tier_r_0 if r["class_match"])
        rate_0 = match_0 / len(tier_r_0) if tier_r_0 else 0
        print(f"- {tier.capitalize()} Tier (0% Noise) Class Match: {match_0}/{len(tier_r_0)} ({rate_0 * 100:.1f}%)")
        
    # Overall noise robustness (5% noise class match rate)
    noise_5_r = [r for r in results if r["noise"] == 0.05]
    match_5 = sum(1 for r in noise_5_r if r["class_match"])
    rate_5 = match_5 / len(noise_5_r) if noise_5_r else 0
    print(f"- Noise Robustness (5% Noise) overall Class Match: {match_5}/{len(noise_5_r)} ({rate_5 * 100:.1f}%)")
    
if __name__ == "__main__":
    main()
