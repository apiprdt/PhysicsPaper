"""
Timing analysis: measures wall-clock time per ADCD stage and compares with PySR.

Output columns:
  - Scenario name
  - Data generation + residual extraction time
  - Full ADCD pipeline time (proposer → physics gates → JAX optimizer)
  - Total wall time
  - PySR equivalent time (from pysr_baseline_results.json)
  - Speedup factor

Usage:
    py -3.11 run_timing_analysis.py

Output:
    timing_results.json
    Console: formatted comparison table
"""

import json
import time
import logging
import numpy as np
import sympy as sp

from adcd.anomaly_scenarios import get_all_scenarios, AnomalyScenario
from adcd.llm_proposer import CorrectionMockProposer, ProposalContext
from adcd.correction_orchestrator import CorrectionOrchestrator
from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.arc_scorer import ARCScorer, AsymptoticRegime
from adcd.pipeline import Stage1Pipeline
from adcd.jax_optimizer import JAXOptimizer

logging.basicConfig(level=logging.WARNING)

NOISE = 0.05  # Fixed noise level for fair comparison (matches PySR baseline)


def time_single_scenario(scenario: AnomalyScenario, noise: float = NOISE,
                         seed: int = 42, max_iter: int = 4) -> dict:
    """Run ADCD pipeline and measure wall-clock time breakdown."""

    # --- Stage 0: Data generation + residual ---
    t0 = time.perf_counter()
    X, y_obs, y_classical, residual = scenario.generate_data(
        noise_level=noise, seed=seed
    )
    t_data_gen = time.perf_counter() - t0

    # --- Full pipeline (proposer + physics gates + JAX) ---
    t1 = time.perf_counter()

    # Setup pipeline components (same as run_correction_discovery.py)
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
    proposer = CorrectionMockProposer(seed=seed)

    orchestrator = CorrectionOrchestrator(
        proposer=proposer,
        pipeline=pipeline,
        optimizer=optimizer,
        max_iterations=max_iter,
        top_k=8,
        convergence_nmse=1e-5,
        verbose=False
    )

    search_res = orchestrator.search_correction(scenario, noise_level=noise, seed=seed)
    t_pipeline = time.perf_counter() - t1

    t_total = t_data_gen + t_pipeline

    return {
        "scenario": scenario.name,
        "tier": scenario.tier,
        "t_data_gen_ms": round(t_data_gen * 1000, 2),
        "t_pipeline_ms": round(t_pipeline * 1000, 2),
        "t_total_ms": round(t_total * 1000, 2),
        "best_expr": search_res.best_expr,
        "converged": search_res.converged,
    }


def main():
    print("=" * 78)
    print("        ADCD TIMING ANALYSIS & PySR COMPARISON")
    print("=" * 78)
    print(f"  Noise level: {NOISE * 100:.0f}%")
    print()

    # Only benchmark non-blind scenarios (matches PySR baseline)
    scenarios = [s for s in get_all_scenarios() if s.tier != "blind"]

    results = []

    print(f"{'Scenario':<25} {'Data Gen (ms)':>14} {'Pipeline (ms)':>14} {'Total (ms)':>12}")
    print("-" * 70)

    for s in scenarios:
        r = time_single_scenario(s)
        results.append(r)
        print(f"{r['scenario']:<25} {r['t_data_gen_ms']:>14.2f} "
              f"{r['t_pipeline_ms']:>14.2f} {r['t_total_ms']:>12.2f}")

    # --- Load PySR baseline for comparison ---
    try:
        with open("pysr_baseline_results.json") as f:
            pysr = json.load(f)

        # Filter for 5% noise
        pysr_5 = {r["scenario"]: r["time_seconds"] for r in pysr
                  if abs(r["noise"] - NOISE) < 1e-9}

        if pysr_5:
            print()
            print("=" * 78)
            print(f"  SPEED COMPARISON: ADCD vs PySR at {NOISE * 100:.0f}% noise")
            print("=" * 78)
            print(f"{'Scenario':<25} {'ADCD (ms)':>12} {'PySR (ms)':>12} {'Speedup':>10}")
            print("-" * 65)

            total_adcd = 0
            total_pysr = 0
            for r in results:
                pysr_t_s = pysr_5.get(r["scenario"], 0)
                pysr_t_ms = pysr_t_s * 1000
                speedup = pysr_t_ms / r["t_total_ms"] if r["t_total_ms"] > 0 else 0
                total_adcd += r["t_total_ms"]
                total_pysr += pysr_t_ms
                print(f"{r['scenario']:<25} {r['t_total_ms']:>12.1f} "
                      f"{pysr_t_ms:>12.1f} {speedup:>9.1f}x")

            avg_speedup = total_pysr / total_adcd if total_adcd > 0 else 0
            print("-" * 65)
            print(f"{'TOTAL':<25} {total_adcd:>12.1f} "
                  f"{total_pysr:>12.1f} {avg_speedup:>9.1f}x")

    except FileNotFoundError:
        print("\npysr_baseline_results.json not found -- skipping PySR comparison")

    # --- Save results ---
    with open("timing_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: timing_results.json ({len(results)} entries)")


if __name__ == "__main__":
    main()
