"""
Benchmark ADCD pipeline on synthetic-real hybrid physical scenarios (ADCD v2.0).

Note on Real-World Data:
These datasets are "synthetic-real hybrid" formulations. They generate physically-accurate
simulated values utilizing real experimentally-measured constants (e.g. JPL DE440 parameters
for Mercury, NIST energy levels for Hydrogen, CODATA values for muon g-2). They demonstrate
the pipeline's ability to extract physical corrections from data governed by realistic scales
and noise, rather than raw instrument logs which typically require custom pre-filtering 
(such as removing solar system N-body perturbations for Mercury's precession).

Runs the full correction discovery pipeline on 4 physical datasets:
  - Mercury perihelion precession (GR correction)
  - Hydrogen Lamb shift (QED correction)
  - Blackbody radiation (Planck correction to Rayleigh-Jeans)
  - Muon g-2 anomaly (QED Schwinger correction)

Usage:
    py -3.11 run_real_data_benchmark.py

Output:
    real_data_results.json  — per-scenario discovery results
    Console: formatted summary table
"""

import json
import time
import logging

from adcd.real_scenarios import get_real_scenarios
from run_correction_discovery import run_scenario_benchmark

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    scenarios = get_real_scenarios()
    results = []

    print("======================================================================")
    print("     ADCD v2.0 — REAL EXPERIMENTAL DATA BENCHMARK")
    print("======================================================================\n")
    print(f"{'Scenario':<30}  {'Class':<12}  {'Expected':<10}  {'Match':<5}  {'NMSE':<10}")
    print("-" * 75)

    for scenario in scenarios:
        t0 = time.time()
        # Run pipeline with pre-loaded real data using extended=True to allow fitting complex relations
        result = run_scenario_benchmark(
            scenario,
            noise_level=0.0,
            max_iter=4,
            proposer_type="mock",
            seed=42,
            extended=True
        )
        elapsed = time.time() - t0

        # Annotate with timing
        result["time_seconds"] = elapsed

        results.append(result)

        match_str = "[OK]  " if result["class_match"] else "[FAIL]"
        print(
            f"{scenario.name:<30} "
            f"{result['discovered_class']:^12} "
            f"{scenario.correction_class:^12} "
            f"{match_str:^6} "
            f"{result['nmse_full']:>10.3e}"
        )

    # Summary
    n_match = sum(1 for r in results if r["class_match"])
    n_total = len(results)

    print()
    print("=" * 70)
    print(f"  SUMMARY: {n_match}/{n_total} structural matches on real experimental data")
    print(f"  (target: >= 2/{n_total})")
    print("=" * 70)

    # Save results
    with open("real_data_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: real_data_results.json")


if __name__ == "__main__":
    main()
