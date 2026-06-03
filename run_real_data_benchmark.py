"""
Benchmark ADCD pipeline on real experimental data (ADCD v2.0).

Runs the full correction discovery pipeline on 4 real physics datasets:
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

from src.real_scenarios import get_real_scenarios
from run_correction_discovery import run_scenario_benchmark

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s [%(levelname)s] %(message)s")


def main():
    print("=" * 70)
    print("     ADCD v2.0 — REAL EXPERIMENTAL DATA BENCHMARK")
    print("=" * 70)
    print()
    print(f"{'Scenario':<30} {'Class':^12} {'Expected':^12} {'Match':^6} {'NMSE':>10}")
    print("-" * 75)

    scenarios = get_real_scenarios()
    results = []

    for scenario in scenarios:
        t0 = time.time()
        result = run_scenario_benchmark(
            scenario,
            noise_level=0.0,   # noise already embedded in the real data model
            max_iter=4,
            proposer_type="mock",
            seed=42,
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
