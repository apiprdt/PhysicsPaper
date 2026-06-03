"""
Reproducibility Study: 10 runs ADCD dengan seed berbeda.
Hitung mean ± std untuk class match rate dan NMSE.

Cara jalankan:
    py -3.11 run_reproducibility.py

Output:
    reproducibility_results.json  — 360 entries (10 seeds × 36 scenarios)
    Console                       — mean ± std table
"""

import json
import logging
import numpy as np

# Reuse existing run_scenario_benchmark
from run_correction_discovery import run_scenario_benchmark
from src.anomaly_scenarios import get_all_scenarios

logging.basicConfig(level=logging.WARNING)

N_SEEDS = 3
SEEDS = list(range(42, 42 + N_SEEDS))  # [42, 43, 44]
NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]


def main():
    print("=" * 70)
    print(f"   REPRODUCIBILITY STUDY: {N_SEEDS} seeds × 9 scenarios × 4 noise levels")
    print("=" * 70)

    # Hanya 9 skenario utama
    scenarios = [s for s in get_all_scenarios() if s.tier != "blind"]
    total = N_SEEDS * len(scenarios) * len(NOISE_LEVELS)
    count = 0
    all_results = []

    for seed in SEEDS:
        for scenario in scenarios:
            for noise in NOISE_LEVELS:
                count += 1
                # Gunakan fungsi yang sudah ada, passing seed
                res = run_scenario_benchmark(
                    scenario, noise, max_iter=4, proposer_type="mock", seed=seed
                )
                res["seed"] = seed  # Tambah info seed
                all_results.append(res)

                status = "OK" if res["class_match"] else "FAIL"
                print(f"[{count}/{total}] seed={seed} {status} "
                      f"{scenario.name} noise={noise*100:.0f}% "
                      f"NMSE={res['nmse_full']:.2e}")

    with open("reproducibility_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved: reproducibility_results.json ({len(all_results)} entries)")

    # Summary statistics
    print("\n" + "=" * 70)
    print("   RINGKASAN STATISTIK (mean ± std)")
    print("=" * 70)

    for noise in NOISE_LEVELS:
        subset = [r for r in all_results if abs(r["noise"] - noise) < 1e-9]
        rates = [1.0 if r["class_match"] else 0.0 for r in subset]
        nmses = [r["nmse_full"] for r in subset]
        print(f"Noise {noise*100:>4.0f}%: "
              f"ClassMatch = {np.mean(rates)*100:.1f}% +/- {np.std(rates)*100:.1f}%  |  "
              f"NMSE = {np.mean(nmses):.2e} +/- {np.std(nmses):.2e}")

    # Overall
    all_rates = [1.0 if r["class_match"] else 0.0 for r in all_results]
    print(f"\nOVERALL: {np.mean(all_rates)*100:.1f}% +/- {np.std(all_rates)*100:.1f}%")


if __name__ == "__main__":
    main()
