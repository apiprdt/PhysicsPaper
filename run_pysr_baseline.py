"""
PySR Baseline: Jalankan unconstrained symbolic regression langsung pada residual.
Membandingkan dengan ADCD untuk membuktikan nilai Physics-Constrained Gates.

Cara jalankan:
    py -3.11 run_pysr_baseline.py

Output:
    pysr_baseline_results.json  — raw results (36 entries)
    Console                     — tabel perbandingan ADCD vs PySR
"""

import os
# Expose Julia to the execution environment path
os.environ['PATH'] = r'C:\Users\user\AppData\Local\Programs\Julia-1.12.6\bin;' + os.environ['PATH']

import json
import time
import logging
import numpy as np
import sympy as sp
from pysr import PySRRegressor

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.metrics import classify_structure

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("PySRBaseline")

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]
TIMEOUT_PER_RUN = 25  # detik per skenario, maksimal


def run_pysr_on_residual(scenario, noise_level: float, seed: int = 42) -> dict:
    """Jalankan PySR langsung pada residual (tanpa Physics Gates apapun)."""
    # 1. Generate data pakai API yang sama dengan ADCD
    X_dict, y_obs, y_classical, residual = scenario.generate_data(
        noise_level=noise_level, seed=seed
    )

    # 2. Build feature matrix
    feature_names = list(scenario.classical_variables)
    X_array = np.column_stack([X_dict[v] for v in feature_names])

    # 3. Jalankan PySR
    model = PySRRegressor(
        niterations=15,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "sin", "cos", "log", "tanh", "sqrt"],
        maxsize=15,
        populations=8,
        population_size=20,
        parsimony=0.0032,
        timeout_in_seconds=TIMEOUT_PER_RUN,
        deterministic=True,
        random_state=seed,
        procs=0,
        multithreading=False,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )

    t0 = time.time()
    try:
        model.fit(X_array, residual, variable_names=feature_names)
        elapsed = time.time() - t0

        best_expr_str = str(model.sympy())
        y_pred_residual = model.predict(X_array)

        # NMSE pada residual
        var_res = np.var(residual)
        nmse_residual = float(
            np.mean((y_pred_residual - residual) ** 2) / var_res
        ) if var_res > 1e-30 else 1.0

        # NMSE rekonstruksi penuh
        if scenario.correction_type == "multiplicative":
            y_recon = y_classical * (1.0 + y_pred_residual)
        else:
            y_recon = y_classical + y_pred_residual
        var_y = np.var(y_obs)
        nmse_full = float(
            np.mean((y_recon - y_obs) ** 2) / var_y
        ) if var_y > 1e-30 else 1.0

        # Klasifikasi struktur
        discovered_class = classify_structure(best_expr_str)
        class_match = (discovered_class == scenario.correction_class)
        complexity = int(model.get_best()["complexity"])

    except Exception as e:
        logger.error(f"PySR gagal: {scenario.name} noise={noise_level}: {e}")
        best_expr_str = "FAILED"
        nmse_residual = 1.0
        nmse_full = 1.0
        discovered_class = "failed"
        class_match = False
        complexity = 0
        elapsed = time.time() - t0

    return {
        "scenario": scenario.name,
        "tier": scenario.tier,
        "noise": noise_level,
        "method": "PySR",
        "discovered_expr": best_expr_str,
        "nmse_residual": nmse_residual,
        "nmse_full": nmse_full,
        "true_class": scenario.correction_class,
        "discovered_class": discovered_class,
        "class_match": class_match,
        "complexity": complexity,
        "time_seconds": elapsed,
    }


def main():
    print("=" * 70)
    print("     PySR BASELINE BENCHMARK")
    print("     Unconstrained SR vs ADCD Physics-Gated SR")
    print("=" * 70)

    # Hanya benchmark 9 skenario utama (bukan blind scenarios)
    scenarios = [s for s in get_all_scenarios() if s.tier != "blind"]
    results = []
    total = len(scenarios) * len(NOISE_LEVELS)

    for i, scenario in enumerate(scenarios):
        for j, noise in enumerate(NOISE_LEVELS):
            idx = i * len(NOISE_LEVELS) + j + 1
            res = run_pysr_on_residual(scenario, noise)
            results.append(res)
            status = "OK" if res["class_match"] else "FAIL"
            print(f"[{idx}/{total}] {status} {scenario.name} noise={noise*100:.0f}% "
                  f"class={res['discovered_class']} NMSE={res['nmse_full']:.2e}")

    # Simpan hasil
    with open("pysr_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: pysr_baseline_results.json ({len(results)} entries)")

    # Load ADCD results untuk perbandingan
    try:
        with open("scratch_correction_results.json") as f:
            adcd = json.load(f)
    except FileNotFoundError:
        print("scratch_correction_results.json tidak ditemukan — skip comparison")
        return

    # Tabel perbandingan per noise level
    print("\n" + "=" * 70)
    print("  PERBANDINGAN: ADCD vs PySR")
    print("=" * 70)
    print(f"{'Noise':>6} | {'ADCD':>8} | {'PySR':>8} | {'Delta':>8}")
    print("-" * 40)

    for noise in NOISE_LEVELS:
        adcd_n = [r for r in adcd if abs(r["noise"] - noise) < 1e-9]
        pysr_n = [r for r in results if abs(r["noise"] - noise) < 1e-9]
        adcd_match = sum(1 for r in adcd_n if r["class_match"])
        pysr_match = sum(1 for r in pysr_n if r["class_match"])
        delta = adcd_match - pysr_match
        sign = "+" if delta >= 0 else ""
        print(f"{noise*100:>5.0f}% | {adcd_match:>4}/9   | {pysr_match:>4}/9   | {sign}{delta:>+4}")

    adcd_tot = sum(1 for r in adcd if r["class_match"])
    pysr_tot = sum(1 for r in results if r["class_match"])
    print("-" * 40)
    print(f"{'Total':>6} | {adcd_tot:>4}/36  | {pysr_tot:>4}/36  | {adcd_tot-pysr_tot:>+4}")
    print(f"         | {adcd_tot/36*100:.1f}%    | {pysr_tot/36*100:.1f}%    |")


if __name__ == "__main__":
    main()
