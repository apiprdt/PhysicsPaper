"""
MLP Baseline: 3-layer neural network pada residual data.
Tujuan: Buktikan bahwa ADCD menghasilkan interpretable expressions
        sementara MLP hanya memberikan black-box numerik.

Cara jalankan:
    py -3.11 run_mlp_baseline.py

Output:
    mlp_baseline_results.json  — raw results (36 entries)
    Console                    — tabel perbandingan ADCD vs MLP
"""

import json
import time
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

from adcd.anomaly_scenarios import get_all_scenarios

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]


def run_mlp_on_residual(scenario, noise_level: float, seed: int = 42) -> dict:
    """Jalankan MLP 3-layer pada residual. Hitung NMSE rekonstruksi penuh."""
    X_dict, y_obs, y_classical, residual = scenario.generate_data(
        noise_level=noise_level, seed=seed
    )

    # Build feature matrix
    feature_names = list(scenario.classical_variables)
    X_array = np.column_stack([X_dict[v] for v in feature_names])

    # Normalize input features (penting untuk MLP)
    scaler_X = StandardScaler()
    X_scaled = scaler_X.fit_transform(X_array)

    # MLP 3-layer: 64-32-16 neurons, ReLU activation
    mlp = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=2000,
        random_state=seed,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=50,
        tol=1e-6
    )

    t0 = time.time()
    mlp.fit(X_scaled, residual)
    elapsed = time.time() - t0

    y_pred_residual = mlp.predict(X_scaled)

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

    return {
        "scenario": scenario.name,
        "tier": scenario.tier,
        "noise": noise_level,
        "method": "MLP-3L",
        "nmse_residual": nmse_residual,
        "nmse_full": nmse_full,
        # MLP TIDAK menghasilkan symbolic expression — ini SENGAJA
        # Ini adalah poin utama paper: ADCD = interpretable, MLP = black box
        "discovered_expr": "black_box",
        "class_match": False,  # MLP tidak bisa di-classify structurally
        "interpretable": False,
        "time_seconds": elapsed,
    }


def main():
    print("=" * 70)
    print("     MLP BASELINE BENCHMARK (Black-Box Comparison)")
    print("=" * 70)

    scenarios = [s for s in get_all_scenarios() if s.tier != "blind"]
    results = []
    total = len(scenarios) * len(NOISE_LEVELS)

    for i, scenario in enumerate(scenarios):
        for j, noise in enumerate(NOISE_LEVELS):
            idx = i * len(NOISE_LEVELS) + j + 1
            res = run_mlp_on_residual(scenario, noise)
            results.append(res)
            print(f"[{idx}/{total}] {scenario.name} noise={noise*100:.0f}% "
                  f"NMSE_full={res['nmse_full']:.2e} "
                  f"NMSE_res={res['nmse_residual']:.2e} "
                  f"time={res['time_seconds']:.2f}s")

    with open("mlp_baseline_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: mlp_baseline_results.json ({len(results)} entries)")

    # Load ADCD results untuk perbandingan
    try:
        with open("scratch_correction_results.json") as f:
            adcd = json.load(f)
    except FileNotFoundError:
        print("scratch_correction_results.json tidak ditemukan — skip comparison")
        return

    # Tabel: ADCD vs MLP pada NMSE (bukan class match — MLP tidak punya)
    print("\n" + "=" * 70)
    print("  NMSE COMPARISON: ADCD vs MLP (lower = better fit)")
    print("  Catatan: MLP tidak bisa menghasilkan symbolic expression.")
    print("           ADCD lebih tinggi NMSE tapi FULLY INTERPRETABLE.")
    print("=" * 70)
    print(f"{'Noise':>6} | {'ADCD avg NMSE':>14} | {'MLP avg NMSE':>14} | {'ADCD Class Match':>18}")
    print("-" * 65)

    for noise in NOISE_LEVELS:
        adcd_n = [r for r in adcd if abs(r["noise"] - noise) < 1e-9]
        mlp_n = [r for r in results if abs(r["noise"] - noise) < 1e-9]
        adcd_nmse_avg = np.mean([r["nmse_full"] for r in adcd_n])
        mlp_nmse_avg = np.mean([r["nmse_full"] for r in mlp_n])
        adcd_match = sum(1 for r in adcd_n if r["class_match"])
        print(f"{noise*100:>5.0f}% | {adcd_nmse_avg:>14.2e} | {mlp_nmse_avg:>14.2e} | {adcd_match}/9 ({adcd_match/9*100:.0f}%)")


if __name__ == "__main__":
    main()
