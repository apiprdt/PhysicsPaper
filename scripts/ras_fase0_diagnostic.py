"""
Fase 0 — RAS Bias Diagnostic on 9 Real Benchmark Scenarios
===========================================================
ADCD_Upgrade_Plan_v2.md §4.1 Fase 0:

  "Implementasikan §3.3, uji ulang tabel §3.1 dan §3.2 tapi terhadap
   9 skenario benchmark ASLI, semua 16 seed. Kalau bias serupa muncul
   di data nyata, lanjut ke Fase 1."

This script:
  1. Loads all 9 primary benchmark scenarios (3 textbook + 3 cross-domain
     + 3 synthetic) at noise levels [0%, 1%, 5%, 10%] and seeds 0-15.
  2. For each (scenario, noise, seed): generates data, computes RAS via
     the NEW nonlinear fit, and records gamma_est, gamma_std, fit_quality,
     suggested_class.
  3. Compares gamma_est to the ground-truth correction exponent for
     scenarios where the correction is a pure power-law (polynomial class).
  4. Prints a summary table and saves results to ras_fase0_results.json.

Run:
    py -3.11 scripts/ras_fase0_diagnostic.py
"""
import json
import sys
import numpy as np

# Ensure src/ is on path when run from project root
sys.path.insert(0, "src")

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.residual_analyzer import compute_ras

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
NOISE_LEVELS  = [0.00, 0.01, 0.05, 0.10]
SEEDS         = list(range(16))          # 16 seeds as per paper benchmark
N_POINTS      = 200
OUTPUT_FILE   = "ras_fase0_results.json"

# Ground-truth gamma for scenarios with pure power-law corrections.
# Format: scenario_name -> expected gamma
# Only scenarios where correction_class == "polynomial" and the correction
# is a *single* power-law term are included; exponential / rational corrections
# cannot be characterised by a single gamma.
KNOWN_GAMMA = {
    "Relativistic KE":   2.0,   # theta_0 * (v/c)^2  -> gamma=2 in v
    "Anharmonic Spring": 4.0,   # theta_0 * x^4      -> gamma=4 in x
    "Net Radiation":     4.0,   # ~T^4 correction    -> gamma=4 in T (approximate)
    "Nonlinear Drag":    2.0,   # alpha * v^2         -> gamma=2 in v
}

# ─────────────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────────────

def _classical_limit_val(scenario, X: dict) -> float:
    """
    Parse scenario.classical_limit_direction to a numeric limit value.
    Returns the actual limit value (0.0 for direction "0", max(x) for "oo").
    """
    direction = scenario.classical_limit_direction.strip()
    var = scenario.classical_limit_variable
    if direction in ("0", "0.0"):
        return 0.0
    elif direction in ("oo", "inf", "infinity"):
        return float(np.max(X.get(var, [1.0])))  # treat limit as beyond data
    else:
        try:
            return float(direction)
        except ValueError:
            return 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def run_fase0():
    all_scenarios = get_all_scenarios()

    # Keep only the 9 primary benchmark scenarios (tier in textbook/cross_domain/synthetic)
    primary = [s for s in all_scenarios
               if s.tier in ("textbook", "cross_domain", "synthetic")
               and not s.name.startswith(("Subtle", "Wrong", "Missing", "Spurious"))]

    print(f"Found {len(primary)} primary scenarios")
    for s in primary:
        print(f"  [{s.tier:12s}] {s.name}")
    print()

    records = []
    summary_by_scenario = {}

    for scenario in primary:
        var = scenario.classical_limit_variable
        gamma_true = KNOWN_GAMMA.get(scenario.name)
        errs = []

        for noise in NOISE_LEVELS:
            for seed in SEEDS:
                X, y_obs, y_classical, residual = scenario.generate_data(
                    n_points=N_POINTS, noise_level=noise, seed=seed
                )
                x_vals = X.get(var, np.ones(N_POINTS))
                limit_val = _classical_limit_val(scenario, X)

                ras = compute_ras(x_vals=x_vals, delta_vals=residual,
                                  limit_val=limit_val)

                gamma_est = ras.get("leading_exponent")
                gamma_std = ras.get("gamma_std")
                fit_q     = ras.get("fit_quality", 0.0)
                cls       = ras.get("suggested_class", "unknown")

                error_pct = None
                if gamma_true is not None and gamma_est is not None:
                    error_pct = abs(gamma_est - gamma_true) / gamma_true * 100

                if error_pct is not None:
                    errs.append(error_pct)

                records.append({
                    "scenario":     scenario.name,
                    "tier":         scenario.tier,
                    "limit_var":    var,
                    "limit_val":    limit_val,
                    "noise":        noise,
                    "seed":         seed,
                    "gamma_true":   gamma_true,
                    "gamma_est":    gamma_est,
                    "gamma_std":    gamma_std,
                    "fit_quality":  fit_q,
                    "suggested_class": cls,
                    "error_pct":    error_pct,
                })

        # Per-scenario summary (only for known-gamma scenarios)
        if errs:
            summary_by_scenario[scenario.name] = {
                "gamma_true":     gamma_true,
                "mean_error_pct": float(np.mean(errs)),
                "max_error_pct":  float(np.max(errs)),
                "n_failed":       sum(1 for r in records
                                     if r["scenario"] == scenario.name
                                     and r["gamma_est"] is None),
                "n_total":        len(NOISE_LEVELS) * len(SEEDS),
            }

    # ─── Print summary table ──────────────────────────────────────────────
    print("=" * 70)
    print(f"{'Scenario':<22} {'gamma_true':>7} {'mean err%':>10} {'max err%':>9} {'failed':>7}")
    print("-" * 70)
    for name, s in summary_by_scenario.items():
        print(f"{name:<22} {s['gamma_true']:>7.1f} "
              f"{s['mean_error_pct']:>10.1f} {s['max_error_pct']:>9.1f} "
              f"{s['n_failed']:>6}/{s['n_total']}")
    print("=" * 70)

    # Per-noise breakdown for key scenarios
    print()
    print("Per-noise breakdown (mean error % across 16 seeds):")
    print(f"{'Scenario':<22} {'noise':>6} {'mean gamma_est':>11} {'mean err%':>10}")
    print("-" * 55)
    for scenario in primary:
        name = scenario.name
        if name not in KNOWN_GAMMA:
            continue
        gamma_true = KNOWN_GAMMA[name]
        for noise in NOISE_LEVELS:
            subset = [r for r in records
                      if r["scenario"] == name
                      and r["noise"] == noise
                      and r["gamma_est"] is not None]
            if not subset:
                continue
            mean_est = np.mean([r["gamma_est"] for r in subset])
            mean_err = np.mean([r["error_pct"] for r in subset])
            print(f"{name:<22} {noise:>6.0%} {mean_est:>11.3f} {mean_err:>10.1f}")
        print()

    # Save full records
    out = {
        "config": {
            "noise_levels": NOISE_LEVELS,
            "seeds":        SEEDS,
            "n_points":     N_POINTS,
            "known_gamma":  KNOWN_GAMMA,
        },
        "summary": summary_by_scenario,
        "records": records,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nFull results saved to {OUTPUT_FILE}")
    print("\nFase 0 verdict:")
    bad = {k: v for k, v in summary_by_scenario.items() if v["mean_error_pct"] > 10}
    if bad:
        print(f"  BIAS DETECTED in {len(bad)} scenario(s): {list(bad.keys())}")
        print("  -> Proceed to Fase 1 (soft-weighting Grammar Proposer)")
    else:
        print("  No significant bias detected (all mean errors < 10%).")
        print("  -> Adopt nonlinear RAS as standalone improvement; stop here.")


if __name__ == "__main__":
    run_fase0()
