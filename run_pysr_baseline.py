"""
PySR Baseline: unconstrained symbolic regression on residuals.
Compare against ADCD to quantify physics-gated search efficiency.

Usage:
    py -3.11 run_pysr_baseline.py --profile fast
    py -3.11 run_pysr_baseline.py --profile fair
    py -3.11 run_pysr_baseline.py --profile generous

Output:
    pysr_baseline_{profile}.json
"""

import os
import argparse
import json
import time
import logging
import numpy as np
from pysr import PySRRegressor

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.metrics import classify_structure

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("PySRBaseline")

NOISE_LEVELS = [0.0, 0.01, 0.05, 0.10]

PYSR_PROFILES = {
    "fast": {
        "niterations": 15,
        "maxsize": 15,
        "timeout_in_seconds": 25,
        "description": "Matched wall-clock budget (legacy comparison)",
    },
    "fair": {
        "niterations": 100,
        "maxsize": 30,
        "timeout_in_seconds": 60,
        "description": "PySR near-default budget (primary fair comparison)",
    },
    "generous": {
        "niterations": 200,
        "maxsize": 40,
        "timeout_in_seconds": 120,
        "description": "Upper-bound PySR search budget",
    },
}


def _count_pysr_expressions(model) -> int:
    """Best-effort count of expressions evaluated by PySR."""
    try:
        eqs = model.equations_
        if eqs is not None and len(eqs) > 0:
            return int(len(eqs))
    except Exception:
        pass
    try:
        best = model.get_best()
        if best is not None and "complexity" in best:
            return int(best["complexity"])
    except Exception:
        pass
    return 0


def run_pysr_on_residual(scenario, noise_level: float, profile: str, seed: int = 42) -> dict:
    """Run PySR on the residual (no physics gates)."""
    cfg = PYSR_PROFILES[profile]

    X_dict, y_obs, y_classical, residual = scenario.generate_data(
        noise_level=noise_level, seed=seed
    )

    feature_names = list(scenario.classical_variables)
    X_array = np.column_stack([X_dict[v] for v in feature_names])

    model = PySRRegressor(
        niterations=cfg["niterations"],
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["exp", "sin", "cos", "log", "tanh", "sqrt"],
        maxsize=cfg["maxsize"],
        populations=8,
        population_size=20,
        parsimony=0.0032,
        timeout_in_seconds=cfg["timeout_in_seconds"],
        deterministic=True,
        random_state=seed,
        procs=0,
        multithreading=False,
        verbosity=0,
        progress=False,
        temp_equation_file=True,
    )

    t0 = time.time()
    expressions_evaluated = 0
    try:
        model.fit(X_array, residual, variable_names=feature_names)
        elapsed = time.time() - t0
        expressions_evaluated = _count_pysr_expressions(model)

        best_expr_str = str(model.sympy())
        y_pred_residual = model.predict(X_array)

        var_res = np.var(residual)
        nmse_residual = float(
            np.mean((y_pred_residual - residual) ** 2) / var_res
        ) if var_res > 1e-30 else 1.0

        if scenario.correction_type == "multiplicative":
            y_recon = y_classical * (1.0 + y_pred_residual)
        else:
            y_recon = y_classical + y_pred_residual
        var_y = np.var(y_obs)
        nmse_full = float(
            np.mean((y_recon - y_obs) ** 2) / var_y
        ) if var_y > 1e-30 else 1.0

        discovered_class = classify_structure(best_expr_str)
        class_match = (discovered_class == scenario.correction_class)
        complexity = int(model.get_best()["complexity"])

    except Exception as e:
        logger.error(f"PySR failed: {scenario.name} noise={noise_level}: {e}")
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
        "profile": profile,
        "discovered_expr": best_expr_str,
        "nmse_residual": nmse_residual,
        "nmse_full": nmse_full,
        "true_class": scenario.correction_class,
        "discovered_class": discovered_class,
        "class_match": class_match,
        "complexity": complexity,
        "expressions_evaluated": expressions_evaluated,
        "time_seconds": elapsed,
        "pysr_config": {
            "niterations": cfg["niterations"],
            "maxsize": cfg["maxsize"],
            "timeout_in_seconds": cfg["timeout_in_seconds"],
        },
    }


def _print_comparison(results: list, adcd_path: str) -> None:
    try:
        with open(adcd_path) as f:
            adcd = json.load(f)
    except FileNotFoundError:
        print(f"{adcd_path} not found — skipping ADCD comparison")
        return

    print("\n" + "=" * 70)
    print("  COMPARISON: ADCD vs PySR")
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

    pysr_times = [r["time_seconds"] for r in results]
    pysr_exprs = [r.get("expressions_evaluated", 0) for r in results]
    print(f"\nPySR mean wall-clock: {np.mean(pysr_times):.1f}s/scenario")
    if any(pysr_exprs):
        print(f"PySR mean expressions in hall-of-fame: {np.mean(pysr_exprs):.0f}")


def main():
    parser = argparse.ArgumentParser(description="PySR baseline benchmark for ADCD comparison")
    parser.add_argument(
        "--profile",
        type=str,
        choices=list(PYSR_PROFILES.keys()),
        default="fair",
        help="PySR search budget profile (default: fair)",
    )
    parser.add_argument(
        "--adcd-results",
        type=str,
        default="scratch_correction_results.json",
        help="Path to ADCD benchmark JSON for side-by-side comparison",
    )
    args = parser.parse_args()

    if os.name == "nt":
        julia_bin = r"C:\Users\user\AppData\Local\Programs\Julia-1.12.6\bin"
        if os.path.isdir(julia_bin):
            os.environ["PATH"] = julia_bin + ";" + os.environ.get("PATH", "")

    cfg = PYSR_PROFILES[args.profile]
    print("=" * 70)
    print(f"     PySR BASELINE — profile={args.profile.upper()}")
    print(f"     {cfg['description']}")
    print(f"     niter={cfg['niterations']} maxsize={cfg['maxsize']} timeout={cfg['timeout_in_seconds']}s")
    print("=" * 70)

    scenarios = [s for s in get_all_scenarios() if s.tier != "blind"]
    results = []
    total = len(scenarios) * len(NOISE_LEVELS)

    for i, scenario in enumerate(scenarios):
        for j, noise in enumerate(NOISE_LEVELS):
            idx = i * len(NOISE_LEVELS) + j + 1
            res = run_pysr_on_residual(scenario, noise, profile=args.profile)
            results.append(res)
            status = "OK" if res["class_match"] else "FAIL"
            print(
                f"[{idx}/{total}] {status} {scenario.name} noise={noise*100:.0f}% "
                f"class={res['discovered_class']} NMSE={res['nmse_full']:.2e} "
                f"time={res['time_seconds']:.1f}s exprs={res['expressions_evaluated']}"
            )

    out_path = f"pysr_baseline_{args.profile}.json"
    if args.profile == "fast":
        out_path_legacy = "pysr_baseline_results.json"
        with open(out_path_legacy, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved (legacy): {out_path_legacy}")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved: {out_path} ({len(results)} entries)")

    _print_comparison(results, args.adcd_results)


if __name__ == "__main__":
    main()
