#!/usr/bin/env python3
"""
B2: Correction magnitude scaling experiment.

Varies the true correction amplitude on Relativistic KE to test when
correction-first search outperforms tabula rasa (PySR optional).

Usage:
    py -3.11 run_correction_scaling.py
    py -3.11 run_correction_scaling.py --include-pysr --profile fast
"""

import argparse
import copy
import json
import time

from adcd.anomaly_scenarios import get_all_scenarios
from run_correction_discovery import run_scenario_benchmark

SCALES = [0.01, 0.05, 0.1, 0.25, 0.5, 1.0]


def _scaled_scenario(base, scale: float):
    """Scale multiplicative correction amplitude."""
    sc = copy.deepcopy(base)
    sc.correction_constants = dict(sc.correction_constants)
    sc.correction_constants["theta_0"] = sc.correction_constants.get("theta_0", 0.75) * scale
    sc.name = f"{base.name} (scale={scale})"
    return sc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-pysr", action="store_true")
    parser.add_argument("--profile", default="fast", choices=["fast", "fair", "generous"])
    parser.add_argument("--noise", type=float, default=0.05)
    args = parser.parse_args()

    base = next(s for s in get_all_scenarios() if s.name == "Relativistic KE")
    results = []

    print("=" * 70)
    print("CORRECTION MAGNITUDE SCALING (Relativistic KE)")
    print("=" * 70)

    for scale in SCALES:
        scenario = _scaled_scenario(base, scale)
        t0 = time.time()
        adcd_res = run_scenario_benchmark(
            scenario, args.noise, max_iter=4, proposer_type="mock", seed=42
        )
        entry = {
            "scale": scale,
            "method": "ADCD",
            "class_match": adcd_res["class_match"],
            "nmse_full": adcd_res["nmse_full"],
            "time_seconds": time.time() - t0,
        }
        results.append(entry)
        print(
            f"scale={scale:.2f} ADCD match={entry['class_match']} "
            f"NMSE={entry['nmse_full']:.2e} t={entry['time_seconds']:.1f}s"
        )

        if args.include_pysr:
            try:
                from run_pysr_baseline import run_pysr_on_residual
                pysr_res = run_pysr_on_residual(scenario, args.noise, profile=args.profile)
                results.append({
                    "scale": scale,
                    "method": "PySR",
                    "profile": args.profile,
                    "class_match": pysr_res["class_match"],
                    "nmse_full": pysr_res["nmse_full"],
                    "time_seconds": pysr_res["time_seconds"],
                })
                print(
                    f"         PySR match={pysr_res['class_match']} "
                    f"NMSE={pysr_res['nmse_full']:.2e}"
                )
            except ImportError:
                print("         PySR not available — skip")

    out = "correction_scaling_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
