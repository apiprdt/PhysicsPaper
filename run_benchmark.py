import sys
import argparse
import time
import numpy as np
import sympy as sp
from adcd.anomaly_scenarios import get_all_scenarios
from run_correction_discovery import run_scenario_benchmark

def main():
    parser = argparse.ArgumentParser(description="Run ADCD standard scenarios benchmark.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise", type=float, default=0.05)
    parser.add_argument("--proposer", type=str, default="mock", choices=["mock", "gemini", "hybrid"])
    args = parser.parse_args()
    
    print(f"Running benchmark with seed={args.seed}, noise={args.noise}, proposer={args.proposer}...")
    scenarios = [s for s in get_all_scenarios() if s.tier in ("textbook", "cross_domain", "synthetic")]
    
    success = 0
    for scenario in scenarios:
        res = run_scenario_benchmark(
            scenario, args.noise, max_iter=4, proposer_type=args.proposer, seed=args.seed
        )
        if res["class_match"]:
            success += 1
            status = "OK"
        else:
            status = "FAIL"
        print(f"[{status}] {scenario.name} (class: {scenario.correction_class}) -> discovered: {res['discovered_expr']} | NMSE_full={res['nmse_full']:.2e}")
        
    pct = 100.0 * success / len(scenarios)
    print(f"\nResult: {success}/{len(scenarios)} ({pct:.1f}%)")

if __name__ == "__main__":
    main()
