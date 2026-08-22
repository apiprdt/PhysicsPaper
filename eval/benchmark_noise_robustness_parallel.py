"""
ADCD Parallel Noise Robustness Benchmark (Julia Engine Optimized)
Executes full formal 4-step ADCD validation protocol concurrently across
all noise levels and seeds for locked scenarios with 3-tier epistemic verdict.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.run_adcd_v3_validation_blind import (
    DOMAIN_RESTRICTIONS,
    ScenarioThresholdConfig,
    run_scenario_protocol,
)

NOISE_SWEEP = [0.01, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
SEEDS = [42, 43, 44]

DEFAULT_CLEAN_DOMAINS: Dict[str, float] = {
    "Time Dilation": 0.99,       # Full relativistic domain
    "Screened Coulomb": 4.0,     # Standard domain
    "Entropy Expansion": 3.0,    # Broad domain (breaks Taylor degeneracy)
}


def _run_single_task(task: Tuple[str, float, int, str, float]) -> Dict[str, Any]:
    scenario_name, noise, seed, engine, actual_dmax = task
    scenarios = {s.name: s for s in get_all_scenarios()}
    scenario = scenarios[scenario_name]
    scenario.engine = engine

    t_cfg = ScenarioThresholdConfig.for_scenario(scenario, noise_level=noise)
    start_t = time.time()
    res = run_scenario_protocol(
        scenario=scenario,
        seed=seed,
        threshold_cfg=t_cfg,
        noise_level=noise,
        domain_max=actual_dmax,
    )
    elapsed = time.time() - start_t

    primary_check = res.checks.get("primary_search", {})
    expr_str = primary_check.get("top_candidate", "None")
    nmse = primary_check.get("nmse", float("inf"))
    cls_name = primary_check.get("discovered_class", "unknown")
    match_level = primary_check.get("match_level", "none")
    is_match = match_level in ["exact", "class_only"]

    return {
        "scenario": scenario_name,
        "noise": noise,
        "seed": seed,
        "tier": res.tier,
        "all_passed": res.all_passed,
        "is_match": is_match,
        "expr_str": expr_str,
        "nmse": nmse,
        "cls_name": cls_name,
        "match_level": match_level,
        "checks": res.checks,
        "elapsed": elapsed,
    }


def main():
    parser = argparse.ArgumentParser(description="ADCD Parallel Noise Robustness Benchmark")
    parser.add_argument("--engine", type=str, default="julia", choices=["python", "julia"], help="Execution engine")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all", "Time Dilation", "Screened Coulomb", "Entropy Expansion"],
        help="Scenario name or 'all'",
    )
    parser.add_argument("--workers", type=int, default=6, help="Concurrent workers")
    parser.add_argument("--domain-max", type=float, default=None, help="Custom domain max override")
    args = parser.parse_args()

    target_scenarios = (
        ["Time Dilation", "Screened Coulomb", "Entropy Expansion"]
        if args.scenario == "all"
        else [args.scenario]
    )

    tasks = []
    for sc_name in target_scenarios:
        actual_dmax = args.domain_max if args.domain_max is not None else DEFAULT_CLEAN_DOMAINS.get(
            sc_name, DOMAIN_RESTRICTIONS.get(sc_name, {}).get("domain_max", 1.0)
        )
        for noise in NOISE_SWEEP:
            for seed in SEEDS:
                tasks.append((sc_name, noise, seed, args.engine, actual_dmax))

    total_tasks = len(tasks)
    print("=" * 85)
    print(f" ADCD PARALLEL NOISE ROBUSTNESS BENCHMARK (Engine: {args.engine}, Workers: {args.workers})")
    print(f" Total Runs Scheduled: {total_tasks} ({len(target_scenarios)} scenarios x {len(NOISE_SWEEP)} noise x {len(SEEDS)} seeds)")
    print("=" * 85)

    start_total = time.time()
    results_by_task = []
    completed_count = 0

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(_run_single_task, t): t for t in tasks}
        for future in as_completed(futures):
            res = future.result()
            results_by_task.append(res)
            completed_count += 1
            print(
                f"[{completed_count:>2}/{total_tasks}] "
                f"{res['scenario']:<18} | Noise: {res['noise']:.2f} | Seed: {res['seed']} | "
                f"Tier: {res['tier']:<19} | NMSE: {res['nmse']:.4f} | ({res['elapsed']:.1f}s)"
            )

    total_elapsed = time.time() - start_total
    print("\n" + "=" * 85)
    print(f" ALL RUNS COMPLETED IN {total_elapsed:.1f}s (Average: {total_elapsed/total_tasks:.2f}s/run)")
    print("=" * 85)

    # Aggregation & Summary
    summary_tables = {}
    for sc_name in target_scenarios:
        sc_results = [r for r in results_by_task if r["scenario"] == sc_name]
        summary_rows = []
        for noise in NOISE_SWEEP:
            noise_runs = [r for r in sc_results if abs(r["noise"] - noise) < 1e-6]
            n_runs = len(noise_runs)
            id_count = sum(1 for r in noise_runs if r["tier"] == "IDENTIFIABLE")
            unres_count = sum(1 for r in noise_runs if r["tier"] == "DETECTED_UNRESOLVED")
            withheld_count = sum(1 for r in noise_runs if r["tier"] == "WITHHELD")
            match_count = sum(1 for r in noise_runs if r["is_match"])
            summary_rows.append({
                "noise": noise,
                "identifiable_rate": id_count / n_runs if n_runs > 0 else 0.0,
                "detected_unresolved_rate": unres_count / n_runs if n_runs > 0 else 0.0,
                "withheld_rate": withheld_count / n_runs if n_runs > 0 else 0.0,
                "match_rate": match_count / n_runs if n_runs > 0 else 0.0,
            })
        summary_tables[sc_name] = summary_rows

        print("\n" + "-" * 85)
        print(f" NOISE ROBUSTNESS SUMMARY (3-TIER VERDICT): {sc_name}")
        print("-" * 85)
        print(f"{'Noise':<8} | {'Identifiable':<14} | {'Detected (Unres)':<18} | {'Withheld':<10} | {'True Match':<12}")
        print("-" * 85)
        for r in summary_rows:
            print(
                f"{r['noise']:<8.2f} | "
                f"{r['identifiable_rate'] * 100:>11.0f}% | "
                f"{r['detected_unresolved_rate'] * 100:>15.0f}% | "
                f"{r['withheld_rate'] * 100:>8.0f}% | "
                f"{r['match_rate'] * 100:>10.0f}%"
            )
        print("-" * 85)

    # Save full JSON report
    os.makedirs("run_outputs", exist_ok=True)
    out_file = os.path.join("run_outputs", f"noise_robustness_{args.engine}_parallel.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(
            {
                "engine": args.engine,
                "total_elapsed_seconds": total_elapsed,
                "summary": summary_tables,
                "raw_runs": results_by_task,
            },
            f,
            indent=2,
            default=str,
        )
    print(f"\nFull benchmark report saved to: {out_file}\n")


if __name__ == "__main__":
    main()
