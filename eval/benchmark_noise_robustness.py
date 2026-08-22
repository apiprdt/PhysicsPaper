"""
ADCD Noise Robustness Benchmark
Evaluates true IDENTIFIABLE vs WITHHELD transition across noise levels
using the full formal 4-step ADCD validation protocol.
"""

from __future__ import annotations

import argparse
from typing import Dict, List, Optional

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


def run_benchmark_for_scenario(
    scenario_name: str,
    engine: str = "julia",
    domain_max: Optional[float] = None,
) -> Optional[List[dict]]:
    scenarios = {s.name: s for s in get_all_scenarios()}
    if scenario_name not in scenarios:
        print(f"Error: Scenario '{scenario_name}' not found.")
        return None

    scenario = scenarios[scenario_name]
    scenario.engine = engine

    actual_dmax = domain_max if domain_max is not None else DEFAULT_CLEAN_DOMAINS.get(
        scenario_name, DOMAIN_RESTRICTIONS.get(scenario_name, {}).get("domain_max", 1.0)
    )

    results = []

    print("\n" + "=" * 76)
    print(f" ADCD V3 NOISE ROBUSTNESS BENCHMARK : {scenario_name} (Engine: {engine})")
    print(f" Domain Range (Max): {actual_dmax}")
    print("=" * 76)

    for noise in NOISE_SWEEP:
        print(f"\n--- Testing Noise Level: {noise:.2f} (Domain Max: {actual_dmax}) ---")
        t_cfg = ScenarioThresholdConfig.for_scenario(scenario, noise_level=noise)

        identifiable_count = 0
        match_count = 0
        withheld_count = 0

        for seed in SEEDS:
            res = run_scenario_protocol(
                scenario=scenario,
                seed=seed,
                threshold_cfg=t_cfg,
                noise_level=noise,
                domain_max=actual_dmax,
            )

            primary_check = res.checks.get("primary_search", {})
            expr_str = primary_check.get("top_candidate", "None")
            nmse = primary_check.get("nmse", float("inf"))
            cls_name = primary_check.get("discovered_class", "unknown")
            match_level = primary_check.get("match_level", "none")
            is_match = match_level in ["exact", "class_only"]

            if res.all_passed:
                identifiable_count += 1
                if is_match:
                    match_count += 1
                print(f"  Seed {seed}: IDENTIFIABLE -> NMSE={nmse:.4f} | {expr_str} (Class: {cls_name}, Match: {match_level})")
            else:
                withheld_count += 1
                # Identifikasi penyebab WITHHELD
                reasons = [k for k, v in res.checks.items() if not v.get("pass", False)]
                print(f"  Seed {seed}: WITHHELD     -> NMSE={nmse:.4f} | Failed Checks: {reasons}")

        results.append({
            "noise": noise,
            "identifiable_rate": identifiable_count / len(SEEDS),
            "match_rate": match_count / len(SEEDS),
            "withheld_rate": withheld_count / len(SEEDS),
        })

    print("\n" + "-" * 76)
    print(f" NOISE ROBUSTNESS SUMMARY: {scenario_name}")
    print("-" * 76)
    print(f"{'Noise':<8} | {'Identifiable':<15} | {'True Match':<15} | {'Withheld':<10}")
    print("-" * 76)
    for r in results:
        print(
            f"{r['noise']:<8.2f} | "
            f"{r['identifiable_rate'] * 100:>12.0f}% | "
            f"{r['match_rate'] * 100:>12.0f}% | "
            f"{r['withheld_rate'] * 100:>8.0f}%"
        )
    print("-" * 76)
    return results


def main():
    parser = argparse.ArgumentParser(description="ADCD Noise Robustness Benchmark")
    parser.add_argument("--engine", type=str, default="julia", choices=["python", "julia"], help="Execution engine")
    parser.add_argument(
        "--scenario",
        type=str,
        default="all",
        choices=["all", "Time Dilation", "Screened Coulomb", "Entropy Expansion"],
        help="Scenario name or 'all'",
    )
    parser.add_argument("--domain-max", type=float, default=None, help="Custom domain max override")
    args = parser.parse_args()

    target_scenarios = (
        ["Time Dilation", "Screened Coulomb", "Entropy Expansion"]
        if args.scenario == "all"
        else [args.scenario]
    )

    all_summaries = {}
    for sc_name in target_scenarios:
        res = run_benchmark_for_scenario(sc_name, engine=args.engine, domain_max=args.domain_max)
        if res is not None:
            all_summaries[sc_name] = res

    if len(target_scenarios) > 1:
        print("\n" + "=" * 76)
        print(" ALL SCENARIOS NOISE ROBUSTNESS OVERVIEW")
        print("=" * 76)
        print(f"{'Scenario':<20} | {'Noise':<6} | {'Identifiable':<13} | {'True Match':<13} | {'Withheld':<9}")
        print("-" * 76)
        for sc_name, results in all_summaries.items():
            for r in results:
                print(
                    f"{sc_name:<20} | "
                    f"{r['noise']:<6.2f} | "
                    f"{r['identifiable_rate'] * 100:>10.0f}% | "
                    f"{r['match_rate'] * 100:>10.0f}% | "
                    f"{r['withheld_rate'] * 100:>7.0f}%"
                )
        print("-" * 76)


if __name__ == "__main__":
    main()
