#!/usr/bin/env python3
"""
verify_paper_claims.py
======================
Single-source-of-truth script that reads directly from all result JSON files
and prints all headline numbers used in the paper.

Run this after any benchmark update to ensure paper claims match results.

    py -3.11 scripts/verify_paper_claims.py

Exit code 0 = all claims verified.
Exit code 1 = at least one mismatch detected.
"""

import json
import sys
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

CONVERGENCE_NMSE = 1e-5   # correction_orchestrator.py default
QUANTITATIVE_NMSE = 1e-4  # paper threshold for "quantitative success"

OK  = "[OK] "
FAIL = "[FAIL]"
MISS = "[MISSING]"


def load(path):
    p = ROOT / path
    if not p.exists():
        print(f"{MISS} {path} - run the corresponding benchmark first")
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def section(title):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


all_ok = True


# --- 1. Standard Benchmark (reproducibility_results.json, seed=42) -----------
section("Standard Benchmark -- Mock Proposer, seed=42")
repro_all = load("reproducibility_results.json")
if repro_all:
    seed42 = [r for r in repro_all if r.get("seed") == 42]
    noise_groups = {}
    for r in seed42:
        n = round(r["noise"], 3)
        noise_groups.setdefault(n, []).append(r)
    for noise, results in sorted(noise_groups.items()):
        n_match = sum(r["class_match"] for r in results)
        n_total = len(results)
        pct = 100 * n_match / n_total
        print(f"  noise={noise*100:.0f}%: {n_match}/{n_total} = {pct:.1f}%")

    n_match_all = sum(r["class_match"] for r in seed42)
    n_total_all = len(seed42)
    pct_all = 100 * n_match_all / n_total_all
    passed = (n_match_all == 34 and n_total_all == 36)
    flag = OK if passed else FAIL
    print(f"\n  {flag} seed=42 overall: {n_match_all}/{n_total_all} = {pct_all:.1f}%"
          f"  (paper claims 94.4% = 34/36)")
    all_ok &= passed


# --- 2. Multi-seed Reproducibility -------------------------------------------
section("Multi-seed Reproducibility")
if repro_all:
    seed_rates = {}
    for r in repro_all:
        s = r.get("seed")
        seed_rates.setdefault(s, []).append(r["class_match"])
    rates = [sum(v) / len(v) * 100 for v in seed_rates.values()]
    mean_rate = sum(rates) / len(rates)
    std_rate = statistics.stdev(rates)
    print(f"  Seeds tested: {sorted(seed_rates.keys())}")
    expected_by_seed = {0: 86.1, 7: 75.0, 21: 77.8, 42: 94.4, 99: 80.6}
    for seed in sorted(seed_rates.keys()):
        n_match = sum(seed_rates[seed])
        n_total = len(seed_rates[seed])
        pct = 100 * n_match / n_total
        exp = expected_by_seed.get(seed)
        if exp is not None:
            ok_seed = abs(pct - exp) < 0.15
            flag = OK if ok_seed else FAIL
            print(f"  {flag} seed={seed}: {n_match}/{n_total} = {pct:.1f}%  (paper claims {exp:.1f}%)")
            all_ok &= ok_seed
    passed = (abs(mean_rate - 82.8) < 1.5 and abs(std_rate - 7.7) < 1.0)
    flag = OK if passed else FAIL
    print(f"  {flag} Mean: {mean_rate:.1f}% +/- {std_rate:.1f}%"
          f"  (paper claims 82.8% +/- 7.7%)")
    all_ok &= passed


# --- 2b. Hybrid Proposer (seed=42, optional frozen results) ------------------
section("Hybrid Proposer -- seed=42 (supplementary)")
hybrid = load("hybrid_seed42_results.json")
if hybrid:
    n_match = sum(r["class_match"] for r in hybrid)
    n_total = len(hybrid)
    pct = 100 * n_match / n_total
    passed = (n_match == 33 and n_total == 36)
    flag = OK if passed else FAIL
    print(f"  {flag} Hybrid seed=42: {n_match}/{n_total} = {pct:.1f}%"
          f"  (paper claims 91.7% = 33/36)")
    all_ok &= passed
else:
    print(f"  {FAIL} hybrid_seed42_results.json not found."
          f"  Run: python run_correction_discovery.py --proposer hybrid")
    all_ok = False


# --- 3. Real-World Established Scenarios -------------------------------------
section("Real-World Established Scenarios (4 scenarios, excl. Binary Pulsar)")
real = load("real_data_results.json")
if real:
    established = [r for r in real if "Binary Pulsar" not in r["scenario"]]

    n_struct = sum(1 for r in established if r["class_match"])
    n_quant  = sum(1 for r in established if r["nmse_full"] < QUANTITATIVE_NMSE)
    n_conv   = sum(1 for r in established if r["nmse_residual"] < CONVERGENCE_NMSE)
    n_total  = len(established)

    for label, actual, expected in [
        ("Structural",   n_struct, 4),
        ("Quantitative", n_quant,  3),
        ("Converged",    n_conv,   2),
    ]:
        passed = (actual == expected)
        flag = OK if passed else FAIL
        print(f"  {flag} {label:<14}: {actual}/{n_total}  (paper claims {expected}/{n_total})")
        all_ok &= passed

    print(f"\n  Per-scenario breakdown:")
    for r in established:
        conv_flag   = "conv:Y " if r["nmse_residual"] < CONVERGENCE_NMSE  else "conv:N "
        quant_flag  = "quant:Y" if r["nmse_full"]     < QUANTITATIVE_NMSE else "quant:N"
        struct_flag = "struct:Y" if r["class_match"] else "struct:N"
        print(f"    {r['scenario'].replace('Real: ', ''):<25} "
              f"NMSE_res={r['nmse_residual']:.2e}  NMSE_full={r['nmse_full']:.2e}  "
              f"{struct_flag}  {quant_flag}  {conv_flag}")


# --- 4. Binary Pulsar Sensitivity -------------------------------------------
section("Binary Pulsar Sensitivity Study")
pulsar = load("binary_pulsar_sensitivity.json")
if pulsar:
    print(f"  Variants run: {len(pulsar)}")
    for r in pulsar:
        match = "Y" if r.get("class_match") else "N"
        print(f"    {r['variant']:<12} class_match={match}  "
              f"NMSE={r.get('nmse_full', float('nan')):.2e}")
    full_results = [r for r in pulsar if r["variant"] == "full"]
    if full_results:
        passed = not full_results[0]["class_match"]
        flag = OK if passed else FAIL
        print(f"  {flag} full variant fails structurally  (paper claims: yes)")
        all_ok &= passed


# --- 5. PySR Baseline --------------------------------------------------------
section("PySR Fair Baseline")
pysr = load("pysr_baseline_fair.json")
if pysr:
    noise_match = {}
    for r in pysr:
        n = round(r["noise"], 3)
        noise_match.setdefault(n, []).append(r["class_match"])
    print(f"  {'Noise':<8}  {'Match':<8}  {'Accuracy'}")
    for noise in sorted(noise_match):
        matches = noise_match[noise]
        n_m = sum(matches)
        n_t = len(matches)
        print(f"  {noise*100:.0f}%{'':<6}  {n_m}/{n_t}{'':<5}  {100*n_m/n_t:.1f}%")

    noise5_matches = sum(noise_match.get(0.05, []))
    noise5_total   = len(noise_match.get(0.05, []))
    passed = (noise5_matches == 1 and noise5_total == 9)
    flag = OK if passed else FAIL
    print(f"\n  {flag} At 5% noise: {noise5_matches}/{noise5_total} = "
          f"{100*noise5_matches/noise5_total:.1f}%  (paper claims 11.1% = 1/9)")
    all_ok &= passed

    noise0_mm = [r for r in pysr if abs(r["noise"]) < 1e-9 and not r["class_match"]]
    trig_count = sum(1 for r in noise0_mm if r["discovered_class"] == "trigonometric")
    passed = (trig_count == 5 and len(noise0_mm) == 5)
    flag = OK if passed else FAIL
    print(f"  {flag} At 0% noise: {len(noise0_mm)} mismatches, "
          f"{trig_count}/{len(noise0_mm)} trigonometric  "
          f"(paper claims all 5 are trigonometric)")
    all_ok &= passed


# --- Summary -----------------------------------------------------------------
section("Verification Summary")
if all_ok:
    print("  [ALL OK] All verified claims match result files.")
else:
    print("  [FAIL] One or more claims DO NOT match result files.")
    print("         Review mismatches above before updating the paper.")

sys.exit(0 if all_ok else 1)
