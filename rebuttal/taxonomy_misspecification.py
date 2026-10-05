"""
rebuttal/taxonomy_misspecification.py
=======================================
Answers Reviewer Q3 and Concern C3:

    "What is the impact of supplying an incorrectly formulated taxonomy key?
     Does the system produce WITHHELD or overfit and produce IDENTIFIABLE?"

Strategy:
    Run a 3x3 cross-test matrix:
    - 3 scenarios (Screened Coulomb, Entropy Expansion, Time Dilation)
    - For each: correct taxonomy + 2 wrong taxonomies

    A good epistemic safety system should produce WITHHELD for ALL wrong-taxonomy
    inputs, because the positive control gate will fail when the correct primitive
    family is excluded from the search.

Usage:
    cd PhysicsPaper
    $env:PYTHONPATH = "src"
    python rebuttal/taxonomy_misspecification.py
"""

from __future__ import annotations

import copy
import io
import json
import os
import sys
import time

# Force UTF-8 unbuffered output regardless of Windows console codepage
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.run_adcd_v3_validation_blind import run_scenario_protocol

SEED = 42

# ---------------------------------------------------------------------------
# Misspecification test matrix
# Each entry: (test_label, wrong_domain_key, description)
# ---------------------------------------------------------------------------
MISSPEC_MATRIX = {
    "Screened Coulomb": [
        ("correct",  "yukawa_debye_screening",       "Correct: D_exp + D_rat"),
        ("wrong_A",  "wave_resonance",               "Wrong A: D_osc + D_rat (oscillatory)"),
        ("wrong_B",  "critical_scaling",             "Wrong B: D_pow only (power law)"),
    ],
    "Entropy Expansion": [
        ("correct",  "boltzmann_thermodynamics",     "Correct: D_exp + D_log"),
        ("wrong_A",  "lorentz_special_relativity",   "Wrong A: D_lor only (Lorentz)"),
        ("wrong_B",  "wave_resonance",               "Wrong B: D_osc + D_rat (oscillatory)"),
    ],
    "Time Dilation": [
        ("correct",  "lorentz_special_relativity",   "Correct: D_lor only"),
        ("wrong_A",  "boltzmann_thermodynamics",     "Wrong A: D_exp + D_log (thermal)"),
        ("wrong_B",  "yukawa_debye_screening",       "Wrong B: D_exp + D_rat (screening)"),
    ],
}


def run_one_test(scenario, domain_override: str, label: str, description: str, test_num: int = 1, total_tests: int = 9) -> dict:
    """
    Run the full 4-step protocol on `scenario` but with `domain_override`
    as the taxonomy key (which may be wrong).
    """
    sc_copy = copy.deepcopy(scenario)
    sc_copy.domain = domain_override

    t0 = time.time()
    now_str = time.strftime("%H:%M:%S")
    print(f"\n[{now_str}] >>> [Test {test_num}/{total_tests}] {scenario.name} | [{label}] domain='{domain_override}' — {description}", flush=True)
    try:
        result = run_scenario_protocol(sc_copy, seed=SEED, use_taxonomy_prior=True)
    except AssertionError as e:
        # Unknown domain key → treat as total failure
        print(f"      !! AssertionError (unknown domain key): {e}", flush=True)
        return {
            "label":         label,
            "domain":        domain_override,
            "description":   description,
            "error":         str(e),
            "verdict":       "ERROR",
            "all_passed":    False,
            "pc_pass":       False,
            "pc_nmse":       None,
            "ablation_pass": False,
            "delta_bic":     None,
            "top_expr":      None,
            "primitives":    [],
            "elapsed_s":     round(time.time() - t0, 1),
        }

    checks  = result.checks
    verdict = "IDENTIFIABLE" if result.all_passed else "WITHHELD"

    ps          = checks.get("primary_search", {})
    top_expr    = ps.get("top_candidate", "")
    nmse_best   = ps.get("nmse", None)

    pc          = checks.get("positive_control", {})
    pc_pass     = pc.get("pass", False)
    pc_nmse     = pc.get("nmse", None)

    ac          = checks.get("ablation_control", {})
    abl_pass    = ac.get("pass", False)
    delta_bic   = ac.get("bic_diff", None)

    primitives  = checks.get("budget_disclosure", {}).get("primitives", [])
    elapsed_s   = round(time.time() - t0, 1)

    v_sym = "✅ IDENTIFIABLE" if verdict == "IDENTIFIABLE" else "❌ WITHHELD"
    now_end = time.strftime("%H:%M:%S")
    print(f"[{now_end}] <<< [Test {test_num}/{total_tests}] Verdict: {v_sym} | PC={'PASS' if pc_pass else 'FAIL'} "
          f"(NMSE={f'{pc_nmse:.3e}' if pc_nmse else 'N/A'}) | "
          f"ΔBIC={f'{delta_bic:.2f}' if delta_bic else 'N/A'} | Time: {elapsed_s:.1f}s ({elapsed_s/60:.1f}m)", flush=True)
    print(f"      Primitives used: {primitives}", flush=True)
    if top_expr:
        print(f"      Top expression: {top_expr[:80]}...", flush=True)

    return {
        "label":         label,
        "domain":        domain_override,
        "description":   description,
        "verdict":       verdict,
        "all_passed":    result.all_passed,
        "pc_pass":       pc_pass,
        "pc_nmse":       pc_nmse,
        "nmse_best":     nmse_best,
        "ablation_pass": abl_pass,
        "delta_bic":     delta_bic,
        "top_expr":      top_expr,
        "primitives":    primitives,
        "elapsed_s":     elapsed_s,
    }


def main():
    print("=" * 70, flush=True)
    print(" TAXONOMY MISSPECIFICATION — Rebuttal for Reviewer Q3 / C3", flush=True)
    print("=" * 70, flush=True)

    scenarios   = {s.name: s for s in get_all_scenarios()}
    all_results = {}

    total_tests = sum(len(tests) for tests in MISSPEC_MATRIX.values())
    test_counter = 0

    checkpoint_path = os.path.join(
        os.path.dirname(__file__), "results", "taxonomy_misspec_checkpoint.json"
    )
    os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

    if os.path.exists(checkpoint_path):
        try:
            with open(checkpoint_path, "r", encoding="utf-8") as f_chk:
                all_results = json.load(f_chk)
            completed_count = sum(len(v) for v in all_results.values())
            print(f"[Resume] Loaded checkpoint from {checkpoint_path} ({completed_count} tests already completed)", flush=True)
        except Exception:
            all_results = {}

    for scenario_name, tests in MISSPEC_MATRIX.items():
        sc = scenarios[scenario_name]
        print(f"\n{'─'*70}", flush=True)
        print(f"  Scenario: {scenario_name}", flush=True)
        print(f"{'─'*70}", flush=True)

        rows = []
        for label, domain_key, description in tests:
            test_counter += 1

            # Check if this exact test is already completed in checkpoint
            already_done = None
            if scenario_name in all_results:
                for r in all_results[scenario_name]:
                    if r.get("label") == label and r.get("domain") == domain_key:
                        already_done = r
                        break
            if already_done is not None:
                v_sym = "✅ IDENTIFIABLE" if already_done.get("verdict") == "IDENTIFIABLE" else "❌ WITHHELD"
                print(f"[{time.strftime('%H:%M:%S')}] >>> [Test {test_counter}/{total_tests}] {scenario_name} | [{label}] domain='{domain_key}' — ALREADY DONE: {v_sym} (skipping)", flush=True)
                rows.append(already_done)
                continue

            row = run_one_test(sc, domain_key, label, description, test_num=test_counter, total_tests=total_tests)
            rows.append(row)
            all_results[scenario_name] = rows

            # Save checkpoint after every single test!
            with open(checkpoint_path, "w", encoding="utf-8") as f_chk:
                json.dump(all_results, f_chk, indent=2, default=str)
            print(f"      [Checkpoint Saved] -> {checkpoint_path} ({test_counter}/{total_tests} completed)", flush=True)

        all_results[scenario_name] = rows

    # --- Save final results ---
    out_path = os.path.join(
        os.path.dirname(__file__), "results", "taxonomy_misspecification.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n[Final Results Saved] -> {out_path}", flush=True)

    # --- Summary table ---
    print(f"\n\n{'='*70}")
    print(f" MISSPECIFICATION SUMMARY TABLE")
    print(f"{'='*70}")
    print(f"  {'Scenario':<25} {'Taxonomy Key':<35} {'Verdict':<16} {'PC':<6} {'ΔBIC':>8}")
    print(f"  {'─'*25} {'─'*35} {'─'*16} {'─'*6} {'─'*8}")
    for sc_name, rows in all_results.items():
        for r in rows:
            is_correct = r["label"] == "correct"
            prefix     = "  " if is_correct else "  ↳"
            v_str      = r["verdict"]
            pc_str     = "PASS" if r.get("pc_pass") else "FAIL"
            dbic_str   = f"{r['delta_bic']:.2f}" if r.get("delta_bic") else "N/A"
            mark       = " ✅" if (is_correct and v_str == "IDENTIFIABLE") else \
                         " ⚠️" if (not is_correct and v_str == "IDENTIFIABLE") else \
                         " ✅" if (not is_correct and v_str == "WITHHELD") else ""
            print(f"{prefix} {sc_name if is_correct else '':<25} "
                  f"{r['domain']:<35} {v_str:<16} {pc_str:<6} {dbic_str:>8}{mark}")

    # --- Safety check ---
    print(f"\n\n  ── Safety Assessment ──────────────────────────────────────")
    n_wrong_tests    = 0
    n_wrong_withheld = 0
    n_wrong_overfit  = 0

    for sc_name, rows in all_results.items():
        for r in rows:
            if r["label"] != "correct":
                n_wrong_tests += 1
                if r["verdict"] == "WITHHELD":
                    n_wrong_withheld += 1
                else:
                    n_wrong_overfit += 1
                    print(f"  ⚠️  OVERFIT DETECTED: {sc_name} + domain='{r['domain']}' → {r['verdict']}")
                    print(f"     This means the system falsely claimed IDENTIFIABLE with wrong physics!")

    if n_wrong_overfit == 0:
        print(f"\n  ✅ Epistemic safety holds: All {n_wrong_tests} wrong-taxonomy tests → WITHHELD")
        print(f"     The system correctly refuses to overclaim when given wrong physics priors.")
    else:
        print(f"\n  ⚠️  Safety breach: {n_wrong_overfit}/{n_wrong_tests} wrong-taxonomy tests → IDENTIFIABLE")
        print(f"     These cases require deeper investigation and disclosure in the paper.")

    print(f"\nFull results saved to: {out_path}")

    # --- LaTeX table ---
    print("\n\n% ---- LaTeX table for rebuttal ----")
    print(r"\begin{table}[ht]")
    print(r"\small")
    print(r"\caption{Taxonomy misspecification results. Wrong-taxonomy inputs should")
    print(r"produce WITHHELD (epistemic safety holds) rather than IDENTIFIABLE (overfit).}")
    print(r"\label{tab:misspec}")
    print(r"\centering")
    print(r"\begin{tabular}{@{}llllrr@{}}")
    print(r"\toprule")
    print(r"Scenario & Taxonomy key & Primitives & Verdict & PC & $\Delta$BIC \\")
    print(r"\midrule")
    for sc_name, rows in all_results.items():
        first = True
        for r in rows:
            sc_label = sc_name if first else ""
            first    = False
            is_corr  = r["label"] == "correct"
            v_str    = (r"{\textbf{IDENTIFIABLE}}" if r["verdict"] == "IDENTIFIABLE"
                        else r"WITHHELD")
            pc_str   = "PASS" if r.get("pc_pass") else "FAIL"
            dbic_str = f"{r['delta_bic']:.2f}" if r.get("delta_bic") else "--"
            prim_str = ",".join(r.get("primitives", [])) or "--"
            mark     = r"$\leftarrow$ correct" if is_corr else ""
            print(f"  {sc_label:<20} & {r['domain']:<30} & {prim_str:<15} & "
                  f"{v_str} & {pc_str} & {dbic_str} {mark} \\\\")
        print(r"  \midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
