"""
rebuttal/multi_seed_stability.py
=================================
Answers Reviewer Q2, Q5 and Concern C2:

    "Are the IDENTIFIABLE/WITHHELD labels stable over 5-10 noise realizations?"
    "Multi-seed sensitivity analysis and tease apart ΔBIC penalty from NLL."

Strategy:
    Run the full 4-step protocol (with taxonomy prior) across 10 seeds.
    For each seed, record: verdict, NMSE, ΔBIC, and full BIC decomposition.
    Compute stability metrics (% consistent verdict, NMSE CV%, ΔBIC min/mean/std).

Usage:
    cd PhysicsPaper
    $env:PYTHONPATH = "src"
    python rebuttal/multi_seed_stability.py

    # Optionally run a quick smoke test with fewer seeds:
    python rebuttal/multi_seed_stability.py --quick
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import sys
import time

import numpy as np

# Force UTF-8 unbuffered output regardless of Windows console codepage
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.asymptotic_dictionary_proposer_v3 import PRIMITIVE_REGISTRY
from adcd.quickfit import DOMAIN_TAXONOMY
from adcd.run_adcd_v3_validation_blind import (
    _guess_true_primitive,
    _run_search,
    run_scenario_protocol,
)

SEEDS_FULL  = [0, 42, 99, 256, 512]   # 5 seeds — balanced coverage, includes paper's original seed
SEEDS_QUICK = [0, 42, 99]             # smoke test: 3 seeds only
N_POINTS    = 200


def decompose_bic(nmse: float, n_params: int, n_points: int, n_candidates: int):
    nmse_floored   = max(nmse, 1e-6)
    nll_term       = float(n_points * math.log(nmse_floored))
    complexity_term = float(n_params * math.log(n_points))
    penalty_term    = float(2.0 * math.log(max(1, n_candidates)))
    return nll_term, complexity_term, penalty_term


def run_one_seed(scenario, seed: int, true_primitive: str, n_cand_ablated_ref: int):
    """
    Run full protocol for one scenario+seed.
    Returns a dict of all relevant metrics.
    """
    t0 = time.time()
    result = run_scenario_protocol(scenario, seed=seed, use_taxonomy_prior=True)
    elapsed = time.time() - t0

    checks  = result.checks
    verdict = "IDENTIFIABLE" if result.all_passed else "WITHHELD"

    # Primary search
    ps         = checks.get("primary_search", {})
    nmse_best  = ps.get("nmse", float("nan"))
    bic_best   = ps.get("bic", float("nan"))
    theta_best = ps.get("theta_fit", {})
    k_best     = len([k for k in theta_best if k.startswith("theta_")])
    top_expr   = ps.get("top_candidate", "")
    n_cand_dom = checks.get("budget_disclosure", {}).get("search_space_size", 1)

    # Positive control
    pc_nmse = checks.get("positive_control", {}).get("nmse", float("nan"))
    pc_pass = checks.get("positive_control", {}).get("pass", False)

    # Ablation control
    ac        = checks.get("ablation_control", {})
    bic_abl   = ac.get("ablated_bic", float("nan"))
    delta_bic = ac.get("bic_diff", float("nan"))
    abl_pass  = ac.get("pass", False)

    # BIC decomposition
    if math.isfinite(nmse_best) and math.isfinite(bic_best):
        nll_d, comp_d, pen_d = decompose_bic(nmse_best, k_best, N_POINTS, n_cand_dom)
    else:
        nll_d = comp_d = pen_d = float("nan")

    # For ablated BIC we use the reference space size gathered in main()
    # (running full ablated search per seed is too expensive; we approximate
    # using the reference n_cand_ablated which is seed-independent since
    # the grammar is deterministic given the taxonomy exclude list)
    if math.isfinite(bic_abl) and not math.isnan(bic_abl):
        # Reverse-engineer ablated NMSE from Extended BIC:
        # bic_abl = n_points * ln(nmse_abl) + k_abl * ln(N) + 2*ln(|C_abl|)
        # We cannot perfectly reverse without k_abl, so we record delta directly
        pen_a = float(2.0 * math.log(max(1, n_cand_ablated_ref)))
        delta_penalty = pen_a - pen_d if math.isfinite(pen_d) else float("nan")
        delta_nll_approx = delta_bic - delta_penalty if math.isfinite(delta_penalty) else float("nan")
    else:
        delta_penalty    = float("nan")
        delta_nll_approx = float("nan")

    return {
        "seed":            seed,
        "verdict":         verdict,
        "elapsed_s":       round(elapsed, 1),
        # Primary
        "nmse_best":       nmse_best,
        "bic_best":        bic_best,
        "top_expr":        top_expr,
        "k_domain":        k_best,
        "n_cand_domain":   n_cand_dom,
        "nll_domain":      nll_d,
        "complexity_domain": comp_d,
        "penalty_domain":  pen_d,
        # Positive control
        "pc_nmse":         pc_nmse,
        "pc_pass":         pc_pass,
        # Ablation
        "delta_bic":       delta_bic,
        "bic_ablated":     bic_abl,
        "ablation_pass":   abl_pass,
        "n_cand_ablated":  n_cand_ablated_ref,
        "penalty_ablated": float(2.0 * math.log(max(1, n_cand_ablated_ref))),
        "delta_penalty_approx": delta_penalty,
        "delta_nll_approx":     delta_nll_approx,
    }


def compute_summary(per_seed_rows: list[dict]) -> dict:
    verdicts = [r["verdict"] for r in per_seed_rows]
    n_total  = len(verdicts)
    n_ident  = verdicts.count("IDENTIFIABLE")
    n_with   = verdicts.count("WITHHELD")

    nmses    = [r["nmse_best"] for r in per_seed_rows if math.isfinite(r["nmse_best"])]
    dbics    = [r["delta_bic"] for r in per_seed_rows if math.isfinite(r["delta_bic"])]
    dnlls    = [r["delta_nll_approx"] for r in per_seed_rows if math.isfinite(r["delta_nll_approx"])]

    def safe_stats(vals):
        if not vals:
            return dict(mean=float("nan"), std=float("nan"), min=float("nan"),
                        max=float("nan"), cv_pct=float("nan"))
        arr = np.array(vals)
        m   = float(np.mean(arr))
        s   = float(np.std(arr))
        return dict(
            mean   = round(m, 6),
            std    = round(s, 6),
            min    = round(float(np.min(arr)), 6),
            max    = round(float(np.max(arr)), 6),
            cv_pct = round(s / abs(m) * 100 if m != 0 else float("nan"), 2),
        )

    return {
        "n_seeds":          n_total,
        "n_identifiable":   n_ident,
        "n_withheld":       n_with,
        "stability_pct":    round(max(n_ident, n_with) / n_total * 100, 1),
        "dominant_verdict": "IDENTIFIABLE" if n_ident >= n_with else "WITHHELD",
        "nmse_stats":       safe_stats(nmses),
        "delta_bic_stats":  safe_stats(dbics),
        "delta_nll_approx_stats": safe_stats(dnlls),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="Run with 3 seeds only (quick smoke test)")
    args = parser.parse_args()

    seeds = SEEDS_QUICK if args.quick else SEEDS_FULL
    print("=" * 70)
    print(f" MULTI-SEED STABILITY — Rebuttal for Reviewer Q2 / C2")
    print(f" Seeds: {seeds}  ({'QUICK' if args.quick else 'FULL'})")
    print("=" * 70)

    scenarios      = {s.name: s for s in get_all_scenarios()}
    locked         = ["Screened Coulomb", "Entropy Expansion", "Time Dilation"]
    all_results    = {}

    for name in locked:
        sc = scenarios[name]
        true_primitive  = _guess_true_primitive(sc.correction_expr)
        domain_prims    = DOMAIN_TAXONOMY.get(sc.domain, list(PRIMITIVE_REGISTRY.keys()))
        taxonomy_exclude = [p for p in PRIMITIVE_REGISTRY if p not in domain_prims]

        # Pre-compute ablated space size once (grammar is deterministic, seed-independent)
        print(f"\n[{name}] Pre-computing ablated search space size...")
        _, n_cand_ablated_ref, _ = _run_search(
            sc, exclude_primitives=[true_primitive], seed=42, n_candidates=0
        )
        print(f"  |C_ablated| = {n_cand_ablated_ref}")

        per_seed = []
        for i, seed in enumerate(seeds):
            print(f"\n[{name}] Seed {seed} ({i+1}/{len(seeds)})...")
            row = run_one_seed(sc, seed, true_primitive, n_cand_ablated_ref)
            per_seed.append(row)

            v_sym = "✅" if row["verdict"] == "IDENTIFIABLE" else "❌"
            pc_sym = "✅" if row["pc_pass"] else "❌"
            abl_sym = "✅" if row["ablation_pass"] else "❌"
            print(f"  Verdict: {row['verdict']} {v_sym} | "
                  f"PC {pc_sym} (NMSE={row['pc_nmse']:.3e}) | "
                  f"Abl {abl_sym} (ΔBIC={row['delta_bic']:.2f}) | "
                  f"Time: {row['elapsed_s']:.0f}s", flush=True)

            # Incremental checkpoint after every seed!
            chk_data = dict(all_results)
            chk_data[name] = {"per_seed": per_seed}
            chk_path = os.path.join(os.path.dirname(__file__), "results", "multi_seed_checkpoint.json")
            os.makedirs(os.path.dirname(chk_path), exist_ok=True)
            with open(chk_path, "w", encoding="utf-8") as f_chk:
                json.dump(chk_data, f_chk, indent=2, default=str)
            print(f"      [Checkpoint Saved] -> {chk_path} ({name}: seed {seed} recorded)", flush=True)

        summary = compute_summary(per_seed)
        all_results[name] = {
            "true_primitive":      true_primitive,
            "n_cand_ablated_ref":  n_cand_ablated_ref,
            "per_seed":            per_seed,
            "summary":             summary,
        }

        print(f"\n  ── {name} Summary ──────────────────────────────────────")
        print(f"  Verdict stability: {summary['stability_pct']:.0f}% "
              f"({summary['n_identifiable']}×IDENTIFIABLE, "
              f"{summary['n_withheld']}×WITHHELD out of {summary['n_seeds']})")
        print(f"  NMSE:   mean={summary['nmse_stats']['mean']:.3e} ± "
              f"{summary['nmse_stats']['std']:.3e}  "
              f"(CV={summary['nmse_stats']['cv_pct']:.1f}%)")
        print(f"  ΔBIC:   mean={summary['delta_bic_stats']['mean']:.2f} ± "
              f"{summary['delta_bic_stats']['std']:.2f}  "
              f"min={summary['delta_bic_stats']['min']:.2f}  "
              f"max={summary['delta_bic_stats']['max']:.2f}")
        print(f"  ΔNLL≈:  mean={summary['delta_nll_approx_stats']['mean']:.2f} ± "
              f"{summary['delta_nll_approx_stats']['std']:.2f}  "
              f"(penalty removed)")

        if summary["delta_bic_stats"]["min"] > 10:
            print(f"  ✅ ΔBIC > 10 for ALL seeds — verdict robust")
        elif summary["delta_bic_stats"]["max"] > 10:
            print(f"  ⚠️  ΔBIC dips below 10 on some seeds — partial robustness")
        else:
            print(f"  ❌ ΔBIC < 10 on most/all seeds — verdict not robust")

    # --- Save results ---
    out_path = os.path.join(
        os.path.dirname(__file__), "results", "multi_seed_stability.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # --- Final consolidated table ---
    print(f"\n\n{'='*70}")
    print(f" CONSOLIDATED STABILITY SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Scenario':<25} {'Stability%':>11} {'ΔBIC min':>10} "
          f"{'ΔBIC mean':>10} {'ΔNLL≈ mean':>11}")
    print(f"  {'─'*25} {'─'*11} {'─'*10} {'─'*10} {'─'*11}")
    for name in locked:
        s  = all_results[name]["summary"]
        db = s["delta_bic_stats"]
        dn = s["delta_nll_approx_stats"]
        print(f"  {name:<25} {s['stability_pct']:>10.0f}% "
              f"{db['min']:>10.2f} {db['mean']:>10.2f} "
              f"{dn['mean']:>11.2f}")

    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()
