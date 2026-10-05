"""
rebuttal/bic_decomposition.py
==============================
Answers Reviewer Q1 and Concern C1:

    "Can you specify |C_ablated| values for each case and the unpenalized NLL?"
    "The penalty part of ΔBIC is inflated regardless of model fit."

Strategy:
    Extended_BIC = k*ln(N) - 2*ln(L̂) + 2*ln(|C|)
                 = [complexity] + [fit_quality] + [search_penalty]

    ΔBIC = ΔBIC_NLL + ΔComplexity + ΔPenalty

We extract each component separately so the reviewer can see how much of
ΔBIC is genuine fit improvement vs. how much is a penalty asymmetry artefact.

Usage:
    cd PhysicsPaper
    $env:PYTHONPATH = "src"
    py -3.11 rebuttal/bic_decomposition.py
"""

from __future__ import annotations

import io
import json
import math
import os
import sys

# Force UTF-8 output regardless of Windows console codepage
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.asymptotic_dictionary_proposer_v3 import PRIMITIVE_REGISTRY
from adcd.quickfit import DOMAIN_TAXONOMY
from adcd.run_adcd_v3_validation_blind import (
    DOMAIN_RESTRICTIONS,
    _guess_true_primitive,
    _run_search,
)

N_POINTS = 200
SEED = 42

SEP  = "=" * 70
SEP2 = "-" * 70
SEP3 = "  " + "-" * 67


def decompose_bic(nmse: float, n_params: int, n_points: int, n_candidates: int):
    """
    Returns (nll_term, complexity_term, penalty_term).

    Extended_BIC = k*ln(N) + N*ln(nmse) + 2*ln(|C|)
    nll_term       = N*ln(nmse)    [= -2*log_likelihood]
    complexity_term = k*ln(N)
    penalty_term    = 2*ln(|C|)
    """
    nmse_floored    = max(nmse, 1e-6)
    nll_term        = float(n_points * math.log(nmse_floored))
    complexity_term = float(n_params * math.log(n_points))
    penalty_term    = float(2.0 * math.log(max(1, n_candidates)))
    return nll_term, complexity_term, penalty_term


def main():
    print(SEP)
    print(" BIC DECOMPOSITION -- Rebuttal for Reviewer Q1 / Concern C1")
    print(SEP)

    report_path = os.path.join(
        os.path.dirname(__file__), "..", "run_outputs",
        "adcd_v3_taxonomy_validation_report.json"
    )
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)

    scenarios = {s.name: s for s in get_all_scenarios()}
    locked = ["Screened Coulomb", "Entropy Expansion", "Time Dilation"]

    rows = []

    for name in locked:
        sc     = scenarios[name]
        checks = report[name]["checks"]

        # ---- Domain-guided search (Rank-1) ----
        ps         = checks["primary_search"]
        nmse_best  = ps["nmse"]
        bic_best   = ps["bic"]
        theta_best = ps["theta_fit"]
        k_best     = len([k for k in theta_best if k.startswith("theta_")])
        n_cand_dom = checks["budget_disclosure"]["search_space_size"]

        nll_d, comp_d, pen_d = decompose_bic(nmse_best, k_best, N_POINTS, n_cand_dom)
        total_d    = nll_d + comp_d + pen_d
        recon_err  = abs(total_d - bic_best)
        verify_ok  = recon_err < 0.05

        # ---- Ablated search: re-run to get |C_ablated| and fit quality ----
        true_primitive = _guess_true_primitive(sc.correction_expr)

        print(f"\n[{name}] Getting ablated search space size (excludes '{true_primitive}')...")
        _, space_size_ablated, _ = _run_search(
            sc,
            exclude_primitives=[true_primitive],
            seed=SEED,
            n_candidates=0,
        )

        print(f"[{name}] Running full ablated search for nmse/k decomposition...")
        ranked_ablated, n_cand_abl, _ = _run_search(
            sc,
            exclude_primitives=[true_primitive],
            seed=SEED,
        )

        if ranked_ablated:
            _, nmse_abl, bic_abl_check, theta_abl = ranked_ablated[0]
            k_abl = len([k for k in theta_abl if k.startswith("theta_")])
        else:
            nmse_abl, k_abl = float("nan"), 0

        # Use stored ablated BIC (from original report) as the authoritative value
        ac         = checks["ablation_control"]
        bic_ablated = ac["ablated_bic"]
        bic_diff_orig = ac["bic_diff"]

        nll_a, comp_a, pen_a = decompose_bic(nmse_abl, k_abl, N_POINTS, n_cand_abl)

        # ---- Decompose ΔBIC ----
        delta_bic        = bic_ablated - bic_best
        delta_nll        = nll_a - nll_d
        delta_complexity = comp_a - comp_d
        delta_penalty    = pen_a - pen_d

        row = {
            "scenario":            name,
            "n_cand_domain":       n_cand_dom,
            "k_domain":            k_best,
            "nmse_domain":         nmse_best,
            "nll_domain":          nll_d,
            "complexity_domain":   comp_d,
            "penalty_domain":      pen_d,
            "bic_domain":          bic_best,
            "n_cand_ablated":      n_cand_abl,
            "k_ablated":           k_abl,
            "nmse_ablated":        nmse_abl,
            "nll_ablated":         nll_a,
            "complexity_ablated":  comp_a,
            "penalty_ablated":     pen_a,
            "bic_ablated":         bic_ablated,
            "delta_bic":           delta_bic,
            "delta_nll":           delta_nll,
            "delta_complexity":    delta_complexity,
            "delta_penalty":       delta_penalty,
            "delta_sum_check":     delta_nll + delta_complexity + delta_penalty,
            "verify_dom_ok":       verify_ok,
            "paper_delta_bic":     bic_diff_orig,
            "nll_exceeds_kr10":    math.isfinite(delta_nll) and delta_nll >= 10.0,
        }
        rows.append(row)

        # ---- Pretty print ----
        print(f"\n{SEP2}")
        print(f"  Scenario: {name}")
        print(SEP2)
        header = f"  {'Component':<30} {'Domain-Guided':>16} {'Ablated':>16} {'Delta':>12}"
        print(header)
        print(f"  {'-'*30} {'-'*16} {'-'*16} {'-'*12}")
        print(f"  {'|C| (search space size)':<30} {n_cand_dom:>16d} {n_cand_abl:>16d} {n_cand_abl - n_cand_dom:>+12d}")
        print(f"  {'k (free parameters)':<30} {k_best:>16d} {k_abl:>16d} {k_abl - k_best:>+12d}")
        print(f"  {'NMSE':<30} {nmse_best:>16.4e} {nmse_abl:>16.4e} {'':>12}")
        print(f"  {'-'*30} {'-'*16} {'-'*16} {'-'*12}")
        print(f"  {'NLL term  [N*ln(NMSE)]':<30} {nll_d:>16.2f} {nll_a:>16.2f} {delta_nll:>+12.2f}")
        print(f"  {'Complexity [k*ln(N)]':<30} {comp_d:>16.2f} {comp_a:>16.2f} {delta_complexity:>+12.2f}")
        print(f"  {'Penalty [2*ln(|C|)]':<30} {pen_d:>16.2f} {pen_a:>16.2f} {delta_penalty:>+12.2f}")
        print(f"  {'-'*30} {'-'*16} {'-'*16} {'-'*12}")
        print(f"  {'Extended BIC (total)':<30} {bic_best:>16.2f} {bic_ablated:>16.2f} {delta_bic:>+12.2f}")
        print(f"\n  Paper reports  dBIC = {bic_diff_orig:.2f}")
        print(f"  Decomposition: dBIC = {delta_nll:.2f} [NLL] + {delta_complexity:.2f} [complexity] + {delta_penalty:.2f} [penalty]")
        check_sum = delta_nll + delta_complexity + delta_penalty
        print(f"  Sum check:     {check_sum:.2f}  (matches paper: {abs(check_sum - bic_diff_orig) < 0.5})")
        pct = delta_penalty / delta_bic * 100 if delta_bic != 0 else float("nan")
        print(f"\n  >> dNLL (pure fit advantage, penalty-free): {delta_nll:.2f}")
        print(f"  >> dPenalty (search-space artefact):        {delta_penalty:.2f}  ({pct:.1f}% of total dBIC)")
        if delta_nll >= 10.0:
            print(f"  >> VERDICT: dNLL alone exceeds K-R threshold (10) -- verdict robust after penalty removal")
        else:
            print(f"  >> VERDICT: dNLL alone BELOW K-R threshold (10) -- verdict partially driven by penalty asymmetry")

    # ---- Save results ----
    out_path = os.path.join(
        os.path.dirname(__file__), "results", "bic_decomposition_table.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, default=str)

    print(f"\n\n{SEP}")
    print(f" Results saved to: {out_path}")
    print(SEP)

    # ---- LaTeX table ----
    print("\n\n% ---- LaTeX table for rebuttal (Table R1) ----")
    print(r"\begin{table}[ht]")
    print(r"\small")
    print(r"\caption{BIC decomposition into pure fit quality ($\Delta$NLL) and")
    print(r"multiple-testing penalty ($\Delta$Pen.) components. $\Delta$NLL$\geq$10")
    print(r"indicates the identifiability signal is robust after removing the penalty asymmetry.}")
    print(r"\label{tab:bic_decomp}")
    print(r"\centering")
    print(r"\begin{tabular}{@{}lrrrrrrc@{}}")
    print(r"\toprule")
    print(r"Scenario & $|\mathcal{C}_d|$ & $|\mathcal{C}_a|$ & "
          r"$\Delta$Pen. & $\Delta$NLL & $\Delta$BIC & $\Delta$NLL${\geq}10$? \\")
    print(r"\midrule")
    for r in rows:
        dn   = r["delta_nll"]
        ok   = r"Yes" if r["nll_exceeds_kr10"] else r"\textbf{No}"
        print(f"  {r['scenario']:<25} & {r['n_cand_domain']:3d} & {r['n_cand_ablated']:3d} & "
              f"{r['delta_penalty']:+.2f} & {dn:+.2f} & {r['delta_bic']:.2f} & {ok} \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    main()
