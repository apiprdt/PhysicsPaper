"""
rebuttal/generate_rebuttal_tables.py
=====================================
Aggregates results from all 3 rebuttal experiments into:
  - Formatted LaTeX tables
  - A summary JSON for the rebuttal document
  - Console printout of key numbers for copy-paste into the rebuttal text

Usage:
    cd PhysicsPaper
    $env:PYTHONPATH = "src"
    python rebuttal/generate_rebuttal_tables.py

Run AFTER all three experiment scripts have completed.
"""

from __future__ import annotations

import io
import json
import math
import os
import sys

# Force UTF-8 output regardless of Windows console codepage
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
SCENARIOS   = ["Screened Coulomb", "Entropy Expansion", "Time Dilation"]
THRESH_BIC  = 10.0


def load(filename: str) -> dict:
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        print(f"  [WARNING] {filename} not found — run the corresponding script first.")
        return {}
    with open(path) as f:
        return json.load(f)


def fmt(val, decimals: int = 2, fallback: str = "—") -> str:
    try:
        v = float(val)
        if math.isnan(v) or math.isinf(v):
            return fallback
        return f"{v:.{decimals}f}"
    except (TypeError, ValueError):
        return fallback


def main():
    print("=" * 70)
    print(" REBUTTAL TABLE GENERATOR")
    print("=" * 70)

    bic_data   = load("bic_decomposition_table.json")
    seed_data  = load("multi_seed_stability.json")
    missp_data = load("taxonomy_misspecification.json")

    summary = {}

    # =========================================================================
    # TABLE 1: BIC Decomposition (Q1 / C1)
    # =========================================================================
    print("\n\n── Table R1: BIC Decomposition ─────────────────────────────────────")
    print(f"  {'Scenario':<25} {'|C_d|':>6} {'|C_a|':>6} {'ΔPen':>8} {'ΔNLL':>8} {'ΔBIC':>8} {'NLL≥10?':>8}")
    print(f"  {'─'*25} {'─'*6} {'─'*6} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for row in (bic_data if isinstance(bic_data, list) else []):
        name    = row.get("scenario", "?")
        n_d     = row.get("n_cand_domain", 0)
        n_a     = row.get("n_cand_ablated", 0)
        dp      = row.get("delta_penalty", float("nan"))
        dn      = row.get("delta_nll", float("nan"))
        db      = row.get("delta_bic", float("nan"))
        nll_ok  = "YES ✅" if (math.isfinite(dn) and dn >= THRESH_BIC) else "NO ⚠️"
        print(f"  {name:<25} {n_d:>6d} {n_a:>6d} {fmt(dp):>8} {fmt(dn):>8} {fmt(db):>8} {nll_ok:>8}")
        summary[name] = summary.get(name, {})
        summary[name]["bic_decomp"] = {
            "n_cand_domain": n_d, "n_cand_ablated": n_a,
            "delta_penalty": dp,  "delta_nll": dn,
            "delta_bic": db,      "nll_exceeds_threshold": math.isfinite(dn) and dn >= THRESH_BIC,
        }

    if not bic_data:
        print("  (No data — run bic_decomposition.py first)")

    # LaTeX
    print("\n% LaTeX Table R1:")
    print(r"\begin{tabular}{@{}lrrrrrrc@{}}")
    print(r"\toprule")
    print(r"Scenario & $|\mathcal{C}_d|$ & $|\mathcal{C}_a|$ & "
          r"$\Delta$Pen. & $\Delta$NLL & $\Delta$BIC & $\Delta$NLL${\geq}10$? \\")
    print(r"\midrule")
    for row in (bic_data if isinstance(bic_data, list) else []):
        name    = row.get("scenario", "?")
        n_d     = row.get("n_cand_domain", 0)
        n_a     = row.get("n_cand_ablated", 0)
        dp      = fmt(row.get("delta_penalty"), 2)
        dn      = row.get("delta_nll", float("nan"))
        dn_str  = fmt(dn, 2)
        db      = fmt(row.get("delta_bic"), 2)
        nll_ok  = r"\checkmark" if (math.isfinite(dn) and dn >= THRESH_BIC) else r"\times"
        print(f"  {name} & {n_d} & {n_a} & {dp} & {dn_str} & {db} & ${nll_ok}$ \\\\")
    print(r"\bottomrule\end{tabular}")

    # =========================================================================
    # TABLE 2: Multi-Seed Stability (Q2 / C2)
    # =========================================================================
    print("\n\n── Table R2: Multi-Seed Stability ──────────────────────────────────")
    if seed_data:
        print(f"  {'Scenario':<25} {'Seeds':>6} {'Stable%':>8} {'ΔBIC min':>10} "
              f"{'ΔBIC mean±std':>18} {'ΔNLL≈ mean':>12}")
        print(f"  {'─'*25} {'─'*6} {'─'*8} {'─'*10} {'─'*18} {'─'*12}")

        for name in SCENARIOS:
            if name not in seed_data:
                continue
            s   = seed_data[name]["summary"]
            db  = s["delta_bic_stats"]
            dn  = s["delta_nll_approx_stats"]
            n   = s["n_seeds"]
            stb = s["stability_pct"]
            print(f"  {name:<25} {n:>6d} {stb:>7.0f}% "
                  f"{fmt(db['min']):>10} "
                  f"{fmt(db['mean'])}±{fmt(db['std']):>18} "
                  f"{fmt(dn['mean']):>12}")
            summary[name] = summary.get(name, {})
            summary[name]["seed_stability"] = {
                "n_seeds": n,
                "stability_pct": stb,
                "dominant_verdict": s["dominant_verdict"],
                "delta_bic_min": db["min"],
                "delta_bic_mean": db["mean"],
                "delta_nll_mean": dn["mean"],
            }

        # Per-seed detail for each scenario
        for name in SCENARIOS:
            if name not in seed_data:
                continue
            print(f"\n  {name} — per-seed detail:")
            print(f"    {'Seed':>6} {'Verdict':<16} {'NMSE':>10} {'ΔBIC':>8} {'ΔNLL≈':>8} {'PC':>5}")
            print(f"    {'─'*6} {'─'*16} {'─'*10} {'─'*8} {'─'*8} {'─'*5}")
            for r in seed_data[name]["per_seed"]:
                v_mark = "✅" if r["verdict"] == "IDENTIFIABLE" else "❌"
                print(f"    {r['seed']:>6} {r['verdict']:<14}{v_mark} "
                      f"{fmt(r['nmse_best'], 4):>10} "
                      f"{fmt(r['delta_bic']):>8} "
                      f"{fmt(r.get('delta_nll_approx')):>8} "
                      f"{'✅' if r['pc_pass'] else '❌':>5}")

        # LaTeX
        print("\n% LaTeX Table R2:")
        print(r"\begin{tabular}{@{}lrrrr@{}}")
        print(r"\toprule")
        print(r"Scenario & $n_{\text{seeds}}$ & Stability & "
              r"$\Delta\text{BIC}_{\min}$ & $\overline{\Delta\text{BIC}}$ \\")
        print(r"\midrule")
        for name in SCENARIOS:
            if name not in seed_data:
                continue
            s   = seed_data[name]["summary"]
            db  = s["delta_bic_stats"]
            print(f"  {name} & {s['n_seeds']} & "
                  f"{s['stability_pct']:.0f}\\% & "
                  f"{fmt(db['min'])} & "
                  f"{fmt(db['mean'])}$\\pm${fmt(db['std'])} \\\\")
        print(r"\bottomrule\end{tabular}")
    else:
        print("  (No data — run multi_seed_stability.py first)")

    # =========================================================================
    # TABLE 3: Taxonomy Misspecification (Q3 / C3)
    # =========================================================================
    print("\n\n── Table R3: Taxonomy Misspecification ─────────────────────────────")
    if missp_data:
        n_wrong = 0
        n_safe  = 0
        n_overfit = 0

        print(f"  {'Scenario':<25} {'Type':<9} {'Domain':<30} {'Verdict':<16} {'PC':>5} {'ΔBIC':>8}")
        print(f"  {'─'*25} {'─'*9} {'─'*30} {'─'*16} {'─'*5} {'─'*8}")
        for sc_name in SCENARIOS:
            if sc_name not in missp_data:
                continue
            for r in missp_data[sc_name]:
                is_corr = r["label"] == "correct"
                if not is_corr:
                    n_wrong += 1
                    if r["verdict"] == "WITHHELD":
                        n_safe += 1
                    else:
                        n_overfit += 1
                t_str = "CORRECT" if is_corr else f"WRONG ({r['label'][-1]})"
                v_str = r["verdict"]
                pc_str = "✅" if r.get("pc_pass") else "❌"
                print(f"  {sc_name if is_corr else '':<25} {t_str:<9} "
                      f"{r['domain']:<30} {v_str:<16} {pc_str:>5} "
                      f"{fmt(r.get('delta_bic')):>8}")

        print(f"\n  Safety: {n_safe}/{n_wrong} wrong-taxonomy inputs → WITHHELD "
              f"({'✅ All safe' if n_overfit == 0 else f'⚠️ {n_overfit} overfit(s)!'})")

        summary["misspecification_safety"] = {
            "n_wrong_tests": n_wrong,
            "n_withheld":    n_safe,
            "n_overfit":     n_overfit,
            "all_safe":      n_overfit == 0,
        }
    else:
        print("  (No data — run taxonomy_misspecification.py first)")

    # =========================================================================
    # KEY NUMBERS FOR REBUTTAL TEXT (copy-paste ready)
    # =========================================================================
    print("\n\n── Key Numbers for Rebuttal Text ───────────────────────────────────")
    print("  Copy these into your rebuttal response:")
    print()

    if isinstance(bic_data, list) and bic_data:
        for row in bic_data:
            dn = row.get("delta_nll", float("nan"))
            dp = row.get("delta_penalty", float("nan"))
            db = row.get("delta_bic", float("nan"))
            pct = (dp / db * 100) if db else float("nan")
            print(f"  {row['scenario']}:")
            print(f"    ΔBIC = {fmt(db)}  (ΔNLL={fmt(dn)}, ΔPenalty={fmt(dp)}, "
                  f"penalty is {fmt(pct, 1)}% of total)")
            pass_str = "exceeds K-R threshold independently" if (math.isfinite(dn) and dn >= 10) \
                       else "does NOT exceed K-R threshold without penalty — caveat needed"
            print(f"    ΔNLL alone: {pass_str}")

    # =========================================================================
    # Save summary
    # =========================================================================
    out_path = os.path.join(RESULTS_DIR, "rebuttal_summary.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\n\nSummary saved to: {out_path}")


if __name__ == "__main__":
    main()
