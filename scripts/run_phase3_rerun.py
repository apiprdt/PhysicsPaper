"""
run_phase3_rerun.py
===================
Executes the three canonical v1 scenarios under the new Blind Three-Tier Protocol (Phase 3).
Computes actual equal-pool Delta-BIC values and generates the verdict-change table.
"""

from __future__ import annotations

import io
import json
import os
import sys

# Force UTF-8 unbuffered output regardless of Windows console codepage
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", write_through=True)
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace", write_through=True)

import numpy as np

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.metrics import classify_structure
from adcd.run_adcd_v3_validation_blind import _run_search
from adcd.validation_v2 import (
    CandidateFit,
    run_three_tier_audit,
)


def run_phase3_evaluation():
    scenarios_by_name = {s.name: s for s in get_all_scenarios()}
    target_scenarios = ["Screened Coulomb", "Entropy Expansion", "Time Dilation"]

    print("=" * 80)
    print("PHASE 3: BLIND THREE-TIER PROTOCOL EVALUATION (EQUAL POOLS)")
    print("=" * 80)

    results = {}

    for name in target_scenarios:
        sc = scenarios_by_name[name]
        print(f"\nEvaluating: {name} (seed=42)...")

        # 1. Run search on the domain pool (same pool for all candidates)
        # Using domain taxonomy prior (as in the v1 reference results)
        taxonomy_exclude = []  # In taxonomy prior mode
        from adcd.quickfit import DOMAIN_TAXONOMY
        from adcd.asymptotic_dictionary_proposer_v3 import PRIMITIVE_REGISTRY
        if sc.domain in DOMAIN_TAXONOMY:
            allowed = set(DOMAIN_TAXONOMY[sc.domain])
            taxonomy_exclude = [p for p in PRIMITIVE_REGISTRY if p not in allowed]

        ranked_cands, space_size, _ = _run_search(sc, exclude_primitives=taxonomy_exclude, seed=42)

        # 2. Compute null model (baseline with no correction)
        X, y_obs, y_cl, res = sc.generate_data(n_points=200, noise_level=0.01, seed=42)
        res_null = (y_obs - y_cl) / y_cl if sc.correction_type == "multiplicative" else (y_obs - y_cl)
        nmse_null = float(np.mean(res_null ** 2) / np.var(y_obs))
        # Null model BIC: k=0, no correction parameters
        # Under v1 extended BIC: N * ln(max(nmse, 1e-6))
        n_points = len(y_obs)
        bic_null = float(n_points * np.log(max(nmse_null, 1e-6)))

        # 3. Wrap candidates into CandidateFit
        candidate_fits = []
        for expr_str, nmse, bic, theta_fit in ranked_cands:
            cand_family = classify_structure(expr_str, theta_fit)
            k_params = len([k for k in theta_fit if k.startswith("theta_")])
            candidate_fits.append(
                CandidateFit(
                    expr=expr_str,
                    bic=bic,
                    nmse=nmse,
                    k_params=k_params,
                    family=cand_family,
                    theta_fit=theta_fit,
                )
            )

        # 4. Execute Blind Three-Tier Protocol
        protocol_res = run_three_tier_audit(
            candidates=candidate_fits,
            null_fit={"bic": bic_null, "nmse": nmse_null},
            search_space_size=space_size,
        )

        results[name] = {
            "space_size": space_size,
            "null_bic": bic_null,
            "m1": protocol_res.best_candidate.expr if protocol_res.best_candidate else None,
            "m1_family": protocol_res.best_candidate.family if protocol_res.best_candidate else None,
            "m1_bic": protocol_res.best_candidate.bic if protocol_res.best_candidate else None,
            "m2": protocol_res.rival_candidate.expr if protocol_res.rival_candidate else None,
            "m2_family": protocol_res.rival_candidate.family if protocol_res.rival_candidate else None,
            "m2_bic": protocol_res.rival_candidate.bic if protocol_res.rival_candidate else None,
            "delta_bic_null": protocol_res.delta_bic_vs_null,
            "delta_bic_rival": protocol_res.delta_bic_vs_rival,
            "verdict": protocol_res.verdict.value,
            "equiv_class_size": len(protocol_res.equivalence_class),
            "shared_series": protocol_res.shared_series_explanation,
        }

        print(f"  Result: {protocol_res.verdict.value}")
        print(f"  Delta-BIC (null -> M1): {protocol_res.delta_bic_vs_null:.2f}")
        print(f"  Delta-BIC (rival M2 -> M1): {protocol_res.delta_bic_vs_rival:.2f}")
        print(f"  Best Candidate (M1): {results[name]['m1']} ({results[name]['m1_family']})")
        print(f"  Best Rival (M2): {results[name]['m2']} ({results[name]['m2_family']})")

    # Save output JSON
    os.makedirs("reports", exist_ok=True)
    with open("reports/phase3_rerun_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    run_phase3_evaluation()
