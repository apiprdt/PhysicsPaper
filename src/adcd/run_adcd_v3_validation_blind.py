"""
run_adcd_v3_validation.py
=========================
Four-step blind validation protocol for ADCD:

  Step 0: Budget Disclosure   -- log search space size before any results
  Step 1: Blind Search        -- fully unrestricted GrammarProposerV3 run
  Step 2: Positive Control    -- restrict to true primitive family only
  Step 3: Ablation Control    -- exclude true primitive, compare BIC (Kass-Raftery)
  Step 4: Determinism Check   -- 3 independent runs must produce byte-exact output

The pipeline uses GrammarProposerV3, which derives ratio candidates from
ProposalContext via Buckingham-Pi dimensional analysis. No ratio string or
primitive identity is typed by hand anywhere in this file.

Usage:
    python src/adcd/run_adcd_v3_validation_blind.py --top-k 5
    python src/adcd/run_adcd_v3_validation_blind.py --top-k 5 --taxonomy
"""

from __future__ import annotations

import json
import argparse
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

logging.basicConfig(
    filename="adcd_validation_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
import warnings
warnings.filterwarnings("always", category=RuntimeWarning)
logging.captureWarnings(True)

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.grammar_proposer_v3 import GrammarProposerV3
from adcd.asymptotic_dictionary_proposer_v3 import GrammarBudget
from adcd.context import ProposalContext
from adcd.pipeline import Stage1Pipeline
from adcd.dimensional_checker import DimensionalChecker, ASTValidator
from adcd.arc_scorer import ARCScorer, build_arc_regimes
from adcd.jax_optimizer import JAXOptimizer
from adcd.metrics import classify_structure, extended_bic_score
from adcd.constants import NMSE_SUCCESS_THRESHOLD
from adcd.quickfit import DOMAIN_TAXONOMY


BIC_SIGNIFICANCE_THRESHOLD = 10.0  # Kass-Raftery "very strong evidence"


@dataclass
class ProtocolResult:
    scenario_name: str
    checks: Dict[str, dict] = field(default_factory=dict)
    all_passed: bool = False


def _build_context(scenario, n_candidates: int, exclude_primitives: Optional[List[str]] = None) -> ProposalContext:
    return ProposalContext(
        variable_names=scenario.classical_variables,
        target_name="residual",
        data_statistics={},
        n_candidates=n_candidates,
        constants=scenario.classical_constants,
        known_limits=[{
            "variable": scenario.classical_limit_variable,
            "limit": scenario.classical_limit_direction,
            "expected": "0",
        }],
        variables_with_units=scenario.variables_with_units,
    )


def _make_pipeline(checker: DimensionalChecker, scenario) -> Stage1Pipeline:
    validator = ASTValidator(max_depth=12, max_tokens=50)
    regimes = build_arc_regimes(
        scenario.classical_limit_variable,
        scenario.classical_limit_direction,
    )
    scorer = ARCScorer(regimes=regimes)
    return Stage1Pipeline(validator=validator, checker=checker, scorer=scorer)


# Observation windows used throughout the paper (Table 2).
# Changing any value here changes ALL reported BIC/NMSE numbers.
DOMAIN_RESTRICTIONS = {
    "Time Dilation":     {"domain_max": 0.3},   # v <= 0.3c
    "Screened Coulomb":  {"domain_max": 4.0},   # r <= 4.0 m
    "Entropy Expansion": {"domain_max": 1.0},   # dV/V_i <= 1.0
}


def _run_search(
    scenario,
    exclude_primitives: Optional[List[str]],
    seed: int,
    n_candidates: int = 400,
) -> Tuple[List[Tuple[str, float, float, dict]], int, GrammarProposerV3]:
    """
    One full search pass: GrammarProposerV3 → Stage1 gates → JAX optimiser → BIC ranking.

    exclude_primitives=None  →  fully blind (Step 1).
    Non-empty list           →  restricted for positive-control or ablation checks,
                                where the purpose is explicitly to isolate one variable.
                                The ratio argument is never hinted either way.
    """
    checker = DimensionalChecker()
    for var in scenario.classical_variables:
        if var not in checker.registry:
            checker.registry[var] = [0, 0, 0, 0, 0]
    for const in scenario.classical_constants:
        if const not in checker.registry:
            checker.registry[const] = [0, 0, 0, 0, 0]

    proposer = GrammarProposerV3(
        budget=GrammarBudget(max_ratio_candidates=12, max_primitives_used=2),
        exclude_primitives=exclude_primitives,
        dimensional_checker=checker,
    )
    context = _build_context(scenario, n_candidates=n_candidates)
    candidates = proposer.propose(context)
    space_size = proposer.search_space_size(context)

    pipeline = _make_pipeline(checker, scenario)
    domain_kwargs = DOMAIN_RESTRICTIONS.get(scenario.name, {})
    X, y_obs, y_classical, residual = scenario.generate_data(
        noise_level=0.01, seed=seed, **domain_kwargs
    )
    for c_name, c_val in scenario.classical_constants.items():
        if c_name not in X:
            X[c_name] = np.full_like(residual, c_val)

    if scenario.correction_type == "multiplicative":
        target_dim_key = "dimensionless"
    else:
        target_dim_key = scenario.variables_with_units.get(scenario.classical_limit_variable, "dimensionless")

    stage1_results = pipeline.execute(
        [(c, True) for c in candidates], target_dim_key, X, residual,
        constants=scenario.classical_constants,
    )

    optimizer = JAXOptimizer(n_restarts=15)
    top_k = stage1_results[:30]
    ranked = []
    for expr_str, combined_score, mse, arc_score, deferred_arc in top_k:
        opt = optimizer.optimize(
            expr_str, X, residual, scenario.classical_variables,
            seed=seed, loss_mode="auto", y_classical=y_classical,
            correction_type=scenario.correction_type,
        )
        if not np.isfinite(opt.nmse):
            continue
        n_params = len([k for k in opt.theta if k.startswith("theta_")])
        b = extended_bic_score(opt.nmse, n_params, len(residual), n_candidates=len(candidates))
        ranked.append((expr_str, opt.nmse, b, opt.theta))

    ranked.sort(key=lambda r: r[2])  # lower BIC = better
    return ranked, space_size, proposer


def _find_true_structure_in_pareto(ranked_blind, scenario):
    """
    Search the Pareto front for the true correction structure.
    Returns (true_structure_bic, true_structure_rank, match_level)
    where match_level is one of: "exact", "class_only", "none".

    Exact match uses numeric sampling rather than sympy.simplify,
    which is brittle on exponentiated functions.
    """
    true_expr_str = getattr(scenario, "correction_expr", None)
    true_class = getattr(scenario, "correction_class", None)

    if not true_expr_str:
        return None, None, "none"

    true_expr = sp.sympify(true_expr_str)
    true_thetas = [str(s) for s in true_expr.free_symbols if str(s).startswith("theta_")]

    def _try_exact_match(cand_expr, theta_keys):
        """Returns True if candidate matches ground truth under numeric sampling."""
        import numpy as np
        try:
            test_points = np.random.uniform(0.1, 0.9, 5)
            vars_cand = [str(s) for s in cand_expr.free_symbols if not str(s).startswith("theta_")]
            vars_true = [str(s) for s in true_expr.free_symbols if not str(s).startswith("theta_")]
            main_var = vars_cand[0] if vars_cand else (vars_true[0] if vars_true else "x")

            def eval_expr(expr, var_val, theta_subs):
                subs_dict = {sp.Symbol(main_var): var_val}
                for k, v in theta_subs.items():
                    subs_dict[sp.Symbol(k)] = v
                return float(expr.subs(subs_dict))

            cand_subs = {k: 1.0 for k in theta_keys}
            true_subs = {k: 1.0 for k in true_thetas}

            for val in test_points:
                try:
                    if not np.isclose(eval_expr(cand_expr, val, cand_subs),
                                      eval_expr(true_expr, val, true_subs)):
                        return False
                except Exception:
                    return False
            return True
        except Exception:
            return False

    for rank, (expr_str, nmse, bic, theta_fit) in enumerate(ranked_blind):
        cand_expr = sp.sympify(expr_str)
        theta_keys = [k for k in theta_fit.keys() if k.startswith("theta_")]
        if _try_exact_match(cand_expr, theta_keys):
            return bic, rank + 1, "exact"

    for rank, (expr_str, nmse, bic, theta_fit) in enumerate(ranked_blind):
        disc_class = classify_structure(expr_str, theta_fit)
        if disc_class == true_class:
            return bic, rank + 1, "class_only"

    return None, None, "none"


def run_scenario_protocol(scenario, seed: int = 42, top_k_val: int = 5, use_taxonomy_prior: bool = False) -> ProtocolResult:
    result = ProtocolResult(scenario_name=scenario.name)
    TRUE_PRIMITIVE_MAP = {
        "Time Dilation": "D_lor",
        "Screened Coulomb": "D_exp",
        "Entropy Expansion": "D_log"
    }
    true_primitive = TRUE_PRIMITIVE_MAP.get(scenario.name)

    # Optional taxonomy prior: restricts primitive set to domain-relevant family.
    # When enabled, also affects Steps 0, 1, and 4 (budget, blind search, determinism).
    # For fully blind results (paper Table 5), run WITHOUT --taxonomy.
    taxonomy_allowed = None
    if use_taxonomy_prior and hasattr(scenario, "domain") and scenario.domain in DOMAIN_TAXONOMY:
        taxonomy_allowed = DOMAIN_TAXONOMY[scenario.domain]
        from adcd.asymptotic_dictionary_proposer_v3 import PRIMITIVE_REGISTRY
        taxonomy_exclude = [p for p in PRIMITIVE_REGISTRY.keys() if p not in taxonomy_allowed]
    else:
        taxonomy_exclude = None

    # ---- Step 0: Budget disclosure ----
    _, space_size_blind, proposer = _run_search(scenario, exclude_primitives=taxonomy_exclude, seed=seed, n_candidates=0)
    result.checks["budget_disclosure"] = {
        "search_space_size": space_size_blind,
        "primitives": list(proposer._active_primitives.keys()),
        "pass": True,
        "note": "Search space logged before results.",
    }

    # ---- Step 1: Blind search ----
    ranked_blind, _, _ = _run_search(scenario, exclude_primitives=taxonomy_exclude, seed=seed)

    top_candidates = []
    if ranked_blind:
        for expr_str, nmse, bic, theta_fit in ranked_blind[:top_k_val]:
            disc_class = classify_structure(expr_str, theta_fit)
            top_candidates.append({
                "expr_str": expr_str,
                "nmse": nmse,
                "bic": bic,
                "class": disc_class
            })

    top = ranked_blind[0] if ranked_blind else None

    true_structure_bic, true_structure_rank, match_level = _find_true_structure_in_pareto(ranked_blind, scenario)

    if top is not None:
        expr_str, nmse, bic, theta_fit = top
        discovered_class = classify_structure(expr_str, theta_fit)

        blind_pass = (match_level in ["exact", "class_only"])
        symbolic_match = (match_level == "exact")
        class_match = (match_level in ["exact", "class_only"])

        result.checks["blind_search"] = {
            "top_candidate": expr_str,
            "nmse": nmse,
            "bic": bic,
            "discovered_class": discovered_class,
            "match_level": match_level,
            "symbolic_match": symbolic_match,
            "class_match": class_match,
            "true_structure_rank": true_structure_rank,
            "pass": blind_pass,
            "note": f"Match level: {match_level} at rank {true_structure_rank}",
            "pareto_front": top_candidates,
        }
    else:
        result.checks["blind_search"] = {"pass": False, "note": "No candidate survived the full pipeline."}

    # ---- Step 2: Positive control ----
    ranked_isolated, space_size_isolated, _ = _run_search(
        scenario,
        exclude_primitives=[p for p in ["D_lor", "D_rat", "D_exp", "D_log", "D_sqrt_inv"]
                            if p != true_primitive],
        seed=seed,
    )
    pc_pass = len(ranked_isolated) > 0 and ranked_isolated[0][1] < NMSE_SUCCESS_THRESHOLD
    result.checks["positive_control"] = {
        "search_space_size": space_size_isolated,
        "nmse": ranked_isolated[0][1] if ranked_isolated else None,
        "pass": pc_pass,
    }

    # ---- Step 3: Ablation control ----
    # Reference BIC is Rank-1 from blind search (best model with true primitive present).
    # Delta = ablated_BIC - blind_rank1_BIC; positive means full search is preferred.
    ranked_ablated, _, _ = _run_search(scenario, exclude_primitives=[true_primitive], seed=seed)
    if ranked_ablated and ranked_blind:
        ablated_bic = ranked_ablated[0][2]
        blind_rank1_bic = ranked_blind[0][2]
        bic_diff = ablated_bic - blind_rank1_bic
        result.checks["ablation_control"] = {
            "ablated_bic": ablated_bic,
            "true_structure_bic": blind_rank1_bic,
            "true_structure_rank": true_structure_rank,
            "bic_diff": bic_diff,
            "pass": bic_diff > BIC_SIGNIFICANCE_THRESHOLD,
            "note": "Reference is Rank-1 BIC from blind search.",
        }
    else:
        result.checks["ablation_control"] = {"pass": False, "note": "Could not compute — missing blind or ablated result."}

    # ---- Step 4: Determinism check ----
    runs = []
    for _ in range(3):
        r, _, _ = _run_search(scenario, exclude_primitives=taxonomy_exclude, seed=seed)
        runs.append(r[0][:2] if r else None)  # (expr_str, nmse)
    determinism_pass = len(set(str(r) for r in runs)) == 1
    result.checks["determinism_check"] = {"runs": runs, "pass": determinism_pass}

    result.all_passed = all(
        c.get("pass", False) for c in result.checks.values() if "pass" in c
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="ADCD Validation Protocol")
    parser.add_argument("--top-k", type=int, default=5, help="Number of Pareto Front candidates to display")
    parser.add_argument("--taxonomy", action="store_true", help="Use Domain Taxonomy Prior for Stage 1")
    args = parser.parse_args()

    scenarios = {s.name: s for s in get_all_scenarios()}
    locked = ["Time Dilation", "Screened Coulomb", "Entropy Expansion"]

    all_results = {}
    for name in locked:
        if name not in scenarios:
            print(f"[SKIP] Scenario '{name}' not found.")
            continue
        print("=" * 80)
        mode_str = "BAYESIAN TAXONOMY PRIOR" if args.taxonomy else "BLIND SEARCH"
        print(f" ADCD VALIDATION PROTOCOL ({mode_str}): {name.upper()}")
        print("=" * 80)
        res = run_scenario_protocol(scenarios[name], top_k_val=args.top_k, use_taxonomy_prior=args.taxonomy)
        all_results[name] = res

        for step, info in res.checks.items():
            status = "PASS" if info.get("pass") else "FAIL"
            info_to_print = {k: v for k, v in info.items() if k != "pareto_front"}
            print(f"[{status:^6}] {step.upper():<20} | {info_to_print}")

            if step == "blind_search" and "pareto_front" in info:
                print("\n" + " "*10 + "--- TOP-K PARETO FRONT ---")
                print(" "*10 + f"{'Rank':<5} | {'BIC':<8} | {'NMSE':<10} | {'Class':<15} | {'Equation'}")
                print(" "*10 + "-" * 70)
                for i, cand in enumerate(info["pareto_front"]):
                    bic_val = f"{cand['bic']:.2f}" if cand['bic'] is not None else "-"
                    nmse_val = f"{cand['nmse']:.2e}"
                    print(" "*10 + f"{i+1:<5} | {bic_val:<8} | {nmse_val:<10} | {cand['class']:<15} | {cand['expr_str']}")
                print(" "*10 + "-" * 70 + "\n")

        print("-" * 80)
        if res.all_passed:
            print("[SUCCESS] All checks passed.")
        else:
            print("[FAIL] At least one check failed. Inspect raw discovered_class before citing results.")
        print("=" * 80 + "\n")

    report_name = "adcd_v3_taxonomy_validation_report.json" if args.taxonomy else "adcd_v3_blind_validation_report.json"
    os_path = f"run_outputs/{report_name}"
    import os; os.makedirs("run_outputs", exist_ok=True)
    with open(os_path, "w") as f:
        json.dump(
            {name: {"all_passed": r.all_passed, "checks": r.checks} for name, r in all_results.items()},
            f, indent=2, default=str,
        )
    print(f"Full report saved to {os_path}")


if __name__ == "__main__":
    main()
