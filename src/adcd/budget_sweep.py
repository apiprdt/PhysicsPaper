"""
budget_sweep.py
=================

FOR DIAGNOSTIC USE ONLY — do not use sweep results to select the budget used in the paper.
"""

from __future__ import annotations
import json
from dataclasses import dataclass
from typing import List, Optional
import numpy as np

DEFAULT_TOKEN_BUDGET_SWEEP = [10, 15, 20, 25, 30, 35, 40, 50]
DEFAULT_DEPTH_BUDGET_SWEEP = [7, 8, 9, 12]

@dataclass
class SweepPoint:
    max_tokens: int
    max_depth: int
    search_space_size: int
    top_candidate: Optional[str]
    nmse: Optional[float]
    bic: Optional[float]
    discovered_class: Optional[str]
    matches_ground_truth_class: Optional[bool]


def run_budget_sweep(
    scenario,
    token_budgets: List[int] = None,
    depth_budgets: List[int] = None,
    seed: int = 42,
) -> List[SweepPoint]:
    from adcd.grammar_proposer_v3 import GrammarProposerV3
    from adcd.asymptotic_dictionary_proposer_v3 import GrammarBudget
    from adcd.context import ProposalContext
    from adcd.pipeline import Stage1Pipeline
    from adcd.dimensional_checker import DimensionalChecker, ASTValidator
    from adcd.arc_scorer import ARCScorer, build_arc_regimes
    from adcd.jax_optimizer import JAXOptimizer
    from adcd.metrics import classify_structure, extended_bic_score

    token_budgets = token_budgets or DEFAULT_TOKEN_BUDGET_SWEEP
    depth_budgets = depth_budgets or DEFAULT_DEPTH_BUDGET_SWEEP

    X, y_obs, y_classical, residual = scenario.generate_data(noise_level=0.01, seed=seed)
    for c_name, c_val in scenario.classical_constants.items():
        if c_name not in X:
            X[c_name] = np.full_like(residual, c_val)
    target_dim_key = "dimensionless" if scenario.correction_type == "multiplicative" \
        else scenario.variables_with_units.get(scenario.classical_limit_variable, "dimensionless")

    true_class = getattr(scenario, "correction_class", None)

    results: List[SweepPoint] = []

    for max_depth in depth_budgets:
        for max_tokens in token_budgets:
            checker = DimensionalChecker()
            for var in scenario.classical_variables:
                checker.registry.setdefault(var, [0, 0, 0])
            for const in scenario.classical_constants:
                checker.registry.setdefault(const, [0, 0, 0])

            proposer = GrammarProposerV3(
                budget=GrammarBudget(max_ratio_candidates=8, max_primitives_used=2, max_depth=max_depth, max_tokens=max_tokens),
                dimensional_checker=checker,
            )
            context = ProposalContext(
                variable_names=scenario.classical_variables,
                target_name="residual", data_statistics={}, n_candidates=0,
                constants=scenario.classical_constants,
                known_limits=[{"variable": scenario.classical_limit_variable,
                               "limit": scenario.classical_limit_direction, "expected": "0"}],
            )
            candidates = proposer.propose(context)
            space_size = proposer.search_space_size(context)

            validator = ASTValidator(max_depth=max_depth, max_tokens=max_tokens)
            scorer = ARCScorer(regimes=build_arc_regimes(limit_variables=[scenario.classical_limit_variable]))
            pipeline = Stage1Pipeline(validator=validator, checker=checker, scorer=scorer)

            stage1 = pipeline.execute(
                [(c, True) for c in candidates], target_dim_key, X, residual,
                constants=scenario.classical_constants,
            )

            if not stage1:
                results.append(SweepPoint(max_tokens, max_depth, space_size,
                                          None, None, None, None, None))
                continue

            optimizer = JAXOptimizer(n_restarts=15)
            ranked = []
            for expr_str, _, _, _, _ in stage1[:30]:
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
            ranked.sort(key=lambda r: r[2])

            if not ranked:
                results.append(SweepPoint(max_tokens, max_depth, space_size,
                                          None, None, None, None, None))
                continue

            expr_str, nmse, bic, theta_fit = ranked[0]
            disc_class = classify_structure(expr_str, theta_fit)
            match = (disc_class == true_class) if true_class else None

            results.append(SweepPoint(
                max_tokens=max_tokens, max_depth=max_depth,
                search_space_size=space_size, top_candidate=expr_str,
                nmse=nmse, bic=bic, discovered_class=disc_class,
                matches_ground_truth_class=match,
            ))

    return results


def print_sweep_table(scenario_name: str, results: List[SweepPoint]) -> None:
    print(f"\n{'='*100}\nBUDGET SWEEP (full, unfiltered): {scenario_name}\n{'='*100}")
    print(f"{'tokens':>7} {'depth':>6} {'space':>7} {'nmse':>12} {'bic':>12} {'class':>15} {'match':>7}")
    for r in results:
        nmse_s = f"{r.nmse:.4e}" if r.nmse is not None else "FAIL"
        bic_s = f"{r.bic:.2f}" if r.bic is not None else "-"
        cls_s = r.discovered_class or "-"
        match_s = "" if r.matches_ground_truth_class is None else ("YES" if r.matches_ground_truth_class else "no")
        print(f"{r.max_tokens:>7} {r.max_depth:>6} {r.search_space_size:>7} {nmse_s:>12} {bic_s:>12} {cls_s:>15} {match_s:>7}")


if __name__ == "__main__":
    from adcd.anomaly_scenarios import get_all_scenarios

    scenarios = {s.name: s for s in get_all_scenarios()}
    all_sweeps = {}
    for name in ["Time Dilation", "Screened Coulomb", "Entropy Expansion"]:
        if name not in scenarios:
            continue
        res = run_budget_sweep(scenarios[name])
        print_sweep_table(name, res)
        all_sweeps[name] = [r.__dict__ for r in res]

    with open("budget_sweep_report.json", "w") as f:
        json.dump(all_sweeps, f, indent=2, default=str)
    print("Full sweep saved to budget_sweep_report.json")
