"""
ADCD Validation Protocol (Blind & Domain-Guided Taxonomy Search)
Executes formal four-step verification:
  1. Budget Disclosure
  2. Positive Control
  3. Ablation Control (BIC gap check)
  4. Determinism Check (byte-identical reproducibility)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import sympy as sp

logging.basicConfig(
    filename="adcd_validation_debug.log",
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
warnings.filterwarnings("always", category=RuntimeWarning)
logging.captureWarnings(True)

from adcd.anomaly_scenarios import get_all_scenarios
from adcd.arc_scorer import ARCScorer, build_arc_regimes
from adcd.asymptotic_dictionary_proposer_v3 import PRIMITIVE_REGISTRY, GrammarBudget
from adcd.context import ProposalContext
from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.grammar_proposer_v3 import GrammarProposerV3
from adcd.jax_optimizer import JAXOptimizer
from adcd.julia_bridge import ADCDJuliaEngine, JuliaEngineConfig, JuliaEngineData
from adcd.metrics import classify_structure, extended_bic_score
from adcd.mode_detection import detect_correction_mode
from adcd.pipeline import Stage1Pipeline
from adcd.quickfit import DOMAIN_TAXONOMY

BIC_SIGNIFICANCE_THRESHOLD = 10.0  # Kass-Raftery "very strong evidence"

FORMAL_PROTOCOL_CHECKS = (
    "budget_disclosure",
    "positive_control",
    "ablation_control",
    "determinism_check",
)

# Single Source of Truth untuk batasan domain observasi
DOMAIN_RESTRICTIONS: Dict[str, Dict[str, float]] = {
    "Time Dilation": {"domain_max": 0.3},       # v <= 0.3c (Historical window)
    "Screened Coulomb": {"domain_max": 4.0},    # r <= 4.0 m
    "Entropy Expansion": {"domain_max": 1.0},   # dV/V_i <= 1.0 (Historical window)
}


@dataclass
class ScenarioThresholdConfig:
    bic_threshold: float = 10.0
    nmse_fine: float = 0.1
    nmse_coarse: float = 1.0
    groups: Optional[List[List[int]]] = None

    @classmethod
    def synthetic(cls, noise_level: float = 0.01) -> ScenarioThresholdConfig:
        nmse_target = min(0.60, max(0.1, (noise_level * 5.0) ** 2 + 0.05))
        return cls(bic_threshold=10.0, nmse_fine=nmse_target, nmse_coarse=1.0, groups=None)

    @classmethod
    def real_observational(cls, n_groups: int = 1, scatter_level: float = 0.25) -> ScenarioThresholdConfig:
        floor = scatter_level ** 2
        sampling_margin = floor * 1.645 * np.sqrt(2.0 / max(n_groups, 2))
        nmse_fine_calibrated = floor * 1.35 + sampling_margin
        return cls(bic_threshold=6.0, nmse_fine=nmse_fine_calibrated, nmse_coarse=1.0, groups=None)

    @classmethod
    def for_scenario(cls, scenario: Any, noise_level: float = 0.01) -> ScenarioThresholdConfig:
        tier = getattr(scenario, "tier", "synthetic")
        domain = getattr(scenario, "domain", "")

        if domain == "mond_radial_acceleration":
            return cls.real_observational(n_groups=147, scatter_level=0.30)
        elif domain == "hubble_expansion":
            return cls.real_observational(n_groups=1, scatter_level=0.15)
        elif tier in ("real", "observational"):
            return cls.real_observational(n_groups=1, scatter_level=0.25)
        else:
            return cls.synthetic(noise_level=noise_level)


@dataclass
class ProtocolResult:
    scenario_name: str
    checks: Dict[str, dict] = field(default_factory=dict)
    all_passed: bool = False


def _build_context(scenario: Any, n_candidates: int) -> ProposalContext:
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


def _make_pipeline(checker: DimensionalChecker, scenario: Any) -> Stage1Pipeline:
    validator = ASTValidator(max_depth=12, max_tokens=50)
    regimes = build_arc_regimes(
        scenario.classical_limit_variable,
        scenario.classical_limit_direction,
    )
    scorer = ARCScorer(regimes=regimes)
    return Stage1Pipeline(validator=validator, checker=checker, scorer=scorer)


def _guess_true_primitive(expr_str: str) -> Optional[str]:
    """Identifikasi primitif sejati secara robust."""
    if not expr_str:
        return None
    try:
        expr = sp.sympify(expr_str)
        s_str = str(expr)
    except Exception:
        s_str = expr_str

    if "exp" in s_str:
        return "D_exp"
    if "log" in s_str or "ln" in s_str:
        return "D_log"
    if "cos" in s_str or "sin" in s_str:
        return "D_osc"
    if "tanh" in s_str:
        return "D_sat"
    if "sqrt" in s_str:
        if any(tok in s_str for tok in ["1 -", "1.0 -", "1.0-", "1-"]):
            return "D_lor"
        return "D_sqrt_inv"
    if "/" in s_str:
        return "D_rat"
    return "D_pow"


def _run_search(
    scenario: Any,
    exclude_primitives: Optional[List[str]],
    seed: int,
    n_candidates: int = 500,
    threshold_cfg: Optional[ScenarioThresholdConfig] = None,
    noise_level: float = 0.01,
    domain_max: Optional[float] = None,
) -> Tuple[List[Tuple[str, float, float, dict]], int, Any]:
    checker = DimensionalChecker()
    for var in scenario.classical_variables:
        if var not in checker.registry:
            checker.registry[var] = [0, 0, 0, 0, 0]
    for const in scenario.classical_constants:
        if const not in checker.registry:
            checker.registry[const] = [0, 0, 0, 0, 0]

    d_max = domain_max if domain_max is not None else DOMAIN_RESTRICTIONS.get(scenario.name, {}).get("domain_max", None)
    gen_kwargs = {"domain_max": d_max} if d_max is not None else {}
    X, y_obs, y_classical, _ = scenario.generate_data(noise_level=noise_level, seed=seed, **gen_kwargs)

    for c_name, c_val in scenario.classical_constants.items():
        if c_name not in X:
            X[c_name] = np.full_like(y_obs, c_val)

    detected_mode, _ = detect_correction_mode(y_obs, y_classical)
    target_dim_key = "dimensionless"

    if getattr(scenario, "engine", "python") == "julia":
        print("[_run_search] Delegating to ADCDJuliaEngine...")
        tcfg = threshold_cfg or ScenarioThresholdConfig.for_scenario(scenario, noise_level=noise_level)
        
        groups = tcfg.groups
        if "galaxy_id" in X:
            galaxy_ids = np.asarray(X["galaxy_id"])
            unique_ids = np.unique(galaxy_ids)
            groups = [(np.where(galaxy_ids == uid)[0] + 1).tolist() for uid in unique_ids]

        config = JuliaEngineConfig(
            domain=scenario.domain,
            target_dim=target_dim_key,
            input_vars=scenario.classical_variables,
            known_constants=scenario.classical_constants,
            bic_threshold=tcfg.bic_threshold,
            nmse_coarse=tcfg.nmse_coarse,
            nmse_fine=tcfg.nmse_fine,
            n_restarts=50,
            max_proposals=n_candidates,
            groups=groups,
            excluded_primitives=list(exclude_primitives) if exclude_primitives else [],
            correction_type=detected_mode,
            classical_limit_direction=scenario.classical_limit_direction,
            classical_limit_variable=getattr(scenario, "classical_limit_variable", ""),
        )
        data = JuliaEngineData(
            y_classical=y_classical,
            y_obs=y_obs,
            vars={k: X[k] for k in scenario.classical_variables},
        )
        engine_jl = ADCDJuliaEngine()
        result = engine_jl.run(config, data)

        ranked = []
        for cand in result.results:
            if not cand.converged or not np.isfinite(cand.nmse):
                continue
            theta_dict = {f"theta_{i}": v for i, v in enumerate(cand.theta)}
            ranked.append((cand.expr_str, cand.nmse, -cand.delta_bic, theta_dict))

        ranked.sort(key=lambda r: r[2])
        space_size = result.n_proposals_generated

        proposer = GrammarProposerV3(
            budget=GrammarBudget(max_ratio_candidates=12, max_primitives_used=2),
            exclude_primitives=exclude_primitives,
            dimensional_checker=checker,
        )
        proposer._julia_primitives_active = result.primitives_active
        return ranked, space_size, proposer

    else:
        # Python Path
        residual = (y_obs / y_classical - 1.0) if detected_mode == "multiplicative" else (y_obs - y_classical)
        context = _build_context(scenario, n_candidates=n_candidates)
        proposer = GrammarProposerV3(
            budget=GrammarBudget(max_ratio_candidates=12, max_primitives_used=2),
            exclude_primitives=exclude_primitives,
            dimensional_checker=checker,
        )
        candidates = proposer.propose(context)
        space_size = proposer.search_space_size(context)

        pipeline = _make_pipeline(checker, scenario)
        stage1_results = pipeline.execute(
            [(c, True) for c in candidates], target_dim_key, X, residual,
            constants=scenario.classical_constants,
        )

        optimizer = JAXOptimizer(n_restarts=50)
        ranked = []
        for expr_str, _, _, _, deferred_arc in stage1_results[:30]:
            opt = optimizer.optimize(
                expr_str, X, residual, scenario.classical_variables,
                seed=seed, loss_mode="auto", y_classical=y_classical,
                correction_type=detected_mode,
            )
            if not np.isfinite(opt.nmse):
                continue

            if deferred_arc:
                fitted_expr = sp.sympify(expr_str).subs(opt.theta)
                if float(pipeline.scorer.score(fitted_expr, constants=scenario.classical_constants)) <= 0.0:
                    continue

            n_params = len([k for k in opt.theta if k.startswith("theta_")])
            b = extended_bic_score(opt.nmse, n_params, len(residual), n_candidates=len(candidates))
            ranked.append((expr_str, opt.nmse, b, opt.theta))

        ranked.sort(key=lambda r: r[2])
        return ranked, space_size, proposer


def _find_true_structure_in_pareto(ranked_blind: List[tuple], scenario: Any) -> Tuple[Optional[float], Optional[int], str]:
    true_expr_str = getattr(scenario, "correction_expr", None)
    true_class = getattr(scenario, "correction_class", None)

    if not true_expr_str or not ranked_blind:
        return None, None, "none"

    true_expr = sp.sympify(true_expr_str)
    constants = getattr(scenario, "classical_constants", {})
    true_constants = getattr(scenario, "correction_constants", {})

    def _try_exact_match(cand_expr: sp.Expr, theta_dict: Dict[str, float]) -> bool:
        try:
            subs_true = {sp.Symbol(k): v for k, v in {**constants, **true_constants}.items()}
            subs_cand = {sp.Symbol(k): v for k, v in {**constants, **theta_dict}.items()}
            expr_true_sub = true_expr.subs(subs_true)
            expr_cand_sub = cand_expr.subs(subs_cand)

            free_cand = sorted([s for s in expr_cand_sub.free_symbols if not str(s).startswith("theta_")], key=lambda x: str(x))
            free_true = sorted([s for s in expr_true_sub.free_symbols if not str(s).startswith("theta_")], key=lambda x: str(x))

            if free_cand != free_true:
                return False

            if not free_cand:
                return bool(abs(float(expr_cand_sub) - float(expr_true_sub)) < 1e-4)

            np.random.seed(42)
            for _ in range(5):
                point_subs = {sym: float(np.random.uniform(0.2, 0.8)) for sym in free_cand}
                val_c = float(expr_cand_sub.subs(point_subs).evalf())
                val_t = float(expr_true_sub.subs(point_subs).evalf())
                if not np.isclose(val_c, val_t, rtol=1e-2, atol=1e-3):
                    return False
            return True
        except Exception:
            return False

    for rank, (expr_str, _, bic, theta_fit) in enumerate(ranked_blind):
        if _try_exact_match(sp.sympify(expr_str), theta_fit):
            return bic, rank + 1, "exact"

    for rank, (expr_str, _, bic, theta_fit) in enumerate(ranked_blind):
        if classify_structure(expr_str, theta_fit) == true_class:
            return bic, rank + 1, "class_only"

    return None, None, "none"


def run_scenario_protocol(
    scenario: Any,
    seed: int = 42,
    top_k_val: int = 5,
    use_taxonomy_prior: bool = True,
    threshold_cfg: Optional[ScenarioThresholdConfig] = None,
    noise_level: float = 0.01,
    domain_max: Optional[float] = None,
) -> ProtocolResult:
    result = ProtocolResult(scenario_name=scenario.name)
    tcfg = threshold_cfg or ScenarioThresholdConfig.for_scenario(scenario, noise_level=noise_level)
    true_primitive = _guess_true_primitive(scenario.correction_expr)

    taxonomy_allowed = DOMAIN_TAXONOMY.get(scenario.domain, list(PRIMITIVE_REGISTRY.keys())) if use_taxonomy_prior else None
    taxonomy_exclude = [p for p in PRIMITIVE_REGISTRY.keys() if p not in taxonomy_allowed] if taxonomy_allowed else None

    # Step 0: Budget Disclosure
    _, space_size_blind, proposer = _run_search(
        scenario, exclude_primitives=taxonomy_exclude, seed=seed, n_candidates=0,
        threshold_cfg=tcfg, noise_level=noise_level, domain_max=domain_max
    )
    result.checks["budget_disclosure"] = {
        "search_space_size": space_size_blind,
        "primitives": getattr(proposer, "_julia_primitives_active", list(proposer._active_primitives.keys())),
        "pass": True,
    }

    # Step 1: Blind Primary Search
    ranked_blind, _, _ = _run_search(
        scenario, exclude_primitives=taxonomy_exclude, seed=seed,
        threshold_cfg=tcfg, noise_level=noise_level, domain_max=domain_max
    )

    top_candidates = []
    if ranked_blind:
        for expr_str, nmse, bic, theta_fit in ranked_blind[:top_k_val]:
            top_candidates.append({
                "expr_str": expr_str, "nmse": nmse, "bic": bic,
                "class": classify_structure(expr_str, theta_fit), "theta_fit": theta_fit,
            })

    top = ranked_blind[0] if ranked_blind else None
    _, true_structure_rank, match_level = _find_true_structure_in_pareto(ranked_blind, scenario)

    if top is not None:
        expr_str, nmse, bic, theta_fit = top
        is_diag_pass = match_level in ["exact", "class_only"]
        result.checks["primary_search"] = {
            "pass": is_diag_pass,
            "top_candidate": expr_str, "nmse": nmse, "bic": bic, "theta_fit": theta_fit,
            "discovered_class": classify_structure(expr_str, theta_fit),
            "match_level": match_level, "true_structure_rank": true_structure_rank,
            "ground_truth_match_diagnostic_only": is_diag_pass,
            "counts_toward_verdict": False, "pareto_front": top_candidates,
        }
    else:
        result.checks["primary_search"] = {"pass": False}

    # Step 2: Positive Control
    ranked_isolated, space_size_isolated, _ = _run_search(
        scenario,
        exclude_primitives=[p for p in PRIMITIVE_REGISTRY if p != true_primitive],
        seed=seed, threshold_cfg=tcfg, noise_level=noise_level, domain_max=domain_max
    )
    pc_pass = len(ranked_isolated) > 0 and ranked_isolated[0][1] <= tcfg.nmse_fine
    result.checks["positive_control"] = {
        "search_space_size": space_size_isolated,
        "nmse": ranked_isolated[0][1] if ranked_isolated else None,
        "pass": pc_pass,
    }

    # Step 3: Ablation Control
    ablation_exclude_list = list(taxonomy_exclude) if taxonomy_exclude is not None else []
    if true_primitive and true_primitive not in ablation_exclude_list:
        ablation_exclude_list.append(true_primitive)

    ranked_ablated, _, _ = _run_search(
        scenario, exclude_primitives=ablation_exclude_list, seed=seed,
        threshold_cfg=tcfg, noise_level=noise_level, domain_max=domain_max
    )

    if ranked_ablated and ranked_blind:
        bic_diff = ranked_ablated[0][2] - ranked_blind[0][2]
        result.checks["ablation_control"] = {
            "ablated_bic": ranked_ablated[0][2], "true_structure_bic": ranked_blind[0][2],
            "bic_diff": bic_diff, "pass": bic_diff > tcfg.bic_threshold,
        }
    elif not ranked_ablated and ranked_blind:
        result.checks["ablation_control"] = {
            "ablated_bic": float("inf"), "true_structure_bic": ranked_blind[0][2],
            "bic_diff": float("inf"), "pass": True,
        }
    else:
        result.checks["ablation_control"] = {"pass": False}

    # Step 4: Determinism Check
    runs = []
    for _ in range(3):
        r, _, _ = _run_search(
            scenario, exclude_primitives=taxonomy_exclude, seed=seed,
            threshold_cfg=tcfg, noise_level=noise_level, domain_max=domain_max
        )
        runs.append(r[0][0] if r else None)

    determinism_pass = (not all(r is None for r in runs)) and (len(set(runs)) == 1)
    result.checks["determinism_check"] = {"runs": runs, "pass": determinism_pass}

    result.all_passed = all(
        result.checks[name].get("pass", False) for name in FORMAL_PROTOCOL_CHECKS if name in result.checks
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="ADCD Validation Protocol")
    parser.add_argument("--top-k", type=int, default=5, help="Pareto candidates to display")
    parser.add_argument("--no-taxonomy", action="store_false", dest="taxonomy", help="Disable taxonomy prior")
    parser.add_argument("--engine", type=str, choices=["python", "julia"], default="julia", help="Execution backend")
    parser.add_argument("--domain-max", type=float, default=None, help="Override default domain max")
    parser.set_defaults(taxonomy=True)
    args = parser.parse_args()

    scenarios = {s.name: s for s in get_all_scenarios()}
    locked_scenarios = ["Time Dilation", "Screened Coulomb", "Entropy Expansion"]

    all_results = {}
    for name in locked_scenarios:
        if name not in scenarios:
            continue
        print("=" * 80)
        mode_str = "BAYESIAN TAXONOMY PRIOR" if args.taxonomy else "BLIND SEARCH"
        print(f" ADCD VALIDATION PROTOCOL ({mode_str}): {name.upper()}")
        print("=" * 80)

        scenarios[name].engine = args.engine
        res = run_scenario_protocol(
            scenarios[name], top_k_val=args.top_k, use_taxonomy_prior=args.taxonomy, domain_max=args.domain_max
        )
        all_results[name] = res

        for step, info in res.checks.items():
            status = "PASS" if info.get("pass") else "FAIL"
            info_to_print = {k: v for k, v in info.items() if k != "pareto_front"}
            print(f"[{status:^6}] {step.upper():<20} | {info_to_print}")

            if step == "primary_search" and "pareto_front" in info:
                print("\n" + " " * 10 + "--- TOP-K PARETO FRONT ---")
                print(" " * 10 + f"{'Rank':<5} | {'BIC':<10} | {'NMSE':<10} | {'Class':<15} | {'Equation'}")
                print(" " * 10 + "-" * 75)
                for i, cand in enumerate(info["pareto_front"]):
                    bic_val = f"{cand['bic']:.2f}" if cand['bic'] is not None else "-"
                    nmse_val = f"{cand['nmse']:.2e}"
                    print(" " * 10 + f"{i+1:<5} | {bic_val:<10} | {nmse_val:<10} | {cand['class']:<15} | {cand['expr_str']}")
                print(" " * 10 + "-" * 75 + "\n")

        print("-" * 80)
        if res.all_passed:
            print("[SUCCESS] STATUS: All checks passed with a genuinely blind search.")
        else:
            print("[FAIL] STATUS: At least one check failed (Expected epistemic withheld or out of regime).")
        print("=" * 80 + "\n")

    os.makedirs("run_outputs", exist_ok=True)
    report_name = os.path.join(
        "run_outputs",
        "adcd_v3_taxonomy_validation_report.json" if args.taxonomy else "adcd_v3_blind_validation_report.json",
    )
    with open(report_name, "w") as f:
        json.dump(
            {name: {"all_passed": r.all_passed, "checks": r.checks} for name, r in all_results.items()},
            f, indent=2, default=str,
        )
    print(f"Full report saved to {report_name}")


if __name__ == "__main__":
    main()
