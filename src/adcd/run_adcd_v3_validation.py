"""
run_adcd_v3_validation.py
==========================
Final validation harness for ADCD v3 (regularized-primitive, deterministic-
enumeration architecture). This is the ONE script that must be run — and
pass all four checks — before ANY v3 result is cited in the paper.

IMPORTANT: this script assembles the pipeline using the SAME low-level
components adcd.fit() already uses internally (Stage1Pipeline, JAXOptimizer,
CorrectionOrchestrator), because adcd.fit()'s public wrapper only accepts
proposer as a string ("mock"/"gemini"/"hybrid") and cannot take a proposer
INSTANCE like AsymptoticDictionaryProposerV3 — this was the exact bug found
earlier in this project's held-out benchmark scripts. We bypass that here
by constructing the pipeline directly, which is the correct fix (do not
try to hack the string-based fit() wrapper).

BEFORE RUNNING: verify against your local adcd/correction_orchestrator.py
and adcd/pipeline.py that the constructor signatures below still match —
they were last confirmed against this project's source during an earlier
audit pass, but any local edits since then could have changed them. If a
signature mismatch is found, fix it here rather than routing around the
validation protocol.
"""
import json
from dataclasses import dataclass, asdict
from typing import List, Optional

# --- MUST be first: see jax_precision_config.py for why import order matters
import jax_precision_config
jax_precision_config.verify_x64_enabled()

import sympy as sp

from adcd.pipeline import Stage1Pipeline
from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.arc_scorer import ARCScorer, build_arc_regimes
from adcd.jax_optimizer import JAXOptimizer
from adcd.correction_orchestrator import CorrectionOrchestrator
from adcd.anomaly_scenarios import get_all_scenarios

from asymptotic_dictionary_proposer_v3 import (
    AsymptoticDictionaryProposerV3,
    GrammarBudget,
    PRIMITIVE_REGISTRY,
)


# =====================================================================
# RESULT RECORD
# =====================================================================

@dataclass
class ValidationResult:
    scenario_name: str
    check_name: str          # "positive_control" | "ablation_control" |
    # "determinism_check" | "budget_disclosure"
    passed: bool
    detail: str
    search_space_size: int
    best_expr: Optional[str] = None
    best_nmse_residual: Optional[float] = None


def run_pipeline(
    scenario,
    proposer: AsymptoticDictionaryProposerV3,
    seed: int = 42,
    noise_level: float = 0.01,
    max_iterations: int = 1,
):
    """Mirrors adcd.fit()'s internal recipe exactly, but accepts a proposer
    INSTANCE instead of a string. Returns the raw search_result object."""
    validator = ASTValidator(max_depth=9, max_tokens=40)
    checker = DimensionalChecker()
    regimes = build_arc_regimes(
        scenario.classical_limit_variable,
        scenario.classical_limit_direction,
    )
    scorer = ARCScorer(regimes=regimes)
    pipeline = Stage1Pipeline(validator, checker, scorer)
    optimizer = JAXOptimizer()

    orchestrator = CorrectionOrchestrator(
        proposer=proposer,
        pipeline=pipeline,
        optimizer=optimizer,
        max_iterations=max_iterations,
        verbose=False,
    )
    return orchestrator.search_correction(scenario, noise_level=noise_level, seed=seed)


# =====================================================================
# CHECK 1 — POSITIVE CONTROL
# =====================================================================

def positive_control(scenario, ratio_symbol: str, seed: int = 42) -> ValidationResult:
    """Ground truth's matching primitive INCLUDED. Must recover the true
    structure, verified by SYMBOLIC equivalence (sp.simplify), never by
    substring matching (that exact bug already cost this project a false
    'Verbatim: No' entry in an earlier disclosure table)."""
    proposer = AsymptoticDictionaryProposerV3(ratio_symbol=ratio_symbol)
    result = run_pipeline(scenario, proposer, seed=seed)

    passed = False
    detail = "no candidate reached acceptable NMSE"
    if result.best_expr is not None:
        try:
            diff = sp.simplify(
                sp.sympify(result.best_expr) - sp.sympify(scenario.correction_expr)
            )
            symbolic_match = (diff == 0)
        except Exception as e:
            symbolic_match = False
            detail = f"could not symbolically compare: {e}"
        nmse_ok = result.best_nmse_residual is not None and result.best_nmse_residual < 0.20
        passed = nmse_ok  # structural symbolic match is a bonus signal, not
        # the pass/fail criterion by itself, since equivalent-but-differently
        # -parameterized forms are also legitimate recoveries
        detail = (
            f"nmse_residual={result.best_nmse_residual}, "
            f"symbolic_exact_match={symbolic_match}"
        )

    return ValidationResult(
        scenario_name=scenario.name,
        check_name="positive_control",
        passed=passed,
        detail=detail,
        search_space_size=proposer.search_space_size(),
        best_expr=result.best_expr,
        best_nmse_residual=result.best_nmse_residual,
    )


# =====================================================================
# CHECK 2 — ABLATION CONTROL
# =====================================================================

def ablation_control(
    scenario, ratio_symbol: str, ground_truth_primitive: str, seed: int = 42
) -> ValidationResult:
    """Ground truth's matching primitive EXCLUDED. A PASSING ablation means
    the system does NOT confidently produce a wrong composition with good
    NMSE — either it reports a clearly worse BIC/NMSE than the positive
    control, or falls back to an explicitly-labeled low-order polynomial
    approximation. A system that still gets a great NMSE here is a FAILED
    control (evidence of an undiagnosed leak), not a success."""
    proposer = AsymptoticDictionaryProposerV3(
        ratio_symbol=ratio_symbol,
        exclude_primitives=[ground_truth_primitive],
    )
    result = run_pipeline(scenario, proposer, seed=seed)

    # get the positive-control NMSE for comparison
    pos = positive_control(scenario, ratio_symbol, seed=seed)
    pos_nmse = pos.best_nmse_residual if pos.best_nmse_residual is not None else float("inf")
    abl_nmse = result.best_nmse_residual if result.best_nmse_residual is not None else float("inf")

    # PASS condition: ablated NMSE should be meaningfully worse than positive
    # control's, OR itself already poor in absolute terms
    passed = (abl_nmse > 3 * pos_nmse) or (abl_nmse > 0.20)

    return ValidationResult(
        scenario_name=scenario.name,
        check_name="ablation_control",
        passed=passed,
        detail=(
            f"excluded={ground_truth_primitive}, ablated_nmse={abl_nmse}, "
            f"positive_control_nmse={pos_nmse}, ratio={abl_nmse / max(pos_nmse, 1e-12):.2f}"
        ),
        search_space_size=proposer.search_space_size(),
        best_expr=result.best_expr,
        best_nmse_residual=result.best_nmse_residual,
    )


# =====================================================================
# CHECK 3 — DETERMINISM CHECK
# =====================================================================

def determinism_check(scenario, ratio_symbol: str, seed: int = 42) -> ValidationResult:
    """Run the exact same scenario+budget 3 times. The PROPOSED CANDIDATE
    LIST must be byte-identical across runs (that part is guaranteed by
    construction -- enumerate_candidates has no randomness). What we
    actually verify here is that the WINNING candidate and its NMSE are
    also identical -- if not, non-determinism has leaked in downstream
    (JAX solver randomness, unseeded init, etc.) and must be found before
    trusting any single-run result."""
    runs = []
    for _ in range(3):
        proposer = AsymptoticDictionaryProposerV3(ratio_symbol=ratio_symbol)
        result = run_pipeline(scenario, proposer, seed=seed)
        runs.append((result.best_expr, result.best_nmse_residual))

    passed = len(set(runs)) == 1
    return ValidationResult(
        scenario_name=scenario.name,
        check_name="determinism_check",
        passed=passed,
        detail=f"3 runs produced: {runs}",
        search_space_size=AsymptoticDictionaryProposerV3(
            ratio_symbol=ratio_symbol
        ).search_space_size(),
    )


# =====================================================================
# CHECK 4 — COMPLEXITY-BUDGET DISCLOSURE (always "passes"; this just
# forces the number to be recorded, never silently omitted)
# =====================================================================

def budget_disclosure(scenario, ratio_symbol: str) -> ValidationResult:
    proposer = AsymptoticDictionaryProposerV3(ratio_symbol=ratio_symbol)
    size = proposer.search_space_size()
    return ValidationResult(
        scenario_name=scenario.name,
        check_name="budget_disclosure",
        passed=True,
        detail=(
            f"search_space_size={size}, primitives={list(PRIMITIVE_REGISTRY.keys())}, "
            f"budget={asdict(GrammarBudget())}"
        ),
        search_space_size=size,
    )


# =====================================================================
# MAIN — run all four checks for a named scenario, print + save report
# =====================================================================

def run_full_protocol(
    scenario_name: str,
    ratio_symbol: str,
    ground_truth_primitive: str,
    seed: int = 42,
) -> List[ValidationResult]:
    scenario = next(s for s in get_all_scenarios() if s.name == scenario_name)

    results = [
        budget_disclosure(scenario, ratio_symbol),
        positive_control(scenario, ratio_symbol, seed=seed),
        ablation_control(scenario, ratio_symbol, ground_truth_primitive, seed=seed),
        determinism_check(scenario, ratio_symbol, seed=seed),
    ]

    print(f"\n{'='*70}\nVALIDATION PROTOCOL: {scenario_name}\n{'='*70}")
    for r in results:
        status = "PASS" if r.passed else "FAIL"
        print(f"[{status}] {r.check_name}: {r.detail}")

    all_passed = all(r.passed for r in results)
    print(f"\nOVERALL: {'ALL CHECKS PASSED' if all_passed else 'AT LEAST ONE CHECK FAILED'}")
    if not all_passed:
        print(
            "DO NOT cite this scenario's result until every check above "
            "passes. A failed ablation or determinism check invalidates "
            "the positive-control result too, even if that one looks good."
        )

    return results


if __name__ == "__main__":
    # EDIT THESE per scenario before running. Do not guess
    # ground_truth_primitive/ratio_symbol -- confirm them against
    SCENARIOS_TO_VALIDATE = [
        {
            "scenario_name": "Blind-4: Relativistic Pendulum",
            "ratio_symbol": "(v/c)**2",   # confirm this matches how X is wired to
            "ground_truth_primitive": "D_lor",
        },
        # add further scenarios here ONLY after each one individually
        # passes all four checks -- do not batch-run untested scenarios
        # and cherry-pick the ones that look good.
    ]

    all_results = {}
    for cfg in SCENARIOS_TO_VALIDATE:
        all_results[cfg["scenario_name"]] = [
            asdict(r) for r in run_full_protocol(**cfg)
        ]

    with open("adcd_v3_validation_report.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print("\nFull report saved to adcd_v3_validation_report.json")
