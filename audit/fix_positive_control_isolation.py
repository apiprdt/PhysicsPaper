"""
fix_positive_control_isolation.py
====================================
CONFIRMED BUG (from cross-referencing adcd_v3_validation_report.json and
identifiability_sweep_report_*.json against the actual source code):

positive_control() in run_adcd_v3_validation.py did this:

    proposer = AsymptoticDictionaryProposerV3(ratio_symbol=ratio_symbol)
    proposer.budget.allowed_primitives = [ground_truth_primitive]

GrammarBudget has no `allowed_primitives` field. Python silently created an
unused instance attribute -- enumerate_candidates() never reads it. So
"positive control" was actually running the FULL 5-primitive search, not a
search isolated to the ground-truth primitive.

DIRECT EVIDENCE: the reported Blind-4 "positive control" winning expression

    1.11e-17*theta_25*v**2*(theta_26*(-1.0 + 1.0/(1.0 - 1.11e-17*v**2)) + 1.0)
      / (sqrt(1.0 - 1.11e-17*v**2)*(sqrt(1.0 - 1.11e-17*v**2) + 1.0))

contains BOTH the D_lor rationalized form (the sqrt(1-u)*(sqrt(1-u)+1)
denominator) AND the D_rat form (1.0/(1.0-u) inside the parentheses) at
once -- proof the search was never restricted to D_lor alone.

CONSEQUENCE: every "ratio = ablated_nmse / positive_control_nmse" reported
so far compares "full 5-primitive pool" vs "full pool minus one primitive"
-- NOT "isolated ground truth" vs "ground truth removed", which is what the
protocol document (Section: MANDATORY VALIDATION PROTOCOL) actually
specifies. This explains the repeated EXACT ratio=1.0 entries in both
identifiability_sweep_report_D_lor_ablated.json and
..._D_rat_ablated.json: whenever the unrestricted search's BIC-preferred
candidate happens not to use the "ablated" primitive at all, positive and
ablated runs converge on the identical expression by construction, not by
coincidence.

FIX: replace the no-op attribute assignment with exclude_primitives set to
every OTHER primitive -- the mechanism that is actually implemented and
already used correctly by ablation_control().

ACTION REQUIRED: every ratio reported in
  - adcd_v3_validation_report.json
  - identifiability_sweep_report_D_lor_ablated.json
  - identifiability_sweep_report_D_rat_ablated.json
must be treated as INVALID until re-run with the fixed function below.
Do not average, cite, or plot the old numbers alongside new ones -- rerun
the full sweep from scratch with this fix and report only the new numbers.
"""

import sympy as sp

from run_adcd_v3_validation import (
    ValidationResult,
    run_pipeline,
)
from asymptotic_dictionary_proposer_v3 import (
    AsymptoticDictionaryProposerV3,
    PRIMITIVE_REGISTRY,
)


def positive_control_fixed(
    scenario, ratio_symbol: str, ground_truth_primitive: str, seed: int = 42
) -> ValidationResult:
    """
    Correct positive control: ONLY the ground-truth primitive is available
    to the search (every other primitive excluded). This is the genuine
    "if the true structure is available in isolation, is it recovered"
    test -- the isolation that the original code's `allowed_primitives`
    line intended but failed to implement.
    """
    all_other_primitives = [p for p in PRIMITIVE_REGISTRY if p != ground_truth_primitive]
    if not all_other_primitives:
        raise ValueError(
            f"'{ground_truth_primitive}' not found in PRIMITIVE_REGISTRY "
            f"(available: {list(PRIMITIVE_REGISTRY.keys())}) -- check spelling "
            f"before trusting any result."
        )

    proposer = AsymptoticDictionaryProposerV3(
        ratio_symbol=ratio_symbol,
        exclude_primitives=all_other_primitives,
    )
    result = run_pipeline(scenario, proposer, seed=seed)

    passed = False
    detail = "no candidate reached acceptable NMSE"
    symbolic_match = False
    if result.best_expr is not None:
        try:
            diff = sp.simplify(
                sp.sympify(result.best_expr) - sp.sympify(scenario.correction_expr)
            )
            symbolic_match = (diff == 0)
        except Exception as e:
            detail = f"could not symbolically compare: {e}"

        nmse_ok = result.best_nmse_residual is not None and result.best_nmse_residual < 0.20
        passed = nmse_ok
        detail = (
            f"nmse_residual={result.best_nmse_residual}, "
            f"symbolic_exact_match={symbolic_match}, "
            f"TRULY isolated to '{ground_truth_primitive}' "
            f"({len(all_other_primitives)} other primitives excluded, "
            f"search_space_size={proposer.search_space_size()})"
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


def ablation_control_fixed(
    scenario, ratio_symbol: str, ground_truth_primitive: str, seed: int = 42
) -> ValidationResult:
    """
    Ablation control, now correctly compared against the FIXED (isolated)
    positive control -- not against a positive control that shares the
    same contaminated full-pool search space as this ablated run.
    """
    proposer = AsymptoticDictionaryProposerV3(
        ratio_symbol=ratio_symbol,
        exclude_primitives=[ground_truth_primitive],
    )
    result = run_pipeline(scenario, proposer, seed=seed)

    pos = positive_control_fixed(scenario, ratio_symbol, ground_truth_primitive, seed=seed)
    pos_nmse = pos.best_nmse_residual if pos.best_nmse_residual is not None else float("inf")
    abl_nmse = result.best_nmse_residual if result.best_nmse_residual is not None else float("inf")

    passed = (abl_nmse > 3 * pos_nmse) or (abl_nmse > 0.20)

    return ValidationResult(
        scenario_name=scenario.name,
        check_name="ablation_control",
        passed=passed,
        detail=(
            f"excluded={ground_truth_primitive} only (4 primitives still "
            f"available, matches original intent), ablated_nmse={abl_nmse}, "
            f"TRULY-isolated positive_control_nmse={pos_nmse}, "
            f"ratio={abl_nmse / max(pos_nmse, 1e-12):.2f}"
        ),
        search_space_size=proposer.search_space_size(),
        best_expr=result.best_expr,
        best_nmse_residual=result.best_nmse_residual,
    )


if __name__ == "__main__":
    from adcd.anomaly_scenarios import get_all_scenarios

    scenario = next(
        s for s in get_all_scenarios() if s.name == "Blind-4: Relativistic Pendulum"
    )

    print("=" * 70)
    print("RE-RUNNING WITH FIXED ISOLATION")
    print("=" * 70)
    pos = positive_control_fixed(scenario, "(v/c)**2", "D_lor")
    print(f"[{'PASS' if pos.passed else 'FAIL'}] positive_control (fixed): {pos.detail}")

    abl = ablation_control_fixed(scenario, "(v/c)**2", "D_lor")
    print(f"[{'PASS' if abl.passed else 'FAIL'}] ablation_control (fixed): {abl.detail}")

    print()
    print("Compare pos.best_expr against the OLD contaminated result:")
    print(f"  OLD (contaminated): mixed D_lor+D_rat expression")
    print(f"  NEW (isolated)    : {pos.best_expr}")
    print()
    print(
        "If the NEW positive-control expression is now PURELY D_lor-shaped "
        "(no D_rat term mixed in), the fix worked as intended. If it's "
        "still mixed, something else is leaking primitives across the "
        "exclude_primitives boundary -- investigate "
        "AsymptoticDictionaryProposerV3._active_primitives construction next."
    )
