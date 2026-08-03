"""
run_pure_grammar_search.py (NEW FILE)
========================================
PER EXPLICIT INSTRUCTION: this is the ONLY proposer configuration that
should be used to produce any result claimed as a "discovery." It contains
ZERO LLM involvement and ZERO hardcoded functional templates.

WHY THIS MATTERS (validating the instruction, not just complying with it):
An LLM proposer (Gemini/Anthropic/OpenAI-compatible, all present elsewhere
in llm_proposer.py) does not derive a formula from the data in front of it
-- it recalls text patterns from its training corpus. The Radial
Acceleration Relation, the Lorentz factor, Yukawa potentials, and every
other functional form in this project's benchmark are published, widely
discussed physics. A large language model has almost certainly seen the
literal functional form during training. If an LLM proposer is in the
pipeline (even mixed 40/40/20 with other sources, as the original
`HybridCorrectionProposer` did), any successful "discovery" is
UNFALSIFIABLE as evidence of algorithmic discovery -- it is indistinguishable
from the model simply typing out a memorized answer. This is the exact same
class of problem as the oracle-fed `ratio_symbol` bug found earlier in this
audit (Finding #1), just entering through a different door. Removing LLM
involvement entirely is the correct, principled fix -- not a partial
mitigation.

`CorrectionMockProposer` (also in llm_proposer.py) is NOT an LLM, but it IS
built from hand-written string templates ("theta_0 * exp(-{v1}/theta_1)",
etc.) -- exactly the "hard-coded mathematical templates" this whole project
was originally rebuilt to move away from (see the very first line of the
audit brief this session started from). It is kept in the codebase ONLY as
an optional, clearly-labeled NEGATIVE CONTROL / naive baseline ("does the
deterministic grammar outperform blind random templates?"), never as part
of a proposer whose output is reported as a discovered result.

THE ONLY PROPOSER WIRED IN BELOW: `GrammarProposerV3`
(asymptotic_dictionary_proposer_v3_fixed.py). It builds candidates by:
  1. Mechanically enumerating dimensionless ratios via Buckingham-Pi
     dimensional analysis over the variables/constants it is given.
  2. Mechanically substituting those ratios into a small, fixed library of
     REGULARIZED PRIMITIVES (D_lor, D_rat, D_exp, D_log, D_sqrt_inv) plus
     generic sum/product/reciprocal compositions.
Nothing in this process can "recall" a specific scenario's answer -- it has
no access to scenario names, ground truth expressions, or any text corpus.
It either finds a structure through exhaustive, disclosed enumeration, or it
does not.
"""

from adcd.pipeline import Stage1Pipeline
from adcd.dimensional_checker import DimensionalChecker, ASTValidator
from adcd.arc_scorer import ARCScorer, build_arc_regimes
from adcd.jax_optimizer import JAXOptimizer
from adcd.asymptotic_dictionary_proposer_v3 import GrammarProposerV3, GrammarBudget
from adcd.correction_orchestrator import CorrectionOrchestrator


def build_pure_grammar_orchestrator(
    max_ratio_candidates: int = 8,
    max_primitives_used: int = 2,
    n_restarts: int = 15,
    max_iterations: int = 3,
) -> CorrectionOrchestrator:
    """
    Assembles a CorrectionOrchestrator whose ONLY candidate source is the
    deterministic GrammarProposerV3. No proposer parameter here can be set
    to anything LLM-backed -- the class simply is not imported.
    """
    checker = DimensionalChecker()
    validator = ASTValidator(max_depth=7, max_tokens=25)
    regimes = build_arc_regimes()  # generic "vanish at classical limit" regime,
    # identical machinery for every scenario --
    # see arc_scorer.py fix notes.
    scorer = ARCScorer(regimes=regimes)
    pipeline = Stage1Pipeline(validator=validator, checker=checker, scorer=scorer)

    proposer = GrammarProposerV3(
        budget=GrammarBudget(
            max_ratio_candidates=max_ratio_candidates,
            max_primitives_used=max_primitives_used,
        ),
        dimensional_checker=checker,
    )

    optimizer = JAXOptimizer(n_restarts=n_restarts)

    return CorrectionOrchestrator(
        proposer=proposer,        # <-- the ONLY candidate source
        pipeline=pipeline,
        optimizer=optimizer,
        max_iterations=max_iterations,
        verbose=True,
    )


if __name__ == "__main__":
    import sys
    print(
        "This module wires a pure, LLM-free, template-free "
        "CorrectionOrchestrator. Import build_pure_grammar_orchestrator() "
        "and call .search_correction(scenario, noise_level=..., seed=...) "
        "on it. See the audit report for a worked RAR example."
    )
    sys.exit(0)
