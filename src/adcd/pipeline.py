"""
pipeline.py (AUDIT-HARDENED)
=============================
FIX LOG:

BUG FOUND (critical -- defeats the ARC classical-limit safety gate):
Original code:

    if arc_score <= 0.0:
        if has_params:
            arc_score = 1.0     # <-- BYPASS: treated as a PERFECT score
        else:
            reject

`has_params` is True whenever the candidate contains ANY free theta_N symbol
(set upstream by `_substitute_thetas`, which is True for essentially every
candidate CorrectionMockProposer ever emits -- virtually all of its templates
start with "theta_0 * ..."). The consequence: candidates that fail to vanish
at the classical limit even at theta=1 were not just given a second chance,
they were handed the BEST POSSIBLE arc_score (1.0) and then never re-checked.

The comment claimed "This will be rigorously checked again AFTER Stage 2
optimization" -- but grep across the entire 18-file codebase confirms no
such re-check exists anywhere. `arc_score` is simply carried forward
unchanged from Stage 1 all the way to the final BIC ranking. This means the
single most important physical constraint this whole project is built
around -- "a correction MUST vanish at the classical limit" -- is, for
virtually every real candidate, never actually verified against the FITTED
parameters. Only the untested theta=1 probe (which is not what the model
claims at the end) goes through the gate at all, and even that is discarded
in favor of an automatic pass.

FIX:
1. `has_params` no longer forces arc_score to 1.0. Instead the candidate is
   marked `deferred_arc=True` and allowed to proceed to Stage 2 UNSCORED
   (arc_score stays at its true, computed value, which may be 0). This
   preserves the legitimate original intent (some ground-truth corrections
   need exact cancellation between two terms that theta=1 cannot produce,
   e.g. 2*(c/v)^2*(...) - 1) without pretending that intent has already been
   satisfied.
2. `GateStats` gains `deferred_arc` and `arc_relaxed_dim` counters so every
   run reports exactly how many candidates were waved through pending
   Stage-2 re-verification -- this number must now be reported alongside
   any "success" claim.
3. The actual Stage-2 re-verification is implemented in
   `correction_orchestrator_fixed.py` (`_reverify_arc_at_fitted_theta`),
   which recomputes the ARC score at the FITTED theta and drops any
   candidate that still fails to vanish at the classical limit. This file
   only prepares the honest bookkeeping; the enforcement lives where the
   fitted theta actually exists.
"""

import sympy as sp
import numpy as np
from dataclasses import dataclass, fields, asdict
from typing import List, Tuple, Union, Dict, Optional
from adcd.dimensional_checker import ASTValidator, DimensionalChecker, validate_transcendental_args
from adcd.arc_scorer import ARCScorer
from adcd.coarse_evaluator import CoarseEvaluator


@dataclass
class GateStats:
    """Per-gate survival counts for Stage 1 filter cascade telemetry."""

    input_count: int = 0
    parse_fail: int = 0
    ast_reject: int = 0
    dim_reject: int = 0
    transcendental_reject: int = 0
    arc_reject: int = 0
    coarse_reject: int = 0
    output_count: int = 0

    # NEW: honest bookkeeping for candidates waved through pending re-verification
    deferred_arc: int = 0          # arc_score was 0 at theta=1 but candidate has free params
    arc_relaxed_dim: int = 0       # passed dimensional check only via theta-scaling relaxation

    llm_input: int = 0
    llm_output: int = 0
    grammar_input: int = 0
    grammar_output: int = 0
    mock_input: int = 0
    mock_output: int = 0

    @property
    def after_parse(self) -> int:
        return self.input_count - self.parse_fail

    @property
    def after_ast(self) -> int:
        return self.after_parse - self.ast_reject

    @property
    def after_dim(self) -> int:
        return self.after_ast - self.dim_reject

    @property
    def after_transcendental(self) -> int:
        return self.after_dim - self.transcendental_reject

    @property
    def after_arc(self) -> int:
        return self.after_transcendental - self.arc_reject

    @property
    def after_coarse(self) -> int:
        return self.after_arc - self.coarse_reject

    def merge(self, other: "GateStats") -> None:
        for f in fields(self):
            setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))

    def survival_rates(self) -> Dict[str, float]:
        if self.input_count == 0:
            return {}

        def rate(survivors: int, entered: int) -> float:
            return survivors / entered if entered > 0 else 1.0

        return {
            "parse": rate(self.after_parse, self.input_count),
            "ast": rate(self.after_ast, self.after_parse),
            "dimensional": rate(self.after_dim, self.after_ast),
            "transcendental": rate(self.after_transcendental, self.after_dim),
            "arc": rate(self.after_arc, self.after_transcendental),
            "coarse": rate(self.output_count, self.after_arc),
            "overall": rate(self.output_count, self.input_count),
            # NEW: fraction of the FINAL output pool that was never actually
            # ARC-verified and is only pending Stage-2 re-check.
            "fraction_output_deferred_arc": (
                self.deferred_arc / self.output_count if self.output_count > 0 else 0.0
            ),
        }

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["after_parse"] = self.after_parse
        d["after_ast"] = self.after_ast
        d["after_dim"] = self.after_dim
        d["after_transcendental"] = self.after_transcendental
        d["after_arc"] = self.after_arc
        d["after_coarse"] = self.after_coarse
        d["survival_rates"] = self.survival_rates()
        return d


class Stage1Pipeline:
    """Orchestrates the cascading coarse screening workflow."""

    def __init__(self, validator: ASTValidator, checker: DimensionalChecker, scorer: ARCScorer):
        self.validator = validator
        self.checker = checker
        self.scorer = scorer
        self.locals = {s: sp.Symbol(s) for s in checker.registry}

    def execute(
        self,
        candidates: Union[List[str], List[Tuple[str, bool]]],
        target_dimension_key: str,
        X: Dict[str, np.ndarray] = None,
        y_obs: np.ndarray = None,
        beta: float = 1.0,
        constants: Dict[str, float] = None,
        stats: Optional[GateStats] = None,
        candidate_sources: Optional[Dict[str, str]] = None,
    ) -> List[Tuple[str, float, float, float, bool]]:
        """
        Returns list of (candidate, combined_score, mse, arc_score, deferred_arc)
        sorted by combined_score descending. `deferred_arc=True` means arc_score
        was 0 at the screening probe (theta=1) and MUST be re-verified at the
        fitted theta before being reported as a discovery -- see
        `correction_orchestrator_fixed._reverify_arc_at_fitted_theta`.
        """
        screened_candidates = []

        evaluator = None
        if X is not None and y_obs is not None:
            evaluator = CoarseEvaluator(X, y_obs, constants=constants)

        for item in candidates:
            if isinstance(item, tuple):
                raw_cand, has_params = item
            else:
                raw_cand = item
                has_params = False

            if stats is not None:
                stats.input_count += 1
                if candidate_sources and raw_cand in candidate_sources:
                    src = candidate_sources[raw_cand]
                    if src in ("gemini", "llm"):
                        stats.llm_input += 1
                    elif src == "grammar":
                        stats.grammar_input += 1
                    elif src == "mock":
                        stats.mock_input += 1

            try:
                expr = sp.sympify(raw_cand, locals=self.locals)
            except Exception:
                if stats is not None:
                    stats.parse_fail += 1
                continue

            if not self.validator.verify(expr):
                if stats is not None:
                    stats.ast_reject += 1
                continue

            is_dim_ok = True
            if target_dimension_key is not None:
                is_dim_ok = self.checker.verify(expr, target_dimension_key)
                if is_dim_ok and getattr(self.checker, "last_relaxed", False) and stats is not None:
                    stats.arc_relaxed_dim += 1

            if not is_dim_ok:
                if stats is not None:
                    stats.dim_reject += 1
                continue

            if not validate_transcendental_args(expr, self.checker):
                if stats is not None:
                    stats.transcendental_reject += 1
                continue

            try:
                arc_score = float(self.scorer.score(expr, constants=constants))
            except Exception:
                if stats is not None:
                    stats.arc_reject += 1
                continue

            deferred_arc = False
            if arc_score <= 0.0:
                if has_params:
                    # FIXED: do NOT fabricate a perfect score. Mark as deferred
                    # and let it through UNSCORED (arc_score stays 0.0) so that
                    # BIC/likelihood ranking downstream does not treat it as
                    # ARC-verified. It must clear `_reverify_arc_at_fitted_theta`
                    # after Stage 2 or it gets dropped there.
                    deferred_arc = True
                    if stats is not None:
                        stats.deferred_arc += 1
                else:
                    if stats is not None:
                        stats.arc_reject += 1
                    continue

            mse = 0.0
            nmse = 0.0
            if evaluator is not None:
                mse, nmse = evaluator.evaluate(expr, has_params=has_params)
                if not np.isfinite(mse):
                    if stats is not None:
                        stats.coarse_reject += 1
                    continue

            # For deferred candidates, use a neutral placeholder (0.5) for the
            # coarse ranking pass ONLY -- never report this as the final
            # arc_score. It exists purely so the candidate isn't sorted to the
            # very bottom before Stage 2 gets a chance to fit it properly.
            ranking_arc_score = 0.5 if deferred_arc else arc_score
            combined_score = ranking_arc_score * float(np.exp(-beta * nmse))
            screened_candidates.append((raw_cand, combined_score, mse, arc_score, deferred_arc))
            if stats is not None:
                stats.output_count += 1
                if candidate_sources and raw_cand in candidate_sources:
                    src = candidate_sources[raw_cand]
                    if src in ("gemini", "llm"):
                        stats.llm_output += 1
                    elif src == "grammar":
                        stats.grammar_output += 1
                    elif src == "mock":
                        stats.mock_output += 1

        return sorted(screened_candidates, key=lambda x: x[1], reverse=True)
