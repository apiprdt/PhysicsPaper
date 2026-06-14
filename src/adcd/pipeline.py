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

    # Source-aware tracking (optional)
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
        """Fraction of inputs surviving each gate (1.0 if no inputs at gate entry)."""
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
    """
    Orchestrates the cascading coarse screening workflow to protect low-resource CPU
    hardware from processing mathematically flawed candidates.

    Example:
        >>> from adcd.pipeline import Stage1Pipeline
        >>> from adcd.dimensional_checker import ASTValidator, DimensionalChecker
        >>> from adcd.arc_scorer import ARCScorer
        >>> validator = ASTValidator()
        >>> checker = DimensionalChecker()
        >>> scorer = ARCScorer(regimes=[...])
        >>> pipeline = Stage1Pipeline(validator, checker, scorer)
    """
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
    ) -> List[Tuple[str, float, float, float]]:
        """
        Runs candidate strings through the filter cascade.

        Args:
            candidates: A list of candidate strings, or list of tuples (cand_str, has_params).
            target_dimension_key: The target dimension registry key (e.g. "dimensionless").
            X: Dictionary of independent variable arrays.
            y_obs: The observed residual array to fit.
            beta: Exponential weight factor scaling the influence of coarse NMSE.
            constants: Physical constants to substitute during evaluation.
            stats: Optional GateStats object mutated in-place with per-gate counts.

        Returns:
            List of sorted tuples (candidate, combined_score, mse, arc_score) in descending order of combined_score.

        Example:
            >>> results = pipeline.execute(["theta_0 * x**2", "x"], "dimensionless", X, y)
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
            if target_dimension_key is not None and target_dimension_key != "dimensionless":
                if has_params:
                    try:
                        self.checker._get_dim_vector(expr)
                        is_dim_ok = True
                    except TypeError:
                        is_dim_ok = False
                else:
                    is_dim_ok = self.checker.verify(expr, target_dimension_key)
            else:
                is_dim_ok = self.checker.verify(expr, target_dimension_key)

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
            if arc_score <= 0.0:
                if stats is not None:
                    stats.arc_reject += 1
                continue

            mse = 0.0
            nmse = 0.0
            if evaluator is not None:
                mse, nmse = evaluator.evaluate(expr, has_params=has_params)
                if np.isinf(mse):
                    if stats is not None:
                        stats.coarse_reject += 1
                    continue

            combined_score = arc_score * float(np.exp(-beta * nmse))
            screened_candidates.append((raw_cand, combined_score, mse, arc_score))
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
