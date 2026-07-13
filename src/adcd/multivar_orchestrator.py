"""Multivariable correction discovery orchestrator (ADCD Phase 2)."""

from __future__ import annotations

import copy
import logging
from typing import Dict, List, Optional, Union

import numpy as np
import sympy as sp
from scipy.stats import pearsonr

from adcd.anomaly_scenarios import AnomalyScenario, get_mv_scenario  # noqa: F401
from adcd.buckingham_pi import BuckinghamPiEngine
from adcd.correction_orchestrator import CorrectionOrchestrator, CorrectionSearchResult
from adcd.dimensional_checker import ASTValidator, DimensionalChecker
from adcd.jax_optimizer import JAXOptimizer
from adcd.llm_proposer import BaseProposer, ProposalContext
from adcd.pipeline import Stage1Pipeline
from adcd.product_grammar import (
    limit_specs_from_scenario,
    pi_groups_for_scenario,
    ProductGrammar,
)
from adcd.residual_factorizer_v2 import ResidualFactorizerV2
from adcd.sequential_arc import SequentialARCChecker

logger = logging.getLogger(__name__)


class SequentialARCScorer:
    """ARCScorer-compatible wrapper using independent limit checks."""

    def __init__(
        self,
        limit_vars: List[str],
        limit_dirs: List[str],
        checker: Optional[SequentialARCChecker] = None,
    ) -> None:
        self.limit_vars = limit_vars
        self.limit_dirs = limit_dirs
        self.checker = checker or SequentialARCChecker()
        self.total_weight = 1.0

    def score(
        self,
        candidate_expr: Union[str, sp.Expr],
        constants: Optional[Dict[str, float]] = None,
    ) -> float:
        try:
            expr = (
                sp.sympify(candidate_expr)
                if isinstance(candidate_expr, str)
                else candidate_expr
            )
        except Exception:
            return 0.0
        result = self.checker.check(
            expr,
            limit_vars=self.limit_vars,
            limit_dirs=self.limit_dirs,
            constants=constants or {},
        )
        return 1.0 if result.passes else 0.0


class ProductGrammarProposer(BaseProposer):
    """Propose multivariable candidates from ProductGrammar Pi-group templates."""

    def __init__(self, scenario: AnomalyScenario, seed: int = 42) -> None:
        self.scenario = scenario
        self.seed = seed
        self.sources: Dict[str, str] = {}
        self._grammar = ProductGrammar()
        self._limit_specs = limit_specs_from_scenario(scenario)
        self._pi_groups = pi_groups_for_scenario(scenario)

    def propose(self, context: ProposalContext) -> List[str]:
        candidates = self._grammar.generate(
            self._pi_groups,
            self._limit_specs,
            n_candidates=max(context.n_candidates, 80),
            constants=self.scenario.classical_constants,
            verify_arc=True,
        )

        extra: List[str] = []
        vars_avail = list(context.variable_names)
        refs = self.scenario.classical_constants

        for pi in self._pi_groups:
            pi_str = str(sp.simplify(pi))
            extra.append(f"theta_0 * ({pi_str})")
            extra.append(f"theta_0 * ({pi_str})**2")
            extra.append(f"theta_0 * exp(-({pi_str})/theta_1)")

        if len(vars_avail) >= 2:
            for v1 in vars_avail:
                for v2 in vars_avail:
                    if v1 == v2:
                        continue
                    # Resolve symbol names for constants instead of float values to preserve dimensional units
                    r1_str = f"{v1}_ref" if f"{v1}_ref" in refs else (f"{v1}_0" if f"{v1}_0" in refs else "1.0")
                    r2_str = f"{v2}_ref" if f"{v2}_ref" in refs else (f"{v2}_0" if f"{v2}_0" in refs else "1.0")
                    
                    extra.extend(
                        [
                            f"theta_0 * ({v1}/{r1_str}) * exp(-{v2}/{r2_str})",
                            f"theta_0 * ({v1}/{r1_str})**2 * ({v2}/{r2_str})",
                            f"theta_0 * ({v1}/{r1_str}) * ({v2}/{r2_str})",
                            f"theta_0 * ({v1}/{r1_str})**2 / ({v2}/{r2_str})**2",
                            f"theta_0 * ({v1}/{r1_str}) * ({r2_str}/{v2})**0.5",
                        ]
                    )

        seen: set[str] = set()
        merged: List[str] = []
        for cand in extra + candidates:
            if cand not in seen:
                seen.add(cand)
                merged.append(cand)
                self.sources[cand] = "grammar"

        # Expand pool to allow Stage 1 pipeline to screen candidates thoroughly
        return merged[:250]


class MultivariableOrchestrator:
    """Coordinate Pi-sparse selection, factorization, and product-grammar search."""

    def __init__(self, seed: int = 42, verbose: bool = True) -> None:
        self.seed = seed
        self.verbose = verbose

    def _select_relevant_pis(
        self,
        pi_groups: List[sp.Expr],
        X: Dict[str, np.ndarray],
        delta: np.ndarray,
        threshold: float = 0.1,
    ) -> List[sp.Expr]:
        relevant: List[sp.Expr] = []
        for pi in pi_groups:
            try:
                syms = [sp.Symbol(s) for s in pi.free_symbols]
                subs = {s: 1.0 for s in syms if str(s).startswith("theta")}
                pi_sub = pi.subs(subs)
                fn = sp.lambdify(syms, pi_sub, modules=["numpy"])
                args = [X[str(s)] for s in syms]
                pi_vals = np.asarray(fn(*args), dtype=float)
                if not np.all(np.isfinite(pi_vals)):
                    continue
                corr, _ = pearsonr(pi_vals, delta)
                if abs(corr) > threshold:
                    relevant.append(pi)
            except Exception:
                continue
        return relevant if relevant else pi_groups

    def discover_multivariable_correction(
        self,
        scenario: AnomalyScenario,
        X: Dict[str, np.ndarray],
        delta: np.ndarray,
        noise_level: float = 0.0,
        max_iterations: int = 5,
    ) -> CorrectionSearchResult:
        pi_engine = BuckinghamPiEngine()
        pi_engine.register_from_scenario(scenario)
        pi_groups = pi_engine.compute_pi_groups()
        relevant_pis = self._select_relevant_pis(pi_groups, X, delta)

        factorizer = ResidualFactorizerV2()
        fact_result = factorizer.test_separability(
            {k: X[k] for k in scenario.classical_variables if k in X},
            delta,
        )

        if self.verbose:
            logger.info(
                "MV strategy: factorization=%s (R²=%.3f), Pi groups=%d",
                fact_result.factorization_type,
                fact_result.explained_variance,
                len(relevant_pis),
            )

        return run_adcd_mv(
            scenario,
            noise=noise_level,
            seed=self.seed,
            max_iterations=max_iterations,
            verbose=self.verbose,
        )


def _build_mv_pipeline(scenario: AnomalyScenario, seed: int = 42) -> tuple:
    limit_vars = [v.strip() for v in scenario.classical_limit_variable.split(",")]
    limit_dirs = [d.strip() for d in scenario.classical_limit_direction.split(",")]
    if len(limit_dirs) < len(limit_vars):
        limit_dirs.extend([limit_dirs[-1]] * (len(limit_vars) - len(limit_dirs)))

    validator = ASTValidator(max_depth=8, max_tokens=30)
    checker = DimensionalChecker()
    for var in scenario.classical_variables:
        if var not in checker.registry:
            base = var.replace("_ref", "").replace("_0", "")
            if base in checker.registry:
                checker.registry[var] = checker.registry[base]
            else:
                checker.registry[var] = [0, 0, 0]
    for const in scenario.classical_constants:
        if const not in checker.registry:
            base = const.replace("_ref", "").replace("_0", "")
            if base in checker.registry:
                checker.registry[const] = checker.registry[base]
            else:
                checker.registry[const] = [0, 0, 0]

    scorer = SequentialARCScorer(limit_vars, limit_dirs, SequentialARCChecker(seed=seed))
    pipeline = Stage1Pipeline(validator, checker, scorer)
    optimizer = JAXOptimizer(log_param=True)
    proposer = ProductGrammarProposer(scenario, seed=seed)
    return pipeline, optimizer, proposer


def run_adcd_mv(
    scenario: AnomalyScenario,
    noise: float = 0.0,
    seed: int = 42,
    max_iterations: int = 5,
    verbose: bool = False,
) -> CorrectionSearchResult:
    """Run multivariable ADCD discovery on a Phase 2 scenario."""
    pipeline, optimizer, proposer = _build_mv_pipeline(scenario, seed=seed)
    orchestrator = CorrectionOrchestrator(
        proposer=proposer,
        pipeline=pipeline,
        optimizer=optimizer,
        max_iterations=max_iterations,
        verbose=verbose,
    )

    orchestrator._register_scenario_symbols(scenario)
    for const in scenario.classical_constants:
        if const not in pipeline.checker.registry:
            base = const.replace("_ref", "").replace("_0", "")
            if base in pipeline.checker.registry:
                pipeline.checker.registry[const] = pipeline.checker.registry[base]
            else:
                pipeline.checker.registry[const] = [0, 0, 0]
            pipeline.locals[const] = sp.Symbol(const)

    original_execute = pipeline.execute

    def execute_dimensionless(candidates, target_dimension_key, X=None, y_obs=None, **kwargs):
        return original_execute(candidates, "dimensionless", X, y_obs, **kwargs)

    pipeline.execute = execute_dimensionless

    scenario_for_search = scenario
    if "," in scenario.classical_limit_variable:
        scenario_for_search = copy.copy(scenario)
        scenario_for_search.classical_limit_variable = (
            scenario.classical_limit_variable.split(",")[0].strip()
        )

    try:
        result = orchestrator.search_correction(
            scenario_for_search, noise_level=noise, seed=seed
        )
    finally:
        # Always restore the original method, even if search_correction raises,
        # so the pipeline object is not left with a permanently patched `execute`.
        pipeline.execute = original_execute
    return result
