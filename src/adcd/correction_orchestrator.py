"""
correction_orchestrator.py (AUDIT-HARDENED)
==============================================
FIX LOG:

1. POST-STAGE-2 ARC RE-VERIFICATION (closes the pipeline.py ARC-bypass gap):
   pipeline_fixed.py marks candidates `deferred_arc=True` instead of
   fabricating a perfect arc_score when a candidate's theta=1 probe fails
   the classical-limit check. This file now ACTUALLY re-verifies those
   candidates after Stage 2 fitting, at the FITTED theta -- the check the
   original code's comment claimed happened but, confirmed by grep across
   the whole codebase, never did. Any candidate that still fails to vanish
   at the classical limit with its real, fitted parameters is dropped.

2. REAL residual_features (closes the dead-code gap in llm_proposer.py):
   `res_feat = None` is replaced with an actual call to
   `compute_residual_features()` using the scenario's classical-limit
   variable and the observed residual -- purely data-driven, no ground
   truth involved.

3. `correction_type` is threaded into `IdentifiabilityAnalyzer.analyze()` so
   its SNR computation reconstructs y_obs correctly (see identifiability_fixed.py).

4. New honesty flags (`best_arc_reverified`, `n_rejected_at_arc_reverify`,
   `used_extreme_scale_restart`) are surfaced in `CorrectionSearchResult`.
"""

import time
import logging
import numpy as np
import sympy as sp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from adcd.llm_proposer import BaseProposer, ProposalContext
from adcd.pipeline import Stage1Pipeline, GateStats
from adcd.jax_optimizer import JAXOptimizer
from adcd.anomaly_scenarios import AnomalyScenario
from adcd.metrics import evaluate_correction, CorrectionEvaluation, bic_score
from adcd.bayesian_ranker import BayesianCorrectionOutput
from adcd.identifiability import IdentifiabilityReport, IdentifiabilityAnalyzer
from adcd.residual_features import compute_residual_features

logger = logging.getLogger(__name__)


@dataclass
class CorrectionIterationResult:
    iteration: int
    n_proposed: int
    n_survived_stage1: int
    n_optimized_stage2: int
    best_expr: str
    best_nmse_residual: float
    best_nmse_full: float
    top_5: List[Tuple[str, float]]
    time_seconds: float


@dataclass
class CorrectionSearchResult:
    best_expr: str
    best_nmse_residual: float
    best_nmse_full: float
    best_theta: Dict[str, float]
    history: List[CorrectionIterationResult]
    total_candidates_proposed: int
    total_candidates_survived_stage1: int
    total_time_seconds: float
    converged: bool
    evaluation: Optional[CorrectionEvaluation] = None
    gate_stats: Optional[GateStats] = None
    total_candidates_optimized: int = 0
    bayesian_output: Optional[BayesianCorrectionOutput] = None
    identifiability_report: Optional[IdentifiabilityReport] = None
    best_arc_reverified: bool = False
    n_rejected_at_arc_reverify: int = 0
    used_extreme_scale_restart: bool = False


class CorrectionOrchestrator:
    def __init__(
        self,
        proposer: BaseProposer,
        pipeline: Stage1Pipeline,
        optimizer: JAXOptimizer,
        max_iterations: int = 5,
        top_k: int = 10,
        convergence_nmse: float = 1e-5,
        verbose: bool = True,
    ):
        self.proposer = proposer
        self.pipeline = pipeline
        self.optimizer = optimizer
        self.max_iterations = max_iterations
        self.top_k = top_k
        self.convergence_nmse = convergence_nmse
        self.verbose = bool(verbose) if not isinstance(verbose, str) else (verbose == "debug" or verbose.lower() == "true")

        if "dimensionless" not in self.pipeline.checker.registry:
            self.pipeline.checker.registry["dimensionless"] = [0, 0, 0]
            self.pipeline.locals["dimensionless"] = sp.Symbol("dimensionless")

    def _register_scenario_symbols(self, scenario: AnomalyScenario):
        for var in scenario.classical_variables:
            if var not in self.pipeline.checker.registry:
                base = var.replace("_ref", "").replace("_0", "")
                self.pipeline.checker.registry[var] = self.pipeline.checker.registry.get(base, [0, 0, 0])
            if var not in self.pipeline.locals:
                self.pipeline.locals[var] = sp.Symbol(var)
        for const in scenario.classical_constants:
            if const not in self.pipeline.checker.registry:
                base = const.replace("_ref", "").replace("_0", "")
                self.pipeline.checker.registry[const] = self.pipeline.checker.registry.get(base, [0, 0, 0])
            if const not in self.pipeline.locals:
                self.pipeline.locals[const] = sp.Symbol(const)

    def _substitute_thetas(self, expr_str: str, val: float = 1.0) -> str:
        try:
            expr = sp.sympify(expr_str, locals=self.pipeline.locals)
            subs_dict = {s: val for s in expr.free_symbols if str(s).startswith("theta_")}
            if subs_dict:
                return str(expr.subs(subs_dict))
        except Exception:
            pass
        return expr_str

    def _reverify_arc_at_fitted_theta(
        self, expr_str: str, theta_fit: Dict[str, float], constants: Dict[str, float]
    ) -> float:
        """FIX: recompute the ARC score with the ACTUAL fitted theta values,
        not the theta=1 screening probe. This is the re-verification the
        original code's comment promised but never implemented."""
        try:
            expr = sp.sympify(expr_str, locals=self.pipeline.locals)
            subs = {sp.Symbol(k): v for k, v in theta_fit.items() if k.startswith("theta_")}
            fitted_expr = expr.subs(subs)
            return float(self.pipeline.scorer.score(fitted_expr, constants=constants))
        except Exception as e:
            logger.warning(f"ARC re-verification failed for '{expr_str}': {e}")
            return 0.0

    def search_correction(
        self, scenario: AnomalyScenario, noise_level: float = 0.0, seed: int = 42
    ) -> CorrectionSearchResult:
        start_time = time.time()
        self._register_scenario_symbols(scenario)

        X, y_obs, y_classical, residual = scenario.generate_data(noise_level=noise_level, seed=seed)
        for c_name, c_val in scenario.classical_constants.items():
            if c_name not in X:
                X[c_name] = np.full_like(y_obs, c_val)

        # FIX: real residual features (no ground truth used).
        res_feat = None
        limit_vars = [v.strip() for v in str(scenario.classical_limit_variable).split(",") if v.strip()]
        leading_var_name = limit_vars[0] if limit_vars else None
        if leading_var_name and leading_var_name in X:
            try:
                res_feat = compute_residual_features(X[leading_var_name], residual)
            except Exception as e:
                logger.warning(f"residual feature computation failed: {e}")

        data_statistics = {}
        for var in scenario.classical_variables:
            if var in X:
                arr = X[var]
                data_statistics[var] = {
                    "mean": float(np.mean(arr)), "std": float(np.std(arr)),
                    "min": float(np.min(arr)), "max": float(np.max(arr)),
                }

        if scenario.correction_type == "multiplicative":
            target_dim_key = "dimensionless"
        else:
            limit_var = scenario.classical_limit_variable
            target_dim_key = scenario.variables_with_units.get(limit_var, "dimensionless")
            if target_dim_key not in self.pipeline.checker.registry and target_dim_key != "dimensionless":
                target_dim_key = "dimensionless"

        best_expr, best_nmse_residual, best_bic = "", float("inf"), float("inf")
        best_theta: Dict[str, float] = {}
        best_arc_reverified = False
        used_extreme_scale_restart = False
        n_rejected_at_arc_reverify = 0
        stuck_count = 0
        prev_best_nmse_res = float("inf")

        total_candidates_proposed = 0
        total_candidates_survived_stage1 = 0
        total_candidates_optimized = 0
        gate_stats = GateStats()

        history: List[CorrectionIterationResult] = []
        previous_best: List[Tuple[str, float]] = []
        all_candidates_bic: List[Tuple[str, float]] = []
        stage2_results_with_bic: List[tuple] = []

        classical_limit_cond = f"{scenario.classical_limit_variable} -> {scenario.classical_limit_direction}"

        for iteration in range(self.max_iterations):
            iter_start_time = time.time()

            context = ProposalContext(
                variable_names=scenario.classical_variables,
                target_name="residual",
                data_statistics=data_statistics,
                n_candidates=25,
                iteration=iteration,
                stuck_count=stuck_count,
                domain=scenario.domain,
                classical_expr=scenario.classical_expr,
                variables_with_units=scenario.variables_with_units,
                anomaly_description=f"Observed in {scenario.anomaly_regime}. Mode: {scenario.correction_type}",
                known_limits=[{
                    "variable": scenario.classical_limit_variable,
                    "limit": scenario.classical_limit_direction,
                    "expected": "0",
                }],
                classical_limit_condition=classical_limit_cond,
                max_nodes=10,
                structural_hints=[],
                previous_best=previous_best if previous_best else None,
                constants=scenario.classical_constants,
                residual_features=res_feat,
                X_data=X,
                residual_data=residual,
            )

            proposed_candidates = self.proposer.propose(context)
            n_proposed = len(proposed_candidates)
            total_candidates_proposed += n_proposed

            subbed_candidates, orig_by_subbed, candidate_sources = [], {}, {}
            for cand in proposed_candidates:
                sub_expr = self._substitute_thetas(cand, 1.0)
                has_params = (sub_expr != cand)
                subbed_candidates.append((sub_expr, has_params))
                orig_by_subbed.setdefault(sub_expr, []).append(cand)
                if hasattr(self.proposer, "sources") and cand in self.proposer.sources:
                    candidate_sources[sub_expr] = self.proposer.sources[cand]

            # pipeline.execute() now returns 5-tuples: (cand, combined_score,
            # mse, arc_score, deferred_arc) -- see pipeline_fixed.py.
            stage1_results = self.pipeline.execute(
                subbed_candidates, target_dim_key, X, residual,
                constants=scenario.classical_constants, stats=gate_stats,
                candidate_sources=candidate_sources,
            )

            seen_sub_exprs, reconstructed_results = set(), []
            for sub_expr, combined_score, mse, arc_score, deferred_arc in stage1_results:
                if sub_expr in orig_by_subbed and sub_expr not in seen_sub_exprs:
                    seen_sub_exprs.add(sub_expr)
                    orig_cand = orig_by_subbed[sub_expr][0]
                    reconstructed_results.append((orig_cand, combined_score, mse, arc_score, deferred_arc))

            n_survived = len(reconstructed_results)
            total_candidates_survived_stage1 += n_survived

            if n_survived == 0:
                stuck_count += 1
                history.append(CorrectionIterationResult(
                    iteration=iteration, n_proposed=n_proposed, n_survived_stage1=0,
                    n_optimized_stage2=0, best_expr=best_expr,
                    best_nmse_residual=best_nmse_residual, best_nmse_full=float("inf"),
                    top_5=[], time_seconds=time.time() - iter_start_time,
                ))
                continue

            top_k_candidates = reconstructed_results[:self.top_k]
            subbed_top_k = []
            for cand, score, mse, arc, deferred in top_k_candidates:
                try:
                    expr = sp.sympify(cand, locals=self.pipeline.locals)
                    subs_dict = {sp.Symbol(k): v for k, v in scenario.classical_constants.items()}
                    subbed_cand = str(expr.subs(subs_dict)) if subs_dict else cand
                except Exception:
                    subbed_cand = cand
                subbed_top_k.append((subbed_cand, score, mse, arc, deferred))

            stage2_results = self.optimizer.optimize_batch(
                [(c, s, m, a) for c, s, m, a, d in subbed_top_k],
                X, residual, scenario.classical_variables, loss_mode='auto',
                y_classical=y_classical, correction_type=scenario.correction_type,
            )
            total_candidates_optimized += len(stage2_results)
            deferred_flags = {c: d for c, s, m, a, d in subbed_top_k}

            stage2_results_with_bic = []
            for expr_str, stage2_combined, opt_nmse, arc_score, opt_result in stage2_results:
                is_deferred = deferred_flags.get(expr_str, False)
                if is_deferred:
                    reverified_score = self._reverify_arc_at_fitted_theta(
                        expr_str, opt_result.theta, scenario.classical_constants
                    )
                    if reverified_score <= 0.0:
                        n_rejected_at_arc_reverify += 1
                        continue  # fails the classical-limit constraint for real -> drop

                n_params = len([k for k in opt_result.theta.keys() if k.startswith("theta_")])
                val_eval = evaluate_correction(expr_str, scenario, X, y_obs, y_classical, opt_result.theta)
                b_score = bic_score(val_eval.nmse_residual, n_params, len(residual))
                stage2_results_with_bic.append(
                    (expr_str, stage2_combined, val_eval.nmse_residual, arc_score, opt_result, b_score, val_eval, is_deferred)
                )
                if np.isfinite(b_score):
                    all_candidates_bic.append((expr_str, b_score))

            if stage2_results_with_bic:
                stage2_results_with_bic.sort(
                    key=lambda x: (x[5], len(list(sp.preorder_traversal(sp.sympify(x[0])))))
                )
                (iter_best_expr, _, iter_best_nmse, _, iter_opt_res,
                 iter_best_bic, _, iter_was_deferred) = stage2_results_with_bic[0]

                if iter_best_bic < best_bic:
                    best_bic, best_nmse_residual, best_expr = iter_best_bic, iter_best_nmse, iter_best_expr
                    best_theta = iter_opt_res.theta
                    best_arc_reverified = True  # reaching here means it either
                    # wasn't deferred, or it WAS
                    # deferred and passed re-verify
                    used_extreme_scale_restart = getattr(iter_opt_res, "extreme_scale_restart", False)

                iter_feedback = [(r[0], r[5]) for r in stage2_results_with_bic if np.isfinite(r[2])]
                previous_best.extend(iter_feedback)
                previous_best = sorted(previous_best, key=lambda x: x[1])[:20]

            stuck_count = 0 if best_nmse_residual < prev_best_nmse_res * 0.99 else stuck_count + 1
            prev_best_nmse_res = best_nmse_residual

            if best_expr:
                temp_eval = evaluate_correction(best_expr, scenario, X, y_obs, y_classical, best_theta)
                best_nmse_full = temp_eval.nmse_full
            else:
                best_nmse_full = float("inf")

            top_5 = [(r[0], r[2]) for r in stage2_results_with_bic[:5]]
            history.append(CorrectionIterationResult(
                iteration=iteration, n_proposed=n_proposed, n_survived_stage1=n_survived,
                n_optimized_stage2=len(stage2_results), best_expr=best_expr,
                best_nmse_residual=best_nmse_residual, best_nmse_full=best_nmse_full,
                top_5=top_5, time_seconds=time.time() - iter_start_time,
            ))

            if self.verbose:
                nmse_str = f"{best_nmse_residual:.2e}" if best_nmse_residual < float('inf') else "inf"
                bar = "#" * int((iteration + 1) / self.max_iterations * 20) + "." * (20 - int((iteration + 1) / self.max_iterations * 20))
                expr_display = best_expr if len(best_expr) <= 32 else best_expr[:29] + "..."
                gate_pct = f"{n_survived/n_proposed*100:.0f}%" if n_proposed else "n/a"
                print(
                    f"  [{bar}] Iter {iteration+1}/{self.max_iterations}  |  {n_proposed} proposed -> "
                    f"{n_survived} passed ({gate_pct})  |  NMSE: {nmse_str}  |  best: {expr_display}  |  "
                    f"rejected-at-ARC-reverify(cum): {n_rejected_at_arc_reverify}"
                )

            if best_nmse_residual < self.convergence_nmse:
                if len([k for k in best_theta.keys() if k.startswith("theta_")]) <= 1:
                    break

        total_time = time.time() - start_time
        final_evaluation = evaluate_correction(best_expr, scenario, X, y_obs, y_classical, best_theta)
        converged = final_evaluation.nmse_residual < self.convergence_nmse

        bayesian_out, ident_report = None, None
        if all_candidates_bic:
            try:
                unique_cands = {}
                for expr, bic in all_candidates_bic:
                    if expr not in unique_cands or bic < unique_cands[expr]:
                        unique_cands[expr] = bic
                deduped = list(unique_cands.items())
                if deduped:
                    from adcd.bayesian_ranker import BayesianReranker
                    reranker = BayesianReranker(threshold_ratio=0.05)
                    bayesian_out = reranker.rank(deduped)

                    analyzer = IdentifiabilityAnalyzer()
                    ident_report = analyzer.analyze(
                        bayesian_output=bayesian_out, residual=residual,
                        y_classical=y_classical, noise_level=noise_level,
                        correction_type=scenario.correction_type,  # FIX: threaded through
                    )
                    logger.debug(f"[Phase3] {ident_report.summary}")
            except Exception as e:
                logger.warning(f"[Phase3] Bayesian analysis failed: {e}")

        return CorrectionSearchResult(
            best_expr=best_expr, best_nmse_residual=best_nmse_residual,
            best_nmse_full=final_evaluation.nmse_full, best_theta=best_theta,
            history=history, total_candidates_proposed=total_candidates_proposed,
            total_candidates_survived_stage1=total_candidates_survived_stage1,
            total_time_seconds=total_time, converged=converged,
            evaluation=final_evaluation, gate_stats=gate_stats,
            total_candidates_optimized=total_candidates_optimized,
            bayesian_output=bayesian_out, identifiability_report=ident_report,
            best_arc_reverified=best_arc_reverified,
            n_rejected_at_arc_reverify=n_rejected_at_arc_reverify,
            used_extreme_scale_restart=used_extreme_scale_restart,
        )
