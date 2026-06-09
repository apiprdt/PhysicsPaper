import time
import logging
import numpy as np
import sympy as sp
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from adcd.llm_proposer import BaseProposer, ProposalContext
from adcd.pipeline import Stage1Pipeline
from adcd.jax_optimizer import JAXOptimizer

logger = logging.getLogger(__name__)

@dataclass
class IterationResult:
    iteration: int
    n_proposed: int
    n_survived_stage1: int
    n_optimized_stage2: int
    best_expr: str
    best_nmse: float
    top_5: List[Tuple[str, float]]  # List of (expr, nmse)
    time_seconds: float

@dataclass
class SearchResult:
    best_expr: str
    best_nmse: float
    best_theta: Dict[str, float]
    best_likelihood: float
    history: List[IterationResult]
    total_candidates_proposed: int
    total_candidates_survived_stage1: int
    total_time_seconds: float
    converged: bool

class SROrchestrator:
    def __init__(
        self,
        proposer: BaseProposer,
        pipeline: Stage1Pipeline,
        optimizer: JAXOptimizer,
        max_iterations: int = 10,
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
        self.verbose = verbose

    def _substitute_thetas(self, expr_str: str, val: float = 1.0) -> str:
        """Substitute free parameter symbols (theta_N) with a default value for Stage 1 screening."""
        try:
            expr = sp.sympify(expr_str, locals=self.pipeline.locals)
            subs_dict = {s: val for s in expr.free_symbols if str(s).startswith("theta_")}
            if subs_dict:
                return str(expr.subs(subs_dict))
        except Exception:
            pass
        return expr_str

    def search(
        self,
        variable_names: List[str],
        target_name: str,
        target_dimension: Optional[str],
        X: Dict[str, np.ndarray],
        y_obs: np.ndarray,
        data_vars: List[str],
        constants: Dict[str, float] = None,
        seed: int = 42,
        # --- Rich physics metadata (populated from FeynmanProblem) ---
        domain: str = "classical physics",
        classical_expr: str = "",
        variables_with_units: Dict[str, str] = None,
        anomaly_description: str = "None",
        known_limits: List[dict] = None,
        classical_limit_condition: str = "",
        structural_hints: List[str] = None,
    ) -> SearchResult:
        start_time = time.time()
        
        # 1. Compute dataset statistics for ProposalContext
        data_statistics = {}
        for var in variable_names:
            if var in X:
                arr = X[var]
                data_statistics[var] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                    "min": float(np.min(arr)),
                    "max": float(np.max(arr)),
                }

        # Global search state tracking
        best_expr: str = ""
        best_nmse: float = float("inf")
        best_theta: Dict[str, float] = {}
        best_likelihood: float = 0.0
        stuck_count: int = 0
        prev_best_nmse: float = float("inf")
        
        total_candidates_proposed = 0
        total_candidates_survived_stage1 = 0
        
        history: List[IterationResult] = []
        previous_best: List[Tuple[str, float]] = []

        for iteration in range(self.max_iterations):
            iter_start_time = time.time()
            
            # Step 1: Build ProposalContext with all rich metadata + feedback
            context = ProposalContext(
                variable_names=variable_names,
                target_name=target_name,
                data_statistics=data_statistics,
                n_candidates=25,
                iteration=iteration,
                stuck_count=stuck_count,
                domain=domain,
                classical_expr=classical_expr,
                variables_with_units=variables_with_units or {},
                anomaly_description=anomaly_description,
                known_limits=known_limits or [],
                classical_limit_condition=classical_limit_condition,
                max_nodes=15,
                structural_hints=structural_hints or [],
                previous_best=previous_best if previous_best else None,
                constants=constants or {},
            )
            
            # Step 2: Propose candidates (Stage 0)
            proposed_candidates = self.proposer.propose(context)
            n_proposed = len(proposed_candidates)
            total_candidates_proposed += n_proposed

            # Step 3: Run Stage 1 Coarse Screening
            # Substitute theta_N with 1.0 for Stage 1, map back to originals afterward.
            subbed_candidates = []
            orig_by_subbed = {}
            for cand in proposed_candidates:
                sub_expr = self._substitute_thetas(cand, 1.0)
                has_params = (sub_expr != cand)
                subbed_candidates.append((sub_expr, has_params))
                if sub_expr not in orig_by_subbed:
                    orig_by_subbed[sub_expr] = []
                orig_by_subbed[sub_expr].append(cand)

            # If target_dimension is None (e.g. Coulomb, Projectile), skip dimensional check
            # by passing a fake dimension key that we register as matching everything.
            effective_target_dim = target_dimension
            if effective_target_dim is None:
                # Skip Stage 1 dimensional gate: run pipeline without dimensional filtering
                # We do this by executing candidates through pipeline with target_dimension=None
                # which our pipeline handles by skipping the DimensionalChecker gate.
                pass

            stage1_results = self.pipeline.execute(
                subbed_candidates,
                effective_target_dim,
                X,
                y_obs,
                constants=constants
            )
            
            # Reconstruct Stage 1 results back to original templates.
            # Deduplicate by sub_expr so theta-index variants of the same mathematical
            # form (e.g. theta_0*t and theta_89*t) each occupy only one top_k slot.
            seen_sub_exprs: set = set()
            reconstructed_results = []
            for sub_expr, combined_score, mse, arc_score in stage1_results:
                if sub_expr in orig_by_subbed:
                    if sub_expr not in seen_sub_exprs:
                        seen_sub_exprs.add(sub_expr)
                        # Pick the canonical (first-registered) original template
                        orig_cand = orig_by_subbed[sub_expr][0]
                        reconstructed_results.append((orig_cand, combined_score, mse, arc_score))

            n_survived = len(reconstructed_results)
            total_candidates_survived_stage1 += n_survived

            if n_survived == 0:
                stuck_count += 1
                logger.warning(f"[Iter {iteration}] No survivors from Stage 1. stuck_count={stuck_count}")
                iter_time = time.time() - iter_start_time
                history.append(
                    IterationResult(
                        iteration=iteration,
                        n_proposed=n_proposed,
                        n_survived_stage1=0,
                        n_optimized_stage2=0,
                        best_expr=best_expr,
                        best_nmse=best_nmse,
                        top_5=[],
                        time_seconds=iter_time
                    )
                )
                continue

            # Step 4: Pass Top K candidates to Stage 2 JAX Optimization
            top_k_candidates = reconstructed_results[:self.top_k]
            stage2_results = self.optimizer.optimize_batch(
                top_k_candidates,
                X,
                y_obs,
                data_vars
            )

            # Step 5: Update global best if improved
            if stage2_results:
                best_iter_cand = min(stage2_results, key=lambda x: x[2])
                iter_best_expr, iter_best_combined_score, iter_best_nmse, iter_best_arc, iter_opt_res = best_iter_cand

                if iter_best_nmse < best_nmse:
                    best_nmse = iter_best_nmse
                    best_expr = iter_best_expr
                    best_theta = iter_opt_res.theta
                    best_likelihood = iter_opt_res.likelihood

            # Step 6: Track stuck_count for Explore/Exploit/Escape mode switching
            if best_nmse < prev_best_nmse * 0.99:  # Require >1% improvement to reset
                stuck_count = 0
            else:
                stuck_count += 1
            prev_best_nmse = best_nmse

            # Step 7: Build feedback loop previous_best for next iteration
            iter_feedback = [(r[0], r[2]) for r in stage2_results if np.isfinite(r[2])]
            previous_best.extend(iter_feedback)
            previous_best = sorted(previous_best, key=lambda x: x[1])[:20]

            # Build IterationResult
            iter_time = time.time() - iter_start_time
            top_5 = [(r[0], r[2]) for r in stage2_results[:5]]
            
            iter_res = IterationResult(
                iteration=iteration,
                n_proposed=n_proposed,
                n_survived_stage1=n_survived,
                n_optimized_stage2=len(stage2_results),
                best_expr=best_expr,
                best_nmse=best_nmse,
                top_5=top_5,
                time_seconds=iter_time
            )
            history.append(iter_res)

            if self.verbose:
                print(f"[Iter {iteration}/{self.max_iterations}] Proposed: {n_proposed} | Stage 1 survivors: {n_survived} | Stage 2 top NMSE: {best_nmse:.6f} | stuck: {stuck_count}")
                for idx, (expr, combined, nmse, arc, opt) in enumerate(stage2_results[:3]):
                    print(f"  #{idx+1}: {expr} (NMSE={nmse:.6f}, theta={opt.theta})")

            # Early convergence check
            if best_nmse < self.convergence_nmse:
                if self.verbose:
                    print(f"[CONVERGED] Converged at iteration {iteration} (NMSE={best_nmse:.2e} < {self.convergence_nmse:.2e})")
                break

        total_time = time.time() - start_time
        converged = best_nmse < self.convergence_nmse

        return SearchResult(
            best_expr=best_expr,
            best_nmse=best_nmse,
            best_theta=best_theta,
            best_likelihood=best_likelihood,
            history=history,
            total_candidates_proposed=total_candidates_proposed,
            total_candidates_survived_stage1=total_candidates_survived_stage1,
            total_time_seconds=total_time,
            converged=converged
        )
